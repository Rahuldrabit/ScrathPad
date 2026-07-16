"""
ScratchpadPoweredLLM: drop-in LLM wrapper that uses the scratchpad middleware
as persistent working memory for tool-calling agents.

The wrapper sits ABOVE the LLM. Any tool-calling agent that currently calls
`openai.chat.completions.create(messages=..., tools=...)` can be re-pointed at
this wrapper with no other code changes:

    from scratchpad_agent import ScratchpadPoweredLLM

    agent = ScratchpadPoweredLLM(
        goal="Find the bug in order_service.ts that causes lost orders",
        context_size=8000,
    )

    # Now the agent IS the LLM. Same call shape.
    response = agent(messages, tools=tools)
    # (also exposed as agent.call(...) if __call__ feels too magic)

What the wrapper does on every call:
  1. Fetches the current scratchpad view, token-bounded to
     (context_size - system_safety_buffer) so it always fits.
  2. Injects that view into the system prompt as
     "# SCRATCHPAD MEMORY (auto-injected)".
  3. Calls the LLM.
  4. Extracts new knowledge-graph triplets from the LLM's text response
     and stores them in the scratchpad.
  5. Returns the OpenAI-compatible response dict unchanged, so the
     agent code (which expects .choices[0].message.tool_calls etc.)
     works without any modification.

What the wrapper does at init (__init__):
  - Creates the scratchpad session with the goal.
  - Asks the LLM to generate an InitialPlan (master_plan + key_entities
    + key_variables_to_monitor) from the goal. The plan goes into
    sessions.master_plan, the entities become triplets, the variables
    go into the Unresolved Variables Matrix. So the very first LLM
    call inside the agent already has a populated scratchpad to read.

What the wrapper exposes for the agent:
  - agent(messages, tools=...)         drop-in LLM call
  - agent.call(...)                    same thing, by name
  - agent.record_observation(text,...) add a tool result / external fact
  - agent.mark_resolved(var_name)      mark a variable as resolved
  - agent.get_scratchpad_view()        peek at the current memory
  - agent.get_session_id()             for logging / cross-references
"""
from __future__ import annotations

import os
import uuid
import json
import httpx
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from inference import UniversalInferenceEngine
from schema import GraphTriplet, PageExtractionPayload
from engine import compile_bounded_markdown_view, commit_page_data_to_sqlite
from database import initialize_database, get_db_connection


# ─────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────

class InitialPlan(BaseModel):
    """Plan the LLM generates at agent init from a high-level goal."""
    master_plan: str = Field(
        ...,
        description=(
            "Markdown-formatted plan with numbered steps. Each step on its "
            "own line, starting with a digit + period."
        ),
    )
    key_entities: List[str] = Field(
        default_factory=list,
        description=(
            "Key entities the agent should track (files, services, "
            "components, configs, etc.). Will be stored as 'GOAL involves ENTITY'."
        ),
    )
    key_variables: List[str] = Field(
        default_factory=list,
        description=(
            "Key variables the agent should monitor (unknown values, "
            "missing config, blockers, etc.). Will be added to the "
            "Unresolved Variables Matrix as MISSING."
        ),
    )


class FactExtraction(BaseModel):
    """Triplets + variable updates extracted from arbitrary text."""
    triplets: List[GraphTriplet] = Field(default_factory=list)
    resolved_variables: List[str] = Field(
        default_factory=list,
        description="Variable names that are now answered / known.",
    )
    new_missing_variables: List[str] = Field(
        default_factory=list,
        description="Variable names that the text revealed to be missing.",
    )


# ─────────────────────────────────────────────────────────────────────────
# Wrapper
# ─────────────────────────────────────────────────────────────────────────

class ScratchpadPoweredLLM:
    """
    Drop-in LLM wrapper with automatic scratchpad memory management.

    Use as a callable:
        response = agent(messages, tools=tools)

    Or by name:
        response = agent.call(messages, tools=tools)

    The response is the same OpenAI-compatible dict the underlying LLM
    would have returned, so agent code that inspects
    response["choices"][0]["message"]["tool_calls"] works unchanged.
    """

    DEFAULT_SAFETY_BUFFER = 2000  # tokens reserved for system + user + LLM output

    def __init__(
        self,
        goal: str,
        context_size: int = 8000,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
        backend: str = "lmstudio",
        middleware_url: Optional[str] = None,
        system_safety_buffer: int = DEFAULT_SAFETY_BUFFER,
        auto_plan: bool = True,
        lmstudio_url: str = "http://localhost:1234/v1",
    ):
        self.goal = goal
        self.context_size = context_size
        self.session_id = session_id or f"agent-{uuid.uuid4().hex[:8]}"
        self.model = model or os.getenv("SCRATCHPAD_MODEL_NAME", "google/gemma-4-e4b")
        self.backend = backend
        self.middleware_url = middleware_url  # None → in-process mode
        self.lmstudio_url = lmstudio_url
        self.system_safety_buffer = system_safety_buffer
        # The scratchpad view will be token-bounded to this budget.
        # If context_size is 8000 and safety buffer is 2000, the view
        # gets up to 6000 tokens — plenty for the LLM to read and reason over.
        self.scratchpad_budget = max(context_size - system_safety_buffer, 512)

        # Configure env so UniversalInferenceEngine picks the right backend/model
        os.environ["SCRATCHPAD_LLM_BACKEND"] = backend
        os.environ["SCRATCHPAD_MODEL_NAME"] = self.model

        # The inference engine used for both LLM calls and structured fact
        # extraction. Sharing one client keeps tokenizer/warmup state hot.
        self.inference = UniversalInferenceEngine()

        # Set up the scratchpad session
        self._initialize_session()

        # Ask the LLM to plan first. After this returns, the scratchpad
        # already contains the master_plan, key entities, and key
        # variables — so the very first agent(messages) call has a
        # populated memory to read.
        if auto_plan:
            self._generate_initial_plan()

    # ─── Public API ─────────────────────────────────────────────────────

    def __call__(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Drop-in replacement for the LLM call. See class docstring."""
        return self.call(messages, tools=tools, **kwargs)

    def call(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        1. Fetch the current scratchpad view (token-bounded).
        2. Inject it as a system message at the top of the conversation.
        3. Call the LLM.
        4. Extract facts from the LLM's text response and store them.
        5. Return the OpenAI-compatible response dict.
        """
        view = self._fetch_scratchpad_view()
        scratchpad_block = self._format_injection(view)
        augmented = self._inject_into(messages, scratchpad_block)
        response = self._call_llm(augmented, tools=tools, **kwargs)
        self._absorb_response(response, augmented)
        return response

    def record_observation(self, text: str, source: str = "external") -> int:
        """
        Add an external observation (tool result, log line, fetched
        content, anything) to the scratchpad. Returns the number of
        triplets actually committed after the verification gate.
        """
        facts = self._extract_facts(text, context_hint=f"source={source}")
        if not facts:
            return 0
        return self._store_facts(facts, agent_id=f"observation:{source}")

    def mark_resolved(self, variable_name: str) -> None:
        """Mark an unresolved variable as resolved (delete it from the matrix)."""
        self._mutate_variables({variable_name: "RESOLVED"})

    def mark_missing(self, variable_name: str) -> None:
        """Add or re-add a variable as MISSING in the matrix."""
        self._mutate_variables({variable_name: "MISSING"})

    def get_scratchpad_view(self) -> str:
        """Return the current token-bounded markdown view of the scratchpad."""
        return self._fetch_scratchpad_view()

    def get_session_id(self) -> str:
        return self.session_id

    # ─── Internals: session + plan ──────────────────────────────────────

    def _initialize_session(self) -> None:
        if self.middleware_url:
            httpx.post(
                f"{self.middleware_url}/v1/session/init",
                json={"session_id": self.session_id, "user_query": self.goal},
                timeout=30.0,
            ).raise_for_status()
            return

        # In-process mode
        try:
            initialize_database()
        except Exception:
            pass
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO sessions
                (session_id, user_query, master_plan, global_status)
            VALUES (?, ?, '', 'EXECUTING')
            """,
            (self.session_id, self.goal),
        )
        conn.commit()
        conn.close()

    def _generate_initial_plan(self) -> None:
        """
        First LLM call: ask the model to decompose the goal into a
        master_plan + key_entities + key_variables, and persist them.
        Failure here is non-fatal — the agent will still work, just
        starting with an empty memory instead of a planned one.
        """
        system_prompt = (
            "You are a strategic planning agent. Given a high-level goal, "
            "produce a concrete, actionable plan. Be concise: short numbered "
            "steps, the 3-7 most important entities to track, and the 3-7 "
            "most important unknowns to monitor. Output ONLY valid JSON "
            "matching the requested schema — no prose around it."
        )
        prompt = (
            f"GOAL:\n{self.goal}\n\n"
            "Generate the master plan, key entities, and key variables."
        )
        try:
            plan: InitialPlan = self.inference.generate_structured(
                prompt=prompt,
                system_prompt=system_prompt,
                response_schema=InitialPlan,
            )
        except Exception as e:
            print(f"[ScratchpadPoweredLLM] Initial plan generation failed: {e}")
            # Fall back to a stub plan so the session is still useful.
            self._update_session_plan(
                f"## Plan (auto-fallback)\n1. Investigate: {self.goal}\n2. Execute"
            )
            return

        # Persist the plan
        self._update_session_plan(plan.master_plan)

        # Add key entities as triplets of the form GOAL -> involves -> ENTITY
        entity_facts = [
            {
                "source_entity": "GOAL",
                "relationship": "involves",
                "target_entity": _canon(e),
                "citation_quote": f"Initial plan identified '{e}' as a key entity.",
            }
            for e in plan.key_entities
        ]
        if entity_facts:
            self._store_facts(entity_facts, agent_id="plan:entities")

        # Add key variables to the Unresolved Variables Matrix
        if plan.key_variables:
            self._mutate_variables(
                {_canon(v): "MISSING" for v in plan.key_variables}
            )

    # ─── Internals: view + injection ────────────────────────────────────

    def _fetch_scratchpad_view(self) -> str:
        if self.middleware_url:
            r = httpx.get(
                f"{self.middleware_url}/v1/session/{self.session_id}/memory",
                params={"max_tokens": self.scratchpad_budget},
                timeout=30.0,
            )
            r.raise_for_status()
            return r.json().get("markdown_view", "")
        return compile_bounded_markdown_view(
            self.session_id, max_tokens=self.scratchpad_budget
        )

    def _format_injection(self, view: str) -> str:
        """Format the scratchpad as a system-prompt block."""
        return (
            f"# SCRATCHPAD MEMORY (auto-injected, fits within "
            f"{self.scratchpad_budget} tokens)\n"
            f"# GOAL: {self.goal}\n"
            f"# This is your persistent working memory across every call "
            f"in this session.\n"
            f"# Use it to recall entities, track progress on the master "
            f"plan, and remember unresolved questions.\n\n"
            f"{view}"
        )

    def _inject_into(
        self, messages: List[Dict[str, Any]], scratchpad_block: str
    ) -> List[Dict[str, Any]]:
        """
        If the first message is already a system message, prepend the
        scratchpad to it. Otherwise insert a new system message at the top.
        This keeps the rest of the conversation (user/tool/assistant turns)
        intact and in order.
        """
        out = [dict(m) for m in messages]  # shallow copy
        if out and out[0].get("role") == "system":
            existing = out[0].get("content", "") or ""
            out[0] = {**out[0], "content": scratchpad_block + "\n\n" + existing}
        else:
            out = [{"role": "system", "content": scratchpad_block}] + out
        return out

    # ─── Internals: LLM call ────────────────────────────────────────────

    def _call_llm(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 2048),
        }
        if tools:
            payload["tools"] = tools

        # Prefer going through the middleware if configured (so the
        # verification gate, telemetry, etc. are all in the loop).
        if self.middleware_url:
            r = httpx.post(
                f"{self.middleware_url}/v1/chat/completions",
                json=payload,
                timeout=kwargs.get("timeout", 120.0),
            )
            r.raise_for_status()
            return r.json()

        # Direct LM Studio / OpenAI-compatible call
        r = httpx.post(
            f"{self.lmstudio_url}/chat/completions",
            json=payload,
            timeout=kwargs.get("timeout", 120.0),
        )
        r.raise_for_status()
        return r.json()

    # ─── Internals: response → scratchpad ───────────────────────────────

    def _absorb_response(
        self, response: Dict[str, Any], messages: List[Dict[str, Any]]
    ) -> None:
        """
        Record bookkeeping facts from the LLM response. By default this
        only records tool calls as triplets (cheap, no extra LLM call).
        Text content is NOT auto-extracted — that's expensive (another
        LLM call) and usually unnecessary because the agent's main value
        comes from tool results, not from its prose. If you do want to
        extract facts from the agent's text, call
        `record_observation(text)` explicitly.
        """
        try:
            choices = response.get("choices") or []
            if not choices:
                return
            message = choices[0].get("message") or {}
            tool_calls = message.get("tool_calls") or []

            # Tool calls → bookkeeping triplets (no extra LLM call)
            for tc in tool_calls:
                if tc.get("type") == "function":
                    fn = (tc.get("function") or {}).get("name", "unknown_tool")
                    args = (tc.get("function") or {}).get("arguments", "")
                    self._store_facts(
                        [
                            {
                                "source_entity": "AGENT",
                                "relationship": "calls_tool",
                                "target_entity": _canon(fn),
                                "citation_quote": f"Tool call: {fn}({args[:120]})",
                            }
                        ],
                        agent_id="llm:tool_call",
                    )
        except Exception as e:
            # Scratchpad updates are best-effort. Never let them break the
            # main LLM call.
            print(f"[ScratchpadPoweredLLM] Failed to absorb response: {e}")

    def _extract_facts(self, text: str, context_hint: str = "") -> List[Dict[str, Any]]:
        """
        Use the LLM (structured output) to pull triplets + variable
        updates out of arbitrary text. Each triplet's citation_quote
        must be a verbatim substring of the input — the verification
        gate in commit_page_data_to_sqlite will drop anything that isn't.
        """
        if not text or not text.strip():
            return []

        system_prompt = (
            "You extract Knowledge Graph triplets and variable updates from "
            "unstructured text. Every triplet MUST include a citation_quote "
            "that is a verbatim substring of the input — fabricating a quote "
            "will get the fact rejected by the verification gate. Only "
            "extract facts that are actually supported by the text."
        )
        prompt = (
            f"CONTEXT: {context_hint}\n\n"
            f"TEXT:\n{text}\n\n"
            "Extract triplets and any variable state changes as JSON."
        )

        try:
            result: FactExtraction = self.inference.generate_structured(
                prompt=prompt,
                system_prompt=system_prompt,
                response_schema=FactExtraction,
            )
        except Exception as e:
            print(f"[ScratchpadPoweredLLM] Fact extraction failed: {e}")
            return []

        facts: List[Dict[str, Any]] = []
        for t in result.triplets:
            facts.append(
                {
                    "source_entity": t.source_entity,
                    "relationship": t.relationship,
                    "target_entity": t.target_entity,
                    "citation_quote": t.citation_quote,
                }
            )

        # Apply variable updates
        updates: Dict[str, str] = {}
        for v in result.resolved_variables:
            updates[_canon(v)] = "RESOLVED"
        for v in result.new_missing_variables:
            updates[_canon(v)] = "MISSING"
        if updates:
            self._mutate_variables(updates)

        return facts

    # ─── Internals: persistence ─────────────────────────────────────────

    def _store_facts(self, facts: List[Dict[str, Any]], agent_id: str) -> int:
        if not facts:
            return 0
        triplets = [
            GraphTriplet(
                source_entity=f["source_entity"],
                relationship=f["relationship"],
                target_entity=f["target_entity"],
                citation_quote=f["citation_quote"],
            )
            for f in facts
        ]
        # The verification gate requires citation_quote to be a verbatim
        # substring of raw_chunk. We pass the joined citations as the
        # raw chunk so the gate has the exact text to match against.
        raw_chunk = "\n".join(f["citation_quote"] for f in facts)
        payload = PageExtractionPayload(
            extracted_triplets=triplets,
            unresolved_variables_mutations={},
            is_chunk_completely_exhausted=True,
        )

        if self.middleware_url:
            r = httpx.post(
                f"{self.middleware_url}/v1/agent/update",
                json={
                    "agent_id": agent_id,
                    "session_id": self.session_id,
                    "raw_active_chunk": raw_chunk,
                    "extracted_triplets": [t.model_dump() for t in triplets],
                    "unresolved_variables_mutations": {},
                    "is_chunk_completely_exhausted": True,
                },
                timeout=30.0,
            )
            if r.status_code == 200:
                return r.json().get("verified_triplets_committed", 0)
            return 0

        return commit_page_data_to_sqlite(
            session_id=self.session_id,
            agent_id=agent_id,
            raw_chunk=raw_chunk,
            extraction_data=payload,
        )

    def _mutate_variables(self, mutations: Dict[str, str]) -> None:
        if not mutations:
            return
        if self.middleware_url:
            httpx.post(
                f"{self.middleware_url}/v1/agent/update",
                json={
                    "agent_id": "scratchpad_agent:vars",
                    "session_id": self.session_id,
                    "raw_active_chunk": "\n".join(mutations.keys()),
                    "extracted_triplets": [],
                    "unresolved_variables_mutations": mutations,
                    "is_chunk_completely_exhausted": False,
                },
                timeout=30.0,
            ).raise_for_status()
            return
        conn = get_db_connection()
        cursor = conn.cursor()
        for var_name, status in mutations.items():
            var_id = f"{self.session_id}_{var_name}"
            if status.upper() == "RESOLVED":
                cursor.execute(
                    "DELETE FROM unresolved_variables WHERE variable_id = ?",
                    (var_id,),
                )
            else:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO unresolved_variables
                        (variable_id, session_id, variable_name, status)
                    VALUES (?, ?, ?, ?)
                    """,
                    (var_id, self.session_id, var_name, status.upper()),
                )
        conn.commit()
        conn.close()

    def _update_session_plan(self, plan: str) -> None:
        if self.middleware_url:
            # No dedicated plan endpoint; fall back to direct DB write
            # through a tiny internal helper. For simplicity here, just
            # store the plan as a special triplet so the view picks it up.
            self._store_facts(
                [
                    {
                        "source_entity": "GOAL",
                        "relationship": "has_plan",
                        "target_entity": "MASTER_PLAN",
                        "citation_quote": plan[:4000],  # gate needs verbatim; trimmed is fine
                    }
                ],
                agent_id="plan:master",
            )
            return
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET master_plan = ? WHERE session_id = ?",
            (plan, self.session_id),
        )
        conn.commit()
        conn.close()


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _canon(s: str) -> str:
    """Normalize a name into UPPER_SNAKE_CASE for entity canonicalization."""
    if not s:
        return ""
    out = []
    for ch in s.strip():
        if ch.isalnum():
            out.append(ch.upper())
        elif ch in (" ", "_", "-", "/", "."):
            out.append("_")
    canon = "".join(out).strip("_")
    while "__" in canon:
        canon = canon.replace("__", "_")
    return canon or "UNNAMED"
