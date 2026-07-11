from typing import Optional
from inference import UniversalInferenceEngine
from engine import commit_page_data_to_sqlite, compile_bounded_markdown_view
from schema import PageExtractionPayload, L1ExtractionPayload


class ScratchpadMiddleware:
    """
    Universal convenience wrapper for the "hand me messy text" pattern.
    Structured agents that already extract their own triplets should call
    the engine.py functions directly (see agent_sdk.py); this class exists
    for callers that don't want to do their own extraction step.

    Both commit_messy_input() and get_clean_context() now delegate to the
    same canonical engine.py functions that /v1/agent/update and
    /v1/session/{id}/memory use, instead of maintaining a second,
    independently-bugged copy of citation verification and view compilation.

    Previously this class had its own commit path (plain INSERT with no
    ON CONFLICT handling, no rejection telemetry, silently swallowed
    exceptions behind a print()) and its own view path (no Unresolved
    Variables section at all, no token budget despite declaring
    self.token_window and self.t_system_safety_buffer in __init__ - neither
    attribute was ever read anywhere else in the class).
    """

    def __init__(self, token_window: int = 8192, t_system_safety_buffer: int = 1500):
        self.inference_engine = UniversalInferenceEngine()
        self.token_window = token_window
        self.t_system_safety_buffer = t_system_safety_buffer

    def process_turn(
        self,
        session_id: str,
        messy_input: str,
        agent_id: str = "middleware_auto_extract",
    ) -> str:
        """
        Universal wrapper: extracts structured knowledge from raw messy
        text, commits it through the same verification gate every other
        commit path uses, and returns a clean, token-bounded context.
        """
        self.commit_messy_input(session_id, messy_input, agent_id=agent_id)
        return self.get_clean_context(session_id)

    def commit_messy_input(
        self,
        session_id: str,
        raw_text: str,
        agent_id: str = "middleware_auto_extract",
    ) -> int:
        """
        Extracts triplets from unstructured text via the SLM, then commits
        them through engine.commit_page_data_to_sqlite - the same citation
        check, entity canonicalization, and rejection path every structured
        agent update goes through. Returns the number of triplets that
        actually passed verification and were committed.

        Note this now lets exceptions propagate instead of catching them and
        only printing. main.py's /v1/middleware/process endpoint already
        converts exceptions into a proper HTTP 500; silently swallowing a
        failed extraction here meant that endpoint could return "success"
        with a stale, unchanged context after a real failure.
        """
        system_prompt = (
            "Extract clear entity-relationship triplets from the messy, "
            "unstructured text."
        )
        extracted_payload: L1ExtractionPayload = self.inference_engine.generate_structured(
            prompt=raw_text,
            system_prompt=system_prompt,
            response_schema=L1ExtractionPayload,
        )
        page_payload = PageExtractionPayload(
            extracted_triplets=extracted_payload.triplets,
            unresolved_variables_mutations={},
            is_chunk_completely_exhausted=True,
        )
        return commit_page_data_to_sqlite(
            session_id=session_id,
            agent_id=agent_id,
            raw_chunk=raw_text,
            extraction_data=page_payload,
        )

    def get_clean_context(self, session_id: str) -> str:
        """
        Returns the token-bounded Markdown view (Unresolved Variables always
        included in full, knowledge_graph rows filled in by relevance_score
        until budget runs out). self.token_window and
        self.t_system_safety_buffer are now actually used - previously they
        were declared in __init__ and never referenced again anywhere in
        this class, so get_clean_context() had no real budget at all.
        """
        max_tokens = max(self.token_window - self.t_system_safety_buffer, 256)
        return compile_bounded_markdown_view(session_id, max_tokens=max_tokens)
