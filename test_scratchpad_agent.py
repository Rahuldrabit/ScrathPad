"""
Full tool-calling agent test for ScratchpadPoweredLLM.

Simulates a real ReAct-style tool-calling agent that:
  1. Has a high-level goal
  2. Uses the scratchpad wrapper as its LLM
  3. Has access to read_file, search_code, and finish tools
  4. Runs a multi-turn loop until the agent decides it's done
  5. Verifies the scratchpad is correctly managed throughout

The agent code never touches the scratchpad directly — the wrapper does
all of it. The agent only writes:
  - the tool result back into the conversation (OpenAI protocol)
  - one call to agent.record_observation(...) to log the result into memory

Run:
    $env:SCRATCHPAD_LLM_BACKEND = "lmstudio"
    $env:SCRATCHPAD_MODEL_NAME = "google/gemma-4-e4b"
    python -u test_scratchpad_agent.py
"""
import os
import sys
import json
import sqlite3
import textwrap

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


GOAL = (
    "Find and fix the bug in order_service.ts that causes orders to be "
    "lost after payment. Confirm the fix by tracing the full order flow."
)

SESSION_ID = "fix-order-bug-agent-001"
TEST_DB = "test_agent_demo.db"

# Mock codebase
MOCK_FILES = {
    "order_service.ts": textwrap.dedent("""
        import { db } from './postgres_client';
        import { queue } from './redis_queue';
        import { logger } from './logging_service';

        export async function processOrder(orderId: string) {
          const order = await db.query('SELECT * FROM orders WHERE id = $1', [orderId]);
          if (!order) throw new Error('Order not found');

          // BUG: We push to the kitchen display but never wait for an ACK.
          // If the kitchen display is down, the order is silently lost.
          await queue.add('kitchen_display', { orderId, items: order.items });

          await db.query(
            'INSERT INTO payment_ledger (order_id, amount, status) VALUES ($1, $2, $3)',
            [orderId, order.total, 'PENDING']
          );

          // BUG: We never mark the order as 'paid' after the ledger write.
          return { ok: true };
        }
    """),
    "redis_queue.ts": textwrap.dedent("""
        // redis_queue.ts — wraps ioredis stream operations
        import Redis from 'ioredis';
        const r = new Redis();
        export const queue = {
          add: async (stream: string, data: any) => {
            return r.xadd(`stream:${stream}`, '*', 'payload', JSON.stringify(data));
          }
        };
    """),
    "kitchen_display_consumer.ts": textwrap.dedent("""
        // kitchen_display_consumer.ts — reads from the stream
        import { queue } from './redis_queue';
        // ... consumer loop that may be down for maintenance
    """),
}

MOCK_SEARCH = {
    "order_id": [
        "order_service.ts: const order = await db.query('SELECT * FROM orders WHERE id = $1', [orderId]);",
        "order_service.ts: throw new Error('Order not found');",
    ],
    "queue.add": [
        "order_service.ts: await queue.add('kitchen_display', { orderId, items: order.items });",
    ],
    "kitchen_display": [
        "order_service.ts: await queue.add('kitchen_display', { orderId, items: order.items });",
        "kitchen_display_consumer.ts: // consumer loop that may be down for maintenance",
    ],
    "payment_ledger": [
        "order_service.ts: 'INSERT INTO payment_ledger (order_id, amount, status) VALUES ($1, $2, $3)'",
    ],
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a source file by path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search the codebase for a keyword and return matches.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Call when the goal is achieved. Provide summary + fix.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "root_cause": {"type": "string"},
                    "proposed_fix": {"type": "string"},
                },
                "required": ["summary", "root_cause", "proposed_fix"],
            },
        },
    },
]


def tool_read_file(args):
    return MOCK_FILES.get(args.get("path", ""), f"ERROR: file not found: {args.get('path','')}")


def tool_search_code(args):
    matches = MOCK_SEARCH.get(args.get("query", ""), [])
    return "\n".join(matches) if matches else f"No matches for: {args.get('query','')}"


def tool_finish(args):
    return f"FINISHED — summary: {args.get('summary','')} | cause: {args.get('root_cause','')} | fix: {args.get('proposed_fix','')}"


TOOL_DISPATCH = {
    "read_file": tool_read_file,
    "search_code": tool_search_code,
    "finish": tool_finish,
}


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────
def reset_db():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def q(sql, args=()):
    conn = sqlite3.connect(TEST_DB)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(sql, args)
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def count(table, where=""):
    return q(f"SELECT COUNT(*) as c FROM {table} {where}")[0]["c"]


# ─────────────────────────────────────────────────────────────────────────
# The agent loop
# ─────────────────────────────────────────────────────────────────────────
def run_agent_loop(agent, max_turns: int = 6):
    """
    Standard ReAct-style tool-calling loop. The agent code never
    touches the scratchpad directly — the wrapper does it.
    """
    conversation = [
        {
            "role": "user",
            "content": (
                f"GOAL: {GOAL}\n\nUse the available tools to investigate. "
                f"Call finish() when you have identified the root cause and "
                f"proposed a fix."
            ),
        }
    ]
    history = []
    finished_normally = False

    for turn in range(max_turns):
        print(f"\n───── TURN {turn + 1} ─────", flush=True)

        # The wrapper auto-injects the scratchpad, calls the LLM, and
        # records any tool calls. The agent just sees a normal LLM response.
        try:
            response = agent(conversation, tools=TOOLS, max_tokens=512, timeout=240.0)
        except Exception as e:
            print(f"  LLM call timed out / failed on turn {turn + 1}: {type(e).__name__}", flush=True)
            print(f"  Conversation had {len(conversation)} messages, "
                  f"~{sum(len(str(m.get('content',''))) for m in conversation)} chars of text", flush=True)
            break

        message = response["choices"][0]["message"]
        content = message.get("content", "")
        tool_calls = message.get("tool_calls", [])

        if content:
            print(f"  Agent: {content[:200]}{'...' if len(content) > 200 else ''}", flush=True)

        # Append the assistant turn (OpenAI protocol)
        conversation.append(message)

        if not tool_calls:
            print("  Agent finished without a tool call.", flush=True)
            break

        # Execute each tool and feed the result back
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            try:
                fn_args = json.loads(tc["function"].get("arguments", "{}"))
            except json.JSONDecodeError:
                fn_args = {}
            print(f"  Tool: {fn_name}({json.dumps(fn_args)[:120]})", flush=True)

            handler = TOOL_DISPATCH.get(fn_name)
            tool_result = handler(fn_args) if handler else f"Unknown tool: {fn_name}"
            print(f"  Result: {str(tool_result)[:200]}{'...' if len(str(tool_result)) > 200 else ''}", flush=True)

            # ONE method call to add the tool result to the scratchpad.
            # (This is the only scratchpad touch the agent code needs.)
            agent.record_observation(
                f"TOOL: {fn_name}\nARGS: {json.dumps(fn_args)}\nRESULT:\n{tool_result}",
                source=f"tool:{fn_name}",
            )

            # Standard OpenAI tool-result message
            conversation.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": str(tool_result),
            })

        history.append({
            "turn": turn,
            "tool_calls": [tc["function"]["name"] for tc in tool_calls],
        })

        if any(tc["function"]["name"] == "finish" for tc in tool_calls):
            finished_normally = True
            print("  Agent called finish() — done.", flush=True)
            break

    return {"turns": history, "finished_normally": finished_normally,
            "final_messages": len(conversation)}


# ─────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────
results = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((name, status))
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""), flush=True)


def main():
    from scratchpad_agent import ScratchpadPoweredLLM
    from engine import count_tokens as tok

    reset_db()

    import database
    database.DB_PATH = TEST_DB
    database.initialize_database()

    os.environ["SCRATCHPAD_LLM_BACKEND"] = "lmstudio"
    os.environ["SCRATCHPAD_MODEL_NAME"] = "google/gemma-4-e4b"

    # ── Step 1: Create the wrapper (auto-plans from goal) ──
    print("=" * 60, flush=True)
    print("STEP 1: Initialize wrapper → LLM creates initial plan from goal", flush=True)
    print("=" * 60, flush=True)
    agent = ScratchpadPoweredLLM(
        goal=GOAL,
        context_size=4096,
        session_id=SESSION_ID,
        system_safety_buffer=1500,
        auto_plan=True,
    )
    print(f"  Session: {agent.get_session_id()}", flush=True)
    print(f"  Scratchpad budget: {agent.scratchpad_budget} tokens", flush=True)

    sessions = q("SELECT * FROM sessions WHERE session_id = ?", (SESSION_ID,))
    check("session row exists", len(sessions) == 1)
    check("session goal stored", GOAL in sessions[0]["user_query"])
    plan_text = sessions[0]["master_plan"] or ""
    check("master_plan is non-empty (LLM generated a plan)",
          bool(plan_text.strip()), f"{len(plan_text)} chars")
    if plan_text:
        print(f"\n  Generated plan ({len(plan_text)} chars):\n  {plan_text}\n", flush=True)

    kg_after_plan = count("knowledge_graph", f"WHERE session_id = '{SESSION_ID}'")
    vars_after_plan = count("unresolved_variables", f"WHERE session_id = '{SESSION_ID}'")
    check("plan populated the knowledge graph with >=1 triplet",
          kg_after_plan >= 1, f"{kg_after_plan} triplets after plan")
    check("plan registered key variables as MISSING",
          vars_after_plan >= 1, f"{vars_after_plan} variables after plan")

    # ── Step 2: First view fetch (proves context-window sizing) ──
    print("\n" + "=" * 60, flush=True)
    print("STEP 2: Initial scratchpad view (fits the context window)", flush=True)
    print("=" * 60, flush=True)
    view1 = agent.get_scratchpad_view()
    check("view is non-empty", len(view1) > 0, f"{len(view1)} chars")
    check("view fits within scratchpad budget",
          tok(view1) <= agent.scratchpad_budget + 50,
          f"{tok(view1)} tokens, budget={agent.scratchpad_budget}")
    check("view contains 'UNRESOLVED VARIABLES' section",
          "UNRESOLVED VARIABLES" in view1)
    check("view contains 'KNOWLEDGE GRAPH' section",
          "KNOWLEDGE GRAPH" in view1)
    print(f"  View1 ({tok(view1)} tokens, budget {agent.scratchpad_budget}):", flush=True)
    print(f"  {view1[:500]}...\n", flush=True)

    # ── Step 3: Run the full multi-turn tool-calling agent loop ──
    print("=" * 60, flush=True)
    print("STEP 3: Multi-turn tool-calling agent loop (up to 6 turns)", flush=True)
    print("=" * 60, flush=True)
    summary = run_agent_loop(agent, max_turns=4)
    n_turns = len(summary["turns"])
    check("agent ran at least 1 turn", n_turns >= 1, f"{n_turns} turns")
    check("agent ran multiple turns", n_turns >= 2, f"{n_turns} turns")

    # ── Step 4: Verify the scratchpad grew during the loop ──
    print("\n" + "=" * 60, flush=True)
    print("STEP 4: Scratchpad state after the agent loop", flush=True)
    print("=" * 60, flush=True)
    kg_final = count("knowledge_graph", f"WHERE session_id = '{SESSION_ID}'")
    check("knowledge_graph grew during the agent loop",
          kg_final > kg_after_plan,
          f"{kg_after_plan} → {kg_final} triplets")

    # Tool calls should be recorded as triplets
    tool_call_rows = q(
        f"SELECT * FROM knowledge_graph WHERE session_id = '{SESSION_ID}' "
        f"AND relationship = 'calls_tool'"
    )
    check("tool calls recorded as triplets in the scratchpad",
          len(tool_call_rows) >= 1,
          f"{len(tool_call_rows)} 'calls_tool' triplet(s)")
    if tool_call_rows:
        called_tools = {r["target_entity"] for r in tool_call_rows}
        print(f"  Tools called: {sorted(called_tools)}", flush=True)

    # ── Step 5: Verify the second view still fits the budget ──
    print("\n" + "=" * 60, flush=True)
    print("STEP 5: Final scratchpad view (still fits budget after growth)", flush=True)
    print("=" * 60, flush=True)
    view2 = agent.get_scratchpad_view()
    check("view2 is non-empty", len(view2) > 0, f"{len(view2)} chars")
    check("view2 still fits in scratchpad budget after growth",
          tok(view2) <= agent.scratchpad_budget + 50,
          f"{tok(view2)} tokens, budget={agent.scratchpad_budget}")
    check("view2 reflects the new observations",
          "ORDER_SERVICE" in view2 or "PROCESSORDER" in view2 or
          "QUEUE" in view2 or "KITCHEN" in view2 or "PAYMENT_LEDGER" in view2,
          "looking for any discovered entity from the tool results")
    print(f"  View2 ({tok(view2)} tokens):", flush=True)
    print(f"  {view2[:700]}...\n", flush=True)

    # ── Step 6: mark_resolved API still works mid-loop ──
    print("=" * 60, flush=True)
    print("STEP 6: mark_missing + mark_resolved API", flush=True)
    print("=" * 60, flush=True)
    agent.mark_missing("POSTMORTEM_DRAFT")
    vars_after_mark = count("unresolved_variables", f"WHERE session_id = '{SESSION_ID}'")
    check("mark_missing adds a variable", vars_after_mark >= 1)
    agent.mark_resolved("POSTMORTEM_DRAFT")
    remaining = q(
        "SELECT * FROM unresolved_variables WHERE session_id = ? AND variable_name = ?",
        (SESSION_ID, "POSTMORTEM_DRAFT"),
    )
    check("mark_resolved removes the variable", len(remaining) == 0)

    # ── Summary ──
    print("\n" + "=" * 60, flush=True)
    failed = [n for n, s in results if s == "FAIL"]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed.", flush=True)
    if failed:
        print("FAILED:", failed, flush=True)
    else:
        print("All checks passed — wrapper + agent loop is healthy.", flush=True)

    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
