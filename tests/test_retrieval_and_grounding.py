"""
Regression tests for:
  1. Query-aware k-hop retrieval boost (engine.py::_apply_query_aware_boost)
  2. Answer-grounding gate (scratchpad_agent.py::_check_response_grounding)

Standalone, no pytest required, matching this repo's existing style:

    python3 tests/test_retrieval_and_grounding.py

Uses a throwaway on-disk sqlite file, never the real scratchpad_memory.db.
"""
import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

TEST_DB_PATH = "/tmp/_scratchpad_retrieval_grounding_test.db"
results = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((name, status))
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))


def setup_db():
    import database
    database.DB_PATH = TEST_DB_PATH
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    database.initialize_database()
    return database


def test_query_aware_retrieval():
    """
    Reproduces the exact failure mode the fix targets: a low-relevance-score
    root cause competing against many high-relevance-score but irrelevant
    facts under a tight token budget. Without query-awareness, the root
    cause loses. With it, a query naming an entity in its neighborhood
    correctly surfaces it.
    """
    database = setup_db()
    import engine

    conn = database.get_db_connection()
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO sessions (session_id) VALUES ('sess1')")

    def insert(src, rel, tgt, score):
        cur.execute(
            """INSERT INTO knowledge_graph
               (edge_id, session_id, source_entity, relationship, target_entity,
                citation_quote, is_active, relevance_score)
               VALUES (?, 'sess1', ?, ?, ?, 'x', TRUE, ?)""",
            (str(uuid.uuid4()), src, rel, tgt, score),
        )

    insert("SESSION_MIDDLEWARE", "connects_to", "AUTH_SERVICE", 0.10)
    insert("AUTH_SERVICE", "connects_to", "API_GATEWAY", 0.15)
    for i in range(20):
        insert(f"BILLING_JOB_{i}", "connects_to", f"BILLING_QUEUE_{i}", 0.90)
    conn.commit()
    conn.close()

    without_query = engine.compile_bounded_markdown_view("sess1", max_tokens=200, query=None)
    with_query = engine.compile_bounded_markdown_view(
        "sess1", max_tokens=200, query="Why did API_GATEWAY return errors?", k_hops=2
    )

    check(
        "query-aware retrieval: root cause LOST under old pure-relevance ranking (confirms the bug existed)",
        "SESSION_MIDDLEWARE" not in without_query,
    )
    check(
        "query-aware retrieval: root cause FOUND once query names something in its neighborhood",
        "SESSION_MIDDLEWARE" in with_query and "AUTH_SERVICE" in with_query,
    )


def test_answer_grounding_gate():
    """
    The gate exists because every prior verification gate protects the
    GRAPH, not the FINAL ANSWER. Confirms it flags an invented entity name
    and does not flag an answer built only from verified entities.
    """
    database = setup_db()
    import inference
    from scratchpad_agent import ScratchpadPoweredLLM

    fake_answers = []

    def fake_call_llm(self, messages, tools=None, **kwargs):
        return {"choices": [{"message": {"role": "assistant", "content": fake_answers[-1]}}]}

    ScratchpadPoweredLLM._call_llm = fake_call_llm
    ScratchpadPoweredLLM._generate_initial_plan = lambda self: None
    ScratchpadPoweredLLM._absorb_response = lambda self, response, messages: None

    agent = ScratchpadPoweredLLM(
        goal="test", session_id="ground_test", backend="lmstudio", auto_plan=False
    )

    conn = database.get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO knowledge_graph
           (edge_id, session_id, source_entity, relationship, target_entity,
            citation_quote, is_active)
           VALUES (?, 'ground_test', 'AUTH_SERVICE', 'connects_to', 'POSTGRES_PRIMARY', 'x', TRUE)""",
        (str(uuid.uuid4()),),
    )
    conn.commit()
    conn.close()

    fake_answers.append("The root cause was BILLING_QUEUE_OVERFLOW causing AUTH_SERVICE to fail.")
    r1 = agent.call([{"role": "user", "content": "why did it fail?"}])
    check(
        "grounding gate: flags an invented entity not in the graph",
        r1["_scratchpad_grounding"]["grounded"] is False
        and "BILLING_QUEUE_OVERFLOW" in r1["_scratchpad_grounding"]["unknown_entities"],
        str(r1["_scratchpad_grounding"]),
    )

    fake_answers.append("AUTH_SERVICE connects to POSTGRES_PRIMARY, which caused the failure.")
    r2 = agent.call([{"role": "user", "content": "why did it fail?"}])
    check(
        "grounding gate: does not flag an answer using only known entities",
        r2["_scratchpad_grounding"]["grounded"] is True,
        str(r2["_scratchpad_grounding"]),
    )


if __name__ == "__main__":
    test_query_aware_retrieval()
    test_answer_grounding_gate()

    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)

    print()
    failed = [name for name, status in results if status == "FAIL"]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed.")
    if failed:
        print("FAILED:", failed)
        sys.exit(1)
