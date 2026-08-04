"""
Full regression suite for this round of fixes:
  1. direction_check auto-repair bypass removed (was circular: reconstructed
     the value from the same fields it's supposed to verify against)
  2. direction_check restructured to a nested object - malformed/narrative
     values are now impossible to construct, not just harder to slip past
  3. Query-aware k-hop retrieval boost
  4. Answer-grounding gate on the final synthesized response

Standalone, no pytest required:
    python3 tests/test_direction_and_retrieval.py
"""
import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

TEST_DB_PATH = "/tmp/_scratchpad_final_test.db"
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


def test_narrative_direction_check_structurally_impossible():
    """
    Reproduces the exact three direction_check values from the real
    rejection log ('rolled out', 'routine health check passed', 'Root
    cause identified.') and confirms none of them can even be constructed
    as a GraphTriplet anymore - Pydantic rejects them at construction,
    before validate_direction or any commit-time guardrail ever runs.
    """
    from pydantic import ValidationError
    from schema import GraphTriplet

    for val in ["rolled out", "routine health check passed", "Root cause identified."]:
        try:
            GraphTriplet(
                direction_check=val,
                source_entity="X", source_type="SERVICE", relationship="connects_to",
                target_entity="Y", target_type="SERVICE", citation_quote="x",
            )
            check(f"narrative direction_check {val!r} structurally rejected", False)
        except ValidationError:
            check(f"narrative direction_check {val!r} structurally rejected", True)


def test_direction_check_fuzzy_matching_still_works():
    from schema import GraphTriplet, DirectionCheck, validate_direction

    def make(subj, obj, src, tgt):
        return GraphTriplet(
            direction_check=DirectionCheck(stated_subject=subj, stated_verb="connects_to", stated_object=obj),
            source_entity=src, source_type="SERVICE", relationship="connects_to",
            target_entity=tgt, target_type="SERVICE", citation_quote="x",
        )

    check(
        "phrasing drift (extra word each side) passes",
        validate_direction(make("the AUTH_SERVICE", "the connection pool", "AUTH_SERVICE", "CONNECTION_POOL")) is True,
    )
    check(
        "exact match still passes (regression check)",
        validate_direction(make("AUTH_SERVICE", "POSTGRES_PRIMARY", "AUTH_SERVICE", "POSTGRES_PRIMARY")) is True,
    )
    check(
        "genuine subject/object swap still correctly rejected",
        validate_direction(make("POSTGRES_PRIMARY", "AUTH_SERVICE", "AUTH_SERVICE", "POSTGRES_PRIMARY")) is False,
    )


def test_auto_repair_bypass_removed_end_to_end():
    """
    Full commit path: a genuinely backwards fact must be rejected, not
    silently laundered back into acceptance by reconstructing
    direction_check from the fields it's meant to check against.
    """
    database = setup_db()
    from schema import GraphTriplet, DirectionCheck, PageExtractionPayload
    from engine import commit_page_data_to_sqlite

    import sqlite3
    conn = database.get_db_connection()
    conn.execute("INSERT OR IGNORE INTO sessions (session_id) VALUES ('bypass_test')")
    conn.commit()
    conn.close()

    backwards = GraphTriplet(
        direction_check=DirectionCheck(stated_subject="API_GATEWAY", stated_verb="connects_to", stated_object="SESSION_MIDDLEWARE"),
        source_entity="SESSION_MIDDLEWARE", source_type="SERVICE", relationship="connects_to",
        target_entity="API_GATEWAY", target_type="SERVICE",
        citation_quote="api-gateway returned 502s",
    )
    payload = PageExtractionPayload(
        extracted_triplets=[backwards],
        unresolved_variables_mutations={},
        is_chunk_completely_exhausted=True,
    )
    committed = commit_page_data_to_sqlite(
        session_id="bypass_test", agent_id="test",
        raw_chunk="api-gateway returned 502s", extraction_data=payload,
    )
    check(
        "genuinely backwards fact rejected, not committed (bypass removed)",
        committed == 0,
        f"committed={committed}",
    )


def test_query_aware_retrieval():
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
        "root cause lost under old pure-relevance ranking (confirms the bug existed)",
        "SESSION_MIDDLEWARE" not in without_query,
    )
    check(
        "root cause found once query names something in its neighborhood",
        "SESSION_MIDDLEWARE" in with_query and "AUTH_SERVICE" in with_query,
    )


def test_answer_grounding_gate():
    database = setup_db()
    import inference
    from scratchpad_agent import ScratchpadPoweredLLM

    fake_answers = []

    def fake_call_llm(self, messages, tools=None, **kwargs):
        return {"choices": [{"message": {"role": "assistant", "content": fake_answers[-1]}}]}

    ScratchpadPoweredLLM._call_llm = fake_call_llm
    ScratchpadPoweredLLM._generate_initial_plan = lambda self: None
    ScratchpadPoweredLLM._absorb_response = lambda self, response, messages: None

    agent = ScratchpadPoweredLLM(goal="test", session_id="ground_test", backend="lmstudio", auto_plan=False)

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
        "grounding gate flags an invented entity not in the graph",
        r1["_scratchpad_grounding"]["grounded"] is False
        and "BILLING_QUEUE_OVERFLOW" in r1["_scratchpad_grounding"]["unknown_entities"],
    )

    fake_answers.append("AUTH_SERVICE connects to POSTGRES_PRIMARY, which caused the failure.")
    r2 = agent.call([{"role": "user", "content": "why did it fail?"}])
    check(
        "grounding gate does not flag an answer using only known entities",
        r2["_scratchpad_grounding"]["grounded"] is True,
    )


if __name__ == "__main__":
    test_narrative_direction_check_structurally_impossible()
    test_direction_check_fuzzy_matching_still_works()
    test_auto_repair_bypass_removed_end_to_end()
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
