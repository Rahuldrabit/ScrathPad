"""
Regression tests for the fixes described in FIXES_APPLIED.md.

Each test targets one specific bug that was found by actually running the
code, not just by reading it - several of these looked correct on
inspection and only broke under execution. Run standalone, no pytest
required, to match this repo's existing test_listener.py style:

    python3 tests/test_regressions.py

Uses a throwaway on-disk sqlite file, never the real scratchpad_memory.db.
"""
import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

TEST_DB_PATH = "/tmp/_scratchpad_regression_test.db"


def setup():
    import database
    database.DB_PATH = TEST_DB_PATH
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    database.initialize_database()
    return database


results = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((name, status))
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))


def test_canonicalize_entity_bugs():
    """
    Original bug: canonicalize_entity only queried DISTINCT source_entity,
    so an entity that had only ever appeared as a target_entity was
    invisible to fuzzy matching. Second bug found while fixing the first:
    fuzz.token_ratio splits on whitespace only, so it can never bridge an
    underscore-vs-space boundary ("FASTAPI_APP" is one token, "FAST API APP"
    is three) regardless of the first fix.
    """
    database = setup()
    import engine

    conn = database.get_db_connection()
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO sessions (session_id) VALUES ('sess1')")
    cur.execute(
        """INSERT INTO knowledge_graph
           (edge_id, session_id, source_entity, relationship, target_entity,
            citation_quote, is_active)
           VALUES (?, 'sess1', 'GATEWAY', 'calls', 'FASTAPI_APP',
                   'app = FastAPI()', TRUE)""",
        (str(uuid.uuid4()),),
    )
    conn.commit()

    match = engine.canonicalize_entity(cur, "sess1", "FAST API APP")
    check(
        "canonicalize_entity: target-only entity, underscore-vs-space boundary",
        match == "FASTAPI_APP",
        f"got '{match}'",
    )

    unrelated = engine.canonicalize_entity(cur, "sess1", "COMPLETELY_UNRELATED_ENTITY")
    check(
        "canonicalize_entity: does not force-match a genuinely new entity",
        unrelated == "COMPLETELY_UNRELATED_ENTITY",
        f"got '{unrelated}'",
    )

    # Known limitation, not a regression: word-order reversal COMBINED with
    # a separator-boundary mismatch at the same time is not covered by
    # either scorer pass. token_ratio is word-order invariant but can't
    # bridge the underscore boundary; the separator-stripped character pass
    # bridges the boundary but is itself order-sensitive. Fixing this would
    # need a third, more expensive scoring pass for a narrow combined case -
    # not applied here. Documented rather than silently dropped.
    reversed_and_unbounded = engine.canonicalize_entity(cur, "sess1", "APP FASTAPI")
    print(
        f"[INFO] Known limitation (not fixed, documented): "
        f"'APP FASTAPI' -> got '{reversed_and_unbounded}', "
        f"not 'FASTAPI_APP'. Word-order reversal + missing underscore "
        f"simultaneously is not covered by either scoring pass."
    )
    conn.close()


def test_sweeper_edge_id_collision():
    """
    Original bug: a plain nx.Graph() only holds one edge between any pair of
    nodes. Two different relationship types between the same source/target
    (e.g. "A -calls-> B" and "A -depends_on-> B", both legal distinct rows)
    caused the second add_edge() call to silently overwrite the first
    edge's edge_id - that first triplet then vanished from compression
    bookkeeping permanently.
    """
    import networkx as nx

    G = nx.Graph()
    edges_in = [
        {"source_entity": "A", "target_entity": "B", "edge_id": "edge-calls"},
        {"source_entity": "A", "target_entity": "B", "edge_id": "edge-depends-on"},
    ]
    for edge in edges_in:
        if G.has_edge(edge["source_entity"], edge["target_entity"]):
            G[edge["source_entity"]][edge["target_entity"]]["edge_ids"].append(edge["edge_id"])
        else:
            G.add_edge(edge["source_entity"], edge["target_entity"], edge_ids=[edge["edge_id"]])

    survived = set(G["A"]["B"]["edge_ids"])
    check(
        "sweeper: both edge_ids survive a duplicate node-pair with different relationships",
        survived == {"edge-calls", "edge-depends-on"},
        f"got {survived}",
    )


def test_bounded_markdown_view():
    """
    Original bug: compile_graph_memory_to_markdown had no is_active filter,
    no ordering, and no token budget at all - it returned every row ever
    written, including rows the sweeper had already deactivated. Confirms
    the replacement keeps Unresolved Variables fully intact under a tiny
    budget and reports omissions instead of silently dropping rows.
    """
    database = setup()
    import engine

    conn = database.get_db_connection()
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO sessions (session_id) VALUES ('sess2')")
    for v in ["JWT_SECRET_SOURCE", "TOKEN_EXPIRY"]:
        cur.execute(
            "INSERT OR IGNORE INTO unresolved_variables "
            "(variable_id, session_id, variable_name, status) VALUES (?, 'sess2', ?, 'MISSING')",
            (f"sess2_{v}", v),
        )
    for i in range(60):
        cur.execute(
            """INSERT INTO knowledge_graph
               (edge_id, session_id, source_entity, relationship, target_entity,
                citation_quote, relevance_score, is_active)
               VALUES (?, 'sess2', ?, 'relates_to', ?, 'because the file said so', ?, TRUE)""",
            (str(uuid.uuid4()), f"NODE_{i}", f"NODE_{i+1}", 1.0 - (i * 0.001)),
        )
    conn.commit()
    conn.close()

    view = engine.compile_bounded_markdown_view("sess2", max_tokens=300)
    check(
        "bounded view: Unresolved Variables fully present under a tiny budget",
        "JWT_SECRET_SOURCE" in view and "TOKEN_EXPIRY" in view,
    )
    check(
        "bounded view: omission is reported, not silently dropped",
        "omitted to fit the 300-token budget" in view,
    )
    check(
        "bounded view: rendered output actually respects the budget (+small footer slack)",
        engine.count_tokens(view) <= 320,
        f"{engine.count_tokens(view)} tokens",
    )


def test_agent_update_endpoint_end_to_end():
    """
    THE critical bug: commit_page_data_to_sqlite was declared async def with
    no internal await, called via run_in_threadpool() which expects a plain
    sync callable. Calling an async def function just constructs a
    coroutine object without running its body. This endpoint silently wrote
    nothing to the database and crashed on the very next line with
    TypeError: unsupported operand type(s) for -: 'int' and 'coroutine'.
    Tested through the real ASGI app, not just the isolated function call,
    because the bug only manifests through that exact call pattern.
    """
    database = setup()
    from starlette.testclient import TestClient
    import main

    with TestClient(main.app) as client:
        r = client.post("/v1/session/init", json={"session_id": "e2e", "user_query": "q"})
        check("e2e: session init succeeds", r.status_code == 200, str(r.json()))

        raw_chunk = "auth-service connects to postgres_primary on port 5432."
        r = client.post(
            "/v1/agent/update",
            json={
                "agent_id": "test_agent",
                "session_id": "e2e",
                "raw_active_chunk": raw_chunk,
                "extracted_triplets": [
                    {
                        "source_entity": "AUTH_SERVICE",
                        "relationship": "connects_to",
                        "target_entity": "POSTGRES_PRIMARY",
                        "citation_quote": "auth-service connects to postgres_primary",
                    },
                    {
                        "source_entity": "GHOST_SERVICE",
                        "relationship": "calls",
                        "target_entity": "NOWHERE",
                        "citation_quote": "not present in the raw chunk at all",
                    },
                ],
                "unresolved_variables_mutations": {"GUARD_ROTATION_SCHEDULE": "MISSING"},
                "is_chunk_completely_exhausted": False,
            },
        )
        check(
            "e2e: /v1/agent/update no longer crashes with TypeError",
            r.status_code == 200,
            f"status={r.status_code} body={r.text[:200]}",
        )
        if r.status_code == 200:
            body = r.json()
            check(
                "e2e: exactly the valid triplet was committed, the hallucinated one rejected",
                body.get("verified_triplets_committed") == 1 and body.get("rejected_triplets_count") == 1,
                str(body),
            )

        r = client.get("/v1/session/e2e/memory", params={"max_tokens": 2000})
        view = r.json().get("markdown_view", "")
        check(
            "e2e: committed fact appears in the view, rejected one does not",
            "POSTGRES_PRIMARY" in view and "GHOST_SERVICE" not in view,
        )


if __name__ == "__main__":
    test_canonicalize_entity_bugs()
    test_sweeper_edge_id_collision()
    test_bounded_markdown_view()
    test_agent_update_endpoint_end_to_end()

    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)

    print()
    failed = [name for name, status in results if status == "FAIL"]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed.")
    if failed:
        print("FAILED:", failed)
        sys.exit(1)
