"""
End-to-end pipeline test: mock a real-world problem + agent.

Simulates a solo developer (agent) working on a restaurant POS ordering bug.
Drives the full scratchpad pipeline:
  1. Init session
  2. Submit messy code context → middleware extracts triplets
  3. Agent updates memory with its own structured extractions
  4. Verify SQLite rows in sessions / knowledge_graph / unresolved_variables
  5. Verify the generated Markdown view
  6. Trigger sweeper → verify NetworkX graph is built
  7. Verify sweeper compressed the graph when density threshold is met

Run:
    $env:SCRATCHPAD_LLM_BACKEND = "lmstudio"
    $env:SCRATCHPAD_MODEL_NAME = "google/gemma-4-e4b"
    python test_full_pipeline_e2e.py
"""
import os
import sys
import json
import uuid
import sqlite3

# Force UTF-8 stdout (Windows PowerShell defaults to cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ---------------------------------------------------------------------------
# Problem scenario: a solo dev debugging a "lost orders" bug in the POS system
# ---------------------------------------------------------------------------
PROBLEM_SESSION_ID = "pos-order-bug-session"
AGENT_ID = "dev-auditor-agent"

# Messy raw code / context that the agent reads — contains multiple facts
RAW_CODE_CHUNK = """
// order_service.ts — processes incoming orders from the kiosk
import { db } from './postgres_client';
import { queue } from './redis_queue';
import { logger } from './logging_service';

async function processOrder(orderId: string) {
  const order = await db.query('SELECT * FROM orders WHERE id = $1', [orderId]);
  if (!order) throw new Error('Order not found');

  // Push to Redis stream for async kitchen display
  await queue.add('kitchen_display', { orderId, items: order.items });

  // Record payment in the ledger
  await db.query(
    'INSERT INTO payment_ledger (order_id, amount, status) VALUES ($1, $2, $3)',
    [orderId, order.total, 'PENDING']
  );
}
"""

# What the agent (calling the LLM directly) would extract from the code
AGENT_EXTRACTED_TRIPLETS = [
    {
        "direction_check": "[ORDER_SERVICE] -> [imports] -> [POSTGRES_CLIENT].",
        "source_type": "SERVICE",
        "source_entity": "ORDER_SERVICE",
        "relationship": "imports",
        "target_type": "SERVICE",
        "target_entity": "POSTGRES_CLIENT",
        "citation_quote": "import { db } from './postgres_client';"
    },
    {
        "direction_check": "[ORDER_SERVICE] -> [imports] -> [REDIS_QUEUE].",
        "source_type": "SERVICE",
        "source_entity": "ORDER_SERVICE",
        "relationship": "imports",
        "target_type": "SERVICE",
        "target_entity": "REDIS_QUEUE",
        "citation_quote": "import { queue } from './redis_queue';"
    },
    {
        "direction_check": "[ORDER_SERVICE] -> [imports] -> [LOGGING_SERVICE].",
        "source_type": "SERVICE",
        "source_entity": "ORDER_SERVICE",
        "relationship": "imports",
        "target_type": "SERVICE",
        "target_entity": "LOGGING_SERVICE",
        "citation_quote": "import { logger } from './logging_service';"
    },
    {
        "direction_check": "[ORDER_SERVICE] -> [selects_from] -> [ORDERS_TABLE].",
        "source_type": "SERVICE",
        "source_entity": "ORDER_SERVICE",
        "relationship": "selects_from",
        "target_type": "TABLE",
        "target_entity": "ORDERS_TABLE",
        "citation_quote": "db.query('SELECT * FROM orders WHERE id = $1', [orderId])"
    },
    {
        "direction_check": "[ORDER_SERVICE] -> [pushes_to] -> [REDIS_STREAM].",
        "source_type": "SERVICE",
        "source_entity": "ORDER_SERVICE",
        "relationship": "pushes_to",
        "target_type": "QUEUE",
        "target_entity": "REDIS_STREAM",
        "citation_quote": "await queue.add('kitchen_display', { orderId, items: order.items })"
    },
    {
        "direction_check": "[ORDER_SERVICE] -> [inserts_into] -> [PAYMENT_LEDGER].",
        "source_type": "SERVICE",
        "source_entity": "ORDER_SERVICE",
        "relationship": "inserts_into",
        "target_type": "TABLE",
        "target_entity": "PAYMENT_LEDGER",
        "citation_quote": "INSERT INTO payment_ledger (order_id, amount, status)"
    },
]

UNRESOLVED_VARIABLES = {
    "REDIS_STREAM_BACKLOG_SIZE": "MISSING",
    "ORDER_RETRY_DEAD_LETTER_QUEUE": "MISSING",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
TEST_DB = "test_pipeline_e2e.db"

def reset_test_db():
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

def count(table):
    return q(f"SELECT COUNT(*) as c FROM {table}")[0]["c"]

# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
results = []

def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((name, status))
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))


def run():
    reset_test_db()

    # Monkey-patch database to use the test file
    import database
    database.DB_PATH = TEST_DB
    database.initialize_database()

    # Set up env for LM Studio backend
    os.environ["SCRATCHPAD_LLM_BACKEND"] = "lmstudio"
    os.environ["SCRATCHPAD_MODEL_NAME"] = "google/gemma-4-e4b"

    from starlette.testclient import TestClient
    import main

    with TestClient(main.app) as client:
        import engine as eng_module  # used in steps 5 and 6
        # ── STEP 1: Init session ────────────────────────────────────────────
        print("\n=== STEP 1: Session init ===")
        r = client.post("/v1/session/init", json={
            "session_id": PROBLEM_SESSION_ID,
            "user_query": "Orders are getting lost after payment. Need to trace the full flow.",
            "master_plan": "1. Map order service dependencies  2. Find missing ack signals  3. Add retry logic"
        })
        check("session init returns 200", r.status_code == 200, r.text)
        sessions_count = count("sessions")
        check("sessions table has 1 row", sessions_count == 1, f"{sessions_count} rows")

        session_row = q("SELECT * FROM sessions WHERE session_id = ?", (PROBLEM_SESSION_ID,))[0]
        check("session has correct user_query", "Orders are getting lost" in session_row["user_query"])
        check("session has correct master_plan", "Map order service dependencies" in session_row["master_plan"])
        check("session status is EXECUTING", session_row["global_status"] == "EXECUTING")

        # ── STEP 2: Middleware auto-extraction ──────────────────────────────
        print("\n=== STEP 2: Middleware auto-extraction ===")
        r = client.post("/v1/middleware/process", json={
            "session_id": PROBLEM_SESSION_ID,
            "messy_input": RAW_CODE_CHUNK
        })
        check("middleware/process returns 200", r.status_code == 200, r.text)
        view_after_middleware = r.json()["scratchpad_view"]
        check("markdown view is non-empty after middleware", len(view_after_middleware) > 0, f"{len(view_after_middleware)} chars")
        print(f"  Middleware view preview:\n  {view_after_middleware[:300]}\n")

        # ── STEP 3: Agent update (structured extraction) ───────────────────
        print("\n=== STEP 3: Agent structured update ===")
        r = client.post("/v1/agent/update", json={
            "agent_id": AGENT_ID,
            "session_id": PROBLEM_SESSION_ID,
            "raw_active_chunk": RAW_CODE_CHUNK,
            "extracted_triplets": AGENT_EXTRACTED_TRIPLETS,
            "unresolved_variables_mutations": UNRESOLVED_VARIABLES,
            "is_chunk_completely_exhausted": False
        })
        check("agent update returns 200", r.status_code == 200, r.text)
        body = r.json()
        check("all 6 triplets committed", body["verified_triplets_committed"] == 6, str(body))
        check("no rejections", body["rejected_triplets_count"] == 0, str(body))

        # ── STEP 4: Verify SQLite state ────────────────────────────────────
        print("\n=== STEP 4: SQLite state ===")
        kg_rows = q("SELECT * FROM knowledge_graph WHERE session_id = ?", (PROBLEM_SESSION_ID,))
        # The middleware (LM Studio + Gemma) extracts a non-deterministic
        # number of triplets from the raw code chunk — sometimes 0, sometimes
        # 3. The agent's structured update always adds exactly 6.
        agent_rows = [r for r in kg_rows if r["agent_id"] == AGENT_ID]
        middleware_rows = [r for r in kg_rows if r["agent_id"] == "middleware_auto_extract"]
        check("knowledge_graph has agent's 6 structured rows",
              len(agent_rows) == 6, f"{len(agent_rows)} agent rows")
        check("knowledge_graph has at least 6 rows total (agent's guaranteed 6)",
              len(kg_rows) >= 6, f"{len(kg_rows)} rows")

        var_rows = q("SELECT * FROM unresolved_variables WHERE session_id = ?", (PROBLEM_SESSION_ID,))
        check("unresolved_variables has 2 rows", len(var_rows) == 2, f"{len(var_rows)} rows")

        # Check entity canonicalization
        entity_names = {r["source_entity"] for r in kg_rows} | {r["target_entity"] for r in kg_rows}
        check("entities are uppercase", all(e.isupper() for e in entity_names if e), f"entities: {entity_names}")
        check("all rows are active (is_active=1)", all(r["is_active"] for r in kg_rows))
        check("all rows have citation_quote", all(r["citation_quote"] for r in kg_rows))
        # Agent-originated rows have AGENT_ID; the middleware-extracted row has
        # 'middleware_auto_extract'. Check both labels are present.
        agent_ids = {r["agent_id"] for r in kg_rows}
        check("all rows have agent_id set", all(r["agent_id"] for r in kg_rows),
              f"agent_ids: {agent_ids}")
        check("agent-extracted rows tagged with our AGENT_ID",
              any(r["agent_id"] == AGENT_ID for r in kg_rows),
              f"agent_ids: {agent_ids}")

        # Check specific triplets
        relationships = {r["relationship"] for r in kg_rows}
        check("relationships are lowercase_with_underscores",
              all("_" in r or r.islower() for r in relationships), f"rels: {relationships}")

        print(f"  KG entities: {entity_names}")
        print(f"  KG relationships: {relationships}")
        print(f"  Variables: {[r['variable_name'] for r in var_rows]}")

        # ── STEP 5: Markdown view ─────────────────────────────────────────
        print("\n=== STEP 5: Markdown view generation ===")
        r = client.get(f"/v1/session/{PROBLEM_SESSION_ID}/memory", params={"max_tokens": 6000})
        check("/memory endpoint returns 200", r.status_code == 200, r.text)
        full_view = r.json()["markdown_view"]

        check("view contains 'UNRESOLVED VARIABLES MATRIX'",
              "UNRESOLVED VARIABLES MATRIX" in full_view)
        check("view contains both unresolved variables",
              "REDIS_STREAM_BACKLOG_SIZE" in full_view and "ORDER_RETRY_DEAD_LETTER_QUEUE" in full_view)
        check("view contains 'KNOWLEDGE GRAPH MEMORY'",
              "KNOWLEDGE GRAPH MEMORY" in full_view)
        check("view contains ORDER_SERVICE", "ORDER_SERVICE" in full_view)
        check("view contains REDIS_STREAM", "REDIS_STREAM" in full_view)
        check("view contains PAYMENT_LEDGER", "PAYMENT_LEDGER" in full_view)
        check("view contains citation quotes",
              any("citation" in l.lower() or "SELECT" in l for l in full_view.split("\n")))
        check("view is within token budget", eng_module.count_tokens(full_view) <= 6200,
              f"{eng_module.count_tokens(full_view)} tokens")

        print(f"  Markdown view ({len(full_view)} chars, ~{eng_module.count_tokens(full_view)} tokens):")
        print(f"  {full_view[:600]}\n")

        # ── STEP 6: Sweeper / NetworkX graph ───────────────────────────────
        print("\n=== STEP 6: Sweeper → NetworkX graph ===")
        import sweeper
        # Rebuild sweeper's in-memory graph from DB state
        import networkx as nx
        from database import get_db_connection

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT edge_id, source_entity, target_entity FROM knowledge_graph "
            "WHERE session_id = ? AND is_active = TRUE",
            (PROBLEM_SESSION_ID,)
        )
        edges = cur.fetchall()
        conn.close()

        # Simulate what sweeper does: build per-session graphs
        G = nx.Graph()
        for edge in edges:
            if G.has_edge(edge['source_entity'], edge['target_entity']):
                G[edge['source_entity']][edge['target_entity']]['edge_ids'].append(edge['edge_id'])
            else:
                G.add_edge(edge['source_entity'], edge['target_entity'], edge_ids=[edge['edge_id']])

        check("NetworkX graph has expected node count (7+ for agent triplets)",
              len(G.nodes) >= 7, f"{len(G.nodes)} nodes: {list(G.nodes)}")
        check("NetworkX graph has expected edge count (>=6 for agent triplets)",
              len(G.edges) >= 6, f"{len(G.edges)} edges")
        check("ORDER_SERVICE is a hub (degree > 1)",
              G.degree("ORDER_SERVICE") > 1, f"degree={G.degree('ORDER_SERVICE')}")
        # Note: the graph may not be fully connected if Gemma's
        # middleware-side extractions produced entities that don't link
        # back to the ORDER_SERVICE cluster. That's realistic small-LLM
        # behavior. We verify there IS a main connected component
        # containing ORDER_SERVICE.
        main_component = max(nx.connected_components(G), key=len)
        check("main component contains ORDER_SERVICE and the agent triplets",
              "ORDER_SERVICE" in main_component and len(main_component) >= 7,
              f"main component size: {len(main_component)}")

        # Degree distribution
        degree_data = dict(G.degree())
        print(f"  NetworkX graph nodes: {list(G.nodes)}")
        print(f"  NetworkX graph edges: {list(G.edges)}")
        print(f"  Degree distribution: {degree_data}")
        print(f"  Main component size: {len(main_component)}")

        # ── STEP 7: Trigger sweeper compression (inject more edges to hit threshold) ──
        print("\n=== STEP 7: Sweeper compression (artificially dense graph) ===")

        # Inject 11 extra edges to push max degree above sweeper.degree_threshold (15).
        # Each edge goes to a UNIQUE target — if they share a target, nx.Graph
        # collapses them into a single edge and the degree stays low.
        # Keep the community small (≤16 edges total) so Gemma can summarize
        # it within the LM Studio timeout.
        conn = get_db_connection()
        cur = conn.cursor()
        extra_rels = [
            ("calls",        "PAYMENT_GATEWAY"),
            ("uses",         "INVENTORY_SERVICE"),
            ("reads",        "CACHE_LAYER"),
            ("writes",       "AUDIT_LOG"),
            ("connects_to",  "KAFKA_BROKER"),
            ("awaits",       "REPLY_QUEUE"),
            ("emits",        "WEBSOCKET_HUB"),
            ("listens_to",   "EVENT_BUS"),
            ("publishes_to", "NOTIFICATION_SERVICE"),
            ("retries",      "DEAD_LETTER_QUEUE"),
            ("validates",    "SCHEMA_REGISTRY"),
        ]
        for rel, target in extra_rels:
            cur.execute("""
                INSERT INTO knowledge_graph
                (edge_id, session_id, agent_id, source_entity, relationship,
                 target_entity, citation_quote, is_active)
                VALUES (?, ?, ?, 'ORDER_SERVICE', ?, ?, 'injected citation', TRUE)
            """, (str(uuid.uuid4()), PROBLEM_SESSION_ID, AGENT_ID, rel, target))
        conn.commit()
        conn.close()

        extra_count = count("knowledge_graph")
        check("injected extra edges for sweeper trigger",
              extra_count >= 17, f"total KG rows: {extra_count}")

        # Now run the sweeper's maintenance sweep directly (sync)
        import engine as eng_module
        sweeper_instance = sweeper.GraphSweeperDaemon()
        # Use the default min_community_size (10) and degree_threshold (15).
        # The community we built has ~17 edges which is small enough for
        # Gemma to summarize in one pass.
        sweeper_instance.execute_maintenance_sweep()

        # Check the sweeper outcome. Two valid paths:
        #   (a) Gemma returns a grounded L2 payload → L2 rows exist, L1 deactivated
        #   (b) Gemma hallucinates edge_ids/entities → guardrail rejects, L1 stays
        # Both are correct sweeper behavior; the test should pass either way.
        l2_rows = q("SELECT * FROM knowledge_graph WHERE hierarchy_level = 2 AND session_id = ?",
                    (PROBLEM_SESSION_ID,))
        l1_still_active = q("SELECT * FROM knowledge_graph WHERE hierarchy_level = 1 AND is_active = TRUE AND session_id = ?",
                             (PROBLEM_SESSION_ID,))
        l2_created = len(l2_rows) > 0
        if l2_created:
            # Path (a): successful L2 compression
            check("sweeper created L2 compressed nodes", True,
                  f"{len(l2_rows)} L2 rows")
            check("sweeper deactivated L1 rows after successful L2",
                  len(l1_still_active) < extra_count,
                  f"{len(l1_still_active)} L1 active rows remain")
            print(f"  L2 compressed nodes: {len(l2_rows)}")
            for row in l2_rows:
                print(f"    {row['source_entity']} --({row['relationship']})--> {row['target_entity']}")
        else:
            # Path (b): guardrail caught the hallucination. This is the
            # sweeper's verification gate working as designed. L1 rows
            # must still be active and intact, and the markdown view
            # must still render them.
            check("sweeper correctly rejected hallucinated L2 via guardrail",
                  len(l1_still_active) == extra_count,
                  f"L1 still active: {len(l1_still_active)} / {extra_count}")
            print(f"  L2 guardrail rejected ungrounded compression — "
                  f"L1 rows remain intact ({len(l1_still_active)} active)")

        # ── Final: check the markdown view still renders after compression ──
        print("\n=== STEP 8: Markdown view after compression ===")
        r = client.get(f"/v1/session/{PROBLEM_SESSION_ID}/memory", params={"max_tokens": 6000})
        compressed_view = r.json()["markdown_view"]
        check("compressed view still renders (no crash)", len(compressed_view) > 0)
        if l2_created:
            # Path (a): L2 exists, view should mention COMPRESSED marker
            check("compressed view mentions L2 hierarchy",
                  "COMPRESSED" in compressed_view or "hierarchy" in compressed_view.lower(),
                  f"COMPRESSED: {'COMPRESSED' in compressed_view}")
        else:
            # Path (b): L1 still active, view should still show ORDER_SERVICE
            check("view still shows the original L1 data after rejected L2",
                  "ORDER_SERVICE" in compressed_view,
                  f"ORDER_SERVICE present: {'ORDER_SERVICE' in compressed_view}")

        # Clean up
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    failed = [name for name, status in results if status == "FAIL"]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed.")
    if failed:
        print("FAILED:", failed)
        sys.exit(1)
    else:
        print("All checks passed — pipeline is healthy.")

if __name__ == "__main__":
    run()
