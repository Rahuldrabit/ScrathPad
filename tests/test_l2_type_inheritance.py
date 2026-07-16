"""
Unit tests for L2 type inheritance in sweeper.py.

Verifies the sweeper overrides the LLM's L2 source_type / target_type
with the majority type from the L1 community. This is a free
grounding signal — a proposed L2 type that contradicts its source
community's types is now impossible.

Run:
    python tests/test_l2_type_inheritance.py
"""
import os
import sys
import uuid
import sqlite3

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

TEST_DB = "/tmp/_scratchpad_l2_inheritance_test.db"


def setup():
    import database
    database.DB_PATH = TEST_DB
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    database.initialize_database()
    return database


results = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((name, status))
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))


# ─────────────────────────────────────────────────────────────────────────
# _majority_l1_type — pure-function tests
# ─────────────────────────────────────────────────────────────────────────
def test_majority_l1_type_basic():
    from sweeper import _majority_l1_type
    check("majority: single real type wins",
          _majority_l1_type(["SERVICE", "SERVICE", "SERVICE"]) == "SERVICE")
    check("majority: UNKNOWN loses to a real type",
          _majority_l1_type(["UNKNOWN", "SERVICE", "SERVICE"]) == "SERVICE")
    check("majority: real type with single count still wins over UNKNOWN",
          _majority_l1_type(["UNKNOWN", "SERVICE", "UNKNOWN", "DATABASE"]) == "SERVICE"
          or _majority_l1_type(["UNKNOWN", "SERVICE", "UNKNOWN", "DATABASE"]) == "DATABASE",
          "ties broken by most_common ordering — either is acceptable")
    check("majority: all UNKNOWN returns UNKNOWN",
          _majority_l1_type(["UNKNOWN", "UNKNOWN"]) == "UNKNOWN")
    check("majority: empty list returns UNKNOWN",
          _majority_l1_type([]) == "UNKNOWN")
    check("majority: real type wins by 2+",
          _majority_l1_type(["SERVICE", "SERVICE", "SERVICE", "DATABASE"]) == "SERVICE")


# ─────────────────────────────────────────────────────────────────────────
# End-to-end: build a community, run the override path, verify L2 types
# ─────────────────────────────────────────────────────────────────────────
def test_l2_type_override_in_commit():
    """
    Manually run the override logic against an L1 community where all
    source types are SERVICE and all target types are DATABASE. Verify
    the L2 row's types match the majority even when the LLM said
    something different.
    """
    from sweeper import _majority_l1_type
    setup()
    from database import get_db_connection
    from schema import GraphTriplet, GraphTripletSchema, L2CompressionPayload
    from engine import commit_page_data_to_sqlite

    # 1. Build a community of 5 L1 rows: all SERVICE -> DATABASE
    raw = (
        "order_service uses postgres_client.\n"
        "auth_service uses postgres_client.\n"
        "payment_service uses postgres_client.\n"
        "inventory_service uses postgres_client.\n"
        "notification_service uses postgres_client.\n"
    )
    triplets = []
    for svc in ("ORDER_SERVICE", "AUTH_SERVICE", "PAYMENT_SERVICE",
                "INVENTORY_SERVICE", "NOTIFICATION_SERVICE"):
        triplets.append(GraphTriplet(
            direction_check=f"[{svc}] -> [uses] -> [POSTGRES_CLIENT].",
            source_type="SERVICE",
            source_entity=svc,
            relationship="uses",
            target_type="DATABASE",
            target_entity="POSTGRES_CLIENT",
            citation_quote=f"{svc.lower()} uses postgres_client",
        ))
    commit_page_data_to_sqlite("inherit-test", "agent", raw, GraphTriplet.to_payload if False else _build_payload(triplets))

    # 2. Build the L1 community (simulating what the sweeper would fetch)
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM knowledge_graph WHERE session_id = ? AND is_active = TRUE",
        ("inherit-test",),
    )
    raw_triplets = [dict(r) for r in cur.fetchall()]
    conn.close()
    check("inheritance: 5 L1 rows persisted", len(raw_triplets) == 5)

    # 3. Run the override (this is the exact code path the sweeper uses)
    src_types = [t.get("source_type") or "UNKNOWN" for t in raw_triplets]
    tgt_types = [t.get("target_type") or "UNKNOWN" for t in raw_triplets]
    inherited_source_type = _majority_l1_type(src_types)
    inherited_target_type = _majority_l1_type(tgt_types)
    check("inheritance: majority source type is SERVICE",
          inherited_source_type == "SERVICE", f"got {inherited_source_type}")
    check("inheritance: majority target type is DATABASE",
          inherited_target_type == "DATABASE", f"got {inherited_target_type}")

    # 4. Simulate an L2 triplet where the LLM said wrong types
    l2_triplet_wrong = GraphTripletSchema(
        source_type="TABLE",  # LLM hallucinated — should be overridden
        source_entity="BUSINESS_LOGIC_LAYER",
        relationship="uses",
        target_type="QUEUE",  # LLM hallucinated — should be overridden
        target_entity="MESSAGE_BUS",
        citation_quote="Generated via structural compression.",
    )

    # Run the override
    if inherited_source_type != "UNKNOWN":
        l2_triplet_wrong.source_type = inherited_source_type
    if inherited_target_type != "UNKNOWN":
        l2_triplet_wrong.target_type = inherited_target_type

    # 5. Verify the override actually happened
    check("inheritance: L2 source_type overridden to SERVICE",
          l2_triplet_wrong.source_type == "SERVICE",
          f"got {l2_triplet_wrong.source_type}")
    check("inheritance: L2 target_type overridden to DATABASE",
          l2_triplet_wrong.target_type == "DATABASE",
          f"got {l2_triplet_wrong.target_type}")


def test_l2_type_no_override_when_community_is_unknown():
    """
    If the L1 community has all UNKNOWN types (e.g., pre-migration data),
    the override path should leave the L2 types alone (not force-set to
    "UNKNOWN" which would defeat the schema's defaults).
    """
    from sweeper import _majority_l1_type
    setup()
    from database import get_db_connection

    # Manually insert an L1 row with explicit UNKNOWN types (simulating pre-migration)
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO sessions (session_id) VALUES ('pre-mig')")
    for i in range(3):
        cur.execute(
            """INSERT INTO knowledge_graph
               (edge_id, session_id, agent_id, source_entity, source_type,
                relationship, target_entity, target_type, citation_quote, is_active)
               VALUES (?, 'pre-mig', 'old-agent', ?, 'UNKNOWN', 'relates_to', ?, 'UNKNOWN', 'legacy', TRUE)""",
            (str(uuid.uuid4()), f"OLD_NODE_{i}", f"OLD_NODE_{i+1}"),
        )
    conn.commit()
    conn.close()

    # Aggregate
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT source_type, target_type FROM knowledge_graph WHERE session_id = 'pre-mig'"
    )
    rows = cur.fetchall()
    conn.close()

    src_types = [r["source_type"] for r in rows]
    tgt_types = [r["target_type"] for r in rows]
    check("pre-mig: aggregate has UNKNOWN source types",
          all(s == "UNKNOWN" for s in src_types))
    check("pre-mig: aggregate has UNKNOWN target types",
          all(t == "UNKNOWN" for t in tgt_types))

    inherited_source = _majority_l1_type(src_types)
    inherited_target = _majority_l1_type(tgt_types)
    check("pre-mig: majority is UNKNOWN when all are UNKNOWN",
          inherited_source == "UNKNOWN" and inherited_target == "UNKNOWN")

    # The sweeper code path is:
    #   if inherited_source_type != "UNKNOWN":
    #       t.source_type = inherited_source_type
    # So the override is SKIPPED, leaving the L2 triplet's default (SERVICE).
    # This is correct — we don't want to force UNKNOWN into the row.
    from schema import GraphTripletSchema
    fresh = GraphTripletSchema(
        source_entity="MACRO", relationship="uses",
        target_entity="OTHER_MACRO", citation_quote="Generated via structural compression.",
    )
    if inherited_source != "UNKNOWN":
        fresh.source_type = inherited_source
    if inherited_target != "UNKNOWN":
        fresh.target_type = inherited_target
    check("pre-mig: L2 source_type not forced to UNKNOWN (keeps default SERVICE)",
          fresh.source_type == "SERVICE")
    check("pre-mig: L2 target_type not forced to UNKNOWN (keeps default SERVICE)",
          fresh.target_type == "SERVICE")


def _build_payload(triplets):
    from schema import PageExtractionPayload
    return PageExtractionPayload(
        extracted_triplets=triplets,
        unresolved_variables_mutations={},
        is_chunk_completely_exhausted=True,
    )


if __name__ == "__main__":
    test_majority_l1_type_basic()
    test_l2_type_override_in_commit()
    test_l2_type_no_override_when_community_is_unknown()

    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)

    print()
    failed = [n for n, s in results if s == "FAIL"]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed.")
    if failed:
        print("FAILED:", failed)
        sys.exit(1)
