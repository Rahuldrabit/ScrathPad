"""
Unit tests for the deterministic verification layers in engine.py.

These tests don't make any LLM calls. They construct GraphTriplet
objects directly and verify the rejection pipeline behavior:
  - Layer 1: closed relationship vocabulary
  - Layer 2: in-schema CoT (direction_check consistency)
  - Layer 3: citation substring match (existing)
  - Layer 4: citation quality score
  - Layer 5: atomicity (no compound conjunctions)
  - Layer 6: contradiction check (FUNCTIONAL_RELATIONSHIPS only)
  - Rejection log gets written for every rejected triplet

Run:
    python tests/test_verification_layers.py
"""
import os
import sys
import uuid
import sqlite3
import textwrap

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

TEST_DB = "/tmp/_scratchpad_verification_test.db"


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


def make_triplet(**overrides):
    """Build a valid GraphTriplet with sane defaults."""
    from schema import GraphTriplet
    defaults = {
        "direction_check": "[AUTH_SERVICE] -> [connects_to] -> [POSTGRES_PRIMARY].",
        "source_type": "SERVICE",
        "source_entity": "AUTH_SERVICE",
        "relationship": "connects_to",
        "target_type": "DATABASE",
        "target_entity": "POSTGRES_PRIMARY",
        "citation_quote": "auth-service connects to postgres_primary",
    }
    defaults.update(overrides)
    return GraphTriplet(**defaults)


def make_payload(triplets):
    from schema import PageExtractionPayload
    return PageExtractionPayload(
        extracted_triplets=triplets,
        unresolved_variables_mutations={},
        is_chunk_completely_exhausted=True,
    )


# ─────────────────────────────────────────────────────────────────────────
# Layer 1: closed relationship vocabulary
# ─────────────────────────────────────────────────────────────────────────
def test_layer1_vocabulary():
    from schema import validate_relationship, ALLOWED_RELATIONSHIPS
    check("layer1: imports is in vocabulary", validate_relationship("imports"))
    check("layer1: runs_on_port is in vocabulary", validate_relationship("runs_on_port"))
    check("layer1: empty string rejected", not validate_relationship(""))
    check("layer1: invented verb rejected",
          not validate_relationship("frobnicate_thingamajig"),
          f"ALLOWED count={len(ALLOWED_RELATIONSHIPS)}")
    check("layer1: case-insensitive",
          validate_relationship("CALLS"),
          "CALLS should normalize to calls")
    check("layer1: ALLOWED includes the 3 functional ones",
          {"runs_on_port", "has_primary_ip", "hosted_in_region"}.issubset(ALLOWED_RELATIONSHIPS))


# ─────────────────────────────────────────────────────────────────────────
# Layer 2: in-schema CoT (direction_check)
# ─────────────────────────────────────────────────────────────────────────
def test_layer2_direction_check():
    from schema import validate_direction, parse_direction

    # Valid triplet
    t = make_triplet()
    check("layer2: matching direction passes", validate_direction(t))

    # Source/target swapped
    bad = make_triplet(
        direction_check="[POSTGRES_PRIMARY] -> [connects_to] -> [AUTH_SERVICE].",
    )
    check("layer2: swapped direction rejected", not validate_direction(bad))

    # Malformed direction_check
    bad2 = make_triplet(direction_check="auth-service connects to postgres_primary")
    check("layer2: missing brackets rejected", not validate_direction(bad2))

    # Parser
    parsed = parse_direction("[A] -> [b] -> [C].")
    check("layer2: parse_direction returns triple",
          parsed is not None and parsed == ("A", "b", "C"))


# ─────────────────────────────────────────────────────────────────────────
# Layer 4: citation quality score
# ─────────────────────────────────────────────────────────────────────────
def test_layer4_citation_score():
    from engine import citation_score, CITATION_SCORE_THRESHOLD

    raw = "auth-service connects to postgres_primary on port 5432."

    # Good citations
    good1 = "auth-service connects to postgres_primary"
    s1 = citation_score(good1, raw)
    check(f"layer4: good citation scores >= threshold ({s1:.2f} >= {CITATION_SCORE_THRESHOLD})",
          s1 >= CITATION_SCORE_THRESHOLD)

    # Trivial citations
    trivial1 = "import"
    s2 = citation_score(trivial1, "import x;")
    check(f"layer4: 'import' alone scores low ({s2:.2f})",
          s2 < CITATION_SCORE_THRESHOLD)

    trivial2 = "TODO"
    s3 = citation_score(trivial2, "// TODO: fix this")
    check(f"layer4: 'TODO' alone scores low ({s3:.2f})",
          s3 < CITATION_SCORE_THRESHOLD)

    # Not a substring at all
    s4 = citation_score("nonexistent phrase", raw)
    check("layer4: missing substring scores 0.0", s4 == 0.0)

    # Repeated boilerplate
    s5 = citation_score("//", "//\n//\n//\n//\n//\n//")
    check("layer4: repeated boilerplate is penalized",
          s5 < CITATION_SCORE_THRESHOLD,
          f"got {s5:.2f}")


# ─────────────────────────────────────────────────────────────────────────
# Layer 5: atomicity (compound fact detection)
# ─────────────────────────────────────────────────────────────────────────
def test_layer5_atomicity():
    from engine import COMPOUND_CONJUNCTIONS

    # Compound fact: " and " in citation
    raw = "foo connects to bar and baz writes to qux."
    compound = make_triplet(
        direction_check="[FOO] -> [connects_to] -> [BAR].",
        source_entity="FOO",
        target_entity="BAR",
        citation_quote="foo connects to bar and baz",
    )
    payload = make_payload([compound])
    setup()
    from engine import commit_page_data_to_sqlite
    saved = commit_page_data_to_sqlite("test-session", "test-agent", raw, payload)
    check("layer5: compound fact (with ' and ') rejected", saved == 0)

    # Verify the rejection log captured it
    conn = sqlite3.connect(TEST_DB)
    cur = conn.cursor()
    cur.execute("SELECT rejection_reason FROM rejected_triplets ORDER BY id DESC LIMIT 1")
    reason = cur.fetchone()[0]
    conn.close()
    check(f"layer5: rejection reason logged as 'compound_fact'",
          reason == "compound_fact", f"got: {reason}")


# ─────────────────────────────────────────────────────────────────────────
# Layer 6: contradiction check (FUNCTIONAL_RELATIONSHIPS only)
# ─────────────────────────────────────────────────────────────────────────
def test_layer6_contradiction():
    setup()
    from engine import commit_page_data_to_sqlite

    raw = "auth-service runs_on_port 5432."

    # First triplet: A runs_on_port 5432
    t1 = make_triplet(
        direction_check="[AUTH_SERVICE] -> [runs_on_port] -> [PORT_5432].",
        source_entity="AUTH_SERVICE",
        target_entity="PORT_5432",
        relationship="runs_on_port",
        citation_quote="auth-service runs_on_port 5432",
    )
    saved1 = commit_page_data_to_sqlite("sess1", "agent", raw, make_payload([t1]))
    check("layer6: first runs_on_port triplet accepted", saved1 == 1)

    # Second triplet: A runs_on_port 6000 (contradicts!)
    t2 = make_triplet(
        direction_check="[AUTH_SERVICE] -> [runs_on_port] -> [PORT_6000].",
        source_entity="AUTH_SERVICE",
        target_entity="PORT_6000",
        relationship="runs_on_port",
        citation_quote="(fabricated: A runs_on_port 6000) auth-service runs_on_port 5432",
    )
    saved2 = commit_page_data_to_sqlite("sess1", "agent", raw, make_payload([t2]))
    check("layer6: contradictory second port rejected", saved2 == 0)

    # Same (source, relationship, target) is NOT a contradiction
    t3 = make_triplet(
        direction_check="[AUTH_SERVICE] -> [runs_on_port] -> [PORT_5432].",
        source_entity="AUTH_SERVICE",
        target_entity="PORT_5432",
        relationship="runs_on_port",
        citation_quote="auth-service runs_on_port 5432",
    )
    saved3 = commit_page_data_to_sqlite("sess1", "agent", raw, make_payload([t3]))
    check("layer6: re-asserting same (source, rel, target) accepted", saved3 == 1)

    # Non-functional relationships are NOT checked for contradiction
    # A imports B + A imports C is fine
    raw2 = "service_a imports b_module.\nservice_a imports c_module."
    t4 = make_triplet(
        direction_check="[SERVICE_A] -> [imports] -> [B_MODULE].",
        source_entity="SERVICE_A",
        target_entity="B_MODULE",
        relationship="imports",
        citation_quote="service_a imports b_module",
    )
    t5 = make_triplet(
        direction_check="[SERVICE_A] -> [imports] -> [C_MODULE].",
        source_entity="SERVICE_A",
        target_entity="C_MODULE",
        relationship="imports",
        citation_quote="service_a imports c_module",
    )
    saved4 = commit_page_data_to_sqlite("sess2", "agent", raw2, make_payload([t4, t5]))
    check("layer6: 'imports' is non-functional, two different targets allowed",
          saved4 == 2)


# ─────────────────────────────────────────────────────────────────────────
# End-to-end commit pipeline
# ─────────────────────────────────────────────────────────────────────────
def test_full_pipeline():
    setup()
    from engine import commit_page_data_to_sqlite

    raw = textwrap.dedent("""
        order_service imports postgres_client.
        order_service uses queue.add for kitchen_display.
        order_service runs_on_port 8080.
        TODO: refactor this.
    """).strip()

    payload = make_payload([
        # Valid
        make_triplet(
            direction_check="[ORDER_SERVICE] -> [imports] -> [POSTGRES_CLIENT].",
            source_entity="ORDER_SERVICE",
            target_entity="POSTGRES_CLIENT",
            relationship="imports",
            citation_quote="order_service imports postgres_client",
        ),
        # Bad: invented relationship
        make_triplet(
            direction_check="[ORDER_SERVICE] -> [frobnicate] -> [X].",
            source_entity="ORDER_SERVICE",
            target_entity="X",
            relationship="frobnicate",
            citation_quote="order_service frobnicate x.",
        ),
        # Bad: citation not in raw
        make_triplet(
            direction_check="[ORDER_SERVICE] -> [imports] -> [NONEXISTENT].",
            source_entity="ORDER_SERVICE",
            target_entity="NONEXISTENT",
            relationship="imports",
            citation_quote="not in the raw text at all",
        ),
        # Bad: trivial citation (passes substring because "TODO" is in raw,
        # but fails the quality score)
        make_triplet(
            direction_check="[ORDER_SERVICE] -> [uses] -> [TODO].",
            source_entity="ORDER_SERVICE",
            target_entity="TODO",
            relationship="uses",
            citation_quote="TODO",
        ),
        # Bad: direction_check inconsistent
        make_triplet(
            direction_check="[WRONG] -> [imports] -> [CLIENT].",
            source_entity="ORDER_SERVICE",
            target_entity="CLIENT",
            relationship="imports",
            citation_quote="order_service imports client.",
        ),
    ])

    saved = commit_page_data_to_sqlite("e2e", "agent", raw, payload)
    check("pipeline: 1 valid + 4 rejected = 1 saved", saved == 1, f"saved={saved}")

    # Verify rejection log has 4 entries
    conn = sqlite3.connect(TEST_DB)
    cur = conn.cursor()
    cur.execute("SELECT rejection_reason, COUNT(*) FROM rejected_triplets GROUP BY rejection_reason")
    reasons = dict(cur.fetchall())
    conn.close()
    check("pipeline: rejection log has 4 entries",
          sum(reasons.values()) == 4,
          f"reasons: {reasons}")
    check("pipeline: 'bad_relationship_vocab' logged for invented verb",
          reasons.get("bad_relationship_vocab", 0) == 1)
    check("pipeline: 'citation_mismatch' logged for hallucinated citation",
          reasons.get("citation_mismatch", 0) == 1)
    check("pipeline: 'low_citation_score' logged for trivial citation",
          reasons.get("low_citation_score", 0) == 1)
    check("pipeline: 'direction_check_mismatch' logged for bad CoT",
          reasons.get("direction_check_mismatch", 0) == 1)


# ─────────────────────────────────────────────────────────────────────────
# Database migration
# ─────────────────────────────────────────────────────────────────────────
def test_migration():
    """Migration is idempotent and adds type columns."""
    setup()
    conn = sqlite3.connect(TEST_DB)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(knowledge_graph)")
    cols = {row[1] for row in cur.fetchall()}
    conn.close()
    check("migration: source_type column added", "source_type" in cols)
    check("migration: target_type column added", "target_type" in cols)

    # Re-initialize should be a no-op
    import database
    database.initialize_database()
    conn = sqlite3.connect(TEST_DB)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(knowledge_graph)")
    cols2 = {row[1] for row in cur.fetchall()}
    conn.close()
    check("migration: re-initialize is idempotent", cols == cols2)


# ─────────────────────────────────────────────────────────────────────────
# Markdown view includes type tags
# ─────────────────────────────────────────────────────────────────────────
def test_view_includes_types():
    setup()
    from engine import commit_page_data_to_sqlite, compile_bounded_markdown_view

    raw = "auth_service connects_to postgres_primary."
    commit_page_data_to_sqlite("view-test", "agent", raw, make_payload([
        make_triplet(
            direction_check="[AUTH_SERVICE] -> [connects_to] -> [POSTGRES_PRIMARY].",
            source_entity="AUTH_SERVICE",
            target_entity="POSTGRES_PRIMARY",
            relationship="connects_to",
            citation_quote="auth_service connects_to postgres_primary",
        )
    ]))
    view = compile_bounded_markdown_view("view-test", max_tokens=2000)
    check("view: includes source type tag <service>",
          "<service>" in view.lower(),
          f"view preview: {view[:200]}")
    check("view: includes target type tag <database>",
          "<database>" in view.lower())


if __name__ == "__main__":
    test_layer1_vocabulary()
    test_layer2_direction_check()
    test_layer4_citation_score()
    test_layer5_atomicity()
    test_layer6_contradiction()
    test_full_pipeline()
    test_migration()
    test_view_includes_types()

    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)

    print()
    failed = [n for n, s in results if s == "FAIL"]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed.")
    if failed:
        print("FAILED:", failed)
        sys.exit(1)
