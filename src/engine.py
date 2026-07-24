import uuid
import asyncio
import yaml
from pathlib import Path
from database import get_db_connection
from schema import (
    PageExtractionPayload,
    ALLOWED_RELATIONSHIPS,
    FUNCTIONAL_RELATIONSHIPS,
    validate_relationship,
    validate_direction,
    validate_entity_shape,
)
from rapidfuzz import fuzz, process as rf_process
from telemetry import telemetry_manager

# --- Token counting: real tokenizer when available, heuristic fallback ---
#
# tiktoken's encoding files are NOT bundled in the pip package - "cl100k_base"
# fetches its BPE ranks from a Microsoft blob URL the first time it's used,
# then caches the result locally. On air-gapped or network-restricted
# hardware (the actual target environment for this project) that fetch can
# fail, and importing this module would previously crash the whole app.
# We try once at import time and fall back to a calibrated character
# heuristic if it's unavailable, instead of hard-depending on network access
# just to count tokens.
_ENCODER = None
_TOKENIZER_MODE = "heuristic"

def _load_encoder():
    global _ENCODER, _TOKENIZER_MODE
    try:
        import tiktoken
        _ENCODER = tiktoken.get_encoding("cl100k_base")
        _TOKENIZER_MODE = "tiktoken"
    except Exception as e:
        print(f"[TOKENIZER] tiktoken unavailable ({e}); using heuristic token counting.")
        _TOKENIZER_MODE = "heuristic"

_load_encoder()


def count_tokens(text: str) -> int:
    if _ENCODER is not None:
        return len(_ENCODER.encode(text))
    # ~4 characters per token is the standard rough approximation for
    # English prose. It slightly overestimates for code, which is the safe
    # direction to be wrong in here: it means we truncate a little early
    # rather than silently exceed the real budget.
    return max(1, len(text) // 4)


from text_matching import strip_separators as _strip_separators


def canonicalize_entity(cursor, session_id: str, incoming_entity: str, threshold: int = 85) -> str:
    """
    Collapses synonyms to prevent graph fragmentation.

    BUGFIX 1: the original query only looked at DISTINCT source_entity. Any
    entity that had so far only appeared as a target_entity (e.g. FASTAPI_APP
    in "A -> calls -> FASTAPI_APP") was invisible to the match pool, so a
    later triplet introducing it as a *source* under a slightly different
    name would never find it and would fragment the graph anyway. This now
    unions both columns.

    BUGFIX 2: fuzz.token_ratio splits on whitespace only, so it can never
    bridge an underscore-vs-space boundary difference - "FASTAPI_APP" is one
    token, "FAST API APP" is three, and their token-level similarity scores
    low even though they're the same entity. Since this system's entities
    mix SCREAMING_SNAKE_CASE (from code) with space-separated phrasing (from
    an SLM extracting from prose), we run a second pass with all separators
    stripped and a plain character ratio, and keep whichever scorer is more
    confident. token_ratio stays first because it is word-order invariant
    ("APP FASTAPI" vs "FASTAPI APP"), which the stripped-character pass is
    NOT - it is sensitive to order, so neither pass alone covers both cases.
    """
    cursor.execute(
        """
        SELECT DISTINCT source_entity AS entity
        FROM knowledge_graph WHERE session_id = ? AND is_active = TRUE
        UNION
        SELECT DISTINCT target_entity AS entity
        FROM knowledge_graph WHERE session_id = ? AND is_active = TRUE
        """,
        (session_id, session_id),
    )
    existing_entities = [row[0] for row in cursor.fetchall()]

    normalized_incoming = incoming_entity.strip().upper()

    if not existing_entities:
        return normalized_incoming

    best_match, best_score = None, 0

    token_match = rf_process.extractOne(
        normalized_incoming, existing_entities, scorer=fuzz.token_ratio
    )
    if token_match and token_match[1] > best_score:
        best_match, best_score = token_match[0], token_match[1]

    stripped_incoming = _strip_separators(normalized_incoming)
    stripped_lookup = {_strip_separators(e): e for e in existing_entities}
    char_match = rf_process.extractOne(
        stripped_incoming, list(stripped_lookup.keys()), scorer=fuzz.ratio
    )
    if char_match and char_match[1] > best_score:
        best_match, best_score = stripped_lookup[char_match[0]], char_match[1]

    if best_match and best_score >= threshold:
        return best_match

    return normalized_incoming

def verify_citation(raw_chunk: str, citation: str) -> bool:
    """Anti-hallucination guardrail. Ensures exact matches only."""
    if not citation or citation.strip() == "":
        return False
    return citation.strip() in raw_chunk


# Layer 4: citation quality scoring.
# Beyond "is it a substring?" — also how long, how unique, how informative.
# Citations that are too short, repeated elsewhere, or trivial are
# suspicious and likely fabricated.
MIN_CITATION_LEN = 8
MAX_CITATION_LEN = 240
CITATION_SCORE_THRESHOLD = 0.5

# Atomicity check: compound facts joining two claims with a conjunction
# are the #1 source of hallucinations. Forcing atomic facts means the
# verification gate can match each one cleanly.
COMPOUND_CONJUNCTIONS = (" and ", " plus ", " as well as ", " also ")


def citation_score(citation: str, raw_chunk: str) -> float:
    """
    Quality score in [0.0, 1.0]. Returns 0.0 if citation is missing
    or not a substring of raw_chunk. Otherwise:
      + 0.5 base (passing substring check is the main signal)
      + 0.3 if the citation appears exactly once in the chunk (specific)
      - 0.4 if the citation is < 20 chars OR < 3 words (trivial)
    Threshold 0.5: typical good citations ("auth-service connects to
    postgres_primary", "INSERT INTO payment_ledger (order_id, ...)")
    score 0.8+. Trivial citations like "import", "TODO", "//" score 0.1-0.4.
    """
    if not citation or citation not in raw_chunk:
        return 0.0
    score = 0.5
    if raw_chunk.count(citation) == 1:
        score += 0.3
    if len(citation) < 20 or len(citation.split()) < 3:
        score -= 0.4
    return max(score, 0.0)


# Layer 6: contradiction check.
# Only meaningful for FUNCTIONAL_RELATIONSHIPS, where a single source
# can only have one value for a given key (one port, one IP, one region).
# "A imports B" + "A imports C" is fine. "A runs_on_port 5432" +
# "A runs_on_port 6000" is a contradiction.
def check_contradiction(cursor, session_id: str, source_entity: str,
                        relationship: str, target_entity: str) -> tuple:
    """
    Returns (contradicts: bool, reason: str).
    For FUNCTIONAL_RELATIONSHIPS only: if there's already a different
    target for the same (source, relationship) pair, that's a contradiction.
    """
    rel = relationship.lower().strip()
    if rel not in FUNCTIONAL_RELATIONSHIPS:
        return False, ""
    cursor.execute(
        """
        SELECT DISTINCT target_entity FROM knowledge_graph
        WHERE session_id = ? AND source_entity = ?
          AND relationship = ? AND is_active = TRUE
        """,
        (session_id, source_entity, rel),
    )
    existing = {row["target_entity"] for row in cursor.fetchall()}
    if existing and target_entity not in existing:
        existing_first = next(iter(existing))
        return True, (
            f"contradicts existing: {source_entity} {rel} -> {existing_first} "
            f"(attempting to overwrite with {target_entity})"
        )
    return False, ""


# ─────────────────────────────────────────────────────────────────────────
# Layer 8: Pinned facts (YAML-configured immutable constants)
#
# Some facts are known constants and must never be overwritten by LLM
# extractions, even if the LLM contradicts them (e.g. POSTGRES_PRIMARY
# runs_on_port 5432). If a pinned fact exists for the same (source,
# relationship) pair but with a DIFFERENT target, the new triplet is a
# contradiction against ground truth and is rejected.
#
# Lazy-loaded from pinned_facts.yaml next to this module; cached for the
# process lifetime. If the file is absent or empty, the layer is a no-op.
# ─────────────────────────────────────────────────────────────────────────
_PINNED_FACTS: list = []
_PINNED_FACTS_LOADED = False


def _load_pinned_facts():
    """Lazy-load pinned facts from pinned_facts.yaml. Cached for process lifetime."""
    global _PINNED_FACTS, _PINNED_FACTS_LOADED
    if _PINNED_FACTS_LOADED:
        return
    path = Path(__file__).parent / "pinned_facts.yaml"
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            _PINNED_FACTS = data.get("facts", []) or []
        except Exception as e:
            print(f"[GUARDRAIL] Failed to load pinned_facts.yaml ({e}); treating as empty.")
            _PINNED_FACTS = []
    _PINNED_FACTS_LOADED = True


def reload_pinned_facts():
    """
    Force a reload of pinned_facts.yaml on the next check. Exposed so tests
    can swap in a temp YAML and so operators can pick up edits without a
    full process restart.
    """
    global _PINNED_FACTS_LOADED
    _PINNED_FACTS_LOADED = False


def check_pinned_contradiction(source: str, relationship: str, target: str) -> bool:
    """
    Returns True if (source, relationship, target) contradicts any pinned
    fact. 'Contradicts' means: a pinned fact exists with the same
    (source, relationship) but a DIFFERENT target. A matching target is
    NOT a contradiction — the LLM agreeing with ground truth is fine.
    """
    _load_pinned_facts()
    rel = (relationship or "").lower().strip()
    src = (source or "").strip().upper()
    tgt = (target or "").strip().upper()
    for p in _PINNED_FACTS:
        if (
            str(p.get("source_entity", "")).upper() == src
            and str(p.get("relationship", "")).lower() == rel
        ):
            if str(p.get("target_entity", "")).upper() != tgt:
                return True
    return False


# Rejection log table — keep one row per rejected triplet so we can
# analyze failure modes and tighten the filters over time.
REJECTION_TABLE = """
    CREATE TABLE IF NOT EXISTS rejected_triplets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        proposed_json TEXT,
        rejection_reason TEXT,
        raw_chunk TEXT,
        rejected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
"""


def _log_rejection(cursor, session_id, triplet, reason, raw_chunk, score=None):
    """Persist a rejected triplet for later analysis."""
    cursor.execute(
        """
        INSERT INTO rejected_triplets
            (session_id, proposed_json, rejection_reason, raw_chunk)
        VALUES (?, ?, ?, ?)
        """,
        (session_id, triplet.model_dump_json(), reason, (raw_chunk or "")[:4000]),
    )
    # Broadcast a live telemetry event for operators watching the dashboard.
    # engine.py runs synchronous sqlite I/O (called via run_in_threadpool in
    # the FastAPI path), so we may or may not be inside a running event loop.
    # Schedule the broadcast when there is one; skip silently otherwise (e.g.
    # during unit tests or in-process agent mode). A failed broadcast must
    # never break the commit transaction.
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(telemetry_manager.broadcast(session_id, {
            "event": "guardrail_rejection",
            "telemetry": {
                "active_agent": getattr(triplet, "source_entity", None),
                "rejection_reason": reason,
            },
            "rejection_details": {
                "proposed_source": getattr(triplet, "source_entity", None),
                "proposed_relationship": getattr(triplet, "relationship", None),
                "proposed_target": getattr(triplet, "target_entity", None),
                "citation_score": score,
            },
        }))
    except RuntimeError:
        # No running event loop — e.g. unit tests or in-process agent mode.
        pass


def commit_page_data_to_sqlite(
    session_id: str,
    agent_id: str,
    raw_chunk: str,
    extraction_data: PageExtractionPayload,
    extractor: str = "agent",
    pass_number: int = 0,
) -> int:
    """
    Runs the deterministic verification engine and bulk upserts verified facts.

    Verification layers, applied in order to each triplet:
      1. direction_check consistency  (in-schema CoT, post-parse)
      2. closed relationship vocabulary (ALLOWED_RELATIONSHIPS)
      3. citation substring match      (existing)
      4. citation quality score        (length / uniqueness / triviality)
      5. atomicity                     (no compound conjunctions)
      6. contradiction check           (FUNCTIONAL_RELATIONSHIPS only)

    NOTE: this is intentionally a plain function, not async def. It has no
    internal await - it is pure blocking sqlite3 I/O - and main.py calls it
    via run_in_threadpool(), which expects a plain synchronous callable.
    Calling an async def function returns an unawaited coroutine object
    without running its body; run_in_threadpool then hands that coroutine
    back as if it were the real return value. The endpoint's very next line
    (`len(...) - saved_triplets_count`) then crashes with
    "TypeError: unsupported operand type(s) for -: 'int' and 'coroutine'" -
    and, more importantly, nothing was ever actually written to the database,
    because the function body never ran. Confirmed by direct reproduction,
    not just by inspection: see the verification harness for this exact
    call pattern with and without async.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    verified_count = 0

    try:
        cursor.execute("BEGIN TRANSACTION;")

        # Ensure session exists to prevent foreign key constraints from failing
        cursor.execute("INSERT OR IGNORE INTO sessions (session_id) VALUES (?)", (session_id,))

        # Ensure the rejection log table exists (idempotent)
        cursor.execute(REJECTION_TABLE)

        for triplet in extraction_data.extracted_triplets:
            reason = None
            # Compute the score once. Used by Layer 4 and persisted on
            # commit so we can later answer "this row is on the edge
            # of being rejected — what happened?"
            score = citation_score(triplet.citation_quote, raw_chunk)

            # Layer 1: in-schema CoT — model said who-acts-on-whom,
            # now verify what it filled in matches.
            if not validate_direction(triplet):
                reason = "direction_check_mismatch"

            # Layer 2: closed relationship vocabulary
            elif not validate_relationship(triplet.relationship):
                reason = "bad_relationship_vocab"

            # Layer 2b: per-EntityType shape validation
            # Cheap regex gate: the entity string must look like its
            # declared type. Catches obvious garbage like a raw SQL
            # fragment labeled as a SERVICE.
            elif not validate_entity_shape(triplet.source_entity, triplet.source_type):
                reason = "bad_source_entity_shape"
            elif not validate_entity_shape(triplet.target_entity, triplet.target_type):
                reason = "bad_target_entity_shape"

            # Layer 3: citation substring match (existing)
            elif not verify_citation(raw_chunk, triplet.citation_quote):
                reason = "citation_mismatch"

            # Layer 4: citation quality score
            elif score < CITATION_SCORE_THRESHOLD:
                reason = "low_citation_score"

            # Layer 5: atomicity — no compound conjunctions
            elif any(c in (triplet.citation_quote or "").lower() for c in COMPOUND_CONJUNCTIONS):
                reason = "compound_fact"

            if reason is not None:
                _log_rejection(cursor, session_id, triplet, reason, raw_chunk, score=score)
                extra = f" | direction_check={triplet.direction_check!r}" if reason == "direction_check_mismatch" else ""
                print(f"[GUARDRAIL] Rejected: {triplet.source_entity} -> {triplet.target_entity} | {reason}{extra}")
                continue

            src = canonicalize_entity(cursor, session_id, triplet.source_entity)
            tgt = canonicalize_entity(cursor, session_id, triplet.target_entity)

            # Layer 6: contradiction check (FUNCTIONAL_RELATIONSHIPS only)
            contradicts, why = check_contradiction(
                cursor, session_id, src, triplet.relationship, tgt
            )
            if contradicts:
                _log_rejection(cursor, session_id, triplet, why, raw_chunk, score=score)
                print(f"[GUARDRAIL] Rejected: {why}")
                continue

            # Layer 8: pinned fact contradiction (YAML-configured constants)
            if check_pinned_contradiction(src, triplet.relationship, tgt):
                why = (
                    f"contradicts pinned fact: {src} {triplet.relationship} -> {tgt}"
                )
                _log_rejection(
                    cursor, session_id, triplet, "contradicts_pinned_fact",
                    raw_chunk, score=score,
                )
                print(f"[GUARDRAIL] Rejected: {why}")
                continue

            edge_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT OR REPLACE INTO knowledge_graph
                (edge_id, session_id, agent_id, source_entity, source_type,
                 relationship, target_entity, target_type, citation_quote,
                 is_active, extractor, pass_number, raw_citation_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, TRUE, ?, ?, ?)
            """, (edge_id, session_id, agent_id, src,
                  triplet.source_type,
                  triplet.relationship.lower().strip(), tgt,
                  triplet.target_type,
                  triplet.citation_quote.strip(),
                  extractor, pass_number, score))

            verified_count += 1

        for var_name, status in extraction_data.unresolved_variables_mutations.items():
            var_id = f"{session_id}_{var_name}"
            if status.upper() == "RESOLVED":
                cursor.execute("DELETE FROM unresolved_variables WHERE variable_id = ?", (var_id,))
            else:
                cursor.execute("""
                    INSERT OR IGNORE INTO unresolved_variables (variable_id, session_id, variable_name, status)
                    VALUES (?, ?, ?, ?)
                """, (var_id, session_id, var_name.upper().strip(), status.upper().strip()))

        conn.commit()
    except Exception as e:
        conn.execute("ROLLBACK;")
        print(f"[DATABASE ERROR] Transaction aborted: {e}")
        raise e
    finally:
        conn.close()

    return verified_count

def _apply_query_aware_boost(rows: list, query: str, k_hops: int) -> list:
    """
    Re-orders active graph rows so ones relevant to the CURRENT query are
    prioritized ahead of pure global relevance_score ranking.

    Without this, compile_bounded_markdown_view ranks every active fact by
    a single static relevance_score with no notion of "relevant to what's
    being asked right now." Under a tight token budget on a long session,
    this reproduces the exact "lost in the middle" failure the scratchpad
    exists to prevent, just moved from the raw context window to the
    compression layer: an early root-cause fact can lose out to later,
    more-recently-referenced facts that happen to score higher globally
    but have nothing to do with the current question.

    Builds a lightweight in-memory graph from the rows themselves (not
    persisted, not the sweeper's NetworkX layer - this is query-scoped and
    thrown away after use), finds which known entities are named in the
    query text, and expands a k-hop neighborhood around them. Rows with
    either endpoint in that neighborhood are boosted into a top tier;
    everything else keeps its normal relevance_score ordering below that
    tier. If no known entity is named in the query at all, falls back to
    pure global ranking rather than guessing at relevance.

    Entity matching against the query text uses the same separator-
    stripped normalization as canonicalize_entity/validate_direction
    (strip_separators, uppercase) - a raw substring check would miss
    "API_GATEWAY" against a query asking about "the api gateway", for the
    identical underscore/space/case reasons already fixed elsewhere in
    this file.
    """
    import networkx as nx
    from text_matching import strip_separators

    if not query or not rows:
        return sorted(rows, key=lambda r: r["relevance_score"] or 0.0, reverse=True)

    G = nx.Graph()
    for r in rows:
        G.add_edge(r["source_entity"], r["target_entity"])

    query_normalized = strip_separators(query.upper())
    mentioned = {
        e for e in G.nodes()
        if strip_separators(e.upper()) in query_normalized
    }

    if not mentioned:
        return sorted(rows, key=lambda r: r["relevance_score"] or 0.0, reverse=True)

    neighborhood = set(mentioned)
    frontier = set(mentioned)
    for _ in range(max(k_hops, 0)):
        next_frontier = set()
        for node in frontier:
            next_frontier.update(G.neighbors(node))
        neighborhood.update(next_frontier)
        frontier = next_frontier

    def sort_key(r):
        in_neighborhood = r["source_entity"] in neighborhood or r["target_entity"] in neighborhood
        return (1 if in_neighborhood else 0, r["relevance_score"] or 0.0)

    return sorted(rows, key=sort_key, reverse=True)


def compile_bounded_markdown_view(
    session_id: str,
    max_tokens: int = 6000,
    footer_reserve_tokens: int = 60,
    query: str = None,
    k_hops: int = 2,
) -> str:
    """
    Assembles a token-bounded GitHub-flavored Markdown view from SQLite.

    Priority order (matches the "never truncate the executive summary"
    design rule): the Unresolved Variables Matrix is always included in
    full - it stays small by nature and is the thing an agent least wants
    to lose. Remaining budget after that goes to knowledge_graph rows,
    ranked highest-priority first, until the budget runs out. Anything cut
    is reported in a footer rather than silently vanishing.

    When `query` is given, rows within `k_hops` of any entity named in the
    query are boosted ahead of pure global relevance_score ranking (see
    _apply_query_aware_boost). When query is None, ranking is unchanged
    from before - pure global relevance_score, for backward compatibility
    with any caller that doesn't have a specific question to bias toward.

    Replaces the old compile_graph_memory_to_markdown, which had no
    is_active filter, no ordering, and no budget at all - it would return
    every row ever written, including ones the sweeper had already
    deactivated.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT source_entity, source_type, relationship, target_entity,
               target_type, citation_quote, hierarchy_level, edge_id,
               relevance_score
        FROM knowledge_graph
        WHERE session_id = ? AND is_active = TRUE
        """,
        (session_id,),
    )
    graph_rows = _apply_query_aware_boost(cursor.fetchall(), query, k_hops)

    cursor.execute(
        "SELECT variable_name, status FROM unresolved_variables WHERE session_id = ?",
        (session_id,),
    )
    var_rows = cursor.fetchall()
    conn.close()

    # --- Section 1: Unresolved Variables (fixed priority, never cut) ---
    var_lines = ["## 1. UNRESOLVED VARIABLES MATRIX"]
    if not var_rows:
        var_lines.append("*(No active variables currently tracked)*")
    else:
        for row in var_rows:
            var_lines.append(f"- [?] `{row['variable_name']}` (Status: {row['status']})")
    var_section = "\n".join(var_lines)

    remaining_budget = max_tokens - count_tokens(var_section) - footer_reserve_tokens
    if remaining_budget < 0:
        remaining_budget = 0

    # --- Section 2: Knowledge graph, top-K by relevance until budget runs out ---
    graph_lines = ["## 2. KNOWLEDGE GRAPH MEMORY (Verified Facts)"]
    included = 0
    if not graph_rows:
        graph_lines.append("*(No active knowledge graph nodes established for this session)*")
    else:
        used = count_tokens(graph_lines[0])
        for row in graph_rows:
            # Type tag shown when known — UNKNOWN means the row was
            # written before the type columns existed (pre-migration).
            stype = (row["source_type"] or "UNKNOWN").lower()
            ttype = (row["target_type"] or "UNKNOWN").lower()
            type_tag_src = f" <{stype}>" if stype != "unknown" else ""
            type_tag_tgt = f" <{ttype}>" if ttype != "unknown" else ""
            if row["hierarchy_level"] == 2:
                line = (
                    f"* `[{row['source_entity']}]{type_tag_src}` --({row['relationship']})--> "
                    f"`[{row['target_entity']}]{type_tag_tgt}` "
                    f"`[COMPRESSED | drill-down id: {row['edge_id']}]`"
                )
            else:
                line = (
                    f"* `[{row['source_entity']}]{type_tag_src}` --({row['relationship']})--> "
                    f"`[{row['target_entity']}]{type_tag_tgt}`\n"
                    f"  └── Source Citation: \"{row['citation_quote']}\""
                )
            line_tokens = count_tokens(line)
            if used + line_tokens > remaining_budget:
                break
            graph_lines.append(line)
            used += line_tokens
            included += 1

    omitted = len(graph_rows) - included
    if omitted > 0:
        graph_lines.append(
            f"\n*({omitted} lower-relevance fact(s) omitted to fit the {max_tokens}-token "
            f"budget. Call drill-down on a COMPRESSED node, or raise max_tokens, to see more.)*"
        )
    graph_section = "\n".join(graph_lines)

    return f"{var_section}\n\n{graph_section}"