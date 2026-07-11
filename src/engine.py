import uuid
from database import get_db_connection
from schema import PageExtractionPayload
from rapidfuzz import fuzz, process as rf_process

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


import re

def _strip_separators(s: str) -> str:
    return re.sub(r'[\s_\-]+', '', s)


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

def commit_page_data_to_sqlite(session_id: str, agent_id: str, raw_chunk: str, extraction_data: PageExtractionPayload) -> int:
    """
    Runs the deterministic verification engine and bulk upserts verified facts.

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
        
        for triplet in extraction_data.extracted_triplets:
            if not verify_citation(raw_chunk, triplet.citation_quote):
                print(f"[GUARDRAIL] Rejected hallucinated triplet: {triplet.source_entity} -> {triplet.target_entity}")
                continue 
                
            src = canonicalize_entity(cursor, session_id, triplet.source_entity)
            tgt = canonicalize_entity(cursor, session_id, triplet.target_entity)
                
            edge_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT OR REPLACE INTO knowledge_graph 
                (edge_id, session_id, agent_id, source_entity, relationship, target_entity, citation_quote, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, TRUE)
            """, (edge_id, session_id, agent_id, src, 
                  triplet.relationship.lower().strip(), tgt, 
                  triplet.citation_quote.strip()))
            
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

def compile_bounded_markdown_view(
    session_id: str,
    max_tokens: int = 6000,
    footer_reserve_tokens: int = 60,
) -> str:
    """
    Assembles a token-bounded GitHub-flavored Markdown view from SQLite.

    Priority order (matches the "never truncate the executive summary"
    design rule): the Unresolved Variables Matrix is always included in
    full - it stays small by nature and is the thing an agent least wants
    to lose. Remaining budget after that goes to knowledge_graph rows,
    highest relevance_score first, until the budget runs out. Anything cut
    is reported in a footer rather than silently vanishing.

    Replaces the old compile_graph_memory_to_markdown, which had no
    is_active filter, no ordering, and no budget at all - it would return
    every row ever written, including ones the sweeper had already
    deactivated.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT source_entity, relationship, target_entity, citation_quote,
               hierarchy_level, edge_id
        FROM knowledge_graph
        WHERE session_id = ? AND is_active = TRUE
        ORDER BY relevance_score DESC
        """,
        (session_id,),
    )
    graph_rows = cursor.fetchall()

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
            if row["hierarchy_level"] == 2:
                line = (
                    f"* `[{row['source_entity']}]` --({row['relationship']})--> "
                    f"`[{row['target_entity']}]` `[COMPRESSED | drill-down id: {row['edge_id']}]`"
                )
            else:
                line = (
                    f"* `[{row['source_entity']}]` --({row['relationship']})--> "
                    f"`[{row['target_entity']}]`\n"
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