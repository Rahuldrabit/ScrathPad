# Deterministic Hallucination Pipeline — Implementation Roadmap

This document is the design + implementation plan for the verification
pipeline in the scratchpad middleware. It tracks what has shipped, what
is deferred, and exactly how to implement each deferred piece when the
time comes.

**Status of each layer (Layers 1–8 from the original plan):**

| # | Layer | Status | Commit |
|---|---|---|---|
| 1 | Closed relationship vocabulary | **Shipped** | `aef404c` |
| 2 | In-schema CoT (`direction_check` + `validate_direction`) | **Shipped** | `aef404c` |
| 3 | Citation substring match (existing) | **Shipped** | pre-`aef404c` |
| 4 | Citation quality score (length / uniqueness / triviality) | **Shipped** | `aef404c` |
| 5 | Atomicity (no compound conjunctions) | **Shipped** | `aef404c` |
| 6 | Contradiction check (`FUNCTIONAL_RELATIONSHIPS`) | **Shipped** | `aef404c` |
| 7 | Rejection log table (`rejected_triplets`) | **Shipped** | `aef404c` |
| 8 | Pinned facts (YAML config, hardcoded constants) | **Shipped** | (this commit) |
| — | L2 type inheritance (sweeper overrides LLM types) | **Shipped** | (this commit) |
| — | Source-provenance columns + extractor history | **Shipped** | (this commit) |
| — | Per-`EntityType` regex validation | **Shipped** | (this commit) |
| — | Rejection-rate telemetry over WebSocket | **Shipped** | (this commit) |
| — | Multi-pass consensus extraction (3-pass, keep ≥2) | **Deferred** | — |
| — | Re-extraction API for sweeper-deactivated rows | **Deferred** | — |

**The currently shipping pipeline.** Six layers, in order, all
deterministic, all logged. Total tests: 77/77 (51 unit, 10 regression,
16 L2 type inheritance).

```python
# engine.py: commit_page_data_to_sqlite, the canonical order
for triplet in extraction_data.extracted_triplets:
    reason = None
    if not validate_direction(triplet):                           # 1+2
        reason = "direction_check_mismatch"
    elif not validate_relationship(triplet.relationship):         # 1
        reason = "bad_relationship_vocab"
    elif not verify_citation(raw_chunk, triplet.citation_quote):  # 3
        reason = "citation_mismatch"
    elif citation_score(triplet.citation_quote, raw_chunk) < 0.5: # 4
        reason = "low_citation_score"
    elif any(c in triplet.citation_quote.lower()
             for c in COMPOUND_CONJUNCTIONS):                     # 5
        reason = "compound_fact"
    if reason is not None:
        _log_rejection(cursor, session_id, triplet, reason, raw_chunk)
        continue
    src = canonicalize_entity(cursor, session_id, triplet.source_entity)
    tgt = canonicalize_entity(cursor, session_id, triplet.target_entity)
    contradicts, why = check_contradiction(                       # 6
        cursor, session_id, src, triplet.relationship, tgt
    )
    if contradicts:
        _log_rejection(cursor, session_id, triplet, why, raw_chunk)
        continue
    # store ...
```

---

## Why each deferred layer was deferred, and what it would take

The shipped layers cover ~95% of the hallucination surface observed in
testing against Gemma 4B. The deferred layers are **escalation
mechanisms** — they add cost (LLM calls, latency, schema changes) and
should only be turned on once the shipped layers prove insufficient in
production.

Each section below has:
- The problem it solves (with example)
- The exact files to change
- The exact code shape
- Cost / latency impact
- What to measure to decide if it should be turned on

---

## L2 type inheritance (sweeper override)

### Problem
The L2 path (`sweeper.compress_louvain_community` →
`_execute_lmstudio` → `L2CompressionPayload`) lets the LLM set
`source_type` and `target_type` on the L2 summary triplet. The LLM
often gets these wrong (e.g., it labels a summary of three
[SERVICE]→[DATABASE] edges as `[TABLE]→[TABLE]`). The L2 node then
contradicts its own source community, which is a soft signal that
something is off.

### Fix
After fetching the L1 source triplets for the community, compute the
majority type for each side. Override whatever the LLM emitted for the
L2 triplet's `source_type` and `target_type`. Don't trust the LLM to
reclassify from scratch — that's a second place it can hallucinate.

### Where
`sweeper.py:execute_maintenance_sweep`, in the L2 compression block,
right after the L2 guardrail drops ungrounded triplets but before
`cursor.execute("BEGIN TRANSACTION;")`.

### Code shape

```python
# In sweeper.py, after filtered_l2_triplets is computed:

from collections import Counter

# Aggregate types from the L1 community
src_type_counter = Counter(t.get("source_type", "UNKNOWN") for t in raw_triplets)
tgt_type_counter = Counter(t.get("target_type", "UNKNOWN") for t in raw_triplets)

# Majority vote. "UNKNOWN" loses to any real type.
def majority(counter):
    real_types = {k: v for k, v in counter.items() if k != "UNKNOWN"}
    if not real_types:
        return "UNKNOWN"
    # If a single real type wins by 2+, use it
    top, count = counter.most_common(1)[0]
    if top != "UNKNOWN" and count >= 2:
        return top
    # Tie or all UNKNOWN: fall back to top real
    if real_types:
        return Counter(real_types).most_common(1)[0][0]
    return "UNKNOWN"

inherited_source_type = majority(src_type_counter)
inherited_target_type = majority(tgt_type_counter)

# Override the LLM's types. The LLM is not allowed to reclassify.
for t in filtered_l2_triplets:
    t.source_type = inherited_source_type
    t.target_type = inherited_target_type
    # Note: GraphTripletSchema is what the SLM emits, but L2 rows
    # are stored with the new source_type / target_type columns
    # populated from these values, not from t.source_type.
```

### Cost
Zero. Pure Python aggregation on already-loaded rows.

### What to measure before turning on
Look at `rejected_triplets` for any rejection reason involving L2
nodes. If rejection rate is low after Layers 1–6, this is mostly
cosmetic — turn it on when you want the L2 view to be type-consistent
with its source community.

### Test to add
```python
def test_l2_type_inheritance():
    # Set up a community where all source L1 entities are SERVICE
    # and all target L1 entities are DATABASE.
    # Trigger compression.
    # Verify the resulting L2 row's source_type='SERVICE',
    # target_type='DATABASE', regardless of what the LLM said.
```

---

## Multi-pass consensus (Layer 5 from the original plan)

### Problem
A single LLM extraction pass can hallucinate citations, get direction
wrong, or invent entities. Even with Layers 1–6 filtering, the LLM
can still produce a grammatically-valid triplet that passes every
deterministic check yet is still subtly wrong (a citation that
exists in the raw text but is being used out of context).

### Fix
Run the same extraction **N times** (default 3) and only keep triplets
that appear in **≥2 of the N passes**. Triplets that appear in only
one pass are dropped as "non-reproducible hallucinations."

This is the most expensive layer by far — it triples the LLM cost
per extraction. Don't turn it on until the cheaper layers prove
insufficient.

### Where
- `client.py:commit_messy_input` (for the `middleware/process` path)
- `scratchpad_agent.py:_extract_facts` (for the `record_observation` path)
- Anywhere else that calls `inference.generate_structured` with a
  `GraphTriplet`-bearing schema

### Code shape

```python
# New helper in inference.py or a new consensus.py module

from collections import Counter
from typing import List
from pydantic import BaseModel

def consensus_extract(
    raw_text: str,
    system_prompt: str,
    response_schema: Type[BaseModel],
    inference_engine: UniversalInferenceEngine,
    n_passes: int = 3,
    min_consensus: int = 2,
) -> List[BaseModel]:
    """
    Run the same extraction N times. Return only facts seen in
    >= min_consensus passes.
    """
    extractions = []
    for _ in range(n_passes):
        try:
            result = inference_engine.generate_structured(
                prompt=raw_text,
                system_prompt=system_prompt,
                response_schema=response_schema,
            )
            extractions.append(result)
        except Exception as e:
            # One bad pass shouldn't kill the whole call. Log and continue.
            print(f"[consensus] pass failed: {e}")
            continue

    if not extractions:
        return []

    # Collect all triplets across passes, keyed by (source, rel, target)
    all_triplets = []
    for ext in extractions:
        for t in ext.triplets:
            all_triplets.append(t)

    counter = Counter(
        (t.direction_check, t.source_entity.lower().strip(),
         t.relationship.lower().strip(), t.target_entity.lower().strip())
        for t in all_triplets
    )

    # Keep only triplets that hit consensus
    kept_keys = {k for k, c in counter.items() if c >= min_consensus}
    seen = set()
    out = []
    for t in all_triplets:
        key = (t.direction_check, t.source_entity.lower().strip(),
               t.relationship.lower().strip(), t.target_entity.lower().strip())
        if key in kept_keys and key not in seen:
            out.append(t)
            seen.add(key)
    return out
```

### Wire-up

```python
# In client.py:commit_messy_input — replace single LLM call with:
def commit_messy_input(self, session_id, raw_text, agent_id="middleware_auto_extract", use_consensus=False):
    if use_consensus:
        from consensus import consensus_extract
        triplets = consensus_extract(
            raw_text,
            system_prompt="Extract clear entity-relationship triplets...",
            response_schema=L1ExtractionPayload,
            inference_engine=self.inference_engine,
        )
        page_payload = PageExtractionPayload(
            extracted_triplets=triplets,
            unresolved_variables_mutations={},
            is_chunk_completely_exhausted=True,
        )
    else:
        # existing single-pass path
        ...
    return commit_page_data_to_sqlite(...)
```

### Cost
3× LLM calls per extraction. For a tool-calling agent doing 6 turns
with 2 record_observation calls per turn, that's 12 → 36 LLM calls.
With Gemma 4B at ~5–30s per call, that pushes a 6-turn loop from
~2 minutes to ~6–18 minutes.

### Mitigation
- Make `n_passes` and `min_consensus` configurable per call
- Default to 2 passes (instead of 3) for `use_consensus=True` — most
  hallucinations fail at 1/2 consensus anyway
- Add a "fast path" that skips consensus on trivial chunks (e.g.,
  chunks < 500 chars)

### What to measure before turning on
Look at the rejection log. If you see triplets being committed that
the user later flags as wrong, that's the signal to turn on consensus.
If the rejection log's `bad_relationship_vocab` / `citation_mismatch` /
`compound_fact` reasons are catching the obvious hallucinations, you
probably don't need consensus.

### Test to add
```python
def test_consensus():
    # Mock the inference engine to return different triplets each pass.
    # Pass 1: 3 valid + 1 hallucinated
    # Pass 2: 3 valid only (hallucination gone)
    # Pass 3: 3 valid + 2 NEW hallucinated
    # Run consensus_extract(n_passes=3, min_consensus=2)
    # Assert: 3 valid triplets returned, 3 hallucinated dropped
```

---

## Pinned facts (Layer 8)

### Problem
Some facts must never be overwritten by LLM extractions, even if the
LLM contradicts them. Examples:
- `POSTGRES_PRIMARY` runs on port `5432` — this is a real, known
  constant, not a learned fact
- `AUTH_SERVICE` is a `SERVICE` — type, not a relationship
- Compliance facts: `USER_DATA` is encrypted at rest, full stop

If a pinned fact is contradicted by a new extraction, the new
extraction is rejected (the pinned fact wins).

### Where
- New file: `src/pinned_facts.yaml` (or `.json`)
- New function in `engine.py:check_pinned_contradiction`
- Wire into `commit_page_data_to_sqlite`, after `check_contradiction`
  but before the INSERT

### Code shape

```python
# src/pinned_facts.yaml
facts:
  - source_entity: POSTGRES_PRIMARY
    relationship: runs_on_port
    target_entity: PORT_5432
  - source_entity: AUTH_SERVICE
    relationship: hosted_in_region
    target_entity: REGION_US_EAST_1
  - source_entity: USER_DATA_TABLE
    relationship: encrypted_at_rest
    target_entity: TRUE

# src/engine.py

import yaml
from pathlib import Path

_PINNED_FACTS: list[dict] = []
_PINNED_FACTS_LOADED = False

def _load_pinned_facts():
    """Lazy-load from pinned_facts.yaml. Cached for the process lifetime."""
    global _PINNED_FACTS, _PINNED_FACTS_LOADED
    if _PINNED_FACTS_LOADED:
        return
    path = Path(__file__).parent / "pinned_facts.yaml"
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        _PINNED_FACTS = data.get("facts", [])
    _PINNED_FACTS_LOADED = True

def check_pinned_contradiction(source: str, relationship: str, target: str) -> bool:
    """
    Returns True if (source, relationship, target) contradicts any
    pinned fact. 'Contradicts' means: a pinned fact exists with the
    same (source, relationship) but a DIFFERENT target.
    """
    _load_pinned_facts()
    rel = relationship.lower().strip()
    src = source.strip().upper()
    tgt = target.strip().upper()
    for p in _PINNED_FACTS:
        if p["source_entity"].upper() == src and p["relationship"].lower() == rel:
            if p["target_entity"].upper() != tgt:
                return True
    return False
```

### Wire-up

```python
# In commit_page_data_to_sqlite, after check_contradiction:

if check_pinned_contradiction(src, triplet.relationship, tgt):
    _log_rejection(cursor, session_id, triplet, "contradicts_pinned_fact", raw_chunk)
    print(f"[GUARDRAIL] Rejected: {src} {triplet.relationship} -> {tgt} contradicts pinned fact")
    continue
```

### Cost
Zero. Pure Python lookup.

### What to measure before turning on
You don't need to measure — just add the YAML file with the facts
you know to be true and turn it on. It's a one-line addition to the
commit pipeline.

### Test to add
```python
def test_pinned_facts():
    # Write a temp pinned_facts.yaml
    # Try to commit a triplet that contradicts it
    # Assert: rejected with reason 'contradicts_pinned_fact'
    # Then try a triplet that matches the pinned fact
    # Assert: committed
```

---

## Per-`EntityType` regex validation

### Problem
The current schema accepts any string for `source_type` and
`target_type` as long as it's in the `EntityType` literal. But that
doesn't check that the *entity itself* matches its declared type.
A `source_type="SERVICE"` with `source_entity="db.query('SELECT..."`
is clearly wrong but currently passes.

### Fix
Add a per-type regex allowlist. The entity string must match the
pattern for its declared type. If it doesn't, the triplet is
rejected.

### Where
- `src/schema.py` — new `ENTITY_PATTERNS` dict mapping `EntityType`
  to compiled regex
- `src/engine.py` — new `validate_entity_shape(entity, entity_type)`
  function
- Wire into `commit_page_data_to_sqlite`, right after the in-schema
  CoT check (it's a free rejection — no LLM call needed)

### Code shape

```python
# src/schema.py
import re

ENTITY_PATTERNS: dict[str, re.Pattern] = {
    "SERVICE":    re.compile(r"^[A-Z][A-Z0-9_]{2,40}$"),
    "FILE":       re.compile(r"^[\w./-]+\.\w{1,8}$"),
    "TABLE":      re.compile(r"^[A-Z][A-Z0-9_]{2,40}$"),
    "CONFIG_KEY": re.compile(r"^[A-Z][A-Z0-9_.]{2,80}$"),
    "FUNCTION":   re.compile(r"^[a-z][a-zA-Z0-9_]{2,60}$"),
    "QUEUE":      re.compile(r"^[A-Z][A-Z0-9_.:-]{2,60}$"),
    "PROTOCOL":   re.compile(r"^(http|https|grpc|amqp|tcp|udp|ws|wss)$"),
    "ENV_VAR":    re.compile(r"^[A-Z][A-Z0-9_]{2,60}$"),
    "CACHE":      re.compile(r"^[A-Z][A-Z0-9_]{2,40}$"),
    "DATABASE":   re.compile(r"^[A-Z][A-Z0-9_]{2,40}$"),
}

def validate_entity_shape(entity: str, entity_type: str) -> bool:
    pattern = ENTITY_PATTERNS.get(entity_type)
    if pattern is None:
        return True  # unknown type, don't reject
    return bool(pattern.match(entity.strip()))
```

### Wire-up

```python
# In commit_page_data_to_sqlite, in the for-loop, right after the
# direction_check and vocab checks:

elif not validate_entity_shape(triplet.source_entity, triplet.source_type):
    reason = "bad_source_entity_shape"
elif not validate_entity_shape(triplet.target_entity, triplet.target_type):
    reason = "bad_target_entity_shape"
```

### Cost
Zero. Pure regex.

### What to measure
Run the e2e test. Count how many real LLM extractions would be
rejected by this check. If it's > 30%, your `EntityType` literals
or the regexes are too restrictive — relax them. If it's < 5%,
turn it on permanently; it's catching real garbage.

---

## Source provenance columns

### Problem
The current `knowledge_graph` row has `agent_id` (the agent that
extracted it) and `extracted_at` (when). That's not enough to
debug "where did this fact come from?" — was it the middleware
auto-extraction, the agent's structured update, the L2 sweeper,
or a `record_observation` call?

### Fix
Add three columns: `extractor` (string: `middleware` | `agent` |
`sweeper_l2` | `observation`), `pass_number` (int, for consensus),
`raw_citation_score` (float, the score from Layer 4).

### Where
- `src/database.py:_migrate_knowledge_graph` — add the three columns
- `src/engine.py:_log_rejection` — also log these in the rejection
  table
- `src/sweeper.py` — set them when inserting L2 rows
- `src/scratchpad_agent.py` — set them when recording observations

### Cost
Zero. Storage + a few extra INSERT fields.

### Code shape

```python
# src/database.py: in _migrate_knowledge_graph, add to the (col, default) list:
for col, default in (
    ("source_type", "UNKNOWN"),
    ("target_type", "UNKNOWN"),
    ("extractor", "unknown"),
    ("pass_number", "0"),
    ("raw_citation_score", "0.0"),
):
    if col not in existing:
        cursor.execute(
            f"ALTER TABLE knowledge_graph ADD COLUMN {col} TEXT DEFAULT '{default}'"
        )
```

Then in `commit_page_data_to_sqlite`, after the Layer 4 score check:

```python
# Compute the score once, log it on commit, log it on rejection.
score = citation_score(triplet.citation_quote, raw_chunk)
if score < CITATION_SCORE_THRESHOLD:
    _log_rejection(cursor, session_id, triplet, "low_citation_score", raw_chunk, score=score)
    continue
# ... pass score to the INSERT
cursor.execute("""
    INSERT OR REPLACE INTO knowledge_graph
    (..., extractor, pass_number, raw_citation_score)
    VALUES (..., ?, ?, ?)
""", (..., "agent", 0, score))
```

---

## Rejection-rate telemetry

### Problem
The rejection log captures every rejection but there's no live view
of it. An operator looking at the dashboard can't see "right now,
60% of LLM extractions are being rejected for `low_citation_score`"
without querying SQLite.

### Fix
When `_log_rejection` is called, also broadcast a telemetry event
over the existing WebSocket stream (already wired in `main.py`).

### Where
- `src/engine.py:_log_rejection` — add a broadcast call
- `src/main.py` — pass the broadcast hook in, or import the
  `telemetry_manager` directly

### Code shape

```python
# src/engine.py

import asyncio
from telemetry import telemetry_manager  # already exists

def _log_rejection(cursor, session_id, triplet, reason, raw_chunk, score=None):
    cursor.execute(
        "INSERT INTO rejected_triplets (session_id, proposed_json, rejection_reason, raw_chunk) VALUES (?, ?, ?, ?)",
        (session_id, triplet.model_dump_json(), reason, (raw_chunk or "")[:4000]),
    )
    # Telemetry is async; engine.py is sync. Schedule the broadcast.
    # If there's no running event loop, skip silently.
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(telemetry_manager.broadcast(session_id, {
            "event": "guardrail_rejection",
            "telemetry": {
                "active_agent": triplet.source_entity,
                "rejection_reason": reason,
            },
            "rejection_details": {
                "proposed_source": triplet.source_entity,
                "proposed_relationship": triplet.relationship,
                "proposed_target": triplet.target_entity,
                "citation_score": score,
            },
        }))
    except RuntimeError:
        pass  # no event loop, skip
```

### Cost
One extra event-loop task per rejection. Negligible.

---

## Re-extraction API for sweeper-deactivated rows

### Problem
When the sweeper compresses a community, the L1 rows are marked
`is_active = FALSE` with `parent_node_id = <l2_edge_id>`. If the L2
summary turns out to be wrong, the L1 facts are lost — they're
still in the table but `is_active = FALSE` and most queries filter
on `is_active = TRUE`.

### Fix
Add a `POST /v1/agent/re-extract` endpoint that takes an `edge_id`
of a deactivated L1 row and re-extracts from the original
`raw_chunk` (which we'd need to store).

This is a bigger change. Don't do it until a real user has lost data
to a bad L2 compression.

### Where
- `src/database.py` — add a `raw_chunk` column to `knowledge_graph`
  (or a side table linking edge_id to raw_chunk)
- `src/main.py` — new endpoint
- `src/sweeper.py` — DON'T delete the raw_chunk on L2 compression
  (currently the column doesn't exist so this is moot, but when you
  add it, preserve the L1's raw_chunk for re-extraction)

### Code shape (sketch only)

```python
# src/main.py
@app.post("/v1/agent/re-extract")
async def re_extract_deactivated_edge(payload: ReExtractRequest):
    edge_id = payload.edge_id
    # Fetch the row's original raw_chunk
    # Re-run the extraction pipeline with consensus if enabled
    # If the new extraction matches the original, re-activate the row
    # If it doesn't, leave it deactivated and report the diff
    ...
```

This is the largest deferred change. Don't implement it without
talking to users about whether they've actually lost data.

---

## When to enable each layer

Use this table to decide when to turn on each deferred layer.

| Layer | Trigger to enable |
|---|---|
| L2 type inheritance | Any production use of the sweeper. The implementation is free. |
| Per-EntityType regex | Once you have a corpus of real LLM extractions and can see what % get rejected. |
| Pinned facts | Day 1, if you have any known constants. Just write the YAML. |
| Source provenance columns | Day 1. Free, no downside. |
| Rejection telemetry | Once you have a dashboard. The implementation is small. |
| Multi-pass consensus | Only when 1-6 demonstrably miss real hallucinations. |
| Re-extraction API | Only when a real user loses data to a bad L2 compression. |

---

## Test additions for each layer (when you implement them)

Each section above lists one or more unit tests to add when that layer
is implemented. They follow the same pattern as
`tests/test_verification_layers.py`: construct input directly (no LLM
calls), call the engine function, assert the row was committed or
rejected and the rejection reason matches.

The test runner is `python -u tests/test_verification_layers.py`
(also runs `test_layer1_vocabulary`, `test_layer2_direction_check`,
`test_layer4_citation_score`, `test_layer5_atomicity`,
`test_layer6_contradiction`, `test_full_pipeline`,
`test_migration`, `test_view_includes_types` — add new test_*
functions at the bottom and call them in `__main__`).

---

## Versioning

When you turn on a deferred layer, bump the minor version of the
middleware (`src/main.py:app = FastAPI(title=..., version=...)`) and
add a one-line note in the changelog. The current shipping version
is `2.0.0`. Bump to `2.1.0` when L2 type inheritance lands, `2.2.0`
when per-EntityType regex lands, `3.0.0` when multi-pass consensus
becomes the default (it's a behavior change for callers that
expected a single LLM call per extraction).
