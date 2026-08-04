# Scratchpad Hallucination + Reasoning Eval

## Files
- `synthetic_incident.py` — closed synthetic test corpus (ground truth known exactly, by construction)
- `scoring.py` — deterministic scorer (no LLM judge). Verified against 4 hand-built failure modes.
- `verify_scorer.py` — run this to see the scorer correctly discriminate good / hallucinated / lost-in-the-middle / out-of-order answers
- `ab_harness.py` — runs the same document+question through WITHOUT vs WITH ScratchpadPoweredLLM
- `verify_harness_wiring.py` — proves the harness plumbing works end-to-end (mocked LLM call, real everything else)
- `src_patch/scratchpad_agent.py` — **your file, with a real bug fixed** (see below). Overwrite your `src/scratchpad_agent.py` with this.

## The bug found while building this

`_store_facts()` constructed `GraphTriplet` objects using only the original
four fields (`source_entity`, `relationship`, `target_entity`, `citation_quote`).
Since `GraphTriplet` now requires `direction_check`, `source_type`, `target_type`
with no defaults, every call to `record_observation()` — and the internal
plan-entity and tool-call bookkeeping paths — would crash with a Pydantic
`ValidationError` the moment real data flowed through. Confirmed by actually
running it, not just reading it.

Compounding it: `_extract_facts()` was discarding the `direction_check`/
`source_type`/`target_type` fields the SLM had *already* correctly generated
under grammar constraint, before `_store_facts()` needed them again downstream.

Fix: `_extract_facts()` now carries those fields through instead of dropping
them. `_store_facts()` now falls back to deriving `direction_check` from the
triplet's own fields (Python already knows the direction for internal/
deterministic bookkeeping facts — no SLM reasoning needed there) and defaults
`source_type`/`target_type` to `"SERVICE"` when a caller doesn't supply them.

Verified: `verify_harness_wiring.py` exercises `record_observation()` through
the real citation gate, the real `validate_direction` check, the real
`ALLOWED_RELATIONSHIPS` check, and the real bounded view compiler — all pass.

## Running it for real

```bash
# 1. Prove the scorer works (no LLM needed)
python3 verify_scorer.py

# 2. Prove the harness plumbing works (no LLM needed, mocked extraction)
python3 verify_harness_wiring.py

# 3. Run the actual A/B comparison (needs your LM Studio / Ollama running)
cp src_patch/scratchpad_agent.py /path/to/ScrathPad/src/scratchpad_agent.py
cd /path/to/ScrathPad
python3 /path/to/ab_harness.py
```

Expected scorecard shape:

| condition | pass_rate | avg_completeness | hallucination_free_rate |
|---|---|---|---|
| without_scratchpad | low | low (misses early root-cause hop) | varies |
| with_scratchpad | high | high | high |

If `with_scratchpad` doesn't clearly outperform `without_scratchpad` on your
real model, that's a genuinely useful negative result — it tells you the
compression or retrieval isn't preserving what the reasoning test needs, and
`missing_hops` in the scorer output tells you exactly which hop is getting
lost.
