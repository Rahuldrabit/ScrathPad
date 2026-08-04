"""
A/B harness: same underlying LLM, same question, same token budget -
the only variable is whether ScratchpadPoweredLLM mediates the context or
not. Uses whatever local backend you already have running (LM Studio /
Ollama / etc via scratchpad_agent.py's existing backend plumbing) - this
harness does not call any model itself, it just constructs the two
conditions correctly and scores whatever comes back.

Usage (against your own running LM Studio / Ollama instance):

    python3 ab_harness.py

Requires scratchpad_agent.py, engine.py, database.py, schema.py,
inference.py to be importable (i.e. run this from inside src/, or add
src/ to PYTHONPATH).
"""
import sys
import os
import uuid
import httpx

sys.path.insert(0, r"D:\ScratchPad\src\src")

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_file = open(os.path.join(os.path.dirname(__file__), "ab_harness.log"), "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

from synthetic_incident import DOCUMENT, GROUND_TRUTH_CHAIN, FORBIDDEN_CAUSAL_ENTITIES, QUESTION
from scoring import score_answer


def chunk_document(document: str, chunk_size: int = 400) -> list[str]:
    """Split into paragraph-ish chunks, simulating how a real ingestion
    pipeline would feed the document in incrementally rather than all at
    once - this matters for the WITHOUT-scratchpad condition specifically,
    since it determines what "no external memory" actually means here."""
    blocks = [b.strip() for b in document.split("\n\n") if b.strip()]
    chunks, current = [], ""
    for b in blocks:
        if len(current) + len(b) > chunk_size and current:
            chunks.append(current)
            current = b
        else:
            current = f"{current}\n\n{b}" if current else b
    if current:
        chunks.append(current)
    return chunks


def run_without_scratchpad(document: str, question: str, token_budget: int,
                            llm_url: str, model: str, variant: str = "recent_only") -> str:
    """
    Baseline: no ScratchpadPoweredLLM, no persistent structured memory.

    variant="recent_only"    -> only the last ~token_budget*4 characters of
                                 the document are visible (simulates a
                                 rolling window / no external memory across
                                 chunks - the realistic "no scratchpad"
                                 condition for an incrementally-ingested doc)
    variant="naive_truncate" -> as much of the document as fits from the
                                 start (simulates "dump the whole thing in
                                 and hope it fits")
    """
    approx_chars = token_budget * 4
    if variant == "recent_only":
        context = document[-approx_chars:]
    else:
        context = document[:approx_chars]

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": f"Answer using only the following log excerpt:\n\n{context}"},
            {"role": "user", "content": question},
        ],
        "temperature": 0.1,
        "max_tokens": 500,
    }
    r = httpx.post(f"{llm_url}/chat/completions", json=payload, timeout=120.0)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def run_with_scratchpad(document: str, question: str, token_budget: int,
                         model: str, backend: str = "lmstudio",
                         lmstudio_url: str = "http://localhost:1234/v1") -> str:
    """
    Real condition: ingest the document chunk by chunk through
    ScratchpadPoweredLLM.record_observation() (citation-verified extraction,
    same as any real tool-result ingestion), then ask the question through
    the same wrapper so it answers from the compiled, verified,
    token-bounded view rather than raw text.
    """
    from scratchpad_agent import ScratchpadPoweredLLM

    agent = ScratchpadPoweredLLM(
        goal=f"Investigate: {question}",
        context_size=token_budget + 2000,  # +buffer, matches the class's own accounting
        session_id=f"eval-{uuid.uuid4().hex[:8]}",
        model=model,
        backend=backend,
        lmstudio_url=lmstudio_url,
        auto_plan=False,  # skip the planning call for a pure ingestion test
    )

    for chunk in chunk_document(document):
        agent.record_observation(chunk, source="incident_log")

    response = agent([{"role": "user", "content": question}])
    return response["choices"][0]["message"]["content"]


def run_comparison(n_trials: int = 1, llm_url: str = "http://localhost:1234/v1",
                    model: str = "google/gemma-4-e4b", token_budget: int = 1500):
    results = {"without_scratchpad": [], "with_scratchpad": []}

    for i in range(n_trials):
        answer_a = run_without_scratchpad(DOCUMENT, QUESTION, token_budget, llm_url, model)
        results["without_scratchpad"].append(
            score_answer(answer_a, GROUND_TRUTH_CHAIN, FORBIDDEN_CAUSAL_ENTITIES)
        )

        answer_b = run_with_scratchpad(DOCUMENT, QUESTION, token_budget, model)
        results["with_scratchpad"].append(
            score_answer(answer_b, GROUND_TRUTH_CHAIN, FORBIDDEN_CAUSAL_ENTITIES)
        )

    return results


def print_scorecard(results: dict) -> None:
    print(f"{'condition':<20} {'pass_rate':<12} {'avg_completeness':<18} {'hallucination_free_rate':<24}")
    for condition, runs in results.items():
        n = len(runs)
        pass_rate = sum(r["overall_pass"] for r in runs) / n
        avg_completeness = sum(r["reasoning"]["chain_completeness"] for r in runs) / n
        halluc_free = sum(r["hallucination"]["hallucination_free"] for r in runs) / n
        print(f"{condition:<20} {pass_rate:<12.0%} {avg_completeness:<18.0%} {halluc_free:<24.0%}")


if __name__ == "__main__":
    results = run_comparison(n_trials=5)
    print_scorecard(results)
