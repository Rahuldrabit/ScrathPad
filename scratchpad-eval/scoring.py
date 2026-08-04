"""
Deterministic scoring for the closed synthetic corpus. No LLM judge - the
same "verify against ground truth, don't trust the model's word" principle
already used by engine.py::verify_citation, applied at evaluation time
instead of extraction time.
"""
import re

CAUSAL_MARKERS = ["caused", "leads to", "led to", "resulted in", "due to",
                   "because of", "responsible for", "triggered", "traced to"]
NEGATION_MARKERS = [" not ", " n't ", " no ", " never ", " unrelated",
                     " unaffected", " were not", " wasn't", " isn't"]


def score_reasoning_chain(answer: str, ground_truth_chain: list[str]) -> dict:
    """
    Checks whether every hop in the true causal chain appears in the answer
    (checked against ALL occurrences, not just the first - an entity
    mentioned twice, once in a contextual clause and once in its actual
    causal role, is present either way), and separately, whether there
    exists a valid reading of those occurrences in the correct order. A
    model that names the right final entity without the intermediate hops
    (a lucky guess, or a model that only saw the end of the document)
    scores partial credit on completeness, not full credit - this is the
    whole point: it tests genuine multi-hop reasoning, not pattern-matching
    to the most recent or most salient fact.
    """
    lower = answer.lower()
    occurrences = {
        hop: [m.start() for m in re.finditer(re.escape(hop.lower()), lower)]
        for hop in ground_truth_chain
    }
    missing = [hop for hop in ground_truth_chain if not occurrences[hop]]
    present_count = len(ground_truth_chain) - len(missing)

    correct_order = None
    if not missing:
        search_from = 0
        correct_order = True
        for hop in ground_truth_chain:
            valid_positions = [p for p in occurrences[hop] if p >= search_from]
            if not valid_positions:
                correct_order = False
                break
            search_from = min(valid_positions) + len(hop)

    return {
        "hops_found": present_count,
        "total_hops": len(ground_truth_chain),
        "chain_completeness": present_count / len(ground_truth_chain),
        "all_hops_present": len(missing) == 0,
        "correct_order": correct_order,
        "missing_hops": missing,
    }


def score_hallucination(answer: str, forbidden_entities: list[str], window: int = 60) -> dict:
    """
    Proximity-based, polarity-aware check: a forbidden entity mentioned near
    a negation marker ("billing-service was NOT affected") is a correct,
    faithful answer. The same entity mentioned near a causal marker with no
    negation nearby ("billing-service caused...") is a hallucinated causal
    claim about something the source document explicitly ruled out.

    This is a heuristic, not real NLU - it's a proximity window, not a parse
    tree. It's built to be reliable against a test corpus engineered to be
    unambiguous (as this one is), not as a general-purpose fact-checker.
    Sample-audit a subset of real runs by hand before trusting it blindly on
    less controlled text.
    """
    lower = answer.lower()
    violations = []
    for entity in forbidden_entities:
        entity_l = entity.lower()
        start = 0
        while True:
            idx = lower.find(entity_l, start)
            if idx == -1:
                break
            ctx = lower[max(0, idx - window): idx + len(entity_l) + window]
            has_causal = any(c in ctx for c in CAUSAL_MARKERS)
            has_negation = any(n in ctx for n in NEGATION_MARKERS)
            if has_causal and not has_negation:
                violations.append({"entity": entity, "context": ctx.strip()})
            start = idx + len(entity_l)

    return {
        "forbidden_causal_claims": violations,
        "hallucination_free": len(violations) == 0,
    }


def score_answer(answer: str, ground_truth_chain: list[str], forbidden_entities: list[str]) -> dict:
    reasoning = score_reasoning_chain(answer, ground_truth_chain)
    hallucination = score_hallucination(answer, forbidden_entities)
    return {
        "reasoning": reasoning,
        "hallucination": hallucination,
        "overall_pass": reasoning["all_hops_present"] and reasoning["correct_order"] and hallucination["hallucination_free"],
    }
