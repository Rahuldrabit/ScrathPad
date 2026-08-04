"""
Prove the scorer itself works before trusting it on real model output.
Four hand-constructed answers, each representing a distinct failure mode
we expect real SLM answers to fall into.
"""
from synthetic_incident import GROUND_TRUTH_CHAIN, FORBIDDEN_CAUSAL_ENTITIES
from scoring import score_answer

CASES = {
    "GOOD (full correct reasoning)": """
        The root cause was the session-middleware v2.3 deployment, which
        introduced a new httpx-based connection pool. That pool leaked
        connections under auth-service's retry settings, causing a
        connection leak that exhausted the auth-service pool by 14:01.
        This in turn caused api-gateway to receive upstream timeouts
        (502 errors) starting at 14:02. billing-service and payment-service
        were not affected.
    """,
    "HALLUCINATED (invents a forbidden causal link)": """
        The api-gateway 502 errors were caused by auth-service pool
        exhaustion, which was triggered by billing-service consuming
        too many shared connections during its batch job.
    """,
    "LOST IN THE MIDDLE (misses the root cause, only sees local symptoms)": """
        The api-gateway received 502 errors because auth-service's
        connection pool hit 100% utilization and stopped releasing
        connections around 14:01.
    """,
    "OUT OF ORDER (right entities, reversed/incoherent causality)": """
        api-gateway's 502 errors caused auth-service's connection pool to
        become exhausted, which happened because of a connection leak
        related to httpx in session-middleware.
    """,
}

for label, answer in CASES.items():
    result = score_answer(answer, GROUND_TRUTH_CHAIN, FORBIDDEN_CAUSAL_ENTITIES)
    print(f"=== {label} ===")
    print(f"  chain_completeness : {result['reasoning']['chain_completeness']:.0%}  "
          f"({result['reasoning']['hops_found']}/{result['reasoning']['total_hops']} hops)")
    print(f"  missing_hops       : {result['reasoning']['missing_hops']}")
    print(f"  correct_order      : {result['reasoning']['correct_order']}")
    print(f"  hallucination_free : {result['hallucination']['hallucination_free']}")
    if result['hallucination']['forbidden_causal_claims']:
        print(f"  violation detected : {result['hallucination']['forbidden_causal_claims'][0]['entity']!r} "
              f"linked causally in: \"...{result['hallucination']['forbidden_causal_claims'][0]['context'][:80]}...\"")
    print(f"  OVERALL PASS       : {result['overall_pass']}")
    print()
