"""
Closed synthetic test corpus. Ground truth is known EXACTLY because we wrote
it - no LLM judge needed to score answers. This mirrors the same principle
already in engine.py::verify_citation: don't trust the model's word for it,
check against something you can verify deterministically.
"""

# The document is deliberately long, with the true causal chain split across
# widely separated sections, and distractor sections in between that mention
# plausible-sounding services that are explicitly NOT involved. Padding
# sections simulate realistic incident-log noise.

DOCUMENT = """
[09:14:02] DEPLOY: session-middleware v2.3 rolled out to all auth-service pods.
[09:14:05] DEPLOY: session-middleware v2.3 changelog: switched connection pooling
           library from urllib3 to httpx.

[09:20:11] INFO: billing-service nightly batch job started.
[09:20:45] INFO: billing-service nightly batch job completed successfully.
[09:31:02] INFO: payment-service processed 118 transactions, p99 latency 41ms.
[09:45:19] INFO: notification-service queue depth: 3 (nominal).
[10:02:44] INFO: billing-service nightly batch job report emailed to finance-team.
[10:15:00] INFO: routine health check passed for all 14 registered services.
[11:00:00] INFO: payment-service processed 204 transactions, p99 latency 38ms.
[12:30:00] INFO: notification-service queue depth: 5 (nominal).
[13:00:00] INFO: scheduled backup completed for postgres_primary.

[14:01:03] ALERT: auth-service connection pool utilization reached 100/100.
[14:01:04] WARN: auth-service connection pool has not released a connection
           in the last 6 minutes.

[14:01:30] INFO: payment-service processed 342 transactions in the last hour.
[14:01:31] INFO: payment-service latency p99: 45ms (nominal, unaffected).
[14:01:50] INFO: billing-service scheduled job queued for 22:00, no anomalies.

[14:02:11] ERROR: api-gateway received 502 Bad Gateway from auth-service
           (upstream timeout).
[14:02:12] ERROR: api-gateway received 502 Bad Gateway from auth-service
           (upstream timeout).
[14:02:15] ERROR: api-gateway circuit breaker OPEN for auth-service.

[14:45:00] NOTE (on-call engineer): Root cause identified. The httpx connection
           pool introduced in session-middleware v2.3 does not release idle
           connections under auth-service's current retry configuration,
           causing a slow connection leak. This exhausted the pool by 14:01,
           which caused api-gateway to receive upstream timeouts starting
           14:02.

[14:45:05] NOTE (on-call engineer): billing-service and payment-service were
           NOT affected and were not investigated further - their connection
           pools use a separate configuration unrelated to session-middleware.
"""

# Ground truth causal chain, in order. An answer that gets this right must
# name every hop, in roughly this order - not just land on the right final
# entity by luck.
GROUND_TRUTH_CHAIN = [
    "session-middleware",
    "httpx",
    "connection leak",
    "auth-service",
    "api-gateway",
]

# Explicitly ruled out in the document itself. A correct answer either
# doesn't mention these, or mentions them only to say they were NOT involved.
# Claiming either one was CAUSALLY involved is a hallucination.
FORBIDDEN_CAUSAL_ENTITIES = ["billing-service", "payment-service"]

QUESTION = "What was the root cause of the api-gateway 502 errors at 14:02, and which services were involved in the chain?"
