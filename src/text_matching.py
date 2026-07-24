"""
Shared fuzzy string matching. Used by:
  - engine.py::canonicalize_entity  (best match among many candidates)
  - schema.py::validate_direction   (are these two specific strings the
    same entity, allowing for phrasing drift)

Kept in its own module rather than living in engine.py, because engine.py
already imports from schema.py - schema.py importing back from engine.py
would be a circular import.
"""
import re
from rapidfuzz import fuzz

SEPARATOR_RE = re.compile(r"[\s_\-]+")


def strip_separators(s: str) -> str:
    return SEPARATOR_RE.sub("", s)


def fuzzy_equal(a: str, b: str, threshold: int = 85) -> bool:
    """
    True if a and b refer to the same thing, tolerating the kind of
    phrasing drift a real model produces across two independently
    generated fields in the same response: extra words ("AUTH_SERVICE
    pods" vs "AUTH_SERVICE"), underscore-vs-space boundaries, minor
    rewording, or case differences.

    Two scorers, kept and combined the same way canonicalize_entity does:
    token_ratio is word-order invariant but can't bridge an
    underscore-vs-space boundary; the separator-stripped character ratio
    bridges that boundary but is itself order-sensitive. Neither alone
    covers both cases. Threshold stays conservative (85) deliberately:
    a rejected-but-true fact is a lost opportunity, an accepted-but-false
    one silently corrupts the graph - the second is worse, so ties go to
    rejection.
    """
    a_n, b_n = a.strip().upper(), b.strip().upper()
    if a_n == b_n:
        return True
    token_score = fuzz.token_ratio(a_n, b_n)
    char_score = fuzz.ratio(strip_separators(a_n), strip_separators(b_n))
    return max(token_score, char_score) >= threshold
