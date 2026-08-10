"""
Security & privacy helpers for the Telecom RAG system.

The system is designed to keep telecom data on-prem: retrieval, embedding, and
re-ranking run fully locally; only the final generation step calls the Groq API
(and only the user query + retrieved PUBLIC spec/synthetic context is sent —
never raw private logs unless the operator explicitly indexes them).

This module provides:
  * sanitize_query  — bound input size and strip prompt-injection / control noise
  * redact          — mask common PII/identifier patterns before display/logging
"""
import re

MAX_QUERY_CHARS = 2000

# Phrases that try to override the system instruction (basic prompt-injection guard).
_INJECTION_PATTERNS = [
    re.compile(r"ignore (all|previous|the above) instructions", re.IGNORECASE),
    re.compile(r"disregard (all|previous|the) .*?(instruction|prompt)", re.IGNORECASE),
    re.compile(r"you are now (a|an) ", re.IGNORECASE),
    re.compile(r"system prompt", re.IGNORECASE),
]

# Identifier/PII patterns to redact from displayed/logged output.
_REDACTIONS = [
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[NUMBER]"),          # long digit strings (IMSI/card-like)
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "[IP]"),         # IPv4
]


class SecurityError(ValueError):
    """Raised when an input fails validation."""


def sanitize_query(query: str) -> str:
    """Validate and clean a user query. Raises SecurityError on hard failures."""
    if not isinstance(query, str) or not query.strip():
        raise SecurityError("Query must be a non-empty string.")

    cleaned = query.strip()
    if len(cleaned) > MAX_QUERY_CHARS:
        cleaned = cleaned[:MAX_QUERY_CHARS]

    # Drop control characters (keep normal whitespace).
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", cleaned)

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(cleaned):
            raise SecurityError("Query rejected: possible prompt-injection content.")

    return cleaned


def redact(text: str) -> str:
    """Mask common PII / identifier patterns for safe display and logging."""
    if not text:
        return text
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text
