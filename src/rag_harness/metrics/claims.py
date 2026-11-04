"""Atomic claim decomposition and abstention detection for RAG answers."""

from __future__ import annotations

import re
from typing import List

ABSTENTION_PATTERNS = [
    r"\bi\s+(?:do\s+not|don't)\s+(?:have\s+enough\s+information|know)\b",
    r"\bnot\s+(?:mentioned|provided|found|specified|stated)\s+in\s+the\s+(?:provided\s+)?(?:context|documents|passages)\b",
    r"\bcannot\s+(?:be\s+determined|answer|verify)\s+from\s+the\s+(?:provided\s+)?context\b",
    r"\binsufficient\s+(?:context|information|evidence)\b",
]


def is_abstention_response(answer: str) -> bool:
    """Return True if the RAG response explicitly abstains due to missing context."""
    text = answer.strip().lower()
    if not text:
        return True
    return any(re.search(pattern, text) for pattern in ABSTENTION_PATTERNS)


def decompose_into_claims(answer: str) -> List[str]:
    """Split a generated RAG answer into atomic, verifiable factual statements.

    Handles:
    - Markdown bullet lists and numbered lists
    - Multi-sentence paragraphs (preserving decimal numbers and common abbreviations)
    - Compound sentences joined by coordinating conjunctions when both clauses carry facts
    """
    cleaned = answer.strip()
    if not cleaned:
        return []

    # Protect decimals and common abbreviations before sentence splitting
    protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", cleaned)
    for abbr in ("e.g.", "i.e.", "vs.", "Mr.", "Dr.", "Inc.", "Ltd.", "v."):
        protected = protected.replace(abbr, abbr.replace(".", "<DOT>"))

    # Split on newlines / bullet points first, then sentence boundaries
    raw_lines = re.split(r"[\r\n]+", protected)
    sentences: List[str] = []

    for line in raw_lines:
        line = re.sub(r"^\s*(?:[\-\*\•]|\d+[\.\)])\s*", "", line).strip()
        if not line:
            continue
        parts = re.split(r"(?<=[\.\!\?])\s+|;\s+", line)
        for part in parts:
            restored = part.replace("<DOT>", ".").strip()
            if restored:
                sentences.append(restored)

    atomic_claims: List[str] = []
    for sent in sentences:
        # Split compound factual clauses connected by ", while ", ", whereas ", ", and it ", " but "
        subclauses = re.split(
            r",?\s+(?:while|whereas|but|however)\s+|,\s+and\s+(?=(?:it|they|the|enterprise|standard|pro|users|customers|requests)\b)",
            sent,
            flags=re.IGNORECASE,
        )
        for clause in subclauses:
            clause = clause.strip().rstrip(".,;:")
            if len(clause.split()) >= 3:
                atomic_claims.append(clause + ".")

    return atomic_claims if atomic_claims else [cleaned]
