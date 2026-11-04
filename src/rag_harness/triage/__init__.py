"""Failure taxonomy and representative failure report builder."""

from rag_harness.triage.classifier import (
    BUCKET_METADATA,
    build_representative_failures,
    classify_failure,
    compute_answer_completeness,
)

__all__ = [
    "BUCKET_METADATA",
    "build_representative_failures",
    "classify_failure",
    "compute_answer_completeness",
]
