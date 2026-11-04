"""Metrics package for retrieval and dual-method faithfulness evaluation."""

from rag_harness.metrics.claims import decompose_into_claims, is_abstention_response
from rag_harness.metrics.ensemble import DualFaithfulnessEngine
from rag_harness.metrics.llm_judge import LLMJudgeEvaluator
from rag_harness.metrics.nli_judge import NLIEvaluator
from rag_harness.metrics.retrieval import compute_context_fact_coverage, evaluate_retrieval

__all__ = [
    "decompose_into_claims",
    "is_abstention_response",
    "DualFaithfulnessEngine",
    "LLMJudgeEvaluator",
    "NLIEvaluator",
    "compute_context_fact_coverage",
    "evaluate_retrieval",
]
