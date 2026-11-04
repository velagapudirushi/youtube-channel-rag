"""rag-eval-harness: Standalone, framework-agnostic RAG Evaluation CLI & Python Library."""

from rag_harness.adapters import (
    CallablePipelineAdapter,
    HTTPPipelineAdapter,
    TraceFileAdapter,
    load_python_pipeline,
    load_question_set,
)
from rag_harness.evaluator import RAGEvaluator
from rag_harness.metrics import (
    DualFaithfulnessEngine,
    LLMJudgeEvaluator,
    NLIEvaluator,
    decompose_into_claims,
    evaluate_retrieval,
)
from rag_harness.schemas import (
    ClaimEvaluation,
    ClaimVerdict,
    FailureCategory,
    FaithfulnessResult,
    HarnessReport,
    NLILabel,
    QuestionItem,
    QueryEvalResult,
    RAGTrace,
    RepresentativeFailureBucket,
    RetrievalMetrics,
    RetrievedChunk,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "CallablePipelineAdapter",
    "HTTPPipelineAdapter",
    "TraceFileAdapter",
    "load_python_pipeline",
    "load_question_set",
    "RAGEvaluator",
    "DualFaithfulnessEngine",
    "LLMJudgeEvaluator",
    "NLIEvaluator",
    "decompose_into_claims",
    "evaluate_retrieval",
    "ClaimEvaluation",
    "ClaimVerdict",
    "FailureCategory",
    "FaithfulnessResult",
    "HarnessReport",
    "NLILabel",
    "QuestionItem",
    "QueryEvalResult",
    "RAGTrace",
    "RepresentativeFailureBucket",
    "RetrievalMetrics",
    "RetrievedChunk",
]
