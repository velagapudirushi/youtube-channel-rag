"""Pydantic v2 data contracts for the standalone RAG Evaluation Harness."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ClaimVerdict(str, Enum):
    """Structured LLM-as-a-Judge verdict for an atomic claim."""
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIABLE = "UNVERIFIABLE"


class NLILabel(str, Enum):
    """Natural Language Inference (NLI) entailment label."""
    ENTAILMENT = "ENTAILMENT"
    CONTRADICTION = "CONTRADICTION"
    NEUTRAL = "NEUTRAL"


class FailureCategory(str, Enum):
    """Actionable failure taxonomy bucket for RAG debugging."""
    NONE = "NONE"
    RETRIEVAL_MISS = "RETRIEVAL_MISS"
    LOST_IN_THE_MIDDLE = "LOST_IN_THE_MIDDLE"
    INTRINSIC_CONTRADICTION = "INTRINSIC_CONTRADICTION"
    EXTRINSIC_HALLUCINATION = "EXTRINSIC_HALLUCINATION"
    JUDGE_NLI_DISAGREEMENT = "JUDGE_NLI_DISAGREEMENT"
    INCOMPLETE_SYNTHESIS = "INCOMPLETE_SYNTHESIS"


class RetrievedChunk(BaseModel):
    """A single document chunk returned by a RAG retriever."""
    chunk_id: str = Field(..., description="Unique identifier of the chunk or passage")
    text: str = Field(..., description="Raw text content of the retrieved chunk")
    score: float = Field(default=0.0, description="Retriever similarity or reranker score")
    rank: int = Field(default=1, description="1-indexed rank position in retrieval output")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QuestionItem(BaseModel):
    """A benchmark question item in a RAG evaluation dataset."""
    query_id: str = Field(..., description="Unique ID for the evaluation query")
    question: str = Field(..., description="The user query sent to the RAG pipeline")
    ground_truth_answer: str = Field(default="", description="Reference golden answer")
    golden_chunk_ids: List[str] = Field(
        default_factory=list,
        description="IDs of chunks that contain the ground-truth evidence",
    )
    key_facts: List[str] = Field(
        default_factory=list,
        description="Atomic key facts expected in a complete answer",
    )
    category: str = Field(default="general", description="Query domain or difficulty tag")


class RAGTrace(BaseModel):
    """Execution trace produced by running a RAG pipeline on a single query."""
    query_id: str
    question: str
    generated_answer: str
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    latency_ms: float = Field(default=0.0)
    pipeline_name: str = Field(default="default-rag")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalMetrics(BaseModel):
    """Information Retrieval metrics for a single query at cutoff K."""
    k: int = 5
    recall_at_k: float = Field(..., ge=0.0, le=1.0)
    precision_at_k: float = Field(..., ge=0.0, le=1.0)
    mrr: float = Field(..., ge=0.0, le=1.0, description="Reciprocal Rank of first relevant chunk")
    ndcg_at_k: float = Field(..., ge=0.0, le=1.0, description="Normalized Discounted Cumulative Gain")
    hit_rate: float = Field(..., ge=0.0, le=1.0, description="1.0 if at least one golden chunk retrieved")
    context_fact_coverage: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Fraction of reference key facts present in retrieved context text",
    )
    first_relevant_rank: Optional[int] = Field(
        default=None,
        description="1-indexed rank of the first relevant golden chunk (None if missed)",
    )


class NLIProbabilities(BaseModel):
    """Calibrated probability distribution over NLI classes."""
    entailment: float = Field(..., ge=0.0, le=1.0)
    neutral: float = Field(..., ge=0.0, le=1.0)
    contradiction: float = Field(..., ge=0.0, le=1.0)


class ClaimEvaluation(BaseModel):
    """Dual-method verification of a single atomic claim extracted from the answer."""
    claim_text: str
    llm_verdict: ClaimVerdict
    llm_confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    llm_reasoning: str = Field(default="")
    nli_label: NLILabel
    nli_probabilities: NLIProbabilities
    best_chunk_id: Optional[str] = None
    agreement: bool = Field(..., description="True when LLM Judge and NLI model agree on support")
    disagreement_type: Optional[str] = Field(
        default=None,
        description="Type of divergence when LLM Judge and NLI disagree",
    )
    calibrated_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cross-checked faithfulness score combining LLM Judge and NLI",
    )


class FaithfulnessResult(BaseModel):
    """Aggregate faithfulness evaluation for a generated RAG response."""
    llm_judge_score: float = Field(..., ge=0.0, le=1.0)
    nli_entailment_score: float = Field(..., ge=0.0, le=1.0)
    cross_checked_score: float = Field(..., ge=0.0, le=1.0)
    claims: List[ClaimEvaluation] = Field(default_factory=list)
    disagreement_count: int = 0
    has_judge_disagreement: bool = False
    contradiction_count: int = 0
    unverifiable_count: int = 0
    is_abstention: bool = False


class QueryEvalResult(BaseModel):
    """Complete evaluation record for a single RAG query trace."""
    query_id: str
    question: str
    category: str = "general"
    pipeline_name: str
    generated_answer: str
    ground_truth_answer: str = ""
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    golden_chunk_ids: List[str] = Field(default_factory=list)
    retrieval: RetrievalMetrics
    faithfulness: FaithfulnessResult
    answer_completeness: float = Field(default=1.0, ge=0.0, le=1.0)
    failure_category: FailureCategory = FailureCategory.NONE
    failure_explanation: str = ""
    remediation_hint: str = ""
    latency_ms: float = 0.0
    passed: bool = True


class FailureExample(BaseModel):
    """Concise snapshot of a failing query for the Representative Failures report."""
    query_id: str
    question: str
    generated_answer: str
    ground_truth_answer: str
    retrieved_chunk_ids: List[str]
    golden_chunk_ids: List[str]
    first_relevant_rank: Optional[int]
    llm_judge_score: float
    nli_entailment_score: float
    cross_checked_score: float
    flagged_claims: List[ClaimEvaluation]
    failure_explanation: str
    remediation_hint: str


class RepresentativeFailureBucket(BaseModel):
    """Grouped failure taxonomy bucket with representative real trace failures."""
    category: FailureCategory
    title: str
    count: int
    percentage: float
    description: str
    remediation: str
    examples: List[FailureExample] = Field(default_factory=list)


class HarnessReport(BaseModel):
    """Top-level evaluation report returned by the RAG Eval Harness."""
    pipeline_name: str
    dataset_name: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    k: int = 5
    total_queries: int
    passed_queries: int
    pass_rate: float
    mean_recall_at_k: float
    mean_precision_at_k: float
    mean_mrr: float
    mean_ndcg_at_k: float
    mean_hit_rate: float
    mean_context_coverage: float
    mean_llm_faithfulness: float
    mean_nli_faithfulness: float
    mean_cross_checked_faithfulness: float
    mean_answer_completeness: float
    total_claims_evaluated: int
    total_judge_disagreements: int
    disagreement_rate: float
    mean_latency_ms: float
    failure_distribution: Dict[str, int] = Field(default_factory=dict)
    representative_failures: List[RepresentativeFailureBucket] = Field(default_factory=list)
    query_results: List[QueryEvalResult] = Field(default_factory=list)
