"""Demo pipelines and benchmark corpus for public RAG architecture evaluation."""

from rag_harness.demos.corpus import BENCHMARK_QUESTIONS, KNOWLEDGE_CORPUS
from rag_harness.demos.public_repos import (
    PUBLIC_REPO_PIPELINES,
    CorrectiveSelfRAGPipeline,
    HybridBM25RerankRAGPipeline,
    NaiveChromaRAGPipeline,
)

__all__ = [
    "BENCHMARK_QUESTIONS",
    "KNOWLEDGE_CORPUS",
    "PUBLIC_REPO_PIPELINES",
    "CorrectiveSelfRAGPipeline",
    "HybridBM25RerankRAGPipeline",
    "NaiveChromaRAGPipeline",
]
