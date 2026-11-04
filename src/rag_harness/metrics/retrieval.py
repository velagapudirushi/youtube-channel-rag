"""Information Retrieval metrics engine for RAG evaluation (Recall@K, Precision@K, MRR, NDCG@K)."""

from __future__ import annotations

import math
import re
from typing import List, Sequence

from rag_harness.schemas import QuestionItem, RetrievalMetrics, RetrievedChunk


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def compute_context_fact_coverage(
    chunks: Sequence[RetrievedChunk],
    key_facts: Sequence[str],
    ground_truth_answer: str = "",
) -> float:
    """Compute what fraction of reference key facts appear in the retrieved context."""
    if not chunks:
        return 0.0

    combined_context = _normalize_text(" ".join(c.text for c in chunks))
    facts_to_check = list(key_facts)

    if not facts_to_check and ground_truth_answer:
        # Extract salient numbers, quoted terms, or 4+ char content tokens from ground truth
        tokens = [
            tok
            for tok in re.findall(r"\b[a-zA-Z0-9_\-\$%\.]+\b", ground_truth_answer.lower())
            if len(tok) >= 4 or any(ch.isdigit() for ch in tok)
        ]
        facts_to_check = tokens[:8]

    if not facts_to_check:
        return 1.0

    matched = 0
    for fact in facts_to_check:
        norm_fact = _normalize_text(fact)
        if not norm_fact:
            continue
        if norm_fact in combined_context:
            matched += 1
        else:
            # Check token-level coverage for multi-word key facts
            fact_tokens = [
                t for t in re.findall(r"\b[a-z0-9\.\-%\$]+\b", norm_fact) if len(t) > 2 or any(c.isdigit() for c in t)
            ]
            if fact_tokens and sum(1 for t in fact_tokens if t in combined_context) / len(fact_tokens) >= 0.75:
                matched += 1

    return round(matched / len(facts_to_check), 4)


def evaluate_retrieval(
    question_item: QuestionItem,
    retrieved_chunks: Sequence[RetrievedChunk],
    k: int = 5,
) -> RetrievalMetrics:
    """Compute Recall@K, Precision@K, MRR, NDCG@K, HitRate@K, and Context Fact Coverage."""
    top_k_chunks = list(retrieved_chunks[:k])
    golden_set = set(question_item.golden_chunk_ids)

    fact_coverage = compute_context_fact_coverage(
        top_k_chunks,
        question_item.key_facts,
        question_item.ground_truth_answer,
    )

    # If the dataset does not provide explicit golden_chunk_ids, fall back to fact coverage heuristics
    if not golden_set:
        inferred_hit = 1.0 if fact_coverage >= 0.6 else 0.0
        return RetrievalMetrics(
            k=k,
            recall_at_k=fact_coverage,
            precision_at_k=round(fact_coverage / max(len(top_k_chunks), 1), 4) if inferred_hit else 0.0,
            mrr=inferred_hit,
            ndcg_at_k=fact_coverage,
            hit_rate=inferred_hit,
            context_fact_coverage=fact_coverage,
            first_relevant_rank=1 if inferred_hit else None,
        )

    if not top_k_chunks:
        return RetrievalMetrics(
            k=k,
            recall_at_k=0.0,
            precision_at_k=0.0,
            mrr=0.0,
            ndcg_at_k=0.0,
            hit_rate=0.0,
            context_fact_coverage=0.0,
            first_relevant_rank=None,
        )

    # Track binary relevance at each rank position (1..k)
    relevances: List[int] = []
    seen_golden: set[str] = set()
    first_relevant_rank: int | None = None

    for idx, chunk in enumerate(top_k_chunks, start=1):
        is_rel = chunk.chunk_id in golden_set and chunk.chunk_id not in seen_golden
        if is_rel:
            seen_golden.add(chunk.chunk_id)
            relevances.append(1)
            if first_relevant_rank is None:
                first_relevant_rank = idx
        else:
            relevances.append(0)

    num_relevant_retrieved = sum(relevances)
    recall_at_k = num_relevant_retrieved / len(golden_set)
    precision_at_k = num_relevant_retrieved / len(top_k_chunks)
    mrr = (1.0 / first_relevant_rank) if first_relevant_rank is not None else 0.0
    hit_rate = 1.0 if num_relevant_retrieved > 0 else 0.0

    # Compute DCG@K and IDCG@K
    dcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevances, start=1))
    ideal_hits = min(len(golden_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg_at_k = (dcg / idcg) if idcg > 0 else 0.0

    return RetrievalMetrics(
        k=k,
        recall_at_k=round(recall_at_k, 4),
        precision_at_k=round(precision_at_k, 4),
        mrr=round(mrr, 4),
        ndcg_at_k=round(ndcg_at_k, 4),
        hit_rate=round(hit_rate, 4),
        context_fact_coverage=fact_coverage,
        first_relevant_rank=first_relevant_rank,
    )
