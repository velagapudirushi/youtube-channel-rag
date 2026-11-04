"""Root-cause failure taxonomy classifier and representative failure report builder."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from rag_harness.metrics.retrieval import _normalize_text
from rag_harness.schemas import (
    FailureCategory,
    FailureExample,
    FaithfulnessResult,
    QuestionItem,
    QueryEvalResult,
    RepresentativeFailureBucket,
    RetrievalMetrics,
    RetrievedChunk,
)

BUCKET_METADATA: Dict[FailureCategory, Dict[str, str]] = {
    FailureCategory.JUDGE_NLI_DISAGREEMENT: {
        "title": "Judge-NLI Disagreement (LLM Judge Fooled by Plausible Hallucination)",
        "description": (
            "The LLM-as-a-Judge marked the claim SUPPORTED due to passage-level keyword overlap, "
            "but the sentence-aligned NLI Entailment cross-encoder caught a contradiction or scope swap."
        ),
        "remediation": (
            "Enforce sentence-level citation grounding in the generation prompt and gate CI on "
            "NLI entailment cross-checks rather than single-model LLM judge scores."
        ),
    },
    FailureCategory.INTRINSIC_CONTRADICTION: {
        "title": "Intrinsic Contradiction (Misread Numbers, Negations, or Tiers)",
        "description": (
            "The retriever fetched the relevant passage, but the generator misread a numeric figure, "
            "inverted a negation polarity, or confused policy rules across adjacent tiers."
        ),
        "remediation": (
            "Keep structured tables/policies intact during chunking (semantic/markdown-header chunking) "
            "and instruct the generator to verify entity scope before stating numbers."
        ),
    },
    FailureCategory.EXTRINSIC_HALLUCINATION: {
        "title": "Extrinsic Hallucination (Unsupported Facts Invented Outside Context)",
        "description": (
            "The generator fabricated specific claims, commands, or parameters that do not exist "
            "anywhere in the retrieved context chunks."
        ),
        "remediation": (
            "Add strict abstention instructions ('Answer ONLY using provided chunks; if missing, state "
            "what cannot be verified') and lower generation temperature to 0.0."
        ),
    },
    FailureCategory.RETRIEVAL_MISS: {
        "title": "Retrieval Miss (Golden Evidence Absent from Top-K Chunks)",
        "description": (
            "The retriever failed to surface the ground-truth document chunk in the top-K results, "
            "starving the generator of required facts."
        ),
        "remediation": (
            "Switch from pure dense embeddings to Hybrid Search (BM25 + Dense with Reciprocal Rank Fusion) "
            "and add query expansion for exact error codes, CLI flags, and acronyms."
        ),
    },
    FailureCategory.LOST_IN_THE_MIDDLE: {
        "title": "Context Ignored / Lost-in-the-Middle (Evidence Retrieved but Unused)",
        "description": (
            "The relevant golden chunk was successfully retrieved in the top-K window (often at rank 3–5), "
            "but the generator ignored those facts or falsely abstained."
        ),
        "remediation": (
            "Add a Cross-Encoder Reranker (e.g., BGE-Reranker or Cohere Rerank) to push the golden chunk "
            "to Rank 1–2 and trim low-relevance distractor chunks from the prompt."
        ),
    },
    FailureCategory.INCOMPLETE_SYNTHESIS: {
        "title": "Incomplete Multi-Chunk Synthesis (Partial Answer)",
        "description": (
            "The query required combining facts across multiple chunks, and the generator answered "
            "only part of the question while omitting critical constraints."
        ),
        "remediation": (
            "Increase top-K after reranking or use decompose-then-synthesize prompting for multi-hop queries."
        ),
    },
}


def compute_answer_completeness(
    generated_answer: str,
    question_item: QuestionItem,
) -> float:
    """Compute fraction of reference key facts present in the generated answer."""
    if not generated_answer.strip():
        return 0.0
    if not question_item.key_facts:
        return 1.0

    norm_ans = _normalize_text(generated_answer.replace("-", " "))
    matched = 0
    for fact in question_item.key_facts:
        norm_fact = _normalize_text(fact.replace("-", " "))
        if not norm_fact:
            continue
        if norm_fact in norm_ans:
            matched += 1
        else:
            tokens = [t.rstrip("s") for t in norm_fact.split() if len(t) > 2 or any(c.isdigit() for c in t)]
            if tokens and sum(1 for t in tokens if t in norm_ans) / len(tokens) >= 0.75:
                matched += 1

    return round(matched / len(question_item.key_facts), 4)


def classify_failure(
    question_item: QuestionItem,
    retrieved_chunks: Sequence[RetrievedChunk],
    retrieval: RetrievalMetrics,
    faithfulness: FaithfulnessResult,
    answer_completeness: float,
) -> Tuple[FailureCategory, str, str, bool]:
    """Assign a root-cause FailureCategory, diagnostic explanation, remediation hint, and pass/fail."""
    # 1. Check for critical Judge-NLI disagreement where LLM Judge missed a contradiction or unverified claim
    critical_disagreements = [
        c
        for c in faithfulness.claims
        if not c.agreement
        and c.disagreement_type in ("LLM_MISSED_CONTRADICTION", "LLM_OVERCONFIDENT_UNVERIFIED")
    ]
    if critical_disagreements:
        first_c = critical_disagreements[0]
        meta = BUCKET_METADATA[FailureCategory.JUDGE_NLI_DISAGREEMENT]
        explanation = (
            f"LLM Judge scored claim as {first_c.llm_verdict.value}, but NLI Cross-Encoder flagged "
            f"{first_c.nli_label.value} (p_contra={first_c.nli_probabilities.contradiction:.2f}, "
            f"p_neutral={first_c.nli_probabilities.neutral:.2f}) on: \"{first_c.claim_text}\""
        )
        return FailureCategory.JUDGE_NLI_DISAGREEMENT, explanation, meta["remediation"], False

    # 2. Check for Intrinsic Contradiction (both or NLI detected contradiction when relevant chunk was retrieved)
    if faithfulness.contradiction_count > 0:
        contra_claims = [
            c for c in faithfulness.claims if c.nli_label.value == "CONTRADICTION" or c.llm_verdict.value == "CONTRADICTED"
        ]
        sample = contra_claims[0].claim_text if contra_claims else ""
        meta = BUCKET_METADATA[FailureCategory.INTRINSIC_CONTRADICTION]
        explanation = (
            f"Generated answer contradicts retrieved evidence ({faithfulness.contradiction_count} contradicted "
            f"claim(s)): \"{sample}\""
        )
        return FailureCategory.INTRINSIC_CONTRADICTION, explanation, meta["remediation"], False

    # 3. Check for Retrieval Miss (golden chunks missed in top-K)
    if retrieval.hit_rate == 0.0 or (retrieval.recall_at_k == 0.0 and retrieval.context_fact_coverage < 0.5):
        meta = BUCKET_METADATA[FailureCategory.RETRIEVAL_MISS]
        if faithfulness.is_abstention:
            explanation = (
                f"Retriever missed golden chunk(s) {question_item.golden_chunk_ids}; generator faithfully "
                f"abstained, but user query went unanswered."
            )
        elif faithfulness.unverifiable_count > 0 or faithfulness.cross_checked_score < 0.75:
            explanation = (
                f"Retriever missed golden chunk(s) {question_item.golden_chunk_ids} (Recall@{retrieval.k}=0.00), "
                f"causing generator to hallucinate unsupported facts (Faithfulness={faithfulness.cross_checked_score:.2f})."
            )
        else:
            explanation = (
                f"Retriever failed to fetch golden chunk(s) {question_item.golden_chunk_ids} in top-{retrieval.k} "
                f"(retrieved: {[c.chunk_id for c in retrieved_chunks[:retrieval.k]]})."
            )
        return FailureCategory.RETRIEVAL_MISS, explanation, meta["remediation"], False

    # 4. Check for Extrinsic Hallucination (relevant chunk retrieved, but model still invented extra unsupported facts)
    if faithfulness.unverifiable_count > 0 and faithfulness.cross_checked_score < 0.75:
        unver_claims = [c for c in faithfulness.claims if c.calibrated_score < 0.5]
        sample = unver_claims[0].claim_text if unver_claims else ""
        meta = BUCKET_METADATA[FailureCategory.EXTRINSIC_HALLUCINATION]
        explanation = (
            f"Generator fabricated {faithfulness.unverifiable_count} unverifiable claim(s) not grounded in "
            f"retrieved context: \"{sample}\""
        )
        return FailureCategory.EXTRINSIC_HALLUCINATION, explanation, meta["remediation"], False

    # 5. Check for Lost-in-the-Middle / Context Ignored
    if retrieval.hit_rate > 0.0 and (
        faithfulness.is_abstention
        or (answer_completeness < 0.65 and (retrieval.first_relevant_rank or 1) >= 2)
    ):
        meta = BUCKET_METADATA[FailureCategory.LOST_IN_THE_MIDDLE]
        rank_str = f"rank {retrieval.first_relevant_rank}" if retrieval.first_relevant_rank else "top-K"
        explanation = (
            f"Golden evidence was retrieved at {rank_str}, but generator ignored key facts "
            f"(completeness={answer_completeness:.0%}, abstained={faithfulness.is_abstention})."
        )
        return FailureCategory.LOST_IN_THE_MIDDLE, explanation, meta["remediation"], False

    # 6. Check for Incomplete Multi-Chunk Synthesis
    if retrieval.recall_at_k < 1.0 or answer_completeness < 0.70:
        meta = BUCKET_METADATA[FailureCategory.INCOMPLETE_SYNTHESIS]
        explanation = (
            f"Partial synthesis: Recall@{retrieval.k}={retrieval.recall_at_k:.2f}, "
            f"Answer Fact Completeness={answer_completeness:.0%}."
        )
        return FailureCategory.INCOMPLETE_SYNTHESIS, explanation, meta["remediation"], False

    return (
        FailureCategory.NONE,
        "All retrieval and dual-judge faithfulness checks passed.",
        "",
        True,
    )


def build_representative_failures(
    query_results: Sequence[QueryEvalResult],
    max_examples_per_bucket: int = 3,
) -> List[RepresentativeFailureBucket]:
    """Group failed queries into root-cause buckets with worst-K representative examples."""
    failed = [r for r in query_results if not r.passed and r.failure_category != FailureCategory.NONE]
    if not failed:
        return []

    grouped: Dict[FailureCategory, List[QueryEvalResult]] = defaultdict(list)
    for res in failed:
        grouped[res.failure_category].append(res)

    total_queries = max(len(query_results), 1)
    buckets: List[RepresentativeFailureBucket] = []

    # Order by frequency (descending), breaking ties by severity
    severity_order = [
        FailureCategory.JUDGE_NLI_DISAGREEMENT,
        FailureCategory.INTRINSIC_CONTRADICTION,
        FailureCategory.EXTRINSIC_HALLUCINATION,
        FailureCategory.RETRIEVAL_MISS,
        FailureCategory.LOST_IN_THE_MIDDLE,
        FailureCategory.INCOMPLETE_SYNTHESIS,
    ]

    sorted_categories = sorted(
        grouped.keys(),
        key=lambda cat: (-len(grouped[cat]), severity_order.index(cat) if cat in severity_order else 99),
    )

    for cat in sorted_categories:
        items = grouped[cat]
        # Sort examples so the lowest cross_checked_score / MRR appear first
        items_sorted = sorted(
            items,
            key=lambda x: (
                -x.faithfulness.disagreement_count,
                x.faithfulness.cross_checked_score,
                x.retrieval.mrr,
            ),
        )
        examples: List[FailureExample] = []
        for item in items_sorted[:max_examples_per_bucket]:
            flagged = [
                c
                for c in item.faithfulness.claims
                if not c.agreement or c.calibrated_score < 0.70
            ]
            examples.append(
                FailureExample(
                    query_id=item.query_id,
                    question=item.question,
                    generated_answer=item.generated_answer,
                    ground_truth_answer=item.ground_truth_answer,
                    retrieved_chunk_ids=[c.chunk_id for c in item.retrieved_chunks],
                    golden_chunk_ids=item.golden_chunk_ids,
                    first_relevant_rank=item.retrieval.first_relevant_rank,
                    llm_judge_score=item.faithfulness.llm_judge_score,
                    nli_entailment_score=item.faithfulness.nli_entailment_score,
                    cross_checked_score=item.faithfulness.cross_checked_score,
                    flagged_claims=flagged,
                    failure_explanation=item.failure_explanation,
                    remediation_hint=item.remediation_hint,
                )
            )

        meta = BUCKET_METADATA.get(
            cat,
            {
                "title": cat.value,
                "description": "Pipeline failed evaluation checks.",
                "remediation": "Inspect trace logs.",
            },
        )
        buckets.append(
            RepresentativeFailureBucket(
                category=cat,
                title=meta["title"],
                count=len(items),
                percentage=round(100.0 * len(items) / total_queries, 1),
                description=meta["description"],
                remediation=meta["remediation"],
                examples=examples,
            )
        )

    return buckets
