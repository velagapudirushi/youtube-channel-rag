"""Main RAG Evaluation Harness orchestrator."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from rag_harness.adapters import RAGPipelineAdapter
from rag_harness.metrics.ensemble import DualFaithfulnessEngine
from rag_harness.metrics.llm_judge import LLMJudgeEvaluator
from rag_harness.metrics.nli_judge import NLIEvaluator
from rag_harness.metrics.retrieval import evaluate_retrieval
from rag_harness.schemas import (
    FailureCategory,
    HarnessReport,
    QuestionItem,
    QueryEvalResult,
    RAGTrace,
)
from rag_harness.triage.classifier import (
    build_representative_failures,
    classify_failure,
    compute_answer_completeness,
)


class RAGEvaluator:
    """Standalone evaluator that grades any RAG pipeline against a question set."""

    def __init__(
        self,
        k: int = 5,
        judge_model: str = "rubric-llm-judge-v1",
        nli_backend: str = "local",
        cache_path: Optional[Path] = None,
        max_failure_examples: int = 3,
    ) -> None:
        self.k = k
        self.max_failure_examples = max_failure_examples
        self.llm_judge = LLMJudgeEvaluator(model=judge_model, cache_path=cache_path)
        self.nli_judge = NLIEvaluator(backend=nli_backend)
        self.faithfulness_engine = DualFaithfulnessEngine(
            llm_judge=self.llm_judge,
            nli_judge=self.nli_judge,
        )

    def evaluate_single(
        self,
        question_item: QuestionItem,
        trace: RAGTrace,
    ) -> QueryEvalResult:
        """Evaluate a single RAG execution trace against its QuestionItem."""
        retrieval = evaluate_retrieval(
            question_item=question_item,
            retrieved_chunks=trace.retrieved_chunks,
            k=self.k,
        )
        faithfulness = self.faithfulness_engine.evaluate(
            generated_answer=trace.generated_answer,
            retrieved_chunks=trace.retrieved_chunks[: self.k],
        )
        completeness = compute_answer_completeness(
            generated_answer=trace.generated_answer,
            question_item=question_item,
        )
        failure_cat, explanation, remediation, passed = classify_failure(
            question_item=question_item,
            retrieved_chunks=trace.retrieved_chunks,
            retrieval=retrieval,
            faithfulness=faithfulness,
            answer_completeness=completeness,
        )

        return QueryEvalResult(
            query_id=question_item.query_id,
            question=question_item.question,
            category=question_item.category,
            pipeline_name=trace.pipeline_name,
            generated_answer=trace.generated_answer,
            ground_truth_answer=question_item.ground_truth_answer,
            retrieved_chunks=trace.retrieved_chunks,
            golden_chunk_ids=question_item.golden_chunk_ids,
            retrieval=retrieval,
            faithfulness=faithfulness,
            answer_completeness=completeness,
            failure_category=failure_cat,
            failure_explanation=explanation,
            remediation_hint=remediation,
            latency_ms=trace.latency_ms,
            passed=passed,
        )

    def run(
        self,
        adapter: RAGPipelineAdapter,
        questions: Sequence[QuestionItem],
        dataset_name: str = "benchmark",
        trace_log_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[int, int, QueryEvalResult], None]] = None,
    ) -> HarnessReport:
        """Execute `adapter` over all `questions`, log traces, and build a full HarnessReport."""
        results: List[QueryEvalResult] = []
        traces: List[RAGTrace] = []

        total = len(questions)
        for idx, item in enumerate(questions, start=1):
            trace = adapter.run_query(item)
            traces.append(trace)
            eval_res = self.evaluate_single(item, trace)
            results.append(eval_res)
            if progress_callback is not None:
                progress_callback(idx, total, eval_res)

        if trace_log_path is not None:
            trace_log_path.parent.mkdir(parents=True, exist_ok=True)
            with trace_log_path.open("w", encoding="utf-8") as f:
                for tr in traces:
                    f.write(tr.model_dump_json() + "\n")

        return self. summarize_results(
            results=results,
            pipeline_name=adapter.pipeline_name,
            dataset_name=dataset_name,
        )

    def summarize_results(
        self,
        results: Sequence[QueryEvalResult],
        pipeline_name: str,
        dataset_name: str = "benchmark",
    ) -> HarnessReport:
        """Aggregate individual QueryEvalResults into a HarnessReport with representative failures."""
        n = max(len(results), 1)
        passed_count = sum(1 for r in results if r.passed)

        total_claims = sum(len(r.faithfulness.claims) for r in results)
        total_disagreements = sum(r.faithfulness.disagreement_count for r in results)
        disagreement_rate = (
            round(total_disagreements / total_claims, 4) if total_claims > 0 else 0.0
        )

        failure_counts = Counter(
            r.failure_category.value
            for r in results
            if r.failure_category != FailureCategory.NONE
        )

        rep_failures = build_representative_failures(
            results,
            max_examples_per_bucket=self.max_failure_examples,
        )

        return HarnessReport(
            pipeline_name=pipeline_name,
            dataset_name=dataset_name,
            k=self.k,
            total_queries=len(results),
            passed_queries=passed_count,
            pass_rate=round(passed_count / n, 4),
            mean_recall_at_k=round(sum(r.retrieval.recall_at_k for r in results) / n, 4),
            mean_precision_at_k=round(sum(r.retrieval.precision_at_k for r in results) / n, 4),
            mean_mrr=round(sum(r.retrieval.mrr for r in results) / n, 4),
            mean_ndcg_at_k=round(sum(r.retrieval.ndcg_at_k for r in results) / n, 4),
            mean_hit_rate=round(sum(r.retrieval.hit_rate for r in results) / n, 4),
            mean_context_coverage=round(sum(r.retrieval.context_fact_coverage for r in results) / n, 4),
            mean_llm_faithfulness=round(sum(r.faithfulness.llm_judge_score for r in results) / n, 4),
            mean_nli_faithfulness=round(sum(r.faithfulness.nli_entailment_score for r in results) / n, 4),
            mean_cross_checked_faithfulness=round(
                sum(r.faithfulness.cross_checked_score for r in results) / n, 4
            ),
            mean_answer_completeness=round(sum(r.answer_completeness for r in results) / n, 4),
            total_claims_evaluated=total_claims,
            total_judge_disagreements=total_disagreements,
            disagreement_rate=disagreement_rate,
            mean_latency_ms=round(sum(r.latency_ms for r in results) / n, 2),
            failure_distribution=dict(failure_counts),
            representative_failures=rep_failures,
            query_results=list(results),
        )
