"""
Evaluate the YouTube Channel RAG Pipeline using the Standalone RAG Evaluation Harness (`rag-eval-harness`).

Computes:
- Information Retrieval Metrics: Recall@K, Precision@K, MRR, NDCG@K, HitRate@K, Context Fact Coverage
- Dual-Engine Faithfulness: LLM-as-a-Judge × NLI Entailment Cross-Check
- Answer Completeness & Root-Cause Failure Taxonomy
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add youtube-channel-rag and bundled src/rag_harness (plus sibling fallback) to sys.path
PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_SRC_DIR = PROJECT_DIR / "src"
SIBLING_HARNESS_SRC_DIR = PROJECT_DIR.parent / "rag-eval-harness" / "src"

for p in (PROJECT_DIR, LOCAL_SRC_DIR, SIBLING_HARNESS_SRC_DIR):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from channel_rag import (
    SAMPLE_CHANNELS,
    _index_video_list,
    get_chroma_collection,
    query_channel,
)
from rag_harness.adapters import CallablePipelineAdapter
from rag_harness.evaluator import RAGEvaluator
from rag_harness.reporting.markdown import generate_markdown_report
from rag_harness.schemas import HarnessReport, QuestionItem

YOUTUBE_RAG_BENCHMARK_QUESTIONS: list[QuestionItem] = [
    QuestionItem(
        query_id="fcc_01_python_list_tuple",
        question="What does this channel teach about Python lists vs tuples and mutability?",
        ground_truth_answer=(
            "Lists in Python are mutable ordered sequences created with square brackets where elements can be modified, "
            "appended, or removed, whereas tuples are immutable ordered sequences created with parentheses whose elements "
            "cannot be changed once created, making tuples faster and usable as dictionary keys."
        ),
        golden_chunk_ids=["rfscVS0vtbw_chunk_0", "rfscVS0vtbw_chunk_1"],
        key_facts=["mutable", "immutable", "square brackets", "parentheses"],
        category="freeCodeCamp - Python",
    ),
    QuestionItem(
        query_id="fcc_02_javascript_es6",
        question="How are variables declared in modern ES6 JavaScript and what asynchronous features are covered?",
        ground_truth_answer=(
            "In modern ES6 JavaScript, variables are declared using let for block-scoped mutable variables and const for constants "
            "instead of var, and asynchronous API calls use promises and async/await."
        ),
        golden_chunk_ids=["PkZNo7MFNFg_chunk_0"],
        key_facts=["let", "const", "promises", "async/await"],
        category="freeCodeCamp - JavaScript",
    ),
    QuestionItem(
        query_id="fcc_03_ml_paradigms",
        question="What are the three main paradigms of Machine Learning and what supervised classification algorithms are covered?",
        ground_truth_answer=(
            "The three main paradigms are Supervised Learning, Unsupervised Learning, and Reinforcement Learning. "
            "Supervised classification covers K-Nearest Neighbors (KNN), Naive Bayes, Logistic Regression, Support Vector Machines (SVM), and Neural Networks."
        ),
        golden_chunk_ids=["NWONeJKn6kc_chunk_0"],
        key_facts=["Supervised Learning", "Unsupervised Learning", "Reinforcement Learning", "Support Vector Machines"],
        category="freeCodeCamp - Machine Learning",
    ),
    QuestionItem(
        query_id="fcc_04_rag_architecture",
        question="What are the three stages of the core RAG pipeline and what advanced RAG techniques are taught?",
        ground_truth_answer=(
            "The core RAG pipeline consists of Indexing, Retrieval, and Generation. Advanced techniques include "
            "Multi-Query retrieval, RAG-Fusion with Reciprocal Rank Fusion (RRF), HyDE, and query routing."
        ),
        golden_chunk_ids=["sVcwVQRHIc8_chunk_0"],
        key_facts=["Indexing", "Retrieval", "Generation", "Reciprocal Rank Fusion"],
        category="freeCodeCamp - RAG",
    ),
    QuestionItem(
        query_id="karpathy_01_self_attention",
        question="How does self-attention work in a Transformer using Query, Key, and Value vectors?",
        ground_truth_answer=(
            "Every token emits Query, Key, and Value vectors. We compute scaled dot-product attention between Queries and Keys, "
            "apply a causal mask and softmax to get attention weights, and aggregate the Value vectors."
        ),
        golden_chunk_ids=["kCc8FmEb1nY_chunk_0", "kCc8FmEb1nY_chunk_1"],
        key_facts=["Query vector", "Key vector", "Value vector", "scaled dot-product attention"],
        category="Karpathy - Transformers",
    ),
    QuestionItem(
        query_id="karpathy_02_llm_os",
        question="What are the stages of training an LLM and how does an LLM act like an Operating System with RAM and disk?",
        ground_truth_answer=(
            "Training happens in Stage 1 Pre-training and Stage 2 Fine-tuning and Alignment (SFT and RLHF). "
            "In the LLM Operating System analogy, the context window acts as RAM (working memory) and RAG acts as the file system or disk storage."
        ),
        golden_chunk_ids=["zjkBMFhNj_g_chunk_0", "zjkBMFhNj_g_chunk_1"],
        key_facts=["Pre-training", "Fine-tuning", "context window", "RAM"],
        category="Karpathy - LLM Intro",
    ),
    QuestionItem(
        query_id="karpathy_03_bpe_tokenizer",
        question="How does Byte Pair Encoding (BPE) tokenization work in GPT models?",
        ground_truth_answer=(
            "Byte Pair Encoding starts with 256 raw byte tokens and iteratively merges the most frequent adjacent pair of tokens "
            "into new vocabulary tokens until reaching target vocabulary size such as 50,257 for GPT-2 or 100,277 for GPT-4."
        ),
        golden_chunk_ids=["zduSFxRajkE_chunk_0"],
        key_facts=["Byte Pair Encoding", "tokenization", "256", "50,257"],
        category="Karpathy - Tokenization",
    ),
]


def run_youtube_rag_evaluation(k: int = 5) -> HarnessReport:
    """Index both freeCodeCamp and Andrej Karpathy channels into an evaluation ChromaDB collection and run the RAG Eval Harness."""
    eval_col = get_chroma_collection(
        collection_name="youtube_rag_eval_suite",
        persist_directory=None,
        in_memory=True,
    )

    all_videos = (
        SAMPLE_CHANNELS["freeCodeCamp.org (@freecodecamp)"]
        + SAMPLE_CHANNELS["Andrej Karpathy (@AndrejKarpathy)"]
    )
    _index_video_list(
        videos=all_videos,
        channel_url="multi-channel-benchmark",
        chunk_size=140,
        effective_key="",
        embed_mode="local",
        active_collection=eval_col,
    )

    def _rag_callable(question: str) -> dict:
        return query_channel(
            question=question,
            n_results=k,
            api_key="",
            target_collection=eval_col,
            enable_dynamic_channel_search=False,
        )

    adapter = CallablePipelineAdapter(
        fn=_rag_callable,
        pipeline_name="youtube-channel-rag",
    )

    evaluator = RAGEvaluator(k=k)
    report = evaluator.run(
        adapter=adapter,
        questions=YOUTUBE_RAG_BENCHMARK_QUESTIONS,
        dataset_name="youtube-channel-rag-benchmark-v1",
        trace_log_path=PROJECT_DIR / "eval_traces.jsonl",
    )

    md_report = generate_markdown_report(report)
    (PROJECT_DIR / "EVALUATION_REPORT.md").write_text(md_report, encoding="utf-8")
    (PROJECT_DIR / "eval_report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    rep = run_youtube_rag_evaluation(k=5)
    print("\n" + "=" * 78)
    print(f"📊 RAG EVAL HARNESS REPORT — Pipeline: {rep.pipeline_name}")
    print("=" * 78)
    print(f"Pass Rate                   : {rep.pass_rate:.1%} ({rep.passed_queries}/{rep.total_queries} queries)")
    print(f"Mean Recall@{rep.k}               : {rep.mean_recall_at_k:.1%}")
    print(f"Mean MRR                    : {rep.mean_mrr:.3f}")
    print(f"Mean NDCG@{rep.k}                 : {rep.mean_ndcg_at_k:.3f}")
    print(f"Mean Hit Rate@{rep.k}             : {rep.mean_hit_rate:.1%}")
    print(f"Mean Context Fact Coverage  : {rep.mean_context_coverage:.1%}")
    print(f"Mean LLM-Judge Faithfulness : {rep.mean_llm_faithfulness:.1%}")
    print(f"Mean NLI Entailment Score   : {rep.mean_nli_faithfulness:.1%}")
    print(f"Cross-Checked Faithfulness  : {rep.mean_cross_checked_faithfulness:.1%}")
    print(f"Mean Answer Completeness    : {rep.mean_answer_completeness:.1%}")
    print(f"Judge-NLI Disagreements     : {rep.total_judge_disagreements}/{rep.total_claims_evaluated} ({rep.disagreement_rate:.1%})")
    print(f"Mean Query Latency          : {rep.mean_latency_ms:.2f} ms")
    print("=" * 78)
    for qr in rep.query_results:
        status = "✅ PASS" if qr.passed else f"❌ FAIL ({qr.failure_category.value})"
        print(
            f"{status} | {qr.query_id:<26} | Recall@5={qr.retrieval.recall_at_k:.2f} | "
            f"MRR={qr.retrieval.mrr:.2f} | Faithfulness={qr.faithfulness.cross_checked_score:.2f} | "
            f"Completeness={qr.answer_completeness:.2f}"
        )
