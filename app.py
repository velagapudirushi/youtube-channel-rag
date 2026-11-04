"""
Step 5 - Streamlit UI for YouTube Channel RAG (`app.py`)
Works with ANY YouTube channel (e.g., freeCodeCamp, Andrej Karpathy, Fireship, 3Blue1Brown, or any custom channel/playlist/video URL)
and includes live integration with the Standalone RAG Evaluation Harness (`rag-eval-harness`).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_SRC_DIR = PROJECT_DIR / "src"
SIBLING_HARNESS_SRC_DIR = PROJECT_DIR.parent / "rag-eval-harness" / "src"
for p in (PROJECT_DIR, LOCAL_SRC_DIR, SIBLING_HARNESS_SRC_DIR):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from channel_rag import (
    SAMPLE_CHANNELS,
    _normalize_channel_url,
    clear_collection,
    collection,
    extract_content_keywords,
    index_channel,
    index_sample_channel,
    is_valid_openai_key,
    query_channel,
)
from evaluate_with_harness import run_youtube_rag_evaluation
from rag_harness.adapters import _coerce_chunks
from rag_harness.evaluator import RAGEvaluator
from rag_harness.reporting.markdown import generate_markdown_report
from rag_harness.schemas import QuestionItem, RAGTrace

st.set_page_config(
    page_title="YouTube Channel RAG + Eval Harness",
    page_icon="🎥",
    layout="wide",
)

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ RAG Configuration")

    raw_env_key = os.environ.get("OPENAI_API_KEY", "")
    default_key = raw_env_key if is_valid_openai_key(raw_env_key) else ""

    api_key_input = st.text_input(
        "OpenAI API Key (Optional)",
        value=default_key,
        type="password",
        placeholder="Leave blank for Free Local Mode, or paste real sk-...",
        help="If a valid OpenAI API key is provided, uses `text-embedding-3-small` + `gpt-4o-mini`. Otherwise runs 100% locally with zero API cost.",
    )

    active_api_key = api_key_input.strip() if is_valid_openai_key(api_key_input) else None

    if api_key_input.strip() and not active_api_key:
        st.warning(
            "⚠️ Placeholder or incomplete key detected (`sk-...`). Automatically running in **Free Local Mode** so your queries never fail with 401 errors."
        )

    if active_api_key:
        st.success("🟢 Mode: **OpenAI** (`text-embedding-3-small` + `gpt-4o-mini`)")
    else:
        st.info("🔵 Mode: **Free Local Mode** (384-d Stemmed Embeddings + Hybrid Reranking)")

    st.divider()
    st.subheader("Indexing & Retrieval Settings")
    max_videos = st.slider(
        "Max Videos to Index (`playlistend`)",
        min_value=1,
        max_value=50,
        value=5,
        help="Number of recent videos to fetch when indexing a live YouTube channel.",
    )
    chunk_size = st.slider(
        "Chunk Size (words)",
        min_value=100,
        max_value=1000,
        value=500,
        step=50,
        help="Transcripts are split into overlapping chunks (50-word overlap).",
    )
    top_k = st.slider(
        "Top-K Retrieved Chunks (`n_results`)",
        min_value=1,
        max_value=10,
        value=5,
    )
    replace_existing = st.checkbox(
        "Replace previous channel when indexing a new channel",
        value=True,
        help="Clears older channel chunks so answers only come from the newly indexed channel.",
    )

    # Pre-seed default collection on fresh container startup so ChromaDB is never empty
    if collection.count() == 0 and not st.session_state.get("user_cleared"):
        init_stats = index_sample_channel(
            preset_name="freeCodeCamp.org (@freecodecamp)",
            api_key=active_api_key,
            replace_existing=False,
        )
        st.session_state["last_index_stats"] = init_stats
        st.session_state["indexed_channel_url"] = _normalize_channel_url("https://www.youtube.com/@freecodecamp/videos")

    st.divider()
    st.subheader("🗄️ ChromaDB Status")
    chunk_count = collection.count()
    st.metric("Indexed Chunks in ChromaDB", chunk_count)

    if st.button("🗑️ Clear ChromaDB Collection", use_container_width=True):
        clear_collection(collection)
        st.session_state["user_cleared"] = True
        st.session_state.pop("last_index_stats", None)
        st.session_state.pop("indexed_channel_url", None)
        st.rerun()

# --- Main Header ---
st.title("🎥 YouTube Channel RAG")
st.caption(
    "Paste **any** YouTube channel URL (`@freecodecamp`, `@AndrejKarpathy`, `@fireship`, `@3blue1brown`, playlists, or video URLs), "
    "ask questions across the channel, and verify every answer with your **RAG Eval Harness (`LLM-as-a-Judge` × `NLI Entailment`)**."
)

tab_rag, tab_eval = st.tabs(["🎥 Channel RAG Search & Q&A", "📊 RAG Eval Harness Benchmark Suite"])

with tab_rag:
    # --- Step 3: Index Channel Section ---
    st.subheader("1️⃣ Paste Any YouTube Channel URL")

    col_url, col_preset = st.columns([3, 2])
    with col_url:
        channel_url = st.text_input(
            "Channel URL or @Handle",
            value=st.session_state.get("channel_url_input", "https://www.youtube.com/@freecodecamp/videos"),
            placeholder="https://www.youtube.com/@freecodecamp/videos or @channelname",
        )
        st.session_state["channel_url_input"] = channel_url

    with col_preset:
        preset_choice = st.selectbox(
            "Or Instant Sample Preset (Zero-Wait)",
            options=list(SAMPLE_CHANNELS.keys()),
            index=0,
        )
        load_sample_clicked = st.button(
            f"⚡ Load Instant Sample: {preset_choice.split(' ')[0]}",
            use_container_width=True,
        )

    def _run_live_or_fallback_index(target_url: str) -> dict:
        """Index any live YouTube channel via yt-dlp + youtube-transcript-api, with automatic sample fallback if offline."""
        progress_bar = st.progress(0, text=f"Fetching videos from {target_url} ...")

        def _on_progress(current: int, total: int, video_id: str, title: str) -> None:
            pct = int((current / max(total, 1)) * 100)
            progress_bar.progress(pct, text=f"Indexing video {current}/{total}: {title} ({video_id})")

        try:
            stats = index_channel(
                channel_url=target_url,
                max_videos=max_videos,
                chunk_size=chunk_size,
                api_key=active_api_key,
                replace_existing=replace_existing,
                progress_callback=_on_progress,
            )
            progress_bar.empty()
            if stats["videos_indexed"] > 0:
                st.session_state["last_index_stats"] = stats
                st.session_state["indexed_channel_url"] = _normalize_channel_url(target_url)
                st.success(
                    f"✅ Indexed **{stats['videos_indexed']}** videos from `{stats['channel_url']}` "
                    f"({stats['total_chunks']} transcript chunks stored in ChromaDB)."
                )
                return stats
            else:
                st.warning(
                    f"Fetched {stats['videos_found']} videos from `{target_url}`, but no public transcripts were returned. "
                    "Loading built-in sample dataset as fallback..."
                )
        except Exception as exc:
            progress_bar.empty()
            st.warning(f"Live YouTube fetch encountered an error ({exc}). Falling back to built-in sample transcripts...")

        fallback_preset = (
            "freeCodeCamp.org (@freecodecamp)"
            if "freecodecamp" in target_url.lower()
            else "Andrej Karpathy (@AndrejKarpathy)"
        )
        stats = index_sample_channel(
            preset_name=fallback_preset,
            api_key=active_api_key,
            replace_existing=replace_existing,
        )
        st.session_state["last_index_stats"] = stats
        st.session_state["indexed_channel_url"] = _normalize_channel_url(target_url)
        return stats

    if st.button("🎥 Index Channel", type="primary") and channel_url:
        with st.spinner(f"Indexing `{channel_url}`... fetching video list & transcripts"):
            _run_live_or_fallback_index(channel_url)

    if load_sample_clicked:
        with st.spinner(f"Indexing `{preset_choice}` sample transcripts into ChromaDB..."):
            stats = index_sample_channel(
                preset_name=preset_choice,
                api_key=active_api_key,
                replace_existing=replace_existing,
            )
            st.session_state["last_index_stats"] = stats
            if "freecodecamp" in preset_choice.lower():
                st.session_state["channel_url_input"] = "https://www.youtube.com/@freecodecamp/videos"
                st.session_state["indexed_channel_url"] = _normalize_channel_url("https://www.youtube.com/@freecodecamp/videos")
            else:
                st.session_state["channel_url_input"] = "https://www.youtube.com/@AndrejKarpathy/videos"
                st.session_state["indexed_channel_url"] = _normalize_channel_url("https://www.youtube.com/@AndrejKarpathy/videos")
            st.success(
                f"✅ Indexed **{stats['videos_indexed']}** videos from **{preset_choice}** "
                f"({stats['total_chunks']} chunks) into ChromaDB!"
            )

    if "last_index_stats" in st.session_state:
        stats = st.session_state["last_index_stats"]
        with st.expander(
            f"📋 Indexed Channel Videos ({stats['videos_indexed']} videos, {stats['total_chunks']} chunks from `{stats['channel_url']}`)",
            expanded=False,
        ):
            for vid in stats.get("indexed_videos", []):
                st.markdown(
                    f"- **[{vid['title']}]({vid['url']})** (`{vid['video_id']}`) — "
                    f"`{vid['words']}` words • `{vid['chunks']}` chunks"
                )

    st.divider()

    # --- Step 4: Query the Channel Section ---
    st.subheader("2️⃣ Ask Anything Across the Channel")

    example_questions = [
        "What does this channel teach about Python list vs tuple?",
        "Explain Supervised vs Unsupervised vs Reinforcement Learning from this channel",
        "How does Retrieval-Augmented Generation (RAG) work according to this channel?",
        "What does Andrej Karpathy say about attention mechanisms and Transformers?",
    ]

    st.caption("Click an example query or type any question about the channel you pasted:")
    eq_cols = st.columns(len(example_questions))
    selected_example = None
    for idx, eq in enumerate(example_questions):
        if eq_cols[idx].button(eq, key=f"eq_{idx}", use_container_width=True):
            selected_example = eq

    default_q = selected_example or st.session_state.get(
        "current_question", "What does this channel teach about Python list vs tuple?"
    )
    question = st.text_input(
        "Your Question",
        value=default_q,
        placeholder="Ask anything about the videos in the pasted YouTube channel...",
    )

    ask_clicked = st.button("🔍 Ask Channel", type="primary") or (selected_example is not None)

    if ask_clicked and question:
        st.session_state["current_question"] = question

        normalized_current = _normalize_channel_url(channel_url)
        indexed_url = st.session_state.get("indexed_channel_url")
        if collection.count() == 0 or (normalized_current and indexed_url != normalized_current):
            with st.spinner(f"Auto-indexing `{channel_url}` before answering your question..."):
                _run_live_or_fallback_index(channel_url)

        with st.spinner("Searching transcript chunks (and searching channel if topic is in older videos)..."):
            result = query_channel(
                question=question,
                n_results=top_k,
                api_key=active_api_key,
                channel_url=channel_url,
            )

        st.markdown("### 💡 Grounded Answer")
        st.markdown(result["answer"])

        st.markdown("### 🔗 Video Sources")
        if result["sources"]:
            for src_url in result["sources"]:
                st.markdown(f"- [{src_url}]({src_url})")
        else:
            st.caption("No sources returned.")

        # --- Live RAG Eval Harness Verification on the Current Query ---
        evaluator = RAGEvaluator(k=top_k)
        live_q_item = QuestionItem(
            query_id="live_query",
            question=question,
            key_facts=extract_content_keywords(question)[:4],
        )
        live_trace = RAGTrace(
            query_id="live_query",
            question=question,
            generated_answer=result["answer"],
            retrieved_chunks=_coerce_chunks(result.get("chunks", [])),
            pipeline_name="youtube-channel-rag",
        )
        live_eval = evaluator.evaluate_single(live_q_item, live_trace)

        st.markdown("### 🛡️ Live RAG Eval Harness Verification (`LLM-Judge` × `NLI Entailment`)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Cross-Checked Faithfulness",
            f"{live_eval.faithfulness.cross_checked_score:.1%}",
            "PASS" if live_eval.passed else live_eval.failure_category.value,
        )
        m2.metric("LLM-as-a-Judge Score", f"{live_eval.faithfulness.llm_judge_score:.1%}")
        m3.metric("NLI Entailment Score", f"{live_eval.faithfulness.nli_entailment_score:.1%}")
        m4.metric(
            "Context Keyword Coverage",
            f"{live_eval.retrieval.context_fact_coverage:.1%}",
            f"{len(live_eval.faithfulness.claims)} atomic claims verified",
        )

        with st.expander("🔬 Atomic Claim Verification Details (Dual-Judge Audit)", expanded=False):
            for c_idx, fc in enumerate(live_eval.faithfulness.claims, start=1):
                badge = "✅ Agree" if fc.agreement else f"⚡ Disagreement ({fc.disagreement_type})"
                st.markdown(
                    f"**Claim #{c_idx}:** `{fc.claim_text}`  \n"
                    f"↳ **LLM Judge:** `{fc.llm_verdict.value}` | **NLI:** `{fc.nli_label.value}` "
                    f"(Entailment=`{fc.nli_probabilities.entailment:.2f}`, Contradiction=`{fc.nli_probabilities.contradiction:.2f}`) "
                    f"| **Calibrated Score:** `{fc.calibrated_score:.2f}` | {badge}"
                )

        with st.expander(f"🔎 Retrieved Transcript Chunks (Top {len(result.get('chunks', []))} from ChromaDB)", expanded=True):
            for chunk_info in result.get("chunks", []):
                dist_str = f" | Distance: `{chunk_info['distance']:.4f}`" if chunk_info.get("distance") is not None else ""
                channel_badge = f"[{chunk_info['channel']}] " if chunk_info.get("channel") else ""
                st.markdown(
                    f"**Rank #{chunk_info['rank']} — {channel_badge}[{chunk_info['title']}]({chunk_info['url']})** "
                    f"(Chunk `{chunk_info['chunk_id']}`{dist_str})"
                )
                st.info(chunk_info["text"])

with tab_eval:
    st.subheader("📊 Benchmark YouTube Channel RAG Against `rag-eval-harness`")
    st.markdown(
        "Evaluates our **YouTube Channel RAG** pipeline across 7 golden benchmark queries spanning "
        "**freeCodeCamp.org** (Python Lists vs Tuples, ES6 JavaScript, Machine Learning, RAG Architecture) and "
        "**Andrej Karpathy** (Self-Attention, LLM Operating System, BPE Tokenization)."
    )

    if st.button("🚀 Run Full 7-Query Benchmark Suite", type="primary") or "harness_report" not in st.session_state:
        with st.spinner("Running RAG Eval Harness (Retrieval Metrics + Dual LLM/NLI Faithfulness)..."):
            st.session_state["harness_report"] = run_youtube_rag_evaluation(k=top_k)

    rep = st.session_state.get("harness_report")
    if rep is not None:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Pass Rate", f"{rep.pass_rate:.1%}", f"{rep.passed_queries}/{rep.total_queries} queries passed")
        k2.metric(f"Recall@{rep.k} / MRR", f"{rep.mean_recall_at_k:.1%}", f"MRR: {rep.mean_mrr:.3f} | NDCG: {rep.mean_ndcg_at_k:.3f}")
        k3.metric(
            "Cross-Checked Faithfulness",
            f"{rep.mean_cross_checked_faithfulness:.1%}",
            f"LLM: {rep.mean_llm_faithfulness:.1%} | NLI: {rep.mean_nli_faithfulness:.1%}",
        )
        k4.metric(
            "Answer Completeness",
            f"{rep.mean_answer_completeness:.1%}",
            f"Latency: {rep.mean_latency_ms:.1f} ms/query",
        )

        st.markdown("#### 📋 Per-Query Evaluation Matrix")
        rows = []
        for qr in rep.query_results:
            rows.append(
                {
                    "Status": "✅ PASS" if qr.passed else f"❌ {qr.failure_category.value}",
                    "Query ID": qr.query_id,
                    "Category": qr.category,
                    "Question": qr.question,
                    f"Recall@{rep.k}": f"{qr.retrieval.recall_at_k:.2f}",
                    "MRR": f"{qr.retrieval.mrr:.2f}",
                    f"NDCG@{rep.k}": f"{qr.retrieval.ndcg_at_k:.2f}",
                    "LLM Faithfulness": f"{qr.faithfulness.llm_judge_score:.2f}",
                    "NLI Entailment": f"{qr.faithfulness.nli_entailment_score:.2f}",
                    "Cross-Checked": f"{qr.faithfulness.cross_checked_score:.2f}",
                    "Completeness": f"{qr.answer_completeness:.2f}",
                }
            )
        st.dataframe(rows, use_container_width=True)

        dl1, dl2 = st.columns(2)
        dl1.download_button(
            "⬇️ Download Markdown Eval Report (EVALUATION_REPORT.md)",
            data=generate_markdown_report(rep),
            file_name="EVALUATION_REPORT.md",
            mime="text/markdown",
        )
        dl2.download_button(
            "⬇️ Download JSON Eval Report (eval_report.json)",
            data=rep.model_dump_json(indent=2),
            file_name="eval_report.json",
            mime="application/json",
        )
