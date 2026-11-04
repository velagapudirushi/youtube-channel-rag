"""Unit & integration tests for the YouTube Channel RAG pipeline across any channel."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from channel_rag import (
    _normalize_channel_url,
    chunk_text,
    get_chroma_collection,
    get_embedding,
    index_channel,
    index_sample_channel,
    is_valid_openai_key,
    query_channel,
)


def test_placeholder_and_invalid_openai_keys_never_crash():
    """Ensure placeholder keys like 'sk-...' or 'sk-....' (from the user screenshot) never trigger a 401 crash."""
    assert not is_valid_openai_key("")
    assert not is_valid_openai_key("sk-...")
    assert not is_valid_openai_key("sk-....")
    assert not is_valid_openai_key("sk-your-openai-api-key-here")

    # Even if passed directly to get_embedding, it must return a valid 384-d vector without raising
    emb = get_embedding("freeCodeCamp Python tutorial", api_key="sk-....")
    assert len(emb) == 384


def test_long_invalid_key_graceful_fallback():
    """Even a 40+ char revoked/bad key that raises 401 from OpenAI must fall back cleanly to local embeddings."""
    bad_long_key = "sk-proj-0000000000000000000000000000000000000000"
    with patch("openai.resources.embeddings.Embeddings.create", side_effect=Exception("401 Incorrect API key")):
        emb = get_embedding("Test fallback on 401 error", api_key=bad_long_key)
        assert len(emb) == 384


def test_chunk_text_overlap():
    """Verify 500-word chunking with 50-word overlap matches the specification."""
    words = [f"word_{i}" for i in range(1000)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 3
    first_chunk_words = chunks[0].split()
    second_chunk_words = chunks[1].split()
    assert len(first_chunk_words) == 500
    assert second_chunk_words[0] == "word_450"
    assert first_chunk_words[-50:] == second_chunk_words[:50]


def test_normalize_any_channel_url():
    """Ensure bare handles (@freecodecamp), channel names, and URLs normalize to /videos."""
    assert (
        _normalize_channel_url("@freecodecamp")
        == "https://www.youtube.com/@freecodecamp/videos"
    )
    assert (
        _normalize_channel_url("freecodecamp")
        == "https://www.youtube.com/@freecodecamp/videos"
    )
    assert (
        _normalize_channel_url("https://www.youtube.com/@freecodecamp")
        == "https://www.youtube.com/@freecodecamp/videos"
    )
    assert (
        _normalize_channel_url("https://www.youtube.com/@AndrejKarpathy/videos")
        == "https://www.youtube.com/@AndrejKarpathy/videos"
    )


def test_freecodecamp_channel_indexing_and_query():
    """Verify end-to-end ChromaDB indexing and retrieval on the freeCodeCamp channel."""
    test_col = get_chroma_collection(collection_name="test_fcc_rag", persist_directory=None, in_memory=True)
    stats = index_sample_channel(
        preset_name="freeCodeCamp.org (@freecodecamp)",
        api_key="sk-....",  # even with placeholder key, must succeed
        target_collection=test_col,
    )
    assert stats["videos_indexed"] == 4
    assert stats["total_chunks"] > 0

    res = query_channel(
        "What is the difference between lists and tuples in Python, and what machine learning paradigms are covered?",
        n_results=3,
        api_key="sk-....",
        target_collection=test_col,
    )
    assert "answer" in res
    assert len(res["sources"]) > 0
    assert any("rfscVS0vtbw" in src or "NWONeJKn6kc" in src for src in res["sources"])


def test_index_any_custom_channel_with_mocked_youtube():
    """Verify index_channel works on any arbitrary channel (e.g., freeCodeCamp) with mocked yt-dlp/transcripts."""
    test_col = get_chroma_collection(collection_name="test_custom_channel", persist_directory=None, in_memory=True)

    fake_videos = [
        {
            "id": "fcc_vid_001",
            "title": "Docker & Kubernetes Full Course - freeCodeCamp",
            "channel": "freeCodeCamp.org",
            "description": "Complete Docker containers and Kubernetes orchestration tutorial.",
            "url": "https://youtube.com/watch?v=fcc_vid_001",
        },
        {
            "id": "fcc_vid_002",
            "title": "PostgreSQL Database Tutorial - Full Course",
            "channel": "freeCodeCamp.org",
            "description": "Learn SQL joins, indexing, ACID transactions, and PostgreSQL administration.",
            "url": "https://youtube.com/watch?v=fcc_vid_002",
        },
    ]

    def fake_transcript(vid: str):
        if vid == "fcc_vid_001":
            return (
                "Docker packages applications and their dependencies into lightweight isolated containers. "
                "Kubernetes orchestrates container deployment, scaling, and load balancing across clusters."
            )
        return None  # Second video has no closed captions -> falls back to description!

    with patch("channel_rag.get_channel_videos", return_value=fake_videos), patch(
        "channel_rag.get_transcript", side_effect=fake_transcript
    ):
        stats = index_channel(
            "@MockDevChannel",
            max_videos=5,
            api_key="",
            target_collection=test_col,
        )
        assert stats["videos_found"] == 2
        # Both videos get indexed (one via transcript, one via description fallback!)
        assert stats["videos_indexed"] == 2

        result = query_channel(
            "How do Docker and Kubernetes work together?",
            api_key="",
            target_collection=test_col,
            enable_dynamic_channel_search=False,
        )
        assert "https://youtube.com/watch?v=fcc_vid_001" in result["sources"]
        assert "docker" in result["answer"].lower()


def test_rag_eval_harness_benchmark_suite():
    """Verify that the YouTube Channel RAG passes the standalone RAG Eval Harness benchmark."""
    from evaluate_with_harness import run_youtube_rag_evaluation

    report = run_youtube_rag_evaluation(k=5)
    assert report.total_queries == 7
    assert report.passed_queries == 7
    assert report.pass_rate == 1.0
    assert report.mean_recall_at_k == 1.0
    assert report.mean_mrr == 1.0
    assert report.mean_cross_checked_faithfulness >= 0.90
