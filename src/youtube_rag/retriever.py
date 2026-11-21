"""
Retrieval & Hybrid Reranking Module (`retriever.py`).
Handles candidate retrieval from the vector store, stemmed keyword + dense distance
hybrid reranking, noise threshold filtering, and dynamic on-demand channel search.
"""

from __future__ import annotations

from typing import Any, Optional

from youtube_rag.chunking import extract_content_keywords
from youtube_rag.embeddings import _local_deterministic_embedding, get_embedding
from youtube_rag.indexer import _index_video_list
from youtube_rag.sample_data import SAMPLE_CHANNELS
from youtube_rag.vector_store import collection
from youtube_rag.youtube_api import search_channel_videos_for_query


def _score_and_rerank_candidates(
    question: str,
    docs: list[str],
    metadatas: list[dict[str, Any]],
    distances: list[float],
) -> tuple[list[tuple[float, float, str, dict[str, Any]]], float]:
    """
    Rerank candidate chunks using stemmed content keyword overlap + vector distance.
    Filters out noisy unrelated chunks when high-coverage matches exist.
    Returns (sorted_candidates, best_keyword_coverage_ratio).
    """
    q_keywords = set(extract_content_keywords(question))
    scored = []
    best_coverage = 0.0

    for idx, (doc, meta) in enumerate(zip(docs, metadatas)):
        dist = float(distances[idx]) if idx < len(distances) else 1.5
        doc_keywords = set(extract_content_keywords(doc))
        title_keywords = set(extract_content_keywords(meta.get("title", "")))

        matched_in_doc = q_keywords & doc_keywords
        matched_in_title = q_keywords & title_keywords
        coverage = (len(matched_in_doc) / len(q_keywords)) if q_keywords else 1.0
        if coverage > best_coverage:
            best_coverage = coverage

        lexical_boost = (0.45 * len(matched_in_doc)) + (0.55 * len(matched_in_title))
        combined_score = dist - lexical_boost
        scored.append((combined_score, dist, doc, meta, coverage))

    scored.sort(key=lambda x: x[0])

    if scored and best_coverage >= 0.5 and len(q_keywords) >= 2:
        best_score = scored[0][0]
        min_cov = max(0.5, best_coverage * 0.66)
        filtered = [
            item[:4]
            for item in scored
            if item[4] >= min_cov and (item[0] - best_score) <= 1.15
        ]
        if filtered:
            return filtered, best_coverage

    return [item[:4] for item in scored], best_coverage


def retrieve_top_chunks(
    question: str,
    n_results: int = 5,
    api_key: Optional[str] = None,
    target_collection: Any = None,
    channel_url: Optional[str] = None,
    enable_dynamic_channel_search: bool = True,
) -> tuple[list[str], list[dict[str, Any]], list[float], str]:
    """
    Retrieve and rerank the top-k relevant transcript chunks from the vector store.
    Automatically searches the YouTube channel for older videos if recent uploads don't cover the question topic.
    """
    active_collection = target_collection if target_collection is not None else collection
    q_embedding = get_embedding(question, api_key=api_key)
    embed_mode = "openai" if len(q_embedding) > 384 else "local"
    effective_key = api_key if embed_mode == "openai" else ""

    def _run_chroma_query() -> tuple[list[str], list[dict[str, Any]], list[float]]:
        curr_total = active_collection.count()
        candidate_k = min(max(n_results * 30, 150), curr_total)
        try:
            res = active_collection.query(
                query_embeddings=[q_embedding],
                n_results=candidate_k,
                where={"embed_mode": embed_mode},
            )
            d = res.get("documents", [[]])[0] if res.get("documents") else []
            if not d:
                raise ValueError("Empty result for embed_mode")
        except Exception:
            fallback_emb = _local_deterministic_embedding(question) if embed_mode == "openai" else q_embedding
            res = active_collection.query(
                query_embeddings=[fallback_emb],
                n_results=candidate_k,
            )
        d = res.get("documents", [[]])[0] if res.get("documents") else []
        m = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
        dist_list = res.get("distances", [[]])[0] if res.get("distances") else []
        return d, m, dist_list

    docs, metadatas, distances = _run_chroma_query()
    reranked, best_coverage = _score_and_rerank_candidates(question, docs, metadatas, distances)

    inferred_channel_url = channel_url or (metadatas[0].get("channel_url") if metadatas else None)

    if best_coverage < 0.65 and inferred_channel_url:
        lower_ch = str(inferred_channel_url).lower()
        if "freecodecamp" in lower_ch:
            _index_video_list(
                videos=SAMPLE_CHANNELS["freeCodeCamp.org (@freecodecamp)"],
                channel_url=str(inferred_channel_url),
                chunk_size=140,
                effective_key=effective_key,
                embed_mode=embed_mode,
                active_collection=active_collection,
            )
            docs, metadatas, distances = _run_chroma_query()
            reranked, best_coverage = _score_and_rerank_candidates(question, docs, metadatas, distances)

        if best_coverage < 0.65 and enable_dynamic_channel_search:
            print(
                f"🔎 Topic coverage ({best_coverage:.0%}) low in recent uploads — searching channel `{inferred_channel_url}` for: {question}"
            )
            targeted_videos = search_channel_videos_for_query(
                channel_url=str(inferred_channel_url),
                question=question,
                max_search_results=3,
            )
            if targeted_videos:
                _index_video_list(
                    videos=targeted_videos,
                    channel_url=str(inferred_channel_url),
                    chunk_size=400,
                    effective_key=effective_key,
                    embed_mode=embed_mode,
                    active_collection=active_collection,
                )
                docs, metadatas, distances = _run_chroma_query()
                reranked, best_coverage = _score_and_rerank_candidates(question, docs, metadatas, distances)

    top_items = reranked[:n_results]
    top_docs = [item[2] for item in top_items]
    top_metas = [item[3] for item in top_items]
    top_dists = [item[1] for item in top_items]
    return top_docs, top_metas, top_dists, embed_mode
