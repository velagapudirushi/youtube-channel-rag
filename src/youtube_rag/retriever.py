"""
LangChain Hybrid Retrieval Module (`retriever.py`).
Implements:
1. `LangChainBM25Retriever(BaseRetriever)` — Sparse Okapi BM25 retriever powered by `rank_bm25.BM25Okapi`.
2. `ChromaDenseRetriever(BaseRetriever)` — Dense vector similarity retriever over ChromaDB.
3. `LangChainHybridRRFEnsembleRetriever(BaseRetriever)` — Combines BM25 + Chroma Dense Vector Search
   using Reciprocal Rank Fusion (RRF) and returns ONLY the Top-K `langchain_core.documents.Document` objects.
4. Dynamic On-Demand Channel Search (`search_channel_videos_for_query`): If the user asks about a video not in
   the latest 10 uploads, automatically searches the remaining channel videos on YouTube (`/@channel/search?query=...`),
   indexes the new `Document` chunks into ChromaDB + BM25, and retrieves the Top-K chunks!
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from rank_bm25 import BM25Okapi

from youtube_rag.chunking import extract_content_keywords
from youtube_rag.embeddings import get_embedding
from youtube_rag.indexer import _index_video_list
from youtube_rag.sample_data import SAMPLE_CHANNELS
from youtube_rag.vector_store import ChromaDenseRetriever, collection
from youtube_rag.youtube_api import search_channel_videos_for_query


class LangChainBM25Retriever(BaseRetriever):
    """
    LangChain `BaseRetriever` implementing Okapi BM25 sparse retrieval via `rank_bm25.BM25Okapi`.
    Automatically applies Inverse Document Frequency (IDF) and document-length normalization.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    documents: list[Document]
    k: int = 5

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> list[Document]:
        if not self.documents:
            return []

        tokenized_corpus = [
            extract_content_keywords(doc.page_content) or doc.page_content.lower().split()
            for doc in self.documents
        ]
        bm25 = BM25Okapi(tokenized_corpus)
        query_tokens = extract_content_keywords(query) or query.lower().split()
        scores = bm25.get_scores(query_tokens)

        scored_docs: list[tuple[float, int, Document]] = []
        for idx, (score, doc) in enumerate(zip(scores, self.documents)):
            meta = dict(doc.metadata or {})
            meta["bm25_score"] = round(float(score), 4)
            scored_docs.append((float(score), idx, Document(page_content=doc.page_content, metadata=meta)))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [item[2] for item in scored_docs[: self.k]]


class LangChainHybridRRFEnsembleRetriever(BaseRetriever):
    """
    LangChain `EnsembleRetriever` combining `LangChainBM25Retriever` (sparse) and
    `ChromaDenseRetriever` (dense) via Reciprocal Rank Fusion (RRF):
        RRF_Score(d) = sum_{r in {BM25, Chroma}} weight_r / (c + rank_r(d))
    Slices and returns ONLY the Top-K `Document` chunks.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bm25_retriever: LangChainBM25Retriever
    dense_retriever: ChromaDenseRetriever
    weights: tuple[float, float] = (0.45, 0.55)
    c: int = 60
    k: int = 5

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> list[Document]:
        bm25_docs = self.bm25_retriever.invoke(query)
        dense_docs = self.dense_retriever.invoke(query)

        w_bm25, w_dense = self.weights
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        def _doc_key(d: Document) -> str:
            m = d.metadata or {}
            return str(m.get("doc_id") or f"{m.get('video_id', '')}_chunk_{m.get('chunk_index', 0)}" or d.page_content[:80])

        for rank, d in enumerate(bm25_docs, start=1):
            key = _doc_key(d)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (w_bm25 / (self.c + rank))
            doc_map[key] = d

        for rank, d in enumerate(dense_docs, start=1):
            key = _doc_key(d)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (w_dense / (self.c + rank))
            if key in doc_map:
                merged_meta = dict(doc_map[key].metadata)
                merged_meta.update(d.metadata)
                doc_map[key] = Document(page_content=d.page_content, metadata=merged_meta)
            else:
                doc_map[key] = d

        # Sort by RRF score descending and add keyword coverage bonus so exact topic matches rank #1
        q_keywords = set(extract_content_keywords(query))
        ranked_entries: list[tuple[float, float, Document]] = []
        best_coverage = 0.0

        for key, rrf_val in rrf_scores.items():
            d = doc_map[key]
            doc_kw = set(extract_content_keywords(d.page_content))
            cov = (len(q_keywords & doc_kw) / len(q_keywords)) if q_keywords else 1.0
            if cov > best_coverage:
                best_coverage = cov
            meta = dict(d.metadata or {})
            meta["rrf_score"] = round(rrf_val, 6)
            meta["keyword_coverage"] = round(cov, 4)
            if "dense_distance" not in meta:
                meta["dense_distance"] = round(1.0 / (1.0 + rrf_val * 30.0), 4)
            ranked_entries.append((rrf_val + 0.05 * cov, cov, Document(page_content=d.page_content, metadata=meta)))

        ranked_entries.sort(key=lambda x: x[0], reverse=True)

        # Filter out unrelated noise chunks when high-coverage chunks exist
        if ranked_entries and best_coverage >= 0.5 and len(q_keywords) >= 2:
            min_cov = max(0.30, best_coverage * 0.45)
            filtered = [item[2] for item in ranked_entries if item[1] >= min_cov]
            if filtered:
                return filtered[: self.k]

        return [item[2] for item in ranked_entries[: self.k]]


def _score_and_rerank_candidates(
    question: str,
    docs: list[str],
    metadatas: list[dict[str, Any]],
    distances: list[float],
) -> tuple[list[tuple[float, float, str, dict[str, Any]]], float]:
    """Backwards-compatible helper scoring raw candidate lists."""
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
        filtered = [item[:4] for item in scored if item[4] >= min_cov and (item[0] - best_score) <= 1.15]
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
    Execute LangChain Hybrid Search (`LangChainBM25Retriever` + `ChromaDenseRetriever` via
    `LangChainHybridRRFEnsembleRetriever`) and return ONLY the Top-K chunks.
    If the topic is not in the latest 10 indexed videos, automatically searches the remaining
    videos on that YouTube channel (`/@channel/search?query=...`), indexes them into ChromaDB,
    and re-runs the LangChain Hybrid RRF Ensemble Retriever!
    """
    active_collection = target_collection if target_collection is not None else collection
    q_embedding = get_embedding(question, api_key=api_key)
    embed_mode = "openai" if len(q_embedding) > 384 else "local"
    effective_key = api_key if embed_mode == "openai" else ""

    def _run_langchain_hybrid_rrf() -> tuple[list[Document], float]:
        all_lc_docs = active_collection.as_langchain_documents(embed_mode=embed_mode)
        if not all_lc_docs:
            all_lc_docs = active_collection.as_langchain_documents(embed_mode=None)
        if not all_lc_docs:
            return [], 0.0

        candidate_pool_k = min(max(n_results * 3, 15), len(all_lc_docs))
        bm25_ret = LangChainBM25Retriever(documents=all_lc_docs, k=candidate_pool_k)
        dense_ret = ChromaDenseRetriever(
            target_collection=active_collection,
            k=candidate_pool_k,
            api_key=effective_key,
            embed_mode=embed_mode,
        )
        hybrid_ensemble = LangChainHybridRRFEnsembleRetriever(
            bm25_retriever=bm25_ret,
            dense_retriever=dense_ret,
            weights=(0.45, 0.55),
            k=n_results,
        )
        top_k_docs = hybrid_ensemble.invoke(question)[:n_results]
        best_cov = max((float(d.metadata.get("keyword_coverage", 0.0)) for d in top_k_docs), default=0.0)
        return top_k_docs, best_cov

    top_lc_docs, best_coverage = _run_langchain_hybrid_rrf()
    inferred_channel_url = channel_url or (
        top_lc_docs[0].metadata.get("channel_url") if top_lc_docs else None
    )

    # If the topic is NOT in the latest 10 indexed videos (coverage < 65%), search the remaining channel videos!
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
            top_lc_docs, best_coverage = _run_langchain_hybrid_rrf()

        if best_coverage < 0.65 and enable_dynamic_channel_search:
            print(
                f"🔎 Topic coverage ({best_coverage:.0%}) low in latest uploads — LangChain retriever searching remaining videos on `{inferred_channel_url}` for: {question}"
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
                top_lc_docs, best_coverage = _run_langchain_hybrid_rrf()

    top_docs = [d.page_content for d in top_lc_docs[:n_results]]
    top_metas = [dict(d.metadata) for d in top_lc_docs[:n_results]]
    top_dists = [float(d.metadata.get("dense_distance", 0.5)) for d in top_lc_docs[:n_results]]
    return top_docs, top_metas, top_dists, embed_mode
