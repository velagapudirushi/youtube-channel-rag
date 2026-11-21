"""
Answer Generation & Query Orchestration Module (`generator.py`).
Handles Step 4: synthesizing grounded answers from retrieved transcript chunks
using OpenAI `gpt-4o-mini` (or the Local Extractive Synthesizer fallback).
"""

from __future__ import annotations

import re
from typing import Any, Optional

from youtube_rag.chunking import _strip_chunk_header, extract_content_keywords
from youtube_rag.embeddings import _get_openai_client
from youtube_rag.retriever import retrieve_top_chunks
from youtube_rag.vector_store import collection


def _synthesize_extractive_answer(question: str, docs: list[str], metadatas: list[dict[str, Any]]) -> str:
    """
    Generate a clean, well-structured grounded answer directly from the retrieved transcript chunks
    when running without an OpenAI API key (or if the OpenAI API key fails).
    """
    if not docs:
        return "No indexed transcript chunks were found. Please index a YouTube channel first."

    q_keywords = set(extract_content_keywords(question))

    scored_sentences: list[tuple[float, str, str, str]] = []
    for rank_idx, (doc, meta) in enumerate(zip(docs, metadatas)):
        title = meta.get("title") or meta.get("video_id", "Unknown Video")
        url = meta.get("url", "")
        clean_doc = _strip_chunk_header(doc)
        title_keywords = set(extract_content_keywords(title))
        title_bonus = len(q_keywords & title_keywords) * 1.2
        rank_bonus = max(0.0, (len(docs) - rank_idx) * 0.25)

        sentences = re.split(r"(?<=[.!?])\s+", clean_doc)
        for sent_idx, sent in enumerate(sentences):
            sent_clean = sent.strip()
            if len(sent_clean.split()) < 6:
                continue
            if (sent_idx == 0 or sent_idx == len(sentences) - 1) and not sent_clean.endswith((".", "!", "?")):
                continue
            sent_keywords = set(extract_content_keywords(sent_clean))
            overlap = len(q_keywords & sent_keywords)
            if q_keywords and overlap == 0:
                continue
            score = (overlap * 3.0) + title_bonus + rank_bonus
            scored_sentences.append((score, sent_clean, title, url))

    scored_sentences.sort(key=lambda x: x[0], reverse=True)

    if not scored_sentences:
        top_meta = metadatas[0] if metadatas else {}
        return (
            f"Based on **{top_meta.get('title', 'the channel')}** "
            f"({top_meta.get('url', '')}):\n\n> {_strip_chunk_header(docs[0])[:600]}"
        )

    lines: list[str] = []
    seen: list[str] = []
    for _, sent, title, url in scored_sentences:
        norm_sent = sent.lower()
        if any(norm_sent in existing or existing in norm_sent for existing in seen):
            continue
        seen.append(norm_sent)
        lines.append(f"- {sent}")
        if len(seen) >= 5:
            break

    return "\n".join(lines)


def query_channel(
    question: str,
    n_results: int = 5,
    api_key: Optional[str] = None,
    target_collection: Any = None,
    channel_url: Optional[str] = None,
    enable_dynamic_channel_search: bool = True,
    model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """
    Search across all indexed videos and generate a grounded answer with video citations.
    """
    active_collection = target_collection if target_collection is not None else collection

    if active_collection.count() == 0:
        return {
            "answer": "No videos are indexed in the collection yet. Index a channel or load a sample channel first.",
            "sources": [],
            "chunks": [],
        }

    docs, metadatas, distances, embed_mode = retrieve_top_chunks(
        question=question,
        n_results=n_results,
        api_key=api_key,
        target_collection=active_collection,
        channel_url=channel_url,
        enable_dynamic_channel_search=enable_dynamic_channel_search,
    )

    if not docs:
        return {
            "answer": "No matching transcript chunks found. Please click 'Index Channel' to index your channel.",
            "sources": [],
            "chunks": [],
        }

    context_blocks = []
    for doc, meta in zip(docs, metadatas):
        title = meta.get("title") or meta.get("video_id", "Video")
        url = meta.get("url", "")
        context_blocks.append(f"[Video: {title} | URL: {url}]\n{_strip_chunk_header(doc)}")

    context = "\n\n".join(context_blocks)
    sources = list(dict.fromkeys(m["url"] for m in metadatas if "url" in m))

    prompt = f"""Answer the question using ONLY the YouTube transcript context below.
Include which video(s) the information came from.

Context:
{context}

Question: {question}
Answer:"""

    client = _get_openai_client(api_key) if embed_mode == "openai" else None
    answer = None
    if client is not None:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response.choices[0].message.content
        except Exception as exc:
            print(f"⚠️ OpenAI chat completion failed ({exc}); falling back to local synthesis.")

    if not answer:
        answer = _synthesize_extractive_answer(question, docs, metadatas)

    retrieved_chunks = []
    for idx, (doc, meta) in enumerate(zip(docs, metadatas)):
        dist = distances[idx] if idx < len(distances) else None
        vid = meta.get("video_id", "")
        c_idx = meta.get("chunk_index", 0)
        sim_score = round(1.0 / (1.0 + max(0.0, float(dist))), 4) if dist is not None else 1.0
        retrieved_chunks.append(
            {
                "chunk_id": f"{vid}_chunk_{c_idx}",
                "rank": idx + 1,
                "score": sim_score,
                "video_id": vid,
                "title": meta.get("title", vid),
                "channel": meta.get("channel", ""),
                "url": meta.get("url", ""),
                "chunk_index": c_idx,
                "distance": dist,
                "text": _strip_chunk_header(doc),
                "metadata": meta,
            }
        )

    return {
        "answer": answer,
        "sources": sources,
        "chunks": retrieved_chunks,
    }
