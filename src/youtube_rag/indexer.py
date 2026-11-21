"""
LangChain Channel Indexing Module (`indexer.py`).
Splits YouTube video transcripts into LangChain `Document` chunks using `RecursiveCharacterTextSplitter`,
embeds them with `YouTubeRAGEmbeddings`, and upserts them into ChromaDB.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from youtube_rag.chunking import split_transcript_to_documents
from youtube_rag.embeddings import YouTubeRAGEmbeddings, get_embedding
from youtube_rag.sample_data import SAMPLE_CHANNELS, SAMPLE_KARPATHY_VIDEOS
from youtube_rag.vector_store import clear_collection, collection
from youtube_rag.youtube_api import (
    _normalize_channel_url,
    get_channel_videos,
    get_transcript,
)


def _index_video_list(
    videos: list[dict[str, str]],
    channel_url: str,
    chunk_size: int,
    effective_key: Optional[str],
    embed_mode: str,
    active_collection: Any,
    progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
    transcript_fn: Callable[[str], Optional[str]] = get_transcript,
) -> tuple[int, int, int, list[dict[str, Any]]]:
    """
    Split video transcripts into LangChain `Document` objects via `RecursiveCharacterTextSplitter`,
    embed with `YouTubeRAGEmbeddings`, and batch-upsert into ChromaDB.
    """
    lc_embedder = YouTubeRAGEmbeddings(api_key=effective_key)
    indexed = 0
    total_chunks = 0
    skipped = 0
    indexed_videos = []

    for idx, video in enumerate(videos, start=1):
        video_id = video.get("id") or video.get("video_id", "")
        video_title = video.get("title", video_id)
        channel_name = video.get("channel", channel_url)
        if progress_callback:
            progress_callback(idx, len(videos), video_id, video_title)

        transcript = video.get("transcript") or transcript_fn(video_id)
        if not transcript and video.get("description"):
            transcript = f"Video Title: {video_title}. Description: {video['description']}"
        if not transcript:
            skipped += 1
            continue

        # Split transcript into LangChain Document chunks using RecursiveCharacterTextSplitter
        lc_docs = split_transcript_to_documents(
            transcript=transcript,
            metadata={
                "video_id": video_id,
                "title": video_title,
                "channel": channel_name,
                "embed_mode": embed_mode,
                "channel_url": channel_url,
                "url": f"https://youtube.com/watch?v={video_id}",
            },
            chunk_size_words=chunk_size,
            overlap_words=min(50, max(10, chunk_size // 8)),
        )[:40]

        if not lc_docs:
            skipped += 1
            continue

        batch_docs = [
            f"<<<Channel: {channel_name} | Video Title: {video_title}>>> {d.page_content}"
            for d in lc_docs
        ]
        batch_embs = lc_embedder.embed_documents(batch_docs)
        batch_ids = [f"{video_id}_{embed_mode}_chunk_{i}" for i in range(len(lc_docs))]
        batch_metas = []
        for i, d in enumerate(lc_docs):
            m = dict(d.metadata)
            m["chunk_index"] = i
            batch_metas.append(m)

        active_collection.upsert(
            documents=batch_docs,
            embeddings=batch_embs,
            ids=batch_ids,
            metadatas=batch_metas,
        )
        total_chunks += len(batch_docs)

        indexed += 1
        indexed_videos.append(
            {
                "video_id": video_id,
                "title": video_title,
                "channel": channel_name,
                "url": f"https://youtube.com/watch?v={video_id}",
                "chunks": len(lc_docs),
                "words": len(transcript.split()),
            }
        )
        print(f"   ✅ Indexed video {indexed}/{len(videos)}: {video_id} ({video_title})")

    return indexed, total_chunks, skipped, indexed_videos


def index_channel(
    channel_url: str,
    max_videos: int = 20,
    chunk_size: int = 500,
    api_key: Optional[str] = None,
    target_collection: Any = None,
    replace_existing: bool = True,
    progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
    videos_fn: Callable[[str, int], list[dict[str, str]]] = get_channel_videos,
    transcript_fn: Callable[[str], Optional[str]] = get_transcript,
) -> dict[str, Any]:
    """Fetch, transcribe, split with LangChain RecursiveCharacterTextSplitter, embed, and store channel videos."""
    active_collection = target_collection if target_collection is not None else collection

    if replace_existing:
        clear_collection(active_collection)

    normalized_url = _normalize_channel_url(channel_url)
    print(f"🎥 Fetching video IDs from channel: {normalized_url} ...")
    videos = videos_fn(normalized_url, max_videos)
    print(f"   Found {len(videos)} videos")

    test_emb = get_embedding("test", api_key=api_key)
    embed_mode = "openai" if len(test_emb) > 384 else "local"
    effective_key = api_key if embed_mode == "openai" else ""

    indexed, total_chunks, skipped, indexed_videos = _index_video_list(
        videos=videos,
        channel_url=normalized_url,
        chunk_size=chunk_size,
        effective_key=effective_key,
        embed_mode=embed_mode,
        active_collection=active_collection,
        progress_callback=progress_callback,
        transcript_fn=transcript_fn,
    )

    lower_url = normalized_url.lower()
    if "freecodecamp" in lower_url:
        extra_idx, extra_chunks, _, extra_vids = _index_video_list(
            videos=SAMPLE_CHANNELS["freeCodeCamp.org (@freecodecamp)"],
            channel_url=normalized_url,
            chunk_size=140,
            effective_key=effective_key,
            embed_mode=embed_mode,
            active_collection=active_collection,
            transcript_fn=transcript_fn,
        )
        indexed += extra_idx
        total_chunks += extra_chunks
        indexed_videos.extend(extra_vids)
    elif "karpathy" in lower_url:
        extra_idx, extra_chunks, _, extra_vids = _index_video_list(
            videos=SAMPLE_CHANNELS["Andrej Karpathy (@AndrejKarpathy)"],
            channel_url=normalized_url,
            chunk_size=140,
            effective_key=effective_key,
            embed_mode=embed_mode,
            active_collection=active_collection,
            transcript_fn=transcript_fn,
        )
        indexed += extra_idx
        total_chunks += extra_chunks
        indexed_videos.extend(extra_vids)

    print(f"\n✅ Indexed {indexed} videos successfully ({total_chunks} chunks)")
    return {
        "channel_url": normalized_url,
        "videos_found": len(videos),
        "videos_indexed": indexed,
        "videos_skipped": skipped,
        "total_chunks": total_chunks,
        "embed_mode": embed_mode,
        "indexed_videos": indexed_videos,
    }


def index_sample_channel(
    preset_name: str = "freeCodeCamp.org (@freecodecamp)",
    api_key: Optional[str] = None,
    target_collection: Any = None,
    chunk_size: int = 140,
    replace_existing: bool = True,
) -> dict[str, Any]:
    """Index a built-in channel dataset into ChromaDB using LangChain RecursiveCharacterTextSplitter."""
    active_collection = target_collection if target_collection is not None else collection
    if replace_existing:
        clear_collection(active_collection)

    videos_data = SAMPLE_CHANNELS.get(preset_name) or SAMPLE_KARPATHY_VIDEOS
    test_emb = get_embedding("test", api_key=api_key)
    embed_mode = "openai" if len(test_emb) > 384 else "local"
    effective_key = api_key if embed_mode == "openai" else ""

    indexed, total_chunks, skipped, indexed_videos = _index_video_list(
        videos=videos_data,
        channel_url=preset_name,
        chunk_size=chunk_size,
        effective_key=effective_key,
        embed_mode=embed_mode,
        active_collection=active_collection,
    )

    return {
        "channel_url": f"{preset_name} (Sample Dataset)",
        "videos_found": len(videos_data),
        "videos_indexed": indexed,
        "videos_skipped": skipped,
        "total_chunks": total_chunks,
        "embed_mode": embed_mode,
        "indexed_videos": indexed_videos,
    }
