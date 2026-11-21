"""
YouTube Channel RAG — CLI Entrypoint & Public Facade (`channel_rag.py`).
All core responsibilities are separated into dedicated modules inside `src/youtube_rag/`:
- `src/youtube_rag/youtube_api.py`  : YouTube channel video discovery (`yt-dlp`) & 3-tier transcript extraction (`youtube-transcript-api`)
- `src/youtube_rag/chunking.py`     : 500-word sliding window chunking (50-word overlap), stopword filtering, & plural stemming
- `src/youtube_rag/embeddings.py`   : OpenAI `text-embedding-3-small` API calls & 384-d local stemmed n-gram fallback embeddings
- `src/youtube_rag/vector_store.py` : `EmbeddedChromaCollection` vector database (L2 similarity search, metadata filtering, `/tmp` persistence)
- `src/youtube_rag/indexer.py`      : Channel & sample dataset indexing pipeline
- `src/youtube_rag/retriever.py`    : Hybrid dense + lexical candidate reranking & on-demand channel search
- `src/youtube_rag/generator.py`    : Grounded answer synthesis (`gpt-4o-mini` + local extractive fallback)
- `src/youtube_rag/sample_data.py`  : Pre-loaded evergreen transcripts for freeCodeCamp.org and Andrej Karpathy
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Optional

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from youtube_rag.chunking import (
    STOPWORDS,
    _stem_token,
    _strip_chunk_header,
    chunk_text,
    extract_content_keywords,
)
from youtube_rag.embeddings import (
    _get_openai_client,
    _local_deterministic_embedding,
    get_embedding,
    is_valid_openai_key,
)
from youtube_rag.generator import _synthesize_extractive_answer, query_channel
from youtube_rag.indexer import (
    _index_video_list,
    index_channel as _core_index_channel,
    index_sample_channel,
)
from youtube_rag.retriever import (
    _score_and_rerank_candidates,
    retrieve_top_chunks,
)
from youtube_rag.sample_data import SAMPLE_CHANNELS, SAMPLE_KARPATHY_VIDEOS
from youtube_rag.vector_store import (
    DEFAULT_CHROMA_PATH,
    EmbeddedChromaCollection,
    clear_collection,
    collection,
    get_chroma_collection,
)
from youtube_rag.youtube_api import (
    _channel_base_url,
    _extract_transcript_via_ytdlp,
    _normalize_channel_url,
    get_channel_video_ids,
    get_channel_videos,
    get_transcript,
    search_channel_videos_for_query,
)


def index_channel(
    channel_url: str,
    max_videos: int = 20,
    chunk_size: int = 500,
    api_key: Optional[str] = None,
    target_collection: Any = None,
    replace_existing: bool = True,
    progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
) -> dict[str, Any]:
    """Delegate channel indexing to `youtube_rag.indexer.index_channel`."""
    return _core_index_channel(
        channel_url=channel_url,
        max_videos=max_videos,
        chunk_size=chunk_size,
        api_key=api_key,
        target_collection=target_collection,
        replace_existing=replace_existing,
        progress_callback=progress_callback,
        videos_fn=get_channel_videos,
        transcript_fn=get_transcript,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YouTube Channel RAG CLI — Works with any YouTube channel")
    parser.add_argument(
        "--channel",
        type=str,
        default="https://www.youtube.com/@freecodecamp/videos",
        help="Any YouTube channel URL or handle (e.g. https://www.youtube.com/@freecodecamp/videos)",
    )
    parser.add_argument(
        "--sample",
        type=str,
        choices=["freecodecamp", "karpathy"],
        default=None,
        help="Index a built-in sample channel ('freecodecamp' or 'karpathy') for instant offline testing",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=5,
        help="Maximum number of channel videos to index",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="what does this channel teach about python list vs tuple",
        help="Question to ask across the indexed channel",
    )
    args = parser.parse_args()

    if args.sample == "freecodecamp":
        stats = index_sample_channel("freeCodeCamp.org (@freecodecamp)")
        print(f"✅ Indexed {stats['videos_indexed']} freeCodeCamp sample videos ({stats['total_chunks']} chunks)")
    elif args.sample == "karpathy":
        stats = index_sample_channel("Andrej Karpathy (@AndrejKarpathy)")
        print(f"✅ Indexed {stats['videos_indexed']} Karpathy sample videos ({stats['total_chunks']} chunks)")
    else:
        index_channel(args.channel, max_videos=args.max_videos)

    result = query_channel(args.question, channel_url=args.channel)
    print("\n" + "=" * 70)
    print(f"Question: {args.question}\n")
    print(result["answer"])
    print("\nSources:", result["sources"])
