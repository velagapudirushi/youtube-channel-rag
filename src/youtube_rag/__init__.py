"""
Modular YouTube Channel RAG Package (`youtube_rag`).

Modules:
- `youtube_rag.youtube_api`  : YouTube channel discovery (`yt-dlp`) & 3-tier transcript extraction (`youtube-transcript-api`)
- `youtube_rag.chunking`     : 500-word sliding window chunking (50-word overlap), stopword filtering, & plural stemming
- `youtube_rag.embeddings`   : OpenAI `text-embedding-3-small` API calls & 384-d local stemmed n-gram fallback embeddings
- `youtube_rag.vector_store` : `EmbeddedChromaCollection` vector database (L2 similarity search, metadata filtering, `/tmp` persistence)
- `youtube_rag.indexer`      : Channel & sample dataset indexing pipeline
- `youtube_rag.retriever`    : Hybrid dense + lexical candidate reranking & on-demand channel search
- `youtube_rag.generator`    : Grounded answer synthesis (`gpt-4o-mini` + local extractive fallback)
- `youtube_rag.sample_data`  : Pre-loaded evergreen transcripts for freeCodeCamp.org and Andrej Karpathy
"""

from __future__ import annotations

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
    index_channel,
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

__all__ = [
    "DEFAULT_CHROMA_PATH",
    "EmbeddedChromaCollection",
    "SAMPLE_CHANNELS",
    "SAMPLE_KARPATHY_VIDEOS",
    "STOPWORDS",
    "_channel_base_url",
    "_extract_transcript_via_ytdlp",
    "_get_openai_client",
    "_index_video_list",
    "_local_deterministic_embedding",
    "_normalize_channel_url",
    "_score_and_rerank_candidates",
    "_stem_token",
    "_strip_chunk_header",
    "_synthesize_extractive_answer",
    "chunk_text",
    "clear_collection",
    "collection",
    "extract_content_keywords",
    "get_channel_video_ids",
    "get_channel_videos",
    "get_chroma_collection",
    "get_embedding",
    "get_transcript",
    "index_channel",
    "index_sample_channel",
    "is_valid_openai_key",
    "query_channel",
    "retrieve_top_chunks",
    "search_channel_videos_for_query",
]
