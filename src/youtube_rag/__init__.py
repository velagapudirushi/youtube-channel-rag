"""
Modular YouTube Channel RAG Package (`youtube_rag`) — Powered by LangChain, ChromaDB & Hybrid BM25+Vector Search (RRF).

Modules:
- `youtube_rag.youtube_api`  : YouTube channel discovery (`yt-dlp`), on-demand channel search & 3-tier transcript extraction (`youtube-transcript-api`)
- `youtube_rag.chunking`     : LangChain `RecursiveCharacterTextSplitter` -> `Document` splitting, stopword filtering, & plural stemming
- `youtube_rag.embeddings`   : LangChain `YouTubeRAGEmbeddings(Embeddings)` (OpenAI `text-embedding-3-small` + 384-d local fallback)
- `youtube_rag.vector_store` : `EmbeddedChromaCollection` & LangChain `ChromaDenseRetriever(BaseRetriever)`
- `youtube_rag.indexer`      : LangChain `Document` indexing pipeline into ChromaDB
- `youtube_rag.retriever`    : LangChain `LangChainBM25Retriever` (`rank-bm25`) + `LangChainHybridRRFEnsembleRetriever` (Reciprocal Rank Fusion -> Top-K)
- `youtube_rag.generator`    : LangChain LCEL (`ChatPromptTemplate | RunnableLambda | StrOutputParser`) grounded answer synthesis
- `youtube_rag.sample_data`  : Pre-loaded evergreen transcripts for freeCodeCamp.org and Andrej Karpathy
"""

from __future__ import annotations

from youtube_rag.chunking import (
    STOPWORDS,
    _stem_token,
    _strip_chunk_header,
    chunk_text,
    extract_content_keywords,
    split_transcript_to_documents,
)
from youtube_rag.embeddings import (
    YouTubeRAGEmbeddings,
    _get_openai_client,
    _local_deterministic_embedding,
    get_embedding,
    is_valid_openai_key,
)
from youtube_rag.generator import (
    RAG_PROMPT_TEMPLATE,
    _synthesize_extractive_answer,
    query_channel,
)
from youtube_rag.indexer import (
    _index_video_list,
    index_channel,
    index_sample_channel,
)
from youtube_rag.retriever import (
    LangChainBM25Retriever,
    LangChainHybridRRFEnsembleRetriever,
    _score_and_rerank_candidates,
    retrieve_top_chunks,
)
from youtube_rag.sample_data import SAMPLE_CHANNELS, SAMPLE_KARPATHY_VIDEOS
from youtube_rag.vector_store import (
    DEFAULT_CHROMA_PATH,
    ChromaDenseRetriever,
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
    "ChromaDenseRetriever",
    "DEFAULT_CHROMA_PATH",
    "EmbeddedChromaCollection",
    "LangChainBM25Retriever",
    "LangChainHybridRRFEnsembleRetriever",
    "RAG_PROMPT_TEMPLATE",
    "SAMPLE_CHANNELS",
    "SAMPLE_KARPATHY_VIDEOS",
    "STOPWORDS",
    "YouTubeRAGEmbeddings",
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
    "split_transcript_to_documents",
]
