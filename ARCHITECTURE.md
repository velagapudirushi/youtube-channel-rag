# Architecture, Codebase Structure & Design Decisions (`youtube-channel-rag`)

This document explains **how `youtube-channel-rag` works under the hood**, **how the codebase is organized**, **what alternative technical options exist**, **why we chose every component in this stack**, and **how it integrates with `rag-eval-harness`**.

---

## 1. Visual System Architecture

![System Architecture Pipeline](docs/assets/architecture_pipeline.svg)

### End-to-End Execution Flow
1. **Universal Channel & Handle Normalization (`_normalize_channel_url` in `channel_rag.py`):**
   Accepts *any* YouTube input format without requiring channel IDs:
   * `@freecodecamp` or `freecodecamp` $\to$ `https://www.youtube.com/@freecodecamp/videos`
   * `https://www.youtube.com/@AndrejKarpathy` $\to$ `https://www.youtube.com/@AndrejKarpathy/videos`
   * Playlist URLs (`playlist?list=...`) and single video URLs (`watch?v=...`) are preserved directly.
2. **Dual-Path Video Discovery (`get_channel_videos` & `search_channel_videos_for_query`):**
   * **Path A (Recent Uploads Discovery):** Uses `yt-dlp` (`extract_flat="in_playlist"`) to fetch video IDs, titles, channel names, and descriptions in ~1–2 seconds without downloading audio/video streams.
   * **Path B (On-Demand Channel-Scoped Search):** Large channels like `freeCodeCamp.org` have 1,800+ videos. If a user asks about a topic (e.g., *"Python lists vs tuples"*) that isn't in the 10 newest uploads from this week, `query_channel()` detects low topic keyword coverage (`< 65%`) and automatically queries YouTube's channel search endpoint (`https://www.youtube.com/@<channel>/search?query=...`) via `yt-dlp` to locate and index the relevant videos on the fly.
3. **3-Tier Transcript & Syllabus Extraction (`get_transcript`):**
   * **Tier 1A:** Queries `youtube-transcript-api` across English regional codes (`en`, `en-US`, `en-GB`, `en-IN`, `en-CA`, `en-AU`).
   * **Tier 1B:** Lists all available caption tracks (`YouTubeTranscriptApi().list(video_id)`) to capture auto-generated speech-to-text captions.
   * **Tier 2 (`_extract_transcript_via_ytdlp`):** Downloads `json3` subtitle streams directly via `yt-dlp` or extracts structured video chapters + course syllabus descriptions.
4. **Contextual Sliding-Window Chunking (`chunk_text`):**
   * Splits transcripts into **500-word chunks** with a **50-word sliding overlap** (`stride = 450` words) so sentences at chunk boundaries are never severed.
   * Prepends `<<<Channel: {channel} | Video Title: {title}>>>` to every chunk so each passage carries global video/topic context during embedding and retrieval.
5. **Zero-Crash Dual-Mode Embeddings (`get_embedding`):**
   * **Primary Mode (OpenAI):** Calls `text-embedding-3-small` (1536-d) when a valid `OPENAI_API_KEY` is configured.
   * **Automatic Fallback Mode (Local 384-d Stemmed N-Gram Hash):** Filters 100+ stopwords and spoken transcript filler (`"uh"`, `"um"`, `"right"`, `"okay"`), stems plurals (`lists` $\to$ `list`, `tuples` $\to$ `tuple`), and hashes unigrams + bigrams into a unit-normalized 384-d vector.
6. **Hybrid Lexical + Dense Reranking & Noise Filtering (`_score_and_rerank_candidates`):**
   * Queries `ChromaDB` (`PersistentClient`) for candidate chunks and combines vector distance with stemmed keyword overlap in the chunk body and video title.
   * Filters out low-coverage noise chunks (e.g., an unrelated security video that only mentions the word `"list"` in `"OWASP top 10 list"`).
7. **Live Evaluation with `rag-eval-harness` (`evaluate_with_harness.py` & `src/rag_harness/`):**
   * Evaluates every query with Information Retrieval metrics (`Recall@K`, `Precision@K`, `MRR`, `NDCG@K`, `Context Fact Coverage`) and Dual-Engine Faithfulness (`LLM-as-a-Judge` $\times$ `NLI Entailment`).

![RAG Eval Harness Benchmark Results](docs/assets/eval_benchmark.svg)

---

## 2. Project Directory & File Structure

```text
youtube-channel-rag/
├── app.py                                 # Interactive 2-tab Streamlit Web UI (Channel RAG Q&A + Live RAG Eval Harness)
├── channel_rag.py                         # Core RAG pipeline (Steps 2–4: discovery, transcripts, chunking, ChromaDB, hybrid search)
├── evaluate_with_harness.py               # Standalone benchmark runner connecting YouTube RAG to `rag-eval-harness`
├── ARCHITECTURE.md                        # Visual system architecture, file map & engineering trade-offs (this file)
├── README.md                              # Complete project guide, quickstart & "Why X vs. Why Not Y" deep dive
├── EVALUATION_REPORT.md                   # Generated Markdown evaluation report from `rag-eval-harness` (100% Pass Rate)
├── eval_report.json                       # Full structured JSON evaluation report (IR + Dual-Judge Faithfulness)
├── eval_traces.jsonl                      # Recorded execution traces for all 7 benchmark queries
├── requirements.txt                       # Python dependencies (`youtube-transcript-api`, `yt-dlp`, `chromadb`, `openai`, `streamlit`)
├── pyproject.toml                         # Python project metadata & pytest configuration
├── LICENSE                                # MIT License
├── .env.example                           # Example environment variable template for `OPENAI_API_KEY`
├── .gitignore                             # Git ignore rules for `.chroma_db/`, caches, and virtual environments
├── .github/workflows/
│   └── ci.yml                             # GitHub Actions CI workflow running unit tests & RAG Eval Harness benchmark
├── docs/assets/
│   ├── architecture_pipeline.svg          # Vector diagram of the 4-stage YouTube RAG + Eval Harness architecture
│   └── eval_benchmark.svg                 # Vector chart of benchmark metrics (Recall@5, MRR, NDCG@5, Faithfulness)
├── src/rag_harness/                       # Bundled Standalone RAG Evaluation Harness (`RAGEvaluator`, Dual-Judge NLI/LLM)
│   ├── __init__.py
│   ├── adapters.py
│   ├── evaluator.py
│   ├── schemas.py
│   ├── metrics/
│   │   ├── claims.py
│   │   ├── ensemble.py
│   │   ├── llm_judge.py
│   │   ├── nli_judge.py
│   │   └── retrieval.py
│   ├── reporting/
│   │   ├── html_dashboard.py
│   │   ├── markdown.py
│   │   └── terminal.py
│   └── triage/
│       └── classifier.py
└── tests/
    └── test_rag.py                        # 7 unit & integration tests (chunking, URL normalization, 401 safety, freeCodeCamp & Eval Harness)
```

---

## 3. Architectural Trade-Offs: Why We Chose Each Component vs. Alternatives

| Layer | Chosen Technology | Alternatives Considered | Why We Chose Our Approach |
| :--- | :--- | :--- | :--- |
| **Video Discovery** | `yt-dlp` (`extract_flat` + `/@channel/search`) | YouTube Data API v3, `pytube`, Selenium, RSS | Zero API keys, no 10,000-unit daily quota limits, supports `@handles` natively, and enables on-demand search inside 1,800-video channels like `freeCodeCamp`. |
| **Transcripts** | `youtube-transcript-api` + `yt-dlp` JSON3/Chapters fallback | OpenAI Whisper ASR, Official Captions API | Fetches existing caption tracks in `<200ms` per video at `$0` cost instead of spending hours of GPU compute transcribing multi-hour courses with Whisper. |
| **Chunking** | 500-word sliding window (50-word overlap) + Title Header | Character splitting, sentence-only splitting, full-video stuffing | Spoken auto-captions often lack punctuation (breaking sentence splitters). 500 words (~3.5 mins of speech) with a 50-word overlap preserves complete explanations without diluting embeddings. |
| **Embeddings** | OpenAI `text-embedding-3-small` + 384-d Stemmed Local Fallback | `text-embedding-3-large`, `ada-002`, heavy local GPU models | `text-embedding-3-small` is 5x cheaper and higher-scoring than `ada-002`, while our stopword-filtered 384-d fallback guarantees zero-crash execution even without an OpenAI key. |
| **Vector Store** | `ChromaDB` (`PersistentClient`) | Pinecone, FAISS, Qdrant, Weaviate, `pgvector` | Embedded on-disk vector store with built-in document storage, metadata filtering, and idempotent `.upsert()`—requiring zero Docker containers or external cloud keys. |
| **Retrieval** | Hybrid Vector Distance + Stemmed Lexical Reranking | Pure cosine similarity | Pure dense embeddings can get distracted by spoken filler words; hybrid lexical boost + coverage thresholding ensures exact technical matches rank `#1` (`MRR = 1.000`). |
| **Evaluation** | Dual-Engine `rag-eval-harness` (`LLM-Judge` $\times$ `NLI`) | Manual eyeballing, single LLM judge | Verifies both retrieval quality (`Recall@5 = 100%`, `MRR = 1.000`) and atomic claim faithfulness (`98.0%`) on every code change. |
