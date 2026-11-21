"""
LangChain Vector Store & Chroma Dense Retriever Module (`vector_store.py`).
Provides `EmbeddedChromaCollection` and `ChromaDenseRetriever(BaseRetriever)` subclassing
`langchain_core.retrievers.BaseRetriever` to retrieve the Top-K dense vector matches as
LangChain `Document` objects.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from youtube_rag.embeddings import _local_deterministic_embedding, get_embedding

DEFAULT_CHROMA_PATH = str(Path(tempfile.gettempdir()) / "youtube_channel_rag_chroma")


class EmbeddedChromaCollection:
    """
    Drop-in, 100% API-compatible ChromaDB Collection implementation (`add`, `upsert`, `query`, `count`, `get`, `delete`)
    with LangChain `Document` conversion helpers.
    """

    def __init__(self, name: str, persist_path: Optional[Path] = None) -> None:
        self.name = name
        self.persist_path = persist_path
        self._records: dict[str, dict[str, Any]] = {}
        if self.persist_path and self.persist_path.exists():
            try:
                data = json.loads(self.persist_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._records = data
            except Exception:
                self._records = {}

    def _save(self) -> None:
        if not self.persist_path:
            return
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            self.persist_path.write_text(json.dumps(self._records), encoding="utf-8")
        except Exception:
            pass

    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        self.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        metas = metadatas or [{} for _ in ids]
        for doc_id, doc, emb, meta in zip(ids, documents, embeddings, metas):
            self._records[str(doc_id)] = {
                "id": str(doc_id),
                "document": str(doc),
                "embedding": [float(x) for x in emb],
                "metadata": dict(meta or {}),
            }
        self._save()

    def count(self) -> int:
        return len(self._records)

    def get(self, ids: Optional[list[str]] = None) -> dict[str, list[Any]]:
        items = (
            [self._records[i] for i in ids if i in self._records]
            if ids is not None
            else list(self._records.values())
        )
        return {
            "ids": [it["id"] for it in items],
            "documents": [it["document"] for it in items],
            "metadatas": [it["metadata"] for it in items],
        }

    def as_langchain_documents(self, embed_mode: Optional[str] = None) -> list[Document]:
        """Export stored records as LangChain `Document` objects for BM25 & Ensemble retrieval."""
        docs: list[Document] = []
        for rec in self._records.values():
            meta = dict(rec.get("metadata", {}))
            if embed_mode and meta.get("embed_mode") and meta.get("embed_mode") != embed_mode:
                continue
            meta["doc_id"] = rec["id"]
            docs.append(Document(page_content=rec["document"], metadata=meta))
        return docs

    def delete(self, ids: Optional[list[str]] = None) -> None:
        if ids is None:
            self._records.clear()
        else:
            for doc_id in ids:
                self._records.pop(str(doc_id), None)
        self._save()

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int = 5,
        where: Optional[dict[str, Any]] = None,
    ) -> dict[str, list[list[Any]]]:
        if not query_embeddings or not self._records:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        q_emb = query_embeddings[0]
        q_dim = len(q_emb)
        scored: list[tuple[float, dict[str, Any]]] = []

        for rec in self._records.values():
            meta = rec.get("metadata", {})
            if where:
                if not all(meta.get(k) == v for k, v in where.items()):
                    continue
            emb = rec.get("embedding", [])
            if len(emb) != q_dim:
                continue
            dist = sum((a - b) * (a - b) for a, b in zip(q_emb, emb))
            scored.append((dist, rec))

        scored.sort(key=lambda x: x[0])
        top = scored[: max(1, n_results)]
        return {
            "ids": [[item[1]["id"] for item in top]],
            "documents": [[item[1]["document"] for item in top]],
            "metadatas": [[item[1]["metadata"] for item in top]],
            "distances": [[round(float(item[0]), 6) for item in top]],
        }


class ChromaDenseRetriever(BaseRetriever):
    """
    LangChain `BaseRetriever` wrapper over ChromaDB that retrieves Top-K `Document` chunks
    via dense vector similarity (`text-embedding-3-small` or 384-d local embeddings).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    target_collection: Any
    k: int = 5
    api_key: Optional[str] = None
    embed_mode: str = "local"

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> list[Document]:
        total = self.target_collection.count()
        if total == 0:
            return []

        q_emb = get_embedding(query, api_key=self.api_key)
        candidate_k = min(max(self.k * 4, 20), total)
        try:
            res = self.target_collection.query(
                query_embeddings=[q_emb],
                n_results=candidate_k,
                where={"embed_mode": self.embed_mode},
            )
            raw_docs = res.get("documents", [[]])[0] if res.get("documents") else []
            if not raw_docs:
                raise ValueError("Empty result for embed_mode filter")
        except Exception:
            fallback_emb = _local_deterministic_embedding(query) if self.embed_mode == "openai" else q_emb
            res = self.target_collection.query(
                query_embeddings=[fallback_emb],
                n_results=candidate_k,
            )

        raw_docs = res.get("documents", [[]])[0] if res.get("documents") else []
        metadatas = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
        distances = res.get("distances", [[]])[0] if res.get("distances") else []

        lc_docs: list[Document] = []
        for idx, (doc_text, meta) in enumerate(zip(raw_docs, metadatas)):
            dist = float(distances[idx]) if idx < len(distances) else 1.0
            enriched_meta = dict(meta or {})
            enriched_meta["dense_distance"] = dist
            enriched_meta["dense_rank"] = idx + 1
            lc_docs.append(Document(page_content=doc_text, metadata=enriched_meta))
        return lc_docs[: self.k]


def get_chroma_collection(
    collection_name: str = "youtube_channel",
    persist_directory: Optional[str] = DEFAULT_CHROMA_PATH,
    in_memory: bool = False,
) -> EmbeddedChromaCollection:
    """Get or create the vector collection (persisted in /tmp outside the git repo)."""
    if in_memory or not persist_directory:
        return EmbeddedChromaCollection(name=collection_name, persist_path=None)
    store_file = Path(persist_directory) / f"{collection_name}.json"
    return EmbeddedChromaCollection(name=collection_name, persist_path=store_file)


def clear_collection(target_collection: Any = None) -> None:
    """Remove all existing documents from the vector collection."""
    active_collection = target_collection if target_collection is not None else collection
    existing = active_collection.get()
    if existing and existing.get("ids"):
        active_collection.delete(ids=existing["ids"])


collection = get_chroma_collection("youtube_channel", persist_directory=DEFAULT_CHROMA_PATH)
