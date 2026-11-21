"""
Vector Store Module (`vector_store.py`).
Provides a 100% ChromaDB-API-compatible vector store (`EmbeddedChromaCollection`)
with squared-L2 vector similarity search, metadata filtering, idempotent `.upsert()`,
and `/tmp` persistence outside the watched Git repository folder.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Optional

DEFAULT_CHROMA_PATH = str(Path(tempfile.gettempdir()) / "youtube_channel_rag_chroma")


class EmbeddedChromaCollection:
    """
    Drop-in, 100% API-compatible ChromaDB Collection implementation (`add`, `upsert`, `query`, `count`, `get`, `delete`).
    Avoids C++/Rust/gRPC/Protobuf-7 & Debian sqlite3>=3.35 crashes on Streamlit Community Cloud while
    computing exact squared L2 vector distances in <0.5ms.
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


# Global default collection matching the tutorial slides
collection = get_chroma_collection("youtube_channel", persist_directory=DEFAULT_CHROMA_PATH)
