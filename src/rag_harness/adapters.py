"""Universal adapters for connecting any RAG pipeline (Python callable, HTTP endpoint, or JSONL trace) to the harness."""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Protocol, Sequence, Union

import httpx

from rag_harness.schemas import QuestionItem, RAGTrace, RetrievedChunk


class RAGPipelineAdapter(Protocol):
    """Protocol implemented by all RAG pipeline adapters."""

    pipeline_name: str

    def run_query(self, item: QuestionItem) -> RAGTrace:
        ...


def _coerce_chunks(raw_chunks: Sequence[Any]) -> List[RetrievedChunk]:
    chunks: List[RetrievedChunk] = []
    for idx, c in enumerate(raw_chunks, start=1):
        if isinstance(c, RetrievedChunk):
            chunk = c.model_copy(update={"rank": idx})
        elif isinstance(c, dict):
            chunk = RetrievedChunk(
                chunk_id=str(c.get("chunk_id") or c.get("id") or c.get("doc_id") or f"chunk_{idx}"),
                text=str(c.get("text") or c.get("content") or c.get("page_content") or ""),
                score=float(c.get("score", 0.0)),
                rank=int(c.get("rank", idx)),
                metadata=dict(c.get("metadata", {})),
            )
        elif isinstance(c, str):
            chunk = RetrievedChunk(chunk_id=f"chunk_{idx}", text=c, score=0.0, rank=idx)
        else:
            # Support LangChain Document / LlamaIndex NodeWithScore objects
            text = getattr(c, "page_content", None) or getattr(c, "text", None) or str(c)
            meta = getattr(c, "metadata", {}) or {}
            cid = meta.get("chunk_id") or meta.get("id") or getattr(c, "id_", None) or f"chunk_{idx}"
            score = float(getattr(c, "score", 0.0) or 0.0)
            chunk = RetrievedChunk(chunk_id=str(cid), text=str(text), score=score, rank=idx, metadata=meta)
        chunks.append(chunk)
    return chunks


class CallablePipelineAdapter:
    """Wraps any Python function or class instance into a standardized RAGPipelineAdapter."""

    def __init__(self, fn: Callable[[str], Any], pipeline_name: str = "python-callable-rag") -> None:
        self.fn = fn
        self.pipeline_name = pipeline_name

    def run_query(self, item: QuestionItem) -> RAGTrace:
        t0 = time.perf_counter()
        raw = self.fn(item.question)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        if isinstance(raw, RAGTrace):
            return raw.model_copy(
                update={
                    "query_id": item.query_id,
                    "pipeline_name": self.pipeline_name,
                    "latency_ms": raw.latency_ms or elapsed_ms,
                }
            )

        if isinstance(raw, dict):
            answer = str(
                raw.get("generated_answer")
                or raw.get("answer")
                or raw.get("response")
                or raw.get("result")
                or ""
            )
            raw_chunks = (
                raw.get("retrieved_chunks")
                or raw.get("contexts")
                or raw.get("chunks")
                or raw.get("source_documents")
                or []
            )
            latency = float(raw.get("latency_ms", elapsed_ms))
            return RAGTrace(
                query_id=item.query_id,
                question=item.question,
                generated_answer=answer,
                retrieved_chunks=_coerce_chunks(raw_chunks),
                latency_ms=latency,
                pipeline_name=self.pipeline_name,
                metadata=dict(raw.get("metadata", {})),
            )

        raise TypeError(f"Unsupported pipeline return type: {type(raw)}")


class HTTPPipelineAdapter:
    """Evaluates any live RAG HTTP endpoint (`POST /query` with `{'question': ...}`)."""

    def __init__(
        self,
        endpoint_url: str,
        pipeline_name: str = "http-rag-endpoint",
        headers: Dict[str, str] | None = None,
        timeout_sec: float = 30.0,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.pipeline_name = pipeline_name
        self.headers = headers or {}
        self.timeout_sec = timeout_sec

    def run_query(self, item: QuestionItem) -> RAGTrace:
        t0 = time.perf_counter()
        with httpx.Client(timeout=self.timeout_sec) as client:
            resp = client.post(
                self.endpoint_url,
                headers=self.headers,
                json={"query_id": item.query_id, "question": item.question},
            )
            resp.raise_for_status()
            payload = resp.json()
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        answer = str(payload.get("generated_answer") or payload.get("answer") or payload.get("response") or "")
        raw_chunks = payload.get("retrieved_chunks") or payload.get("contexts") or payload.get("chunks") or []
        return RAGTrace(
            query_id=item.query_id,
            question=item.question,
            generated_answer=answer,
            retrieved_chunks=_coerce_chunks(raw_chunks),
            latency_ms=float(payload.get("latency_ms", elapsed_ms)),
            pipeline_name=self.pipeline_name,
        )


class TraceFileAdapter:
    """Evaluates pre-recorded RAG traces from a `.jsonl` or `.json` file (language-agnostic)."""

    def __init__(self, trace_path: Union[str, Path], pipeline_name: str | None = None) -> None:
        self.trace_path = Path(trace_path)
        self.pipeline_name = pipeline_name or self.trace_path.stem
        self._traces_by_id: Dict[str, RAGTrace] = {}
        self._traces_by_question: Dict[str, RAGTrace] = {}
        self._load()

    def _load(self) -> None:
        text = self.trace_path.read_text(encoding="utf-8").strip()
        if not text:
            return
        if text.startswith("["):
            records = json.loads(text)
        else:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]

        for idx, rec in enumerate(records, start=1):
            qid = str(rec.get("query_id") or f"q_{idx}")
            qtext = str(rec.get("question") or rec.get("query") or "")
            ans = str(rec.get("generated_answer") or rec.get("answer") or rec.get("response") or "")
            raw_chunks = rec.get("retrieved_chunks") or rec.get("contexts") or rec.get("chunks") or []
            pname = str(rec.get("pipeline_name") or self.pipeline_name)
            trace = RAGTrace(
                query_id=qid,
                question=qtext,
                generated_answer=ans,
                retrieved_chunks=_coerce_chunks(raw_chunks),
                latency_ms=float(rec.get("latency_ms", 0.0)),
                pipeline_name=pname,
                metadata=dict(rec.get("metadata", {})),
            )
            self._traces_by_id[qid] = trace
            if qtext:
                self._traces_by_question[qtext.strip().lower()] = trace
            if idx == 1 and rec.get("pipeline_name"):
                self.pipeline_name = str(rec["pipeline_name"])

    def run_query(self, item: QuestionItem) -> RAGTrace:
        if item.query_id in self._traces_by_id:
            return self._traces_by_id[item.query_id]
        norm_q = item.question.strip().lower()
        if norm_q in self._traces_by_question:
            return self._traces_by_question[norm_q]
        raise KeyError(f"Query ID '{item.query_id}' not found in trace file {self.trace_path}")


def load_python_pipeline(module_spec: str, pipeline_name: str | None = None) -> CallablePipelineAdapter:
    """Load a Python callable pipeline from `path/to/file.py:func_name` or `module.name:func_name`."""
    if ":" in module_spec:
        target_path, attr_name = module_spec.rsplit(":", 1)
    else:
        target_path, attr_name = module_spec, "run_rag"

    file_candidate = Path(target_path)
    if file_candidate.exists():
        spec = importlib.util.spec_from_file_location(file_candidate.stem, file_candidate)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {file_candidate}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(target_path)

    fn_or_obj = getattr(mod, attr_name)
    if callable(fn_or_obj) and not isinstance(fn_or_obj, type):
        callable_fn = fn_or_obj
    elif isinstance(fn_or_obj, type):
        instance = fn_or_obj()
        callable_fn = getattr(instance, "query", None) or getattr(instance, "run", None) or instance
    else:
        callable_fn = getattr(fn_or_obj, "query", None) or getattr(fn_or_obj, "run", None) or fn_or_obj

    inferred_name = pipeline_name or getattr(mod, "PIPELINE_NAME", None) or file_candidate.stem
    return CallablePipelineAdapter(callable_fn, pipeline_name=str(inferred_name))


def load_question_set(path: Union[str, Path]) -> List[QuestionItem]:
    """Load a question benchmark dataset from a `.json` or `.jsonl` file."""
    p = Path(path)
    raw_text = p.read_text(encoding="utf-8").strip()
    if raw_text.startswith("["):
        records = json.loads(raw_text)
    else:
        records = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
    return [QuestionItem.model_validate(r) for r in records]
