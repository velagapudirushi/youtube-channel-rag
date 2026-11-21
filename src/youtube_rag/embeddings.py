"""
Embeddings Module (`embeddings.py`).
Handles OpenAI `text-embedding-3-small` API calls, API key validation (preventing 401 crashes
on placeholder keys like 'sk-...'), and deterministic 384-d local stemmed n-gram fallback embeddings.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Optional

from openai import OpenAI

from youtube_rag.chunking import extract_content_keywords


def is_valid_openai_key(api_key: Optional[str]) -> bool:
    """
    Validate whether a candidate OpenAI API key is a real key and not a placeholder
    such as 'sk-...', 'sk-....', or 'sk-your-openai-api-key-here'.
    """
    if not api_key:
        return False
    key = api_key.strip()
    if len(key) < 30:
        return False
    if "..." in key or "your-" in key.lower() or "placeholder" in key.lower():
        return False
    if not key.startswith("sk-"):
        return False
    return True


def _get_openai_client(api_key: Optional[str] = None) -> Optional[OpenAI]:
    """Return an initialized OpenAI client only if a valid non-placeholder API key is available."""
    candidate = (api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")).strip()
    if not is_valid_openai_key(candidate):
        return None
    return OpenAI(api_key=candidate)


def _local_deterministic_embedding(text: str, dim: int = 384) -> list[float]:
    """
    Stopword-filtered, stemmed local word + bigram hashing embedding (384-d unit vector).
    By removing spoken filler words ("uh", "um", "what", "this") and normalizing plurals
    ("lists" -> "list", "tuples" -> "tuple"), cosine similarity focuses strictly on technical content.
    """
    vec = [0.0] * dim
    tokens = extract_content_keywords(text)
    if not tokens:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
    if not tokens:
        return vec

    features: list[tuple[str, float]] = [(t, 1.0) for t in tokens]
    for i in range(len(tokens) - 1):
        features.append((f"{tokens[i]}_{tokens[i+1]}", 1.25))

    for feat, weight in features:
        digest = hashlib.md5(feat.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        vec[idx] += sign * weight

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def get_embedding(text: str, api_key: Optional[str] = None) -> list[float]:
    """
    Compute embedding using OpenAI's `text-embedding-3-small` model when a valid OPENAI_API_KEY
    is available, and automatically fall back to local 384-d embeddings if the key is missing or invalid.
    """
    client = _get_openai_client(api_key)
    if client is not None:
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
            )
            return response.data[0].embedding
        except Exception as exc:
            print(f"⚠️ OpenAI embedding call failed ({exc}); falling back to local 384-d embedding.")
    return _local_deterministic_embedding(text)
