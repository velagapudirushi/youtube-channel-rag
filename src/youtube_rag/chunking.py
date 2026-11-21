"""
Text Chunking, Stopword Filtering, and Plural Stemming Module (`chunking.py`).
Splits raw YouTube transcripts into 500-word sliding windows with a 50-word overlap
and extracts meaningful content keywords for embedding and hybrid reranking.
"""

from __future__ import annotations

import re

# Stopwords and spoken transcript filler words excluded from local embeddings & keyword reranking
STOPWORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "chanel", "channel", "channels", "channle", "could", "course", "courses",
    "creator", "did", "difference", "different", "do", "does", "doing", "done",
    "each", "even", "every", "explain", "explains", "for", "from", "get", "getting",
    "give", "go", "going", "good", "got", "had", "has", "have", "having", "he",
    "her", "here", "hers", "him", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "just", "know", "learn", "let", "lets", "like", "look", "make",
    "many", "may", "me", "might", "more", "most", "much", "must", "my", "need",
    "no", "not", "now", "of", "off", "okay", "on", "one", "only", "or", "other",
    "our", "out", "over", "really", "right", "said", "same", "say", "says", "see",
    "she", "should", "so", "some", "something", "such", "take", "talk", "teach",
    "teaches", "tell", "than", "that", "the", "their", "them", "then", "there",
    "these", "they", "thing", "things", "think", "this", "those", "through",
    "time", "to", "too", "tutorial", "uh", "um", "up", "us", "use", "used",
    "using", "very", "video", "videos", "vs", "versus", "want", "was", "wat",
    "way", "we", "well", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "would", "yeah", "you", "your", "youtube",
}


def _stem_token(word: str) -> str:
    """Lightweight plural/suffix normalizer so 'lists'=='list', 'tuples'=='tuple', 'transformers'=='transformer'."""
    w = word.lower()
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 4 and not w.endswith(
        ("tuples", "variables", "classes", "files", "types", "modules", "sequences")
    ):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is", "as", "os")):
        return w[:-1]
    return w


def extract_content_keywords(text: str) -> list[str]:
    """Extract meaningful non-stopword stemmed tokens from a question or document."""
    raw_tokens = re.findall(r"[a-z0-9]+", text.lower())
    result = []
    for tok in raw_tokens:
        if tok in STOPWORDS or len(tok) < 2:
            continue
        stemmed = _stem_token(tok)
        if stemmed not in STOPWORDS and len(stemmed) >= 2:
            result.append(stemmed)
    return result


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split transcript into overlapping word chunks (500 words, 50-word overlap)."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


def _strip_chunk_header(doc: str) -> str:
    """Strip the '<<<Channel: ... | Video Title: ...>>>' or '[Channel: ...]' prefix for clean display."""
    cleaned = re.sub(r"^<<<Channel:.*?>>>\s*", "", doc)
    cleaned = re.sub(r"^\[Channel:[^\n]*?\]\s*(?:\]?\s*)", "", cleaned)
    return cleaned.strip()
