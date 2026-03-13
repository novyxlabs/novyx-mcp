"""
Local embedding strategies for Novyx MCP semantic search.

Provides vector embeddings for local mode without requiring an API.
Supports two strategies:

1. sentence-transformers (all-MiniLM-L6-v2) — high quality, 384-dim
   Install: pip install novyx-mcp[local]

2. TF-IDF hashing trick — zero dependencies, usable similarity
   Uses stdlib only, fixed 384-dim vectors via hashing
"""

from __future__ import annotations

import math
import re
import struct
from typing import Optional


# Dimension matching Novyx Cloud (all-MiniLM-L6-v2)
EMBEDDING_DIM = 384

# Common English stop words to filter from TF-IDF
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "it", "of", "in", "to", "and", "or", "for",
    "on", "at", "by", "with", "from", "as", "this", "that", "was", "are",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can", "shall",
    "not", "no", "but", "if", "so", "than", "too", "very", "just",
    "about", "into", "over", "after", "before", "between", "through",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "they",
    "them", "its", "his", "her", "their", "what", "which", "who", "whom",
})


class LocalEmbedder:
    """Abstraction over embedding strategies for local semantic search."""

    def __init__(self) -> None:
        self._model = None
        self._strategy: str = "none"
        self._try_load()

    def _try_load(self) -> None:
        """Attempt to load the best available embedding strategy."""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self._strategy = "transformer"
            return
        except ImportError:
            pass
        # TF-IDF fallback — always available
        self._strategy = "tfidf"

    @property
    def strategy(self) -> str:
        """Current embedding strategy: 'transformer', 'tfidf', or 'none'."""
        return self._strategy

    def embed(self, text: str) -> Optional[list[float]]:
        """Generate an embedding vector for the given text.

        Returns:
            A list of floats (384-dim) or None if no strategy is available.
        """
        if self._strategy == "transformer":
            vec = self._model.encode(text).tolist()
            return vec
        elif self._strategy == "tfidf":
            return self._tfidf_vector(text)
        return None

    def similarity(self, a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _tfidf_vector(self, text: str) -> list[float]:
        """Hash-based TF-IDF vector using only stdlib.

        Tokenizes text, hashes each token to a bucket in a fixed-dimension
        vector, and accumulates term frequency weights. Produces usable
        (not great) semantic similarity with zero external dependencies.
        """
        tokens = _tokenize(text)
        if not tokens:
            return [0.0] * EMBEDDING_DIM

        vec = [0.0] * EMBEDDING_DIM
        for token in tokens:
            # Hash to bucket, use two hashes for sign (simhash-style)
            bucket = hash(token) % EMBEDDING_DIM
            sign = 1 if hash(token + "_sign") % 2 == 0 else -1
            vec[bucket] += sign * 1.0

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]

        return vec


def pack_embedding(vec: list[float]) -> bytes:
    """Pack a float list into a compact bytes blob for SQLite storage."""
    return struct.pack(f"{len(vec)}f", *vec)


def unpack_embedding(data: bytes) -> list[float]:
    """Unpack a bytes blob back into a float list."""
    count = len(data) // 4
    return list(struct.unpack(f"{count}f", data))


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase, split on non-alpha, remove stop words."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 1]
