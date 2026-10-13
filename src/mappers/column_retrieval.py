"""Column retrieval for the Single-Prompt mapper: lexical, semantic or hybrid.

The lexical selector (``predicate_filter.select_relevant_columns``) misses columns
whose description uses other words than the sentence (e.g. "high-cost mortgage"
vs. "HOEPA status"). The semantic selector ranks columns by cosine similarity
between embeddings of the sentence and of each column document (name,
description, domain values); embeddings come from OpenRouter. The hybrid
selector fuses both rankings (reciprocal rank fusion).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.mappers.predicate_filter import select_relevant_columns

DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
RRF_K = 60  # reciprocal rank fusion constant


def column_document(column: Dict[str, Any]) -> str:
    """Text embedded for a column: name (split), description and small domains."""
    parts = [column["name"].replace("_", " ")]
    if column.get("description"):
        parts.append(" ".join(str(column["description"]).split()))
    if column.get("domain_values") is not None:
        parts.append(f"Values: {column['domain_values']}")
    return ". ".join(parts)


class OpenRouterEmbedder:
    """Embeddings through OpenRouter (OpenAI-compatible /embeddings), with a
    small on-disk cache keyed by model and text, and retries on transient errors."""

    def __init__(self, model: str = DEFAULT_EMBEDDING_MODEL, cache_dir: Optional[str] = None):
        from openai import OpenAI

        from src.llm.client import OPENROUTER_BASE_URL

        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=OPENROUTER_BASE_URL)
        self.cache_dir = Path(cache_dir or os.environ.get("NORMA_EMBED_CACHE", Path.home() / ".cache" / "norma" / "embeddings"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, text: str) -> Path:
        return self.cache_dir / (hashlib.sha256(f"{self.model}\n{text}".encode()).hexdigest() + ".json")

    def __call__(self, texts: Sequence[str]) -> List[List[float]]:
        out: List[Optional[List[float]]] = [None] * len(texts)
        missing = []
        for i, text in enumerate(texts):
            path = self._key(text)
            if path.exists():
                out[i] = json.loads(path.read_text())
            else:
                missing.append(i)
        for start in range(0, len(missing), 96):
            batch = missing[start : start + 96]
            for attempt in range(5):
                try:
                    response = self.client.embeddings.create(model=self.model, input=[texts[i] for i in batch])
                    break
                except Exception as error:  # transient API errors: back off and retry
                    if attempt == 4:
                        raise
                    print(f"Warning: embedding request failed ({error}); retrying")
                    time.sleep(5 * 2**attempt)
            for i, item in zip(batch, sorted(response.data, key=lambda d: d.index)):
                out[i] = list(item.embedding)
                self._key(texts[i]).write_text(json.dumps(out[i]))
        return out  # type: ignore[return-value]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class ColumnSelector:
    """Select the columns relevant to a sentence.

    mode: "lexical" (current behaviour), "semantic" (embeddings) or "hybrid"
    (reciprocal rank fusion of both rankings).
    """

    def __init__(
        self,
        tables: List[Dict[str, Any]],
        mode: str = "lexical",
        embedder: Optional[Callable[[Sequence[str]], List[List[float]]]] = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        if mode not in ("lexical", "semantic", "hybrid"):
            raise ValueError(f"Unknown column retrieval mode: {mode}")
        self.tables = tables
        self.mode = mode
        self.columns = [c for t in tables for c in t.get("columns", [])]
        self.names = [c["name"] for c in self.columns]
        self._vectors: Optional[List[List[float]]] = None
        self.embedder = embedder
        if mode != "lexical" and embedder is None:
            self.embedder = OpenRouterEmbedder(embedding_model)

    def _column_vectors(self) -> List[List[float]]:
        if self._vectors is None:
            self._vectors = self.embedder([column_document(c) for c in self.columns])
        return self._vectors

    def semantic_ranking(self, sentence: str) -> List[str]:
        query = self.embedder([sentence])[0]
        scores = [_cosine(query, v) for v in self._column_vectors()]
        return [name for _, name in sorted(zip(scores, self.names), key=lambda x: (-x[0], x[1]))]

    def select(self, sentence: str, max_columns: int = 12) -> List[str]:
        if self.mode == "lexical":
            return select_relevant_columns(sentence, self.tables, max_columns)
        semantic = self.semantic_ranking(sentence)
        if self.mode == "semantic":
            return semantic[:max_columns]
        lexical = select_relevant_columns(sentence, self.tables, len(self.names))
        fused: Dict[str, float] = {}
        for ranking in (lexical, semantic):
            for rank, name in enumerate(ranking):
                fused[name] = fused.get(name, 0.0) + 1.0 / (RRF_K + rank + 1)
        return [n for n, _ in sorted(fused.items(), key=lambda x: (-x[1], x[0]))][:max_columns]
