"""OpenAI embedding utilities for Phase 1 indexing."""

from __future__ import annotations

import argparse
import logging
import math
import os
from dataclasses import dataclass
from typing import Iterable

from dotenv import load_dotenv


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = "text-embedding-3-small"
    batch_size: int = 64


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


class OpenAIEmbedder:
    """Thin wrapper around OpenAI embeddings with deterministic batching."""

    def __init__(self, config: EmbeddingConfig | None = None, api_key: str | None = None) -> None:
        load_dotenv()
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install openai to generate embeddings.") from exc

        self.config = config or EmbeddingConfig(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for embedding generation.")

        self.client = OpenAI(api_key=resolved_api_key)

    @property
    def model(self) -> str:
        return self.config.model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        clean_texts = [text if text.strip() else " " for text in texts]
        embeddings: list[list[float]] = []

        for batch in batched(clean_texts, self.config.batch_size):
            LOGGER.debug("Embedding batch of %s texts with %s", len(batch), self.model)
            response = self.client.embeddings.create(model=self.model, input=batch)
            ordered = sorted(response.data, key=lambda item: item.index)
            embeddings.extend(item.embedding for item in ordered)

        return embeddings

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding vectors must have the same dimensionality.")

    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an embedding for a text snippet.")
    parser.add_argument("text", help="Text to embed.")
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embedder = OpenAIEmbedder(EmbeddingConfig(model=args.model))
    embedding = embedder.embed_query(args.text)
    print({"model": embedder.model, "dimensions": len(embedding), "preview": embedding[:5]})


if __name__ == "__main__":
    main()
