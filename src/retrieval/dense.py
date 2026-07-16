"""Dense retrieval from ChromaDB using OpenAI query embeddings."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.indexing.embedder import EmbeddingConfig, OpenAIEmbedder
from src.indexing.vector_store import chroma_distance_to_similarity, get_chroma_collection
from src.retrieval.types import RetrievalResult, make_result


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DenseRetrievalConfig:
    persist_dir: str = "data/chroma"
    collection_name: str = "internal_docs"
    embedding_model: str = "text-embedding-3-small"
    top_k: int = 10


class DenseRetriever:
    """Query the Chroma vector index with an embedded user question."""

    def __init__(self, collection: Any, embedder: OpenAIEmbedder) -> None:
        self.collection = collection
        self.embedder = embedder

    @classmethod
    def from_config(cls, config: DenseRetrievalConfig) -> "DenseRetriever":
        collection = get_chroma_collection(config.persist_dir, config.collection_name)
        embedder = OpenAIEmbedder(EmbeddingConfig(model=config.embedding_model))
        return cls(collection=collection, embedder=embedder)

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("query cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if self.collection.count() == 0:
            LOGGER.warning("Chroma collection is empty.")
            return []

        query_embedding = self.embedder.embed_query(query)
        raw = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return parse_chroma_results(raw)


def parse_chroma_results(raw: dict[str, Any]) -> list[RetrievalResult]:
    ids = (raw.get("ids") or [[]])[0]
    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]

    results: list[RetrievalResult] = []
    for index, item_id in enumerate(ids):
        metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
        content = str(documents[index] or "") if index < len(documents) else ""
        distance = float(distances[index]) if index < len(distances) else 1.0
        results.append(
            make_result(
                result_id_value=str(item_id),
                content=content,
                metadata=metadata,
                score=chroma_distance_to_similarity(distance),
                rank=index + 1,
                retriever="dense",
            )
        )
    return results


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run dense retrieval against Chroma.")
    parser.add_argument("query", help="User question.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--persist-dir", default=os.getenv("CHROMA_PERSIST_DIR", "data/chroma"))
    parser.add_argument("--collection", default=os.getenv("CHROMA_COLLECTION", "internal_docs"))
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    retriever = DenseRetriever.from_config(
        DenseRetrievalConfig(
            persist_dir=str(Path(args.persist_dir)),
            collection_name=args.collection,
            embedding_model=args.model,
            top_k=args.top_k,
        )
    )
    print(json.dumps(retriever.retrieve(args.query, top_k=args.top_k), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
