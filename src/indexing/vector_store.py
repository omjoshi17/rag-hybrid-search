"""Persist embedded chunks into ChromaDB with cosine-similarity deduplication."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.data.loader import read_jsonl
from src.indexing.embedder import EmbeddingConfig, OpenAIEmbedder, cosine_similarity


LOGGER = logging.getLogger(__name__)

ChunkRecord = dict[str, Any]


@dataclass
class DuplicateMatch:
    chunk_id: str
    similarity: float
    source: str | None


@dataclass
class IndexingStats:
    seen: int = 0
    inserted: int = 0
    skipped_empty: int = 0
    skipped_duplicate: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "seen": self.seen,
            "inserted": self.inserted,
            "skipped_empty": self.skipped_empty,
            "skipped_duplicate": self.skipped_duplicate,
        }


def get_chroma_collection(persist_dir: str | Path, collection_name: str):
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("Install chromadb to persist vector indexes.") from exc

    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_path))
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def reset_chroma_collection(persist_dir: str | Path, collection_name: str):
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("Install chromadb to persist vector indexes.") from exc

    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_path))
    try:
        client.delete_collection(collection_name)
    except ValueError:
        pass
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def normalize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    normalized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            normalized[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            normalized[key] = value
        else:
            normalized[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return normalized


def chunk_id(chunk: ChunkRecord) -> str:
    metadata = chunk.get("metadata") or {}
    if not metadata.get("chunk_id"):
        raise ValueError("Chunk metadata is missing chunk_id.")
    return str(metadata["chunk_id"])


def chroma_distance_to_similarity(distance: float) -> float:
    # Collections are created with hnsw:space=cosine, where smaller distance is better.
    return 1.0 - distance


def find_duplicate_in_chroma(collection, embedding: list[float], threshold: float) -> DuplicateMatch | None:
    if collection.count() == 0:
        return None

    result = collection.query(
        query_embeddings=[embedding],
        n_results=1,
        include=["distances", "metadatas"],
    )
    distances = result.get("distances") or [[]]
    if not distances[0]:
        return None

    similarity = chroma_distance_to_similarity(float(distances[0][0]))
    if similarity <= threshold:
        return None

    ids = result.get("ids") or [[]]
    metadatas = result.get("metadatas") or [[]]
    metadata = metadatas[0][0] if metadatas and metadatas[0] else {}
    return DuplicateMatch(
        chunk_id=str(ids[0][0]) if ids and ids[0] else "",
        similarity=similarity,
        source=metadata.get("source") if isinstance(metadata, dict) else None,
    )


def find_duplicate_in_pending(
    pending: list[tuple[str, list[float], dict[str, Any]]],
    embedding: list[float],
    threshold: float,
) -> DuplicateMatch | None:
    for pending_id, pending_embedding, pending_metadata in pending:
        similarity = cosine_similarity(embedding, pending_embedding)
        if similarity > threshold:
            return DuplicateMatch(
                chunk_id=pending_id,
                similarity=similarity,
                source=pending_metadata.get("source"),
            )
    return None


def add_pending(collection, pending: list[tuple[str, list[float], dict[str, Any], str]]) -> int:
    if not pending:
        return 0

    collection.add(
        ids=[item[0] for item in pending],
        embeddings=[item[1] for item in pending],
        metadatas=[normalize_metadata(item[2]) for item in pending],
        documents=[item[3] for item in pending],
    )
    return len(pending)


def index_chunks(
    chunks: list[ChunkRecord],
    collection,
    embedder: OpenAIEmbedder,
    dedupe_threshold: float = 0.95,
    add_batch_size: int = 32,
) -> IndexingStats:
    stats = IndexingStats()
    pending: list[tuple[str, list[float], dict[str, Any], str]] = []

    texts = [str(chunk.get("content") or "") for chunk in chunks]
    embeddings = embedder.embed_texts(texts) if texts else []

    for chunk, embedding in zip(chunks, embeddings):
        stats.seen += 1
        content = str(chunk.get("content") or "").strip()
        metadata = dict(chunk.get("metadata") or {})
        current_id = chunk_id(chunk)

        if not content:
            stats.skipped_empty += 1
            continue

        duplicate = find_duplicate_in_pending(
            [(item[0], item[1], item[2]) for item in pending],
            embedding,
            dedupe_threshold,
        ) or find_duplicate_in_chroma(collection, embedding, dedupe_threshold)

        if duplicate:
            stats.skipped_duplicate += 1
            LOGGER.info(
                "Skipping duplicate chunk %s; closest=%s similarity=%.4f source=%s",
                current_id,
                duplicate.chunk_id,
                duplicate.similarity,
                duplicate.source,
            )
            continue

        metadata["embedding_model"] = embedder.model
        pending.append((current_id, embedding, metadata, content))

        if len(pending) >= add_batch_size:
            stats.inserted += add_pending(collection, pending)
            pending = []

    stats.inserted += add_pending(collection, pending)
    return stats


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Index chunks into local ChromaDB.")
    parser.add_argument("--chunks", default="data/processed/chunks_recursive.jsonl", help="Chunk JSONL path.")
    parser.add_argument("--persist-dir", default=os.getenv("CHROMA_PERSIST_DIR", "data/chroma"))
    parser.add_argument("--collection", default=os.getenv("CHROMA_COLLECTION", "internal_docs"))
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--dedupe-threshold", type=float, default=0.95)
    parser.add_argument("--reset", action="store_true", help="Delete and recreate the Chroma collection first.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    chunks = read_jsonl(args.chunks)
    collection = (
        reset_chroma_collection(args.persist_dir, args.collection)
        if args.reset
        else get_chroma_collection(args.persist_dir, args.collection)
    )
    embedder = OpenAIEmbedder(EmbeddingConfig(model=args.model))
    stats = index_chunks(chunks, collection, embedder, dedupe_threshold=args.dedupe_threshold)
    LOGGER.info("Indexing complete: %s", stats.as_dict())


if __name__ == "__main__":
    main()
