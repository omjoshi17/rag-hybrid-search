"""Run the Phase 1 ingestion, chunking, and optional indexing pipeline."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.chunker import ChunkingConfig, chunk_documents
from src.data.loader import load_documents, write_jsonl


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description="Run Phase 1: load, chunk, embed, and index docs.")
    parser.add_argument("--docs-dir", default="data/docs", help="Directory containing raw documents.")
    parser.add_argument("--processed-dir", default="data/processed", help="Where JSONL outputs should be written.")
    parser.add_argument(
        "--strategy",
        default="recursive",
        choices=["recursive", "markdown_headers", "semantic_markdown", "fixed", "all"],
        help="Chunking strategy.",
    )
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--skip-index", action="store_true", help="Only load and chunk; do not call OpenAI/Chroma.")
    parser.add_argument("--persist-dir", default=os.getenv("CHROMA_PERSIST_DIR", "data/chroma"))
    parser.add_argument("--collection", default=os.getenv("CHROMA_COLLECTION", "internal_docs"))
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--dedupe-threshold", type=float, default=0.95)
    parser.add_argument("--reset", action="store_true", help="Delete and recreate the Chroma collection before indexing.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    docs_dir = resolve_project_path(args.docs_dir)
    processed_dir = resolve_project_path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    documents_path = processed_dir / "documents.jsonl"
    chunks_path = processed_dir / f"chunks_{args.strategy}.jsonl"

    LOGGER.info("Loading documents from %s", docs_dir)
    documents = load_documents(docs_dir)
    write_jsonl(documents, documents_path)
    LOGGER.info("Wrote %s normalized document records to %s", len(documents), documents_path)

    config = ChunkingConfig(strategy=args.strategy, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks = chunk_documents(documents, config)
    write_jsonl(chunks, chunks_path)
    write_jsonl(chunks, processed_dir / "chunks_latest.jsonl")
    LOGGER.info("Wrote %s chunks to %s", len(chunks), chunks_path)

    if args.skip_index:
        LOGGER.info("Skipping indexing because --skip-index was set.")
        return

    from src.indexing.embedder import EmbeddingConfig, OpenAIEmbedder
    from src.indexing.vector_store import get_chroma_collection, index_chunks, reset_chroma_collection

    persist_dir = resolve_project_path(args.persist_dir)
    collection = (
        reset_chroma_collection(persist_dir, args.collection)
        if args.reset
        else get_chroma_collection(persist_dir, args.collection)
    )
    embedder = OpenAIEmbedder(EmbeddingConfig(model=args.embedding_model))
    stats = index_chunks(chunks, collection, embedder, dedupe_threshold=args.dedupe_threshold)
    LOGGER.info("Indexed chunks into Chroma collection '%s': %s", args.collection, stats.as_dict())


if __name__ == "__main__":
    main()
