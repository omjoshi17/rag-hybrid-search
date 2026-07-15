"""Chunk normalized document records for indexing.

Runnable examples:

    python -m src.data.chunker --documents data/processed/documents.jsonl
    python -m src.data.chunker --docs-dir data/docs --strategy markdown_headers
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.data.loader import load_documents, read_jsonl, write_jsonl


LOGGER = logging.getLogger(__name__)

MARKDOWN_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

DocumentRecord = dict[str, Any]
ChunkRecord = dict[str, Any]


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 150

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")


def recursive_split(text: str, config: ChunkingConfig) -> list[str]:
    """Split text with LangChain's RecursiveCharacterTextSplitter."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise RuntimeError("Install langchain-text-splitters to use recursive chunking.") from exc

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n# ", "\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
    )
    return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]


def fixed_split(text: str, config: ChunkingConfig) -> list[str]:
    chunks: list[str] = []
    step = config.chunk_size - config.chunk_overlap
    start = 0
    while start < len(text):
        chunk = text[start : start + config.chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def markdown_sections(text: str) -> list[tuple[str, str | None]]:
    """Split Markdown text into sections and return (section_text, heading_path)."""
    lines = text.splitlines()
    sections: list[tuple[str, str | None]] = []
    heading_stack: list[str] = []
    current_lines: list[str] = []
    current_heading: str | None = None

    def flush() -> None:
        nonlocal current_lines, current_heading
        section_text = "\n".join(current_lines).strip()
        if section_text:
            sections.append((section_text, current_heading))
        current_lines = []

    for line in lines:
        match = MARKDOWN_HEADER_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            heading_stack[:] = heading_stack[: level - 1]
            heading_stack.append(title)
            current_heading = " > ".join(heading_stack)
        current_lines.append(line)

    flush()
    return sections or [(text.strip(), None)]


def markdown_header_split(text: str, metadata: dict[str, Any], config: ChunkingConfig) -> list[tuple[str, str | None]]:
    if metadata.get("content_type") != "markdown":
        return [(chunk, metadata.get("section_heading")) for chunk in recursive_split(text, config)]

    chunks: list[tuple[str, str | None]] = []
    for section_text, heading in markdown_sections(text):
        if len(section_text) <= config.chunk_size:
            chunks.append((section_text, heading))
            continue
        for chunk in recursive_split(section_text, config):
            chunks.append((chunk, heading))
    return chunks


def stable_chunk_id(metadata: dict[str, Any], strategy: str, chunk_index: int, content: str) -> str:
    payload = {
        "source": metadata.get("source"),
        "page": metadata.get("page"),
        "strategy": strategy,
        "chunk_index": chunk_index,
        "content_sha1": hashlib.sha1(content.encode("utf-8")).hexdigest(),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    source = str(metadata.get("source") or "unknown").replace("/", "_").replace("\\", "_")
    return f"{source}:{strategy}:{chunk_index}:{digest}"


def build_chunk(
    content: str,
    source_metadata: dict[str, Any],
    strategy: str,
    chunk_index: int,
    section_heading: str | None,
) -> ChunkRecord:
    metadata = dict(source_metadata)
    metadata.update(
        {
            "chunk_id": stable_chunk_id(source_metadata, strategy, chunk_index, content),
            "chunk_index": chunk_index,
            "strategy": strategy,
            "section_heading": section_heading,
            "char_count": len(content),
        }
    )
    return {"content": content, "metadata": metadata}


def chunk_single_document(document: DocumentRecord, config: ChunkingConfig) -> list[ChunkRecord]:
    text = str(document.get("content") or "").strip()
    if not text:
        return []

    metadata = dict(document.get("metadata") or {})
    strategy = config.strategy

    if strategy == "recursive":
        raw_chunks = [(chunk, metadata.get("section_heading")) for chunk in recursive_split(text, config)]
    elif strategy == "fixed":
        raw_chunks = [(chunk, metadata.get("section_heading")) for chunk in fixed_split(text, config)]
    elif strategy in {"markdown_headers", "semantic_markdown"}:
        raw_chunks = markdown_header_split(text, metadata, config)
    else:
        raise ValueError(f"Unsupported chunking strategy: {strategy}")

    return [
        build_chunk(content, metadata, strategy, chunk_index, section_heading)
        for chunk_index, (content, section_heading) in enumerate(raw_chunks)
    ]


def chunk_documents(documents: Iterable[DocumentRecord], config: ChunkingConfig) -> list[ChunkRecord]:
    documents = list(documents)
    if config.strategy == "all":
        chunks: list[ChunkRecord] = []
        for strategy in ("recursive", "semantic_markdown", "fixed"):
            strategy_config = ChunkingConfig(
                strategy=strategy,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
            chunks.extend(chunk_documents(documents, strategy_config))
        return chunks

    output: list[ChunkRecord] = []
    for document in documents:
        output.extend(chunk_single_document(document, config))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk normalized documents.")
    parser.add_argument("--documents", default=None, help="Input JSONL from loader.py.")
    parser.add_argument("--docs-dir", default="data/docs", help="Raw docs directory if --documents is omitted.")
    parser.add_argument("--out", default="data/processed/chunks_recursive.jsonl", help="Output chunks JSONL path.")
    parser.add_argument(
        "--strategy",
        default="recursive",
        choices=["recursive", "markdown_headers", "semantic_markdown", "fixed", "all"],
        help="Chunking strategy to apply.",
    )
    parser.add_argument("--chunk-size", type=int, default=1000, help="Maximum characters per chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Overlapping characters between chunks.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    documents = read_jsonl(args.documents) if args.documents else load_documents(args.docs_dir)
    config = ChunkingConfig(strategy=args.strategy, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks = chunk_documents(documents, config)
    write_jsonl(chunks, args.out)
    LOGGER.info("Wrote %s chunks to %s", len(chunks), args.out)


if __name__ == "__main__":
    main()
