"""Sparse keyword retrieval with BM25 over Phase 1 chunks."""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.loader import read_jsonl
from src.retrieval.types import RetrievalResult, make_result


LOGGER = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+")


@dataclass(frozen=True)
class SparseRetrievalConfig:
    chunks_path: str = "data/processed/chunks_recursive.jsonl"
    top_k: int = 10
    include_zero_scores: bool = False


def tokenize(text: str) -> list[str]:
    """Tokenize for BM25 while keeping technical identifiers intact."""
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


class SparseRetriever:
    """BM25 retriever built from the exact chunks used by dense retrieval."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError("Install rank-bm25 to use sparse retrieval.") from exc

        self.chunks = chunks
        self.tokenized_corpus = [tokenize(str(chunk.get("content") or "")) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    @classmethod
    def from_jsonl(cls, chunks_path: str | Path) -> "SparseRetriever":
        return cls(read_jsonl(chunks_path))

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        include_zero_scores: bool = False,
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("query cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if not self.chunks:
            LOGGER.warning("Sparse corpus is empty.")
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        ranked_indexes = sorted(range(len(scores)), key=lambda index: float(scores[index]), reverse=True)

        results: list[RetrievalResult] = []
        for index in ranked_indexes:
            score = float(scores[index])
            if score <= 0 and not include_zero_scores:
                continue
            chunk = self.chunks[index]
            metadata = dict(chunk.get("metadata") or {})
            results.append(
                make_result(
                    result_id_value=str(metadata.get("chunk_id") or index),
                    content=str(chunk.get("content") or ""),
                    metadata=metadata,
                    score=score,
                    rank=len(results) + 1,
                    retriever="sparse",
                )
            )
            if len(results) >= top_k:
                break

        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BM25 sparse retrieval over chunk JSONL.")
    parser.add_argument("query", help="User question.")
    parser.add_argument("--chunks", default="data/processed/chunks_recursive.jsonl")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--include-zero-scores", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    retriever = SparseRetriever.from_jsonl(args.chunks)
    results = retriever.retrieve(args.query, top_k=args.top_k, include_zero_scores=args.include_zero_scores)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
