"""Shared result helpers for retrieval components."""

from __future__ import annotations

from typing import Any


RetrievalResult = dict[str, Any]


def result_id(result: RetrievalResult) -> str:
    metadata = result.get("metadata") or {}
    return str(result.get("id") or metadata.get("chunk_id") or "")


def make_result(
    *,
    result_id_value: str,
    content: str,
    metadata: dict[str, Any],
    score: float,
    rank: int,
    retriever: str,
) -> RetrievalResult:
    return {
        "id": result_id_value,
        "content": content,
        "metadata": metadata,
        "score": float(score),
        "rank": int(rank),
        "retriever": retriever,
    }
