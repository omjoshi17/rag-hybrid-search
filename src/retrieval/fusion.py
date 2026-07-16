"""Reciprocal Rank Fusion for dense and sparse retrieval results."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.data.loader import read_jsonl
from src.retrieval.types import RetrievalResult, result_id


DEFAULT_RRF_K = 60


def rrf_contribution(rank_position: int, weight: float = 1.0, rrf_k: int = DEFAULT_RRF_K) -> float:
    if rank_position <= 0:
        raise ValueError("rank_position must be positive.")
    if rrf_k < 0:
        raise ValueError("rrf_k cannot be negative.")
    if weight < 0:
        raise ValueError("weight cannot be negative.")
    return weight / (rrf_k + rank_position)


def fuse_ranked_results(
    ranked_lists: Mapping[str, Sequence[RetrievalResult]],
    *,
    weights: Mapping[str, float] | None = None,
    rrf_k: int = DEFAULT_RRF_K,
    top_k: int | None = 20,
) -> list[RetrievalResult]:
    """Merge ranked result lists using weighted Reciprocal Rank Fusion."""
    weights = weights or {}
    merged: dict[str, RetrievalResult] = {}
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}

    for retriever_name, results in ranked_lists.items():
        weight = float(weights.get(retriever_name, 1.0))
        for rank_position, result in enumerate(results, start=1):
            key = result_id(result) or f"{retriever_name}:{rank_position}"
            if key not in merged:
                merged[key] = {
                    "id": key,
                    "content": result.get("content", ""),
                    "metadata": dict(result.get("metadata") or {}),
                    "retriever": "hybrid",
                    "source_retrievers": [],
                }
                scores[key] = 0.0
                ranks[key] = {}

            scores[key] += rrf_contribution(rank_position, weight=weight, rrf_k=rrf_k)
            ranks[key][retriever_name] = rank_position
            source_retrievers = merged[key]["source_retrievers"]
            if retriever_name not in source_retrievers:
                source_retrievers.append(retriever_name)

    fused = sorted(merged.values(), key=lambda result: scores[result["id"]], reverse=True)
    if top_k is not None:
        fused = fused[:top_k]

    for rank_position, result in enumerate(fused, start=1):
        key = result["id"]
        result["score"] = scores[key]
        result["rank"] = rank_position
        result["rrf"] = {
            "k": rrf_k,
            "source_ranks": ranks[key],
            "weights": {name: float(weights.get(name, 1.0)) for name in ranks[key]},
        }

    return fused


def fuse_dense_sparse(
    dense_results: Sequence[RetrievalResult],
    sparse_results: Sequence[RetrievalResult],
    *,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    rrf_k: int = DEFAULT_RRF_K,
    top_k: int | None = 20,
) -> list[RetrievalResult]:
    return fuse_ranked_results(
        {"dense": dense_results, "sparse": sparse_results},
        weights={"dense": dense_weight, "sparse": sparse_weight},
        rrf_k=rrf_k,
        top_k=top_k,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse dense and sparse JSONL result files with RRF.")
    parser.add_argument("--dense", required=True, help="JSONL file of dense results.")
    parser.add_argument("--sparse", required=True, help="JSONL file of sparse results.")
    parser.add_argument("--dense-weight", type=float, default=0.7)
    parser.add_argument("--sparse-weight", type=float, default=0.3)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dense_results = read_jsonl(Path(args.dense))
    sparse_results = read_jsonl(Path(args.sparse))
    fused = fuse_dense_sparse(
        dense_results,
        sparse_results,
        dense_weight=args.dense_weight,
        sparse_weight=args.sparse_weight,
        rrf_k=args.rrf_k,
        top_k=args.top_k,
    )
    print(json.dumps(fused, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
