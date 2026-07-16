"""Cross-encoder reranking for fused retrieval candidates."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from src.data.loader import read_jsonl
from src.retrieval.types import RetrievalResult


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class PredictsPairs(Protocol):
    def predict(self, pairs: list[tuple[str, str]]) -> Any:
        ...


@dataclass(frozen=True)
class RerankerConfig:
    model_name: str = DEFAULT_RERANKER_MODEL
    candidate_k: int = 20
    top_k: int = 5


class CrossEncoderReranker:
    """Rerank fused candidates by direct query-chunk relevance."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, model: PredictsPairs | None = None) -> None:
        if model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError("Install sentence-transformers to use cross-encoder reranking.") from exc
            model = CrossEncoder(model_name)

        self.model_name = model_name
        self.model = model

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        *,
        candidate_k: int = 20,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("query cannot be empty.")
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        candidate_pool = list(candidates)[:candidate_k]
        if not candidate_pool:
            return []

        pairs = [(query, str(candidate.get("content") or "")) for candidate in candidate_pool]
        scores = to_float_list(self.model.predict(pairs))
        scored = list(zip(candidate_pool, scores))
        scored.sort(key=lambda item: item[1], reverse=True)

        reranked: list[RetrievalResult] = []
        for rank_position, (candidate, rerank_score) in enumerate(scored[:top_k], start=1):
            result = dict(candidate)
            result["fusion_score"] = float(candidate.get("score", 0.0))
            result["score"] = rerank_score
            result["cross_encoder_score"] = rerank_score
            result["cross_encoder_model"] = self.model_name
            result["rank"] = rank_position
            result["retriever"] = "reranked"
            reranked.append(result)

        return reranked


def to_float_list(scores: Any) -> list[float]:
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    if isinstance(scores, (int, float)):
        return [float(scores)]
    return [float(score) for score in scores]


def rerank_results(
    query: str,
    candidates: Sequence[RetrievalResult],
    *,
    model_name: str = DEFAULT_RERANKER_MODEL,
    candidate_k: int = 20,
    top_k: int = 5,
) -> list[RetrievalResult]:
    reranker = CrossEncoderReranker(model_name=model_name)
    return reranker.rerank(query, candidates, candidate_k=candidate_k, top_k=top_k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerank fused candidates with a cross-encoder.")
    parser.add_argument("query", help="User question.")
    parser.add_argument("--fused", required=True, help="JSONL file of fused candidates.")
    parser.add_argument("--model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = read_jsonl(Path(args.fused))
    reranked = rerank_results(
        args.query,
        candidates,
        model_name=args.model,
        candidate_k=args.candidate_k,
        top_k=args.top_k,
    )
    print(json.dumps(reranked, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
