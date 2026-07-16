"""End-to-end Phase 2 hybrid retrieval orchestration."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from src.retrieval.dense import DenseRetrievalConfig, DenseRetriever
from src.retrieval.fusion import fuse_dense_sparse
from src.retrieval.reranker import DEFAULT_RERANKER_MODEL, CrossEncoderReranker
from src.retrieval.sparse import SparseRetriever
from src.retrieval.types import RetrievalResult


@dataclass(frozen=True)
class HybridRetrievalConfig:
    persist_dir: str = "data/chroma"
    collection_name: str = "internal_docs"
    chunks_path: str = "data/processed/chunks_recursive.jsonl"
    embedding_model: str = "text-embedding-3-small"
    reranker_model: str = DEFAULT_RERANKER_MODEL
    dense_top_k: int = 10
    sparse_top_k: int = 10
    fusion_top_k: int = 20
    final_top_k: int = 5
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    rrf_k: int = 60
    rerank: bool = True


class HybridRetrievalEngine:
    """Run dense retrieval, sparse retrieval, RRF fusion, and optional reranking."""

    def __init__(
        self,
        *,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        config: HybridRetrievalConfig,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.config = config
        self.reranker = reranker

    @classmethod
    def from_config(cls, config: HybridRetrievalConfig) -> "HybridRetrievalEngine":
        dense_retriever = DenseRetriever.from_config(
            DenseRetrievalConfig(
                persist_dir=config.persist_dir,
                collection_name=config.collection_name,
                embedding_model=config.embedding_model,
                top_k=config.dense_top_k,
            )
        )
        sparse_retriever = SparseRetriever.from_jsonl(config.chunks_path)
        reranker = CrossEncoderReranker(config.reranker_model) if config.rerank else None
        return cls(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            config=config,
            reranker=reranker,
        )

    def retrieve(self, query: str) -> dict[str, list[RetrievalResult]]:
        dense_results = self.dense_retriever.retrieve(query, top_k=self.config.dense_top_k)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=self.config.sparse_top_k)
        fused_results = fuse_dense_sparse(
            dense_results,
            sparse_results,
            dense_weight=self.config.dense_weight,
            sparse_weight=self.config.sparse_weight,
            rrf_k=self.config.rrf_k,
            top_k=self.config.fusion_top_k,
        )

        final_results = (
            self.reranker.rerank(
                query,
                fused_results,
                candidate_k=self.config.fusion_top_k,
                top_k=self.config.final_top_k,
            )
            if self.reranker
            else fused_results[: self.config.final_top_k]
        )

        return {
            "dense": dense_results,
            "sparse": sparse_results,
            "fused": fused_results,
            "results": final_results,
        }


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run Phase 2 hybrid retrieval.")
    parser.add_argument("query", help="User question.")
    parser.add_argument("--persist-dir", default=os.getenv("CHROMA_PERSIST_DIR", "data/chroma"))
    parser.add_argument("--collection", default=os.getenv("CHROMA_COLLECTION", "internal_docs"))
    parser.add_argument("--chunks", default="data/processed/chunks_recursive.jsonl")
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--dense-weight", type=float, default=0.7)
    parser.add_argument("--sparse-weight", type=float, default=0.3)
    parser.add_argument("--no-rerank", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = HybridRetrievalEngine.from_config(
        HybridRetrievalConfig(
            persist_dir=args.persist_dir,
            collection_name=args.collection,
            chunks_path=args.chunks,
            embedding_model=args.embedding_model,
            reranker_model=args.reranker_model,
            dense_weight=args.dense_weight,
            sparse_weight=args.sparse_weight,
            rerank=not args.no_rerank,
        )
    )
    print(json.dumps(engine.retrieve(args.query), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
