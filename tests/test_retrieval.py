import unittest

from src.retrieval.fusion import fuse_dense_sparse, rrf_contribution
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.sparse import tokenize


def result(chunk_id: str, rank: int, score: float = 1.0) -> dict:
    return {
        "id": chunk_id,
        "content": f"content for {chunk_id}",
        "metadata": {"chunk_id": chunk_id},
        "rank": rank,
        "score": score,
    }


class FakeCrossEncoder:
    def predict(self, pairs):
        return [3.0 if "best" in content else 1.0 for _, content in pairs]


class RetrievalTests(unittest.TestCase):
    def test_tokenize_keeps_technical_terms(self) -> None:
        self.assertEqual(tokenize("OPENAI_API_KEY failed with ERR-42"), ["openai_api_key", "failed", "with", "err-42"])

    def test_rrf_uses_rank_positions(self) -> None:
        self.assertAlmostEqual(rrf_contribution(1, weight=2.0, rrf_k=60), 2.0 / 61)

    def test_fusion_merges_duplicate_chunk_ids(self) -> None:
        dense = [result("dense-only", 1), result("shared", 2)]
        sparse = [result("shared", 1), result("sparse-only", 2)]

        fused = fuse_dense_sparse(dense, sparse, dense_weight=1.0, sparse_weight=1.0, top_k=3)

        self.assertEqual(fused[0]["id"], "shared")
        self.assertEqual(fused[0]["rrf"]["source_ranks"], {"dense": 2, "sparse": 1})

    def test_reranker_sorts_by_cross_encoder_score(self) -> None:
        candidates = [
            {"id": "a", "content": "okay chunk", "metadata": {}, "score": 0.4, "rank": 1},
            {"id": "b", "content": "best chunk", "metadata": {}, "score": 0.3, "rank": 2},
        ]
        reranker = CrossEncoderReranker(model=FakeCrossEncoder())

        reranked = reranker.rerank("question", candidates, top_k=2)

        self.assertEqual(reranked[0]["id"], "b")
        self.assertEqual(reranked[0]["rank"], 1)
        self.assertEqual(reranked[0]["fusion_score"], 0.3)


if __name__ == "__main__":
    unittest.main()
