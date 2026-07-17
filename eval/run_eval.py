"""Run Phase 4 automated evaluation over the golden dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.llm import GenerationConfig, GroundedRAGPipeline
from src.retrieval.hybrid import HybridRetrievalConfig
from src.retrieval.reranker import DEFAULT_RERANKER_MODEL


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Golden dataset must be a JSON list.")
    return data


def source_names(response: dict[str, Any]) -> set[str]:
    sources: set[str] = set()
    for result in response.get("retrieval", {}).get("results", []):
        source = (result.get("metadata") or {}).get("source")
        if source:
            sources.add(str(source))
            sources.add(Path(str(source)).name)
    return sources


def required_docs_hit(required_docs: list[str], retrieved_sources: set[str]) -> bool:
    if not required_docs:
        return True
    return all(doc in retrieved_sources or Path(doc).name in retrieved_sources for doc in required_docs)


def evaluate_case(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    required_docs = list(case.get("required_docs") or [])
    sources = source_names(response)
    verification = response.get("citation_verification", {})
    confidence = response.get("confidence", {})
    return {
        "question": case["question"],
        "expected_answer": case.get("expected_answer", ""),
        "answer": response.get("answer", ""),
        "required_docs": required_docs,
        "retrieved_sources": sorted(sources),
        "retrieval_hit": required_docs_hit(required_docs, sources),
        "citation_accuracy": float(verification.get("citation_accuracy", 0.0)),
        "supported_citations": int(verification.get("supported_citations", 0)),
        "total_citations": int(verification.get("total_citations", 0)),
        "confidence": confidence,
        "used_fallback": bool(response.get("used_fallback", False)),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, float | int]:
    answerable = [record for record in records if record["required_docs"]]
    return {
        "total_cases": len(records),
        "answerable_cases": len(answerable),
        "overall_retrieval_hit_rate": mean([record["retrieval_hit"] for record in records]) if records else 0.0,
        "answerable_retrieval_hit_rate": mean([record["retrieval_hit"] for record in answerable]) if answerable else 0.0,
        "average_citation_accuracy": mean([record["citation_accuracy"] for record in records]) if records else 0.0,
        "fallback_rate": mean([record["used_fallback"] for record in records]) if records else 0.0,
    }


def run_eval(dataset: list[dict[str, Any]], pipeline: GroundedRAGPipeline, limit: int | None = None) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    cases = dataset[:limit] if limit else dataset
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['question']}")
        response = pipeline.ask(case["question"])
        records.append(evaluate_case(case, response))
    return {"summary": summarize(records), "records": records}


def parse_args() -> argparse.Namespace:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline against golden questions.")
    parser.add_argument("--dataset", default="eval/golden_dataset.json")
    parser.add_argument("--output", default="eval/results.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--persist-dir", default=os.getenv("CHROMA_PERSIST_DIR", "data/chroma"))
    parser.add_argument("--collection", default=os.getenv("CHROMA_COLLECTION", "internal_docs"))
    parser.add_argument("--chunks", default="data/processed/chunks_recursive.jsonl")
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--answer-model", default="gpt-4o")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--confidence-threshold", type=float, default=1.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(PROJECT_ROOT / args.dataset)
    pipeline = GroundedRAGPipeline.from_configs(
        HybridRetrievalConfig(
            persist_dir=str(PROJECT_ROOT / args.persist_dir),
            collection_name=args.collection,
            chunks_path=str(PROJECT_ROOT / args.chunks),
            embedding_model=args.embedding_model,
            reranker_model=args.reranker_model,
        ),
        GenerationConfig(
            answer_model=args.answer_model,
            judge_model=args.judge_model,
            confidence_threshold=args.confidence_threshold,
        ),
    )
    report = run_eval(dataset, pipeline, limit=args.limit)
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
