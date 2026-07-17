"""Grounded answer generation pipeline for Phase 3."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Protocol

from dotenv import load_dotenv

from src.generation.citation_judge import CitationJudge, calculate_confidence
from src.generation.prompts import build_grounded_messages
from src.retrieval.hybrid import HybridRetrievalConfig, HybridRetrievalEngine
from src.retrieval.reranker import DEFAULT_RERANKER_MODEL


class MessageCompleter(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str:
        ...


@dataclass(frozen=True)
class GenerationConfig:
    answer_model: str = "gpt-4o"
    judge_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 900
    confidence_threshold: float = 1.2


class OpenAIChatLLM:
    """Small OpenAI chat wrapper used by generation and citation judging."""

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 900) -> None:
        load_dotenv()
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install openai to use the generation pipeline.") from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for LLM calls.")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""


class GroundedRAGPipeline:
    """Retrieve context, generate a cited answer, verify citations, score confidence."""

    def __init__(
        self,
        retrieval_engine: HybridRetrievalEngine,
        answer_llm: MessageCompleter,
        citation_judge: CitationJudge,
        config: GenerationConfig,
    ) -> None:
        self.retrieval_engine = retrieval_engine
        self.answer_llm = answer_llm
        self.citation_judge = citation_judge
        self.config = config

    @classmethod
    def from_configs(
        cls,
        retrieval_config: HybridRetrievalConfig,
        generation_config: GenerationConfig,
    ) -> "GroundedRAGPipeline":
        answer_llm = OpenAIChatLLM(
            generation_config.answer_model,
            temperature=generation_config.temperature,
            max_tokens=generation_config.max_tokens,
        )
        judge_llm = OpenAIChatLLM(generation_config.judge_model, temperature=0.0, max_tokens=20)
        return cls(
            retrieval_engine=HybridRetrievalEngine.from_config(retrieval_config),
            answer_llm=answer_llm,
            citation_judge=CitationJudge(judge_llm, model_name=generation_config.judge_model),
            config=generation_config,
        )

    def ask(self, question: str) -> dict:
        retrieval = self.retrieval_engine.retrieve(question)
        chunks = retrieval["results"]
        if not chunks:
            return {
                "answer": safe_fallback_response(chunks),
                "raw_answer": "",
                "confidence": {"relevance_average": 0.0, "citation_accuracy": 0.0, "composite_score": 0.0},
                "citation_verification": {"checks": [], "total_citations": 0, "supported_citations": 0},
                "retrieval": retrieval,
                "used_fallback": True,
            }

        messages, documents = build_grounded_messages(question, chunks)
        raw_answer = self.answer_llm.complete(messages).strip()
        verification = self.citation_judge.verify_answer(raw_answer, documents)
        confidence = calculate_confidence(chunks, verification)
        used_fallback = confidence["composite_score"] < self.config.confidence_threshold

        return {
            "answer": safe_fallback_response(chunks) if used_fallback else raw_answer,
            "raw_answer": raw_answer,
            "confidence": confidence,
            "citation_verification": verification,
            "retrieval": retrieval,
            "context_documents": documents,
            "used_fallback": used_fallback,
        }


def safe_fallback_response(chunks: list[dict]) -> str:
    sources = sorted({str((chunk.get("metadata") or {}).get("source")) for chunk in chunks if chunk.get("metadata")})
    if not sources:
        return "I do not have enough verified context to answer this question safely."
    return "I do not have enough verified context to answer this question safely. Relevant sources to check: " + ", ".join(sources)


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run Phase 3 grounded RAG generation.")
    parser.add_argument("question")
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
    pipeline = GroundedRAGPipeline.from_configs(
        HybridRetrievalConfig(
            persist_dir=args.persist_dir,
            collection_name=args.collection,
            chunks_path=args.chunks,
            embedding_model=args.embedding_model,
            reranker_model=args.reranker_model,
        ),
        GenerationConfig(
            answer_model=args.answer_model,
            judge_model=args.judge_model,
            confidence_threshold=args.confidence_threshold,
        ),
    )
    print(json.dumps(pipeline.ask(args.question), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
