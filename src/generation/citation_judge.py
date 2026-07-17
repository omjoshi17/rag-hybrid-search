"""Citation extraction, support judging, and confidence scoring."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Protocol


CITATION_RE = re.compile(r"\[(?:Document\s*)?(\d+)\]", re.IGNORECASE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


class MessageLLM(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str:
        ...


@dataclass(frozen=True)
class CitationCheck:
    sentence: str
    document_index: int
    source: str
    supported: bool


def extract_citation_numbers(text: str) -> list[int]:
    return [int(match.group(1)) for match in CITATION_RE.finditer(text)]


def strip_citations(text: str) -> str:
    return CITATION_RE.sub("", text).strip()


def split_sentences(answer: str) -> list[str]:
    return [sentence.strip() for sentence in SENTENCE_RE.split(answer.strip()) if sentence.strip()]


def cited_sentences(answer: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for sentence in split_sentences(answer):
        citations = extract_citation_numbers(sentence)
        if citations:
            items.append({"sentence": sentence, "citations": citations})
    return items


class CitationJudge:
    """Use a secondary LLM call to verify each sentence/citation pair."""

    def __init__(self, llm: MessageLLM, model_name: str = "gpt-4o-mini") -> None:
        self.llm = llm
        self.model_name = model_name

    def verify_claim(self, sentence: str, cited_chunk: str) -> bool:
        messages = [
            {
                "role": "system",
                "content": "Return only True or False. True means the cited context fully supports the sentence.",
            },
            {
                "role": "user",
                "content": f"Sentence:\n{strip_citations(sentence)}\n\nCited context:\n{cited_chunk}",
            },
        ]
        verdict = self.llm.complete(messages).strip().lower()
        return verdict.startswith("true")

    def verify_answer(self, answer: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
        by_index = {int(document["document_index"]): document for document in documents}
        checks: list[CitationCheck] = []

        for item in cited_sentences(answer):
            sentence = item["sentence"]
            for citation in item["citations"]:
                document = by_index.get(citation)
                if not document:
                    checks.append(CitationCheck(sentence, citation, "", False))
                    continue
                supported = self.verify_claim(sentence, str(document.get("content") or ""))
                source = str((document.get("metadata") or {}).get("source") or "")
                checks.append(CitationCheck(sentence, citation, source, supported))

        total = len(checks)
        supported_count = sum(1 for check in checks if check.supported)
        return {
            "checks": [check.__dict__ for check in checks],
            "total_citations": total,
            "supported_citations": supported_count,
            "citation_accuracy": supported_count / total if total else 0.0,
        }


def normalize_relevance(score: float) -> float:
    """Map retrieval/reranker scores to 0..1 for confidence scoring."""
    if 0.0 <= score <= 1.0:
        return score
    return 1.0 / (1.0 + math.exp(-score))


def average_relevance(chunks: list[dict[str, Any]]) -> float:
    scores: list[float] = []
    for chunk in chunks:
        raw = chunk.get("cross_encoder_score", chunk.get("score"))
        if raw is not None:
            scores.append(normalize_relevance(float(raw)))
    return sum(scores) / len(scores) if scores else 0.0


def calculate_confidence(chunks: list[dict[str, Any]], verification: dict[str, Any]) -> dict[str, float]:
    relevance = average_relevance(chunks)
    citation_accuracy = float(verification.get("citation_accuracy", 0.0))
    composite = relevance + citation_accuracy
    return {
        "relevance_average": relevance,
        "citation_accuracy": citation_accuracy,
        "composite_score": composite,
    }
