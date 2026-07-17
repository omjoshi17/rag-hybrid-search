"""Request and response schemas for the RAG API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    dense_weight: float = Field(0.7, ge=0.0, le=1.0)
    sparse_weight: float = Field(0.3, ge=0.0, le=1.0)
    confidence_threshold: float = Field(1.2, ge=0.0)
    rerank: bool = True


class AskResponse(BaseModel):
    answer: str
    raw_answer: str = ""
    confidence: dict[str, float]
    citation_verification: dict[str, Any]
    retrieved_chunks: list[dict[str, Any]]
    used_fallback: bool


class IngestResponse(BaseModel):
    documents_loaded: int
    chunks_created: int
    indexing: dict[str, int] | None = None
    chunks_path: str


class DocumentSummary(BaseModel):
    source: str
    chunks: int


class DocumentsResponse(BaseModel):
    documents: list[DocumentSummary]
