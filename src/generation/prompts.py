"""Grounded prompts and context formatting for Phase 3 generation."""

from __future__ import annotations

from typing import Any


SYSTEM_PROMPT = """You are a careful RAG assistant for internal documentation.

Use only the provided context. If the context is not enough, say so directly.
Every factual claim must include a citation immediately after the claim using
the exact format [Document X]. Do not cite sources that do not support the
claim. Do not use outside knowledge."""


def format_source(metadata: dict[str, Any]) -> str:
    source = metadata.get("source") or "unknown"
    page = metadata.get("page")
    section = metadata.get("section_heading")
    parts = [f"source={source}"]
    if page not in (None, ""):
        parts.append(f"page={page}")
    if section:
        parts.append(f"section={section}")
    return ", ".join(parts)


def format_context(chunks: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Format retrieved chunks as numbered context blocks."""
    documents: list[dict[str, Any]] = []
    blocks: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        metadata = dict(chunk.get("metadata") or {})
        content = str(chunk.get("content") or "").strip()
        document = {
            "document_index": index,
            "content": content,
            "metadata": metadata,
            "retrieval_score": float(chunk.get("score", 0.0)),
            "cross_encoder_score": chunk.get("cross_encoder_score"),
        }
        documents.append(document)
        blocks.append(f"[Document {index}] ({format_source(metadata)})\n{content}")

    return "\n\n".join(blocks), documents


def build_grounded_messages(question: str, chunks: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    context, documents = format_context(chunks)
    user_prompt = f"""Question:
{question}

Context:
{context}

Answer with citations after each factual claim."""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}], documents
