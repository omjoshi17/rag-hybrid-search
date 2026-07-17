"""FastAPI service for the RAG hybrid search pipeline."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from src.api.schemas import AskRequest, AskResponse, DocumentSummary, DocumentsResponse, IngestResponse
from src.data.chunker import ChunkingConfig, chunk_documents
from src.data.loader import load_documents, read_jsonl, write_jsonl
from src.generation.llm import GenerationConfig, GroundedRAGPipeline
from src.retrieval.hybrid import HybridRetrievalConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

app = FastAPI(
    title="RAG Hybrid Search API",
    version="1.0.0",
    description="Hybrid RAG API with ingestion, retrieval, grounded generation, citations, and confidence scoring.",
)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def default_docs_dir() -> Path:
    return project_path(os.getenv("RAG_DOCS_DIR", "data/docs"))


def default_processed_dir() -> Path:
    return project_path(os.getenv("RAG_PROCESSED_DIR", "data/processed"))


def default_chunks_path(strategy: str = "recursive") -> Path:
    return default_processed_dir() / f"chunks_{strategy}.jsonl"


def safe_filename(filename: str) -> str:
    name = Path(filename).name
    return re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip() or "uploaded_doc"


def build_pipeline(request: AskRequest) -> GroundedRAGPipeline:
    retrieval_config = HybridRetrievalConfig(
        persist_dir=str(project_path(os.getenv("CHROMA_PERSIST_DIR", "data/chroma"))),
        collection_name=os.getenv("CHROMA_COLLECTION", "internal_docs"),
        chunks_path=str(project_path(os.getenv("RAG_CHUNKS_PATH", "data/processed/chunks_recursive.jsonl"))),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        dense_weight=request.dense_weight,
        sparse_weight=request.sparse_weight,
        rerank=request.rerank,
    )
    generation_config = GenerationConfig(
        answer_model=os.getenv("ANSWER_MODEL", "gpt-4o"),
        judge_model=os.getenv("JUDGE_MODEL", "gpt-4o-mini"),
        confidence_threshold=request.confidence_threshold,
    )
    return GroundedRAGPipeline.from_configs(retrieval_config, generation_config)


def run_ingestion(strategy: str, chunk_size: int, chunk_overlap: int, reset: bool) -> IngestResponse:
    docs_dir = default_docs_dir()
    processed_dir = default_processed_dir()
    processed_dir.mkdir(parents=True, exist_ok=True)

    documents = load_documents(docs_dir)
    documents_path = processed_dir / "documents.jsonl"
    write_jsonl(documents, documents_path)

    config = ChunkingConfig(strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunk_documents(documents, config)
    chunks_path = default_chunks_path(strategy)
    write_jsonl(chunks, chunks_path)

    from src.indexing.embedder import EmbeddingConfig, OpenAIEmbedder
    from src.indexing.vector_store import get_chroma_collection, index_chunks, reset_chroma_collection

    persist_dir = project_path(os.getenv("CHROMA_PERSIST_DIR", "data/chroma"))
    collection_name = os.getenv("CHROMA_COLLECTION", "internal_docs")
    collection = reset_chroma_collection(persist_dir, collection_name) if reset else get_chroma_collection(persist_dir, collection_name)
    embedder = OpenAIEmbedder(EmbeddingConfig(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")))
    stats = index_chunks(chunks, collection, embedder).as_dict()

    return IngestResponse(
        documents_loaded=len(documents),
        chunks_created=len(chunks),
        indexing=stats,
        chunks_path=str(chunks_path.relative_to(PROJECT_ROOT)),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    try:
        response = await run_in_threadpool(lambda: build_pipeline(request).ask(request.question))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AskResponse(
        answer=response["answer"],
        raw_answer=response.get("raw_answer", ""),
        confidence=response.get("confidence", {}),
        citation_verification=response.get("citation_verification", {}),
        retrieved_chunks=response.get("retrieval", {}).get("results", []),
        used_fallback=bool(response.get("used_fallback", False)),
    )


@app.post("/v1/ingest", response_model=IngestResponse)
async def ingest(
    files: Annotated[list[UploadFile] | None, File(description="Documents to add to data/docs/.")] = None,
    strategy: Annotated[str, Form()] = "recursive",
    chunk_size: Annotated[int, Form(ge=100)] = 1000,
    chunk_overlap: Annotated[int, Form(ge=0)] = 150,
    reset: Annotated[bool, Form()] = False,
) -> IngestResponse:
    docs_dir = default_docs_dir()
    docs_dir.mkdir(parents=True, exist_ok=True)

    for upload in files or []:
        target = docs_dir / safe_filename(upload.filename)
        target.write_bytes(await upload.read())

    try:
        return await run_in_threadpool(lambda: run_ingestion(strategy, chunk_size, chunk_overlap, reset))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/v1/documents", response_model=DocumentsResponse)
def documents() -> DocumentsResponse:
    chunks_path = project_path(os.getenv("RAG_CHUNKS_PATH", "data/processed/chunks_recursive.jsonl"))
    if not chunks_path.exists():
        return DocumentsResponse(documents=[])

    counts: dict[str, int] = {}
    for chunk in read_jsonl(chunks_path):
        source = str((chunk.get("metadata") or {}).get("source") or "unknown")
        counts[source] = counts.get(source, 0) + 1

    return DocumentsResponse(
        documents=[DocumentSummary(source=source, chunks=count) for source, count in sorted(counts.items())]
    )
