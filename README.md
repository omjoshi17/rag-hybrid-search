# RAG Hybrid Search

Production-style Retrieval-Augmented Generation over internal documents. Phases 1-4 are implemented: ingestion, chunking, OpenAI embeddings, ChromaDB persistence, hybrid retrieval, grounded generation, citation verification, confidence scoring, and automated evaluation.

## Current Status

| Phase | Status | What is implemented |
| --- | --- | --- |
| Phase 1: Ingestion and Chunking | Complete | Document loading, cleaning, chunking, OpenAI embeddings, ChromaDB indexing, and duplicate skipping with cosine similarity `> 0.95`. |
| Phase 2: Hybrid Retrieval | Complete | Dense Chroma retrieval, BM25 keyword retrieval, weighted RRF, and top-5 cross-encoder reranking. |
| Phase 3: Generation and Citations | Complete | GPT-4o grounded prompts, numbered context formatting, inline citations, citation judge, confidence scoring, and safe fallback. |
| Phase 4: Evaluation Framework | Complete | 50-case golden dataset and automated retrieval/citation evaluation runner. |
| Phase 5: API and Dashboard | Next | FastAPI endpoints, Streamlit UI, and Docker Compose. |

## Phase 1: Ingestion and Chunking

- Load Markdown, text, HTML, and PDF files from `data/docs/`.
- Use BeautifulSoup for HTML, `pdfplumber` with PyPDF2 fallback for PDFs, and Markdown/text parsing for plain files.
- Normalize documents into dictionaries shaped like:

```json
{"content": "...", "metadata": {"source": "file.pdf", "page": 1}}
```

- Chunk documents with switchable strategies:
  - `recursive`: LangChain `RecursiveCharacterTextSplitter`.
  - `semantic_markdown`: structure-aware splitting on Markdown headers.
  - `markdown_headers`: readable alias for `semantic_markdown`.
  - `fixed`: deterministic fixed-size windows with overlap.
- Add `chunk_id`, `chunk_index`, `strategy`, `section_heading`, and `char_count` to metadata.
- Embed chunks with `text-embedding-3-small`.
- Persist chunks to local ChromaDB.
- Skip near-duplicate chunks when cosine similarity against existing Chroma chunks is greater than `0.95`.

## Phase 2: Hybrid Retrieval Engine

- Dense retrieval embeds a user query and returns the top 10 ChromaDB chunks.
- Sparse retrieval builds `BM25Okapi` over the same chunk JSONL produced by Phase 1 and returns the top 10 keyword matches.
- Reciprocal Rank Fusion merges dense and sparse lists by rank position, not raw score addition.
- Cross-encoder reranking scores the top 20 fused candidates with `cross-encoder/ms-marco-MiniLM-L-6-v2` and returns the final top 5 chunks.

## Phase 3: Generation and Citation Layer

- Formats retrieved chunks as `[Document X]` context blocks.
- Uses GPT-4o to answer only from provided context.
- Requires citations immediately after factual claims.
- Verifies each cited sentence against the cited chunk with a secondary judge call.
- Scores confidence as normalized cross-encoder relevance plus verified citation rate.
- Returns a safe fallback when confidence is below threshold.

## Phase 4: Evaluation Framework

- Includes 50 golden Q&A objects in `eval/golden_dataset.json`.
- Runs the full pipeline for each question in `eval/run_eval.py`.
- Records required-document retrieval hits, citation accuracy, fallback rate, and per-case results.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY`.

## Run The Phase 1 Pipeline

Put source files in `data/docs/`, then run:

```bash
python scripts/phase1_ingest.py --strategy recursive
```

Useful variants:

```bash
python scripts/phase1_ingest.py --strategy semantic_markdown
python scripts/phase1_ingest.py --strategy fixed --chunk-size 900 --chunk-overlap 150
python scripts/phase1_ingest.py --skip-index
```

Standalone module commands are also available:

```bash
python -m src.data.loader --docs-dir data/docs --out data/processed/documents.jsonl
python -m src.data.chunker --documents data/processed/documents.jsonl --out data/processed/chunks_recursive.jsonl
python -m src.indexing.vector_store --chunks data/processed/chunks_recursive.jsonl
```

## Run The Phase 2 Retrieval Engine

After Phase 1 has indexed Chroma and written chunk JSONL, run dense, sparse, fusion, or the full hybrid flow:

```bash
python -m src.retrieval.dense "How fast do reset links expire?"
python -m src.retrieval.sparse "VPN access approval" --chunks data/processed/chunks_recursive.jsonl
python -m src.retrieval.hybrid "When should critical incidents be escalated?"
```

The hybrid engine performs:

- Dense Chroma retrieval: top 10
- Sparse BM25 retrieval: top 10
- Weighted Reciprocal Rank Fusion: top 20 candidates
- Cross-encoder reranking: final top 5

For quick local checks without downloading the reranker model:

```bash
python -m src.retrieval.hybrid "When should critical incidents be escalated?" --no-rerank
```

## Run The Phase 3 Generation Pipeline

After Phase 1 indexing is complete, ask a grounded question:

```bash
python -m src.generation.llm "When should critical incidents be escalated?"
```

The response includes the final answer, raw answer, confidence score, citation checks, retrieved chunks, and fallback status.

## Run Phase 4 Evaluation

Run the golden dataset:

```bash
python eval/run_eval.py --limit 5
python eval/run_eval.py --output eval/results.json
```

The full evaluation requires `OPENAI_API_KEY`, indexed Chroma data, and the reranker model available locally or downloadable by `sentence-transformers`.

## Project Layout

```text
rag-hybrid-search/
|-- data/
|   |-- docs/
|   |-- processed/
|   `-- chroma/
|-- eval/
|   |-- golden_dataset.json
|   `-- run_eval.py
|-- scripts/
|   `-- phase1_ingest.py
|-- src/
|   |-- data/
|   |   |-- loader.py
|   |   `-- chunker.py
|   |-- indexing/
|   |   |-- embedder.py
|   |   `-- vector_store.py
|   |-- retrieval/
|   |   |-- dense.py
|   |   |-- sparse.py
|   |   |-- fusion.py
|   |   |-- reranker.py
|   |   |-- hybrid.py
|   |   `-- types.py
|   `-- generation/
|       |-- prompts.py
|       |-- llm.py
|       `-- citation_judge.py
|-- tests/
|-- requirements.txt
`-- README.md
```

## Roadmap

- Phase 1: complete.
- Phase 2: complete.
- Phase 3: complete.
- Phase 4: complete.
- Phase 5: FastAPI service, Streamlit dashboard, and Docker Compose.
- Phase 6: portfolio polish and demo walkthrough.
