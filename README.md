# RAG Hybrid Search

Production-style Retrieval-Augmented Generation over internal documents. Phase 1 is implemented: multi-format document loading, configurable chunking, OpenAI embeddings, ChromaDB persistence, and cosine-similarity deduplication before insert.

## Phase 1 Scope

- Load Markdown, text, HTML, and PDF files from `data/docs/`.
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

## Project Layout

```text
rag-hybrid-search/
├── data/
│   ├── docs/
│   ├── processed/
│   └── chroma/
├── eval/
├── scripts/
│   └── phase1_ingest.py
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── chunker.py
│   ├── indexing/
│   │   ├── embedder.py
│   │   └── vector_store.py
│   ├── retrieval/
│   └── generation/
├── tests/
├── requirements.txt
└── README.md
```

## Roadmap

- Phase 2: dense retrieval, BM25 sparse retrieval, Reciprocal Rank Fusion, and cross-encoder reranking.
- Phase 3: grounded answer generation, inline citations, citation verification, and confidence scoring.
- Phase 4: golden dataset and automated evaluation.
- Phase 5: FastAPI service, Streamlit dashboard, and Docker Compose.
- Phase 6: portfolio polish and demo walkthrough.
