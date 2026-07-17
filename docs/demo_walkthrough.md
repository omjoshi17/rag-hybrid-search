# Demo Walkthrough

Use this flow for a short portfolio demo.

1. Start with the sample docs in `data/docs/`.
2. Run ingestion and indexing:

```bash
python scripts/phase1_ingest.py --strategy recursive --reset
```

3. Start the API:

```bash
uvicorn src.api.main:app --reload
```

4. Start the dashboard:

```bash
streamlit run frontend/app.py
```

5. Ask: `When should critical incidents be escalated?`
6. Show the answer citations, confidence score, and retrieved chunks.
7. Move the dense/sparse weight slider and ask the same question again.
8. Run a quick eval:

```bash
python eval/run_eval.py --limit 5
```

9. Explain the production safeguards:

- Hybrid retrieval catches both semantic and exact keyword matches.
- RRF merges by rank position instead of incompatible raw scores.
- Cross-encoder reranking improves precision before generation.
- The LLM is forced to cite numbered context documents.
- A citation judge checks whether cited chunks support claims.
- Low confidence triggers a safe fallback instead of hallucination.
