"""Streamlit dashboard for the RAG Hybrid Search API."""

from __future__ import annotations

import os

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def api_url(path: str) -> str:
    return f"{API_BASE_URL.rstrip('/')}{path}"


def post_json(path: str, payload: dict) -> dict:
    response = requests.post(api_url(path), json=payload, timeout=180)
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title="RAG Hybrid Search", layout="wide")
st.title("RAG Hybrid Search")

with st.sidebar:
    st.header("Retrieval tuning")
    dense_weight = st.slider("Dense weight", 0.0, 1.0, 0.7, 0.05)
    sparse_weight = round(1.0 - dense_weight, 2)
    st.metric("Sparse weight", sparse_weight)
    confidence_threshold = st.slider("Confidence threshold", 0.0, 2.0, 1.2, 0.05)
    rerank = st.toggle("Cross-encoder rerank", value=True)

    st.header("Ingest documents")
    uploads = st.file_uploader("Upload docs", accept_multiple_files=True, type=["md", "txt", "html", "htm", "pdf"])
    strategy = st.selectbox("Chunk strategy", ["recursive", "semantic_markdown", "fixed"])
    reset = st.checkbox("Reset vector index before ingest")
    if st.button("Ingest", type="secondary"):
        files = [("files", (file.name, file.getvalue())) for file in uploads]
        data = {"strategy": strategy, "reset": str(reset).lower()}
        try:
            response = requests.post(api_url("/v1/ingest"), files=files, data=data, timeout=300)
            response.raise_for_status()
            st.success(response.json())
        except Exception as exc:
            st.error(exc)

question = st.text_input("Ask a question", placeholder="When should critical incidents be escalated?")

if st.button("Ask", type="primary", disabled=not question.strip()):
    payload = {
        "question": question,
        "dense_weight": dense_weight,
        "sparse_weight": sparse_weight,
        "confidence_threshold": confidence_threshold,
        "rerank": rerank,
    }
    try:
        result = post_json("/v1/ask", payload)
        st.subheader("Answer")
        st.write(result["answer"])

        c1, c2, c3 = st.columns(3)
        confidence = result.get("confidence", {})
        c1.metric("Composite", round(confidence.get("composite_score", 0.0), 3))
        c2.metric("Relevance", round(confidence.get("relevance_average", 0.0), 3))
        c3.metric("Citation accuracy", round(confidence.get("citation_accuracy", 0.0), 3))

        with st.expander("Citation checks", expanded=False):
            st.json(result.get("citation_verification", {}))

        with st.expander("Retrieved chunks", expanded=True):
            for chunk in result.get("retrieved_chunks", []):
                metadata = chunk.get("metadata", {})
                st.markdown(f"**{metadata.get('source', 'unknown')}** | score `{chunk.get('score', 0):.4f}`")
                st.write(chunk.get("content", ""))
    except Exception as exc:
        st.error(exc)

with st.expander("Indexed documents"):
    try:
        docs = requests.get(api_url("/v1/documents"), timeout=30).json()
        st.json(docs)
    except Exception as exc:
        st.info(f"API not reachable yet: {exc}")
