"""Load raw documents into normalized text records.

This module is intentionally runnable as a standalone script:

    python -m src.data.loader --docs-dir data/docs --out data/processed/documents.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".md",
    ".markdown",
    ".txt",
    ".html",
    ".htm",
    ".pdf",
}

FRONT_MATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)
MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", re.MULTILINE)

DocumentRecord = dict[str, Any]


def clean_text(text: str) -> str:
    """Normalize whitespace without destroying Markdown structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def source_name(file_path: Path, docs_dir: Path) -> str:
    try:
        return file_path.relative_to(docs_dir).as_posix()
    except ValueError:
        return file_path.name


def first_markdown_heading(text: str) -> str | None:
    match = MARKDOWN_HEADING_RE.search(text)
    return match.group(1).strip() if match else None


def base_metadata(file_path: Path, docs_dir: Path, content_type: str) -> dict[str, Any]:
    return {
        "source": source_name(file_path, docs_dir),
        "path": str(file_path),
        "page": None,
        "content_type": content_type,
    }


def load_markdown(file_path: Path, docs_dir: Path) -> list[DocumentRecord]:
    raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
    raw_text = FRONT_MATTER_RE.sub("", raw_text)
    content = clean_text(raw_text)
    metadata = base_metadata(file_path, docs_dir, "markdown")
    metadata["section_heading"] = first_markdown_heading(content)
    return [{"content": content, "metadata": metadata}]


def load_text(file_path: Path, docs_dir: Path) -> list[DocumentRecord]:
    content = clean_text(file_path.read_text(encoding="utf-8", errors="ignore"))
    return [{"content": content, "metadata": base_metadata(file_path, docs_dir, "text")}]


def load_html(file_path: Path, docs_dir: Path) -> list[DocumentRecord]:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError("Install beautifulsoup4 to load HTML files.") from exc

    html = file_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else None
    content = clean_text(soup.get_text("\n"))
    metadata = base_metadata(file_path, docs_dir, "html")
    metadata["title"] = title
    return [{"content": content, "metadata": metadata}]


def load_pdf_with_pdfplumber(file_path: Path, docs_dir: Path) -> list[DocumentRecord]:
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("pdfplumber is not installed.") from exc

    records: list[DocumentRecord] = []
    with pdfplumber.open(file_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            content = clean_text(page.extract_text() or "")
            metadata = base_metadata(file_path, docs_dir, "pdf")
            metadata["page"] = page_index
            records.append({"content": content, "metadata": metadata})
    return records


def load_pdf_with_pypdf2(file_path: Path, docs_dir: Path) -> list[DocumentRecord]:
    try:
        from PyPDF2 import PdfReader
    except ImportError as exc:
        raise RuntimeError("Install pdfplumber or PyPDF2 to load PDF files.") from exc

    records: list[DocumentRecord] = []
    reader = PdfReader(str(file_path))
    for page_index, page in enumerate(reader.pages, start=1):
        content = clean_text(page.extract_text() or "")
        metadata = base_metadata(file_path, docs_dir, "pdf")
        metadata["page"] = page_index
        records.append({"content": content, "metadata": metadata})
    return records


def load_pdf(file_path: Path, docs_dir: Path) -> list[DocumentRecord]:
    try:
        return load_pdf_with_pdfplumber(file_path, docs_dir)
    except RuntimeError as pdfplumber_error:
        LOGGER.warning("pdfplumber unavailable for %s: %s", file_path, pdfplumber_error)
        return load_pdf_with_pypdf2(file_path, docs_dir)


def load_file(file_path: Path, docs_dir: Path) -> list[DocumentRecord]:
    suffix = file_path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return load_markdown(file_path, docs_dir)
    if suffix == ".txt":
        return load_text(file_path, docs_dir)
    if suffix in {".html", ".htm"}:
        return load_html(file_path, docs_dir)
    if suffix == ".pdf":
        return load_pdf(file_path, docs_dir)
    raise ValueError(f"Unsupported file type: {file_path.suffix}")


def iter_document_files(docs_dir: Path) -> list[Path]:
    if not docs_dir.exists():
        raise FileNotFoundError(f"Document directory does not exist: {docs_dir}")

    files = [
        path
        for path in docs_dir.rglob("*")
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


def load_documents(docs_dir: str | Path = "data/docs", include_empty: bool = False) -> list[DocumentRecord]:
    docs_path = Path(docs_dir)
    documents: list[DocumentRecord] = []

    for file_path in iter_document_files(docs_path):
        LOGGER.info("Loading %s", file_path)
        for record in load_file(file_path, docs_path):
            content = record.get("content", "")
            if content or include_empty:
                documents.append(record)

    return documents


def write_jsonl(records: list[DocumentRecord], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(input_path: str | Path) -> list[DocumentRecord]:
    records: list[DocumentRecord] = []
    with Path(input_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number} in {input_path}") from exc
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load raw docs into normalized JSONL records.")
    parser.add_argument("--docs-dir", default="data/docs", help="Directory containing raw documents.")
    parser.add_argument("--out", default="data/processed/documents.jsonl", help="Output JSONL path.")
    parser.add_argument("--include-empty", action="store_true", help="Keep records with empty extracted text.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    documents = load_documents(args.docs_dir, include_empty=args.include_empty)
    write_jsonl(documents, args.out)
    LOGGER.info("Wrote %s document records to %s", len(documents), args.out)


if __name__ == "__main__":
    main()
