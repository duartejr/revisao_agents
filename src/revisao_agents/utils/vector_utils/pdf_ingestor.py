# src/revisao_agents/utils/pdf_ingestor.py
"""
PDF ingestor that processes a folder of PDFs, extracts text with pdfplumber,
and indexes chunks in MongoDB via CorpusMongoDB.build().

The `url` field of each extracted document is filled with the absolute PDF
path so it remains compatible with the existing search and verification
pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pdfplumber

from ...config import EXTRACT_MIN_CHARS
from .mongodb_corpus import CorpusMongoDB

# ── Text Extraction ─────────────────────────────────────────────────────────


def _extract_pdf_text(pdf_path: Path) -> str:
    """
    Extracts all text from a PDF using pdfplumber.

    Pages without extractable text are silently ignored.
    Returns an empty string if the file cannot be read.

    Args:
        pdf_path: Path to the PDF file.
    Returns:
        Extracted text from all pages, concatenated with double newlines.
    """
    pages: list[str] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text.strip())
    except Exception as e:
        print(f"   ⚠️  Error reading {pdf_path.name}: {e}")
        return ""
    return "\n\n".join(pages)


# ── Main Function ──────────────────────────────────────────────────────────


def ingest_pdf_folder(folder_path: str) -> dict:
    """
    Processes all PDFs in a folder (recursively) and indexes them in MongoDB.

    The `url` field of each chunk is the absolute path of the PDF, maintaining
    compatibility with the existing pipeline (vector search, citations,
    references). Already indexed files are detected by url_exists() and skipped automatically.

    Args:
        folder_path: Path to the folder containing the PDFs.

    Returns:
        {
            "indexed"     : int,  # new PDFs indexed in this run
            "skipped"     : int,  # PDFs with insufficient text (< EXTRACT_MIN_CHARS)
            "already"     : int,  # PDFs already present in MongoDB
            "total_chunks": int,  # chunks inserted in this session
            "errors"      : int,  # PDFs that failed to read
        }
    """
    folder_path_obj = Path(folder_path).resolve()

    pdf_files = sorted(folder_path_obj.rglob("*.pdf"))
    if not pdf_files:
        print(f"   ℹ️  No PDFs found in: {folder_path_obj}")
        return {
            "indexed": 0,
            "skipped": 0,
            "already": 0,
            "total_chunks": 0,
            "errors": 0,
        }

    print(f"\n📂 Folder: {folder_path}")
    print(f"   {len(pdf_files)} PDF(s) found\n")

    corpus = CorpusMongoDB()
    extracted_documents: list[dict] = []

    counts = {"indexed": 0, "skipped": 0, "already": 0, "errors": 0}

    for pdf_path in pdf_files:
        abs_path = str(pdf_path)
        filename = pdf_path.stem  # filename without extension

        # Check if already indexed (url = absolute path of the PDF)
        if corpus.url_exists(abs_path):
            print(f"   ⏭️  Already indexed: {pdf_path.name}")
            counts["already"] += 1
            continue

        print(f"   📄 Extracting: {pdf_path.name}")
        text = _extract_pdf_text(pdf_path)

        if not text:
            print("      ❌ Empty text — invalid or protected file")
            counts["errors"] += 1
            continue

        if len(text) < EXTRACT_MIN_CHARS:
            print(f"      ⚠️  Text too short ({len(text)} chars < {EXTRACT_MIN_CHARS}) — ignored")
            counts["skipped"] += 1
            continue

        print(f"      ✅ {len(text):,} chars extracted")
        extracted_documents.append(
            {
                "url": abs_path,
                "content": text,
                "title": filename,
            }
        )
        counts["indexed"] += 1

    if not extracted_documents:
        total_chunks = 0
        print("\n   ℹ️  No new PDFs to index.")
    else:
        print(f"\n🔷 Indexing {len(extracted_documents)} PDF(s) in MongoDB…")
        before = corpus._total_chunks
        corpus.build(extracted_documents, snippets=[], prefix="pdf")
        total_chunks = corpus._total_chunks - before
        print(f"   ✅ {total_chunks} chunk(s) inserted into MongoDB.")

    counts["total_chunks"] = total_chunks
    return counts
