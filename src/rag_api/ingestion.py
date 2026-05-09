"""
PDF Ingestion Pipeline - Multimodal RAG
Steps:
  1. Extract text per page    (pymupdf4llm -> Markdown-aware)
  2. Extract tables per page  (pdfplumber  -> Markdown string)
  3. Extract images per page  (PyMuPDF/fitz -> PNG, optional)
  4. Summarize images         (Gemini Vision, if ENABLE_VISION=True)
  5. Chunking                 (split text; keep tables & summaries as full chunks)
  6. Embed & store            (ChromaDB with rate-limit-safe batching)
"""

import time
import logging
from typing import Any

import chromadb
import fitz  # PyMuPDF
import pymupdf4llm
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.rag_api.config import (
    CHROMA_DIR,
    CHROMA_COLLECTION_NAME,
    IMAGES_TEMP_DIR,
    TEXT_CHUNK_SIZE,
    TEXT_CHUNK_OVERLAP,
    MIN_IMAGE_WIDTH,
    MIN_IMAGE_HEIGHT,
    ENABLE_VISION,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_BATCH_DELAY,
)
from src.rag_api.llm_utils import get_embeddings, summarize_image

logger = logging.getLogger(__name__)

SOURCE_LABEL = "Laporan Keuangan Bank Mandiri 2025"


# ─────────────────────────────────────────────────────────
# 1. Extract Text
# ─────────────────────────────────────────────────────────

def extract_text_elements(pdf_path: str) -> list[dict]:
    """
    Extract text per page using pymupdf4llm (Markdown-aware).
    Returns: list of {"page_number": int, "text": str, "type": "text"}
    """
    logger.info("Extracting text...")
    try:
        pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    except Exception as e:
        logger.error(f"pymupdf4llm error: {e}")
        return []

    elements = []
    for page_data in pages:
        text = page_data.get("text", "").strip()
        page_num = page_data.get("metadata", {}).get("page_number", 0)
        if text and page_num > 0:
            elements.append({"page_number": page_num, "text": text, "type": "text"})

    logger.info(f"  -> {len(elements)} text pages extracted")
    return elements


# ─────────────────────────────────────────────────────────
# 2. Extract Tables
# ─────────────────────────────────────────────────────────

def _rows_to_markdown(table: list[list]) -> str:
    """Convert a pdfplumber table (list of rows) to a Markdown table string."""
    if not table or not table[0]:
        return ""

    def clean(cell) -> str:
        return str(cell or "").replace("\n", " ").strip()

    header    = [clean(c) for c in table[0]]
    separator = [":---"] * len(header)
    rows      = [header, separator]

    for row in table[1:]:
        cells = [clean(c) for c in row]
        while len(cells) < len(header):  # pad short rows
            cells.append("")
        rows.append(cells)

    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def extract_table_elements(pdf_path: str) -> list[dict]:
    """
    Extract tables per page using pdfplumber.
    Returns: list of {"page_number": int, "text": str, "type": "table"}
    """
    logger.info("Extracting tables...")
    elements = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for table in page.extract_tables():
                if not table:
                    continue
                md = _rows_to_markdown(table)
                if md.strip():
                    elements.append({
                        "page_number": page_num,
                        "text": f"**Table on Page {page_num}:**\n\n{md}",
                        "type": "table",
                    })

    logger.info(f"  -> {len(elements)} tables extracted")
    return elements


# ─────────────────────────────────────────────────────────
# 3. Extract & Summarize Images
# ─────────────────────────────────────────────────────────

def extract_image_elements(pdf_path: str) -> list[dict]:
    """
    Render each PDF page as PNG and summarize with Gemini Vision.
    Designed for PDFs using vector graphics instead of embedded raster images.

    - Pages with fewer than 5 vector drawings are skipped (likely plain text).
    - Returns empty list immediately if ENABLE_VISION=False.

    Returns: list of {"page_number": int, "text": str, "type": "image_summary", "image_path": str}
    """
    if not ENABLE_VISION:
        logger.info("Vision is disabled (ENABLE_VISION=False). Skipping image extraction.")
        return []

    logger.info("Rendering PDF pages as images for Vision summarization...")
    elements = []
    doc = fitz.open(pdf_path)
    IMAGES_TEMP_DIR.mkdir(parents=True, exist_ok=True)

    for page_idx in range(len(doc)):
        page        = doc[page_idx]
        page_number = page_idx + 1
        drawings    = page.get_drawings()

        if len(drawings) < 5:
            logger.info(f"  Page {page_number}: skipped ({len(drawings)} drawings — likely plain text)")
            continue

        # Render to PNG at 2x zoom (~144 DPI)
        img_path = IMAGES_TEMP_DIR / f"page{page_number:02d}_render.png"
        page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0)).save(str(img_path))

        logger.info(f"  Summarizing page {page_number} ({len(drawings)} vector drawings)...")
        summary = summarize_image(str(img_path))

        if summary:
            elements.append({
                "page_number": page_number,
                "text":        f"[Visual Summary — Page {page_number} (Chart/Graph/Table)]:\n{summary}",
                "type":        "image_summary",
                "image_path":  str(img_path),
            })
        else:
            logger.warning(f"  Page {page_number}: Vision returned empty summary, skipping.")

    doc.close()
    logger.info(f"  -> {len(elements)} image summaries created")
    return elements


# ─────────────────────────────────────────────────────────
# 4. Build Documents (Chunking)
# ─────────────────────────────────────────────────────────

def build_documents(
    text_elements:  list[dict],
    table_elements: list[dict],
    image_elements: list[dict],
) -> list[Document]:
    """
    Convert raw elements into LangChain Documents.
    - Text    : split with RecursiveCharacterTextSplitter
    - Table   : one table  = one Document (never split)
    - Image   : one image  = one Document (never split)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_CHUNK_SIZE,
        chunk_overlap=TEXT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    documents: list[Document] = []

    for elem in text_elements:
        if not elem["text"].strip():
            continue
        for i, chunk in enumerate(splitter.split_text(elem["text"])):
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "page_number": elem["page_number"],
                    "chunk_type":  "text",
                    "chunk_index": i,
                    "source":      SOURCE_LABEL,
                },
            ))

    for elem in table_elements:
        if elem["text"].strip():
            documents.append(Document(
                page_content=elem["text"],
                metadata={
                    "page_number": elem["page_number"],
                    "chunk_type":  "table",
                    "source":      SOURCE_LABEL,
                },
            ))

    for elem in image_elements:
        if elem["text"].strip():
            documents.append(Document(
                page_content=elem["text"],
                metadata={
                    "page_number": elem["page_number"],
                    "chunk_type":  "image_summary",
                    "image_path":  elem.get("image_path", ""),
                    "source":      SOURCE_LABEL,
                },
            ))

    return documents


# ─────────────────────────────────────────────────────────
# 5. Embed & Store to ChromaDB
# ─────────────────────────────────────────────────────────

def store_to_chroma(documents: list[Document]) -> Chroma:
    """
    Embed documents and persist to ChromaDB.
    Uses small batches with delays to stay within API rate limits.
    Retries once automatically on 429 errors.
    """
    logger.info(f"Storing {len(documents)} documents to ChromaDB...")
    embeddings = get_embeddings()

    # Drop existing collection to prevent duplicates
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    existing = [c.name for c in client.list_collections()]
    if CHROMA_COLLECTION_NAME in existing:
        client.delete_collection(CHROMA_COLLECTION_NAME)
        logger.info("  Existing collection removed.")

    vectorstore   = Chroma(
        client=client,
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
    )
    total_batches = (len(documents) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

    for i in range(0, len(documents), EMBEDDING_BATCH_SIZE):
        batch     = documents[i : i + EMBEDDING_BATCH_SIZE]
        batch_num = i // EMBEDDING_BATCH_SIZE + 1
        logger.info(f"  Embedding batch {batch_num}/{total_batches} ({len(batch)} docs)...")

        try:
            vectorstore.add_documents(batch)
        except Exception as e:
            if "429" in str(e):
                logger.warning("  Rate limit hit — waiting 60 seconds before retry...")
                time.sleep(60)
                vectorstore.add_documents(batch)  # one retry
            else:
                raise

        # Delay between batches (skip after last batch)
        if i + EMBEDDING_BATCH_SIZE < len(documents):
            time.sleep(EMBEDDING_BATCH_DELAY)

    logger.info("  ChromaDB successfully updated.")
    return vectorstore


# ─────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────

def ingest_pdf(pdf_path: str) -> dict[str, Any]:
    """
    Run the full end-to-end ingestion pipeline for a single PDF.
    Called by the POST /ingest endpoint.
    """
    pdf_path = str(pdf_path)
    logger.info(f"=== Starting ingestion: {pdf_path} ===")

    text_elements  = extract_text_elements(pdf_path)
    table_elements = extract_table_elements(pdf_path)
    image_elements = extract_image_elements(pdf_path)
    documents      = build_documents(text_elements, table_elements, image_elements)

    n_text  = sum(1 for d in documents if d.metadata["chunk_type"] == "text")
    n_table = sum(1 for d in documents if d.metadata["chunk_type"] == "table")
    n_image = sum(1 for d in documents if d.metadata["chunk_type"] == "image_summary")
    logger.info(f"Total chunks: {len(documents)} [text={n_text}, table={n_table}, image={n_image}]")

    store_to_chroma(documents)

    return {
        "status": "success",
        "stats": {
            "pages_processed":      len(text_elements),
            "text_chunks":          n_text,
            "table_chunks":         n_table,
            "image_summary_chunks": n_image,
            "total_chunks":         len(documents),
        },
    }