"""
Centralized configuration for the Multimodal RAG pipeline.
All constants and settings are loaded from environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Environment ──────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent.parent
load_dotenv(BASE_DIR / ".env")

# ── Paths ────────────────────────────────────────────────
DATA_DIR        = BASE_DIR / "data"
CHROMA_DIR      = BASE_DIR / "chroma_db"
OUTPUT_DIR      = BASE_DIR / "output"
IMAGES_TEMP_DIR = CHROMA_DIR / "extracted_images"

for _dir in [DATA_DIR, CHROMA_DIR, OUTPUT_DIR, IMAGES_TEMP_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Gemini API ───────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found! Please ensure the .env file is properly configured.")

# Model names
GEMINI_LLM_MODEL       = "gemini-2.5-flash"       # Main LLM for answer generation
GEMINI_VISION_MODEL    = "gemini-2.5-flash"       # Vision model for image summarization
GEMINI_GRADING_MODEL   = "gemini-2.5-flash-lite"  # Lightweight model for relevance grading
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"  # Embedding model for vector store

# ── Feature Flags ────────────────────────────────────────
ENABLE_VISION = False  # Set to True if PDF contains scanned images or non-text charts

# ── ChromaDB ─────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "bank_mandiri_2025"

# ── Chunking ─────────────────────────────────────────────
TEXT_CHUNK_SIZE    = 800
TEXT_CHUNK_OVERLAP = 150

# ── Retrieval ────────────────────────────────────────────
TOP_K_RETRIEVAL = 9  # Total chunks retrieved per query (3 per chunk type)

# ── Embedding Batching ───────────────────────────────────
EMBEDDING_BATCH_SIZE  = 5   # documents per batch
EMBEDDING_BATCH_DELAY = 4   # seconds between batches

# ── Image Filter ─────────────────────────────────────────
MIN_IMAGE_WIDTH  = 100  # Ignore decorative images smaller than this (px)
MIN_IMAGE_HEIGHT = 100