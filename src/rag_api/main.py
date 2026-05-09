"""
FastAPI Application — Multimodal RAG API
Endpoint:
  GET  /        -> Health check
  POST /ingest  -> Upload PDF and process to ChromaDB
  POST /query   -> Q&A based on documents
"""

import os
import logging
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.rag_api.ingestion import ingest_pdf
from src.rag_api.graph import get_rag_graph
from src.rag_api.config import CHROMA_DIR, CHROMA_COLLECTION_NAME

# ─────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Lifespan (startup/shutdown)
# ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Multimodal RAG API starting up...")
    # Pre-compile LangGraph graph during startup
    get_rag_graph()
    logger.info("LangGraph RAG graph ready.")
    yield
    logger.info("API shutting down.")


# ─────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────
app = FastAPI(
    title="Multimodal RAG API — Bank Mandiri 2025",
    description=(
        "REST API for Multimodal RAG pipeline.\n\n"
        "**Endpoint:**\n"
        "- `POST /ingest` - Upload PDF and process to ChromaDB\n"
        "- `POST /query` - Ask questions based on documents"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        example="Berapa nominal pertumbuhan kredit di sektor tambang?",
    )


class IngestStats(BaseModel):
    pages_processed: int
    text_chunks: int
    table_chunks: int
    image_summary_chunks: int
    total_chunks: int


class IngestResponse(BaseModel):
    filename: str
    status: str
    stats: IngestStats


class QueryResponse(BaseModel):
    question: str
    answer: str
    source_pages: list[int]
    chunk_types_used: list[str]


# ─────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────

@app.get("/", tags=["Health"], summary="Health Check")
def root():
    """Check API status."""
    import chromadb
    db_ready = CHROMA_DIR.exists()
    doc_count = 0
    if db_ready:
        try:
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            collections = [c.name for c in client.list_collections()]
            if CHROMA_COLLECTION_NAME in collections:
                col = client.get_collection(CHROMA_COLLECTION_NAME)
                doc_count = col.count()
        except Exception:
            pass

    return {
        "status": "ok",
        "service": "Multimodal RAG API — Bank Mandiri 2025",
        "vector_db_ready": db_ready,
        "documents_indexed": doc_count,
    }


@app.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["Ingestion"],
    summary="Upload & Process PDF",
)
async def ingest_endpoint(file: UploadFile = File(...)):
    """
    Upload Bank Mandiri PDF Report.

    Pipeline process:
    1. **Extract text** per page (pymupdf4llm)
    2. **Extract table** per page (pdfplumber -> Markdown)
    3. **Extract & summarize image** with Gemini Vision
    4. **Chunking** + **Embedding** -> save to ChromaDB
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Save temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        logger.info(f"Processing file: {file.filename} ({len(content) / 1024:.1f} KB)")
        result = ingest_pdf(tmp_path)
        return IngestResponse(filename=file.filename, **result)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["Query"],
    summary="Q&A Based on Documents",
)
async def query_endpoint(request: QueryRequest):
    """
    Ask a question. System will retrieve from ChromaDB
    and generate an answer using Gemini LLM.

    Response includes **source_pages** for retrieval debugging.
    """
    # Check if vector DB is populated
    if not CHROMA_DIR.exists():
        raise HTTPException(
            status_code=400,
            detail="Vector database is empty. Run POST /ingest first.",
        )

    try:
        graph = get_rag_graph()
        state = graph.invoke({"question": request.question})

        return QueryResponse(
            question=request.question,
            answer=state["answer"],
            source_pages=state.get("source_pages", []),
            chunk_types_used=state.get("chunk_types_used", []),
        )
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")
