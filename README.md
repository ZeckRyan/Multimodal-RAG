# Multimodal RAG - Bank Mandiri 2025 Report

> **AI Engineer Intern Technical Test**
>
> A multimodal RAG system capable of answering questions from PDF documents containing text, tables, and charts - as well as a text extraction pipeline from presentation slides to HTML.

---

## Project Structure

```text
├── data/
│   ├── Laporan Keuangan Bank Mandiri 2025.pdf
│   └── Sampel Slide Presentasi/        <- slide images (5 JPG files)
├── src/
│   ├── rag_api/
│   │   ├── config.py       <- centralized config (paths, model names)
│   │   ├── llm_utils.py    <- Gemini LLM + Embedding + Vision
│   │   ├── ingestion.py    <- PDF parsing + chunking + ChromaDB
│   │   ├── graph.py        <- LangGraph RAG workflow
│   │   └── main.py         <- FastAPI endpoints
│   └── cv_pipeline/
│       ├── text_extractor.ipynb   <- end-to-end CV notebook
│       ├── inpainting.py          <- OpenCV text removal
│       └── html_generator.py     <- HTML/CSS overlay generator
├── output/                 <- slide HTML results (auto-generated)
├── chroma_db/              <- ChromaDB vector store (auto-generated)
├── .env                    <- API keys (DO NOT commit!)
├── .env.example
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Component         | Library                                              |
| ----------------- | ---------------------------------------------------- |
| API Framework     | FastAPI + Uvicorn                                    |
| Orchestration     | **LangGraph** (agentic workflow)                     |
| LLM & Vision      | **Google Gemini 1.5 Flash**                          |
| Embedding         | `models/text-embedding-004` (Gemini)                 |
| PDF Parser        | `pymupdf4llm` + `pdfplumber` + `PyMuPDF`             |
| Vector DB         | **ChromaDB** (persistent, no Docker)                 |
| OCR               | EasyOCR                                              |
| Image Processing  | OpenCV (inpainting)                                  |
| HTML Generation   | Jinja2                                               |

---

## How to Run

### 1. Environment Setup

```bash
# Clone repo & enter folder
cd Multimodal-RAG

# Create & activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows

# Install all dependencies
pip install -r requirements.txt
```

### 2. API Key Configuration

Create a `.env` file from the template:

```bash
copy .env.example .env
```

Fill in `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Run API (Part A)

```bash
# From project root, with .venv active:
uvicorn src.rag_api.main:app --reload --host 0.0.0.0 --port 8000
```

Open Swagger UI: **<http://localhost:8000/docs>**

---

## API Endpoints

### `GET /`

Health check & vector database status.

```json
{
  "status": "ok",
  "vector_db_ready": true,
  "documents_indexed": 142
}
```

---

### `POST /ingest`

Upload PDF and process to ChromaDB.

**Request**: `multipart/form-data` with `file` field (PDF)

**Running Pipeline:**

1. Extract text per page (markdown-aware)
2. Extract tables per page -> Markdown
3. Crop & summarize images/charts with **Gemini Vision**
4. Chunking + Embedding + save to ChromaDB

**Response:**

```json
{
  "filename": "Laporan Keuangan Bank Mandiri 2025.pdf",
  "status": "success",
  "stats": {
    "pages_processed": 9,
    "text_chunks": 87,
    "table_chunks": 12,
    "image_summary_chunks": 15,
    "total_chunks": 114
  }
}
```

---

### `POST /query`

Ask questions based on the document.

**Request:**

```json
{
  "question": "Berapa persentase komposisi DPK Bank Mandiri tahun 2024?"
}
```

**Response:**

```json
{
  "question": "...",
  "answer": "Berdasarkan halaman 6, komposisi DPK Bank Mandiri...",
  "source_pages": [6],
  "chunk_types_used": ["image_summary"]
}
```

---

## LangGraph Workflow

```text
User Question
     │
  [retrieve]  ->  similarity_search to ChromaDB (Top-6 chunks)
     │
  [grade_relevance]  ->  LLM filter irrelevant chunks
     │
  [generate]  ->  Gemini synthesizes answer + page metadata
     │
  JSON Response
```

---

## Part B: CV Pipeline

### How to Run Notebook

```bash
# Ensure .venv is active, from project root:
jupyter notebook src/cv_pipeline/text_extractor.ipynb
```

Or run Jupyter Lab:

```bash
jupyter lab
```

### Output per Slide

| File              | Description                                             |
| ----------------- | ------------------------------------------------------- |
| `*_bbox_viz.jpg`  | OCR bounding box visualization                          |
| `*_clean_bg.jpg`  | Clean background (post-inpainting)                      |
| `*_result.html`   | **Final HTML** - open in browser, click to edit         |

### HTML Output Features & Bonus Points

- Text positioned with CSS `position: absolute` matching OCR coordinates.
- Text color extracted from original image pixels.
- Font size estimated from bounding box height.
- **Bonus 1 (Penanganan Teks Ganda)**: Clean background via **OpenCV TELEA inpainting** completely removes original text duplication.
- **Bonus 2 (Fungsionalitas Edit Teks)**: Added a **Floating WYSIWYG Toolbar** allowing users to click text and edit font family, font size, bold, italic, and underline natively in the browser (`contenteditable`).
- **Bonus 3 (Responsiveness)**: JavaScript auto-scaling ensures the HTML perfectly fits 90% of any viewport regardless of the original 4K/8K slide resolution.

---

## Evaluation Questions Example

| #   | Question                                                        | Source           |
| --- | --------------------------------------------------------------- | ---------------- |
| 1   | Apa peran Unit Pelindungan Nasabah menurut POJK No.22/2023?     | Page 7           |
| 2   | Apakah penagihan boleh dilakukan jam 21.00?                     | Page 7           |
| 3   | Berapa pertumbuhan kredit sektor tambang & konstruksi?          | Table Page 4     |
| 4   | Berapa komposisi DPK Bank Mandiri 2024 & 2025?                  | Chart Page 6     |
| 5   | Bagaimana alur penanganan pengaduan nasabah?                    | Info. Page 8     |
| 6   | Apa saja saluran pengaduan Bank Mandiri?                        | Image Page 9     |

---

## Architecture Decisions

### Why `pymupdf4llm` instead of LlamaParse?

LlamaParse is paid ($0.003/page). The `pymupdf4llm + pdfplumber + PyMuPDF` stack provides equivalent results for free: markdown-aware text, structured tables, and image extraction.

### Why LangGraph?

LangGraph enables an extensible agentic workflow: the `grade_relevance` node filters out irrelevant chunks before sending them to the LLM, increasing answer accuracy.

### Why ChromaDB?

Local setup without Docker, persistent storage, directly compatible with LangChain.

---

## Author

**Technical Test - AI Engineer Intern**

Stack: FastAPI · LangGraph · Gemini 1.5 Flash · ChromaDB · EasyOCR · OpenCV
