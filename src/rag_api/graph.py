"""
LangGraph RAG Workflow
State Machine:
  retrieve -> grade_relevance -> generate -> END

- retrieve        : Get Top-K chunks from ChromaDB
- grade_relevance : Filter irrelevant chunks (agentic bonus)
- generate        : Synthesize answer with Gemini LLM
"""

import logging
from typing import TypedDict, Annotated
import operator

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END
from src.rag_api.llm_utils import get_llm, get_grading_llm, get_embeddings


from src.rag_api.config import (
    CHROMA_DIR,
    CHROMA_COLLECTION_NAME,
    TOP_K_RETRIEVAL,
)
from src.rag_api.llm_utils import get_llm, get_embeddings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# State Schema
# ─────────────────────────────────────────────────────────

class RAGState(TypedDict):
    question: str
    retrieved_docs: list[Document]
    relevant_docs: list[Document]
    answer: str
    source_pages: list[int]
    chunk_types_used: list[str]


# ─────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────

def _get_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
        collection_name=CHROMA_COLLECTION_NAME,
    )


# ─────────────────────────────────────────────────────────
# Node 1: Retrieve
# ─────────────────────────────────────────────────────────

# Node 1: Retrieve - replace standard similarity_search with MMR + multi-query
def retrieve_node(state: RAGState) -> dict:
    question = state["question"]
    vectorstore = _get_vectorstore()
    k = max(2, TOP_K_RETRIEVAL // 3)

    all_docs = []
    for chunk_type in ["text", "table", "image_summary"]:
        try:
            results = vectorstore.similarity_search(
                question, k=k,
                filter={"chunk_type": chunk_type}
            )
            for doc in results:
                page = doc.metadata.get("page_number")
                score_preview = doc.page_content[:60].replace("\n", " ")
                logger.info(f"  [{chunk_type}] page {page}: {score_preview}...")
            all_docs.extend(results)
        except Exception as e:
            logger.warning(f"  [{chunk_type}] error: {e}")

    # Deduplication
    seen, unique_docs = set(), []
    for doc in all_docs:
        key = doc.page_content[:80]
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    pages = sorted(set(d.metadata.get("page_number") for d in unique_docs))
    logger.info(f"[retrieve] Total: {len(unique_docs)} chunks, pages: {pages}")
    return {"retrieved_docs": unique_docs}


# ─────────────────────────────────────────────────────────
# Node 2: Grade Relevance (Agentic Filter)
# ─────────────────────────────────────────────────────────

# graph.py - replace grade_relevance_node
def grade_relevance_node(state: RAGState) -> dict:
    question = state["question"]
    docs = state["retrieved_docs"]
    llm = get_grading_llm(temperature=0.0)

    # Create all snippets at once
    snippets = ""
    for i, doc in enumerate(docs):
        chunk_type = doc.metadata.get("chunk_type", "text")
        snippets += f"\n[{i}] ({chunk_type}): {doc.page_content[:400]}\n"

    # 1 LLM call for all chunks at once
    prompt = (
        f"Question: {question}\n\n"
        f"Here are {len(docs)} document snippets:\n{snippets}\n\n"
        "For each snippet, determine if it is relevant to the question.\n"
        f"Answer ONLY with a comma-separated list of relevant indices.\n"
        f"Example: 0,2,4\n"
        f"If none are relevant, answer: all"
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    answer = response.content.strip().lower()

    if "all" in answer or not answer:
        relevant = docs
    else:
        try:
            indices = [int(x.strip()) for x in answer.split(",") if x.strip().isdigit()]
            relevant = [docs[i] for i in indices if i < len(docs)]
        except Exception:
            relevant = docs

    if not relevant:
        relevant = docs

    logger.info(f"[grade] {len(relevant)}/{len(docs)} chunks passed the filter")
    return {"relevant_docs": relevant}


# ─────────────────────────────────────────────────────────
# Node 3: Generate
# ─────────────────────────────────────────────────────────

def generate_node(state: RAGState) -> dict:
    """Synthesize answer using Gemini LLM based on relevant chunks."""
    question = state["question"]
    docs = state.get("relevant_docs") or state.get("retrieved_docs", [])
    llm = get_llm(temperature=0.1)

    # Build context from chunks + metadata
    context_parts = []
    source_pages: set[int] = set()
    chunk_types: set[str] = set()

    for doc in docs:
        page_num = doc.metadata.get("page_number", "?")
        chunk_type = doc.metadata.get("chunk_type", "text")
        source_pages.add(page_num)
        chunk_types.add(chunk_type)
        context_parts.append(
            f"[Source: Page {page_num} | Type: {chunk_type}]\n{doc.page_content}"
        )

    context = "\n\n" + ("─" * 60) + "\n\n".join(context_parts)

    system_prompt = (
    "You are a professional and informative Bank Mandiri AI assistant. "
    "Answer user questions based on the provided document context. "
    "If the context contains relevant information, use it directly to answer - "
    "do not say 'not available' if the information is present. "
    "Do not hallucinate information that is completely absent from the context. "
    "Always include source page references. "
    "Answer in clear and structured Indonesian."
)

    user_prompt = (
        f"Document context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Provide a complete and accurate answer based on the context above. "
        "Include source page numbers in the answer."
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    logger.info(f"[generate] Answer generated ({len(response.content)} characters)")

    return {
        "answer": response.content,
        "source_pages": sorted(p for p in source_pages if isinstance(p, int)),
        "chunk_types_used": sorted(chunk_types),
    }


# ─────────────────────────────────────────────────────────
# Graph Builder
# ─────────────────────────────────────────────────────────

def build_rag_graph():
    """Compile LangGraph StateGraph: retrieve -> grade -> generate -> END."""
    workflow = StateGraph(RAGState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_relevance", grade_relevance_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_relevance")
    workflow.add_edge("grade_relevance", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


# Singleton - graph is compiled only once at startup
_rag_graph = None


def get_rag_graph():
    global _rag_graph
    if _rag_graph is None:
        logger.info("Compiling LangGraph RAG workflow...")
        _rag_graph = build_rag_graph()
    return _rag_graph
