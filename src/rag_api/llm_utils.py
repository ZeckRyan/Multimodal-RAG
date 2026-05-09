"""
LLM & Embedding utilities using Google Gemini.
- get_llm()         : ChatGoogleGenerativeAI instance
- get_embeddings()  : GoogleGenerativeAIEmbeddings instance
- summarize_image() : Send image to Gemini Vision, get text description
"""

import logging
from pathlib import Path
import PIL.Image
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from src.rag_api.config import (
    GEMINI_API_KEY,
    GEMINI_LLM_MODEL,
    GEMINI_VISION_MODEL,
    GEMINI_GRADING_MODEL,
    GEMINI_EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)

# Configure Gemini globally
genai.configure(api_key=GEMINI_API_KEY)


def get_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """Return LangChain-wrapped Gemini LLM."""
    return ChatGoogleGenerativeAI(
        model=GEMINI_LLM_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=temperature,
        convert_system_message_to_human=True,  # Gemini compatibility
    )

def get_grading_llm(temperature=0.0):
    return ChatGoogleGenerativeAI(
        model=GEMINI_GRADING_MODEL,
        temperature=temperature,
        google_api_key=GEMINI_API_KEY,
    )

def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return Gemini embedding model."""
    return GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBEDDING_MODEL,
        google_api_key=GEMINI_API_KEY,
    )


def summarize_image(image_path: str) -> str:
    """
    Send image to Gemini Vision and get a detailed description.
    Used during ingestion to interpret charts/graphs/infographics.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        logger.warning(f"Image not found: {image_path}")
        return ""

    try:
        img = PIL.Image.open(str(image_path))
        model = genai.GenerativeModel(GEMINI_VISION_MODEL)

        prompt = (
            "You are an expert financial and data analyst. "
            "Describe completely and in detail ALL information contained in this image. "
            "Make sure to:\n"
            "1. State all numbers, percentages, and numerical data explicitly.\n"
            "2. If this is a chart/graph, mention all values on every category/bar/slice.\n"
            "3. If this is a table, transcribe all rows and columns.\n"
            "4. If this is an infographic or flowchart, explain each step/element sequentially.\n"
            "5. Mention the title, axis labels, legends, and footnotes if any.\n"
            "Use clear and structured English."
        )

        response = model.generate_content([prompt, img])
        logger.info(f"Image summarized: {image_path.name}")
        return response.text

    except Exception as e:
        logger.error(f"Failed to summarize image {image_path}: {e}")
        return f"[Failed to process image: {image_path.name}]"
