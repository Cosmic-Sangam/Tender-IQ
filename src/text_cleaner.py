"""
TenderIQ — Text Cleaning & Smart Chunking
==========================================
"""

import re
import json
from pathlib import Path
from src.logger_config import get_logger

logger = get_logger(__name__)


def clean_text(text: str) -> str:
    if not text or not text.strip():
        return ""

    # Fix hyphenated line breaks
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    # Remove standalone page numbers
    text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'[Pp]age\s+\d+\s+of\s+\d+', '', text)
    # Clean lines
    lines = [line.strip() for line in text.split('\n')]
    lines = [line for line in lines if line]
    return '\n'.join(lines)


def clean_document(extracted_json_path: str) -> dict:
    logger.info(f"Cleaning: {Path(extracted_json_path).name}")
    with open(extracted_json_path, "r", encoding="utf-8") as f:
        document = json.load(f)

    cleaned_pages = []
    all_cleaned = []

    for page in document["pages"]:
        cleaned = clean_text(page["text"])
        wc = len(cleaned.split()) if cleaned else 0
        cleaned_pages.append({
            "page_number": page["page_number"],
            "text": cleaned,
            "word_count": wc
        })
        if cleaned:
            all_cleaned.append(cleaned)

    result = {
        "document_id": document["document_id"],
        "file_name": document["file_name"],
        "total_pages": document["total_pages"],
        "pages": cleaned_pages,
        "full_text": "\n\n".join(all_cleaned),
        "total_words": sum(p["word_count"] for p in cleaned_pages)
    }
    logger.info(f"Cleaned: {result['total_words']:,} words")
    return result


def save_cleaned_document(cleaned_doc: dict, output_dir: str) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{cleaned_doc['document_id']}_cleaned.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(cleaned_doc, f, indent=2, ensure_ascii=False)
    return str(filepath)


def create_text_chunks(cleaned_doc: dict, chunk_size: int = 5) -> list:
    """
    Split document into overlapping chunks of 5 sentences.

    WHY 5? Bigger chunks keep related info together (e.g.,
    "EMD is Rs. 30 Lakh" stays with "EMD must be submitted by...").
    Overlap ensures no information falls between chunk boundaries.
    """
    chunks = []
    chunk_id = 0

    for page in cleaned_doc["pages"]:
        if not page["text"].strip():
            continue

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+|\n', page["text"])
        sentences = [s.strip() for s in sentences if len(s.strip()) > 8]

        # Create overlapping chunks (stride = chunk_size - 1)
        stride = max(1, chunk_size - 1)
        for i in range(0, len(sentences), stride):
            chunk_sentences = sentences[i:i + chunk_size]
            chunk_text = " ".join(chunk_sentences)

            if len(chunk_text.strip()) > 20:
                chunks.append({
                    "chunk_id": chunk_id,
                    "document_id": cleaned_doc["document_id"],
                    "page_number": page["page_number"],
                    "text": chunk_text,
                    "word_count": len(chunk_text.split())
                })
                chunk_id += 1

    logger.info(f"Created {len(chunks)} chunks from {cleaned_doc['document_id']}")
    return chunks