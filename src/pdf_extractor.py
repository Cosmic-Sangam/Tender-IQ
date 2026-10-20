"""
TenderIQ — PDF Text Extraction with Table Support
===================================================
Extracts text and tables from PDFs while preserving page numbers.
"""

import fitz
import json
import os
from pathlib import Path
from datetime import datetime
from src.logger_config import get_logger

logger = get_logger(__name__)


def extract_tables_from_page(page) -> str:
    """
    Extract tables from a PDF page and convert to readable text.

    WHY: Many tenders store critical data (EMD, evaluation criteria,
    BoQ, eligibility) inside tables. Plain text extraction misses
    the structure. This function reads table cells and formats them
    as labeled key-value lines.

    HOW: PyMuPDF's find_tables() detects table regions, reads cells
    row by row, and we join them into readable text.
    """
    table_text_parts = []

    try:
        # find_tables() is available in PyMuPDF >= 1.23.0
        tables = page.find_tables()

        for table in tables:
            rows = table.extract()
            if not rows or len(rows) < 2:
                continue

            # First row is usually the header
            headers = [str(cell).strip() if cell else "" for cell in rows[0]]

            for row in rows[1:]:
                cells = [str(cell).strip() if cell else "" for cell in row]
                # Pair each cell with its header
                for header, cell in zip(headers, cells):
                    if cell and header and cell != header:
                        table_text_parts.append(f"{header}: {cell}")
                    elif cell:
                        table_text_parts.append(cell)

    except AttributeError:
        # Older PyMuPDF version without find_tables()
        logger.debug("Table extraction not available in this PyMuPDF version")
    except Exception as e:
        logger.debug(f"Table extraction error: {e}")

    return "\n".join(table_text_parts)


def extract_pdf(pdf_path: str, document_id: str = None) -> dict:
    """
    Extract text and tables from a PDF, preserving page numbers.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF: {pdf_path}")

    if document_id is None:
        document_id = pdf_path.stem

    logger.info(f"Extracting: {pdf_path.name}")

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        raise RuntimeError(f"Cannot open PDF: {e}")

    pages_data = []
    all_text_parts = []

    for page_index in range(doc.page_count):
        page_number = page_index + 1
        page = doc[page_index]

        # Extract text blocks
        raw_blocks = page.get_text("blocks")
        text_blocks = []
        for b in raw_blocks:
            if b[6] == 0:
                text = b[4].strip()
                if len(text) > 2:
                    text_blocks.append({"y_position": b[1], "text": text})

        text_blocks.sort(key=lambda x: x["y_position"])
        page_text = "\n".join(b["text"] for b in text_blocks)

        # Extract tables and append to page text
        table_text = extract_tables_from_page(page)
        if table_text:
            page_text = page_text + "\n\n[TABLE DATA]\n" + table_text
            logger.debug(f"Page {page_number}: extracted table data")

        word_count = len(page_text.split()) if page_text.strip() else 0

        pages_data.append({
            "page_number": page_number,
            "text": page_text,
            "word_count": word_count
        })

        if page_text.strip():
            all_text_parts.append(page_text)

    doc.close()

    full_text = "\n\n".join(all_text_parts)
    total_words = sum(p["word_count"] for p in pages_data)

    result = {
        "document_id": document_id,
        "file_name": pdf_path.name,
        "extraction_time": datetime.now().isoformat(),
        "total_pages": len(pages_data),
        "pages": pages_data,
        "full_text": full_text,
        "total_words": total_words
    }

    logger.info(f"Extracted {result['total_pages']} pages, {total_words:,} words")
    return result


def save_extraction(result: dict, output_dir: str) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{result['document_id']}_extracted.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return str(filepath)


def extract_all_pdfs(input_dir: str, output_dir: str) -> list:
    input_dir = Path(input_dir)
    pdf_files = sorted(input_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDFs in {input_dir}")
        return []

    logger.info(f"Processing {len(pdf_files)} PDFs")
    results = []
    for pdf_file in pdf_files:
        try:
            result = extract_pdf(str(pdf_file))
            save_extraction(result, output_dir)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed: {pdf_file.name} — {e}")

    logger.info(f"Done: {len(results)}/{len(pdf_files)} PDFs")
    return results