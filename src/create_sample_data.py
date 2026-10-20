"""
TenderIQ — Create Sample PDFs from Text Files
==============================================

WHAT THIS DOES:
    Converts text files in data/sample_tenders/ into PDF files
    in data/raw_pdfs/. These PDFs are used to test the extraction
    pipeline with known content.

WHY:
    Real tender PDFs are hard to find consistently.
    Sample PDFs give us:
    - Controlled, predictable content
    - Same NLP pipeline works on both real and sample PDFs
    - Perfect for testing since we know what to expect

USAGE:
    python src/create_sample_data.py
"""

import sys
import os
from pathlib import Path

# Add project root to path so we can import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz  # PyMuPDF
from src.logger_config import get_logger

logger = get_logger(__name__)


# PDF layout constants
PAGE_WIDTH = 595       # A4 width in points
PAGE_HEIGHT = 842      # A4 height in points
MARGIN_LEFT = 60
MARGIN_TOP = 70
MARGIN_BOTTOM = 70
LINE_HEIGHT = 14
MAX_CHARS_PER_LINE = 85


def _wrap_line(line: str, max_chars: int) -> list[str]:
    """
    Wrap a long line into multiple shorter lines.

    We split at word boundaries so we don't break words in the middle.
    """
    if len(line) <= max_chars:
        return [line]

    words = line.split()
    wrapped = []
    current = ""

    for word in words:
        # Would adding this word exceed the limit?
        if len(current) + len(word) + 1 > max_chars:
            if current:
                wrapped.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()

    if current:
        wrapped.append(current)

    return wrapped


def text_to_pdf(text: str, output_path: str, title: str) -> int:
    """
    Convert text content to a multi-page PDF.

    Args:
        text: Text content to write
        output_path: Where to save the PDF
        title: PDF document title (metadata)

    Returns:
        Number of pages created
    """
    logger.debug(f"Creating PDF: {output_path}")

    doc = fitz.open()
    max_y = PAGE_HEIGHT - MARGIN_BOTTOM

    # Start first page
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    current_y = MARGIN_TOP

    lines = text.split('\n')

    for raw_line in lines:
        line = raw_line.rstrip()

        # Determine font size based on content
        fontsize = 10
        if line.startswith(('NOTICE', 'REQUEST', 'TENDER')):
            fontsize = 14
        elif line and len(line) > 2 and line[0].isdigit() and '.' in line[:3]:
            fontsize = 11

        # Wrap long lines
        wrapped_lines = _wrap_line(line, MAX_CHARS_PER_LINE) if line else [""]

        for wrapped in wrapped_lines:
            # Need new page?
            if current_y + LINE_HEIGHT > max_y:
                page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
                current_y = MARGIN_TOP

            # Write line (skip empty but keep spacing)
            if wrapped.strip():
                page.insert_text(
                    (MARGIN_LEFT, current_y),
                    wrapped,
                    fontsize=fontsize,
                    fontname="helv"
                )

            current_y += LINE_HEIGHT

    total_pages = doc.page_count

    # Set metadata
    doc.set_metadata({
        "title": title,
        "author": "TenderIQ Sample Data",
        "subject": "Sample Tender Document"
    })

    doc.save(output_path)
    doc.close()

    logger.debug(f"PDF saved: {output_path} ({total_pages} pages)")
    return total_pages


def create_all_sample_pdfs() -> int:
    """
    Convert all sample .txt files into PDF files.

    Returns:
        Number of PDFs created
    """
    input_dir = Path("data/sample_tenders")
    output_dir = Path("data/raw_pdfs")
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(input_dir.glob("*.txt"))

    if not txt_files:
        logger.warning(f"No .txt files found in {input_dir}")
        logger.warning("Make sure sample tender text files exist first.")
        return 0

    logger.info(f"Found {len(txt_files)} text files to convert")

    created_count = 0

    for txt_file in txt_files:
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                text = f.read()

            pdf_name = txt_file.stem + ".pdf"
            pdf_path = output_dir / pdf_name

            pages = text_to_pdf(text, str(pdf_path), title=txt_file.stem)
            logger.info(f"  ✓ Created {pdf_name} ({pages} pages)")
            created_count += 1

        except Exception as e:
            logger.error(f"  ✗ Failed to create PDF from {txt_file.name}: {e}")

    logger.info(f"Successfully created {created_count} sample PDFs")
    return created_count


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  CREATING SAMPLE PDFs")
    logger.info("=" * 60)
    count = create_all_sample_pdfs()
    logger.info(f"Done. {count} PDFs available in data/raw_pdfs/")