"""
TenderIQ — Day 1 Runner
=========================
Extracts text from all PDFs in data/raw_pdfs/, cleans them,
and creates chunks for later NLP processing.

USAGE:
    python run_day1.py
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.logger_config import get_logger, log_section
from src.pdf_extractor import extract_all_pdfs
from src.text_cleaner import clean_document, save_cleaned_document, create_text_chunks

logger = get_logger("run_day1")


def main():
    logger.info("")
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║" + "  TenderIQ — DAY 1: EXTRACTION PIPELINE  ".center(58) + "║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info("")

    # ═══════════════════════════════════════════
    # CHECK: Do we have PDFs?
    # ═══════════════════════════════════════════
    pdf_dir = Path("data/raw_pdfs")
    pdf_files = sorted(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []

    if not pdf_files:
        logger.error("No PDFs found in data/raw_pdfs/")
        logger.error("Please place your tender PDFs there first.")
        logger.error("")
        logger.error("Steps:")
        logger.error("  1. Download tender PDFs from any government portal")
        logger.error("  2. Place them in: data/raw_pdfs/")
        logger.error("  3. Optionally run: python rename_pdfs.py")
        logger.error("  4. Run this script again: python run_day1.py")
        return

    logger.info(f"Found {len(pdf_files)} PDFs in data/raw_pdfs/")

    # ═══════════════════════════════════════════
    # STEP 1: Extract text from all PDFs
    # ═══════════════════════════════════════════
    log_section(logger, "STEP 1: Extracting Text from PDFs")
    extraction_results = extract_all_pdfs(
        input_dir="data/raw_pdfs",
        output_dir="data/extracted"
    )

    if not extraction_results:
        logger.error("Extraction failed. Check the errors above.")
        return

    # ═══════════════════════════════════════════
    # STEP 2: Clean extracted text
    # ═══════════════════════════════════════════
    log_section(logger, "STEP 2: Cleaning Extracted Text")

    extracted_dir = Path("data/extracted")
    json_files = sorted(extracted_dir.glob("*_extracted.json"))

    all_chunks = []
    cleaned_count = 0

    for json_file in json_files:
        try:
            cleaned = clean_document(str(json_file))
            save_cleaned_document(cleaned, "data/extracted")

            chunks = create_text_chunks(cleaned, chunk_size=5)
            all_chunks.extend(chunks)
            cleaned_count += 1
        except Exception as e:
            logger.error(f"Failed to clean {json_file.name}: {e}")

    # ═══════════════════════════════════════════
    # STEP 3: Save all chunks
    # ═══════════════════════════════════════════
    log_section(logger, "STEP 3: Saving Chunks")

    chunks_path = "data/extracted/all_chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(all_chunks)} chunks to {chunks_path}")

    # ═══════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════
    log_section(logger, "DAY 1 SUMMARY")
    logger.info(f"  PDFs extracted       : {len(extraction_results)}")
    logger.info(f"  Documents cleaned    : {cleaned_count}")
    logger.info(f"  Total chunks         : {len(all_chunks)}")
    logger.info(f"  Total words          : {sum(c['word_count'] for c in all_chunks):,}")

    # Show per-document stats
    logger.info("")
    logger.info("  Per-document breakdown:")
    for r in extraction_results:
        logger.info(
            f"    {r['document_id']:<40} "
            f"{r['total_pages']:>3} pages  "
            f"{r['total_words']:>7,} words"
        )

    logger.info("")
    logger.info("✅ Day 1 Complete!")
    logger.info("   Next: python run_day2.py")
    logger.info("")


if __name__ == "__main__":
    main()