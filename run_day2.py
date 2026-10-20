"""
TenderIQ — Day 2 Runner
=========================
Annotates entities in cleaned documents and builds spaCy training dataset.

USAGE:
    python run_day2.py

PREREQUISITES:
    Day 1 must be completed (cleaned JSON files in data/extracted/)
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.logger_config import get_logger, log_section
from src.ner_annotator import annotate_all_documents, save_annotations
from src.dataset_builder import build_complete_dataset

logger = get_logger("run_day2")


def main():
    logger.info("")
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║" + "  TenderIQ — DAY 2: NER DATASET  ".center(58) + "║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info("")

    # Check Day 1 output
    extracted_dir = Path("data/extracted")
    cleaned_files = list(extracted_dir.glob("*_cleaned.json"))

    if not cleaned_files:
        logger.error("No cleaned JSON files found in data/extracted/")
        logger.error("Please run Day 1 first: python run_day1.py")
        return

    logger.info(f"Found {len(cleaned_files)} cleaned documents from Day 1")

    # ═══════════════════════════════════════════
    # STEP 1: Annotate entities
    # ═══════════════════════════════════════════
    log_section(logger, "STEP 1: Annotating Entities")

    annotated_data = annotate_all_documents("data/extracted")

    if not annotated_data:
        logger.error("No annotations produced. Check your data.")
        return

    save_annotations(annotated_data, "data/ner_dataset/raw_annotations.json")

    # ═══════════════════════════════════════════
    # STEP 2: Build spaCy dataset
    # ═══════════════════════════════════════════
    log_section(logger, "STEP 2: Building spaCy Dataset")

    stats = build_complete_dataset(
        annotated_data,
        output_dir="data/ner_dataset"
    )

    # ═══════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════
    log_section(logger, "DAY 2 SUMMARY")

    total_sentences = sum(s["total_sentences"] for s in stats.values())
    total_entities = sum(s["total_entities"] for s in stats.values())

    logger.info(f"  Total annotated sentences : {total_sentences}")
    logger.info(f"  Total entities            : {total_entities}")
    logger.info("")

    for split_name, split_stats in stats.items():
        logger.info(f"  {split_name.upper()} split:")
        logger.info(
            f"    Sentences: {split_stats['total_sentences']} "
            f"(with entities: {split_stats['sentences_with_entities']})"
        )
        logger.info(f"    Entities:  {split_stats['total_entities']}")
        if split_stats["entity_labels"]:
            for label, count in sorted(
                split_stats["entity_labels"].items(), key=lambda x: -x[1]
            ):
                logger.info(f"      {label:<28} {count:>3}")
        logger.info("")

    logger.info("  Files created:")
    ner_dir = Path("data/ner_dataset")
    for f in sorted(ner_dir.glob("*")):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            logger.info(f"    {f.name:<35} ({size_kb:>6.1f} KB)")

    logger.info("")
    logger.info("✅ Day 2 Complete!")
    logger.info("   Next: python run_day3.py")
    logger.info("")


if __name__ == "__main__":
    main()