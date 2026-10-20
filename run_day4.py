"""
TenderIQ — Day 4 Runner
=========================
Runs the hybrid extraction pipeline (NER + Rules) on all documents.

USAGE:
    python run_day4.py
"""

import sys, os, json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.logger_config import get_logger, log_section
from src.hybrid_pipeline import HybridPipeline, save_structured_output

logger = get_logger("run_day4")


def main():
    logger.info("TenderIQ — DAY 4: HYBRID PIPELINE")

    # Check prerequisites
    if not Path("models/tender_ner_v1").exists():
        logger.error("Model not found. Run Day 3 first.")
        return

    cleaned_files = list(Path("data/extracted").glob("*_cleaned.json"))
    if not cleaned_files:
        logger.error("No cleaned files. Run Day 1 first.")
        return

    # Initialize pipeline
    log_section(logger, "STEP 1: Initializing Hybrid Pipeline")
    pipeline = HybridPipeline("models/tender_ner_v1")

    # Process all documents
    log_section(logger, "STEP 2: Processing Documents")
    results = pipeline.process_all_documents("data/extracted")

    # Save output
    log_section(logger, "STEP 3: Saving Results")
    save_structured_output(results, "data/extracted/structured_output.json")

    # Display results
    log_section(logger, "DAY 4 RESULTS")

    for r in results:
        logger.info(f"\n  Document: {r['document_id']}")
        logger.info(f"  {'─' * 45}")

        # NER results
        logger.info("  NER Entities:")
        for label, items in r["structured_data"].items():
            if items:
                logger.info(f"    {label}: {items}")

        # Rule results
        logger.info("  Rule-Extracted Fields:")
        for field, value in r["rule_extracted"].items():
            logger.info(f"    {field}: {value}")

        logger.info(f"  Stats: {r['statistics']['total_ner_entities']} NER, "
                     f"{r['statistics']['total_rule_fields']} Rules")

    logger.info("\nDay 4 Complete!")
    logger.info("Next: Day 5 — Embeddings + Semantic Search")


if __name__ == "__main__":
    main()