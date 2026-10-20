"""
TenderIQ — Day 6 Runner
=========================
Tests the RAG pipeline with sample questions.

USAGE:
    python run_day6.py

PREREQUISITES:
    Day 5 must be completed (embeddings exist)
    .env file with GROQ_API_KEY (for LLM answers)
"""

import sys, os, json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.logger_config import get_logger, log_section
from src.rag import TenderRAG

logger = get_logger("run_day6")


def main():
    logger.info("TenderIQ — DAY 6: RAG (Retrieval Augmented Generation)")

    # Check prerequisites
    if not Path("data/extracted/embeddings.npy").exists():
        logger.error("Embeddings not found. Run Day 5 first.")
        return

    # Initialize RAG
    log_section(logger, "STEP 1: Initializing RAG Pipeline")
    rag = TenderRAG()

    # Test questions
    test_questions = [
        "What is the minimum turnover requirement?",
        "Which certifications does the bidder need?",
        "What is the bid submission deadline?",
        "What is the EMD amount?",
        "What technology stack is required?",
        "What are the penalty clauses for delays?",
        "How many years of experience is required?",
        "What documents must be submitted with the bid?",
        "What is the estimated project cost?",
        "Who is the issuing organization?",
    ]

    # ═══════════════════════════════════════════
    # STEP 2: Ask questions
    # ═══════════════════════════════════════════
    log_section(logger, "STEP 2: Asking Questions via RAG")

    has_llm = rag.llm_client is not None

    for i, question in enumerate(test_questions, 1):
        logger.info(f"\n{'─' * 55}")
        logger.info(f"Q{i}: {question}")
        logger.info(f"{'─' * 55}")

        if has_llm:
            result = rag.ask(question, top_k=5)
        else:
            logger.info("  (No LLM — showing retrieved chunks only)")
            result = rag.ask_without_llm(question, top_k=3)

        logger.info(f"\n  Answer:")
        # Wrap long answers
        answer_lines = result["answer"].split("\n")
        for line in answer_lines:
            logger.info(f"    {line}")

        logger.info(f"\n  Sources: {result['sources']}")

        if result["evidence"]:
            logger.info(f"  Top evidence (score: {result['evidence'][0]['score']:.4f}):")
            logger.info(f"    {result['evidence'][0]['text'][:120]}...")

    # ═══════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════
    log_section(logger, "DAY 6 SUMMARY")
    logger.info(f"  Questions tested : {len(test_questions)}")
    logger.info(f"  LLM available    : {'Yes (Groq)' if has_llm else 'No (retrieval only)'}")
    logger.info(f"  Evidence tracking: Yes (page numbers + source text)")
    logger.info("")
    logger.info("Day 6 Complete!")
    logger.info("Next: Day 7 — Streamlit Application")


if __name__ == "__main__":
    main()