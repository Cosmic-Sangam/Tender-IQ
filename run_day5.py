"""
TenderIQ — Day 5 Runner
=========================
Builds embeddings and tests semantic search.

USAGE:
    python run_day5.py

PREREQUISITES:
    Day 1 must be completed (data/extracted/all_chunks.json)
"""

import sys, os, json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.logger_config import get_logger, log_section
from src.embeddings import build_embeddings_from_chunks
from src.semantic_search import SemanticSearchEngine

logger = get_logger("run_day5")


def main():
    logger.info("TenderIQ — DAY 5: EMBEDDINGS + SEMANTIC SEARCH")

    # Check prerequisites
    chunks_path = "data/extracted/all_chunks.json"
    if not Path(chunks_path).exists():
        logger.error(f"Missing {chunks_path}. Run Day 1 first.")
        return

    # ═══════════════════════════════════════════
    # STEP 1: Build embeddings
    # ═══════════════════════════════════════════
    log_section(logger, "STEP 1: Building Embeddings")

    vectors, metadata = build_embeddings_from_chunks(
        chunks_path=chunks_path,
        output_dir="data/extracted"
    )

    if len(vectors) == 0:
        logger.error("No embeddings created. Cannot continue.")
        return

    # ═══════════════════════════════════════════
    # STEP 2: Test semantic search
    # ═══════════════════════════════════════════
    log_section(logger, "STEP 2: Testing Semantic Search")

    search_engine = SemanticSearchEngine("data/extracted")

    test_queries = [
        "What is the minimum turnover requirement?",
        "Which certifications are required?",
        "What is the bid submission deadline?",
        "What is the EMD amount?",
        "What technology stack should be used?",
        "What are the penalty clauses?",
        "How many years of experience is needed?",
        "What documents must be submitted?",
    ]

    for query in test_queries:
        logger.info(f"\n  Query: '{query}'")
        results = search_engine.search(query, top_k=3)

        if results:
            for r in results:
                logger.info(
                    f"    #{r['rank']} (score: {r['score']:.4f}) "
                    f"[{r['document_id']}, Page {r['page_number']}]"
                )
                logger.info(f"       {r['text'][:100]}...")
        else:
            logger.info("    No results found")

    # ═══════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════
    log_section(logger, "DAY 5 SUMMARY")
    logger.info(f"  Embeddings built  : {vectors.shape[0]} vectors")
    logger.info(f"  Vector dimensions : {vectors.shape[1]}")
    logger.info(f"  Documents indexed : {len(set(m['document_id'] for m in metadata))}")
    logger.info(f"  Files saved:")
    logger.info(f"    data/extracted/embeddings.npy")
    logger.info(f"    data/extracted/chunk_metadata.json")
    logger.info("")
    logger.info("Day 5 Complete!")
    logger.info("Next: Day 6 — Basic RAG (Retrieval Augmented Generation)")


if __name__ == "__main__":
    main()