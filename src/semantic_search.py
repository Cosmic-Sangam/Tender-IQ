"""
TenderIQ — Semantic Search Engine
====================================
Supports robust, case-insensitive document-level filtering.
"""

import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics.pairwise import cosine_similarity
from src.logger_config import get_logger
from src.embeddings import EmbeddingEngine

logger = get_logger(__name__)


class SemanticSearchEngine:
    """Semantic search over tender documents with document-scope filtering."""

    def __init__(self, embeddings_dir="data/extracted", model_name=None):
        logger.info("Initializing Semantic Search Engine...")
        self.engine = EmbeddingEngine(model_name)
        self.vectors, self.metadata = self.engine.load_embeddings(embeddings_dir)
        logger.info(f"Search engine ready. {len(self.metadata)} chunks indexed.")

    def search(self, query, top_k=5, min_score=0.0, filter_document_id=None):
        """
        Find most relevant chunks for a query.
        
        Args:
            query: The user's question
            top_k: Number of results to return
            min_score: Minimum similarity threshold
            filter_document_id: Optional - restrict search to this doc only
        """
        if not query or not query.strip():
            return []

        if len(self.vectors) == 0:
            logger.warning("No vectors indexed on disk.")
            return []

        # Encode query
        query_vector = self.engine.encode_single(query)

        # Compute similarity with ALL vectors
        similarities = cosine_similarity(
            query_vector.reshape(1, -1),
            self.vectors
        )[0]

        # Create (index, score, metadata) triples
        candidates = [
            (i, float(similarities[i]), self.metadata[i])
            for i in range(len(self.metadata))
        ]

        # Apply strict, case-insensitive document filter if specified
        if filter_document_id:
            target_filter = str(filter_document_id).strip().lower()
            
            # Filter the candidates list
            candidates = [
                c for c in candidates
                if str(c[2].get("document_id", "")).strip().lower() == target_filter
            ]
            
            logger.info(
                f"🎯 Filter applied: '{target_filter}'. "
                f"Matched chunks remaining: {len(candidates)}"
            )

        # Sort by score descending
        candidates.sort(key=lambda x: -x[1])

        # Take top-k above threshold
        results = []
        for rank, (idx, score, meta) in enumerate(candidates[:top_k], 1):
            if score < min_score:
                continue
            results.append({
                "rank": rank,
                "score": round(score, 4),
                "document_id": meta["document_id"],
                "page_number": meta["page_number"],
                "text": meta["text"],
                "word_count": meta["word_count"]
            })

        return results

    def search_by_document(self, query, document_id, top_k=3):
        """Search within a specific document."""
        return self.search(query, top_k=top_k, filter_document_id=document_id)