"""
TenderIQ — Semantic Search Engine
=================================
Vector search with fuzzy document ID filtering.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logger_config import get_logger
from src.embeddings import EmbeddingEngine

logger = get_logger(__name__)


def _normalize_doc_id(doc_id: str) -> str:
    """Aggressive normalization so any variant matches:
    '13_Tender%20Document_cleaned' -> '13tenderdocument'
    '13_tender_document' -> '13tenderdocument'
    """
    if not doc_id:
        return ""
    s = str(doc_id).lower()
    s = s.replace("_cleaned", "").replace("_extracted", "").replace(".json", "").replace(".pdf", "")
    for char in [" ", "_", "-", "%20", "%"]:
        s = s.replace(char, "")
    return s


class SemanticSearchEngine:
    """Cosine similarity search over embedded tender chunks."""

    def __init__(self, data_dir: str = "data/extracted", model_name: Optional[str] = None):
        logger.info("Initializing Semantic Search Engine...")
        self.data_dir = Path(data_dir)
        self.embedding_engine = EmbeddingEngine(model_name=model_name)
        
        self.vectors, self.metadata = self.embedding_engine.load_embeddings(str(self.data_dir))
        logger.info(f"Search engine ready. {len(self.metadata)} chunks indexed.")

    def search(
        self,
        query: str,
        top_k: int = 8,
        min_score: float = 0.0,
        filter_document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search top-k most relevant chunks using cosine similarity."""
        if len(self.vectors) == 0 or len(self.metadata) == 0:
            logger.warning("No embeddings loaded in vector store.")
            return []

        # 1. Encode query
        query_vector = self.embedding_engine.encode_texts([query])[0]

        # 2. Compute Cosine Similarity
        # Normalize vectors for cosine sim
        norm_vectors = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-10)
        norm_query = query_vector / (np.linalg.norm(query_vector) + 1e-10)
        
        scores = np.dot(norm_vectors, norm_query)

        # 3. Apply Document Filter (FUZZY & FLEXIBLE)
        candidate_indices = list(range(len(self.metadata)))
        if filter_document_id:
            target_norm = _normalize_doc_id(filter_document_id)
            filtered_indices = [
                i for i, meta in enumerate(self.metadata)
                if target_norm in _normalize_doc_id(meta.get("document_id", ""))
                or _normalize_doc_id(meta.get("document_id", "")) in target_norm
            ]
            
            if filtered_indices:
                candidate_indices = filtered_indices
                logger.info(f"🎯 Filter applied: '{filter_document_id}'. Matched {len(candidate_indices)} chunks.")
            else:
                logger.warning(f"⚠️ Doc filter '{filter_document_id}' matched 0 chunks! Falling back to global search.")

        # 4. Sort candidates by score
        scored_candidates = [
            (idx, float(scores[idx]))
            for idx in candidate_indices
        ]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # 5. Format Top Results
        results = []
        for idx, score in scored_candidates[:top_k]:
            if score < min_score:
                continue
            
            meta = self.metadata[idx]
            results.append({
                "chunk_id": meta.get("chunk_id", f"chunk_{idx}"),
                "document_id": meta.get("document_id", "unknown"),
                "page_number": meta.get("page_number", 1),
                "text": meta.get("text", ""),
                "score": round(score, 4)
            })

        return results