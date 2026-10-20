"""
TenderIQ — Text Embeddings with Incremental Updates
=====================================================
Only encodes NEW chunks. Never re-encodes the full database.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from src.logger_config import get_logger

logger = get_logger(__name__)


class EmbeddingEngine:
    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        sample = self.model.encode(["test"])
        self.vector_dim = sample.shape[1]
        logger.info(f"  Model loaded. Vector dimension: {self.vector_dim}")

    def encode(self, texts: list) -> np.ndarray:
        if not texts:
            return np.array([])
        valid = [t if t.strip() else "empty" for t in texts]
        logger.info(f"Encoding {len(valid)} texts...")
        vectors = self.model.encode(
            valid,
            show_progress_bar=len(valid) > 20,
            normalize_embeddings=True
        )
        logger.info(f"  Encoded. Shape: {vectors.shape}")
        return vectors

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def save_embeddings(self, vectors, metadata, output_dir="data/extracted"):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(output_dir / "embeddings.npy"), vectors)
        with open(output_dir / "chunk_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {vectors.shape[0]} embeddings")

    def load_embeddings(self, input_dir="data/extracted"):
        input_dir = Path(input_dir)
        vectors_path = input_dir / "embeddings.npy"
        meta_path = input_dir / "chunk_metadata.json"

        if not vectors_path.exists():
            return np.zeros((0, 384)), []

        vectors = np.load(str(vectors_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        logger.info(f"Loaded {vectors.shape[0]} existing embeddings")
        return vectors, metadata


def build_embeddings_from_chunks(
    chunks_path="data/extracted/all_chunks.json",
    output_dir="data/extracted",
    model_name=None
):
    """Full rebuild (only use for initial setup or after wipe)."""
    logger.info("Building FULL embeddings from all chunks...")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        logger.error("No chunks found.")
        return np.array([]), []

    texts = [c["text"] for c in chunks]
    metadata = [
        {
            "chunk_id": c["chunk_id"],
            "document_id": c["document_id"],
            "page_number": c["page_number"],
            "text": c["text"],
            "word_count": c["word_count"]
        }
        for c in chunks
    ]

    engine = EmbeddingEngine(model_name)
    vectors = engine.encode(texts)
    engine.save_embeddings(vectors, metadata, output_dir)
    return vectors, metadata


def append_document_embeddings(
    new_chunks: list,
    output_dir: str = "data/extracted",
    model_name: str = None
):
    """
    INCREMENTAL UPDATE: Only encode the NEW document's chunks
    and append them to the existing embeddings file.

    This is 100x faster than rebuilding everything.
    """
    if not new_chunks:
        logger.info("No new chunks to encode.")
        return

    engine = EmbeddingEngine(model_name)

    # Load existing
    existing_vectors, existing_meta = engine.load_embeddings(output_dir)

    # Get document IDs already indexed
    existing_docs = set(m["document_id"] for m in existing_meta)
    new_doc_id = new_chunks[0]["document_id"]

    # Skip if already indexed
    if new_doc_id in existing_docs:
        logger.info(f"Document '{new_doc_id}' already indexed. Skipping.")
        return

    # Encode ONLY the new chunks
    logger.info(f"Incrementally encoding {len(new_chunks)} new chunks for '{new_doc_id}'...")
    new_texts = [c["text"] for c in new_chunks]
    new_vectors = engine.encode(new_texts)

    new_meta = [
        {
            "chunk_id": c["chunk_id"],
            "document_id": c["document_id"],
            "page_number": c["page_number"],
            "text": c["text"],
            "word_count": c["word_count"]
        }
        for c in new_chunks
    ]

    # Append to existing
    if existing_vectors.shape[0] > 0:
        combined_vectors = np.vstack([existing_vectors, new_vectors])
        combined_meta = existing_meta + new_meta
    else:
        combined_vectors = new_vectors
        combined_meta = new_meta

    # Save
    engine.save_embeddings(combined_vectors, combined_meta, output_dir)
    logger.info(
        f"✓ Appended {len(new_chunks)} chunks. "
        f"Total now: {combined_vectors.shape[0]}"
    )