"""
TenderIQ — Semantic Search Engine
=================================
Robust vector search + fuzzy document filter.
Compatible with multiple EmbeddingEngine method names.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logger_config import get_logger

logger = get_logger(__name__)


def _normalize_doc_id(doc_id: str) -> str:
    """
    Aggressive normalization so UI ids match stored ids.
    Examples that all become the same key:
      13_Tender%20Document
      13_Tender Document_cleaned
      13_tender_document
      13tenderdocument
    """
    if not doc_id:
        return ""
    s = str(doc_id).lower().strip()
    for token in ("_cleaned", "_extracted", ".json", ".pdf", ".npy"):
        s = s.replace(token, "")
    for ch in (" ", "_", "-", "%20", "%", ".", "/"):
        s = s.replace(ch, "")
    return s


def _cosine_scores(vectors: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
    """Cosine similarity of one query against a matrix of vectors."""
    if vectors is None or len(vectors) == 0:
        return np.array([], dtype=np.float32)

    v = np.asarray(vectors, dtype=np.float32)
    q = np.asarray(query_vec, dtype=np.float32).reshape(-1)

    # Handle shape mismatch gracefully
    if v.ndim == 1:
        v = v.reshape(1, -1)
    if q.shape[0] != v.shape[1]:
        raise ValueError(
            f"Query dim {q.shape[0]} != embedding dim {v.shape[1]}. "
            "Rebuild embeddings with run_day5.py."
        )

    v_norm = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-10)
    q_norm = q / (np.linalg.norm(q) + 1e-10)
    return np.dot(v_norm, q_norm).astype(np.float32)


class SemanticSearchEngine:
    """Cosine similarity search over embedded tender chunks."""

    def __init__(self, embeddings_dir: str = "data/extracted", model_name: Optional[str] = None):
        logger.info("Initializing Semantic Search Engine...")
        self.data_dir = Path(embeddings_dir)
        self.model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"

        self.embedding_engine = None
        self._st_model = None  # direct sentence-transformers fallback
        self.vectors = np.zeros((0, 384), dtype=np.float32)
        self.metadata: List[Dict[str, Any]] = []

        self._init_embedding_engine(model_name)
        self._load_index()

        logger.info(f"Search engine ready. {len(self.metadata)} chunks indexed.")

    # ─────────────────────────────────────────────────
    # Init / load
    # ─────────────────────────────────────────────────
    def _init_embedding_engine(self, model_name: Optional[str]):
        """Try project EmbeddingEngine; fall back to sentence-transformers."""
        try:
            from src.embeddings import EmbeddingEngine
            # Support both constructor styles
            try:
                self.embedding_engine = EmbeddingEngine(model_name=model_name) if model_name else EmbeddingEngine()
            except TypeError:
                self.embedding_engine = EmbeddingEngine()
            logger.info("Using project EmbeddingEngine")
        except Exception as e:
            logger.warning(f"EmbeddingEngine import failed ({e}); will use sentence-transformers directly.")
            self.embedding_engine = None

    def _load_index(self):
        """Load embeddings.npy + chunk_metadata.json (or rebuild metadata from all_chunks)."""
        npy_path = self.data_dir / "embeddings.npy"
        meta_path = self.data_dir / "chunk_metadata.json"
        chunks_path = self.data_dir / "all_chunks.json"

        vectors = None
        metadata: List[Dict[str, Any]] = []

        # Preferred: project loader
        if self.embedding_engine is not None:
            for method_name in ("load_embeddings", "load", "load_index"):
                method = getattr(self.embedding_engine, method_name, None)
                if callable(method):
                    try:
                        out = method(str(self.data_dir))
                        if isinstance(out, tuple) and len(out) >= 2:
                            vectors, metadata = out[0], out[1]
                        elif isinstance(out, dict):
                            vectors = out.get("vectors") or out.get("embeddings")
                            metadata = out.get("metadata") or out.get("meta") or []
                        break
                    except Exception as e:
                        logger.debug(f"{method_name} failed: {e}")

        # Direct file load
        if vectors is None and npy_path.exists():
            try:
                vectors = np.load(str(npy_path))
            except Exception as e:
                logger.error(f"Failed to load embeddings.npy: {e}")
                vectors = None

        if not metadata and meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load chunk_metadata.json: {e}")

        # If metadata missing text, merge from all_chunks.json
        if metadata and chunks_path.exists():
            try:
                with open(chunks_path, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                # Build lookup by chunk_id or by (doc, page, idx)
                by_id = {}
                for c in chunks:
                    cid = str(c.get("chunk_id", ""))
                    if cid:
                        by_id[cid] = c
                for m in metadata:
                    if not m.get("text"):
                        cid = str(m.get("chunk_id", ""))
                        if cid in by_id:
                            m["text"] = by_id[cid].get("text", "")
                        # also copy page/doc if missing
                        src = by_id.get(cid, {})
                        m.setdefault("document_id", src.get("document_id", m.get("document_id", "unknown")))
                        m.setdefault("page_number", src.get("page_number", m.get("page_number", 1)))
            except Exception as e:
                logger.debug(f"all_chunks merge skipped: {e}")

        # Last resort: metadata ONLY from all_chunks (no vectors yet)
        if (not metadata) and chunks_path.exists():
            try:
                with open(chunks_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception:
                metadata = []

        if vectors is None:
            vectors = np.zeros((0, 384), dtype=np.float32)
        else:
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)

        # Align lengths
        n = min(len(vectors), len(metadata)) if len(metadata) else len(vectors)
        if len(metadata) and len(vectors) and len(vectors) != len(metadata):
            logger.warning(
                f"embeddings ({len(vectors)}) != metadata ({len(metadata)}); truncating to {n}"
            )
            vectors = vectors[:n]
            metadata = metadata[:n]

        self.vectors = vectors
        self.metadata = metadata if metadata else [{} for _ in range(len(vectors))]

        logger.info(f"Loaded {len(self.metadata)} existing embeddings")

    # ─────────────────────────────────────────────────
    # Encoding (THE FIX for AttributeError)
    # ─────────────────────────────────────────────────
    def _get_st_model(self):
        if self._st_model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            name = self.model_name
            # allow short name
            if name and not name.startswith("sentence-transformers/") and "/" not in name:
                # already a hub id like all-MiniLM-L6-v2
                pass
            self._st_model = SentenceTransformer(
                name.replace("sentence-transformers/", "") if name.startswith("sentence-transformers/") else name
            )
            logger.info("  Model loaded (direct sentence-transformers).")
        return self._st_model

    def _encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string.
        Tries every common API on EmbeddingEngine, then ST fallback.
        """
        text = (query or "").strip() or " "
        eng = self.embedding_engine

        if eng is not None:
            # 1) encode_texts([q]) -> ndarray (n, d)
            for name in ("encode_texts", "encode_text", "embed_texts", "embed", "encode"):
                fn = getattr(eng, name, None)
                if not callable(fn):
                    continue
                try:
                    out = fn([text])
                    arr = np.asarray(out, dtype=np.float32)
                    if arr.ndim == 2:
                        return arr[0]
                    if arr.ndim == 1:
                        return arr
                except TypeError:
                    # maybe encode(text) not encode([text])
                    try:
                        out = fn(text)
                        arr = np.asarray(out, dtype=np.float32)
                        if arr.ndim == 2:
                            return arr[0]
                        return arr.reshape(-1)
                    except Exception:
                        continue
                except Exception as e:
                    logger.debug(f"eng.{name} failed: {e}")
                    continue

            # 2) engine.model.encode
            model = getattr(eng, "model", None) or getattr(eng, "encoder", None) or getattr(eng, "st_model", None)
            if model is not None and hasattr(model, "encode"):
                try:
                    out = model.encode([text], convert_to_numpy=True)
                    arr = np.asarray(out, dtype=np.float32)
                    return arr[0] if arr.ndim == 2 else arr.reshape(-1)
                except Exception as e:
                    logger.debug(f"engine.model.encode failed: {e}")

        # 3) Direct sentence-transformers (always works)
        model = self._get_st_model()
        out = model.encode([text], convert_to_numpy=True)
        arr = np.asarray(out, dtype=np.float32)
        logger.info(f"Encoding 1 texts... shape={arr.shape}")
        return arr[0] if arr.ndim == 2 else arr.reshape(-1)

    # ─────────────────────────────────────────────────
    # Search
    # ─────────────────────────────────────────────────
    def search(
        self,
        query: str,
        top_k: int = 8,
        min_score: float = 0.0,
        filter_document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return top-k chunks. Never raises on empty filter — falls back to global."""
        if self.vectors is None or len(self.vectors) == 0 or not self.metadata:
            logger.warning("No embeddings loaded in vector store.")
            return []

        try:
            query_vector = self._encode_query(query)
        except Exception as e:
            logger.error(f"Query encoding failed: {e}")
            return []

        try:
            scores = _cosine_scores(self.vectors, query_vector)
        except Exception as e:
            logger.error(f"Similarity failed: {e}")
            return []

        candidate_indices = list(range(len(self.metadata)))

        # Fuzzy document filter
        if filter_document_id:
            target = _normalize_doc_id(filter_document_id)
            filtered = []
            for i, meta in enumerate(self.metadata):
                did = _normalize_doc_id(str(meta.get("document_id", "")))
                if not did:
                    continue
                # substring both ways catches stem vs full name
                if target == did or target in did or did in target:
                    filtered.append(i)

            if filtered:
                candidate_indices = filtered
                logger.info(
                    f"🎯 Filter applied: '{filter_document_id}'. "
                    f"Matched chunks remaining: {len(candidate_indices)}"
                )
            else:
                # IMPORTANT: do not return empty — fall back so UI still answers
                logger.warning(
                    f"⚠️ Doc filter '{filter_document_id}' matched 0 chunks. "
                    f"Falling back to ALL documents ({len(candidate_indices)} chunks)."
                )

        ranked: List[Tuple[int, float]] = [
            (i, float(scores[i])) for i in candidate_indices
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)

        results: List[Dict[str, Any]] = []
        for idx, score in ranked[: max(1, int(top_k))]:
            if score < float(min_score):
                continue
            meta = self.metadata[idx] if idx < len(self.metadata) else {}
            text = meta.get("text") or meta.get("chunk_text") or meta.get("content") or ""
            results.append(
                {
                    "chunk_id": meta.get("chunk_id", f"chunk_{idx}"),
                    "document_id": meta.get("document_id", "unknown"),
                    "page_number": meta.get("page_number", meta.get("page", 1)),
                    "text": text,
                    "score": round(score, 4),
                }
            )

        return results

    def reload(self):
        """Reload vectors after incremental embed / delete."""
        self._load_index()
        logger.info(f"Search index reloaded. {len(self.metadata)} chunks.")