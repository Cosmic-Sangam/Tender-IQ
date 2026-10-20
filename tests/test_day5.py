"""
TenderIQ — Day 5 Tests
RUN: python tests/test_day5.py
"""

import sys, os, json, tempfile
import numpy as np
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def test_embedding_engine_loads():
    """Embedding model loads successfully."""
    from src.embeddings import EmbeddingEngine
    engine = EmbeddingEngine()
    assert engine.model is not None
    assert engine.vector_dim == 384
    print("  ✓ Embedding model loaded (384 dimensions)")


def test_encode_single_text():
    """Single text produces a 384-d vector."""
    from src.embeddings import EmbeddingEngine
    engine = EmbeddingEngine()
    vector = engine.encode_single("The bidder must have ISO 9001.")
    assert vector.shape == (384,)
    print("  ✓ Single text encoded correctly")


def test_encode_multiple_texts():
    """Multiple texts produce correct shape."""
    from src.embeddings import EmbeddingEngine
    engine = EmbeddingEngine()
    texts = ["Hello world", "Tender requirements", "ISO certification"]
    vectors = engine.encode(texts)
    assert vectors.shape == (3, 384)
    print("  ✓ Multiple texts encoded correctly")


def test_similar_texts_have_high_similarity():
    """Similar meanings should produce similar vectors."""
    from src.embeddings import EmbeddingEngine
    from sklearn.metrics.pairwise import cosine_similarity

    engine = EmbeddingEngine()

    text_a = "The bidder must have minimum turnover of Rs. 50 Crore"
    text_b = "Required annual revenue is at least 50 Crore rupees"
    text_c = "The weather is sunny today in Bangalore"

    vectors = engine.encode([text_a, text_b, text_c])

    # Similarity between A and B (both about turnover)
    sim_ab = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
    # Similarity between A and C (unrelated)
    sim_ac = cosine_similarity([vectors[0]], [vectors[2]])[0][0]

    assert sim_ab > sim_ac, (
        f"Similar texts should be closer: AB={sim_ab:.4f} vs AC={sim_ac:.4f}"
    )
    print(f"  ✓ Similarity works (AB={sim_ab:.4f} > AC={sim_ac:.4f})")


def test_save_load_embeddings():
    """Save and load round trip works."""
    from src.embeddings import EmbeddingEngine

    engine = EmbeddingEngine()
    vectors = engine.encode(["Test text one", "Test text two"])
    metadata = [
        {"chunk_id": 0, "document_id": "test", "page_number": 1, "text": "Test text one", "word_count": 3},
        {"chunk_id": 1, "document_id": "test", "page_number": 2, "text": "Test text two", "word_count": 3},
    ]

    with tempfile.TemporaryDirectory() as tmp:
        engine.save_embeddings(vectors, metadata, tmp)

        # Verify files exist
        assert Path(tmp, "embeddings.npy").exists()
        assert Path(tmp, "chunk_metadata.json").exists()

        # Load back
        loaded_vectors, loaded_meta = engine.load_embeddings(tmp)

        assert loaded_vectors.shape == vectors.shape
        assert len(loaded_meta) == 2
        assert loaded_meta[0]["page_number"] == 1

    print("  ✓ Save/load embeddings round trip works")


def test_empty_input():
    """Empty input returns empty array."""
    from src.embeddings import EmbeddingEngine
    engine = EmbeddingEngine()
    vectors = engine.encode([])
    assert len(vectors) == 0
    print("  ✓ Empty input handled")


def test_search_engine_loads():
    """Search engine initializes with saved embeddings."""
    if not Path("data/extracted/embeddings.npy").exists():
        print("  ⚠ Skipped: run_day5.py not run yet")
        return

    from src.semantic_search import SemanticSearchEngine
    search = SemanticSearchEngine("data/extracted")
    assert len(search.metadata) > 0
    assert search.vectors.shape[0] > 0
    print(f"  ✓ Search engine loaded ({len(search.metadata)} chunks)")


def test_search_returns_results():
    """Search returns relevant results."""
    if not Path("data/extracted/embeddings.npy").exists():
        print("  ⚠ Skipped: run_day5.py not run yet")
        return

    from src.semantic_search import SemanticSearchEngine
    search = SemanticSearchEngine("data/extracted")

    results = search.search("What certifications are required?", top_k=3)

    assert len(results) > 0
    assert results[0]["score"] > 0
    assert "page_number" in results[0]
    assert "text" in results[0]
    print(f"  ✓ Search returns results (top score: {results[0]['score']:.4f})")


def test_search_results_have_page_numbers():
    """Every search result includes page number for evidence."""
    if not Path("data/extracted/embeddings.npy").exists():
        print("  ⚠ Skipped: run_day5.py not run yet")
        return

    from src.semantic_search import SemanticSearchEngine
    search = SemanticSearchEngine("data/extracted")

    results = search.search("minimum turnover", top_k=5)

    for r in results:
        assert r["page_number"] > 0, "Page number must be positive"

    print("  ✓ All results have page numbers")


def test_search_empty_query():
    """Empty query returns empty results."""
    if not Path("data/extracted/embeddings.npy").exists():
        print("  ⚠ Skipped: run_day5.py not run yet")
        return

    from src.semantic_search import SemanticSearchEngine
    search = SemanticSearchEngine("data/extracted")

    assert search.search("") == []
    assert search.search("   ") == []
    print("  ✓ Empty query handled")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  TenderIQ — Day 5 Tests")
    print("=" * 60 + "\n")

    tests = [
        test_embedding_engine_loads,
        test_encode_single_text,
        test_encode_multiple_texts,
        test_similar_texts_have_high_similarity,
        test_save_load_embeddings,
        test_empty_input,
        test_search_engine_loads,
        test_search_returns_results,
        test_search_results_have_page_numbers,
        test_search_empty_query,
    ]

    passed = failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"  ✗ {test.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ✗ {test.__name__}: {type(e).__name__}: {e}")

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}\n")