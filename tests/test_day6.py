"""
TenderIQ — Day 6 Tests
RUN: python tests/test_day6.py
"""

import sys, os
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def test_rag_initializes():
    """RAG pipeline initializes without errors."""
    if not Path("data/extracted/embeddings.npy").exists():
        print("  ⚠ Skipped: embeddings not found")
        return

    from src.rag import TenderRAG
    rag = TenderRAG()
    assert rag.search_engine is not None
    print("  ✓ RAG pipeline initialized")


def test_rag_retrieval_only():
    """ask_without_llm returns relevant chunks."""
    if not Path("data/extracted/embeddings.npy").exists():
        print("  ⚠ Skipped: embeddings not found")
        return

    from src.rag import TenderRAG
    rag = TenderRAG()

    result = rag.ask_without_llm("What certifications are required?", top_k=3)

    assert result["found"] is True
    assert len(result["evidence"]) > 0
    assert len(result["answer"]) > 0
    assert "page_number" in result["evidence"][0]
    print(f"  ✓ Retrieval-only mode works ({len(result['evidence'])} chunks)")


def test_rag_evidence_has_page_numbers():
    """Every evidence item has a page number."""
    if not Path("data/extracted/embeddings.npy").exists():
        print("  ⚠ Skipped: embeddings not found")
        return

    from src.rag import TenderRAG
    rag = TenderRAG()

    result = rag.ask_without_llm("minimum turnover", top_k=3)

    for ev in result["evidence"]:
        assert ev["page_number"] > 0
        assert ev["document_id"] != ""
        assert ev["score"] > 0

    print("  ✓ Evidence has page numbers and sources")


def test_rag_empty_question():
    """Empty question returns appropriate response."""
    if not Path("data/extracted/embeddings.npy").exists():
        print("  ⚠ Skipped: embeddings not found")
        return

    from src.rag import TenderRAG
    rag = TenderRAG()

    result = rag.ask("")
    assert result["found"] is False
    print("  ✓ Empty question handled")


def test_rag_sources_list():
    """Sources list is populated correctly."""
    if not Path("data/extracted/embeddings.npy").exists():
        print("  ⚠ Skipped: embeddings not found")
        return

    from src.rag import TenderRAG
    rag = TenderRAG()

    result = rag.ask_without_llm("deadline", top_k=3)

    assert len(result["sources"]) > 0
    assert "Page" in result["sources"][0]
    print(f"  ✓ Sources list: {result['sources']}")


def test_context_building():
    """Context is built correctly from search results."""
    if not Path("data/extracted/embeddings.npy").exists():
        print("  ⚠ Skipped: embeddings not found")
        return

    from src.rag import TenderRAG
    rag = TenderRAG()

    mock_results = [
        {"document_id": "test", "page_number": 5, "text": "ISO 9001 required.", "score": 0.8}
    ]

    context = rag._build_context(mock_results)
    assert "test" in context
    assert "Page 5" in context
    assert "ISO 9001" in context
    print("  ✓ Context building works")


def test_context_empty_results():
    """Empty results produce 'not found' context."""
    if not Path("data/extracted/embeddings.npy").exists():
        print("  ⚠ Skipped: embeddings not found")
        return

    from src.rag import TenderRAG
    rag = TenderRAG()

    context = rag._build_context([])
    assert "not" in context.lower() or "No" in context
    print("  ✓ Empty context handled")


def test_rag_with_llm_if_available():
    """Test full RAG with LLM if API key is set."""
    if not Path("data/extracted/embeddings.npy").exists():
        print("  ⚠ Skipped: embeddings not found")
        return

    from src.rag import TenderRAG
    rag = TenderRAG()

    if rag.llm_client is None:
        print("  ⚠ Skipped: No GROQ_API_KEY in .env")
        return

    result = rag.ask("What certifications are required?", top_k=3)

    assert result["found"] is True
    assert len(result["answer"]) > 20
    assert "[LLM not configured" not in result["answer"]
    print(f"  ✓ Full RAG with LLM works")
    print(f"    Answer: {result['answer'][:100]}...")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  TenderIQ — Day 6 Tests")
    print("=" * 60 + "\n")

    tests = [
        test_rag_initializes,
        test_rag_retrieval_only,
        test_rag_evidence_has_page_numbers,
        test_rag_empty_question,
        test_rag_sources_list,
        test_context_building,
        test_context_empty_results,
        test_rag_with_llm_if_available,
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