"""
TenderIQ — Day 1 Tests
========================

Tests for:
- PDF extraction
- Text cleaning
- Chunking with page number preservation

RUN:
    python tests/test_day1.py
"""

import sys
import os
import json
import tempfile

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import fitz
from src.logger_config import get_logger
from src.pdf_extractor import extract_pdf, save_extraction
from src.text_cleaner import clean_text, clean_document, create_text_chunks

logger = get_logger("test_day1")


def create_test_pdf(path, texts):
    """Helper: create a test PDF with given page texts."""
    doc = fitz.open()
    for text in texts:
        page = doc.new_page()
        if text:
            page.insert_text((72, 100), text, fontsize=11)
    doc.save(path)
    doc.close()


# ═══════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════

def test_extraction_structure():
    """Extraction returns all required fields."""
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = os.path.join(tmp, "test.pdf")
        create_test_pdf(pdf_path, ["Test content for extraction."])

        result = extract_pdf(pdf_path, "test_001")

        required = ["document_id", "pages", "total_pages",
                    "full_text", "extraction_time", "total_words"]
        for field in required:
            assert field in result, f"Missing field: {field}"

        assert result["document_id"] == "test_001"
        assert result["total_pages"] == 1
        assert result["pages"][0]["page_number"] == 1

        print("  ✓ Extraction structure correct")


def test_page_numbers_are_1_based():
    """Page numbers must start at 1, not 0."""
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = os.path.join(tmp, "test.pdf")
        create_test_pdf(pdf_path, ["Page 1", "Page 2", "Page 3"])

        result = extract_pdf(pdf_path)
        page_nums = [p["page_number"] for p in result["pages"]]

        assert page_nums == [1, 2, 3], f"Got {page_nums}"
        assert 0 not in page_nums

        print("  ✓ Page numbers are 1-based")


def test_text_extraction_works():
    """Extracted text is not empty."""
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = os.path.join(tmp, "test.pdf")
        create_test_pdf(pdf_path, ["The bidder must have ISO 9001."])

        result = extract_pdf(pdf_path)

        assert result["total_words"] > 0
        assert len(result["pages"][0]["text"]) > 0

        print("  ✓ Text extraction works")


def test_missing_file_raises_error():
    """Missing PDF raises FileNotFoundError."""
    try:
        extract_pdf("does_not_exist.pdf")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        print("  ✓ FileNotFoundError raised correctly")


def test_non_pdf_raises_error():
    """Non-PDF file raises ValueError."""
    with tempfile.TemporaryDirectory() as tmp:
        txt = os.path.join(tmp, "not_pdf.txt")
        with open(txt, "w") as f:
            f.write("not a pdf")

        try:
            extract_pdf(txt)
            assert False, "Should have raised ValueError"
        except ValueError:
            print("  ✓ ValueError raised for non-PDF")


def test_text_cleaning_fixes_hyphenation():
    """Hyphenated line breaks are fixed."""
    dirty = "The bidder must have certifi-\ncation."
    clean = clean_text(dirty)
    assert "certification" in clean
    assert "certifi-" not in clean

    print("  ✓ Hyphenation fixed")


def test_text_cleaning_normalizes_whitespace():
    """Multiple spaces become single spaces."""
    dirty = "bidder    must   have   ISO"
    clean = clean_text(dirty)
    assert "    " not in clean
    assert "   " not in clean

    print("  ✓ Whitespace normalized")


def test_text_cleaning_removes_page_numbers():
    """Standalone page numbers are removed."""
    dirty = "The bidder must qualify.\n5\nNext requirement."
    clean = clean_text(dirty)
    # Line "5" alone should be removed
    assert "\n5\n" not in "\n" + clean + "\n"

    print("  ✓ Page number artifacts removed")


def test_empty_input_handled():
    """Empty text doesn't crash."""
    assert clean_text("") == ""
    assert clean_text("   ") == ""
    assert clean_text("\n\n\n") == ""

    print("  ✓ Empty input handled")


def test_save_load_round_trip():
    """Save then load produces identical data."""
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = os.path.join(tmp, "test.pdf")
        create_test_pdf(pdf_path, ["Save load test content."])

        original = extract_pdf(pdf_path, "round_trip_test")
        saved_path = save_extraction(original, tmp)

        with open(saved_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["document_id"] == original["document_id"]
        assert loaded["total_pages"] == original["total_pages"]

        print("  ✓ Save/load round trip works")


def test_chunking_preserves_page_numbers():
    """Chunks retain their source page number."""
    cleaned = {
        "document_id": "test",
        "pages": [
            {
                "page_number": 7,  # Non-trivial page number
                "text": "The bidder must have ISO 9001. "
                        "Minimum turnover Rs. 10 Crore. "
                        "Project duration 18 months. "
                        "EMD required Rs. 50 Lakh."
            }
        ]
    }

    chunks = create_text_chunks(cleaned, chunk_size=2)

    assert len(chunks) > 0
    for chunk in chunks:
        assert "page_number" in chunk
        assert chunk["page_number"] == 7  # Preserved from source!

    print("  ✓ Chunks preserve page numbers")


def test_multiple_pages_extracted():
    """Multi-page PDF: all pages extracted."""
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = os.path.join(tmp, "multi.pdf")
        create_test_pdf(pdf_path, [
            "First page content here.",
            "Second page content here.",
            "Third page content here.",
            "Fourth page content here.",
            "Fifth page content here."
        ])

        result = extract_pdf(pdf_path)

        assert result["total_pages"] == 5
        assert len(result["pages"]) == 5

        print("  ✓ Multi-page extraction works")


# ═══════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════

def run_all_tests():
    print("\n" + "=" * 60)
    print("  TenderIQ — Day 1 Tests")
    print("=" * 60 + "\n")

    tests = [
        test_extraction_structure,
        test_page_numbers_are_1_based,
        test_text_extraction_works,
        test_missing_file_raises_error,
        test_non_pdf_raises_error,
        test_text_cleaning_fixes_hyphenation,
        test_text_cleaning_normalizes_whitespace,
        test_text_cleaning_removes_page_numbers,
        test_empty_input_handled,
        test_save_load_round_trip,
        test_chunking_preserves_page_numbers,
        test_multiple_pages_extracted,
    ]

    passed = 0
    failed = 0
    errors = []

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            failed += 1
            errors.append((test.__name__, str(e)))
            print(f"  ✗ {test.__name__}: {e}")
        except Exception as e:
            failed += 1
            errors.append((test.__name__, str(e)))
            print(f"  ✗ {test.__name__}: {type(e).__name__}: {e}")

    print("\n" + "-" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("-" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)