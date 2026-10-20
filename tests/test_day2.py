"""
TenderIQ — Day 2 Tests
========================

Tests for NER annotation and dataset building.

RUN:
    python tests/test_day2.py
"""

import sys
import os
import json
import tempfile

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.logger_config import get_logger
from src.ner_annotator import find_entities_in_text, annotate_text
from src.dataset_builder import (
    convert_to_spacy_format,
    split_dataset,
    compute_dataset_stats
)

logger = get_logger("test_day2")


# ═══════════════════════════════════════════════
# ANNOTATION TESTS
# ═══════════════════════════════════════════════

def test_certification_detection():
    """ISO and CMMI certifications are detected."""
    text = "The bidder must possess valid ISO 9001:2015 certification."
    entities = find_entities_in_text(text)

    cert_entities = [e for e in entities if e["label"] == "CERTIFICATION"]
    assert len(cert_entities) >= 1, "Should find at least 1 certification"
    assert "ISO 9001:2015" in cert_entities[0]["text"]

    print("  ✓ CERTIFICATION detected")


def test_cmmi_detection():
    """CMMI levels are detected."""
    text = "Bidder must have CMMI Level 3 or above certification."
    entities = find_entities_in_text(text)

    cert_entities = [e for e in entities if e["label"] == "CERTIFICATION"]
    assert len(cert_entities) >= 1
    assert "CMMI Level 3" in cert_entities[0]["text"]

    print("  ✓ CMMI Level detected")


def test_financial_requirement_detection():
    """Financial amounts in Rs. are detected."""
    text = "Minimum turnover of Rs. 50 Crore during the last three years."
    entities = find_entities_in_text(text)

    fin_entities = [e for e in entities if e["label"] == "FINANCIAL_REQUIREMENT"]
    assert len(fin_entities) >= 1, "Should find financial requirement"
    assert "Rs. 50 Crore" in fin_entities[0]["text"]

    print("  ✓ FINANCIAL_REQUIREMENT detected")


def test_deadline_detection():
    """Dates with month names are detected."""
    text = "Last date for submission of bids: 15th February 2024, 3:00 PM"
    entities = find_entities_in_text(text)

    deadline_entities = [e for e in entities if e["label"] == "DEADLINE"]
    assert len(deadline_entities) >= 1, "Should find deadline"

    print("  ✓ DEADLINE detected")


def test_technical_requirement_detection():
    """Tech stack items are detected."""
    text = "The application must use Java Spring Boot and PostgreSQL version 14."
    entities = find_entities_in_text(text)

    tech_entities = [e for e in entities if e["label"] == "TECHNICAL_REQUIREMENT"]
    assert len(tech_entities) >= 2, "Should find at least 2 tech requirements"

    print("  ✓ TECHNICAL_REQUIREMENT detected")


def test_penalty_detection():
    """Penalty clauses are detected."""
    text = "A penalty of Rs. 50,000 per hour of downtime shall be levied."
    entities = find_entities_in_text(text)

    penalty_entities = [e for e in entities if e["label"] == "PENALTY"]
    assert len(penalty_entities) >= 1

    print("  ✓ PENALTY detected")


def test_document_detection():
    """Required documents are detected."""
    text = "Submit GST registration certificate and PAN card copy."
    entities = find_entities_in_text(text)

    doc_entities = [e for e in entities if e["label"] == "DOCUMENT"]
    assert len(doc_entities) >= 1

    print("  ✓ DOCUMENT detected")


def test_entity_positions_correct():
    """Entity start/end positions match the actual text."""
    text = "Must have ISO 9001:2015 and Rs. 10 Crore turnover."
    entities = find_entities_in_text(text)

    for ent in entities:
        # Verify that text[start:end] matches the entity text
        extracted = text[ent["start"]:ent["end"]]
        assert extracted == ent["text"], (
            f"Position mismatch: '{extracted}' != '{ent['text']}'"
        )

    print("  ✓ Entity positions are correct")


def test_no_overlapping_entities():
    """Entities should not overlap with each other."""
    text = "ISO 9001:2015 certification and CMMI Level 3 required."
    entities = find_entities_in_text(text)

    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            e1 = entities[i]
            e2 = entities[j]
            # Check no overlap
            assert e1["end"] <= e2["start"] or e2["end"] <= e1["start"], (
                f"Overlapping entities: '{e1['text']}' and '{e2['text']}'"
            )

    print("  ✓ No overlapping entities")


def test_empty_text_no_crash():
    """Empty text should return empty list, not crash."""
    entities = find_entities_in_text("")
    assert entities == []

    entities = find_entities_in_text("   ")
    assert entities == []

    print("  ✓ Empty text handled")


# ═══════════════════════════════════════════════
# DATASET BUILDER TESTS
# ═══════════════════════════════════════════════

def test_spacy_format_conversion():
    """Annotation format converts to spaCy format correctly."""
    annotated = [
        {
            "text": "Must have ISO 9001 certification.",
            "entities": [
                {"start": 10, "end": 18, "label": "CERTIFICATION", "text": "ISO 9001"}
            ]
        }
    ]

    spacy_data = convert_to_spacy_format(annotated)

    assert len(spacy_data) == 1
    text, ann = spacy_data[0]
    assert text == "Must have ISO 9001 certification."
    assert len(ann["entities"]) == 1
    assert ann["entities"][0] == (10, 18, "CERTIFICATION")

    print("  ✓ spaCy format conversion works")


def test_dataset_split_ratios():
    """Split produces approximately correct ratios."""
    # Create dummy data
    dummy_data = [
        (f"Sentence {i}.", {"entities": []})
        for i in range(100)
    ]

    splits = split_dataset(dummy_data, 0.7, 0.15, 0.15)

    assert len(splits["train"]) == 70
    assert len(splits["dev"]) == 15
    assert len(splits["test"]) == 15

    # All data should be accounted for
    total = sum(len(s) for s in splits.values())
    assert total == 100

    print("  ✓ Dataset split ratios correct")


def test_split_reproducibility():
    """Same seed produces same split."""
    dummy_data = [
        (f"Sentence {i}.", {"entities": []})
        for i in range(50)
    ]

    split1 = split_dataset(dummy_data, seed=42)
    split2 = split_dataset(dummy_data, seed=42)

    # Same seed = same split
    assert split1["train"][0][0] == split2["train"][0][0]

    print("  ✓ Split is reproducible")


def test_dataset_stats():
    """Statistics are computed correctly."""
    dummy_data = {
        "train": [
            ("Text with ISO 9001.", {"entities": [(10, 18, "CERTIFICATION")]}),
            ("No entities here.", {"entities": []}),
        ],
        "test": [
            ("Rs. 10 Crore.", {"entities": [(0, 12, "FINANCIAL_REQUIREMENT")]}),
        ]
    }

    stats = compute_dataset_stats(dummy_data)

    assert stats["train"]["total_sentences"] == 2
    assert stats["train"]["total_entities"] == 1
    assert stats["train"]["sentences_with_entities"] == 1
    assert stats["test"]["total_entities"] == 1

    print("  ✓ Dataset statistics correct")


# ═══════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════

def run_all_tests():
    print("\n" + "=" * 60)
    print("  TenderIQ — Day 2 Tests")
    print("=" * 60 + "\n")

    tests = [
        test_certification_detection,
        test_cmmi_detection,
        test_financial_requirement_detection,
        test_deadline_detection,
        test_technical_requirement_detection,
        test_penalty_detection,
        test_document_detection,
        test_entity_positions_correct,
        test_no_overlapping_entities,
        test_empty_text_no_crash,
        test_spacy_format_conversion,
        test_dataset_split_ratios,
        test_split_reproducibility,
        test_dataset_stats,
    ]

    passed = 0
    failed = 0

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

    print("\n" + "-" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("-" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)