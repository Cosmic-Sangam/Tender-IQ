"""
TenderIQ — Day 3 Tests
========================

Tests for NER training, evaluation, and prediction.

RUN:
    python tests/test_day3.py

NOTE:
    These tests require that Day 3 has been run at least once
    (i.e., trained model exists at models/tender_ner_v1/).
"""

import sys
import os
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.logger_config import get_logger
from src.ner_evaluator import compute_metrics

logger = get_logger("test_day3")


# ═══════════════════════════════════════════════
# METRICS TESTS (no model needed)
# ═══════════════════════════════════════════════

def test_perfect_precision():
    """When no false positives, precision = 1.0"""
    metrics = compute_metrics(tp=10, fp=0, fn=5)
    assert metrics["precision"] == 1.0
    print("  ✓ Perfect precision calculated correctly")


def test_perfect_recall():
    """When no false negatives, recall = 1.0"""
    metrics = compute_metrics(tp=10, fp=3, fn=0)
    assert metrics["recall"] == 1.0
    print("  ✓ Perfect recall calculated correctly")


def test_zero_predictions():
    """When model predicts nothing."""
    metrics = compute_metrics(tp=0, fp=0, fn=10)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    print("  ✓ Zero predictions handled")


def test_f1_calculation():
    """F1 should be harmonic mean of P and R."""
    # P = 10 / (10+10) = 0.5
    # R = 10 / (10+10) = 0.5
    # F1 = 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5
    metrics = compute_metrics(tp=10, fp=10, fn=10)
    assert abs(metrics["precision"] - 0.5) < 0.01
    assert abs(metrics["recall"] - 0.5) < 0.01
    assert abs(metrics["f1"] - 0.5) < 0.01
    print("  ✓ F1 calculation correct")


def test_asymmetric_metrics():
    """High precision, low recall example."""
    # P = 9/10 = 0.9
    # R = 9/100 = 0.09
    # F1 = 2 * 0.9 * 0.09 / (0.9 + 0.09) ≈ 0.164
    metrics = compute_metrics(tp=9, fp=1, fn=91)
    assert abs(metrics["precision"] - 0.9) < 0.01
    assert abs(metrics["recall"] - 0.09) < 0.01
    assert metrics["f1"] < 0.2  # F1 should reflect poor recall
    print("  ✓ Asymmetric metrics computed correctly")


# ═══════════════════════════════════════════════
# PREDICTOR TESTS (requires trained model)
# ═══════════════════════════════════════════════

def test_predictor_loads():
    """Trained model can be loaded."""
    model_dir = "models/tender_ner_v1"

    if not Path(model_dir).exists():
        print(f"  ⚠ Skipped: model not found at {model_dir}")
        print(f"    Run: python run_day3.py")
        return

    from src.ner_predictor import TenderNERPredictor
    predictor = TenderNERPredictor(model_dir)

    assert predictor.nlp is not None
    assert len(predictor.labels) > 0

    print(f"  ✓ Model loaded ({len(predictor.labels)} labels)")


def test_predictor_returns_entities():
    """Predictor returns entities for text with known patterns."""
    model_dir = "models/tender_ner_v1"

    if not Path(model_dir).exists():
        print("  ⚠ Skipped: model not found")
        return

    from src.ner_predictor import TenderNERPredictor
    predictor = TenderNERPredictor(model_dir)

    text = "The bidder must have ISO 9001:2015 certification."
    entities = predictor.predict(text)

    # Should return a list (may or may not find entities depending on training)
    assert isinstance(entities, list)

    # If entities found, they should have required fields
    for ent in entities:
        assert "text" in ent
        assert "start" in ent
        assert "end" in ent
        assert "label" in ent
        # Positions should be valid
        assert text[ent["start"]:ent["end"]] == ent["text"]

    print(f"  ✓ Predictor works ({len(entities)} entities in sample)")


def test_predictor_handles_empty():
    """Empty text returns empty list."""
    model_dir = "models/tender_ner_v1"

    if not Path(model_dir).exists():
        print("  ⚠ Skipped: model not found")
        return

    from src.ner_predictor import TenderNERPredictor
    predictor = TenderNERPredictor(model_dir)

    assert predictor.predict("") == []
    assert predictor.predict("   ") == []

    print("  ✓ Empty text handled")


def test_predictor_batch():
    """Batch prediction works."""
    model_dir = "models/tender_ner_v1"

    if not Path(model_dir).exists():
        print("  ⚠ Skipped: model not found")
        return

    from src.ner_predictor import TenderNERPredictor
    predictor = TenderNERPredictor(model_dir)

    texts = [
        "ISO 9001 required.",
        "Rs. 10 Crore turnover.",
        "Deadline 15th March 2024."
    ]
    results = predictor.predict_batch(texts)

    assert len(results) == 3
    assert all(isinstance(r, list) for r in results)

    print("  ✓ Batch prediction works")


def test_entities_by_label():
    """get_entities_by_label groups correctly."""
    model_dir = "models/tender_ner_v1"

    if not Path(model_dir).exists():
        print("  ⚠ Skipped: model not found")
        return

    from src.ner_predictor import TenderNERPredictor
    predictor = TenderNERPredictor(model_dir)

    text = "ISO 9001 and CMMI Level 3 required. Rs. 10 Crore turnover."
    grouped = predictor.get_entities_by_label(text)

    # Should return a dict
    assert isinstance(grouped, dict)
    # All values should be lists
    for label, entities in grouped.items():
        assert isinstance(entities, list)

    print("  ✓ Entities grouped by label")


# ═══════════════════════════════════════════════
# EVALUATION TESTS
# ═══════════════════════════════════════════════

def test_evaluation_report_exists():
    """After running Day 3, evaluation report should exist."""
    report_path = Path("docs/model_evaluation.md")

    if not report_path.exists():
        print("  ⚠ Skipped: run_day3.py not run yet")
        return

    content = report_path.read_text(encoding="utf-8")
    assert "Precision" in content
    assert "Recall" in content
    assert "F1" in content

    print("  ✓ Evaluation report exists and has metrics")


# ═══════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════

def run_all_tests():
    print("\n" + "=" * 60)
    print("  TenderIQ — Day 3 Tests")
    print("=" * 60 + "\n")

    tests = [
        # Metric calculation tests
        test_perfect_precision,
        test_perfect_recall,
        test_zero_predictions,
        test_f1_calculation,
        test_asymmetric_metrics,
        # Predictor tests
        test_predictor_loads,
        test_predictor_returns_entities,
        test_predictor_handles_empty,
        test_predictor_batch,
        test_entities_by_label,
        # Report test
        test_evaluation_report_exists,
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