"""
TenderIQ — Day 4 Tests
RUN: python tests/test_day4.py
"""

import sys, os
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.rule_extractor import RuleExtractor


def test_tender_id_extraction():
    ext = RuleExtractor()
    text = "Tender Reference Number: NIC/IT/2024/0847"
    result = ext.extract(text)
    assert "tender_id" in result
    assert "NIC/IT/2024/0847" in result["tender_id"]["value"]
    print("  ✓ Tender ID extracted")


def test_emd_extraction():
    ext = RuleExtractor()
    text = "EMD of Rs. 30 Lakh must be submitted along with the bid."
    result = ext.extract(text)
    assert "emd" in result
    assert "30 Lakh" in result["emd"]["value"]
    print("  ✓ EMD extracted")


def test_estimated_cost_extraction():
    ext = RuleExtractor()
    text = "The estimated cost of the project is Rs. 15 Crore."
    result = ext.extract(text)
    assert "estimated_cost" in result
    assert "15 Crore" in result["estimated_cost"]["value"]
    print("  ✓ Estimated cost extracted")


def test_turnover_extraction():
    ext = RuleExtractor()
    text = "Average annual turnover of at least Rs. 50 Crore during last three years."
    result = ext.extract(text)
    assert "minimum_turnover" in result
    print("  ✓ Turnover extracted")


def test_deadline_extraction():
    ext = RuleExtractor()
    text = "Bid submission deadline: 15th February 2024, 3:00 PM"
    result = ext.extract(text)
    assert "bid_deadline" in result
    assert "15th February 2024" in result["bid_deadline"]["value"]
    print("  ✓ Deadline extracted")


def test_penalty_extraction():
    ext = RuleExtractor()
    text = "A penalty of Rs. 50,000 per hour of downtime shall be levied."
    result = ext.extract(text)
    assert "penalty_amount" in result
    print("  ✓ Penalty extracted")


def test_uptime_extraction():
    ext = RuleExtractor()
    text = "The system must have 99.9% uptime availability."
    result = ext.extract(text)
    assert "uptime_requirement" in result
    print("  ✓ Uptime extracted")


def test_empty_text():
    ext = RuleExtractor()
    result = ext.extract("")
    assert result == {}
    print("  ✓ Empty text handled")


def test_page_number_preserved():
    ext = RuleExtractor()
    text = "EMD of Rs. 30 Lakh required."
    result = ext.extract(text, page_number=7)
    assert result["emd"]["page"] == 7
    print("  ✓ Page number preserved")


def test_hybrid_pipeline_loads():
    if not Path("models/tender_ner_v1").exists():
        print("  ⚠ Skipped: model not found")
        return
    from src.hybrid_pipeline import HybridPipeline
    pipeline = HybridPipeline("models/tender_ner_v1")
    assert pipeline.ner is not None
    assert pipeline.rules is not None
    print("  ✓ Hybrid pipeline initialized")


def test_hybrid_process_text():
    if not Path("models/tender_ner_v1").exists():
        print("  ⚠ Skipped: model not found")
        return
    from src.hybrid_pipeline import HybridPipeline
    pipeline = HybridPipeline("models/tender_ner_v1")
    text = "ISO 9001 required. EMD of Rs. 30 Lakh. Deadline: 15th February 2024, 3:00 PM"
    result = pipeline.process_text(text, page_number=1)
    assert "ner_entities" in result
    assert "rule_fields" in result
    print("  ✓ Hybrid text processing works")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  TenderIQ — Day 4 Tests")
    print("=" * 60 + "\n")

    tests = [
        test_tender_id_extraction,
        test_emd_extraction,
        test_estimated_cost_extraction,
        test_turnover_extraction,
        test_deadline_extraction,
        test_penalty_extraction,
        test_uptime_extraction,
        test_empty_text,
        test_page_number_preserved,
        test_hybrid_pipeline_loads,
        test_hybrid_process_text,
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