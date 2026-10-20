"""
TenderIQ — Smart Fallback Hybrid Pipeline
=========================================
Executes deterministic rule patterns first, then implements a fallback matching
engine that queries the custom NER model if rules yield empty extractions.
"""

import json
from pathlib import Path
from collections import defaultdict
from src.logger_config import get_logger
from src.ner_predictor import TenderNERPredictor
from src.rule_extractor import RuleExtractor

logger = get_logger(__name__)


class HybridPipeline:
    """Combines NER + Rules with a robust semantic fallback mapping."""

    def __init__(self, model_dir: str = "models/tender_ner_v1"):
        logger.info("Initializing Hybrid Pipeline...")
        self.ner = TenderNERPredictor(model_dir)
        self.rules = RuleExtractor()
        logger.info("Hybrid Pipeline ready")

    def process_document(self, cleaned_json_path: str) -> dict:
        logger.info(f"Processing document: {Path(cleaned_json_path).name}")

        with open(cleaned_json_path, "r", encoding="utf-8") as f:
            document = json.load(f)

        doc_id = document["document_id"]

        # Run NER across all pages
        all_ner = defaultdict(list)
        ner_details = []

        for page in document["pages"]:
            if not page["text"].strip():
                continue

            entities = self.ner.predict(page["text"])
            for ent in entities:
                ent["page"] = page["page_number"]
                ner_details.append(ent)
                if ent["text"] not in all_ner[ent["label"]]:
                    all_ner[ent["label"]].append(ent["text"])

        # Run deterministic Rules
        rule_fields = self.rules.extract_all_pages(document["pages"])

        # ── SMART FALLBACK ENGINE ──
        # If rules fail to extract exact fields, fallback to mapping NER labels
        def get_fallback_entity(target_label: str, index: int = 0):
            """Returns the N-th detected statistical NER entity for a given label."""
            vals = [ent for ent in ner_details if ent["label"] == target_label]
            if len(vals) > index:
                selected = vals[index]
                return {
                    "value": f"{selected['text']} (Suggested via ML)",
                    "page": selected["page"]
                }
            return None

        # Fallback mapping rules
        if "minimum_turnover" not in rule_fields:
            fallback = get_fallback_entity("FINANCIAL_REQUIREMENT", index=0)
            if fallback:
                rule_fields["minimum_turnover"] = fallback

        if "emd" not in rule_fields:
            # Pick the second financial entity if turnover took the first, else the first
            idx = 1 if "minimum_turnover" in rule_fields and "Suggested via ML" in str(rule_fields["minimum_turnover"]) else 0
            fallback = get_fallback_entity("FINANCIAL_REQUIREMENT", index=idx)
            if fallback:
                rule_fields["emd"] = fallback

        if "bid_deadline" not in rule_fields:
            fallback = get_fallback_entity("DEADLINE")
            if fallback:
                rule_fields["bid_deadline"] = fallback

        if "experience_years" not in rule_fields:
            fallback = get_fallback_entity("EXPERIENCE")
            if fallback:
                rule_fields["experience_years"] = fallback

        # Build final unified outputs
        result = {
            "document_id": doc_id,
            "file_name": document["file_name"],
            "total_pages": document["total_pages"],
            "structured_data": {
                "certifications": all_ner.get("CERTIFICATION", []),
                "financial_requirements": all_ner.get("FINANCIAL_REQUIREMENT", []),
                "technical_requirements": all_ner.get("TECHNICAL_REQUIREMENT", []),
                "deadlines": all_ner.get("DEADLINE", []),
                "documents_required": all_ner.get("DOCUMENT", []),
                "eligibility": all_ner.get("ELIGIBILITY", []),
                "experience": all_ner.get("EXPERIENCE", []),
                "penalties": all_ner.get("PENALTY", []),
            },
            "rule_extracted": {
                field: data["value"]
                for field, data in rule_fields.items()
            },
            "rule_extracted_with_pages": rule_fields,
            "statistics": {
                "total_ner_entities": len(ner_details),
                "total_rule_fields": len(rule_fields),
                "ner_labels_found": len(all_ner),
            }
        }

        return result
    
    def process_all_documents(self, extracted_dir: str) -> list:
        """Process all cleaned JSON documents in a directory."""
        from pathlib import Path
        
        extracted_path = Path(extracted_dir)
        json_files = sorted(list(extracted_path.glob("*_cleaned.json")))
        
        if not json_files:
            logger.warning(f"No *_cleaned.json files found in {extracted_dir}")
            return []

        logger.info(f"Found {len(json_files)} documents to process.")
        results = []

        for json_file in json_files:
            res = self.process_document(str(json_file))
            results.append(res)

        return results

@staticmethod
def save_structured_output(results: list, output_path: str) -> str:
    """Save structured extraction results as JSON."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved structured output to {out_p}")
    return str(out_p)