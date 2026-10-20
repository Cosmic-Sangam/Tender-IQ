"""
TenderIQ — Hybrid NER Predictor (spaCy + Pattern Layer)
=======================================================
Combines trained statistical model weights with domain-specific
pattern matching to extract entities from real-world documents.
"""

import sys
import os
import json
import re
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spacy
from src.logger_config import get_logger

logger = get_logger(__name__)

# Broad patterns for procurement, government bids (GeM/CPPP), and IT RFPs
DOMAIN_PATTERNS = [
    # ── DEADLINES & DATES ──
    (r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?\b', 'DEADLINE'),
    (r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}(?:,\s*\d{1,2}:\d{2}\s*(?:AM|PM)?)?\b', 'DEADLINE'),
    (r'\b\d{1,3}\s*(?:\(Days\)|\s*Days?)\b', 'DEADLINE'),

    # ── FINANCIAL & MONETARY ──
    (r'(?:Rs\.?|INR|₹|\$|USD|EUR|€)\s*[\d,]+(?:\.\d+)?(?:\s*(?:Crore|Cr|Lakh|Lacs?|Million|Thousand|K))?', 'FINANCIAL_REQUIREMENT'),
    (r'\b\d+(?:\.\d+)?\s*%\s*(?:of\s+[\w\s]+)?(?:value|cost|turnover|contract|EMD|security)?\b', 'FINANCIAL_REQUIREMENT'),

    # ── ORGANISATION / ELIGIBILITY ──
    (r'(?:Indian\s+Institute\s+of\s+Information\s+Technology|IIIT|IIT|NIT|National\s+Informatics\s+Centre|NIC|DRDO|ISRO|SBI|CAIR)\b[^\n,;]*', 'ORGANISATION'),
    (r'Ministry\s+of\s+[A-Za-z\s&]+', 'ORGANISATION'),
    (r'Department\s+of\s+[A-Za-z\s&]+', 'ORGANISATION'),
    (r'(?:MSE|Startup|OEM)\s+Relaxation[^\n,;]*', 'ELIGIBILITY'),
    (r'(?:registered\s+company|partnership\s+firm|authorized\s+dealer)\b', 'ELIGIBILITY'),

    # ── CERTIFICATIONS ──
    (r'\bISO\s+\d{4,5}(?::\d{4})?\b', 'CERTIFICATION'),
    (r'\bCMMI\s+Level\s+\d+\b', 'CERTIFICATION'),
    (r'\bSOC\s+\d+\s+Type\s+[IVX]+\b', 'CERTIFICATION'),
    (r'\bBIS|CE|STQC|RoHS\b', 'CERTIFICATION'),

    # ── TECHNICAL & PRODUCT CATEGORIES ──
    (r'Item\s+Category\s*[:/]\s*([A-Za-z0-9\s\(\)\-_]+)', 'TECHNICAL_REQUIREMENT'),
    (r'\b(?:Waste\s+Container|GPU|NVIDIA|PostgreSQL|React\.js|Java|Spring\s+Boot|Python|Docker|Kubernetes|Cloud)\b[^\n,;]*', 'TECHNICAL_REQUIREMENT'),
    (r'\b\d+(?:\.\d+)?\s*%\s*(?:uptime|availability)\b', 'TECHNICAL_REQUIREMENT'),
    (r'\b[\d,]+\s*(?:concurrent\s+users|units|qty|quantity|sets|nos)\b', 'TECHNICAL_REQUIREMENT'),

    # ── DOCUMENTS ──
    (r'(?:BoQ\s+specification|ATC|PAN\s+card|GST\s+certificate|Audited\s+financial\s+statements|Certificate\s+of\s+Incorporation)', 'DOCUMENT'),
    (r'Document\s+required\s+from\s+seller', 'DOCUMENT'),

    # ── PENALTIES ──
    (r'penalty\s+of\s+[^\n,;]+', 'PENALTY'),
    (r'Liquidated\s+damages[^\n,;]+', 'PENALTY'),
]


class TenderNERPredictor:
    """Combines statistical spaCy weights with regex-based domain pattern matching."""

    def __init__(self, model_dir="models/tender_ner_v1"):
        self.model_dir = Path(model_dir)
        self.nlp = None
        self.labels = [
            "CERTIFICATION", "FINANCIAL_REQUIREMENT", "DEADLINE",
            "TECHNICAL_REQUIREMENT", "EXPERIENCE", "PENALTY",
            "DOCUMENT", "ELIGIBILITY", "ORGANISATION"
        ]

        if self.model_dir.exists():
            try:
                self.nlp = spacy.load(str(self.model_dir))
                logger.info(f"Loaded trained spaCy NER model from {model_dir}")
            except Exception as e:
                logger.warning(f"Could not load spaCy model: {e}")

    def predict(self, text: str) -> list:
        if not text or not text.strip():
            return []

        entities = []
        occupied_spans = []

        # 1. Run statistical model if available
        if self.nlp:
            try:
                doc = self.nlp(text[:10000])  # limit chunk size for fast prediction
                for ent in doc.ents:
                    entities.append({
                        "text": ent.text.strip(),
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "label": ent.label_
                    })
                    occupied_spans.append((ent.start_char, ent.end_char))
            except Exception as e:
                logger.debug(f"spaCy prediction error: {e}")

        # 2. Run domain pattern layer (fills in gaps on unseen/bilingual documents)
        for pattern, label in DOMAIN_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start, end = match.start(), match.end()
                matched_text = match.group(0).strip()

                if len(matched_text) < 2:
                    continue

                # Check if this span overlaps with an already-extracted entity
                overlaps = any(
                    (start < occ_end and end > occ_start)
                    for occ_start, occ_end in occupied_spans
                )

                if not overlaps:
                    entities.append({
                        "text": matched_text,
                        "start": start,
                        "end": end,
                        "label": label
                    })
                    occupied_spans.append((start, end))

        entities.sort(key=lambda x: x["start"])
        return entities

    def get_entities_by_label(self, text: str) -> dict:
        entities = self.predict(text)
        grouped = {}
        for ent in entities:
            lbl = ent["label"]
            if lbl not in grouped:
                grouped[lbl] = []
            if ent["text"] not in grouped[lbl]:
                grouped[lbl].append(ent["text"])
        return grouped