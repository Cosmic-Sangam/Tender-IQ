"""
TenderIQ — NER Annotator (Expanded Patterns for 15+ Tender Formats)
"""

import sys, os, re, json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.logger_config import get_logger

logger = get_logger(__name__)

ENTITY_PATTERNS = [
    # CERTIFICATION
    (r'ISO\s+\d{4,5}(?::\d{4})?', 'CERTIFICATION'),
    (r'CMMI\s+Level\s+\d+', 'CERTIFICATION'),
    (r'SOC\s+\d+\s+Type\s+[IVX]+', 'CERTIFICATION'),
    (r'\bBIS\b|CE\s+mark|STQC|RoHS|ISI\s+mark', 'CERTIFICATION'),

    # FINANCIAL_REQUIREMENT
    (r'(?:Rs\.?|INR|₹|\$|USD|EUR|€)\s*[\d,]+(?:\.\d+)?\s*(?:Crore|Cr|Lakh|Lacs?|Million|Mn|Thousand|K|Billion)?', 'FINANCIAL_REQUIREMENT'),
    (r'\d+(?:\.\d+)?%\s*(?:of\s+(?:the\s+)?(?:total\s+)?(?:contract|order|project|quarterly|EMD)\s*(?:value|cost|payment)?)?', 'FINANCIAL_REQUIREMENT'),

    # DEADLINE
    (r'\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}(?:,\s*\d{1,2}:\d{2}\s*(?:AM|PM))?', 'DEADLINE'),
    (r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?', 'DEADLINE'),
    (r'\d+\s+(?:weeks?|months?|days?)\s+from\s+(?:the\s+)?(?:date\s+of\s+)?(?:purchase\s+order|commencement|award|signing)', 'DEADLINE'),

    # TECHNICAL_REQUIREMENT
    (r'(?:Java\s+Spring\s+Boot|React\.js|Angular|Node\.js|Django|Flask|FastAPI|\.NET|Python|Docker|Kubernetes)', 'TECHNICAL_REQUIREMENT'),
    (r'(?:PostgreSQL|MySQL|MongoDB|Oracle|SQL\s+Server|Redis)(?:\s+version\s+\d+)?', 'TECHNICAL_REQUIREMENT'),
    (r'(?:NIC\s+Cloud|MeghRaj\s+Cloud|AWS|Azure|GCP|Oracle\s+Cloud)', 'TECHNICAL_REQUIREMENT'),
    (r'(?:NVIDIA\s+\w+|AMD\s+EPYC\s+\d+|Intel\s+Xeon\s+\w+\s+\d+)', 'TECHNICAL_REQUIREMENT'),
    (r'\d+(?:,\d+)*\s+concurrent\s+users', 'TECHNICAL_REQUIREMENT'),
    (r'\d+(?:\.\d+)?%\s+uptime', 'TECHNICAL_REQUIREMENT'),
    (r'\d+\s+TB\s+(?:NVMe\s+)?SSD', 'TECHNICAL_REQUIREMENT'),
    (r'\d+\s+GB\s+(?:DDR\d+\s+)?(?:ECC\s+)?(?:RAM|GPUs?)', 'TECHNICAL_REQUIREMENT'),
    (r'Item\s+Category[:\s]+[A-Za-z0-9\s\(\)\-]+', 'TECHNICAL_REQUIREMENT'),

    # EXPERIENCE
    (r'\d+\s+years?\s+of\s+experience', 'EXPERIENCE'),
    (r'minimum\s+of\s+\d+\s+years', 'EXPERIENCE'),
    (r'\d+\s+similar\s+projects?', 'EXPERIENCE'),
    (r'at\s+least\s+\d+\s+(?:AI/ML\s+)?projects?', 'EXPERIENCE'),

    # PENALTY
    (r'penalty\s+of\s+(?:Rs\.?|₹|INR|\$)\s*[\d,]+(?:\s+per\s+\w+)?', 'PENALTY'),
    (r'\d+(?:\.\d+)?%\s+per\s+week', 'PENALTY'),
    (r'[Ll]iquidated\s+damages?\s+of\s+\d+(?:\.\d+)?%', 'PENALTY'),

    # DOCUMENT
    (r'(?:GST\s+(?:registration\s+)?certificate|PAN\s+card|Certificate\s+of\s+Incorporation|Registration\s+Certificate|Power\s+of\s+Attorney|Audited\s+(?:financial\s+statements|balance\s+sheets|accounts)|Work\s+order\s+copies|completion\s+certificates|OEM\s+authorization\s+letter|BoQ\s+specification|ATC|MSME\s+certificate|Startup\s+certificate)', 'DOCUMENT'),

    # ELIGIBILITY
    (r'registered\s+company\s+in\s+India', 'ELIGIBILITY'),
    (r'company\s+registered\s+under\s+the\s+Companies\s+Act', 'ELIGIBILITY'),
    (r'OEM\s+or\s+authorized\s+dealer', 'ELIGIBILITY'),
    (r'partnership\s+firm\s+registered', 'ELIGIBILITY'),
    (r'(?:MSE|Startup)\s+Relaxation', 'ELIGIBILITY'),
    (r'Ministry\s+of\s+[A-Za-z\s&]+', 'ELIGIBILITY'),
    (r'Department\s+of\s+[A-Za-z\s&]+', 'ELIGIBILITY'),
    (r'(?:Indian\s+Institute|IIT|IIIT|NIT|NIC|DRDO|ISRO|SBI|CAIR|CDAC|BARC)[^\n,;]*', 'ELIGIBILITY'),
]


def find_entities_in_text(text: str) -> list:
    entities = []
    used_spans = []

    for pattern, label in ENTITY_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start, end = match.start(), match.end()
            matched = match.group().strip()
            if len(matched) < 3:
                continue
            overlaps = any(start < ue and end > us for us, ue in used_spans)
            if not overlaps:
                entities.append({"text": matched, "start": start, "end": end, "label": label})
                used_spans.append((start, end))

    entities.sort(key=lambda e: e["start"])
    return entities


def annotate_text(text, document_id="", page_number=0):
    entities = find_entities_in_text(text)
    return {"document_id": document_id, "page_number": page_number, "text": text, "entities": entities, "entity_count": len(entities)}


def annotate_document(cleaned_json_path):
    logger.info(f"Annotating: {Path(cleaned_json_path).name}")
    with open(cleaned_json_path, "r", encoding="utf-8") as f:
        document = json.load(f)

    annotated = []
    total = 0
    for page in document["pages"]:
        if not page["text"].strip():
            continue
        sentences = re.split(r'(?<=[.!?])\s+|\n', page["text"])
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
        for s in sentences:
            a = annotate_text(s, document["document_id"], page["page_number"])
            annotated.append(a)
            total += a["entity_count"]

    logger.info(f"  → {len(annotated)} sentences, {total} entities")
    return annotated


def annotate_all_documents(extracted_dir):
    extracted_dir = Path(extracted_dir)
    cleaned_files = sorted(extracted_dir.glob("*_cleaned.json"))
    if not cleaned_files:
        return []

    all_annotated = []
    for cf in cleaned_files:
        all_annotated.extend(annotate_document(str(cf)))

    label_counts = {}
    for item in all_annotated:
        for e in item["entities"]:
            label_counts[e["label"]] = label_counts.get(e["label"], 0) + 1

    logger.info("Entity distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        logger.info(f"    {label:<28} {count:>4}")

    return all_annotated


def save_annotations(data, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(data)} annotations to {output_path}")