"""
TenderIQ — All-in-One Fix & Generator for Day 1 & Day 2
=========================================================
Run this script to automatically generate all sample data,
extract PDFs, clean text, annotate entities, and build the
spaCy dataset files.

USAGE:
    python fix_dataset.py
"""

import os
import sys
import json
from pathlib import Path

# Ensure root directory is in path
ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)
sys.path.insert(0, str(ROOT_DIR))

print(f"Working directory: {ROOT_DIR}")

# ─────────────────────────────────────────────────────────────
# 1. Ensure Folder Structure
# ─────────────────────────────────────────────────────────────
folders = [
    "data/sample_tenders",
    "data/raw_pdfs",
    "data/extracted",
    "data/ner_dataset",
    "logs",
    "models",
    "src"
]
for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 2. Check / Create Sample Tender Text Files
# ─────────────────────────────────────────────────────────────
sample_1 = """NOTICE INVITING TENDER
National Informatics Centre (NIC), New Delhi
Tender Ref: NIC/IT/2024/0847

1. ELIGIBILITY CRITERIA
1.1 The bidder must be a registered company in India.
1.2 Average annual turnover of at least Rs. 50 Crore during the last three financial years.
1.3 Minimum net worth of Rs. 10 Crore as on 31st March 2023.
1.4 Completed at least 3 similar projects of value not less than Rs. 5 Crore each in last 5 years.
1.5 Must possess valid ISO 9001:2015 certification for quality management.
1.6 Must possess valid ISO 27001:2013 certification for information security.
1.7 Must possess valid CMMI Level 3 or above certification.

2. TECHNICAL REQUIREMENTS
2.1 Developed using Java Spring Boot backend and React.js frontend.
2.2 Database: PostgreSQL version 14 or above.
2.3 Must support 10,000 concurrent users with 99.9% uptime.
2.4 Deployed on NIC Cloud or MeghRaj Cloud.

3. FINANCIAL REQUIREMENTS
3.1 Estimated cost: Rs. 15 Crore.
3.2 EMD of Rs. 30 Lakh as bank guarantee.
3.3 Performance guarantee: 10% of total contract value.
3.4 Penalty: 1% per week for delays, maximum 10%.

4. SUBMISSION DOCUMENTS
4.1 GST registration certificate and PAN card.
4.2 Certificate of Incorporation.
4.3 Audited financial statements for last 3 years.

5. IMPORTANT DATES
5.1 Bid submission deadline: 15th February 2024, 3:00 PM.
5.2 Technical bid opening: 16th February 2024, 11:00 AM.
"""

sample_2 = """REQUEST FOR PROPOSAL
State Bank of India (SBI), Corporate Centre, Mumbai
RFP Reference: SBI/IT/RFP/2024/036

1. ELIGIBILITY
1.1 Registered company with minimum 10 years of experience in IT services.
1.2 Average annual turnover of Rs. 100 Crore in last 3 years.
1.3 Implemented at least 2 AI/ML projects with value at least Rs. 10 Crore each.
1.4 Possess ISO 9001 and ISO 27001 certifications.
1.5 Possess CMMI Level 5 certification.

2. FINANCIAL TERMS
2.1 Estimated project value: Rs. 25 Crore.
2.2 EMD: Rs. 50 Lakh bank guarantee.
2.3 Liquidated damages of 0.5% per week for delay.

3. DOCUMENTS REQUIRED
3.1 ISO 9001:2015 certificate and ISO 27001:2013 certificate.
3.2 GST registration certificate and PAN card copy.
3.3 Audited balance sheets for last 3 years.

4. TIMELINE
4.1 Bid submission deadline: 28th February 2024, 4:00 PM.
4.2 Technical bid opening: 1st March 2024.
4.3 Project completion: 18 months from commencement date.
"""

sample_3 = """TENDER NOTICE
DRDO - Centre for Artificial Intelligence and Robotics (CAIR), Bangalore
Tender No: CAIR/PROC/2024/CS/001

1. ELIGIBILITY
1.1 OEM or authorized dealer with valid authorization letter.
1.2 Annual turnover of at least Rs. 25 Crore in last three years.
1.3 Supplying HPC systems to at least 2 Government organizations in India.
1.4 Minimum 5 years of experience in high-performance computing.

2. TECHNICAL SPECIFICATIONS
2.1 Minimum 8 NVIDIA A100 80GB GPUs.
2.2 Dual AMD EPYC 7763 processors per node.
2.3 200 TB NVMe SSD array with RAID.
2.4 Operating System: Ubuntu 22.04 LTS.

3. FINANCIALS & DATES
3.1 Estimated cost: Rs. 8 Crore.
3.2 EMD: Rs. 16 Lakh.
3.3 Tender fee: Rs. 5,000.
3.4 Bid submission deadline: 20th February 2024, 2:00 PM.
3.5 Delivery period: 12 weeks from date of purchase order.
"""

tenders = {
    "data/sample_tenders/sample_tender_001.txt": sample_1,
    "data/sample_tenders/sample_tender_002.txt": sample_2,
    "data/sample_tenders/sample_tender_003.txt": sample_3,
}

for path_str, content in tenders.items():
    p = Path(path_str)
    if not p.exists():
        p.write_text(content, encoding="utf-8")
        print(f"Created sample text: {p}")

# ─────────────────────────────────────────────────────────────
# 3. Run Pipeline Steps Directly
# ─────────────────────────────────────────────────────────────
from src.create_sample_data import create_all_sample_pdfs
from src.pdf_extractor import extract_all_pdfs
from src.text_cleaner import clean_document, save_cleaned_document, create_text_chunks
from src.ner_annotator import annotate_all_documents, save_annotations
from src.dataset_builder import build_complete_dataset

print("\n--- STEP 1: Creating PDFs ---")
create_all_sample_pdfs()

print("\n--- STEP 2: Extracting PDFs ---")
extract_all_pdfs("data/raw_pdfs", "data/extracted")

print("\n--- STEP 3: Cleaning Documents ---")
extracted_dir = Path("data/extracted")
for json_file in sorted(extracted_dir.glob("*_extracted.json")):
    cleaned = clean_document(str(json_file))
    save_cleaned_document(cleaned, "data/extracted")

print("\n--- STEP 4: Annotating Entities ---")
annotated_data = annotate_all_documents("data/extracted")
save_annotations(annotated_data, "data/ner_dataset/raw_annotations.json")

print("\n--- STEP 5: Building spaCy Dataset ---")
stats = build_complete_dataset(annotated_data, "data/ner_dataset")

# ─────────────────────────────────────────────────────────────
# 4. Verify Files in data/ner_dataset
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("VERIFYING data/ner_dataset/ CONTENTS:")
print("=" * 50)

dataset_files = list(Path("data/ner_dataset").glob("*"))
if not dataset_files:
    print("❌ ERROR: Folder is STILL empty. Please check console errors above.")
else:
    for f in sorted(dataset_files):
        size_kb = f.stat().st_size / 1024
        print(f"  ✓ {f.name:<30} ({size_kb:>6.1f} KB)")
    print("\n🎉 SUCCESS! All dataset files are generated and ready for Day 3.")