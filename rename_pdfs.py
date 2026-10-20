"""
Auto-rename all PDFs in data/raw_pdfs/ to clean lowercase format.

USAGE:
    python rename_pdfs.py

BEFORE:
    GeM Bid - Waste Container (V3).pdf
    CPPP_Software Development_2024-25.pdf
    SBI IT Services RFP (Mumbai).pdf

AFTER:
    gem_bid_waste_container_v3.pdf
    cppp_software_development_2024_25.pdf
    sbi_it_services_rfp_mumbai.pdf
"""

import os
import re
from pathlib import Path


def clean_filename(name: str) -> str:
    """Convert any filename to clean lowercase_underscore format."""
    # Remove extension
    stem = Path(name).stem

    # Replace common separators with underscores
    stem = re.sub(r'[\s\-\(\)\[\]\{\},;:]+', '_', stem)

    # Remove special characters (keep only letters, numbers, underscores)
    stem = re.sub(r'[^a-zA-Z0-9_]', '', stem)

    # Convert to lowercase
    stem = stem.lower()

    # Remove multiple underscores
    stem = re.sub(r'_+', '_', stem)

    # Remove leading/trailing underscores
    stem = stem.strip('_')

    # Truncate if too long (max 60 chars)
    if len(stem) > 60:
        stem = stem[:60].rstrip('_')

    return stem + ".pdf"


def rename_all_pdfs(folder: str = "data/raw_pdfs"):
    folder_path = Path(folder)

    if not folder_path.exists():
        print(f"Folder not found: {folder}")
        return

    pdf_files = list(folder_path.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        return

    print(f"\nFound {len(pdf_files)} PDFs to rename:\n")

    renamed = 0
    for pdf in pdf_files:
        new_name = clean_filename(pdf.name)

        if pdf.name == new_name:
            print(f"  ✓ Already clean: {pdf.name}")
            continue

        new_path = pdf.parent / new_name

        # Avoid overwriting existing files
        counter = 1
        while new_path.exists():
            stem = Path(new_name).stem
            new_path = pdf.parent / f"{stem}_{counter}.pdf"
            counter += 1

        pdf.rename(new_path)
        print(f"  ✏️  {pdf.name}")
        print(f"     → {new_path.name}")
        renamed += 1

    print(f"\n✅ Renamed {renamed} files. {len(pdf_files) - renamed} already clean.\n")


if __name__ == "__main__":
    rename_all_pdfs()