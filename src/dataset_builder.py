"""
TenderIQ — NER Dataset Builder
================================

WHAT THIS DOES:
    Converts annotated text into spaCy's training format and
    splits the data into train / validation / test sets.

WHAT IS SPACY FORMAT?
    spaCy expects training data as a list of tuples:
    [
        (
            "The bidder must have ISO 9001 certification.",
            {
                "entities": [(20, 28, "CERTIFICATION")]
            }
        ),
        ...
    ]

    Each tuple is (text, {"entities": [(start, end, label), ...]})

WHY SPLIT THE DATA?
    - Training set: Model learns from this (70%)
    - Validation set: Tune hyperparameters during training (15%)
    - Test set: Final evaluation, never seen during training (15%)

WHY DOCUMENT-LEVEL SPLITTING?
    If we split randomly by sentence, sentences from the SAME tender
    could end up in both train and test sets. The model would then
    be tested on text it has already "seen" — this is DATA LEAKAGE.

    By splitting at the DOCUMENT level, the test set contains
    entirely unseen tenders. This gives honest evaluation metrics.

USAGE:
    from src.dataset_builder import build_spacy_dataset, split_dataset
"""

import sys
import os
import json
import random
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spacy
from spacy.tokens import DocBin

from src.logger_config import get_logger

logger = get_logger(__name__)


def convert_to_spacy_format(annotated_data: list[dict]) -> list[tuple]:
    """
    Convert our annotation format to spaCy's training format.

    Our format:
        {"text": "...", "entities": [{"start": 5, "end": 12, "label": "CERT"}, ...]}

    spaCy format:
        ("...", {"entities": [(5, 12, "CERT"), ...]})

    Args:
        annotated_data: List of annotated sentence dictionaries

    Returns:
        List of (text, annotations) tuples in spaCy format
    """
    spacy_data = []
    skipped = 0

    for item in annotated_data:
        text = item["text"]
        entities = item.get("entities", [])

        if not text.strip():
            skipped += 1
            continue

        # Convert entity dicts to tuples: (start, end, label)
        entity_tuples = []
        for ent in entities:
            start = ent["start"]
            end = ent["end"]
            label = ent["label"]

            # Validate span boundaries
            if start < 0 or end > len(text) or start >= end:
                logger.warning(
                    f"Invalid span ({start}, {end}) in text of "
                    f"length {len(text)}. Skipping entity."
                )
                continue

            entity_tuples.append((start, end, label))

        # Include ALL sentences, even those without entities.
        # The model needs to learn what is NOT an entity too.
        spacy_data.append((text, {"entities": entity_tuples}))

    if skipped > 0:
        logger.debug(f"Skipped {skipped} empty sentences")

    logger.info(
        f"Converted {len(spacy_data)} sentences to spaCy format "
        f"({sum(1 for _, a in spacy_data if a['entities'])} with entities)"
    )

    return spacy_data


def split_dataset(
    spacy_data: list[tuple],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> dict:
    """
    Split dataset into train / validation / test sets.

    We use RANDOM SPLITTING at the sentence level here because
    we only have 3 sample documents. With more documents (10+),
    you should switch to document-level splitting.

    For now, this gives us enough data in each split to train
    and evaluate the model.

    Args:
        spacy_data: List of (text, annotations) tuples
        train_ratio: Fraction for training (default 70%)
        val_ratio: Fraction for validation (default 15%)
        test_ratio: Fraction for testing (default 15%)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with 'train', 'dev', 'test' keys
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
        raise ValueError("Ratios must sum to 1.0")

    # Shuffle with fixed seed for reproducibility
    # This means you get the same split every time you run this
    random.seed(seed)
    shuffled = list(spacy_data)
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    splits = {
        "train": shuffled[:train_end],
        "dev": shuffled[train_end:val_end],
        "test": shuffled[val_end:]
    }

    logger.info("Dataset split:")
    for name, data in splits.items():
        with_entities = sum(1 for _, a in data if a["entities"])
        total_entities = sum(len(a["entities"]) for _, a in data)
        logger.info(
            f"    {name:<6}: {len(data):>3} sentences "
            f"({with_entities} with entities, "
            f"{total_entities} total entities)"
        )

    return splits


def save_spacy_binary(
    spacy_data: list[tuple],
    output_path: str,
    model_name: str = "en_core_web_sm"
) -> str:
    """
    Save spaCy-format data as a binary .spacy file.

    WHY BINARY FORMAT?
    spaCy's DocBin format is:
    - Faster to load during training
    - More compact than JSON
    - The official format spaCy expects

    HOW IT WORKS:
    1. Load a blank spaCy model (for tokenization)
    2. Create Doc objects from text + entity annotations
    3. Pack all Docs into a DocBin
    4. Serialize to disk

    Args:
        spacy_data: List of (text, annotations) tuples
        output_path: Where to save the .spacy file
        model_name: Base spaCy model for tokenization

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load spaCy model for tokenization
    # We only need the tokenizer — not the NER or other components
    nlp = spacy.load(model_name, disable=["ner", "parser", "tagger",
                                          "lemmatizer", "attribute_ruler"])

    doc_bin = DocBin()
    error_count = 0

    for text, annotations in spacy_data:
        try:
            # Create a Doc object from text
            doc = nlp.make_doc(text)

            # Get entity spans from annotations
            entities = annotations.get("entities", [])

            # Convert (start, end, label) character offsets to Span objects
            # spacy.util.filter_spans removes overlapping spans
            spans = []
            for start, end, label in entities:
                span = doc.char_span(start, end, label=label)
                if span is not None:
                    spans.append(span)
                else:
                    # char_span returns None if the span doesn't align
                    # with token boundaries. This happens when an entity
                    # starts or ends in the middle of a token.
                    logger.debug(
                        f"Span misalignment: ({start}, {end}, '{label}') "
                        f"in '{text[:50]}...'"
                    )
                    error_count += 1

            # Filter overlapping spans (keep the first one)
            filtered_spans = spacy.util.filter_spans(spans)
            doc.ents = filtered_spans

            doc_bin.add(doc)

        except Exception as e:
            logger.warning(f"Error processing sentence: {e}")
            error_count += 1

    # Save to disk
    doc_bin.to_disk(str(output_path))

    logger.info(
        f"Saved {len(spacy_data)} docs to {output_path} "
        f"({error_count} span alignment issues)"
    )
    return str(output_path)


def save_json_split(
    spacy_data: list[tuple],
    output_path: str
) -> str:
    """
    Save a data split as human-readable JSON.

    This is useful for inspecting the data manually and for
    debugging. The .spacy binary is what training uses, but
    JSON lets you read the data.

    Args:
        spacy_data: List of (text, annotations) tuples
        output_path: Where to save

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert tuples back to dictionaries for JSON serialization
    json_data = []
    for text, annotations in spacy_data:
        entities = []
        for start, end, label in annotations.get("entities", []):
            entities.append({
                "text": text[start:end],
                "start": start,
                "end": end,
                "label": label
            })

        json_data.append({
            "text": text,
            "entities": entities,
            "entity_count": len(entities)
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    logger.debug(f"Saved JSON: {output_path}")
    return str(output_path)


def compute_dataset_stats(splits: dict) -> dict:
    """
    Compute statistics about the dataset splits.

    Returns:
        Dictionary with statistics for each split
    """
    stats = {}

    for split_name, data in splits.items():
        total_sentences = len(data)
        sentences_with_entities = sum(
            1 for _, a in data if a["entities"]
        )
        total_entities = sum(len(a["entities"]) for _, a in data)

        # Count entities by label
        label_counts = Counter()
        for _, annotations in data:
            for start, end, label in annotations["entities"]:
                label_counts[label] += 1

        stats[split_name] = {
            "total_sentences": total_sentences,
            "sentences_with_entities": sentences_with_entities,
            "sentences_without_entities": total_sentences - sentences_with_entities,
            "total_entities": total_entities,
            "entity_labels": dict(label_counts)
        }

    return stats


def build_complete_dataset(
    annotated_data: list[dict],
    output_dir: str
) -> dict:
    """
    Build the complete NER dataset: convert, split, save.

    This is the main function that orchestrates the entire
    dataset building process.

    Args:
        annotated_data: Raw annotated data from ner_annotator
        output_dir: Where to save all dataset files

    Returns:
        Dataset statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building NER dataset...")

    # Step 1: Convert to spaCy format
    spacy_data = convert_to_spacy_format(annotated_data)

    if len(spacy_data) < 10:
        logger.warning(
            f"Very small dataset: only {len(spacy_data)} sentences. "
            f"NER model performance will be limited."
        )

    # Step 2: Split into train / dev / test
    splits = split_dataset(spacy_data)

    # Step 3: Save each split in both formats
    for split_name, split_data in splits.items():
        # Binary format (for spaCy training)
        save_spacy_binary(
            split_data,
            output_dir / f"{split_name}.spacy"
        )

        # JSON format (for human inspection)
        save_json_split(
            split_data,
            output_dir / f"{split_name}.json"
        )

    # Step 4: Compute and save statistics
    stats = compute_dataset_stats(splits)

    stats_path = output_dir / "annotation_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved dataset statistics to {stats_path}")

    return stats