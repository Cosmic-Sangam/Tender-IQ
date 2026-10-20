"""
TenderIQ — NER Model Trainer
==============================

WHAT THIS DOES:
    Trains a custom Named Entity Recognition (NER) model using
    spaCy on our annotated tender data.

HOW SPACY TRAINING WORKS:
    1. Start with a base model (en_core_web_sm)
    2. Keep only the tokenizer, remove the existing NER
    3. Add a new blank NER component
    4. Register our custom entity labels
    5. Iterate through training data multiple times (epochs)
    6. For each batch, adjust model weights to reduce errors
    7. Track loss (lower = better)
    8. Save the trained model to disk

WHY START FROM en_core_web_sm?
    Transfer learning. The base model already knows English
    grammar, tokenization, and word representations. We just
    teach it OUR entity types on top. Much faster and better
    than training from scratch.

USAGE:
    from src.ner_trainer import train_ner_model
    train_ner_model("data/ner_dataset", "models/tender_ner_v1")
"""

import sys
import os
import random
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spacy
from spacy.util import minibatch, compounding
from spacy.training import Example
from spacy.tokens import DocBin

from src.logger_config import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════
# ENTITY LABELS
# ═══════════════════════════════════════════════════════
# Must match the labels used in ner_annotator.py

ENTITY_LABELS = [
    "CERTIFICATION",
    "FINANCIAL_REQUIREMENT",
    "DEADLINE",
    "TECHNICAL_REQUIREMENT",
    "EXPERIENCE",
    "PENALTY",
    "DOCUMENT",
    "ELIGIBILITY"
]


def load_training_data(spacy_file: str, base_model: str = "en_core_web_sm") -> list:
    """
    Load training data from a .spacy binary file.

    HOW IT WORKS:
    1. Load a blank spaCy pipeline (for tokenization)
    2. Load the DocBin from disk
    3. Convert DocBin to Doc objects
    4. Return the Docs

    Args:
        spacy_file: Path to .spacy binary file
        base_model: Base spaCy model name

    Returns:
        List of spaCy Doc objects
    """
    logger.info(f"Loading training data from {spacy_file}")

    # Load the base model just to get the vocabulary
    nlp = spacy.load(base_model)

    # Load the binary training data
    doc_bin = DocBin().from_disk(spacy_file)
    docs = list(doc_bin.get_docs(nlp.vocab))

    logger.info(f"Loaded {len(docs)} documents")

    # Count entities per label
    label_counts = {}
    for doc in docs:
        for ent in doc.ents:
            label_counts[ent.label_] = label_counts.get(ent.label_, 0) + 1

    logger.info("Entity counts in this data:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        logger.info(f"    {label:<28} {count:>3}")

    return docs


def create_training_examples(docs: list, nlp) -> list:
    """
    Convert Doc objects into training Examples.

    spaCy's Example class holds:
    - reference: the gold-standard Doc (correct entities)
    - predicted: what the model currently predicts

    During training, spaCy compares predicted vs reference and
    adjusts weights to reduce the difference.

    Args:
        docs: List of gold-annotated Doc objects
        nlp: The spaCy pipeline being trained

    Returns:
        List of Example objects
    """
    examples = []

    for doc in docs:
        # Create a fresh prediction Doc from the same text
        # This will be what the model predicts on
        predicted = nlp.make_doc(doc.text)

        # Example combines the model's prediction with the gold standard
        example = Example(predicted, doc)
        examples.append(example)

    return examples


def train_ner_model(
    dataset_dir: str = "data/ner_dataset",
    output_model_dir: str = "models/tender_ner_v1",
    base_model: str = "en_core_web_sm",
    n_iter: int = 50,
    dropout: float = 0.35,
    batch_size_start: float = 4.0,
    batch_size_max: float = 32.0,
    seed: int = 42
) -> dict:
    """
    Train a custom NER model on tender data.

    Args:
        dataset_dir: Directory with train.spacy, dev.spacy
        output_model_dir: Where to save the trained model
        base_model: Base spaCy model for transfer learning
        n_iter: Number of training iterations (epochs)
        dropout: Dropout rate (0.0-1.0). Higher = more regularization.
                 Prevents overfitting on small datasets.
        batch_size_start: Initial batch size
        batch_size_max: Maximum batch size (grows over iterations)
        seed: Random seed for reproducibility

    Returns:
        Training statistics dictionary
    """
    logger.info("=" * 60)
    logger.info("  STARTING NER MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"Base model      : {base_model}")
    logger.info(f"Iterations      : {n_iter}")
    logger.info(f"Dropout         : {dropout}")
    logger.info(f"Output directory: {output_model_dir}")

    # Set random seeds for reproducibility
    random.seed(seed)
    spacy.util.fix_random_seed(seed)

    # ─────────────────────────────────────────
    # STEP 1: Load base model and configure
    # ─────────────────────────────────────────
    logger.info("Step 1: Loading base model...")
    nlp = spacy.load(base_model)

    # Remove existing NER (we're training our own)
    if "ner" in nlp.pipe_names:
        nlp.remove_pipe("ner")
        logger.debug("Removed existing NER component")

    # Add a fresh NER component
    ner = nlp.add_pipe("ner", last=True)
    logger.debug("Added fresh NER component")

    # Register all our custom labels with the NER component
    for label in ENTITY_LABELS:
        ner.add_label(label)
    logger.info(f"Registered {len(ENTITY_LABELS)} entity labels")

    # ─────────────────────────────────────────
    # STEP 2: Load training data
    # ─────────────────────────────────────────
    logger.info("Step 2: Loading training data...")
    train_docs = load_training_data(
        f"{dataset_dir}/train.spacy",
        base_model=base_model
    )

    if len(train_docs) < 5:
        logger.error(
            f"Too few training examples ({len(train_docs)}). "
            f"Need at least 5. Cannot train."
        )
        raise ValueError("Insufficient training data")

    # ─────────────────────────────────────────
    # STEP 3: Convert to Examples
    # ─────────────────────────────────────────
    train_examples = create_training_examples(train_docs, nlp)
    logger.info(f"Created {len(train_examples)} training examples")

    # ─────────────────────────────────────────
    # STEP 4: Initialize the NER component
    # ─────────────────────────────────────────
    # This creates the neural network weights based on our labels
    logger.info("Step 3: Initializing NER neural network...")

    # Disable other pipes so only NER gets trained
    # This isolates NER training from other components
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    with nlp.disable_pipes(*other_pipes):
        # Initialize with our training examples
        # This sets up the network with correct output labels
        optimizer = nlp.initialize(lambda: train_examples)
        logger.debug("Initialization complete")

        # ─────────────────────────────────────
        # STEP 5: Training loop
        # ─────────────────────────────────────
        logger.info("Step 4: Training loop starting...")
        logger.info("")
        logger.info(f"{'Iteration':<12} {'Loss':<12} {'Progress'}")
        logger.info("-" * 40)

        training_stats = {
            "iterations": [],
            "losses": [],
            "start_time": datetime.now().isoformat()
        }

        # Batch size grows over iterations (compounding)
        # Start small (learn slowly, precisely)
        # Grow larger (learn faster once basics are down)
        batch_sizes = compounding(
            batch_size_start,
            batch_size_max,
            1.001
        )

        for iteration in range(n_iter):
            # Shuffle data each iteration
            # This prevents the model from memorizing order
            random.shuffle(train_examples)

            losses = {}
            batches = minibatch(train_examples, size=batch_sizes)

            for batch in batches:
                # Update model weights based on this batch
                # drop=dropout adds regularization to prevent overfitting
                nlp.update(
                    batch,
                    drop=dropout,
                    losses=losses,
                    sgd=optimizer
                )

            ner_loss = losses.get("ner", 0.0)

            # Progress indicator
            progress = "█" * int((iteration + 1) / n_iter * 20)
            progress = progress.ljust(20, "░")

            logger.info(
                f"{iteration + 1:<12} {ner_loss:<12.4f} [{progress}]"
            )

            training_stats["iterations"].append(iteration + 1)
            training_stats["losses"].append(ner_loss)

        training_stats["end_time"] = datetime.now().isoformat()
        training_stats["final_loss"] = training_stats["losses"][-1]

    logger.info("-" * 40)
    logger.info(f"Training complete. Final loss: {training_stats['final_loss']:.4f}")

    # ─────────────────────────────────────────
    # STEP 6: Save the trained model
    # ─────────────────────────────────────────
    logger.info(f"Step 5: Saving model to {output_model_dir}")

    output_path = Path(output_model_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    nlp.to_disk(output_path)

    # Save training metadata alongside model
    metadata = {
        "model_name": output_path.name,
        "base_model": base_model,
        "trained_on": datetime.now().isoformat(),
        "n_iterations": n_iter,
        "dropout": dropout,
        "seed": seed,
        "n_training_examples": len(train_examples),
        "entity_labels": ENTITY_LABELS,
        "final_loss": training_stats["final_loss"],
        "loss_history": training_stats["losses"]
    }

    metadata_path = output_path / "training_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Model saved to: {output_path}")
    logger.info(f"Training metadata: {metadata_path}")

    return training_stats