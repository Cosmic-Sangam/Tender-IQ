
import sys
import os
import random
import json

from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import spacy

from spacy.util import minibatch, compounding
from spacy.training import Example
from spacy.tokens import DocBin

from src.logger_config import get_logger
from src.ner_evaluator import (
    evaluate_model,
    save_evaluation_report,
    save_evaluation_json
)
from src.ner_predictor import TenderNERPredictor


logger = get_logger("run_day3")


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


def main():

    logger.info("TenderIQ — DAY 3: NER TRAINING")

    # ---------------------------------------------------------
    # Check Day 2 files
    # ---------------------------------------------------------

    for file_name in ["train.spacy", "dev.spacy", "test.spacy"]:

        file_path = Path("data/ner_dataset") / file_name

        if not file_path.exists():
            logger.error(
                f"Missing data/ner_dataset/{file_name}. "
                "Run Day 2 first."
            )
            return

    # ---------------------------------------------------------
    # Load base model
    # ---------------------------------------------------------

    logger.info("Loading en_core_web_sm...")

    nlp = spacy.load("en_core_web_sm")

    if "ner" in nlp.pipe_names:
        nlp.remove_pipe("ner")

    ner = nlp.add_pipe("ner", last=True)

    for label in ENTITY_LABELS:
        ner.add_label(label)

    # ---------------------------------------------------------
    # Load training data
    # ---------------------------------------------------------

    logger.info("Loading training data...")

    doc_bin = DocBin().from_disk(
        "data/ner_dataset/train.spacy"
    )

    train_docs = list(
        doc_bin.get_docs(nlp.vocab)
    )

    logger.info(
        f"Loaded {len(train_docs)} training docs"
    )

    # ---------------------------------------------------------
    # Create spaCy Examples
    # ---------------------------------------------------------

    examples = []

    for doc in train_docs:

        predicted_doc = nlp.make_doc(doc.text)

        example = Example(
            predicted_doc,
            doc
        )

        examples.append(example)

    # ---------------------------------------------------------
    # Train NER model
    # ---------------------------------------------------------

    logger.info(
        "Training NER model (30 iterations)..."
    )

    other_pipes = [
        pipe_name
        for pipe_name in nlp.pipe_names
        if pipe_name != "ner"
    ]

    losses_history = []

    with nlp.disable_pipes(*other_pipes):

        optimizer = nlp.initialize(
            lambda: examples
        )

        batch_sizes = compounding(
            4.0,
            32.0,
            1.001
        )

        for iteration in range(30):

            random.shuffle(examples)

            losses = {}

            for batch in minibatch(
                examples,
                size=batch_sizes
            ):

                nlp.update(
                    batch,
                    drop=0.35,
                    losses=losses,
                    sgd=optimizer
                )

            loss = losses.get("ner", 0.0)

            # Convert NumPy/spaCy numeric value
            # into normal Python float.
            loss = float(loss)

            losses_history.append(loss)

            if (
                (iteration + 1) % 5 == 0
                or iteration == 0
            ):

                logger.info(
                    f"  Iteration {iteration + 1:>2}/30  "
                    f"Loss: {loss:.4f}"
                )

    final_loss = float(losses_history[-1])

    logger.info(
        f"Training done. Final loss: {final_loss:.4f}"
    )

    # ---------------------------------------------------------
    # Save trained model
    # ---------------------------------------------------------

    model_dir = "models/tender_ner_v1"

    Path(model_dir).mkdir(
        parents=True,
        exist_ok=True
    )

    nlp.to_disk(model_dir)

    logger.info(
        f"Model saved to {model_dir}"
    )

    # ---------------------------------------------------------
    # Save training metadata
    # ---------------------------------------------------------

    meta = {
        "n_iterations": 30,
        "final_loss": final_loss,
        "entity_labels": ENTITY_LABELS,
        "n_training_examples": len(examples)
    }

    metadata_path = Path(model_dir) / "training_metadata.json"

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            meta,
            f,
            indent=2
        )

    logger.info(
        f"Training metadata saved to {metadata_path}"
    )

    # ---------------------------------------------------------
    # Evaluate model
    # ---------------------------------------------------------

    logger.info(
        "Evaluating on test set..."
    )

    results = evaluate_model(
        model_dir,
        "data/ner_dataset/test.spacy"
    )

    save_evaluation_report(results)
    save_evaluation_json(results)

    logger.info(
        "Evaluation complete."
    )

    # ---------------------------------------------------------
    # Live prediction test
    # ---------------------------------------------------------

    logger.info(
        "Live predictions:"
    )

    predictor = TenderNERPredictor(
        model_dir
    )

    test_texts = [
        "The bidder must have ISO 9001:2015 and CMMI Level 3.",
        "Minimum turnover Rs. 100 Crore required.",
        "Deadline: 28th February 2024, 4:00 PM.",
        "Penalty of Rs. 50,000 per hour of downtime."
    ]

    for text in test_texts:

        ents = predictor.predict(text)

        logger.info(
            f"  Text: {text}"
        )

        for entity in ents:

            logger.info(
                f"    [{entity['label']}] "
                f"'{entity['text']}'"
            )

        if not ents:

            logger.info(
                "    (no entities)"
            )

    # ---------------------------------------------------------
    # Finished
    # ---------------------------------------------------------

    logger.info(
        "Day 3 Complete!"
    )


if __name__ == "__main__":
    main()