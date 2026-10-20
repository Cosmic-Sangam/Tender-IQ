import sys, os, json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spacy
from spacy.tokens import DocBin
from src.logger_config import get_logger

logger = get_logger(__name__)


def compute_metrics(tp=0, fp=0, fn=0):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn
    }


def evaluate_model(model_dir, test_spacy_file):
    logger.info("Loading model...")
    nlp = spacy.load(model_dir)
    logger.info("Loading test data...")
    doc_bin = DocBin().from_disk(test_spacy_file)
    test_docs = list(doc_bin.get_docs(nlp.vocab))
    if len(test_docs) == 0:
        return {}
    logger.info(f"Test set: {len(test_docs)} sentences")

    tp_label = defaultdict(int)
    fp_label = defaultdict(int)
    fn_label = defaultdict(int)
    errors = {"false_positives": [], "false_negatives": [], "correct": []}

    for gold_doc in test_docs:
        pred_doc = nlp(gold_doc.text)
        gold_ents = {(e.start_char, e.end_char, e.label_) for e in gold_doc.ents}
        pred_ents = {(e.start_char, e.end_char, e.label_) for e in pred_doc.ents}

        for _, _, label in gold_ents & pred_ents:
            tp_label[label] += 1
        for _, _, label in pred_ents - gold_ents:
            fp_label[label] += 1
        for _, _, label in gold_ents - pred_ents:
            fn_label[label] += 1

        for s, e, l in list(gold_ents & pred_ents)[:5]:
            errors["correct"].append({"text": gold_doc.text[s:e], "label": l})
        for s, e, l in list(pred_ents - gold_ents)[:5]:
            errors["false_positives"].append({"text": pred_doc.text[s:e], "label": l})
        for s, e, l in list(gold_ents - pred_ents)[:5]:
            errors["false_negatives"].append({"text": gold_doc.text[s:e], "label": l})

    total_tp = sum(tp_label.values())
    total_fp = sum(fp_label.values())
    total_fn = sum(fn_label.values())
    overall = compute_metrics(tp=total_tp, fp=total_fp, fn=total_fn)

    all_labels = set(tp_label) | set(fp_label) | set(fn_label)
    per_label = {}
    for label in sorted(all_labels):
        per_label[label] = compute_metrics(tp=tp_label[label], fp=fp_label[label], fn=fn_label[label])

    logger.info(f"Precision: {overall['precision']:.4f}  Recall: {overall['recall']:.4f}  F1: {overall['f1']:.4f}")
    for label in sorted(per_label):
        m = per_label[label]
        logger.info(f"  {label:<28} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")

    return {"overall": overall, "per_label": per_label, "test_size": len(test_docs), "error_examples": errors}


def save_evaluation_report(results, output_path="docs/model_evaluation.md"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    lines = ["# NER Evaluation Report", "", f"*{datetime.now()}*", "",
             "## Overall", "",
             f"- Precision: {results['overall']['precision']:.4f}",
             f"- Recall: {results['overall']['recall']:.4f}",
             f"- F1: {results['overall']['f1']:.4f}", "",
             "## Per-Label", "", "| Label | P | R | F1 |", "|---|---|---|---|"]
    for label in sorted(results["per_label"]):
        m = results["per_label"][label]
        lines.append(f"| {label} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Report saved: {output_path}")


def save_evaluation_json(results, output_path="models/tender_ner_v1/evaluation.json"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"JSON saved: {output_path}")