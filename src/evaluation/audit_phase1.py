"""Read-only data audit plus shared rescoring of historical predictions."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from src.evaluation.spans import LABELS, calculate_span_metrics

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; use a new path to preserve audit history.")
    query_path = ROOT / "data/annotations/query_sample.csv"
    gold_path = ROOT / "data/annotations/chunk_annotations_reviewed.csv"
    intent_path = ROOT / "data/annotation_outputs/intent_annotations_annotated.csv"
    queries = read(query_path)
    query_map = {r["query_id"]: r["query"] for r in queries}
    gold = [r for r in read(gold_path) if r["review_status"].strip().lower() == "reviewed"]
    issues = []
    for row in gold:
        text = query_map.get(row["query_id"])
        start, end = int(row["start_char"]), int(row["end_char"])
        if text is None or not 0 <= start < end <= len(text) or row["chunk_label"] not in LABELS:
            issues.append(row)
    intents = read(intent_path)
    report = {
        "scope": "Development pilot audit, not independent held-out performance",
        "queries": len(query_map), "reviewed_chunk_queries": len({r['query_id'] for r in gold}),
        "reviewed_chunk_spans": len(gold), "invalid_gold": issues,
        "chunk_labels": dict(Counter(r["chunk_label"] for r in gold)),
        "intent_labels": dict(Counter(r["intent_label"] for r in intents)),
        "intent_annotators": dict(Counter(r["annotator"] for r in intents)),
        "intent_review_status": dict(Counter(r["review_status"] for r in intents)),
        "intent_provenance_warning": "Assistant pilot labels require independent human verification, regardless of reviewed status.",
        "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (query_path, gold_path, intent_path)},
        "historical_rescoring": {},
    }
    if issues:
        raise ValueError(f"Invalid gold spans: {issues}")
    ids = {r["query_id"] for r in gold}
    for name, relative in {
        "rule_day2": "experiments/day2/chunk_predictions.csv",
        "gemini_day4": "experiments/day4/gemini_chunk_predictions.csv",
        "gemini_day5": "experiments/day5/gemini_chunk_predictions.csv",
    }.items():
        predictions = [r for r in read(ROOT / relative) if r["query_id"] in ids]
        scores = calculate_span_metrics(gold, predictions)
        scores["per_label"] = {label: calculate_span_metrics(
            [r for r in gold if r["chunk_label"] == label],
            [r for r in predictions if r["chunk_label"] == label]) for label in LABELS}
        report["historical_rescoring"][name] = scores
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
