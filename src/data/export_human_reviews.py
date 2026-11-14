"""Export the latest explicit approval per query into a new training corpus."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from src.data.human_review import validate_review

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", type=Path, default=ROOT / "data/human_reviews")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    records = {}
    for path in sorted(args.reviews.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["review_status"] != "human_reviewed" or not row["complete_query_annotation"]:
            raise ValueError(f"Incomplete approval: {path}")
        if row["query_sha256"] != hashlib.sha256(row["query"].encode()).hexdigest():
            raise ValueError(f"Query changed after approval: {path}")
        validate_review(row["query"], row["chunks"], row["intent_label"], row["annotator"])
        qid = row["query_id"]
        if qid in records and row["reviewed_at_utc"] == records[qid]["reviewed_at_utc"]:
            raise ValueError(f"Two approvals share a timestamp for {qid}; adjudicate which record is current")
        if qid not in records or row["reviewed_at_utc"] > records[qid]["reviewed_at_utc"]:
            records[qid] = row
    if not records:
        parser.error("No explicit human approval records found")
    args.output.mkdir(parents=True)
    def write(name, fields, rows):
        with (args.output / name).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    rows = [records[k] for k in sorted(records)]
    write("queries.csv", ["query_id", "query"], rows)
    write("intents.csv", ["query_id", "query", "intent_label", "annotator", "review_status", "notes"],
          [dict(r, review_status="reviewed") for r in rows])
    write("chunks.csv", ["query_id", "chunk_label", "chunk_text", "start_char", "end_char", "review_status"],
          [dict(chunk, query_id=r["query_id"], review_status="reviewed") for r in rows for chunk in r["chunks"]])
    registry = {r["query_id"]: {"complete": True, "span_count": len(r["chunks"]),
                               "reviewed_at_utc": r["reviewed_at_utc"], "annotator": r["annotator"]} for r in rows}
    (args.output / "review_registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(f"Exported {len(rows)} human-reviewed queries to {args.output}")


if __name__ == "__main__":
    main()
