"""Validation and explicit human approval records, separate from pilot data."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.evaluation.spans import LABELS

INTENTS = ("PRODUCT_SEARCH", "ATTRIBUTE_SEARCH", "USE_CASE_SEARCH", "OCCASION_SEARCH", "CONTEXT_CONSTRAINT_SEARCH")


def validate_review(query, chunks, intent, annotator):
    if not annotator.strip() or "assistant" in annotator.lower():
        raise ValueError("Enter the human reviewer's name or identifier")
    if intent not in INTENTS:
        raise ValueError("Choose a permitted primary intent")
    clean = []
    for row in chunks:
        start, end = row.get("start_char"), row.get("end_char")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(query):
            raise ValueError(f"Invalid offsets in {row}")
        if row.get("chunk_label") not in LABELS:
            raise ValueError("Choose a permitted chunk label")
        clean.append({"start_char": start, "end_char": end,
                      "chunk_label": row["chunk_label"], "chunk_text": query[start:end]})
    clean.sort(key=lambda r: r["start_char"])
    if any(a["end_char"] > b["start_char"] for a, b in zip(clean, clean[1:])):
        raise ValueError("Overlapping spans: resolve them before approving flat BIO gold")
    return clean


def save_review(directory, query_id, query, chunks, intent, annotator, notes):
    clean = validate_review(query, chunks, intent, annotator)
    stamp = datetime.now(timezone.utc)
    record = {"query_id": str(query_id), "query": query, "chunks": clean,
              "intent_label": intent, "annotator": annotator.strip(),
              "review_status": "human_reviewed", "reviewed_at_utc": stamp.isoformat(),
              "guideline_version": "0.2", "notes": notes,
              "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
              "complete_query_annotation": True}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    safe_id = hashlib.sha256(str(query_id).encode()).hexdigest()[:16]
    target = directory / f"{safe_id}_{stamp.strftime('%Y%m%dT%H%M%S%f')}_{uuid4().hex[:12]}.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False)
    return target
