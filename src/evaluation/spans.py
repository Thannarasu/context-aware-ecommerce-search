"""Shared exact character-span evaluation; no model or API imports."""
import math

LABELS = ("PRODUCT", "ATTRIBUTE", "PURPOSE", "OCCASION", "CONTEXT", "CONSTRAINT")


def assign_predicted_spans(query, chunks):
    """Resolve text-only chunks left to right, without access to annotations.

    Explicit offsets are accepted only when they reproduce the exact text.
    Repeated predictions beyond the available occurrences remain invalid.
    """
    used = set()
    output = []
    for chunk in chunks:
        text, label = chunk.get("text", ""), chunk.get("label", "")
        start, end = chunk.get("start_char"), chunk.get("end_char")
        explicit = start is not None or end is not None
        valid = (type(start) is int and type(end) is int
                 and 0 <= start < end <= len(query) and query[start:end] == text)
        if not explicit and text:
            offset = 0
            while True:
                position = query.find(text, offset)
                if position < 0:
                    break
                candidate = (position, position + len(text))
                if candidate not in used:
                    start, end = candidate
                    valid = True
                    break
                offset = position + 1
        valid = valid and (start, end) not in used and label in LABELS
        if valid:
            used.add((start, end))
        output.append({"chunk_text": text, "chunk_label": label,
                       "start_char": start if valid else None,
                       "end_char": end if valid else None, "span_found": valid})
    return output


def _records(rows):
    return rows.to_dict("records") if hasattr(rows, "to_dict") else list(rows)


def calculate_span_metrics(gold, predictions):
    """Deduplicate valid spans; count unalignable outputs as false positives.

    All gold rows supplied are scored, even for queries with no predictions.
    Invalid gold offsets are an error, rather than silently discarded.
    """
    def key(row):
        start, end = float(row["start_char"]), float(row["end_char"])
        if not (math.isfinite(start) and math.isfinite(end)
                and start.is_integer() and end.is_integer() and 0 <= start < end):
            raise ValueError("Invalid character offsets")
        if row["chunk_label"] not in LABELS:
            raise ValueError("Unknown chunk label")
        return str(row["query_id"]), int(start), int(end), row["chunk_label"]

    gold_set = {key(row) for row in _records(gold)}
    pred_set, invalid = set(), 0
    for row in _records(predictions):
        try:
            pred_set.add(key(row))
        except (ValueError, TypeError, KeyError, OverflowError):
            invalid += 1
    tp = len(gold_set & pred_set)
    fp, fn = len(pred_set - gold_set) + invalid, len(gold_set - pred_set)
    return {"gold_spans": len(gold_set), "predicted_spans": len(pred_set) + invalid,
            "invalid_predictions": invalid, "true_positive": tp,
            "false_positive": fp, "false_negative": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
