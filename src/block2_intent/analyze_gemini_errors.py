

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ANNOTATION_DIR = ROOT / "data" / "annotations"
EXPERIMENT_DIR = ROOT / "experiments" / "day4"

QUERY_FILE = ANNOTATION_DIR / "query_sample.csv"
GOLD_FILE = ANNOTATION_DIR / "chunk_annotations_reviewed.csv"
PREDICTION_FILE = EXPERIMENT_DIR / "gemini_chunk_predictions.csv"

ERROR_REPORT_FILE = EXPERIMENT_DIR / "gemini_error_analysis.csv"
SUMMARY_FILE = EXPERIMENT_DIR / "gemini_error_summary.json"
QUERY_AUDIT_FILE = EXPERIMENT_DIR / "gemini_query_audit.csv"


def normalize_id(value):
    """Normalize IDs so CSV values like 10039 and 10039.0 can match."""
    if pd.isna(value):
        return ""
    value = str(value).strip()
    return value[:-2] if value.endswith(".0") else value


def load_inputs():
    """Load input files and validate their required columns."""
    for path in (QUERY_FILE, GOLD_FILE, PREDICTION_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")

    queries = pd.read_csv(QUERY_FILE, dtype={"query_id": str})
    gold = pd.read_csv(GOLD_FILE, dtype={"query_id": str})
    predictions = pd.read_csv(PREDICTION_FILE, dtype={"query_id": str})

    if not {"query_id", "query"}.issubset(queries.columns):
        raise ValueError("query_sample.csv must contain query_id and query.")

    required_gold = {
        "query_id", "chunk_label", "start_char", "end_char", "review_status"
    }
    if not required_gold.issubset(gold.columns):
        raise ValueError(
            f"Gold file missing columns: {sorted(required_gold - set(gold.columns))}"
        )

    required_predictions = {
        "query_id", "chunk_text", "chunk_label", "start_char", "end_char"
    }
    if not required_predictions.issubset(predictions.columns):
        raise ValueError(
            "Prediction file missing columns: "
            f"{sorted(required_predictions - set(predictions.columns))}"
        )

    for frame in (queries, gold, predictions):
        frame["query_id"] = frame["query_id"].map(normalize_id)

    gold = gold[
        gold["review_status"].astype(str).str.strip().str.lower().eq("reviewed")
    ].copy()

    queries = queries.drop_duplicates("query_id", keep="first")
    query_map = dict(zip(queries["query_id"], queries["query"].astype(str)))

    unknown_gold = sorted(set(gold["query_id"]) - set(query_map))
    unknown_predictions = sorted(set(predictions["query_id"]) - set(query_map))
    if unknown_gold:
        raise ValueError(
            f"Gold IDs not found in query_sample.csv: {unknown_gold[:10]}"
        )
    if unknown_predictions:
        raise ValueError(
            f"Prediction IDs not found in query_sample.csv: {unknown_predictions[:10]}"
        )

    reviewed_ids = set(gold["query_id"])
    predictions = predictions[predictions["query_id"].isin(reviewed_ids)].copy()
    evaluated_ids = set(predictions["query_id"])
    gold = gold[gold["query_id"].isin(evaluated_ids)].copy()

    return queries, gold, predictions, query_map


def clean_span_rows(frame, source_name, query_map, require_text=False):
    """Convert spans to standard records and collect invalid-offset warnings."""
    records = []
    invalid = []

    for row_number, row in enumerate(frame.to_dict("records"), start=2):
        query_id = normalize_id(row.get("query_id"))
        query = query_map.get(query_id, "")
        label = str(row.get("chunk_label", "")).strip()

        try:
            if pd.isna(row.get("start_char")) or pd.isna(row.get("end_char")):
                raise ValueError("missing offset")
            start = int(float(row["start_char"]))
            end = int(float(row["end_char"]))
        except (TypeError, ValueError, OverflowError):
            invalid.append({
                "source": source_name,
                "row_number": row_number,
                "query_id": query_id,
                "problem": "Missing or non-numeric offsets",
                "chunk_text": str(row.get("chunk_text", "")),
            })
            continue

        if start < 0 or end <= start or end > len(query):
            invalid.append({
                "source": source_name,
                "row_number": row_number,
                "query_id": query_id,
                "problem": f"Invalid offsets [{start}:{end}] for query length {len(query)}",
                "chunk_text": str(row.get("chunk_text", "")),
            })
            continue

        actual_text = query[start:end]
        supplied_text = str(row.get("chunk_text", "")) if require_text else actual_text
        if require_text and supplied_text != actual_text:
            invalid.append({
                "source": source_name,
                "row_number": row_number,
                "query_id": query_id,
                "problem": f"chunk_text differs from query substring [{start}:{end}]",
                "chunk_text": supplied_text,
            })

        records.append({
            "query_id": query_id,
            "query": query,
            "start_char": start,
            "end_char": end,
            "chunk_label": label,
            "chunk_text": actual_text,
        })

    return records, invalid


def span_key(row):
    return (
        row["query_id"], row["start_char"],
        row["end_char"], row["chunk_label"]
    )


def overlaps(a, b):
    """Whether two half-open spans share at least one character."""
    return max(a["start_char"], b["start_char"]) < min(
        a["end_char"], b["end_char"]
    )


def contained(inner, outer):
    return (
        outer["start_char"] <= inner["start_char"]
        and inner["end_char"] <= outer["end_char"]
    )


def main():
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    _, gold_df, pred_df, query_map = load_inputs()

    gold_rows, invalid_gold = clean_span_rows(
        gold_df, "gold", query_map, require_text=False
    )
    pred_rows, invalid_predictions = clean_span_rows(
        pred_df, "prediction", query_map, require_text=True
    )

    gold_by_query = {}
    pred_by_query = {}
    for row in gold_rows:
        gold_by_query.setdefault(row["query_id"], []).append(row)
    for row in pred_rows:
        pred_by_query.setdefault(row["query_id"], []).append(row)

    gold_keys = [span_key(row) for row in gold_rows]
    pred_keys = [span_key(row) for row in pred_rows]
    gold_set, pred_set = set(gold_keys), set(pred_keys)
    duplicate_gold_rows = len(gold_keys) - len(gold_set)
    duplicate_prediction_rows = len(pred_keys) - len(pred_set)

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    report = []
    matched_gold = set()
    matched_pred = set()

    def add_report(error_type, g=None, p=None, notes=""):
        base = g if g is not None else p
        query_id = base["query_id"]
        query = base["query"]
        report.append({
            "query_id": query_id,
            "query": query,
            "error_type": error_type,
            "gold_text": g["chunk_text"] if g else "",
            "gold_label": g["chunk_label"] if g else "",
            "gold_start": g["start_char"] if g else None,
            "gold_end": g["end_char"] if g else None,
            "predicted_text": p["chunk_text"] if p else "",
            "predicted_label": p["chunk_label"] if p else "",
            "predicted_start": p["start_char"] if p else None,
            "predicted_end": p["end_char"] if p else None,
            "notes": notes,
        })

    gold_unmatched = [r for r in gold_rows if span_key(r) not in pred_set]
    pred_unmatched = [r for r in pred_rows if span_key(r) not in gold_set]

    for key in sorted(gold_set & pred_set):
        qid, start, end, label = key
        row = {
            "query_id": qid, "query": query_map[qid],
            "start_char": start, "end_char": end, "chunk_label": label,
            "chunk_text": query_map[qid][start:end],
        }
        add_report("EXACT_MATCH", row, row, "Exact query ID, offsets and label match.")
        matched_gold.add(key)
        matched_pred.add(key)

    # Same boundaries but different labels.
    for g in gold_unmatched:
        gkey = span_key(g)
        if gkey in matched_gold:
            continue
        candidates = [
            p for p in pred_unmatched
            if span_key(p) not in matched_pred
            and p["query_id"] == g["query_id"]
            and p["start_char"] == g["start_char"]
            and p["end_char"] == g["end_char"]
            and p["chunk_label"] != g["chunk_label"]
        ]
        if candidates:
            p = candidates[0]
            add_report("LABEL_MISMATCH", g, p, "Same boundaries, different labels.")
            matched_gold.add(gkey)
            matched_pred.add(span_key(p))

    # Likely over-segmentation: multiple predictions fully inside one gold span.
    for g in gold_unmatched:
        gkey = span_key(g)
        if gkey in matched_gold:
            continue
        candidates = [
            p for p in pred_unmatched
            if span_key(p) not in matched_pred
            and p["query_id"] == g["query_id"] and contained(p, g)
        ]
        if len(candidates) >= 2:
            p_text = " | ".join(p["chunk_text"] for p in candidates)
            p_labels = " | ".join(p["chunk_label"] for p in candidates)
            add_report(
                "LIKELY_OVER_SEGMENTATION", g, None,
                f"Predictions inside gold span: {p_text}; labels: {p_labels}. "
                "Heuristic; manually review."
            )
            matched_gold.add(gkey)
            matched_pred.update(span_key(p) for p in candidates)

    # Likely under-segmentation: one prediction fully contains multiple gold spans.
    for p in pred_unmatched:
        pkey = span_key(p)
        if pkey in matched_pred:
            continue
        candidates = [
            g for g in gold_unmatched
            if span_key(g) not in matched_gold
            and g["query_id"] == p["query_id"] and contained(g, p)
        ]
        if len(candidates) >= 2:
            g_text = " | ".join(g["chunk_text"] for g in candidates)
            g_labels = " | ".join(g["chunk_label"] for g in candidates)
            add_report(
                "LIKELY_UNDER_SEGMENTATION", None, p,
                f"Gold spans inside prediction: {g_text}; labels: {g_labels}. "
                "Heuristic; manually review."
            )
            matched_pred.add(pkey)
            matched_gold.update(span_key(g) for g in candidates)

    # Remaining overlapping spans: boundary mismatch or overlap plus label mismatch.
    for g in gold_unmatched:
        gkey = span_key(g)
        if gkey in matched_gold:
            continue
        candidates = [
            p for p in pred_unmatched
            if span_key(p) not in matched_pred
            and p["query_id"] == g["query_id"] and overlaps(p, g)
        ]
        if candidates:
            p = max(
                candidates,
                key=lambda x: (
                    x["chunk_label"] == g["chunk_label"],
                    min(x["end_char"], g["end_char"])
                    - max(x["start_char"], g["start_char"]),
                ),
            )
            same_label = p["chunk_label"] == g["chunk_label"]
            add_report(
                "BOUNDARY_MISMATCH" if same_label else "OVERLAP_AND_LABEL_MISMATCH",
                g, p,
                "Overlapping non-exact spans; inspect boundaries and labels manually.",
            )
            matched_gold.add(gkey)
            matched_pred.add(span_key(p))

    for g in gold_unmatched:
        if span_key(g) not in matched_gold:
            add_report(
                "MISSED_GOLD_SPAN", g, None,
                "No exact or overlapping unmatched prediction was paired.",
            )

    for p in pred_unmatched:
        if span_key(p) not in matched_pred:
            add_report(
                "EXTRA_PREDICTION", None, p,
                "No exact or overlapping unmatched gold span was paired.",
            )

    report_columns = [
        "query_id", "query", "error_type", "gold_text", "gold_label",
        "gold_start", "gold_end", "predicted_text", "predicted_label",
        "predicted_start", "predicted_end", "notes",
    ]
    report_df = pd.DataFrame(report, columns=report_columns)
    if not report_df.empty:
        report_df = report_df.sort_values(
            ["query_id", "gold_start", "predicted_start", "error_type"],
            na_position="last",
        )
    report_df.to_csv(ERROR_REPORT_FILE, index=False)

    all_labels = sorted(
        {r["chunk_label"] for r in gold_rows}
        | {r["chunk_label"] for r in pred_rows}
    )
    per_label = {}
    for label in all_labels:
        gset = {span_key(r) for r in gold_rows if r["chunk_label"] == label}
        pset = {span_key(r) for r in pred_rows if r["chunk_label"] == label}
        ltp, lfp, lfn = len(gset & pset), len(pset - gset), len(gset - pset)
        lp = ltp / (ltp + lfp) if ltp + lfp else 0.0
        lr = ltp / (ltp + lfn) if ltp + lfn else 0.0
        lf1 = 2 * lp * lr / (lp + lr) if lp + lr else 0.0
        per_label[label] = {
            "gold_spans": len(gset), "predicted_spans": len(pset),
            "true_positive": ltp, "false_positive": lfp, "false_negative": lfn,
            "precision": round(lp, 6), "recall": round(lr, 6), "f1": round(lf1, 6),
        }

    query_audit = []
    for qid in sorted(set(gold_by_query) | set(pred_by_query)):
        gset = {span_key(r) for r in gold_by_query.get(qid, [])}
        pset = {span_key(r) for r in pred_by_query.get(qid, [])}
        query_audit.append({
            "query_id": qid, "query": query_map.get(qid, ""),
            "gold_spans": len(gset), "predicted_spans": len(pset),
            "exact_matches": len(gset & pset),
            "false_positives": len(pset - gset),
            "false_negatives": len(gset - pset),
        })
    pd.DataFrame(query_audit).to_csv(QUERY_AUDIT_FILE, index=False)

    summary = {
        "model": "gemini-3.1-flash-lite",
        "queries_evaluated": len(query_audit),
        "gold_spans_unique": len(gold_set),
        "predicted_spans_unique": len(pred_set),
        "true_positive": tp, "false_positive": fp, "false_negative": fn,
        "precision": round(precision, 6), "recall": round(recall, 6),
        "f1": round(f1, 6),
        "duplicate_gold_rows": duplicate_gold_rows,
        "duplicate_prediction_rows": duplicate_prediction_rows,
        "invalid_gold_rows": invalid_gold,
        "invalid_prediction_rows": invalid_predictions,
        "per_label": per_label,
        "error_report_rows": len(report_df),
        "error_type_counts": (
            report_df["error_type"].value_counts().to_dict()
            if not report_df.empty else {}
        ),
        "note": (
            "Error types other than exact matches are diagnostic heuristics. "
            "Review examples manually before making research claims."
        ),
    }
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Gemini error analysis complete ===")
    print(f"Queries evaluated: {len(query_audit)}")
    print(f"Unique gold/predicted spans: {len(gold_set)}/{len(pred_set)}")
    print(f"TP={tp}, FP={fp}, FN={fn}")
    print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    print("\nError type counts:")
    print(json.dumps(summary["error_type_counts"], indent=2))
    print("\nFiles saved:")
    print(f"  Error report: {ERROR_REPORT_FILE}")
    print(f"  Summary:      {SUMMARY_FILE}")
    print(f"  Query audit:  {QUERY_AUDIT_FILE}")

    if invalid_gold or invalid_predictions:
        print("\nWARNING: Invalid rows were found; see the summary JSON.")
    if duplicate_gold_rows or duplicate_prediction_rows:
        print("\nWARNING: Duplicate span keys were found; metrics use unique spans.")


if __name__ == "__main__":
    main()
