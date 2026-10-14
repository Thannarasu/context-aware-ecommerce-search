
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[2]

GOLD_FILE = ROOT / "data" / "annotations" / "chunk_annotations_reviewed.csv"
PRED_FILE = ROOT / "experiments" / "day2" / "chunk_predictions.csv"


def main():
    gold = pd.read_csv(GOLD_FILE)
    pred = pd.read_csv(PRED_FILE)

    required = {"query_id", "chunk_label", "start_char", "end_char"}
    for name, df in [("gold", gold), ("predictions", pred)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    if "review_status" in gold.columns:
        gold = gold[gold["review_status"].eq("reviewed")].copy()

    # Only score queries with reviewed gold annotations.
    reviewed_ids = set(gold["query_id"].unique())
    pred = pred[pred["query_id"].isin(reviewed_ids)].copy()

    if gold.empty:
        raise ValueError(
            "No reviewed gold spans found. Review annotations first "
            "and save them to chunk_annotations_reviewed.csv."
        )

    key_cols = ["query_id", "start_char", "end_char", "chunk_label"]

    gold_keys = set(map(tuple, gold[key_cols].itertuples(index=False, name=None)))
    pred_keys = set(map(tuple, pred[key_cols].itertuples(index=False, name=None)))

    true_positive = len(gold_keys & pred_keys)
    false_positive = len(pred_keys - gold_keys)
    false_negative = len(gold_keys - pred_keys)

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    print(f"Reviewed queries represented in gold: {len(reviewed_ids)}")
    print(f"Gold spans: {len(gold_keys)}")
    print(f"Predicted spans on reviewed queries: {len(pred_keys)}")
    print(f"Exact-match TP: {true_positive}")
    print(f"Exact-match FP: {false_positive}")
    print(f"Exact-match FN: {false_negative}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")

    # Per-label diagnostic: exact span and label must both match.
    labels = sorted(set(gold["chunk_label"]) | set(pred["chunk_label"]))
    rows = []
    for label in labels:
        g = gold[gold["chunk_label"].eq(label)]
        p = pred[pred["chunk_label"].eq(label)]

        g_set = set(map(tuple, g[key_cols].itertuples(index=False, name=None)))
        p_set = set(map(tuple, p[key_cols].itertuples(index=False, name=None)))

        tp = len(g_set & p_set)
        fp = len(p_set - g_set)
        fn = len(g_set - p_set)

        rows.append({
            "chunk_label": label,
            "gold_count": len(g_set),
            "predicted_count": len(p_set),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        })

    report = pd.DataFrame(rows)
    report.to_csv(ROOT / "experiments" / "day2" / "chunk_metrics_by_label.csv", index=False)
    print("\nPer-label metrics:")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
