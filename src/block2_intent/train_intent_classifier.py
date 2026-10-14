
from pathlib import Path

import json
import warnings

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report


ROOT = Path(__file__).resolve().parents[2]

INTENT_FILE = ROOT / "data" / "annotation_outputs" / "intent_annotations_annotated.csv"
OUTPUT_DIR = ROOT / "experiments" / "day3"

CHUNK_FILE_CANDIDATES = [
    ROOT / "experiments" / "day2" / "chunk_predictions.csv",
    ROOT / "chunk_predictions_improved.csv",
    ROOT / "chunk_predictions.csv",
]

# True: run a pipeline smoke test using current provisional labels.
# False: use only rows explicitly marked "reviewed".
USE_PROVISIONAL_LABELS = True

RANDOM_STATE = 42
TEST_SIZE = 0.25


def find_chunk_file():
    for path in CHUNK_FILE_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find chunk predictions. Expected one of:\n"
        + "\n".join(str(p) for p in CHUNK_FILE_CANDIDATES)
    )


def load_data():
    if not INTENT_FILE.exists():
        raise FileNotFoundError(f"Intent annotations not found: {INTENT_FILE}")

    intents = pd.read_csv(INTENT_FILE)
    required = {"query_id", "query", "intent_label", "review_status"}
    missing = required - set(intents.columns)
    if missing:
        raise ValueError(f"Intent file is missing columns: {sorted(missing)}")

    intents = intents.dropna(subset=["query_id", "query", "intent_label"]).copy()
    intents["query_id"] = intents["query_id"].astype(str)
    intents["query"] = intents["query"].astype(str).str.strip()
    intents["intent_label"] = intents["intent_label"].astype(str).str.strip()

    if USE_PROVISIONAL_LABELS:
        warnings.warn(
            "EXPLORATORY RUN: using provisional intent labels. "
            "Do not report these metrics as validated model performance."
        )
    else:
        intents = intents[intents["review_status"].eq("reviewed")].copy()

    intents = intents[
        intents["query"].ne("") & intents["intent_label"].ne("")
    ].drop_duplicates("query_id")

    if intents.empty:
        raise ValueError(
            "No eligible intent labels. Review annotations and set "
            "review_status='reviewed', or enable provisional mode for a smoke test."
        )

    chunk_path = find_chunk_file()
    chunks = pd.read_csv(chunk_path)

    chunk_required = {"query_id", "chunk_label", "chunk_text"}
    chunk_missing = chunk_required - set(chunks.columns)
    if chunk_missing:
        raise ValueError(
            f"Chunk prediction file is missing columns: {sorted(chunk_missing)}"
        )

    chunks["query_id"] = chunks["query_id"].astype(str)
    chunks = chunks.dropna(subset=["chunk_label", "chunk_text"]).copy()

    return intents, chunks, chunk_path


def build_chunk_enriched_text(row, chunks_by_query):
    query = row["query"]
    query_chunks = chunks_by_query.get(row["query_id"], [])

    if not query_chunks:
        return query

    # Preserve the original query and append explicitly typed predicted spans.
    # This lets the model learn which words came from which chunk class.
    additions = []
    for label, text in query_chunks:
        additions.append(f"[CHUNK_{label}] {text}")

    return query + " " + " ".join(additions)


def make_classifier():
    features = FeatureUnion([
        (
            "word_tfidf",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
            ),
        ),
        (
            "char_tfidf",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=1,
                sublinear_tf=True,
            ),
        ),
    ])

    return Pipeline([
        ("features", features),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def evaluate_experiment(name, train_text, test_text, y_train, y_test, test_ids):
    model = make_classifier()
    model.fit(train_text, y_train)

    predictions = model.predict(test_text)
    labels = sorted(set(y_test) | set(predictions))

    metrics = {
        "experiment": name,
        "test_queries": len(y_test),
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
    }

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    prediction_df = pd.DataFrame({
        "query_id": test_ids,
        "actual_intent": y_test,
        "predicted_intent": predictions,
        "correct": np.asarray(y_test) == predictions,
    })

    return metrics, report, prediction_df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    intents, chunks, chunk_path = load_data()

    # Aggregate predictions in source-text order.
    if "start_char" in chunks.columns:
        chunks = chunks.sort_values(
            ["query_id", "start_char"], kind="stable"
        )

    chunks_by_query = (
        chunks.groupby("query_id")[["chunk_label", "chunk_text"]]
        .apply(lambda frame: list(frame.itertuples(index=False, name=None)))
        .to_dict()
    )

    intents["raw_text"] = intents["query"]
    intents["chunk_text_input"] = intents.apply(
        lambda row: build_chunk_enriched_text(row, chunks_by_query),
        axis=1,
    )

    y = intents["intent_label"].to_numpy()
    row_indices = np.arange(len(intents))

    class_counts = intents["intent_label"].value_counts()
    if class_counts.min() < 2:
        raise ValueError(
            "At least one intent class has fewer than 2 examples. "
            "Review/add examples or combine classes only if the taxonomy justifies it."
        )

    train_idx, test_idx = train_test_split(
        row_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_df = intents.iloc[train_idx]
    test_df = intents.iloc[test_idx]
    y_train = train_df["intent_label"].to_numpy()
    y_test = test_df["intent_label"].to_numpy()
    test_ids = test_df["query_id"].to_numpy()

    print(f"Intent annotation file: {INTENT_FILE}")
    print(f"Chunk predictions file: {chunk_path}")
    print(f"Eligible labelled queries: {len(intents)}")
    print(f"Training queries: {len(train_df)}")
    print(f"Test queries: {len(test_df)}")
    print(f"Provisional-label mode: {USE_PROVISIONAL_LABELS}")
    print("\nIntent label distribution:")
    print(class_counts.to_string())

    results = []
    for name, column in [
        ("E1_raw_query", "raw_text"),
        ("E2_predicted_chunks", "chunk_text_input"),
    ]:
        metrics, report, predictions = evaluate_experiment(
            name=name,
            train_text=train_df[column].tolist(),
            test_text=test_df[column].tolist(),
            y_train=y_train,
            y_test=y_test,
            test_ids=test_ids,
        )

        results.append(metrics)
        predictions.to_csv(
            OUTPUT_DIR / f"{name}_predictions.csv", index=False
        )

        with open(
            OUTPUT_DIR / f"{name}_classification_report.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(report, file, indent=2)

        print(f"\n{name}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Macro-F1:  {metrics['macro_f1']:.4f}")
        print(f"Weighted-F1: {metrics['weighted_f1']:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "intent_metrics_comparison.csv", index=False)

    run_info = {
        "intent_file": str(INTENT_FILE),
        "chunk_file": str(chunk_path),
        "provisional_labels_used": USE_PROVISIONAL_LABELS,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "split_query_ids": {
            "train": train_df["query_id"].tolist(),
            "test": test_df["query_id"].tolist(),
        },
    }
    with open(OUTPUT_DIR / "run_info.json", "w", encoding="utf-8") as file:
        json.dump(run_info, file, indent=2)

    print("\nComparison:")
    print(results_df.to_string(index=False))
    print(f"\nOutputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
