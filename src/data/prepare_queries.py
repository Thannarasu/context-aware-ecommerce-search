
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# Configuration
# --------------------------------------------------
INPUT_FILE = Path(
    "data/raw/shopping_queries_dataset_examples.parquet"
)

OUTPUT_DIR = Path("data/processed")
ANNOTATION_DIR = Path("data/annotations")

QUERY_OUTPUT = OUTPUT_DIR / "unique_queries_us_train.csv"
SAMPLE_OUTPUT = ANNOTATION_DIR / "query_sample.csv"
INTENT_TEMPLATE = ANNOTATION_DIR / "intent_annotations.csv"
CHUNK_TEMPLATE = ANNOTATION_DIR / "chunk_annotations.csv"

SAMPLE_SIZE = 150
RANDOM_SEED = 42

# --------------------------------------------------
# 1. Read only required columns
# --------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

    columns = [
        "query",
        "query_id",
        "product_locale",
        "split",
    ]

    print("Reading ESCI examples dataset...")

    df = pd.read_parquet(
        INPUT_FILE,
        columns=columns
    )

    print(f"Rows loaded: {len(df):,}")

    # --------------------------------------------------
    # 2. Validate required fields
    # --------------------------------------------------
    df = df.dropna(
        subset=["query", "query_id", "product_locale", "split"]
    ).copy()

    df["query"] = df["query"].astype(str).str.strip()
    df = df[df["query"].str.len() > 0]

    print("\nLocale distribution:")
    print(df["product_locale"].value_counts().head(10))

    print("\nSplit distribution:")
    print(df["split"].value_counts())

    # --------------------------------------------------
    # 3. Select US-English locale and training split
    # --------------------------------------------------
    filtered = df[
        (df["product_locale"] == "us")
        & (df["split"] == "train")
    ].copy()

    print(f"\nUS training example rows: {len(filtered):,}")

    if filtered.empty:
        raise ValueError(
            "No rows found for product_locale='us' "
            "and split='train'. Check actual values."
        )

    # --------------------------------------------------
    # 4. Verify query_id to query consistency
    # --------------------------------------------------
    query_counts = (
        filtered.groupby("query_id")["query"]
        .nunique()
    )

    inconsistent_ids = query_counts[query_counts > 1]

    print(
        "Query IDs with multiple query texts:",
        len(inconsistent_ids)
    )

    if not inconsistent_ids.empty:
        raise ValueError(
            "Some query IDs map to multiple query texts. "
            "Investigate before sampling."
        )

    # --------------------------------------------------
    # 5. Deduplicate at query level
    # --------------------------------------------------
    queries = (
        filtered[
            ["query_id", "query", "product_locale", "split"]
        ]
        .drop_duplicates(subset=["query_id"])
        .reset_index(drop=True)
    )

    queries["query_length"] = (
        queries["query"].str.split().str.len()
    )

    queries.to_csv(QUERY_OUTPUT, index=False)

    print(f"\nUnique training queries: {len(queries):,}")
    print("\nQuery length summary:")
    print(queries["query_length"].describe())

    # --------------------------------------------------
    # 6. Generate reproducible annotation sample
    # --------------------------------------------------
    sample = queries.sample(
        n=min(SAMPLE_SIZE, len(queries)),
        random_state=RANDOM_SEED
    ).sort_values("query_id").reset_index(drop=True)

    sample.to_csv(SAMPLE_OUTPUT, index=False)

    # --------------------------------------------------
    # 7. Create intent annotation template
    # --------------------------------------------------
    intent = sample[["query_id", "query"]].copy()
    intent["intent_label"] = ""
    intent["annotator"] = ""
    intent["review_status"] = "pending"
    intent["notes"] = ""

    intent.to_csv(INTENT_TEMPLATE, index=False)

    # --------------------------------------------------
    # 8. Create chunk annotation template
    # One row per annotated chunk
    # --------------------------------------------------
    chunk = pd.DataFrame(columns=[
        "query_id",
        "query",
        "chunk_text",
        "chunk_label",
        "start_char",
        "end_char",
        "annotator",
        "review_status",
        "notes",
    ])

    chunk.to_csv(CHUNK_TEMPLATE, index=False)

    print("\nPreparation completed successfully.")
    print(f"Unique queries: {QUERY_OUTPUT}")
    print(f"Query sample: {SAMPLE_OUTPUT}")
    print(f"Intent template: {INTENT_TEMPLATE}")
    print(f"Chunk template: {CHUNK_TEMPLATE}")
    print(f"Sample size: {len(sample)}")


if __name__ == "__main__":
    main()
