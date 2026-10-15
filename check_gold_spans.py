
import pandas as pd

queries = pd.read_csv("data/annotations/query_sample.csv", dtype={"query_id": str})
gold = pd.read_csv(
    "data/annotations/chunk_annotations_reviewed.csv",
    dtype={"query_id": str}
)
predictions = pd.read_csv(
    "experiments/day4/gemini_chunk_predictions.csv",
    dtype={"query_id": str}
)

gold = gold[gold["review_status"].eq("reviewed")]

print("Reviewed gold rows:", len(gold))
print("Prediction rows:", len(predictions))

for query_id in predictions["query_id"].unique():
    query_rows = queries[queries["query_id"] == query_id]
    gold_rows = gold[gold["query_id"] == query_id]
    pred_rows = predictions[predictions["query_id"] == query_id]

    print(f"\nQuery ID: {query_id}")

    if query_rows.empty:
        print("  ERROR: Query ID not found in query_sample.csv")
        continue

    query = query_rows.iloc[0]["query"]
    print("  Query:", query)

    print("  Gold spans:")
    if gold_rows.empty:
        print("    NONE FOUND")
    else:
        for _, row in gold_rows.iterrows():
            start = int(row["start_char"])
            end = int(row["end_char"])
            text = query[start:end]
            print(f"    {text!r} [{row['chunk_label']}] [{start}:{end}]")

    print("  Gemini predictions:")
    for _, row in pred_rows.iterrows():
        print(
            f"    {row['chunk_text']!r} "
            f"[{row['chunk_label']}] "
            f"[{row['start_char']}:{row['end_char']}]"
        )
