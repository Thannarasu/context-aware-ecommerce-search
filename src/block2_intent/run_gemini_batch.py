import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd

from src.evaluation.spans import assign_predicted_spans, calculate_span_metrics


def analyze_query(query):
    from src.block2_intent.gemini_query_understanding import analyze_query as run
    return run(query)


ROOT = Path(__file__).resolve().parents[2]
ANNOTATION_DIR = ROOT / "data" / "annotations"
OUTPUT_DIR = ROOT / "experiments" / "day4"

QUERY_FILE = ANNOTATION_DIR / "query_sample.csv"
GOLD_CHUNK_FILE = ANNOTATION_DIR / "chunk_annotations_reviewed.csv"

# Gemini free-tier limits can vary by model and account. This conservative
# interval spaces API attempts apart; it does not override the provider quota.
MIN_REQUEST_INTERVAL_SECONDS = 12.5
_last_api_request_time = None

QUERY_RESULT_COLUMNS = [
    "query_id",
    "query",
    "primary_intent",
    "intent_explanation",
    "needs_clarification",
    "missing_information",
    "latency_seconds",
    "status",
    "error",
]
CHUNK_RESULT_COLUMNS = [
    "query_id",
    "query",
    "chunk_text",
    "chunk_label",
    "start_char",
    "end_char",
    "span_found",
]
ERROR_COLUMNS = ["query_id", "query", "error"]


def pace_api_request():
    """Wait before every API attempt to avoid sending requests too quickly."""
    global _last_api_request_time

    now = time.perf_counter()
    if _last_api_request_time is not None:
        elapsed = now - _last_api_request_time
        wait_seconds = MIN_REQUEST_INTERVAL_SECONDS - elapsed
        if wait_seconds > 0:
            print(f"  Waiting {wait_seconds:.1f}s to pace the next API request...")
            time.sleep(wait_seconds)

    _last_api_request_time = time.perf_counter()


def retry_delay_from_message(message):
    """Extract a retry delay from common Gemini error message formats."""
    patterns = [
        r"retry in\s+([0-9]+(?:\.[0-9]+)?)\s*s",
        r"retryDelay['\"]?\s*[:=]\s*['\"]?([0-9]+(?:\.[0-9]+)?)s",
        r"retry_delay\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return float(match.group(1)) + 2.0
    return 62.0


def analyze_query_with_retry(query, max_attempts=4):
    """Call Gemini, retrying rate-limit and temporary server errors safely."""
    for attempt in range(max_attempts):
        try:
            # Pace every actual attempt, including retries.
            pace_api_request()
            return analyze_query(query)

        except Exception as exc:
            message = str(exc)
            is_rate_limit = "429" in message or "RESOURCE_EXHAUSTED" in message
            is_temporary = any(
                token in message
                for token in ("503", "UNAVAILABLE", "500", "INTERNAL", "502", "BAD_GATEWAY")
            )

            if not (is_rate_limit or is_temporary):
                raise

            if attempt == max_attempts - 1:
                raise

            if is_rate_limit:
                wait_seconds = retry_delay_from_message(message)
            else:
                wait_seconds = min(5 * (2 ** attempt), 30)

            print(
                f"  Temporary Gemini API error ({'rate limit' if is_rate_limit else 'server error'}). "
                f"Retrying in {wait_seconds:.0f}s "
                f"(attempt {attempt + 2}/{max_attempts})."
            )
            time.sleep(wait_seconds)

    raise RuntimeError("Gemini retries exhausted.")




def main():
    parser = argparse.ArgumentParser(
        description="Run Gemini query understanding against reviewed chunk annotations."
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="New experiment directory; historical runs are not overwritten.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum reviewed queries to process; use 0 for all.",
    )
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be 0 or a positive integer.")

    global OUTPUT_DIR
    OUTPUT_DIR = args.output_dir.resolve()
    if OUTPUT_DIR.exists():
        parser.error("Choose a new --output-dir to preserve previous experiments.")
    OUTPUT_DIR.mkdir(parents=True)

    if not QUERY_FILE.exists():
        raise FileNotFoundError(f"Query file not found: {QUERY_FILE}")
    if not GOLD_CHUNK_FILE.exists():
        raise FileNotFoundError(f"Reviewed gold annotation file not found: {GOLD_CHUNK_FILE}")

    queries = pd.read_csv(QUERY_FILE)
    gold = pd.read_csv(GOLD_CHUNK_FILE)

    required_query_cols = {"query_id", "query"}
    required_gold_cols = {
        "query_id", "chunk_label", "start_char", "end_char", "review_status"
    }

    if not required_query_cols.issubset(queries.columns):
        raise ValueError(
            f"Expected query columns {required_query_cols}; found {set(queries.columns)}"
        )
    if not required_gold_cols.issubset(gold.columns):
        raise ValueError(
            f"Expected gold columns {required_gold_cols}; found {set(gold.columns)}"
        )

    # Evaluate only spans explicitly marked as reviewed.
    gold = gold[gold["review_status"].astype(str).str.strip().str.lower().eq("reviewed")].copy()
    if gold.empty:
        raise ValueError("No reviewed gold chunk spans were found.")

    queries["query_id"] = queries["query_id"].astype(str)
    gold["query_id"] = gold["query_id"].astype(str)
    queries["query"] = queries["query"].fillna("").astype(str)

    reviewed_ids = set(gold["query_id"])
    queries = queries[queries["query_id"].isin(reviewed_ids)].copy()
    queries = queries.drop_duplicates("query_id")
    queries = queries.sort_values("query_id").reset_index(drop=True)

    if args.limit > 0:
        queries = queries.head(args.limit)
    if queries.empty:
        raise ValueError("No matching reviewed queries were found.")

    query_map = dict(zip(queries["query_id"], queries["query"]))
    gold = gold[gold["query_id"].isin(query_map)].copy()

    query_results = []
    chunk_results = []
    error_results = []
    total_start = time.perf_counter()

    for index, row in queries.iterrows():
        query_id = str(row["query_id"])
        query = str(row["query"])
        query_gold = gold[gold["query_id"].eq(query_id)]

        print(f"[{index + 1}/{len(queries)}] Analyzing query_id={query_id}: {query}")
        start_time = time.perf_counter()

        try:
            result = analyze_query_with_retry(query)
            elapsed = time.perf_counter() - start_time
            chunks = result.get("chunks", []) or []
            predicted_spans = assign_predicted_spans(query, chunks)

            query_results.append({
                "query_id": query_id,
                "query": query,
                "primary_intent": result.get("primary_intent"),
                "intent_explanation": result.get("intent_explanation"),
                "needs_clarification": result.get("needs_clarification"),
                "missing_information": json.dumps(
                    result.get("missing_information", []), ensure_ascii=False
                ),
                "latency_seconds": round(elapsed, 3),
                "status": "success",
                "error": "",
            })

            for chunk in predicted_spans:
                chunk_results.append({
                    "query_id": query_id,
                    "query": query,
                    **chunk,
                })

            print(f"  Success: {len(chunks)} chunks, {elapsed:.2f}s")

        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            message = f"{type(exc).__name__}: {exc}"

            query_results.append({
                "query_id": query_id,
                "query": query,
                "primary_intent": None,
                "intent_explanation": None,
                "needs_clarification": None,
                "missing_information": None,
                "latency_seconds": round(elapsed, 3),
                "status": "error",
                "error": message,
            })
            error_results.append({
                "query_id": query_id,
                "query": query,
                "error": message,
            })
            print(f"  ERROR: {message}")

    # Explicit columns prevent KeyError when every API call fails or no chunks
    # are returned. Empty CSVs will still have useful headers.
    query_df = pd.DataFrame(query_results, columns=QUERY_RESULT_COLUMNS)
    chunk_df = pd.DataFrame(chunk_results, columns=CHUNK_RESULT_COLUMNS)
    error_df = pd.DataFrame(error_results, columns=ERROR_COLUMNS)

    query_path = OUTPUT_DIR / "gemini_query_predictions.csv"
    chunk_path = OUTPUT_DIR / "gemini_chunk_predictions.csv"
    errors_path = OUTPUT_DIR / "gemini_errors.csv"
    metrics_path = OUTPUT_DIR / "gemini_chunk_metrics.json"
    summary_path = OUTPUT_DIR / "gemini_run_summary.json"

    query_df.to_csv(query_path, index=False)
    chunk_df.to_csv(chunk_path, index=False)
    error_df.to_csv(errors_path, index=False)

    successful_ids = set(
        query_df.loc[query_df["status"].eq("success"), "query_id"].astype(str)
    )
    evaluated_gold = gold.copy()
    evaluated_predictions = chunk_df.copy()

    metrics = calculate_span_metrics(evaluated_gold, evaluated_predictions)
    metrics["successful_queries"] = len(successful_ids)
    metrics["failed_queries"] = len(error_results)
    metrics["queries_evaluated_for_metrics"] = len(queries)
    metrics["query_success_rate"] = len(successful_ids) / len(queries)

    per_label = {}
    labels = sorted(
        set(evaluated_gold["chunk_label"].dropna().astype(str))
        | set(evaluated_predictions["chunk_label"].dropna().astype(str))
    )
    for label in labels:
        label_gold = evaluated_gold[evaluated_gold["chunk_label"].astype(str).eq(label)]
        label_pred = evaluated_predictions[
            evaluated_predictions["chunk_label"].astype(str).eq(label)
        ]
        per_label[label] = calculate_span_metrics(label_gold, label_pred)
    metrics["per_label"] = per_label

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    summary = {
        "model": os.getenv("GEMINI_MODEL", ""),
        "requested_limit": args.limit,
        "queries_attempted": len(query_df),
        "queries_succeeded": len(successful_ids),
        "queries_failed": len(error_results),
        "total_elapsed_seconds": round(time.perf_counter() - total_start, 3),
        "prediction_file": str(query_path),
        "chunk_file": str(chunk_path),
        "errors_file": str(errors_path),
        "metrics_file": str(metrics_path),
    }

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\n=== Gemini batch summary ===")
    print(json.dumps(summary, indent=2))
    print("\n=== Exact-span metrics ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nOutputs saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
