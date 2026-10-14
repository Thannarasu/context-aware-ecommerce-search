import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "annotations" / "query_sample.csv"
OUTPUT_FILE = PROJECT_ROOT / "experiments" / "day2" / "chunk_predictions.csv"

PHRASE_RULES = {
    "PRODUCT": [
        "airpods pro case", "airpod case", "apple iphone xr", "case",
        "aquarium lid", "atomic habits", "b brand face mask", "baby knobby ball",
        "badgley mischka shoes", "bar q grill charcoal", "bathing suit surong",
        "belt strap", "boho wall decor", "boho decor",
        "building writers learning without tears", "building writers learning",
        "c.c bun beanies", "calcutta d", "camping sleeping pad",
        "car seat organizer", "carhartt", "force tappen cargo pant",
        "black out curtains", "cupcake toppers", "kitchen faucet", "shed plans",
        "sleepers", "steering wheel", "wheel cover", "wrench", "belt", "faucet",
        "sleeping pad", "decor", "shoes", "mask", "lid", "ball", "charcoal",
        "beanies", "shirts", "mascara", "curtains", "scrapers", "scraper",
        "tappers", "tappen cargo pant", "cargo pant", "toddler tennis shoes",
        "navy blue silk fabric", "fabric", "phone case", "phone cases",
        "tongue scrapers", "steering wheel cover",
    ],
    "ATTRIBUTE": [
        "0-3 months", "1 3/4 inch", "tooled", "1 hole", "12x16", "garage door",
        "7/8", "timeless", "go with any outfit", "gucci", "disney",
        "carbon fiber", "waterproof", "black", "blue", "cotton", "leather",
        "stainless steel", "navy blue", "silk", "yellow", "detachable",
    ],
    "PURPOSE": [
        "o2 sensor", "for camping", "for hiking", "for running",
        "for travel", "for work", "for school",
    ],
    "OCCASION": [
        "wedding", "birthday", "christmas", "anniversary", "valentine's day",
    ],
    "CONTEXT": [
        "boy", "girl", "for toddlers", "for adults", "get warm easily",
        "women", "for women", "front seat", "men's",
    ],
}

REGEX_RULES = [
    ("CONSTRAINT", r"\bwithout\b(?:\s+\w+){1,4}"),
    ("CONSTRAINT", r"\bunder\s+\$?\d+(?:\.\d{1,2})?(?:\s+dollars?)?\b"),
    ("CONSTRAINT", r"\b(?:less than|no more than|at most)\s+\$?\d+(?:\.\d{1,2})?\b"),
    ("CONSTRAINT", r"\bnot a thin fabric but not thick either\b"),
    ("CONTEXT", r"\bget warm easily\b"),
]

LABEL_PRIORITY = {
    "CONSTRAINT": 0, "OCCASION": 1, "PURPOSE": 2,
    "CONTEXT": 3, "ATTRIBUTE": 4, "PRODUCT": 5,
}


def phrase_matches(query, phrase, label):
    """Return case-insensitive phrase matches with original character offsets."""
    pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
    return [
        {
            "chunk_text": match.group(0),
            "chunk_label": label,
            "start_char": match.start(),
            "end_char": match.end(),
            "rule": f"phrase:{phrase}",
        }
        for match in re.finditer(pattern, query, flags=re.IGNORECASE)
    ]


def extract_chunks(query):
    candidates = []
    for label, phrases in PHRASE_RULES.items():
        for phrase in phrases:
            candidates.extend(phrase_matches(query, phrase, label))

    for label, pattern in REGEX_RULES:
        for match in re.finditer(pattern, query, flags=re.IGNORECASE):
            candidates.append({
                "chunk_text": match.group(0),
                "chunk_label": label,
                "start_char": match.start(),
                "end_char": match.end(),
                "rule": f"regex:{pattern}",
            })

    # Prefer the longest span, then use deterministic label priority for ties.
    candidates.sort(key=lambda item: (
        -(item["end_char"] - item["start_char"]),
        LABEL_PRIORITY[item["chunk_label"]],
        item["start_char"],
    ))

    selected = []
    for candidate in candidates:
        overlaps = any(
            candidate["start_char"] < existing["end_char"]
            and existing["start_char"] < candidate["end_char"]
            for existing in selected
        )
        if not overlaps:
            selected.append(candidate)

    return sorted(selected, key=lambda item: item["start_char"])


def main():
    queries = pd.read_csv(INPUT_FILE)
    required = {"query_id", "query"}
    missing = required - set(queries.columns)
    if missing:
        raise ValueError(f"Missing required input columns: {sorted(missing)}")

    records = []
    unique_queries = queries[["query_id", "query"]].drop_duplicates("query_id")
    for row in unique_queries.itertuples(index=False):
        if not isinstance(row.query, str) or not row.query.strip():
            continue
        for chunk in extract_chunks(row.query):
            records.append({
                "query_id": row.query_id,
                "query": row.query,
                **chunk,
            })

    output = pd.DataFrame(records, columns=[
        "query_id", "query", "chunk_text", "chunk_label",
        "start_char", "end_char", "rule",
    ])
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)

    # Sanity-check that each saved span reproduces its source text.
    invalid = output[
        output.apply(
            lambda row: row["query"][int(row["start_char"]):int(row["end_char"])]
            != row["chunk_text"],
            axis=1,
        )
    ]
    print(f"Unique queries processed: {unique_queries['query_id'].nunique()}")
    print(f"Predicted chunks: {len(output)}")
    print(f"Invalid offsets: {len(invalid)}")
    print(f"Saved: {OUTPUT_FILE}")
    if not output.empty:
        print("\nPredictions by label:")
        print(output["chunk_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
