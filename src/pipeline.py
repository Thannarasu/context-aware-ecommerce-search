"""Common inference entry point for the Phase 1 application."""
import os
import json
from functools import lru_cache
from pathlib import Path
from time import perf_counter

from src.evaluation.spans import assign_predicted_spans


@lru_cache(maxsize=2)
def _transformer(path):
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    torch.set_num_threads(2)
    return (AutoTokenizer.from_pretrained(path, local_files_only=True),
            AutoModelForTokenClassification.from_pretrained(path, local_files_only=True).eval())


def analyze(query, method="rule_based"):
    if not query.strip():
        raise ValueError("Enter a shopping query.")
    if len(query) > 2000:
        raise ValueError("Please limit the query to 2,000 characters.")
    started = perf_counter()
    if method == "gemini":
        from src.block2_intent.run_gemini_batch import analyze_query_with_retry
        result = analyze_query_with_retry(query)
        result["chunks"] = [dict(c, **s) for c, s in zip(
            result["chunks"], assign_predicted_spans(query.strip(), result["chunks"]))]
    else:
        if method == "rule_based":
            from src.block1_chunking.rule_based_chunker import extract_chunks
            spans = extract_chunks(query)
        elif method == "distilbert":
            path = Path(os.getenv("CHUNK_MODEL_PATH", "models/distilbert_chunker"))
            if not (path / "config.json").exists():
                raise RuntimeError("Trained transformer artifact is unavailable. Train and package it first.")
            import torch
            from src.block1_chunking.bio import decode_labels
            tokenizer, model = _transformer(str(path))
            inputs = tokenizer(query, return_offsets_mapping=True, return_tensors="pt", truncation=False)
            if inputs["input_ids"].shape[1] > 128:
                raise ValueError("Transformer queries must fit within 128 tokens.")
            offsets = inputs.pop("offset_mapping")[0].tolist()
            with torch.no_grad():
                tags = model(**inputs).logits.argmax(-1)[0].tolist()
            spans = decode_labels(query, offsets, tags)
        else:
            raise ValueError(f"Unknown method: {method}")
        chunks = [dict(s, text=s["chunk_text"], label=s["chunk_label"],
                       explanation=f"Extracted using {method}.") for s in spans]
        # Intent requires a trained classifier; do not present a label heuristic
        # as the trained experimental model.
        intent, explanation = "UNAVAILABLE", "Intent classifier artifact has not been trained and packaged."
        provenance = "unavailable"
        intent_path = Path(os.getenv("INTENT_MODEL_PATH", "models/intent_raw.joblib"))
        if intent_path.exists():
            import joblib
            intent = str(_intent_model(str(intent_path)).predict([query])[0])
            explanation = "Prediction from the saved raw-query intent classifier."
            metadata_path = intent_path.with_suffix(".metadata.json")
            if metadata_path.exists():
                provenance = json.loads(metadata_path.read_text(encoding="utf-8")).get("mode", "unknown")
            else:
                provenance = "unknown"
        missing = [] if any(c["label"] == "PRODUCT" for c in chunks) else ["Product category"]
        result = {"chunks": chunks, "primary_intent": intent,
                  "intent_explanation": explanation, "needs_clarification": bool(missing),
                  "missing_information": missing, "intent_model_status": provenance}
        if method == "distilbert":
            result["chunk_model_status"] = "development_pilot" if (path / "pilot.json").exists() else "unknown"
    result.update(method=method, latency_seconds=round(perf_counter() - started, 3))
    return result


@lru_cache(maxsize=2)
def _intent_model(path):
    import joblib
    return joblib.load(path)
