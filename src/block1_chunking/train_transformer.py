"""Fine-tune DistilBERT and compare against rules on identical query splits.

Run with --pilot for the existing development annotations. For a final run,
provide an independently reviewed corpus and a pre-frozen split manifest.
"""
import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from src.block1_chunking.bio import BIO_LABELS, decode_labels, encode_labels
from src.evaluation.spans import LABELS, calculate_span_metrics

ROOT = Path(__file__).resolve().parents[2]


def load_corpus(query_file, gold_file, review_registry=None):
    from src.evaluation.audit_phase1 import read
    queries = {r["query_id"]: r["query"] for r in read(query_file)}
    spans = defaultdict(list)
    for row in read(gold_file):
        if row["review_status"].strip().lower() != "reviewed":
            continue
        start, end = int(row["start_char"]), int(row["end_char"])
        if row["query_id"] not in queries or not 0 <= start < end <= len(queries[row["query_id"]]):
            raise ValueError(f"Invalid gold span: {row}")
        if row["chunk_label"] not in LABELS:
            raise ValueError(f"Unknown label: {row}")
        spans[row["query_id"]].append(dict(row, start_char=start, end_char=end))
    if review_registry:
        registry = json.loads(Path(review_registry).read_text(encoding="utf-8"))
        for qid, record in registry.items():
            if not record.get("complete") or qid not in queries:
                raise ValueError("Incomplete query review registry")
            if len(spans[qid]) != record["span_count"]:
                raise ValueError("Registry span count does not match gold")
    if len(spans) < 10:
        raise ValueError("At least ten reviewed queries required even for a pilot")
    return queries, spans


def make_splits(queries, spans, seed):
    groups = defaultdict(list)
    for qid in sorted(spans):
        groups[" ".join(queries[qid].lower().split())].append(qid)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    if n < 10:
        raise ValueError("At least ten distinct query groups required")
    a, b = int(n * .6), int(n * .8)
    return {name: [qid for key in part for qid in groups[key]]
            for name, part in zip(("train", "validation", "test"), (keys[:a], keys[a:b], keys[b:]))}


def validate_splits(splits, queries, spans):
    seen_ids, seen_texts = set(), set()
    for name in ("train", "validation", "test"):
        ids = splits[name]
        if not ids or len(set(ids)) != len(ids) or any(q not in spans for q in ids):
            raise ValueError(f"Invalid split: {name}")
        texts = {" ".join(queries[q].lower().split()) for q in ids}
        if seen_ids.intersection(ids) or seen_texts.intersection(texts):
            raise ValueError("Query ID or normalized query leaks between splits")
        seen_ids.update(ids)
        seen_texts.update(texts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=ROOT / "data/annotations/query_sample.csv")
    parser.add_argument("--gold", type=Path, default=ROOT / "data/annotations/chunk_annotations_reviewed.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--review-registry", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--checkpoint", default="distilbert/distilbert-base-uncased")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--mask-unalignable", action="store_true",
                        help="Pilot only: mask tokens overlapping unrepresentable gold boundaries; retain original evaluation gold")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.mask_unalignable and not args.pilot:
        parser.error("--mask-unalignable is restricted to pilot runs")
    if args.output.exists() or args.epochs < 1 or args.batch_size < 1:
        parser.error("Use a new output directory and positive epochs/batch size")
    if not args.pilot and not args.splits:
        parser.error("Final runs require a pre-frozen --splits manifest")
    if not args.pilot and args.gold.resolve() == (ROOT / "data/annotations/chunk_annotations_reviewed.csv").resolve():
        parser.error("Existing development gold requires --pilot; use independent gold for final evaluation")
    queries, spans = load_corpus(args.queries, args.gold, args.review_registry)
    splits = json.loads(args.splits.read_text(encoding="utf-8")) if args.splits else make_splits(queries, spans, args.seed)
    validate_splits(splits, queries, spans)
    args.output.mkdir(parents=True)
    metadata = {"mode": "development_pilot" if args.pilot else "independent_evaluation",
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.queries, args.gold)},
                "label_support": {name: {label: sum(r["chunk_label"] == label for q in ids for r in spans[q])
                                         for label in LABELS} for name, ids in splits.items()}}
    (args.output / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    (args.output / "run_info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if args.prepare_only:
        print(json.dumps(metadata, indent=2))
        return

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer, set_seed
    from torch.utils.data import DataLoader
    set_seed(args.seed)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True, local_files_only=args.local_files_only)
    model = AutoModelForTokenClassification.from_pretrained(
        args.checkpoint, num_labels=len(BIO_LABELS),
        id2label=dict(enumerate(BIO_LABELS)), label2id={v: i for i, v in enumerate(BIO_LABELS)},
        local_files_only=args.local_files_only)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    encoded, alignment_issues = {}, []
    for qid in set(sum(splits.values(), [])):
        item = tokenizer(queries[qid], return_offsets_mapping=True, max_length=128, truncation=False)
        if len(item["input_ids"]) > 128:
            raise ValueError(f"Query {qid} exceeds 128 tokens; configure a deliberate long-query policy")
        offsets = item.pop("offset_mapping")
        try:
            item["labels"] = encode_labels(offsets, spans[qid])
        except ValueError:
            if not args.mask_unalignable:
                raise
            valid, invalid = [], []
            for span in spans[qid]:
                try:
                    encode_labels(offsets, [span])
                    valid.append(span)
                except ValueError:
                    invalid.append(span)
            item["labels"] = encode_labels(offsets, valid)
            for span in invalid:
                alignment_issues.append(dict(span, query=queries[qid],
                    selected_text=queries[qid][span["start_char"]:span["end_char"]]))
                for i, (a, b) in enumerate(offsets):
                    if a != b and a < span["end_char"] and span["start_char"] < b:
                        item["labels"][i] = -100
        encoded[qid] = item, offsets
    (args.output / "alignment_issues.json").write_text(json.dumps(alignment_issues, indent=2), encoding="utf-8")

    from transformers import DataCollatorForTokenClassification
    loader = DataLoader([encoded[q][0] for q in splits["train"]], batch_size=args.batch_size,
                        shuffle=True, collate_fn=DataCollatorForTokenClassification(tokenizer))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

    def predict(ids):
        model.eval()
        records = []
        with torch.no_grad():
            for qid in ids:
                item, offsets = encoded[qid]
                tensors = {k: torch.tensor([v], device=device) for k, v in item.items() if k != "labels"}
                tags = model(**tensors).logits.argmax(-1)[0].cpu().tolist()
                records.extend(dict(r, query_id=qid, query=queries[qid])
                               for r in decode_labels(queries[qid], offsets, tags))
        return records

    def score(ids, predictions):
        gold = [r for qid in ids for r in spans[qid]]
        result = calculate_span_metrics(gold, predictions)
        result["per_label"] = {label: calculate_span_metrics(
            [r for r in gold if r["chunk_label"] == label],
            [r for r in predictions if r["chunk_label"] == label]) for label in LABELS}
        return result

    history, best = [], -1
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad()
            loss = model(**{k: v.to(device) for k, v in batch.items()}).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        metrics = score(splits["validation"], predict(splits["validation"]))
        history.append({"epoch": epoch + 1, "train_loss": sum(losses) / len(losses), "validation": metrics})
        print(json.dumps(history[-1]), flush=True)
        if metrics["f1"] > best:
            best = metrics["f1"]
            model.save_pretrained(args.output / "model")
            tokenizer.save_pretrained(args.output / "model")
    model = AutoModelForTokenClassification.from_pretrained(args.output / "model").to(device)
    predictions = predict(splits["test"])
    from src.block1_chunking.rule_based_chunker import extract_chunks
    rules = [dict(r, query_id=qid) for qid in splits["test"] for r in extract_chunks(queries[qid])]
    result = {"mode": metadata["mode"], "transformer": score(splits["test"], predictions),
              "rules": score(splits["test"], rules), "history": history,
              "torch_version": torch.__version__, "device": str(device)}
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with open(args.output / "predictions.csv", "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["query_id", "query", "chunk_text", "chunk_label", "start_char", "end_char"])
        writer.writeheader()
        writer.writerows(predictions)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
