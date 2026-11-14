"""Character offsets <-> BIO labels for a fast tokenizer."""
from src.evaluation.spans import LABELS

BIO_LABELS = ["O"] + [f"{prefix}-{label}" for label in LABELS for prefix in ("B", "I")]


def encode_labels(offsets, spans):
    labels = [-100 if start == end else 0 for start, end in offsets]
    occupied = set()
    for span in sorted(spans, key=lambda s: int(s["start_char"])):
        start, end = int(span["start_char"]), int(span["end_char"])
        indices = [i for i, (a, b) in enumerate(offsets) if a != b and a < end and start < b]
        if not indices or offsets[indices[0]][0] != start or offsets[indices[-1]][1] != end:
            raise ValueError(f"Span boundary cannot be represented by tokenizer: {span}")
        if occupied.intersection(indices):
            raise ValueError("Overlapping gold spans cannot be encoded with flat BIO labels")
        for n, i in enumerate(indices):
            labels[i] = BIO_LABELS.index(f"{'B' if n == 0 else 'I'}-{span['chunk_label']}")
        occupied.update(indices)
    return labels


def decode_labels(query, offsets, tags):
    output, current = [], None
    for (start, end), tag in zip(offsets, tags):
        if start == end:
            continue
        name = BIO_LABELS[int(tag)]
        if name == "O":
            current = None
            continue
        prefix, label = name.split("-", 1)
        if prefix == "B" or current is None or current["chunk_label"] != label:
            current = {"chunk_label": label, "start_char": int(start), "end_char": int(end)}
            output.append(current)
        else:
            current["end_char"] = int(end)
    return [dict(row, chunk_text=query[row["start_char"]:row["end_char"]]) for row in output]
