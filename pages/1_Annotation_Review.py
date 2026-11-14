"""Local human annotation workflow. Not copied into the public demo image."""
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data.human_review import INTENTS, save_review
from src.evaluation.audit_phase1 import read

st.set_page_config(page_title="Phase 1 annotation review", layout="wide")
st.title("Human annotation review")
st.write("Review the complete query, correct the spans and primary intent, then explicitly approve the record.")
st.caption("Suggestions are provisional. Approvals are saved locally as new JSON records; original annotations are preserved.")
with st.expander("Review guidance", expanded=True):
    st.markdown("""
- PRODUCT: explicit product/category. ATTRIBUTE: brand, material, color, size, or specification.
- PURPOSE: intended use. OCCASION: event. CONTEXT: user situation. CONSTRAINT: restriction, exclusion, or budget.
- Use exact character offsets: start is included, end is excluded. Separate attributes from the product category consistently.
- Select one dominant primary intent. Do not infer missing requirements. Explain ambiguous choices in the notes.
- Check the displayed substring carefully: current pilot annotations contain two cases selecting `heel` inside `wheel`.
""")
queries = read(ROOT / "data/annotations/query_sample.csv")
intents = {r["query_id"]: r for r in read(ROOT / "data/annotation_outputs/intent_annotations_annotated.csv")}
gold = read(ROOT / "data/annotations/chunk_annotations_reviewed.csv")
query_id = st.selectbox("Query", [r["query_id"] for r in queries],
                        format_func=lambda q: f"{q}: {next(r['query'] for r in queries if r['query_id'] == q)}")
query = next(r["query"] for r in queries if r["query_id"] == query_id)
st.subheader(query)
st.code("\n".join(f"{i:3}: {char!r}" for i, char in enumerate(query)), language=None)
chunks = [{"start_char": int(r["start_char"]), "end_char": int(r["end_char"]),
           "chunk_label": r["chunk_label"]} for r in gold if r["query_id"] == query_id]
if not chunks:
    from src.block1_chunking.rule_based_chunker import extract_chunks
    chunks = [{k: r[k] for k in ("start_char", "end_char", "chunk_label")} for r in extract_chunks(query)]
    st.caption("Chunk suggestions below come from the rule baseline; complete human chunk gold is not available for this query.")
edited = st.text_area("Chunk spans as JSON (edit offsets/labels; [] confirms no spans)",
                      value=json.dumps(chunks, indent=2), height=260, key=f"chunks_{query_id}")
parsed = None
try:
    parsed = json.loads(edited)
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON list")
    st.dataframe([dict(r, selected_text=query[int(r["start_char"]):int(r["end_char"])]) for r in parsed])
except (ValueError, KeyError, TypeError) as exc:
    st.error(f"Correct the chunk JSON: {exc}")
    parsed = None
suggested = intents.get(query_id, {}).get("intent_label", INTENTS[0])
intent = st.selectbox("Primary intent", INTENTS, index=INTENTS.index(suggested), key=f"intent_{query_id}")
annotator = st.text_input("Your reviewer name / identifier")
notes = st.text_area("Review rationale or unresolved ambiguity", key=f"notes_{query_id}")
approved = st.checkbox("I have checked every span and the intent; this is a complete annotation for this query.", key=f"approved_{query_id}")
if st.button("Save human approval", disabled=not approved or parsed is None):
    try:
        path = save_review(ROOT / "data/human_reviews", query_id, query, parsed, intent, annotator, notes)
        st.success(f"Saved {path.name}")
    except (ValueError, TypeError, KeyError) as exc:
        st.error(str(exc))
