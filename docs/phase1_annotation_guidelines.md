# Phase 1 annotation guideline, working version 0.2

Status: proposed for human/mentor review. Existing labels are development evidence, not automatically independent ground truth.

Annotate exact half-open character intervals `[start_char, end_char)` in the original unmodified query. Do not normalize text before recording offsets. Each reviewed query must have a complete annotation, including explicit confirmation when no spans exist. The current span-only gold CSV cannot represent a reviewed zero-span query; extend it with a query review registry before introducing such cases.

## Labels

| Label | Meaning | Example |
| --- | --- | --- |
| PRODUCT | Explicit product/category noun phrase | backpack |
| ATTRIBUTE | Brand, material, color, size, style, specification | waterproof |
| PURPOSE | Intended activity/function | for hiking |
| OCCASION | Named event | for a wedding |
| CONTEXT | User circumstance or environment | I get warm easily |
| CONSTRAINT | Explicit restriction, negation, or budget | under $80; without leather |

Keep separately expressed attributes separate from the product category. Use a consistent decision for brands and model names across the corpus. Avoid overlapping spans for the flat BIO experiment. Include relation words such as 'for' when they form part of a purpose/occasion phrase. Exclude surrounding punctuation and unrelated filler. Do not infer unmentioned products or invent spans. Record ambiguity rather than silently forcing a decision.

Primary intent is a single dominant requirement, using PRODUCT_SEARCH, ATTRIBUTE_SEARCH, USE_CASE_SEARCH, OCCASION_SEARCH, or CONTEXT_CONSTRAINT_SEARCH. Multiple chunks do not automatically imply a context intent. When requirements compete, record the annotator rationale and adjudicate consistently.

## Human review workflow

Review all 150 existing assistant-generated intent suggestions, even though their CSV review_status says reviewed. Use a separate human annotator identity and a review date; preserve original suggestions and provenance. Have a second reviewer or the supervisor adjudicate ambiguous examples and a sample of other cases. Report agreement only when independent annotations actually exist.

Review complete chunk annotations for the remaining queries. Add genuine examples for underrepresented labels: the current 60-span gold has no OCCASION and one PURPOSE span. Do not manufacture test coverage by relabeling existing examples. Keep authored/synthetic challenge queries separate from sampled ESCI queries and report their source.

Freeze query-group splits before method improvements. Existing rules and historical prompts have seen the pilot sample, so splitting this sample now does not make it independent of rule development. New independently reviewed queries are needed for final claims. Never tune on final test errors.

## Guided local review

Run `python -m streamlit run app.py` from the repository and open Annotation Review in the sidebar. Enter your human reviewer identifier. Inspect/edit each chunk's JSON offsets and label, check the selected text preview, choose the primary intent, explain ambiguity, then approve the complete annotation. Empty chunk lists are supported and mean explicitly reviewed zero-span queries. Original suggestions are not overwritten; every approval creates a versioned JSON record in ignored `data/human_reviews/`.

Export completed approvals with `python -m src.data.export_human_reviews --output data/human_reviewed_v1`. This selects the latest approval per query, validates provenance, and produces queries.csv, chunks.csv, intents.csv, and review_registry.json. Back up these private review records separately because they are excluded from Git and the public demo container. Use the review registry when training with zero-span queries. At least ten distinct reviewed queries are required for even the transformer pilot.

We will review the difficult decisions together, but a suggestion from the assistant does not itself count as independent human approval. Work through a small pilot first, resolve guideline disagreements, then annotate the remaining sample.
