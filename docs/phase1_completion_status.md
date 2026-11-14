# Phase 1 implementation tracker

Started: 30 October 2026. Submission requirements: thesis_submission_requirements.md.

## Implemented in the first work batch

- Repaired preparation-script syntax error and undefined app call.
- Shared inference API with rule, Gemini, and local DistilBERT modes.
- Shared exact-span evaluator; invalid predictions count as false positives, missing query outputs retain false negatives.
- Removed gold-informed text alignment from Gemini inference; historical outputs remain unchanged.
- New Gemini runs require a new output directory and explicit model configuration.
- Reproducible corpus audit, source hashes, historical rescoring, and all-label support counts.
- BIO encoding/decoding, duplicate-safe pilot split generation, and DistilBERT training CLI with validation checkpoint selection and a single test evaluation.
- Intent training requires explicit opt-in to assistant pilot labels and saves trained models and input hashes.
- Container build definition, isolated direct dependency file, secret exclusions, and deployment instructions.
- Regression tests for repeated spans, failed queries, invalid offsets, BIO round trips, and duplicate query splits.

## Audit evidence

`experiments/phase1_audit_20261030.json` reproduces rule F1 = 0.8318584, Gemini day4 F1 = 0.4961240, day5 F1 = 0.4516129 on 28 development queries with 60 gold spans. This verifies historical numbers, not independent performance. Historical Gemini offsets may still reflect the old gold-informed alignment.

All 150 intent rows have annotator assistant_pilot_v0.1 despite reviewed status. They need human verification. Chunk gold has zero OCCASION and one PURPOSE span. Pilot splits therefore cannot validate all six labels.

## Remaining completion gates

Verification: 11 regression tests passed; all Python sources compiled. Local Streamlit health returned `ok`. Streamlit AppTest exercised a rule-based analysis with no application exceptions and loaded the annotation review screen without exceptions. Local transformer inference loaded the saved pilot weights and returned structured spans. Docker CLI was not available, so the container build has not been verified. No cloud deployment or fresh Gemini API evaluation was performed.

User clarified hosting preference as GCP/Azure trial credits for Phase 1, with paid billing considered for the complete project later. GCP Cloud Run is the proposed first choice; actual trial account eligibility and deployment remain pending. User and assistant will work through annotation guidance together. The local Annotation Review screen saves explicit human approvals separately; `src.data.export_human_reviews` exports approved records into a new training corpus with a complete-query registry.

First transformer execution: `phase1_distilbert_pilot_run1` stopped on unrepresentable gold boundaries. `phase1_distilbert_pilot_run2` completed three epochs on CPU using the explicit pilot-only mask policy. Split sizes: 16 train, 6 validation, 6 test. Original evaluation gold was retained; two unrepresentable spans were masked in token loss and logged in alignment_issues.json. Transformer test F1 = 0.0869565 versus rules = 0.8666667 on the six pilot test queries. Rules have prior development exposure, so this is not an independent head-to-head claim. No hyperparameters were selected based on this test result.

Intent pilot rerun `phase1_intent_pilot_20261030` reproduced E1 accuracy 0.789474 / macro-F1 0.563810 and E2 accuracy 0.763158 / macro-F1 0.577489. Saved artifacts were packaged locally for application testing. The interface labels their development provenance. These are not validated human-ground-truth results.

1. Independently review intent labels and expand complete chunk annotations, including rare classes and reviewed zero-span queries.
2. Freeze independent evaluation data, split manifest, annotation version, and method configurations before tuning.
3. Validate full transformer training/inference and save real run evidence; do not treat a prepare-only manifest as a trained model.
4. Improve rule and Gemini behavior on development data, then evaluate all methods on identical independent test queries.
5. Add transformer-predicted chunk intent experiments with out-of-fold training features; do not train the intent model on optimistic in-sample transformer outputs.
6. Verify container build/runtime and publish to an eligible selected hosting account; no cloud URL is live yet.
7. Author thesis in the supplied template, confirm programme/front matter, obtain mentor feedback, plagiarism report below 20%, and real candidate/supervisor signatures.

## Timeline

Thesis authoring update (30 October 2026): a six-chapter mentor-review draft is saved in `../phase1-thesis-draft/main.tex`, using the supplied report structure and logo. It includes the actual pilot results, a focused seven-paper bibliography, and an evidence appendix. Key metrics are snapshotted in the draft's `evidence/` folder. Programme wording, paper size, approved declaration/certificate wording, signatures, and plagiarism reporting remain pending. This is not a submitted or supervisor-approved thesis.

- 1 Nov: complete first draft to mentor.
- 5 Nov, 23:59 IST: LMS draft and plagiarism report.
- 13 Nov: panel comments.
- 18 Nov: corrected final thesis (submission time not yet specified).
- 20 Nov: Phase 1 final review and demonstration.

Final thesis evidence must be ready before 18 Nov; reserve 13-17 Nov for panel revisions and final verification.
