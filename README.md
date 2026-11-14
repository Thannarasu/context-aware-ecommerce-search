# Context-Aware E-commerce Query Understanding

Phase 1 research prototype: rule-based and Gemini chunking, DistilBERT token classification, and raw/chunk-assisted intent classification.

Use Python 3.11 with `pip install -r requirements-phase1.txt`. The legacy requirements file is retained as historical environment information.

```
python -m unittest discover -s tests -v
python -m src.evaluation.audit_phase1 --output experiments/new_audit.json
python -m src.block1_chunking.train_transformer --pilot --prepare-only --output experiments/new_manifest
python -m src.block1_chunking.train_transformer --pilot --epochs 5 --splits experiments/new_manifest/splits.json --output experiments/new_transformer_run
python -m src.block2_intent.train_intent_classifier --allow-provisional --output-dir experiments/new_intent_pilot
python -m src.block2_intent.run_gemini_batch --limit 10 --output-dir experiments/new_gemini_run
python -m streamlit run app.py
```

Use new experiment paths; scripts reject overwriting prior runs. Remove `--allow-provisional` and pass `--intent-file` after independent human review. Pilot results are exploratory and must be identified as such in the thesis.

Set GEMINI_API_KEY and GEMINI_MODEL through environment variables or a local `.env`. Local rule-based inference does not require API access. Set CHUNK_MODEL_PATH to the transformer's saved `model` directory and INTENT_MODEL_PATH to a saved raw-query `.joblib` model. Missing artifacts produce an explicit unavailable message.

See `docs/phase1_completion_status.md`, `docs/phase1_annotation_guidelines.md`, `docs/phase1_cloud.md`, and `docs/thesis_submission_requirements.md` for scope, remaining work, deployment, and deadlines.
