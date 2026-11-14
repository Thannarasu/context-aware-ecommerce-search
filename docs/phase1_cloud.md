# Phase 1 container deployment

Status: deployment package prepared locally; no hosted application has been created.

## Updated hosting preference, 30 October 2026

User prefers GCP or Azure trial credits for the Phase 1 demonstration and a paid billing plan later for the complete project. Proposed first choice: GCP Cloud Run with the existing Docker container; Azure Container Apps is an alternative. No paid subscription upgrade or cloud resource has been created.

Official offer information checked 30 October: Google describes $300 trial credit over 90 days for eligible new accounts (https://cloud.google.com/signup-faqs); Azure advertises $200 for 30 days (https://azure.microsoft.com/en-in/pricing/details/container-apps/). Eligibility, available credits, and activation date must be checked in the actual account. Registry, build, storage, and network costs must also fit the trial; Gemini API usage has its own provider/account billing and is not automatically covered by hosting credits.

For Cloud Run, start benchmarking at 2 vCPU / 4 GiB memory, max instances 1, minimum instances 0, port 7860, and low concurrency. These are proposed settings, not measured sizing. Bind GEMINI_API_KEY from Secret Manager and configure model ID separately. Cloud Run supports WebSockets but applies request timeouts to them; Streamlit connections need reconnection handling and a deliberately configured timeout. Active connections consume billable resources. See https://docs.cloud.google.com/run/docs/triggering/websockets .

Build a versioned image with the trained artifacts, deploy a test revision, check health/latency and secrets, and only then publish the demonstration endpoint. Set budget alerts, check remaining credits, and remove or scale down demo resources after the review. Budget alerts do not impose a hard spending cap. Human annotation stays local in the review app; do not save approvals on ephemeral cloud container storage.

## Local validation

Install `requirements-phase1.txt` in an isolated Python 3.11 environment. Train artifacts first; never package a randomly initialized classification head as a trained model.

Set `CHUNK_MODEL_PATH` to the saved transformer model directory. Set `INTENT_MODEL_PATH` to the saved `E1_raw_query.joblib` artifact. These paths can point directly into experiment outputs for local development. For the container, copy the chosen model and tokenizer into `models/distilbert_chunker/` and the raw intent model into `models/intent_raw.joblib`. Package only artifacts from trusted runs. The Docker build copies the models directory but never raw datasets or experiment files.

```
python -m streamlit run app.py
docker build -t phase1-search .
docker run --rm -p 7860:7860 --env-file .env phase1-search
```

Open http://localhost:7860. The container health endpoint is `/_stcore/health`. Verify rule-based inference without an API key, saved intent predictions, transformer inference, Gemini success, and useful messages when model artifacts or credentials are missing. Benchmark CPU latency before selecting hosting resources.

## Hugging Face Docker Space

Use an account eligible to create a Docker Space; current plan eligibility and budget must be confirmed before provisioning. Set `sdk: docker` and `app_port: 7860` in the Space README YAML. Upload the application, src, Dockerfile, requirements-phase1.txt, and packaged model artifacts. Configure GEMINI_API_KEY as a secret and GEMINI_MODEL as a variable. Do not publish `.env`, raw datasets, or private annotations.

Reference: https://huggingface.co/docs/hub/spaces-sdks-docker

The CPU methods and Gemini calls run through a shared Python inference entry point. No separate API server is required for this Phase 1 Streamlit workflow. Outputs contain method, latency, chunk text/offsets, intent, and clarification fields. Intent artifacts and chunkers are separate choices; the current application uses the raw-query intent artifact for local chunkers. A trained chunk-assisted classifier must receive its matching enriched representation before it is added as a UI option.

## Release evidence

Record code commit, dependency versions, artifact hashes, model/input versions, cloud URL, health-check outcome, warm/cold latency, sample outputs, and quota/error behavior. Keep a local runnable build and a recorded demonstration. Final cloud verification is pending an actual build, selected hosting account, and model artifact validation.
