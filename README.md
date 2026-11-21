# OmniPulse MBS

An early-stage multimodal telemetry workspace based on the supplied Technical Specification 1.0. This repository contains a Next.js interface, a FastAPI API, baseline/rPPG processing primitives, hybrid knowledge retrieval, and optional local infrastructure services.

## Run locally

1. Copy `.env.example` to `.env`. Set `GROQ_API_KEY` to enable Groq Whisper Large v3 transcription; `GEMINI_API_KEY` is optional and enables aggregate-only report summaries. Knowledge Base embeddings and retrieval run locally and need no Gemini key. Keep keys in the repository-root `.env`, never in frontend environment variables.
2. Qdrant runs embedded with persistent local storage by default; no Docker service is needed. The embedded mode is intended for a single API process and a local knowledge base. Set `QDRANT_MODE=server` and start `docker compose up -d qdrant` only if you want a standalone server or multiple API workers. PostgreSQL, Redis, and MinIO remain optional scaffolding for the current local API.
3. In `backend/`, run `python -m venv .venv`, activate it, install `pip install -e ".[documents,signals,ai,inference,retrieval]"`, then run `uvicorn app.main:app --reload --port 8000`. The inference extra installs the Groq SDK and openSMILE; speech recognition runs remotely through Groq without local ASR model weights.
4. Copy `frontend/.env.example` to `frontend/.env.local`. In `frontend/`, run `npm install` and `npm run dev`. Open `http://localhost:3000`.

To build the API image from the repository root, run `docker build -f backend/Dockerfile -t omnipulse-api .`.

The API uses a local SQLite database by default. Dense knowledge retrieval uses the local FastEmbed `BAAI/bge-small-en-v1.5` model (384 dimensions) and embedded Qdrant; Knowledge Base uploads and searches do not call Gemini. The model downloads on first use and is cached in `backend/data/fastembed`. Gemini is used only for optional aggregate-only report summaries. Redis, MinIO, and PostgreSQL are not required by this API path. API docs are at `/docs`. The browser uses a random per-session capture token; `PROCESSOR_SHARED_SECRET` is only for the separate trusted-worker endpoints and is not needed for the browser capture path.
Gemini report summaries use `GEMINI_API_KEY` and the configurable `GEMINI_MODEL` (default `gemini-3.8-flash`). Gemini receives aggregate baseline comparisons only, without participant identifiers, transcripts, event flags, or raw media. It selects metric highlights in structured JSON; the API validates the selected metric names and renders factual comparisons from stored measurements. If the key is missing or Gemini is unavailable, reports still complete and retain their measured-data tables with a summary status indicating the fallback.
The live browser pipeline runs MediaPipe face, pose, and hand tasks locally. Put the three `.task` model files in `frontend/public/models/` as `face_landmarker.task`, `pose_landmarker.task`, and `hand_landmarker.task`; the Next.js app serves them locally. The MediaPipe WASM runtime still loads from jsDelivr. Camera frames never go to the API; only ROI color averages and derived visual measurements are sent for POS/CHROM rPPG and baseline processing. The browser sends short mono 16 kHz WAV windows to the configured API. openSMILE eGeMAPS runs on the API, and every fourth capture window the API forwards its rolling 8-second audio context to Groq's `whisper-large-v3` for transcription and word/segment timestamps. OmniPulse does not write raw audio to disk; audio is sent to Groq for this request and is subject to Groq's service terms. Camera access requires localhost or HTTPS.
To populate a local Knowledge Base from the supplied Markdown seeds, generated retrieval templates, and Europe PMC open-access abstracts, stop the API first (embedded Qdrant allows one process at a time), then run from `backend/`:

```powershell
.\.venv\Scripts\python.exe scripts\build_massive_kb.py
```

The builder is repeatable: it skips sources already stored, saves an ignored local backup at `backend/data/massive_kb_backup.json`, and reports the indexed totals. Use `--skip-europepmc` for an offline seed/template-only pass, `--dry-run` to count chunks without writing, or `--refresh-existing` to re-embed source records and refresh metadata. Seed material, generated templates, and imported abstracts are marked **UNVERIFIED**; generated templates are synthetic and are not empirical evidence. Review evidence before using it in research conclusions. You can also add documents through the Knowledge Base UI; uploads remain unverified until human review.
The PostgreSQL/TimescaleDB schema and local service stack are scaffolding; the running API currently persists to SQLite. Keep this development service bound to a trusted local environment. Authentication and production media encryption are not configured.

### Trusted processor contract

Set `PROCESSOR_SHARED_SECRET` in the repository-root `.env` and send it as `X-Processor-Token` from a trusted worker. After `POST /api/v1/sessions/{id}/calibrate`, complete calibration with measured samples (at least three finite samples for one feature):

```json
{
  "face_tracking_uptime_pct": 96.4,
  "duration_sec": 90,
  "baseline_features": {
    "hr_bpm": [71.8, 72.1, 72.0],
    "jitter_local": [0.011, 0.012, 0.010],
    "au04": [0.12, 0.14, 0.13]
  }
}
```

The API stores each feature's median, MAD, and sample count. The browser capture path computes baseline features and sends accepted telemetry windows to `/api/v1/sessions/{id}/capture/window`; client-supplied z-scores and composite indices are replaced with server-computed values. The processor-only `/telemetry` route remains available for trusted external workers. The server emits a cross-modal incongruence candidate only when its computed CMCI is below `0.55` in two consecutive window indices. Unsupported or absent values stay unavailable.

## What's implemented

- Dashboard, camera/microphone consent and pre-check wizard, 90-second baseline capture, browser-side MediaPipe face/pose/hand tracking, API-side Groq Whisper Large v3/openSMILE audio analysis, live telemetry, report explorer, and knowledge-base workspace.
- Session lifecycle and telemetry APIs, WebSocket telemetry delivery, structured JSON and CSV exports, and optional PDF rendering.
- Robust median/MAD normalization, cross-modal congruency calculations, EAR/gaze primitives, and an rPPG POS/CHROM estimator with explicit quality gates.
- Markdown, text, JSON, PDF, and DOCX knowledge ingestion (PDF/DOCX parsers are optional dependencies), hierarchical chunks, metadata, deduplication, a pseudoscience review gate, local FastEmbed vectors in embedded Qdrant, and lexical/dense hybrid retrieval with local reciprocal-rank fusion.
- Docker Compose definitions for TimescaleDB, Redis, Qdrant, and MinIO.

## Scope and safeguards

The visual and signal cards marked **Illustrative** use synthetic design-preview values, not captured human measurements. The live session path runs real MediaPipe tasks in the browser, local openSMILE feature extraction in the API process, and Groq Whisper Large v3 transcription when `GROQ_API_KEY` is configured. If an optional package is missing, a service is unavailable, or input quality is insufficient, the related value stays unavailable and its inference status is reported. This is a research telemetry pipeline: citation-grounded interpretation from reviewed evidence, general user authentication, encryption key management, and background GPU workers still require deployment-specific integration.

Outputs describe observable, baseline-relative signal changes only. They do not assess honesty, deception, guilt, or mental-health diagnoses. Optical heart-rate estimates are not medical measurements.
