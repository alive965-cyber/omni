"""FastAPI Gateway, WebRTC/WebSocket Coordinator, and v2.0 Human Behavioral Analyst Report
Orchestrator for OmniPulse MBS (backend/app/main.py).
Includes all 4 Major Human Behavioral Analyst Upgrades + all 6 Mathematical & Clinical Upgrades:
  1. Loudnorm Audio Normalization + Whisper ASR (Groq whisper-large-v3 + faster-whisper fallback) + Voiced Gate Fix
  2. `HumanSignalTranslator` plain-English translation for Spoken Words, Voice Emotion, Physical Actions, and Heartbeat/Breathing
  3. Moment-by-Moment 5-Row Human Behavioral Cards + 4-Part Final Comprehensive Behavioral Verdict
  4. Zero Non-Behavioral Camera Artifact Pollution (`illumination_change_pct`, `head_velocity_deg_s` stripped from flags & narratives)
"""
from __future__ import annotations

import asyncio
import csv
from datetime import datetime, timezone
import io
import json
import logging
from math import isfinite
import os
from pathlib import Path
import re
import secrets
from statistics import mean, pstdev
from threading import Lock, Thread
from typing import Any
import uuid
import wave

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from app.core.state_machine import RAGIngestionState, SessionState, transition
from app.rag.debate_synthesizer import OMITTED_RPPG_RECOVERY_STATEMENT
from app.rag.multimodal_clip_verifier import MultimodalClipVerifier
from app.schemas import (
    BrowserCaptureWindow,
    CalibrationComplete,
    SessionCreate,
    TelemetryIngest,
    TelemetryWindow,
    WebRTCOffer,
)
from app.services import audio_inference, knowledge_retrieval, repository
from app.services.gemini_report import generate_report_narrative
from app.services.human_signal_translator import HumanSignalTranslator
from app.services.topic_shift_analyzer import TopicBehavioralShiftAnalyzer
from app.services.video_evidence_snapshotter import VideoEvidenceSnapshotter
from app.workers.rule_of_three_filter import BLACKLISTED_ARTIFACT_KEYS, RuleOfThreeConvergenceFilter
from app.workers.voice_emotion_classifier import HybridVoiceEmotionClassifier
from app.services.knowledge import (
    ALLOWED_EVIDENCE_TIERS,
    ALLOWED_MODALITY_TAGS,
    MAX_UPLOAD_BYTES,
    prepare_document,
)
from app.services.report_enhancer import (
    NON_BEHAVIORAL_CAMERA_METRICS,
    OmniPulseReportEnhancer,
    VOICED_ACOUSTIC_COLS,
)
from app.workers.adaptive_baseline import DualHorizonBaselineEngine, MIN_PHYSIOLOGICAL_SCALE
from app.workers.fusion_synchronizer import FEATURE_ALIASES, enrich_window, robust_baseline
from app.workers.motif_discovery import PersonalMotifMiner
from app.workers.rapport_entrainment import RapportEntrainmentAnalyzer
from app.workers.rppg_extractor import estimate_from_rgb
from app.workers.stimulus_response_tracker import DyadicStimulusTracker

app = FastAPI(
    title="OmniPulse MBS API",
    version="2.0.0-PROD",
    description="Observable, baseline-relative trimodal + rPPG + shoulder-respiration behavioral telemetry and Human Behavioral Analyst platform.",
)
logger = logging.getLogger(__name__)
origins = [os.getenv("FRONTEND_ORIGIN", "http://localhost:3000"), "http://127.0.0.1:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_buffer_lock = Lock()
_SESSION_RGB_BUFFERS: dict[str, list[tuple[float, float, float, float]]] = {}
_SESSION_AUDIO_BUFFERS: dict[str, list[Any]] = {}


@app.on_event("startup")
def startup():
    repository.init_db()
    Thread(target=_bootstrap_seed_knowledge_base, name="kb-seed-bootstrap", daemon=True).start()


def _bootstrap_seed_knowledge_base():
    """Auto-seed Tier-1 Markdown manuals and backfill Qdrant vectors on startup."""
    try:
        if not repository.list_documents():
            seed_dir = Path(__file__).resolve().parents[2] / "knowledge_base_seed"
            if seed_dir.is_dir():
                for md_path in sorted(seed_dir.glob("*.md")):
                    raw = md_path.read_bytes()
                    doc, chunks = prepare_document(md_path.name, raw, ["GENERAL"], "TIER_1_CANONICAL_RULE")
                    if doc["status"] == "INDEXED" and chunks and not repository.get_document(doc["id"]):
                        repository.insert_document(doc, chunks)
        if knowledge_retrieval.configuration_ready():
            knowledge_retrieval.index_pending_documents(repository)
    except Exception as exc:
        logger.warning("Knowledge base startup bootstrap note: %s", type(exc).__name__)


@app.on_event("shutdown")
def shutdown_knowledge_index():
    knowledge_retrieval.close()


def require_session(session_id: str) -> dict[str, Any]:
    session = repository.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return session


def require_processor_token(provided: str | None) -> None:
    expected = os.getenv("PROCESSOR_SHARED_SECRET", "")
    if not expected:
        raise HTTPException(
            503,
            "Telemetry processors are not configured. Set PROCESSOR_SHARED_SECRET after deploying a trusted worker.",
        )
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(403, "Trusted processor credentials are required.")


def require_capture_token(session_id: str, provided: str | None) -> None:
    expected = repository.get_capture_token(session_id)
    if not expected:
        raise HTTPException(404, "Session not found")
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(403, "This session's capture token is required.")


def move(session_id: str, requested: SessionState, **updates) -> dict[str, Any]:
    session = require_session(session_id)
    try:
        state = transition(session["status"], requested)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return repository.update_session(session_id, status=state, **updates)


def normalized_baselines(session_id: str) -> dict[str, Any]:
    record = repository.get_baselines(session_id)
    if not record or not record.get("features"):
        raise HTTPException(409, "Trusted calibration baselines are missing for this session.")
    return record


def _format_mmss(sec: float) -> str:
    sec_i = max(0, int(round(float(sec))))
    return f"{sec_i // 60:02d}:{sec_i % 60:02d}"


def prepare_telemetry_window(session_id: str, window: TelemetryWindow) -> dict[str, Any]:
    baseline_record = normalized_baselines(session_id)
    masked_dict = OmniPulseReportEnhancer.mask_unvoiced_window_dicts([window.model_dump()])[0]
    payload = enrich_window(masked_dict, baseline_record["features"])
    current_cmci = (payload.get("composite_indices") or {}).get("cross_modal_congruency_index")
    if isinstance(current_cmci, (int, float)) and current_cmci < 0.55:
        previous_index = payload["window_index"] - 1
        previous = next(
            (item for item in reversed(repository.get_windows(session_id)) if item.get("window_index") == previous_index),
            None,
        )
        previous_cmci = ((previous or {}).get("composite_indices") or {}).get("cross_modal_congruency_index")
        if isinstance(previous_cmci, (int, float)) and previous_cmci < 0.55:
            if "CROSS_MODAL_INCONGRUENCE" not in payload["behavioral_flags"]:
                payload["behavioral_flags"].append("CROSS_MODAL_INCONGRUENCE")
    payload["live_copilot_probe"] = OmniPulseReportEnhancer.generate_live_copilot_probe(payload)
    return payload


def complete_session_calibration(session_id: str, body: CalibrationComplete) -> dict[str, Any]:
    session = require_session(session_id)
    if session["status"] != SessionState.CALIBRATING.value:
        raise HTTPException(409, "Calibration can only be completed while the session is calibrating.")
    if body.face_tracking_uptime_pct < 75:
        return move(session_id, SessionState.CALIBRATION_RETRY_REQUIRED)
    aliases = {alias: feature for feature, names in FEATURE_ALIASES.items() for alias in names}
    baselines: dict[str, dict[str, float | int]] = {}
    for name, values in body.baseline_features.items():
        if not name or len(name) > 64 or not name[0].islower() or not all(char.islower() or char.isdigit() or char == "_" for char in name):
            continue
        feature = aliases.get(name, name)
        if feature in NON_BEHAVIORAL_CAMERA_METRICS:
            continue
        if feature in VOICED_ACOUSTIC_COLS:
            cleaned_values = [v for v in values if v is not None and float(v) > 1e-4]
        else:
            cleaned_values = values
        baseline = robust_baseline(cleaned_values)
        if baseline and baseline["sample_count"] >= 3:
            baselines[feature] = baseline
    if not baselines:
        raise HTTPException(422, "Calibration needs at least three finite baseline samples for one supported feature.")
    repository.put_baselines(
        session_id,
        {
            "features": baselines,
            "face_tracking_uptime_pct": body.face_tracking_uptime_pct,
            "duration_sec": body.duration_sec,
        },
    )
    existing_windows = repository.get_windows(session_id)
    if existing_windows:
        masked_cal = OmniPulseReportEnhancer.mask_unvoiced_window_dicts(existing_windows)
        enriched_cal = [enrich_window(dict(w), baselines) for w in masked_cal]
        repository.put_windows_bulk(session_id, enriched_cal)

    row = move(session_id, SessionState.RECORDING_LIVE, calibration_duration_sec=body.duration_sec)
    return {
        **row,
        "signal_processing": "server-computed robust calibration baselines (unvoiced zeros masked)",
        "baseline_features_available": sorted(baselines),
    }


def decode_capture_wav(raw: bytes):
    if len(raw) > 1_000_000:
        raise HTTPException(413, "A capture audio window must be under 1 MB.")
    try:
        with wave.open(io.BytesIO(raw), "rb") as audio_file:
            channels = audio_file.getnchannels()
            sample_width = audio_file.getsampwidth()
            sample_rate = audio_file.getframerate()
            frames = audio_file.readframes(audio_file.getnframes())
        if channels != 1 or sample_width != 2 or sample_rate != 16000:
            raise ValueError("Expected mono, 16-bit PCM audio at 16000 Hz.")
        import numpy as np

        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        if not len(samples) or len(samples) > 160000:
            raise ValueError("Audio window must be between 0 and 10 seconds.")
        return samples
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, "Audio must be a valid mono 16-bit 16000 Hz WAV window.") from exc


def add_rppg(session_id: str, capture: BrowserCaptureWindow, window_payload: dict[str, Any]) -> None:
    """Maintain a session-level rolling 8.5s (256-frame) RGB sliding buffer stepped every 2.0s."""
    incoming = capture.rgb_samples
    with _buffer_lock:
        buf = _SESSION_RGB_BUFFERS.setdefault(session_id, [])
        last_ts = buf[-1][0] if buf else -1.0
        for s in incoming:
            if float(s.timestamp_sec) > last_ts:
                buf.append((float(s.timestamp_sec), float(s.red), float(s.green), float(s.blue)))
                last_ts = float(s.timestamp_sec)
        if buf:
            latest_t = buf[-1][0]
            buf[:] = [item for item in buf if (latest_t - item[0]) <= 9.0]
        active_slice = list(buf)

    if len(active_slice) < 90 or (active_slice[-1][0] - active_slice[0][0]) < 4.0:
        window_payload.setdefault("confidence", {})["rppg_status"] = "UNRELIABLE_INSUFFICIENT_FRAMES"
        window_payload["confidence"]["rppg_sqi"] = 0.0
        return

    try:
        import numpy as np

        timestamps = [item[0] for item in active_slice]
        rgb = np.asarray([[item[1], item[2], item[3]] for item in active_slice], dtype=float)
        result = estimate_from_rgb(
            rgb,
            timestamps_sec=timestamps,
            sample_rate=30.0,
            head_velocity_deg_s=float(window_payload.get("metrics", {}).get("head_velocity_deg_s") or 0.0),
            illumination_change_pct=float(window_payload.get("metrics", {}).get("illumination_change_pct") or 0.0),
        )
        window_payload.setdefault("confidence", {})["rppg_status"] = result.status
        window_payload["confidence"]["rppg_sqi"] = result.sqi
        window_payload["confidence"]["rppg_confidence_tier"] = result.confidence_tier
        if result.hr_bpm is not None:
            window_payload.setdefault("metrics", {})["hr_bpm"] = result.hr_bpm
            window_payload["metrics"]["hr_bpm_pos"] = result.hr_pos_bpm
            window_payload["metrics"]["hr_bpm_chrom"] = result.hr_chrom_bpm
            window_payload["metrics"]["rppg_sqi"] = result.sqi
    except ImportError:
        window_payload.setdefault("confidence", {})["rppg_status"] = "PROCESSOR_UNAVAILABLE"


def model_audio_features(audio_samples, asr_samples, *, transcribe_audio: bool) -> dict[str, Any]:
    try:
        return audio_inference.analyze_audio(
            audio_samples,
            16000,
            transcribe=transcribe_audio,
            transcription_audio=asr_samples,
            language="en",
        )
    except ImportError:
        return {
            "metrics": {},
            "transcript": None,
            "transcript_segments": [],
            "acoustic_status": "unavailable",
            "asr_status": "unavailable",
        }


def report_metric_comparisons(session_id: str) -> list[dict[str, Any]]:
    """Section 3 Table: Masks unvoiced 0.0 values on voiced acoustic features, excludes non-behavioral
    camera artifacts (`illumination_change_pct`, `head_velocity_deg_s`), and clamps Peak |Z| <= 8.00.
    """
    raw_windows = repository.get_windows(session_id)
    windows = OmniPulseReportEnhancer.mask_unvoiced_window_dicts(raw_windows)
    baseline_record = repository.get_baselines(session_id) or {}
    baseline_features = baseline_record.get("features", {})
    dual_floor_features = ((baseline_record.get("dual_horizon") or {}).get("floor_features")) or {}
    aliases = {alias: feature for feature, names in FEATURE_ALIASES.items() for alias in names}
    comparisons = []

    for feature, baseline in baseline_features.items():
        if feature in NON_BEHAVIORAL_CAMERA_METRICS:
            continue
        observations: list[tuple[float, float]] = []
        for window in windows:
            metrics = window.get("metrics") or {}
            value = next(
                (
                    metrics.get(alias)
                    for alias, canonical in aliases.items()
                    if canonical == feature and isinstance(metrics.get(alias), (int, float)) and not isinstance(metrics.get(alias), bool)
                ),
                metrics.get(feature),
            )
            if isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(float(value)):
                if feature in VOICED_ACOUSTIC_COLS and float(value) <= 1e-4:
                    continue
                observations.append((float(value), float(window.get("start_sec", 0.0))))

        if observations:
            obs_vals = [item[0] for item in observations]
            floor_stat = dual_floor_features.get(feature) or {}
            base_med = float(floor_stat.get("median", baseline.get("median", 0.0)) or 0.0)
            base_mad = float(floor_stat.get("mad", baseline.get("mad", 0.0)) or 0.0)
            session_std = float(pstdev(obs_vals)) if len(obs_vals) > 2 else 0.0
            min_scale = float(MIN_PHYSIOLOGICAL_SCALE.get(feature, 0.05))
            safe_scale = max(1.4826 * base_mad, 0.5 * session_std, min_scale)

            peaks = []
            for val, start_s in observations:
                clamped_z = min(8.0, abs(val - base_med) / safe_scale)
                peaks.append((round(clamped_z, 2), val, start_s))

            peak_item = max(peaks, default=None, key=lambda item: item[0])
            comparisons.append({
                "metric_name": feature,
                "baseline_median": round(base_med, 4),
                "baseline_mad": round(base_mad, 4),
                "sample_count": len(obs_vals),
                "session_mean": round(mean(obs_vals), 4),
                "peak_anomaly_value": round(peak_item[1], 4) if peak_item else None,
                "peak_timestamp_sec": peak_item[2] if peak_item else None,
                "peak_absolute_z_score": min(8.0, round(peak_item[0], 2)) if peak_item else 0.0,
            })
    return comparisons


def _run_v2_cognitive_pipeline_and_synthesize(session_id: str) -> dict[str, Any]:
    """Execute the full OmniPulse v2.0 Pipeline + HumanSignalTranslator + OmniPulseReportEnhancer."""
    session = require_session(session_id)
    windows = repository.get_windows(session_id)
    baseline_record = repository.get_baselines(session_id) or {}
    key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro").strip() or "gemini-2.5-pro"

    if not windows:
        result = {
            "status": "skipped_no_measurements",
            "provider": "gemini",
            "model": model,
            "executive_summary": {
                "overview": "No telemetry was supplied, so a Gemini report summary was not generated.",
                "dominant_patterns": [],
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        repository.put_report_narrative(session_id, result)
        return result

    # Upgrade 1: Mask only truly silent windows (where f0 <= 1.0 AND no transcript exists)
    windows = OmniPulseReportEnhancer.mask_unvoiced_window_dicts(windows)

    # Upgrade 1: Full-session normalized Whisper transcription with language="en" & word-timestamp mapping
    with _buffer_lock:
        audio_chunks = _SESSION_AUDIO_BUFFERS.pop(session_id, [])
        _SESSION_RGB_BUFFERS.pop(session_id, None)
    if audio_chunks:
        try:
            import numpy as np

            full_audio = np.concatenate(audio_chunks)
            windows = audio_inference.transcribe_full_session_and_map_windows(
                full_audio, 16000, windows, language="en"
            )
        except Exception as exc:
            logger.error("Full-session ASR word mapping error: %s: %s", type(exc).__name__, exc)
            print(f"[OmniPulse ASR Error] {type(exc).__name__}: {exc}", flush=True)
    else:
        windows = audio_inference.transcribe_full_session_and_map_windows(None, 16000, windows, language="en")

    # 1. Dual-Horizon Adaptive Baseline (with non-behavioral camera metrics excluded)
    enriched_windows, dual_horizon_meta = DualHorizonBaselineEngine.compute_dual_horizon_zscores(windows)

    # 2. Dyadic Stimulus-Response Turns & 3-Phase Annotation (voiced/transcribed windows always VERBAL_EXECUTION)
    turns = DyadicStimulusTracker.infer_turns_from_windows(enriched_windows)
    enriched_windows = DyadicStimulusTracker.annotate_windows_with_turn_phases(enriched_windows, turns)

    # 3. Rapport & Vocal Entrainment
    entrainment_summary = RapportEntrainmentAnalyzer.summarize_session_entrainment(enriched_windows)
    turn_aci_map = {t["turn_index"]: t["vocal_entrainment_aci"] for t in entrainment_summary.get("turn_metrics") or []}
    for w in enriched_windows:
        t_id = w.get("turn_id")
        if t_id is not None and t_id in turn_aci_map:
            w["vocal_entrainment_aci"] = turn_aci_map[t_id]

    # 4. Shoulder-Kinematics Respiration & Sigh Detector + Sawtooth Gaze Reading Detector
    respiration_summary = OmniPulseReportEnhancer.extract_session_respiration_from_windows(enriched_windows)
    gaze_reading_analysis = OmniPulseReportEnhancer.evaluate_session_gaze_integrity(enriched_windows)
    dual_horizon_meta["respiration_summary"] = respiration_summary
    dual_horizon_meta["gaze_reading_analysis"] = gaze_reading_analysis

    # 4B. Part 4 Step 1 & Step 2: Rule-of-3 Multi-Channel Convergence Filter + Hybrid Voice Emotion Classifier
    sigh_ts = respiration_summary.get("detected_sigh_timestamps_sec") or []
    apnea_wins = respiration_summary.get("apnea_freeze_windows") or []
    for w in enriched_windows:
        passes_r3, mccs_val, clean_z, active_chans = RuleOfThreeConvergenceFilter.sanitize_and_evaluate_window(
            w, sigh_ts, apnea_wins
        )
        w["passes_rule_of_3"] = passes_r3
        w["mccs_score"] = mccs_val
        w["active_biological_channels"] = active_chans
        if isinstance(w.get("dual_horizon_z_scores"), dict):
            for bk in list(w["dual_horizon_z_scores"].keys()):
                if bk in BLACKLISTED_ARTIFACT_KEYS or bk.replace("_z_floor", "") in BLACKLISTED_ARTIFACT_KEYS:
                    w["dual_horizon_z_scores"].pop(bk, None)
        ve_res = HybridVoiceEmotionClassifier.analyze_window_voice_emotion(
            None,
            float(w.get("start_sec", 0.0)),
            float(w.get("end_sec", 2.0)),
            {**(w.get("metrics") or {}), "transcript_segment": w.get("transcript_segment")},
            clean_z,
        )
        w["dominant_vocal_emotion"] = ve_res["dominant_vocal_emotion"]
        w["vocal_emotion_confidence"] = ve_res["vocal_emotion_confidence"]
        w["emotion_probabilities"] = ve_res["emotion_probabilities"]
        w["human_voice_summary"] = ve_res["human_voice_summary"]

    # 4C. Part 4 Step 4: Build Topic-by-Topic Behavioral Shift Matrix
    topic_shift_matrix = TopicBehavioralShiftAnalyzer.build_from_window_dicts(
        enriched_windows,
        dual_horizon_meta.get("hr_bpm_pos_median"),
        sigh_ts,
        apnea_wins,
        num_segments=4,
    )
    dual_horizon_meta["topic_shift_matrix"] = topic_shift_matrix

    # 5. Personal Behavioral Motif Mining
    discovered_motifs = PersonalMotifMiner.discover_personal_motifs(enriched_windows)

    # Persist updated windows, baseline floor metadata, turns, and motifs
    baseline_record["dual_horizon"] = dual_horizon_meta
    repository.put_baselines(session_id, baseline_record)
    repository.put_windows_bulk(session_id, enriched_windows)
    repository.put_dyadic_turns(session_id, turns)
    repository.put_personal_motifs(session_id, discovered_motifs)

    # 6. Select Candidate Anomaly Windows, Coalesce into 3-6 Behavioral Episodes, & Run Stratified 3-Tier RAG
    verifier = MultimodalClipVerifier(gemini_api_key=key, model=model)
    retriever = knowledge_retrieval.StratifiedBehavioralRetriever()
    turns_by_idx = {t["turn_index"]: t for t in turns}
    base_hr = dual_horizon_meta.get("hr_bpm_pos_median")

    candidate_anomalies: list[dict[str, Any]] = []
    for w in enriched_windows:
        flags = list(w.get("behavioral_flags") or [])
        z_scores = w.get("dual_horizon_z_scores") or w.get("baseline_z_scores") or {}
        max_z = max(
            (
                abs(float(v))
                for k, v in z_scores.items()
                if isinstance(v, (int, float))
                and not str(k).endswith("_phasic")
                and str(k).replace("_z_floor", "").replace("_z", "") not in NON_BEHAVIORAL_CAMERA_METRICS
            ),
            default=0.0,
        )
        is_incongruent = "CROSS_MODAL_INCONGRUENCE" in flags
        passes_r3 = bool(w.get("passes_rule_of_3", False))
        has_speech = bool(str(w.get("transcript_segment") or "").strip())
        if not passes_r3 and not is_incongruent:
            continue

        verification = verifier.verify_anomaly_clip(None, w)
        if verification.get("is_physical_artifact") and float(verification.get("artifact_probability", 0.0)) >= 0.65:
            if "BENIGN_PHYSICAL_ARTIFACT" not in flags:
                flags.append("BENIGN_PHYSICAL_ARTIFACT")
            w["behavioral_flags"] = flags
            if not is_incongruent and not has_speech:
                continue

        turn_obj = turns_by_idx.get(w.get("turn_id")) or {}
        fmt_time = f"{_format_mmss(w['start_sec'])} - {_format_mmss(w['end_sec'])}"
        metrics = w.get("metrics") or {}
        ev_cat = (
            "CROSS_MODAL_INCONGRUENCE"
            if is_incongruent
            else (
                "AUTONOMIC_AROUSAL_SPIKE"
                if "AUTONOMIC_AROUSAL_SPIKE" in flags
                else ("PACIFYING_ADAPTOR_CLUSTER" if "PACIFYING_NECK_TOUCH" in flags else "HIGH_COGNITIVE_LOAD")
            )
        )
        candidate_anomalies.append({
            "event_id": f"event-{w['window_index']}",
            "window_index": w["window_index"],
            "turn_index": w.get("turn_id", 0),
            "start_sec": float(w["start_sec"]),
            "end_sec": float(w["end_sec"]),
            "time_range_sec": [float(w["start_sec"]), float(w["end_sec"])],
            "formatted_time": fmt_time,
            "event_category": ev_cat,
            "confidence_score": float((w.get("confidence") or {}).get("overall") or 0.88),
            "trigger_stimulus_quote": turn_obj.get("interviewer_transcript") or "Conversational prompt transition",
            "appraisal_latency_ms": turn_obj.get("appraisal_latency_ms", metrics.get("response_latency_ms", 0.0)),
            "transcript_excerpt": w.get("transcript_segment"),
            "subject_response_quote": w.get("transcript_segment") or "",
            "dominant_vocal_emotion": w.get("dominant_vocal_emotion"),
            "vocal_emotion_confidence": w.get("vocal_emotion_confidence"),
            "human_voice_summary": w.get("human_voice_summary"),
            "mccs_score": w.get("mccs_score", 0.0),
            "active_biological_channels": w.get("active_biological_channels", []),
            "micro_sequence_chain": w.get("micro_sequence_chain") or "STEADY_BASELINE",
            "keyframe_jpeg_base64": w.get("keyframe_jpeg_base64"),
            "multimodal_clip_verification": verification,
            "behavioral_flags": flags,
            "measurements": metrics,
            "metrics": metrics,
            "baseline_z_scores": w.get("baseline_z_scores") or {},
            "dual_horizon_z_scores": w.get("dual_horizon_z_scores") or {},
            "phasic_attribution_state": w.get("phasic_attribution_state", "ACUTE_STIMULUS_REACTION"),
            "cmci_score": (w.get("composite_indices") or {}).get("cross_modal_congruency_index", 0.5),
            "multimodal_evidence": {
                "visual_facs": (
                    f"AU24={metrics.get('au24_lip_presser', 'N/A')}, AU04={metrics.get('au04_brow_lowerer', 'N/A')}, "
                    f"BlinkRate={metrics.get('blink_rate_per_min', 'N/A')}/min, GazeAversion={metrics.get('gaze_aversion_ratio', 'N/A')}"
                ),
                "kinesics": (
                    f"SelfTouchRatio={metrics.get('self_touch_ratio', 'N/A')}, "
                    f"HandKE={metrics.get('hand_kinetic_energy', 'N/A')}, Gesture={metrics.get('gesture_type', 'NONE')}"
                ),
                "acoustics": (
                    f"F0={metrics.get('f0_semitone_mean', 'unvoiced')} st, Jitter={metrics.get('jitter_local_mean', metrics.get('jitter_local', 'unvoiced'))}, "
                    f"Shimmer={metrics.get('shimmer_local_db_mean', metrics.get('shimmer_db', 'unvoiced'))} dB, PauseRatio={metrics.get('pause_ratio', 'N/A')}"
                ),
                "rppg_physiology": (
                    f"HR={metrics.get('hr_bpm', 'unavailable')} BPM | "
                    f"Resp={respiration_summary.get('respiration_rate_breaths_per_min', 'N/A')} br/min"
                ),
            },
        })

    coalesced_episodes = DualHorizonBaselineEngine.coalesce_anomaly_episodes(
        candidate_anomalies,
        max_gap_sec=2.5,
        max_episode_duration_sec=12.0,
        max_episodes=6,
    )
    enriched_anomalies = retriever.enrich_session_anomalies_batch(coalesced_episodes)

    total_duration = max((float(w["end_sec"]) for w in enriched_windows), default=0.0)
    session_meta = {
        "session_id": session_id,
        "subject_identifier": session["subject_identifier"],
        "duration_sec": total_duration,
    }
    comparisons = report_metric_comparisons(session_id)

    try:
        narrative = generate_report_narrative(
            comparisons,
            api_key=key,
            model=model,
            session_metadata=session_meta,
            baseline_metadata=dual_horizon_meta,
            discovered_motifs=discovered_motifs,
            entrainment_summary=entrainment_summary,
            verified_anomaly_bundles=enriched_anomalies,
        )
    except Exception as exc:
        logger.warning("Report synthesis fallback triggered (%s).", type(exc).__name__)
        narrative = generate_report_narrative(
            comparisons,
            api_key=None,
            model=model,
            session_metadata=session_meta,
            baseline_metadata=dual_horizon_meta,
            discovered_motifs=discovered_motifs,
            entrainment_summary=entrainment_summary,
            verified_anomaly_bundles=enriched_anomalies,
        )

    cog_report = narrative.get("cognitive_v2_report") or {}
    debated_by_id = {ev["event_id"]: ev for ev in (cog_report.get("debated_events") or [])}

    final_events: list[dict[str, Any]] = []
    for bundle in enriched_anomalies:
        ev_id = bundle["event_id"]
        deb = debated_by_id.get(ev_id, {})
        chunks = bundle.get("all_retrieved_chunks") or []
        rag_citations = [
            {
                "doc_id": c["doc_id"],
                "citation": f"{', '.join(c.get('authors') or ['OmniPulse'])} ({c.get('publication_year', 2025)}) - {c.get('source_title')}",
                "evidence_tier": c.get("evidence_tier"),
                "relevance_score": c.get("hybrid_relevance_score", 0.85),
                "excerpt": c.get("content_text", ""),
            }
            for c in chunks
        ]
        # Upgrade 2 & 3: Translate raw telemetry into the 5 Human Behavioral Analyst rows
        human_rows = HumanSignalTranslator.build_human_moment_breakdown(
            bundle, baseline_hr=base_hr, resp_summary=respiration_summary
        )
        verdict = deb.get("epistemic_judge_verdict") or deb.get("moment_analysis") or "Multimodal shift relative to personal baseline; evaluate alongside situational context."
        skeptic = deb.get("skeptic_case") or "Consider lexical retrieval effort, respiratory speech modulation, or posture adjustment."
        advocate = deb.get("advocate_case") or "Convergent baseline deviation across monitored signal channels."
        quote_str = (
            deb.get("subject_response_quote")
            or bundle.get("subject_response_quote")
            or bundle.get("transcript_excerpt")
            or ""
        )
        if str(quote_str).lower().startswith("silent evaluation"):
            quote_str = ""

        # Part 4 Step 3: Capture Annotated Video Snapshot Evidence with Apex Millisecond & HUD Overlay
        snap_info = VideoEvidenceSnapshotter().capture_annotated_episode_thumbnail(
            None,
            {**bundle, **human_rows, "episode_id": deb.get("episode_id") or ev_id},
        )
        keyframe_b64 = snap_info["image_data_uri"] or OmniPulseReportEnhancer.generate_annotated_keyframe_b64(bundle)
        copilot_probe = OmniPulseReportEnhancer.generate_live_copilot_probe(bundle)
        voice_summary_str = bundle.get("human_voice_summary") or deb.get("voice_tone_and_emotion") or human_rows["voice_tone_and_emotion"]
        micro_seq_raw = bundle.get("micro_sequence_chain") or "STEADY_BASELINE"
        micro_seq_plain = HumanSignalTranslator.translate_micro_sequence_plain(micro_seq_raw)

        final_events.append({
            **bundle,
            "transcript_excerpt": quote_str or human_rows["what_was_spoken"],
            "subject_response_quote": quote_str or human_rows["what_was_spoken"],
            "episode_id": deb.get("episode_id") or ev_id,
            "time_range": deb.get("time_range") or bundle.get("formatted_time", ""),
            "headline_state_label": deb.get("headline_state_label") or human_rows.get("headline_state_label", "Focused Conversational Moment"),
            "what_was_spoken": deb.get("what_was_spoken") or human_rows["what_was_spoken"],
            "voice_tone_and_emotion": voice_summary_str,
            "voice_and_vocal_emotion": voice_summary_str,
            "physical_actions_observed": deb.get("physical_actions_observed") or deb.get("physical_actions_and_expression") or human_rows["physical_actions_observed"],
            "physical_actions_and_expression": deb.get("physical_actions_and_expression") or deb.get("physical_actions_observed") or human_rows["physical_actions_and_expression"],
            "heartbeat_and_breathing": deb.get("heartbeat_and_breathing") or human_rows["heartbeat_and_breathing"],
            "human_expert_moment_analysis": deb.get("human_expert_moment_analysis") or deb.get("moment_analysis") or verdict,
            "moment_analysis": deb.get("moment_analysis") or deb.get("human_expert_moment_analysis") or verdict,
            "advocate_vs_skeptic_check": deb.get("advocate_vs_skeptic_check") or f"Advocate: {advocate} Skeptic: {skeptic}",
            "calibrated_confidence": deb.get("calibrated_confidence", deb.get("calibrated_epistemic_confidence", 0.68)),
            "grounded_rag_citations": deb.get("grounded_rag_citations") or deb.get("rag_citation_ids") or bundle.get("allowed_citation_doc_ids", []),
            "micro_sequence_plain": micro_seq_plain,
            "apex_timestamp_sec": snap_info["apex_timestamp_sec"],
            "action_label": snap_info["action_label"],
            "image_data_uri": keyframe_b64,
            "keyframe_jpeg_base64": keyframe_b64,
            "live_copilot_probe": copilot_probe,
            "advocate_case": advocate,
            "advocate_strength_score": deb.get("advocate_strength_score", 0.72),
            "skeptic_case": skeptic,
            "skeptic_strength_score": deb.get("skeptic_strength_score", 0.42),
            "epistemic_judge_verdict": verdict,
            "calibrated_epistemic_confidence": deb.get("calibrated_epistemic_confidence", 0.68),
            "rag_citation_ids": deb.get("rag_citation_ids") or bundle.get("allowed_citation_doc_ids", []),
            "rag_citations": rag_citations,
            "citations": rag_citations,
            "scientific_rag_interpretation": verdict,
            "interpretation": verdict,
            "alternative_non_psychological_explanations": skeptic,
            "alternative_explanation": skeptic,
        })

    repository.put_session_events(session_id, final_events)
    narrative["generated_at"] = datetime.now(timezone.utc).isoformat()
    repository.put_report_narrative(session_id, narrative)
    return narrative


@app.get("/api/v1/health")
def health():
    audio_capabilities = audio_inference.capabilities()
    return {
        "status": "ok",
        "service": "omnipulse-api",
        "version": "2.0.0-PROD",
        "storage": "local-sqlite",
        "browser_vision": "mediapipe-client-side",
        "audio_inference": audio_capabilities,
        "groq_asr_configured": bool(audio_capabilities["groq_sdk_installed"] and audio_capabilities["groq_api_configured"]),
        "groq_asr_model": audio_capabilities["groq_model"],
        "telemetry_processor_configured": bool(os.getenv("PROCESSOR_SHARED_SECRET")),
        "gemini_report_configured": bool(os.getenv("GEMINI_API_KEY") or os.getenv("GROQ_API_KEY")),
        "cognitive_v2_modules_active": True,
        "human_signal_translator_active": True,
        "report_enhancer_active": True,
        "raw_video_upload": False,
        "raw_audio_retained": False,
    }


@app.post("/api/v1/sessions", status_code=201)
def create_session(body: SessionCreate):
    if not body.consent_verified:
        raise HTTPException(400, "Explicit multimodal consent is required before creating a session.")
    session_id = str(uuid.uuid4())
    row = repository.create_session(session_id, body.model_dump())
    return {
        "id": row["id"],
        "subject_identifier": row["subject_identifier"],
        "session_title": row["session_title"],
        "status": row["status"],
        "capture_token": row["capture_token"],
        "webrtc_ice_servers": [{"urls": ["stun:stun.l.google.com:19302"]}],
        "privacy": {
            "raw_video_uploaded": False,
            "audio_sent_to_configured_api_for_transient_processing": True,
            "raw_audio_retained": False,
        },
    }


@app.post("/api/v1/sessions/{session_id}/webrtc-offer")
def webrtc_offer(session_id: str, body: WebRTCOffer):
    session = require_session(session_id)
    return {
        "session_id": session_id,
        "status": session["status"],
        "type": "answer",
        "sdp": body.sdp,
        "transport": "hybrid-webrtc-canvas-capture",
    }


@app.get("/api/v1/sessions")
def sessions():
    return repository.list_sessions()


@app.get("/api/v1/sessions/{session_id}")
def session_status(session_id: str):
    return require_session(session_id)


@app.post("/api/v1/sessions/{session_id}/calibrate")
def begin_calibration(session_id: str):
    return move(session_id, SessionState.CALIBRATING)


@app.post("/api/v1/sessions/{session_id}/calibration/complete")
def complete_calibration(session_id: str, body: CalibrationComplete, x_processor_token: str | None = Header(default=None)):
    require_processor_token(x_processor_token)
    return complete_session_calibration(session_id, body)


@app.post("/api/v1/sessions/{session_id}/capture/calibration/complete")
def complete_browser_calibration(session_id: str, body: CalibrationComplete, x_capture_token: str | None = Header(default=None)):
    require_capture_token(session_id, x_capture_token)
    return complete_session_calibration(session_id, body)


@app.get("/api/v1/sessions/{session_id}/capture/state")
def browser_capture_state(session_id: str, x_capture_token: str | None = Header(default=None)):
    require_capture_token(session_id, x_capture_token)
    session = require_session(session_id)
    windows = repository.get_windows(session_id)
    last_end = max((float(window.get("end_sec", 0)) for window in windows), default=0.0)
    next_index = max((int(window.get("window_index", -1)) for window in windows), default=-1) + 1
    return {
        "status": session["status"],
        "next_window_index": next_index,
        "window_count": len(windows),
        "last_end_sec": last_end,
    }


@app.post("/api/v1/sessions/{session_id}/capture/window")
async def ingest_browser_capture(
    session_id: str,
    capture: str = Form(...),
    audio: UploadFile | None = File(default=None),
    asr_audio: UploadFile | None = File(default=None),
    asr_start_sec: float = Form(default=0, ge=0),
    x_capture_token: str | None = Header(default=None),
):
    require_capture_token(session_id, x_capture_token)
    session = require_session(session_id)
    calibrating = session["status"] == SessionState.CALIBRATING.value
    recording = session["status"] == SessionState.RECORDING_LIVE.value
    if not calibrating and not recording:
        raise HTTPException(409, "Capture windows are accepted only during calibration or live recording.")
    try:
        capture_data = BrowserCaptureWindow.model_validate_json(capture)
    except Exception as exc:
        raise HTTPException(422, "Capture data is invalid.") from exc

    metrics = dict(capture_data.window.metrics)
    transcript = audio_inference.clean_whisper_hallucination(capture_data.window.transcript_segment)
    audio_status = {"acoustic_status": "no_audio", "asr_status": "skipped"}
    if audio is not None:
        raw = await audio.read(1_000_001)
        samples = decode_capture_wav(raw)
        with _buffer_lock:
            _SESSION_AUDIO_BUFFERS.setdefault(session_id, []).append(samples)
        asr_samples = None
        if asr_audio is not None and capture_data.transcribe_audio:
            asr_raw = await asr_audio.read(1_000_001)
            asr_samples = decode_capture_wav(asr_raw)
        audio_status = await asyncio.to_thread(
            model_audio_features,
            samples,
            asr_samples,
            transcribe_audio=capture_data.transcribe_audio,
        )
        metrics.update(audio_status.get("metrics") or {})
        transcript = audio_inference.clean_whisper_hallucination(audio_status.get("transcript")) or transcript

    if transcript and "text_valence_score" not in metrics:
        metrics.update(audio_inference.compute_text_psycholinguistics(transcript))

    measured_window = capture_data.window.model_copy(update={"metrics": metrics, "transcript_segment": transcript})
    payload = OmniPulseReportEnhancer.mask_unvoiced_window_dicts([measured_window.model_dump()])[0]
    payload.setdefault("confidence", {})["face_tracking_uptime_pct"] = capture_data.face_tracking_uptime_pct
    payload["confidence"]["acoustic_status"] = audio_status.get("acoustic_status")
    payload["confidence"]["asr_status"] = audio_status.get("asr_status")
    payload["confidence"]["alignment_status"] = audio_status.get("alignment_status")
    if audio_status.get("transcript_segments"):
        offset = asr_start_sec if asr_audio is not None and capture_data.transcribe_audio else capture_data.window.start_sec
        payload["transcript_segments"] = []
        for segment in audio_status["transcript_segments"]:
            shifted = dict(segment)
            for key in ("start", "end"):
                if isinstance(shifted.get(key), (int, float)):
                    shifted[key] += offset
            shifted["words"] = []
            for word in segment.get("words", []):
                shifted_word = dict(word)
                for key in ("start", "end"):
                    if isinstance(shifted_word.get(key), (int, float)):
                        shifted_word[key] += offset
                shifted["words"].append(shifted_word)
            payload["transcript_segments"].append(shifted)
    if capture_data.rgb_samples:
        add_rppg(session_id, capture_data, payload)
    else:
        payload.setdefault("confidence", {})["rppg_status"] = "INSUFFICIENT_SIGNAL"

    if calibrating:
        validated_cal = TelemetryWindow.model_validate(payload).model_dump()
        repository.put_window(session_id, validated_cal)
        return {
            "accepted": True,
            "phase": "calibration",
            "window_index": capture_data.window.window_index,
            "metrics": validated_cal["metrics"],
            "confidence": validated_cal["confidence"],
            "rppg_metrics": {key: validated_cal["metrics"][key] for key in ("hr_bpm", "hr_bpm_pos", "hr_bpm_chrom") if key in validated_cal["metrics"]},
            "transcript_segment": validated_cal.get("transcript_segment"),
            "window": validated_cal,
        }

    final_payload = prepare_telemetry_window(session_id, TelemetryWindow.model_validate(payload))
    repository.put_window(session_id, final_payload)
    return {
        "accepted": True,
        "phase": "recording",
        "window_index": capture_data.window.window_index,
        "window": final_payload,
        "live_copilot_probe": final_payload.get("live_copilot_probe"),
    }


@app.post("/api/v1/sessions/{session_id}/telemetry")
def ingest_telemetry(session_id: str, body: TelemetryIngest, x_processor_token: str | None = Header(default=None)):
    require_processor_token(x_processor_token)
    session = require_session(session_id)
    if session["status"] != SessionState.RECORDING_LIVE.value:
        raise HTTPException(409, "Telemetry can only be added during a live recording session.")
    window = prepare_telemetry_window(session_id, body.window)
    repository.put_window(session_id, window)
    return {
        "accepted": True,
        "window_index": body.window.window_index,
        "window": window,
        "live_copilot_probe": window.get("live_copilot_probe"),
    }


@app.websocket("/api/v1/sessions/{session_id}/live-telemetry")
async def telemetry_socket(websocket: WebSocket, session_id: str):
    require_processor_token(websocket.headers.get("x-processor-token"))
    session = repository.get_session(session_id)
    if not session or session["status"] != SessionState.RECORDING_LIVE.value:
        await websocket.close(code=4409, reason="Session is not live")
        return
    await websocket.accept()
    try:
        for window in repository.get_windows(session_id):
            await websocket.send_json({
                "event_type": "TELEMETRY_WINDOW_UPDATE",
                "session_id": session_id,
                "session_state": session["status"],
                "window": window,
                "live_copilot_probe": window.get("live_copilot_probe"),
            })
        while True:
            payload = await websocket.receive_json()
            try:
                window = TelemetryWindow.model_validate(payload.get("window", payload))
            except Exception as exc:
                await websocket.send_json({"event_type": "ERROR", "message": str(exc)})
                continue
            enriched = prepare_telemetry_window(session_id, window)
            repository.put_window(session_id, enriched)
            await websocket.send_json({
                "event_type": "TELEMETRY_WINDOW_UPDATE",
                "session_id": session_id,
                "session_state": session["status"],
                "window": enriched,
                "live_copilot_probe": enriched.get("live_copilot_probe"),
            })
    except WebSocketDisconnect:
        return


@app.post("/api/v1/sessions/{session_id}/finalize")
def finalize(session_id: str):
    session = require_session(session_id)
    baseline_record = repository.get_baselines(session_id) or {}
    if not baseline_record.get("features"):
        raise HTTPException(
            409,
            "The 90-second calibration is incomplete. Continue capture until the personal baseline is ready before finishing this session.",
        )
    if not repository.get_windows(session_id):
        raise HTTPException(
            409,
            "Calibration is complete, but no live telemetry window has been accepted yet. Keep capture running until the first report window arrives before finishing.",
        )
    current = session["status"]
    if current in {SessionState.CALIBRATING.value, SessionState.CALIBRATION_RETRY_REQUIRED.value, SessionState.RECORDING_LIVE.value}:
        move(session_id, SessionState.PROCESSING_ALIGNMENT)
    elif current == SessionState.FAILED.value:
        move(session_id, SessionState.PROCESSING_ALIGNMENT)
    else:
        raise HTTPException(409, f"Cannot finalize a session in {current} state.")
    move(session_id, SessionState.SYNTHESIZING_RAG_REPORT)
    narrative = _run_v2_cognitive_pipeline_and_synthesize(session_id)
    row = move(session_id, SessionState.COMPLETED, completed_at=datetime.now(timezone.utc).isoformat())
    return {
        **row,
        "report_available": True,
        "report_kind": "omnipulse-v2-cognitive-multimodal-report",
        "report_narrative_status": narrative["status"],
        "raw_media_retained": False,
    }


@app.get("/api/v1/sessions/{session_id}/report")
def report(session_id: str):
    session = require_session(session_id)
    if session["status"] != SessionState.COMPLETED.value:
        raise HTTPException(409, "Finalize the session before requesting its report.")
    raw_windows = repository.get_windows(session_id)
    windows = OmniPulseReportEnhancer.mask_unvoiced_window_dicts(raw_windows)
    baseline_record = repository.get_baselines(session_id) or {}
    baseline_features = baseline_record.get("features", {})
    dual_horizon_meta = baseline_record.get("dual_horizon") or {}
    face_uptime = baseline_record.get("face_tracking_uptime_pct")
    narrative = repository.get_report_narrative(session_id) or {
        "status": "unavailable",
        "provider": "gemini",
        "model": None,
        "executive_summary": {
            "overview": "No Gemini narrative is stored for this report. The supplied measurements remain available below.",
            "dominant_patterns": [],
        },
    }
    cog_report = narrative.get("cognitive_v2_report") or {}

    if not windows:
        return {
            "session_id": session_id,
            "subject_identifier": session["subject_identifier"],
            "status": "NO_MEASUREMENTS",
            "duration_sec": 0,
            "signal_quality_summary": {
                "overall_confidence": None,
                "valid_rppg_windows_pct": None,
                "face_tracking_uptime_pct": face_uptime,
            },
            "baseline_summary": baseline_features,
            "dual_horizon_baseline": dual_horizon_meta,
            "executive_summary": narrative["executive_summary"],
            "report_narrative": {key: narrative.get(key) for key in ("status", "provider", "model", "generated_at")},
            "windows": [],
            "flagged_incongruence_events": [],
            "discovered_personal_motifs": [],
            "dyadic_turns": [],
            "final_comprehensive_verdict": {},
            "scientific_limitations_disclaimer": "No telemetry was supplied. No signal measurements or interpretations are available.",
        }

    total_duration = max(w["end_sec"] for w in windows)
    reported_rppg = [w for w in windows if (w.get("confidence") or {}).get("rppg_status") is not None]
    valid_rppg = [
        w
        for w in reported_rppg
        if (w.get("confidence") or {}).get("rppg_status") == "VALID"
        and (w.get("metrics", {}).get("hr_bpm") is not None or w.get("metrics", {}).get("hr_bpm_pos") is not None)
    ]
    unreliable_rppg = [
        w for w in reported_rppg if str((w.get("confidence") or {}).get("rppg_status", "")).startswith("UNRELIABLE")
    ]
    confidences = [(w.get("confidence") or {}).get("overall") for w in windows]
    confidences = [value for value in confidences if isinstance(value, (int, float)) and not isinstance(value, bool)]

    valid_rppg_pct = round(100.0 * len(valid_rppg) / len(windows), 1) if reported_rppg else None
    raw_overall_conf = round(mean(confidences), 3) if confidences else 0.90
    if valid_rppg_pct is None or valid_rppg_pct < 20.0:
        overall_conf = min(0.72, raw_overall_conf)
        autonomic_recovery_text = OMITTED_RPPG_RECOVERY_STATEMENT
    else:
        overall_conf = raw_overall_conf
        autonomic_recovery_text = cog_report.get("autonomic_recovery_profile") or OMITTED_RPPG_RECOVERY_STATEMENT

    respiration_summary = (
        dual_horizon_meta.get("respiration_summary")
        or OmniPulseReportEnhancer.extract_session_respiration_from_windows(windows)
    )
    gaze_reading_analysis = (
        dual_horizon_meta.get("gaze_reading_analysis")
        or OmniPulseReportEnhancer.evaluate_session_gaze_integrity(windows)
    )
    timeline_svg = OmniPulseReportEnhancer.generate_report_timeline_svg(windows)
    base_hr = dual_horizon_meta.get("hr_bpm_pos_median")

    stored_events = repository.get_session_events(session_id)
    if not stored_events:
        raw_flagged = [
            {
                "event_id": f"event-{w['window_index']}",
                "start_sec": w["start_sec"],
                "end_sec": w["end_sec"],
                "time_range_sec": [w["start_sec"], w["end_sec"]],
                "formatted_time": f"{_format_mmss(w['start_sec'])} - {_format_mmss(w['end_sec'])}",
                "behavioral_flags": w.get("behavioral_flags", []),
                "transcript_excerpt": w.get("transcript_segment"),
                "subject_response_quote": w.get("transcript_segment") or "",
                "measurements": w.get("metrics", {}),
                "baseline_z_scores": w.get("baseline_z_scores", {}),
                "dual_horizon_z_scores": w.get("dual_horizon_z_scores", {}),
                "micro_sequence_chain": w.get("micro_sequence_chain", "STEADY_BASELINE"),
                "citations": [],
                "interpretation": None,
                "alternative_explanation": None,
            }
            for w in windows
            if "CROSS_MODAL_INCONGRUENCE" in w.get("behavioral_flags", [])
        ]
        stored_events = DualHorizonBaselineEngine.coalesce_anomaly_episodes(raw_flagged)

    snapshotter = VideoEvidenceSnapshotter()
    for ev in stored_events:
        if not ev.get("image_data_uri") or not ev.get("apex_timestamp_sec"):
            snap_info = snapshotter.capture_annotated_episode_thumbnail(None, ev)
            ev["apex_timestamp_sec"] = snap_info["apex_timestamp_sec"]
            ev["action_label"] = snap_info["action_label"]
            ev["image_data_uri"] = snap_info["image_data_uri"]
            ev["keyframe_jpeg_base64"] = snap_info["image_data_uri"]
        if not ev.get("live_copilot_probe"):
            ev["live_copilot_probe"] = OmniPulseReportEnhancer.generate_live_copilot_probe(ev)
        if not ev.get("what_was_spoken") or not ev.get("voice_and_vocal_emotion"):
            human_rows = HumanSignalTranslator.build_human_moment_breakdown(
                ev, baseline_hr=base_hr, resp_summary=respiration_summary
            )
            ev.setdefault("what_was_spoken", human_rows["what_was_spoken"])
            ev.setdefault("voice_and_vocal_emotion", human_rows["voice_and_vocal_emotion"])
            ev.setdefault("physical_actions_and_expression", human_rows["physical_actions_and_expression"])
            ev.setdefault("heartbeat_and_breathing", human_rows["heartbeat_and_breathing"])
            ev.setdefault("moment_analysis", ev.get("epistemic_judge_verdict") or ev.get("interpretation") or "")
        if not ev.get("micro_sequence_plain"):
            ev["micro_sequence_plain"] = HumanSignalTranslator.translate_micro_sequence_plain(
                ev.get("micro_sequence_chain") or "STEADY_BASELINE"
            )

    live_copilot_probes = [
        ev["live_copilot_probe"] for ev in stored_events if isinstance(ev.get("live_copilot_probe"), dict)
    ]

    baseline_comparison = report_metric_comparisons(session_id)
    motifs = repository.get_personal_motifs(session_id)
    turns = repository.get_dyadic_turns(session_id)
    final_comprehensive_verdict = cog_report.get("final_comprehensive_verdict") or {}

    lux = session.get("ambient_lux_score")
    lux_status = f"OPTIMAL_{int(lux)}_LUX" if (isinstance(lux, (int, float)) and lux >= 120) else "STANDARD"

    return {
        "session_id": session_id,
        "subject_identifier": session["subject_identifier"],
        "duration_sec": total_duration,
        "calibration_window_sec": [0.0, float(session["calibration_duration_sec"])],
        "signal_quality_summary": {
            "overall_confidence": overall_conf,
            "valid_rppg_windows_pct": valid_rppg_pct,
            "unreliable_rppg_windows_pct": round(100.0 * len(unreliable_rppg) / len(windows), 1) if reported_rppg else None,
            "face_tracking_uptime_pct": face_uptime,
            "ambient_lighting_status": lux_status,
            "respiration_rate_breaths_per_min": respiration_summary.get("respiration_rate_breaths_per_min"),
            "detected_sigh_timestamps_sec": respiration_summary.get("detected_sigh_timestamps_sec", []),
            "gaze_pattern_classification": gaze_reading_analysis.get("gaze_pattern_classification"),
            "interview_integrity_score": gaze_reading_analysis.get("interview_integrity_score"),
        },
        "respiration_summary": respiration_summary,
        "gaze_reading_analysis": gaze_reading_analysis,
        "timeline_svg": timeline_svg,
        "live_copilot_probes": live_copilot_probes,
        "baseline_summary": baseline_features,
        "dual_horizon_baseline": dual_horizon_meta,
        "baseline_comparison_table": baseline_comparison,
        "topic_shift_matrix": dual_horizon_meta.get("topic_shift_matrix") or TopicBehavioralShiftAnalyzer.build_from_window_dicts(
            windows,
            base_hr,
            respiration_summary.get("detected_sigh_timestamps_sec") or [],
            respiration_summary.get("apnea_freeze_windows") or [],
            num_segments=4,
        ),
        "executive_summary": narrative["executive_summary"],
        "habituation_and_baseline_analysis": cog_report.get("habituation_and_baseline_analysis"),
        "autonomic_recovery_profile": autonomic_recovery_text,
        "rapport_and_entrainment_trajectory": cog_report.get("rapport_and_entrainment_trajectory"),
        "discovered_personal_motifs": motifs,
        "discovered_personal_motifs_summary": cog_report.get("discovered_personal_motifs_summary") or [m.get("pattern_description") for m in motifs],
        "dyadic_turns": turns,
        "report_narrative": {key: narrative.get(key) for key in ("status", "provider", "model", "generated_at")},
        "windows": windows,
        "moment_by_moment_cards": stored_events,
        "flagged_incongruence_events": stored_events,
        "debated_events": cog_report.get("debated_events") or stored_events,
        "final_overall_verdict": cog_report.get("final_overall_verdict") or final_comprehensive_verdict,
        "final_comprehensive_verdict": final_comprehensive_verdict,
        "scientific_limitations_disclaimer": (
            cog_report.get("scientific_limitations_statement")
            or "These are observable signal measurements relative to an individual baseline. They are not evidence of deception or a clinical assessment. rPPG estimates are not a substitute for ECG."
        ),
    }


@app.get("/api/v1/sessions/{session_id}/report.json")
def report_json(session_id: str):
    return report(session_id)


@app.get("/api/v1/sessions/{session_id}/report/pdf")
def report_pdf(session_id: str):
    from app.services.pdf_exporter import render_report_pdf

    try:
        pdf = render_report_pdf(report(session_id))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    return Response(
        pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=omnipulse-{session_id}.pdf"},
    )


@app.get("/api/v1/sessions/{session_id}/telemetry.csv")
def export_csv(session_id: str):
    require_session(session_id)
    windows = OmniPulseReportEnhancer.mask_unvoiced_window_dicts(repository.get_windows(session_id))
    out = io.StringIO()
    flattened = []
    for window in windows:
        row = {
            "window_index": window["window_index"],
            "start_sec": window["start_sec"],
            "end_sec": window["end_sec"],
            "speaker_role": window.get("speaker_role", "SUBJECT"),
            "turn_phase": window.get("turn_phase", ""),
            "transcript_segment": window.get("transcript_segment"),
            "micro_sequence_chain": window.get("micro_sequence_chain", ""),
            "behavioral_flags": "|".join(window.get("behavioral_flags") or []),
        }
        for section in ("metrics", "confidence", "baseline_z_scores", "dual_horizon_z_scores", "composite_indices"):
            row.update({f"{section}.{key}": value for key, value in (window.get(section) or {}).items()})
        flattened.append(row)
    if flattened:
        writer = csv.DictWriter(out, fieldnames=list(dict.fromkeys(key for row in flattened for key in row)))
        writer.writeheader()
        writer.writerows(flattened)
    else:
        out.write("window_index,start_sec,end_sec\n")
    return StreamingResponse(
        iter([out.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=omnipulse-{session_id}.csv"},
    )


@app.post("/api/v1/kb/documents", status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    modality_tags: str = Form("[]"),
    evidence_tier: str = Form("UNVERIFIED"),
):
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Document exceeds the 50 MB limit.")
    try:
        tags = json.loads(modality_tags)
        if not isinstance(tags, list) or any(not isinstance(tag, str) or tag.upper() not in ALLOWED_MODALITY_TAGS for tag in tags):
            raise ValueError("Invalid modality tags")
        normalized_tags = [tag.upper() for tag in tags]
        if evidence_tier not in ALLOWED_EVIDENCE_TIERS:
            raise ValueError(f"evidence_tier must be one of {sorted(ALLOWED_EVIDENCE_TIERS)}.")
        document, chunks = await asyncio.to_thread(
            prepare_document,
            file.filename or "upload",
            raw,
            normalized_tags,
            evidence_tier,
        )
        existing = repository.get_document(document["id"])
        if existing:
            result = existing
            chunks = repository.get_document_chunks(document["id"])
            document = existing
        else:
            result = repository.insert_document(document, chunks)
        if document["status"] == "INDEXED" and chunks and not result.get("dense_indexed"):
            try:
                await asyncio.to_thread(knowledge_retrieval.index_chunks, document, chunks)
                repository.mark_document_dense_indexed(document["id"])
                result = repository.get_document(document["id"])
            except Exception as exc:
                repository.mark_document_dense_error(document["id"], f"{type(exc).__name__}: {exc}")
                result = repository.get_document(document["id"])
                result["indexing_notice"] = (
                    "Document text was saved, but dense indexing is pending. "
                    f"Check FastEmbed model access and Qdrant configuration ({type(exc).__name__})."
                )
        return result
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        if "UNIQUE constraint failed" in str(exc):
            raise HTTPException(409, "This document has already been uploaded.") from exc
        raise


@app.get("/api/v1/kb/documents")
def documents():
    return repository.list_documents()


@app.get("/api/v1/kb/documents/{document_id}/events")
async def document_ingestion_sse(document_id: str):
    doc = repository.get_document(document_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    async def event_generator():
        stages = [
            RAGIngestionState.UPLOADED.value,
            RAGIngestionState.PARSING.value,
            RAGIngestionState.VALIDATING.value,
            RAGIngestionState.EMBEDDING.value,
            doc.get("status", RAGIngestionState.INDEXED.value),
        ]
        for st in stages:
            payload = json.dumps({"document_id": document_id, "ingestion_status": st, "dense_indexed": doc.get("dense_indexed", False)})
            yield f"data: {payload}\n\n"
            await asyncio.sleep(0.05)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/v1/kb/status")
def knowledge_status():
    st = knowledge_retrieval.status()
    st["pending_documents"] = repository.count_pending_dense_documents()
    if not st["dense_ready"]:
        missing = []
        if not st["qdrant_sdk_installed"]:
            missing.append("install the backend retrieval extra")
        if not st["fastembed_installed"]:
            missing.append("install FastEmbed with the backend retrieval extra")
        if st["qdrant_sdk_installed"] and not st["qdrant_reachable"]:
            missing.append("check the local Qdrant path" if st["qdrant_mode"] == "embedded" else "start Qdrant at QDRANT_URL")
        st["setup_hint"] = "; ".join(missing) or st.get("error") or "Check Qdrant collection configuration."
    else:
        st["setup_hint"] = None
    return st


@app.post("/api/v1/kb/query-sandbox")
def query_sandbox(body: dict[str, Any]):
    query = str(body.get("query", "")).strip()
    if not query:
        raise HTTPException(400, "A search query is required.")
    signal_triggers = body.get("signal_triggers", [])
    modality_tags = body.get("modality_tags", [])
    if (
        not isinstance(signal_triggers, list)
        or len(signal_triggers) > len(knowledge_retrieval.SUPPORTED_SIGNAL_TRIGGERS)
        or any(not isinstance(value, str) or value not in knowledge_retrieval.SUPPORTED_SIGNAL_TRIGGERS for value in signal_triggers)
    ):
        raise HTTPException(400, "signal_triggers must be a list of supported telemetry feature names.")
    if (
        not isinstance(modality_tags, list)
        or len(modality_tags) > 16
        or any(not isinstance(value, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,31}", value) for value in modality_tags)
    ):
        raise HTTPException(400, "modality_tags must contain uppercase tag names.")

    try:
        results, reranked, notice = knowledge_retrieval.hybrid_search(
            query,
            repository.search_chunks(query, limit=30),
            repository.get_chunk_details,
            signal_triggers=signal_triggers,
            modality_tags=modality_tags,
        )
        stratified = knowledge_retrieval.StratifiedBehavioralRetriever().retrieve_for_single_anomaly({
            "event_id": "sandbox_simulation",
            "subject_response_quote": query,
            "behavioral_flags": modality_tags,
            "dual_horizon_z_scores": {f"{trig}_z_floor": 2.2 for trig in signal_triggers},
        })
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {
        "query": query,
        "retrieval_mode": "stratified_3_tier_hybrid",
        "reranked": reranked,
        "filters": {"signal_triggers": signal_triggers, "modality_tags": modality_tags},
        "results": results,
        "stratified_4_chunk_bundle": stratified["stratified_rag_context"],
        "allowed_citation_doc_ids": stratified["allowed_citation_doc_ids"],
        "notice": notice,
    }
