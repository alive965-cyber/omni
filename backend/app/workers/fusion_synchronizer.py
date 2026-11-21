"""Temporal Synchronizer, Outlier-Resistant Robust Z-Score Engine, Composite Behavioral Indices
(VSI, VTLI), Cross-Modal Congruency Index (CMCI), and Micro-Sequence Integration
(PDF Specification Section 4.8 + v2.0 Cognitive Upgrades).
Fixes Bug #1: MAD=0.0 division-by-epsilon Z-score explosion using MIN_PHYSIOLOGICAL_SCALE + session STD fallback + [-8.0, +8.0] clamp.
"""
from __future__ import annotations

from math import exp, isfinite
from statistics import median, pstdev
from typing import Any

from app.workers.adaptive_baseline import MIN_PHYSIOLOGICAL_SCALE
from app.workers.micro_sequence_encoder import MicroSequenceEncoder


def robust_baseline(values: list[float | None]) -> dict[str, float | int] | None:
    """Section 4.8.1: Outlier-Resistant Robust Median, MAD & STD Baseline."""
    clean = [float(x) for x in values if x is not None and isfinite(float(x))]
    if not clean:
        return None
    center = float(median(clean))
    mad = float(median([abs(x - center) for x in clean]))
    std = float(pstdev(clean)) if len(clean) > 2 else 0.0
    return {"median": center, "mad": mad, "std": std, "sample_count": len(clean)}


def robust_z_score(value: float | None, baseline: dict | None, feature: str = "") -> float | None:
    """Section 4.8.1 (Bug #1 Fix):
    safe_scale = max(1.4826 * mad_val, 0.5 * session_std, MIN_PHYSIOLOGICAL_SCALE[feature])
    Z_k(t) = clip((x_k(t) - median_k) / safe_scale, -8.0, +8.0).
    """
    if value is None or baseline is None or baseline.get("sample_count", 0) < 3:
        return None
    val_f = float(value)
    med_f = float(baseline.get("median", 0.0))
    mad_f = float(baseline.get("mad", 0.0))
    std_f = float(baseline.get("std", 0.0))
    min_scale = float(MIN_PHYSIOLOGICAL_SCALE.get(feature, 0.05))
    safe_scale = max(1.4826 * mad_f, 0.5 * std_f, min_scale)
    if abs(val_f - med_f) < 1e-9:
        return 0.0
    raw_z = (val_f - med_f) / safe_scale
    return round(max(-8.0, min(8.0, raw_z)), 2)


def classify_deviation(z_score: float | None) -> str:
    """Section 4.8.1: |Z| < 1.5 -> WITHIN_PERSONAL_BASELINE; 1.5 <= |Z| < 2.5 -> MODERATE_DEVIATION; |Z| >= 2.5 -> SIGNIFICANT_DEVIATION."""
    if z_score is None:
        return "INSUFFICIENT_DATA"
    if abs(z_score) >= 2.5:
        return "SIGNIFICANT_DEVIATION"
    if abs(z_score) >= 1.5:
        return "MODERATE_DEVIATION"
    return "WITHIN_PERSONAL_BASELINE"


def logistic_index(weighted_z: float | None) -> float | None:
    """Section 4.8.2: 100 * sigma_logistic(weighted_z)."""
    if weighted_z is None:
        return None
    return 100.0 / (1.0 + exp(-max(-30.0, min(30.0, float(weighted_z)))))


def cross_modal_congruency(text_valence: float | None, vocal_stress: float | None, visual_load: float | None) -> float | None:
    """Section 4.8.3: Cross-Modal Congruency Index (CMCI in [0.0, 1.0]).

    Divergence(t) = I(text_valence(t) > 0.35) * max(0, (VSI(t) + VTLI(t)) / 200 - 0.60)
    CMCI(t) = 1.0 - clip(Divergence(t) * 2.5, 0.0, 1.0)
    """
    if text_valence is None or vocal_stress is None or visual_load is None:
        return None
    divergence = (1.0 if text_valence > 0.35 else 0.0) * max(0.0, (vocal_stress + visual_load) / 200.0 - 0.60)
    return max(0.0, min(1.0, 1.0 - divergence * 2.5))


FEATURE_ALIASES = {
    "hr_bpm": ("hr_bpm", "hr_bpm_pos"),
    "f0_semitone_mean": ("f0_semitone_mean", "f0_semitone", "audio_pitch_f0_mean"),
    "f0_semitone_std": ("f0_semitone_std", "audio_pitch_f0_std"),
    "jitter_local_mean": ("jitter_local_mean", "jitter_local", "audio_jitter_mean"),
    "shimmer_local_db_mean": ("shimmer_local_db_mean", "shimmer_local_db", "shimmer_db", "audio_shimmer_mean"),
    "hnr_db_mean": ("hnr_db_mean", "hnr_db", "audio_hnr_mean"),
    "loudness_mean": ("loudness_mean", "audio_loudness_mean"),
    "loudness_peaks_per_sec": ("loudness_peaks_per_sec", "audio_loudness_peaks_per_sec"),
    "spectral_flux_mean": ("spectral_flux_mean",),
    "au01_inner_brow": ("au01_inner_brow", "au01"),
    "au02_outer_brow": ("au02_outer_brow", "au02"),
    "au04_brow_lowerer": ("au04_brow_lowerer", "au04"),
    "au06_cheek_raiser": ("au06_cheek_raiser", "au06"),
    "au12_lip_corner_puller": ("au12_lip_corner_puller", "au12_smile", "au12"),
    "au15_lip_corner_depressor": ("au15_lip_corner_depressor", "au15"),
    "au20_lip_stretcher": ("au20_lip_stretcher", "au20"),
    "au24_lip_presser": ("au24_lip_presser", "au24"),
    "au25_jaw_drop": ("au25_jaw_drop", "au25"),
    "facial_asymmetry_index": ("facial_asymmetry_index",),
    "self_touch_ratio": ("self_touch_ratio", "video_self_touch_ratio"),
    "hand_kinetic_energy": ("hand_kinetic_energy", "video_hand_motion_mean"),
    "postural_lean_z": ("postural_lean_z",),
    "blink_rate_per_min": ("blink_rate_per_min",),
    "blink_duration_ms_mean": ("blink_duration_ms_mean", "blink_duration_ms"),
    "gaze_aversion_ratio": ("gaze_aversion_ratio",),
    "saccade_velocity_deg_s": ("saccade_velocity_deg_s",),
    "speech_rate_wpm": ("speech_rate_wpm",),
    "pause_ratio": ("pause_ratio",),
    "response_latency_ms": ("response_latency_ms",),
    "disfluency_count": ("disfluency_count",),
    "first_person_pronoun_ratio": ("first_person_pronoun_ratio",),
    "text_valence_score": ("text_valence_score",),
}


def _value(metrics: dict[str, Any], key: str) -> float | None:
    for alias in FEATURE_ALIASES.get(key, (key,)):
        value = metrics.get(alias)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(float(value)):
            return float(value)
    return None


def enrich_window(window: dict[str, Any], baselines: dict[str, dict]) -> dict[str, Any]:
    """Compute baseline-relative robust Z-scores (clamped to [-8.0, +8.0]), VSI, VTLI, CMCI, micro-sequences, and behavioral flags."""
    metrics = window.get("metrics") or {}
    conf = window.get("confidence") or {}
    head_yaw = _value(metrics, "head_yaw_deg")
    head_pitch = _value(metrics, "head_pitch_deg")
    face_conf = conf.get("face_tracking", conf.get("face_confidence"))

    # Section 11.2 Error Handling Matrix: Subject leaves frame or turns head > 35 deg
    visual_geometry_degraded = False
    if (isinstance(face_conf, (int, float)) and float(face_conf) < 0.5) or (head_yaw is not None and abs(head_yaw) > 35.0) or (head_pitch is not None and abs(head_pitch) > 25.0):
        visual_geometry_degraded = True
        conf["face_confidence_status"] = "LOW"
        window["confidence"] = conf

    visual_feature_keys = {
        "au01_inner_brow", "au02_outer_brow", "au04_brow_lowerer", "au06_cheek_raiser",
        "au12_lip_corner_puller", "au15_lip_corner_depressor", "au20_lip_stretcher",
        "au24_lip_presser", "au25_jaw_drop", "facial_asymmetry_index",
        "blink_rate_per_min", "blink_duration_ms_mean", "gaze_aversion_ratio", "saccade_velocity_deg_s",
    }

    z_scores: dict[str, float | None] = {}
    for feature in baselines:
        if visual_geometry_degraded and feature in visual_feature_keys:
            z_scores[feature] = None
            continue
        value = _value(metrics, feature)
        z_scores[feature] = robust_z_score(value, baselines.get(feature), feature=feature)

    # Composite Vocal Stress Index (VSI):
    # VSI(t) = 100 * sigma_logistic(0.35 * Z_jitter + 0.25 * Z_shimmer + 0.25 * Z_F0 - 0.15 * Z_HNR)
    vocal_keys = ("jitter_local_mean", "shimmer_local_db_mean", "f0_semitone_mean", "hnr_db_mean")
    vocal_values = [z_scores.get(key) for key in vocal_keys]
    vsi = None if any(value is None for value in vocal_values) else logistic_index(
        0.35 * vocal_values[0] + 0.25 * vocal_values[1] + 0.25 * vocal_values[2] - 0.15 * vocal_values[3]
    )

    # Composite Visual Tension & Cognitive Load Index (VTLI):
    # VTLI(t) = 100 * sigma_logistic(0.25 * Z_AU04 + 0.25 * Z_AU24 + 0.20 * Z_self_touch + 0.15 * Z_blink_rate + 0.15 * Z_gaze_aversion)
    visual_keys = ("au04_brow_lowerer", "au24_lip_presser", "self_touch_ratio", "blink_rate_per_min", "gaze_aversion_ratio")
    visual_values = [z_scores.get(key) for key in visual_keys]
    vtli = None if any(value is None for value in visual_values) else logistic_index(
        0.25 * visual_values[0] + 0.25 * visual_values[1] + 0.20 * visual_values[2] + 0.15 * visual_values[3] + 0.15 * visual_values[4]
    )

    sentiment = metrics.get("text_valence_score")
    sentiment = float(sentiment) if isinstance(sentiment, (int, float)) and not isinstance(sentiment, bool) and isfinite(float(sentiment)) else None
    cmci = cross_modal_congruency(sentiment, vsi, vtli)
    indices = {
        "vocal_stress_index": vsi,
        "visual_tension_load_index": vtli,
        "cross_modal_congruency_index": cmci,
    }

    # Derived flags are always regenerated from the measurements on the server.
    auto_Semantic_Flags = {
        "CROSS_MODAL_INCONGRUENCE",
        "ELEVATED_VOCAL_JITTER",
        "AU24_LIP_PRESS_SPIKE",
        "PACIFYING_NECK_TOUCH",
        "HIGH_COGNITIVE_LOAD",
        "AUTONOMIC_AROUSAL_SPIKE",
    }
    flags = [
        flag
        for flag in (window.get("behavioral_flags") or [])
        if not flag.startswith(("SIGNIFICANT_DEVIATION_", "MODERATE_DEVIATION_")) and flag not in auto_Semantic_Flags
    ]
    non_behavioral_artifacts = {"illumination_change_pct", "head_velocity_deg_s", "head_pitch_deg", "head_yaw_deg"}
    for key, score in z_scores.items():
        if key in non_behavioral_artifacts:
            continue
        if score is not None and abs(score) >= 1.5:
            level = "SIGNIFICANT" if abs(score) >= 2.5 else "MODERATE"
            flag = f"{level}_DEVIATION_{key.upper()}"
            if flag not in flags:
                flags.append(flag)

    # Canonical semantic flags (PDF Section 8.2 & 6.4)
    if (z_scores.get("jitter_local_mean") or 0.0) >= 1.8 and "ELEVATED_VOCAL_JITTER" not in flags:
        flags.append("ELEVATED_VOCAL_JITTER")
    if (z_scores.get("au24_lip_presser") or 0.0) >= 1.8 and "AU24_LIP_PRESS_SPIKE" not in flags:
        flags.append("AU24_LIP_PRESS_SPIKE")
    if ((_value(metrics, "self_touch_ratio") or 0.0) >= 0.30 or (z_scores.get("self_touch_ratio") or 0.0) >= 1.5) and "PACIFYING_NECK_TOUCH" not in flags:
        flags.append("PACIFYING_NECK_TOUCH")
    if (z_scores.get("au04_brow_lowerer") or 0.0) >= 1.6 and (z_scores.get("gaze_aversion_ratio") or 0.0) >= 1.5 and "HIGH_COGNITIVE_LOAD" not in flags:
        flags.append("HIGH_COGNITIVE_LOAD")
    if (z_scores.get("hr_bpm") or 0.0) >= 1.8 and "AUTONOMIC_AROUSAL_SPIKE" not in flags:
        flags.append("AUTONOMIC_AROUSAL_SPIKE")

    # Upgrade 3: 100ms sub-second micro-sequence encoding
    slices_100ms = window.get("slices_100ms")
    if isinstance(slices_100ms, list) and slices_100ms:
        micro_info = MicroSequenceEncoder.encode_100ms_slice_sequence(slices_100ms)
    else:
        micro_info = MicroSequenceEncoder.synthesize_from_window_metrics(metrics)

    window["micro_sequence_tokens"] = micro_info["sequence_tokens"]
    window["micro_sequence_chain"] = micro_info["sequence_string"]
    window["smile_offset_velocity"] = micro_info["smile_offset_velocity"]
    window["micro_pattern_tag"] = micro_info["pattern_tag"]

    window["baseline_z_scores"] = z_scores
    window["composite_indices"] = indices
    window["behavioral_flags"] = flags
    return window
