"""
backend/app/workers/rule_of_three_filter.py
Implements the FBI / Ekman "Rule of 3" Multi-Channel Biological Convergence Filter.
Strips camera/motion artifacts (`illumination_change_pct`, `head_velocity_deg_s`)
and requires >= 3 converging signals across >= 2 independent biological channels.
"""
from __future__ import annotations

from typing import Dict, List, Any, Tuple

BLACKLISTED_ARTIFACT_KEYS = {
    "illumination_change_pct",
    "illumination_change_pct_z_floor",
    "head_velocity_deg_s",
    "head_velocity_deg_s_z_floor",
    "head_yaw_deg",
    "head_yaw_deg_z_floor",
    "head_pitch_deg",
    "head_pitch_deg_z_floor",
    "face_confidence",
    "rppg_sqi",
    "audio_snr_db",
}

BIOLOGICAL_CHANNELS: Dict[str, List[str]] = {
    "C1_OCULAR_UPPER_FACE": [
        "au01_inner_brow",
        "au02_outer_brow",
        "au04_brow_lowerer",
        "blink_rate_per_min",
        "blink_duration_ms_mean",
        "gaze_aversion_ratio",
        "saccade_velocity_deg_s",
    ],
    "C2_LOWER_FACE_INHIBITORY": [
        "au06_cheek_raiser",
        "au12_lip_corner_puller",
        "au15_lip_corner_depressor",
        "au20_lip_stretcher",
        "au24_lip_presser",
        "facial_asymmetry_index",
    ],
    "C3_KINESIC_ADAPTORS": [
        "self_touch_ratio",
        "hand_kinetic_energy",
        "postural_lean_z",
    ],
    "C4_ACOUSTIC_PSYCHOLINGUISTIC": [
        "f0_semitone_mean",
        "f0_semitone_std",
        "jitter_local_mean",
        "shimmer_local_db_mean",
        "hnr_db_mean",
        "loudness_mean",
        "speech_rate_wpm",
        "pause_ratio",
        "disfluency_count",
        "response_latency_ms",
        "first_person_pronoun_ratio",
    ],
    "C5_CARDIORESPIRATORY": [
        "hr_bpm_pos",
        "hr_bpm",
        "respiration_rate_breaths_per_min",
    ],
}


class RuleOfThreeConvergenceFilter:
    @classmethod
    def sanitize_and_evaluate_window(
        cls,
        win_dict: Dict[str, Any],
        sigh_timestamps: List[float],
        apnea_windows: List[List[float]],
        z_threshold: float = 1.65,
    ) -> Tuple[bool, float, Dict[str, float], List[str]]:
        """
        1. Strips blacklisted camera/motion variables from `dual_horizon_z_scores`.
        2. Evaluates whether the window passes the Rule of 3 (>=3 signals across >=2 biological channels).
        Returns: (passes_gate, mccs_score, clean_z_scores, active_channel_names)
        """
        raw_z: Dict[str, Any] = win_dict.get("dual_horizon_z_scores") or win_dict.get("baseline_z_scores") or {}
        clean_z: Dict[str, float] = {}

        for k, v in raw_z.items():
            if str(k).endswith("_delta_z_phasic"):
                continue
            base_k = str(k).replace("_z_floor", "").replace("_delta_z_phasic", "").replace("_z", "")
            if base_k == "postural_lean":
                base_k = "postural_lean_z"
            if k in BLACKLISTED_ARTIFACT_KEYS or base_k in BLACKLISTED_ARTIFACT_KEYS:
                continue
            if v is not None and isinstance(v, (int, float)) and not isinstance(v, bool):
                clean_z[base_k] = round(float(v), 2)

        active_channels = set()
        spiking_magnitudes: List[float] = []

        # Check all 5 biological channels
        for chan_name, feat_list in BIOLOGICAL_CHANNELS.items():
            for feat in feat_list:
                z_val = clean_z.get(feat, 0.0)
                if abs(z_val) >= z_threshold:
                    active_channels.add(chan_name)
                    spiking_magnitudes.append(abs(z_val))

        # Check discrete physiological events (Self-Touch, Sigh, Breath-Hold Apnea)
        start_s = float(win_dict.get("start_sec", 0.0))
        end_s = float(win_dict.get("end_sec", 0.0))
        metrics = win_dict.get("metrics") or {}
        self_touch = win_dict.get("self_touch_ratio")
        if self_touch is None:
            self_touch = metrics.get("self_touch_ratio", 0.0)

        if float(self_touch or 0.0) >= 0.22:
            active_channels.add("C3_KINESIC_ADAPTORS")
            spiking_magnitudes.append(3.5)

        if any(start_s - 0.5 <= float(s) <= end_s + 0.5 for s in (sigh_timestamps or [])):
            active_channels.add("C5_CARDIORESPIRATORY")
            spiking_magnitudes.append(3.0)

        if any(len(a) >= 2 and float(a[0]) <= end_s and float(a[1]) >= start_s for a in (apnea_windows or [])):
            active_channels.add("C5_CARDIORESPIRATORY")
            spiking_magnitudes.append(2.8)

        # Rule of 3 Gate: At least 3 spiking signals across at least 2 distinct biological channels
        passes_gate = (len(spiking_magnitudes) >= 3) and (len(active_channels) >= 2)

        spiking_magnitudes.sort(reverse=True)
        top_3_mean = sum(spiking_magnitudes[:3]) / 3.0 if len(spiking_magnitudes) >= 3 else (
            sum(spiking_magnitudes) / max(1, len(spiking_magnitudes))
        )
        mccs_score = round(len(active_channels) * top_3_mean, 3)

        return passes_gate, mccs_score, clean_z, sorted(active_channels)
