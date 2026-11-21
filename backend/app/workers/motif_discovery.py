"""Implements OmniPulse v2.0 Upgrade 5: Unsupervised Intra-Session 'Personal Tell' Motif Mining
(motif_discovery.py).
"""
from __future__ import annotations

from math import isfinite
from typing import Any
import numpy as np
from scipy.stats import pointbiserialr


class PersonalMotifMiner:
    """Discovers subject-specific behavioral habits ('tells') that repeatedly co-occur
    with high autonomic/vocal strain windows during a session using Point-Biserial
    Correlation (r_pb >= 0.38) and Risk Ratio (RR >= 2.5).
    """

    CANDIDATE_BEHAVIORS = (
        ("au24_lip_presser", "FACS", "Repeated Lip Compression (AU24)"),
        ("au04_brow_lowerer", "FACS", "Glabellar Brow Furrowing (AU04)"),
        ("self_touch_ratio", "VISUAL_KINESIC", "Pacifying Neck/Face Touch"),
        ("gaze_aversion_ratio", "VISUAL_KINESIC", "Sustained Gaze Aversion Shift"),
        ("blink_rate_per_min", "VISUAL_KINESIC", "Ocular Blink Rate Flurry"),
        ("head_roll_deg", "VISUAL_KINESIC", "Lateral Head Tilt / Roll Shift"),
        ("pause_ratio", "ACOUSTIC", "Intra-Turn Hesitation Pause Cluster"),
        ("disfluency_count", "PSYCHOLINGUISTIC", "Lexical Filler Surge ('uh/um')"),
    )

    @classmethod
    def discover_personal_motifs(cls, windows: list[dict[str, Any]] | Any) -> list[dict[str, Any]]:
        """Accepts either a list of telemetry window dicts or a pandas DataFrame."""
        if hasattr(windows, "to_dict"):
            records = windows.to_dict(orient="records")
        else:
            records = list(windows or [])

        subject_windows = [
            w for w in records if str(w.get("speaker_role", "SUBJECT")).upper() in {"SUBJECT", "SPEAKER_01", ""}
        ]
        if len(subject_windows) < 8:
            return []

        n = len(subject_windows)
        high_strain_mask = np.zeros(n, dtype=int)
        for idx, w in enumerate(subject_windows):
            indices = w.get("composite_indices") or {}
            vsi = float(indices.get("vocal_stress_index") or w.get("vsi_score") or 0.0)
            dual_z = w.get("dual_horizon_z_scores") or {}
            base_z = w.get("baseline_z_scores") or {}
            hr_z = float(dual_z.get("hr_bpm_z_floor") or dual_z.get("hr_bpm_pos_z_floor") or base_z.get("hr_bpm") or 0.0)
            pause_z = float(dual_z.get("pause_ratio_z_floor") or base_z.get("pause_ratio") or 0.0)
            if vsi >= 68.0 or hr_z >= 1.6 or pause_z >= 1.6:
                high_strain_mask[idx] = 1

        if int(high_strain_mask.sum()) < 2 or int(n - high_strain_mask.sum()) < 3:
            return []

        discovered_motifs: list[dict[str, Any]] = []
        for col, modality, label in cls.CANDIDATE_BEHAVIORS:
            vals: list[float] = []
            for w in subject_windows:
                metrics = w.get("metrics") or {}
                raw_v = metrics.get(col, w.get(col, 0.0))
                v = abs(float(raw_v)) if (isinstance(raw_v, (int, float)) and not isinstance(raw_v, bool) and isfinite(float(raw_v))) else 0.0
                vals.append(v)

            series = np.asarray(vals, dtype=float)
            if float(np.std(series)) < 1e-5:
                continue

            r_pb, p_val = pointbiserialr(high_strain_mask, series)
            if np.isnan(r_pb) or float(r_pb) < 0.38 or float(p_val) > 0.08:
                continue

            thresh = float(np.quantile(series, 0.70))
            active = series > max(thresh, 1e-3)
            p_given_strain = float(np.mean(active[high_strain_mask == 1]))
            p_given_calm = float(np.mean(active[high_strain_mask == 0])) + 0.05
            risk_ratio = p_given_strain / p_given_calm

            if risk_ratio >= 2.5:
                active_ts = [
                    round(float(subject_windows[i].get("start_sec", i * 2.0)), 1)
                    for i in range(n)
                    if active[i] and high_strain_mask[i] == 1
                ]
                if len(active_ts) >= 2:
                    discovered_motifs.append({
                        "motif_name": label,
                        "modality": modality,
                        "feature_key": col,
                        "point_biserial_r": round(float(r_pb), 3),
                        "risk_ratio": round(float(risk_ratio), 2),
                        "occurrence_timestamps_sec": active_ts[:8],
                        "pattern_description": (
                            f"Idiosyncratic session pattern: {label} is {risk_ratio:.1f}x more likely to occur "
                            f"when this specific speaker experiences elevated vocal/cardiovascular strain (r = {float(r_pb):.2f})."
                        ),
                    })

        discovered_motifs.sort(key=lambda item: item["point_biserial_r"], reverse=True)
        return discovered_motifs[:4]
