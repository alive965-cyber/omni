"""Implements OmniPulse v2.0 Upgrade 3: 100ms (10Hz) Sub-Second Micro-Sequence Ordering
and Smile Snap-Off Physics (micro_sequence_encoder.py).
"""
from __future__ import annotations

from typing import Any
import numpy as np


class MicroSequenceEncoder:
    """Processes 10Hz (100ms) high-resolution facial/gaze/kinesic slices inside a 2.0s window
    to preserve exact sub-second causal ordering (e.g., Micro-Frown -> Lip Press -> Masked Smile)
    and compute Smile Offset Velocity (v_smile_drop > 1.80 units/sec -> SNAP_OFF_SMILE_DROP).
    """

    ONSET_THRESHOLDS = {
        "AU04_BROW_FURROW": ("au04", 0.45),
        "AU15_LIP_CORNER_DROP": ("au15", 0.40),
        "AU20_LIP_STRETCH": ("au20", 0.42),
        "AU24_LIP_PRESS": ("au24", 0.50),
        "AU12_DUCHENNE_SMILE": ("au12_duchenne", 0.45),
        "AU12_NON_DUCHENNE_SMILE": ("au12_masked", 0.45),
        "GAZE_AVERT_SHIFT": ("gaze_aversion", 0.55),
        "NECK_OR_FACE_SELF_TOUCH": ("self_touch", 0.50),
    }

    @classmethod
    def encode_100ms_slice_sequence(cls, slices_100ms: list[dict[str, float]]) -> dict[str, Any]:
        """Encode 20 x 100ms slices inside a 2.0s window into a chronological action string and grammar tag."""
        if not slices_100ms:
            return {
                "sequence_tokens": [],
                "sequence_string": "NO_MICRO_EVENTS",
                "smile_offset_velocity": 0.0,
                "pattern_tag": "STEADY_BASELINE",
            }

        events: list[tuple[int, str, float]] = []
        active_locks: set[str] = set()

        for s in slices_100ms:
            s_copy = dict(s)
            t_ms = int(round(float(s_copy.get("offset_ms", 0.0))))
            au12_val = float(s_copy.get("au12", s_copy.get("au12_lip_corner_puller", 0.0)))
            au06_val = float(s_copy.get("au06", s_copy.get("au06_cheek_raiser", 0.0)))

            s_copy["au12"] = au12_val
            s_copy["au04"] = float(s_copy.get("au04", s_copy.get("au04_brow_lowerer", 0.0)))
            s_copy["au15"] = float(s_copy.get("au15", s_copy.get("au15_lip_corner_depressor", 0.0)))
            s_copy["au20"] = float(s_copy.get("au20", s_copy.get("au20_lip_stretcher", 0.0)))
            s_copy["au24"] = float(s_copy.get("au24", s_copy.get("au24_lip_presser", 0.0)))
            s_copy["gaze_aversion"] = float(s_copy.get("gaze_aversion", s_copy.get("gaze_aversion_ratio", 0.0)))
            s_copy["self_touch"] = float(s_copy.get("self_touch", s_copy.get("self_touch_ratio", 0.0)))

            s_copy["au12_duchenne"] = au12_val if au06_val >= 0.22 else 0.0
            s_copy["au12_masked"] = au12_val if au06_val < 0.15 else 0.0

            for token_name, (key, thresh) in cls.ONSET_THRESHOLDS.items():
                val = float(s_copy.get(key, 0.0))
                if val >= thresh and token_name not in active_locks:
                    events.append((t_ms, token_name, round(val, 2)))
                    active_locks.add(token_name)
                elif val < thresh * 0.70 and token_name in active_locks:
                    active_locks.remove(token_name)

        au12_curve = np.array([float(s.get("au12", s.get("au12_lip_corner_puller", 0.0))) for s in slices_100ms], dtype=float)
        smile_offset_vel = cls._compute_smile_drop_velocity(au12_curve, dt_sec=0.1)
        if smile_offset_vel > 1.80:
            apex_indices = np.flatnonzero(au12_curve >= 0.90 * float(np.max(au12_curve)))
            last_apex_idx = int(apex_indices[-1]) if len(apex_indices) else int(np.argmax(au12_curve))
            snap_ms = int((last_apex_idx + 1) * 100)
            events.append((snap_ms, "SNAP_OFF_SMILE_DROP", round(smile_offset_vel, 2)))

        events.sort(key=lambda item: item[0])
        ordered_tokens = [e[1] for e in events]
        sequence_str = " -> ".join(f"[{e[1]} @ +{e[0]}ms]" for e in events) if events else "STEADY_BASELINE"
        return {
            "sequence_tokens": ordered_tokens,
            "sequence_string": sequence_str,
            "smile_offset_velocity": round(smile_offset_vel, 3),
            "pattern_tag": cls._classify_micro_grammar(ordered_tokens),
        }

    @classmethod
    def synthesize_from_window_metrics(cls, metrics: dict[str, Any]) -> dict[str, Any]:
        """Fallback micro-sequence encoder when only window-level features are supplied without 10Hz slices."""
        tokens: list[tuple[int, str]] = []
        gaze = float(metrics.get("gaze_aversion_ratio") or 0.0)
        au04 = float(metrics.get("au04_brow_lowerer") or metrics.get("au04") or 0.0)
        au24 = float(metrics.get("au24_lip_presser") or metrics.get("au24") or 0.0)
        touch = float(metrics.get("self_touch_ratio") or 0.0)
        au12 = float(metrics.get("au12_lip_corner_puller") or metrics.get("au12_smile") or 0.0)
        au06 = float(metrics.get("au06_cheek_raiser") or 0.0)

        if gaze >= 0.55:
            tokens.append((180, "GAZE_AVERT_SHIFT"))
        if au04 >= 0.20:
            tokens.append((290, "AU04_BROW_FURROW"))
        if au24 >= 0.30:
            tokens.append((490, "AU24_LIP_PRESS"))
        if touch >= 0.22:
            tokens.append((680, "NECK_OR_FACE_SELF_TOUCH"))
        if au12 >= 0.35:
            if au06 >= 0.22:
                tokens.append((820, "AU12_DUCHENNE_SMILE"))
            elif au06 < 0.15:
                tokens.append((820, "AU12_NON_DUCHENNE_SMILE"))

        ordered_tokens = [t[1] for t in tokens]
        sequence_str = (
            "[STIMULUS_END @ +0ms] -> " + " -> ".join(f"[{name} @ +{ms}ms]" for ms, name in tokens)
            if tokens
            else "STEADY_BASELINE"
        )
        return {
            "sequence_tokens": ordered_tokens,
            "sequence_string": sequence_str,
            "smile_offset_velocity": float(metrics.get("smile_offset_velocity") or 0.0),
            "pattern_tag": cls._classify_micro_grammar(ordered_tokens),
        }

    @staticmethod
    def _compute_smile_drop_velocity(au12_series: np.ndarray, dt_sec: float = 0.1) -> float:
        if len(au12_series) < 4 or float(np.max(au12_series)) < 0.45:
            return 0.0
        peak_val = float(np.max(au12_series))
        apex_candidates = np.flatnonzero(au12_series >= 0.90 * peak_val)
        apex_idx = int(apex_candidates[-1]) if len(apex_candidates) else int(np.argmax(au12_series))
        if apex_idx >= len(au12_series) - 1:
            return 0.0
        post_apex = au12_series[apex_idx:]
        # Find first post-apex sample that drops back to baseline (<= 0.12) or overall post-apex minimum
        baseline_hits = np.flatnonzero(post_apex <= 0.12)
        min_idx = int(baseline_hits[0]) if len(baseline_hits) else int(np.argmin(post_apex))
        if min_idx == 0:
            return 0.0
        drop_amount = float(post_apex[0] - post_apex[min_idx])
        drop_time = min_idx * dt_sec
        return (drop_amount / drop_time) if drop_amount >= 0.30 else 0.0

    @staticmethod
    def _classify_micro_grammar(tokens: list[str]) -> str:
        negative_leads = {"AU04_BROW_FURROW", "AU15_LIP_CORNER_DROP", "AU20_LIP_STRETCH", "AU24_LIP_PRESS"}
        for idx, tok in enumerate(tokens):
            if tok in negative_leads and "AU12_NON_DUCHENNE_SMILE" in tokens[idx + 1 :]:
                return "NEGATIVE_APPRAISAL_MASKED_BY_SOCIAL_SMILE"
            if tok == "AU12_NON_DUCHENNE_SMILE" and "SNAP_OFF_SMILE_DROP" in tokens[idx + 1 :]:
                return "DELIBERATE_POSED_SMILE_SNAP_OFF"
        if "GAZE_AVERT_SHIFT" in tokens and "AU04_BROW_FURROW" in tokens:
            return "COGNITIVE_APPRAISAL_SEQUENCE"
        return "CONCORDANT_OR_STANDARD_SEQUENCE"
