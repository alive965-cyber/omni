"""Implements OmniPulse v2.0 Upgrade 7: Interlocutor Mirroring & Vocal Entrainment
(rapport_entrainment.py).
"""
from __future__ import annotations

from typing import Any
import numpy as np


class RapportEntrainmentAnalyzer:
    """Computes rolling dyadic vocal convergence (Acoustic Convergence Index, ACI in [-1.0, +1.0])
    across speaking rate, loudness, and F0 variability, plus Duchenne Smile Reciprocity.
    """

    @staticmethod
    def compute_turn_entrainment(dyadic_turns_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
        distances: list[float] = []
        results: list[dict[str, Any]] = []

        for idx, t in enumerate(dyadic_turns_metrics):
            d_wpm = abs(float(t.get("interviewer_wpm", 150.0)) - float(t.get("subject_wpm", 150.0))) / 50.0
            d_loud = abs(float(t.get("interviewer_loudness", 0.6)) - float(t.get("subject_loudness", 0.6))) / 0.5
            d_f0 = abs(float(t.get("interviewer_f0_std", 0.2)) - float(t.get("subject_f0_std", 0.2))) / 0.25
            dyad_dist = float(np.sqrt(0.4 * (d_wpm ** 2) + 0.3 * (d_loud ** 2) + 0.3 * (d_f0 ** 2)))
            distances.append(dyad_dist)

            window_dists = distances[max(0, idx - 4) : idx + 1]
            if len(window_dists) >= 3 and float(np.std(window_dists)) > 1e-5:
                corr = float(np.corrcoef(np.arange(len(window_dists)), window_dists)[0, 1])
                aci = round(-corr, 3) if not np.isnan(corr) else 0.0
            else:
                aci = 0.0

            reciprocity = "NO_PROMPT_SMILE"
            if bool(t.get("interviewer_smiled", False)):
                s_au12 = float(t.get("subject_au12", 0.0))
                s_au06 = float(t.get("subject_au06", 0.0))
                if s_au12 >= 0.40 and s_au06 >= 0.22:
                    reciprocity = "CONCORDANT_DUCHENNE_RECIPROCITY"
                elif s_au12 >= 0.40 and s_au06 < 0.15:
                    reciprocity = "POLITE_NON_DUCHENNE_RECIPROCITY"
                else:
                    reciprocity = "UNRECIPROCATED_FLAT_AFFECT"

            results.append({
                "turn_index": int(t.get("turn_index", idx)),
                "dyadic_acoustic_distance": round(dyad_dist, 3),
                "vocal_entrainment_aci": aci,
                "entrainment_state": (
                    "ACTIVE_VOCAL_ENTRAINMENT"
                    if aci > 0.35
                    else ("VOCAL_DIVERGENCE_FRICTION" if aci < -0.35 else "NEUTRAL_DYADIC_PACE")
                ),
                "smile_reciprocity_state": reciprocity,
            })
        return results

    @classmethod
    def summarize_session_entrainment(cls, windows: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute session-wide vocal entrainment trajectory and reciprocity summary from windows."""
        if not windows:
            return {
                "mean_aci": 0.0,
                "trajectory_state": "INSUFFICIENT_TURNS",
                "turn_metrics": [],
            }
        turn_inputs: list[dict[str, Any]] = []
        ref_wpm = 148.0
        ref_loud = 0.65
        ref_f0_std = 0.20

        for idx, w in enumerate(windows):
            m = w.get("metrics") or {}
            wpm = float(m.get("speech_rate_wpm") or 0.0)
            if wpm <= 0.0:
                continue
            loud = float(m.get("loudness_mean") or 0.65)
            f0_std = float(m.get("f0_semitone_std") or 0.20)
            au12 = float(m.get("au12_lip_corner_puller") or m.get("au12_smile") or 0.0)
            au06 = float(m.get("au06_cheek_raiser") or 0.0)
            turn_inputs.append({
                "turn_index": len(turn_inputs),
                "interviewer_wpm": ref_wpm,
                "subject_wpm": wpm,
                "interviewer_loudness": ref_loud,
                "subject_loudness": loud,
                "interviewer_f0_std": ref_f0_std,
                "subject_f0_std": f0_std,
                "interviewer_smiled": au12 >= 0.35,
                "subject_au12": au12,
                "subject_au06": au06,
            })
            # Smoothly update reference pace
            ref_wpm = 0.7 * ref_wpm + 0.3 * wpm

        turn_results = cls.compute_turn_entrainment(turn_inputs)
        acis = [r["vocal_entrainment_aci"] for r in turn_results if r["vocal_entrainment_aci"] != 0.0]
        mean_aci = round(sum(acis) / len(acis), 3) if acis else 0.0
        if mean_aci > 0.35:
            state = "ACTIVE_VOCAL_ENTRAINMENT"
        elif mean_aci < -0.35:
            state = "VOCAL_DIVERGENCE_FRICTION"
        else:
            state = "STABLE_CONVERSATIONAL_SYNCHRONY"

        return {
            "mean_aci": mean_aci,
            "trajectory_state": state,
            "turn_metrics": turn_results,
        }
