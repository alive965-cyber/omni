"""
backend/app/workers/adaptive_baseline.py
Fixes:
 - Bug #1: MAD=0.0 division-by-epsilon explosion (uses MIN_PHYSIOLOGICAL_SCALE + session STD fallback + [-8, +8] clamp)
 - Bug #4: Coalesces contiguous 2s anomaly windows into distinct Behavioral Episodes
 - Bug #5: Preserves all windows from 0.0s and scales resting floor to 25% on short recordings (<240s)
"""
from __future__ import annotations

from math import isfinite
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

MIN_PHYSIOLOGICAL_SCALE: Dict[str, float] = {
    "au01_inner_brow": 0.025,
    "au02_outer_brow": 0.025,
    "au04_brow_lowerer": 0.020,
    "au06_cheek_raiser": 0.020,
    "au12_lip_corner_puller": 0.025,
    "au15_lip_corner_depressor": 0.015,
    "au20_lip_stretcher": 0.020,
    "au24_lip_presser": 0.025,
    "au25_jaw_drop": 0.020,
    "facial_asymmetry_index": 0.015,
    "self_touch_ratio": 0.080,
    "hand_kinetic_energy": 0.020,
    "postural_lean_z": 0.050,
    "blink_rate_per_min": 4.500,
    "blink_duration_ms_mean": 25.000,
    "gaze_aversion_ratio": 0.080,
    "saccade_velocity_deg_s": 2.000,
    "f0_semitone_mean": 1.200,
    "f0_semitone_std": 0.025,
    "jitter_local_mean": 0.004,
    "shimmer_local_db_mean": 0.120,
    "hnr_db_mean": 0.800,
    "loudness_mean": 0.040,
    "speech_rate_wpm": 20.000,
    "disfluency_count": 0.500,
    "pause_ratio": 0.080,
    "response_latency_ms": 250.000,
    "first_person_pronoun_ratio": 0.100,
    "text_valence_score": 0.150,
    "hr_bpm_pos": 3.000,
    # Canonical aliases for seamless cross-worker lookup
    "hr_bpm": 3.000,
    "hr_bpm_chrom": 3.000,
    "jitter_local": 0.004,
    "shimmer_local_db": 0.120,
    "shimmer_db": 0.120,
    "hnr_db": 0.800,
    "blink_duration_ms": 25.000,
    "head_roll_deg": 2.000,
    "loudness_peaks_per_sec": 0.250,
    "spectral_flux_mean": 0.020,
    "epistemic_hedge_count": 0.500,
}


class DualHorizonBaselineEngine:
    @staticmethod
    def _face_conf_from_dict(w: Dict[str, Any]) -> float:
        if isinstance(w.get("face_confidence"), (int, float)):
            return float(w["face_confidence"])
        conf = w.get("confidence") or {}
        for k in ("face_tracking", "face_confidence", "overall"):
            val = conf.get(k)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return float(val)
        uptime = conf.get("face_tracking_uptime_pct")
        if isinstance(uptime, (int, float)) and not isinstance(uptime, bool):
            return float(uptime) / 100.0
        return 0.95

    @classmethod
    def compute_dual_horizon_zscores(
        cls,
        df_windows: pd.DataFrame | List[Dict[str, Any]],
        window_hop_sec: float = 2.0,
        floor_duration_sec: float | None = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Computes Dual-Horizon Resting Floor (Horizon A) and 12s-15s Pre-Stimulus Phasic Delta (Horizon B).
        Supports both a pandas DataFrame (returning `(df, baseline_metadata)`) and a list of telemetry window dicts
        (returning `(enriched_windows_list, baseline_metadata)`).
        """
        input_is_dataframe = isinstance(df_windows, pd.DataFrame)

        if not input_is_dataframe:
            raw_list = list(df_windows or [])
            if not raw_list:
                return [], {
                    "floor_start_sec": 0.0,
                    "floor_end_sec": 0.0,
                    "floor_duration_sec": 0.0,
                    "rppg_valid_coverage_pct": 0.0,
                    "habituation_delta_bpm": None,
                    "floor_features": {},
                }
            ordered_dicts = [dict(w) for w in sorted(raw_list, key=lambda item: float(item.get("start_sec", 0.0)))]
            flat_rows: List[Dict[str, Any]] = []
            for w in ordered_dicts:
                metrics = w.get("metrics") or {}
                row: Dict[str, Any] = {
                    "window_index": w.get("window_index", 0),
                    "start_sec": float(w.get("start_sec", 0.0)),
                    "end_sec": float(w.get("end_sec", 0.0)),
                    "face_confidence": cls._face_conf_from_dict(w),
                }
                for k, v in metrics.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and isfinite(float(v)):
                        row[k] = float(v)
                if "hr_bpm_pos" not in row and isinstance(row.get("hr_bpm"), (int, float)):
                    row["hr_bpm_pos"] = row["hr_bpm"]
                if "hr_bpm" not in row and isinstance(row.get("hr_bpm_pos"), (int, float)):
                    row["hr_bpm"] = row["hr_bpm_pos"]
                flat_rows.append(row)
            df = pd.DataFrame(flat_rows)
        else:
            df = df_windows.copy()
            if df.empty:
                return df, {
                    "floor_start_sec": 0.0,
                    "floor_end_sec": 0.0,
                    "floor_duration_sec": 0.0,
                    "rppg_valid_coverage_pct": 0.0,
                    "habituation_delta_bpm": None,
                    "floor_features": {},
                }

        # Ensure ALL windows from 0.0s to end_sec are preserved and unvoiced audio zeros are masked to NaN
        from app.services.report_enhancer import OmniPulseReportEnhancer
        df = OmniPulseReportEnhancer.mask_unvoiced_audio_zeros(df.sort_values("start_sec").reset_index(drop=True))
        total_duration = float(df["end_sec"].max()) if not df.empty else 0.0

        # Dynamic floor length: 25% of video for short clips (<240s), up to 90s for full 10-min sessions
        if floor_duration_sec is None:
            if total_duration < 240.0:
                floor_duration_sec = max(20.0, min(60.0, round(total_duration * 0.25, 1)))
            else:
                floor_duration_sec = 90.0
        else:
            if total_duration < 240.0:
                floor_duration_sec = max(20.0, min(60.0, round(total_duration * 0.25, 1)))

        floor_win_count = max(1, min(len(df), max(4, int(floor_duration_sec / max(window_hop_sec, 0.5)))))

        # Locate the calmest contiguous stretch across the recording
        arousal_cols = ["jitter_local_mean", "au04_brow_lowerer", "au24_lip_presser", "self_touch_ratio"]
        composite_arousal = np.zeros(len(df))
        for col in arousal_cols:
            if col in df.columns:
                valid_col = df[col].dropna()
                s = df[col].fillna(valid_col.median() if not valid_col.empty else 0.0)
                rng = float(s.max() - s.min()) + 1e-5
                composite_arousal += (s - float(s.min())) / rng

        df["_arousal_proxy"] = composite_arousal
        best_score = float("inf")
        best_start_idx = 0

        max_start = max(1, len(df) - floor_win_count + 1)
        for i in range(max_start):
            chunk = df.iloc[i : i + floor_win_count]
            if "face_confidence" in chunk.columns and chunk["face_confidence"].mean() < 0.70:
                continue
            score = float(chunk["_arousal_proxy"].mean())
            if score < best_score:
                best_score = score
                best_start_idx = i

        floor_df = df.iloc[best_start_idx : best_start_idx + floor_win_count]
        initial_df = df[df["end_sec"] <= min(60.0, total_duration * 0.4)]
        if initial_df.empty:
            initial_df = df.iloc[:floor_win_count]

        hr_col = "hr_bpm_pos" if "hr_bpm_pos" in df.columns else ("hr_bpm" if "hr_bpm" in df.columns else None)
        has_valid_hr = (
            hr_col is not None
            and int(df[hr_col].dropna().count()) >= max(3, int(len(df) * 0.2))
            and not initial_df[hr_col].dropna().empty
            and not floor_df[hr_col].dropna().empty
        )

        rppg_cov_pct = round(
            float(df[hr_col].notna().mean() * 100.0) if hr_col is not None else 0.0,
            1,
        )

        baseline_metadata: Dict[str, Any] = {
            "floor_start_sec": round(float(floor_df["start_sec"].min()), 1),
            "floor_end_sec": round(float(floor_df["end_sec"].max()), 1),
            "floor_duration_sec": floor_duration_sec,
            "rppg_valid_coverage_pct": rppg_cov_pct,
            "habituation_delta_bpm": (
                round(float(initial_df[hr_col].median() - floor_df[hr_col].median()), 2)
                if has_valid_hr and hr_col is not None
                else None
            ),
        }

        pre_lookback = max(2, int(12.0 / max(window_hop_sec, 0.5)))
        floor_features: Dict[str, Dict[str, float | int]] = {}

        # Evaluate all configured physiological scales plus any extra numeric telemetry columns
        target_scales: Dict[str, float] = dict(MIN_PHYSIOLOGICAL_SCALE)
        reserved_cols = {"window_index", "start_sec", "end_sec", "face_confidence", "_arousal_proxy", "illumination_change_pct", "head_velocity_deg_s", "head_pitch_deg", "head_yaw_deg"}
        for c in df.columns:
            if c not in target_scales and c not in reserved_cols and not c.endswith(("_z_floor", "_delta_z_phasic")):
                if pd.api.types.is_numeric_dtype(df[c]):
                    target_scales[c] = 0.05

        for col, min_scale in target_scales.items():
            if col not in df.columns:
                continue
            valid_floor = floor_df[col].dropna()
            valid_all = df[col].dropna()
            if valid_all.empty:
                df[f"{col}_z_floor"] = 0.0
                df[f"{col}_delta_z_phasic"] = 0.0
                continue

            median_val = float(valid_floor.median()) if not valid_floor.empty else float(valid_all.median())
            mad_val = (
                float(np.median(np.abs(valid_floor - median_val)))
                if not valid_floor.empty
                else float(np.median(np.abs(valid_all - median_val)))
            )
            session_std = float(valid_all.std()) if len(valid_all) > 2 else 0.0
            if not isfinite(session_std):
                session_std = 0.0

            # CRITICAL FIX FOR BUG #1: Never divide by 1e-5 when MAD == 0.0
            safe_scale = max(1.4826 * mad_val, 0.5 * session_std, float(min_scale))

            z_floor = ((df[col] - median_val) / safe_scale).clip(lower=-8.0, upper=8.0).round(2)
            df[f"{col}_z_floor"] = z_floor.fillna(0.0)

            rolling_pre = df[f"{col}_z_floor"].shift(1).rolling(window=pre_lookback, min_periods=1).mean().fillna(0.0)
            df[f"{col}_delta_z_phasic"] = (df[f"{col}_z_floor"] - rolling_pre).clip(lower=-8.0, upper=8.0).round(2)

            baseline_metadata[f"{col}_median"] = round(median_val, 4)
            baseline_metadata[f"{col}_mad"] = round(mad_val, 4)
            baseline_metadata[f"{col}_safe_scale"] = round(safe_scale, 4)
            baseline_metadata[f"{col}_floor_median"] = round(median_val, 4)
            baseline_metadata[f"{col}_floor_mad"] = round(mad_val, 4)
            floor_features[col] = {
                "median": round(median_val, 4),
                "mad": round(mad_val, 4),
                "std": round(session_std, 4),
                "safe_scale": round(safe_scale, 4),
                "sample_count": int(len(valid_floor) if not valid_floor.empty else len(valid_all)),
            }

        baseline_metadata["floor_features"] = floor_features
        df.drop(columns=["_arousal_proxy"], inplace=True, errors="ignore")

        if input_is_dataframe:
            return df, baseline_metadata

        # Enrich list[dict[str, Any]] windows in-place with clamped [-8.0, +8.0] Z-scores
        for idx, w in enumerate(ordered_dicts):
            row_s = df.iloc[idx]
            dual_z: Dict[str, float] = {}
            phasic_deltas: Dict[str, float] = {}
            base_z: Dict[str, float | None] = dict(w.get("baseline_z_scores") or {})
            max_abs_floor_z = 0.0
            max_abs_phasic_z = 0.0

            for col in floor_features:
                zf = float(np.clip(float(row_s.get(f"{col}_z_floor", 0.0)), -8.0, 8.0))
                zp = float(np.clip(float(row_s.get(f"{col}_delta_z_phasic", 0.0)), -8.0, 8.0))
                dual_z[f"{col}_z_floor"] = round(zf, 2)
                dual_z[f"{col}_delta_z_phasic"] = round(zp, 2)
                phasic_deltas[col] = round(zp, 2)
                # Ensure baseline_z_scores also uses the safe clamped scale
                if col in base_z and base_z[col] is not None:
                    base_z[col] = round(float(np.clip(float(base_z[col]), -8.0, 8.0)), 2)
                else:
                    base_z[col] = round(zf, 2)
                max_abs_floor_z = max(max_abs_floor_z, abs(zf))
                max_abs_phasic_z = max(max_abs_phasic_z, abs(zp))

            # Clamp any remaining keys in baseline_z_scores to [-8.0, +8.0]
            for k, v in list(base_z.items()):
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    base_z[k] = round(float(np.clip(float(v), -8.0, 8.0)), 2)

            if max_abs_floor_z >= 1.6 and max_abs_phasic_z >= 1.2:
                attribution = "ACUTE_STIMULUS_REACTION"
            elif max_abs_floor_z >= 1.6 and max_abs_phasic_z < 0.6:
                attribution = "CARRYOVER_RUMINATION"
                flags = list(w.get("behavioral_flags") or [])
                if "CARRYOVER_RUMINATION" not in flags:
                    flags.append("CARRYOVER_RUMINATION")
                w["behavioral_flags"] = flags
            else:
                attribution = "WITHIN_BASELINE"

            w["baseline_z_scores"] = base_z
            w["dual_horizon_z_scores"] = dual_z
            w["phasic_delta_z_json"] = phasic_deltas
            w["phasic_attribution_state"] = attribution

        return ordered_dicts, baseline_metadata

    @staticmethod
    def coalesce_anomaly_episodes(
        flagged_windows: List[Dict[str, Any]],
        max_gap_sec: float = 2.5,
        max_episode_duration_sec: float = 12.0,
        max_episodes: int = 6,
    ) -> List[Dict[str, Any]]:
        """
        Merges back-to-back 2s windows (e.g., 01:26-01:32, 01:32-01:34, 01:34-01:36)
        into coherent Behavioral Episodes and returns the top `max_episodes` by severity.
        """
        if not flagged_windows:
            return []

        def _start(item: Dict[str, Any]) -> float:
            if "start_sec" in item and item["start_sec"] is not None:
                return float(item["start_sec"])
            tr = item.get("time_range_sec") or [0.0, 2.0]
            return float(tr[0])

        def _end(item: Dict[str, Any]) -> float:
            if "end_sec" in item and item["end_sec"] is not None:
                return float(item["end_sec"])
            tr = item.get("time_range_sec") or [0.0, 2.0]
            return float(tr[-1])

        def _clean_quote(q: Any) -> str:
            s = str(q or "").strip()
            if s in {".", "..", "...", "Tsk.", "Tsk", "Mas... Não?", "Mas... Nao?"}:
                return ""
            return s

        wins = sorted(flagged_windows, key=_start)
        merged: List[Dict[str, Any]] = []
        curr = wins[0].copy()
        curr["start_sec"] = _start(curr)
        curr["end_sec"] = _end(curr)
        curr["time_range_sec"] = [curr["start_sec"], curr["end_sec"]]
        curr["formatted_time"] = (
            f"{int(curr['start_sec']//60):02d}:{int(curr['start_sec']%60):02d} - "
            f"{int(curr['end_sec']//60):02d}:{int(curr['end_sec']%60):02d}"
        )
        curr["subject_response_quote"] = _clean_quote(
            curr.get("subject_response_quote") or curr.get("transcript_excerpt")
        )
        curr["transcript_excerpt"] = curr["subject_response_quote"] or None
        curr["dual_horizon_z_scores"] = {
            k: round(float(np.clip(float(v), -8.0, 8.0)), 2)
            for k, v in dict(curr.get("dual_horizon_z_scores", {})).items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        curr["baseline_z_scores"] = {
            k: round(float(np.clip(float(v), -8.0, 8.0)), 2)
            for k, v in dict(curr.get("baseline_z_scores", {})).items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        curr["behavioral_flags"] = list(curr.get("behavioral_flags", []))

        for nxt in wins[1:]:
            nxt_start = _start(nxt)
            nxt_end = _end(nxt)
            gap = nxt_start - float(curr["end_sec"])
            new_dur = nxt_end - float(curr["start_sec"])

            if gap <= max_gap_sec and new_dur <= max_episode_duration_sec:
                curr["end_sec"] = nxt_end
                curr["time_range_sec"] = [float(curr["start_sec"]), nxt_end]
                curr["formatted_time"] = (
                    f"{int(curr['start_sec']//60):02d}:{int(curr['start_sec']%60):02d} - "
                    f"{int(curr['end_sec']//60):02d}:{int(curr['end_sec']%60):02d}"
                )
                # Merge quotes cleanly
                nxt_q = _clean_quote(nxt.get("subject_response_quote") or nxt.get("transcript_excerpt"))
                curr_q = _clean_quote(curr.get("subject_response_quote") or curr.get("transcript_excerpt"))
                if nxt_q and nxt_q not in curr_q:
                    curr["subject_response_quote"] = f"{curr_q} {nxt_q}".strip()
                    curr["transcript_excerpt"] = curr["subject_response_quote"]

                # Retain maximum |Z| across merged windows (clamped to [-8.0, +8.0])
                for k, v in (nxt.get("dual_horizon_z_scores") or {}).items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        clamped_v = round(float(np.clip(float(v), -8.0, 8.0)), 2)
                        if abs(clamped_v) > abs(float(curr["dual_horizon_z_scores"].get(k, 0.0))):
                            curr["dual_horizon_z_scores"][k] = clamped_v

                for k, v in (nxt.get("baseline_z_scores") or {}).items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        clamped_v = round(float(np.clip(float(v), -8.0, 8.0)), 2)
                        if abs(clamped_v) > abs(float(curr["baseline_z_scores"].get(k, 0.0))):
                            curr["baseline_z_scores"][k] = clamped_v

                # Merge flags
                curr["behavioral_flags"] = sorted(
                    set(curr["behavioral_flags"] + list(nxt.get("behavioral_flags", [])))
                )
                if "PACIFYING_NECK_TOUCH" in curr["behavioral_flags"]:
                    curr["event_category"] = "PACIFYING_ADAPTOR_CLUSTER"
                elif "CROSS_MODAL_INCONGRUENCE" in curr["behavioral_flags"]:
                    curr["event_category"] = "CROSS_MODAL_INCONGRUENCE"

                # Concatenate active 100ms micro-sequences
                nxt_seq = nxt.get("micro_sequence_chain", "STEADY_BASELINE")
                if nxt_seq and nxt_seq != "STEADY_BASELINE":
                    if curr.get("micro_sequence_chain") in (None, "", "STEADY_BASELINE"):
                        curr["micro_sequence_chain"] = nxt_seq
                    elif nxt_seq not in curr["micro_sequence_chain"]:
                        curr["micro_sequence_chain"] = f"{curr['micro_sequence_chain']} -> {nxt_seq}"
            else:
                merged.append(curr)
                curr = nxt.copy()
                curr["start_sec"] = nxt_start
                curr["end_sec"] = nxt_end
                curr["time_range_sec"] = [nxt_start, nxt_end]
                curr["formatted_time"] = (
                    f"{int(curr['start_sec']//60):02d}:{int(curr['start_sec']%60):02d} - "
                    f"{int(curr['end_sec']//60):02d}:{int(curr['end_sec']%60):02d}"
                )
                curr["subject_response_quote"] = _clean_quote(
                    curr.get("subject_response_quote") or curr.get("transcript_excerpt")
                )
                curr["transcript_excerpt"] = curr["subject_response_quote"] or None
                curr["dual_horizon_z_scores"] = {
                    k: round(float(np.clip(float(v), -8.0, 8.0)), 2)
                    for k, v in dict(curr.get("dual_horizon_z_scores", {})).items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                }
                curr["baseline_z_scores"] = {
                    k: round(float(np.clip(float(v), -8.0, 8.0)), 2)
                    for k, v in dict(curr.get("baseline_z_scores", {})).items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                }
                curr["behavioral_flags"] = list(curr.get("behavioral_flags", []))

        merged.append(curr)

        # Rank episodes by sum of top 4 |Z| scores and return top `max_episodes` chronologically
        def _episode_severity(ep: Dict[str, Any]) -> float:
            z_dict = ep.get("dual_horizon_z_scores") or ep.get("baseline_z_scores") or {}
            non_beh = {"illumination_change_pct", "head_velocity_deg_s", "head_pitch_deg", "head_yaw_deg"}
            vals = sorted(
                [
                    abs(float(v))
                    for k, v in z_dict.items()
                    if v is not None
                    and not str(k).endswith("_phasic")
                    and str(k).replace("_z_floor", "") not in non_beh
                ],
                reverse=True,
            )
            quote_bonus = 1.5 if ep.get("subject_response_quote") else 0.0
            return sum(vals[:4]) + quote_bonus

        merged.sort(key=_episode_severity, reverse=True)
        top_episodes = sorted(merged[:max_episodes], key=lambda x: float(x["start_sec"]))
        for idx, ep in enumerate(top_episodes, 1):
            ep["event_id"] = f"episode-{idx:02d}"
        return top_episodes


coalesce_adjacent_anomaly_windows = DualHorizonBaselineEngine.coalesce_anomaly_episodes
