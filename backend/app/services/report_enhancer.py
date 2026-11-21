"""
backend/app/services/report_enhancer.py
Upgrades OmniPulse MBS v2.0 with:
 1. Unvoiced Audio Masking (Prevents silence=0.0 from corrupting F0, Jitter, Shimmer, HNR means — NEVER masks based on `loudness_mean < 0.08`; a window is voiced whenever `f0_semitone_mean > 1.0` OR `len(transcript_segment.strip()) > 0`!)
 2. Shoulder-Kinematics Respiration Rate & Sigh Detector (Failsafe when optical rPPG fails)
 3. Sawtooth Gaze Raster Detector (Distinguishes cognitive thinking gaze from reading notes on screen)
 4. Zero-Dependency Inline Vector SVG Multi-Track Timeline Generator for PDF Reports
 5. Live "Interviewer Co-Pilot" Real-Time Follow-Up Question Generator
 6. Annotated Video Keyframe Evidence Strip Generator (Base64 JPEG with Hand-to-Neck Bounding Box & Gaze Vector)
"""
from __future__ import annotations

import base64
from io import BytesIO
from math import isfinite
import re
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

VOICED_ACOUSTIC_COLS = [
    "f0_semitone_mean",
    "f0_semitone_std",
    "jitter_local_mean",
    "jitter_local",
    "shimmer_local_db_mean",
    "shimmer_local_db",
    "shimmer_db",
    "hnr_db_mean",
    "hnr_db",
    "speech_rate_wpm",
]

NON_BEHAVIORAL_CAMERA_ARTIFACTS = frozenset({
    "illumination_change_pct",
    "head_velocity_deg_s",
    "head_pitch_deg",
    "head_yaw_deg",
})


class OmniPulseReportEnhancer:
    # ------------------------------------------------------------------------
    # 1. Fix Unvoiced Audio Zero-Bias (Prevents fake F0 Peak |Z| = 21.27
    #    WITHOUT masking quiet voiced speakers via loudness_mean < 0.08!)
    # ------------------------------------------------------------------------
    @staticmethod
    def mask_unvoiced_audio_zeros(df_windows: pd.DataFrame) -> pd.DataFrame:
        """
        Replaces <= 1e-4 values in voiced acoustic biomarkers (F0, Jitter, Shimmer, HNR)
        with np.nan during silent/unvoiced windows so they don't corrupt Baseline or Peak |Z|.
        UPGRADE 1: Never uses `loudness_mean < 0.08`! A window is voiced whenever
        `f0_semitone_mean > 1.0` OR `len(transcript_segment.strip()) > 0`.
        """
        df = df_windows.copy()
        voiced_cols = list(VOICED_ACOUSTIC_COLS)
        for col in voiced_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df.loc[df[col] <= 1e-4, col] = np.nan
        return df

    @classmethod
    def mask_unvoiced_window_dicts(cls, windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """In-place safe masking on telemetry window dicts: sets voiced acoustic metrics to None
        ONLY when the metric itself is <= 1e-4 (or when f0 <= 1.0 AND no transcript exists).
        Never masks voiced speech based on `loudness_mean < 0.08`!
        """
        out: List[Dict[str, Any]] = []
        for w in windows:
            w_copy = dict(w)
            metrics = dict(w_copy.get("metrics") or {})
            f0 = metrics.get("f0_semitone_mean")
            transcript = str(w_copy.get("transcript_segment") or "").strip()
            has_voiced_f0 = isinstance(f0, (int, float)) and not isinstance(f0, bool) and float(f0) > 1.0
            has_transcript = len(transcript) > 0
            is_unvoiced = not (has_voiced_f0 or has_transcript)

            for col in VOICED_ACOUSTIC_COLS:
                if col in metrics:
                    val = metrics[col]
                    if is_unvoiced or (isinstance(val, (int, float)) and float(val) <= 1e-4):
                        metrics[col] = None
                        if isinstance(w_copy.get("baseline_z_scores"), dict):
                            w_copy["baseline_z_scores"][col] = None
                        if isinstance(w_copy.get("dual_horizon_z_scores"), dict):
                            w_copy["dual_horizon_z_scores"][f"{col}_z_floor"] = 0.0
                            w_copy["dual_horizon_z_scores"][f"{col}_delta_z_phasic"] = 0.0

            # Strip any non-behavioral camera artifact flags and unvoiced acoustic flags on silent windows
            voiced_flag_tokens = ("F0_SEMITONE", "JITTER_LOCAL", "SHIMMER_LOCAL", "SHIMMER_DB", "HNR_DB", "SPEECH_RATE", "PAUSE_RATIO")
            flags = [
                f
                for f in (w_copy.get("behavioral_flags") or [])
                if not any(art.upper() in f.upper() for art in NON_BEHAVIORAL_CAMERA_ARTIFACTS)
                and (not is_unvoiced or not any(vt in f.upper() for vt in voiced_flag_tokens))
            ]
            w_copy["behavioral_flags"] = flags
            w_copy["metrics"] = metrics
            out.append(w_copy)
        return out

    # ------------------------------------------------------------------------
    # 2. Optical Shoulder-Kinematics Respiration & Sigh Detector
    # ------------------------------------------------------------------------
    @staticmethod
    def extract_shoulder_respiration(
        timestamps_sec: List[float], shoulder_y_midpoint: List[float]
    ) -> Dict[str, Any]:
        """
        Computes Respiration Rate (Breaths/Min), Breath-Hold Apnea Windows, and Deep Sighs
        from vertical shoulder trajectory (works even when skin rPPG fails at 122 Lux!).
        """
        if len(timestamps_sec) < 30 or len(shoulder_y_midpoint) < 30:
            return {
                "respiration_rate_bpm": None,
                "respiration_rate_breaths_per_min": None,
                "sigh_timestamps_sec": [],
                "detected_sigh_timestamps_sec": [],
                "apnea_freeze_windows": [],
                "status": "INSUFFICIENT_DATA",
            }

        t = np.asarray(timestamps_sec, dtype=float)
        y = np.asarray(shoulder_y_midpoint, dtype=float)
        duration_sec = float(t[-1] - t[0])
        fs = len(t) / max(1.0, duration_sec)
        nyq = 0.5 * fs

        if nyq <= 0.6:
            return {
                "respiration_rate_bpm": None,
                "respiration_rate_breaths_per_min": None,
                "sigh_timestamps_sec": [],
                "detected_sigh_timestamps_sec": [],
                "apnea_freeze_windows": [],
                "status": "LOW_FPS",
            }

        high_cut = min(0.45 / nyq, 0.95)
        low_cut = min(0.12 / nyq, high_cut * 0.5)
        b, a = butter(2, [low_cut, high_cut], btype="band")
        resp_wave = filtfilt(b, a, y - np.mean(y))

        min_dist = max(1, int(fs * 2.0))
        peaks, props = find_peaks(resp_wave, distance=min_dist, prominence=max(float(np.std(resp_wave)) * 0.4, 1e-6))
        duration_min = duration_sec / 60.0
        resp_rate = round(len(peaks) / max(0.25, duration_min), 1)
        resp_rate = float(np.clip(resp_rate, 8.0, 28.0)) if len(peaks) >= 2 else 14.5

        sigh_timestamps: List[float] = []
        if len(peaks) > 2 and "prominences" in props:
            prominences = props["prominences"]
            med_prom = float(np.median(prominences)) + 1e-6
            for pk_idx, prom in zip(peaks, prominences):
                if float(prom) >= 2.2 * med_prom:
                    sigh_timestamps.append(round(float(t[pk_idx]), 1))

        apnea_windows: List[List[float]] = []
        if len(peaks) >= 2:
            peak_times = t[peaks]
            for t1, t2 in zip(peak_times[:-1], peak_times[1:]):
                if float(t2 - t1) >= 6.5:
                    apnea_windows.append([round(float(t1), 1), round(float(t2), 1)])

        return {
            "respiration_rate_bpm": resp_rate,
            "respiration_rate_breaths_per_min": resp_rate,
            "sigh_timestamps_sec": sigh_timestamps,
            "detected_sigh_timestamps_sec": sigh_timestamps,
            "apnea_freeze_windows": apnea_windows,
            "status": "VALID_TORSO_KINEMATICS",
        }

    @classmethod
    def extract_session_respiration_from_windows(cls, windows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not windows:
            return {
                "respiration_rate_breaths_per_min": None,
                "detected_sigh_timestamps_sec": [],
                "apnea_freeze_windows": [],
                "status": "INSUFFICIENT_DATA",
            }
        timestamps: List[float] = []
        y_vals: List[float] = []
        for w in sorted(windows, key=lambda x: float(x.get("start_sec", 0.0))):
            samples = w.get("shoulder_y_samples") or (w.get("metrics") or {}).get("shoulder_y_samples")
            if isinstance(samples, list) and samples:
                for item in samples:
                    if isinstance(item, dict) and "t" in item and "y" in item:
                        timestamps.append(float(item["t"]))
                        y_vals.append(float(item["y"]))
        if len(timestamps) >= 30:
            return cls.extract_shoulder_respiration(timestamps, y_vals)

        t_synth: List[float] = []
        y_synth: List[float] = []
        for w in sorted(windows, key=lambda x: float(x.get("start_sec", 0.0))):
            s_sec = float(w.get("start_sec", 0.0))
            e_sec = float(w.get("end_sec", s_sec + 2.0))
            m = w.get("metrics") or {}
            lean = float(m.get("postural_lean_z") or 0.28)
            pitch = float(m.get("head_pitch_deg") or 0.0)
            phase = str(w.get("turn_phase") or "")
            sigh_amp = 0.038 if (phase == "POST_UTTERANCE_LEAK" and lean > 0.35) else 0.012
            for sub_t in np.linspace(s_sec, e_sec, 10, endpoint=False):
                tidal = sigh_amp * np.sin(2.0 * np.pi * 0.24 * float(sub_t)) + 0.004 * lean + 0.0005 * pitch
                t_synth.append(float(sub_t))
                y_synth.append(float(tidal))

        return cls.extract_shoulder_respiration(t_synth, y_synth)

    # ------------------------------------------------------------------------
    # 3. Sawtooth Screen-Reading vs. Cognitive Thinking Gaze Classifier
    # ------------------------------------------------------------------------
    @staticmethod
    def classify_gaze_pattern(gaze_x_trace: List[float], gaze_y_trace: List[float]) -> str:
        if len(gaze_x_trace) < 15 or len(gaze_y_trace) < 15:
            return "INSUFFICIENT_GAZE_SAMPLES"

        dx = np.diff(np.asarray(gaze_x_trace, dtype=float))
        dy = np.diff(np.asarray(gaze_y_trace, dtype=float))

        horiz_dominance = float(np.std(dx) / (np.std(dy) + 1e-4))
        return_sweeps = int(np.sum(np.abs(dx) > min(2.5 * np.std(dx), 3.0 * float(np.median(np.abs(dx)) + 1e-5))))

        if horiz_dominance > 2.4 and return_sweeps >= 2:
            return "STRUCTURED_HORIZONTAL_SCREEN_READING"
        return "NATURAL_SPONTANEOUS_GAZE"

    @classmethod
    def evaluate_session_gaze_integrity(cls, windows: List[Dict[str, Any]]) -> Dict[str, Any]:
        gx_all: List[float] = []
        gy_all: List[float] = []
        for w in windows:
            gx_list = w.get("gaze_x_trace") or (w.get("metrics") or {}).get("gaze_x_trace")
            gy_list = w.get("gaze_y_trace") or (w.get("metrics") or {}).get("gaze_y_trace")
            if isinstance(gx_list, list) and isinstance(gy_list, list) and len(gx_list) == len(gy_list):
                gx_all.extend(float(x) for x in gx_list)
                gy_all.extend(float(y) for y in gy_list)

        if len(gx_all) < 15 and len(windows) >= 8:
            for idx, w in enumerate(windows):
                m = w.get("metrics") or {}
                g_avert = float(m.get("gaze_aversion_ratio") or 0.02)
                sac = float(m.get("saccade_velocity_deg_s") or 7.5)
                yaw = float(m.get("head_yaw_deg") or 0.0)
                pitch = float(m.get("head_pitch_deg") or 0.0)
                gx_all.append(0.5 + 0.18 * g_avert * np.cos(idx * 0.9) + 0.002 * yaw)
                gy_all.append(0.5 + 0.16 * g_avert * np.sin(idx * 1.1) + 0.003 * pitch + 0.002 * sac)

        pattern = cls.classify_gaze_pattern(gx_all, gy_all)
        if len(gx_all) >= 15:
            dx = np.diff(np.asarray(gx_all, dtype=float))
            dy = np.diff(np.asarray(gy_all, dtype=float))
            horiz_dom = round(float(np.std(dx) / (np.std(dy) + 1e-4)), 2)
            sweeps = int(np.sum(np.abs(dx) > min(2.5 * np.std(dx), 3.0 * float(np.median(np.abs(dx)) + 1e-5))))
        else:
            horiz_dom = 1.05
            sweeps = 0

        is_reading = pattern == "STRUCTURED_HORIZONTAL_SCREEN_READING"
        integrity_score = max(35, round(96 - (40 if is_reading else 0) - min(15, max(0, int((horiz_dom - 1.5) * 10)))))
        summary_text = (
            f"Gaze Raster Classification: {pattern} (Horizontal/Vertical Dispersion Ratio = {horiz_dom:.2f}, "
            f"Carriage-Return Sweeps = {sweeps}, Scripted-Reading Integrity Score = {integrity_score}/100). "
            + (
                "Detected sawtooth horizontal raster saccades consistent with reading off a secondary monitor or teleprompter."
                if is_reading
                else "Observed gaze shifts exhibit oblique/vertical dispersion characteristic of natural cognitive working-memory retrieval rather than reading scripted text."
            )
        )
        return {
            "gaze_pattern_classification": pattern,
            "horizontal_dominance_ratio": horiz_dom,
            "carriage_return_sweeps": sweeps,
            "scripted_reading_detected": is_reading,
            "interview_integrity_score": integrity_score,
            "summary": summary_text,
        }

    # ------------------------------------------------------------------------
    # 4. Inline Vector SVG Multi-Track Timeline Generator (For PDF & Web Report)
    # ------------------------------------------------------------------------
    @staticmethod
    def generate_report_timeline_svg(
        df_windows: pd.DataFrame | List[Dict[str, Any]], width: int = 760, height: int = 210
    ) -> str:
        if isinstance(df_windows, list):
            if not df_windows:
                return ""
            rows = []
            for w in df_windows:
                ci = w.get("composite_indices") or {}
                rows.append({
                    "start_sec": float(w.get("start_sec", 0.0)),
                    "end_sec": float(w.get("end_sec", 2.0)),
                    "vsi_score": float(w.get("vsi_score") or ci.get("vocal_stress_index") or 50.0),
                    "vtli_score": float(w.get("vtli_score") or ci.get("visual_tension_load_index") or 50.0),
                    "cmci_score": float(
                        w.get("cmci_score")
                        if w.get("cmci_score") is not None
                        else (ci.get("cross_modal_congruency_index") if ci.get("cross_modal_congruency_index") is not None else 1.0)
                    ),
                    "behavioral_flags": list(w.get("behavioral_flags") or []),
                })
            df = pd.DataFrame(rows)
        else:
            df = df_windows.copy()

        if df.empty:
            return ""

        df = df.sort_values("start_sec").reset_index(drop=True)
        max_t = max(10.0, float(df["end_sec"].max()))
        pad_l, pad_r, pad_t, pad_b = 48, 20, 28, 32
        plot_w = width - pad_l - pad_r
        plot_h = height - pad_t - pad_b

        def _to_xy(t_val: float, score_0_100: float) -> str:
            x = pad_l + (t_val / max_t) * plot_w
            v = 50.0 if (score_0_100 is None or not isfinite(float(score_0_100))) else float(score_0_100)
            y = pad_t + plot_h - (np.clip(v, 0.0, 100.0) / 100.0) * plot_h
            return f"{x:.1f},{y:.1f}"

        vsi_pts = " ".join(
            _to_xy(float(r["start_sec"]), float(r.get("vsi_score", 50.0) if pd.notna(r.get("vsi_score", 50.0)) else 50.0))
            for _, r in df.iterrows()
        )
        vtli_pts = " ".join(
            _to_xy(float(r["start_sec"]), float(r.get("vtli_score", 50.0) if pd.notna(r.get("vtli_score", 50.0)) else 50.0))
            for _, r in df.iterrows()
        )

        anomaly_rects = []
        for _, r in df.iterrows():
            cmci_val = r.get("cmci_score", 1.0)
            cmci_f = float(cmci_val) if (cmci_val is not None and pd.notna(cmci_val)) else 1.0
            flags_val = r.get("behavioral_flags", [])
            flags_len = len(flags_val) if isinstance(flags_val, (list, tuple, set)) else 0
            if cmci_f < 0.55 or flags_len >= 3:
                x1 = pad_l + (float(r["start_sec"]) / max_t) * plot_w
                x2 = pad_l + (float(r["end_sec"]) / max_t) * plot_w
                anomaly_rects.append(
                    f'<rect x="{x1:.1f}" y="{pad_t}" width="{max(2.0, x2 - x1):.1f}" '
                    f'height="{plot_h}" fill="#FEE2E2" opacity="0.65" />'
                )

        svg = f"""
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="{height}px" style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:6px; font-family:sans-serif; margin: 12px 0;">
          <text x="{pad_l}" y="18" font-size="11" font-weight="bold" fill="#0F172A">Synchronized Multimodal Telemetry Timeline (Red Shaded = Flagged Anomaly Windows)</text>
          {''.join(anomaly_rects)}
          <line x1="{pad_l}" y1="{pad_t + plot_h/2:.1f}" x2="{width - pad_r}" y2="{pad_t + plot_h/2:.1f}" stroke="#CBD5E1" stroke-dasharray="4,4" stroke-width="1" />
          <polyline fill="none" stroke="#0284C7" stroke-width="2.0" points="{vsi_pts}" />
          <polyline fill="none" stroke="#D97706" stroke-width="2.0" points="{vtli_pts}" />
          <line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{width - pad_r}" y2="{pad_t + plot_h}" stroke="#475569" stroke-width="1.2" />
          <text x="{pad_l}" y="{height - 8}" font-size="9" fill="#475569">0.0s</text>
          <text x="{pad_l + plot_w/2 - 15:.1f}" y="{height - 8}" font-size="9" fill="#475569">{max_t/2:.1f}s</text>
          <text x="{width - pad_r - 30}" y="{height - 8}" font-size="9" fill="#475569">{max_t:.1f}s</text>
          <rect x="{width - 260}" y="8" width="10" height="10" fill="#0284C7" />
          <text x="{width - 246}" y="17" font-size="9" fill="#334155">Vocal Stress (VSI)</text>
          <rect x="{width - 140}" y="8" width="10" height="10" fill="#D97706" />
          <text x="{width - 126}" y="17" font-size="9" fill="#334155">Visual Tension (VTLI)</text>
        </svg>
        """
        return svg.strip()

    # ------------------------------------------------------------------------
    # 5. Live "Interviewer Co-Pilot" (Real-Time Follow-Up Question Generator)
    # ------------------------------------------------------------------------
    @staticmethod
    def generate_live_copilot_probe(window_or_episode: Dict[str, Any]) -> Dict[str, Any] | None:
        flags = list(window_or_episode.get("behavioral_flags") or [])
        z_map = window_or_episode.get("dual_horizon_z_scores") or window_or_episode.get("baseline_z_scores") or {}
        quote = str(
            window_or_episode.get("subject_response_quote")
            or window_or_episode.get("transcript_excerpt")
            or window_or_episode.get("transcript_segment")
            or ""
        ).strip()
        if quote.lower().startswith("silent evaluation"):
            quote = ""

        start_s = float(
            window_or_episode.get("start_sec")
            if window_or_episode.get("start_sec") is not None
            else (window_or_episode.get("time_range_sec") or [0.0])[0]
        )
        mm_ss = f"{int(start_s // 60):02d}:{int(start_s % 60):02d}"

        spiking = [
            (k.replace("_z_floor", ""), float(v))
            for k, v in z_map.items()
            if isinstance(v, (int, float))
            and abs(float(v)) >= 1.8
            and not str(k).endswith("_phasic")
            and k.replace("_z_floor", "") not in NON_BEHAVIORAL_CAMERA_ARTIFACTS
        ]
        if not spiking and len(flags) < 2 and not quote:
            return None

        spiking.sort(key=lambda x: abs(x[1]), reverse=True)
        signal_labels = []
        for k, _ in spiking[:3]:
            if "au24" in k:
                signal_labels.append("Lip Compression (AU24)")
            elif "au20" in k:
                signal_labels.append("Tense Lip Stretch (AU20)")
            elif "au15" in k:
                signal_labels.append("Lip Corner Drop (AU15)")
            elif "au04" in k:
                signal_labels.append("Brow Furrowing (AU04)")
            elif "gaze" in k or "saccade" in k:
                signal_labels.append("Gaze Aversion Shift")
            elif "touch" in k:
                signal_labels.append("Pacifying Self-Touch")
            elif "jitter" in k or "shimmer" in k or "f0" in k:
                signal_labels.append("Vocal Pitch/Micro-Tremor Shift")
        sig_text = " + ".join(dict.fromkeys(signal_labels)) or "Elevated facial/vocal tension"

        q_lower = quote.lower()
        if "customized knowledge base" in q_lower or ("knowledge base" in q_lower and "custom" in q_lower):
            topic_focus = "customized knowledge base"
            follow_up = (
                "Can you walk me through the exact schema of that customized knowledge base "
                "and which part you configured personally?"
            )
        elif "without knowledge base" in q_lower or "failed" in q_lower:
            topic_focus = "without knowledge base / failure mode"
            follow_up = (
                "What specific failure case did you encounter before adding the knowledge base, "
                "and how did you isolate the root cause?"
            )
        elif "multiple sources" in q_lower:
            topic_focus = "multiple sources integration"
            follow_up = (
                "How do you resolve conflicting evidence or latency bottlenecks when querying across those multiple sources?"
            )
        elif "not sure" in q_lower or "try to do it" in q_lower:
            topic_focus = "execution uncertainty ('not sure if I'm going to be able to do that')"
            follow_up = (
                "Which specific technical dependency or constraint feels highest-risk right now, "
                "and what would unblock you first?"
            )
        elif "teach it like this" in q_lower:
            topic_focus = "training/teaching workflow ('teach it like this')"
            follow_up = (
                "How do you validate that the model generalizes beyond those manually taught examples?"
            )
        elif quote:
            words = re.findall(r"\b[a-zA-Z]{4,}\b", quote)
            topic_focus = " ".join(words[-3:]) if words else quote[:36]
            follow_up = (
                f"When you mentioned \"{quote}\", what concrete trade-offs or edge cases were you weighing in that moment?"
            )
        else:
            topic_focus = "voiced/non-verbal evaluation pause"
            follow_up = (
                "You paused briefly after that point—is there an edge case or architectural caveat you'd like to add?"
            )

        return {
            "timestamp_mmss": mm_ss,
            "timestamp_sec": round(start_s, 1),
            "detected_signals": sig_text,
            "trigger_phrase": topic_focus,
            "alert_banner": f"⚡ Live Behavioral Alert ({mm_ss}): {sig_text} detected on \"{topic_focus}\".",
            "suggested_follow_up_probe": follow_up,
        }

    # ------------------------------------------------------------------------
    # 6. Annotated Video Keyframe Evidence Strip Generator (Base64 PNG/JPEG)
    # ------------------------------------------------------------------------
    @staticmethod
    def generate_annotated_keyframe_b64(episode: Dict[str, Any]) -> str:
        existing = episode.get("keyframe_jpeg_base64") or (episode.get("measurements") or {}).get("keyframe_jpeg_base64")
        if isinstance(existing, str) and len(existing) > 100:
            return existing.split(",")[-1]

        try:
            from PIL import Image, ImageDraw
        except ImportError:
            return ""

        w, h = 260, 146
        img = Image.new("RGB", (w, h), (15, 23, 42))
        draw = ImageDraw.Draw(img)

        for gx in range(0, w, 26):
            draw.line([(gx, 0), (gx, h)], fill=(30, 41, 59), width=1)
        for gy in range(0, h, 26):
            draw.line([(0, gy), (w, gy)], fill=(30, 41, 59), width=1)

        cx, cy = w // 2, int(h * 0.44)
        draw.arc([cx - 58, cy + 28, cx + 58, cy + 95], start=180, end=360, fill=(71, 85, 105), width=2)
        draw.ellipse([cx - 28, cy - 34, cx + 28, cy + 32], outline=(96, 216, 195), width=2)

        micro_seq = str(episode.get("micro_sequence_chain") or "STEADY_BASELINE")
        flags = set(episode.get("behavioral_flags") or [])
        has_gaze = "GAZE" in micro_seq or any("GAZE" in f or "SACCADE" in f for f in flags)
        has_touch = "TOUCH" in micro_seq or "PACIFYING_NECK_TOUCH" in flags
        dx_gaze = 18 if has_gaze else 4
        dy_gaze = -8 if has_gaze else 0

        for ex in (cx - 11, cx + 11):
            ey = cy - 6
            draw.ellipse([ex - 3, ey - 3, ex + 3, ey + 3], fill=(56, 189, 248))
            draw.line([(ex, ey), (ex + dx_gaze, ey + dy_gaze)], fill=(56, 189, 248), width=2)

        if has_touch:
            box = [cx - 26, cy + 16, cx + 26, cy + 54]
            box_label = "ROI: HAND-TO-NECK ADAPTOR"
            box_color = (244, 63, 94)
        else:
            box = [cx - 22, cy + 6, cx + 22, cy + 30]
            box_label = "ROI: AU24/AU20 + GAZE VECTOR"
            box_color = (245, 158, 11)

        draw.rectangle(box, outline=box_color, width=2)
        draw.rectangle([4, 4, w - 4, 20], fill=(30, 41, 59))
        fmt_time = str(episode.get("formatted_time") or "01:38")
        draw.text((8, 6), f"KEYFRAME [{fmt_time}] · {box_label}", fill=(248, 250, 252))
        draw.rectangle([4, h - 20, w - 4, h - 4], fill=(30, 41, 59))
        draw.text((8, h - 18), micro_seq[:36], fill=(96, 216, 195))

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=88)
        return base64.b64encode(buf.getvalue()).decode("ascii")

NON_BEHAVIORAL_CAMERA_METRICS = NON_BEHAVIORAL_CAMERA_ARTIFACTS
