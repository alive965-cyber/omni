"""
backend/app/services/topic_shift_analyzer.py
Segments the recording into 4 contiguous Conversational Topics/Phases and computes
a Topic-by-Topic Behavioral Shift Matrix across Words, Voice Emotion, Heartbeat, Breathing, and Body Language.
"""
from __future__ import annotations

from typing import List, Dict, Any
import numpy as np
import pandas as pd


class TopicBehavioralShiftAnalyzer:
    @staticmethod
    def build_topic_shift_matrix(
        df_windows: pd.DataFrame,
        baseline_hr: float | None,
        sigh_timestamps: List[float],
        apnea_windows: List[List[float]],
        num_segments: int = 4,
    ) -> List[Dict[str, Any]]:
        if df_windows is None or df_windows.empty:
            return []

        df = df_windows.sort_values("start_sec").reset_index(drop=True)
        chunks = np.array_split(df, min(num_segments, len(df)))
        topic_rows: List[Dict[str, Any]] = []

        for idx, chunk in enumerate(chunks, 1):
            if chunk.empty:
                continue
            t_start = float(chunk["start_sec"].min())
            t_end = float(chunk["end_sec"].max())
            time_label = (
                f"{int(t_start // 60):02d}:{int(t_start % 60):02d} – "
                f"{int(t_end // 60):02d}:{int(t_end % 60):02d}"
            )

            # 1. Extract Spoken Topic Excerpt
            texts = [
                str(t).strip()
                for t in (chunk["transcript_segment"] if "transcript_segment" in chunk.columns else [])
                if t is not None and str(t).strip() and str(t).strip().lower() not in ("nan", "none")
            ]
            combined_text = " ".join(texts).strip()
            if combined_text:
                topic_excerpt = f'"{combined_text[:85]}..."' if len(combined_text) > 85 else f'"{combined_text}"'
            else:
                topic_excerpt = "Silent Listening / Non-Verbal Appraisal Phase"

            # 2. Voice Emotion & Pace
            emotions = [
                str(e)
                for e in (chunk["dominant_vocal_emotion"] if "dominant_vocal_emotion" in chunk.columns else [])
                if e is not None and str(e) not in ("", "nan", "None", "Silent / Listening Pause")
            ]
            dom_emotion = max(set(emotions), key=emotions.count) if emotions else "Silent / Controlled Pause"
            if "speech_rate_wpm" in chunk.columns:
                wpm_series = pd.to_numeric(chunk["speech_rate_wpm"], errors="coerce").fillna(0.0)
                voiced_wpm = wpm_series[wpm_series > 0]
                mean_wpm = f"{voiced_wpm.mean():.0f} WPM" if not voiced_wpm.empty else "Silent"
            else:
                mean_wpm = "Silent"

            # 3. Heartbeat & Breathing in this Topic
            hr_col = "hr_bpm_pos" if "hr_bpm_pos" in chunk.columns else "hr_bpm"
            valid_hr = pd.to_numeric(chunk[hr_col], errors="coerce").dropna() if hr_col in chunk.columns else pd.Series(dtype=float)
            valid_hr = valid_hr[valid_hr > 40.0]
            if not valid_hr.empty:
                mean_hr = float(valid_hr.mean())
                delta_str = f" ({mean_hr - baseline_hr:+.0f} vs base)" if baseline_hr else ""
                hr_summary = f"{mean_hr:.0f} BPM{delta_str}"
            else:
                hr_summary = "Motion Masked"

            topic_sighs = [s for s in (sigh_timestamps or []) if t_start <= float(s) <= t_end]
            topic_apneas = [
                a for a in (apnea_windows or []) if len(a) >= 2 and float(a[0]) <= t_end and float(a[1]) >= t_start
            ]
            breath_notes = []
            if topic_sighs:
                breath_notes.append(f"{len(topic_sighs)} Deep Sigh(s)")
            if topic_apneas:
                breath_notes.append(f"{len(topic_apneas)} Breath-Hold(s)")
            if not breath_notes:
                breath_notes.append("Steady Breathing")

            # 4. Physical Actions Observed
            actions = []
            st_max = float(pd.to_numeric(chunk["self_touch_ratio"], errors="coerce").max()) if "self_touch_ratio" in chunk.columns else 0.0
            au24_max = float(pd.to_numeric(chunk["au24_lip_presser_z_floor"], errors="coerce").max()) if "au24_lip_presser_z_floor" in chunk.columns else 0.0
            au04_max = float(pd.to_numeric(chunk["au04_brow_lowerer_z_floor"], errors="coerce").max()) if "au04_brow_lowerer_z_floor" in chunk.columns else 0.0
            au01_max = float(pd.to_numeric(chunk["au01_inner_brow_z_floor"], errors="coerce").max()) if "au01_inner_brow_z_floor" in chunk.columns else 0.0
            au12_max = float(pd.to_numeric(chunk["au12_lip_corner_puller_z_floor"], errors="coerce").max()) if "au12_lip_corner_puller_z_floor" in chunk.columns else 0.0
            blink_max = float(pd.to_numeric(chunk["blink_rate_per_min_z_floor"], errors="coerce").max()) if "blink_rate_per_min_z_floor" in chunk.columns else 0.0

            if st_max >= 0.22:
                actions.append("Neck/Face Self-Touch")
            if au24_max >= 1.8:
                actions.append("Tight Lip Press")
            if au04_max >= 1.6:
                actions.append("Brow Furrow Focus")
            if au01_max >= 1.8:
                actions.append("Raised Eyebrows")
            if au12_max >= 1.8:
                actions.append("Warm Genuine Smile")
            if blink_max >= 1.8:
                actions.append("Rapid Blink Burst")
            if not actions:
                actions.append("Calm Baseline Posture")

            # 5. Overall State Verdict for this Topic
            mean_vtli = float(pd.to_numeric(chunk["vtli_score"], errors="coerce").mean()) if "vtli_score" in chunk.columns else 50.0
            if "Neck/Face Self-Touch" in actions or "Tight Lip Press" in actions or topic_sighs:
                state_verdict = "Elevated Tension & Self-Regulation"
            elif "Warm Genuine Smile" in actions and dom_emotion == "Animated & Expressive":
                state_verdict = "Positive Relief & High Engagement"
            elif mean_vtli >= 62.0 or topic_apneas or "Brow Furrow Focus" in actions:
                state_verdict = "High Cognitive Load / Active Focus"
            else:
                state_verdict = "Comfortable & Baseline Congruent"

            topic_rows.append({
                "phase_index": f"Phase {idx} ({time_label})",
                "spoken_topic_excerpt": topic_excerpt,
                "voice_and_emotion": f"{dom_emotion} ({mean_wpm})",
                "heartbeat_and_breathing": f"{hr_summary} | {', '.join(breath_notes)}",
                "key_physical_actions": ", ".join(actions),
                "topic_state_verdict": state_verdict,
            })

        return topic_rows

    @classmethod
    def build_from_window_dicts(
        cls,
        windows: List[Dict[str, Any]],
        baseline_hr: float | None,
        sigh_timestamps: List[float],
        apnea_windows: List[List[float]],
        num_segments: int = 4,
    ) -> List[Dict[str, Any]]:
        """Helper that flattens window dicts (metrics, dual_horizon_z_scores, composite_indices)
        into a DataFrame and runs `build_topic_shift_matrix`.
        """
        if not windows:
            return []
        flat_rows = []
        for w in windows:
            metrics = w.get("metrics") or {}
            z_map = w.get("dual_horizon_z_scores") or w.get("baseline_z_scores") or {}
            ci = w.get("composite_indices") or {}
            flat_rows.append({
                "start_sec": float(w.get("start_sec", 0.0)),
                "end_sec": float(w.get("end_sec", 0.0)),
                "transcript_segment": w.get("transcript_segment") or "",
                "dominant_vocal_emotion": w.get("dominant_vocal_emotion") or "",
                "speech_rate_wpm": metrics.get("speech_rate_wpm", w.get("speech_rate_wpm", 0.0)),
                "hr_bpm_pos": metrics.get("hr_bpm_pos", metrics.get("hr_bpm", w.get("hr_bpm_pos"))),
                "hr_bpm": metrics.get("hr_bpm", w.get("hr_bpm")),
                "self_touch_ratio": metrics.get("self_touch_ratio", w.get("self_touch_ratio", 0.0)),
                "au24_lip_presser_z_floor": z_map.get("au24_lip_presser_z_floor", z_map.get("au24_lip_presser_z", 0.0)),
                "au04_brow_lowerer_z_floor": z_map.get("au04_brow_lowerer_z_floor", z_map.get("au04_brow_lowerer_z", 0.0)),
                "au01_inner_brow_z_floor": z_map.get("au01_inner_brow_z_floor", z_map.get("au01_inner_brow_z", 0.0)),
                "au12_lip_corner_puller_z_floor": z_map.get("au12_lip_corner_puller_z_floor", z_map.get("au12_lip_corner_puller_z", 0.0)),
                "blink_rate_per_min_z_floor": z_map.get("blink_rate_per_min_z_floor", z_map.get("blink_rate_per_min_z", 0.0)),
                "vtli_score": w.get("vtli_score", ci.get("visual_tension_load_index", 50.0)),
            })
        df = pd.DataFrame(flat_rows)
        return cls.build_topic_shift_matrix(df, baseline_hr, sigh_timestamps, apnea_windows, num_segments=num_segments)
