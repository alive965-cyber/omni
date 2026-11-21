"""
backend/app/services/human_signal_translator.py
Translates raw sensor Z-scores and metrics into plain human behavioral descriptions:
 1. Spoken Words (`what_was_spoken`)
 2. Voice Quality & Vocal Emotion (`voice_tone_and_emotion`)
 3. Physical Actions (`physical_actions_observed`)
 4. Heartbeat & Breathing State (`heartbeat_and_breathing`)
 5. Human-Readable Headline State Label (`headline_state_label`)
"""
from __future__ import annotations

import re
from typing import Dict, Any, List


def _metric(win: Dict[str, Any], key: str, default: Any = 0.0) -> Any:
    if key in win and win[key] is not None:
        return win[key]
    for sub in ("metrics", "measurements"):
        sub_dict = win.get(sub)
        if isinstance(sub_dict, dict) and key in sub_dict and sub_dict[key] is not None:
            return sub_dict[key]
    return default


def _zval(z: Dict[str, Any], key: str) -> float:
    for candidate in (key, f"{key}_z_floor", key.replace("_z_floor", "")):
        val = z.get(candidate)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
    return 0.0


class HumanSignalTranslator:
    @staticmethod
    def describe_voice_and_emotion(win: Dict[str, Any], z: Dict[str, float]) -> Dict[str, str]:
        f0 = float(_metric(win, "f0_semitone_mean", 0.0) or 0.0)
        wpm = float(_metric(win, "speech_rate_wpm", 0.0) or 0.0)
        transcript = (
            win.get("transcript_segment")
            or win.get("subject_response_quote")
            or win.get("transcript_excerpt")
            or ""
        ).strip()
        if transcript.lower().startswith("silent evaluation"):
            transcript = ""

        f0_z = _zval(z, "f0_semitone_mean_z_floor")
        f0_std_z = _zval(z, "f0_semitone_std_z_floor")
        jitter_z = _zval(z, "jitter_local_mean_z_floor")
        shimmer_z = _zval(z, "shimmer_local_db_mean_z_floor")
        hnr_z = _zval(z, "hnr_db_mean_z_floor")
        loud_z = _zval(z, "loudness_mean_z_floor")
        pause_ratio = float(_metric(win, "pause_ratio", 0.0) or 0.0)

        if f0 <= 1.0 and not transcript:
            return {
                "voice_tone_summary": "Silent / Non-verbal listening or thinking pause (no active speech).",
                "detected_vocal_emotion": "Silent Appraisal / Thinking Pause",
            }

        traits = []
        emotion_labels = []

        # Pitch & Tone
        if f0_z >= 1.5:
            traits.append(f"vocal pitch rose noticeably above their resting baseline ({f0:.1f} semitones)")
            emotion_labels.append("Heightened Arousal / Urgency")
        elif f0_z <= -1.5:
            traits.append(f"vocal pitch dropped lower than their normal tone ({f0:.1f} semitones)")
            emotion_labels.append("Serious / Guarded Tone")

        # Expressiveness vs Monotone
        if f0_std_z <= -1.3:
            traits.append("flat, monotone pitch delivery (carefully controlled expression)")
            emotion_labels.append("Emotionally Guarded / Controlled")
        elif f0_std_z >= 1.5:
            traits.append("animated, expressive pitch inflection")
            emotion_labels.append("Animated / Expressive")

        # Micro-tremor / Vocal Stress (Jitter & Shimmer)
        if jitter_z >= 1.6 or shimmer_z >= 1.6:
            traits.append("subtle tremor and wavering heard in the voice")
            emotion_labels.append("Vocal Nervousness / Internal Strain")
        elif hnr_z <= -1.5:
            traits.append("tense, slightly breathy vocal quality")
            emotion_labels.append("Vocal Tension")

        # Pace & Pauses
        if wpm > 155:
            traits.append(f"fast, eager speaking pace ({wpm:.0f} WPM)")
            emotion_labels.append("Rushed / Eager")
        elif 0 < wpm < 110 or pause_ratio > 0.35:
            pace_label = f"{wpm:.0f} WPM" if wpm > 0 else "measured cadence"
            traits.append(f"slower, hesitant pacing with thinking pauses ({pace_label})")
            emotion_labels.append("Hesitant / Careful Word Selection")

        if loud_z >= 1.5:
            traits.append("louder, assertive voice volume")
        elif loud_z <= -1.5:
            traits.append("softer, lowered voice volume")

        if not traits:
            pace_str = f"{wpm:.0f} WPM" if wpm > 0 else f"pitch {f0:.1f} semitones"
            traits.append(f"steady, natural speaking voice ({pace_str}) matching personal baseline")
            emotion_labels.append("Calm & Assured")

        if not emotion_labels:
            emotion_labels.append("Focused & Measured Delivery")

        return {
            "voice_tone_summary": "; ".join(traits).capitalize() + ".",
            "detected_vocal_emotion": " + ".join(dict.fromkeys(emotion_labels)),
        }

    @staticmethod
    def describe_physical_actions(win: Dict[str, Any], z: Dict[str, float]) -> str:
        actions = []
        au01_z = _zval(z, "au01_inner_brow_z_floor")
        au02_z = _zval(z, "au02_outer_brow_z_floor")
        au04_z = _zval(z, "au04_brow_lowerer_z_floor")
        au12_z = _zval(z, "au12_lip_corner_puller_z_floor")
        au06_val = float(_metric(win, "au06_cheek_raiser", 0.0) or 0.0)
        au20_z = _zval(z, "au20_lip_stretcher_z_floor")
        au24_z = _zval(z, "au24_lip_presser_z_floor")
        touch_z = _zval(z, "self_touch_ratio_z_floor")
        hand_z = _zval(z, "hand_kinetic_energy_z_floor")
        blink_z = _zval(z, "blink_rate_per_min_z_floor")
        gaze_z = _zval(z, "gaze_aversion_ratio_z_floor")
        lean_z = _zval(z, "postural_lean_z_z_floor")

        if touch_z >= 1.8 or "SELF_TOUCH" in str(win.get("micro_sequence_chain", "")):
            actions.append("brought their hand to their neck/face in a self-soothing gesture")
        if au24_z >= 1.8:
            actions.append("pressed their lips tightly shut (holding back words)")
        if au20_z >= 1.8:
            actions.append("pulled their lips sideways in tension")
        if au04_z >= 1.6:
            actions.append("knitted and furrowed their eyebrows in deep concentration")
        if au01_z >= 1.8 or au02_z >= 1.8:
            actions.append("raised both eyebrows noticeably in emphasis")
        if au12_z >= 1.8:
            if au06_val >= 0.20:
                actions.append("showed a genuine, warm smile reaching the eyes")
            else:
                actions.append("put on a polite, surface-level social smile")
        if gaze_z >= 1.6 or "GAZE_AVERT" in str(win.get("micro_sequence_chain", "")) or "GAZE_AVERSION" in str(win.get("micro_sequence_chain", "")):
            actions.append("broke eye contact with the camera while thinking through their words")
        if blink_z >= 1.8:
            actions.append("blinked rapidly in a brief flurry while processing the thought")
        if hand_z >= 2.0 and touch_z < 1.8:
            actions.append("used expressive hand gestures while explaining")
        if lean_z >= 1.8:
            actions.append("leaned forward toward the screen with engaged focus")
        elif lean_z <= -1.8:
            actions.append("shifted posture slightly backward away from the screen")

        if not actions:
            return "Maintained steady eye contact with the camera, relaxed facial muscles, and calm upright posture."
        return "Speaker " + ", ".join(actions) + "."

    @staticmethod
    def describe_heartbeat_and_breathing(
        win: Dict[str, Any],
        z: Dict[str, float],
        baseline_hr: float | None,
        resp_rate: float | None,
        sigh_times: List[float],
        apnea_windows: List[List[float]],
    ) -> str:
        tr = win.get("time_range_sec") or [win.get("start_sec", 0.0), win.get("end_sec", 0.0)]
        start_s = float(win.get("start_sec") if win.get("start_sec") is not None else tr[0])
        end_s = float(win.get("end_sec") if win.get("end_sec") is not None else tr[-1])
        hr = _metric(win, "hr_bpm_pos", None) or _metric(win, "hr_bpm", None)

        parts = []
        if hr is not None and isinstance(hr, (int, float)) and float(hr) > 40:
            hr_val = float(hr)
            if baseline_hr and abs(hr_val - baseline_hr) >= 5.0:
                direction = "rose to" if hr_val > baseline_hr else "settled to"
                parts.append(
                    f"Heart rate {direction} {hr_val:.0f} BPM (vs. {baseline_hr:.0f} BPM personal resting baseline)"
                )
            else:
                parts.append(f"Heart rate remained steady at {hr_val:.0f} BPM (matching personal resting baseline)")
        else:
            parts.append("Optical heart-rate lock paused briefly during natural head/hand movement")

        had_sigh = [s for s in (sigh_times or []) if start_s - 1.5 <= float(s) <= end_s + 1.5]
        had_apnea = [
            a for a in (apnea_windows or []) if len(a) >= 2 and float(a[0]) <= end_s and float(a[1]) >= start_s
        ]

        if had_sigh:
            parts.append(f"released a deep relief sigh at {float(had_sigh[0]):.1f}s")
        if had_apnea:
            parts.append(
                f"briefly held their breath between {float(had_apnea[0][0]):.1f}s and {float(had_apnea[0][1]):.1f}s while concentrating"
            )
        elif resp_rate is not None:
            parts.append(f"upper-torso breathing remained steady at {float(resp_rate):.1f} breaths/min")

        return "; ".join(parts) + "."

    @staticmethod
    def translate_micro_sequence_plain(micro_seq: str | None) -> str:
        """Translates raw 100ms tokens into plain English chronological descriptions."""
        if not micro_seq or micro_seq in ("STEADY_BASELINE", "NO_MICRO_EVENTS"):
            return "Steady facial posture without abrupt sub-second micro-expressions."

        token_labels = {
            "NECK_OR_FACE_SELF_TOUCH": "brought hand to neck/face",
            "SELF_TOUCH_ONSET": "brought hand to neck/face",
            "GAZE_AVERT_SHIFT": "looked away from the camera",
            "GAZE_AVERSION": "looked away from the camera",
            "AU04_BROW_FURROW": "furrowed eyebrows in concentration",
            "AU04_BROW_LOWER": "furrowed eyebrows in concentration",
            "AU24_LIP_PRESS": "pressed lips tightly shut",
            "AU20_LIP_STRETCH": "pulled lips sideways in tension",
            "AU15_LIP_CORNER_DROP": "lowered lip corners",
            "AU12_DUCHENNE_SMILE": "warm genuine smile reached the eyes",
            "AU12_SMILE_ONSET": "started smiling",
            "AU06_CHEEK_RAISE": "cheeks lifted into an eye-crinkling smile",
            "AU12_NON_DUCHENNE_SMILE": "put on a polite social smile",
            "SNAP_OFF_SMILE_DROP": "abruptly dropped the smile",
        }
        matches = re.findall(r"\[?([A-Z0-9_]+)\s*@\s*\+?(\d+ms)\]?", str(micro_seq))
        if not matches:
            return "Steady facial posture without abrupt sub-second micro-expressions."

        steps = []
        filtered = [(tok, ms) for tok, ms in matches if tok != "STIMULUS_END"]
        for idx, (tok, ms) in enumerate(filtered):
            human_act = token_labels.get(tok, tok.lower().replace("_", " "))
            prefix = "First" if idx == 0 else ("then" if idx < len(filtered) - 1 else "and finally")
            steps.append(f"{prefix} {human_act} at +{ms}")

        if not steps:
            return "Steady facial posture without abrupt sub-second micro-expressions."
        return "Chronological reaction: " + ", ".join(steps) + "."

    @staticmethod
    def infer_headline_state_label(
        what_was_spoken: str,
        detected_vocal_emotion: str,
        phys_desc: str,
    ) -> str:
        low_s = what_was_spoken.lower()
        low_p = phys_desc.lower()
        low_v = detected_vocal_emotion.lower()
        if "genuine, warm smile" in low_p or "animated" in low_v:
            return "Genuine Visual Satisfaction & Animated Engagement"
        if "neck/face" in low_p and ("pressed their lips" in low_p or "not sure" in low_s):
            return "Candid Uncertainty & Self-Soothing Restraint"
        if "pressed their lips" in low_p and "furrowed" in low_p:
            return "Intense Technical Concentration & Guarded Phrasing"
        if "neck/face" in low_p:
            return "Active Explanation with Brief Self-Soothing Touch"
        if "silent" in low_v:
            return "Silent Post-Statement Appraisal & Reflection"
        return "Calm & Assured Conversational Flow"

    @classmethod
    def build_human_moment_breakdown(
        cls,
        episode: Dict[str, Any],
        baseline_hr: float | None = None,
        resp_summary: Dict[str, Any] | None = None,
    ) -> Dict[str, str]:
        """Builds the 5 human-friendly behavioral rows + headline_state_label for a Moment Card."""
        z_map = episode.get("dual_horizon_z_scores") or episode.get("baseline_z_scores") or {}
        resp = resp_summary or {}
        resp_rate = resp.get("respiration_rate_breaths_per_min") or resp.get("respiration_rate_bpm")
        sigh_times = resp.get("detected_sigh_timestamps_sec") or resp.get("sigh_timestamps_sec") or []
        apnea_windows = resp.get("apnea_freeze_windows") or []

        raw_quote = str(
            episode.get("subject_response_quote")
            or episode.get("transcript_excerpt")
            or episode.get("transcript_segment")
            or ""
        ).strip()
        if raw_quote and not raw_quote.lower().startswith("silent evaluation"):
            what_was_spoken = f"“{raw_quote}”"
        else:
            f0_val = float(_metric(episode, "f0_semitone_mean", 0.0) or 0.0)
            if f0_val > 1.0:
                what_was_spoken = (
                    f"Active spoken segment (voiced pitch {f0_val:.1f} semitones; unscripted conversational turn)"
                )
            else:
                what_was_spoken = "Silent listening / post-statement reflection pause between spoken sentences."

        voice_info = cls.describe_voice_and_emotion(episode, z_map)
        phys_desc = cls.describe_physical_actions(episode, z_map)
        micro_plain = cls.translate_micro_sequence_plain(episode.get("micro_sequence_chain"))
        full_physical = f"{phys_desc} ({micro_plain})" if "Chronological reaction:" in micro_plain else phys_desc
        hb_breath_desc = cls.describe_heartbeat_and_breathing(
            episode, z_map, baseline_hr, resp_rate, sigh_times, apnea_windows
        )
        headline_label = cls.infer_headline_state_label(
            what_was_spoken, voice_info["detected_vocal_emotion"], phys_desc
        )

        return {
            "headline_state_label": headline_label,
            "what_was_spoken": what_was_spoken,
            "voice_tone_summary": voice_info["voice_tone_summary"],
            "detected_vocal_emotion": voice_info["detected_vocal_emotion"],
            "voice_tone_and_emotion": f"{voice_info['detected_vocal_emotion']} — {voice_info['voice_tone_summary']}",
            "voice_and_vocal_emotion": f"{voice_info['detected_vocal_emotion']} — {voice_info['voice_tone_summary']}",
            "physical_actions_observed": full_physical,
            "physical_actions_and_expression": full_physical,
            "heartbeat_and_breathing": hb_breath_desc,
        }
