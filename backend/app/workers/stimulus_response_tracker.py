"""Implements OmniPulse v2.0 Upgrade 1: Dyadic Stimulus-Response Causal Coupling
(stimulus_response_tracker.py).
UPGRADE 1: Marks any window with `f0_semitone_mean > 1.0` OR `len(transcript_segment.strip()) > 0`
as `VERBAL_EXECUTION` so active speech is never mislabeled as `SILENCE`.
"""
from __future__ import annotations

from typing import Any


class DyadicStimulusTracker:
    HIGH_SALIENCE_LEXICON = frozenset({
        "deadline", "delay", "conflict", "budget", "failure", "mistake", "salary",
        "weakness", "disagree", "pressure", "risk", "gap", "termination", "overload",
        "personally", "blame", "outage", "incident", "escalation", "audit",
    })

    @staticmethod
    def _is_voiced(w: dict[str, Any]) -> bool:
        text = str(w.get("transcript_segment") or "").strip()
        if text:
            return True
        metrics = w.get("metrics") or {}
        f0 = metrics.get("f0_semitone_mean")
        return isinstance(f0, (int, float)) and not isinstance(f0, bool) and float(f0) > 1.0

    @classmethod
    def build_dyadic_turns(cls, diarized_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        turns: list[dict[str, Any]] = []
        turn_idx = 0
        i = 0
        while i < len(diarized_segments) - 1:
            curr_seg = diarized_segments[i]
            next_seg = diarized_segments[i + 1]
            curr_spk = str(curr_seg.get("speaker") or "").upper()
            next_spk = str(next_seg.get("speaker") or "").upper()

            if curr_spk in {"INTERVIEWER", "SPEAKER_00", "SPEAKER_A"} and next_spk in {"SUBJECT", "SPEAKER_01", "SPEAKER_B"}:
                c_start = float(curr_seg.get("start", 0.0))
                c_end = float(curr_seg.get("end", c_start))
                n_start = float(next_seg.get("start", c_end))
                n_end = float(next_seg.get("end", n_start))
                latency_ms = max(0.0, (n_start - c_end) * 1000.0)
                trigger_word, trigger_ts = cls._find_trigger_word(curr_seg)

                turns.append({
                    "turn_index": turn_idx,
                    "interviewer_start_sec": round(c_start, 3),
                    "interviewer_end_sec": round(c_end, 3),
                    "interviewer_transcript": str(curr_seg.get("text") or "").strip(),
                    "trigger_keyword": trigger_word,
                    "trigger_timestamp_sec": round(trigger_ts, 3),
                    "appraisal_latency_ms": round(latency_ms, 1),
                    "subject_start_sec": round(n_start, 3),
                    "subject_end_sec": round(n_end, 3),
                    "subject_transcript": str(next_seg.get("text") or "").strip(),
                    "post_utterance_leak_window": [round(n_end, 3), round(n_end + 1.5, 3)],
                })
                turn_idx += 1
                i += 2
            else:
                i += 1
        return turns

    @classmethod
    def infer_turns_from_windows(cls, windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not windows:
            return []
        diarized: list[dict[str, Any]] = []
        for idx, w in enumerate(windows):
            text = str(w.get("transcript_segment") or "").strip()
            if not text:
                continue
            role = str(w.get("speaker_role") or "").upper()
            if role not in {"INTERVIEWER", "SUBJECT"}:
                if "?" in text or text.lower().startswith(("did you", "can you", "why did", "how did", "what happened", "tell me")):
                    role = "INTERVIEWER"
                else:
                    role = "SUBJECT"
            seg_words = []
            for seg in w.get("transcript_segments") or []:
                seg_words.extend(seg.get("words") or [])
            diarized.append({
                "speaker": role,
                "start": float(w.get("start_sec", idx * 2.0)),
                "end": float(w.get("end_sec", (idx + 1) * 2.0)),
                "text": text,
                "words": seg_words,
                "window_index": w.get("window_index", idx),
            })

        turns = cls.build_dyadic_turns(diarized)
        if turns:
            return turns

        fallback_turns: list[dict[str, Any]] = []
        turn_idx = 0
        for idx, w in enumerate(windows):
            text = str(w.get("transcript_segment") or "").strip()
            if not text:
                continue
            w_start = float(w.get("start_sec", idx * 2.0))
            w_end = float(w.get("end_sec", (idx + 1) * 2.0))
            prev_w = windows[idx - 1] if idx > 0 else None
            prev_text = str((prev_w or {}).get("transcript_segment") or "").strip()
            latency_ms = float((w.get("metrics") or {}).get("response_latency_ms") or 420.0)
            trig_kw, trig_ts = cls._find_trigger_word({"text": prev_text or text, "end": max(0.0, w_start - latency_ms / 1000.0), "words": []})
            fallback_turns.append({
                "turn_index": turn_idx,
                "interviewer_start_sec": round(max(0.0, w_start - 2.0), 3),
                "interviewer_end_sec": round(max(0.0, w_start - latency_ms / 1000.0), 3),
                "interviewer_transcript": prev_text or "Preceding conversational prompt",
                "trigger_keyword": trig_kw,
                "trigger_timestamp_sec": round(trig_ts, 3),
                "appraisal_latency_ms": round(latency_ms, 1),
                "subject_start_sec": round(w_start, 3),
                "subject_end_sec": round(w_end, 3),
                "subject_transcript": text,
                "post_utterance_leak_window": [round(w_end, 3), round(w_end + 1.5, 3)],
            })
            turn_idx += 1
        return fallback_turns

    @classmethod
    def annotate_windows_with_turn_phases(
        cls,
        windows: list[dict[str, Any]],
        turns: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """UPGRADE 1: Any window where `_is_voiced(w)` is True (`f0_semitone_mean > 1.0` OR non-empty transcript)
        defaults to `VERBAL_EXECUTION` and `SUBJECT`, never `SILENCE`.
        """
        for w in windows:
            w_start = float(w.get("start_sec", 0.0))
            w_end = float(w.get("end_sec", 0.0))
            w_mid = 0.5 * (w_start + w_end)
            voiced = cls._is_voiced(w)
            assigned_phase = "VERBAL_EXECUTION" if voiced else "SILENCE"
            assigned_role = w.get("speaker_role") or ("SUBJECT" if voiced else "SILENCE")
            assigned_turn = None

            for t in turns:
                t_idx = t["turn_index"]
                if t["subject_start_sec"] <= w_mid <= t["subject_end_sec"]:
                    assigned_phase = "VERBAL_EXECUTION"
                    assigned_role = "SUBJECT"
                    assigned_turn = t_idx
                    break
                leak_start, leak_end = t.get("post_utterance_leak_window", [t["subject_end_sec"], t["subject_end_sec"] + 1.5])
                if leak_start < w_mid <= leak_end and not voiced:
                    assigned_phase = "POST_UTTERANCE_LEAK"
                    assigned_role = "SILENCE"
                    assigned_turn = t_idx
                    break
                if t["interviewer_start_sec"] <= w_mid <= t["interviewer_end_sec"] and not voiced:
                    assigned_phase = "STIMULUS_LISTENING"
                    assigned_role = "INTERVIEWER"
                    assigned_turn = t_idx
                    break
                if t["interviewer_end_sec"] < w_mid < t["subject_start_sec"] and not voiced:
                    assigned_phase = "APPRAISAL_LATENCY"
                    assigned_role = "SILENCE"
                    assigned_turn = t_idx
                    break

            w["speaker_role"] = assigned_role
            w["turn_id"] = assigned_turn
            w["turn_phase"] = assigned_phase
        return windows

    @classmethod
    def _find_trigger_word(cls, interviewer_segment: dict[str, Any]) -> tuple[str, float]:
        words = interviewer_segment.get("words") or []
        for w in words:
            clean_w = str(w.get("word") or "").lower().strip(".,?!;:\"'()")
            if clean_w in cls.HIGH_SALIENCE_LEXICON:
                return clean_w, float(w.get("start", interviewer_segment.get("end", 0.0)))
        text = str(interviewer_segment.get("text") or "")
        for token in text.split():
            clean_w = token.lower().strip(".,?!;:\"'()")
            if clean_w in cls.HIGH_SALIENCE_LEXICON:
                return clean_w, float(interviewer_segment.get("end", 0.0))
        return "TURN_COMPLETION", float(interviewer_segment.get("end", 0.0))
