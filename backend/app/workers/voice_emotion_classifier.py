"""
backend/app/workers/voice_emotion_classifier.py
Fuses Deep Neural Speech Emotion Recognition (wav2vec2/HuBERT) with openSMILE
baseline-calibrated vocal biomarkers to output exact Vocal Emotion probabilities
and human-readable voice descriptions.
"""
from __future__ import annotations

import os
from typing import Dict, Any
import numpy as np

try:
    import soundfile as sf
except Exception:
    sf = None

_SER_PIPELINE = None
if os.getenv("ENABLE_HUBERT_SER", "").lower() in ("1", "true", "yes"):
    try:
        from transformers import pipeline

        _SER_PIPELINE = pipeline(
            "audio-classification",
            model="superb/hubert-large-superb-er",
            top_k=5,
        )
    except Exception:
        _SER_PIPELINE = None


def _get_z(z_scores: Dict[str, Any], key: str) -> float:
    for cand in (key, f"{key}_z_floor", f"{key}_z"):
        val = z_scores.get(cand)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
    return 0.0


class HybridVoiceEmotionClassifier:
    EMOTION_CATEGORIES = [
        "Calm & Assured",
        "Hesitant & Unsure",
        "Nervous / Vocal Strain",
        "Guarded / Monotone",
        "Animated & Expressive",
    ]

    @classmethod
    def analyze_window_voice_emotion(
        cls,
        wav_path: str | None,
        start_sec: float,
        end_sec: float,
        win_metrics: Dict[str, Any],
        z_scores: Dict[str, float],
    ) -> Dict[str, Any]:
        f0_mean = float(win_metrics.get("f0_semitone_mean") or 0.0)
        transcript = str(win_metrics.get("transcript_segment") or "").strip()

        # Check if window is genuinely unvoiced/silent
        if f0_mean <= 1.0 and not transcript:
            return {
                "is_voiced": False,
                "dominant_vocal_emotion": "Silent / Listening Pause",
                "vocal_emotion_confidence": 1.0,
                "emotion_probabilities": {},
                "human_voice_summary": "No active speech in this window (silent listening or thinking pause).",
            }

        # 1. Compute openSMILE Biomarker Logits
        z_f0 = _get_z(z_scores, "f0_semitone_mean")
        z_f0_std = _get_z(z_scores, "f0_semitone_std")
        z_jit = _get_z(z_scores, "jitter_local_mean")
        z_shim = _get_z(z_scores, "shimmer_local_db_mean")
        z_hnr = _get_z(z_scores, "hnr_db_mean")
        z_loud = _get_z(z_scores, "loudness_mean")
        z_wpm = _get_z(z_scores, "speech_rate_wpm")
        pause_ratio = float(win_metrics.get("pause_ratio") or 0.0)
        disfluencies = int(win_metrics.get("disfluency_count") or 0)

        jit_ex = max(0.0, z_jit - 0.9)
        shim_ex = max(0.0, z_shim - 0.9)
        f0_ex = max(0.0, z_f0 - 0.9)
        hnr_strain = -0.4 * min(0.0, z_hnr + 0.9) if (jit_ex > 0 or f0_ex > 0) else 0.0
        bio_logits = np.array(
            [
                1.45 - 0.5 * jit_ex - 0.4 * f0_ex + 0.2 * max(-0.5, z_hnr),  # Calm & Assured
                1.8 * max(0.0, pause_ratio - 0.20) + 0.6 * disfluencies - 0.3 * z_wpm + 0.3 * jit_ex,  # Hesitant & Unsure
                0.7 * jit_ex + 0.5 * shim_ex + 0.5 * f0_ex + hnr_strain,  # Nervous / Strain
                -0.8 * z_f0_std - 0.4 * z_loud + 0.8 * pause_ratio,  # Guarded / Monotone
                0.7 * max(0.0, z_f0_std - 0.8) + 0.5 * max(0.0, z_loud - 0.5) + 0.4 * max(0.0, z_wpm - 0.5) - 0.3 * jit_ex,  # Animated & Expressive
            ],
            dtype=np.float64,
        )

        # 2. Optional Neural SER Stream (HuBERT / wav2vec2)
        neural_probs = np.ones(5, dtype=np.float64) / 5.0
        if _SER_PIPELINE is not None and wav_path and sf is not None and os.path.exists(wav_path):
            try:
                info = sf.info(wav_path)
                sr = info.samplerate
                start_frame = int(start_sec * sr)
                frames_to_read = int((end_sec - start_sec) * sr)
                audio_slice, _ = sf.read(wav_path, start=start_frame, frames=frames_to_read, dtype="float32")
                if len(audio_slice) > sr * 0.5:
                    preds = _SER_PIPELINE({"raw": audio_slice, "sampling_rate": sr})
                    label_map = {p["label"].lower(): float(p["score"]) for p in preds}
                    neural_probs = np.array(
                        [
                            label_map.get("neu", 0.25),
                            label_map.get("sad", 0.15) + 0.5 * label_map.get("fear", 0.1),
                            label_map.get("ang", 0.15) + label_map.get("fear", 0.15),
                            label_map.get("sad", 0.20) + 0.5 * label_map.get("neu", 0.1),
                            label_map.get("hap", 0.25),
                        ],
                        dtype=np.float64,
                    )
                    neural_probs = neural_probs / (np.sum(neural_probs) + 1e-8)
            except Exception:
                pass

        # 3. Softmax Fusion
        combined_logits = 0.55 * bio_logits + 0.45 * np.log(neural_probs + 1e-4)
        exp_l = np.exp(combined_logits - np.max(combined_logits))
        probs = exp_l / np.sum(exp_l)

        prob_dict = {cat: round(float(p), 3) for cat, p in zip(cls.EMOTION_CATEGORIES, probs)}
        top_idx = int(np.argmax(probs))
        dominant_emotion = cls.EMOTION_CATEGORIES[top_idx]
        confidence = round(float(probs[top_idx]), 2)

        # 4. Build Human-Readable Voice Tone Description
        wpm = float(win_metrics.get("speech_rate_wpm") or 0.0)
        details = []
        if z_f0 >= 1.5:
            details.append(f"pitch rose noticeably higher than normal ({f0_mean:.1f} semitones)")
        elif z_f0 <= -1.5:
            details.append(f"pitch dropped lower than baseline ({f0_mean:.1f} semitones)")
        else:
            details.append("natural pitch level")

        if z_jit >= 1.6 or z_shim >= 1.6:
            details.append("audible vocal micro-tremor/wavering")
        elif z_hnr <= -1.5:
            details.append("slightly tense/pressed vocal tone")
        else:
            details.append("steady vocal cord clarity")

        if pause_ratio >= 0.35 or disfluencies >= 1:
            details.append(f"hesitant pacing with {int(pause_ratio * 100)}% pause time ({wpm:.0f} words/min)" if wpm > 0 else f"hesitant pacing with {int(pause_ratio * 100)}% pause time")
        elif wpm > 0:
            details.append(f"speaking pace of {wpm:.0f} words/min")

        human_summary = (
            f"Voice Emotion: {dominant_emotion} ({int(confidence * 100)}% match) — "
            + ", ".join(details)
            + "."
        )

        return {
            "is_voiced": True,
            "dominant_vocal_emotion": dominant_emotion,
            "vocal_emotion_confidence": confidence,
            "emotion_probabilities": prob_dict,
            "human_voice_summary": human_summary,
        }
