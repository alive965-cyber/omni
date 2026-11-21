"""In-memory and file-based openSMILE eGeMAPSv02 acoustic extraction (with loudnorm normalization & DSP fallback),
Groq Whisper Large-v3 verbatim transcription + local faster-whisper fallback (locked to language="en"
with explicit console logging and zero silent error swallowing),
LIWC pronoun/hedge psycholinguistics, and semantic valence scoring.
UPGRADE 1:
 - Normalizes audio volume with FFmpeg `loudnorm=I=-16:TP=-1.5:LRA=11` at 16kHz mono before openSMILE and Whisper.
 - Removes the `loudness_mean < 0.08` unvoiced cutoff! A window is voiced whenever `f0_semitone_mean > 1.0` OR `len(transcript_segment.strip()) > 0`.
"""
from __future__ import annotations

import importlib.util
import io
import logging
from math import isfinite, log10, log2
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
from typing import Any
import wave

from app.workers.audio_extractor import is_window_voiced, normalize_audio_file_ffmpeg, normalize_pcm_samples

logger = logging.getLogger("omnipulse.audio_inference")

_client_lock = threading.Lock()
_groq_client = None
_faster_whisper_model = None
_smile = None

_EGEMAPS = {
    "F0semitoneFrom27.5Hz_sma3nz_amean": "f0_semitone_mean",
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm": "f0_semitone_std",
    "jitterLocal_sma3nz_amean": "jitter_local_mean",
    "shimmerLocaldB_sma3nz_amean": "shimmer_local_db_mean",
    "HNRdBACF_sma3nz_amean": "hnr_db_mean",
    "loudness_sma3_amean": "loudness_mean",
    "loudnessPeaksPerSec": "loudness_peaks_per_sec",
    "spectralFlux_sma3_amean": "spectral_flux_mean",
}

FILLER_REGEX = re.compile(
    r"\b(um+|uh+|er+|ah+|hmm+|like|you\s+know|i\s+mean|sort\s+of|basically|haan|matlab|accha)\b",
    re.IGNORECASE,
)
FIRST_PERSON_SINGULAR_REGEX = re.compile(r"\b(i|me|my|mine|myself)\b", re.IGNORECASE)
FIRST_PERSON_PLURAL_DISTANCING_REGEX = re.compile(r"\b(we|us|our|ours|the\s+team|one)\b", re.IGNORECASE)
EPISTEMIC_HEDGE_REGEX = re.compile(
    r"\b(sort\s+of|kind\s+of|technically|to\s+be\s+honest|honestly|basically|essentially|probably|maybe|i\s+guess)\b",
    re.IGNORECASE,
)

POSITIVE_VALENCE_WORDS = frozenset({
    "comfortable", "confident", "ready", "smooth", "great", "good", "fine", "resolved",
    "completed", "on-track", "easy", "clear", "happy", "agree", "certain", "strong",
    "love", "excellent", "success", "stable", "sure", "definitely", "achieved", "nice",
})
NEGATIVE_VALENCE_WORDS = frozenset({
    "difficult", "hard", "delay", "delayed", "problem", "issue", "conflict", "stress",
    "worried", "unsure", "failed", "failure", "risk", "concerned", "impossible", "late",
    "overloaded", "struggling", "disagree", "afraid", "nervous", "uncertain", "blocked",
})
NEGATION_WORDS = frozenset({"not", "never", "no", "neither", "hardly", "barely", "cannot", "cant", "won't", "don't", "didn't", "isn't", "wasn't"})

WHISPER_HALLUCINATION_PHRASES = frozenset({
    "mas... não?",
    "mas... nao?",
    "mas não?",
    "obrigado.",
    "obrigado",
    "subtitles by",
    "thank you for watching",
    "thanks for watching",
    "amara.org",
    ".",
    "..",
    "...",
    "tsk.",
    "tsk",
})

VERBATIM_DISFLUENCY_PROMPT = (
    "Umm, uhh, ah, er, hmm... Like, you know, I mean, sort of, basically—transcribe strictly "
    "in English verbatim including every hesitation, stutter, false start, and filler word."
)


def clean_whisper_hallucination(text: str | None) -> str | None:
    if not text:
        return None
    cleaned = text.strip()
    if not cleaned:
        return None
    if cleaned.lower() in WHISPER_HALLUCINATION_PHRASES:
        return None
    if len(cleaned.split()) <= 2 and any(ch in cleaned.lower() for ch in ("ã", "õ", "ç", "¿", "¡")):
        return None
    return cleaned


def capabilities() -> dict[str, bool | str]:
    return {
        "opensmile_installed": importlib.util.find_spec("opensmile") is not None,
        "opensmile_initialized": _smile is not None,
        "dsp_fallback_available": importlib.util.find_spec("numpy") is not None,
        "ffmpeg_installed": shutil.which("ffmpeg") is not None,
        "groq_sdk_installed": importlib.util.find_spec("groq") is not None,
        "groq_api_configured": bool(os.getenv("GROQ_API_KEY", "").strip()),
        "faster_whisper_installed": importlib.util.find_spec("faster_whisper") is not None,
        "groq_model": os.getenv("GROQ_ASR_MODEL", "whisper-large-v3").strip() or "whisper-large-v3",
        "language_locked": os.getenv("GROQ_ASR_LANGUAGE", "en").strip() or "en",
    }


def compute_text_psycholinguistics(text: str | None) -> dict[str, float | int]:
    cleaned = clean_whisper_hallucination(text)
    if not cleaned:
        return {
            "text_valence_score": 0.0,
            "first_person_pronoun_ratio": 0.0,
            "epistemic_hedge_count": 0,
        }
    tokens = re.findall(r"\b[\w']+\b", cleaned.lower())
    if not tokens:
        return {
            "text_valence_score": 0.0,
            "first_person_pronoun_ratio": 0.0,
            "epistemic_hedge_count": 0,
        }

    fp_singular = len(FIRST_PERSON_SINGULAR_REGEX.findall(cleaned))
    fp_plural = len(FIRST_PERSON_PLURAL_DISTANCING_REGEX.findall(cleaned))
    pronoun_denom = max(1, fp_singular + fp_plural)
    fp_ratio = round(fp_singular / pronoun_denom, 4) if (fp_singular + fp_plural) > 0 else round(fp_singular / len(tokens), 4)

    hedges = len(EPISTEMIC_HEDGE_REGEX.findall(cleaned))

    score = 0.0
    hits = 0
    for idx, tok in enumerate(tokens):
        clean_tok = tok.strip("'")
        prev_window = tokens[max(0, idx - 3) : idx]
        negated = any(p.strip("'") in NEGATION_WORDS or p.endswith("n't") for p in prev_window)
        if clean_tok in POSITIVE_VALENCE_WORDS:
            score += -0.65 if negated else 0.72
            hits += 1
        elif clean_tok in NEGATIVE_VALENCE_WORDS:
            score += 0.45 if negated else -0.72
            hits += 1

    valence = max(-1.0, min(1.0, score / max(1, hits))) if hits > 0 else 0.0
    return {
        "text_valence_score": round(valence, 4),
        "first_person_pronoun_ratio": fp_ratio,
        "epistemic_hedge_count": hedges,
    }


def _dsp_acoustic_fallback(audio, sample_rate: int) -> dict[str, float | None]:
    """Deterministic NumPy/SciPy acoustic biomarker estimator (F0 semitones, Jitter, Shimmer dB, HNR dB, Loudness).
    UPGRADE 1: Normalizes audio volume first and NEVER uses `loudness_mean < 0.08` to discard voiced pitch!
    """
    import numpy as np

    raw_x = np.asarray(audio, dtype=np.float64).reshape(-1)
    if len(raw_x) < sample_rate // 4:
        return {}
    raw_rms = float(np.sqrt(np.mean(raw_x ** 2)))
    if raw_rms < 1e-5:
        return {
            "f0_semitone_mean": None,
            "f0_semitone_std": None,
            "jitter_local_mean": None,
            "jitter_local": None,
            "shimmer_local_db_mean": None,
            "shimmer_db": None,
            "hnr_db_mean": None,
            "hnr_db": None,
            "loudness_mean": 0.0,
            "loudness_peaks_per_sec": 0.0,
        }

    # Normalize volume before running autocorrelation pitch & perturbation analysis
    x = np.asarray(normalize_pcm_samples(raw_x.astype(np.float32), sample_rate), dtype=np.float64)
    rms = float(np.sqrt(np.mean(x ** 2)) + 1e-9)

    frame_len = int(0.040 * sample_rate)
    hop_len = int(0.015 * sample_rate)
    min_lag = max(1, int(sample_rate / 400.0))
    max_lag = max(min_lag + 2, int(sample_rate / 75.0))

    f0_semitones: list[float] = []
    periods: list[float] = []
    amps: list[float] = []
    hnrs: list[float] = []
    flux_vals: list[float] = []
    prev_spec = None

    for start in range(0, len(x) - frame_len, hop_len):
        frame = x[start : start + frame_len]
        f_rms = float(np.sqrt(np.mean(frame ** 2)))
        spec = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
        if prev_spec is not None:
            flux_vals.append(float(np.sqrt(np.mean((spec - prev_spec) ** 2))))
        prev_spec = spec
        if f_rms < rms * 0.22:
            continue
        centered = frame - np.mean(frame)
        corr = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
        if corr[0] <= 1e-9 or len(corr) <= max_lag:
            continue
        corr_norm = corr / corr[0]
        search_region = corr_norm[min_lag:max_lag]
        if len(search_region) == 0:
            continue
        best_offset = int(np.argmax(search_region))
        r_max = float(search_region[best_offset])
        if r_max < 0.24:
            continue
        lag = min_lag + best_offset
        f0_hz = float(sample_rate) / float(lag)
        if 70.0 <= f0_hz <= 400.0:
            f0_semitones.append(12.0 * log2(f0_hz / 27.5))
            periods.append(1.0 / f0_hz)
            amps.append(float(np.max(np.abs(centered))))
            r_clipped = min(0.999, max(0.01, r_max))
            hnrs.append(10.0 * log10(r_clipped / (1.0 - r_clipped)))

    loudness_est = round(min(5.0, rms * 4.0), 3)
    if not f0_semitones:
        return {
            "f0_semitone_mean": None,
            "f0_semitone_std": None,
            "jitter_local_mean": None,
            "jitter_local": None,
            "shimmer_local_db_mean": None,
            "shimmer_db": None,
            "hnr_db_mean": None,
            "hnr_db": None,
            "loudness_mean": loudness_est,
            "loudness_peaks_per_sec": 0.0,
        }

    f0_arr = np.asarray(f0_semitones, dtype=float)
    f0_mean = float(np.mean(f0_arr))
    f0_std = float(np.std(f0_arr) / max(f0_mean, 1.0))

    if len(periods) >= 2:
        p_arr = np.asarray(periods, dtype=float)
        jitter = float(np.mean(np.abs(np.diff(p_arr))) / max(float(np.mean(p_arr)), 1e-6))
    else:
        jitter = 0.01

    if len(amps) >= 2:
        a_arr = np.asarray(amps, dtype=float) + 1e-7
        shimmer_db = float(np.mean(np.abs(20.0 * np.log10(a_arr[1:] / a_arr[:-1]))))
    else:
        shimmer_db = 0.25

    hnr_val = float(np.mean(hnrs)) if hnrs else 10.0
    duration_sec = max(len(x) / sample_rate, 0.1)
    peaks_per_sec = len(f0_semitones) / duration_sec * 0.25

    return {
        "f0_semitone_mean": round(f0_mean, 3),
        "f0_semitone_std": round(f0_std, 4),
        "jitter_local_mean": round(jitter, 4),
        "jitter_local": round(jitter, 4),
        "shimmer_local_db_mean": round(shimmer_db, 4),
        "shimmer_db": round(shimmer_db, 4),
        "hnr_db_mean": round(hnr_val, 3),
        "hnr_db": round(hnr_val, 3),
        "loudness_mean": loudness_est,
        "loudness_peaks_per_sec": round(peaks_per_sec, 3),
        "spectral_flux_mean": round(float(np.mean(flux_vals)) if flux_vals else 0.0, 4),
    }


def _extract_egemaps(audio, sample_rate: int) -> dict[str, Any]:
    global _smile
    try:
        import numpy as np
    except ImportError:
        return {"status": "unavailable", "metrics": {}}
    if len(audio) < max(int(sample_rate * 0.25), 4000):
        return {"status": "insufficient_audio", "metrics": {}}

    # Normalize audio before openSMILE so quiet speakers still yield voiced F0, Jitter, Shimmer, HNR
    norm_audio = normalize_pcm_samples(np.asarray(audio, dtype=np.float32), sample_rate)

    try:
        import opensmile

        if _smile is None:
            _smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
        frame = _smile.process_signal(np.asarray(norm_audio, dtype=np.float32), sample_rate)
        row = frame.iloc[0]
        metrics: dict[str, float | None] = {}
        for source_name, target_name in _EGEMAPS.items():
            value = row.get(source_name)
            if value is not None and isfinite(float(value)):
                metrics[target_name] = round(float(value), 4)
        if "jitter_local_mean" in metrics:
            metrics["jitter_local"] = metrics["jitter_local_mean"]
        if "shimmer_local_db_mean" in metrics:
            metrics["shimmer_db"] = metrics["shimmer_local_db_mean"]
        if "hnr_db_mean" in metrics:
            metrics["hnr_db"] = metrics["hnr_db_mean"]
        # UPGRADE 1: Never mask voiced pitch using loudness_mean < 0.08! Only mask if f0_semitone_mean <= 1.0
        if (metrics.get("f0_semitone_mean") or 0.0) <= 1.0:
            for vc in ("f0_semitone_mean", "f0_semitone_std", "jitter_local_mean", "jitter_local", "shimmer_local_db_mean", "shimmer_db", "hnr_db_mean", "hnr_db"):
                metrics[vc] = None
        return {"status": "ready" if metrics else "no_features", "metrics": metrics}
    except ImportError:
        dsp_metrics = _dsp_acoustic_fallback(norm_audio, sample_rate)
        return {"status": "ready_dsp_fallback" if dsp_metrics else "unavailable", "metrics": dsp_metrics}
    except Exception as exc:
        logger.warning("openSMILE note (%s); using DSP acoustic extractor", type(exc).__name__)
        dsp_metrics = _dsp_acoustic_fallback(norm_audio, sample_rate)
        if dsp_metrics:
            return {"status": "ready_dsp_fallback", "metrics": dsp_metrics}
        return {"status": f"error:{type(exc).__name__}", "metrics": {}}


def _client():
    global _groq_client
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        return None
    with _client_lock:
        if _groq_client is None:
            from groq import Groq

            _groq_client = Groq(api_key=key)
    return _groq_client


def _wav_bytes(audio, sample_rate: int) -> bytes:
    import numpy as np

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


def compress_for_groq_free_tier(input_wav_path: str, output_ogg_path: str) -> str:
    """Normalizes audio volume via FFmpeg loudnorm (`I=-16:TP=-1.5:LRA=11`) and encodes to 16kHz Mono Opus/OGG."""
    cmd = [
        "ffmpeg", "-y", "-i", input_wav_path,
        "-vn",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ac", "1", "-ar", "16000",
        "-c:a", "libopus", "-b:a", "48k", output_ogg_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return output_ogg_path


def _prepare_groq_upload_tuple(audio, sample_rate: int) -> tuple[str, bytes]:
    """Normalizes audio with FFmpeg loudnorm (`loudnorm=I=-16:TP=-1.5:LRA=11` at 16kHz mono) before Whisper."""
    norm_samples = normalize_pcm_samples(audio, sample_rate)
    raw_wav = _wav_bytes(norm_samples, sample_rate)
    if len(raw_wav) <= 2_000_000 or shutil.which("ffmpeg") is None:
        return ("normalized_audio.wav", raw_wav)
    with tempfile.TemporaryDirectory() as tmpdir:
        in_wav = str(Path(tmpdir) / "input.wav")
        out_ogg = str(Path(tmpdir) / "normalized_audio.ogg")
        Path(in_wav).write_bytes(raw_wav)
        try:
            compress_for_groq_free_tier(in_wav, out_ogg)
            return ("normalized_audio.ogg", Path(out_ogg).read_bytes())
        except Exception as exc:
            logger.warning("FFmpeg Opus compression fallback to WAV: %s", exc)
            return ("normalized_audio.wav", raw_wav)


def _transcribe_local_faster_whisper(audio, sample_rate: int, language: str = "en") -> dict[str, Any] | None:
    """Offline fallback using local faster-whisper (`base` or `small`) if Groq API is offline or unconfigured."""
    global _faster_whisper_model
    if importlib.util.find_spec("faster_whisper") is None:
        return None
    try:
        from faster_whisper import WhisperModel

        with _client_lock:
            if _faster_whisper_model is None:
                model_size = os.getenv("LOCAL_WHISPER_MODEL", "base").strip() or "base"
                _faster_whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")

        norm_samples = normalize_pcm_samples(audio, sample_rate)
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = str(Path(tmpdir) / "normalized_audio.wav")
            Path(wav_path).write_bytes(_wav_bytes(norm_samples, sample_rate))
            seg_gen, _ = _faster_whisper_model.transcribe(
                wav_path,
                language=language or "en",
                word_timestamps=True,
                initial_prompt=VERBATIM_DISFLUENCY_PROMPT,
            )
            segments = []
            words = []
            texts = []
            for seg in seg_gen:
                seg_txt = clean_whisper_hallucination(seg.text)
                if not seg_txt:
                    continue
                texts.append(seg_txt)
                seg_words = []
                for w in getattr(seg, "words", None) or []:
                    w_dict = {"word": w.word, "start": float(w.start), "end": float(w.end)}
                    seg_words.append(w_dict)
                    words.append(w_dict)
                segments.append({
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": seg_txt,
                    "words": seg_words,
                })
            full_text = clean_whisper_hallucination(" ".join(texts))
            if full_text:
                print(f"[OmniPulse faster-whisper ASR] Full transcript: {full_text}", flush=True)
                logger.info("[OmniPulse faster-whisper ASR] Full transcript: %s", full_text)
            return {
                "text": full_text,
                "segments": segments,
                "words": words,
            }
    except Exception as exc:
        logger.error("[OmniPulse faster-whisper Error] %s: %s", type(exc).__name__, exc)
        print(f"[OmniPulse faster-whisper Error] {type(exc).__name__}: {exc}", flush=True)
        return None


def _transcribe(audio, sample_rate: int, language: str | None = "en") -> dict[str, Any]:
    """UPGRADE 1: Normalizes audio volume, runs Groq `whisper-large-v3` (`language="en"`) or local
    `faster-whisper`, logs the full transcript to console, and NEVER swallows ASR errors silently.
    """
    if len(audio) < max(1, int(sample_rate * 0.05)):
        return {"status": "insufficient_audio", "alignment_status": "skipped", "transcript": None, "segments": [], "words": [], "metrics": {}}

    duration = max(float(len(audio)) / sample_rate, 0.001)
    language_code = (language or os.getenv("GROQ_ASR_LANGUAGE", "en")).strip() or "en"
    raw_segments: list[dict[str, Any]] = []
    words: list[dict[str, Any]] = []
    text: str | None = None
    asr_status = "not_configured"

    if importlib.util.find_spec("groq") is not None and os.getenv("GROQ_API_KEY", "").strip():
        try:
            client = _client()
            upload_file = _prepare_groq_upload_tuple(audio, sample_rate)
            kwargs: dict[str, Any] = {
                "file": upload_file,
                "model": os.getenv("GROQ_ASR_MODEL", "whisper-large-v3").strip() or "whisper-large-v3",
                "language": language_code,
                "prompt": VERBATIM_DISFLUENCY_PROMPT,
                "response_format": "verbose_json",
                "timestamp_granularities": ["word", "segment"],
                "temperature": 0.0,
            }
            response = client.audio.transcriptions.create(**kwargs)
            raw = response.model_dump() if hasattr(response, "model_dump") else dict(response)
            raw_segments = raw.get("segments") or []
            words = raw.get("words") or [word for segment in raw_segments for word in (segment.get("words") or [])]
            text = clean_whisper_hallucination(str(raw.get("text") or "").strip())
            asr_status = "ready"
            if text:
                print(f"[OmniPulse Groq Whisper ASR] ({duration:.1f}s): {text}", flush=True)
                logger.info("[OmniPulse Groq Whisper ASR] (%0.1fs): %s", duration, text)
        except Exception as exc:
            logger.error("[OmniPulse Groq Whisper ASR Error] %s: %s", type(exc).__name__, exc)
            print(f"[OmniPulse Groq Whisper ASR Error] {type(exc).__name__}: {exc}", flush=True)
            asr_status = f"error:{type(exc).__name__}"

    if text is None:
        local_out = _transcribe_local_faster_whisper(audio, sample_rate, language=language_code)
        if local_out and local_out.get("text"):
            text = local_out["text"]
            raw_segments = local_out.get("segments") or []
            words = local_out.get("words") or []
            asr_status = "ready_local_faster_whisper"

    if text is None:
        return {
            "status": asr_status,
            "alignment_status": "skipped",
            "transcript": None,
            "segments": [],
            "words": [],
            "metrics": {},
        }

    word_text = [str(word.get("word") or "").strip() for word in words if clean_whisper_hallucination(str(word.get("word") or ""))]
    if not word_text and text:
        word_text = re.findall(r"\b[\w']+\b", text)

    pauses = 0.0
    previous_end = None
    first_word_start_sec: float | None = None
    for word in words:
        start, end = word.get("start"), word.get("end")
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            if first_word_start_sec is None:
                first_word_start_sec = max(0.0, float(start))
            if previous_end is not None:
                gap = max(0.0, float(start) - previous_end)
                if gap >= 0.25:
                    pauses += gap
            previous_end = float(end)

    metrics: dict[str, float | int] = {}
    if word_text:
        metrics["speech_rate_wpm"] = round(len(word_text) * 60.0 / duration, 3)
        metrics["disfluency_count"] = len(FILLER_REGEX.findall(text))
    if words:
        metrics["pause_ratio"] = round(min(1.0, pauses / duration), 4)
    if first_word_start_sec is not None:
        metrics["response_latency_ms"] = round(first_word_start_sec * 1000.0, 1)

    metrics.update(compute_text_psycholinguistics(text))

    segments = []
    for segment in raw_segments:
        seg_txt = clean_whisper_hallucination(segment.get("text", ""))
        if not seg_txt:
            continue
        segments.append({
            "start": segment.get("start"),
            "end": segment.get("end"),
            "text": seg_txt,
            "words": [
                {key: word.get(key) for key in ("word", "start", "end", "score") if word.get(key) is not None}
                for word in (segment.get("words") or [])
            ],
        })
    if not segments and words:
        segments = [{
            "start": min((word.get("start", 0.0) for word in words), default=0.0),
            "end": max((word.get("end", duration) for word in words), default=duration),
            "text": text,
            "words": [
                {key: word.get(key) for key in ("word", "start", "end", "score") if word.get(key) is not None}
                for word in words
            ],
        }]
    return {
        "status": asr_status,
        "alignment_status": "ready" if words else "unavailable",
        "transcript": text,
        "segments": segments,
        "words": words,
        "metrics": metrics,
    }


def transcribe_full_session_and_map_windows(
    full_audio_samples,
    sample_rate: int,
    windows: list[dict[str, Any]],
    *,
    language: str = "en",
) -> list[dict[str, Any]]:
    """UPGRADE 1: Normalizes the entire session audio via FFmpeg loudnorm (`loudnorm=I=-16:TP=-1.5:LRA=11`),
    transcribes in a single call (`whisper-large-v3` with `language="en"` or local `faster-whisper`),
    logs the full transcript to console, and maps word/segment timestamps into the 2.0-second windows.
    Also propagates rolling 8-second `transcript_segments` into any adjacent voiced windows that lacked
    direct window-level text.
    """
    for w in windows:
        cleaned = clean_whisper_hallucination(w.get("transcript_segment"))
        w["transcript_segment"] = cleaned
        if not cleaned:
            m = w.get("metrics") or {}
            if "text_valence_score" in m:
                m["text_valence_score"] = 0.0

    # First backfill any window that has aligned words inside `transcript_segments` from rolling 8s browser buffers
    all_aligned_words: list[dict[str, Any]] = []
    for w in windows:
        for seg in w.get("transcript_segments") or []:
            for wd in seg.get("words") or []:
                if isinstance(wd, dict) and wd.get("word") and isinstance(wd.get("start"), (int, float)):
                    all_aligned_words.append(wd)

    if full_audio_samples is not None and len(full_audio_samples) >= sample_rate:
        norm_full = normalize_pcm_samples(full_audio_samples, sample_rate)
        res = _transcribe(norm_full, sample_rate, language=language)
        if res.get("transcript"):
            print(f"[OmniPulse Full-Session ASR] Complete transcript: {res['transcript']}", flush=True)
            logger.info("[OmniPulse Full-Session ASR] Complete transcript: %s", res["transcript"])
        words = res.get("words") or []
        if words:
            all_aligned_words = words
        elif res.get("segments"):
            # Map segment-level text into overlapping 2.0s windows if word timestamps weren't returned
            for seg in res["segments"]:
                s_start = float(seg.get("start") or 0.0)
                s_end = float(seg.get("end") or (s_start + 2.0))
                s_txt = clean_whisper_hallucination(seg.get("text"))
                if not s_txt:
                    continue
                for w in windows:
                    w_start = float(w.get("start_sec", 0.0))
                    w_end = float(w.get("end_sec", w_start + 2.0))
                    if s_start < w_end and s_end > w_start and not w.get("transcript_segment"):
                        w["transcript_segment"] = s_txt
                        w.setdefault("metrics", {}).update(compute_text_psycholinguistics(s_txt))

    if all_aligned_words:
        for w in windows:
            w_start = float(w.get("start_sec", 0.0))
            w_end = float(w.get("end_sec", w_start + 2.0))
            w_dur = max(0.2, w_end - w_start)
            matched_words = []
            for wd in all_aligned_words:
                ws = wd.get("start")
                we = wd.get("end", ws)
                w_tok = str(wd.get("word") or "").strip()
                if not w_tok or not isinstance(ws, (int, float)):
                    continue
                w_mid = 0.5 * (float(ws) + float(we if isinstance(we, (int, float)) else ws))
                if w_start <= w_mid < w_end:
                    matched_words.append(wd)

            if matched_words:
                joined = clean_whisper_hallucination(" ".join(str(wd.get("word") or "").strip() for wd in matched_words))
                if joined:
                    w["transcript_segment"] = joined
                    metrics = w.setdefault("metrics", {})
                    metrics["speech_rate_wpm"] = round(len(matched_words) * 60.0 / w_dur, 3)
                    metrics["disfluency_count"] = len(FILLER_REGEX.findall(joined))
                    metrics.update(compute_text_psycholinguistics(joined))

    return windows


def analyze_audio(
    audio,
    sample_rate: int = 16000,
    *,
    transcribe: bool = True,
    transcription_audio=None,
    language: str | None = "en",
) -> dict[str, Any]:
    """Analyze in-memory audio locally (loudnorm + openSMILE eGeMAPSv02 or DSP fallback),
    then optionally transcribe via Groq whisper-large-v3 or local faster-whisper.
    """
    import numpy as np

    samples = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    acoustic = _extract_egemaps(samples, sample_rate)
    speech_samples = np.asarray(transcription_audio, dtype=np.float32).reshape(-1) if transcription_audio is not None else samples
    speech = (
        _transcribe(speech_samples, sample_rate, language=language or "en")
        if transcribe
        else {"status": "skipped", "alignment_status": "skipped", "transcript": None, "segments": [], "metrics": {}}
    )
    return {
        "metrics": {**acoustic["metrics"], **speech.get("metrics", {})},
        "transcript": speech.get("transcript"),
        "transcript_segments": speech.get("segments", []),
        "acoustic_status": acoustic["status"],
        "asr_status": speech["status"],
        "alignment_status": speech.get("alignment_status", "skipped"),
    }
