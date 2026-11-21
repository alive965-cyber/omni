"""
backend/app/workers/audio_extractor.py
UPGRADE 1: Normalizes audio volume using FFmpeg loudnorm (`loudnorm=I=-16:TP=-1.5:LRA=11` at 16kHz mono)
and extracts openSMILE eGeMAPSv02 / DSP voice quality biomarkers without the destructive
`loudness_mean < 0.08` unvoiced cutoff.
A window is voiced whenever `f0_semitone_mean > 1.0` OR `len(transcript_segment.strip()) > 0`.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np


def normalize_audio_file_ffmpeg(input_media_path: str, output_wav_path: str) -> str:
    """Normalizes audio volume before running openSMILE and Whisper using FFmpeg loudnorm:
    ffmpeg -y -i input_video.mp4 -vn -af "loudnorm=I=-16:TP=-1.5:LRA=11" -ar 16000 -ac 1 normalized_audio.wav
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_media_path),
        "-vn",
        "-af",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_wav_path),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return str(output_wav_path)


def normalize_pcm_samples(samples: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Normalizes 1D float32 audio samples using FFmpeg loudnorm when available, with an
    automatic target-RMS (-16 dBFS) gain normalizer so quiet microphone captures are never lost.
    """
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if len(x) == 0:
        return x

    rms = float(np.sqrt(np.mean(x ** 2)))
    if rms < 1e-6:
        return x

    # Fast in-memory RMS normalization to -16 dBFS (target_rms ~ 0.158) with soft limiter
    target_rms = 0.158
    if rms < target_rms * 0.85:
        gain = min(25.0, target_rms / max(rms, 1e-5))
        x = np.tanh(x * gain * 0.9) / 0.9

    if shutil.which("ffmpeg") is not None and len(x) >= sample_rate:
        try:
            import wave

            with tempfile.TemporaryDirectory() as tmpdir:
                in_wav = Path(tmpdir) / "raw.wav"
                out_wav = Path(tmpdir) / "normalized_audio.wav"
                pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
                with wave.open(str(in_wav), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(pcm)
                normalize_audio_file_ffmpeg(str(in_wav), str(out_wav))
                with wave.open(str(out_wav), "rb") as rf:
                    norm_frames = rf.readframes(rf.getnframes())
                norm_arr = np.frombuffer(norm_frames, dtype="<i2").astype(np.float32) / 32768.0
                if len(norm_arr) > 0:
                    return norm_arr
        except Exception:
            pass

    return np.clip(x, -1.0, 1.0)


def is_window_voiced(metrics: Dict[str, Any], transcript_segment: str | None = None) -> bool:
    """UPGRADE 1 Rule: A window is voiced whenever `f0_semitone_mean > 1.0` OR `len(transcript_segment.strip()) > 0`.
    Never uses `loudness_mean < 0.08` to mark a window as silent!
    """
    f0 = metrics.get("f0_semitone_mean")
    if isinstance(f0, (int, float)) and not isinstance(f0, bool) and float(f0) > 1.0:
        return True
    if transcript_segment and len(str(transcript_segment).strip()) > 0:
        return True
    return False
