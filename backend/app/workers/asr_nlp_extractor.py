"""
backend/app/workers/asr_nlp_extractor.py
UPGRADE 1: Loudnorm-normalized Whisper ASR & Psycholinguistic Extractor.
Runs Groq `whisper-large-v3` (`language="en"`) with automatic fallback to local `faster-whisper`
(`base`/`small`) if offline or unconfigured. Logs the full transcript to console and never swallows
ASR errors silently.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from app.services import audio_inference
from app.workers.audio_extractor import normalize_audio_file_ffmpeg, normalize_pcm_samples

logger = logging.getLogger("omnipulse.asr")


class AsrNlpExtractor:
    """Executes loudnorm audio normalization followed by Groq whisper-large-v3 or local faster-whisper."""

    @staticmethod
    def transcribe_normalized_audio(
        audio_samples,
        sample_rate: int = 16000,
        language: str = "en",
    ) -> Dict[str, Any]:
        norm_samples = normalize_pcm_samples(audio_samples, sample_rate=sample_rate)
        result = audio_inference.analyze_audio(
            norm_samples,
            sample_rate=sample_rate,
            transcribe=True,
            transcription_audio=norm_samples,
            language=language,
        )
        transcript = result.get("transcript")
        if transcript:
            logger.info("[OmniPulse ASR] Transcribed utterance: %s", transcript)
            print(f"[OmniPulse ASR] Transcribed utterance: {transcript}", flush=True)
        return result

    @staticmethod
    def transcribe_video_or_audio_file(
        input_media_path: str,
        windows: List[Dict[str, Any]],
        language: str = "en",
    ) -> List[Dict[str, Any]]:
        import tempfile
        import wave
        from pathlib import Path
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            norm_wav = str(Path(tmpdir) / "normalized_audio.wav")
            normalize_audio_file_ffmpeg(input_media_path, norm_wav)
            with wave.open(norm_wav, "rb") as wf:
                sr = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
            samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
            return audio_inference.transcribe_full_session_and_map_windows(
                samples, sr, windows, language=language
            )
