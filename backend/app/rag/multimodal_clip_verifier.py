"""Implements OmniPulse v2.0 Upgrade 4: Native Video-Clip Visual Common Sense Gate via Gemini 2.5 Pro
(multimodal_clip_verifier.py), plus deterministic autonomic/kinesic artifact verification fallback.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

from pydantic import Field
from app.schemas import CompatBaseModel


class ClipArtifactVerificationSchema(CompatBaseModel):
    is_physical_artifact: bool = Field(
        description="True if the movement/audio spike is caused by a non-psychological physical action like adjusting glasses, itching, coughing, or shifting chair."
    )
    artifact_probability: float = Field(ge=0.0, le=1.0)
    physical_action_observed: str = Field(
        description="Concise description of what the person physically does in the 4-second clip."
    )
    recommended_action: str = Field(
        description="One of: SUPPRESS_FALSE_POSITIVE, RETAIN_AS_GENUINE_BEHAVIORAL_EVENT, DOWNGRADE_CONFIDENCE"
    )


class MultimodalClipVerifier:
    """Slices a 4.0-second audiovisual clip around flagged anomaly windows and verifies
    via Gemini 2.5 Pro (or autonomic cross-modal concordance heuristics when raw MP4 is absent)
    whether a hand-to-face touch or blink flurry is a benign physical maintenance artifact
    (P_artifact >= 0.65 -> BENIGN_PHYSICAL_ARTIFACT).
    """

    def __init__(self, gemini_api_key: str | None = None, model: str = "gemini-2.5-pro"):
        self.api_key = (gemini_api_key or "").strip()
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None and self.api_key:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    @staticmethod
    def slice_4s_anomaly_clip(source_mp4: str, center_sec: float, output_mp4: str) -> str:
        """Extract a 4.0s raw audiovisual snippet [t_event - 1.0s, t_event + 3.0s] at 720p, 15 FPS using FFmpeg."""
        start_sec = max(0.0, float(center_sec) - 1.0)
        cmd = [
            "ffmpeg", "-y", "-ss", str(round(start_sec, 2)),
            "-i", source_mp4, "-t", "4.0",
            "-vf", "scale=-2:720,fps=15",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26",
            "-c:a", "aac", "-b:a", "96k", output_mp4,
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return output_mp4

    def verify_anomaly_clip(self, clip_path: str | None, flagged_signals: dict[str, Any]) -> dict[str, Any]:
        """Verify anomaly clip with Gemini 2.5 Pro if video file and API key exist;
        otherwise evaluate cross-modal autonomic concordance heuristics (AC-8).
        """
        if clip_path and Path(clip_path).is_file() and self.api_key:
            try:
                from google.genai import types

                video_bytes = Path(clip_path).read_bytes()
                prompt = (
                    "You are a visual artifact verification gate for a behavioral telemetry lab.\n"
                    f"Automated sensors flagged this 4-second window with: {json.dumps(flagged_signals)}.\n"
                    "Inspect the video clip carefully. Determine if the hand-to-face movement, eye closure, "
                    "or vocal perturbation is merely a benign physical maintenance action (e.g., pushing up slipping eyeglasses, "
                    "scratching an itch, adjusting a microphone/headset, coughing, sneezing, drinking water) OR a genuine "
                    "non-verbal behavioral expression/adaptor during conversation."
                )
                client = self._get_client()
                response = client.models.generate_content(
                    model=self.model,
                    contents=[
                        types.Part.from_bytes(data=video_bytes, mime_type="video/mp4"),
                        prompt,
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ClipArtifactVerificationSchema,
                        temperature=0.1,
                    ),
                )
                if response.text:
                    parsed = ClipArtifactVerificationSchema.model_validate_json(response.text)
                    return parsed.model_dump()
            except Exception:
                pass

        return self.verify_from_telemetry_heuristics(flagged_signals)

    @staticmethod
    def verify_from_telemetry_heuristics(flagged_signals: dict[str, Any]) -> dict[str, Any]:
        """AC-8 Heuristic Gate: When a participant touches their face/glasses or blinks rapidly
        while heart rate (|Z_HR| < 0.8) and vocal jitter (|Z_jitter| < 0.8) remain completely calm
        and speech is fluent (CMCI >= 0.70), classify as a probable benign physical maintenance action
        (P_artifact >= 0.65 -> SUPPRESS_FALSE_POSITIVE).
        """
        z_scores = flagged_signals.get("dual_horizon_z_scores") or flagged_signals.get("baseline_z_scores") or {}
        metrics = flagged_signals.get("metrics") or flagged_signals.get("measurements") or {}
        flags = set(flagged_signals.get("behavioral_flags") or [])

        hr_z = abs(float(z_scores.get("hr_bpm_z_floor") or z_scores.get("hr_bpm") or 0.0))
        jitter_z = abs(float(z_scores.get("jitter_local_mean_z_floor") or z_scores.get("jitter_local_z_floor") or z_scores.get("jitter_local_mean") or 0.0))
        au24_z = abs(float(z_scores.get("au24_lip_presser_z_floor") or z_scores.get("au24_lip_presser") or 0.0))
        self_touch = float(metrics.get("self_touch_ratio") or 0.0)
        cmci = float((flagged_signals.get("composite_indices") or {}).get("cross_modal_congruency_index") or flagged_signals.get("cmci_score") or 0.5)

        isolated_physical_touch = (
            (self_touch >= 0.30 or "PACIFYING_NECK_TOUCH" in flags or "SIGNIFICANT_DEVIATION_SELF_TOUCH_RATIO" in flags)
            and hr_z < 0.80
            and jitter_z < 0.80
            and au24_z < 1.0
            and cmci >= 0.65
        )
        if isolated_physical_touch:
            return {
                "is_physical_artifact": True,
                "artifact_probability": 0.78,
                "physical_action_observed": "Isolated brief hand-to-face/glasses contact with stable heart rate and calm vocal prosody.",
                "recommended_action": "SUPPRESS_FALSE_POSITIVE",
            }

        convergent_modalities = sum([
            hr_z >= 1.5,
            jitter_z >= 1.5,
            au24_z >= 1.5,
            self_touch >= 0.35,
            cmci < 0.55,
        ])
        if convergent_modalities >= 2:
            prob = round(max(0.05, 0.30 - 0.06 * convergent_modalities), 2)
            return {
                "is_physical_artifact": False,
                "artifact_probability": prob,
                "physical_action_observed": "Sustained multi-channel non-verbal shift co-occurring with vocal and autonomic modulation.",
                "recommended_action": "RETAIN_AS_GENUINE_BEHAVIORAL_EVENT",
            }

        return {
            "is_physical_artifact": False,
            "artifact_probability": 0.35,
            "physical_action_observed": "Single-channel movement variation; evaluate with moderate confidence.",
            "recommended_action": "DOWNGRADE_CONFIDENCE",
        }
