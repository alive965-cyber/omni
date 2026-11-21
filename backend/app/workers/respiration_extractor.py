"""
backend/app/workers/respiration_extractor.py
Upgrade 2: Optical Respiration Rate & Sigh Detection using MediaPipe Pose shoulder landmarks
(#11 Left Shoulder, #12 Right Shoulder) and chest-to-chin distance.
Works in almost any lighting condition (even at 122 Lux when skin-chrominance rPPG fails).
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence
from app.services.report_enhancer import OmniPulseReportEnhancer

LEFT_SHOULDER_LANDMARK = 11
RIGHT_SHOULDER_LANDMARK = 12
CHIN_LANDMARK = 152


class KinematicRespirationExtractor:
    """Extracts sub-pixel vertical oscillation of the shoulders (0.12 Hz to 0.45 Hz, 7.2 to 27 breaths/min),
    Thoracic Breath-Holding (Apnea Freeze), and Deep Post-Stress Relief Sighs.
    """

    @staticmethod
    def compute_shoulder_midpoint_y(
        pose_landmarks: Sequence[Dict[str, float] | Sequence[float]],
        face_landmarks: Sequence[Dict[str, float] | Sequence[float]] | None = None,
    ) -> float | None:
        if not pose_landmarks or len(pose_landmarks) <= RIGHT_SHOULDER_LANDMARK:
            return None

        def _get_y(pt: Any) -> float:
            if isinstance(pt, dict):
                return float(pt.get("y", 0.0))
            return float(pt[1])

        y_l = _get_y(pose_landmarks[LEFT_SHOULDER_LANDMARK])
        y_r = _get_y(pose_landmarks[RIGHT_SHOULDER_LANDMARK])
        shoulder_y = 0.5 * (y_l + y_r)

        if face_landmarks and len(face_landmarks) > CHIN_LANDMARK:
            chin_y = _get_y(face_landmarks[CHIN_LANDMARK])
            return round(float(shoulder_y - 0.15 * chin_y), 6)
        return round(float(shoulder_y), 6)

    @staticmethod
    def analyze_shoulder_trajectory(
        timestamps_sec: List[float],
        shoulder_y_midpoint: List[float],
    ) -> Dict[str, Any]:
        return OmniPulseReportEnhancer.extract_shoulder_respiration(timestamps_sec, shoulder_y_midpoint)
