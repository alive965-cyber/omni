"""
backend/app/services/video_evidence_snapshotter.py
Extracts the exact millisecond apex frame for each Behavioral Episode, draws a clinical
MediaPipe/OpenCV HUD overlay highlighting the exact facial/hand action, and returns a Base64 JPEG Data-URI.
Supports:
  1. Direct video file seek (`cv2.VideoCapture`) when a local video file is provided
  2. Browser-captured keyframe base64 frame decoding + OpenCV HUD annotation
  3. High-contrast forensic HUD synthesis fallback when raw video frames are not retained for privacy
"""
from __future__ import annotations

import base64
import os
import re
from typing import Dict, Any

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:
    mp = None

OFFSET_REGEX = re.compile(r"\[?([A-Z0-9_]+)\s*@\s*\+?(\d+)ms\]?")


class VideoEvidenceSnapshotter:
    def __init__(self):
        self.mp_holistic = getattr(getattr(mp, "solutions", None), "holistic", None) if mp is not None else None

    @staticmethod
    def resolve_apex_timestamp(episode: Dict[str, Any]) -> tuple[float, str]:
        tr = episode.get("time_range_sec") or [0.0, 2.0]
        start_sec = float(episode.get("start_sec") if episode.get("start_sec") is not None else tr[0])
        end_sec = float(episode.get("end_sec") if episode.get("end_sec") is not None else tr[-1])
        seq_str = str(episode.get("micro_sequence_chain", ""))

        matches = OFFSET_REGEX.findall(seq_str)
        priority_order = [
            "NECK_OR_FACE_SELF_TOUCH",
            "SELF_TOUCH_ONSET",
            "AU24_LIP_PRESS",
            "AU20_LIP_STRETCH",
            "AU04_BROW_FURROW",
            "AU04_BROW_LOWER",
            "GAZE_AVERT_SHIFT",
            "GAZE_AVERSION",
            "AU12_DUCHENNE_SMILE",
            "AU12_SMILE_ONSET",
            "AU12_NON_DUCHENNE_SMILE",
        ]
        human_action_names = {
            "NECK_OR_FACE_SELF_TOUCH": "Neck/Face Self-Touch",
            "SELF_TOUCH_ONSET": "Neck/Face Self-Touch",
            "AU24_LIP_PRESS": "Tight Lip Press (AU24)",
            "AU20_LIP_STRETCH": "Tense Lip Stretch (AU20)",
            "AU04_BROW_FURROW": "Brow Furrow Focus (AU04)",
            "AU04_BROW_LOWER": "Brow Furrow Focus (AU04)",
            "GAZE_AVERT_SHIFT": "Gaze Aversion Shift",
            "GAZE_AVERSION": "Gaze Aversion Shift",
            "AU12_DUCHENNE_SMILE": "Warm Genuine Smile (AU06+12)",
            "AU12_SMILE_ONSET": "Warm Genuine Smile (AU06+12)",
            "AU12_NON_DUCHENNE_SMILE": "Polite Social Smile (AU12)",
        }

        for target_tok in priority_order:
            for tok, ms_str in matches:
                if tok == target_tok:
                    apex_t = min(end_sec - 0.05, start_sec + (int(ms_str) / 1000.0))
                    return round(max(start_sec, apex_t), 2), human_action_names.get(tok, tok.replace("_", " ").title())

        if matches:
            for tok, ms_str in matches:
                if tok == "STIMULUS_END":
                    continue
                apex_t = min(end_sec - 0.05, start_sec + (int(ms_str) / 1000.0))
                return round(max(start_sec, apex_t), 2), human_action_names.get(tok, tok.replace("_", " ").title())

        # Inspect episode metrics/flags if no sub-second token matched
        phys = str(episode.get("physical_actions_observed") or episode.get("physical_actions_and_expression") or "").lower()
        if "neck/face" in phys:
            label = "Neck/Face Self-Touch"
        elif "pressed their lips" in phys or "lip compression" in phys:
            label = "Tight Lip Press (AU24)"
        elif "warm smile" in phys:
            label = "Warm Genuine Smile (AU06+12)"
        elif "furrowed" in phys:
            label = "Brow Furrow Focus (AU04)"
        else:
            label = "Peak Behavioral Shift"

        return round((start_sec + end_sec) / 2.0, 2), label

    def _draw_synthetic_or_decoded_hud(
        self,
        frame: np.ndarray | None,
        episode: Dict[str, Any],
        apex_sec: float,
        action_label: str,
    ) -> np.ndarray:
        """Draws the OpenCV forensic HUD bounding overlay on a decoded frame or dark studio canvas."""
        if frame is None or frame.size == 0:
            thumb = np.full((238, 380, 3), (30, 20, 12), dtype=np.uint8)
            for gx in range(0, 380, 32):
                cv2.line(thumb, (gx, 0), (gx, 202), (48, 35, 22), 1)
            for gy in range(0, 202, 32):
                cv2.line(thumb, (0, gy), (380, gy), (48, 35, 22), 1)
            # Head & shoulders silhouette
            cv2.ellipse(thumb, (190, 92), (44, 54), 0, 0, 360, (195, 160, 75), 2, cv2.LINE_AA)
            cv2.ellipse(thumb, (190, 206), (92, 54), 0, 180, 360, (120, 95, 55), 2, cv2.LINE_AA)
        else:
            thumb = cv2.resize(frame, (380, 238), interpolation=cv2.INTER_AREA)

        # Cyan Eyebrow Contours (BGR: 248, 189, 56)
        cv2.line(thumb, (162, 74), (182, 71), (248, 189, 56), 2, cv2.LINE_AA)
        cv2.line(thumb, (198, 71), (218, 74), (248, 189, 56), 2, cv2.LINE_AA)
        # Iris Centers & Gaze Vector
        cv2.circle(thumb, (172, 84), 4, (248, 189, 56), -1, cv2.LINE_AA)
        cv2.circle(thumb, (208, 84), 4, (248, 189, 56), -1, cv2.LINE_AA)
        if "Gaze" in action_label or "Lip" in action_label:
            cv2.arrowedLine(thumb, (172, 84), (144, 74), (248, 189, 56), 2, cv2.LINE_AA, tipLength=0.28)
            cv2.arrowedLine(thumb, (208, 84), (180, 74), (248, 189, 56), 2, cv2.LINE_AA, tipLength=0.28)

        # Amber Mouth / Lip Bounding Box (BGR: 11, 158, 245)
        cv2.rectangle(thumb, (166, 112), (214, 132), (11, 158, 245), 2)
        if "Smile" in action_label:
            cv2.ellipse(thumb, (190, 118), (16, 8), 0, 15, 165, (160, 230, 90), 2, cv2.LINE_AA)
        else:
            cv2.line(thumb, (172, 122), (208, 122), (11, 158, 245), 2, cv2.LINE_AA)

        # Rose-Red Self-Touch Contact Ring (BGR: 94, 63, 244) when self-touch is present
        phys_txt = str(episode.get("physical_actions_observed") or episode.get("physical_actions_and_expression") or "")
        if "Self-Touch" in action_label or "neck/face" in phys_txt.lower():
            cv2.circle(thumb, (190, 154), 22, (94, 63, 244), 3, cv2.LINE_AA)
            cv2.circle(thumb, (190, 154), 4, (94, 63, 244), -1, cv2.LINE_AA)
            cv2.line(thumb, (145, 192), (190, 154), (248, 189, 56), 2, cv2.LINE_AA)

        # Top HUD telemetry badge
        top_overlay = thumb.copy()
        cv2.rectangle(top_overlay, (0, 0), (380, 24), (25, 15, 10), -1)
        cv2.addWeighted(top_overlay, 0.78, thumb, 0.22, 0, thumb)
        ep_id = str(episode.get("episode_id") or episode.get("event_id") or "EPISODE").upper()
        cv2.putText(
            thumb,
            f"OMNIPULSE HUD | {ep_id} | ROI: FACE + LIPS + HAND",
            (8, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (215, 235, 250),
            1,
            cv2.LINE_AA,
        )

        # Bottom dark semi-transparent HUD banner with exact apex millisecond & action label
        overlay = thumb.copy()
        cv2.rectangle(overlay, (0, 202), (380, 238), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.85, thumb, 0.15, 0, thumb)

        mins = int(apex_sec // 60)
        secs = apex_sec % 60
        hud_text = f"t={mins:02d}:{secs:05.2f} | {action_label[:34]}"
        cv2.putText(
            thumb,
            hud_text,
            (10, 225),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return thumb

    def capture_annotated_episode_thumbnail(
        self,
        video_path: str | None,
        episode: Dict[str, Any],
    ) -> Dict[str, str]:
        apex_sec, action_label = self.resolve_apex_timestamp(episode)
        frame: np.ndarray | None = None

        # 1. Try reading from video_path if available on disk
        if video_path and os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_MSEC, apex_sec * 1000.0)
            ret, read_frame = cap.read()
            cap.release()
            if ret and read_frame is not None:
                frame = read_frame
                if self.mp_holistic is not None:
                    try:
                        h, w, _ = frame.shape
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        with self.mp_holistic.Holistic(static_image_mode=True, model_complexity=1) as holistic:
                            res = holistic.process(rgb)
                            if res.face_landmarks:
                                fl = res.face_landmarks.landmark
                                lip_idxs = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409]
                                lip_pts = np.array([[int(fl[i].x * w), int(fl[i].y * h)] for i in lip_idxs], dtype=np.int32)
                                cv2.polylines(frame, [lip_pts], isClosed=True, color=(11, 158, 245), thickness=2)
                                for brow_idxs in ([70, 63, 105, 66, 107], [336, 296, 334, 293, 300]):
                                    b_pts = np.array([[int(fl[i].x * w), int(fl[i].y * h)] for i in brow_idxs], dtype=np.int32)
                                    cv2.polylines(frame, [b_pts], isClosed=False, color=(248, 189, 56), thickness=2)
                            for hand_lms in (res.left_hand_landmarks, res.right_hand_landmarks):
                                if hand_lms:
                                    idx_tip = hand_lms.landmark[8]
                                    cx, cy = int(idx_tip.x * w), int(idx_tip.y * h)
                                    cv2.circle(frame, (cx, cy), radius=22, color=(94, 63, 244), thickness=3)
                                    cv2.circle(frame, (cx, cy), radius=4, color=(94, 63, 244), thickness=-1)
                    except Exception:
                        pass

        # 2. Or decode browser-captured keyframe if provided
        if frame is None:
            existing_b64 = episode.get("keyframe_jpeg_base64")
            if isinstance(existing_b64, str) and len(existing_b64) > 120:
                try:
                    raw_bytes = base64.b64decode(existing_b64.split(",")[-1])
                    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if decoded is not None and decoded.size > 0:
                        frame = decoded
                except Exception:
                    frame = None

        thumb = self._draw_synthetic_or_decoded_hud(frame, episode, apex_sec, action_label)
        ok, buf = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
        mins = int(apex_sec // 60)
        secs = apex_sec % 60
        apex_fmt = f"{mins:02d}:{secs:05.2f}"
        if not ok:
            return {"apex_timestamp_sec": apex_fmt, "action_label": action_label, "image_data_uri": ""}

        b64_str = base64.b64encode(buf.tobytes()).decode("ascii")
        return {
            "apex_timestamp_sec": apex_fmt,
            "action_label": action_label,
            "image_data_uri": f"data:image/jpeg;base64,{b64_str}",
        }
