"""Exhaustive Computer Vision, FACS Action Units, Ocular Blink/Gaze, Head Pose, and Kinesics
mathematical primitives (PDF Specification Sections 4.1 - 4.5).
"""
from __future__ import annotations

from math import atan2, cos, degrees, hypot, isfinite, radians, sqrt
from statistics import median
from typing import Any, Mapping, Sequence

LEFT_EYE_EAR_INDICES = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_EAR_INDICES = (362, 385, 387, 263, 373, 380)
FINGERTIP_INDICES = (4, 8, 12, 16, 20)

# Canonical 3D generic facial model points (in mm) for 6-point Perspective-n-Point (solvePnP)
# Landmarks: [1 (Nose Tip), 152 (Chin), 33 (Left Eye Outer), 263 (Right Eye Outer), 61 (Left Mouth), 291 (Right Mouth)]
CANONICAL_FACE_3D_POINTS = (
    (0.0, 0.0, 0.0),          # #1 Nose tip
    (0.0, -330.0, -65.0),     # #152 Chin menton
    (-225.0, 170.0, -135.0),  # #33 Left eye outer canthus
    (225.0, 170.0, -135.0),   # #263 Right eye outer canthus
    (-150.0, -150.0, -125.0), # #61 Left mouth corner
    (150.0, -150.0, -125.0),  # #291 Right mouth corner
)
PNP_LANDMARK_INDICES = (1, 152, 33, 263, 61, 291)


def _dist2d(p1: Sequence[float], p2: Sequence[float]) -> float:
    return hypot(float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1]))


def _dist3d(p1: Sequence[float], p2: Sequence[float]) -> float:
    dz = (float(p1[2]) - float(p2[2])) if (len(p1) > 2 and len(p2) > 2) else 0.0
    return sqrt((float(p1[0]) - float(p2[0])) ** 2 + (float(p1[1]) - float(p2[1])) ** 2 + dz ** 2)


def facial_scale_normalizers(points: Mapping[int, Sequence[float]]) -> dict[str, float] | None:
    """Section 4.1.1: Compute Inter-Ocular Distance (D_IOD) and Facial Height (D_FH)."""
    if any(idx not in points for idx in (33, 263, 10, 152)):
        return None
    d_iod = _dist3d(points[33], points[263])
    d_fh = _dist3d(points[10], points[152])
    if d_iod <= 1e-6 or d_fh <= 1e-6:
        return None
    return {"d_iod": d_iod, "d_fh": d_fh}


def estimate_head_pose_euler(
    points: Mapping[int, Sequence[float]],
    frame_width: float = 1280.0,
    frame_height: float = 720.0,
) -> dict[str, float | bool] | None:
    """Section 4.1.2: 3D Head Pose Estimation (Pitch, Yaw, Roll) + Reliability Gate.

    Uses OpenCV solvePnP + Rodrigues when cv2 is available, with an exact geometric
    trigonometric fallback when running in headless environments without OpenCV.
    Flags facial geometry confidence as degraded (C_face = 0.4) when |yaw| > 35 deg or |pitch| > 25 deg.
    """
    if any(idx not in points for idx in PNP_LANDMARK_INDICES):
        return None

    pitch_deg: float | None = None
    yaw_deg: float | None = None
    roll_deg: float | None = None

    try:
        import cv2
        import numpy as np

        model_points = np.array(CANONICAL_FACE_3D_POINTS, dtype=np.float64)
        image_points = np.array(
            [
                [
                    float(points[idx][0]) * (frame_width if float(points[idx][0]) <= 2.0 else 1.0),
                    float(points[idx][1]) * (frame_height if float(points[idx][1]) <= 2.0 else 1.0),
                ]
                for idx in PNP_LANDMARK_INDICES
            ],
            dtype=np.float64,
        )
        focal_length = float(frame_width)
        center = (frame_width / 2.0, frame_height / 2.0)
        camera_matrix = np.array(
            [[focal_length, 0.0, center[0]], [0.0, focal_length, center[1]], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        success, rvec, _ = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if success:
            rmat, _ = cv2.Rodrigues(rvec)
            pitch_deg = degrees(atan2(-float(rmat[2, 0]), sqrt(float(rmat[2, 1]) ** 2 + float(rmat[2, 2]) ** 2)))
            yaw_deg = degrees(atan2(float(rmat[1, 0]), float(rmat[0, 0])))
            roll_deg = degrees(atan2(float(rmat[2, 1]), float(rmat[2, 2])))
    except Exception:
        pass

    if pitch_deg is None or yaw_deg is None or roll_deg is None:
        # Geometric 3D landmark fallback
        left_eye, right_eye = points[33], points[263]
        nose, chin = points[1], points[152]
        eye_dx = float(right_eye[0]) - float(left_eye[0])
        eye_dy = float(right_eye[1]) - float(left_eye[1])
        roll_deg = degrees(atan2(eye_dy, eye_dx if abs(eye_dx) > 1e-6 else 1e-6))
        eye_mid_x = 0.5 * (float(left_eye[0]) + float(right_eye[0]))
        eye_mid_y = 0.5 * (float(left_eye[1]) + float(right_eye[1]))
        iod = max(_dist2d(left_eye, right_eye), 1e-6)
        yaw_deg = max(-75.0, min(75.0, ((float(nose[0]) - eye_mid_x) / iod) * 90.0))
        face_v = max(abs(float(chin[1]) - eye_mid_y), 1e-6)
        pitch_deg = max(-60.0, min(60.0, (((float(nose[1]) - eye_mid_y) / face_v) - 0.36) * 110.0))

    degraded = abs(yaw_deg) > 35.0 or abs(pitch_deg) > 25.0
    return {
        "head_pitch_deg": round(float(pitch_deg), 3),
        "head_yaw_deg": round(float(yaw_deg), 3),
        "head_roll_deg": round(float(roll_deg), 3),
        "face_confidence": 0.4 if degraded else 0.98,
        "geometry_degraded": degraded,
    }


def _single_eye_ear(points: Mapping[int, Sequence[float]], indices: tuple[int, int, int, int, int, int]) -> float | None:
    if any(i not in points for i in indices):
        return None
    p = [points[i] for i in indices]
    width = 2.0 * _dist2d(p[0], p[3])
    if width <= 1e-9:
        return None
    return (_dist2d(p[1], p[5]) + _dist2d(p[2], p[4])) / width


def eye_aspect_ratio(
    points: Mapping[int, Sequence[float]],
    head_pitch_deg: float = 0.0,
) -> float | None:
    """Section 4.2: Soukupova & Cech (2016) Eye Aspect Ratio with bilateral averaging and cos(pitch) compensation."""
    left_ear = _single_eye_ear(points, LEFT_EYE_EAR_INDICES)
    right_ear = _single_eye_ear(points, RIGHT_EYE_EAR_INDICES)
    if left_ear is None and right_ear is None:
        return None
    raw_ear = (left_ear + right_ear) / 2.0 if (left_ear is not None and right_ear is not None) else (left_ear or right_ear)
    pitch_comp = max(0.5, abs(cos(radians(float(head_pitch_deg)))))
    return float(raw_ear) * pitch_comp


def compute_adaptive_blink_threshold(baseline_ear_samples: Sequence[float | None]) -> float:
    """Section 4.2: Adaptive Blink Detection Threshold tau_blink = min(0.21, 0.72 * median_open_EAR)."""
    clean = sorted(float(x) for x in baseline_ear_samples if x is not None and isfinite(float(x)) and float(x) > 0.05)
    if not clean:
        return 0.21
    cutoff = max(1, int(len(clean) * 0.10))
    upper_90 = clean[cutoff:] if len(clean) > cutoff else clean
    neutral_median = median(upper_90)
    return round(min(0.21, 0.72 * float(neutral_median)), 4)


def detect_blinks_with_hysteresis(
    ear_series: Sequence[float | None],
    *,
    fps: float = 30.0,
    threshold: float = 0.21,
) -> dict[str, float | int]:
    """Section 4.2: Hysteresis & Duration Validation.

    Valid blink requires EAR(t) < tau_blink for N_frames in [2, 12] at 30 FPS (66 ms to 400 ms).
    Closures exceeding 400 ms are classified separately as Prolonged Eye Closure / Visual Avoidance.
    """
    if not ear_series or fps <= 0:
        return {
            "blink_count": 0,
            "blink_rate_per_min": 0.0,
            "blink_duration_ms_mean": 0.0,
            "prolonged_closure_count": 0,
        }
    min_frames = max(1, int(round(0.066 * fps)))
    max_frames = max(min_frames + 1, int(round(0.400 * fps)))

    valid_durations_ms: list[float] = []
    prolonged_closures = 0
    current_run = 0

    for val in list(ear_series) + [1.0]:
        if val is not None and isfinite(float(val)) and float(val) < threshold:
            current_run += 1
        else:
            if current_run > 0:
                if min_frames <= current_run <= max_frames:
                    valid_durations_ms.append((current_run / fps) * 1000.0)
                elif current_run > max_frames:
                    prolonged_closures += 1
                current_run = 0

    total_duration_sec = max(len(ear_series) / fps, 1e-3)
    blink_rate = (len(valid_durations_ms) / total_duration_sec) * 60.0
    mean_duration = sum(valid_durations_ms) / len(valid_durations_ms) if valid_durations_ms else 0.0
    return {
        "blink_count": len(valid_durations_ms),
        "blink_rate_per_min": round(blink_rate, 2),
        "blink_duration_ms_mean": round(mean_duration, 2),
        "prolonged_closure_count": prolonged_closures,
    }


def compute_iris_gaze(points: Mapping[int, Sequence[float]]) -> tuple[float, float] | None:
    """Section 4.3.1: Horizontal & Vertical Gaze Ratio (Gx, Gy) from MediaPipe Iris landmarks (468, 473)."""
    ratios_x: list[float] = []
    ratios_y: list[float] = []
    if all(idx in points for idx in (468, 33, 133, 159, 145)):
        denom_x = abs(float(points[133][0]) - float(points[33][0]))
        denom_y = abs(float(points[145][1]) - float(points[159][1]))
        if denom_x > 1e-6 and denom_y > 1e-6:
            ratios_x.append((float(points[468][0]) - float(points[33][0])) / denom_x)
            ratios_y.append((float(points[468][1]) - float(points[159][1])) / denom_y)
    if all(idx in points for idx in (473, 362, 263, 386, 374)):
        denom_x = abs(float(points[263][0]) - float(points[362][0]))
        denom_y = abs(float(points[374][1]) - float(points[386][1]))
        if denom_x > 1e-6 and denom_y > 1e-6:
            ratios_x.append((float(points[473][0]) - float(points[362][0])) / denom_x)
            ratios_y.append((float(points[473][1]) - float(points[386][1])) / denom_y)
    if not ratios_x or not ratios_y:
        return None
    gx = max(0.0, min(1.0, sum(ratios_x) / len(ratios_x)))
    gy = max(0.0, min(1.0, sum(ratios_y) / len(ratios_y)))
    return round(gx, 4), round(gy, 4)


def gaze_aversion_ratio(gaze_points: list[tuple[float, float] | None]) -> float | None:
    """Section 4.3.2: Direct Gaze Zone |Gx - 0.5| <= 0.14 and |Gy - 0.5| <= 0.16."""
    valid = [p for p in gaze_points if p is not None]
    if not valid:
        return None
    aversions = sum(abs(x - 0.5) > 0.14 or abs(y - 0.5) > 0.16 for x, y in valid)
    return aversions / len(valid)


def saccade_velocity_deg_s(
    gaze_points: Sequence[tuple[float, float] | None],
    fps: float = 30.0,
) -> float | None:
    """Section 4.3.2: Frame-to-frame gaze displacement ||g(t) - g(t-1)||_2 * FPS."""
    valid = [p for p in gaze_points if p is not None]
    if len(valid) < 2 or fps <= 0:
        return None
    displacements = [
        hypot(float(curr[0]) - float(prev[0]), float(curr[1]) - float(prev[1])) * fps
        for prev, curr in zip(valid, valid[1:])
    ]
    return round(sum(displacements) / len(displacements), 3)


def compute_facs_action_units(
    blendshapes: Mapping[str, float] | None = None,
    landmarks: Mapping[int, Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Section 4.4: Compute 9 Core FACS Action Units (AU01..AU26) + Facial Asymmetry Index (FAI)."""
    bs = dict(blendshapes or {})
    au01 = float(bs.get("browInnerUp", 0.0))
    au02 = 0.5 * (float(bs.get("browOuterUpLeft", 0.0)) + float(bs.get("browOuterUpRight", 0.0)))
    au04 = 0.5 * (float(bs.get("browDownLeft", 0.0)) + float(bs.get("browDownRight", 0.0)))
    au06 = 0.5 * (float(bs.get("cheekSquintLeft", 0.0)) + float(bs.get("cheekSquintRight", 0.0)))
    au12 = 0.5 * (float(bs.get("mouthSmileLeft", 0.0)) + float(bs.get("mouthSmileRight", 0.0)))
    au15 = 0.5 * (float(bs.get("mouthFrownLeft", 0.0)) + float(bs.get("mouthFrownRight", 0.0)))
    au20 = 0.5 * (float(bs.get("mouthStretchLeft", 0.0)) + float(bs.get("mouthStretchRight", 0.0)))
    au24 = 0.5 * (float(bs.get("mouthPressLeft", 0.0)) + float(bs.get("mouthPressRight", 0.0)))
    au25 = float(bs.get("jawOpen", 0.0))

    if landmarks and 13 in landmarks and 14 in landmarks and 10 in landmarks and 152 in landmarks:
        d_fh = max(_dist3d(landmarks[10], landmarks[152]), 1e-6)
        jaw_ratio = min(1.0, (_dist3d(landmarks[13], landmarks[14]) / d_fh) * 3.5)
        au25 = max(au25, jaw_ratio)

    asym_pairs = (
        ("mouthSmileLeft", "mouthSmileRight"),
        ("browDownLeft", "browDownRight"),
        ("cheekSquintLeft", "cheekSquintRight"),
        ("mouthPressLeft", "mouthPressRight"),
    )
    fai = 0.25 * sum(abs(float(bs.get(left, 0.0)) - float(bs.get(right, 0.0))) for left, right in asym_pairs)

    smile_type = "NONE"
    if au12 > 0.50 and au06 < 0.15:
        smile_type = "Non_Duchenne_Social_Smile"
    elif au12 > 0.45 and au06 >= 0.25:
        smile_type = "Duchenne_Concordant_Smile"

    return {
        "au01_inner_brow": round(min(1.0, max(0.0, au01)), 4),
        "au02_outer_brow": round(min(1.0, max(0.0, au02)), 4),
        "au04_brow_lowerer": round(min(1.0, max(0.0, au04)), 4),
        "au06_cheek_raiser": round(min(1.0, max(0.0, au06)), 4),
        "au12_lip_corner_puller": round(min(1.0, max(0.0, au12)), 4),
        "au15_lip_corner_depressor": round(min(1.0, max(0.0, au15)), 4),
        "au20_lip_stretcher": round(min(1.0, max(0.0, au20)), 4),
        "au24_lip_presser": round(min(1.0, max(0.0, au24)), 4),
        "au25_jaw_drop": round(min(1.0, max(0.0, au25)), 4),
        "facial_asymmetry_index": round(min(1.0, max(0.0, fai)), 4),
        "smile_classification": smile_type,
    }


def compute_kinesics_and_posture(
    pose_landmarks: Mapping[int, Sequence[float]] | None,
    left_hand_landmarks: Mapping[int, Sequence[float]] | None = None,
    right_hand_landmarks: Mapping[int, Sequence[float]] | None = None,
    face_landmarks: Mapping[int, Sequence[float]] | None = None,
    prev_wrists: Mapping[str, Sequence[float]] | None = None,
    dt_sec: float = 1.0 / 30.0,
    speech_activity_ratio: float = 0.0,
    baseline_shoulder_z: float | None = None,
) -> dict[str, Any]:
    """Section 4.5: Shoulder-normalized Self-Adaptors, Hand Kinetic Energy, Gesture Disambiguation, and Postural Lean."""
    if not pose_landmarks or 11 not in pose_landmarks or 12 not in pose_landmarks:
        return {
            "self_touch_detected": False,
            "d_adaptor_norm": None,
            "hand_kinetic_energy": 0.0,
            "gesture_type": "NONE",
            "postural_lean_z": 0.0,
            "shoulder_slouch_deg": 0.0,
        }

    p11, p12 = pose_landmarks[11], pose_landmarks[12]
    d_sw = max(_dist3d(p11, p12), 1e-5)
    p_neck = (
        0.5 * (float(p11[0]) + float(p12[0])),
        0.5 * (float(p11[1]) + float(p12[1])),
        0.5 * ((float(p11[2]) if len(p11) > 2 else 0.0) + (float(p12[2]) if len(p12) > 2 else 0.0)),
    )

    # Target anatomical zones: Nose (#1), Chin (#152), Cheeks (#234, #454), Suprasternale Neck proxy
    zones: list[Sequence[float]] = [p_neck]
    if face_landmarks:
        for z_idx in (1, 152, 234, 454):
            if z_idx in face_landmarks:
                zones.append(face_landmarks[z_idx])

    min_dist = float("inf")
    for hand in (left_hand_landmarks, right_hand_landmarks):
        if not hand:
            continue
        for tip_idx in FINGERTIP_INDICES:
            if tip_idx in hand:
                for zone_pt in zones:
                    min_dist = min(min_dist, _dist3d(hand[tip_idx], zone_pt) / d_sw)

    self_touch = min_dist < 0.22

    # Hand Kinetic Energy KE_hands(t)
    ke_hands = 0.0
    curr_wrists: dict[str, Sequence[float]] = {}
    if left_hand_landmarks and 0 in left_hand_landmarks:
        curr_wrists["left"] = left_hand_landmarks[0]
    elif 15 in pose_landmarks:
        curr_wrists["left"] = pose_landmarks[15]
    if right_hand_landmarks and 0 in right_hand_landmarks:
        curr_wrists["right"] = right_hand_landmarks[0]
    elif 16 in pose_landmarks:
        curr_wrists["right"] = pose_landmarks[16]

    if prev_wrists and dt_sec > 1e-5:
        vel_sq_sum = 0.0
        for side, w_pt in curr_wrists.items():
            if side in prev_wrists:
                v = _dist3d(w_pt, prev_wrists[side]) / dt_sec
                vel_sq_sum += v ** 2
        ke_hands = vel_sq_sum / (2.0 * (d_sw ** 2))

    hand_to_hand = False
    if "left" in curr_wrists and "right" in curr_wrists:
        hand_to_hand = (_dist3d(curr_wrists["left"], curr_wrists["right"]) / d_sw) < 0.18

    if self_touch:
        gesture_type = "SELF_TOUCH"
    elif ke_hands > 0.12 and speech_activity_ratio > 0.40 and not hand_to_hand:
        gesture_type = "ILLUSTRATIVE"
    elif (ke_hands > 0.08 and speech_activity_ratio <= 0.40) or (ke_hands > 0.04 and hand_to_hand):
        gesture_type = "FIDGET_ADAPTOR"
    else:
        gesture_type = "NONE"

    mean_z = 0.5 * ((float(p11[2]) if len(p11) > 2 else 0.0) + (float(p12[2]) if len(p12) > 2 else 0.0))
    lean_z = ((baseline_shoulder_z - mean_z) / d_sw) if baseline_shoulder_z is not None else 0.0
    slouch_deg = degrees(atan2(float(p12[1]) - float(p11[1]), float(p12[0]) - float(p11[0]) + 1e-6))

    return {
        "self_touch_detected": self_touch,
        "d_adaptor_norm": round(min_dist, 4) if min_dist != float("inf") else None,
        "hand_kinetic_energy": round(ke_hands, 4),
        "gesture_type": gesture_type,
        "postural_lean_z": round(lean_z, 4),
        "shoulder_slouch_deg": round(slouch_deg, 2),
    }
