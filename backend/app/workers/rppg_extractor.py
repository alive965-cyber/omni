"""
backend/app/workers/rppg_extractor.py
Fixes Bug #3: Extracts RGB signals at native video FPS (25-30 FPS) across the full video
and runs POS + CHROM over an 8.5-second (256-frame) sliding buffer stepped every 2.0 seconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Sequence
import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.interpolate import CubicSpline

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]

try:
    import mediapipe as mp
except ImportError:
    mp = None  # type: ignore[assignment]

FOREHEAD_LANDMARKS = [10, 67, 69, 104, 108, 151, 337, 333, 299, 297]
LEFT_CHEEK_LANDMARKS = [116, 117, 118, 100, 126, 209]
RIGHT_CHEEK_LANDMARKS = [345, 346, 347, 329, 355, 429]

FOREHEAD_ROI_INDICES = tuple(FOREHEAD_LANDMARKS)
LEFT_CHEEK_ROI_INDICES = tuple(LEFT_CHEEK_LANDMARKS)
RIGHT_CHEEK_ROI_INDICES = tuple(RIGHT_CHEEK_LANDMARKS)


@dataclass
class RPPGResult:
    hr_bpm: float | None
    sqi: float | None
    status: str
    hr_pos_bpm: float | None = None
    hr_chrom_bpm: float | None = None
    algorithm_discrepancy_bpm: float | None = None
    confidence_tier: str = "UNRELIABLE"
    bvp_waveform: list[float] = field(default_factory=list)


def unavailable(reason: str = "INSUFFICIENT_SIGNAL") -> RPPGResult:
    return RPPGResult(
        hr_bpm=None,
        sqi=0.0,
        status=reason,
        hr_pos_bpm=None,
        hr_chrom_bpm=None,
        algorithm_discrepancy_bpm=None,
        confidence_tier="UNRELIABLE",
        bvp_waveform=[],
    )


class RollingPosChromRppgExtractor:
    def __init__(self, target_fs: float = 30.0, buffer_seconds: float = 8.5):
        self.target_fs = target_fs
        self.buffer_seconds = buffer_seconds
        self.mp_face_mesh = getattr(mp, "solutions", None).face_mesh if (mp is not None and hasattr(mp, "solutions")) else None

    def extract_session_rppg_windows(
        self, video_path: str, window_size_sec: float = 2.0
    ) -> Dict[int, Dict[str, Any]]:
        """
        Pass 1: Extracts frame-by-frame spatially averaged Forehead+Cheek RGB at native FPS.
        Pass 2: Slides an 8.5s window stepped every 2.0s to compute POS/CHROM BPM & SQI.
        """
        if cv2 is None or self.mp_face_mesh is None:
            return {}
        cap = cv2.VideoCapture(video_path)
        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        timestamps: List[float] = []
        rgb_traces: List[np.ndarray] = []

        with self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                t_sec = frame_idx / native_fps
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = face_mesh.process(rgb_frame)

                if res.multi_face_landmarks:
                    h, w, _ = rgb_frame.shape
                    lm = res.multi_face_landmarks[0].landmark
                    roi_means = []
                    for poly_indices in (FOREHEAD_LANDMARKS, LEFT_CHEEK_LANDMARKS, RIGHT_CHEEK_LANDMARKS):
                        pts = np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in poly_indices], dtype=np.int32)
                        mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.fillConvexPoly(mask, pts, 255)
                        pixels = rgb_frame[mask == 255]
                        if len(pixels) > 20:
                            # Filter out specular glare (R,G,B > 245) and deep shadow (< 25)
                            valid_px = pixels[(pixels.max(axis=1) < 245) & (pixels.min(axis=1) > 25)]
                            if len(valid_px) > 10:
                                roi_means.append(valid_px.mean(axis=0))
                    if roi_means:
                        timestamps.append(t_sec)
                        rgb_traces.append(np.mean(roi_means, axis=0))
                frame_idx += 1

        cap.release()
        if len(timestamps) < int(self.target_fs * 5):
            return {}

        t_arr = np.array(timestamps, dtype=float)
        rgb_arr = np.vstack(rgb_traces)
        total_dur = float(t_arr[-1])
        num_windows = int(total_dur // window_size_sec) + 1
        results: Dict[int, Dict[str, Any]] = {}

        for w_idx in range(num_windows):
            w_end = (w_idx + 1) * window_size_sec
            # Use trailing 8.5s buffer (or initial 8.5s for early windows)
            buf_end = max(self.buffer_seconds, min(w_end, total_dur))
            buf_start = max(0.0, buf_end - self.buffer_seconds)

            mask = (t_arr >= buf_start) & (t_arr <= buf_end)
            if np.sum(mask) < int(self.target_fs * 4.5):
                results[w_idx] = {
                    "hr_bpm_pos": None,
                    "hr_bpm_chrom": None,
                    "rppg_sqi": 0.0,
                    "rppg_status": "UNRELIABLE_INSUFFICIENT_FRAMES",
                }
                continue

            results[w_idx] = self._compute_pos_chrom_on_buffer(t_arr[mask], rgb_arr[mask])

        return results

    def _compute_pos_chrom_on_buffer(self, t_slice: np.ndarray, rgb_slice: np.ndarray) -> Dict[str, Any]:
        # Deduplicate strictly increasing timestamps for CubicSpline
        t_clean = np.asarray(t_slice, dtype=float).reshape(-1)
        rgb_clean = np.asarray(rgb_slice, dtype=float)
        if len(t_clean) < 32 or rgb_clean.ndim != 2 or rgb_clean.shape[1] != 3:
            return {
                "hr_bpm_pos": None,
                "hr_bpm_chrom": None,
                "rppg_sqi": 0.0,
                "rppg_status": "UNRELIABLE_INSUFFICIENT_FRAMES",
            }
        unique_t, unique_idx = np.unique(t_clean, return_index=True)
        if len(unique_t) < 32 or float(unique_t[-1] - unique_t[0]) < 3.5:
            return {
                "hr_bpm_pos": None,
                "hr_bpm_chrom": None,
                "rppg_sqi": 0.0,
                "rppg_status": "UNRELIABLE_INSUFFICIENT_FRAMES",
            }
        t_slice = unique_t
        rgb_slice = rgb_clean[unique_idx]

        # Resample onto uniform 30.0 Hz grid (256 samples across ~8.5s)
        n_samples = 256
        uniform_t = np.linspace(t_slice[0], t_slice[-1], n_samples)
        cs = CubicSpline(t_slice, rgb_slice, axis=0)
        rgb_u = cs(uniform_t)
        fs = n_samples / max(1.0, float(t_slice[-1] - t_slice[0]))

        # POS projection with overlap-add (window l = 48 samples)
        bvp_pos = np.zeros(n_samples)
        sub_l = 48
        for s in range(0, n_samples - sub_l + 1, 2):
            seg = rgb_u[s : s + sub_l]
            c_norm = seg / (np.mean(seg, axis=0, keepdims=True) + 1e-6) - 1.0
            s1 = c_norm[:, 1] - c_norm[:, 2]
            s2 = c_norm[:, 1] + c_norm[:, 2] - 2.0 * c_norm[:, 0]
            alpha = np.std(s1) / (np.std(s2) + 1e-6)
            h = s1 + alpha * s2
            bvp_pos[s : s + sub_l] += h - np.mean(h)

        # CHROM projection
        c_full = rgb_u / (np.mean(rgb_u, axis=0, keepdims=True) + 1e-6) - 1.0
        xs = 3.0 * c_full[:, 0] - 2.0 * c_full[:, 1]
        ys = 1.5 * c_full[:, 0] + c_full[:, 1] - 1.5 * c_full[:, 2]
        bvp_chrom = xs - (np.std(xs) / (np.std(ys) + 1e-6)) * ys

        nyq = 0.5 * fs
        if nyq <= 3.6:
            return {"hr_bpm_pos": None, "hr_bpm_chrom": None, "rppg_sqi": 0.0, "rppg_status": "UNRELIABLE_LOW_FPS"}

        b, a = butter(3, [0.75 / nyq, min(3.20 / nyq, 0.98)], btype="band")
        pos_filt = filtfilt(b, a, bvp_pos)
        chrom_filt = filtfilt(b, a, bvp_chrom)

        freqs, psd_pos = welch(pos_filt, fs=fs, nperseg=min(192, n_samples), nfft=2048)
        _, psd_chrom = welch(chrom_filt, fs=fs, nperseg=min(192, n_samples), nfft=2048)

        band_mask = (freqs >= 0.75) & (freqs <= 3.20)
        f_band = freqs[band_mask]
        p_pos = psd_pos[band_mask]
        p_chrom = psd_chrom[band_mask]

        f_peak_pos = float(f_band[np.argmax(p_pos)])
        f_peak_chrom = float(f_band[np.argmax(p_chrom)])
        bpm_pos = round(f_peak_pos * 60.0, 1)
        bpm_chrom = round(f_peak_chrom * 60.0, 1)

        # Spectral Purity SQI around fundamental + 1st harmonic
        harm_mask = (np.abs(f_band - f_peak_pos) <= 0.14) | (np.abs(f_band - 2.0 * f_peak_pos) <= 0.14)
        sqi = float(np.sum(p_pos[harm_mask]) / (np.sum(p_pos) + 1e-8))

        wf_std = float(np.std(pos_filt))
        wf_norm = (
            ((pos_filt[-min(len(pos_filt), 120) :] - float(np.mean(pos_filt))) / wf_std).round(3).tolist()
            if wf_std > 1e-9
            else []
        )

        if sqi >= 0.36 and abs(bpm_pos - bpm_chrom) <= 14.0:
            return {
                "hr_bpm_pos": bpm_pos,
                "hr_bpm_chrom": bpm_chrom,
                "rppg_sqi": round(sqi, 3),
                "rppg_status": "VALID",
                "bvp_waveform": wf_norm,
            }
        return {
            "hr_bpm_pos": None,
            "hr_bpm_chrom": None,
            "rppg_sqi": round(sqi, 3),
            "rppg_status": "UNRELIABLE_MOTION_OR_LIGHTING",
            "bvp_waveform": wf_norm,
        }


def extract_multi_region_skin_rgb(
    frame_bgr,
    landmarks_norm: Sequence[tuple[float, float] | tuple[float, float, float]],
) -> tuple[float, float, float] | None:
    """Extract spatially averaged RGB from Forehead + Left/Right Cheek ROIs with glare/shadow rejection."""
    if cv2 is None or frame_bgr is None or len(landmarks_norm) < 430:
        return None
    height, width = frame_bgr.shape[:2]
    if height < 16 or width < 16:
        return None
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    roi_means = []
    for poly_indices in (FOREHEAD_LANDMARKS, LEFT_CHEEK_LANDMARKS, RIGHT_CHEEK_LANDMARKS):
        pts = np.array(
            [
                [
                    int(round(float(landmarks_norm[i][0]) * width)),
                    int(round(float(landmarks_norm[i][1]) * height)),
                ]
                for i in poly_indices
                if i < len(landmarks_norm)
            ],
            dtype=np.int32,
        )
        if len(pts) >= 3:
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillConvexPoly(mask, pts, 255)
            pixels = rgb_frame[mask == 255]
            if len(pixels) > 20:
                valid_px = pixels[(pixels.max(axis=1) < 245) & (pixels.min(axis=1) > 25)]
                if len(valid_px) > 10:
                    roi_means.append(valid_px.mean(axis=0))
    if not roi_means:
        return None
    mean_rgb = np.mean(roi_means, axis=0)
    return float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])


def resample_rgb_uniform(
    rgb_samples,
    timestamps_sec: Sequence[float] | None = None,
    *,
    target_fs: float = 30.0,
    target_samples: int = 256,
):
    """Resample C(t) via CubicSpline onto a uniform 30.0 Hz 256-sample grid."""
    x = np.asarray(rgb_samples, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or len(x) < 32 or not np.isfinite(x).all():
        return None
    if timestamps_sec is None:
        return x, float(target_fs)
    t = np.asarray(timestamps_sec, dtype=float).reshape(-1)
    if len(t) != len(x) or not np.isfinite(t).all():
        return None
    unique_t, unique_idx = np.unique(t, return_index=True)
    if len(unique_t) < 32:
        return None
    x_u = x[unique_idx]
    duration = float(unique_t[-1] - unique_t[0])
    if duration < 3.5:
        return None
    uniform_t = np.linspace(float(unique_t[0]), float(unique_t[-1]), target_samples)
    resampled = CubicSpline(unique_t, x_u, axis=0)(uniform_t)
    effective_fs = target_samples / max(1.0, duration)
    return resampled, float(effective_fs)


def estimate_from_rgb(
    rgb_samples,
    *,
    timestamps_sec: Sequence[float] | None = None,
    sample_rate: float = 30.0,
    head_velocity_deg_s: float = 0.0,
    illumination_change_pct: float = 0.0,
    sqi_floor: float = 0.36,
) -> RPPGResult:
    """Estimate heart rate from a rolling 8.5s (256-frame) RGB buffer using RollingPosChromRppgExtractor."""
    x = np.asarray(rgb_samples, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or len(x) < 64 or not np.isfinite(x).all():
        return unavailable("UNRELIABLE_INSUFFICIENT_FRAMES")

    if abs(head_velocity_deg_s) > 28.0 or abs(illumination_change_pct) > 18.0:
        return unavailable("UNRELIABLE_MOTION_OR_LIGHTING")

    if timestamps_sec is not None:
        t_arr = np.asarray(timestamps_sec, dtype=float).reshape(-1)
        if len(t_arr) != len(x):
            return unavailable("UNRELIABLE_SAMPLING")
    else:
        t_arr = np.arange(len(x), dtype=float) / max(sample_rate, 1.0)

    extractor = RollingPosChromRppgExtractor(target_fs=sample_rate, buffer_seconds=8.5)
    out = extractor._compute_pos_chrom_on_buffer(t_arr, x)
    hr_pos = out.get("hr_bpm_pos")
    hr_chrom = out.get("hr_bpm_chrom")
    sqi_val = float(out.get("rppg_sqi") or 0.0)
    status = str(out.get("rppg_status") or "UNRELIABLE_MOTION_OR_LIGHTING")
    wf = out.get("bvp_waveform") or []

    if status == "VALID" and hr_pos is not None and sqi_val >= sqi_floor:
        disc = round(abs(float(hr_pos) - float(hr_chrom)), 2) if hr_chrom is not None else 0.0
        tier = "HIGH" if (sqi_val >= 0.52 and disc <= 5.0) else "MODERATE"
        return RPPGResult(
            hr_bpm=float(hr_pos),
            sqi=sqi_val,
            status="VALID",
            hr_pos_bpm=float(hr_pos),
            hr_chrom_bpm=float(hr_chrom) if hr_chrom is not None else float(hr_pos),
            algorithm_discrepancy_bpm=disc,
            confidence_tier=tier,
            bvp_waveform=wf,
        )

    return RPPGResult(
        hr_bpm=None,
        sqi=sqi_val,
        status=status,
        hr_pos_bpm=hr_pos,
        hr_chrom_bpm=hr_chrom,
        algorithm_discrepancy_bpm=None,
        confidence_tier="UNRELIABLE",
        bvp_waveform=wf,
    )
