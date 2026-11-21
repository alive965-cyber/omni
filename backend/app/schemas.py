"""Pydantic Request & Response Schemas for OmniPulse MBS v1.0 + v2.0 Cognitive/Dyadic Telemetry.
Supports both Pydantic v2 and Pydantic v1 seamlessly.
"""
from __future__ import annotations

from math import isfinite
from typing import Any

import pydantic
from pydantic import BaseModel as _PydanticBaseModel, Field

_PYDANTIC_V2 = int(getattr(pydantic, "__version__", "1").split(".")[0]) >= 2

if _PYDANTIC_V2:
    from pydantic import field_validator
else:
    from pydantic import validator as _v1_validator


class CompatBaseModel(_PydanticBaseModel):
    """Base model providing Pydantic v2 methods (`model_dump`, `model_validate`, `model_validate_json`, `model_copy`)
    even when running on a Pydantic v1 installation.
    """

    if not _PYDANTIC_V2:
        def model_dump(self, *args, **kwargs) -> dict[str, Any]:
            return self.dict(*args, **kwargs)

        def model_copy(self, *, update: dict[str, Any] | None = None, deep: bool = False):
            return self.copy(update=update, deep=deep)

        @classmethod
        def model_validate(cls, obj: Any):
            return cls.parse_obj(obj)

        @classmethod
        def model_validate_json(cls, json_data: str | bytes):
            return cls.parse_raw(json_data)


class SessionCreate(CompatBaseModel):
    subject_identifier: str = Field(min_length=1, max_length=128)
    session_title: str = Field(default="Untitled session", min_length=1, max_length=255)
    consent_verified: bool
    ambient_lux_score: float | None = None
    mic_snr_db: float | None = None


class WebRTCOffer(CompatBaseModel):
    sdp: str = Field(min_length=1)
    type: str = Field(default="offer")


def _validate_finite_baseline_samples(features: dict[str, list[float | None]]):
    for name, values in features.items():
        if not name or len(name) > 64:
            raise ValueError("Baseline feature names must contain 1 to 64 characters.")
        if any(value is not None and not isfinite(value) for value in values):
            raise ValueError("Baseline samples must be finite numbers or null.")
    return features


class CalibrationComplete(CompatBaseModel):
    face_tracking_uptime_pct: float = Field(ge=0, le=100)
    duration_sec: float = Field(ge=60, le=120)
    baseline_features: dict[str, list[float | None]] = Field(default_factory=dict)

    if _PYDANTIC_V2:
        @field_validator("baseline_features")
        @classmethod
        def finite_baseline_samples(cls, features):
            return _validate_finite_baseline_samples(features)
    else:
        @_v1_validator("baseline_features")
        def finite_baseline_samples(cls, features):
            return _validate_finite_baseline_samples(features)


def _validate_finite_measurements(values: dict[str, Any]):
    for value in values.values():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and not isfinite(float(value)):
            raise ValueError("Telemetry measurements must be finite numbers or null.")
    return values


def _validate_finite_confidence(values: dict[str, Any]):
    def check(value):
        if isinstance(value, dict):
            for nested in value.values():
                check(nested)
        elif isinstance(value, list):
            for nested in value:
                check(nested)
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and not isfinite(float(value)):
            raise ValueError("Confidence values must be finite numbers or null.")

    check(values)
    return values


class TelemetryWindow(CompatBaseModel):
    window_index: int = Field(ge=0)
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    confidence: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float | int | str | None] = Field(default_factory=dict)
    baseline_z_scores: dict[str, float | None] = Field(default_factory=dict)
    dual_horizon_z_scores: dict[str, float | None] = Field(default_factory=dict)
    phasic_delta_z_json: dict[str, float | None] = Field(default_factory=dict)
    phasic_attribution_state: str | None = None
    composite_indices: dict[str, float | None] = Field(default_factory=dict)
    behavioral_flags: list[str] = Field(default_factory=list)
    transcript_segment: str | None = None
    transcript_segments: list[dict[str, Any]] = Field(default_factory=list)
    slices_100ms: list[dict[str, float]] = Field(default_factory=list)
    speaker_role: str = Field(default="SUBJECT")
    turn_id: int | None = None
    turn_phase: str | None = None
    micro_sequence_tokens: list[str] = Field(default_factory=list)
    micro_sequence_chain: str | None = None
    smile_offset_velocity: float | None = None
    micro_pattern_tag: str | None = None
    vocal_entrainment_aci: float | None = None
    keyframe_jpeg_base64: str | None = None
    gaze_x_trace: list[float] = Field(default_factory=list)
    gaze_y_trace: list[float] = Field(default_factory=list)
    shoulder_y_samples: list[dict[str, float]] = Field(default_factory=list)
    live_copilot_probe: dict[str, Any] | None = None

    if _PYDANTIC_V2:
        @field_validator("end_sec")
        @classmethod
        def validate_window_order(cls, end_sec: float, info):
            start_sec = info.data.get("start_sec")
            if start_sec is not None and end_sec <= start_sec:
                raise ValueError("end_sec must be greater than start_sec")
            return end_sec

        @field_validator("metrics", "baseline_z_scores", "dual_horizon_z_scores", "phasic_delta_z_json", "composite_indices")
        @classmethod
        def finite_measurements(cls, values):
            return _validate_finite_measurements(values)

        @field_validator("confidence")
        @classmethod
        def finite_confidence_values(cls, values):
            return _validate_finite_confidence(values)
    else:
        @_v1_validator("end_sec")
        def validate_window_order(cls, end_sec: float, values):
            start_sec = values.get("start_sec")
            if start_sec is not None and end_sec <= start_sec:
                raise ValueError("end_sec must be greater than start_sec")
            return end_sec

        @_v1_validator("metrics", "baseline_z_scores", "dual_horizon_z_scores", "phasic_delta_z_json", "composite_indices")
        def finite_measurements(cls, values):
            return _validate_finite_measurements(values)

        @_v1_validator("confidence")
        def finite_confidence_values(cls, values):
            return _validate_finite_confidence(values)


class TelemetryIngest(CompatBaseModel):
    window: TelemetryWindow


class RGBSample(CompatBaseModel):
    timestamp_sec: float = Field(ge=0)
    red: float = Field(ge=0, le=255)
    green: float = Field(ge=0, le=255)
    blue: float = Field(ge=0, le=255)


def _validate_ordered_rgb(samples: list[RGBSample]):
    if len(samples) > 512:
        raise ValueError("rgb_samples cannot exceed 512 samples per capture window.")
    if any(current.timestamp_sec <= previous.timestamp_sec for previous, current in zip(samples, samples[1:])):
        raise ValueError("RGB samples must be in strictly increasing time order.")
    return samples


class BrowserCaptureWindow(CompatBaseModel):
    window: TelemetryWindow
    rgb_samples: list[RGBSample] = Field(default_factory=list)
    face_tracking_uptime_pct: float = Field(ge=0, le=100)
    transcribe_audio: bool = True

    if _PYDANTIC_V2:
        @field_validator("rgb_samples")
        @classmethod
        def ordered_rgb_samples(cls, samples):
            return _validate_ordered_rgb(samples)
    else:
        @_v1_validator("rgb_samples")
        def ordered_rgb_samples(cls, samples):
            return _validate_ordered_rgb(samples)
