from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class GeometryV2Readiness:
    protocol_version: str
    ready: bool
    runtime_fingerprint: Mapping[str, Any]
    policy: Mapping[str, Any]
    weights: Mapping[str, Any]
    failure_reason: Optional[str]


@dataclass(frozen=True)
class GeometryV2InferenceResponse:
    protocol_version: str
    request_id: Optional[str]
    success: bool
    frame_idx: Optional[int]
    selected_variant: Optional[str]
    selection_status: str
    homography_image_to_pitch: Optional[list[list[float]]]
    homography_pitch_to_image: Optional[list[list[float]]]
    keypoint_count: Optional[int]
    line_count: Optional[int]
    reference_free: Optional[bool]
    input_echo: Optional[Mapping[str, Any]]
    runtime_fingerprint: Mapping[str, Any]
    failure_reason: Optional[str]
