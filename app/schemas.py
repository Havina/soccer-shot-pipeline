"""
Pydantic schemas for the Soccer ML backend.

These mirror the frozen Semantic-v2 public result without re-implementing any
of its semantics. All values are copied from ``SoccerClipAnalysisV2Result``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class SemanticState(str, Enum):
    COMPLETE = "COMPLETE"
    SHOOTER_CONFIRMATION_REQUIRED = "SHOOTER_CONFIRMATION_REQUIRED"
    FAILED = "FAILED"


class JobState(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    AWAITING_SHOOTER_CONFIRMATION = "AWAITING_SHOOTER_CONFIRMATION"
    CONFIRMING = "CONFIRMING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class IntegrityReport(BaseModel):
    frozen_v1_exact: bool
    frozen_v2_r2_exact: bool
    v2_file_set_exact: bool
    local_geometry_certification_exact: bool
    app_readiness_freeze_sha256: Optional[str] = None
    app_readiness_freeze_ready: bool
    certified_runtime_fingerprint_sha256: Optional[str] = None
    mismatches: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            self.frozen_v1_exact
            and self.frozen_v2_r2_exact
            and self.v2_file_set_exact
            and self.local_geometry_certification_exact
            and self.app_readiness_freeze_ready
        )


class RuntimeProbe(BaseModel):
    performed: bool
    ready: bool = False
    fingerprint_sha256: Optional[str] = None
    fingerprint_matches_certification: bool = False
    gpu_name: Optional[str] = None
    failure_reason: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    can_analyze: bool
    certification_kind: str = "LOCAL_RTX2060_RECERTIFICATION"
    not_a_claim_of: str = "ORIGINAL_NVIDIA_L40_SERVER_CERTIFICATION"
    integrity: IntegrityReport
    runtime: RuntimeProbe
    active_jobs: int


class ShooterCandidate(BaseModel):
    track_id: int
    auto_score: Optional[float] = None
    is_auto_top: bool = False
    contact_frame_idx: Optional[int] = None
    crop_url: Optional[str] = None


class MatchContext(BaseModel):
    """
    User-supplied display labels. Never used for inference: the pipeline only
    knows ATTACKING (the shooter's side) and DEFENDING; these names and colours
    are applied to those labels for presentation.
    """
    attacking_team: Optional[str] = Field(None, max_length=60)
    attacking_color: Optional[str] = Field(None, max_length=20)
    defending_team: Optional[str] = Field(None, max_length=60)
    defending_color: Optional[str] = Field(None, max_length=20)
    notes: Optional[str] = Field(None, max_length=300)


class ConfirmationRecord(BaseModel):
    shooter_track_id: int
    source: str
    at: str
    outcome: Optional[str] = None


class ShooterDecision(BaseModel):
    auto_status: str
    resolution: str
    resolved_track_id: Optional[int]
    candidates: list[ShooterCandidate]
    auto_margin_fraction: Optional[float]
    required_margin_fraction: float
    manual_confirmation_used: bool
    confirmation_track_id: Optional[int]
    confirmation_used_as_model_evidence: bool


class GeometrySummary(BaseModel):
    success: bool
    selected_frame_idx: Optional[int]
    selected_variant: Optional[str]
    validation_status: str
    keypoint_count: Optional[int]
    line_count: Optional[int]
    reference_free: Optional[bool]
    runtime_fingerprint_sha256: Optional[str]
    runtime_profile: Optional[str]


class PublicMetrics(BaseModel):
    shooter_track_id: Optional[int] = None
    shooter_relative_team: Optional[str] = None
    target_goal: Optional[str] = None
    target_goal_confidence: Optional[float] = None
    distance_to_goal_m: Optional[float] = None
    shot_angle_deg: Optional[float] = None
    nearest_defender_m: Optional[float] = None
    defenders_within_5m: Optional[int] = None
    defender_track_ids: list[int] = Field(default_factory=list)
    xg_probability: Optional[float] = None
    xg_percent: Optional[float] = None
    distance_to_left_goal_m: Optional[float] = None
    distance_to_right_goal_m: Optional[float] = None
    geometry_warnings: list[str] = Field(default_factory=list)


class AnalysisSummary(BaseModel):
    semantic_state: SemanticState
    completed_stage_count: int
    completed_stages: list[str]
    blocked_stage: Optional[str]
    failure_stage: Optional[str]
    failure_reason: Optional[str]
    shooter: ShooterDecision
    geometry: GeometrySummary
    metrics: PublicMetrics
    component_success: dict[str, bool]
    annotated_video_available: bool


class ClipPreprocess(BaseModel):
    """App-layer inspection/trim. Never used as frozen-pipeline inference input."""
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    duration_s: Optional[float] = None
    cut_frames: list[int] = Field(default_factory=list)
    trimmed: bool = False
    trim_start_frame: Optional[int] = None
    trim_end_frame: Optional[int] = None
    original_duration_s: Optional[float] = None
    green_fraction: Optional[float] = None
    warnings: list[str] = Field(default_factory=list)
    overlays: list[str] = Field(default_factory=list)
    geometry_fallback: Optional[dict[str, Any]] = None
    team_assignment_v3: Optional[dict[str, Any]] = None
    goal_end: Optional[dict[str, Any]] = None


class JobResponse(BaseModel):
    job_id: str
    state: JobState
    created_at: str
    updated_at: str
    original_filename: str
    video_sha256: Optional[str] = None
    public_api_calls: int = 0
    error: Optional[str] = None
    automatic: Optional[AnalysisSummary] = None
    confirmed: Optional[AnalysisSummary] = None
    shot_frame_idx: Optional[int] = None
    can_change_shooter: bool = False
    confirmations: list[ConfirmationRecord] = Field(default_factory=list)
    match_context: Optional[MatchContext] = None
    preprocess: Optional[ClipPreprocess] = None
    links: dict[str, str] = Field(default_factory=dict)


class ConfirmShooterRequest(BaseModel):
    shooter_track_id: int = Field(..., description="One of the candidate track ids returned by the automatic pass.")


class ErrorResponse(BaseModel):
    detail: str
    extra: dict[str, Any] = Field(default_factory=dict)
