"""
Canonical public result contracts for the Soccer ML clip-analysis pipeline.

These dataclasses package the public result interfaces established and
certified by Notebook 42.

The classes in this module are intentionally lightweight containers.
Pipeline execution belongs in soccer_ml.clip_pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


__all__ = [
    "PipelineStatus",
    "BallTrackResult",
    "PlayerTrackResult",
    "PitchProjectionResult",
    "ShooterResult",
    "RelativeTeamResult",
    "TargetGoalResult",
    "ShotGeometryResult",
    "DefenderContextResult",
    "XGResult",
    "AnnotationResult",
    "SoccerClipAnalysisResult",
]


@dataclass
class PipelineStatus:
    """Execution state for the canonical clip-analysis pipeline."""

    success: bool = False
    degraded: bool = False
    failed_stage: Optional[str] = None
    degraded_mode_reason: Optional[str] = None
    completed_stages: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class BallTrackResult:
    """Public ball-detection and ball-trajectory result."""

    success: bool = False
    detections: Any = None
    track: Any = None
    raw_detection_frame_count: int = 0
    tracked_frame_count: int = 0
    failure_reason: Optional[str] = None


@dataclass
class PlayerTrackResult:
    """Public player-detection and temporal tracking result."""

    success: bool = False
    detections: Any = None
    tracks: Any = None
    track_count: int = 0
    failure_reason: Optional[str] = None


@dataclass
class PitchProjectionResult:
    """Public pitch-projection result."""

    success: bool = False
    homography: Any = None
    projected_players: Any = None
    players_projected: int = 0
    players_inside_pitch: int = 0
    validation_status: Any = None
    failure_reason: Optional[str] = None


@dataclass
class ShooterResult:
    """Public shooter-assignment result."""

    success: bool = False
    track_id: Optional[int] = None
    image_xy: Optional[Tuple[float, float]] = None
    pitch_xy: Optional[Tuple[float, float]] = None
    assignment_score: Optional[float] = None
    failure_reason: Optional[str] = None


@dataclass
class RelativeTeamResult:
    """Public relative-team assignment result."""

    success: bool = False
    shooter_team_label: Optional[str] = None
    track_labels: Dict[int, str] = field(default_factory=dict)
    failure_reason: Optional[str] = None


@dataclass
class TargetGoalResult:
    """Public target-goal inference result."""

    success: bool = False
    target_goal: Optional[str] = None
    confidence: Optional[float] = None
    failure_reason: Optional[str] = None


@dataclass
class ShotGeometryResult:
    """Public shot-geometry result."""

    success: bool = False
    shooter_pitch_xy: Optional[Tuple[float, float]] = None
    distance_to_goal_m: Optional[float] = None
    shot_angle_deg: Optional[float] = None
    failure_reason: Optional[str] = None


@dataclass
class DefenderContextResult:
    """Public defensive-pressure context result."""

    success: bool = False
    nearest_defender_m: Optional[float] = None
    defenders_within_5m: Optional[int] = None
    defender_track_ids: List[int] = field(default_factory=list)
    failure_reason: Optional[str] = None


@dataclass
class XGResult:
    """Public enhanced expected-goals result."""

    success: bool = False
    probability: Optional[float] = None
    percent: Optional[float] = None
    feature_values: Dict[str, Any] = field(default_factory=dict)
    failure_reason: Optional[str] = None


@dataclass
class AnnotationResult:
    """Public annotated-video result."""

    success: bool = False
    output_path: Optional[str] = None
    frame_count: int = 0
    failure_reason: Optional[str] = None


@dataclass
class SoccerClipAnalysisResult:
    """
    Complete public result returned by the canonical soccer-clip pipeline.

    Component result objects are created independently for every instance
    to prevent shared mutable state.
    """

    video_path: Optional[str] = None

    ball_track: BallTrackResult = field(
        default_factory=BallTrackResult
    )

    player_tracks: PlayerTrackResult = field(
        default_factory=PlayerTrackResult
    )

    pitch_projection: PitchProjectionResult = field(
        default_factory=PitchProjectionResult
    )

    shooter: ShooterResult = field(
        default_factory=ShooterResult
    )

    relative_teams: RelativeTeamResult = field(
        default_factory=RelativeTeamResult
    )

    target_goal: TargetGoalResult = field(
        default_factory=TargetGoalResult
    )

    shot_geometry: ShotGeometryResult = field(
        default_factory=ShotGeometryResult
    )

    defender_context: DefenderContextResult = field(
        default_factory=DefenderContextResult
    )

    xg: XGResult = field(
        default_factory=XGResult
    )

    annotated_clip: AnnotationResult = field(
        default_factory=AnnotationResult
    )

    pipeline_status: PipelineStatus = field(
        default_factory=PipelineStatus
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )
