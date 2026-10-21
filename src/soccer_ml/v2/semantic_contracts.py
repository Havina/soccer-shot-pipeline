from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any, Optional

from soccer_ml.results import SoccerClipAnalysisResult

from .geometry_client import validate_homography_pair


GEOMETRY_V2_RUNTIME_MODE = "GEOMETRY_V2_HTTP_MULTIPART_GPU_SERVICE_V1"

AUTO_SHOOTER_REQUIRED_MARGIN_FRACTION = 0.10

SHOOTER_AUTO_SELECTED = "AUTO_SELECTED"

SHOOTER_AUTO_AMBIGUOUS = "AMBIGUOUS_TOP_TWO"

SHOOTER_AUTO_FAILED = "FAILED"

SHOOTER_RESOLUTION_AUTO = "AUTO_SELECTION"

SHOOTER_RESOLUTION_UNRESOLVED = "UNRESOLVED"

SHOOTER_RESOLUTION_USER_CONFIRMATION = "USER_CONFIRMATION"

SHOOTER_RESOLUTION_FAILED = "FAILED"

SHOOTER_CONFIRMATION_SOURCE_USER = "USER_CONFIRMATION"

GEOMETRY_SUCCESS_STATUS_BY_VARIANT = {
    "PNL_REFINED":
        "PRIMARY_PNL_REFINED",

    "BASE":
        "FALLBACK_BASE",
}

GEOMETRY_FAILURE_STATUSES = {
    "NO_VALID_GEOMETRY",
    "RUNTIME_FAILURE",
    "INPUT_CONTRACT_FAILURE",
}


CANONICAL_V2_STAGE_NAMES = (
    "input_validation",
    "artifact_validation",
    "video_metadata",
    "ball_detection",
    "ball_tracking",
    "player_detection_tracking",
    "pitch_line_detection",
    "homography_search",
    "homography_validation",
    "pitch_projection",
    "shooter_assignment",
    "relative_team_assignment",
    "target_goal_inference",
    "shot_geometry",
    "defender_context",
    "enhanced_xg",
    "annotation",
    "output_validation",
)


SHOOTER_CONFIRMATION_COMPLETED_STAGE_COUNT = 10

SHOOTER_CONFIRMATION_BLOCKED_STAGE = "shooter_assignment"


class SemanticV2State(str, Enum):

    COMPLETE = "COMPLETE"

    SHOOTER_CONFIRMATION_REQUIRED = "SHOOTER_CONFIRMATION_REQUIRED"

    FAILED = "FAILED"


@dataclass(frozen=True)
class GeometryV2Result:

    success: bool

    runtime_mode: str

    selected_variant: Optional[str]

    selected_frame_idx: Optional[int]

    homography_image_to_pitch: Optional[list[list[float]]]

    homography_pitch_to_image: Optional[list[list[float]]]

    keypoint_count: Optional[int]

    line_count: Optional[int]

    reference_free: Optional[bool]

    validation_status: str

    runtime_fingerprint: Optional[Mapping[str, Any]]

    failure_reason: Optional[str]


@dataclass(frozen=True)
class ShooterV2Decision:

    auto_status: str

    resolution: str

    resolved_track_id: Optional[int]

    candidate_track_ids: tuple[int, ...]

    auto_scores: Mapping[int, float]

    auto_top_track_id: Optional[int]

    auto_second_track_id: Optional[int]

    auto_margin_fraction: Optional[float]

    required_margin_fraction: float

    confirmation_track_id: Optional[int]

    confirmation_source: Optional[str]

    manual_confirmation_used: bool

    confirmation_used_as_model_evidence: bool

    failure_reason: Optional[str]

    @property
    def resolved(self) -> bool:

        return (
            self.resolved_track_id
            is not None
        )

    @property
    def auto_top_score(self) -> Optional[float]:

        if self.auto_top_track_id is None:

            return None

        value = self.auto_scores.get(
            self.auto_top_track_id
        )

        return (
            float(
                value
            )
            if value is not None
            else None
        )

    @property
    def auto_second_score(self) -> Optional[float]:

        if self.auto_second_track_id is None:

            return None

        value = self.auto_scores.get(
            self.auto_second_track_id
        )

        return (
            float(
                value
            )
            if value is not None
            else None
        )


@dataclass(frozen=True)
class SemanticV2Status:

    state: SemanticV2State

    completed_stage_count: int

    blocked_stage: Optional[str]

    confirmation_required: bool

    failure_stage: Optional[str]

    failure_reason: Optional[str]


@dataclass(frozen=False, kw_only=True)
class SoccerClipAnalysisV2Result(
    SoccerClipAnalysisResult
):

    geometry_v2: GeometryV2Result

    shooter_v2: ShooterV2Decision

    semantic_v2_status: SemanticV2Status


def _require_track_id(
    value: Any,
    name: str,
) -> int:

    if (
        not isinstance(
            value,
            int,
        )
        or
        isinstance(
            value,
            bool,
        )
        or
        value
        <
        0
    ):

        raise ValueError(
            f"{name} must be a non-negative integer track ID."
        )

    return int(
        value
    )


def _require_optional_nonnegative_int(
    value: Any,
    name: str,
):

    if value is None:

        return

    if (
        not isinstance(
            value,
            int,
        )
        or
        isinstance(
            value,
            bool,
        )
        or
        value
        <
        0
    ):

        raise ValueError(
            f"{name} must be None or a non-negative integer."
        )


def validate_geometry_v2_result(
    value: GeometryV2Result,
) -> GeometryV2Result:

    if not isinstance(
        value,
        GeometryV2Result,
    ):

        raise TypeError(
            "value must be GeometryV2Result."
        )


    if not isinstance(
        value.success,
        bool,
    ):

        raise ValueError(
            "GeometryV2Result.success must be bool."
        )


    if value.runtime_mode != GEOMETRY_V2_RUNTIME_MODE:

        raise ValueError(
            "GeometryV2Result.runtime_mode does not match the frozen Geometry-v2 runtime mode."
        )


    _require_optional_nonnegative_int(
        value.selected_frame_idx,
        "selected_frame_idx",
    )


    _require_optional_nonnegative_int(
        value.keypoint_count,
        "keypoint_count",
    )


    _require_optional_nonnegative_int(
        value.line_count,
        "line_count",
    )


    if value.runtime_fingerprint is not None:

        if not isinstance(
            value.runtime_fingerprint,
            Mapping,
        ):

            raise ValueError(
                "runtime_fingerprint must be None or a mapping."
            )


    if value.success:

        if value.selected_variant not in GEOMETRY_SUCCESS_STATUS_BY_VARIANT:

            raise ValueError(
                "Successful GeometryV2Result returned an unsupported selected_variant."
            )


        expected_status = GEOMETRY_SUCCESS_STATUS_BY_VARIANT[
            value.selected_variant
        ]


        if value.validation_status != expected_status:

            raise ValueError(
                "Successful GeometryV2Result variant/status mismatch."
            )


        if value.selected_frame_idx is None:

            raise ValueError(
                "Successful GeometryV2Result requires selected_frame_idx."
            )


        if value.keypoint_count is None:

            raise ValueError(
                "Successful GeometryV2Result requires keypoint_count."
            )


        if value.line_count is None:

            raise ValueError(
                "Successful GeometryV2Result requires line_count."
            )


        if value.reference_free is not True:

            raise ValueError(
                "Successful GeometryV2Result must be reference-free."
            )


        if value.runtime_fingerprint is None:

            raise ValueError(
                "Successful GeometryV2Result requires runtime_fingerprint."
            )


        if value.failure_reason is not None:

            raise ValueError(
                "Successful GeometryV2Result cannot have failure_reason."
            )


        if (
            value.homography_image_to_pitch
            is None
            or
            value.homography_pitch_to_image
            is None
        ):

            raise ValueError(
                "Successful GeometryV2Result requires both homographies."
            )


        validate_homography_pair(
            value.homography_image_to_pitch,
            value.homography_pitch_to_image,
        )


    else:

        if value.selected_variant is not None:

            raise ValueError(
                "Failed GeometryV2Result cannot select a geometry variant."
            )


        if value.validation_status not in GEOMETRY_FAILURE_STATUSES:

            raise ValueError(
                "Failed GeometryV2Result returned an unsupported failure status."
            )


        if (
            value.homography_image_to_pitch
            is not None
            or
            value.homography_pitch_to_image
            is not None
        ):

            raise ValueError(
                "Failed GeometryV2Result cannot contain selected homographies."
            )


        if (
            not isinstance(
                value.failure_reason,
                str,
            )
            or
            not value.failure_reason
        ):

            raise ValueError(
                "Failed GeometryV2Result requires failure_reason."
            )


    return value


def validate_shooter_v2_decision(
    value: ShooterV2Decision,
) -> ShooterV2Decision:

    if not isinstance(
        value,
        ShooterV2Decision,
    ):

        raise TypeError(
            "value must be ShooterV2Decision."
        )


    if value.auto_status not in {
        SHOOTER_AUTO_SELECTED,
        SHOOTER_AUTO_AMBIGUOUS,
        SHOOTER_AUTO_FAILED,
    }:

        raise ValueError(
            "Unsupported Shooter-v2 auto_status."
        )


    if value.resolution not in {
        SHOOTER_RESOLUTION_AUTO,
        SHOOTER_RESOLUTION_UNRESOLVED,
        SHOOTER_RESOLUTION_USER_CONFIRMATION,
        SHOOTER_RESOLUTION_FAILED,
    }:

        raise ValueError(
            "Unsupported Shooter-v2 resolution."
        )


    if not isinstance(
        value.candidate_track_ids,
        tuple,
    ):

        raise ValueError(
            "candidate_track_ids must be a tuple."
        )


    if len(
        value.candidate_track_ids
    ) != len(
        set(
            value.candidate_track_ids
        )
    ):

        raise ValueError(
            "candidate_track_ids must be unique."
        )


    for track_id in value.candidate_track_ids:

        _require_track_id(
            track_id,
            "candidate_track_id",
        )


    if not isinstance(
        value.auto_scores,
        Mapping,
    ):

        raise ValueError(
            "auto_scores must be a mapping."
        )


    if set(
        value.auto_scores.keys()
    ) != set(
        value.candidate_track_ids
    ):

        raise ValueError(
            "auto_scores keys must exactly match candidate_track_ids."
        )


    for track_id, score in value.auto_scores.items():

        _require_track_id(
            track_id,
            "auto_scores track_id",
        )


        if (
            not isinstance(
                score,
                (int, float),
            )
            or
            isinstance(
                score,
                bool,
            )
            or
            not isfinite(
                float(
                    score
                )
            )
            or
            float(
                score
            )
            <
            0.0
            or
            float(
                score
            )
            >
            1.0
        ):

            raise ValueError(
                "Every Shooter-v2 auto score must be finite and within [0, 1]."
            )


    if (
        not isinstance(
            value.required_margin_fraction,
            (int, float),
        )
        or
        isinstance(
            value.required_margin_fraction,
            bool,
        )
        or
        not isfinite(
            float(
                value.required_margin_fraction
            )
        )
        or
        float(
            value.required_margin_fraction
        )
        <
        0.0
        or
        float(
            value.required_margin_fraction
        )
        >
        1.0
    ):

        raise ValueError(
            "required_margin_fraction must be finite and within [0, 1]."
        )


    if abs(
        float(
            value.required_margin_fraction
        )
        -
        AUTO_SHOOTER_REQUIRED_MARGIN_FRACTION
    ) > 1e-12:

        raise ValueError(
            "Shooter-v2 required margin differs from the frozen 10% policy."
        )


    if value.auto_margin_fraction is not None:

        if (
            not isinstance(
                value.auto_margin_fraction,
                (int, float),
            )
            or
            isinstance(
                value.auto_margin_fraction,
                bool,
            )
            or
            not isfinite(
                float(
                    value.auto_margin_fraction
                )
            )
            or
            float(
                value.auto_margin_fraction
            )
            <
            0.0
        ):

            raise ValueError(
                "auto_margin_fraction must be None or a finite non-negative number."
            )


    for optional_track_name, optional_track_value in [
        (
            "resolved_track_id",
            value.resolved_track_id,
        ),
        (
            "auto_top_track_id",
            value.auto_top_track_id,
        ),
        (
            "auto_second_track_id",
            value.auto_second_track_id,
        ),
        (
            "confirmation_track_id",
            value.confirmation_track_id,
        ),
    ]:

        if optional_track_value is not None:

            _require_track_id(
                optional_track_value,
                optional_track_name,
            )


    if value.auto_top_track_id is not None:

        if value.auto_top_track_id not in value.candidate_track_ids:

            raise ValueError(
                "auto_top_track_id must be a Shooter-v2 candidate."
            )


    if value.auto_second_track_id is not None:

        if value.auto_second_track_id not in value.candidate_track_ids:

            raise ValueError(
                "auto_second_track_id must be a Shooter-v2 candidate."
            )


    if (
        value.auto_top_track_id is not None
        and
        value.auto_second_track_id is not None
        and
        value.auto_top_track_id
        ==
        value.auto_second_track_id
    ):

        raise ValueError(
            "Shooter-v2 top and second candidate cannot be the same track."
        )


    if (
        value.auto_top_track_id is not None
        and
        value.auto_second_track_id is not None
    ):

        top_score = float(
            value.auto_scores[
                value.auto_top_track_id
            ]
        )


        second_score = float(
            value.auto_scores[
                value.auto_second_track_id
            ]
        )


        if top_score < second_score:

            raise ValueError(
                "Shooter-v2 auto_top_track_id must not score below auto_second_track_id."
            )


    if not isinstance(
        value.manual_confirmation_used,
        bool,
    ):

        raise ValueError(
            "manual_confirmation_used must be bool."
        )


    if not isinstance(
        value.confirmation_used_as_model_evidence,
        bool,
    ):

        raise ValueError(
            "confirmation_used_as_model_evidence must be bool."
        )


    if value.confirmation_used_as_model_evidence:

        raise ValueError(
            "Manual confirmation is forbidden as Shooter-v2 model/scoring evidence."
        )


    if value.auto_status == SHOOTER_AUTO_SELECTED:

        if value.auto_top_track_id is None:

            raise ValueError(
                "AUTO_SELECTED requires auto_top_track_id."
            )


        if (
            value.auto_second_track_id is not None
            and
            (
                value.auto_margin_fraction is None
                or
                float(
                    value.auto_margin_fraction
                )
                <=
                float(
                    value.required_margin_fraction
                )
            )
        ):

            raise ValueError(
                "AUTO_SELECTED requires margin strictly above the frozen threshold when a second candidate exists."
            )


        if value.failure_reason is not None:

            raise ValueError(
                "AUTO_SELECTED cannot contain failure_reason."
            )


    if value.auto_status == SHOOTER_AUTO_AMBIGUOUS:

        if (
            value.auto_top_track_id is None
            or
            value.auto_second_track_id is None
            or
            value.auto_margin_fraction is None
        ):

            raise ValueError(
                "AMBIGUOUS_TOP_TWO requires top, second, and margin evidence."
            )


        if float(
            value.auto_margin_fraction
        ) > float(
            value.required_margin_fraction
        ):

            raise ValueError(
                "AMBIGUOUS_TOP_TWO margin cannot exceed the frozen auto-selection threshold."
            )


        if value.failure_reason is not None:

            raise ValueError(
                "AMBIGUOUS_TOP_TWO is not a failure and cannot contain failure_reason."
            )


    if value.auto_status == SHOOTER_AUTO_FAILED:

        if value.resolution != SHOOTER_RESOLUTION_FAILED:

            raise ValueError(
                "FAILED auto decision requires FAILED resolution."
            )


        if value.resolved_track_id is not None:

            raise ValueError(
                "FAILED Shooter-v2 decision cannot have resolved_track_id."
            )


        if (
            not isinstance(
                value.failure_reason,
                str,
            )
            or
            not value.failure_reason
        ):

            raise ValueError(
                "FAILED Shooter-v2 decision requires failure_reason."
            )


    if value.resolution == SHOOTER_RESOLUTION_AUTO:

        if value.auto_status != SHOOTER_AUTO_SELECTED:

            raise ValueError(
                "AUTO_SELECTION resolution requires AUTO_SELECTED auto status."
            )


        if value.resolved_track_id != value.auto_top_track_id:

            raise ValueError(
                "AUTO_SELECTION must resolve to auto_top_track_id."
            )


        if value.manual_confirmation_used:

            raise ValueError(
                "AUTO_SELECTION cannot use manual confirmation."
            )


        if (
            value.confirmation_track_id is not None
            or
            value.confirmation_source is not None
        ):

            raise ValueError(
                "AUTO_SELECTION cannot contain confirmation provenance."
            )


    if value.resolution == SHOOTER_RESOLUTION_UNRESOLVED:

        if value.auto_status != SHOOTER_AUTO_AMBIGUOUS:

            raise ValueError(
                "UNRESOLVED Shooter-v2 state requires AMBIGUOUS_TOP_TWO."
            )


        if value.resolved_track_id is not None:

            raise ValueError(
                "UNRESOLVED Shooter-v2 decision cannot contain resolved_track_id."
            )


        if value.manual_confirmation_used:

            raise ValueError(
                "UNRESOLVED Shooter-v2 decision cannot have manual confirmation."
            )


        if (
            value.confirmation_track_id is not None
            or
            value.confirmation_source is not None
        ):

            raise ValueError(
                "UNRESOLVED Shooter-v2 decision cannot contain confirmation provenance."
            )


    if value.resolution == SHOOTER_RESOLUTION_USER_CONFIRMATION:

        if value.auto_status not in {
            SHOOTER_AUTO_SELECTED,
            SHOOTER_AUTO_AMBIGUOUS,
        }:

            raise ValueError(
                "USER_CONFIRMATION requires preserved successful/ambiguous auto evidence."
            )


        if not value.manual_confirmation_used:

            raise ValueError(
                "USER_CONFIRMATION requires manual_confirmation_used=True."
            )


        if value.confirmation_source != SHOOTER_CONFIRMATION_SOURCE_USER:

            raise ValueError(
                "USER_CONFIRMATION requires confirmation_source=USER_CONFIRMATION."
            )


        if value.confirmation_track_id is None:

            raise ValueError(
                "USER_CONFIRMATION requires confirmation_track_id."
            )


        if value.resolved_track_id != value.confirmation_track_id:

            raise ValueError(
                "USER_CONFIRMATION must resolve to confirmation_track_id."
            )


        if value.failure_reason is not None:

            raise ValueError(
                "Resolved USER_CONFIRMATION cannot contain failure_reason."
            )


    if value.resolution == SHOOTER_RESOLUTION_FAILED:

        if value.auto_status != SHOOTER_AUTO_FAILED:

            raise ValueError(
                "FAILED resolution requires FAILED auto status."
            )


    return value


def validate_semantic_v2_status(
    value: SemanticV2Status,
) -> SemanticV2Status:

    if not isinstance(
        value,
        SemanticV2Status,
    ):

        raise TypeError(
            "value must be SemanticV2Status."
        )


    if not isinstance(
        value.state,
        SemanticV2State,
    ):

        raise ValueError(
            "SemanticV2Status.state must be SemanticV2State."
        )


    if (
        not isinstance(
            value.completed_stage_count,
            int,
        )
        or
        isinstance(
            value.completed_stage_count,
            bool,
        )
        or
        value.completed_stage_count
        <
        0
        or
        value.completed_stage_count
        >
        len(
            CANONICAL_V2_STAGE_NAMES
        )
    ):

        raise ValueError(
            "completed_stage_count is outside the canonical v2 stage range."
        )


    if value.state == SemanticV2State.COMPLETE:

        if value.completed_stage_count != len(
            CANONICAL_V2_STAGE_NAMES
        ):

            raise ValueError(
                "COMPLETE semantic-v2 status requires all 18 stages complete."
            )


        if value.blocked_stage is not None:

            raise ValueError(
                "COMPLETE semantic-v2 status cannot have blocked_stage."
            )


        if value.confirmation_required:

            raise ValueError(
                "COMPLETE semantic-v2 status cannot require confirmation."
            )


        if value.failure_stage is not None:

            raise ValueError(
                "COMPLETE semantic-v2 status cannot have failure_stage."
            )


        if value.failure_reason is not None:

            raise ValueError(
                "COMPLETE semantic-v2 status cannot have failure_reason."
            )


    elif value.state == SemanticV2State.SHOOTER_CONFIRMATION_REQUIRED:

        if (
            value.completed_stage_count
            !=
            SHOOTER_CONFIRMATION_COMPLETED_STAGE_COUNT
        ):

            raise ValueError(
                "SHOOTER_CONFIRMATION_REQUIRED must preserve the completed prefix through pitch_projection."
            )


        if value.blocked_stage != SHOOTER_CONFIRMATION_BLOCKED_STAGE:

            raise ValueError(
                "SHOOTER_CONFIRMATION_REQUIRED must block shooter_assignment."
            )


        if value.confirmation_required is not True:

            raise ValueError(
                "SHOOTER_CONFIRMATION_REQUIRED must require confirmation."
            )


        if value.failure_stage is not None:

            raise ValueError(
                "SHOOTER_CONFIRMATION_REQUIRED is nonfailed and cannot contain failure_stage."
            )


        if value.failure_reason is not None:

            raise ValueError(
                "SHOOTER_CONFIRMATION_REQUIRED is nonfailed and cannot contain failure_reason."
            )


    elif value.state == SemanticV2State.FAILED:

        if value.confirmation_required:

            raise ValueError(
                "FAILED semantic-v2 status cannot simultaneously request shooter confirmation."
            )


        if value.blocked_stage is not None:

            raise ValueError(
                "FAILED semantic-v2 status cannot use blocked_stage."
            )


        if value.failure_stage not in CANONICAL_V2_STAGE_NAMES:

            raise ValueError(
                "FAILED semantic-v2 status requires a canonical failure_stage."
            )


        if (
            not isinstance(
                value.failure_reason,
                str,
            )
            or
            not value.failure_reason
        ):

            raise ValueError(
                "FAILED semantic-v2 status requires failure_reason."
            )


    return value


def validate_soccer_clip_analysis_v2_result(
    value: SoccerClipAnalysisV2Result,
) -> SoccerClipAnalysisV2Result:

    if not isinstance(
        value,
        SoccerClipAnalysisV2Result,
    ):

        raise TypeError(
            "value must be SoccerClipAnalysisV2Result."
        )


    validate_geometry_v2_result(
        value.geometry_v2
    )


    validate_shooter_v2_decision(
        value.shooter_v2
    )


    validate_semantic_v2_status(
        value.semantic_v2_status
    )


    state = value.semantic_v2_status.state


    if state == SemanticV2State.SHOOTER_CONFIRMATION_REQUIRED:

        if value.geometry_v2.success is not True:

            raise ValueError(
                "Shooter confirmation may only be requested after valid Geometry-v2 completion."
            )


        if value.shooter_v2.auto_status != SHOOTER_AUTO_AMBIGUOUS:

            raise ValueError(
                "Shooter confirmation state requires preserved AMBIGUOUS_TOP_TWO auto evidence."
            )


        if value.shooter_v2.resolution != SHOOTER_RESOLUTION_UNRESOLVED:

            raise ValueError(
                "Shooter confirmation state requires unresolved Shooter-v2 decision."
            )


    if state == SemanticV2State.COMPLETE:

        if value.geometry_v2.success is not True:

            raise ValueError(
                "COMPLETE semantic-v2 result requires successful Geometry-v2."
            )


        if value.shooter_v2.resolution not in {
            SHOOTER_RESOLUTION_AUTO,
            SHOOTER_RESOLUTION_USER_CONFIRMATION,
        }:

            raise ValueError(
                "COMPLETE semantic-v2 result requires a resolved Shooter-v2 decision."
            )


        if value.shooter_v2.resolved_track_id is None:

            raise ValueError(
                "COMPLETE semantic-v2 result requires resolved shooter track."
            )


    return value
