from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .geometry_client import GEOMETRY_FRAME_POLICY
from .semantic_contracts import (
    CANONICAL_V2_STAGE_NAMES,
    SHOOTER_CONFIRMATION_BLOCKED_STAGE,
    SHOOTER_CONFIRMATION_COMPLETED_STAGE_COUNT,
    SemanticV2State,
    SemanticV2Status,
    validate_semantic_v2_status,
)


V2_EXECUTION_POLICY = "STATELESS_REPLAY_WITH_OPTIONAL_CONFIRMATION"

GEOMETRY_ANCHOR_SOURCE = "FIRST_RAW_V3_BALL_FRAME"

GEOMETRY_CANDIDATE_FRAME_COUNT = 1

GEOMETRY_TEMPORAL_FALLBACK_ENABLED = False

GEOMETRY_V1_FALLBACK_ENABLED = False

TARGET_GOAL_POLICY = "IMAGE_SPACE_SEMANTIC_MOTION_V2"

TARGET_GOAL_REQUIRES_AIRBORNE_GROUND_PLANE = False


class V2StageAction(str, Enum):

    REUSE_V1_HANDLER = "REUSE_V1_HANDLER"

    WRAP_V1_PLUS_GEOMETRY_RUNTIME_READINESS = (
        "WRAP_V1_PLUS_GEOMETRY_RUNTIME_READINESS"
    )

    REPLACE_V2_PNLCALIB_KEYPOINT_LINE_INFERENCE = (
        "REPLACE_V2_PNLCALIB_KEYPOINT_LINE_INFERENCE"
    )

    REPLACE_V2_PNLCALIB_BASE_PNL_SELECTION = (
        "REPLACE_V2_PNLCALIB_BASE_PNL_SELECTION"
    )

    REPLACE_V2_REFERENCE_FREE_GEOMETRY_VALIDATION = (
        "REPLACE_V2_REFERENCE_FREE_GEOMETRY_VALIDATION"
    )

    REPLACE_V2_PLAYER_PROJECTION = (
        "REPLACE_V2_PLAYER_PROJECTION"
    )

    REPLACE_V2_AMBIGUITY_SAFE_SHOOTER_GATE = (
        "REPLACE_V2_AMBIGUITY_SAFE_SHOOTER_GATE"
    )

    REUSE_AFTER_RESOLVED = "REUSE_AFTER_RESOLVED"

    REPLACE_V2_IMAGE_SPACE_TARGET_GOAL = (
        "REPLACE_V2_IMAGE_SPACE_TARGET_GOAL"
    )

    WRAP_V1_PLUS_AUTHORITATIVE_V2_PROVENANCE = (
        "WRAP_V1_PLUS_AUTHORITATIVE_V2_PROVENANCE"
    )


@dataclass(frozen=True)
class V2StageSpec:

    index: int

    name: str

    action: V2StageAction

    may_execute_before_shooter_resolution: bool

    requires_resolved_shooter: bool

    geometry_v2_stage: bool


V2_STAGE_PLAN = (
    V2StageSpec(
        index=1,
        name="input_validation",
        action=V2StageAction.REUSE_V1_HANDLER,
        may_execute_before_shooter_resolution=True,
        requires_resolved_shooter=False,
        geometry_v2_stage=False,
    ),

    V2StageSpec(
        index=2,
        name="artifact_validation",
        action=V2StageAction.WRAP_V1_PLUS_GEOMETRY_RUNTIME_READINESS,
        may_execute_before_shooter_resolution=True,
        requires_resolved_shooter=False,
        geometry_v2_stage=False,
    ),

    V2StageSpec(
        index=3,
        name="video_metadata",
        action=V2StageAction.REUSE_V1_HANDLER,
        may_execute_before_shooter_resolution=True,
        requires_resolved_shooter=False,
        geometry_v2_stage=False,
    ),

    V2StageSpec(
        index=4,
        name="ball_detection",
        action=V2StageAction.REUSE_V1_HANDLER,
        may_execute_before_shooter_resolution=True,
        requires_resolved_shooter=False,
        geometry_v2_stage=False,
    ),

    V2StageSpec(
        index=5,
        name="ball_tracking",
        action=V2StageAction.REUSE_V1_HANDLER,
        may_execute_before_shooter_resolution=True,
        requires_resolved_shooter=False,
        geometry_v2_stage=False,
    ),

    V2StageSpec(
        index=6,
        name="player_detection_tracking",
        action=V2StageAction.REUSE_V1_HANDLER,
        may_execute_before_shooter_resolution=True,
        requires_resolved_shooter=False,
        geometry_v2_stage=False,
    ),

    V2StageSpec(
        index=7,
        name="pitch_line_detection",
        action=V2StageAction.REPLACE_V2_PNLCALIB_KEYPOINT_LINE_INFERENCE,
        may_execute_before_shooter_resolution=True,
        requires_resolved_shooter=False,
        geometry_v2_stage=True,
    ),

    V2StageSpec(
        index=8,
        name="homography_search",
        action=V2StageAction.REPLACE_V2_PNLCALIB_BASE_PNL_SELECTION,
        may_execute_before_shooter_resolution=True,
        requires_resolved_shooter=False,
        geometry_v2_stage=True,
    ),

    V2StageSpec(
        index=9,
        name="homography_validation",
        action=V2StageAction.REPLACE_V2_REFERENCE_FREE_GEOMETRY_VALIDATION,
        may_execute_before_shooter_resolution=True,
        requires_resolved_shooter=False,
        geometry_v2_stage=True,
    ),

    V2StageSpec(
        index=10,
        name="pitch_projection",
        action=V2StageAction.REPLACE_V2_PLAYER_PROJECTION,
        may_execute_before_shooter_resolution=True,
        requires_resolved_shooter=False,
        geometry_v2_stage=True,
    ),

    V2StageSpec(
        index=11,
        name="shooter_assignment",
        action=V2StageAction.REPLACE_V2_AMBIGUITY_SAFE_SHOOTER_GATE,
        may_execute_before_shooter_resolution=True,
        requires_resolved_shooter=False,
        geometry_v2_stage=False,
    ),

    V2StageSpec(
        index=12,
        name="relative_team_assignment",
        action=V2StageAction.REUSE_AFTER_RESOLVED,
        may_execute_before_shooter_resolution=False,
        requires_resolved_shooter=True,
        geometry_v2_stage=False,
    ),

    V2StageSpec(
        index=13,
        name="target_goal_inference",
        action=V2StageAction.REPLACE_V2_IMAGE_SPACE_TARGET_GOAL,
        may_execute_before_shooter_resolution=False,
        requires_resolved_shooter=True,
        geometry_v2_stage=False,
    ),

    V2StageSpec(
        index=14,
        name="shot_geometry",
        action=V2StageAction.REUSE_AFTER_RESOLVED,
        may_execute_before_shooter_resolution=False,
        requires_resolved_shooter=True,
        geometry_v2_stage=False,
    ),

    V2StageSpec(
        index=15,
        name="defender_context",
        action=V2StageAction.REUSE_AFTER_RESOLVED,
        may_execute_before_shooter_resolution=False,
        requires_resolved_shooter=True,
        geometry_v2_stage=False,
    ),

    V2StageSpec(
        index=16,
        name="enhanced_xg",
        action=V2StageAction.REUSE_AFTER_RESOLVED,
        may_execute_before_shooter_resolution=False,
        requires_resolved_shooter=True,
        geometry_v2_stage=False,
    ),

    V2StageSpec(
        index=17,
        name="annotation",
        action=V2StageAction.REUSE_AFTER_RESOLVED,
        may_execute_before_shooter_resolution=False,
        requires_resolved_shooter=True,
        geometry_v2_stage=False,
    ),

    V2StageSpec(
        index=18,
        name="output_validation",
        action=V2StageAction.WRAP_V1_PLUS_AUTHORITATIVE_V2_PROVENANCE,
        may_execute_before_shooter_resolution=False,
        requires_resolved_shooter=True,
        geometry_v2_stage=False,
    ),
)


EXPECTED_ACTION_BY_STAGE = {
    "input_validation":
        V2StageAction.REUSE_V1_HANDLER,

    "artifact_validation":
        V2StageAction.WRAP_V1_PLUS_GEOMETRY_RUNTIME_READINESS,

    "video_metadata":
        V2StageAction.REUSE_V1_HANDLER,

    "ball_detection":
        V2StageAction.REUSE_V1_HANDLER,

    "ball_tracking":
        V2StageAction.REUSE_V1_HANDLER,

    "player_detection_tracking":
        V2StageAction.REUSE_V1_HANDLER,

    "pitch_line_detection":
        V2StageAction.REPLACE_V2_PNLCALIB_KEYPOINT_LINE_INFERENCE,

    "homography_search":
        V2StageAction.REPLACE_V2_PNLCALIB_BASE_PNL_SELECTION,

    "homography_validation":
        V2StageAction.REPLACE_V2_REFERENCE_FREE_GEOMETRY_VALIDATION,

    "pitch_projection":
        V2StageAction.REPLACE_V2_PLAYER_PROJECTION,

    "shooter_assignment":
        V2StageAction.REPLACE_V2_AMBIGUITY_SAFE_SHOOTER_GATE,

    "relative_team_assignment":
        V2StageAction.REUSE_AFTER_RESOLVED,

    "target_goal_inference":
        V2StageAction.REPLACE_V2_IMAGE_SPACE_TARGET_GOAL,

    "shot_geometry":
        V2StageAction.REUSE_AFTER_RESOLVED,

    "defender_context":
        V2StageAction.REUSE_AFTER_RESOLVED,

    "enhanced_xg":
        V2StageAction.REUSE_AFTER_RESOLVED,

    "annotation":
        V2StageAction.REUSE_AFTER_RESOLVED,

    "output_validation":
        V2StageAction.WRAP_V1_PLUS_AUTHORITATIVE_V2_PROVENANCE,
}


GEOMETRY_V2_STAGE_NAMES = (
    "pitch_line_detection",
    "homography_search",
    "homography_validation",
    "pitch_projection",
)


PRE_RESOLUTION_EXECUTABLE_STAGE_NAMES = (
    CANONICAL_V2_STAGE_NAMES[
        :11
    ]
)


POST_RESOLUTION_REQUIRED_STAGE_NAMES = (
    CANONICAL_V2_STAGE_NAMES[
        11:
    ]
)


CONFIRMATION_REQUIRED_COMPLETED_STAGE_NAMES = (
    CANONICAL_V2_STAGE_NAMES[
        :SHOOTER_CONFIRMATION_COMPLETED_STAGE_COUNT
    ]
)


CONFIRMATION_REQUIRED_FORBIDDEN_STAGE_NAMES = (
    CANONICAL_V2_STAGE_NAMES[
        11:
    ]
)


@dataclass(frozen=True)
class V2TerminalSnapshot:

    completed_stages: tuple[str, ...]

    blocked_stage: Optional[str]

    failed_stage: Optional[str]

    failure_reason: Optional[str]

    shooter_confirmation_track_id: Optional[int]


def stage_spec(
    stage_name: str,
) -> V2StageSpec:

    for spec in V2_STAGE_PLAN:

        if spec.name == stage_name:

            return spec


    raise KeyError(
        f"Unknown canonical v2 stage: {stage_name!r}"
    )


def validate_stage_plan(
    plan=V2_STAGE_PLAN,
):

    if not isinstance(
        plan,
        tuple,
    ):

        raise ValueError(
            "Canonical v2 stage plan must be a tuple."
        )


    if len(
        plan
    ) != len(
        CANONICAL_V2_STAGE_NAMES
    ):

        raise ValueError(
            "Canonical v2 stage plan must contain exactly 18 stages."
        )


    names = tuple(
        spec.name
        for spec in plan
    )


    if names != CANONICAL_V2_STAGE_NAMES:

        raise ValueError(
            "Canonical v2 stage names/order do not match the frozen semantic contract."
        )


    for expected_index, spec in enumerate(
        plan,
        start=1,
    ):

        if not isinstance(
            spec,
            V2StageSpec,
        ):

            raise ValueError(
                "Every v2 stage-plan entry must be V2StageSpec."
            )


        if spec.index != expected_index:

            raise ValueError(
                "Canonical v2 stage indexes must be contiguous 1..18."
            )


        if EXPECTED_ACTION_BY_STAGE.get(
            spec.name
        ) != spec.action:

            raise ValueError(
                f"Unexpected v2 implementation action for stage {spec.name!r}."
            )


        if expected_index <= 11:

            if spec.may_execute_before_shooter_resolution is not True:

                raise ValueError(
                    f"Stage {spec.name!r} must be executable before shooter resolution."
                )


            if spec.requires_resolved_shooter is not False:

                raise ValueError(
                    f"Stage {spec.name!r} cannot require a resolved shooter."
                )


        else:

            if spec.may_execute_before_shooter_resolution is not False:

                raise ValueError(
                    f"Stage {spec.name!r} is forbidden before shooter resolution."
                )


            if spec.requires_resolved_shooter is not True:

                raise ValueError(
                    f"Stage {spec.name!r} must require a resolved shooter."
                )


    geometry_names = tuple(
        spec.name
        for spec in plan
        if spec.geometry_v2_stage
    )


    if geometry_names != GEOMETRY_V2_STAGE_NAMES:

        raise ValueError(
            "Geometry-v2-owned stage set changed."
        )


    if GEOMETRY_FRAME_POLICY != "GEOMETRY_FRAME_POLICY_V2_0":

        raise ValueError(
            "Geometry-v2 frame policy changed from the frozen client contract."
        )


    if GEOMETRY_ANCHOR_SOURCE != "FIRST_RAW_V3_BALL_FRAME":

        raise ValueError(
            "Geometry-v2 anchor source changed."
        )


    if GEOMETRY_CANDIDATE_FRAME_COUNT != 1:

        raise ValueError(
            "Geometry-v2 must use exactly one anchor frame in v2.0."
        )


    if GEOMETRY_TEMPORAL_FALLBACK_ENABLED:

        raise ValueError(
            "Geometry-v2 temporal fallback is forbidden in v2.0."
        )


    if GEOMETRY_V1_FALLBACK_ENABLED:

        raise ValueError(
            "Geometry-v2 fallback to v1 geometry is forbidden."
        )


    if TARGET_GOAL_POLICY != "IMAGE_SPACE_SEMANTIC_MOTION_V2":

        raise ValueError(
            "Target-goal v2 policy changed."
        )


    if TARGET_GOAL_REQUIRES_AIRBORNE_GROUND_PLANE:

        raise ValueError(
            "Target-goal v2 must not require airborne ground-plane projection."
        )


    return plan


def _validate_optional_track_id(
    value: Optional[int],
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
        value < 0
    ):

        raise ValueError(
            "shooter_confirmation_track_id must be None or a non-negative integer."
        )


def _validate_completed_prefix(
    completed_stages: tuple[str, ...],
):

    if not isinstance(
        completed_stages,
        tuple,
    ):

        raise ValueError(
            "completed_stages must be a tuple."
        )


    if len(
        completed_stages
    ) > len(
        CANONICAL_V2_STAGE_NAMES
    ):

        raise ValueError(
            "completed_stages exceeds the canonical 18-stage pipeline."
        )


    expected_prefix = (
        CANONICAL_V2_STAGE_NAMES[
            :len(
                completed_stages
            )
        ]
    )


    if completed_stages != expected_prefix:

        raise ValueError(
            "completed_stages must be an exact canonical prefix."
        )


def validate_terminal_snapshot(
    value: V2TerminalSnapshot,
) -> V2TerminalSnapshot:

    if not isinstance(
        value,
        V2TerminalSnapshot,
    ):

        raise TypeError(
            "value must be V2TerminalSnapshot."
        )


    _validate_completed_prefix(
        value.completed_stages
    )


    _validate_optional_track_id(
        value.shooter_confirmation_track_id
    )


    if (
        value.blocked_stage is not None
        and
        value.failed_stage is not None
    ):

        raise ValueError(
            "A terminal snapshot cannot be blocked and failed simultaneously."
        )


    completed_count = len(
        value.completed_stages
    )


    # -----------------------------------------------------------------------------------------------
    # SHOOTER CONFIRMATION REQUIRED
    # -----------------------------------------------------------------------------------------------

    if value.blocked_stage is not None:

        if value.blocked_stage != SHOOTER_CONFIRMATION_BLOCKED_STAGE:

            raise ValueError(
                "Only shooter_assignment may be a nonfailed terminal block."
            )


        if (
            completed_count
            !=
            SHOOTER_CONFIRMATION_COMPLETED_STAGE_COUNT
        ):

            raise ValueError(
                "Shooter-confirmation block requires stages 1-10 complete."
            )


        if (
            value.completed_stages
            !=
            CONFIRMATION_REQUIRED_COMPLETED_STAGE_NAMES
        ):

            raise ValueError(
                "Shooter-confirmation completed prefix changed."
            )


        if value.failure_reason is not None:

            raise ValueError(
                "Shooter-confirmation block is nonfailed and cannot have failure_reason."
            )


        if value.shooter_confirmation_track_id is not None:

            raise ValueError(
                "A confirmation-required snapshot cannot already contain a confirmation track."
            )


        return value


    # -----------------------------------------------------------------------------------------------
    # FAILED
    # -----------------------------------------------------------------------------------------------

    if value.failed_stage is not None:

        if completed_count >= len(
            CANONICAL_V2_STAGE_NAMES
        ):

            raise ValueError(
                "A failed terminal snapshot cannot already have all 18 stages complete."
            )


        expected_failed_stage = (
            CANONICAL_V2_STAGE_NAMES[
                completed_count
            ]
        )


        if value.failed_stage != expected_failed_stage:

            raise ValueError(
                "failed_stage must be the next canonical stage after the completed prefix."
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
                "Failed terminal snapshot requires failure_reason."
            )


        return value


    # -----------------------------------------------------------------------------------------------
    # COMPLETE
    # -----------------------------------------------------------------------------------------------

    if completed_count != len(
        CANONICAL_V2_STAGE_NAMES
    ):

        raise ValueError(
            "A terminal snapshot without block/failure must contain all 18 completed stages."
        )


    if value.failure_reason is not None:

        raise ValueError(
            "Complete terminal snapshot cannot contain failure_reason."
        )


    return value


def derive_semantic_v2_status(
    value: V2TerminalSnapshot,
) -> SemanticV2Status:

    validate_terminal_snapshot(
        value
    )


    completed_count = len(
        value.completed_stages
    )


    if value.blocked_stage is not None:

        result = SemanticV2Status(
            state=(
                SemanticV2State.SHOOTER_CONFIRMATION_REQUIRED
            ),
            completed_stage_count=completed_count,
            blocked_stage=value.blocked_stage,
            confirmation_required=True,
            failure_stage=None,
            failure_reason=None,
        )


    elif value.failed_stage is not None:

        result = SemanticV2Status(
            state=(
                SemanticV2State.FAILED
            ),
            completed_stage_count=completed_count,
            blocked_stage=None,
            confirmation_required=False,
            failure_stage=value.failed_stage,
            failure_reason=value.failure_reason,
        )


    else:

        result = SemanticV2Status(
            state=(
                SemanticV2State.COMPLETE
            ),
            completed_stage_count=completed_count,
            blocked_stage=None,
            confirmation_required=False,
            failure_stage=None,
            failure_reason=None,
        )


    validate_semantic_v2_status(
        result
    )


    return result


def make_confirmation_required_snapshot() -> V2TerminalSnapshot:

    value = V2TerminalSnapshot(
        completed_stages=(
            CONFIRMATION_REQUIRED_COMPLETED_STAGE_NAMES
        ),
        blocked_stage=(
            SHOOTER_CONFIRMATION_BLOCKED_STAGE
        ),
        failed_stage=None,
        failure_reason=None,
        shooter_confirmation_track_id=None,
    )


    return validate_terminal_snapshot(
        value
    )


def make_complete_snapshot(
    *,
    shooter_confirmation_track_id: Optional[int] = None,
) -> V2TerminalSnapshot:

    value = V2TerminalSnapshot(
        completed_stages=(
            CANONICAL_V2_STAGE_NAMES
        ),
        blocked_stage=None,
        failed_stage=None,
        failure_reason=None,
        shooter_confirmation_track_id=(
            shooter_confirmation_track_id
        ),
    )


    return validate_terminal_snapshot(
        value
    )


def make_failed_snapshot(
    *,
    completed_stage_count: int,
    failure_stage: str,
    failure_reason: str,
    shooter_confirmation_track_id: Optional[int] = None,
) -> V2TerminalSnapshot:

    if (
        not isinstance(
            completed_stage_count,
            int,
        )
        or
        isinstance(
            completed_stage_count,
            bool,
        )
        or
        completed_stage_count < 0
        or
        completed_stage_count >= len(
            CANONICAL_V2_STAGE_NAMES
        )
    ):

        raise ValueError(
            "completed_stage_count must be an integer from 0 through 17 for a failed snapshot."
        )


    value = V2TerminalSnapshot(
        completed_stages=(
            CANONICAL_V2_STAGE_NAMES[
                :completed_stage_count
            ]
        ),
        blocked_stage=None,
        failed_stage=failure_stage,
        failure_reason=failure_reason,
        shooter_confirmation_track_id=(
            shooter_confirmation_track_id
        ),
    )


    return validate_terminal_snapshot(
        value
    )
