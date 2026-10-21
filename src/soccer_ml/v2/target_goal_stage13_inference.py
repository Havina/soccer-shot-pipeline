from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from soccer_ml.results import TargetGoalResult

from .semantic_contracts import CANONICAL_V2_STAGE_NAMES


TARGET_GOAL_V2_STAGE13_NAME = "target_goal_inference"

TARGET_GOAL_V2_STAGE13_ACTION = (
    "REPLACE_V2_IMAGE_SPACE_TARGET_GOAL"
)

TARGET_GOAL_V2_POLICY = (
    "IMAGE_SPACE_SEMANTIC_MOTION_V2"
)

TARGET_GOAL_V2_METADATA_KEY = (
    "semantic_v2_target_goal_stage13"
)

TARGET_GOAL_V2_PITCH_LENGTH_M = 105.0

TARGET_GOAL_V2_GOAL_CENTER_Y_M = 34.0

TARGET_GOAL_V2_EARLY_BALL_ROWS = 4

TARGET_GOAL_V2_REQUIRED_COMPLETED_PREFIX = tuple(
    CANONICAL_V2_STAGE_NAMES[
        :12
    ]
)


_NO_DEFAULT = object()


class TargetGoalV2Stage13Error(
    RuntimeError
):
    pass


class TargetGoalV2Stage13InputError(
    TargetGoalV2Stage13Error
):
    pass


class TargetGoalV2Stage13ContractError(
    TargetGoalV2Stage13Error
):
    pass


@dataclass(
    frozen=True
)
class TargetGoalV2Stage13Evidence:

    target_goal: str

    confidence: float

    left_goal_image_xy: tuple[
        float,
        float,
    ]

    right_goal_image_xy: tuple[
        float,
        float,
    ]

    early_ball_velocity_xy: tuple[
        float,
        float,
    ]

    full_endpoint_motion_xy: tuple[
        float,
        float,
    ]

    comparisons: tuple[
        Mapping[
            str,
            Any,
        ],
        ...,
    ]


def _value(
    obj,
    key,
    default=_NO_DEFAULT,
):

    if isinstance(
        obj,
        Mapping,
    ):

        if key in obj:

            return obj[
                key
            ]

    elif obj is not None and hasattr(
        obj,
        key,
    ):

        return getattr(
            obj,
            key,
        )


    if default is not _NO_DEFAULT:

        return default


    raise TargetGoalV2Stage13InputError(
        "Required field is missing: {}".format(
            key
        )
    )


def _finite_float(
    value,
    *,
    name,
) -> float:

    if isinstance(
        value,
        bool,
    ):

        raise TargetGoalV2Stage13InputError(
            "{} must be a finite real number.".format(
                name
            )
        )


    try:

        result = float(
            value
        )

    except Exception as exc:

        raise TargetGoalV2Stage13InputError(
            "{} must be a finite real number.".format(
                name
            )
        ) from exc


    if not math.isfinite(
        result
    ):

        raise TargetGoalV2Stage13InputError(
            "{} must be finite.".format(
                name
            )
        )


    return result


def _strict_int(
    value,
    *,
    name,
) -> int:

    if isinstance(
        value,
        bool,
    ):

        raise TargetGoalV2Stage13InputError(
            "{} must be an integer.".format(
                name
            )
        )


    try:

        result = int(
            value
        )

    except Exception as exc:

        raise TargetGoalV2Stage13InputError(
            "{} must be an integer.".format(
                name
            )
        ) from exc


    try:

        numeric = float(
            value
        )

    except Exception:

        numeric = float(
            result
        )


    if (
        not math.isfinite(
            numeric
        )
        or
        numeric
        !=
        float(
            result
        )
    ):

        raise TargetGoalV2Stage13InputError(
            "{} must be an exact integer.".format(
                name
            )
        )


    return result


def _two_finite_xy(
    value,
    *,
    name,
) -> tuple[
    float,
    float,
]:

    if not isinstance(
        value,
        (
            list,
            tuple,
            np.ndarray,
        ),
    ):

        raise TargetGoalV2Stage13InputError(
            "{} must contain two coordinates.".format(
                name
            )
        )


    if len(
        value
    ) != 2:

        raise TargetGoalV2Stage13InputError(
            "{} must contain exactly two coordinates.".format(
                name
            )
        )


    x = _finite_float(
        value[
            0
        ],
        name="{}.x".format(
            name
        ),
    )

    y = _finite_float(
        value[
            1
        ],
        name="{}.y".format(
            name
        ),
    )


    return (
        x,
        y,
    )


def _extract_ball_rows(
    ball_track,
) -> list[
    dict[
        str,
        Any,
    ]
]:

    if ball_track is None:

        raise TargetGoalV2Stage13InputError(
            "BallTrackResult is required."
        )


    if _value(
        ball_track,
        "success",
        False,
    ) is not True:

        raise TargetGoalV2Stage13InputError(
            "TargetGoal-v2 requires successful BallTrackResult."
        )


    track = _value(
        ball_track,
        "track",
    )


    if not isinstance(
        track,
        (
            list,
            tuple,
        ),
    ):

        raise TargetGoalV2Stage13InputError(
            "BallTrackResult.track must be a list or tuple."
        )


    if len(
        track
    ) < TARGET_GOAL_V2_EARLY_BALL_ROWS:

        raise TargetGoalV2Stage13InputError(
            "TargetGoal-v2 requires at least four ball observations."
        )


    rows = []


    for index, row in enumerate(
        track
    ):

        frame_idx = _strict_int(
            _value(
                row,
                "frame_idx",
            ),
            name="ball[{}].frame_idx".format(
                index
            ),
        )

        center_x = _finite_float(
            _value(
                row,
                "center_x",
            ),
            name="ball[{}].center_x".format(
                index
            ),
        )

        center_y = _finite_float(
            _value(
                row,
                "center_y",
            ),
            name="ball[{}].center_y".format(
                index
            ),
        )


        rows.append(
            {
                "frame_idx":
                    frame_idx,

                "center_x":
                    center_x,

                "center_y":
                    center_y,
            }
        )


    frames = [
        row[
            "frame_idx"
        ]
        for row in rows
    ]


    if len(
        set(
            frames
        )
    ) != len(
        frames
    ):

        raise TargetGoalV2Stage13ContractError(
            "Ball trajectory contains duplicate frame indices."
        )


    if any(
        later
        <=
        earlier
        for earlier, later
        in zip(
            frames,
            frames[
                1:
            ],
        )
    ):

        raise TargetGoalV2Stage13ContractError(
            "Ball trajectory frames must be strictly increasing."
        )


    return rows


def _validate_pitch_to_image_homography(
    value,
) -> np.ndarray:

    try:

        H = np.asarray(
            value,
            dtype=np.float64,
        )

    except Exception as exc:

        raise TargetGoalV2Stage13InputError(
            "Geometry-v2 pitch-to-image homography is invalid."
        ) from exc


    if H.shape != (
        3,
        3,
    ):

        raise TargetGoalV2Stage13InputError(
            "Geometry-v2 pitch-to-image homography must be 3x3."
        )


    if not np.isfinite(
        H
    ).all():

        raise TargetGoalV2Stage13InputError(
            "Geometry-v2 pitch-to-image homography must be finite."
        )


    if abs(
        float(
            np.linalg.det(
                H
            )
        )
    ) <= 1e-12:

        raise TargetGoalV2Stage13ContractError(
            "Geometry-v2 pitch-to-image homography is singular."
        )


    return H


def _project_static_pitch_point_to_image(
    H_pitch_to_image,
    pitch_xy,
) -> tuple[
    float,
    float,
]:

    H = _validate_pitch_to_image_homography(
        H_pitch_to_image
    )


    x, y = _two_finite_xy(
        pitch_xy,
        name="static_pitch_point",
    )


    projected = (
        H
        @
        np.asarray(
            [
                x,
                y,
                1.0,
            ],
            dtype=np.float64,
        )
    )


    denominator = float(
        projected[
            2
        ]
    )


    if (
        not math.isfinite(
            denominator
        )
        or
        abs(
            denominator
        )
        <=
        1e-12
    ):

        raise TargetGoalV2Stage13ContractError(
            "Static goal-center projection has invalid homogeneous denominator."
        )


    image_x = float(
        projected[
            0
        ]
        /
        denominator
    )

    image_y = float(
        projected[
            1
        ]
        /
        denominator
    )


    if not (
        math.isfinite(
            image_x
        )
        and
        math.isfinite(
            image_y
        )
    ):

        raise TargetGoalV2Stage13ContractError(
            "Static goal-center projection is non-finite."
        )


    return (
        image_x,
        image_y,
    )


def _cosine(
    a,
    b,
) -> float:

    a = np.asarray(
        a,
        dtype=np.float64,
    )

    b = np.asarray(
        b,
        dtype=np.float64,
    )


    denominator = float(
        np.linalg.norm(
            a
        )
        *
        np.linalg.norm(
            b
        )
    )


    if (
        not math.isfinite(
            denominator
        )
        or
        denominator
        <=
        1e-12
    ):

        raise TargetGoalV2Stage13ContractError(
            "Zero-length vector encountered in image-space target-goal inference."
        )


    result = float(
        np.dot(
            a,
            b,
        )
        /
        denominator
    )


    if not math.isfinite(
        result
    ):

        raise TargetGoalV2Stage13ContractError(
            "Image-space cosine is non-finite."
        )


    return result


def _comparison(
    *,
    motion_model,
    origin_name,
    motion_xy,
    origin_xy,
    left_goal_image_xy,
    right_goal_image_xy,
) -> dict[
    str,
    Any,
]:

    motion = np.asarray(
        motion_xy,
        dtype=np.float64,
    )

    origin = np.asarray(
        origin_xy,
        dtype=np.float64,
    )

    left_goal = np.asarray(
        left_goal_image_xy,
        dtype=np.float64,
    )

    right_goal = np.asarray(
        right_goal_image_xy,
        dtype=np.float64,
    )


    if (
        motion.shape
        !=
        (
            2,
        )
        or
        origin.shape
        !=
        (
            2,
        )
    ):

        raise TargetGoalV2Stage13ContractError(
            "Image-space comparison vectors must be two-dimensional."
        )


    if not (
        np.isfinite(
            motion
        ).all()
        and
        np.isfinite(
            origin
        ).all()
    ):

        raise TargetGoalV2Stage13ContractError(
            "Image-space comparison contains non-finite vectors."
        )


    left_vector = (
        left_goal
        -
        origin
    )

    right_vector = (
        right_goal
        -
        origin
    )


    left_cosine = _cosine(
        motion,
        left_vector,
    )

    right_cosine = _cosine(
        motion,
        right_vector,
    )


    left_dot = float(
        np.dot(
            motion,
            left_vector,
        )
    )

    right_dot = float(
        np.dot(
            motion,
            right_vector,
        )
    )


    if not (
        math.isfinite(
            left_dot
        )
        and
        math.isfinite(
            right_dot
        )
    ):

        raise TargetGoalV2Stage13ContractError(
            "Image-space dot product is non-finite."
        )


    if left_cosine > right_cosine:

        winner = "LEFT"

        hemisphere_separated = bool(
            left_dot
            >
            0.0
            and
            right_dot
            <
            0.0
        )

    elif right_cosine > left_cosine:

        winner = "RIGHT"

        hemisphere_separated = bool(
            right_dot
            >
            0.0
            and
            left_dot
            <
            0.0
        )

    else:

        raise TargetGoalV2Stage13ContractError(
            "Image-space target-goal comparison produced an exact tie."
        )


    return {
        "motion_model":
            str(
                motion_model
            ),

        "origin":
            str(
                origin_name
            ),

        "left_cosine":
            left_cosine,

        "right_cosine":
            right_cosine,

        "left_dot":
            left_dot,

        "right_dot":
            right_dot,

        "winner":
            winner,

        "hemisphere_separated":
            hemisphere_separated,
    }


def infer_target_goal_v2_image_space(
    ball_track,
    shooter_image_xy,
    homography_pitch_to_image,
) -> TargetGoalV2Stage13Evidence:

    rows = _extract_ball_rows(
        ball_track
    )


    shooter_xy = _two_finite_xy(
        shooter_image_xy,
        name="shooter_image_xy",
    )


    H_pitch_to_image = (
        _validate_pitch_to_image_homography(
            homography_pitch_to_image
        )
    )


    # Static goal centers only.
    #
    # The airborne ball trajectory is deliberately never projected through
    # the ground-plane homography in this policy.
    left_goal_image_xy = (
        _project_static_pitch_point_to_image(
            H_pitch_to_image,
            (
                0.0,
                TARGET_GOAL_V2_GOAL_CENTER_Y_M,
            ),
        )
    )


    right_goal_image_xy = (
        _project_static_pitch_point_to_image(
            H_pitch_to_image,
            (
                TARGET_GOAL_V2_PITCH_LENGTH_M,
                TARGET_GOAL_V2_GOAL_CENTER_Y_M,
            ),
        )
    )


    if np.linalg.norm(
        np.asarray(
            right_goal_image_xy,
            dtype=np.float64,
        )
        -
        np.asarray(
            left_goal_image_xy,
            dtype=np.float64,
        )
    ) <= 1e-12:

        raise TargetGoalV2Stage13ContractError(
            "Projected static goal centers are indistinguishable."
        )


    early_rows = rows[
        :
        TARGET_GOAL_V2_EARLY_BALL_ROWS
    ]


    early_frames = np.asarray(
        [
            row[
                "frame_idx"
            ]
            for row
            in early_rows
        ],
        dtype=np.float64,
    )


    early_x = np.asarray(
        [
            row[
                "center_x"
            ]
            for row
            in early_rows
        ],
        dtype=np.float64,
    )


    early_y = np.asarray(
        [
            row[
                "center_y"
            ]
            for row
            in early_rows
        ],
        dtype=np.float64,
    )


    early_x_fit = np.polyfit(
        early_frames,
        early_x,
        1,
    )

    early_y_fit = np.polyfit(
        early_frames,
        early_y,
        1,
    )


    early_ball_velocity_xy = (
        float(
            early_x_fit[
                0
            ]
        ),
        float(
            early_y_fit[
                0
            ]
        ),
    )


    first_ball_xy = (
        float(
            rows[
                0
            ][
                "center_x"
            ]
        ),
        float(
            rows[
                0
            ][
                "center_y"
            ]
        ),
    )


    full_endpoint_motion_xy = (
        float(
            rows[
                -1
            ][
                "center_x"
            ]
            -
            rows[
                0
            ][
                "center_x"
            ]
        ),
        float(
            rows[
                -1
            ][
                "center_y"
            ]
            -
            rows[
                0
            ][
                "center_y"
            ]
        ),
    )


    for name, vector in (
        (
            "early_ball_velocity_xy",
            early_ball_velocity_xy,
        ),
        (
            "full_endpoint_motion_xy",
            full_endpoint_motion_xy,
        ),
    ):

        if (
            np.linalg.norm(
                np.asarray(
                    vector,
                    dtype=np.float64,
                )
            )
            <=
            1e-12
        ):

            raise TargetGoalV2Stage13ContractError(
                "{} has zero magnitude.".format(
                    name
                )
            )


    comparisons = []


    for motion_model, motion_xy in (
        (
            "EARLY_LINEAR",
            early_ball_velocity_xy,
        ),
        (
            "FULL_ENDPOINT",
            full_endpoint_motion_xy,
        ),
    ):

        for origin_name, origin_xy in (
            (
                "SHOOTER_FOOT",
                shooter_xy,
            ),
            (
                "FIRST_BALL",
                first_ball_xy,
            ),
        ):

            comparisons.append(
                _comparison(
                    motion_model=motion_model,
                    origin_name=origin_name,
                    motion_xy=motion_xy,
                    origin_xy=origin_xy,
                    left_goal_image_xy=left_goal_image_xy,
                    right_goal_image_xy=right_goal_image_xy,
                )
            )


    winners = tuple(
        row[
            "winner"
        ]
        for row
        in comparisons
    )


    if len(
        set(
            winners
        )
    ) != 1:

        raise TargetGoalV2Stage13ContractError(
            "TargetGoal-v2 image-space comparisons do not agree."
        )


    if not all(
        row[
            "hemisphere_separated"
        ]
        is True
        for row
        in comparisons
    ):

        raise TargetGoalV2Stage13ContractError(
            "TargetGoal-v2 forward/backward hemisphere separation failed."
        )


    target_goal = winners[
        0
    ]


    if target_goal not in {
        "LEFT",
        "RIGHT",
    }:

        raise TargetGoalV2Stage13ContractError(
            "TargetGoal-v2 resolved an unsupported goal side."
        )


    # Frozen Semantic-v2 confidence:
    #
    # four independent image-space comparisons agree and every comparison
    # has strict forward/backward hemisphere separation.
    confidence = 1.0


    return TargetGoalV2Stage13Evidence(
        target_goal=target_goal,
        confidence=confidence,
        left_goal_image_xy=left_goal_image_xy,
        right_goal_image_xy=right_goal_image_xy,
        early_ball_velocity_xy=early_ball_velocity_xy,
        full_endpoint_motion_xy=full_endpoint_motion_xy,
        comparisons=tuple(
            comparisons
        ),
    )


def _validate_stage13_entry(
    result,
):

    if result is None:

        raise TargetGoalV2Stage13InputError(
            "Stage-13 result object is required."
        )


    pipeline_status = _value(
        result,
        "pipeline_status",
    )


    completed_stages = _value(
        pipeline_status,
        "completed_stages",
    )


    if not isinstance(
        completed_stages,
        list,
    ):

        raise TargetGoalV2Stage13InputError(
            "pipeline_status.completed_stages must be a list."
        )


    if tuple(
        completed_stages
    ) != TARGET_GOAL_V2_REQUIRED_COMPLETED_PREFIX:

        raise TargetGoalV2Stage13InputError(
            "Stage-13 requires the exact completed Stage-1..12 prefix."
        )


    if _value(
        pipeline_status,
        "failed_stage",
        None,
    ) is not None:

        raise TargetGoalV2Stage13InputError(
            "Stage-13 cannot execute after a failed stage."
        )


    ball_track = _value(
        result,
        "ball_track",
    )


    if _value(
        ball_track,
        "success",
        False,
    ) is not True:

        raise TargetGoalV2Stage13InputError(
            "Stage-13 requires successful ball tracking."
        )


    geometry_v2 = _value(
        result,
        "geometry_v2",
    )


    if geometry_v2 is None:

        raise TargetGoalV2Stage13InputError(
            "Stage-13 requires Geometry-v2."
        )


    if _value(
        geometry_v2,
        "success",
        False,
    ) is not True:

        raise TargetGoalV2Stage13InputError(
            "Stage-13 requires successful Geometry-v2."
        )


    if _value(
        geometry_v2,
        "reference_free",
        False,
    ) is not True:

        raise TargetGoalV2Stage13ContractError(
            "Stage-13 requires reference-free Geometry-v2."
        )


    H_pitch_to_image = (
        _validate_pitch_to_image_homography(
            _value(
                geometry_v2,
                "homography_pitch_to_image",
            )
        )
    )


    shooter = _value(
        result,
        "shooter",
    )


    if _value(
        shooter,
        "success",
        False,
    ) is not True:

        raise TargetGoalV2Stage13InputError(
            "Stage-13 requires a resolved public ShooterResult."
        )


    shooter_track_id = _strict_int(
        _value(
            shooter,
            "track_id",
        ),
        name="shooter.track_id",
    )


    shooter_image_xy = _two_finite_xy(
        _value(
            shooter,
            "image_xy",
        ),
        name="shooter.image_xy",
    )


    shooter_v2 = _value(
        result,
        "shooter_v2",
    )


    if shooter_v2 is None:

        raise TargetGoalV2Stage13InputError(
            "Stage-13 requires Shooter-v2 resolution."
        )


    resolved_track_id = _strict_int(
        _value(
            shooter_v2,
            "resolved_track_id",
        ),
        name="shooter_v2.resolved_track_id",
    )


    if resolved_track_id != shooter_track_id:

        raise TargetGoalV2Stage13ContractError(
            "Shooter-v2 resolved track does not match public ShooterResult."
        )


    shooter_resolution = str(
        _value(
            shooter_v2,
            "resolution",
        )
    )


    if shooter_resolution not in {
        "AUTO_SELECTION",
        "USER_CONFIRMATION",
    }:

        raise TargetGoalV2Stage13InputError(
            "Stage-13 requires resolved Shooter-v2 semantics."
        )


    relative_teams = _value(
        result,
        "relative_teams",
    )


    if _value(
        relative_teams,
        "success",
        False,
    ) is not True:

        raise TargetGoalV2Stage13InputError(
            "Stage-13 requires completed relative-team assignment."
        )


    target_goal = _value(
        result,
        "target_goal",
    )


    if not isinstance(
        target_goal,
        TargetGoalResult,
    ):

        raise TargetGoalV2Stage13InputError(
            "result.target_goal must be TargetGoalResult."
        )


    if (
        target_goal.success
        is not False
        or
        target_goal.target_goal
        is not None
        or
        target_goal.confidence
        is not None
        or
        target_goal.failure_reason
        is not None
    ):

        raise TargetGoalV2Stage13ContractError(
            "TargetGoalResult was unexpectedly populated before Stage 13."
        )


    metadata = _value(
        result,
        "metadata",
    )


    if not isinstance(
        metadata,
        dict,
    ):

        raise TargetGoalV2Stage13InputError(
            "result.metadata must be a dict."
        )


    if TARGET_GOAL_V2_METADATA_KEY in metadata:

        raise TargetGoalV2Stage13ContractError(
            "Stage-13 metadata is already populated."
        )


    if (
        "semantic_v2_shooter_stage11"
        not in
        metadata
    ):

        raise TargetGoalV2Stage13InputError(
            "Stage-13 requires Shooter-v2 Stage-11 provenance."
        )


    return {
        "ball_track":
            ball_track,

        "H_pitch_to_image":
            H_pitch_to_image,

        "shooter_image_xy":
            shooter_image_xy,

        "shooter_track_id":
            shooter_track_id,

        "shooter_resolution":
            shooter_resolution,

        "metadata":
            metadata,

        "geometry_v2":
            geometry_v2,

        "shooter":
            shooter,

        "shooter_v2":
            shooter_v2,

        "relative_teams":
            relative_teams,
    }


def build_target_goal_v2_stage13_inference_handler():

    def handler(
        result,
    ):

        entry = _validate_stage13_entry(
            result
        )


        evidence = (
            infer_target_goal_v2_image_space(
                entry[
                    "ball_track"
                ],
                entry[
                    "shooter_image_xy"
                ],
                entry[
                    "H_pitch_to_image"
                ],
            )
        )


        result.target_goal.success = True

        result.target_goal.target_goal = (
            evidence.target_goal
        )

        result.target_goal.confidence = float(
            evidence.confidence
        )

        result.target_goal.failure_reason = None


        comparison_rows = [
            dict(
                row
            )
            for row
            in evidence.comparisons
        ]


        result.metadata[
            TARGET_GOAL_V2_METADATA_KEY
        ] = {
            "stage":
                TARGET_GOAL_V2_STAGE13_NAME,

            "action":
                TARGET_GOAL_V2_STAGE13_ACTION,

            "policy":
                TARGET_GOAL_V2_POLICY,

            "target_goal":
                evidence.target_goal,

            "confidence":
                float(
                    evidence.confidence
                ),

            "shooter_track_id":
                int(
                    entry[
                        "shooter_track_id"
                    ]
                ),

            "shooter_resolution":
                entry[
                    "shooter_resolution"
                ],

            "goal_centers_pitch_xy":
                {
                    "LEFT":
                        [
                            0.0,
                            TARGET_GOAL_V2_GOAL_CENTER_Y_M,
                        ],

                    "RIGHT":
                        [
                            TARGET_GOAL_V2_PITCH_LENGTH_M,
                            TARGET_GOAL_V2_GOAL_CENTER_Y_M,
                        ],
                },

            "goal_centers_image_xy":
                {
                    "LEFT":
                        list(
                            evidence.left_goal_image_xy
                        ),

                    "RIGHT":
                        list(
                            evidence.right_goal_image_xy
                        ),
                },

            "early_ball_row_count":
                TARGET_GOAL_V2_EARLY_BALL_ROWS,

            "early_ball_velocity_xy":
                list(
                    evidence.early_ball_velocity_xy
                ),

            "full_endpoint_motion_xy":
                list(
                    evidence.full_endpoint_motion_xy
                ),

            "comparison_count":
                len(
                    comparison_rows
                ),

            "comparison_winners":
                [
                    row[
                        "winner"
                    ]
                    for row
                    in comparison_rows
                ],

            "all_comparisons_agree":
                True,

            "hemisphere_separated_all":
                all(
                    row[
                        "hemisphere_separated"
                    ]
                    is True
                    for row
                    in comparison_rows
                ),

            "comparisons":
                comparison_rows,

            # Semantic-v2 safety provenance.
            "airborne_ball_projected_through_homography":
                False,

            "static_goal_centers_projected_with_homography":
                True,

            "homography_direction":
                "PITCH_TO_IMAGE_STATIC_GOALS_ONLY",

            "manual_target_goal_used":
                False,

            "historical_target_goal_used":
                False,

            "shooter_goal_distance_used_for_direction":
                False,

            "shooter_pitch_xy_used_for_direction":
                False,

            "shooter_image_xy_used_as_origin":
                True,

            "relative_team_labels_used_for_direction":
                False,

            "defender_positions_used_for_direction":
                False,

            "shot_geometry_used_for_direction":
                False,

            "xg_used_for_direction":
                False,

            "reference_homography_used":
                False,

            "geometry_variant_reselected":
                False,

            "geometry_service_called":
                False,

            "new_model_inference_performed":
                False,

            "frozen_v1_target_goal_handler_used":
                False,

            "public_target_goal_result_written":
                True,

            "geometry_object_replaced":
                False,

            "shooter_object_replaced":
                False,

            "relative_teams_object_replaced":
                False,

            "ball_track_object_replaced":
                False,
        }


        # Canonical runner owns stage completion.
        # Stage-13 handler deliberately does NOT mutate pipeline_status.
        return None


    return handler
