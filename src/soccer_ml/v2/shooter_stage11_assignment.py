from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from soccer_ml.results import ShooterResult

from .semantic_contracts import (
    CANONICAL_V2_STAGE_NAMES,
    SHOOTER_RESOLUTION_AUTO,
    ShooterV2Decision,
    validate_shooter_v2_decision,
)


SHOOTER_V2_STAGE11_NAME = "shooter_assignment"

SHOOTER_V2_STAGE11_ACTION = (
    "REPLACE_V2_AMBIGUITY_SAFE_SHOOTER_GATE"
)

SHOOTER_V2_POLICY_NAME = (
    "SHOOTER_V2_TEMPORAL_LAUNCH_ORIGIN_CANDIDATE_V1"
)

SHOOTER_V2_POLICY_VERSION = "1.0.0"

SHOOTER_V2_POLICY_SEMANTIC_SHA256 = (
    "2a87111ea4a01c365067ddf94150169ddc284b762eb3e11c26218ac7b5cf46bc"
)

SHOOTER_V2_REQUIRED_MARGIN_FRACTION = 0.10

SHOOTER_V2_FIRST_BALL_ROWS = 4

SHOOTER_V2_MAX_EARLY_PLAYER_OBSERVATIONS = 4

SHOOTER_V2_MAX_FIRST_FRAME_OFFSET = 1

SHOOTER_V2_MIN_FIRST_FOUR_OBSERVATIONS = 3

SHOOTER_V2_MIN_OVERLAP_FRAMES = 4

SHOOTER_V2_BACKPROJECTION_FRAMES = 8

SHOOTER_V2_REQUIRED_COMPLETED_PREFIX = tuple(
    CANONICAL_V2_STAGE_NAMES[:10]
)

SHOOTER_V2_WEIGHTS = {
    "contact": 0.55,
    "departure": 0.25,
    "monotonic": 0.15,
    "support": 0.05,
}

SHOOTER_V2_TIE_BREAK_ORDER = (
    "final_score_desc",
    "minimum_backprojected_distance_norm_asc",
    "post_launch_distance_slope_desc",
    "track_id_asc",
)

SHOOTER_V2_FORBIDDEN_AUTOMATIC_INPUTS = (
    "human_shooter_track_id",
    "jersey_number",
    "team_label",
    "xg",
    "expected_shot_distance",
    "target_goal",
    "reference_homography",
    "geometry_expected_outcome",
    "track_specific_prior",
)

SHOOTER_V2_FEATURE_FIELDS = (
    "track_id",
    "first_player_frame",
    "player_observation_count",
    "ball_player_overlap_count",
    "minimum_backprojected_distance_norm",
    "minimum_backprojected_distance_px",
    "best_backprojection_lag_frames",
    "best_backprojection_frame",
    "post_launch_distance_slope_norm_per_frame",
    "separation_nondecreasing_fraction",
    "temporal_support_fraction",
    "first_observed_distance_norm",
    "last_observed_distance_norm",
    "observed_separation_gain_norm",
    "score_contact_rank",
    "score_departure_rank",
    "score_monotonic_rank",
    "score_support_rank",
    "final_score",
    "blind_rank",
)


class ShooterV2Stage11Error(RuntimeError):
    pass


class ShooterV2Stage11InputError(ShooterV2Stage11Error):
    pass


class ShooterV2Stage11ContractError(ShooterV2Stage11Error):
    pass


@dataclass(frozen=True)
class ShooterV2Stage11Evidence:

    candidate_count: int

    auto_top_track_id: Optional[int]

    auto_second_track_id: Optional[int]

    auto_margin_fraction: Optional[float]

    required_margin_fraction: float

    auto_status: str

    resolution: str

    resolved_track_id: Optional[int]

    manual_confirmation_used: bool

    confirmation_used_as_model_evidence: bool


_NO_DEFAULT = object()


def _value(
    value: Any,
    key: str,
    default: Any = _NO_DEFAULT,
) -> Any:

    if isinstance(
        value,
        Mapping,
    ):

        if key in value:
            return value[key]

    else:

        if hasattr(
            value,
            key,
        ):
            return getattr(
                value,
                key,
            )

    if default is not _NO_DEFAULT:
        return default

    raise ShooterV2Stage11InputError(
        f"Required field is missing: {key}"
    )


def _strict_int(
    value: Any,
    *,
    field_name: str,
    minimum: Optional[int] = None,
) -> int:

    if isinstance(
        value,
        bool,
    ):

        raise ShooterV2Stage11InputError(
            f"{field_name} must be an integer, not bool."
        )

    try:
        converted = int(
            value
        )
    except Exception as exc:
        raise ShooterV2Stage11InputError(
            f"{field_name} must be an integer."
        ) from exc

    if converted != value:

        try:
            if float(
                converted
            ) != float(
                value
            ):
                raise ShooterV2Stage11InputError(
                    f"{field_name} is not an exact integer."
                )

        except Exception as exc:
            raise ShooterV2Stage11InputError(
                f"{field_name} is not an exact integer."
            ) from exc

    if (
        minimum is not None
        and
        converted < minimum
    ):

        raise ShooterV2Stage11InputError(
            f"{field_name} must be >= {minimum}."
        )

    return converted


def _finite_float(
    value: Any,
    *,
    field_name: str,
) -> float:

    try:
        converted = float(
            value
        )
    except Exception as exc:
        raise ShooterV2Stage11InputError(
            f"{field_name} must be numeric."
        ) from exc

    if not math.isfinite(
        converted
    ):

        raise ShooterV2Stage11InputError(
            f"{field_name} must be finite."
        )

    return converted


def _fit_linear(
    frames: Sequence[int],
    values: Sequence[float],
) -> tuple[float, float]:

    if len(
        frames
    ) != len(
        values
    ):

        raise ShooterV2Stage11ContractError(
            "Linear fit frame/value lengths disagree."
        )

    if len(
        frames
    ) < 2:

        raise ShooterV2Stage11InputError(
            "At least two observations are required for linear motion fit."
        )

    frame_array = np.asarray(
        frames,
        dtype=np.float64,
    )

    value_array = np.asarray(
        values,
        dtype=np.float64,
    )

    if not (
        np.isfinite(
            frame_array
        ).all()
        and
        np.isfinite(
            value_array
        ).all()
    ):

        raise ShooterV2Stage11InputError(
            "Linear fit data must be finite."
        )

    slope, intercept = np.polyfit(
        frame_array,
        value_array,
        1,
    )

    slope = float(
        slope
    )

    intercept = float(
        intercept
    )

    if not (
        math.isfinite(
            slope
        )
        and
        math.isfinite(
            intercept
        )
    ):

        raise ShooterV2Stage11ContractError(
            "Linear fit produced non-finite parameters."
        )

    return (
        slope,
        intercept,
    )


def _predict_linear(
    fit: tuple[float, float],
    frame_idx: int,
) -> float:

    slope, intercept = fit

    value = (
        slope
        *
        float(
            frame_idx
        )
        +
        intercept
    )

    if not math.isfinite(
        value
    ):

        raise ShooterV2Stage11ContractError(
            "Linear prediction is non-finite."
        )

    return float(
        value
    )


def _point_to_bbox_distance(
    point_x: float,
    point_y: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:

    if not (
        x2 >= x1
        and
        y2 >= y1
    ):

        raise ShooterV2Stage11InputError(
            "Player bbox coordinates are invalid."
        )

    dx = max(
        x1 - point_x,
        0.0,
        point_x - x2,
    )

    dy = max(
        y1 - point_y,
        0.0,
        point_y - y2,
    )

    distance = math.hypot(
        dx,
        dy,
    )

    if not math.isfinite(
        distance
    ):

        raise ShooterV2Stage11ContractError(
            "Point-to-bbox distance is non-finite."
        )

    return float(
        distance
    )


def _extract_ball_rows(
    ball_track: Any,
) -> list[dict[str, Any]]:

    if ball_track is None:

        raise ShooterV2Stage11InputError(
            "Stage 11 requires ball_track."
        )

    if _value(
        ball_track,
        "success",
        False,
    ) is not True:

        raise ShooterV2Stage11InputError(
            "Stage 11 requires successful ball_track."
        )

    raw_track = _value(
        ball_track,
        "track",
    )

    if not isinstance(
        raw_track,
        (list, tuple),
    ):

        raise ShooterV2Stage11InputError(
            "ball_track.track must be a list or tuple."
        )

    rows = []

    seen_frames = set()

    for index, raw_row in enumerate(
        raw_track
    ):

        frame_idx = _strict_int(
            _value(
                raw_row,
                "frame_idx",
            ),
            field_name=(
                f"ball_track.track[{index}].frame_idx"
            ),
            minimum=0,
        )

        if frame_idx in seen_frames:

            raise ShooterV2Stage11ContractError(
                "Duplicate ball-track frame encountered."
            )

        center_x = _finite_float(
            _value(
                raw_row,
                "center_x",
            ),
            field_name=(
                f"ball_track.track[{index}].center_x"
            ),
        )

        center_y = _finite_float(
            _value(
                raw_row,
                "center_y",
            ),
            field_name=(
                f"ball_track.track[{index}].center_y"
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

        seen_frames.add(
            frame_idx
        )

    rows.sort(
        key=lambda row:
            row[
                "frame_idx"
            ]
    )

    if len(
        rows
    ) < SHOOTER_V2_FIRST_BALL_ROWS:

        raise ShooterV2Stage11InputError(
            "Shooter-v2 requires at least four canonical ball rows."
        )

    return rows


def _normalized_player_observation(
    raw_observation: Any,
    *,
    parent_track_id: int,
    observation_index: int,
) -> dict[str, Any]:

    frame_idx = _strict_int(
        _value(
            raw_observation,
            "frame_idx",
        ),
        field_name=(
            f"player observation {observation_index}.frame_idx"
        ),
        minimum=0,
    )

    observation_track_id = _strict_int(
        _value(
            raw_observation,
            "track_id",
            parent_track_id,
        ),
        field_name=(
            f"player observation {observation_index}.track_id"
        ),
        minimum=0,
    )

    if observation_track_id != parent_track_id:

        raise ShooterV2Stage11ContractError(
            "Player observation track_id disagrees with parent track."
        )

    x1 = _finite_float(
        _value(
            raw_observation,
            "x1",
        ),
        field_name="player observation x1",
    )

    y1 = _finite_float(
        _value(
            raw_observation,
            "y1",
        ),
        field_name="player observation y1",
    )

    x2 = _finite_float(
        _value(
            raw_observation,
            "x2",
        ),
        field_name="player observation x2",
    )

    y2 = _finite_float(
        _value(
            raw_observation,
            "y2",
        ),
        field_name="player observation y2",
    )

    if not (
        x2 > x1
        and
        y2 > y1
    ):

        raise ShooterV2Stage11InputError(
            "Player observation bbox must have positive width and height."
        )

    return {
        "track_id":
            parent_track_id,

        "frame_idx":
            frame_idx,

        "x1":
            x1,

        "y1":
            y1,

        "x2":
            x2,

        "y2":
            y2,
    }


def _extract_player_tracks(
    player_tracks: Any,
) -> dict[int, list[dict[str, Any]]]:

    if player_tracks is None:

        raise ShooterV2Stage11InputError(
            "Stage 11 requires player_tracks."
        )

    if _value(
        player_tracks,
        "success",
        False,
    ) is not True:

        raise ShooterV2Stage11InputError(
            "Stage 11 requires successful player_tracks."
        )

    raw_tracks = _value(
        player_tracks,
        "tracks",
    )

    normalized: dict[
        int,
        list[dict[str, Any]],
    ] = {}


    if isinstance(
        raw_tracks,
        Mapping,
    ):

        iterator = []

        for raw_track_id, raw_observations in raw_tracks.items():

            track_id = _strict_int(
                raw_track_id,
                field_name="player track mapping key",
                minimum=0,
            )

            iterator.append(
                (
                    track_id,
                    raw_observations,
                )
            )


    elif isinstance(
        raw_tracks,
        (list, tuple),
    ):

        iterator = []

        for track_index, raw_track in enumerate(
            raw_tracks
        ):

            track_id = _strict_int(
                _value(
                    raw_track,
                    "track_id",
                ),
                field_name=(
                    f"player_tracks.tracks[{track_index}].track_id"
                ),
                minimum=0,
            )

            raw_observations = _value(
                raw_track,
                "observations",
            )

            iterator.append(
                (
                    track_id,
                    raw_observations,
                )
            )


    else:

        raise ShooterV2Stage11InputError(
            "player_tracks.tracks must be mapping, list, or tuple."
        )


    for track_id, raw_observations in iterator:

        if track_id in normalized:

            raise ShooterV2Stage11ContractError(
                "Duplicate player track_id encountered."
            )

        if not isinstance(
            raw_observations,
            (list, tuple),
        ):

            raise ShooterV2Stage11InputError(
                "Player observations must be a list or tuple."
            )

        observations = []

        seen_frames = set()

        for observation_index, raw_observation in enumerate(
            raw_observations
        ):

            observation = _normalized_player_observation(
                raw_observation,
                parent_track_id=track_id,
                observation_index=observation_index,
            )

            frame_idx = observation[
                "frame_idx"
            ]

            if frame_idx in seen_frames:

                raise ShooterV2Stage11ContractError(
                    "Duplicate observation frame within player track."
                )

            seen_frames.add(
                frame_idx
            )

            observations.append(
                observation
            )

        observations.sort(
            key=lambda row:
                row[
                    "frame_idx"
                ]
        )

        if observations:

            normalized[
                track_id
            ] = observations


    if not normalized:

        raise ShooterV2Stage11InputError(
            "No usable player tracks are available."
        )

    return normalized


def _candidate_is_eligible(
    observations: Sequence[Mapping[str, Any]],
    *,
    ball_frames: Sequence[int],
    ball_start_frame: int,
) -> bool:

    if not observations:

        return False

    first_player_frame = int(
        observations[
            0
        ][
            "frame_idx"
        ]
    )

    if (
        first_player_frame
        >
        ball_start_frame
        +
        SHOOTER_V2_MAX_FIRST_FRAME_OFFSET
    ):

        return False

    first_four_ball_frames = set(
        ball_frames[
            :SHOOTER_V2_FIRST_BALL_ROWS
        ]
    )

    observation_frames = {
        int(
            row[
                "frame_idx"
            ]
        )
        for row in observations
    }

    early_count = len(
        observation_frames
        &
        first_four_ball_frames
    )

    if (
        early_count
        <
        SHOOTER_V2_MIN_FIRST_FOUR_OBSERVATIONS
    ):

        return False

    overlap_count = len(
        observation_frames
        &
        set(
            ball_frames
        )
    )

    if (
        overlap_count
        <
        SHOOTER_V2_MIN_OVERLAP_FRAMES
    ):

        return False

    return True


def _candidate_features(
    track_id: int,
    observations: Sequence[Mapping[str, Any]],
    *,
    ball_rows: Sequence[Mapping[str, Any]],
    launch_x_fit: tuple[float, float],
    launch_y_fit: tuple[float, float],
) -> dict[str, Any]:

    ball_by_frame = {
        int(
            row[
                "frame_idx"
            ]
        ):
            row
        for row in ball_rows
    }

    ball_frames = sorted(
        ball_by_frame.keys()
    )

    player_by_frame = {
        int(
            row[
                "frame_idx"
            ]
        ):
            row
        for row in observations
    }

    player_frames = sorted(
        player_by_frame.keys()
    )

    early_observations = list(
        observations[
            :SHOOTER_V2_MAX_EARLY_PLAYER_OBSERVATIONS
        ]
    )

    if len(
        early_observations
    ) < 2:

        raise ShooterV2Stage11InputError(
            "Eligible player requires at least two early observations."
        )

    early_frames = [
        int(
            row[
                "frame_idx"
            ]
        )
        for row in early_observations
    ]

    player_fits = {}

    for coordinate in (
        "x1",
        "y1",
        "x2",
        "y2",
    ):

        player_fits[
            coordinate
        ] = _fit_linear(
            early_frames,
            [
                float(
                    row[
                        coordinate
                    ]
                )
                for row
                in early_observations
            ],
        )


    ball_start_frame = int(
        ball_rows[
            0
        ][
            "frame_idx"
        ]
    )

    backprojection_candidates = []


    for lag in range(
        SHOOTER_V2_BACKPROJECTION_FRAMES
        +
        1
    ):

        frame_idx = (
            ball_start_frame
            -
            lag
        )

        ball_x = _predict_linear(
            launch_x_fit,
            frame_idx,
        )

        ball_y = _predict_linear(
            launch_y_fit,
            frame_idx,
        )

        x1 = _predict_linear(
            player_fits[
                "x1"
            ],
            frame_idx,
        )

        y1 = _predict_linear(
            player_fits[
                "y1"
            ],
            frame_idx,
        )

        x2 = _predict_linear(
            player_fits[
                "x2"
            ],
            frame_idx,
        )

        y2 = _predict_linear(
            player_fits[
                "y2"
            ],
            frame_idx,
        )

        predicted_height = (
            y2
            -
            y1
        )

        if (
            x2 <= x1
            or
            predicted_height <= 0.0
        ):

            continue

        distance_px = _point_to_bbox_distance(
            ball_x,
            ball_y,
            x1,
            y1,
            x2,
            y2,
        )

        distance_norm = (
            distance_px
            /
            predicted_height
        )

        if not math.isfinite(
            distance_norm
        ):

            raise ShooterV2Stage11ContractError(
                "Backprojected normalized distance is non-finite."
            )

        backprojection_candidates.append(
            {
                "lag":
                    int(
                        lag
                    ),

                "frame_idx":
                    int(
                        frame_idx
                    ),

                "distance_px":
                    float(
                        distance_px
                    ),

                "distance_norm":
                    float(
                        distance_norm
                    ),
            }
        )


    if not backprojection_candidates:

        raise ShooterV2Stage11InputError(
            "No valid backprojected player bbox was available."
        )


    backprojection_candidates.sort(
        key=lambda row:
            (
                row[
                    "distance_norm"
                ],
                row[
                    "lag"
                ],
            )
    )


    best_contact = backprojection_candidates[
        0
    ]


    overlap_frames = sorted(
        set(
            player_frames
        )
        &
        set(
            ball_frames
        )
    )


    if (
        len(
            overlap_frames
        )
        <
        SHOOTER_V2_MIN_OVERLAP_FRAMES
    ):

        raise ShooterV2Stage11ContractError(
            "Eligible candidate lost required ball/player overlap."
        )


    observed_distances = []


    for frame_idx in overlap_frames:

        ball_row = ball_by_frame[
            frame_idx
        ]

        player_row = player_by_frame[
            frame_idx
        ]

        height = (
            float(
                player_row[
                    "y2"
                ]
            )
            -
            float(
                player_row[
                    "y1"
                ]
            )
        )

        if height <= 0.0:

            raise ShooterV2Stage11InputError(
                "Observed player bbox height must be positive."
            )

        distance_px = _point_to_bbox_distance(
            float(
                ball_row[
                    "center_x"
                ]
            ),
            float(
                ball_row[
                    "center_y"
                ]
            ),
            float(
                player_row[
                    "x1"
                ]
            ),
            float(
                player_row[
                    "y1"
                ]
            ),
            float(
                player_row[
                    "x2"
                ]
            ),
            float(
                player_row[
                    "y2"
                ]
            ),
        )

        observed_distances.append(
            float(
                distance_px
                /
                height
            )
        )


    distance_slope, _ = _fit_linear(
        overlap_frames,
        observed_distances,
    )


    if len(
        observed_distances
    ) >= 2:

        deltas = np.diff(
            np.asarray(
                observed_distances,
                dtype=np.float64,
            )
        )

        nondecreasing_fraction = float(
            np.mean(
                deltas
                >=
                0.0
            )
        )

    else:

        nondecreasing_fraction = 0.0


    temporal_support_fraction = float(
        len(
            overlap_frames
        )
        /
        len(
            ball_frames
        )
    )


    first_distance = float(
        observed_distances[
            0
        ]
    )

    last_distance = float(
        observed_distances[
            -1
        ]
    )


    return {
        "track_id":
            int(
                track_id
            ),

        "first_player_frame":
            int(
                player_frames[
                    0
                ]
            ),

        "player_observation_count":
            int(
                len(
                    observations
                )
            ),

        "ball_player_overlap_count":
            int(
                len(
                    overlap_frames
                )
            ),

        "minimum_backprojected_distance_norm":
            float(
                best_contact[
                    "distance_norm"
                ]
            ),

        "minimum_backprojected_distance_px":
            float(
                best_contact[
                    "distance_px"
                ]
            ),

        "best_backprojection_lag_frames":
            int(
                best_contact[
                    "lag"
                ]
            ),

        "best_backprojection_frame":
            int(
                best_contact[
                    "frame_idx"
                ]
            ),

        "post_launch_distance_slope_norm_per_frame":
            float(
                distance_slope
            ),

        "separation_nondecreasing_fraction":
            float(
                nondecreasing_fraction
            ),

        "temporal_support_fraction":
            float(
                temporal_support_fraction
            ),

        "first_observed_distance_norm":
            first_distance,

        "last_observed_distance_norm":
            last_distance,

        "observed_separation_gain_norm":
            float(
                last_distance
                -
                first_distance
            ),
    }


def _ordinal_rank_scores(
    values_by_track: Mapping[int, float],
    *,
    higher_is_better: bool,
) -> dict[int, float]:

    if not values_by_track:

        raise ShooterV2Stage11InputError(
            "Cannot rank an empty candidate mapping."
        )

    normalized_values = {
        int(
            track_id
        ):
            float(
                value
            )
        for track_id, value
        in values_by_track.items()
    }

    if not all(
        math.isfinite(
            value
        )
        for value in normalized_values.values()
    ):

        raise ShooterV2Stage11InputError(
            "Candidate ranking values must be finite."
        )

    candidate_count = len(
        normalized_values
    )

    if candidate_count == 1:

        only_track = next(
            iter(
                normalized_values
            )
        )

        return {
            only_track:
                1.0
        }


    scores = {}


    for track_id, value in normalized_values.items():

        if higher_is_better:

            strictly_better = sum(
                other_value
                >
                value
                for other_value
                in normalized_values.values()
            )

        else:

            strictly_better = sum(
                other_value
                <
                value
                for other_value
                in normalized_values.values()
            )

        rank = (
            1
            +
            strictly_better
        )

        score = (
            candidate_count
            -
            rank
        ) / (
            candidate_count
            -
            1
        )

        scores[
            track_id
        ] = float(
            score
        )


    return scores


def _rank_feature_rows(
    base_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:

    if not base_rows:

        raise ShooterV2Stage11InputError(
            "No eligible Shooter-v2 candidates were produced."
        )

    track_ids = [
        int(
            row[
                "track_id"
            ]
        )
        for row in base_rows
    ]

    if len(
        track_ids
    ) != len(
        set(
            track_ids
        )
    ):

        raise ShooterV2Stage11ContractError(
            "Duplicate candidate track_id encountered."
        )


    contact_scores = _ordinal_rank_scores(
        {
            int(
                row[
                    "track_id"
                ]
            ):
                float(
                    row[
                        "minimum_backprojected_distance_norm"
                    ]
                )
            for row in base_rows
        },
        higher_is_better=False,
    )


    departure_scores = _ordinal_rank_scores(
        {
            int(
                row[
                    "track_id"
                ]
            ):
                float(
                    row[
                        "post_launch_distance_slope_norm_per_frame"
                    ]
                )
            for row in base_rows
        },
        higher_is_better=True,
    )


    monotonic_scores = _ordinal_rank_scores(
        {
            int(
                row[
                    "track_id"
                ]
            ):
                float(
                    row[
                        "separation_nondecreasing_fraction"
                    ]
                )
            for row in base_rows
        },
        higher_is_better=True,
    )


    support_scores = _ordinal_rank_scores(
        {
            int(
                row[
                    "track_id"
                ]
            ):
                float(
                    row[
                        "temporal_support_fraction"
                    ]
                )
            for row in base_rows
        },
        higher_is_better=True,
    )


    ranked = []


    for row in base_rows:

        track_id = int(
            row[
                "track_id"
            ]
        )

        output = dict(
            row
        )

        output[
            "score_contact_rank"
        ] = float(
            contact_scores[
                track_id
            ]
        )

        output[
            "score_departure_rank"
        ] = float(
            departure_scores[
                track_id
            ]
        )

        output[
            "score_monotonic_rank"
        ] = float(
            monotonic_scores[
                track_id
            ]
        )

        output[
            "score_support_rank"
        ] = float(
            support_scores[
                track_id
            ]
        )

        output[
            "final_score"
        ] = float(
            SHOOTER_V2_WEIGHTS[
                "contact"
            ]
            *
            output[
                "score_contact_rank"
            ]
            +
            SHOOTER_V2_WEIGHTS[
                "departure"
            ]
            *
            output[
                "score_departure_rank"
            ]
            +
            SHOOTER_V2_WEIGHTS[
                "monotonic"
            ]
            *
            output[
                "score_monotonic_rank"
            ]
            +
            SHOOTER_V2_WEIGHTS[
                "support"
            ]
            *
            output[
                "score_support_rank"
            ]
        )

        ranked.append(
            output
        )


    ranked.sort(
        key=lambda row:
            (
                -float(
                    row[
                        "final_score"
                    ]
                ),
                float(
                    row[
                        "minimum_backprojected_distance_norm"
                    ]
                ),
                -float(
                    row[
                        "post_launch_distance_slope_norm_per_frame"
                    ]
                ),
                int(
                    row[
                        "track_id"
                    ]
                ),
            )
    )


    for index, row in enumerate(
        ranked,
        start=1,
    ):

        row[
            "blind_rank"
        ] = int(
            index
        )


    return ranked


def score_shooter_v2_candidates(
    ball_track: Any,
    player_tracks: Any,
) -> list[dict[str, Any]]:

    ball_rows = _extract_ball_rows(
        ball_track
    )

    normalized_tracks = _extract_player_tracks(
        player_tracks
    )

    ball_frames = [
        int(
            row[
                "frame_idx"
            ]
        )
        for row in ball_rows
    ]

    ball_start_frame = int(
        ball_frames[
            0
        ]
    )


    launch_rows = ball_rows[
        :SHOOTER_V2_FIRST_BALL_ROWS
    ]

    launch_frames = [
        int(
            row[
                "frame_idx"
            ]
        )
        for row in launch_rows
    ]


    launch_x_fit = _fit_linear(
        launch_frames,
        [
            float(
                row[
                    "center_x"
                ]
            )
            for row in launch_rows
        ],
    )


    launch_y_fit = _fit_linear(
        launch_frames,
        [
            float(
                row[
                    "center_y"
                ]
            )
            for row in launch_rows
        ],
    )


    candidate_rows = []


    for track_id in sorted(
        normalized_tracks
    ):

        observations = normalized_tracks[
            track_id
        ]

        if not _candidate_is_eligible(
            observations,
            ball_frames=ball_frames,
            ball_start_frame=ball_start_frame,
        ):

            continue

        candidate_rows.append(
            _candidate_features(
                track_id,
                observations,
                ball_rows=ball_rows,
                launch_x_fit=launch_x_fit,
                launch_y_fit=launch_y_fit,
            )
        )


    return _rank_feature_rows(
        candidate_rows
    )


def build_shooter_v2_decision(
    ranked_rows: Sequence[Mapping[str, Any]],
    *,
    shooter_confirmation_track_id: Optional[int] = None,
) -> ShooterV2Decision:

    if len(
        ranked_rows
    ) < 2:

        decision = ShooterV2Decision(
            auto_status="FAILED",
            resolution="FAILED",
            resolved_track_id=None,
            candidate_track_ids=tuple(),
            auto_scores={},
            auto_top_track_id=None,
            auto_second_track_id=None,
            auto_margin_fraction=None,
            required_margin_fraction=(
                SHOOTER_V2_REQUIRED_MARGIN_FRACTION
            ),
            confirmation_track_id=None,
            confirmation_source=None,
            manual_confirmation_used=False,
            confirmation_used_as_model_evidence=False,
            failure_reason=(
                "Shooter-v2 requires at least two eligible candidates."
            ),
        )

        return validate_shooter_v2_decision(
            decision
        )


    top = ranked_rows[
        0
    ]

    second = ranked_rows[
        1
    ]


    top_track_id = _strict_int(
        top[
            "track_id"
        ],
        field_name="top track_id",
        minimum=0,
    )

    second_track_id = _strict_int(
        second[
            "track_id"
        ],
        field_name="second track_id",
        minimum=0,
    )


    if top_track_id == second_track_id:

        raise ShooterV2Stage11ContractError(
            "Top-two Shooter-v2 candidates must be distinct."
        )


    top_score = _finite_float(
        top[
            "final_score"
        ],
        field_name="top final_score",
    )

    second_score = _finite_float(
        second[
            "final_score"
        ],
        field_name="second final_score",
    )


    if not (
        0.0
        <=
        second_score
        <=
        top_score
        <=
        1.0
    ):

        raise ShooterV2Stage11ContractError(
            "Shooter-v2 top-two scores violate [0,1] ordering."
        )


    if top_score > 0.0:

        margin_fraction = float(
            (
                top_score
                -
                second_score
            )
            /
            top_score
        )

    else:

        margin_fraction = 0.0


    candidate_track_ids = (
        top_track_id,
        second_track_id,
    )


    auto_scores = {
        top_track_id:
            top_score,

        second_track_id:
            second_score,
    }


    ambiguous = bool(
        margin_fraction
        <
        SHOOTER_V2_REQUIRED_MARGIN_FRACTION
    )


    if not ambiguous:

        if shooter_confirmation_track_id is not None:

            raise ShooterV2Stage11InputError(
                "Manual confirmation is accepted only for an ambiguous automatic decision."
            )

        decision = ShooterV2Decision(
            auto_status="AUTO_SELECTED",
            resolution=SHOOTER_RESOLUTION_AUTO,
            resolved_track_id=top_track_id,
            candidate_track_ids=candidate_track_ids,
            auto_scores=auto_scores,
            auto_top_track_id=top_track_id,
            auto_second_track_id=second_track_id,
            auto_margin_fraction=margin_fraction,
            required_margin_fraction=(
                SHOOTER_V2_REQUIRED_MARGIN_FRACTION
            ),
            confirmation_track_id=None,
            confirmation_source=None,
            manual_confirmation_used=False,
            confirmation_used_as_model_evidence=False,
            failure_reason=None,
        )

        return validate_shooter_v2_decision(
            decision
        )


    if shooter_confirmation_track_id is None:

        decision = ShooterV2Decision(
            auto_status="AMBIGUOUS_TOP_TWO",
            resolution="UNRESOLVED",
            resolved_track_id=None,
            candidate_track_ids=candidate_track_ids,
            auto_scores=auto_scores,
            auto_top_track_id=top_track_id,
            auto_second_track_id=second_track_id,
            auto_margin_fraction=margin_fraction,
            required_margin_fraction=(
                SHOOTER_V2_REQUIRED_MARGIN_FRACTION
            ),
            confirmation_track_id=None,
            confirmation_source=None,
            manual_confirmation_used=False,
            confirmation_used_as_model_evidence=False,
            failure_reason=None,
        )

        return validate_shooter_v2_decision(
            decision
        )


    confirmation_track_id = _strict_int(
        shooter_confirmation_track_id,
        field_name="shooter_confirmation_track_id",
        minimum=0,
    )


    if confirmation_track_id not in candidate_track_ids:

        raise ShooterV2Stage11InputError(
            "Shooter confirmation must identify one of the ambiguous top-two candidates."
        )


    decision = ShooterV2Decision(
        auto_status="AMBIGUOUS_TOP_TWO",
        resolution="USER_CONFIRMATION",
        resolved_track_id=confirmation_track_id,
        candidate_track_ids=candidate_track_ids,
        auto_scores=auto_scores,
        auto_top_track_id=top_track_id,
        auto_second_track_id=second_track_id,
        auto_margin_fraction=margin_fraction,
        required_margin_fraction=(
            SHOOTER_V2_REQUIRED_MARGIN_FRACTION
        ),
        confirmation_track_id=confirmation_track_id,
        confirmation_source="USER_CONFIRMATION",
        manual_confirmation_used=True,
        confirmation_used_as_model_evidence=False,
        failure_reason=None,
    )

    return validate_shooter_v2_decision(
        decision
    )


def _validate_stage10_projection(
    result: Any,
) -> None:

    projection = _value(
        result,
        "pitch_projection",
    )

    if _value(
        projection,
        "success",
        False,
    ) is not True:

        raise ShooterV2Stage11InputError(
            "Stage 11 requires successful Stage-10 pitch projection."
        )

    projected_players = _value(
        projection,
        "projected_players",
    )

    if not isinstance(
        projected_players,
        (list, tuple),
    ):

        raise ShooterV2Stage11InputError(
            "pitch_projection.projected_players must be list or tuple."
        )

    metadata = _value(
        result,
        "metadata",
    )

    if not isinstance(
        metadata,
        dict,
    ):

        raise ShooterV2Stage11InputError(
            "result.metadata must be a dict."
        )

    if (
        "semantic_v2_shooter_stage11"
        in metadata
    ):

        raise ShooterV2Stage11ContractError(
            "Stage-11 metadata is already populated."
        )

    stage10 = metadata.get(
        "semantic_v2_geometry_stage10"
    )

    if not isinstance(
        stage10,
        dict,
    ):

        raise ShooterV2Stage11InputError(
            "Stage-10 provenance metadata is missing."
        )

    required_stage10 = {
        "stage":
            "pitch_projection",

        "homography_direction":
            "IMAGE_TO_PITCH",

        "stage9_evidence_consumed":
            True,

        "validated_homography_used":
            True,

        "reference_free":
            True,

        "historical_projected_coordinates_used":
            False,

        "reference_homography_used":
            False,

        "manual_projection_used":
            False,

        "known_shooter_used":
            False,

        "shooter_assignment_performed":
            False,

        "relative_team_assignment_performed":
            False,

        "variant_reselection_performed":
            False,

        "temporal_fallback_performed":
            False,

        "frozen_v1_projection_handler_used":
            False,

        "additional_geometry_inference_calls":
            0,

        "geometry_object_replaced":
            False,

        "public_projection_result_written":
            True,
    }


    for key, expected_value in required_stage10.items():

        if stage10.get(
            key
        ) != expected_value:

            raise ShooterV2Stage11ContractError(
                f"Stage-10 provenance mismatch for {key}."
            )


def _validate_stage11_entry(
    result: Any,
) -> tuple[
    tuple[str, ...],
    Optional[str],
]:

    pipeline_status = _value(
        result,
        "pipeline_status",
    )

    if pipeline_status is None:

        raise ShooterV2Stage11InputError(
            "Stage 11 requires pipeline_status."
        )

    completed_stages = _value(
        pipeline_status,
        "completed_stages",
    )

    if not isinstance(
        completed_stages,
        list,
    ):

        raise ShooterV2Stage11InputError(
            "pipeline_status.completed_stages must be a list."
        )

    if tuple(
        completed_stages
    ) != SHOOTER_V2_REQUIRED_COMPLETED_PREFIX:

        raise ShooterV2Stage11InputError(
            "Stage 11 requires the exact completed Stage 1-10 prefix."
        )

    failed_stage = _value(
        pipeline_status,
        "failed_stage",
        None,
    )

    if failed_stage is not None:

        raise ShooterV2Stage11InputError(
            "Stage 11 cannot execute after a failed stage."
        )

    return (
        tuple(
            completed_stages
        ),
        failed_stage,
    )


def _validate_public_shooter_empty(
    result: Any,
) -> None:

    shooter = _value(
        result,
        "shooter",
    )

    if not isinstance(
        shooter,
        ShooterResult,
    ):

        raise ShooterV2Stage11InputError(
            "result.shooter must be ShooterResult."
        )

    if (
        shooter.success
        or
        shooter.track_id is not None
        or
        shooter.image_xy is not None
        or
        shooter.pitch_xy is not None
        or
        shooter.assignment_score is not None
    ):

        raise ShooterV2Stage11ContractError(
            "Public ShooterResult is already populated."
        )


def _projected_player_row(
    result: Any,
    *,
    track_id: int,
) -> Mapping[str, Any]:

    projection = _value(
        result,
        "pitch_projection",
    )

    projected_players = _value(
        projection,
        "projected_players",
    )

    matches = []

    for row in projected_players:

        row_track_id = _strict_int(
            _value(
                row,
                "track_id",
            ),
            field_name="projected player track_id",
            minimum=0,
        )

        if row_track_id == track_id:

            matches.append(
                row
            )


    if len(
        matches
    ) != 1:

        raise ShooterV2Stage11ContractError(
            "Resolved shooter must have exactly one Stage-10 projected row."
        )

    return matches[
        0
    ]


def _write_public_shooter(
    result: Any,
    *,
    decision: ShooterV2Decision,
) -> None:

    if decision.resolved_track_id is None:

        return

    track_id = int(
        decision.resolved_track_id
    )

    row = _projected_player_row(
        result,
        track_id=track_id,
    )

    image_x = _finite_float(
        _value(
            row,
            "bottom_center_x",
        ),
        field_name="resolved shooter bottom_center_x",
    )

    image_y = _finite_float(
        _value(
            row,
            "bottom_center_y",
        ),
        field_name="resolved shooter bottom_center_y",
    )

    pitch_x = _finite_float(
        _value(
            row,
            "pitch_x",
        ),
        field_name="resolved shooter pitch_x",
    )

    pitch_y = _finite_float(
        _value(
            row,
            "pitch_y",
        ),
        field_name="resolved shooter pitch_y",
    )

    assignment_score = _finite_float(
        decision.auto_scores[
            track_id
        ],
        field_name="resolved shooter automatic score",
    )

    result.shooter = ShooterResult(
        success=True,
        track_id=track_id,
        image_xy=(
            image_x,
            image_y,
        ),
        pitch_xy=(
            pitch_x,
            pitch_y,
        ),
        assignment_score=assignment_score,
        failure_reason=None,
    )


def build_shooter_v2_stage11_assignment_handler(
    *,
    shooter_confirmation_track_id: Optional[int] = None,
):

    confirmation_input = (
        shooter_confirmation_track_id
    )


    def shooter_v2_stage11_assignment_handler(
        result: Any,
    ) -> None:

        (
            completed_before,
            failed_before,
        ) = _validate_stage11_entry(
            result
        )

        _validate_stage10_projection(
            result
        )

        _validate_public_shooter_empty(
            result
        )


        if _value(
            result,
            "shooter_v2",
            None,
        ) is not None:

            raise ShooterV2Stage11ContractError(
                "result.shooter_v2 is already populated."
            )


        geometry_before = _value(
            result,
            "geometry_v2",
            None,
        )

        geometry_identity_before = id(
            geometry_before
        )


        ranked_rows = score_shooter_v2_candidates(
            _value(
                result,
                "ball_track",
            ),
            _value(
                result,
                "player_tracks",
            ),
        )


        decision = build_shooter_v2_decision(
            ranked_rows,
            shooter_confirmation_track_id=(
                confirmation_input
            ),
        )


        result.shooter_v2 = decision


        if decision.resolved_track_id is not None:

            _write_public_shooter(
                result,
                decision=decision,
            )


        metadata = _value(
            result,
            "metadata",
        )


        metadata[
            "semantic_v2_shooter_stage11"
        ] = {
            "stage":
                SHOOTER_V2_STAGE11_NAME,

            "action":
                SHOOTER_V2_STAGE11_ACTION,

            "policy_name":
                SHOOTER_V2_POLICY_NAME,

            "policy_version":
                SHOOTER_V2_POLICY_VERSION,

            "policy_semantic_sha256":
                SHOOTER_V2_POLICY_SEMANTIC_SHA256,

            "candidate_count":
                len(
                    ranked_rows
                ),

            "auto_status":
                decision.auto_status,

            "resolution":
                decision.resolution,

            "resolved_track_id":
                decision.resolved_track_id,

            "candidate_track_ids":
                list(
                    decision.candidate_track_ids
                ),

            "auto_scores":
                {
                    str(
                        track_id
                    ):
                        float(
                            score
                        )
                    for track_id, score
                    in decision.auto_scores.items()
                },

            "auto_top_track_id":
                decision.auto_top_track_id,

            "auto_second_track_id":
                decision.auto_second_track_id,

            "auto_margin_fraction":
                decision.auto_margin_fraction,

            "required_margin_fraction":
                decision.required_margin_fraction,

            "confirmation_track_id":
                decision.confirmation_track_id,

            "confirmation_source":
                decision.confirmation_source,

            "manual_confirmation_used":
                decision.manual_confirmation_used,

            "confirmation_used_as_model_evidence":
                decision.confirmation_used_as_model_evidence,

            "automatic_scores_modified_by_confirmation":
                False,

            "automatic_ranking_modified_by_confirmation":
                False,

            "human_truth_used_for_automatic_scoring":
                False,

            "human_truth_used_for_reranking":
                False,

            "jersey_number_used":
                False,

            "team_label_used":
                False,

            "xg_used":
                False,

            "expected_shot_distance_used":
                False,

            "target_goal_used":
                False,

            "reference_homography_used":
                False,

            "geometry_expected_outcome_used":
                False,

            "track_specific_prior_used":
                False,

            "new_model_inference_performed":
                False,

            "geometry_service_called":
                False,

            "frozen_v1_shooter_handler_used":
                False,

            "public_shooter_written":
                bool(
                    decision.resolved_track_id
                    is not None
                ),

            "ranked_candidate_features":
                [
                    {
                        key:
                            (
                                int(
                                    row[
                                        key
                                    ]
                                )
                                if key
                                in {
                                    "track_id",
                                    "first_player_frame",
                                    "player_observation_count",
                                    "ball_player_overlap_count",
                                    "best_backprojection_lag_frames",
                                    "best_backprojection_frame",
                                    "blind_rank",
                                }
                                else
                                float(
                                    row[
                                        key
                                    ]
                                )
                            )
                        for key
                        in SHOOTER_V2_FEATURE_FIELDS
                    }
                    for row in ranked_rows
                ],
        }


        pipeline_status = _value(
            result,
            "pipeline_status",
        )

        completed_after = tuple(
            _value(
                pipeline_status,
                "completed_stages",
            )
        )

        failed_after = _value(
            pipeline_status,
            "failed_stage",
            None,
        )


        if completed_after != completed_before:

            raise ShooterV2Stage11ContractError(
                "Stage-11 handler must not mark orchestration completion."
            )


        if failed_after != failed_before:

            raise ShooterV2Stage11ContractError(
                "Stage-11 handler must not mutate failed_stage."
            )


        if id(
            _value(
                result,
                "geometry_v2",
                None,
            )
        ) != geometry_identity_before:

            raise ShooterV2Stage11ContractError(
                "Stage-11 handler replaced GeometryV2Result."
            )


        validate_shooter_v2_decision(
            result.shooter_v2
        )


        return None


    return shooter_v2_stage11_assignment_handler
