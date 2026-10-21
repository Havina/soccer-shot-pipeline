from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soccer_ml.results import PitchProjectionResult

from .semantic_contracts import (
    CANONICAL_V2_STAGE_NAMES,
    GeometryV2Result,
    validate_geometry_v2_result,
)


GEOMETRY_V2_STAGE10_NAME = "pitch_projection"

GEOMETRY_V2_STAGE10_REQUIRED_COMPLETED_PREFIX = (
    CANONICAL_V2_STAGE_NAMES[
        :9
    ]
)


GEOMETRY_V2_STAGE10_METHOD = (
    "GEOMETRY_V2_VALIDATED_FRAME_BOTTOM_CENTER_PROJECTION"
)


PITCH_LENGTH_M = 105.0

PITCH_WIDTH_M = 68.0

PROJECTION_DENOMINATOR_EPSILON = 1e-12


PUBLIC_PROJECTED_PLAYER_FIELDS = (
    "frame_number",
    "track_id",
    "detection_id",
    "confidence",
    "bottom_center_x",
    "bottom_center_y",
    "pitch_x",
    "pitch_y",
    "inside_pitch",
)


STAGE9_METADATA_KEY = (
    "semantic_v2_geometry_stage9"
)


STAGE10_METADATA_KEY = (
    "semantic_v2_geometry_stage10"
)


EXPECTED_STAGE9_METADATA_KEYS = frozenset(
    {
        "stage",
        "validation_policy",
        "selected_frame_idx",
        "selected_variant",
        "selection_status",
        "request_id",
        "image_to_pitch_shape_valid",
        "image_to_pitch_finite",
        "image_to_pitch_rank",
        "image_to_pitch_rank_valid",
        "image_to_pitch_invertible",
        "image_to_pitch_inverse_finite",
        "image_to_pitch_determinant",
        "pitch_to_image_shape_valid",
        "pitch_to_image_finite",
        "pitch_to_image_rank",
        "pitch_to_image_rank_valid",
        "pitch_to_image_invertible",
        "pitch_to_image_inverse_finite",
        "pitch_to_image_determinant",
        "inverse_consistency_max_abs_delta",
        "inverse_consistency_tolerance",
        "inverse_consistent",
        "reference_free",
        "reference_used_for_validation",
        "reference_homography_used",
        "known_shot_coordinates_used",
        "known_shooter_used",
        "event_specific_oracle_used",
        "expected_result_used",
        "variant_reselection_performed",
        "temporal_fallback_performed",
        "v1_geometry_validation_used",
        "additional_geometry_inference_calls",
        "geometry_object_replaced",
        "semantic_geometry_contract_revalidated",
    }
)


_NO_DEFAULT = object()

_MISSING = object()


class GeometryV2Stage10Error(RuntimeError):
    """Base error for Geometry-v2 Stage-10 projection."""


class GeometryV2Stage10InputError(GeometryV2Stage10Error):
    """Stage-10 upstream state is incomplete or malformed."""


class GeometryV2ProjectionContractError(GeometryV2Stage10Error):
    """Player projection violated the Geometry-v2 Stage-10 contract."""


@dataclass(frozen=True)
class GeometryV2Stage10ProjectionEvidence:

    selected_frame_idx: int

    selected_variant: str

    selection_status: str

    projection_method: str

    players_projected: int

    players_inside_pitch: int

    inside_pitch_fraction: float

    explicit_foot_point_count: int

    bbox_bottom_center_count: int

    all_projections_finite: bool

    reference_free: bool

    additional_inference_calls: int

    geometry_object_replaced: bool


def _value(
    object_or_mapping,
    key,
    default=_NO_DEFAULT,
):

    if isinstance(
        object_or_mapping,
        dict,
    ):

        if key in object_or_mapping:

            return object_or_mapping[
                key
            ]


    elif hasattr(
        object_or_mapping,
        key,
    ):

        return getattr(
            object_or_mapping,
            key,
        )


    if default is not _NO_DEFAULT:

        return default


    raise GeometryV2Stage10InputError(
        f"Required player field is missing: {key}"
    )


def _strict_nonnegative_int(
    value,
    *,
    field_name,
):

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

        raise GeometryV2Stage10InputError(
            f"{field_name} must be a non-negative integer."
        )


    return int(
        value
    )


def _finite_float(
    value,
    *,
    field_name,
):

    try:

        number = float(
            value
        )


    except Exception as exc:

        raise GeometryV2Stage10InputError(
            f"{field_name} must be numeric."
        ) from exc


    if not np.isfinite(
        number
    ):

        raise GeometryV2Stage10InputError(
            f"{field_name} must be finite."
        )


    return number


def _sha256_hex(
    value,
    *,
    field_name,
):

    if (
        not isinstance(
            value,
            str,
        )
        or
        len(
            value
        )
        !=
        64
    ):

        raise GeometryV2Stage10InputError(
            f"{field_name} must be a 64-character hexadecimal string."
        )


    try:

        int(
            value,
            16,
        )

    except Exception as exc:

        raise GeometryV2Stage10InputError(
            f"{field_name} must be hexadecimal."
        ) from exc


    return value.lower()


def _validate_stage10_prefix(
    result,
):

    pipeline_status = getattr(
        result,
        "pipeline_status",
        None,
    )


    if pipeline_status is None:

        raise GeometryV2Stage10InputError(
            "Stage 10 requires result.pipeline_status."
        )


    completed_stages = getattr(
        pipeline_status,
        "completed_stages",
        None,
    )


    if not isinstance(
        completed_stages,
        list,
    ):

        raise GeometryV2Stage10InputError(
            "pipeline_status.completed_stages must be a list."
        )


    if tuple(
        completed_stages
    ) != GEOMETRY_V2_STAGE10_REQUIRED_COMPLETED_PREFIX:

        raise GeometryV2Stage10InputError(
            "Stage 10 requires exact canonical completed prefix stages 1-9."
        )


    if getattr(
        pipeline_status,
        "failed_stage",
        None,
    ) is not None:

        raise GeometryV2Stage10InputError(
            "Stage 10 cannot execute after pipeline failure."
        )


def _snapshot_orchestration_status(
    result,
):

    pipeline_status = getattr(
        result,
        "pipeline_status",
        None,
    )


    if pipeline_status is None:

        raise GeometryV2Stage10InputError(
            "Stage 10 requires result.pipeline_status."
        )


    completed_stages = getattr(
        pipeline_status,
        "completed_stages",
        None,
    )


    if not isinstance(
        completed_stages,
        list,
    ):

        raise GeometryV2Stage10InputError(
            "pipeline_status.completed_stages must remain list-valued."
        )


    return (
        tuple(
            completed_stages
        ),
        getattr(
            pipeline_status,
            "failed_stage",
            None,
        ),
    )


def _validate_status_unchanged(
    result,
    before,
):

    after = _snapshot_orchestration_status(
        result
    )


    if after != before:

        raise GeometryV2ProjectionContractError(
            "Stage 10 modified orchestration-owned completion/failure state."
        )


def _validate_fresh_public_projection_target(
    result,
):

    public_projection = getattr(
        result,
        "pitch_projection",
        None,
    )


    if not isinstance(
        public_projection,
        PitchProjectionResult,
    ):

        raise GeometryV2Stage10InputError(
            "Stage 10 requires the frozen public PitchProjectionResult target."
        )


    fresh = bool(
        public_projection.success
        is False
        and
        public_projection.homography
        is None
        and
        public_projection.projected_players
        is None
        and
        public_projection.players_projected
        ==
        0
        and
        public_projection.players_inside_pitch
        ==
        0
        and
        public_projection.validation_status
        is None
        and
        public_projection.failure_reason
        is None
    )


    if not fresh:

        raise GeometryV2ProjectionContractError(
            "Stage 10 refuses to overwrite a pre-populated pitch projection."
        )


def _validate_stage9_provenance(
    metadata,
    geometry,
):

    if not isinstance(
        metadata,
        dict,
    ):

        raise GeometryV2Stage10InputError(
            "Stage 10 requires result.metadata to be dict-valued."
        )


    if STAGE10_METADATA_KEY in metadata:

        raise GeometryV2ProjectionContractError(
            "Stage-10 authoritative metadata is already populated."
        )


    stage9 = metadata.get(
        STAGE9_METADATA_KEY
    )


    if not isinstance(
        stage9,
        dict,
    ):

        raise GeometryV2Stage10InputError(
            "Stage 10 requires authoritative Stage-9 metadata."
        )


    if frozenset(
        stage9.keys()
    ) != EXPECTED_STAGE9_METADATA_KEYS:

        raise GeometryV2ProjectionContractError(
            "Stage-9 metadata field set differs from the frozen Stage-9 contract."
        )


    if stage9[
        "stage"
    ] != "homography_validation":

        raise GeometryV2ProjectionContractError(
            "Stage-9 provenance names the wrong canonical stage."
        )


    if stage9[
        "validation_policy"
    ] != "REFERENCE_FREE_MATHEMATICAL_GEOMETRY_VALIDATION_V2_0":

        raise GeometryV2ProjectionContractError(
            "Stage-9 validation policy changed."
        )


    selected_frame_idx = _strict_nonnegative_int(
        stage9[
            "selected_frame_idx"
        ],
        field_name="Stage-9 selected_frame_idx",
    )


    if selected_frame_idx != geometry.selected_frame_idx:

        raise GeometryV2ProjectionContractError(
            "Stage-9 selected frame disagrees with GeometryV2Result."
        )


    if stage9[
        "selected_variant"
    ] != geometry.selected_variant:

        raise GeometryV2ProjectionContractError(
            "Stage-9 selected variant disagrees with GeometryV2Result."
        )


    if stage9[
        "selection_status"
    ] != geometry.validation_status:

        raise GeometryV2ProjectionContractError(
            "Stage-9 selection status disagrees with GeometryV2Result."
        )


    request_id = _sha256_hex(
        stage9[
            "request_id"
        ],
        field_name="Stage-9 request_id",
    )


    required_true = (
        "image_to_pitch_shape_valid",
        "image_to_pitch_finite",
        "image_to_pitch_rank_valid",
        "image_to_pitch_invertible",
        "image_to_pitch_inverse_finite",
        "pitch_to_image_shape_valid",
        "pitch_to_image_finite",
        "pitch_to_image_rank_valid",
        "pitch_to_image_invertible",
        "pitch_to_image_inverse_finite",
        "inverse_consistent",
        "reference_free",
        "semantic_geometry_contract_revalidated",
    )


    for key in required_true:

        if stage9[
            key
        ] is not True:

            raise GeometryV2ProjectionContractError(
                f"Stage-9 validated geometry flag must remain true: {key}"
            )


    if (
        _strict_nonnegative_int(
            stage9[
                "image_to_pitch_rank"
            ],
            field_name="Stage-9 image_to_pitch_rank",
        )
        !=
        3
    ):

        raise GeometryV2ProjectionContractError(
            "Stage-9 image-to-pitch rank must remain 3."
        )


    if (
        _strict_nonnegative_int(
            stage9[
                "pitch_to_image_rank"
            ],
            field_name="Stage-9 pitch_to_image_rank",
        )
        !=
        3
    ):

        raise GeometryV2ProjectionContractError(
            "Stage-9 pitch-to-image rank must remain 3."
        )


    tolerance = _finite_float(
        stage9[
            "inverse_consistency_tolerance"
        ],
        field_name="Stage-9 inverse_consistency_tolerance",
    )


    if tolerance != 1e-8:

        raise GeometryV2ProjectionContractError(
            "Stage-9 inverse-consistency tolerance changed."
        )


    inverse_delta = _finite_float(
        stage9[
            "inverse_consistency_max_abs_delta"
        ],
        field_name="Stage-9 inverse_consistency_max_abs_delta",
    )


    if (
        inverse_delta
        <
        0.0
        or
        inverse_delta
        >
        tolerance
    ):

        raise GeometryV2ProjectionContractError(
            "Stage-9 inverse-consistency evidence is outside the frozen tolerance."
        )


    required_false = (
        "reference_used_for_validation",
        "reference_homography_used",
        "known_shot_coordinates_used",
        "known_shooter_used",
        "event_specific_oracle_used",
        "expected_result_used",
        "variant_reselection_performed",
        "temporal_fallback_performed",
        "v1_geometry_validation_used",
        "geometry_object_replaced",
    )


    for key in required_false:

        if stage9[
            key
        ] is not False:

            raise GeometryV2ProjectionContractError(
                f"Forbidden Stage-9 provenance flag became true: {key}"
            )


    additional_calls = _strict_nonnegative_int(
        stage9[
            "additional_geometry_inference_calls"
        ],
        field_name="Stage-9 additional_geometry_inference_calls",
    )


    if additional_calls != 0:

        raise GeometryV2ProjectionContractError(
            "Stage 9 must preserve zero additional Geometry-v2 inference."
        )


    return request_id


def _track_entries(
    tracks,
):

    if isinstance(
        tracks,
        dict,
    ):

        entries = []


        for mapping_track_id, track_value in tracks.items():

            if isinstance(
                track_value,
                (
                    list,
                    tuple,
                ),
            ):

                entries.append(
                    (
                        mapping_track_id,
                        track_value,
                    )
                )


            else:

                observations = _value(
                    track_value,
                    "observations",
                )


                track_id = _value(
                    track_value,
                    "track_id",
                    mapping_track_id,
                )


                entries.append(
                    (
                        track_id,
                        observations,
                    )
                )


        return entries


    if isinstance(
        tracks,
        (
            list,
            tuple,
        ),
    ):

        entries = []


        for track in tracks:

            track_id = _value(
                track,
                "track_id",
            )


            observations = _value(
                track,
                "observations",
            )


            entries.append(
                (
                    track_id,
                    observations,
                )
            )


        return entries


    raise GeometryV2Stage10InputError(
        "player_tracks.tracks must be list-, tuple-, or dict-valued."
    )


def _observation_detection_id(
    observation,
):

    detection_idx = _value(
        observation,
        "detection_idx",
        _MISSING,
    )


    detection_id = _value(
        observation,
        "detection_id",
        _MISSING,
    )


    if (
        detection_idx
        is _MISSING
        and
        detection_id
        is _MISSING
    ):

        raise GeometryV2Stage10InputError(
            "Geometry-frame player observation has no detection identifier."
        )


    if detection_idx is not _MISSING:

        detection_idx = _strict_nonnegative_int(
            detection_idx,
            field_name="observation.detection_idx",
        )


    if detection_id is not _MISSING:

        detection_id = _strict_nonnegative_int(
            detection_id,
            field_name="observation.detection_id",
        )


    if (
        detection_idx
        is not _MISSING
        and
        detection_id
        is not _MISSING
        and
        detection_idx
        !=
        detection_id
    ):

        raise GeometryV2ProjectionContractError(
            "observation.detection_idx and detection_id disagree."
        )


    if detection_idx is not _MISSING:

        return detection_idx


    return detection_id


def _observation_bottom_center(
    observation,
):

    foot_x = _value(
        observation,
        "foot_x",
        _MISSING,
    )


    foot_y = _value(
        observation,
        "foot_y",
        _MISSING,
    )


    if (
        foot_x
        is not _MISSING
        or
        foot_y
        is not _MISSING
    ):

        if (
            foot_x
            is _MISSING
            or
            foot_y
            is _MISSING
        ):

            raise GeometryV2Stage10InputError(
                "Explicit player foot coordinates must provide both foot_x and foot_y."
            )


        return (
            _finite_float(
                foot_x,
                field_name="observation.foot_x",
            ),
            _finite_float(
                foot_y,
                field_name="observation.foot_y",
            ),
            "FOOT_XY",
        )


    x1 = _finite_float(
        _value(
            observation,
            "x1",
        ),
        field_name="observation.x1",
    )


    x2 = _finite_float(
        _value(
            observation,
            "x2",
        ),
        field_name="observation.x2",
    )


    y2 = _finite_float(
        _value(
            observation,
            "y2",
        ),
        field_name="observation.y2",
    )


    if x2 < x1:

        raise GeometryV2Stage10InputError(
            "Player bbox requires x2 >= x1."
        )


    return (
        float(
            (
                x1
                +
                x2
            )
            /
            2.0
        ),
        y2,
        "BBOX_BOTTOM_CENTER",
    )


def _geometry_frame_player_rows(
    result,
    *,
    geometry_frame,
):

    player_tracks = getattr(
        result,
        "player_tracks",
        None,
    )


    if player_tracks is None:

        raise GeometryV2Stage10InputError(
            "Stage 10 requires result.player_tracks."
        )


    if getattr(
        player_tracks,
        "success",
        False,
    ) is not True:

        raise GeometryV2Stage10InputError(
            "Stage 10 requires successful player tracking."
        )


    tracks = getattr(
        player_tracks,
        "tracks",
        None,
    )


    entries = _track_entries(
        tracks
    )


    if len(
        entries
    ) == 0:

        raise GeometryV2Stage10InputError(
            "Stage 10 received zero player tracks."
        )


    rows = []


    for (
        raw_track_id,
        observations,
    ) in entries:

        track_id = _strict_nonnegative_int(
            raw_track_id,
            field_name="track_id",
        )


        if not isinstance(
            observations,
            (
                list,
                tuple,
            ),
        ):

            raise GeometryV2Stage10InputError(
                "Player track observations must be list- or tuple-valued."
            )


        for observation in observations:

            frame_idx = _strict_nonnegative_int(
                _value(
                    observation,
                    "frame_idx",
                ),
                field_name="observation.frame_idx",
            )


            if frame_idx != geometry_frame:

                continue


            observation_track_id = _value(
                observation,
                "track_id",
                track_id,
            )


            observation_track_id = _strict_nonnegative_int(
                observation_track_id,
                field_name="observation.track_id",
            )


            if observation_track_id != track_id:

                raise GeometryV2ProjectionContractError(
                    "Observation track_id disagrees with parent track."
                )


            detection_id = _observation_detection_id(
                observation
            )


            confidence = _finite_float(
                _value(
                    observation,
                    "confidence",
                ),
                field_name="observation.confidence",
            )


            if not (
                0.0
                <=
                confidence
                <=
                1.0
            ):

                raise GeometryV2Stage10InputError(
                    "Player confidence must be within [0, 1]."
                )


            (
                bottom_center_x,
                bottom_center_y,
                point_source,
            ) = _observation_bottom_center(
                observation
            )


            rows.append(
                {
                    "frame_number":
                        frame_idx,

                    "track_id":
                        track_id,

                    "detection_id":
                        detection_id,

                    "confidence":
                        confidence,

                    "bottom_center_x":
                        bottom_center_x,

                    "bottom_center_y":
                        bottom_center_y,

                    "_point_source":
                        point_source,
                }
            )


    if len(
        rows
    ) == 0:

        raise GeometryV2Stage10InputError(
            "Stage 10 received zero tracked players at the selected geometry frame."
        )


    track_ids = [
        row[
            "track_id"
        ]
        for row in rows
    ]


    detection_ids = [
        row[
            "detection_id"
        ]
        for row in rows
    ]


    if len(
        track_ids
    ) != len(
        set(
            track_ids
        )
    ):

        raise GeometryV2ProjectionContractError(
            "Multiple observations for one track exist at the geometry frame."
        )


    if len(
        detection_ids
    ) != len(
        set(
            detection_ids
        )
    ):

        raise GeometryV2ProjectionContractError(
            "Duplicate player detection IDs exist at the geometry frame."
        )


    rows.sort(
        key=lambda row:
            (
                row[
                    "track_id"
                ],
                row[
                    "detection_id"
                ],
            )
    )


    return rows


def _project_image_point(
    H,
    *,
    x,
    y,
):

    source = np.asarray(
        [
            _finite_float(
                x,
                field_name="projection x",
            ),
            _finite_float(
                y,
                field_name="projection y",
            ),
            1.0,
        ],
        dtype=np.float64,
    )


    projected = (
        H
        @
        source
    )


    if not np.isfinite(
        projected
    ).all():

        raise GeometryV2ProjectionContractError(
            "Player homography projection produced non-finite homogeneous coordinates."
        )


    denominator = float(
        projected[
            2
        ]
    )


    if (
        not np.isfinite(
            denominator
        )
        or
        abs(
            denominator
        )
        <=
        PROJECTION_DENOMINATOR_EPSILON
    ):

        raise GeometryV2ProjectionContractError(
            "Player homography projection has invalid homogeneous denominator."
        )


    pitch_x = float(
        projected[
            0
        ]
        /
        denominator
    )


    pitch_y = float(
        projected[
            1
        ]
        /
        denominator
    )


    if not (
        np.isfinite(
            pitch_x
        )
        and
        np.isfinite(
            pitch_y
        )
    ):

        raise GeometryV2ProjectionContractError(
            "Player pitch projection produced non-finite coordinates."
        )


    return (
        pitch_x,
        pitch_y,
    )


def _project_player_rows(
    rows,
    H,
):

    projected_rows = []

    explicit_foot_count = 0

    bbox_bottom_center_count = 0


    for row in rows:

        (
            pitch_x,
            pitch_y,
        ) = _project_image_point(
            H,
            x=row[
                "bottom_center_x"
            ],
            y=row[
                "bottom_center_y"
            ],
        )


        inside_pitch = bool(
            0.0
            <=
            pitch_x
            <=
            PITCH_LENGTH_M
            and
            0.0
            <=
            pitch_y
            <=
            PITCH_WIDTH_M
        )


        point_source = row[
            "_point_source"
        ]


        if point_source == "FOOT_XY":

            explicit_foot_count += 1


        elif point_source == "BBOX_BOTTOM_CENTER":

            bbox_bottom_center_count += 1


        else:

            raise GeometryV2ProjectionContractError(
                "Unknown player projection point source."
            )


        projected_rows.append(
            {
                "frame_number":
                    int(
                        row[
                            "frame_number"
                        ]
                    ),

                "track_id":
                    int(
                        row[
                            "track_id"
                        ]
                    ),

                "detection_id":
                    int(
                        row[
                            "detection_id"
                        ]
                    ),

                "confidence":
                    float(
                        row[
                            "confidence"
                        ]
                    ),

                "bottom_center_x":
                    float(
                        row[
                            "bottom_center_x"
                        ]
                    ),

                "bottom_center_y":
                    float(
                        row[
                            "bottom_center_y"
                        ]
                    ),

                "pitch_x":
                    pitch_x,

                "pitch_y":
                    pitch_y,

                "inside_pitch":
                    inside_pitch,
            }
        )


    return (
        projected_rows,
        explicit_foot_count,
        bbox_bottom_center_count,
    )


def build_geometry_v2_stage10_projection_handler():

    def geometry_v2_stage10_projection_handler(
        result,
    ):

        _validate_stage10_prefix(
            result
        )


        before_status = _snapshot_orchestration_status(
            result
        )


        _validate_fresh_public_projection_target(
            result
        )


        geometry = getattr(
            result,
            "geometry_v2",
            None,
        )


        if not isinstance(
            geometry,
            GeometryV2Result,
        ):

            raise GeometryV2Stage10InputError(
                "Stage 10 requires GeometryV2Result."
            )


        if geometry.success is not True:

            raise GeometryV2ProjectionContractError(
                "Stage 10 cannot project players from failed Geometry-v2 evidence."
            )


        validate_geometry_v2_result(
            geometry
        )


        if geometry.reference_free is not True:

            raise GeometryV2ProjectionContractError(
                "Stage 10 requires reference-free Geometry-v2 evidence."
            )


        geometry_identity_before = id(
            geometry
        )


        metadata = getattr(
            result,
            "metadata",
            None,
        )


        request_id = _validate_stage9_provenance(
            metadata,
            geometry,
        )


        H = np.asarray(
            geometry.homography_image_to_pitch,
            dtype=np.float64,
        )


        if (
            H.shape
            !=
            (
                3,
                3,
            )
            or
            not np.isfinite(
                H
            ).all()
        ):

            raise GeometryV2ProjectionContractError(
                "Stage 10 requires finite 3x3 image-to-pitch homography."
            )


        geometry_frame = _strict_nonnegative_int(
            geometry.selected_frame_idx,
            field_name="geometry.selected_frame_idx",
        )


        player_rows = _geometry_frame_player_rows(
            result,
            geometry_frame=geometry_frame,
        )


        (
            projected_players,
            explicit_foot_count,
            bbox_bottom_center_count,
        ) = _project_player_rows(
            player_rows,
            H,
        )


        players_projected = len(
            projected_players
        )


        players_inside_pitch = sum(
            int(
                row[
                    "inside_pitch"
                ]
                is True
            )
            for row in projected_players
        )


        all_projections_finite = bool(
            all(
                np.isfinite(
                    [
                        row[
                            "bottom_center_x"
                        ],
                        row[
                            "bottom_center_y"
                        ],
                        row[
                            "pitch_x"
                        ],
                        row[
                            "pitch_y"
                        ],
                    ]
                ).all()

                for row in projected_players
            )
        )


        if not all_projections_finite:

            raise GeometryV2ProjectionContractError(
                "Stage 10 produced non-finite public projection output."
            )


        inside_pitch_fraction = float(
            players_inside_pitch
            /
            players_projected
        )


        evidence = GeometryV2Stage10ProjectionEvidence(
            selected_frame_idx=geometry_frame,
            selected_variant=geometry.selected_variant,
            selection_status=geometry.validation_status,
            projection_method=GEOMETRY_V2_STAGE10_METHOD,
            players_projected=players_projected,
            players_inside_pitch=players_inside_pitch,
            inside_pitch_fraction=inside_pitch_fraction,
            explicit_foot_point_count=explicit_foot_count,
            bbox_bottom_center_count=bbox_bottom_center_count,
            all_projections_finite=True,
            reference_free=True,
            additional_inference_calls=0,
            geometry_object_replaced=False,
        )


        result.pitch_projection = PitchProjectionResult(
            success=True,
            homography=H.tolist(),
            projected_players=projected_players,
            players_projected=players_projected,
            players_inside_pitch=players_inside_pitch,
            validation_status="VALID",
            failure_reason=None,
        )


        metadata[
            STAGE10_METADATA_KEY
        ] = {
            "stage":
                GEOMETRY_V2_STAGE10_NAME,

            "method":
                evidence.projection_method,

            "selected_frame_idx":
                evidence.selected_frame_idx,

            "selected_variant":
                evidence.selected_variant,

            "selection_status":
                evidence.selection_status,

            "request_id":
                request_id,

            "homography_direction":
                "IMAGE_TO_PITCH",

            "pitch_length_m":
                PITCH_LENGTH_M,

            "pitch_width_m":
                PITCH_WIDTH_M,

            "projection_denominator_epsilon":
                PROJECTION_DENOMINATOR_EPSILON,

            "players_projected":
                evidence.players_projected,

            "players_inside_pitch":
                evidence.players_inside_pitch,

            "inside_pitch_fraction":
                evidence.inside_pitch_fraction,

            "all_projections_finite":
                evidence.all_projections_finite,

            "explicit_foot_point_count":
                evidence.explicit_foot_point_count,

            "bbox_bottom_center_count":
                evidence.bbox_bottom_center_count,

            "stage9_evidence_consumed":
                True,

            "validated_homography_used":
                True,

            "reference_free":
                evidence.reference_free,

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
                evidence.additional_inference_calls,

            "geometry_object_replaced":
                evidence.geometry_object_replaced,

            "public_projection_result_written":
                True,
        }


        if id(
            getattr(
                result,
                "geometry_v2",
                None,
            )
        ) != geometry_identity_before:

            raise GeometryV2ProjectionContractError(
                "Stage 10 replaced the frozen GeometryV2Result."
            )


        _validate_status_unchanged(
            result,
            before_status,
        )


        return None


    return geometry_v2_stage10_projection_handler
