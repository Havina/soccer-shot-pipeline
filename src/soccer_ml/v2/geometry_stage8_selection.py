from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .semantic_contracts import (
    CANONICAL_V2_STAGE_NAMES,
    GeometryV2Result,
    validate_geometry_v2_result,
)


GEOMETRY_V2_STAGE8_NAME = "homography_search"

GEOMETRY_V2_STAGE8_REQUIRED_COMPLETED_PREFIX = (
    CANONICAL_V2_STAGE_NAMES[
        :7
    ]
)


PNLCALIB_SELECTION_POLICY_NAME = (
    "PNLCALIB_PNL_PRIMARY_BASE_FALLBACK_V1"
)


PNLCALIB_SELECTION_POLICY_SHA256 = (
    "62e0431373c6a665cbe07561c178cf750dd78ea610b11678644a28d7a0d342fd"
)


PNLCALIB_PRIMARY_VARIANT = "PNL_REFINED"

PNLCALIB_FALLBACK_VARIANT = "BASE"


CANONICAL_SELECTION_STATUS_BY_VARIANT = {
    "PNL_REFINED":
        "PRIMARY_PNL_REFINED",

    "BASE":
        "FALLBACK_BASE",
}


STAGE7_METADATA_KEY = (
    "semantic_v2_geometry_stage7"
)


STAGE8_METADATA_KEY = (
    "semantic_v2_geometry_stage8"
)


EXPECTED_STAGE7_METADATA_KEYS = frozenset(
    {
        "stage",
        "anchor_policy",
        "candidate_frame_count_policy",
        "temporal_fallback_enabled",
        "v1_geometry_fallback_enabled",
        "selected_frame_idx",
        "trajectory_source",
        "is_raw_v3_detection",
        "is_interpolated",
        "raw_candidate_count",
        "video_sha256",
        "request_frame_shape",
        "png_size_bytes",
        "request_id",
        "success",
        "selected_variant",
        "selection_status",
        "keypoint_count",
        "line_count",
        "reference_free",
        "failure_reason",
    }
)


class GeometryV2Stage8Error(RuntimeError):
    """Base error for Geometry-v2 Stage-8 selection."""


class GeometryV2Stage8InputError(GeometryV2Stage8Error):
    """Stage-8 upstream state is incomplete or malformed."""


class GeometryV2SelectionContractError(GeometryV2Stage8Error):
    """Stage-7 selection evidence violates the frozen PnLCalib policy."""


@dataclass(frozen=True)
class GeometryV2Stage8SelectionEvidence:

    selected_frame_idx: int

    selected_variant: str

    selection_status: str

    policy_name: str

    policy_sha256: str

    primary_variant: str

    fallback_variant: str

    reference_used_for_selection: bool

    additional_inference_calls: int

    request_id: str


def _validate_nonnegative_int(
    value,
    *,
    field_name: str,
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

        raise GeometryV2Stage8InputError(
            f"{field_name} must be a non-negative integer."
        )


    return int(
        value
    )


def _validate_sha256_hex(
    value,
    *,
    field_name: str,
) -> str:

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

        raise GeometryV2Stage8InputError(
            f"{field_name} must be a 64-character hexadecimal string."
        )


    try:

        int(
            value,
            16,
        )

    except Exception as exc:

        raise GeometryV2Stage8InputError(
            f"{field_name} must be hexadecimal."
        ) from exc


    return value.lower()


def _validate_stage8_prefix(
    result,
):

    pipeline_status = getattr(
        result,
        "pipeline_status",
        None,
    )


    if pipeline_status is None:

        raise GeometryV2Stage8InputError(
            "Stage 8 requires result.pipeline_status."
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

        raise GeometryV2Stage8InputError(
            "pipeline_status.completed_stages must be a list."
        )


    if tuple(
        completed_stages
    ) != GEOMETRY_V2_STAGE8_REQUIRED_COMPLETED_PREFIX:

        raise GeometryV2Stage8InputError(
            "Stage 8 requires exact canonical completed prefix stages 1-7."
        )


    if getattr(
        pipeline_status,
        "failed_stage",
        None,
    ) is not None:

        raise GeometryV2Stage8InputError(
            "Stage 8 cannot execute after pipeline failure."
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

        raise GeometryV2Stage8InputError(
            "Stage 8 requires result.pipeline_status."
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

        raise GeometryV2Stage8InputError(
            "pipeline_status.completed_stages must be list-valued."
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

        raise GeometryV2SelectionContractError(
            "Stage 8 modified orchestration-owned completion/failure state."
        )


def _validate_stage7_metadata(
    metadata,
    geometry: GeometryV2Result,
):

    if not isinstance(
        metadata,
        dict,
    ):

        raise GeometryV2Stage8InputError(
            "Stage 8 requires result.metadata to be dict-valued."
        )


    stage7 = metadata.get(
        STAGE7_METADATA_KEY
    )


    if not isinstance(
        stage7,
        dict,
    ):

        raise GeometryV2Stage8InputError(
            "Stage 8 requires authoritative Stage-7 metadata."
        )


    if frozenset(
        stage7.keys()
    ) != EXPECTED_STAGE7_METADATA_KEYS:

        raise GeometryV2SelectionContractError(
            "Stage-7 metadata field set differs from the frozen Stage-7 contract."
        )


    if stage7[
        "stage"
    ] != "pitch_line_detection":

        raise GeometryV2SelectionContractError(
            "Stage-7 provenance names the wrong canonical stage."
        )


    if stage7[
        "anchor_policy"
    ] != "FIRST_RAW_V3_BALL_FRAME":

        raise GeometryV2SelectionContractError(
            "Stage-7 geometry anchor policy changed."
        )


    if stage7[
        "candidate_frame_count_policy"
    ] != 1:

        raise GeometryV2SelectionContractError(
            "Stage-7 geometry candidate count must remain exactly one."
        )


    if stage7[
        "temporal_fallback_enabled"
    ] is not False:

        raise GeometryV2SelectionContractError(
            "Temporal geometry fallback is forbidden."
        )


    if stage7[
        "v1_geometry_fallback_enabled"
    ] is not False:

        raise GeometryV2SelectionContractError(
            "Frozen-v1 geometry fallback is forbidden."
        )


    if stage7[
        "trajectory_source"
    ] != "RAW_V3":

        raise GeometryV2SelectionContractError(
            "Stage-7 selected frame must originate from RAW_V3."
        )


    if stage7[
        "is_raw_v3_detection"
    ] is not True:

        raise GeometryV2SelectionContractError(
            "Stage-7 selected frame must be a raw-v3 detection."
        )


    if stage7[
        "is_interpolated"
    ] is not False:

        raise GeometryV2SelectionContractError(
            "Stage-7 selected frame may not be interpolated."
        )


    raw_candidate_count = _validate_nonnegative_int(
        stage7[
            "raw_candidate_count"
        ],
        field_name="Stage-7 raw_candidate_count",
    )


    if raw_candidate_count < 1:

        raise GeometryV2SelectionContractError(
            "Stage-7 must contain at least one RAW_V3 candidate."
        )


    selected_frame_idx = _validate_nonnegative_int(
        stage7[
            "selected_frame_idx"
        ],
        field_name="Stage-7 selected_frame_idx",
    )


    if selected_frame_idx != geometry.selected_frame_idx:

        raise GeometryV2SelectionContractError(
            "Stage-7 selected frame disagrees with GeometryV2Result."
        )


    _validate_sha256_hex(
        stage7[
            "video_sha256"
        ],
        field_name="Stage-7 video_sha256",
    )


    request_id = _validate_sha256_hex(
        stage7[
            "request_id"
        ],
        field_name="Stage-7 request_id",
    )


    request_frame_shape = stage7[
        "request_frame_shape"
    ]


    if (
        not isinstance(
            request_frame_shape,
            list,
        )
        or
        len(
            request_frame_shape
        )
        !=
        3
        or
        any(
            (
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
                <=
                0
            )
            for value in request_frame_shape
        )
        or
        request_frame_shape[
            2
        ]
        !=
        3
    ):

        raise GeometryV2SelectionContractError(
            "Stage-7 request frame shape must be positive HxWx3."
        )


    png_size_bytes = _validate_nonnegative_int(
        stage7[
            "png_size_bytes"
        ],
        field_name="Stage-7 png_size_bytes",
    )


    if png_size_bytes <= 0:

        raise GeometryV2SelectionContractError(
            "Stage-7 PNG payload size must be positive."
        )


    if stage7[
        "success"
    ] is not True:

        raise GeometryV2SelectionContractError(
            "Stage 8 cannot execute from failed Stage-7 geometry."
        )


    if stage7[
        "selected_variant"
    ] != geometry.selected_variant:

        raise GeometryV2SelectionContractError(
            "Stage-7 selected variant disagrees with GeometryV2Result."
        )


    if stage7[
        "selection_status"
    ] != geometry.validation_status:

        raise GeometryV2SelectionContractError(
            "Stage-7 selection status disagrees with GeometryV2Result."
        )


    if stage7[
        "keypoint_count"
    ] != geometry.keypoint_count:

        raise GeometryV2SelectionContractError(
            "Stage-7 keypoint count disagrees with GeometryV2Result."
        )


    if stage7[
        "line_count"
    ] != geometry.line_count:

        raise GeometryV2SelectionContractError(
            "Stage-7 line count disagrees with GeometryV2Result."
        )


    if stage7[
        "reference_free"
    ] is not True:

        raise GeometryV2SelectionContractError(
            "Stage-7 geometry must be reference-free."
        )


    if geometry.reference_free is not True:

        raise GeometryV2SelectionContractError(
            "GeometryV2Result must be reference-free."
        )


    if stage7[
        "failure_reason"
    ] is not None:

        raise GeometryV2SelectionContractError(
            "Successful Stage-7 geometry may not carry failure_reason."
        )


    return (
        stage7,
        request_id,
    )


def _validate_selection_pair(
    geometry: GeometryV2Result,
):

    selected_variant = geometry.selected_variant


    if selected_variant not in CANONICAL_SELECTION_STATUS_BY_VARIANT:

        raise GeometryV2SelectionContractError(
            "Geometry-v2 selected an unknown calibration variant."
        )


    expected_status = CANONICAL_SELECTION_STATUS_BY_VARIANT[
        selected_variant
    ]


    if geometry.validation_status != expected_status:

        raise GeometryV2SelectionContractError(
            "Geometry-v2 selected variant/status pair violates frozen PnLCalib policy."
        )


    return (
        selected_variant,
        expected_status,
    )


def build_geometry_v2_stage8_selection_handler():

    def geometry_v2_stage8_selection_handler(
        result,
    ):

        _validate_stage8_prefix(
            result
        )


        before_status = _snapshot_orchestration_status(
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

            raise GeometryV2Stage8InputError(
                "Stage 8 requires GeometryV2Result produced by Stage 7."
            )


        if geometry.success is not True:

            raise GeometryV2SelectionContractError(
                "Stage 8 cannot select from failed Geometry-v2 evidence."
            )


        validate_geometry_v2_result(
            geometry
        )


        geometry_identity_before = id(
            geometry
        )


        metadata = getattr(
            result,
            "metadata",
            None,
        )


        stage7, request_id = _validate_stage7_metadata(
            metadata,
            geometry,
        )


        (
            selected_variant,
            selection_status,
        ) = _validate_selection_pair(
            geometry
        )


        evidence = GeometryV2Stage8SelectionEvidence(
            selected_frame_idx=geometry.selected_frame_idx,
            selected_variant=selected_variant,
            selection_status=selection_status,
            policy_name=PNLCALIB_SELECTION_POLICY_NAME,
            policy_sha256=PNLCALIB_SELECTION_POLICY_SHA256,
            primary_variant=PNLCALIB_PRIMARY_VARIANT,
            fallback_variant=PNLCALIB_FALLBACK_VARIANT,
            reference_used_for_selection=False,
            additional_inference_calls=0,
            request_id=request_id,
        )


        metadata[
            STAGE8_METADATA_KEY
        ] = {
            "stage":
                GEOMETRY_V2_STAGE8_NAME,

            "selection_policy":
                evidence.policy_name,

            "selection_policy_sha256":
                evidence.policy_sha256,

            "primary_variant":
                evidence.primary_variant,

            "fallback_variant":
                evidence.fallback_variant,

            "selected_frame_idx":
                evidence.selected_frame_idx,

            "selected_variant":
                evidence.selected_variant,

            "selection_status":
                evidence.selection_status,

            "request_id":
                evidence.request_id,

            "reference_used_for_selection":
                evidence.reference_used_for_selection,

            "reference_used_for_threshold_tuning":
                False,

            "reference_based_per_event_switching":
                False,

            "expected_result_used_for_selection":
                False,

            "temporal_variant_switching":
                False,

            "additional_geometry_inference_calls":
                evidence.additional_inference_calls,

            "stage7_evidence_consumed":
                True,

            "geometry_object_replaced":
                False,
        }


        if id(
            getattr(
                result,
                "geometry_v2",
                None,
            )
        ) != geometry_identity_before:

            raise GeometryV2SelectionContractError(
                "Stage 8 replaced the frozen Stage-7 GeometryV2Result."
            )


        _validate_status_unchanged(
            result,
            before_status,
        )


        return None


    return geometry_v2_stage8_selection_handler
