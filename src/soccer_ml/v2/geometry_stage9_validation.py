from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .semantic_contracts import (
    CANONICAL_V2_STAGE_NAMES,
    GeometryV2Result,
    validate_geometry_v2_result,
)


GEOMETRY_V2_STAGE9_NAME = "homography_validation"

GEOMETRY_V2_STAGE9_REQUIRED_COMPLETED_PREFIX = (
    CANONICAL_V2_STAGE_NAMES[
        :8
    ]
)


REFERENCE_FREE_VALIDATION_POLICY = (
    "REFERENCE_FREE_MATHEMATICAL_GEOMETRY_VALIDATION_V2_0"
)


HOMOGRAPHY_INVERSE_CONSISTENCY_TOLERANCE = 1e-8


STAGE7_METADATA_KEY = (
    "semantic_v2_geometry_stage7"
)


STAGE8_METADATA_KEY = (
    "semantic_v2_geometry_stage8"
)


STAGE9_METADATA_KEY = (
    "semantic_v2_geometry_stage9"
)


EXPECTED_STAGE8_METADATA_KEYS = frozenset(
    {
        "stage",
        "selection_policy",
        "selection_policy_sha256",
        "primary_variant",
        "fallback_variant",
        "selected_frame_idx",
        "selected_variant",
        "selection_status",
        "request_id",
        "reference_used_for_selection",
        "reference_used_for_threshold_tuning",
        "reference_based_per_event_switching",
        "expected_result_used_for_selection",
        "temporal_variant_switching",
        "additional_geometry_inference_calls",
        "stage7_evidence_consumed",
        "geometry_object_replaced",
    }
)


class GeometryV2Stage9Error(RuntimeError):
    """Base error for Geometry-v2 Stage-9 validation."""


class GeometryV2Stage9InputError(GeometryV2Stage9Error):
    """Stage-9 upstream state is incomplete or malformed."""


class GeometryV2ReferenceFreeValidationError(GeometryV2Stage9Error):
    """Selected Geometry-v2 evidence failed reference-free validation."""


@dataclass(frozen=True)
class HomographyMathematicalValidation:

    shape_valid: bool

    finite: bool

    rank: int

    rank_valid: bool

    invertible: bool

    inverse_finite: bool

    determinant: float


@dataclass(frozen=True)
class GeometryV2Stage9ValidationEvidence:

    selected_frame_idx: int

    selected_variant: str

    selection_status: str

    validation_policy: str

    image_to_pitch: HomographyMathematicalValidation

    pitch_to_image: HomographyMathematicalValidation

    inverse_consistency_max_abs_delta: float

    inverse_consistency_tolerance: float

    inverse_consistent: bool

    reference_free: bool

    reference_used_for_validation: bool

    additional_inference_calls: int

    geometry_object_replaced: bool


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

        raise GeometryV2Stage9InputError(
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

        raise GeometryV2Stage9InputError(
            f"{field_name} must be a 64-character hexadecimal string."
        )


    try:

        int(
            value,
            16,
        )

    except Exception as exc:

        raise GeometryV2Stage9InputError(
            f"{field_name} must be hexadecimal."
        ) from exc


    return value.lower()


def _validate_stage9_prefix(
    result,
):

    pipeline_status = getattr(
        result,
        "pipeline_status",
        None,
    )


    if pipeline_status is None:

        raise GeometryV2Stage9InputError(
            "Stage 9 requires result.pipeline_status."
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

        raise GeometryV2Stage9InputError(
            "pipeline_status.completed_stages must be a list."
        )


    if tuple(
        completed_stages
    ) != GEOMETRY_V2_STAGE9_REQUIRED_COMPLETED_PREFIX:

        raise GeometryV2Stage9InputError(
            "Stage 9 requires exact canonical completed prefix stages 1-8."
        )


    if getattr(
        pipeline_status,
        "failed_stage",
        None,
    ) is not None:

        raise GeometryV2Stage9InputError(
            "Stage 9 cannot execute after pipeline failure."
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

        raise GeometryV2Stage9InputError(
            "Stage 9 requires result.pipeline_status."
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

        raise GeometryV2Stage9InputError(
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

        raise GeometryV2ReferenceFreeValidationError(
            "Stage 9 modified orchestration-owned completion/failure state."
        )


def _matrix_validation(
    matrix_value,
    *,
    field_name: str,
) -> tuple[
    np.ndarray,
    HomographyMathematicalValidation,
]:

    try:

        matrix = np.asarray(
            matrix_value,
            dtype=np.float64,
        )


    except Exception as exc:

        raise GeometryV2ReferenceFreeValidationError(
            f"{field_name} cannot be converted to float64 matrix."
        ) from exc


    shape_valid = bool(
        matrix.shape
        ==
        (
            3,
            3,
        )
    )


    if not shape_valid:

        raise GeometryV2ReferenceFreeValidationError(
            f"{field_name} must be a 3x3 matrix."
        )


    finite = bool(
        np.isfinite(
            matrix
        ).all()
    )


    if not finite:

        raise GeometryV2ReferenceFreeValidationError(
            f"{field_name} contains non-finite values."
        )


    rank = int(
        np.linalg.matrix_rank(
            matrix
        )
    )


    rank_valid = bool(
        rank
        ==
        3
    )


    if not rank_valid:

        raise GeometryV2ReferenceFreeValidationError(
            f"{field_name} must have matrix rank 3."
        )


    determinant = float(
        np.linalg.det(
            matrix
        )
    )


    try:

        inverse = np.linalg.inv(
            matrix
        )


        invertible = True


    except np.linalg.LinAlgError as exc:

        raise GeometryV2ReferenceFreeValidationError(
            f"{field_name} is not invertible."
        ) from exc


    inverse_finite = bool(
        np.isfinite(
            inverse
        ).all()
    )


    if not inverse_finite:

        raise GeometryV2ReferenceFreeValidationError(
            f"{field_name} inverse contains non-finite values."
        )


    evidence = HomographyMathematicalValidation(
        shape_valid=True,
        finite=True,
        rank=rank,
        rank_valid=True,
        invertible=invertible,
        inverse_finite=True,
        determinant=determinant,
    )


    return (
        matrix,
        evidence,
    )


def _projectively_normalized_product(
    product: np.ndarray,
) -> np.ndarray:

    scale = float(
        product[
            2,
            2,
        ]
    )


    if abs(
        scale
    ) > 1e-12:

        return (
            product
            /
            scale
        )


    return product


def _inverse_consistency_delta(
    image_to_pitch: np.ndarray,
    pitch_to_image: np.ndarray,
) -> float:

    identity = np.eye(
        3,
        dtype=np.float64,
    )


    forward_product = (
        image_to_pitch
        @
        pitch_to_image
    )


    reverse_product = (
        pitch_to_image
        @
        image_to_pitch
    )


    if not (
        np.isfinite(
            forward_product
        ).all()
        and
        np.isfinite(
            reverse_product
        ).all()
    ):

        raise GeometryV2ReferenceFreeValidationError(
            "Homography inverse-consistency products contain non-finite values."
        )


    # NOTEBOOK50 CELL-40-LOCAL DEFECT FIX (stage9 r2)
    # --------------------------------------------------------------------
    # Homographies are projective: the frozen Geometry-v2 wire protocol, the
    # frozen geometry client (validate_homography_pair) and the historical
    # Notebook-37 runner all normalise BOTH matrices independently to
    # H[2,2] = 1, so a mutually-inverse pair multiplies to c * I (c != 1 in
    # general), never to I itself. The original check compared the raw
    # products against the identity and was only ever exercised on synthetic
    # identity/exact-inverse pairs (Cell 19: stage9_inverse_consistency_identity_zero).
    # All four historical L40 matrix pairs fail it (delta 0.64 .. 2.72) and so
    # did the first real run. The products are now normalised by their own
    # projective scale before comparison; the 1e-8 tolerance is unchanged.
    forward_product = _projectively_normalized_product(
        forward_product
    )


    reverse_product = _projectively_normalized_product(
        reverse_product
    )


    forward_delta = float(
        np.max(
            np.abs(
                forward_product
                -
                identity
            )
        )
    )


    reverse_delta = float(
        np.max(
            np.abs(
                reverse_product
                -
                identity
            )
        )
    )


    return max(
        forward_delta,
        reverse_delta,
    )


def _validate_stage8_provenance(
    metadata,
    geometry: GeometryV2Result,
):

    if not isinstance(
        metadata,
        dict,
    ):

        raise GeometryV2Stage9InputError(
            "Stage 9 requires result.metadata to be dict-valued."
        )


    stage7 = metadata.get(
        STAGE7_METADATA_KEY
    )


    stage8 = metadata.get(
        STAGE8_METADATA_KEY
    )


    if not isinstance(
        stage7,
        dict,
    ):

        raise GeometryV2Stage9InputError(
            "Stage 9 requires authoritative Stage-7 metadata."
        )


    if not isinstance(
        stage8,
        dict,
    ):

        raise GeometryV2Stage9InputError(
            "Stage 9 requires authoritative Stage-8 metadata."
        )


    if frozenset(
        stage8.keys()
    ) != EXPECTED_STAGE8_METADATA_KEYS:

        raise GeometryV2ReferenceFreeValidationError(
            "Stage-8 metadata field set differs from the frozen Stage-8 contract."
        )


    if stage8[
        "stage"
    ] != "homography_search":

        raise GeometryV2ReferenceFreeValidationError(
            "Stage-8 provenance names the wrong canonical stage."
        )


    if stage8[
        "selection_policy"
    ] != "PNLCALIB_PNL_PRIMARY_BASE_FALLBACK_V1":

        raise GeometryV2ReferenceFreeValidationError(
            "Stage-8 selection policy changed."
        )


    if stage8[
        "selection_policy_sha256"
    ] != "62e0431373c6a665cbe07561c178cf750dd78ea610b11678644a28d7a0d342fd":

        raise GeometryV2ReferenceFreeValidationError(
            "Stage-8 selection policy fingerprint changed."
        )


    if stage8[
        "primary_variant"
    ] != "PNL_REFINED":

        raise GeometryV2ReferenceFreeValidationError(
            "Stage-8 primary variant changed."
        )


    if stage8[
        "fallback_variant"
    ] != "BASE":

        raise GeometryV2ReferenceFreeValidationError(
            "Stage-8 fallback variant changed."
        )


    selected_frame_idx = _validate_nonnegative_int(
        stage8[
            "selected_frame_idx"
        ],
        field_name="Stage-8 selected_frame_idx",
    )


    if selected_frame_idx != geometry.selected_frame_idx:

        raise GeometryV2ReferenceFreeValidationError(
            "Stage-8 selected frame disagrees with GeometryV2Result."
        )


    if stage8[
        "selected_variant"
    ] != geometry.selected_variant:

        raise GeometryV2ReferenceFreeValidationError(
            "Stage-8 selected variant disagrees with GeometryV2Result."
        )


    if stage8[
        "selection_status"
    ] != geometry.validation_status:

        raise GeometryV2ReferenceFreeValidationError(
            "Stage-8 selection status disagrees with GeometryV2Result."
        )


    request_id = _validate_sha256_hex(
        stage8[
            "request_id"
        ],
        field_name="Stage-8 request_id",
    )


    if stage8[
        "reference_used_for_selection"
    ] is not False:

        raise GeometryV2ReferenceFreeValidationError(
            "Reference-assisted Stage-8 selection is forbidden."
        )


    if stage8[
        "reference_used_for_threshold_tuning"
    ] is not False:

        raise GeometryV2ReferenceFreeValidationError(
            "Reference-assisted threshold tuning is forbidden."
        )


    if stage8[
        "reference_based_per_event_switching"
    ] is not False:

        raise GeometryV2ReferenceFreeValidationError(
            "Reference-based per-event switching is forbidden."
        )


    if stage8[
        "expected_result_used_for_selection"
    ] is not False:

        raise GeometryV2ReferenceFreeValidationError(
            "Expected-result selection is forbidden."
        )


    if stage8[
        "temporal_variant_switching"
    ] is not False:

        raise GeometryV2ReferenceFreeValidationError(
            "Temporal variant switching is forbidden."
        )


    additional_calls = _validate_nonnegative_int(
        stage8[
            "additional_geometry_inference_calls"
        ],
        field_name="Stage-8 additional_geometry_inference_calls",
    )


    if additional_calls != 0:

        raise GeometryV2ReferenceFreeValidationError(
            "Stage 8 must perform zero additional Geometry-v2 inference."
        )


    if stage8[
        "stage7_evidence_consumed"
    ] is not True:

        raise GeometryV2ReferenceFreeValidationError(
            "Stage-8 must consume Stage-7 evidence."
        )


    if stage8[
        "geometry_object_replaced"
    ] is not False:

        raise GeometryV2ReferenceFreeValidationError(
            "Stage-8 may not replace the Stage-7 GeometryV2Result."
        )


    if stage7.get(
        "request_id"
    ) != request_id:

        raise GeometryV2ReferenceFreeValidationError(
            "Stage-7 and Stage-8 request IDs disagree."
        )


    if stage7.get(
        "selected_frame_idx"
    ) != selected_frame_idx:

        raise GeometryV2ReferenceFreeValidationError(
            "Stage-7 and Stage-8 selected frame IDs disagree."
        )


    if stage7.get(
        "selected_variant"
    ) != geometry.selected_variant:

        raise GeometryV2ReferenceFreeValidationError(
            "Stage-7 selected variant disagrees with GeometryV2Result."
        )


    if stage7.get(
        "selection_status"
    ) != geometry.validation_status:

        raise GeometryV2ReferenceFreeValidationError(
            "Stage-7 selection status disagrees with GeometryV2Result."
        )


    if stage7.get(
        "reference_free"
    ) is not True:

        raise GeometryV2ReferenceFreeValidationError(
            "Stage-7 geometry must remain reference-free."
        )


    return request_id


def build_geometry_v2_stage9_validation_handler():

    def geometry_v2_stage9_validation_handler(
        result,
    ):

        _validate_stage9_prefix(
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

            raise GeometryV2Stage9InputError(
                "Stage 9 requires GeometryV2Result produced by Stage 7."
            )


        if geometry.success is not True:

            raise GeometryV2ReferenceFreeValidationError(
                "Stage 9 cannot validate failed Geometry-v2 evidence."
            )


        # --------------------------------------------------------------------------------------------
        # Frozen semantic contract is authoritative.
        # This validates runtime mode, variant/status pairing, frame/counts,
        # reference-free provenance, runtime fingerprint, matrix validity,
        # and inverse consistency.
        # --------------------------------------------------------------------------------------------

        validate_geometry_v2_result(
            geometry
        )


        if geometry.reference_free is not True:

            raise GeometryV2ReferenceFreeValidationError(
                "Geometry-v2 Stage 9 requires reference-free geometry."
            )


        geometry_identity_before = id(
            geometry
        )


        metadata = getattr(
            result,
            "metadata",
            None,
        )


        request_id = _validate_stage8_provenance(
            metadata,
            geometry,
        )


        (
            image_to_pitch,
            image_to_pitch_validation,
        ) = _matrix_validation(
            geometry.homography_image_to_pitch,
            field_name="homography_image_to_pitch",
        )


        (
            pitch_to_image,
            pitch_to_image_validation,
        ) = _matrix_validation(
            geometry.homography_pitch_to_image,
            field_name="homography_pitch_to_image",
        )


        inverse_delta = _inverse_consistency_delta(
            image_to_pitch,
            pitch_to_image,
        )


        inverse_consistent = bool(
            inverse_delta
            <=
            HOMOGRAPHY_INVERSE_CONSISTENCY_TOLERANCE
        )


        if not inverse_consistent:

            raise GeometryV2ReferenceFreeValidationError(
                "Geometry-v2 homography pair exceeds inverse-consistency tolerance."
            )


        evidence = GeometryV2Stage9ValidationEvidence(
            selected_frame_idx=geometry.selected_frame_idx,
            selected_variant=geometry.selected_variant,
            selection_status=geometry.validation_status,
            validation_policy=REFERENCE_FREE_VALIDATION_POLICY,
            image_to_pitch=image_to_pitch_validation,
            pitch_to_image=pitch_to_image_validation,
            inverse_consistency_max_abs_delta=inverse_delta,
            inverse_consistency_tolerance=(
                HOMOGRAPHY_INVERSE_CONSISTENCY_TOLERANCE
            ),
            inverse_consistent=True,
            reference_free=True,
            reference_used_for_validation=False,
            additional_inference_calls=0,
            geometry_object_replaced=False,
        )


        metadata[
            STAGE9_METADATA_KEY
        ] = {
            "stage":
                GEOMETRY_V2_STAGE9_NAME,

            "validation_policy":
                evidence.validation_policy,

            "selected_frame_idx":
                evidence.selected_frame_idx,

            "selected_variant":
                evidence.selected_variant,

            "selection_status":
                evidence.selection_status,

            "request_id":
                request_id,

            "image_to_pitch_shape_valid":
                evidence.image_to_pitch.shape_valid,

            "image_to_pitch_finite":
                evidence.image_to_pitch.finite,

            "image_to_pitch_rank":
                evidence.image_to_pitch.rank,

            "image_to_pitch_rank_valid":
                evidence.image_to_pitch.rank_valid,

            "image_to_pitch_invertible":
                evidence.image_to_pitch.invertible,

            "image_to_pitch_inverse_finite":
                evidence.image_to_pitch.inverse_finite,

            "image_to_pitch_determinant":
                evidence.image_to_pitch.determinant,

            "pitch_to_image_shape_valid":
                evidence.pitch_to_image.shape_valid,

            "pitch_to_image_finite":
                evidence.pitch_to_image.finite,

            "pitch_to_image_rank":
                evidence.pitch_to_image.rank,

            "pitch_to_image_rank_valid":
                evidence.pitch_to_image.rank_valid,

            "pitch_to_image_invertible":
                evidence.pitch_to_image.invertible,

            "pitch_to_image_inverse_finite":
                evidence.pitch_to_image.inverse_finite,

            "pitch_to_image_determinant":
                evidence.pitch_to_image.determinant,

            "inverse_consistency_max_abs_delta":
                evidence.inverse_consistency_max_abs_delta,

            "inverse_consistency_tolerance":
                evidence.inverse_consistency_tolerance,

            "inverse_consistent":
                evidence.inverse_consistent,

            "reference_free":
                evidence.reference_free,

            "reference_used_for_validation":
                evidence.reference_used_for_validation,

            "reference_homography_used":
                False,

            "known_shot_coordinates_used":
                False,

            "known_shooter_used":
                False,

            "event_specific_oracle_used":
                False,

            "expected_result_used":
                False,

            "variant_reselection_performed":
                False,

            "temporal_fallback_performed":
                False,

            "v1_geometry_validation_used":
                False,

            "additional_geometry_inference_calls":
                evidence.additional_inference_calls,

            "geometry_object_replaced":
                evidence.geometry_object_replaced,

            "semantic_geometry_contract_revalidated":
                True,
        }


        if id(
            getattr(
                result,
                "geometry_v2",
                None,
            )
        ) != geometry_identity_before:

            raise GeometryV2ReferenceFreeValidationError(
                "Stage 9 replaced the frozen GeometryV2Result."
            )


        _validate_status_unchanged(
            result,
            before_status,
        )


        return None


    return geometry_v2_stage9_validation_handler
