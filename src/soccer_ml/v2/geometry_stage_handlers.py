from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .contracts import (
    GeometryV2InferenceResponse,
)

from .orchestration_contracts import (
    GEOMETRY_ANCHOR_SOURCE,
    GEOMETRY_CANDIDATE_FRAME_COUNT,
    GEOMETRY_TEMPORAL_FALLBACK_ENABLED,
    GEOMETRY_V1_FALLBACK_ENABLED,
)

from .semantic_contracts import (
    CANONICAL_V2_STAGE_NAMES,
    GEOMETRY_V2_RUNTIME_MODE,
    GeometryV2Result,
    validate_geometry_v2_result,
)


GEOMETRY_V2_STAGE7_NAME = "pitch_line_detection"

GEOMETRY_V2_STAGE7_REQUIRED_COMPLETED_PREFIX = (
    CANONICAL_V2_STAGE_NAMES[
        :6
    ]
)

RAW_V3_TRAJECTORY_SOURCE = "RAW_V3"


class GeometryV2StageBoundaryError(RuntimeError):
    """Base error for the Geometry-v2 Stage-7 execution boundary."""


class GeometryV2AnchorError(GeometryV2StageBoundaryError):
    """The FIRST_RAW_V3_BALL_FRAME anchor cannot be resolved safely."""


class GeometryV2InputError(GeometryV2StageBoundaryError):
    """Stage-7 upstream input is malformed or out of contract."""


class GeometryV2InferenceContractError(GeometryV2StageBoundaryError):
    """Injected inference output violates the frozen Geometry-v2 contract."""


class GeometryV2StageFailure(GeometryV2StageBoundaryError):
    """Geometry-v2 inference returned a canonical failure result."""


@dataclass(frozen=True)
class GeometryV2AnchorEvidence:

    frame_idx: int

    trajectory_source: str

    is_raw_v3_detection: bool

    is_interpolated: bool

    raw_candidate_count: int


@dataclass(frozen=True)
class GeometryV2Stage7Evidence:

    anchor: GeometryV2AnchorEvidence

    video_sha256: str

    request_frame_shape: tuple[int, int, int]

    png_size_bytes: int

    request_id: str

    selected_variant: Optional[str]

    selection_status: str

    keypoint_count: Optional[int]

    line_count: Optional[int]

    reference_free: bool


def _row_value(
    row,
    name: str,
):

    if isinstance(
        row,
        dict,
    ):

        if name not in row:

            raise GeometryV2AnchorError(
                f"Ball trajectory row is missing required field {name!r}."
            )


        return row[
            name
        ]


    if not hasattr(
        row,
        name,
    ):

        raise GeometryV2AnchorError(
            f"Ball trajectory row is missing required attribute {name!r}."
        )


    return getattr(
        row,
        name,
    )


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

        raise GeometryV2InputError(
            f"{field_name} must be a non-negative integer."
        )


    return int(
        value
    )


def _validate_video_sha256(
    value,
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

        raise GeometryV2InputError(
            "video_sha256 must be a 64-character hexadecimal SHA-256 string."
        )


    try:

        int(
            value,
            16,
        )

    except Exception as exc:

        raise GeometryV2InputError(
            "video_sha256 must be hexadecimal."
        ) from exc


    return value.lower()


def _sha256_file(
    path_value,
) -> str:

    path = Path(
        path_value
    ).resolve()


    if not path.exists():

        raise GeometryV2InputError(
            f"Input video does not exist: {path}"
        )


    if not path.is_file():

        raise GeometryV2InputError(
            f"Input video is not a regular file: {path}"
        )


    digest = hashlib.sha256()


    with path.open(
        "rb"
    ) as handle:

        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):

            digest.update(
                block
            )


    return digest.hexdigest()


def _validate_frame_bgr(
    frame,
) -> np.ndarray:

    if not isinstance(
        frame,
        np.ndarray,
    ):

        raise GeometryV2InputError(
            "Geometry frame loader must return numpy.ndarray."
        )


    if frame.dtype != np.uint8:

        raise GeometryV2InputError(
            "Geometry frame must have dtype uint8."
        )


    if (
        frame.ndim
        !=
        3
        or
        frame.shape[
            2
        ]
        !=
        3
    ):

        raise GeometryV2InputError(
            "Geometry frame must have shape HxWx3."
        )


    if (
        frame.shape[
            0
        ]
        <=
        0
        or
        frame.shape[
            1
        ]
        <=
        0
    ):

        raise GeometryV2InputError(
            "Geometry frame dimensions must be positive."
        )


    return frame


def _validate_png_bytes(
    value,
) -> bytes:

    if not isinstance(
        value,
        (
            bytes,
            bytearray,
        ),
    ):

        raise GeometryV2InputError(
            "PNG encoder must return bytes."
        )


    value = bytes(
        value
    )


    if not value:

        raise GeometryV2InputError(
            "PNG encoder returned empty payload."
        )


    return value


def select_first_raw_v3_ball_frame(
    result,
) -> GeometryV2AnchorEvidence:

    if GEOMETRY_ANCHOR_SOURCE != "FIRST_RAW_V3_BALL_FRAME":

        raise GeometryV2AnchorError(
            "Frozen Geometry-v2 anchor policy changed."
        )


    if GEOMETRY_CANDIDATE_FRAME_COUNT != 1:

        raise GeometryV2AnchorError(
            "Geometry-v2 v2.0 requires exactly one candidate frame."
        )


    if GEOMETRY_TEMPORAL_FALLBACK_ENABLED:

        raise GeometryV2AnchorError(
            "Geometry-v2 temporal fallback is forbidden."
        )


    if GEOMETRY_V1_FALLBACK_ENABLED:

        raise GeometryV2AnchorError(
            "Geometry-v2 fallback to v1 geometry is forbidden."
        )


    ball_track = getattr(
        result,
        "ball_track",
        None,
    )


    if ball_track is None:

        raise GeometryV2AnchorError(
            "Stage 7 requires result.ball_track."
        )


    if getattr(
        ball_track,
        "success",
        None,
    ) is not True:

        raise GeometryV2AnchorError(
            "Stage 7 requires successful ball tracking."
        )


    trajectory = getattr(
        ball_track,
        "track",
        None,
    )


    if not isinstance(
        trajectory,
        (
            list,
            tuple,
        ),
    ):

        raise GeometryV2AnchorError(
            "BallTrackResult.track must be list/tuple trajectory evidence."
        )


    if not trajectory:

        raise GeometryV2AnchorError(
            "BallTrackResult.track is empty."
        )


    candidates: list[
        GeometryV2AnchorEvidence
    ] = []


    seen_raw_frames = set()


    for row in trajectory:

        frame_idx = _row_value(
            row,
            "frame_idx",
        )


        if (
            not isinstance(
                frame_idx,
                int,
            )
            or
            isinstance(
                frame_idx,
                bool,
            )
            or
            frame_idx
            <
            0
        ):

            raise GeometryV2AnchorError(
                "Ball trajectory frame_idx must be a non-negative integer."
            )


        trajectory_source = _row_value(
            row,
            "trajectory_source",
        )


        is_raw_v3_detection = _row_value(
            row,
            "is_raw_v3_detection",
        )


        is_interpolated = _row_value(
            row,
            "is_interpolated",
        )


        if not isinstance(
            trajectory_source,
            str,
        ):

            raise GeometryV2AnchorError(
                "trajectory_source must be string-valued."
            )


        if not isinstance(
            is_raw_v3_detection,
            bool,
        ):

            raise GeometryV2AnchorError(
                "is_raw_v3_detection must be bool-valued."
            )


        if not isinstance(
            is_interpolated,
            bool,
        ):

            raise GeometryV2AnchorError(
                "is_interpolated must be bool-valued."
            )


        # RAW_V3 is a strict three-field provenance contract.
        if trajectory_source == RAW_V3_TRAJECTORY_SOURCE:

            if (
                is_raw_v3_detection
                is not True
                or
                is_interpolated
                is not False
            ):

                raise GeometryV2AnchorError(
                    "RAW_V3 trajectory row has contradictory provenance flags."
                )


        if (
            is_raw_v3_detection
            and
            trajectory_source
            !=
            RAW_V3_TRAJECTORY_SOURCE
        ):

            raise GeometryV2AnchorError(
                "Raw-v3 flag is inconsistent with trajectory_source."
            )


        is_candidate = bool(
            trajectory_source
            ==
            RAW_V3_TRAJECTORY_SOURCE
            and
            is_raw_v3_detection
            is True
            and
            is_interpolated
            is False
        )


        if not is_candidate:

            continue


        frame_idx = int(
            frame_idx
        )


        if frame_idx in seen_raw_frames:

            raise GeometryV2AnchorError(
                "Duplicate RAW_V3 trajectory frame detected."
            )


        seen_raw_frames.add(
            frame_idx
        )


        candidates.append(
            GeometryV2AnchorEvidence(
                frame_idx=frame_idx,
                trajectory_source=trajectory_source,
                is_raw_v3_detection=True,
                is_interpolated=False,
                raw_candidate_count=0,
            )
        )


    if not candidates:

        raise GeometryV2AnchorError(
            "No RAW_V3 ball frame is available for Geometry-v2."
        )


    candidates.sort(
        key=lambda value:
            value.frame_idx
    )


    first = candidates[
        0
    ]


    return GeometryV2AnchorEvidence(
        frame_idx=first.frame_idx,
        trajectory_source=first.trajectory_source,
        is_raw_v3_detection=True,
        is_interpolated=False,
        raw_candidate_count=len(
            candidates
        ),
    )


def inference_response_to_geometry_result(
    response: GeometryV2InferenceResponse,
) -> GeometryV2Result:

    if not isinstance(
        response,
        GeometryV2InferenceResponse,
    ):

        raise GeometryV2InferenceContractError(
            "Inference callable must return GeometryV2InferenceResponse."
        )


    frame_idx = _validate_nonnegative_int(
        response.frame_idx,
        field_name="response.frame_idx",
    )


    if response.success:

        result = GeometryV2Result(
            success=True,
            runtime_mode=GEOMETRY_V2_RUNTIME_MODE,
            selected_variant=response.selected_variant,
            selected_frame_idx=frame_idx,
            homography_image_to_pitch=(
                response.homography_image_to_pitch
            ),
            homography_pitch_to_image=(
                response.homography_pitch_to_image
            ),
            keypoint_count=response.keypoint_count,
            line_count=response.line_count,
            reference_free=response.reference_free,
            validation_status=response.selection_status,
            runtime_fingerprint=response.runtime_fingerprint,
            failure_reason=None,
        )


    else:

        result = GeometryV2Result(
            success=False,
            runtime_mode=GEOMETRY_V2_RUNTIME_MODE,
            selected_variant=None,
            selected_frame_idx=frame_idx,
            homography_image_to_pitch=None,
            homography_pitch_to_image=None,
            keypoint_count=None,
            line_count=None,
            reference_free=response.reference_free,
            validation_status=response.selection_status,
            runtime_fingerprint=response.runtime_fingerprint,
            failure_reason=response.failure_reason,
        )


    validate_geometry_v2_result(
        result
    )


    return result


def _validate_stage7_prefix(
    result,
):

    pipeline_status = getattr(
        result,
        "pipeline_status",
        None,
    )


    if pipeline_status is None:

        raise GeometryV2InputError(
            "Stage 7 requires result.pipeline_status."
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

        raise GeometryV2InputError(
            "pipeline_status.completed_stages must be a list."
        )


    if tuple(
        completed_stages
    ) != GEOMETRY_V2_STAGE7_REQUIRED_COMPLETED_PREFIX:

        raise GeometryV2InputError(
            "Stage 7 requires exact canonical completed prefix stages 1-6."
        )


    failed_stage = getattr(
        pipeline_status,
        "failed_stage",
        None,
    )


    if failed_stage is not None:

        raise GeometryV2InputError(
            "Stage 7 cannot execute after pipeline failure."
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

        raise GeometryV2InputError(
            "Stage 7 requires result.pipeline_status."
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

        raise GeometryV2InputError(
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

        raise GeometryV2InferenceContractError(
            "Stage-7 handler modified orchestration-owned completion/failure state."
        )


def build_geometry_v2_stage7_handler(
    *,
    inference_callable: Callable[..., GeometryV2InferenceResponse],
    frame_loader: Callable[[str, int], np.ndarray],
    png_encoder: Callable[[np.ndarray], bytes],
    video_sha256_provider: Optional[Callable[[str], str]] = None,
):

    if not callable(
        inference_callable
    ):

        raise GeometryV2InputError(
            "inference_callable must be callable."
        )


    if not callable(
        frame_loader
    ):

        raise GeometryV2InputError(
            "frame_loader must be callable."
        )


    if not callable(
        png_encoder
    ):

        raise GeometryV2InputError(
            "png_encoder must be callable."
        )


    if (
        video_sha256_provider is not None
        and
        not callable(
            video_sha256_provider
        )
    ):

        raise GeometryV2InputError(
            "video_sha256_provider must be callable or None."
        )


    sha_provider = (
        video_sha256_provider
        if video_sha256_provider is not None
        else _sha256_file
    )


    def geometry_v2_stage7_handler(
        result,
    ):

        _validate_stage7_prefix(
            result
        )


        before_status = _snapshot_orchestration_status(
            result
        )


        metadata = getattr(
            result,
            "metadata",
            None,
        )


        if not isinstance(
            metadata,
            dict,
        ):

            raise GeometryV2InputError(
                "Stage 7 requires result.metadata to be a dict."
            )


        video_path = getattr(
            result,
            "video_path",
            None,
        )


        if (
            not isinstance(
                video_path,
                str,
            )
            or
            not video_path
        ):

            raise GeometryV2InputError(
                "Stage 7 requires non-empty result.video_path."
            )


        anchor = select_first_raw_v3_ball_frame(
            result
        )


        video_sha256 = _validate_video_sha256(
            sha_provider(
                video_path
            )
        )


        frame_bgr = _validate_frame_bgr(
            frame_loader(
                video_path,
                anchor.frame_idx,
            )
        )


        frame_png = _validate_png_bytes(
            png_encoder(
                frame_bgr
            )
        )


        response = inference_callable(
            video_sha256=video_sha256,
            frame_idx=anchor.frame_idx,
            frame_bgr=frame_bgr,
            frame_png=frame_png,
        )


        if not isinstance(
            response,
            GeometryV2InferenceResponse,
        ):

            raise GeometryV2InferenceContractError(
                "Geometry inference returned wrong response type."
            )


        if response.frame_idx != anchor.frame_idx:

            raise GeometryV2InferenceContractError(
                "Geometry inference frame echo differs from selected anchor."
            )


        geometry_result = inference_response_to_geometry_result(
            response
        )


        result.geometry_v2 = geometry_result


        evidence = GeometryV2Stage7Evidence(
            anchor=anchor,
            video_sha256=video_sha256,
            request_frame_shape=tuple(
                int(
                    value
                )
                for value in frame_bgr.shape
            ),
            png_size_bytes=len(
                frame_png
            ),
            request_id=response.request_id,
            selected_variant=response.selected_variant,
            selection_status=response.selection_status,
            keypoint_count=response.keypoint_count,
            line_count=response.line_count,
            reference_free=response.reference_free,
        )


        metadata[
            "semantic_v2_geometry_stage7"
        ] = {
            "stage":
                GEOMETRY_V2_STAGE7_NAME,

            "anchor_policy":
                GEOMETRY_ANCHOR_SOURCE,

            "candidate_frame_count_policy":
                GEOMETRY_CANDIDATE_FRAME_COUNT,

            "temporal_fallback_enabled":
                GEOMETRY_TEMPORAL_FALLBACK_ENABLED,

            "v1_geometry_fallback_enabled":
                GEOMETRY_V1_FALLBACK_ENABLED,

            "selected_frame_idx":
                evidence.anchor.frame_idx,

            "trajectory_source":
                evidence.anchor.trajectory_source,

            "is_raw_v3_detection":
                evidence.anchor.is_raw_v3_detection,

            "is_interpolated":
                evidence.anchor.is_interpolated,

            "raw_candidate_count":
                evidence.anchor.raw_candidate_count,

            "video_sha256":
                evidence.video_sha256,

            "request_frame_shape":
                list(
                    evidence.request_frame_shape
                ),

            "png_size_bytes":
                evidence.png_size_bytes,

            "request_id":
                evidence.request_id,

            "success":
                geometry_result.success,

            "selected_variant":
                evidence.selected_variant,

            "selection_status":
                evidence.selection_status,

            "keypoint_count":
                evidence.keypoint_count,

            "line_count":
                evidence.line_count,

            "reference_free":
                evidence.reference_free,

            "failure_reason":
                geometry_result.failure_reason,
        }


        _validate_status_unchanged(
            result,
            before_status,
        )


        if not geometry_result.success:

            raise GeometryV2StageFailure(
                geometry_result.failure_reason
                or
                "Geometry-v2 inference failed."
            )


        return None


    return geometry_v2_stage7_handler
