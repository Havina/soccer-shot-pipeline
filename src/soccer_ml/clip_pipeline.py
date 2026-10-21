"""
Canonical orchestration framework for Soccer ML clip analysis.

This module defines the exact 18-stage pipeline contract established and
certified by Notebook 42.

It intentionally contains no historical Notebook-27, Notebook-28, Notebook-37,
Notebook-41, or Notebook-42 hardcoded result reuse.

Actual stage implementations are registered separately. The orchestration layer
owns stage ordering, completion, failure, and final pipeline status.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .results import PipelineStatus, SoccerClipAnalysisResult


__all__ = [
    "CANONICAL_PIPELINE_STAGES",
    "PipelineContractError",
    "canonical_stage_prefix_valid",
    "expected_next_stage",
    "mark_stage_complete",
    "mark_stage_failed",
    "mark_pipeline_degraded",
    "validate_pipeline_status",
    "SoccerClipPipeline",
    "build_input_validation_handler",
    "build_artifact_validation_handler",
    "build_video_metadata_handler",
    "register_foundation_stage_handlers",
    "build_ball_detection_handler",
    "register_ball_detection_handler",
    "BallTrackPoint",
    "build_ball_tracking_handler",
    "register_ball_tracking_handler",
    "PlayerDetectionRecord",
    "PlayerFieldSupportDecision",
    "PlayerTrackObservation",
    "PlayerTrack",
    "build_player_detection_tracking_handler",
    "register_player_detection_tracking_handler",
    "extract_refined_pitch_markings",
    "build_pitch_line_detection_handler",
    "register_pitch_line_detection_handler",
    "HOMOGRAPHY_SEARCH_VARIANTS",
    "initialize_homography_static_runtime",
    "prepare_homography_search_runtime",
    "factorized_coordinate_homographies",
    "score_pair_product_batched",
    "full_search_candidate_validity",
    "build_homography_search_handler",
    "register_homography_search_handler",
    "build_homography_validation_handler",
    "register_homography_validation_handler",
    "build_pitch_projection_handler",
    "register_pitch_projection_handler",
    "build_shooter_assignment_handler",
    "register_shooter_assignment_handler",
    "build_relative_team_assignment_handler",
    "register_relative_team_assignment_handler",
    "build_target_goal_inference_handler",
    "register_target_goal_inference_handler",
    "build_shot_geometry_handler",
    "register_shot_geometry_handler",
    "build_defender_context_handler",
    "register_defender_context_handler",
    "build_enhanced_xg_handler",
    "register_enhanced_xg_handler",
    "build_annotation_handler",
    "register_annotation_handler",
    "build_output_validation_handler",
    "register_output_validation_handler",
    "canonical_api_stage_plan",
    "run_canonical_clip_pipeline",
]


CANONICAL_PIPELINE_STAGES = (
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


_STAGE_INDEX = {
    stage: index
    for index, stage in enumerate(CANONICAL_PIPELINE_STAGES)
}


StageHandler = Callable[[SoccerClipAnalysisResult], None]


class PipelineContractError(RuntimeError):
    """Raised when the canonical pipeline state or stage order is invalid."""


def canonical_stage_prefix_valid(
    completed_stages: Sequence[str],
) -> bool:
    """Return True when completed stages form an exact canonical prefix."""

    completed = list(completed_stages)

    if len(completed) > len(CANONICAL_PIPELINE_STAGES):
        return False

    return (
        completed
        ==
        list(
            CANONICAL_PIPELINE_STAGES[
                : len(completed)
            ]
        )
    )


def expected_next_stage(
    completed_stages: Sequence[str],
) -> Optional[str]:
    """Return the next legal canonical stage, or None when complete."""

    completed = list(completed_stages)

    if not canonical_stage_prefix_valid(completed):
        raise PipelineContractError(
            "Completed stages do not form a canonical pipeline prefix."
        )

    if len(completed) == len(CANONICAL_PIPELINE_STAGES):
        return None

    return CANONICAL_PIPELINE_STAGES[
        len(completed)
    ]


def mark_stage_complete(
    status: PipelineStatus,
    stage: str,
) -> PipelineStatus:
    """
    Mark exactly the next canonical stage complete.

    Completion order is owned by this orchestration layer.
    """

    if not isinstance(status, PipelineStatus):
        raise TypeError(
            "status must be a PipelineStatus instance."
        )

    if status.failed_stage is not None:
        raise PipelineContractError(
            "Cannot complete a stage after the pipeline has failed."
        )

    if stage not in _STAGE_INDEX:
        raise PipelineContractError(
            f"Unknown canonical stage: {stage!r}"
        )

    expected = expected_next_stage(
        status.completed_stages
    )

    if expected is None:
        raise PipelineContractError(
            "Canonical pipeline is already complete."
        )

    if stage != expected:
        raise PipelineContractError(
            "Stage-order violation: "
            f"expected={expected!r}, observed={stage!r}."
        )

    status.completed_stages.append(
        stage
    )

    status.success = (
        len(status.completed_stages)
        ==
        len(CANONICAL_PIPELINE_STAGES)
        and
        status.failed_stage
        is None
    )

    return status


def mark_stage_failed(
    status: PipelineStatus,
    stage: str,
    reason: str,
) -> PipelineStatus:
    """Record failure of exactly the next canonical stage."""

    if not isinstance(status, PipelineStatus):
        raise TypeError(
            "status must be a PipelineStatus instance."
        )

    if status.failed_stage is not None:
        raise PipelineContractError(
            "Pipeline already has a failed stage."
        )

    if stage not in _STAGE_INDEX:
        raise PipelineContractError(
            f"Unknown canonical stage: {stage!r}"
        )

    expected = expected_next_stage(
        status.completed_stages
    )

    if expected is None:
        raise PipelineContractError(
            "Cannot fail a stage after the canonical pipeline is complete."
        )

    if stage != expected:
        raise PipelineContractError(
            "Failure-stage order violation: "
            f"expected={expected!r}, observed={stage!r}."
        )

    clean_reason = str(
        reason
    ).strip()

    if not clean_reason:
        clean_reason = "Unspecified stage failure."

    status.success = False
    status.failed_stage = stage

    status.warnings.append(
        f"[FAILED:{stage}] {clean_reason}"
    )

    return status


def mark_pipeline_degraded(
    status: PipelineStatus,
    reason: str,
) -> PipelineStatus:
    """Record a degraded-but-continuing pipeline state."""

    if not isinstance(status, PipelineStatus):
        raise TypeError(
            "status must be a PipelineStatus instance."
        )

    clean_reason = str(
        reason
    ).strip()

    if not clean_reason:
        raise PipelineContractError(
            "A degraded pipeline state requires a non-empty reason."
        )

    status.degraded = True
    status.degraded_mode_reason = clean_reason

    warning = (
        f"[DEGRADED] {clean_reason}"
    )

    if warning not in status.warnings:
        status.warnings.append(
            warning
        )

    return status


def validate_pipeline_status(
    status: PipelineStatus,
) -> bool:
    """Validate internal consistency of a PipelineStatus object."""

    if not isinstance(status, PipelineStatus):
        return False

    if not canonical_stage_prefix_valid(
        status.completed_stages
    ):
        return False

    completed_count = len(
        status.completed_stages
    )

    full_pipeline_complete = (
        completed_count
        ==
        len(CANONICAL_PIPELINE_STAGES)
    )

    if status.success:

        if not full_pipeline_complete:
            return False

        if status.failed_stage is not None:
            return False

    if status.failed_stage is not None:

        if status.failed_stage not in _STAGE_INDEX:
            return False

        next_stage = expected_next_stage(
            status.completed_stages
        )

        if next_stage != status.failed_stage:
            return False

        if status.success:
            return False

    if status.degraded:

        if not status.degraded_mode_reason:
            return False

    else:

        if status.degraded_mode_reason is not None:
            return False

    if not isinstance(
        status.warnings,
        list,
    ):
        return False

    return True


class SoccerClipPipeline:
    """
    Canonical 18-stage Soccer ML orchestration container.

    Stage implementations are registered independently. A handler receives the
    mutable SoccerClipAnalysisResult and performs exactly one stage's work.

    Handlers must NOT append to pipeline_status.completed_stages themselves.
    The orchestration layer owns stage completion.
    """

    def __init__(
        self,
        project_root: str | Path,
    ) -> None:

        self.project_root = Path(
            project_root
        ).resolve()

        self._stage_handlers: Dict[
            str,
            StageHandler,
        ] = {}


    @property
    def canonical_stages(self) -> tuple[str, ...]:
        """Return the immutable canonical stage sequence."""

        return CANONICAL_PIPELINE_STAGES


    def register_stage_handler(
        self,
        stage: str,
        handler: StageHandler,
    ) -> None:
        """Register one implementation for one canonical pipeline stage."""

        if stage not in _STAGE_INDEX:
            raise PipelineContractError(
                f"Unknown canonical stage: {stage!r}"
            )

        if not callable(handler):
            raise TypeError(
                "Stage handler must be callable."
            )

        if stage in self._stage_handlers:
            raise PipelineContractError(
                f"Handler already registered for stage: {stage!r}"
            )

        self._stage_handlers[
            stage
        ] = handler


    def registered_stages(self) -> List[str]:
        """Return registered stage names in canonical order."""

        return [
            stage
            for stage
            in CANONICAL_PIPELINE_STAGES
            if stage in self._stage_handlers
        ]


    def missing_stage_handlers(self) -> List[str]:
        """Return canonical stages that do not yet have implementations."""

        return [
            stage
            for stage
            in CANONICAL_PIPELINE_STAGES
            if stage not in self._stage_handlers
        ]


    @property
    def implementation_complete(self) -> bool:
        """True only when all 18 canonical stage handlers are registered."""

        return (
            self.missing_stage_handlers()
            ==
            []
        )


    def new_result(
        self,
        video_path: str | Path,
    ) -> SoccerClipAnalysisResult:
        """Create a fresh public result container for one video."""

        return SoccerClipAnalysisResult(
            video_path=str(
                Path(
                    video_path
                )
            )
        )


    def analyze(
        self,
        video_path: str | Path,
    ) -> SoccerClipAnalysisResult:
        """
        Execute all registered canonical stages.

        This method refuses to start until implementations for all 18 stages
        have been registered.

        Stage handlers should raise an exception when their stage cannot
        complete. The exception is converted into a failed public pipeline
        result at that exact stage.
        """

        missing = self.missing_stage_handlers()

        if missing:

            raise PipelineContractError(
                "Canonical SoccerClipPipeline is not fully implemented. "
                f"Missing stage handlers: {missing}"
            )

        result = self.new_result(
            video_path
        )

        for stage in CANONICAL_PIPELINE_STAGES:

            handler = self._stage_handlers[
                stage
            ]

            try:

                handler(
                    result
                )

                if result.pipeline_status.failed_stage is not None:
                    raise PipelineContractError(
                        "Stage handler directly marked pipeline failure. "
                        "Handlers must raise exceptions instead; "
                        "the orchestration layer owns failure state."
                    )

                mark_stage_complete(
                    result.pipeline_status,
                    stage,
                )

            except Exception as exc:

                if result.pipeline_status.failed_stage is None:

                    mark_stage_failed(
                        result.pipeline_status,
                        stage,
                        str(
                            exc
                        ),
                    )

                return result

        if not validate_pipeline_status(
            result.pipeline_status
        ):

            raise PipelineContractError(
                "Pipeline completed with an invalid PipelineStatus contract."
            )

        return result


# NOTEBOOK43_FOUNDATION_STAGE_HANDLERS_V1
# ====================================================================================================
# REAL FOUNDATION STAGE HANDLERS
# ====================================================================================================


def _sha256_file(path):
    """Return the SHA256 digest for one local file."""

    import hashlib

    digest = hashlib.sha256()

    with open(path, "rb") as file:

        while True:

            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def build_input_validation_handler(
    allowed_suffixes=None,
):
    """
    Build the canonical input-validation stage handler.

    The handler validates that result.video_path references a real,
    non-empty local video file.

    This stage intentionally does not decode the video. Decoding belongs
    to the video_metadata stage.
    """

    if allowed_suffixes is None:

        allowed_suffixes = (
            ".mp4",
            ".mov",
            ".avi",
            ".mkv",
            ".m4v",
            ".webm",
        )


    normalized_suffixes = {
        str(suffix).lower()
        for suffix in allowed_suffixes
    }


    def input_validation_handler(result):

        if result.video_path is None:

            raise ValueError(
                "video_path is required."
            )


        video_path = Path(
            result.video_path
        ).expanduser()


        try:

            video_path = video_path.resolve()

        except Exception:

            video_path = video_path.absolute()


        if not video_path.exists():

            raise FileNotFoundError(
                f"Input video does not exist: {video_path}"
            )


        if not video_path.is_file():

            raise ValueError(
                f"Input video is not a file: {video_path}"
            )


        size_bytes = int(
            video_path.stat().st_size
        )


        if size_bytes <= 0:

            raise ValueError(
                f"Input video is empty: {video_path}"
            )


        suffix = video_path.suffix.lower()


        if suffix not in normalized_suffixes:

            raise ValueError(
                "Unsupported input-video extension: "
                f"{suffix!r}. "
                f"Allowed={sorted(normalized_suffixes)}"
            )


        result.video_path = str(
            video_path
        )


        result.metadata[
            "input_validation"
        ] = {

            "video_path":
                str(
                    video_path
                ),

            "suffix":
                suffix,

            "size_bytes":
                size_bytes,

            "exists":
                True,

            "is_file":
                True,

            "nonempty":
                True,
        }


    return input_validation_handler


def build_artifact_validation_handler(
    project_root,
    required_artifacts,
    expected_sha256=None,
):
    """
    Build the canonical artifact-validation stage handler.

    required_artifacts
    ------------------
    Mapping:
        artifact_name -> absolute or project-root-relative path

    expected_sha256
    ---------------
    Optional mapping:
        artifact_name -> expected SHA256

    Every artifact must:
        - exist
        - be a regular file
        - be non-empty

    If an expected SHA256 is supplied, exact identity is required.
    """

    project_root = Path(
        project_root
    ).resolve()


    required_artifacts = dict(
        required_artifacts
        or {}
    )


    expected_sha256 = {

        str(name):
            str(value).lower()

        for name, value
        in dict(
            expected_sha256
            or {}
        ).items()
    }


    if not required_artifacts:

        raise ValueError(
            "artifact_validation requires at least one required artifact."
        )


    def artifact_validation_handler(result):

        artifact_records = {}


        for artifact_name, raw_path in required_artifacts.items():

            artifact_name = str(
                artifact_name
            )


            artifact_path = Path(
                raw_path
            ).expanduser()


            if not artifact_path.is_absolute():

                artifact_path = (
                    project_root
                    /
                    artifact_path
                )


            artifact_path = artifact_path.resolve()


            if not artifact_path.exists():

                raise FileNotFoundError(
                    "Required artifact does not exist: "
                    f"{artifact_name} -> {artifact_path}"
                )


            if not artifact_path.is_file():

                raise ValueError(
                    "Required artifact is not a file: "
                    f"{artifact_name} -> {artifact_path}"
                )


            size_bytes = int(
                artifact_path.stat().st_size
            )


            if size_bytes <= 0:

                raise ValueError(
                    "Required artifact is empty: "
                    f"{artifact_name} -> {artifact_path}"
                )


            actual_sha256 = _sha256_file(
                artifact_path
            )


            expected_hash = (
                expected_sha256.get(
                    artifact_name
                )
            )


            if expected_hash is None:

                sha256_match = None

            else:

                sha256_match = (
                    actual_sha256.lower()
                    ==
                    expected_hash.lower()
                )


                if not sha256_match:

                    raise ValueError(
                        "Artifact SHA256 mismatch: "
                        f"{artifact_name}. "
                        f"expected={expected_hash}, "
                        f"actual={actual_sha256}"
                    )


            artifact_records[
                artifact_name
            ] = {

                "path":
                    str(
                        artifact_path
                    ),

                "size_bytes":
                    size_bytes,

                "sha256":
                    actual_sha256,

                "expected_sha256":
                    expected_hash,

                "sha256_match":
                    sha256_match,

                "valid":
                    True,
            }


        result.metadata[
            "artifact_validation"
        ] = {

            "artifact_count":
                len(
                    artifact_records
                ),

            "all_valid":
                True,

            "artifacts":
                artifact_records,
        }


    return artifact_validation_handler


def build_video_metadata_handler():
    """
    Build the canonical real video-metadata stage handler.

    OpenCV is imported only when this stage is executed.

    The stage validates:
        - VideoCapture opens
        - frame count > 0
        - FPS > 0
        - width > 0
        - height > 0
        - first frame can actually decode
    """

    def video_metadata_handler(result):

        import cv2


        if result.video_path is None:

            raise ValueError(
                "video_metadata requires a validated video_path."
            )


        video_path = Path(
            result.video_path
        )


        if not video_path.exists():

            raise FileNotFoundError(
                f"Video disappeared before metadata read: {video_path}"
            )


        capture = cv2.VideoCapture(
            str(
                video_path
            )
        )


        try:

            if not capture.isOpened():

                raise RuntimeError(
                    f"OpenCV could not open input video: {video_path}"
                )


            frame_count = int(
                round(
                    capture.get(
                        cv2.CAP_PROP_FRAME_COUNT
                    )
                )
            )


            fps = float(
                capture.get(
                    cv2.CAP_PROP_FPS
                )
            )


            width = int(
                round(
                    capture.get(
                        cv2.CAP_PROP_FRAME_WIDTH
                    )
                )
            )


            height = int(
                round(
                    capture.get(
                        cv2.CAP_PROP_FRAME_HEIGHT
                    )
                )
            )


            fourcc_value = int(
                capture.get(
                    cv2.CAP_PROP_FOURCC
                )
            )


            if frame_count <= 0:

                raise RuntimeError(
                    "Video metadata reported a non-positive frame count."
                )


            if not (fps > 0.0):

                raise RuntimeError(
                    "Video metadata reported a non-positive FPS."
                )


            if width <= 0 or height <= 0:

                raise RuntimeError(
                    "Video metadata reported invalid dimensions."
                )


            decode_ok, first_frame = capture.read()


            if (
                not decode_ok
                or
                first_frame is None
                or
                getattr(
                    first_frame,
                    "size",
                    0,
                )
                <=
                0
            ):

                raise RuntimeError(
                    "Input video opened but the first frame could not be decoded."
                )


        finally:

            capture.release()


        codec_chars = []


        for index in range(
            4
        ):

            value = (
                fourcc_value
                >>
                (
                    8
                    *
                    index
                )
            ) & 0xFF


            if value:

                codec_chars.append(
                    chr(
                        value
                    )
                )


        codec = "".join(
            codec_chars
        )


        duration_seconds = (
            float(
                frame_count
            )
            /
            float(
                fps
            )
        )


        result.metadata[
            "video_metadata"
        ] = {

            "frame_count":
                frame_count,

            "fps":
                fps,

            "width":
                width,

            "height":
                height,

            "duration_seconds":
                duration_seconds,

            "fourcc":
                codec,

            "first_frame_decodable":
                True,
        }


    return video_metadata_handler


def register_foundation_stage_handlers(
    pipeline,
    required_artifacts,
    expected_sha256=None,
    allowed_video_suffixes=None,
):
    """
    Register the first three canonical stage implementations.

    Registration order:
        1. input_validation
        2. artifact_validation
        3. video_metadata
    """

    pipeline.register_stage_handler(
        "input_validation",
        build_input_validation_handler(
            allowed_suffixes=allowed_video_suffixes,
        ),
    )


    pipeline.register_stage_handler(
        "artifact_validation",
        build_artifact_validation_handler(
            project_root=pipeline.project_root,
            required_artifacts=required_artifacts,
            expected_sha256=expected_sha256,
        ),
    )


    pipeline.register_stage_handler(
        "video_metadata",
        build_video_metadata_handler(),
    )


    return pipeline


# NOTEBOOK43_FROZEN_V3_BALL_DETECTION_V1
# ====================================================================================================
# FROZEN V3 BALL-DETECTION STAGE
# ====================================================================================================


def build_ball_detection_handler(
    model_path,
    expected_sha256,
    artifact_name="v3_ball_model",
    confidence_threshold=0.4916697144508362,
    iou_threshold=0.7,
    image_size=1280,
    device=0,
    batch=1,
    half=False,
    max_det=300,
    classes=(0,),
):
    """
    Build the canonical frozen-V3 ball-detection stage.

    This stage performs real YOLO inference over every input-video frame.

    The implementation deliberately uses YOLO.predict rather than tracking.
    Ball temporal tracking belongs to the next canonical stage.
    """

    model_path = Path(
        model_path
    ).expanduser().resolve()


    expected_sha256 = str(
        expected_sha256
    ).lower()


    confidence_threshold = float(
        confidence_threshold
    )


    iou_threshold = float(
        iou_threshold
    )


    image_size = int(
        image_size
    )


    batch = int(
        batch
    )


    max_det = int(
        max_det
    )


    classes = tuple(
        int(
            class_id
        )
        for class_id
        in classes
    )


    if not classes:

        raise ValueError(
            "Ball detection requires at least one class ID."
        )


    def ball_detection_handler(result):

        import math

        import torch

        from ultralytics import YOLO


        # --------------------------------------------------------------------------------------------
        # CANONICAL STAGE-ORDER GUARD
        # --------------------------------------------------------------------------------------------

        expected_prefix = list(
            CANONICAL_PIPELINE_STAGES[
                : _STAGE_INDEX[
                    "ball_detection"
                ]
            ]
        )


        if (
            result.pipeline_status.completed_stages
            !=
            expected_prefix
        ):

            raise PipelineContractError(
                "Ball detection cannot run before required stages: "
                f"{expected_prefix}"
            )


        # --------------------------------------------------------------------------------------------
        # VIDEO CONTRACT
        # --------------------------------------------------------------------------------------------

        if result.video_path is None:

            raise ValueError(
                "Ball detection requires a validated video path."
            )


        video_path = Path(
            result.video_path
        ).resolve()


        if not video_path.exists():

            raise FileNotFoundError(
                f"Input video does not exist: {video_path}"
            )


        video_metadata = result.metadata.get(
            "video_metadata"
        )


        if not isinstance(
            video_metadata,
            dict,
        ):

            raise PipelineContractError(
                "Ball detection requires completed video metadata."
            )


        expected_frame_count = int(
            video_metadata.get(
                "frame_count",
                0,
            )
        )


        if expected_frame_count <= 0:

            raise PipelineContractError(
                "Video metadata does not contain a valid frame count."
            )


        # --------------------------------------------------------------------------------------------
        # ARTIFACT VALIDATION CONTRACT
        # --------------------------------------------------------------------------------------------

        artifact_validation = result.metadata.get(
            "artifact_validation"
        )


        if not isinstance(
            artifact_validation,
            dict,
        ):

            raise PipelineContractError(
                "Ball detection requires completed artifact validation."
            )


        artifact_records = artifact_validation.get(
            "artifacts",
            {},
        )


        model_record = artifact_records.get(
            artifact_name
        )


        if not isinstance(
            model_record,
            dict,
        ):

            raise PipelineContractError(
                f"Validated artifact record missing: {artifact_name}"
            )


        validated_model_path = Path(
            model_record[
                "path"
            ]
        ).resolve()


        if validated_model_path != model_path:

            raise PipelineContractError(
                "Validated V3 model path does not match requested model path."
            )


        validated_sha256 = str(
            model_record.get(
                "sha256",
                "",
            )
        ).lower()


        if (
            validated_sha256
            !=
            expected_sha256
        ):

            raise PipelineContractError(
                "Validated V3 model SHA256 does not match the frozen identity."
            )


        # --------------------------------------------------------------------------------------------
        # LOCAL CUDA CONTRACT
        # --------------------------------------------------------------------------------------------

        if isinstance(
            device,
            int,
        ):

            if device >= 0:

                if not torch.cuda.is_available():

                    raise RuntimeError(
                        "Frozen V3 policy requires CUDA but CUDA is unavailable."
                    )


                if device >= torch.cuda.device_count():

                    raise RuntimeError(
                        "Requested CUDA device does not exist: "
                        f"{device}"
                    )


        # --------------------------------------------------------------------------------------------
        # MODEL LOAD
        # --------------------------------------------------------------------------------------------

        model = YOLO(
            str(
                model_path
            )
        )


        model_names = getattr(
            model,
            "names",
            {},
        )


        if isinstance(
            model_names,
            dict,
        ):

            class_zero_name = (
                model_names.get(
                    0
                )
            )


            if class_zero_name is None:

                class_zero_name = (
                    model_names.get(
                        "0"
                    )
                )

        elif isinstance(
            model_names,
            (
                list,
                tuple,
            ),
        ):

            class_zero_name = (
                model_names[
                    0
                ]
                if len(
                    model_names
                )
                >
                0
                else None
            )

        else:

            class_zero_name = None


        if (
            class_zero_name is not None
            and
            str(
                class_zero_name
            ).strip().lower()
            !=
            "ball"
        ):

            raise RuntimeError(
                "Frozen V3 class-0 contract is invalid: "
                f"{class_zero_name!r}"
            )


        # --------------------------------------------------------------------------------------------
        # CHECKPOINT IMAGE-SIZE AUDIT
        # --------------------------------------------------------------------------------------------

        checkpoint_imgsz = None


        checkpoint = getattr(
            model,
            "ckpt",
            None,
        )


        if isinstance(
            checkpoint,
            dict,
        ):

            train_args = checkpoint.get(
                "train_args"
            )


            if isinstance(
                train_args,
                dict,
            ):

                checkpoint_imgsz = (
                    train_args.get(
                        "imgsz"
                    )
                )


        checkpoint_imgsz_normalized = None


        if checkpoint_imgsz is not None:

            if isinstance(
                checkpoint_imgsz,
                (
                    list,
                    tuple,
                ),
            ):

                checkpoint_values = [
                    int(
                        value
                    )
                    for value
                    in checkpoint_imgsz
                ]


                if len(
                    checkpoint_values
                ) == 1:

                    checkpoint_imgsz_normalized = (
                        checkpoint_values[
                            0
                        ]
                    )

                elif len(
                    set(
                        checkpoint_values
                    )
                ) == 1:

                    checkpoint_imgsz_normalized = (
                        checkpoint_values[
                            0
                        ]
                    )

            else:

                try:

                    checkpoint_imgsz_normalized = int(
                        checkpoint_imgsz
                    )

                except Exception:

                    checkpoint_imgsz_normalized = None


        if (
            checkpoint_imgsz_normalized
            is not None
            and
            checkpoint_imgsz_normalized
            !=
            image_size
        ):

            raise RuntimeError(
                "Frozen V3 image-size contract mismatch: "
                f"checkpoint={checkpoint_imgsz_normalized}, "
                f"requested={image_size}"
            )


        # --------------------------------------------------------------------------------------------
        # REAL FROZEN V3 INFERENCE
        # --------------------------------------------------------------------------------------------

        prediction_stream = model.predict(

            source=str(
                video_path
            ),

            conf=confidence_threshold,

            iou=iou_threshold,

            imgsz=image_size,

            device=device,

            batch=batch,

            half=half,

            max_det=max_det,

            classes=list(
                classes
            ),

            stream=True,

            verbose=False,
        )


        detection_records = []


        frames_processed = 0


        detection_frame_ids = []


        for frame_number, prediction in enumerate(
            prediction_stream
        ):

            frames_processed += 1


            boxes = getattr(
                prediction,
                "boxes",
                None,
            )


            if boxes is None:

                continue


            if len(
                boxes
            ) == 0:

                continue


            xyxy_values = (
                boxes.xyxy
                .detach()
                .cpu()
                .numpy()
            )


            confidence_values = (
                boxes.conf
                .detach()
                .cpu()
                .numpy()
            )


            class_values = (
                boxes.cls
                .detach()
                .cpu()
                .numpy()
            )


            frame_detection_count = 0


            for detection_index in range(
                len(
                    confidence_values
                )
            ):

                class_id = int(
                    class_values[
                        detection_index
                    ]
                )


                if class_id not in classes:

                    continue


                confidence = float(
                    confidence_values[
                        detection_index
                    ]
                )


                x1, y1, x2, y2 = [

                    float(
                        value
                    )

                    for value
                    in xyxy_values[
                        detection_index
                    ].tolist()
                ]


                values_to_validate = [

                    confidence,

                    x1,
                    y1,
                    x2,
                    y2,
                ]


                if not all(
                    math.isfinite(
                        value
                    )
                    for value
                    in values_to_validate
                ):

                    raise RuntimeError(
                        "V3 inference emitted a non-finite detection."
                    )


                if confidence < confidence_threshold:

                    raise RuntimeError(
                        "V3 inference emitted a detection below "
                        "the frozen threshold."
                    )


                if not (
                    x2 > x1
                    and
                    y2 > y1
                ):

                    raise RuntimeError(
                        "V3 inference emitted an invalid bounding box."
                    )


                center_x = (
                    x1
                    +
                    x2
                ) / 2.0


                center_y = (
                    y1
                    +
                    y2
                ) / 2.0


                if isinstance(
                    model_names,
                    dict,
                ):

                    class_name = (
                        model_names.get(
                            class_id
                        )
                    )


                    if class_name is None:

                        class_name = (
                            model_names.get(
                                str(
                                    class_id
                                )
                            )
                        )

                elif isinstance(
                    model_names,
                    (
                        list,
                        tuple,
                    ),
                ):

                    class_name = (
                        model_names[
                            class_id
                        ]
                        if (
                            0
                            <=
                            class_id
                            <
                            len(
                                model_names
                            )
                        )
                        else None
                    )

                else:

                    class_name = None


                detection_records.append(
                    {

                        "frame_number":
                            int(
                                frame_number
                            ),

                        "detection_id":
                            int(
                                frame_detection_count
                            ),

                        "class_id":
                            int(
                                class_id
                            ),

                        "class_name":
                            (
                                str(
                                    class_name
                                )
                                if class_name
                                is not None
                                else None
                            ),

                        "confidence":
                            float(
                                confidence
                            ),

                        "x1":
                            float(
                                x1
                            ),

                        "y1":
                            float(
                                y1
                            ),

                        "x2":
                            float(
                                x2
                            ),

                        "y2":
                            float(
                                y2
                            ),

                        "center_x":
                            float(
                                center_x
                            ),

                        "center_y":
                            float(
                                center_y
                            ),
                    }
                )


                frame_detection_count += 1


            if frame_detection_count > 0:

                detection_frame_ids.append(
                    int(
                        frame_number
                    )
                )


        # --------------------------------------------------------------------------------------------
        # FULL VIDEO COVERAGE
        # --------------------------------------------------------------------------------------------

        if (
            frames_processed
            !=
            expected_frame_count
        ):

            raise RuntimeError(
                "V3 inference frame coverage mismatch: "
                f"expected={expected_frame_count}, "
                f"processed={frames_processed}"
            )


        # --------------------------------------------------------------------------------------------
        # NO-BALL FAILURE CONTRACT
        # --------------------------------------------------------------------------------------------

        if len(
            detection_records
        ) == 0:

            raise RuntimeError(
                "Frozen V3 detector found no ball detections in the input video."
            )


        # --------------------------------------------------------------------------------------------
        # PUBLIC BallTrackResult
        # --------------------------------------------------------------------------------------------

        result.ball_track.success = True


        result.ball_track.detections = (
            detection_records
        )


        result.ball_track.track = None


        result.ball_track.raw_detection_frame_count = (
            len(
                detection_frame_ids
            )
        )


        result.ball_track.tracked_frame_count = 0


        result.ball_track.failure_reason = None


        # --------------------------------------------------------------------------------------------
        # STAGE METADATA
        # --------------------------------------------------------------------------------------------

        confidences = [

            float(
                record[
                    "confidence"
                ]
            )

            for record
            in detection_records
        ]


        result.metadata[
            "ball_detection"
        ] = {

            "method":
                "YOLO_PREDICT_NOT_TRACK",

            "model_path":
                str(
                    model_path
                ),

            "model_sha256":
                expected_sha256,

            "class_zero_name":
                (
                    str(
                        class_zero_name
                    )
                    if class_zero_name
                    is not None
                    else None
                ),

            "confidence_threshold":
                confidence_threshold,

            "iou_threshold":
                iou_threshold,

            "imgsz":
                image_size,

            "checkpoint_imgsz":
                checkpoint_imgsz_normalized,

            "device":
                device,

            "batch":
                batch,

            "half":
                bool(
                    half
                ),

            "max_det":
                max_det,

            "classes":
                list(
                    classes
                ),

            "stream":
                True,

            "frames_processed":
                int(
                    frames_processed
                ),

            "frames_with_detections":
                int(
                    len(
                        detection_frame_ids
                    )
                ),

            "total_detections":
                int(
                    len(
                        detection_records
                    )
                ),

            "detection_frame_ids":
                detection_frame_ids,

            "minimum_confidence":
                float(
                    min(
                        confidences
                    )
                ),

            "maximum_confidence":
                float(
                    max(
                        confidences
                    )
                ),

            "all_classes_valid":
                all(
                    int(
                        record[
                            "class_id"
                        ]
                    )
                    in
                    classes

                    for record
                    in detection_records
                ),

            "all_confidences_at_or_above_threshold":
                all(
                    float(
                        record[
                            "confidence"
                        ]
                    )
                    >=
                    confidence_threshold

                    for record
                    in detection_records
                ),
        }


    return ball_detection_handler


def register_ball_detection_handler(
    pipeline,
    model_path,
    expected_sha256,
    artifact_name="v3_ball_model",
    confidence_threshold=0.4916697144508362,
    iou_threshold=0.7,
    image_size=1280,
    device=0,
    batch=1,
    half=False,
    max_det=300,
    classes=(0,),
):
    """Register the canonical frozen-V3 ball-detection handler."""

    pipeline.register_stage_handler(

        "ball_detection",

        build_ball_detection_handler(

            model_path=model_path,

            expected_sha256=expected_sha256,

            artifact_name=artifact_name,

            confidence_threshold=confidence_threshold,

            iou_threshold=iou_threshold,

            image_size=image_size,

            device=device,

            batch=batch,

            half=half,

            max_det=max_det,

            classes=classes,
        ),
    )


    return pipeline


# NOTEBOOK43_CANONICAL_BALL_TRACKING_V1
# ====================================================================================================
# CANONICAL BALL-TRACKING STAGE
# ====================================================================================================

from dataclasses import dataclass


@dataclass
class BallTrackPoint:
    """One canonical point in the operational ball trajectory."""

    frame_idx: int

    x1: float
    y1: float
    x2: float
    y2: float

    center_x: float
    center_y: float

    width: float
    height: float

    confidence: float

    trajectory_source: str

    is_raw_v3_detection: bool

    is_interpolated: bool

    interpolation_left_frame: Optional[int]

    interpolation_right_frame: Optional[int]

    interpolation_fraction: Optional[float]


def _ball_center_distance(
    left,
    right,
):
    """Euclidean center distance between two raw ball detections."""

    dx = (
        float(
            right["center_x"]
        )
        -
        float(
            left["center_x"]
        )
    )


    dy = (
        float(
            right["center_y"]
        )
        -
        float(
            left["center_y"]
        )
    )


    return float(
        (
            dx * dx
            +
            dy * dy
        )
        ** 0.5
    )


def _ball_detection_frame(
    detection,
):
    """Resolve canonical zero-based clip-frame index."""

    return int(
        detection[
            "frame_number"
        ]
    )


def _build_raw_ball_point(
    detection,
):
    """Convert one V3 detection into a canonical raw BallTrackPoint."""

    x1 = float(
        detection[
            "x1"
        ]
    )

    y1 = float(
        detection[
            "y1"
        ]
    )

    x2 = float(
        detection[
            "x2"
        ]
    )

    y2 = float(
        detection[
            "y2"
        ]
    )


    return BallTrackPoint(

        frame_idx=int(
            detection[
                "frame_number"
            ]
        ),

        x1=x1,

        y1=y1,

        x2=x2,

        y2=y2,

        center_x=float(
            detection[
                "center_x"
            ]
        ),

        center_y=float(
            detection[
                "center_y"
            ]
        ),

        width=float(
            x2
            -
            x1
        ),

        height=float(
            y2
            -
            y1
        ),

        confidence=float(
            detection[
                "confidence"
            ]
        ),

        trajectory_source="RAW_V3",

        is_raw_v3_detection=True,

        is_interpolated=False,

        interpolation_left_frame=None,

        interpolation_right_frame=None,

        interpolation_fraction=None,
    )


def _interpolate_ball_point(
    left_point,
    right_point,
    frame_idx,
):
    """Linearly interpolate one eligible missing ball frame."""

    frame_idx = int(
        frame_idx
    )


    frame_gap = (
        int(
            right_point.frame_idx
        )
        -
        int(
            left_point.frame_idx
        )
    )


    if frame_gap <= 1:

        raise ValueError(
            "Interpolation requires at least one missing frame."
        )


    alpha = (
        float(
            frame_idx
            -
            left_point.frame_idx
        )
        /
        float(
            frame_gap
        )
    )


    if not (
        0.0
        <
        alpha
        <
        1.0
    ):

        raise ValueError(
            "Interpolation fraction must lie strictly between 0 and 1."
        )


    center_x = (
        float(
            left_point.center_x
        )
        +
        alpha
        *
        (
            float(
                right_point.center_x
            )
            -
            float(
                left_point.center_x
            )
        )
    )


    center_y = (
        float(
            left_point.center_y
        )
        +
        alpha
        *
        (
            float(
                right_point.center_y
            )
            -
            float(
                left_point.center_y
            )
        )
    )


    width = (
        float(
            left_point.width
        )
        +
        alpha
        *
        (
            float(
                right_point.width
            )
            -
            float(
                left_point.width
            )
        )
    )


    height = (
        float(
            left_point.height
        )
        +
        alpha
        *
        (
            float(
                right_point.height
            )
            -
            float(
                left_point.height
            )
        )
    )


    return BallTrackPoint(

        frame_idx=frame_idx,

        x1=float(
            center_x
            -
            width
            /
            2.0
        ),

        y1=float(
            center_y
            -
            height
            /
            2.0
        ),

        x2=float(
            center_x
            +
            width
            /
            2.0
        ),

        y2=float(
            center_y
            +
            height
            /
            2.0
        ),

        center_x=float(
            center_x
        ),

        center_y=float(
            center_y
        ),

        width=float(
            width
        ),

        height=float(
            height
        ),

        confidence=float(
            "nan"
        ),

        trajectory_source="LINEAR_INTERPOLATION",

        is_raw_v3_detection=False,

        is_interpolated=True,

        interpolation_left_frame=int(
            left_point.frame_idx
        ),

        interpolation_right_frame=int(
            right_point.frame_idx
        ),

        interpolation_fraction=float(
            alpha
        ),
    )


def build_ball_tracking_handler(
    max_frame_gap=3,
    gate_multiplier=1.25,
):
    """
    Build the canonical Notebook-42 ball-tracking stage.

    Policy
    ------
    1. Derive a characteristic speed from consecutive-frame V3 detections.
    2. Set motion gate = characteristic_speed * gate_multiplier.
    3. Build tracklets using greedy temporal-spatial association.
    4. Rank tracklets by:
         - net displacement/frame
         - path speed/frame
         - detection count
         - mean confidence
    5. Select rank-1 tracklet.
    6. Linearly interpolate eligible gaps within the selected tracklet.
    """

    max_frame_gap = int(
        max_frame_gap
    )


    gate_multiplier = float(
        gate_multiplier
    )


    if max_frame_gap < 1:

        raise ValueError(
            "max_frame_gap must be >= 1."
        )


    if gate_multiplier <= 0.0:

        raise ValueError(
            "gate_multiplier must be > 0."
        )


    def ball_tracking_handler(result):

        import math


        # --------------------------------------------------------------------------------------------
        # STAGE ORDER
        # --------------------------------------------------------------------------------------------

        expected_prefix = list(
            CANONICAL_PIPELINE_STAGES[
                : _STAGE_INDEX[
                    "ball_tracking"
                ]
            ]
        )


        if (
            result.pipeline_status.completed_stages
            !=
            expected_prefix
        ):

            raise PipelineContractError(
                "Ball tracking cannot run before successful ball detection."
            )


        # --------------------------------------------------------------------------------------------
        # PUBLIC DETECTION CONTRACT
        # --------------------------------------------------------------------------------------------

        if not result.ball_track.success:

            raise PipelineContractError(
                "Ball tracking requires successful ball detection."
            )


        raw_detections = result.ball_track.detections


        if not isinstance(
            raw_detections,
            list,
        ):

            raise PipelineContractError(
                "BallTrackResult.detections must be a list."
            )


        if len(
            raw_detections
        ) == 0:

            raise RuntimeError(
                "Ball tracking received zero raw detections."
            )


        required_keys = {
            "frame_number",
            "detection_id",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "center_x",
            "center_y",
        }


        for detection in raw_detections:

            if not required_keys.issubset(
                detection.keys()
            ):

                raise RuntimeError(
                    "Ball detection record is missing required tracking fields."
                )


        detections = sorted(

            raw_detections,

            key=lambda record:
                (
                    int(
                        record[
                            "frame_number"
                        ]
                    ),

                    int(
                        record.get(
                            "detection_id",
                            0,
                        )
                    ),
                ),
        )


        # --------------------------------------------------------------------------------------------
        # CHARACTERISTIC SPEED
        # --------------------------------------------------------------------------------------------
        #
        # Notebook-42 used the maximum center displacement observed between
        # consecutive-frame raw detections.
        #
        # --------------------------------------------------------------------------------------------

        consecutive_speeds = []


        for left_detection, right_detection in zip(

            detections[
                :-1
            ],

            detections[
                1:
            ],

        ):

            left_frame = _ball_detection_frame(
                left_detection
            )


            right_frame = _ball_detection_frame(
                right_detection
            )


            frame_gap = (
                right_frame
                -
                left_frame
            )


            if frame_gap == 1:

                consecutive_speeds.append(
                    _ball_center_distance(
                        left_detection,
                        right_detection,
                    )
                )


        # Fallback only for clips with no directly consecutive raw detections.

        if not consecutive_speeds:

            fallback_speeds = []


            for left_detection, right_detection in zip(

                detections[
                    :-1
                ],

                detections[
                    1:
                ],

            ):

                left_frame = _ball_detection_frame(
                    left_detection
                )


                right_frame = _ball_detection_frame(
                    right_detection
                )


                frame_gap = (
                    right_frame
                    -
                    left_frame
                )


                if (
                    1
                    <=
                    frame_gap
                    <=
                    max_frame_gap
                ):

                    fallback_speeds.append(
                        _ball_center_distance(
                            left_detection,
                            right_detection,
                        )
                        /
                        float(
                            frame_gap
                        )
                    )


            consecutive_speeds = (
                fallback_speeds
            )


        if not consecutive_speeds:

            raise RuntimeError(
                "Unable to estimate ball characteristic speed."
            )


        characteristic_speed = float(
            max(
                consecutive_speeds
            )
        )


        if not (
            math.isfinite(
                characteristic_speed
            )
            and
            characteristic_speed
            >
            0.0
        ):

            raise RuntimeError(
                "Invalid ball characteristic speed."
            )


        spatial_gate_per_frame = float(
            characteristic_speed
            *
            gate_multiplier
        )


        # --------------------------------------------------------------------------------------------
        # GREEDY TRACKLET ASSOCIATION
        # --------------------------------------------------------------------------------------------

        tracklets = []


        next_tracklet_id = 1


        detections_by_frame = {}


        for detection in detections:

            frame_number = int(
                detection[
                    "frame_number"
                ]
            )


            detections_by_frame.setdefault(
                frame_number,
                [],
            ).append(
                detection
            )


        for frame_number in sorted(
            detections_by_frame.keys()
        ):

            frame_detections = (
                detections_by_frame[
                    frame_number
                ]
            )


            candidate_edges = []


            for tracklet in tracklets:

                last_detection = (
                    tracklet[
                        "detections"
                    ][
                        -1
                    ]
                )


                last_frame = int(
                    last_detection[
                        "frame_number"
                    ]
                )


                frame_gap = (
                    int(
                        frame_number
                    )
                    -
                    last_frame
                )


                if not (
                    1
                    <=
                    frame_gap
                    <=
                    max_frame_gap
                ):

                    continue


                for detection_index, detection in enumerate(
                    frame_detections
                ):

                    displacement = (
                        _ball_center_distance(
                            last_detection,
                            detection,
                        )
                    )


                    displacement_per_frame = (
                        displacement
                        /
                        float(
                            frame_gap
                        )
                    )


                    if (
                        displacement_per_frame
                        <=
                        spatial_gate_per_frame
                    ):

                        candidate_edges.append(
                            (
                                float(
                                    displacement_per_frame
                                ),

                                int(
                                    frame_gap
                                ),

                                int(
                                    tracklet[
                                        "tracklet_id"
                                    ]
                                ),

                                int(
                                    detection_index
                                ),

                                tracklet,

                                detection,
                            )
                        )


            candidate_edges.sort(
                key=lambda edge:
                    (
                        edge[
                            0
                        ],

                        edge[
                            1
                        ],

                        edge[
                            2
                        ],

                        edge[
                            3
                        ],
                    )
            )


            assigned_tracklets = set()

            assigned_detection_indices = set()


            for edge in candidate_edges:

                (
                    _,
                    _,
                    tracklet_id,
                    detection_index,
                    tracklet,
                    detection,
                ) = edge


                if (
                    tracklet_id
                    in
                    assigned_tracklets
                ):

                    continue


                if (
                    detection_index
                    in
                    assigned_detection_indices
                ):

                    continue


                tracklet[
                    "detections"
                ].append(
                    detection
                )


                assigned_tracklets.add(
                    tracklet_id
                )


                assigned_detection_indices.add(
                    detection_index
                )


            for detection_index, detection in enumerate(
                frame_detections
            ):

                if (
                    detection_index
                    in
                    assigned_detection_indices
                ):

                    continue


                tracklets.append(
                    {
                        "tracklet_id":
                            int(
                                next_tracklet_id
                            ),

                        "detections":
                            [
                                detection
                            ],
                    }
                )


                next_tracklet_id += 1


        # --------------------------------------------------------------------------------------------
        # TRACKLET METRICS
        # --------------------------------------------------------------------------------------------

        tracklet_summaries = []


        for tracklet in tracklets:

            observations = sorted(

                tracklet[
                    "detections"
                ],

                key=lambda detection:
                    int(
                        detection[
                            "frame_number"
                        ]
                    ),
            )


            first_detection = (
                observations[
                    0
                ]
            )


            last_detection = (
                observations[
                    -1
                ]
            )


            first_frame = int(
                first_detection[
                    "frame_number"
                ]
            )


            last_frame = int(
                last_detection[
                    "frame_number"
                ]
            )


            frame_span = (
                last_frame
                -
                first_frame
            )


            if frame_span > 0:

                net_displacement = (
                    _ball_center_distance(
                        first_detection,
                        last_detection,
                    )
                )


                net_displacement_per_frame = (
                    net_displacement
                    /
                    float(
                        frame_span
                    )
                )

            else:

                net_displacement = 0.0

                net_displacement_per_frame = 0.0


            path_distance = 0.0


            for left_detection, right_detection in zip(

                observations[
                    :-1
                ],

                observations[
                    1:
                ],

            ):

                path_distance += (
                    _ball_center_distance(
                        left_detection,
                        right_detection,
                    )
                )


            if frame_span > 0:

                path_speed_per_frame = (
                    path_distance
                    /
                    float(
                        frame_span
                    )
                )

            else:

                path_speed_per_frame = 0.0


            mean_confidence = (
                sum(
                    float(
                        detection[
                            "confidence"
                        ]
                    )

                    for detection
                    in observations
                )
                /
                float(
                    len(
                        observations
                    )
                )
            )


            summary = {

                "tracklet_id":
                    int(
                        tracklet[
                            "tracklet_id"
                        ]
                    ),

                "detection_count":
                    int(
                        len(
                            observations
                        )
                    ),

                "first_frame":
                    int(
                        first_frame
                    ),

                "last_frame":
                    int(
                        last_frame
                    ),

                "frame_span":
                    int(
                        frame_span
                    ),

                "frame_ids":
                    [
                        int(
                            detection[
                                "frame_number"
                            ]
                        )

                        for detection
                        in observations
                    ],

                "net_displacement":
                    float(
                        net_displacement
                    ),

                "net_displacement_per_frame":
                    float(
                        net_displacement_per_frame
                    ),

                "path_distance":
                    float(
                        path_distance
                    ),

                "path_speed_per_frame":
                    float(
                        path_speed_per_frame
                    ),

                "mean_confidence":
                    float(
                        mean_confidence
                    ),

                "_observations":
                    observations,
            }


            tracklet_summaries.append(
                summary
            )


        if not tracklet_summaries:

            raise RuntimeError(
                "Ball tracking produced no tracklets."
            )


        # --------------------------------------------------------------------------------------------
        # PRIMARY TRACKLET RANKING
        # --------------------------------------------------------------------------------------------

        ranked_tracklets = sorted(

            tracklet_summaries,

            key=lambda summary:
                (
                    -float(
                        summary[
                            "net_displacement_per_frame"
                        ]
                    ),

                    -float(
                        summary[
                            "path_speed_per_frame"
                        ]
                    ),

                    -int(
                        summary[
                            "detection_count"
                        ]
                    ),

                    -float(
                        summary[
                            "mean_confidence"
                        ]
                    ),

                    int(
                        summary[
                            "tracklet_id"
                        ]
                    ),
                ),
        )


        selected_summary = (
            ranked_tracklets[
                0
            ]
        )


        selected_raw_detections = (
            selected_summary[
                "_observations"
            ]
        )


        if len(
            selected_raw_detections
        ) < 2:

            raise RuntimeError(
                "Selected ball tracklet contains fewer than two detections."
            )


        # --------------------------------------------------------------------------------------------
        # BUILD RAW POINTS
        # --------------------------------------------------------------------------------------------

        raw_points = [

            _build_raw_ball_point(
                detection
            )

            for detection
            in selected_raw_detections
        ]


        # --------------------------------------------------------------------------------------------
        # INTERPOLATE ELIGIBLE GAPS
        # --------------------------------------------------------------------------------------------

        trajectory = []


        for index in range(
            len(
                raw_points
            )
            -
            1
        ):

            left_point = (
                raw_points[
                    index
                ]
            )


            right_point = (
                raw_points[
                    index
                    +
                    1
                ]
            )


            trajectory.append(
                left_point
            )


            frame_gap = (
                int(
                    right_point.frame_idx
                )
                -
                int(
                    left_point.frame_idx
                )
            )


            if frame_gap > max_frame_gap:

                raise RuntimeError(
                    "Selected tracklet contains a gap larger than the "
                    "configured interpolation contract."
                )


            for missing_frame in range(

                int(
                    left_point.frame_idx
                )
                +
                1,

                int(
                    right_point.frame_idx
                ),

            ):

                trajectory.append(
                    _interpolate_ball_point(

                        left_point,

                        right_point,

                        missing_frame,
                    )
                )


        trajectory.append(
            raw_points[
                -1
            ]
        )


        trajectory = sorted(
            trajectory,
            key=lambda point:
                int(
                    point.frame_idx
                ),
        )


        # --------------------------------------------------------------------------------------------
        # CONTIGUOUS TRAJECTORY CONTRACT
        # --------------------------------------------------------------------------------------------

        trajectory_frames = [

            int(
                point.frame_idx
            )

            for point
            in trajectory
        ]


        expected_contiguous_frames = list(
            range(
                trajectory_frames[
                    0
                ],

                trajectory_frames[
                    -1
                ]
                +
                1,
            )
        )


        if (
            trajectory_frames
            !=
            expected_contiguous_frames
        ):

            raise RuntimeError(
                "Canonical ball trajectory is not frame-contiguous."
            )


        # --------------------------------------------------------------------------------------------
        # PUBLIC RESULT
        # --------------------------------------------------------------------------------------------

        result.ball_track.track = (
            trajectory
        )


        result.ball_track.tracked_frame_count = (
            len(
                trajectory
            )
        )


        result.ball_track.success = True

        result.ball_track.failure_reason = None


        # --------------------------------------------------------------------------------------------
        # CLEAN TRACKLET SUMMARIES FOR METADATA
        # --------------------------------------------------------------------------------------------

        public_tracklet_summaries = []


        for rank, summary in enumerate(
            ranked_tracklets,
            start=1,
        ):

            public_tracklet_summaries.append(
                {
                    key:
                        value

                    for key, value
                    in summary.items()

                    if key
                    !=
                    "_observations"
                }
                |
                {
                    "rank":
                        int(
                            rank
                        )
                }
            )


        # --------------------------------------------------------------------------------------------
        # STAGE METADATA
        # --------------------------------------------------------------------------------------------

        result.metadata[
            "ball_tracking"
        ] = {

            "method":
                "GREEDY_TEMPORAL_SPATIAL_CONTINUITY",

            "max_frame_gap":
                int(
                    max_frame_gap
                ),

            "max_missing_frames":
                int(
                    max_frame_gap
                    -
                    1
                ),

            "characteristic_speed_px_per_frame":
                float(
                    characteristic_speed
                ),

            "gate_multiplier":
                float(
                    gate_multiplier
                ),

            "spatial_gate_px_per_frame":
                float(
                    spatial_gate_per_frame
                ),

            "association_priority":
                "LOWEST_DISPLACEMENT_PER_FRAME",

            "selection_priority":
                [
                    "NET_DISPLACEMENT_PER_FRAME_DESC",
                    "PATH_SPEED_PER_FRAME_DESC",
                    "DETECTION_COUNT_DESC",
                    "MEAN_CONFIDENCE_DESC",
                ],

            "raw_detection_count":
                int(
                    len(
                        raw_detections
                    )
                ),

            "tracklet_count":
                int(
                    len(
                        tracklets
                    )
                ),

            "tracklets":
                public_tracklet_summaries,

            "selected_tracklet_id":
                int(
                    selected_summary[
                        "tracklet_id"
                    ]
                ),

            "selected_raw_frames":
                [
                    int(
                        point.frame_idx
                    )

                    for point
                    in raw_points
                ],

            "trajectory_frames":
                trajectory_frames,

            "interpolated_frames":
                [
                    int(
                        point.frame_idx
                    )

                    for point
                    in trajectory

                    if point.is_interpolated
                ],

            "trajectory_point_count":
                int(
                    len(
                        trajectory
                    )
                ),

            "raw_trajectory_point_count":
                int(
                    sum(
                        point.is_raw_v3_detection

                        for point
                        in trajectory
                    )
                ),

            "interpolated_point_count":
                int(
                    sum(
                        point.is_interpolated

                        for point
                        in trajectory
                    )
                ),
        }


    return ball_tracking_handler


def register_ball_tracking_handler(
    pipeline,
    max_frame_gap=3,
    gate_multiplier=1.25,
):
    """Register canonical Notebook-42 ball tracking."""

    pipeline.register_stage_handler(

        "ball_tracking",

        build_ball_tracking_handler(

            max_frame_gap=max_frame_gap,

            gate_multiplier=gate_multiplier,
        ),
    )


    return pipeline


# NOTEBOOK43_CANONICAL_PLAYER_DETECTION_TRACKING_V1
# ====================================================================================================
# CANONICAL PLAYER DETECTION + FIELD SUPPORT + TEMPORAL TRACKING
# ====================================================================================================


@dataclass
class PlayerDetectionRecord:
    """One raw frozen-YOLO person detection."""

    frame_idx: int
    detection_idx: int

    x1: float
    y1: float
    x2: float
    y2: float

    center_x: float
    center_y: float

    foot_x: float
    foot_y: float

    width: float
    height: float

    confidence: float
    class_id: int


@dataclass
class PlayerFieldSupportDecision:
    """One field-support decision for one raw player detection."""

    frame_idx: int
    detection_idx: int

    foot_x: float
    foot_y: float

    rounded_foot_x: int
    rounded_foot_y: int

    box_width: float
    box_height: float
    box_height_fraction: float

    roi_x0: int
    roi_y0: int
    roi_x1: int
    roi_y1: int

    roi_width: int
    roi_height: int
    roi_pixel_count: int

    foot_green_fraction: float

    is_closeup: bool
    field_supported: bool

    decision_reason: str


@dataclass
class PlayerTrackObservation:
    """One tracked player observation."""

    track_id: int

    frame_idx: int
    detection_idx: int

    x1: float
    y1: float
    x2: float
    y2: float

    center_x: float
    center_y: float

    foot_x: float
    foot_y: float

    width: float
    height: float

    confidence: float


@dataclass
class PlayerTrack:
    """Canonical temporal player track."""

    track_id: int

    observations: list

    start_frame: int
    end_frame: int

    observed_frames: int
    span_frames: int
    internal_missing_frames: int

    mean_confidence: float

    start_foot_x: float
    start_foot_y: float

    end_foot_x: float
    end_foot_y: float


def _player_bbox_iou(
    left,
    right,
):
    """Axis-aligned bounding-box intersection-over-union."""

    intersection_x1 = max(
        float(
            left.x1
        ),
        float(
            right.x1
        ),
    )


    intersection_y1 = max(
        float(
            left.y1
        ),
        float(
            right.y1
        ),
    )


    intersection_x2 = min(
        float(
            left.x2
        ),
        float(
            right.x2
        ),
    )


    intersection_y2 = min(
        float(
            left.y2
        ),
        float(
            right.y2
        ),
    )


    intersection_width = max(
        0.0,
        intersection_x2
        -
        intersection_x1,
    )


    intersection_height = max(
        0.0,
        intersection_y2
        -
        intersection_y1,
    )


    intersection_area = (
        intersection_width
        *
        intersection_height
    )


    left_area = max(
        0.0,
        float(
            left.x2
        )
        -
        float(
            left.x1
        ),
    ) * max(
        0.0,
        float(
            left.y2
        )
        -
        float(
            left.y1
        ),
    )


    right_area = max(
        0.0,
        float(
            right.x2
        )
        -
        float(
            right.x1
        ),
    ) * max(
        0.0,
        float(
            right.y2
        )
        -
        float(
            right.y1
        ),
    )


    union_area = (
        left_area
        +
        right_area
        -
        intersection_area
    )


    if union_area <= 0.0:

        return 0.0


    return float(
        intersection_area
        /
        union_area
    )


def _player_normalized_movement_per_frame(
    previous,
    current,
    frame_gap,
):
    """
    Bottom-center movement normalized by max pair box height and frame gap.
    """

    frame_gap = int(
        frame_gap
    )


    if frame_gap <= 0:

        raise ValueError(
            "frame_gap must be positive."
        )


    displacement = (
        (
            float(
                current.foot_x
            )
            -
            float(
                previous.foot_x
            )
        )
        ** 2
        +
        (
            float(
                current.foot_y
            )
            -
            float(
                previous.foot_y
            )
        )
        ** 2
    ) ** 0.5


    normalizer = max(
        float(
            previous.height
        ),
        float(
            current.height
        ),
        1.0,
    )


    return float(
        displacement
        /
        normalizer
        /
        float(
            frame_gap
        )
    )


def _player_observation_from_detection(
    track_id,
    detection,
):
    """Convert one raw detection into one tracked observation."""

    return PlayerTrackObservation(

        track_id=int(
            track_id
        ),

        frame_idx=int(
            detection.frame_idx
        ),

        detection_idx=int(
            detection.detection_idx
        ),

        x1=float(
            detection.x1
        ),

        y1=float(
            detection.y1
        ),

        x2=float(
            detection.x2
        ),

        y2=float(
            detection.y2
        ),

        center_x=float(
            detection.center_x
        ),

        center_y=float(
            detection.center_y
        ),

        foot_x=float(
            detection.foot_x
        ),

        foot_y=float(
            detection.foot_y
        ),

        width=float(
            detection.width
        ),

        height=float(
            detection.height
        ),

        confidence=float(
            detection.confidence
        ),
    )


def _finalize_player_track(
    track_id,
    observations,
):
    """Build one public PlayerTrack from its ordered observations."""

    observations = sorted(

        list(
            observations
        ),

        key=lambda observation:
            int(
                observation.frame_idx
            ),
    )


    if not observations:

        raise ValueError(
            "Cannot finalize an empty player track."
        )


    start = observations[
        0
    ]


    end = observations[
        -1
    ]


    start_frame = int(
        start.frame_idx
    )


    end_frame = int(
        end.frame_idx
    )


    observed_frames = len(
        observations
    )


    span_frames = (
        end_frame
        -
        start_frame
        +
        1
    )


    internal_missing_frames = (
        span_frames
        -
        observed_frames
    )


    mean_confidence = float(
        sum(
            float(
                observation.confidence
            )

            for observation
            in observations
        )
        /
        float(
            observed_frames
        )
    )


    return PlayerTrack(

        track_id=int(
            track_id
        ),

        observations=observations,

        start_frame=start_frame,

        end_frame=end_frame,

        observed_frames=int(
            observed_frames
        ),

        span_frames=int(
            span_frames
        ),

        internal_missing_frames=int(
            internal_missing_frames
        ),

        mean_confidence=float(
            mean_confidence
        ),

        start_foot_x=float(
            start.foot_x
        ),

        start_foot_y=float(
            start.foot_y
        ),

        end_foot_x=float(
            end.foot_x
        ),

        end_foot_y=float(
            end.foot_y
        ),
    )


def build_player_detection_tracking_handler(
    model_path,
    expected_sha256,
    artifact_name="player_detector",
    class_id=0,
    confidence_threshold=0.20,
    iou_threshold=0.50,
    image_size=1280,
    device=0,
    batch=1,
    half=False,
    max_det=300,
    field_hsv_lower=(30, 25, 30),
    field_hsv_upper=(95, 255, 255),
    field_kernel_size=5,
    foot_roi_horizontal_factor=0.30,
    foot_roi_min_half_width=5,
    foot_roi_y_above=5,
    foot_roi_y_below=15,
    field_green_fraction_threshold=0.20,
    closeup_height_fraction=0.30,
    max_frame_gap=2,
    minimum_association_iou=0.10,
    max_normalized_movement_per_frame=0.35,
):
    """
    Build canonical Notebook-42 player_detection_tracking stage.

    Internally performs:
        1. frozen six-frame person detection
        2. field-support decisions
        3. deterministic temporal association
    """

    model_path = Path(
        model_path
    ).expanduser().resolve()


    expected_sha256 = str(
        expected_sha256
    ).lower()


    class_id = int(
        class_id
    )


    confidence_threshold = float(
        confidence_threshold
    )


    iou_threshold = float(
        iou_threshold
    )


    image_size = int(
        image_size
    )


    batch = int(
        batch
    )


    max_det = int(
        max_det
    )


    field_hsv_lower = tuple(
        int(
            value
        )
        for value
        in field_hsv_lower
    )


    field_hsv_upper = tuple(
        int(
            value
        )
        for value
        in field_hsv_upper
    )


    field_kernel_size = int(
        field_kernel_size
    )


    foot_roi_horizontal_factor = float(
        foot_roi_horizontal_factor
    )


    foot_roi_min_half_width = int(
        foot_roi_min_half_width
    )


    foot_roi_y_above = int(
        foot_roi_y_above
    )


    foot_roi_y_below = int(
        foot_roi_y_below
    )


    field_green_fraction_threshold = float(
        field_green_fraction_threshold
    )


    closeup_height_fraction = float(
        closeup_height_fraction
    )


    max_frame_gap = int(
        max_frame_gap
    )


    minimum_association_iou = float(
        minimum_association_iou
    )


    max_normalized_movement_per_frame = float(
        max_normalized_movement_per_frame
    )


    def player_detection_tracking_handler(result):

        import cv2
        import math
        import numpy as np
        import torch

        from ultralytics import YOLO


        # --------------------------------------------------------------------------------------------
        # STAGE ORDER
        # --------------------------------------------------------------------------------------------

        expected_prefix = list(
            CANONICAL_PIPELINE_STAGES[
                : _STAGE_INDEX[
                    "player_detection_tracking"
                ]
            ]
        )


        if (
            result.pipeline_status.completed_stages
            !=
            expected_prefix
        ):

            raise PipelineContractError(
                "Player detection/tracking cannot run before "
                "successful canonical ball tracking."
            )


        # --------------------------------------------------------------------------------------------
        # BALL-DERIVED ANALYSIS FRAMES
        # --------------------------------------------------------------------------------------------

        if not result.ball_track.success:

            raise PipelineContractError(
                "Player analysis requires successful ball tracking."
            )


        ball_trajectory = (
            result.ball_track.track
        )


        if not isinstance(
            ball_trajectory,
            list,
        ):

            raise PipelineContractError(
                "Player analysis requires a list-valued ball trajectory."
            )


        if len(
            ball_trajectory
        ) == 0:

            raise RuntimeError(
                "Ball trajectory contains zero points."
            )


        analysis_frames = sorted(
            {
                int(
                    point.frame_idx
                )

                for point
                in ball_trajectory
            }
        )


        if not analysis_frames:

            raise RuntimeError(
                "Could not derive player-analysis frames."
            )


        # --------------------------------------------------------------------------------------------
        # VALIDATED PLAYER ARTIFACT
        # --------------------------------------------------------------------------------------------

        artifact_validation = result.metadata.get(
            "artifact_validation"
        )


        if not isinstance(
            artifact_validation,
            dict,
        ):

            raise PipelineContractError(
                "Player analysis requires completed artifact validation."
            )


        artifact_records = artifact_validation.get(
            "artifacts",
            {},
        )


        model_record = artifact_records.get(
            artifact_name
        )


        if not isinstance(
            model_record,
            dict,
        ):

            raise PipelineContractError(
                f"Validated artifact record missing: {artifact_name}"
            )


        validated_model_path = Path(
            model_record[
                "path"
            ]
        ).resolve()


        if validated_model_path != model_path:

            raise PipelineContractError(
                "Validated player model path does not match configured model."
            )


        if (
            str(
                model_record.get(
                    "sha256",
                    "",
                )
            ).lower()
            !=
            expected_sha256
        ):

            raise PipelineContractError(
                "Validated player model SHA256 does not match frozen identity."
            )


        # --------------------------------------------------------------------------------------------
        # CUDA
        # --------------------------------------------------------------------------------------------

        if isinstance(
            device,
            int,
        ):

            if device >= 0:

                if not torch.cuda.is_available():

                    raise RuntimeError(
                        "Frozen player policy requires CUDA but CUDA is unavailable."
                    )


                if device >= torch.cuda.device_count():

                    raise RuntimeError(
                        f"Requested CUDA device does not exist: {device}"
                    )


        # --------------------------------------------------------------------------------------------
        # DECODE BALL-DERIVED FRAMES
        # --------------------------------------------------------------------------------------------

        if result.video_path is None:

            raise PipelineContractError(
                "Player analysis requires validated video_path."
            )


        video_path = Path(
            result.video_path
        )


        capture = cv2.VideoCapture(
            str(
                video_path
            )
        )


        if not capture.isOpened():

            raise RuntimeError(
                f"Could not open input video: {video_path}"
            )


        decoded_frames = {}


        maximum_required_frame = max(
            analysis_frames
        )


        required_frame_set = set(
            analysis_frames
        )


        frame_idx = 0


        try:

            while frame_idx <= maximum_required_frame:

                ok, frame = capture.read()


                if not ok:

                    break


                if frame_idx in required_frame_set:

                    decoded_frames[
                        int(
                            frame_idx
                        )
                    ] = frame.copy()


                frame_idx += 1


        finally:

            capture.release()


        if sorted(
            decoded_frames.keys()
        ) != analysis_frames:

            raise RuntimeError(
                "Player-analysis frame decode coverage is incomplete. "
                f"required={analysis_frames}, "
                f"decoded={sorted(decoded_frames.keys())}"
            )


        # --------------------------------------------------------------------------------------------
        # FROZEN PLAYER MODEL
        # --------------------------------------------------------------------------------------------

        model = YOLO(
            str(
                model_path
            )
        )


        model_names = getattr(
            model,
            "names",
            {},
        )


        if isinstance(
            model_names,
            dict,
        ):

            person_class_name = model_names.get(
                class_id
            )


            if person_class_name is None:

                person_class_name = model_names.get(
                    str(
                        class_id
                    )
                )

        elif isinstance(
            model_names,
            (
                list,
                tuple,
            ),
        ):

            person_class_name = (
                model_names[
                    class_id
                ]
                if (
                    0
                    <=
                    class_id
                    <
                    len(
                        model_names
                    )
                )
                else None
            )

        else:

            person_class_name = None


        if (
            person_class_name is not None
            and
            str(
                person_class_name
            ).strip().lower()
            !=
            "person"
        ):

            raise RuntimeError(
                "Player detector class contract is invalid: "
                f"class {class_id} -> {person_class_name!r}"
            )


        # --------------------------------------------------------------------------------------------
        # RAW PERSON INFERENCE
        # --------------------------------------------------------------------------------------------

        raw_detections = []


        inferred_frames = []


        for analysis_frame in analysis_frames:

            frame = decoded_frames[
                analysis_frame
            ]


            predictions = model.predict(

                source=frame,

                conf=confidence_threshold,

                iou=iou_threshold,

                imgsz=image_size,

                device=device,

                batch=batch,

                half=half,

                max_det=max_det,

                classes=[
                    class_id
                ],

                verbose=False,
            )


            if len(
                predictions
            ) != 1:

                raise RuntimeError(
                    "Expected one YOLO result for one player-analysis frame."
                )


            prediction = predictions[
                0
            ]


            boxes = prediction.boxes


            inferred_frames.append(
                int(
                    analysis_frame
                )
            )


            if boxes is None:

                continue


            xyxy_values = (
                boxes.xyxy
                .detach()
                .cpu()
                .numpy()
            )


            confidence_values = (
                boxes.conf
                .detach()
                .cpu()
                .numpy()
            )


            class_values = (
                boxes.cls
                .detach()
                .cpu()
                .numpy()
            )


            for detection_idx in range(
                len(
                    confidence_values
                )
            ):

                detected_class_id = int(
                    class_values[
                        detection_idx
                    ]
                )


                if detected_class_id != class_id:

                    continue


                confidence = float(
                    confidence_values[
                        detection_idx
                    ]
                )


                x1, y1, x2, y2 = [

                    float(
                        value
                    )

                    for value
                    in xyxy_values[
                        detection_idx
                    ].tolist()
                ]


                if not all(
                    math.isfinite(
                        value
                    )

                    for value
                    in [
                        confidence,
                        x1,
                        y1,
                        x2,
                        y2,
                    ]
                ):

                    raise RuntimeError(
                        "Player inference produced non-finite values."
                    )


                if confidence < confidence_threshold:

                    raise RuntimeError(
                        "Player inference returned a detection below "
                        "the frozen threshold."
                    )


                if not (
                    x2 > x1
                    and
                    y2 > y1
                ):

                    raise RuntimeError(
                        "Player inference returned an invalid bounding box."
                    )


                center_x = (
                    x1
                    +
                    x2
                ) / 2.0


                center_y = (
                    y1
                    +
                    y2
                ) / 2.0


                width = (
                    x2
                    -
                    x1
                )


                height = (
                    y2
                    -
                    y1
                )


                raw_detections.append(
                    PlayerDetectionRecord(

                        frame_idx=int(
                            analysis_frame
                        ),

                        detection_idx=int(
                            detection_idx
                        ),

                        x1=float(
                            x1
                        ),

                        y1=float(
                            y1
                        ),

                        x2=float(
                            x2
                        ),

                        y2=float(
                            y2
                        ),

                        center_x=float(
                            center_x
                        ),

                        center_y=float(
                            center_y
                        ),

                        foot_x=float(
                            center_x
                        ),

                        foot_y=float(
                            y2
                        ),

                        width=float(
                            width
                        ),

                        height=float(
                            height
                        ),

                        confidence=float(
                            confidence
                        ),

                        class_id=int(
                            detected_class_id
                        ),
                    )
                )


        if inferred_frames != analysis_frames:

            raise RuntimeError(
                "Player inference did not cover all analysis frames."
            )


        # --------------------------------------------------------------------------------------------
        # FIELD-SUPPORT MASKS
        # --------------------------------------------------------------------------------------------

        field_kernel = cv2.getStructuringElement(

            cv2.MORPH_ELLIPSE,

            (
                field_kernel_size,
                field_kernel_size,
            ),
        )


        frame_field_masks = {}


        for analysis_frame in analysis_frames:

            frame = decoded_frames[
                analysis_frame
            ]


            hsv = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2HSV,
            )


            raw_mask = cv2.inRange(

                hsv,

                np.array(
                    field_hsv_lower,
                    dtype=np.uint8,
                ),

                np.array(
                    field_hsv_upper,
                    dtype=np.uint8,
                ),
            )


            opened_mask = cv2.morphologyEx(

                raw_mask,

                cv2.MORPH_OPEN,

                field_kernel,
            )


            final_mask = cv2.morphologyEx(

                opened_mask,

                cv2.MORPH_CLOSE,

                field_kernel,
            )


            frame_field_masks[
                analysis_frame
            ] = final_mask


        # --------------------------------------------------------------------------------------------
        # FIELD-SUPPORT DECISIONS
        # --------------------------------------------------------------------------------------------

        field_support_decisions = []


        supported_detection_keys = set()


        for detection in raw_detections:

            frame = decoded_frames[
                int(
                    detection.frame_idx
                )
            ]


            field_mask = frame_field_masks[
                int(
                    detection.frame_idx
                )
            ]


            frame_height, frame_width = frame.shape[
                :2
            ]


            rounded_foot_x = int(
                round(
                    float(
                        detection.foot_x
                    )
                )
            )


            rounded_foot_y = int(
                round(
                    float(
                        detection.foot_y
                    )
                )
            )


            roi_half_width = max(

                int(
                    foot_roi_min_half_width
                ),

                int(
                    round(
                        foot_roi_horizontal_factor
                        *
                        float(
                            detection.width
                        )
                    )
                ),
            )


            roi_x0 = max(
                0,
                rounded_foot_x
                -
                roi_half_width,
            )


            roi_x1 = min(
                int(
                    frame_width
                ),
                rounded_foot_x
                +
                roi_half_width
                +
                1,
            )


            roi_y0 = max(
                0,
                rounded_foot_y
                -
                foot_roi_y_above,
            )


            roi_y1 = min(
                int(
                    frame_height
                ),
                rounded_foot_y
                +
                foot_roi_y_below
                +
                1,
            )


            roi_width = max(
                0,
                roi_x1
                -
                roi_x0,
            )


            roi_height = max(
                0,
                roi_y1
                -
                roi_y0,
            )


            roi_pixel_count = (
                roi_width
                *
                roi_height
            )


            if roi_pixel_count > 0:

                roi_mask = field_mask[
                    roi_y0:
                    roi_y1,

                    roi_x0:
                    roi_x1,
                ]


                foot_green_fraction = float(
                    np.count_nonzero(
                        roi_mask
                    )
                    /
                    float(
                        roi_mask.size
                    )
                )

            else:

                foot_green_fraction = 0.0


            box_height_fraction = float(
                float(
                    detection.height
                )
                /
                float(
                    frame_height
                )
            )


            is_closeup = bool(
                box_height_fraction
                >
                closeup_height_fraction
            )


            field_supported = bool(
                (
                    foot_green_fraction
                    >=
                    field_green_fraction_threshold
                )
                and
                not is_closeup
            )


            if is_closeup:

                decision_reason = (
                    "CLOSEUP_EXCLUSION"
                )

            elif (
                foot_green_fraction
                <
                field_green_fraction_threshold
            ):

                decision_reason = (
                    "INSUFFICIENT_FIELD_SUPPORT"
                )

            else:

                decision_reason = (
                    "SUPPORTED"
                )


            decision = PlayerFieldSupportDecision(

                frame_idx=int(
                    detection.frame_idx
                ),

                detection_idx=int(
                    detection.detection_idx
                ),

                foot_x=float(
                    detection.foot_x
                ),

                foot_y=float(
                    detection.foot_y
                ),

                rounded_foot_x=int(
                    rounded_foot_x
                ),

                rounded_foot_y=int(
                    rounded_foot_y
                ),

                box_width=float(
                    detection.width
                ),

                box_height=float(
                    detection.height
                ),

                box_height_fraction=float(
                    box_height_fraction
                ),

                roi_x0=int(
                    roi_x0
                ),

                roi_y0=int(
                    roi_y0
                ),

                roi_x1=int(
                    roi_x1
                ),

                roi_y1=int(
                    roi_y1
                ),

                roi_width=int(
                    roi_width
                ),

                roi_height=int(
                    roi_height
                ),

                roi_pixel_count=int(
                    roi_pixel_count
                ),

                foot_green_fraction=float(
                    foot_green_fraction
                ),

                is_closeup=bool(
                    is_closeup
                ),

                field_supported=bool(
                    field_supported
                ),

                decision_reason=str(
                    decision_reason
                ),
            )


            field_support_decisions.append(
                decision
            )


            if field_supported:

                supported_detection_keys.add(
                    (
                        int(
                            detection.frame_idx
                        ),

                        int(
                            detection.detection_idx
                        ),
                    )
                )


        # --------------------------------------------------------------------------------------------
        # SUPPORTED DETECTION SUBSET
        # --------------------------------------------------------------------------------------------

        supported_detections = [

            detection

            for detection
            in raw_detections

            if (
                int(
                    detection.frame_idx
                ),

                int(
                    detection.detection_idx
                ),
            )
            in
            supported_detection_keys
        ]


        # --------------------------------------------------------------------------------------------
        # GROUP BY FRAME
        # --------------------------------------------------------------------------------------------

        supported_by_frame = {}


        for detection in supported_detections:

            supported_by_frame.setdefault(

                int(
                    detection.frame_idx
                ),

                [],
            ).append(
                detection
            )


        for frame_key in supported_by_frame:

            supported_by_frame[
                frame_key
            ] = sorted(

                supported_by_frame[
                    frame_key
                ],

                key=lambda detection:
                    int(
                        detection.detection_idx
                    ),
            )


        # --------------------------------------------------------------------------------------------
        # TEMPORAL ASSOCIATION
        # --------------------------------------------------------------------------------------------

        working_tracks = {}


        next_track_id = 1


        for frame_idx in analysis_frames:

            frame_detections = (
                supported_by_frame.get(
                    frame_idx,
                    [],
                )
            )


            candidates = []


            for track_id, observations in working_tracks.items():

                if not observations:

                    continue


                previous = observations[
                    -1
                ]


                frame_gap = (
                    int(
                        frame_idx
                    )
                    -
                    int(
                        previous.frame_idx
                    )
                )


                if not (
                    1
                    <=
                    frame_gap
                    <=
                    max_frame_gap
                ):

                    continue


                for detection in frame_detections:

                    iou_value = (
                        _player_bbox_iou(
                            previous,
                            detection,
                        )
                    )


                    normalized_movement = (
                        _player_normalized_movement_per_frame(

                            previous,

                            detection,

                            frame_gap,
                        )
                    )


                    if (
                        iou_value
                        >=
                        minimum_association_iou
                        and
                        normalized_movement
                        <=
                        max_normalized_movement_per_frame
                    ):

                        candidates.append(
                            {
                                "track_id":
                                    int(
                                        track_id
                                    ),

                                "detection_idx":
                                    int(
                                        detection.detection_idx
                                    ),

                                "iou":
                                    float(
                                        iou_value
                                    ),

                                "normalized_movement":
                                    float(
                                        normalized_movement
                                    ),

                                "detection":
                                    detection,
                            }
                        )


            candidates.sort(

                key=lambda candidate:
                    (
                        -float(
                            candidate[
                                "iou"
                            ]
                        ),

                        float(
                            candidate[
                                "normalized_movement"
                            ]
                        ),

                        int(
                            candidate[
                                "track_id"
                            ]
                        ),

                        int(
                            candidate[
                                "detection_idx"
                            ]
                        ),
                    )
            )


            assigned_tracks = set()

            assigned_detection_ids = set()


            for candidate in candidates:

                track_id = int(
                    candidate[
                        "track_id"
                    ]
                )


                detection_idx = int(
                    candidate[
                        "detection_idx"
                    ]
                )


                if track_id in assigned_tracks:

                    continue


                if detection_idx in assigned_detection_ids:

                    continue


                detection = candidate[
                    "detection"
                ]


                observation = (
                    _player_observation_from_detection(

                        track_id,

                        detection,
                    )
                )


                working_tracks[
                    track_id
                ].append(
                    observation
                )


                assigned_tracks.add(
                    track_id
                )


                assigned_detection_ids.add(
                    detection_idx
                )


            # ----------------------------------------------------------------------------------------
            # UNMATCHED DETECTIONS START NEW TRACKS
            # ----------------------------------------------------------------------------------------

            for detection in frame_detections:

                detection_idx = int(
                    detection.detection_idx
                )


                if detection_idx in assigned_detection_ids:

                    continue


                track_id = int(
                    next_track_id
                )


                next_track_id += 1


                working_tracks[
                    track_id
                ] = [

                    _player_observation_from_detection(

                        track_id,

                        detection,
                    )
                ]


        # --------------------------------------------------------------------------------------------
        # FINALIZE TRACKS
        # --------------------------------------------------------------------------------------------

        player_tracks = [

            _finalize_player_track(

                track_id,

                working_tracks[
                    track_id
                ],
            )

            for track_id
            in sorted(
                working_tracks.keys()
            )
        ]


        tracked_observation_count = sum(

            int(
                track.observed_frames
            )

            for track
            in player_tracks
        )


        # --------------------------------------------------------------------------------------------
        # ASSIGNMENT CONSERVATION
        # --------------------------------------------------------------------------------------------

        tracked_keys = [

            (
                int(
                    observation.frame_idx
                ),

                int(
                    observation.detection_idx
                ),
            )

            for track
            in player_tracks

            for observation
            in track.observations
        ]


        if (
            len(
                tracked_keys
            )
            !=
            len(
                set(
                    tracked_keys
                )
            )
        ):

            raise RuntimeError(
                "A player detection was assigned to multiple tracks."
            )


        if set(
            tracked_keys
        ) != set(
            supported_detection_keys
        ):

            raise RuntimeError(
                "Tracked observations do not conserve field-supported detections."
            )


        # --------------------------------------------------------------------------------------------
        # PUBLIC PlayerTrackResult
        # --------------------------------------------------------------------------------------------

        result.player_tracks.success = True


        result.player_tracks.detections = (
            raw_detections
        )


        result.player_tracks.tracks = (
            player_tracks
        )


        result.player_tracks.track_count = int(
            len(
                player_tracks
            )
        )


        result.player_tracks.failure_reason = None


        # --------------------------------------------------------------------------------------------
        # METADATA
        # --------------------------------------------------------------------------------------------

        raw_counts_by_frame = {

            int(
                frame_idx
            ):
                int(
                    sum(
                        detection.frame_idx
                        ==
                        frame_idx

                        for detection
                        in raw_detections
                    )
                )

            for frame_idx
            in analysis_frames
        }


        supported_counts_by_frame = {

            int(
                frame_idx
            ):
                int(
                    sum(
                        detection.frame_idx
                        ==
                        frame_idx

                        for detection
                        in supported_detections
                    )
                )

            for frame_idx
            in analysis_frames
        }


        result.metadata[
            "player_detection_tracking"
        ] = {

            "analysis_frames":
                analysis_frames,

            "decoded_frames":
                sorted(
                    decoded_frames.keys()
                ),

            "inferred_frames":
                inferred_frames,

            "inference_policy":
                {
                    "method":
                        "YOLO_PREDICT_NOT_TRACK",

                    "model_path":
                        str(
                            model_path
                        ),

                    "model_sha256":
                        expected_sha256,

                    "class_id":
                        int(
                            class_id
                        ),

                    "confidence_threshold":
                        float(
                            confidence_threshold
                        ),

                    "iou_threshold":
                        float(
                            iou_threshold
                        ),

                    "imgsz":
                        int(
                            image_size
                        ),

                    "device":
                        device,

                    "batch":
                        int(
                            batch
                        ),

                    "half":
                        bool(
                            half
                        ),

                    "max_det":
                        int(
                            max_det
                        ),

                    "yolo_tracking_used":
                        False,
                },

            "raw_detection_count":
                int(
                    len(
                        raw_detections
                    )
                ),

            "raw_counts_by_frame":
                raw_counts_by_frame,

            "field_support_policy":
                {
                    "hsv_lower":
                        list(
                            field_hsv_lower
                        ),

                    "hsv_upper":
                        list(
                            field_hsv_upper
                        ),

                    "kernel_size":
                        int(
                            field_kernel_size
                        ),

                    "morphology":
                        "ELLIPSE_OPEN_THEN_CLOSE",

                    "foot_roi_horizontal_factor":
                        float(
                            foot_roi_horizontal_factor
                        ),

                    "foot_roi_min_half_width":
                        int(
                            foot_roi_min_half_width
                        ),

                    "foot_roi_y_above":
                        int(
                            foot_roi_y_above
                        ),

                    "foot_roi_y_below":
                        int(
                            foot_roi_y_below
                        ),

                    "green_fraction_threshold":
                        float(
                            field_green_fraction_threshold
                        ),

                    "closeup_height_fraction":
                        float(
                            closeup_height_fraction
                        ),
                },

            "field_support_decisions":
                field_support_decisions,

            "supported_detection_count":
                int(
                    len(
                        supported_detections
                    )
                ),

            "supported_counts_by_frame":
                supported_counts_by_frame,

            "unsupported_detection_count":
                int(
                    len(
                        raw_detections
                    )
                    -
                    len(
                        supported_detections
                    )
                ),

            "closeup_exclusion_count":
                int(
                    sum(
                        decision.is_closeup

                        for decision
                        in field_support_decisions
                    )
                ),

            "low_field_support_exclusion_count":
                int(
                    sum(
                        (
                            not decision.is_closeup
                            and
                            not decision.field_supported
                        )

                        for decision
                        in field_support_decisions
                    )
                ),

            "tracking_policy":
                {
                    "maximum_missing_frames":
                        int(
                            max_frame_gap
                            -
                            1
                        ),

                    "maximum_frame_gap":
                        int(
                            max_frame_gap
                        ),

                    "minimum_association_iou":
                        float(
                            minimum_association_iou
                        ),

                    "max_normalized_movement_per_frame":
                        float(
                            max_normalized_movement_per_frame
                        ),

                    "movement_point":
                        "BBOX_BOTTOM_CENTER",

                    "movement_normalizer":
                        "MAX_PAIR_BOX_HEIGHT",

                    "association_gate":
                        "IOU_AND_NORMALIZED_MOVEMENT",

                    "association_priority":
                        "HIGHEST_IOU_THEN_LOWEST_NORMALIZED_MOVEMENT",
                },

            "tracked_observation_count":
                int(
                    tracked_observation_count
                ),

            "player_track_count":
                int(
                    len(
                        player_tracks
                    )
                ),

            "supported_detection_keys":
                sorted(
                    [
                        [
                            int(
                                frame_idx
                            ),

                            int(
                                detection_idx
                            ),
                        ]

                        for frame_idx, detection_idx
                        in supported_detection_keys
                    ]
                ),
        }


    return player_detection_tracking_handler


def register_player_detection_tracking_handler(
    pipeline,
    model_path,
    expected_sha256,
    artifact_name="player_detector",
    class_id=0,
    confidence_threshold=0.20,
    iou_threshold=0.50,
    image_size=1280,
    device=0,
    batch=1,
    half=False,
    max_det=300,
):
    """Register the canonical Notebook-42 player-analysis stage."""

    pipeline.register_stage_handler(

        "player_detection_tracking",

        build_player_detection_tracking_handler(

            model_path=model_path,

            expected_sha256=expected_sha256,

            artifact_name=artifact_name,

            class_id=class_id,

            confidence_threshold=confidence_threshold,

            iou_threshold=iou_threshold,

            image_size=image_size,

            device=device,

            batch=batch,

            half=half,

            max_det=max_det,
        ),
    )


    return pipeline

# NOTEBOOK43_EXACT_REFERENCE_FREE_PITCH_LINE_DETECTION_V1
# ====================================================================================================
# EXACT NOTEBOOK-37 / NOTEBOOK-42 REFINED PITCH-LINE DETECTOR
# ====================================================================================================

# Exact detector runtime dependencies.
import cv2
import math
import numpy as np

LINE_MERGE_MAX_ANGLE_DIFFERENCE = 3.0

LINE_MERGE_MAX_DISTANCE_PIXELS = 14.0

MIN_LOCAL_BRIGHTNESS_ADVANTAGE = 14.0

REFINED_GREEN_LOWER = np.array(
    [
        25,
        35,
        25,
    ],
    dtype=np.uint8,
)

REFINED_GREEN_UPPER = np.array(
    [
        100,
        255,
        255,
    ],
    dtype=np.uint8,
)

REFINED_HOUGH_MAX_LINE_GAP = 12

REFINED_HOUGH_MIN_LINE_LENGTH = 55

REFINED_HOUGH_THRESHOLD = 55

REFINED_INTERSECTION_MIN_ANGLE = 25.0

REFINED_WHITE_MAX_SATURATION = 105

REFINED_WHITE_MIN_VALUE = 115

def segment_angle_degrees(
    x1,
    y1,
    x2,
    y2,
):

    angle = np.degrees(
        np.arctan2(
            float(
                y2
            )
            -
            float(
                y1
            ),
            float(
                x2
            )
            -
            float(
                x1
            ),
        )
    )


    angle = float(
        angle
        %
        180.0
    )


    return angle

def orientation_difference_degrees(
    angle_a,
    angle_b,
):

    difference = abs(
        float(
            angle_a
        )
        -
        float(
            angle_b
        )
    )


    return float(
        min(
            difference,
            180.0
            -
            difference,
        )
    )

def infinite_line_intersection(
    line_a,
    line_b,
):

    x1, y1, x2, y2 = map(
        float,
        line_a
    )


    x3, y3, x4, y4 = map(
        float,
        line_b
    )


    p1 = np.array(
        [
            x1,
            y1,
            1.0,
        ],
        dtype=float,
    )


    p2 = np.array(
        [
            x2,
            y2,
            1.0,
        ],
        dtype=float,
    )


    p3 = np.array(
        [
            x3,
            y3,
            1.0,
        ],
        dtype=float,
    )


    p4 = np.array(
        [
            x4,
            y4,
            1.0,
        ],
        dtype=float,
    )


    line1 = np.cross(
        p1,
        p2,
    )


    line2 = np.cross(
        p3,
        p4,
    )


    intersection = np.cross(
        line1,
        line2,
    )


    if abs(
        float(
            intersection[
                2
            ]
        )
    ) < 1e-10:

        return None


    intersection = (
        intersection
        /
        intersection[
            2
        ]
    )


    x = float(
        intersection[
            0
        ]
    )


    y = float(
        intersection[
            1
        ]
    )


    if not (
        np.isfinite(
            x
        )
        and
        np.isfinite(
            y
        )
    ):

        return None


    return (
        x,
        y,
    )

def point_to_infinite_line_distance(
    point_x,
    point_y,
    line,
):

    x1 = float(
        line[
            "x1"
        ]
    )

    y1 = float(
        line[
            "y1"
        ]
    )

    x2 = float(
        line[
            "x2"
        ]
    )

    y2 = float(
        line[
            "y2"
        ]
    )


    dx = (
        x2
        -
        x1
    )

    dy = (
        y2
        -
        y1
    )


    denominator = float(
        np.sqrt(
            dx ** 2
            +
            dy ** 2
        )
    )


    if denominator < 1e-9:

        return np.inf


    numerator = abs(
        dy
        *
        float(
            point_x
        )
        -
        dx
        *
        float(
            point_y
        )
        +
        x2
        *
        y1
        -
        y2
        *
        x1
    )


    return float(
        numerator
        /
        denominator
    )

def build_pitch_roi(
    frame,
):

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV,
    )


    green_mask = cv2.inRange(
        hsv,
        REFINED_GREEN_LOWER,
        REFINED_GREEN_UPPER,
    )


    # Fill small holes from players / pitch markings.
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            31,
            31,
        ),
    )


    closed_green = cv2.morphologyEx(
        green_mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1,
    )


    # Remove tiny disconnected regions.
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            9,
            9,
        ),
    )


    cleaned_green = cv2.morphologyEx(
        closed_green,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1,
    )


    contours, _ = cv2.findContours(
        cleaned_green,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )


    pitch_roi = np.zeros_like(
        green_mask
    )


    if len(
        contours
    ) == 0:

        return (
            pitch_roi,
            green_mask,
        )


    largest_contour = max(
        contours,
        key=cv2.contourArea,
    )


    hull = cv2.convexHull(
        largest_contour
    )


    cv2.fillConvexPoly(
        pitch_roi,
        hull,
        255,
    )


    # Slight erosion avoids advertising boards / stands along
    # the pitch boundary entering the line detector.
    erosion_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            7,
            7,
        ),
    )


    pitch_roi = cv2.erode(
        pitch_roi,
        erosion_kernel,
        iterations=1,
    )


    return (
        pitch_roi,
        green_mask,
    )

def consolidate_segments(
    segments,
):

    if len(
        segments
    ) == 0:

        return []


    ordered = sorted(
        segments,
        key=
            lambda row:
                row[
                    "length"
                ],
        reverse=True,
    )


    accepted = []


    for candidate in ordered:

        candidate_mid_x = (
            candidate[
                "x1"
            ]
            +
            candidate[
                "x2"
            ]
        ) / 2.0


        candidate_mid_y = (
            candidate[
                "y1"
            ]
            +
            candidate[
                "y2"
            ]
        ) / 2.0


        duplicate = False


        for existing in accepted:

            angle_difference = (
                orientation_difference_degrees(
                    candidate[
                        "angle_degrees"
                    ],
                    existing[
                        "angle_degrees"
                    ],
                )
            )


            if (
                angle_difference
                >
                LINE_MERGE_MAX_ANGLE_DIFFERENCE
            ):

                continue


            center_to_existing_line = (
                point_to_infinite_line_distance(
                    candidate_mid_x,
                    candidate_mid_y,
                    existing,
                )
            )


            if (
                center_to_existing_line
                <=
                LINE_MERGE_MAX_DISTANCE_PIXELS
            ):

                duplicate = True

                break


        if not duplicate:

            accepted.append(
                candidate
            )


    return accepted

def extract_refined_pitch_markings(
    frame,
):

    height, width = (
        frame.shape[
            :2
        ]
    )


    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV,
    )


    saturation = hsv[
        :,
        :,
        1
    ].astype(
        np.float32
    )


    value = hsv[
        :,
        :,
        2
    ].astype(
        np.float32
    )


    pitch_roi, raw_green_mask = (
        build_pitch_roi(
            frame
        )
    )


    # --------------------------------------------------------
    # Local brightness model
    #
    # A pitch marking should normally be brighter than the
    # grass immediately around it.
    # --------------------------------------------------------

    local_value = cv2.GaussianBlur(
        value,
        (
            0,
            0,
        ),
        sigmaX=9.0,
        sigmaY=9.0,
    )


    brightness_advantage = (
        value
        -
        local_value
    )


    white_candidate = (
        (
            saturation
            <=
            REFINED_WHITE_MAX_SATURATION
        )
        &
        (
            value
            >=
            REFINED_WHITE_MIN_VALUE
        )
        &
        (
            brightness_advantage
            >=
            MIN_LOCAL_BRIGHTNESS_ADVANTAGE
        )
        &
        (
            pitch_roi
            >
            0
        )
    )


    white_mask = (
        white_candidate
        .astype(
            np.uint8
        )
        *
        255
    )


    # --------------------------------------------------------
    # Clean isolated speckles without aggressively joining
    # unrelated white objects.
    # --------------------------------------------------------

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            3,
            3,
        ),
    )


    cleaned_white = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1,
    )


    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            2,
            2,
        ),
    )


    cleaned_white = cv2.morphologyEx(
        cleaned_white,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1,
    )


    edges = cv2.Canny(
        cleaned_white,
        50,
        140,
    )


    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=
            np.pi
            /
            180.0,
        threshold=
            REFINED_HOUGH_THRESHOLD,
        minLineLength=
            REFINED_HOUGH_MIN_LINE_LENGTH,
        maxLineGap=
            REFINED_HOUGH_MAX_LINE_GAP,
    )


    raw_segments = []


    if raw_lines is not None:

        normalized_lines = (
            np.asarray(
                raw_lines
            )
            .reshape(
                -1,
                4
            )
        )


        for line in normalized_lines:

            x1, y1, x2, y2 = map(
                float,
                line
            )


            length = float(
                np.hypot(
                    x2 - x1,
                    y2 - y1,
                )
            )


            if (
                length
                <
                REFINED_HOUGH_MIN_LINE_LENGTH
            ):

                continue


            angle = (
                segment_angle_degrees(
                    x1,
                    y1,
                    x2,
                    y2,
                )
            )


            raw_segments.append(
                {
                    "x1":
                        x1,

                    "y1":
                        y1,

                    "x2":
                        x2,

                    "y2":
                        y2,

                    "length":
                        length,

                    "angle_degrees":
                        angle,
                }
            )


    merged_segments = (
        consolidate_segments(
            raw_segments
        )
    )


    # Keep the strongest geometrically distinct lines.
    geometry_segments = (
        sorted(
            merged_segments,
            key=
                lambda row:
                    row[
                        "length"
                    ],
            reverse=True,
        )
        [
            :30
        ]
    )


    # --------------------------------------------------------
    # Orientation bins
    # --------------------------------------------------------

    orientation_bins = set()


    for segment in geometry_segments:

        orientation_bin = int(
            np.floor(
                segment[
                    "angle_degrees"
                ]
                /
                10.0
            )
        )


        orientation_bins.add(
            orientation_bin
        )


    # --------------------------------------------------------
    # Intersections
    # --------------------------------------------------------

    intersection_bins = set()

    intersection_points = []


    margin_x = (
        0.10
        *
        width
    )


    margin_y = (
        0.10
        *
        height
    )


    for first_index in range(
        len(
            geometry_segments
        )
    ):

        first = (
            geometry_segments[
                first_index
            ]
        )


        for second_index in range(
            first_index + 1,
            len(
                geometry_segments
            )
        ):

            second = (
                geometry_segments[
                    second_index
                ]
            )


            angle_difference = (
                orientation_difference_degrees(
                    first[
                        "angle_degrees"
                    ],
                    second[
                        "angle_degrees"
                    ],
                )
            )


            if (
                angle_difference
                <
                REFINED_INTERSECTION_MIN_ANGLE
            ):

                continue


            intersection = (
                infinite_line_intersection(
                    (
                        first[
                            "x1"
                        ],
                        first[
                            "y1"
                        ],
                        first[
                            "x2"
                        ],
                        first[
                            "y2"
                        ],
                    ),
                    (
                        second[
                            "x1"
                        ],
                        second[
                            "y1"
                        ],
                        second[
                            "x2"
                        ],
                        second[
                            "y2"
                        ],
                    ),
                )
            )


            if intersection is None:

                continue


            intersection_x, intersection_y = (
                intersection
            )


            if not (
                -margin_x
                <=
                intersection_x
                <=
                width
                +
                margin_x
                and
                -margin_y
                <=
                intersection_y
                <=
                height
                +
                margin_y
            ):

                continue


            intersection_bin = (
                int(
                    round(
                        intersection_x
                        /
                        24.0
                    )
                ),

                int(
                    round(
                        intersection_y
                        /
                        24.0
                    )
                ),
            )


            if (
                intersection_bin
                not in
                intersection_bins
            ):

                intersection_bins.add(
                    intersection_bin
                )


                intersection_points.append(
                    (
                        float(
                            intersection_x
                        ),

                        float(
                            intersection_y
                        ),
                    )
                )


    return {
        "pitch_roi":
            pitch_roi,

        "raw_green_mask":
            raw_green_mask,

        "white_mask":
            white_mask,

        "cleaned_white":
            cleaned_white,

        "edges":
            edges,

        "raw_segments":
            raw_segments,

        "merged_segments":
            merged_segments,

        "geometry_segments":
            geometry_segments,

        "orientation_bin_count":
            len(
                orientation_bins
            ),

        "intersection_points":
            intersection_points,

        "pitch_roi_fraction":
            float(
                (
                    pitch_roi
                    >
                    0
                )
                .mean()
            ),

        "raw_green_fraction":
            float(
                (
                    raw_green_mask
                    >
                    0
                )
                .mean()
            ),

        "white_mask_fraction":
            float(
                (
                    cleaned_white
                    >
                    0
                )
                .mean()
            ),
    }



def build_pitch_line_detection_handler():
    """
    Build canonical reference-free pitch-line detection.

    Operational geometry frame:
        first current ball-trajectory frame.

    No shooter ID, reference frame, historical homography, or prior
    projected coordinates are used.
    """

    def pitch_line_detection_handler(result):

        # --------------------------------------------------------------------------------------------
        # CANONICAL STAGE ORDER
        # --------------------------------------------------------------------------------------------

        expected_prefix = list(
            CANONICAL_PIPELINE_STAGES[
                : _STAGE_INDEX[
                    "pitch_line_detection"
                ]
            ]
        )


        if (
            result.pipeline_status.completed_stages
            !=
            expected_prefix
        ):

            raise PipelineContractError(
                "Pitch-line detection cannot run before completed "
                "player detection/tracking."
            )


        # --------------------------------------------------------------------------------------------
        # CURRENT BALL TRAJECTORY
        # --------------------------------------------------------------------------------------------

        if not result.ball_track.success:

            raise PipelineContractError(
                "Pitch-line detection requires successful ball tracking."
            )


        trajectory = result.ball_track.track


        if not isinstance(
            trajectory,
            list,
        ):

            raise PipelineContractError(
                "Pitch-line detection requires list-valued ball trajectory."
            )


        if len(
            trajectory
        ) == 0:

            raise RuntimeError(
                "Pitch-line detection received an empty ball trajectory."
            )


        geometry_frame_idx = int(
            min(
                int(
                    point.frame_idx
                )

                for point
                in trajectory
            )
        )


        # --------------------------------------------------------------------------------------------
        # FRESH VIDEO FRAME
        # --------------------------------------------------------------------------------------------

        video_path = Path(
            result.video_path
        )


        capture = cv2.VideoCapture(
            str(
                video_path
            )
        )


        if not capture.isOpened():

            raise RuntimeError(
                f"Could not open input video: {video_path}"
            )


        decoded_frame = None


        try:

            current_frame_idx = 0


            while current_frame_idx <= geometry_frame_idx:

                ok, frame = capture.read()


                if not ok:

                    break


                if current_frame_idx == geometry_frame_idx:

                    decoded_frame = frame.copy()

                    break


                current_frame_idx += 1


        finally:

            capture.release()


        if decoded_frame is None:

            raise RuntimeError(
                "Could not decode operational pitch-line frame: "
                f"{geometry_frame_idx}"
            )


        frame_height, frame_width = decoded_frame.shape[
            :2
        ]


        # --------------------------------------------------------------------------------------------
        # EXACT FRESH DETECTION
        # --------------------------------------------------------------------------------------------

        detector_output = (
            extract_refined_pitch_markings(
                decoded_frame
            )
        )


        if not isinstance(
            detector_output,
            dict,
        ):

            raise RuntimeError(
                "extract_refined_pitch_markings must return a dict."
            )


        if "geometry_segments" not in detector_output:

            raise RuntimeError(
                "Pitch detector output is missing geometry_segments."
            )


        geometry_segments = (
            detector_output[
                "geometry_segments"
            ]
        )


        if not isinstance(
            geometry_segments,
            list,
        ):

            raise RuntimeError(
                "geometry_segments must be a list."
            )


        if len(
            geometry_segments
        ) == 0:

            raise RuntimeError(
                "Pitch-line detection returned zero geometry segments."
            )


        # --------------------------------------------------------------------------------------------
        # GEOMETRY VALIDATION
        # --------------------------------------------------------------------------------------------

        required_segment_keys = {

            "x1",
            "y1",
            "x2",
            "y2",
            "length",
            "angle_degrees",
        }


        normalized_segments = []


        for segment_index, segment in enumerate(
            geometry_segments
        ):

            if not isinstance(
                segment,
                dict,
            ):

                raise RuntimeError(
                    "Geometry segment is not dictionary-valued."
                )


            if not required_segment_keys.issubset(
                segment.keys()
            ):

                raise RuntimeError(
                    "Geometry segment is missing required fields."
                )


            x1 = float(
                segment[
                    "x1"
                ]
            )


            y1 = float(
                segment[
                    "y1"
                ]
            )


            x2 = float(
                segment[
                    "x2"
                ]
            )


            y2 = float(
                segment[
                    "y2"
                ]
            )


            source_length = float(
                segment[
                    "length"
                ]
            )


            source_angle = float(
                segment[
                    "angle_degrees"
                ]
            )


            values = [

                x1,
                y1,
                x2,
                y2,
                source_length,
                source_angle,
            ]


            if not all(
                math.isfinite(
                    value
                )

                for value
                in values
            ):

                raise RuntimeError(
                    "Pitch-line detector produced non-finite geometry."
                )


            computed_length = float(
                math.hypot(
                    x2
                    -
                    x1,

                    y2
                    -
                    y1,
                )
            )


            computed_angle = float(
                math.degrees(
                    math.atan2(
                        y2
                        -
                        y1,

                        x2
                        -
                        x1,
                    )
                )
                %
                180.0
            )


            if computed_length <= 0.0:

                raise RuntimeError(
                    "Pitch-line detector produced a zero-length segment."
                )


            if not (
                0.0
                <=
                source_angle
                <
                180.0
            ):

                raise RuntimeError(
                    "Pitch-line detector produced invalid orientation."
                )


            if not (
                0.0
                <=
                x1
                <=
                float(
                    frame_width
                    -
                    1
                )
                and
                0.0
                <=
                x2
                <=
                float(
                    frame_width
                    -
                    1
                )
                and
                0.0
                <=
                y1
                <=
                float(
                    frame_height
                    -
                    1
                )
                and
                0.0
                <=
                y2
                <=
                float(
                    frame_height
                    -
                    1
                )
            ):

                raise RuntimeError(
                    "Pitch-line segment endpoint lies outside the image."
                )


            normalized_segments.append(
                {
                    "segment_index":
                        int(
                            segment_index
                        ),

                    "x1":
                        x1,

                    "y1":
                        y1,

                    "x2":
                        x2,

                    "y2":
                        y2,

                    "length":
                        source_length,

                    "angle_degrees":
                        source_angle,

                    "computed_length":
                        computed_length,

                    "computed_angle_degrees":
                        computed_angle,

                    "length_consistent":
                        bool(
                            math.isclose(
                                source_length,
                                computed_length,
                                rel_tol=0.0,
                                abs_tol=1e-6,
                            )
                        ),

                    "angle_consistent":
                        bool(
                            math.isclose(
                                source_angle,
                                computed_angle,
                                rel_tol=0.0,
                                abs_tol=1e-6,
                            )
                        ),
                }
            )


        if not all(
            segment[
                "length_consistent"
            ]

            for segment
            in normalized_segments
        ):

            raise RuntimeError(
                "Pitch-line source lengths disagree with endpoint geometry."
            )


        if not all(
            segment[
                "angle_consistent"
            ]

            for segment
            in normalized_segments
        ):

            raise RuntimeError(
                "Pitch-line source angles disagree with endpoint geometry."
            )


        # --------------------------------------------------------------------------------------------
        # STAGE METADATA
        # --------------------------------------------------------------------------------------------

        result.metadata[
            "pitch_line_detection"
        ] = {

            "method":
                "EXACT_REFERENCE_FREE_REFINED_PITCH_MARKING_DETECTOR",

            "frame_selection_source":
                "FIRST_OPERATIONAL_BALL_TRAJECTORY_FRAME",

            "frame_idx":
                int(
                    geometry_frame_idx
                ),

            "frame_width":
                int(
                    frame_width
                ),

            "frame_height":
                int(
                    frame_height
                ),

            "detector_return_type":
                type(
                    detector_output
                ).__name__,

            "segment_return_path":
                "RETURN.geometry_segments",

            "geometry_segment_count":
                int(
                    len(
                        geometry_segments
                    )
                ),

            # Preserve exact detector rows for downstream homography search.
            "geometry_segments":
                geometry_segments,

            # Canonical normalized audit view.
            "normalized_segments":
                normalized_segments,
        }


    return pitch_line_detection_handler


def register_pitch_line_detection_handler(
    pipeline,
):
    """Register canonical reference-free pitch-line detection."""

    pipeline.register_stage_handler(

        "pitch_line_detection",

        build_pitch_line_detection_handler(),
    )


    return pipeline

# NOTEBOOK43_CANONICAL_HOMOGRAPHY_ENGINE_V3_TRUE_GLOBALS
# ====================================================================================================
# EXACT HOMOGRAPHY ENGINE — TRUE GLOBAL DEPENDENCY CLOSURE
# ====================================================================================================


import itertools
import time
import warnings

from itertools import combinations, product, permutations

import pandas as pd



HOMOGRAPHY_SEARCH_VARIANTS = [
    ("CORE6_PLUS2", 6, 2),
    ("CORE6_PLUS4", 6, 4),
    ("CORE6_PLUS6", 6, 6),
    ("CORE6_PLUS8", 6, 8),
    ("CORE8_PLUS2", 8, 2),
    ("CORE8_PLUS4", 8, 4),
    ("CORE8_PLUS6", 8, 6),
    ("CORE8_PLUS8", 8, 8),
    ("CORE10_PLUS2", 10, 2),
    ("CORE10_PLUS4", 10, 4),
    ("CORE10_PLUS6", 10, 6),
]

robust_family_cache = {}

selected_line_score_cache = {}

exact_frame_player_cache = {}

_NOTEBOOK43_HOMOGRAPHY_STATIC_READY = False


def project_point_with_homography(
    homography,
    image_x,
    image_y,
):

    point = np.array(
        [
            float(
                image_x
            ),
            float(
                image_y
            ),
            1.0,
        ],
        dtype=float,
    )


    projected = (
        homography
        @
        point
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
        <
        1e-12
    ):

        return (
            np.nan,
            np.nan,
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


    return (
        pitch_x,
        pitch_y,
    )

def circular_orientation_difference(
    angle_a,
    angle_b,
):

    difference = abs(
        float(
            angle_a
        )
        -
        float(
            angle_b
        )
    )


    return float(
        min(
            difference,
            180.0
            -
            difference,
        )
    )

def weighted_orientation_mean(
    angles,
    weights,
):

    angles = np.asarray(
        angles,
        dtype=float,
    )


    weights = np.asarray(
        weights,
        dtype=float,
    )


    doubled = np.deg2rad(
        2.0
        *
        angles
    )


    mean_cos = float(
        np.average(
            np.cos(
                doubled
            ),
            weights=
                weights,
        )
    )


    mean_sin = float(
        np.average(
            np.sin(
                doubled
            ),
            weights=
                weights,
        )
    )


    angle = (
        0.5
        *
        np.arctan2(
            mean_sin,
            mean_cos,
        )
    )


    return float(
        np.rad2deg(
            angle
        )
        %
        180.0
    )

def robust_two_orientation_modes(
    automatic_lines,
):

    angles = np.array(
        [
            float(
                line[
                    "angle_degrees"
                ]
            )

            for line in (
                automatic_lines
            )
        ],
        dtype=float,
    )


    lengths = np.array(
        [
            float(
                line[
                    "length"
                ]
            )

            for line in (
                automatic_lines
            )
        ],
        dtype=float,
    )


    weights = np.sqrt(
        np.maximum(
            lengths,
            1.0,
        )
    )


    line_count = len(
        angles
    )


    if line_count < 2:

        raise RuntimeError(
            "Need at least two lines for family extraction."
        )


    # --------------------------------------------------------
    # Angular density around every observed orientation.
    # --------------------------------------------------------

    density_scores = []


    for candidate_angle in (
        angles
    ):

        differences = np.array(
            [
                circular_orientation_difference(
                    candidate_angle,
                    other_angle,
                )

                for other_angle in (
                    angles
                )
            ],
            dtype=float,
        )


        kernel = np.exp(
            -0.5
            *
            (
                differences
                /
                ORIENTATION_MODE_SIGMA_DEG
            ) ** 2
        )


        density_scores.append(
            float(
                np.sum(
                    weights
                    *
                    kernel
                )
            )
        )


    density_scores = np.asarray(
        density_scores,
        dtype=float,
    )


    # --------------------------------------------------------
    # First dominant mode.
    # --------------------------------------------------------

    first_seed_index = int(
        np.argmax(
            density_scores
        )
    )


    first_seed_angle = float(
        angles[
            first_seed_index
        ]
    )


    first_distances = np.array(
        [
            circular_orientation_difference(
                angle,
                first_seed_angle,
            )

            for angle in (
                angles
            )
        ],
        dtype=float,
    )


    first_refine_mask = (
        first_distances
        <=
        ORIENTATION_MODE_REFINE_RADIUS_DEG
    )


    first_mode = (
        weighted_orientation_mean(
            angles[
                first_refine_mask
            ],

            weights[
                first_refine_mask
            ],
        )
    )


    # --------------------------------------------------------
    # Second mode must be meaningfully separated.
    # --------------------------------------------------------

    distances_from_first = np.array(
        [
            circular_orientation_difference(
                angle,
                first_mode,
            )

            for angle in (
                angles
            )
        ],
        dtype=float,
    )


    eligible_second = np.where(
        distances_from_first
        >=
        MIN_MODE_SEPARATION_DEG
    )[
        0
    ]


    if len(
        eligible_second
    ) == 0:

        raise RuntimeError(
            (
                "Could not identify second orientation mode. "
                f"First mode={first_mode:.3f}"
            )
        )


    second_seed_index = int(
        eligible_second[
            np.argmax(
                density_scores[
                    eligible_second
                ]
            )
        ]
    )


    second_seed_angle = float(
        angles[
            second_seed_index
        ]
    )


    second_distances = np.array(
        [
            circular_orientation_difference(
                angle,
                second_seed_angle,
            )

            for angle in (
                angles
            )
        ],
        dtype=float,
    )


    second_refine_mask = (
        second_distances
        <=
        ORIENTATION_MODE_REFINE_RADIUS_DEG
    )


    second_mode = (
        weighted_orientation_mean(
            angles[
                second_refine_mask
            ],

            weights[
                second_refine_mask
            ],
        )
    )


    # --------------------------------------------------------
    # Sort names for reproducibility only.
    #
    # FAMILY_A / FAMILY_B carry NO semantic pitch identity.
    # --------------------------------------------------------

    modes = sorted(
        [
            first_mode,
            second_mode,
        ]
    )


    family_a_mode = float(
        modes[
            0
        ]
    )


    family_b_mode = float(
        modes[
            1
        ]
    )


    assignments = []


    for angle in angles:

        distance_a = (
            circular_orientation_difference(
                angle,
                family_a_mode,
            )
        )


        distance_b = (
            circular_orientation_difference(
                angle,
                family_b_mode,
            )
        )


        minimum_distance = min(
            distance_a,
            distance_b,
        )


        if (
            minimum_distance
            >
            MAX_FAMILY_ASSIGNMENT_DISTANCE_DEG
        ):

            family = (
                "OUTLIER"
            )


        elif (
            distance_a
            <=
            distance_b
        ):

            family = (
                "FAMILY_A"
            )


        else:

            family = (
                "FAMILY_B"
            )


        assignments.append(
            {
                "family":
                    family,

                "distance_to_family_a_deg":
                    float(
                        distance_a
                    ),

                "distance_to_family_b_deg":
                    float(
                        distance_b
                    ),

                "minimum_family_distance_deg":
                    float(
                        minimum_distance
                    ),
            }
        )


    return {
        "family_a_mode_deg":
            family_a_mode,

        "family_b_mode_deg":
            family_b_mode,

        "family_separation_deg":
            circular_orientation_difference(
                family_a_mode,
                family_b_mode,
            ),

        "assignments":
            assignments,
    }

def segment_to_line_coefficients(
    segment,
):

    point_1 = np.array(
        [
            float(
                segment[
                    "x1"
                ]
            ),

            float(
                segment[
                    "y1"
                ]
            ),

            1.0,
        ],
        dtype=float,
    )


    point_2 = np.array(
        [
            float(
                segment[
                    "x2"
                ]
            ),

            float(
                segment[
                    "y2"
                ]
            ),

            1.0,
        ],
        dtype=float,
    )


    line = np.cross(
        point_1,
        point_2,
    )


    normalization = float(
        np.hypot(
            line[
                0
            ],
            line[
                1
            ],
        )
    )


    if normalization < 1e-12:

        return None


    line = (
        line
        /
        normalization
    )


    return line

def line_position_along_family_normal(
    segment,
    family_mode_angle_deg,
    image_center_x,
    image_center_y,
):

    theta = np.deg2rad(
        float(
            family_mode_angle_deg
        )
    )


    normal_x = float(
        -np.sin(
            theta
        )
    )


    normal_y = float(
        np.cos(
            theta
        )
    )


    midpoint_x = (
        float(
            segment[
                "x1"
            ]
        )
        +
        float(
            segment[
                "x2"
            ]
        )
    ) / 2.0


    midpoint_y = (
        float(
            segment[
                "y1"
            ]
        )
        +
        float(
            segment[
                "y2"
            ]
        )
    ) / 2.0


    return float(
        normal_x
        *
        (
            midpoint_x
            -
            image_center_x
        )
        +
        normal_y
        *
        (
            midpoint_y
            -
            image_center_y
        )
    )

def intersection_from_coefficients(
    line_a,
    line_b,
):

    intersection = np.cross(
        line_a,
        line_b,
    )


    denominator = float(
        intersection[
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
        <
        1e-12
    ):

        return None


    intersection = (
        intersection
        /
        denominator
    )


    x = float(
        intersection[
            0
        ]
    )


    y = float(
        intersection[
            1
        ]
    )


    if not (
        np.isfinite(
            x
        )
        and
        np.isfinite(
            y
        )
    ):

        return None


    return (
        x,
        y,
    )

def homography_anchor_signature(
    homography,
    frame_width,
    frame_height,
):

    anchors = [
        (
            0.0,
            0.0,
        ),

        (
            float(
                frame_width
            ),
            0.0,
        ),

        (
            0.0,
            float(
                frame_height
            ),
        ),

        (
            float(
                frame_width
            ),
            float(
                frame_height
            ),
        ),

        (
            float(
                frame_width
            )
            /
            2.0,

            float(
                frame_height
            )
            /
            2.0,
        ),
    ]


    projected = []


    for image_x, image_y in anchors:

        pitch_x, pitch_y = (
            project_point_with_homography(
                homography=
                    homography,

                image_x=
                    image_x,

                image_y=
                    image_y,
            )
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

            return None


        if (
            abs(
                pitch_x
            )
            >
            500.0
            or
            abs(
                pitch_y
            )
            >
            500.0
        ):

            return None


        projected.append(
            [
                float(
                    pitch_x
                ),

                float(
                    pitch_y
                ),
            ]
        )


    return np.asarray(
        projected,
        dtype=float,
    )

def nearest_soft_family(
    angle_degrees,
    family_a_mode_deg,
    family_b_mode_deg,
    maximum_distance_deg,
):

    distance_a = (
        circular_orientation_difference(
            angle_degrees,
            family_a_mode_deg,
        )
    )


    distance_b = (
        circular_orientation_difference(
            angle_degrees,
            family_b_mode_deg,
        )
    )


    minimum_distance = min(
        distance_a,
        distance_b,
    )


    if (
        minimum_distance
        >
        maximum_distance_deg
    ):

        return {
            "soft_family":
                "OUTLIER",

            "distance_to_family_a_deg":
                float(
                    distance_a
                ),

            "distance_to_family_b_deg":
                float(
                    distance_b
                ),

            "minimum_family_distance_deg":
                float(
                    minimum_distance
                ),
        }


    if (
        distance_a
        <=
        distance_b
    ):

        family = (
            "FAMILY_A"
        )


    else:

        family = (
            "FAMILY_B"
        )


    return {
        "soft_family":
            family,

        "distance_to_family_a_deg":
            float(
                distance_a
            ),

        "distance_to_family_b_deg":
            float(
                distance_b
            ),

        "minimum_family_distance_deg":
            float(
                minimum_distance
            ),
    }

def build_selected_line_pairs(
    selected_lines,
):

    pair_rows = []


    for first_index, second_index in combinations(
        range(
            len(
                selected_lines
            )
        ),
        2,
    ):

        first_line = (
            selected_lines.iloc[
                first_index
            ]
        )


        second_line = (
            selected_lines.iloc[
                second_index
            ]
        )


        separation_px = float(
            abs(
                float(
                    first_line[
                        "family_position"
                    ]
                )
                -
                float(
                    second_line[
                        "family_position"
                    ]
                )
            )
        )


        if (
            separation_px
            <
            SPATIAL_PAIR_MIN_SEPARATION_PX
        ):

            continue


        pair_strength = float(
            (
                float(
                    first_line[
                        "proposal_priority"
                    ]
                )
                +
                float(
                    second_line[
                        "proposal_priority"
                    ]
                )
            )
            *
            np.sqrt(
                separation_px
            )
        )


        pair_rows.append(
            {
                "first_auto_line_index":
                    int(
                        first_line[
                            "auto_line_index"
                        ]
                    ),

                "second_auto_line_index":
                    int(
                        second_line[
                            "auto_line_index"
                        ]
                    ),

                "separation_px":
                    separation_px,

                "pair_strength":
                    pair_strength,
            }
        )


    pair_table = pd.DataFrame(
        pair_rows
    )


    if len(
        pair_table
    ) > 0:

        pair_table = (
            pair_table
            .sort_values(
                [
                    "pair_strength",
                    "separation_px",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .reset_index(
                drop=True
            )
        )


    pair_set = {
        frozenset(
            [
                int(
                    row[
                        "first_auto_line_index"
                    ]
                ),

                int(
                    row[
                        "second_auto_line_index"
                    ]
                ),
            ]
        )

        for _, row in (
            pair_table.iterrows()
        )
    }


    return (
        pair_table,
        pair_set,
    )

def select_core_plus_spatial_expansion(
    family_lines,
    core_size,
    expansion_size,
):

    family_lines = (
        family_lines
        .copy()
    )


    family_lines = (
        family_lines[
            family_lines[
                "proposal_priority"
            ]
            .notna()
            &
            family_lines[
                "family_position"
            ]
            .notna()
        ]
        .sort_values(
            [
                "proposal_priority",
                "line_length",
                "auto_line_index",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


    core_size = int(
        core_size
    )


    expansion_size = int(
        expansion_size
    )


    if len(
        family_lines
    ) == 0:

        return (
            family_lines.copy()
        )


    # --------------------------------------------------------
    # Priority core.
    # --------------------------------------------------------

    core_count = min(
        core_size,
        len(
            family_lines
        ),
    )


    core = (
        family_lines
        .head(
            core_count
        )
        .copy()
    )


    selected_original_indices = (
        core.index
        .tolist()
    )


    selected_positions = (
        core[
            "family_position"
        ]
        .astype(float)
        .tolist()
    )


    selection_sources = [
        "PRIORITY_CORE"
        for _ in range(
            core_count
        )
    ]


    novelty_at_selection = [
        np.nan
        for _ in range(
            core_count
        )
    ]


    # --------------------------------------------------------
    # Remaining candidates.
    # --------------------------------------------------------

    remaining_indices = [
        index

        for index in (
            family_lines.index
        )

        if index
        not in
        selected_original_indices
    ]


    additions = 0


    while (
        additions
        <
        expansion_size
        and
        len(
            remaining_indices
        )
        >
        0
    ):

        candidate_rows = []


        for candidate_index in (
            remaining_indices
        ):

            candidate = (
                family_lines.loc[
                    candidate_index
                ]
            )


            candidate_position = float(
                candidate[
                    "family_position"
                ]
            )


            if len(
                selected_positions
            ) == 0:

                novelty = np.inf


            else:

                novelty = float(
                    np.min(
                        np.abs(
                            np.asarray(
                                selected_positions,
                                dtype=float,
                            )
                            -
                            candidate_position
                        )
                    )
                )


            candidate_rows.append(
                {
                    "candidate_index":
                        int(
                            candidate_index
                        ),

                    "novelty_px":
                        novelty,

                    "proposal_priority":
                        float(
                            candidate[
                                "proposal_priority"
                            ]
                        ),

                    "line_length":
                        float(
                            candidate[
                                "line_length"
                            ]
                        ),

                    "auto_line_index":
                        int(
                            candidate[
                                "auto_line_index"
                            ]
                        ),
                }
            )


        candidate_table = pd.DataFrame(
            candidate_rows
        )


        chosen = (
            candidate_table
            .sort_values(
                [
                    "novelty_px",
                    "proposal_priority",
                    "line_length",
                    "auto_line_index",
                ],
                ascending=[
                    False,
                    False,
                    False,
                    True,
                ],
            )
            .iloc[
                0
            ]
        )


        chosen_index = int(
            chosen[
                "candidate_index"
            ]
        )


        chosen_row = (
            family_lines.loc[
                chosen_index
            ]
        )


        selected_original_indices.append(
            chosen_index
        )


        selected_positions.append(
            float(
                chosen_row[
                    "family_position"
                ]
            )
        )


        selection_sources.append(
            "SPATIAL_EXPANSION"
        )


        novelty_at_selection.append(
            float(
                chosen[
                    "novelty_px"
                ]
            )
        )


        remaining_indices.remove(
            chosen_index
        )


        additions += 1


    selected = (
        family_lines.loc[
            selected_original_indices
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    selected[
        "selection_order"
    ] = (
        np.arange(
            len(
                selected
            )
        )
        +
        1
    )


    selected[
        "selection_source"
    ] = (
        selection_sources
    )


    selected[
        "novelty_at_selection_px"
    ] = (
        novelty_at_selection
    )


    return selected

def get_auto_line_coefficients(
    event_id,
    auto_line_index,
):

    table = (
        robust_family_cache[
            str(
                event_id
            )
        ]
    )


    match = (
        table[
            table[
                "auto_line_index"
            ]
            .astype(int)
            .eq(
                int(
                    auto_line_index
                )
            )
        ]
    )


    if len(
        match
    ) != 1:

        raise RuntimeError(
            (
                f"{event_id}: expected one automatic line "
                f"{auto_line_index}; found {len(match)}"
            )
        )


    row = (
        match.iloc[
            0
        ]
    )


    return np.asarray(
        [
            float(
                row[
                    "line_a"
                ]
            ),

            float(
                row[
                    "line_b"
                ]
            ),

            float(
                row[
                    "line_c"
                ]
            ),
        ],
        dtype=float,
    )

def pair_product_source_points(
    event_id,
    pair_product_row,
):

    a1 = (
        get_auto_line_coefficients(
            event_id,
            int(
                pair_product_row[
                    "family_a_line_1"
                ]
            ),
        )
    )


    a2 = (
        get_auto_line_coefficients(
            event_id,
            int(
                pair_product_row[
                    "family_a_line_2"
                ]
            ),
        )
    )


    b1 = (
        get_auto_line_coefficients(
            event_id,
            int(
                pair_product_row[
                    "family_b_line_1"
                ]
            ),
        )
    )


    b2 = (
        get_auto_line_coefficients(
            event_id,
            int(
                pair_product_row[
                    "family_b_line_2"
                ]
            ),
        )
    )


    intersections = [

        intersection_from_coefficients(
            a1,
            b1,
        ),

        intersection_from_coefficients(
            a1,
            b2,
        ),

        intersection_from_coefficients(
            a2,
            b1,
        ),

        intersection_from_coefficients(
            a2,
            b2,
        ),
    ]


    if any(
        point is None
        for point in (
            intersections
        )
    ):

        return None


    source_points = np.asarray(
        intersections,
        dtype=np.float32,
    )


    if not (
        np.isfinite(
            source_points
        )
        .all()
    ):

        return None


    return source_points

def normalize_homography_matrix(
    H,
):

    H = np.asarray(
        H,
        dtype=float,
    )


    if (
        abs(
            float(
                H[
                    2,
                    2
                ]
            )
        )
        >
        1e-12
    ):

        H = (
            H
            /
            H[
                2,
                2
            ]
        )


    return H

def build_pair_uv_homography(
    event_id,
    pair_product_row,
):

    source_points = (
        pair_product_source_points(
            event_id=
                event_id,

            pair_product_row=
                pair_product_row,
        )
    )


    if source_points is None:

        return None


    H_uv = cv2.getPerspectiveTransform(
        source_points.astype(
            np.float32
        ),

        UNIT_RECTANGLE_POINTS,
    )


    H_uv = normalize_homography_matrix(
        H_uv
    )


    if not (
        np.isfinite(
            H_uv
        )
        .all()
    ):

        return None


    return H_uv

def factorized_coordinate_homographies(
    H_uv,
):

    # --------------------------------------------------------
    # Shape:
    #
    #   (2520, 3, 3)
    # --------------------------------------------------------

    H_batch = np.einsum(
        "tij,jk->tik",
        COORDINATE_TEMPLATE_AFFINES,
        H_uv,
    )


    denominators = (
        H_batch[
            :,
            2,
            2
        ]
    )


    valid = (
        np.abs(
            denominators
        )
        >
        1e-12
    )


    H_batch_normalized = (
        H_batch.copy()
    )


    H_batch_normalized[
        valid
    ] = (
        H_batch_normalized[
            valid
        ]
        /
        denominators[
            valid,
            None,
            None,
        ]
    )


    return H_batch_normalized

def image_points_to_uv(
    H_uv,
    image_points,
):

    image_points = np.asarray(
        image_points,
        dtype=float,
    )


    if (
        len(
            image_points
        )
        ==
        0
    ):

        return np.empty(
            (
                0,
                2,
            ),
            dtype=float,
        )


    homogeneous = np.column_stack(
        [
            image_points,
            np.ones(
                len(
                    image_points
                ),
                dtype=float,
            ),
        ]
    )


    projected = (
        H_uv
        @
        homogeneous.T
    ).T


    denominator = (
        projected[
            :,
            2
        ]
    )


    valid = (
        np.isfinite(
            denominator
        )
        &
        (
            np.abs(
                denominator
            )
            >
            1e-12
        )
    )


    output = np.full(
        (
            len(
                image_points
            ),
            2,
        ),
        np.nan,
        dtype=float,
    )


    output[
        valid
    ] = (
        projected[
            valid,
            :2
        ]
        /
        denominator[
            valid,
            None
        ]
    )


    return output

def uv_points_to_all_pitch_templates(
    uv_points,
):

    uv_points = np.asarray(
        uv_points,
        dtype=float,
    )


    if (
        len(
            uv_points
        )
        ==
        0
    ):

        return np.empty(
            (
                len(
                    COORDINATE_TEMPLATE_AFFINES
                ),
                0,
                2,
            ),
            dtype=float,
        )


    uv_h = np.column_stack(
        [
            uv_points,
            np.ones(
                len(
                    uv_points
                ),
                dtype=float,
            ),
        ]
    )


    # --------------------------------------------------------
    # Result:
    #
    #   template x point x coordinate
    # --------------------------------------------------------

    projected = np.einsum(
        "tij,pj->tpi",
        COORDINATE_TEMPLATE_AFFINES,
        uv_h,
    )


    return (
        projected[
            :,
            :,
            :2
        ]
    )

def image_lines_to_uv(
    H_uv,
    image_line_coefficients,
):

    image_line_coefficients = np.asarray(
        image_line_coefficients,
        dtype=float,
    )


    H_uv_inverse_transpose = (
        np.linalg.inv(
            H_uv
        )
        .T
    )


    uv_lines = (
        H_uv_inverse_transpose
        @
        image_line_coefficients.T
    ).T


    return uv_lines

def uv_lines_to_all_pitch_templates(
    uv_lines,
):

    uv_lines = np.asarray(
        uv_lines,
        dtype=float,
    )


    if (
        len(
            uv_lines
        )
        ==
        0
    ):

        return np.empty(
            (
                len(
                    COORDINATE_TEMPLATE_AFFINES
                ),
                0,
                3,
            ),
            dtype=float,
        )


    transformed = np.einsum(
        "tij,lj->tli",
        COORDINATE_TEMPLATE_AFFINE_INV_TRANSPOSE,
        uv_lines,
    )


    return transformed

def build_factorized_pair_geometry(
    event_id,
    pair_product_row,
):

    event_id = str(
        event_id
    )


    H_uv = (
        build_pair_uv_homography(
            event_id=
                event_id,

            pair_product_row=
                pair_product_row,
        )
    )


    if H_uv is None:

        return None


    # --------------------------------------------------------
    # Players -> UV.
    # --------------------------------------------------------

    detections = (
        exact_frame_player_cache[
            event_id
        ][
            "detections"
        ]
        .copy()
    )


    player_subset = (
        detections[
            detections[
                "field_supported"
            ]
        ]
        .copy()
    )


    if (
        len(
            player_subset
        )
        <
        4
    ):

        player_subset = (
            detections[
                ~detections[
                    "closeup_person"
                ]
            ]
            .copy()
        )


    player_image_points = (
        player_subset[
            [
                "bottom_center_x",
                "bottom_center_y",
            ]
        ]
        .to_numpy(
            dtype=float
        )
    )


    player_uv_points = (
        image_points_to_uv(
            H_uv=
                H_uv,

            image_points=
                player_image_points,
        )
    )


    # --------------------------------------------------------
    # Selected automatic lines -> UV.
    # --------------------------------------------------------

    image_line_coefficients = (
        selected_line_score_cache[
            event_id
        ][
            "coefficients"
        ]
    )


    line_families = (
        selected_line_score_cache[
            event_id
        ][
            "families"
        ]
    )


    line_lengths = (
        selected_line_score_cache[
            event_id
        ][
            "line_lengths"
        ]
    )


    line_uv_coefficients = (
        image_lines_to_uv(
            H_uv=
                H_uv,

            image_line_coefficients=
                image_line_coefficients,
        )
    )


    return {
        "H_uv":
            H_uv,

        "player_image_points":
            player_image_points,

        "player_uv_points":
            player_uv_points,

        "line_image_coefficients":
            image_line_coefficients,

        "line_uv_coefficients":
            line_uv_coefficients,

        "line_families":
            line_families,

        "line_lengths":
            line_lengths,
    }

def combine_two_finite_medians(
    first,
    second,
):

    first = np.asarray(
        first,
        dtype=float,
    )


    second = np.asarray(
        second,
        dtype=float,
    )


    first_finite = np.isfinite(
        first
    )


    second_finite = np.isfinite(
        second
    )


    result = np.full(
        len(
            first
        ),
        np.inf,
        dtype=float,
    )


    both = (
        first_finite
        &
        second_finite
    )


    first_only = (
        first_finite
        &
        ~second_finite
    )


    second_only = (
        ~first_finite
        &
        second_finite
    )


    result[
        both
    ] = (
        first[
            both
        ]
        +
        second[
            both
        ]
    ) / 2.0


    result[
        first_only
    ] = (
        first[
            first_only
        ]
    )


    result[
        second_only
    ] = (
        second[
            second_only
        ]
    )


    return result

def batch_family_unique_coverage_metrics(
    transformed_lines,
    line_lengths,
    expected_axis,
    canonical_coordinates,
):

    transformed_lines = np.asarray(
        transformed_lines,
        dtype=float,
    )


    line_lengths = np.asarray(
        line_lengths,
        dtype=float,
    )


    canonical_coordinates = np.asarray(
        canonical_coordinates,
        dtype=float,
    )


    template_count = int(
        transformed_lines.shape[
            0
        ]
    )


    line_count = int(
        transformed_lines.shape[
            1
        ]
    )


    if (
        line_count
        ==
        0
    ):

        return {
            "supported_lines":
                np.zeros(
                    template_count,
                    dtype=int,
                ),

            "distinct_canonical_lines":
                np.zeros(
                    template_count,
                    dtype=int,
                ),

            "duplicate_supported_lines":
                np.zeros(
                    template_count,
                    dtype=int,
                ),

            "duplicate_fraction":
                np.ones(
                    template_count,
                    dtype=float,
                ),

            "median_distinct_residual_m":
                np.full(
                    template_count,
                    np.inf,
                    dtype=float,
                ),

            "median_distinct_axis_leak":
                np.full(
                    template_count,
                    np.inf,
                    dtype=float,
                ),

            "canonical_span_m":
                np.zeros(
                    template_count,
                    dtype=float,
                ),
        }


    a = (
        transformed_lines[
            :,
            :,
            0
        ]
    )


    b = (
        transformed_lines[
            :,
            :,
            1
        ]
    )


    c = (
        transformed_lines[
            :,
            :,
            2
        ]
    )


    normal_norm = np.hypot(
        a,
        b,
    )


    valid_normal = (
        np.isfinite(
            normal_norm
        )
        &
        (
            normal_norm
            >
            1e-12
        )
    )


    normalized_a = np.full_like(
        a,
        np.nan,
        dtype=float,
    )


    normalized_b = np.full_like(
        b,
        np.nan,
        dtype=float,
    )


    normalized_c = np.full_like(
        c,
        np.nan,
        dtype=float,
    )


    np.divide(
        a,
        normal_norm,
        out=
            normalized_a,
        where=
            valid_normal,
    )


    np.divide(
        b,
        normal_norm,
        out=
            normalized_b,
        where=
            valid_normal,
    )


    np.divide(
        c,
        normal_norm,
        out=
            normalized_c,
        where=
            valid_normal,
    )


    if (
        expected_axis
        ==
        "X_CONSTANT"
    ):

        denominator = (
            normalized_a
        )


        axis_leak = np.abs(
            normalized_b
        )


    elif (
        expected_axis
        ==
        "Y_CONSTANT"
    ):

        denominator = (
            normalized_b
        )


        axis_leak = np.abs(
            normalized_a
        )


    else:

        raise ValueError(
            expected_axis
        )


    valid_denominator = (
        valid_normal
        &
        np.isfinite(
            denominator
        )
        &
        (
            np.abs(
                denominator
            )
            >
            1e-10
        )
    )


    coordinate_estimate = np.full_like(
        denominator,
        np.nan,
        dtype=float,
    )


    np.divide(
        -normalized_c,
        denominator,
        out=
            coordinate_estimate,
        where=
            valid_denominator,
    )


    coordinate_difference = np.abs(
        coordinate_estimate[
            :,
            :,
            None
        ]
        -
        canonical_coordinates[
            None,
            None,
            :
        ]
    )


    nearest_index = np.argmin(
        coordinate_difference,
        axis=2,
    )


    nearest_residual = np.take_along_axis(
        coordinate_difference,
        nearest_index[
            :,
            :,
            None
        ],
        axis=2,
    )[
        :,
        :,
        0
    ]


    supported = (
        valid_denominator
        &
        np.isfinite(
            coordinate_estimate
        )
        &
        np.isfinite(
            nearest_residual
        )
        &
        np.isfinite(
            axis_leak
        )
        &
        (
            axis_leak
            <=
            UNIQUE_MAX_AXIS_LEAK
        )
        &
        (
            nearest_residual
            <=
            UNIQUE_MAX_COORDINATE_RESIDUAL_M
        )
    )


    supported_count = (
        supported
        .sum(
            axis=1
        )
        .astype(int)
    )


    canonical_count = int(
        len(
            canonical_coordinates
        )
    )


    represented = np.zeros(
        (
            template_count,
            canonical_count,
        ),
        dtype=bool,
    )


    chosen_residual = np.full(
        (
            template_count,
            canonical_count,
        ),
        np.nan,
        dtype=float,
    )


    chosen_leak = np.full(
        (
            template_count,
            canonical_count,
        ),
        np.nan,
        dtype=float,
    )


    # --------------------------------------------------------
    # Same choice cost as Cell 33:
    #
    # residual / threshold
    # +
    # leak / threshold
    # -
    # tiny preference for longer line
    # --------------------------------------------------------

    base_cost = (
        nearest_residual
        /
        UNIQUE_MAX_COORDINATE_RESIDUAL_M
        +
        axis_leak
        /
        UNIQUE_MAX_AXIS_LEAK
        -
        1e-6
        *
        line_lengths[
            None,
            :
        ]
    )


    row_indices = np.arange(
        template_count
    )


    for canonical_index in range(
        canonical_count
    ):

        mask = (
            supported
            &
            (
                nearest_index
                ==
                canonical_index
            )
        )


        has_support = (
            mask.any(
                axis=1
            )
        )


        represented[
            :,
            canonical_index
        ] = (
            has_support
        )


        masked_cost = np.where(
            mask,
            base_cost,
            np.inf,
        )


        best_line_index = np.argmin(
            masked_cost,
            axis=1,
        )


        best_residual = (
            nearest_residual[
                row_indices,
                best_line_index,
            ]
        )


        best_leak = (
            axis_leak[
                row_indices,
                best_line_index,
            ]
        )


        chosen_residual[
            has_support,
            canonical_index
        ] = (
            best_residual[
                has_support
            ]
        )


        chosen_leak[
            has_support,
            canonical_index
        ] = (
            best_leak[
                has_support
            ]
        )


    distinct_count = (
        represented
        .sum(
            axis=1
        )
        .astype(int)
    )


    duplicate_count = (
        supported_count
        -
        distinct_count
    )


    duplicate_fraction = (
        duplicate_count
        /
        np.maximum(
            supported_count,
            1,
        )
    ).astype(float)


    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore",
            category=RuntimeWarning,
        )


        median_residual = np.nanmedian(
            chosen_residual,
            axis=1,
        )


        median_leak = np.nanmedian(
            chosen_leak,
            axis=1,
        )


    median_residual[
        distinct_count
        ==
        0
    ] = np.inf


    median_leak[
        distinct_count
        ==
        0
    ] = np.inf


    coordinate_grid = np.broadcast_to(
        canonical_coordinates[
            None,
            :
        ],
        represented.shape,
    )


    minimum_coordinate = np.min(
        np.where(
            represented,
            coordinate_grid,
            np.inf,
        ),
        axis=1,
    )


    maximum_coordinate = np.max(
        np.where(
            represented,
            coordinate_grid,
            -np.inf,
        ),
        axis=1,
    )


    canonical_span = np.zeros(
        template_count,
        dtype=float,
    )


    enough_distinct = (
        distinct_count
        >=
        2
    )


    canonical_span[
        enough_distinct
    ] = (
        maximum_coordinate[
            enough_distinct
        ]
        -
        minimum_coordinate[
            enough_distinct
        ]
    )


    return {
        "supported_lines":
            supported_count,

        "distinct_canonical_lines":
            distinct_count,

        "duplicate_supported_lines":
            duplicate_count,

        "duplicate_fraction":
            duplicate_fraction,

        "median_distinct_residual_m":
            median_residual,

        "median_distinct_axis_leak":
            median_leak,

        "canonical_span_m":
            canonical_span,
    }

def batch_unique_coverage_metrics(
    pair_geometry,
):

    uv_lines = (
        pair_geometry[
            "line_uv_coefficients"
        ]
    )


    families = (
        pair_geometry[
            "line_families"
        ]
    )


    line_lengths = (
        pair_geometry[
            "line_lengths"
        ]
    )


    all_pitch_lines = (
        uv_lines_to_all_pitch_templates(
            uv_lines
        )
    )


    template_count = int(
        len(
            BATCH_TEMPLATE_INDEX
        )
    )


    # --------------------------------------------------------
    # Output arrays.
    # --------------------------------------------------------

    distinct_a = np.zeros(
        template_count,
        dtype=int,
    )


    distinct_b = np.zeros(
        template_count,
        dtype=int,
    )


    supported_a = np.zeros(
        template_count,
        dtype=int,
    )


    supported_b = np.zeros(
        template_count,
        dtype=int,
    )


    duplicate_a = np.zeros(
        template_count,
        dtype=int,
    )


    duplicate_b = np.zeros(
        template_count,
        dtype=int,
    )


    residual_a = np.full(
        template_count,
        np.inf,
        dtype=float,
    )


    residual_b = np.full(
        template_count,
        np.inf,
        dtype=float,
    )


    leak_a = np.full(
        template_count,
        np.inf,
        dtype=float,
    )


    leak_b = np.full(
        template_count,
        np.inf,
        dtype=float,
    )


    span_a = np.zeros(
        template_count,
        dtype=float,
    )


    span_b = np.zeros(
        template_count,
        dtype=float,
    )


    family_a_mask = (
        families
        ==
        "FAMILY_A"
    )


    family_b_mask = (
        families
        ==
        "FAMILY_B"
    )


    # --------------------------------------------------------
    # Mapping 0:
    #
    # A -> X
    # B -> Y
    # --------------------------------------------------------

    mapping_zero_indices = np.where(
        BATCH_MAPPING_CODE
        ==
        0
    )[0]


    metrics_a_zero = (
        batch_family_unique_coverage_metrics(
            transformed_lines=
                all_pitch_lines[
                    mapping_zero_indices
                ][
                    :,
                    family_a_mask,
                    :
                ],

            line_lengths=
                line_lengths[
                    family_a_mask
                ],

            expected_axis=
                "X_CONSTANT",

            canonical_coordinates=
                BATCH_CANONICAL_X,
        )
    )


    metrics_b_zero = (
        batch_family_unique_coverage_metrics(
            transformed_lines=
                all_pitch_lines[
                    mapping_zero_indices
                ][
                    :,
                    family_b_mask,
                    :
                ],

            line_lengths=
                line_lengths[
                    family_b_mask
                ],

            expected_axis=
                "Y_CONSTANT",

            canonical_coordinates=
                BATCH_CANONICAL_Y,
        )
    )


    # --------------------------------------------------------
    # Mapping 1:
    #
    # A -> Y
    # B -> X
    # --------------------------------------------------------

    mapping_one_indices = np.where(
        BATCH_MAPPING_CODE
        ==
        1
    )[0]


    metrics_a_one = (
        batch_family_unique_coverage_metrics(
            transformed_lines=
                all_pitch_lines[
                    mapping_one_indices
                ][
                    :,
                    family_a_mask,
                    :
                ],

            line_lengths=
                line_lengths[
                    family_a_mask
                ],

            expected_axis=
                "Y_CONSTANT",

            canonical_coordinates=
                BATCH_CANONICAL_Y,
        )
    )


    metrics_b_one = (
        batch_family_unique_coverage_metrics(
            transformed_lines=
                all_pitch_lines[
                    mapping_one_indices
                ][
                    :,
                    family_b_mask,
                    :
                ],

            line_lengths=
                line_lengths[
                    family_b_mask
                ],

            expected_axis=
                "X_CONSTANT",

            canonical_coordinates=
                BATCH_CANONICAL_X,
        )
    )


    # --------------------------------------------------------
    # Assign both mapping groups.
    # --------------------------------------------------------

    for (
        indices,
        metrics_a,
        metrics_b,
    ) in [
        (
            mapping_zero_indices,
            metrics_a_zero,
            metrics_b_zero,
        ),
        (
            mapping_one_indices,
            metrics_a_one,
            metrics_b_one,
        ),
    ]:

        distinct_a[
            indices
        ] = (
            metrics_a[
                "distinct_canonical_lines"
            ]
        )


        distinct_b[
            indices
        ] = (
            metrics_b[
                "distinct_canonical_lines"
            ]
        )


        supported_a[
            indices
        ] = (
            metrics_a[
                "supported_lines"
            ]
        )


        supported_b[
            indices
        ] = (
            metrics_b[
                "supported_lines"
            ]
        )


        duplicate_a[
            indices
        ] = (
            metrics_a[
                "duplicate_supported_lines"
            ]
        )


        duplicate_b[
            indices
        ] = (
            metrics_b[
                "duplicate_supported_lines"
            ]
        )


        residual_a[
            indices
        ] = (
            metrics_a[
                "median_distinct_residual_m"
            ]
        )


        residual_b[
            indices
        ] = (
            metrics_b[
                "median_distinct_residual_m"
            ]
        )


        leak_a[
            indices
        ] = (
            metrics_a[
                "median_distinct_axis_leak"
            ]
        )


        leak_b[
            indices
        ] = (
            metrics_b[
                "median_distinct_axis_leak"
            ]
        )


        span_a[
            indices
        ] = (
            metrics_a[
                "canonical_span_m"
            ]
        )


        span_b[
            indices
        ] = (
            metrics_b[
                "canonical_span_m"
            ]
        )


    distinct_balance = np.minimum(
        distinct_a,
        distinct_b,
    )


    distinct_total = (
        distinct_a
        +
        distinct_b
    )


    supported_total = (
        supported_a
        +
        supported_b
    )


    duplicate_total = (
        duplicate_a
        +
        duplicate_b
    )


    duplicate_fraction = (
        duplicate_total
        /
        np.maximum(
            supported_total,
            1,
        )
    ).astype(float)


    median_residual = (
        combine_two_finite_medians(
            residual_a,
            residual_b,
        )
    )


    median_leak = (
        combine_two_finite_medians(
            leak_a,
            leak_b,
        )
    )


    return pd.DataFrame(
        {
            "template_index":
                BATCH_TEMPLATE_INDEX,

            "batch_distinct_family_a":
                distinct_a,

            "batch_distinct_family_b":
                distinct_b,

            "batch_distinct_balance":
                distinct_balance,

            "batch_distinct_total":
                distinct_total,

            "batch_supported_total":
                supported_total,

            "batch_duplicate_total":
                duplicate_total,

            "batch_duplicate_fraction":
                duplicate_fraction,

            "batch_median_distinct_residual_m":
                median_residual,

            "batch_median_distinct_axis_leak":
                median_leak,

            "batch_family_a_span_m":
                span_a,

            "batch_family_b_span_m":
                span_b,
        }
    )

def batch_player_pitch_metrics(
    pair_geometry,
):

    player_uv_points = (
        pair_geometry[
            "player_uv_points"
        ]
    )


    all_pitch_points = (
        uv_points_to_all_pitch_templates(
            player_uv_points
        )
    )


    template_count = int(
        all_pitch_points.shape[
            0
        ]
    )


    player_count = int(
        all_pitch_points.shape[
            1
        ]
    )


    if (
        player_count
        ==
        0
    ):

        return pd.DataFrame(
            {
                "template_index":
                    BATCH_TEMPLATE_INDEX,

                "batch_player_count":
                    np.zeros(
                        template_count,
                        dtype=int,
                    ),

                "batch_finite_player_count":
                    np.zeros(
                        template_count,
                        dtype=int,
                    ),

                "batch_inside_pitch_count":
                    np.zeros(
                        template_count,
                        dtype=int,
                    ),

                "batch_inside_pitch_fraction":
                    np.zeros(
                        template_count,
                        dtype=float,
                    ),

                "batch_soft_pitch_count":
                    np.zeros(
                        template_count,
                        dtype=int,
                    ),

                "batch_soft_pitch_fraction":
                    np.zeros(
                        template_count,
                        dtype=float,
                    ),

                "batch_median_outside_distance_m":
                    np.full(
                        template_count,
                        np.inf,
                    ),

                "batch_inside_x_span_m":
                    np.zeros(
                        template_count,
                    ),

                "batch_inside_y_span_m":
                    np.zeros(
                        template_count,
                    ),

                "batch_inside_span_balance":
                    np.zeros(
                        template_count,
                    ),
            }
        )


    x = (
        all_pitch_points[
            :,
            :,
            0
        ]
    )


    y = (
        all_pitch_points[
            :,
            :,
            1
        ]
    )


    finite = (
        np.isfinite(
            x
        )
        &
        np.isfinite(
            y
        )
    )


    finite_count = (
        finite
        .sum(
            axis=1
        )
        .astype(int)
    )


    inside = (
        finite
        &
        (
            x
            >=
            0.0
        )
        &
        (
            x
            <=
            PLAYER_PITCH_LENGTH_M
        )
        &
        (
            y
            >=
            0.0
        )
        &
        (
            y
            <=
            PLAYER_PITCH_WIDTH_M
        )
    )


    soft_inside = (
        finite
        &
        (
            x
            >=
            -PLAYER_PITCH_MARGIN_M
        )
        &
        (
            x
            <=
            PLAYER_PITCH_LENGTH_M
            +
            PLAYER_PITCH_MARGIN_M
        )
        &
        (
            y
            >=
            -PLAYER_PITCH_MARGIN_M
        )
        &
        (
            y
            <=
            PLAYER_PITCH_WIDTH_M
            +
            PLAYER_PITCH_MARGIN_M
        )
    )


    inside_count = (
        inside
        .sum(
            axis=1
        )
        .astype(int)
    )


    soft_count = (
        soft_inside
        .sum(
            axis=1
        )
        .astype(int)
    )


    inside_fraction = (
        inside_count
        /
        np.maximum(
            finite_count,
            1,
        )
    ).astype(float)


    soft_fraction = (
        soft_count
        /
        np.maximum(
            finite_count,
            1,
        )
    ).astype(float)


    dx = np.maximum(
        0.0,
        np.maximum(
            -x,
            x
            -
            PLAYER_PITCH_LENGTH_M,
        ),
    )


    dy = np.maximum(
        0.0,
        np.maximum(
            -y,
            y
            -
            PLAYER_PITCH_WIDTH_M,
        ),
    )


    outside_distance = np.hypot(
        dx,
        dy,
    )


    outside_distance[
        ~finite
    ] = np.nan


    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore",
            category=RuntimeWarning,
        )


        median_outside_distance = np.nanmedian(
            outside_distance,
            axis=1,
        )


    median_outside_distance[
        finite_count
        ==
        0
    ] = np.inf


    x_min = np.min(
        np.where(
            inside,
            x,
            np.inf,
        ),
        axis=1,
    )


    x_max = np.max(
        np.where(
            inside,
            x,
            -np.inf,
        ),
        axis=1,
    )


    y_min = np.min(
        np.where(
            inside,
            y,
            np.inf,
        ),
        axis=1,
    )


    y_max = np.max(
        np.where(
            inside,
            y,
            -np.inf,
        ),
        axis=1,
    )


    x_span = np.zeros(
        template_count,
        dtype=float,
    )


    y_span = np.zeros(
        template_count,
        dtype=float,
    )


    enough_inside = (
        inside_count
        >=
        2
    )


    x_span[
        enough_inside
    ] = (
        x_max[
            enough_inside
        ]
        -
        x_min[
            enough_inside
        ]
    )


    y_span[
        enough_inside
    ] = (
        y_max[
            enough_inside
        ]
        -
        y_min[
            enough_inside
        ]
    )


    x_span_fraction = (
        x_span
        /
        PLAYER_PITCH_LENGTH_M
    )


    y_span_fraction = (
        y_span
        /
        PLAYER_PITCH_WIDTH_M
    )


    span_balance = np.minimum(
        x_span_fraction,
        y_span_fraction,
    )


    return pd.DataFrame(
        {
            "template_index":
                BATCH_TEMPLATE_INDEX,

            "batch_player_count":
                np.full(
                    template_count,
                    player_count,
                    dtype=int,
                ),

            "batch_finite_player_count":
                finite_count,

            "batch_inside_pitch_count":
                inside_count,

            "batch_inside_pitch_fraction":
                inside_fraction,

            "batch_soft_pitch_count":
                soft_count,

            "batch_soft_pitch_fraction":
                soft_fraction,

            "batch_median_outside_distance_m":
                median_outside_distance,

            "batch_inside_x_span_m":
                x_span,

            "batch_inside_y_span_m":
                y_span,

            "batch_inside_span_balance":
                span_balance,
        }
    )

def score_pair_product_batched(
    event_id,
    pair_product_row,
):

    geometry = (
        build_factorized_pair_geometry(
            event_id=
                event_id,

            pair_product_row=
                pair_product_row,
        )
    )


    if geometry is None:

        return None


    unique_metrics = (
        batch_unique_coverage_metrics(
            geometry
        )
    )


    player_metrics = (
        batch_player_pitch_metrics(
            geometry
        )
    )


    output = (
        unique_metrics
        .merge(
            player_metrics,
            on=
                "template_index",
            how="inner",
            validate="one_to_one",
        )
    )


    return (
        output,
        geometry,
    )

def full_search_candidate_validity(
    H_batch,
    template_index,
    frame_width,
    frame_height,
):

    template_index = int(
        template_index
    )


    H = np.asarray(
        H_batch[
            template_index
        ],
        dtype=float,
    )


    if not (
        np.isfinite(
            H
        )
        .all()
    ):

        return {
            "valid":
                False,

            "condition_number":
                np.nan,

            "maximum_abs_anchor_m":
                np.nan,

            "reason":
                "NONFINITE_H",
        }


    H = normalize_homography_matrix(
        H
    )


    try:

        condition_number = float(
            np.linalg.cond(
                H
            )
        )


    except np.linalg.LinAlgError:

        return {
            "valid":
                False,

            "condition_number":
                np.nan,

            "maximum_abs_anchor_m":
                np.nan,

            "reason":
                "CONDITION_NUMBER_FAILED",
        }


    if (
        not np.isfinite(
            condition_number
        )
    ):

        return {
            "valid":
                False,

            "condition_number":
                condition_number,

            "maximum_abs_anchor_m":
                np.nan,

            "reason":
                "NONFINITE_CONDITION",
        }


    if (
        condition_number
        >
        FULL_SEARCH_MAX_CONDITION_NUMBER
    ):

        return {
            "valid":
                False,

            "condition_number":
                condition_number,

            "maximum_abs_anchor_m":
                np.nan,

            "reason":
                "CONDITION_TOO_LARGE",
        }


    anchor_signature = (
        homography_anchor_signature(
            homography=
                H,

            frame_width=
                frame_width,

            frame_height=
                frame_height,
        )
    )


    if anchor_signature is None:

        return {
            "valid":
                False,

            "condition_number":
                condition_number,

            "maximum_abs_anchor_m":
                np.nan,

            "reason":
                "ANCHOR_SIGNATURE_FAILED",
        }


    anchor_array = np.asarray(
        anchor_signature,
        dtype=float,
    )


    if not (
        np.isfinite(
            anchor_array
        )
        .all()
    ):

        return {
            "valid":
                False,

            "condition_number":
                condition_number,

            "maximum_abs_anchor_m":
                np.nan,

            "reason":
                "NONFINITE_ANCHOR",
        }


    maximum_abs_anchor = float(
        np.max(
            np.abs(
                anchor_array
            )
        )
    )


    if (
        maximum_abs_anchor
        >
        FULL_SEARCH_MAX_ABS_ANCHOR_M
    ):

        return {
            "valid":
                False,

            "condition_number":
                condition_number,

            "maximum_abs_anchor_m":
                maximum_abs_anchor,

            "reason":
                "ANCHOR_OUT_OF_RANGE",
        }


    return {
        "valid":
            True,

        "condition_number":
            condition_number,

        "maximum_abs_anchor_m":
            maximum_abs_anchor,

        "reason":
            "VALID",
    }

def initialize_homography_static_runtime():
    """Explicitly construct the frozen 2,520-template static homography runtime."""

    global AUDIT_FAMILY_THRESHOLD_DEG, BATCH_CANONICAL_X, BATCH_CANONICAL_Y, BATCH_MAPPING_CODE, BATCH_TEMPLATE_INDEX, CANONICAL_X_LINES, CANONICAL_Y_LINES, COORDINATE_TEMPLATE_AFFINES, COORDINATE_TEMPLATE_AFFINE_INV_TRANSPOSE, COORD_AUDIT_MAX_ABS_ANCHOR_M, COORD_AUDIT_MAX_CONDITION_NUMBER, FULL_SEARCH_MAX_ABS_ANCHOR_M, FULL_SEARCH_MAX_CONDITION_NUMBER, HYBRID_MIN_PAIR_SEPARATION_PX, MAX_FAMILY_ASSIGNMENT_DISTANCE_DEG, MIN_MODE_SEPARATION_DEG, ORIENTATION_MODE_REFINE_RADIUS_DEG, ORIENTATION_MODE_SIGMA_DEG, PLAYER_PITCH_LENGTH_M, PLAYER_PITCH_MARGIN_M, PLAYER_PITCH_WIDTH_M, REDUNDANCY_RADII_PX, SPATIAL_PAIR_MIN_SEPARATION_PX, UNIQUE_MAX_AXIS_LEAK, UNIQUE_MAX_COORDINATE_RESIDUAL_M, UNIT_RECTANGLE_POINTS, X_COORDINATE_PAIRS, Y_COORDINATE_PAIRS, _NOTEBOOK43_HOMOGRAPHY_STATIC_READY, coordinate_template_table

    if _NOTEBOOK43_HOMOGRAPHY_STATIC_READY:
        return {
            "coordinate_templates": int(len(coordinate_template_table)),
            "template_affines": int(COORDINATE_TEMPLATE_AFFINES.shape[0]),
            "inverse_transpose_affines": int(COORDINATE_TEMPLATE_AFFINE_INV_TRANSPOSE.shape[0]),
            "batch_templates": int(len(BATCH_TEMPLATE_INDEX)),
        }

    # --------------------------------------------------------------------------------
    # PURE_ASSIGNMENT | CANONICAL_X_LINES | Notebook-37 code cell 21 line 113
    # --------------------------------------------------------------------------------
    CANONICAL_X_LINES = np.array(
        [
            0.0,
            5.5,
            16.5,
            52.5,
            88.5,
            99.5,
            105.0,
        ],
        dtype=float,
    )

    # --------------------------------------------------------------------------------
    # PURE_ASSIGNMENT | CANONICAL_Y_LINES | Notebook-37 code cell 21 line 127
    # --------------------------------------------------------------------------------
    CANONICAL_Y_LINES = np.array(
        [
            0.0,
            13.84,
            24.84,
            43.16,
            54.16,
            68.0,
        ],
        dtype=float,
    )

    # --------------------------------------------------------------------------------
    # PURE_ASSIGNMENT | ORIENTATION_MODE_SIGMA_DEG | Notebook-37 code cell 21 line 150
    # --------------------------------------------------------------------------------
    ORIENTATION_MODE_SIGMA_DEG = 4.0

    # --------------------------------------------------------------------------------
    # PURE_ASSIGNMENT | ORIENTATION_MODE_REFINE_RADIUS_DEG | Notebook-37 code cell 21 line 152
    # --------------------------------------------------------------------------------
    ORIENTATION_MODE_REFINE_RADIUS_DEG = 6.0

    # --------------------------------------------------------------------------------
    # PURE_ASSIGNMENT | MIN_MODE_SEPARATION_DEG | Notebook-37 code cell 21 line 154
    # --------------------------------------------------------------------------------
    MIN_MODE_SEPARATION_DEG = 8.0

    # --------------------------------------------------------------------------------
    # PURE_ASSIGNMENT | MAX_FAMILY_ASSIGNMENT_DISTANCE_DEG | Notebook-37 code cell 21 line 156
    # --------------------------------------------------------------------------------
    MAX_FAMILY_ASSIGNMENT_DISTANCE_DEG = 12.0

    # --------------------------------------------------------------------------------
    # PURE_ASSIGNMENT | X_COORDINATE_PAIRS | Notebook-37 code cell 21 line 2342
    # --------------------------------------------------------------------------------
    X_COORDINATE_PAIRS = list(
        combinations(
            CANONICAL_X_LINES.tolist(),
            2,
        )
    )

    # --------------------------------------------------------------------------------
    # PURE_ASSIGNMENT | Y_COORDINATE_PAIRS | Notebook-37 code cell 21 line 2350
    # --------------------------------------------------------------------------------
    Y_COORDINATE_PAIRS = list(
        combinations(
            CANONICAL_Y_LINES.tolist(),
            2,
        )
    )

    # --------------------------------------------------------------------------------
    # PURE_ASSIGNMENT | AUDIT_FAMILY_THRESHOLD_DEG | Notebook-37 code cell 27 line 69
    # --------------------------------------------------------------------------------
    AUDIT_FAMILY_THRESHOLD_DEG = 18.0

    # --------------------------------------------------------------------------------
    # PURE_ASSIGNMENT | REDUNDANCY_RADII_PX | Notebook-37 code cell 27 line 71
    # --------------------------------------------------------------------------------
    REDUNDANCY_RADII_PX = [
        3.0,
        5.0,
        10.0,
        15.0,
    ]

    # --------------------------------------------------------------------------------
    # PURE_ASSIGNMENT | SPATIAL_PAIR_MIN_SEPARATION_PX | Notebook-37 code cell 28 line 86
    # --------------------------------------------------------------------------------
    SPATIAL_PAIR_MIN_SEPARATION_PX = 15.0

    # --------------------------------------------------------------------------------
    # PURE_ASSIGNMENT | HYBRID_MIN_PAIR_SEPARATION_PX | Notebook-37 code cell 29 line 80
    # --------------------------------------------------------------------------------
    HYBRID_MIN_PAIR_SEPARATION_PX = 15.0

    # --------------------------------------------------------------------------------
    # PURE_ASSIGNMENT | COORD_AUDIT_MAX_CONDITION_NUMBER | Notebook-37 code cell 32 line 136
    # --------------------------------------------------------------------------------
    COORD_AUDIT_MAX_CONDITION_NUMBER = 1e10

    # --------------------------------------------------------------------------------
    # PURE_ASSIGNMENT | COORD_AUDIT_MAX_ABS_ANCHOR_M | Notebook-37 code cell 32 line 138
    # --------------------------------------------------------------------------------
    COORD_AUDIT_MAX_ABS_ANCHOR_M = 500.0

    # --------------------------------------------------------------------------------
    # MUTATION_BUILDER | COORDINATE_TEMPLATE_BUILDER | Notebook-37 code cell 32 line 174
    # --------------------------------------------------------------------------------
    coordinate_template_rows = []

    coordinate_pitch_points = []


    template_index = 0


    for mapping_name in [
        "A_X_B_Y",
        "A_Y_B_X",
    ]:

        if (
            mapping_name
            ==
            "A_X_B_Y"
        ):

            coordinate_pairs_a = (
                X_COORDINATE_PAIRS
            )

            coordinate_pairs_b = (
                Y_COORDINATE_PAIRS
            )

            mapping_code = 0


        else:

            coordinate_pairs_a = (
                Y_COORDINATE_PAIRS
            )

            coordinate_pairs_b = (
                X_COORDINATE_PAIRS
            )

            mapping_code = 1


        for coordinate_pair_a in (
            coordinate_pairs_a
        ):

            for coordinate_pair_b in (
                coordinate_pairs_b
            ):

                for reverse_a in [
                    False,
                    True,
                ]:

                    if reverse_a:

                        coordinate_a1 = float(
                            coordinate_pair_a[
                                1
                            ]
                        )

                        coordinate_a2 = float(
                            coordinate_pair_a[
                                0
                            ]
                        )


                    else:

                        coordinate_a1 = float(
                            coordinate_pair_a[
                                0
                            ]
                        )

                        coordinate_a2 = float(
                            coordinate_pair_a[
                                1
                            ]
                        )


                    for reverse_b in [
                        False,
                        True,
                    ]:

                        if reverse_b:

                            coordinate_b1 = float(
                                coordinate_pair_b[
                                    1
                                ]
                            )

                            coordinate_b2 = float(
                                coordinate_pair_b[
                                    0
                                ]
                            )


                        else:

                            coordinate_b1 = float(
                                coordinate_pair_b[
                                    0
                                ]
                            )

                            coordinate_b2 = float(
                                coordinate_pair_b[
                                    1
                                ]
                            )


                        if (
                            mapping_name
                            ==
                            "A_X_B_Y"
                        ):

                            pitch_points = np.asarray(
                                [
                                    [
                                        coordinate_a1,
                                        coordinate_b1,
                                    ],

                                    [
                                        coordinate_a1,
                                        coordinate_b2,
                                    ],

                                    [
                                        coordinate_a2,
                                        coordinate_b1,
                                    ],

                                    [
                                        coordinate_a2,
                                        coordinate_b2,
                                    ],
                                ],
                                dtype=np.float32,
                            )


                        else:

                            pitch_points = np.asarray(
                                [
                                    [
                                        coordinate_b1,
                                        coordinate_a1,
                                    ],

                                    [
                                        coordinate_b2,
                                        coordinate_a1,
                                    ],

                                    [
                                        coordinate_b1,
                                        coordinate_a2,
                                    ],

                                    [
                                        coordinate_b2,
                                        coordinate_a2,
                                    ],
                                ],
                                dtype=np.float32,
                            )


                        coordinate_template_rows.append(
                            {
                                "template_index":
                                    int(
                                        template_index
                                    ),

                                "mapping_code":
                                    int(
                                        mapping_code
                                    ),

                                "axis_mapping":
                                    mapping_name,

                                "family_a_coordinate_1":
                                    coordinate_a1,

                                "family_a_coordinate_2":
                                    coordinate_a2,

                                "family_b_coordinate_1":
                                    coordinate_b1,

                                "family_b_coordinate_2":
                                    coordinate_b2,
                            }
                        )


                        coordinate_pitch_points.append(
                            pitch_points
                        )


                        template_index += 1


    coordinate_template_table = pd.DataFrame(
        coordinate_template_rows
    )

    # --------------------------------------------------------------------------------
    # PURE_ASSIGNMENT | UNIQUE_MAX_AXIS_LEAK | Notebook-37 code cell 33 line 113
    # --------------------------------------------------------------------------------
    UNIQUE_MAX_AXIS_LEAK = 0.35

    # --------------------------------------------------------------------------------
    # PURE_ASSIGNMENT | UNIQUE_MAX_COORDINATE_RESIDUAL_M | Notebook-37 code cell 33 line 115
    # --------------------------------------------------------------------------------
    UNIQUE_MAX_COORDINATE_RESIDUAL_M = 3.0

    # --------------------------------------------------------------------------------
    # PURE_ASSIGNMENT | PLAYER_PITCH_LENGTH_M | Notebook-37 code cell 34 line 91
    # --------------------------------------------------------------------------------
    PLAYER_PITCH_LENGTH_M = 105.0

    # --------------------------------------------------------------------------------
    # PURE_ASSIGNMENT | PLAYER_PITCH_WIDTH_M | Notebook-37 code cell 34 line 92
    # --------------------------------------------------------------------------------
    PLAYER_PITCH_WIDTH_M = 68.0

    # --------------------------------------------------------------------------------
    # PURE_ASSIGNMENT | PLAYER_PITCH_MARGIN_M | Notebook-37 code cell 34 line 94
    # --------------------------------------------------------------------------------
    PLAYER_PITCH_MARGIN_M = 5.0

    # --------------------------------------------------------------------------------
    # PURE_ASSIGNMENT | UNIT_RECTANGLE_POINTS | Notebook-37 code cell 36 line 102
    # --------------------------------------------------------------------------------
    UNIT_RECTANGLE_POINTS = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    # --------------------------------------------------------------------------------
    # MUTATION_BUILDER | TEMPLATE_AFFINE_BUILDER | Notebook-37 code cell 36 line 139
    # --------------------------------------------------------------------------------
    template_affine_matrices = []

    template_affine_rows = []


    for _, template_row in (
        coordinate_template_table
        .sort_values(
            "template_index"
        )
        .iterrows()
    ):

        template_index = int(
            template_row[
                "template_index"
            ]
        )


        mapping_name = str(
            template_row[
                "axis_mapping"
            ]
        )


        a1 = float(
            template_row[
                "family_a_coordinate_1"
            ]
        )


        a2 = float(
            template_row[
                "family_a_coordinate_2"
            ]
        )


        b1 = float(
            template_row[
                "family_b_coordinate_1"
            ]
        )


        b2 = float(
            template_row[
                "family_b_coordinate_2"
            ]
        )


        delta_a = float(
            a2
            -
            a1
        )


        delta_b = float(
            b2
            -
            b1
        )


        if (
            mapping_name
            ==
            "A_X_B_Y"
        ):

            # ----------------------------------------------------
            # x = a1 + delta_a * u
            # y = b1 + delta_b * v
            # ----------------------------------------------------

            T = np.asarray(
                [
                    [
                        delta_a,
                        0.0,
                        a1,
                    ],
                    [
                        0.0,
                        delta_b,
                        b1,
                    ],
                    [
                        0.0,
                        0.0,
                        1.0,
                    ],
                ],
                dtype=float,
            )


        elif (
            mapping_name
            ==
            "A_Y_B_X"
        ):

            # ----------------------------------------------------
            # x = b1 + delta_b * v
            # y = a1 + delta_a * u
            # ----------------------------------------------------

            T = np.asarray(
                [
                    [
                        0.0,
                        delta_b,
                        b1,
                    ],
                    [
                        delta_a,
                        0.0,
                        a1,
                    ],
                    [
                        0.0,
                        0.0,
                        1.0,
                    ],
                ],
                dtype=float,
            )


        else:

            raise ValueError(
                (
                    "Unknown axis mapping: "
                    f"{mapping_name}"
                )
            )


        determinant = float(
            np.linalg.det(
                T
            )
        )


        if (
            abs(
                determinant
            )
            <=
            1e-12
        ):

            raise RuntimeError(
                (
                    "Singular template affine matrix for "
                    f"template {template_index}"
                )
            )


        template_affine_matrices.append(
            T
        )


        template_affine_rows.append(
            {
                "template_index":
                    template_index,

                "axis_mapping":
                    mapping_name,

                "family_a_coordinate_1":
                    a1,

                "family_a_coordinate_2":
                    a2,

                "family_b_coordinate_1":
                    b1,

                "family_b_coordinate_2":
                    b2,

                "delta_a":
                    delta_a,

                "delta_b":
                    delta_b,

                "affine_determinant":
                    determinant,
            }
        )


    COORDINATE_TEMPLATE_AFFINES = np.stack(
        template_affine_matrices,
        axis=0,
    )


    coordinate_template_affine_table = pd.DataFrame(
        template_affine_rows
    )


    COORDINATE_TEMPLATE_AFFINE_INV_TRANSPOSE = (
        np.linalg.inv(
            COORDINATE_TEMPLATE_AFFINES
        )
        .transpose(
            0,
            2,
            1,
        )
    )

    # --------------------------------------------------------------------------------
    # PURE_ASSIGNMENT | BATCH_CANONICAL_X | Notebook-37 code cell 38 line 99
    # --------------------------------------------------------------------------------
    BATCH_CANONICAL_X = np.asarray(
        CANONICAL_X_LINES,
        dtype=float,
    )

    # --------------------------------------------------------------------------------
    # PURE_ASSIGNMENT | BATCH_CANONICAL_Y | Notebook-37 code cell 38 line 105
    # --------------------------------------------------------------------------------
    BATCH_CANONICAL_Y = np.asarray(
        CANONICAL_Y_LINES,
        dtype=float,
    )

    # --------------------------------------------------------------------------------
    # PURE_ASSIGNMENT | BATCH_MAPPING_CODE | Notebook-37 code cell 38 line 111
    # --------------------------------------------------------------------------------
    BATCH_MAPPING_CODE = (
        coordinate_template_table
        .sort_values(
            "template_index"
        )[
            "mapping_code"
        ]
        .to_numpy(
            dtype=int
        )
    )

    # --------------------------------------------------------------------------------
    # PURE_ASSIGNMENT | BATCH_TEMPLATE_INDEX | Notebook-37 code cell 38 line 124
    # --------------------------------------------------------------------------------
    BATCH_TEMPLATE_INDEX = (
        coordinate_template_table
        .sort_values(
            "template_index"
        )[
            "template_index"
        ]
        .to_numpy(
            dtype=int
        )
    )

    # --------------------------------------------------------------------------------
    # PURE_ASSIGNMENT | FULL_SEARCH_MAX_CONDITION_NUMBER | Notebook-37 code cell 40 line 183
    # --------------------------------------------------------------------------------
    FULL_SEARCH_MAX_CONDITION_NUMBER = (
        COORD_AUDIT_MAX_CONDITION_NUMBER
    )

    # --------------------------------------------------------------------------------
    # PURE_ASSIGNMENT | FULL_SEARCH_MAX_ABS_ANCHOR_M | Notebook-37 code cell 40 line 187
    # --------------------------------------------------------------------------------
    FULL_SEARCH_MAX_ABS_ANCHOR_M = (
        COORD_AUDIT_MAX_ABS_ANCHOR_M
    )

    if len(coordinate_template_table) != 2520:
        raise RuntimeError(
            "Expected exactly 2520 coordinate templates."
        )

    if COORDINATE_TEMPLATE_AFFINES.shape != (2520, 3, 3):
        raise RuntimeError(
            "Coordinate-template affine tensor has wrong shape."
        )

    if COORDINATE_TEMPLATE_AFFINE_INV_TRANSPOSE.shape != (2520, 3, 3):
        raise RuntimeError(
            "Inverse-transpose affine tensor has wrong shape."
        )

    if len(BATCH_TEMPLATE_INDEX) != 2520:
        raise RuntimeError(
            "BATCH_TEMPLATE_INDEX has wrong length."
        )

    _NOTEBOOK43_HOMOGRAPHY_STATIC_READY = True

    return {
        "coordinate_templates": int(len(coordinate_template_table)),
        "template_affines": int(COORDINATE_TEMPLATE_AFFINES.shape[0]),
        "inverse_transpose_affines": int(COORDINATE_TEMPLATE_AFFINE_INV_TRANSPOSE.shape[0]),
        "batch_templates": int(len(BATCH_TEMPLATE_INDEX)),
    }

def _notebook43_get_value(
    value,
    key,
    default=None,
):
    """Read one field from either a dict or an attribute-bearing object."""

    if isinstance(
        value,
        dict,
    ):

        return value.get(
            key,
            default,
        )


    return getattr(
        value,
        key,
        default,
    )

def _notebook43_build_robust_family_table(
    geometry_segments,
    geometry_frame,
):
    """Fresh exact hard-family table from current pitch geometry."""

    modes = robust_two_orientation_modes(
        geometry_segments
    )


    rows = []


    for auto_line_index, (
        line,
        assignment,
    ) in enumerate(
        zip(
            geometry_segments,
            modes[
                "assignments"
            ],
        )
    ):

        coefficients = (
            segment_to_line_coefficients(
                line
            )
        )


        if coefficients is None:

            continue


        rows.append(
            {
                "analysis_frame":
                    int(
                        geometry_frame
                    ),

                "auto_line_index":
                    int(
                        auto_line_index
                    ),

                "family":
                    str(
                        assignment[
                            "family"
                        ]
                    ),

                "hard_family":
                    str(
                        assignment[
                            "family"
                        ]
                    ),

                "line_angle_degrees":
                    float(
                        line[
                            "angle_degrees"
                        ]
                    ),

                "line_length":
                    float(
                        line[
                            "length"
                        ]
                    ),

                "family_a_mode_deg":
                    float(
                        modes[
                            "family_a_mode_deg"
                        ]
                    ),

                "family_b_mode_deg":
                    float(
                        modes[
                            "family_b_mode_deg"
                        ]
                    ),

                "minimum_family_distance_deg":
                    float(
                        assignment[
                            "minimum_family_distance_deg"
                        ]
                    ),

                "x1":
                    float(
                        line[
                            "x1"
                        ]
                    ),

                "y1":
                    float(
                        line[
                            "y1"
                        ]
                    ),

                "x2":
                    float(
                        line[
                            "x2"
                        ]
                    ),

                "y2":
                    float(
                        line[
                            "y2"
                        ]
                    ),

                "line_a":
                    float(
                        coefficients[
                            0
                        ]
                    ),

                "line_b":
                    float(
                        coefficients[
                            1
                        ]
                    ),

                "line_c":
                    float(
                        coefficients[
                            2
                        ]
                    ),
            }
        )


    table = pd.DataFrame(
        rows
    )


    return (
        modes,
        table,
    )

def _notebook43_build_line_priority_table(
    robust_table,
    frame_width,
    frame_height,
):
    """
    Fresh oracle-free soft-family priority table.

    Frozen policy:
        maximum soft-family distance = 18 degrees
        proposal priority =
            length * exp(-0.5 * (family_distance / 8)^2)
        ordering =
            priority DESC, length DESC, line index ASC
    """

    if len(
        robust_table
    ) == 0:

        raise RuntimeError(
            "Cannot build line priority from zero geometry lines."
        )


    family_a_mode = float(
        robust_table[
            "family_a_mode_deg"
        ].iloc[
            0
        ]
    )


    family_b_mode = float(
        robust_table[
            "family_b_mode_deg"
        ].iloc[
            0
        ]
    )


    image_center_x = float(
        frame_width
    ) / 2.0


    image_center_y = float(
        frame_height
    ) / 2.0


    priority_rows = []


    for _, line_row in robust_table.iterrows():

        assignment = nearest_soft_family(

            angle_degrees=float(
                line_row[
                    "line_angle_degrees"
                ]
            ),

            family_a_mode_deg=
                family_a_mode,

            family_b_mode_deg=
                family_b_mode,

            maximum_distance_deg=
                18.0,
        )


        soft_family = str(
            assignment[
                "soft_family"
            ]
        )


        minimum_family_distance = float(
            assignment[
                "minimum_family_distance_deg"
            ]
        )


        if soft_family == "FAMILY_A":

            family_mode = (
                family_a_mode
            )

        elif soft_family == "FAMILY_B":

            family_mode = (
                family_b_mode
            )

        else:

            family_mode = np.nan


        if soft_family != "OUTLIER":

            family_position = (
                line_position_along_family_normal(

                    segment=line_row,

                    family_mode_angle_deg=
                        family_mode,

                    image_center_x=
                        image_center_x,

                    image_center_y=
                        image_center_y,
                )
            )


            proposal_priority = float(

                float(
                    line_row[
                        "line_length"
                    ]
                )

                *

                np.exp(
                    -0.5
                    *
                    (
                        minimum_family_distance
                        /
                        8.0
                    )
                    ** 2
                )
            )


        else:

            family_position = np.nan

            proposal_priority = np.nan


        row = dict(
            line_row
        )


        row[
            "soft_family"
        ] = soft_family


        row[
            "minimum_family_distance_deg"
        ] = minimum_family_distance


        row[
            "family_position"
        ] = (
            float(
                family_position
            )
            if np.isfinite(
                family_position
            )
            else np.nan
        )


        row[
            "proposal_priority"
        ] = (
            float(
                proposal_priority
            )
            if np.isfinite(
                proposal_priority
            )
            else np.nan
        )


        priority_rows.append(
            row
        )


    priority_table = pd.DataFrame(
        priority_rows
    )


    priority_table[
        "priority_rank"
    ] = np.nan


    for family_name in [
        "FAMILY_A",
        "FAMILY_B",
    ]:

        mask = (
            priority_table[
                "soft_family"
            ]
            ==
            family_name
        )


        family_rows = (
            priority_table[
                mask
            ]
            .sort_values(
                [
                    "proposal_priority",
                    "line_length",
                    "auto_line_index",
                ],
                ascending=[
                    False,
                    False,
                    True,
                ],
            )
        )


        for rank, dataframe_index in enumerate(
            family_rows.index.tolist(),
            start=1,
        ):

            priority_table.loc[
                dataframe_index,
                "priority_rank",
            ] = int(
                rank
            )


    priority_table[
        "priority_rank"
    ] = (
        priority_table[
            "priority_rank"
        ]
        .astype(
            "Int64"
        )
    )


    # The selector emits this name in its output.
    priority_table[
        "original_priority_rank"
    ] = (
        priority_table[
            "priority_rank"
        ]
    )


    # Preserve frozen spatial-redundancy diagnostics.
    for radius_px in [
        3.0,
        5.0,
        10.0,
        15.0,
    ]:

        column_name = (
            f"neighbors_within_{int(radius_px)}px"
        )


        priority_table[
            column_name
        ] = 0


        for family_name in [
            "FAMILY_A",
            "FAMILY_B",
        ]:

            group_indices = (
                priority_table[
                    priority_table[
                        "soft_family"
                    ]
                    ==
                    family_name
                ]
                .index
                .tolist()
            )


            if not group_indices:

                continue


            positions = (
                priority_table.loc[
                    group_indices,
                    "family_position",
                ]
                .to_numpy(
                    dtype=float
                )
            )


            distance_matrix = np.abs(
                positions[
                    :,
                    None,
                ]
                -
                positions[
                    None,
                    :,
                ]
            )


            neighbor_counts = (
                (
                    distance_matrix
                    <=
                    float(
                        radius_px
                    )
                )
                .sum(
                    axis=1
                )
                -
                1
            )


            priority_table.loc[
                group_indices,
                column_name,
            ] = (
                neighbor_counts
                .astype(int)
            )


    for pool_size in [
        8,
        12,
        16,
        20,
    ]:

        priority_table[
            f"in_top_{pool_size}"
        ] = (
            priority_table[
                "priority_rank"
            ]
            .notna()
            &
            (
                priority_table[
                    "priority_rank"
                ]
                <=
                pool_size
            )
        )


    return priority_table

def _notebook43_build_exact_frame_player_table(
    result,
    geometry_frame,
    frame_height,
):
    """Fresh scorer player table from current PlayerTrackResult."""

    detection_rows = []


    # Map (frame, detection) -> track ID from the current tracking output.
    track_lookup = {}


    for track in result.player_tracks.tracks:

        for observation in track.observations:

            track_lookup[
                (
                    int(
                        observation.frame_idx
                    ),

                    int(
                        observation.detection_idx
                    ),
                )
            ] = int(
                track.track_id
            )


    support_lookup = {}


    support_decisions = (
        result.metadata.get(
            "player_detection_tracking",
            {}
        )
        .get(
            "field_support_decisions",
            []
        )
    )


    for decision in support_decisions:

        frame_idx = int(
            _notebook43_get_value(
                decision,
                "frame_idx",
            )
        )


        detection_idx = int(
            _notebook43_get_value(
                decision,
                "detection_idx",
            )
        )


        support_lookup[
            (
                frame_idx,
                detection_idx,
            )
        ] = decision


    for detection in result.player_tracks.detections:

        if int(
            detection.frame_idx
        ) != int(
            geometry_frame
        ):

            continue


        key = (
            int(
                detection.frame_idx
            ),

            int(
                detection.detection_idx
            ),
        )


        decision = support_lookup.get(
            key
        )


        if decision is None:

            raise RuntimeError(
                "Exact-frame player detection is missing its field-support decision."
            )


        field_supported = bool(
            _notebook43_get_value(
                decision,
                "field_supported",
            )
        )


        if not field_supported:

            continue


        if key not in track_lookup:

            raise RuntimeError(
                "Field-supported exact-frame player detection is missing its tracked observation."
            )


        closeup_person = bool(
            _notebook43_get_value(
                decision,
                "is_closeup",
                False,
            )
        )


        foot_green_fraction = (
            _notebook43_get_value(
                decision,
                "foot_green_fraction",
                np.nan,
            )
        )


        detection_rows.append(
            {
                "track_id":
                    int(
                        track_lookup[
                            key
                        ]
                    ),

                "frame_number":
                    int(
                        detection.frame_idx
                    ),

                "detection_id":
                    int(
                        detection.detection_idx
                    ),

                "confidence":
                    float(
                        detection.confidence
                    ),

                "x1":
                    float(
                        detection.x1
                    ),

                "y1":
                    float(
                        detection.y1
                    ),

                "x2":
                    float(
                        detection.x2
                    ),

                "y2":
                    float(
                        detection.y2
                    ),

                "bottom_center_x":
                    float(
                        detection.foot_x
                    ),

                "bottom_center_y":
                    float(
                        detection.foot_y
                    ),

                "box_height_fraction":
                    float(
                        detection.height
                    )
                    /
                    float(
                        frame_height
                    ),

                "foot_green_fraction":
                    float(
                        foot_green_fraction
                    )
                    if (
                        foot_green_fraction
                        is not None
                        and
                        np.isfinite(
                            float(
                                foot_green_fraction
                            )
                        )
                    )
                    else np.nan,

                "closeup_person":
                    closeup_person,

                "field_supported":
                    field_supported,
            }
        )


    player_table = (
        pd.DataFrame(
            detection_rows
        )
        .sort_values(
            [
                "frame_number",
                "detection_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    return player_table

def _notebook43_pair_product_table(
    family_a_pairs,
    family_b_pairs,
    variant_name,
):
    """Cartesian product of the two fresh selected-line pair tables."""

    rows = []


    for _, pair_a in family_a_pairs.iterrows():

        for _, pair_b in family_b_pairs.iterrows():

            rows.append(
                {
                    "family_a_line_1":
                        int(
                            pair_a[
                                "first_auto_line_index"
                            ]
                        ),

                    "family_a_line_2":
                        int(
                            pair_a[
                                "second_auto_line_index"
                            ]
                        ),

                    "family_b_line_1":
                        int(
                            pair_b[
                                "first_auto_line_index"
                            ]
                        ),

                    "family_b_line_2":
                        int(
                            pair_b[
                                "second_auto_line_index"
                            ]
                        ),

                    "family_a_pair_separation_px":
                        float(
                            pair_a[
                                "separation_px"
                            ]
                        ),

                    "family_b_pair_separation_px":
                        float(
                            pair_b[
                                "separation_px"
                            ]
                        ),

                    "family_a_pair_strength":
                        float(
                            pair_a[
                                "pair_strength"
                            ]
                        ),

                    "family_b_pair_strength":
                        float(
                            pair_b[
                                "pair_strength"
                            ]
                        ),

                    "variant_name":
                        str(
                            variant_name
                        ),
                }
            )


    table = pd.DataFrame(
        rows
    )


    table[
        "pair_product_rank"
    ] = (
        np.arange(
            len(
                table
            )
        )
        +
        1
    )


    return table

def prepare_homography_search_runtime(
    result,
):
    """
    Build the complete fresh runtime required by canonical homography search.

    This does NOT select a homography and does NOT complete the pipeline stage.
    """

    expected_prefix = list(
        CANONICAL_PIPELINE_STAGES[
            : _STAGE_INDEX[
                "homography_search"
            ]
        ]
    )


    if (
        result.pipeline_status.completed_stages
        !=
        expected_prefix
    ):

        raise PipelineContractError(
            "Homography runtime cannot be built before completed "
            "pitch-line detection."
        )


    pitch_metadata = result.metadata.get(
        "pitch_line_detection"
    )


    if not isinstance(
        pitch_metadata,
        dict,
    ):

        raise PipelineContractError(
            "Current pitch-line metadata is missing."
        )


    geometry_segments = pitch_metadata.get(
        "geometry_segments"
    )


    if not isinstance(
        geometry_segments,
        list,
    ):

        raise PipelineContractError(
            "Current geometry_segments must be list-valued."
        )


    if len(
        geometry_segments
    ) == 0:

        raise RuntimeError(
            "Current geometry contains zero segments."
        )


    geometry_frame = int(
        pitch_metadata[
            "frame_idx"
        ]
    )


    frame_width = int(
        pitch_metadata[
            "frame_width"
        ]
    )


    frame_height = int(
        pitch_metadata[
            "frame_height"
        ]
    )


    # --------------------------------------------------------------------------------------------
    # FRESH HARD FAMILIES
    # --------------------------------------------------------------------------------------------

    (
        modes,
        hard_family_table,
    ) = _notebook43_build_robust_family_table(

        geometry_segments=
            geometry_segments,

        geometry_frame=
            geometry_frame,
    )


    # --------------------------------------------------------------------------------------------
    # FRESH SOFT PRIORITY
    # --------------------------------------------------------------------------------------------

    priority_table = (
        _notebook43_build_line_priority_table(

            robust_table=
                hard_family_table,

            frame_width=
                frame_width,

            frame_height=
                frame_height,
        )
    )


    soft_family_mapping = {

        int(
            row[
                "auto_line_index"
            ]
        ):
            str(
                row[
                    "soft_family"
                ]
            )

        for _, row
        in priority_table.iterrows()
    }


    soft_family_table = (
        hard_family_table
        .copy()
    )


    soft_family_table[
        "family"
    ] = (
        soft_family_table[
            "auto_line_index"
        ]
        .astype(int)
        .map(
            soft_family_mapping
        )
        .fillna(
            "OUTLIER"
        )
    )


    # --------------------------------------------------------------------------------------------
    # FRESH EXACT-FRAME PLAYERS
    # --------------------------------------------------------------------------------------------

    player_table = (
        _notebook43_build_exact_frame_player_table(

            result=
                result,

            geometry_frame=
                geometry_frame,

            frame_height=
                frame_height,
        )
    )


    # --------------------------------------------------------------------------------------------
    # ALL 11 FROZEN VARIANTS
    # --------------------------------------------------------------------------------------------

    variant_runtime = {}


    for (
        variant_name,
        core_size,
        expansion_size,
    ) in HOMOGRAPHY_SEARCH_VARIANTS:

        family_a_input = (
            priority_table[
                priority_table[
                    "soft_family"
                ]
                ==
                "FAMILY_A"
            ]
            .copy()
        )


        family_b_input = (
            priority_table[
                priority_table[
                    "soft_family"
                ]
                ==
                "FAMILY_B"
            ]
            .copy()
        )


        selected_a = (
            select_core_plus_spatial_expansion(

                family_a_input,

                core_size,

                expansion_size,
            )
        )


        selected_b = (
            select_core_plus_spatial_expansion(

                family_b_input,

                core_size,

                expansion_size,
            )
        )


        if not isinstance(
            selected_a,
            pd.DataFrame,
        ):

            raise RuntimeError(
                f"{variant_name}: FAMILY_A selector did not return DataFrame."
            )


        if not isinstance(
            selected_b,
            pd.DataFrame,
        ):

            raise RuntimeError(
                f"{variant_name}: FAMILY_B selector did not return DataFrame."
            )


        family_a_pairs_raw = (
            build_selected_line_pairs(
                selected_a
            )
        )


        family_b_pairs_raw = (
            build_selected_line_pairs(
                selected_b
            )
        )


        if not (
            isinstance(
                family_a_pairs_raw,
                tuple,
            )
            and
            len(
                family_a_pairs_raw
            )
            ==
            2
        ):

            raise RuntimeError(
                f"{variant_name}: FAMILY_A pair-builder return contract failed."
            )


        if not (
            isinstance(
                family_b_pairs_raw,
                tuple,
            )
            and
            len(
                family_b_pairs_raw
            )
            ==
            2
        ):

            raise RuntimeError(
                f"{variant_name}: FAMILY_B pair-builder return contract failed."
            )


        family_a_pairs = (
            family_a_pairs_raw[
                0
            ]
            .copy()
        )


        family_b_pairs = (
            family_b_pairs_raw[
                0
            ]
            .copy()
        )


        pair_product_table = (
            _notebook43_pair_product_table(

                family_a_pairs=
                    family_a_pairs,

                family_b_pairs=
                    family_b_pairs,

                variant_name=
                    variant_name,
            )
        )


        runtime_event_id = (
            "NOTEBOOK43_FRESH::"
            +
            str(
                variant_name
            )
        )


        # ----------------------------------------------------------------------------------------
        # SCORER CACHE — CURRENT LINES ONLY
        # ----------------------------------------------------------------------------------------

        selected_a_ids = set(
            selected_a[
                "auto_line_index"
            ]
            .astype(int)
            .tolist()
        )


        selected_b_ids = set(
            selected_b[
                "auto_line_index"
            ]
            .astype(int)
            .tolist()
        )


        scoring_table = (
            soft_family_table[
                (
                    (
                        soft_family_table[
                            "family"
                        ]
                        ==
                        "FAMILY_A"
                    )
                    &
                    (
                        soft_family_table[
                            "auto_line_index"
                        ]
                        .astype(int)
                        .isin(
                            selected_a_ids
                        )
                    )
                )
                |
                (
                    (
                        soft_family_table[
                            "family"
                        ]
                        ==
                        "FAMILY_B"
                    )
                    &
                    (
                        soft_family_table[
                            "auto_line_index"
                        ]
                        .astype(int)
                        .isin(
                            selected_b_ids
                        )
                    )
                )
            ]
            .copy()
        )


        robust_family_cache[
            runtime_event_id
        ] = (
            soft_family_table
            .copy()
        )


        selected_line_score_cache[
            runtime_event_id
        ] = {

            "coefficients":
                scoring_table[
                    [
                        "line_a",
                        "line_b",
                        "line_c",
                    ]
                ]
                .to_numpy(
                    dtype=float
                ),

            "families":
                scoring_table[
                    "family"
                ]
                .astype(str)
                .to_numpy(),

            "line_lengths":
                scoring_table[
                    "line_length"
                ]
                .to_numpy(
                    dtype=float
                ),

            "line_count":
                int(
                    len(
                        scoring_table
                    )
                ),
        }


        exact_frame_player_cache[
            runtime_event_id
        ] = {

            "detections":
                player_table.copy(),

            "analysis_frame":
                int(
                    geometry_frame
                ),
        }


        variant_runtime[
            variant_name
        ] = {

            "runtime_event_id":
                runtime_event_id,

            "core_size":
                int(
                    core_size
                ),

            "expansion_size":
                int(
                    expansion_size
                ),

            "selected_a":
                selected_a,

            "selected_b":
                selected_b,

            "family_a_pairs":
                family_a_pairs,

            "family_b_pairs":
                family_b_pairs,

            "pair_product_table":
                pair_product_table,

            "selected_line_count":
                int(
                    len(
                        scoring_table
                    )
                ),
        }


    return {

        "geometry_frame":
            geometry_frame,

        "frame_width":
            frame_width,

        "frame_height":
            frame_height,

        "modes":
            modes,

        "hard_family_table":
            hard_family_table,

        "priority_table":
            priority_table,

        "soft_family_table":
            soft_family_table,

        "player_table":
            player_table,

        "variants":
            variant_runtime,
    }



# NOTEBOOK43_CANONICAL_FULL_HOMOGRAPHY_SEARCH_V1
# ====================================================================================================
# FULL FRESH HOMOGRAPHY SEARCH + AUTOMATIC OPERATIONAL RECOVERY
# ====================================================================================================


def _notebook43_primary_coordinate_order(
    batch_table,
):
    """
    Frozen oracle-free primary geometry ranking.

    Shooter position is NOT part of this ranking.
    Player location is NOT part of this ranking.
    """

    ranking = (
        batch_table
        .sort_values(
            [
                "batch_distinct_balance",
                "batch_distinct_total",
                "batch_duplicate_fraction",
                "batch_median_distinct_residual_m",
                "batch_median_distinct_axis_leak",
                "template_index",
            ],
            ascending=[
                False,
                False,
                True,
                True,
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


    ranking[
        "within_pair_primary_rank"
    ] = (
        np.arange(
            len(
                ranking
            )
        )
        +
        1
    )


    return ranking


def _notebook43_player_coordinate_order(
    batch_table,
):
    """Frozen player-plausibility diagnostic ranking."""

    ranking = (
        batch_table
        .sort_values(
            [
                "batch_inside_pitch_fraction",
                "batch_inside_pitch_count",
                "batch_soft_pitch_fraction",
                "batch_median_outside_distance_m",
                "batch_distinct_balance",
                "batch_distinct_total",
                "batch_duplicate_fraction",
                "batch_median_distinct_residual_m",
                "template_index",
            ],
            ascending=[
                False,
                False,
                False,
                True,
                False,
                False,
                True,
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


    ranking[
        "within_pair_player_rank"
    ] = (
        np.arange(
            len(
                ranking
            )
        )
        +
        1
    )


    return ranking


def _notebook43_geometry_key(
    row,
):
    """Canonical pair/template identity."""

    return (
        "A"
        +
        str(
            int(
                row[
                    "family_a_line_1"
                ]
            )
        )
        +
        "-"
        +
        str(
            int(
                row[
                    "family_a_line_2"
                ]
            )
        )
        +
        "_B"
        +
        str(
            int(
                row[
                    "family_b_line_1"
                ]
            )
        )
        +
        "-"
        +
        str(
            int(
                row[
                    "family_b_line_2"
                ]
            )
        )
        +
        "_T"
        +
        str(
            int(
                row[
                    "template_index"
                ]
            )
        )
    )


def _notebook43_pair_context_key(
    variant_name,
    row,
):
    """Canonical variant + line-pair context identity."""

    return (
        str(
            variant_name
        )
        +
        "|A"
        +
        str(
            int(
                row[
                    "family_a_line_1"
                ]
            )
        )
        +
        "-"
        +
        str(
            int(
                row[
                    "family_a_line_2"
                ]
            )
        )
        +
        "|B"
        +
        str(
            int(
                row[
                    "family_b_line_1"
                ]
            )
        )
        +
        "-"
        +
        str(
            int(
                row[
                    "family_b_line_2"
                ]
            )
        )
    )


def _notebook43_project_xy(
    H,
    x,
    y,
):
    """Project one image point with one 3x3 homography."""

    source = np.asarray(
        [
            float(
                x
            ),
            float(
                y
            ),
            1.0,
        ],
        dtype=float,
    )


    projected = (
        np.asarray(
            H,
            dtype=float,
        )
        @
        source
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
        1e-12
    ):

        return (
            np.nan,
            np.nan,
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


    return (
        pitch_x,
        pitch_y,
    )


def _notebook43_point_to_bbox_distance(
    point_x,
    point_y,
    observation,
):
    """Euclidean distance from point to axis-aligned player bounding box."""

    x1 = float(
        observation.x1
    )

    y1 = float(
        observation.y1
    )

    x2 = float(
        observation.x2
    )

    y2 = float(
        observation.y2
    )


    dx = max(
        x1
        -
        float(
            point_x
        ),
        0.0,
        float(
            point_x
        )
        -
        x2,
    )


    dy = max(
        y1
        -
        float(
            point_y
        ),
        0.0,
        float(
            point_y
        )
        -
        y2,
    )


    return float(
        math.hypot(
            dx,
            dy,
        )
    )


def _notebook43_provisional_shooter_evidence(
    result,
):
    """
    Reference-free provisional shooter for Stage-8 operational acceptance.

    This DOES NOT populate the public ShooterResult.

    Frozen rule:
        first ball trajectory frame
        point-to-axis-aligned-bbox distance
        normalize by player box height
        lowest normalized distance wins
    """

    if not result.ball_track.success:

        raise PipelineContractError(
            "Operational homography recovery requires successful ball tracking."
        )


    trajectory = result.ball_track.track


    if not isinstance(
        trajectory,
        list,
    ) or len(
        trajectory
    ) == 0:

        raise RuntimeError(
            "Operational homography recovery received no ball trajectory."
        )


    first_ball_point = sorted(
        trajectory,
        key=lambda point:
            int(
                point.frame_idx
            ),
    )[
        0
    ]


    first_frame = int(
        first_ball_point.frame_idx
    )


    ball_x = float(
        first_ball_point.center_x
    )


    ball_y = float(
        first_ball_point.center_y
    )


    candidate_rows = []


    for track in result.player_tracks.tracks:

        matching_observations = [

            observation

            for observation
            in track.observations

            if int(
                observation.frame_idx
            )
            ==
            first_frame
        ]


        if len(
            matching_observations
        ) != 1:

            continue


        observation = (
            matching_observations[
                0
            ]
        )


        pixel_distance = (
            _notebook43_point_to_bbox_distance(

                point_x=
                    ball_x,

                point_y=
                    ball_y,

                observation=
                    observation,
            )
        )


        box_height = max(
            float(
                observation.height
            ),
            1.0,
        )


        normalized_distance = float(
            pixel_distance
            /
            box_height
        )


        candidate_rows.append(
            {
                "track_id":
                    int(
                        track.track_id
                    ),

                "frame_idx":
                    first_frame,

                "pixel_distance":
                    pixel_distance,

                "normalized_distance":
                    normalized_distance,

                "image_x":
                    float(
                        observation.foot_x
                    ),

                "image_y":
                    float(
                        observation.foot_y
                    ),
            }
        )


    if not candidate_rows:

        raise RuntimeError(
            "No provisional shooter candidates were available."
        )


    candidate_table = (
        pd.DataFrame(
            candidate_rows
        )
        .sort_values(
            [
                "normalized_distance",
                "pixel_distance",
                "track_id",
            ],
            ascending=[
                True,
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


    candidate_table[
        "provisional_rank"
    ] = (
        np.arange(
            len(
                candidate_table
            )
        )
        +
        1
    )


    selected = (
        candidate_table.iloc[
            0
        ]
    )


    return {

        "track_id":
            int(
                selected[
                    "track_id"
                ]
            ),

        "frame_idx":
            first_frame,

        "image_xy":
            (
                float(
                    selected[
                        "image_x"
                    ]
                ),
                float(
                    selected[
                        "image_y"
                    ]
                ),
            ),

        "pixel_distance":
            float(
                selected[
                    "pixel_distance"
                ]
            ),

        "normalized_distance":
            float(
                selected[
                    "normalized_distance"
                ]
            ),

        "candidate_table":
            candidate_table,
    }


def _notebook43_pair_row_for_identity(
    pair_table,
    candidate_row,
):
    """Recover exactly one pair-product row from its line identities."""

    match = (
        pair_table[
            (
                pair_table[
                    "family_a_line_1"
                ]
                .astype(int)
                ==
                int(
                    candidate_row[
                        "family_a_line_1"
                    ]
                )
            )
            &
            (
                pair_table[
                    "family_a_line_2"
                ]
                .astype(int)
                ==
                int(
                    candidate_row[
                        "family_a_line_2"
                    ]
                )
            )
            &
            (
                pair_table[
                    "family_b_line_1"
                ]
                .astype(int)
                ==
                int(
                    candidate_row[
                        "family_b_line_1"
                    ]
                )
            )
            &
            (
                pair_table[
                    "family_b_line_2"
                ]
                .astype(int)
                ==
                int(
                    candidate_row[
                        "family_b_line_2"
                    ]
                )
            )
        ]
    )


    if len(
        match
    ) != 1:

        raise RuntimeError(
            "Expected exactly one pair-product identity match."
        )


    return (
        match.iloc[
            0
        ]
    )


def _notebook43_operational_candidate_checks(
    H,
    player_table,
    shooter_image_xy,
):
    """Current-player + provisional-shooter operational acceptance."""

    player_xy = []


    for _, player in player_table.iterrows():

        projected = (
            _notebook43_project_xy(

                H=

                    H,

                x=
                    player[
                        "bottom_center_x"
                    ],

                y=
                    player[
                        "bottom_center_y"
                    ],
            )
        )


        player_xy.append(
            projected
        )


    player_xy = np.asarray(
        player_xy,
        dtype=float,
    )


    all_players_finite = bool(
        np.isfinite(
            player_xy
        ).all()
    )


    if all_players_finite:

        inside_mask = (
            (
                player_xy[
                    :,
                    0
                ]
                >=
                0.0
            )
            &
            (
                player_xy[
                    :,
                    0
                ]
                <=
                105.0
            )
            &
            (
                player_xy[
                    :,
                    1
                ]
                >=
                0.0
            )
            &
            (
                player_xy[
                    :,
                    1
                ]
                <=
                68.0
            )
        )


        inside_count = int(
            inside_mask.sum()
        )


        inside_fraction = float(
            inside_count
            /
            max(
                len(
                    player_xy
                ),
                1,
            )
        )


    else:

        inside_count = 0

        inside_fraction = 0.0


    shooter_pitch_xy = (
        _notebook43_project_xy(

            H=
                H,

            x=
                shooter_image_xy[
                    0
                ],

            y=
                shooter_image_xy[
                    1
                ],
        )
    )


    shooter_finite = bool(
        np.isfinite(
            np.asarray(
                shooter_pitch_xy,
                dtype=float,
            )
        ).all()
    )


    shooter_inside_pitch = bool(
        shooter_finite
        and
        0.0
        <=
        float(
            shooter_pitch_xy[
                0
            ]
        )
        <=
        105.0
        and
        0.0
        <=
        float(
            shooter_pitch_xy[
                1
            ]
        )
        <=
        68.0
    )


    return {

        "all_scoring_players_finite":
            all_players_finite,

        "player_count":
            int(
                len(
                    player_xy
                )
            ),

        "inside_pitch_count":
            inside_count,

        "inside_pitch_fraction":
            inside_fraction,

        "shooter_pitch_x":
            float(
                shooter_pitch_xy[
                    0
                ]
            ),

        "shooter_pitch_y":
            float(
                shooter_pitch_xy[
                    1
                ]
            ),

        "shooter_projection_finite":
            shooter_finite,

        "shooter_inside_pitch":
            shooter_inside_pitch,
    }


def build_homography_search_handler(
    progress=True,
):
    """
    Build full canonical Stage-8 homography search.

    Frozen Notebook-42 homography/results are NOT inputs.
    """

    def homography_search_handler(
        result,
    ):

        # --------------------------------------------------------------------------------------------
        # STAGE ORDER
        # --------------------------------------------------------------------------------------------

        expected_prefix = list(
            CANONICAL_PIPELINE_STAGES[
                : _STAGE_INDEX[
                    "homography_search"
                ]
            ]
        )


        if (
            result.pipeline_status.completed_stages
            !=
            expected_prefix
        ):

            raise PipelineContractError(
                "Homography search cannot run before completed pitch-line detection."
            )


        # --------------------------------------------------------------------------------------------
        # STATIC + FRESH RUNTIME
        # --------------------------------------------------------------------------------------------

        initialize_homography_static_runtime()


        runtime = (
            prepare_homography_search_runtime(
                result
            )
        )


        variants = runtime[
            "variants"
        ]


        player_table = runtime[
            "player_table"
        ]


        frame_width = int(
            runtime[
                "frame_width"
            ]
        )


        frame_height = int(
            runtime[
                "frame_height"
            ]
        )


        # --------------------------------------------------------------------------------------------
        # PROVISIONAL SHOOTER — ACCEPTANCE GATE ONLY
        # --------------------------------------------------------------------------------------------

        provisional_shooter = (
            _notebook43_provisional_shooter_evidence(
                result
            )
        )


        shooter_image_xy = tuple(
            provisional_shooter[
                "image_xy"
            ]
        )


        # --------------------------------------------------------------------------------------------
        # FULL 28,296-PAIR SEARCH
        # --------------------------------------------------------------------------------------------

        pair_summary_rows = []

        unique_winner_rows = []

        player_winner_rows = []


        pair_products_attempted = 0

        pair_products_successful = 0

        pair_geometry_failures = 0

        template_rows_evaluated = 0


        for variant_order, (
            variant_name,
            variant_payload,
        ) in enumerate(
            variants.items()
        ):

            pair_table = (
                variant_payload[
                    "pair_product_table"
                ]
                .sort_values(
                    "pair_product_rank"
                )
                .reset_index(
                    drop=True
                )
            )


            if progress:

                print()

                print(
                    "=" * 108
                )

                print(
                    f"VARIANT: {variant_name}"
                )

                print(
                    "=" * 108
                )

                print(
                    f"Pair products : {len(pair_table)}"
                )


            successful_this_variant = 0

            failed_this_variant = 0


            for pair_counter, (
                _,
                pair_row,
            ) in enumerate(
                pair_table.iterrows(),
                start=1,
            ):

                pair_products_attempted += 1


                batch_result = (
                    score_pair_product_batched(

                        event_id=
                            variant_payload[
                                "runtime_event_id"
                            ],

                        pair_product_row=
                            pair_row,
                    )
                )


                if batch_result is None:

                    pair_geometry_failures += 1

                    failed_this_variant += 1


                    pair_summary_rows.append(
                        {
                            "variant_order":
                                int(
                                    variant_order
                                ),

                            "variant_name":
                                str(
                                    variant_name
                                ),

                            "pair_product_rank":
                                int(
                                    pair_row[
                                        "pair_product_rank"
                                    ]
                                ),

                            "family_a_line_1":
                                int(
                                    pair_row[
                                        "family_a_line_1"
                                    ]
                                ),

                            "family_a_line_2":
                                int(
                                    pair_row[
                                        "family_a_line_2"
                                    ]
                                ),

                            "family_b_line_1":
                                int(
                                    pair_row[
                                        "family_b_line_1"
                                    ]
                                ),

                            "family_b_line_2":
                                int(
                                    pair_row[
                                        "family_b_line_2"
                                    ]
                                ),

                            "coordinate_templates":
                                0,

                            "search_status":
                                "PAIR_GEOMETRY_FAILED",
                        }
                    )


                    continue


                (
                    batch_table,
                    pair_geometry,
                ) = batch_result


                pair_products_successful += 1

                successful_this_variant += 1


                template_rows_evaluated += int(
                    len(
                        batch_table
                    )
                )


                if len(
                    batch_table
                ) != 2520:

                    raise RuntimeError(
                        f"{variant_name}: expected 2520 coordinate templates per pair."
                    )


                # ------------------------------------------------------------------------------------
                # EXACT PRIMARY + PLAYER RANKINGS
                # ------------------------------------------------------------------------------------

                primary_ranking = (
                    _notebook43_primary_coordinate_order(
                        batch_table
                    )
                )


                player_ranking = (
                    _notebook43_player_coordinate_order(
                        batch_table
                    )
                )


                primary_winner = (
                    primary_ranking.iloc[
                        0
                    ]
                    .copy()
                )


                player_winner = (
                    player_ranking.iloc[
                        0
                    ]
                    .copy()
                )


                pair_identity = {

                    "variant_order":
                        int(
                            variant_order
                        ),

                    "variant_name":
                        str(
                            variant_name
                        ),

                    "pair_product_rank":
                        int(
                            pair_row[
                                "pair_product_rank"
                            ]
                        ),

                    "family_a_line_1":
                        int(
                            pair_row[
                                "family_a_line_1"
                            ]
                        ),

                    "family_a_line_2":
                        int(
                            pair_row[
                                "family_a_line_2"
                            ]
                        ),

                    "family_b_line_1":
                        int(
                            pair_row[
                                "family_b_line_1"
                            ]
                        ),

                    "family_b_line_2":
                        int(
                            pair_row[
                                "family_b_line_2"
                            ]
                        ),
                }


                primary_payload = dict(
                    primary_winner
                )


                primary_payload.update(
                    pair_identity
                )


                primary_payload[
                    "geometry_key"
                ] = (
                    _notebook43_geometry_key(
                        primary_payload
                    )
                )


                primary_payload[
                    "pair_context_key"
                ] = (
                    _notebook43_pair_context_key(

                        variant_name=
                            variant_name,

                        row=
                            primary_payload,
                    )
                )


                player_payload = dict(
                    player_winner
                )


                player_payload.update(
                    pair_identity
                )


                player_payload[
                    "geometry_key"
                ] = (
                    _notebook43_geometry_key(
                        player_payload
                    )
                )


                player_payload[
                    "pair_context_key"
                ] = (
                    _notebook43_pair_context_key(

                        variant_name=
                            variant_name,

                        row=
                            player_payload,
                    )
                )


                unique_winner_rows.append(
                    primary_payload
                )


                player_winner_rows.append(
                    player_payload
                )


                pair_summary_rows.append(
                    {
                        **pair_identity,

                        "coordinate_templates":
                            int(
                                len(
                                    batch_table
                                )
                            ),

                        "best_unique_template_index":
                            int(
                                primary_winner[
                                    "template_index"
                                ]
                            ),

                        "best_player_template_index":
                            int(
                                player_winner[
                                    "template_index"
                                ]
                            ),

                        "search_status":
                            "SCORED",
                    }
                )


            if progress:

                print(
                    f"Successful : {successful_this_variant}"
                )

                print(
                    f"Failed     : {failed_this_variant}"
                )


        pair_summary = pd.DataFrame(
            pair_summary_rows
        )


        unique_winners = pd.DataFrame(
            unique_winner_rows
        )


        player_winners = pd.DataFrame(
            player_winner_rows
        )


        if (
            len(
                unique_winners
            )
            !=
            pair_products_successful
        ):

            raise RuntimeError(
                "Unique-winner conservation failed."
            )


        if (
            len(
                player_winners
            )
            !=
            pair_products_successful
        ):

            raise RuntimeError(
                "Player-winner conservation failed."
            )


        # --------------------------------------------------------------------------------------------
        # GLOBAL PRIMARY ORDER
        # --------------------------------------------------------------------------------------------

        global_primary = (
            unique_winners
            .sort_values(
                [
                    "batch_distinct_balance",
                    "batch_distinct_total",
                    "batch_duplicate_fraction",
                    "batch_median_distinct_residual_m",
                    "batch_median_distinct_axis_leak",
                    "variant_name",
                    "pair_product_rank",
                    "template_index",
                ],
                ascending=[
                    False,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
            )
            .reset_index(
                drop=True
            )
        )


        global_primary[
            "global_candidate_rank"
        ] = (
            np.arange(
                len(
                    global_primary
                )
            )
            +
            1
        )


        # --------------------------------------------------------------------------------------------
        # VARIANT PRIMARY CHAMPIONS
        # --------------------------------------------------------------------------------------------

        variant_champions = (
            global_primary
            .sort_values(
                [
                    "variant_name",
                    "global_candidate_rank",
                ]
            )
            .groupby(
                "variant_name",
                as_index=False,
                sort=False,
            )
            .head(
                1
            )
            .copy()
            .reset_index(
                drop=True
            )
        )


        # --------------------------------------------------------------------------------------------
        # PHASE A — FIND FIRST OPERATIONAL RETAINED CONTEXT
        # --------------------------------------------------------------------------------------------

        boundary_rows = []

        operational_bound_rank = None


        for _, retained_row in global_primary.iterrows():

            variant_name = str(
                retained_row[
                    "variant_name"
                ]
            )


            variant_payload = variants[
                variant_name
            ]


            pair_row = (
                _notebook43_pair_row_for_identity(

                    pair_table=
                        variant_payload[
                            "pair_product_table"
                        ],

                    candidate_row=
                        retained_row,
                )
            )


            batch_result = (
                score_pair_product_batched(

                    event_id=
                        variant_payload[
                            "runtime_event_id"
                        ],

                    pair_product_row=
                        pair_row,
                )
            )


            if batch_result is None:

                raise RuntimeError(
                    "Operational-bound rescore unexpectedly failed."
                )


            (
                batch_table,
                pair_geometry,
            ) = batch_result


            H_batch = (
                factorized_coordinate_homographies(

                    pair_geometry[
                        "H_uv"
                    ]
                )
            )


            template_index = int(
                retained_row[
                    "template_index"
                ]
            )


            validity = (
                full_search_candidate_validity(

                    H_batch=
                        H_batch,

                    template_index=
                        template_index,

                    frame_width=
                        frame_width,

                    frame_height=
                        frame_height,
                )
            )


            H = normalize_homography_matrix(
                H_batch[
                    template_index
                ]
            )


            operational_checks = (
                _notebook43_operational_candidate_checks(

                    H=
                        H,

                    player_table=
                        player_table,

                    shooter_image_xy=
                        shooter_image_xy,
                )
            )


            operationally_acceptable = bool(

                validity[
                    "valid"
                ]

                and

                operational_checks[
                    "all_scoring_players_finite"
                ]

                and

                operational_checks[
                    "shooter_inside_pitch"
                ]
            )


            boundary_rows.append(
                {
                    "global_candidate_rank":
                        int(
                            retained_row[
                                "global_candidate_rank"
                            ]
                        ),

                    "variant_name":
                        variant_name,

                    "pair_context_key":
                        str(
                            retained_row[
                                "pair_context_key"
                            ]
                        ),

                    "geometry_key":
                        str(
                            retained_row[
                                "geometry_key"
                            ]
                        ),

                    "template_index":
                        template_index,

                    "homography_valid":
                        bool(
                            validity[
                                "valid"
                            ]
                        ),

                    "all_scoring_players_finite":
                        bool(
                            operational_checks[
                                "all_scoring_players_finite"
                            ]
                        ),

                    "shooter_pitch_x":
                        float(
                            operational_checks[
                                "shooter_pitch_x"
                            ]
                        ),

                    "shooter_pitch_y":
                        float(
                            operational_checks[
                                "shooter_pitch_y"
                            ]
                        ),

                    "shooter_inside_pitch":
                        bool(
                            operational_checks[
                                "shooter_inside_pitch"
                            ]
                        ),

                    "operationally_acceptable":
                        operationally_acceptable,
                }
            )


            if operationally_acceptable:

                operational_bound_rank = int(
                    retained_row[
                        "global_candidate_rank"
                    ]
                )


                break


        if operational_bound_rank is None:

            raise RuntimeError(
                "No operational retained homography context was found."
            )


        recovery_boundary = pd.DataFrame(
            boundary_rows
        )


        # --------------------------------------------------------------------------------------------
        # LEADING UNIQUE PAIR CONTEXTS THROUGH OPERATIONAL BOUND
        # --------------------------------------------------------------------------------------------

        leading_contexts = (
            global_primary[
                global_primary[
                    "global_candidate_rank"
                ]
                <=
                operational_bound_rank
            ]
            .copy()
            .sort_values(
                "global_candidate_rank"
            )
            .drop_duplicates(
                subset=[
                    "pair_context_key",
                ],
                keep="first",
            )
            .reset_index(
                drop=True
            )
        )


        # --------------------------------------------------------------------------------------------
        # PHASE B — FULL OPERATIONAL RECOVERY INSIDE EACH LEADING CONTEXT
        # --------------------------------------------------------------------------------------------

        operational_recovery_rows = []


        for _, context_row in leading_contexts.iterrows():

            variant_name = str(
                context_row[
                    "variant_name"
                ]
            )


            variant_payload = variants[
                variant_name
            ]


            pair_row = (
                _notebook43_pair_row_for_identity(

                    pair_table=
                        variant_payload[
                            "pair_product_table"
                        ],

                    candidate_row=
                        context_row,
                )
            )


            batch_result = (
                score_pair_product_batched(

                    event_id=
                        variant_payload[
                            "runtime_event_id"
                        ],

                    pair_product_row=
                        pair_row,
                )
            )


            if batch_result is None:

                raise RuntimeError(
                    "Operational pair recovery rescore failed."
                )


            (
                batch_table,
                pair_geometry,
            ) = batch_result


            H_batch = (
                factorized_coordinate_homographies(

                    pair_geometry[
                        "H_uv"
                    ]
                )
            )


            primary_ranking = (
                _notebook43_primary_coordinate_order(
                    batch_table
                )
            )


            recovered_row = None


            for _, candidate in primary_ranking.iterrows():

                template_index = int(
                    candidate[
                        "template_index"
                    ]
                )


                validity = (
                    full_search_candidate_validity(

                        H_batch=
                            H_batch,

                        template_index=
                            template_index,

                        frame_width=
                            frame_width,

                        frame_height=
                            frame_height,
                    )
                )


                if not bool(
                    validity[
                        "valid"
                    ]
                ):

                    continue


                H = normalize_homography_matrix(
                    H_batch[
                        template_index
                    ]
                )


                checks = (
                    _notebook43_operational_candidate_checks(

                        H=
                            H,

                        player_table=
                            player_table,

                        shooter_image_xy=
                            shooter_image_xy,
                    )
                )


                if not bool(
                    checks[
                        "all_scoring_players_finite"
                    ]
                ):

                    continue


                if not bool(
                    checks[
                        "shooter_inside_pitch"
                    ]
                ):

                    continue


                recovered_row = dict(
                    candidate
                )


                recovered_row.update(
                    {
                        "variant_name":
                            variant_name,

                        "retained_global_rank":
                            int(
                                context_row[
                                    "global_candidate_rank"
                                ]
                            ),

                        "pair_context_key":
                            str(
                                context_row[
                                    "pair_context_key"
                                ]
                            ),

                        "family_a_line_1":
                            int(
                                pair_row[
                                    "family_a_line_1"
                                ]
                            ),

                        "family_a_line_2":
                            int(
                                pair_row[
                                    "family_a_line_2"
                                ]
                            ),

                        "family_b_line_1":
                            int(
                                pair_row[
                                    "family_b_line_1"
                                ]
                            ),

                        "family_b_line_2":
                            int(
                                pair_row[
                                    "family_b_line_2"
                                ]
                            ),

                        "geometry_key":
                            (
                                _notebook43_geometry_key(
                                    {
                                        **dict(
                                            candidate
                                        ),

                                        "family_a_line_1":
                                            int(
                                                pair_row[
                                                    "family_a_line_1"
                                                ]
                                            ),

                                        "family_a_line_2":
                                            int(
                                                pair_row[
                                                    "family_a_line_2"
                                                ]
                                            ),

                                        "family_b_line_1":
                                            int(
                                                pair_row[
                                                    "family_b_line_1"
                                                ]
                                            ),

                                        "family_b_line_2":
                                            int(
                                                pair_row[
                                                    "family_b_line_2"
                                                ]
                                            ),
                                    }
                                )
                            ),

                        "homography_valid":
                            True,

                        "h_condition_number":
                            float(
                                validity[
                                    "condition_number"
                                ]
                            ),

                        "maximum_abs_anchor_m":
                            float(
                                validity[
                                    "maximum_abs_anchor_m"
                                ]
                            ),

                        "all_scoring_players_finite":
                            bool(
                                checks[
                                    "all_scoring_players_finite"
                                ]
                            ),

                        "shooter_pitch_x":
                            float(
                                checks[
                                    "shooter_pitch_x"
                                ]
                            ),

                        "shooter_pitch_y":
                            float(
                                checks[
                                    "shooter_pitch_y"
                                ]
                            ),

                        "shooter_inside_pitch":
                            bool(
                                checks[
                                    "shooter_inside_pitch"
                                ]
                            ),

                        "operational_template_found":
                            True,

                        "_selected_H":
                            H,
                    }
                )


                break


            if recovered_row is None:

                operational_recovery_rows.append(
                    {
                        "variant_name":
                            variant_name,

                        "retained_global_rank":
                            int(
                                context_row[
                                    "global_candidate_rank"
                                ]
                            ),

                        "pair_context_key":
                            str(
                                context_row[
                                    "pair_context_key"
                                ]
                            ),

                        "operational_template_found":
                            False,

                        "_selected_H":
                            None,
                    }
                )


            else:

                operational_recovery_rows.append(
                    recovered_row
                )


        successful_recoveries = [

            row

            for row
            in operational_recovery_rows

            if bool(
                row.get(
                    "operational_template_found",
                    False,
                )
            )
        ]


        if not successful_recoveries:

            raise RuntimeError(
                "No operational homography was recovered."
            )


        # --------------------------------------------------------------------------------------------
        # OPERATIONAL PRIMARY ORDER
        # --------------------------------------------------------------------------------------------

        operational_primary = pd.DataFrame(
            [
                {
                    key:
                        value

                    for key, value
                    in row.items()

                    if key
                    !=
                    "_selected_H"
                }

                for row
                in successful_recoveries
            ]
        )


        operational_primary = (
            operational_primary
            .sort_values(
                [
                    "batch_distinct_balance",
                    "batch_distinct_total",
                    "batch_duplicate_fraction",
                    "batch_median_distinct_residual_m",
                    "batch_median_distinct_axis_leak",
                    "variant_name",
                    "within_pair_primary_rank",
                    "template_index",
                ],
                ascending=[
                    False,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
            )
            .reset_index(
                drop=True
            )
        )


        operational_primary[
            "operational_primary_rank"
        ] = (
            np.arange(
                len(
                    operational_primary
                )
            )
            +
            1
        )


        selected_public_row = (
            operational_primary.iloc[
                0
            ]
        )


        selected_internal = next(

            row

            for row
            in successful_recoveries

            if (
                str(
                    row[
                        "variant_name"
                    ]
                )
                ==
                str(
                    selected_public_row[
                        "variant_name"
                    ]
                )
                and
                str(
                    row[
                        "geometry_key"
                    ]
                )
                ==
                str(
                    selected_public_row[
                        "geometry_key"
                    ]
                )
            )
        )


        selected_H = normalize_homography_matrix(
            selected_internal[
                "_selected_H"
            ]
        )


        # --------------------------------------------------------------------------------------------
        # FINAL OPERATIONAL ACCEPTANCE AUDIT
        # --------------------------------------------------------------------------------------------

        selected_checks = (
            _notebook43_operational_candidate_checks(

                H=
                    selected_H,

                player_table=
                    player_table,

                shooter_image_xy=
                    shooter_image_xy,
            )
        )


        selected_pair_row = (
            _notebook43_pair_row_for_identity(

                pair_table=
                    variants[
                        str(
                            selected_public_row[
                                "variant_name"
                            ]
                        )
                    ][
                        "pair_product_table"
                    ],

                candidate_row=
                    selected_public_row,
            )
        )


        selected_batch_result = (
            score_pair_product_batched(

                event_id=
                    variants[
                        str(
                            selected_public_row[
                                "variant_name"
                            ]
                        )
                    ][
                        "runtime_event_id"
                    ],

                pair_product_row=
                    selected_pair_row,
            )
        )


        selected_batch_table = (
            selected_batch_result[
                0
            ]
        )


        selected_score_row = (
            selected_batch_table[
                selected_batch_table[
                    "template_index"
                ]
                .astype(int)
                ==
                int(
                    selected_public_row[
                        "template_index"
                    ]
                )
            ]
        )


        if len(
            selected_score_row
        ) != 1:

            raise RuntimeError(
                "Could not recover exactly one selected score row."
            )


        selected_score_row = (
            selected_score_row.iloc[
                0
            ]
        )


        # --------------------------------------------------------------------------------------------
        # SAVE IN RUNTIME METADATA
        # --------------------------------------------------------------------------------------------

        result.metadata[
            "homography_search"
        ] = {

            "method":
                "FULL_FRESH_FACTORIZED_ORACLE_FREE_SEARCH_WITH_AUTOMATIC_OPERATIONAL_RECOVERY",

            "pair_products_attempted":
                int(
                    pair_products_attempted
                ),

            "pair_products_successful":
                int(
                    pair_products_successful
                ),

            "pair_geometry_failures":
                int(
                    pair_geometry_failures
                ),

            "template_rows_evaluated":
                int(
                    template_rows_evaluated
                ),

            "unique_winner_count":
                int(
                    len(
                        unique_winners
                    )
                ),

            "player_winner_count":
                int(
                    len(
                        player_winners
                    )
                ),

            "provisional_shooter":
                {
                    "track_id":
                        int(
                            provisional_shooter[
                                "track_id"
                            ]
                        ),

                    "frame_idx":
                        int(
                            provisional_shooter[
                                "frame_idx"
                            ]
                        ),

                    "image_xy":
                        list(
                            provisional_shooter[
                                "image_xy"
                            ]
                        ),

                    "pixel_distance":
                        float(
                            provisional_shooter[
                                "pixel_distance"
                            ]
                        ),

                    "normalized_distance":
                        float(
                            provisional_shooter[
                                "normalized_distance"
                            ]
                        ),

                    "used_for_primary_ranking":
                        False,

                    "used_for_operational_acceptance":
                        True,
                },

            "operational_recovery_bound_rank":
                int(
                    operational_bound_rank
                ),

            "leading_context_count":
                int(
                    len(
                        leading_contexts
                    )
                ),

            "operational_recovery_count":
                int(
                    len(
                        successful_recoveries
                    )
                ),

            "selected_variant":
                str(
                    selected_public_row[
                        "variant_name"
                    ]
                ),

            "selected_pair_context":
                str(
                    selected_public_row[
                        "pair_context_key"
                    ]
                ),

            "selected_geometry_key":
                str(
                    selected_public_row[
                        "geometry_key"
                    ]
                ),

            "selected_template_index":
                int(
                    selected_public_row[
                        "template_index"
                    ]
                ),

            "selected_homography":
                selected_H.tolist(),

            "selected_metrics":
                {
                    "batch_distinct_balance":
                        float(
                            selected_score_row[
                                "batch_distinct_balance"
                            ]
                        ),

                    "batch_distinct_total":
                        float(
                            selected_score_row[
                                "batch_distinct_total"
                            ]
                        ),

                    "batch_duplicate_fraction":
                        float(
                            selected_score_row[
                                "batch_duplicate_fraction"
                            ]
                        ),

                    "batch_median_distinct_residual_m":
                        float(
                            selected_score_row[
                                "batch_median_distinct_residual_m"
                            ]
                        ),

                    "batch_median_distinct_axis_leak":
                        float(
                            selected_score_row[
                                "batch_median_distinct_axis_leak"
                            ]
                        ),

                    "batch_player_count":
                        int(
                            selected_score_row[
                                "batch_player_count"
                            ]
                        ),

                    "batch_finite_player_count":
                        int(
                            selected_score_row[
                                "batch_finite_player_count"
                            ]
                        ),

                    "batch_inside_pitch_count":
                        int(
                            selected_score_row[
                                "batch_inside_pitch_count"
                            ]
                        ),

                    "batch_inside_pitch_fraction":
                        float(
                            selected_score_row[
                                "batch_inside_pitch_fraction"
                            ]
                        ),
                },

            "selected_acceptance":
                {
                    "all_scoring_players_finite":
                        bool(
                            selected_checks[
                                "all_scoring_players_finite"
                            ]
                        ),

                    "inside_pitch_count":
                        int(
                            selected_checks[
                                "inside_pitch_count"
                            ]
                        ),

                    "inside_pitch_fraction":
                        float(
                            selected_checks[
                                "inside_pitch_fraction"
                            ]
                        ),

                    "shooter_pitch_x":
                        float(
                            selected_checks[
                                "shooter_pitch_x"
                            ]
                        ),

                    "shooter_pitch_y":
                        float(
                            selected_checks[
                                "shooter_pitch_y"
                            ]
                        ),

                    "shooter_projection_finite":
                        bool(
                            selected_checks[
                                "shooter_projection_finite"
                            ]
                        ),

                    "shooter_inside_pitch":
                        bool(
                            selected_checks[
                                "shooter_inside_pitch"
                            ]
                        ),

                    "relative_teams_used":
                        False,
                },

            # Notebook driver can persist these compact search results.
            "_pair_summary":
                pair_summary,

            "_unique_winners":
                unique_winners,

            "_player_winners":
                player_winners,

            "_variant_champions":
                variant_champions,

            "_global_primary":
                global_primary,

            "_recovery_boundary":
                recovery_boundary,

            "_leading_contexts":
                leading_contexts,

            "_operational_recovery_rows":
                operational_recovery_rows,

            "_operational_primary":
                operational_primary,
        }


    return homography_search_handler


def register_homography_search_handler(
    pipeline,
    progress=True,
):
    """Register canonical full Stage-8 homography search."""

    pipeline.register_stage_handler(

        "homography_search",

        build_homography_search_handler(
            progress=progress
        ),
    )


    return pipeline



# NOTEBOOK43_CANONICAL_HOMOGRAPHY_VALIDATION_V1
# ====================================================================================================
# CANONICAL HOMOGRAPHY VALIDATION
# ====================================================================================================


def _notebook43_raw_c_float64_sha256(
    matrix,
):
    """Stable numeric identity for one float64 3x3 homography."""

    # Local dependency keeps this helper self-contained under isolated import.
    import hashlib

    array = np.ascontiguousarray(
        matrix,
        dtype=np.float64,
    )


    return hashlib.sha256(
        array.tobytes(
            order="C"
        )
    ).hexdigest()


def _notebook43_validate_homography_matrix(
    H,
    frame_width,
    frame_height,
):
    """
    Independently apply the canonical reference-free validity gate.

    No historical H or projected coordinates are accepted as inputs.
    """

    # --------------------------------------------------------------------------------------------
    # Cell-13 static homography constants/templates are intentionally lazy.
    #
    # Validation depends on constants including:
    #
    #     FULL_SEARCH_MAX_CONDITION_NUMBER
    #     FULL_SEARCH_MAX_ABS_ANCHOR_M
    #
    # Therefore initialize the static runtime explicitly here.
    #
    # This performs NO homography search and NO candidate scoring.
    # --------------------------------------------------------------------------------------------

    initialize_homography_static_runtime()

    matrix = np.asarray(
        H,
        dtype=np.float64,
    )


    basic_shape_valid = (
        matrix.shape
        ==
        (
            3,
            3,
        )
    )


    if not basic_shape_valid:

        return {
            "valid":
                False,

            "shape_valid":
                False,

            "finite":
                False,

            "determinant":
                np.nan,

            "condition_number":
                np.inf,

            "maximum_abs_anchor_m":
                np.inf,

            "raw_c_float64_sha256":
                None,

            "failure_reason":
                "HOMOGRAPHY_SHAPE_INVALID",
        }


    finite = bool(
        np.isfinite(
            matrix
        ).all()
    )


    if not finite:

        return {
            "valid":
                False,

            "shape_valid":
                True,

            "finite":
                False,

            "determinant":
                np.nan,

            "condition_number":
                np.inf,

            "maximum_abs_anchor_m":
                np.inf,

            "raw_c_float64_sha256":
                None,

            "failure_reason":
                "HOMOGRAPHY_NONFINITE",
        }


    determinant = float(
        np.linalg.det(
            matrix
        )
    )


    condition_number = float(
        np.linalg.cond(
            matrix
        )
    )


    # Use the exact packaged canonical validity implementation.
    validation_payload = (
        full_search_candidate_validity(

            H_batch=
                matrix[
                    None,
                    :,
                    :,
                ],

            template_index=
                0,

            frame_width=
                int(
                    frame_width
                ),

            frame_height=
                int(
                    frame_height
                ),
        )
    )


    valid = bool(
        validation_payload[
            "valid"
        ]
    )


    maximum_abs_anchor_m = float(
        validation_payload[
            "maximum_abs_anchor_m"
        ]
    )


    canonical_condition_number = float(
        validation_payload[
            "condition_number"
        ]
    )


    return {

        "valid":
            valid,

        "shape_valid":
            True,

        "finite":
            finite,

        "determinant":
            determinant,

        "condition_number":
            canonical_condition_number,

        "maximum_abs_anchor_m":
            maximum_abs_anchor_m,

        "raw_c_float64_sha256":
            _notebook43_raw_c_float64_sha256(
                matrix
            ),

        "failure_reason":
            None
            if valid
            else
            "CANONICAL_HOMOGRAPHY_VALIDITY_GATE_FAILED",
    }


def build_homography_validation_handler():
    """Build canonical Stage-9 homography validation."""

    def homography_validation_handler(
        result,
    ):

        expected_prefix = list(
            CANONICAL_PIPELINE_STAGES[
                : _STAGE_INDEX[
                    "homography_validation"
                ]
            ]
        )


        if (
            result.pipeline_status.completed_stages
            !=
            expected_prefix
        ):

            raise PipelineContractError(
                "Homography validation requires completed homography_search."
            )


        search_metadata = result.metadata.get(
            "homography_search"
        )


        if not isinstance(
            search_metadata,
            dict,
        ):

            raise PipelineContractError(
                "homography_search metadata is missing."
            )


        selected_homography = search_metadata.get(
            "selected_homography"
        )


        if selected_homography is None:

            raise PipelineContractError(
                "homography_search did not publish selected_homography."
            )


        pitch_metadata = result.metadata.get(
            "pitch_line_detection"
        )


        if not isinstance(
            pitch_metadata,
            dict,
        ):

            raise PipelineContractError(
                "pitch_line_detection metadata is missing."
            )


        frame_width = int(
            pitch_metadata[
                "frame_width"
            ]
        )


        frame_height = int(
            pitch_metadata[
                "frame_height"
            ]
        )


        H = np.asarray(
            selected_homography,
            dtype=np.float64,
        )


        validation_payload = (
            _notebook43_validate_homography_matrix(

                H=
                    H,

                frame_width=
                    frame_width,

                frame_height=
                    frame_height,
            )
        )


        if not bool(
            validation_payload[
                "valid"
            ]
        ):

            raise RuntimeError(
                "Selected homography failed canonical independent validation."
            )


        # --------------------------------------------------------------------------------------------
        # Preserve the search-selected H exactly.
        # --------------------------------------------------------------------------------------------

        result.metadata[
            "homography_validation"
        ] = {

            "success":
                True,

            "method":
                "INDEPENDENT_REFERENCE_FREE_CANONICAL_VALIDATION",

            "selected_variant":
                str(
                    search_metadata[
                        "selected_variant"
                    ]
                ),

            "selected_pair_context":
                str(
                    search_metadata[
                        "selected_pair_context"
                    ]
                ),

            "selected_geometry_key":
                str(
                    search_metadata[
                        "selected_geometry_key"
                    ]
                ),

            "selected_template_index":
                int(
                    search_metadata[
                        "selected_template_index"
                    ]
                ),

            "homography":
                H.tolist(),

            "shape":
                [
                    3,
                    3,
                ],

            "finite":
                bool(
                    validation_payload[
                        "finite"
                    ]
                ),

            "determinant":
                float(
                    validation_payload[
                        "determinant"
                    ]
                ),

            "condition_number":
                float(
                    validation_payload[
                        "condition_number"
                    ]
                ),

            "maximum_abs_anchor_m":
                float(
                    validation_payload[
                        "maximum_abs_anchor_m"
                    ]
                ),

            "raw_c_float64_sha256":
                str(
                    validation_payload[
                        "raw_c_float64_sha256"
                    ]
                ),

            "validation_status":
                "VALID",

            "failure_reason":
                None,

            "reference_homography_used":
                False,

            "historical_projection_used":
                False,

            "homography_reselected":
                False,

            "pitch_projection_performed":
                False,
        }


        return validation_payload


    return homography_validation_handler


def register_homography_validation_handler(
    pipeline,
):
    """Register canonical Stage-9 homography validation."""

    pipeline.register_stage_handler(

        "homography_validation",

        build_homography_validation_handler(),
    )


    return pipeline

# NOTEBOOK43_CANONICAL_HOMOGRAPHY_VALIDATION_FIX1_LAZY_RUNTIME

# NOTEBOOK43_CANONICAL_HOMOGRAPHY_VALIDATION_FIX2_HASHLIB



# NOTEBOOK43_CANONICAL_PITCH_PROJECTION_V1
# ====================================================================================================
# CANONICAL PITCH PROJECTION
# ====================================================================================================


def _notebook43_project_player_table(
    result,
    H,
    geometry_frame,
    frame_height,
):
    """
    Project current geometry-frame player bottom-center points.

    Inputs:
        current PlayerTrackResult
        current validated homography
        dynamically resolved geometry frame

    No historical projected coordinates are accepted.
    """

    player_table = (
        _notebook43_build_exact_frame_player_table(

            result=
                result,

            geometry_frame=
                int(
                    geometry_frame
                ),

            frame_height=
                int(
                    frame_height
                ),
        )
    )


    if len(
        player_table
    ) == 0:

        raise RuntimeError(
            "Pitch projection received zero current geometry-frame players."
        )


    projected_rows = []


    for _, player in player_table.iterrows():

        pitch_x, pitch_y = (
            _notebook43_project_xy(

                H=
                    H,

                x=
                    float(
                        player[
                            "bottom_center_x"
                        ]
                    ),

                y=
                    float(
                        player[
                            "bottom_center_y"
                        ]
                    ),
            )
        )


        projection_finite = bool(
            np.isfinite(
                np.asarray(
                    [
                        pitch_x,
                        pitch_y,
                    ],
                    dtype=float,
                )
            ).all()
        )


        inside_pitch = bool(
            projection_finite
            and
            0.0
            <=
            float(
                pitch_x
            )
            <=
            105.0
            and
            0.0
            <=
            float(
                pitch_y
            )
            <=
            68.0
        )


        projected_rows.append(
            {
                "frame_number":
                    int(
                        player[
                            "frame_number"
                        ]
                    ),

                "track_id":
                    int(
                        player[
                            "track_id"
                        ]
                    ),

                "detection_id":
                    int(
                        player[
                            "detection_id"
                        ]
                    ),

                "confidence":
                    float(
                        player[
                            "confidence"
                        ]
                    ),

                "bottom_center_x":
                    float(
                        player[
                            "bottom_center_x"
                        ]
                    ),

                "bottom_center_y":
                    float(
                        player[
                            "bottom_center_y"
                        ]
                    ),

                "pitch_x":
                    float(
                        pitch_x
                    ),

                "pitch_y":
                    float(
                        pitch_y
                    ),

                "inside_pitch":
                    inside_pitch,
            }
        )


    projected_table = (
        pd.DataFrame(
            projected_rows
        )
        .sort_values(
            [
                "track_id",
                "detection_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    return projected_table


def build_pitch_projection_handler():
    """Build canonical Stage-10 pitch projection."""

    def pitch_projection_handler(
        result,
    ):

        expected_prefix = list(
            CANONICAL_PIPELINE_STAGES[
                : _STAGE_INDEX[
                    "pitch_projection"
                ]
            ]
        )


        if (
            result.pipeline_status.completed_stages
            !=
            expected_prefix
        ):

            raise PipelineContractError(
                "Pitch projection requires completed homography_validation."
            )


        validation_metadata = result.metadata.get(
            "homography_validation"
        )


        if not isinstance(
            validation_metadata,
            dict,
        ):

            raise PipelineContractError(
                "homography_validation metadata is missing."
            )


        if (
            validation_metadata.get(
                "success"
            )
            is not True
            or
            validation_metadata.get(
                "validation_status"
            )
            !=
            "VALID"
        ):

            raise PipelineContractError(
                "Pitch projection requires a VALID homography."
            )


        selected_homography = (
            validation_metadata.get(
                "homography"
            )
        )


        if selected_homography is None:

            raise PipelineContractError(
                "Validated homography matrix is missing."
            )


        pitch_metadata = result.metadata.get(
            "pitch_line_detection"
        )


        if not isinstance(
            pitch_metadata,
            dict,
        ):

            raise PipelineContractError(
                "pitch_line_detection metadata is missing."
            )


        geometry_frame = int(
            pitch_metadata[
                "frame_idx"
            ]
        )


        frame_height = int(
            pitch_metadata[
                "frame_height"
            ]
        )


        H = np.asarray(
            selected_homography,
            dtype=np.float64,
        )


        if (
            H.shape
            !=
            (
                3,
                3,
            )
        ):

            raise RuntimeError(
                "Validated homography has invalid shape."
            )


        if not np.isfinite(
            H
        ).all():

            raise RuntimeError(
                "Validated homography contains non-finite values."
            )


        projected_table = (
            _notebook43_project_player_table(

                result=
                    result,

                H=
                    H,

                geometry_frame=
                    geometry_frame,

                frame_height=
                    frame_height,
            )
        )


        all_projections_finite = bool(
            np.isfinite(
                projected_table[
                    [
                        "pitch_x",
                        "pitch_y",
                    ]
                ]
                .to_numpy(
                    dtype=float
                )
            )
            .all()
        )


        if not all_projections_finite:

            raise RuntimeError(
                "Pitch projection produced non-finite player coordinates."
            )


        players_projected = int(
            len(
                projected_table
            )
        )


        players_inside_pitch = int(
            projected_table[
                "inside_pitch"
            ]
            .sum()
        )


        inside_pitch_fraction = float(
            players_inside_pitch
            /
            max(
                players_projected,
                1,
            )
        )


        projected_records = (
            projected_table
            .to_dict(
                orient="records"
            )
        )


        # --------------------------------------------------------------------------------------------
        # Populate PUBLIC PitchProjectionResult.
        # --------------------------------------------------------------------------------------------

        result.pitch_projection.success = True


        result.pitch_projection.homography = (
            H.tolist()
        )


        result.pitch_projection.projected_players = (
            projected_records
        )


        result.pitch_projection.players_projected = (
            players_projected
        )


        result.pitch_projection.players_inside_pitch = (
            players_inside_pitch
        )


        result.pitch_projection.validation_status = (
            "VALID"
        )


        result.pitch_projection.failure_reason = (
            None
        )


        # --------------------------------------------------------------------------------------------
        # Runtime metadata.
        # --------------------------------------------------------------------------------------------

        result.metadata[
            "pitch_projection"
        ] = {

            "success":
                True,

            "method":
                "FRESH_GEOMETRY_FRAME_BOTTOM_CENTER_PROJECTION",

            "geometry_frame":
                int(
                    geometry_frame
                ),

            "pitch_length_m":
                105.0,

            "pitch_width_m":
                68.0,

            "players_projected":
                players_projected,

            "players_inside_pitch":
                players_inside_pitch,

            "inside_pitch_fraction":
                inside_pitch_fraction,

            "all_projections_finite":
                all_projections_finite,

            "validated_homography_used":
                True,

            "historical_projected_coordinates_used":
                False,

            "reference_homography_used":
                False,

            "manual_projection_used":
                False,

            "shooter_assignment_performed":
                False,

            "relative_team_assignment_performed":
                False,
        }


        return projected_table


    return pitch_projection_handler


def register_pitch_projection_handler(
    pipeline,
):
    """Register canonical Stage-10 pitch projection."""

    pipeline.register_stage_handler(

        "pitch_projection",

        build_pitch_projection_handler(),
    )


    return pipeline



# NOTEBOOK43_CANONICAL_TEMPORAL_SHOOTER_ASSIGNMENT_V1
# ====================================================================================================
# CANONICAL TEMPORAL SHOOTER ASSIGNMENT
# ====================================================================================================


def _notebook43_ball_to_bbox_distance(
    ball_x,
    ball_y,
    observation,
):
    """
    Euclidean point-to-axis-aligned-bbox distance.

    Distance is zero when the ball point lies inside the player box.
    """

    import math


    x1 = float(
        observation.x1
    )

    y1 = float(
        observation.y1
    )

    x2 = float(
        observation.x2
    )

    y2 = float(
        observation.y2
    )


    dx = max(
        x1
        -
        float(
            ball_x
        ),
        0.0,
        float(
            ball_x
        )
        -
        x2,
    )


    dy = max(
        y1
        -
        float(
            ball_y
        ),
        0.0,
        float(
            ball_y
        )
        -
        y2,
    )


    return float(
        math.hypot(
            dx,
            dy,
        )
    )




def _notebook43_record_value(
    record,
    key,
    default=None,
):
    """Read one key from mapping/dataclass/object records."""

    if isinstance(
        record,
        dict,
    ):

        return record.get(
            key,
            default,
        )


    if hasattr(
        record,
        key,
    ):

        return getattr(
            record,
            key,
        )


    return default


def _notebook43_flatten_detection_records(
    value,
    inherited_frame=None,
):
    """
    Flatten public PlayerTrackResult.detections into detection-like records.

    Supports:
        list/tuple containers
        frame-keyed mappings
        row mappings
        dataclass/object rows
    """

    rows = []


    if value is None:

        return rows


    # --------------------------------------------------------------------------------------------
    # Mapping may itself be one detection row.
    # --------------------------------------------------------------------------------------------

    if isinstance(
        value,
        dict,
    ):

        bbox_keys = {
            "x1",
            "y1",
            "x2",
            "y2",
        }


        if bbox_keys.issubset(
            value.keys()
        ):

            frame_value = (

                value.get(
                    "frame_idx"
                )

                if value.get(
                    "frame_idx"
                )
                is not None

                else value.get(
                    "frame_number",
                    inherited_frame,
                )
            )


            rows.append(
                (
                    value,
                    frame_value,
                )
            )


            return rows


        # Otherwise recurse through mapping values.
        for key, item in value.items():

            child_frame = inherited_frame


            try:

                if isinstance(
                    key,
                    (
                        int,
                        np.integer,
                    ),
                ):

                    child_frame = int(
                        key
                    )


                elif (
                    isinstance(
                        key,
                        str,
                    )
                    and
                    key.strip().lstrip("-").isdigit()
                ):

                    child_frame = int(
                        key
                    )


            except Exception:

                pass


            rows.extend(
                _notebook43_flatten_detection_records(

                    item,

                    inherited_frame=
                        child_frame,
                )
            )


        return rows


    # --------------------------------------------------------------------------------------------
    # Sequence container.
    # --------------------------------------------------------------------------------------------

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):

        for item in value:

            rows.extend(
                _notebook43_flatten_detection_records(

                    item,

                    inherited_frame=
                        inherited_frame,
                )
            )


        return rows


    # --------------------------------------------------------------------------------------------
    # Dataclass/object detection row.
    # --------------------------------------------------------------------------------------------

    if all(
        hasattr(
            value,
            key,
        )

        for key
        in [
            "x1",
            "y1",
            "x2",
            "y2",
        ]
    ):

        frame_value = (

            getattr(
                value,
                "frame_idx",
                None,
            )
        )


        if frame_value is None:

            frame_value = getattr(
                value,
                "frame_number",
                inherited_frame,
            )


        rows.append(
            (
                value,
                frame_value,
            )
        )


    return rows


def _notebook43_resolve_observation_detection_id(
    result,
    observation,
):
    """
    Recover original fresh detection_id for one PlayerTrackObservation.

    Match contract:
        same frame
        same bbox geometry

    Historical artifacts are not consulted.
    """

    frame_idx = int(
        observation.frame_idx
    )


    target_bbox = np.asarray(
        [
            float(
                observation.x1
            ),

            float(
                observation.y1
            ),

            float(
                observation.x2
            ),

            float(
                observation.y2
            ),
        ],
        dtype=float,
    )


    flattened = (
        _notebook43_flatten_detection_records(
            result.player_tracks.detections
        )
    )


    candidate_rows = []


    for record_index, (
        record,
        inherited_frame,
    ) in enumerate(
        flattened
    ):

        record_frame = (
            _notebook43_record_value(
                record,
                "frame_idx",
                None,
            )
        )


        if record_frame is None:

            record_frame = (
                _notebook43_record_value(
                    record,
                    "frame_number",
                    inherited_frame,
                )
            )


        if record_frame is None:

            continue


        if int(
            record_frame
        ) != frame_idx:

            continue


        try:

            bbox = np.asarray(
                [
                    float(
                        _notebook43_record_value(
                            record,
                            "x1",
                        )
                    ),

                    float(
                        _notebook43_record_value(
                            record,
                            "y1",
                        )
                    ),

                    float(
                        _notebook43_record_value(
                            record,
                            "x2",
                        )
                    ),

                    float(
                        _notebook43_record_value(
                            record,
                            "y2",
                        )
                    ),
                ],
                dtype=float,
            )


        except Exception:

            continue


        maximum_bbox_error = float(
            np.max(
                np.abs(
                    bbox
                    -
                    target_bbox
                )
            )
        )


        explicit_detection_id = None


        for key in [

            "detection_id",

            "detection_idx",

            "source_detection_id",

            "source_detection_index",

            "detection_index",

            "prediction_index",
        ]:

            value = (
                _notebook43_record_value(
                    record,
                    key,
                    None,
                )
            )


            if value is not None:

                try:

                    explicit_detection_id = int(
                        value
                    )


                    break


                except Exception:

                    pass


        candidate_rows.append(
            {
                "record_index":
                    int(
                        record_index
                    ),

                "maximum_bbox_error":
                    maximum_bbox_error,

                "explicit_detection_id":
                    explicit_detection_id,

                "record":
                    record,
            }
        )


    if len(
        candidate_rows
    ) == 0:

        raise RuntimeError(
            "Could not find any same-frame fresh detection for track observation."
        )


    candidate_rows = sorted(

        candidate_rows,

        key=lambda row:
            (
                row[
                    "maximum_bbox_error"
                ],

                (
                    row[
                        "explicit_detection_id"
                    ]
                    if row[
                        "explicit_detection_id"
                    ]
                    is not None
                    else
                    10**9
                ),

                row[
                    "record_index"
                ],
            ),
    )


    winner = (
        candidate_rows[
            0
        ]
    )


    # Canonical observation geometry was copied from the detection row;
    # this should therefore be effectively exact.
    if float(
        winner[
            "maximum_bbox_error"
        ]
    ) > 1e-6:

        raise RuntimeError(
            "Fresh detection join bbox mismatch exceeded 1e-6 px."
        )


    if (
        winner[
            "explicit_detection_id"
        ]
        is not None
    ):

        detection_id = int(
            winner[
                "explicit_detection_id"
            ]
        )


        resolution_method = (
            "EXPLICIT_ID_SAME_FRAME_EXACT_BBOX"
        )


    else:

        # ----------------------------------------------------------------------------------------
        # Defensive fallback:
        #
        # if public detection rows do not expose the ID field, assign the
        # deterministic ordinal among fresh same-frame detections using their
        # original flattened order.
        # ----------------------------------------------------------------------------------------

        same_frame_records = []


        for original_index, (
            record,
            inherited_frame,
        ) in enumerate(
            flattened
        ):

            record_frame = (
                _notebook43_record_value(
                    record,
                    "frame_idx",
                    None,
                )
            )


            if record_frame is None:

                record_frame = (
                    _notebook43_record_value(
                        record,
                        "frame_number",
                        inherited_frame,
                    )
                )


            if (
                record_frame is not None
                and
                int(
                    record_frame
                )
                ==
                frame_idx
            ):

                same_frame_records.append(
                    int(
                        original_index
                    )
                )


        winner_original_index = int(
            winner[
                "record_index"
            ]
        )


        if winner_original_index not in same_frame_records:

            raise RuntimeError(
                "Detection ordinal fallback could not resolve matched record."
            )


        detection_id = int(
            same_frame_records.index(
                winner_original_index
            )
        )


        resolution_method = (
            "SAME_FRAME_ORIGINAL_ORDER_FALLBACK"
        )


    return {

        "detection_id":
            detection_id,

        "resolution_method":
            resolution_method,

        "maximum_bbox_error":
            float(
                winner[
                    "maximum_bbox_error"
                ]
            ),
    }



def _notebook43_build_temporal_shooter_evidence(
    result,
):
    """
    Build complete ball-player temporal-distance evidence.

    No shooter ID, team label, pitch coordinate, homography reference,
    shot coordinate, or prior xG is accepted as input.
    """

    if not result.ball_track.success:

        raise PipelineContractError(
            "Shooter assignment requires successful ball tracking."
        )


    if not result.player_tracks.success:

        raise PipelineContractError(
            "Shooter assignment requires successful player tracking."
        )


    trajectory = result.ball_track.track


    if not isinstance(
        trajectory,
        list,
    ) or len(
        trajectory
    ) == 0:

        raise RuntimeError(
            "Shooter assignment received no operational ball trajectory."
        )


    trajectory = sorted(

        trajectory,

        key=lambda point:
            int(
                point.frame_idx
            ),
    )


    trajectory_frames = [

        int(
            point.frame_idx
        )

        for point
        in trajectory
    ]


    trajectory_frame_set = set(
        trajectory_frames
    )


    ball_by_frame = {

        int(
            point.frame_idx
        ):
            point

        for point
        in trajectory
    }


    evidence_rows = []

    summary_rows = []


    for track in result.player_tracks.tracks:

        track_id = int(
            track.track_id
        )


        observations = sorted(

            track.observations,

            key=lambda observation:
                int(
                    observation.frame_idx
                ),
        )


        observation_by_frame = {

            int(
                observation.frame_idx
            ):
                observation

            for observation
            in observations
        }


        overlap_frames = sorted(

            trajectory_frame_set
            &
            set(
                observation_by_frame.keys()
            )
        )


        starts_at_ball_start = bool(

            trajectory_frames[
                0
            ]
            in
            observation_by_frame
        )


        full_trajectory_coverage = bool(

            set(
                trajectory_frames
            )
            .issubset(
                observation_by_frame.keys()
            )
        )


        track_distance_rows = []


        for frame_idx in overlap_frames:

            ball_point = (
                ball_by_frame[
                    frame_idx
                ]
            )


            observation = (
                observation_by_frame[
                    frame_idx
                ]
            )


            distance_pixels = (
                _notebook43_ball_to_bbox_distance(

                    ball_x=
                        float(
                            ball_point.center_x
                        ),

                    ball_y=
                        float(
                            ball_point.center_y
                        ),

                    observation=
                        observation,
                )
            )


            box_height = float(
                observation.y2
                -
                observation.y1
            )


            if (
                not np.isfinite(
                    box_height
                )
                or
                box_height
                <=
                0.0
            ):

                raise RuntimeError(
                    "Shooter assignment encountered invalid player-box height."
                )


            distance_normalized = float(
                distance_pixels
                /
                box_height
            )


            row = {

                "track_id":
                    track_id,

                "frame_idx":
                    int(
                        frame_idx
                    ),

                "detection_id":
                    int(
                        _notebook43_resolve_observation_detection_id(
                            result,
                            observation,
                        )[
                            "detection_id"
                        ]
                    ),

                "detection_id_resolution_method":
                    str(
                        _notebook43_resolve_observation_detection_id(
                            result,
                            observation,
                        )[
                            "resolution_method"
                        ]
                    ),

                "detection_id_bbox_error":
                    float(
                        _notebook43_resolve_observation_detection_id(
                            result,
                            observation,
                        )[
                            "maximum_bbox_error"
                        ]
                    ),

                "confidence":
                    float(
                        observation.confidence
                    ),

                "trajectory_source":
                    str(
                        ball_point.trajectory_source
                    ),

                "ball_center_x":
                    float(
                        ball_point.center_x
                    ),

                "ball_center_y":
                    float(
                        ball_point.center_y
                    ),

                "x1":
                    float(
                        observation.x1
                    ),

                "y1":
                    float(
                        observation.y1
                    ),

                "x2":
                    float(
                        observation.x2
                    ),

                "y2":
                    float(
                        observation.y2
                    ),

                "bottom_center_x":
                    float(
                        observation.foot_x
                    ),

                "bottom_center_y":
                    float(
                        observation.foot_y
                    ),

                "player_box_height":
                    box_height,

                "ball_to_box_distance_pixels":
                    distance_pixels,

                "ball_to_box_distance_normalized":
                    distance_normalized,
            }


            evidence_rows.append(
                row
            )


            track_distance_rows.append(
                row
            )


        if len(
            track_distance_rows
        ) > 0:

            first_row = (
                track_distance_rows[
                    0
                ]
            )


            last_row = (
                track_distance_rows[
                    -1
                ]
            )


            pixel_distances = np.asarray(

                [
                    row[
                        "ball_to_box_distance_pixels"
                    ]

                    for row
                    in track_distance_rows
                ],

                dtype=float,
            )


            normalized_distances = np.asarray(

                [
                    row[
                        "ball_to_box_distance_normalized"
                    ]

                    for row
                    in track_distance_rows
                ],

                dtype=float,
            )


            if len(
                normalized_distances
            ) > 1:

                normalized_deltas = np.diff(
                    normalized_distances
                )


                normalized_nondec_fraction = float(
                    np.mean(
                        normalized_deltas
                        >=
                        -1e-12
                    )
                )


            else:

                normalized_nondec_fraction = np.nan


            first_distance_px = float(
                first_row[
                    "ball_to_box_distance_pixels"
                ]
            )


            first_distance_normalized = float(
                first_row[
                    "ball_to_box_distance_normalized"
                ]
            )


            minimum_distance_px = float(
                np.min(
                    pixel_distances
                )
            )


            last_distance_px = float(
                last_row[
                    "ball_to_box_distance_pixels"
                ]
            )


            last_distance_normalized = float(
                last_row[
                    "ball_to_box_distance_normalized"
                ]
            )


            early_window_size = int(
                min(
                    2,
                    len(
                        normalized_distances
                    ),
                )
            )


            late_window_size = int(
                min(
                    2,
                    len(
                        normalized_distances
                    ),
                )
            )


            early_mean_distance_normalized = float(
                np.mean(
                    normalized_distances[
                        :
                        early_window_size
                    ]
                )
            )


            late_mean_distance_normalized = float(
                np.mean(
                    normalized_distances[
                        -
                        late_window_size
                        :
                    ]
                )
            )


            separation_gain_normalized = float(
                late_mean_distance_normalized
                -
                early_mean_distance_normalized
            )


        else:

            first_distance_px = np.nan

            first_distance_normalized = np.nan

            minimum_distance_px = np.nan

            last_distance_px = np.nan

            last_distance_normalized = np.nan

            early_mean_distance_normalized = np.nan

            late_mean_distance_normalized = np.nan

            separation_gain_normalized = np.nan

            normalized_nondec_fraction = np.nan


        summary_rows.append(
            {
                "track_id":
                    track_id,

                "overlap_frames":
                    int(
                        len(
                            overlap_frames
                        )
                    ),

                "starts_at_ball_start":
                    starts_at_ball_start,

                "full_trajectory_coverage":
                    full_trajectory_coverage,

                "first_distance_px":
                    first_distance_px,

                "first_distance_normalized":
                    first_distance_normalized,

                "minimum_distance_px":
                    minimum_distance_px,

                "last_distance_px":
                    last_distance_px,

                "last_distance_normalized":
                    last_distance_normalized,

                "early_mean_distance_normalized":
                    early_mean_distance_normalized,

                "late_mean_distance_normalized":
                    late_mean_distance_normalized,

                "separation_gain_normalized":
                    separation_gain_normalized,

                "normalized_nondec_fraction":
                    normalized_nondec_fraction,

                "eligible":
                    full_trajectory_coverage,
            }
        )


    evidence_table = (
        pd.DataFrame(
            evidence_rows
        )
        .sort_values(
            [
                "track_id",
                "frame_idx",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    summary_table = (
        pd.DataFrame(
            summary_rows
        )
        .sort_values(
            "track_id"
        )
        .reset_index(
            drop=True
        )
    )


    eligible_table = (
        summary_table[
            summary_table[
                "eligible"
            ]
        ]
        .copy()
        .sort_values(
            [
                "first_distance_normalized",
                "first_distance_px",
                "track_id",
            ],
            ascending=[
                True,
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


    eligible_table[
        "primary_candidate_rank"
    ] = (
        np.arange(
            len(
                eligible_table
            )
        )
        +
        1
    )


    rank_by_track = {

        int(
            row[
                "track_id"
            ]
        ):
            int(
                row[
                    "primary_candidate_rank"
                ]
            )

        for _, row
        in eligible_table.iterrows()
    }


    summary_table[
        "primary_candidate_rank"
    ] = [

        rank_by_track.get(
            int(
                track_id
            ),
            np.nan,
        )

        for track_id
        in summary_table[
            "track_id"
        ]
    ]


    return {

        "trajectory_frames":
            trajectory_frames,

        "evidence_table":
            evidence_table,

        "summary_table":
            summary_table,

        "eligible_table":
            eligible_table,
    }


def build_shooter_assignment_handler():
    """Build canonical Stage-11 temporal shooter assignment."""

    def shooter_assignment_handler(
        result,
    ):

        expected_prefix = list(
            CANONICAL_PIPELINE_STAGES[
                : _STAGE_INDEX[
                    "shooter_assignment"
                ]
            ]
        )


        if (
            result.pipeline_status.completed_stages
            !=
            expected_prefix
        ):

            raise PipelineContractError(
                "Shooter assignment requires completed pitch_projection."
            )


        if not result.pitch_projection.success:

            raise PipelineContractError(
                "Shooter assignment requires successful pitch projection."
            )


        temporal = (
            _notebook43_build_temporal_shooter_evidence(
                result
            )
        )


        eligible_table = (
            temporal[
                "eligible_table"
            ]
        )


        if len(
            eligible_table
        ) == 0:

            raise RuntimeError(
                "Shooter assignment found no full-trajectory player candidates."
            )


        winner = (
            eligible_table.iloc[
                0
            ]
        )


        selected_track_id = int(
            winner[
                "track_id"
            ]
        )


        trajectory_frames = (
            temporal[
                "trajectory_frames"
            ]
        )


        first_frame = int(
            trajectory_frames[
                0
            ]
        )


        selected_track_matches = [

            track

            for track
            in result.player_tracks.tracks

            if int(
                track.track_id
            )
            ==
            selected_track_id
        ]


        if len(
            selected_track_matches
        ) != 1:

            raise RuntimeError(
                "Selected shooter track could not be resolved uniquely."
            )


        selected_track = (
            selected_track_matches[
                0
            ]
        )


        first_observation_matches = [

            observation

            for observation
            in selected_track.observations

            if int(
                observation.frame_idx
            )
            ==
            first_frame
        ]


        if len(
            first_observation_matches
        ) != 1:

            raise RuntimeError(
                "Selected shooter first-frame observation could not be resolved."
            )


        first_observation = (
            first_observation_matches[
                0
            ]
        )


        image_xy = (

            float(
                first_observation.foot_x
            ),

            float(
                first_observation.foot_y
            ),
        )


        projected_rows = [

            row

            for row
            in result.pitch_projection.projected_players

            if int(
                row[
                    "track_id"
                ]
            )
            ==
            selected_track_id
        ]


        if len(
            projected_rows
        ) != 1:

            raise RuntimeError(
                "Selected shooter has no unique pitch-projection row."
            )


        projected_row = (
            projected_rows[
                0
            ]
        )


        pitch_xy = (

            float(
                projected_row[
                    "pitch_x"
                ]
            ),

            float(
                projected_row[
                    "pitch_y"
                ]
            ),
        )


        if not bool(
            projected_row[
                "inside_pitch"
            ]
        ):

            raise RuntimeError(
                "Selected shooter is outside canonical pitch bounds."
            )


        assignment_score = float(
            winner[
                "first_distance_normalized"
            ]
        )


        # --------------------------------------------------------------------------------------------
        # Populate PUBLIC ShooterResult.
        # --------------------------------------------------------------------------------------------

        result.shooter.success = True


        result.shooter.track_id = (
            selected_track_id
        )


        result.shooter.image_xy = (
            image_xy
        )


        result.shooter.pitch_xy = (
            pitch_xy
        )


        result.shooter.assignment_score = (
            assignment_score
        )


        result.shooter.failure_reason = (
            None
        )


        selected_evidence = (
            temporal[
                "evidence_table"
            ][
                temporal[
                    "evidence_table"
                ][
                    "track_id"
                ]
                .astype(int)
                .eq(
                    selected_track_id
                )
            ]
            .copy()
            .sort_values(
                "frame_idx"
            )
            .reset_index(
                drop=True
            )
        )


        result.metadata[
            "shooter_assignment"
        ] = {

            "success":
                True,

            "method":
                "TEMPORAL_FIRST_BALL_TO_BOX_DISTANCE_NORMALIZED",

            "distance_metric":
                "POINT_TO_AXIS_ALIGNED_BBOX_EUCLIDEAN",

            "distance_normalizer":
                "PLAYER_BOX_HEIGHT",

            "eligibility":
                "FULL_BALL_TRAJECTORY_COVERAGE",

            "primary_ranking":
                "FIRST_BALL_TO_BOX_DISTANCE_NORMALIZED_ASCENDING",

            "trajectory_frames":
                [
                    int(
                        frame
                    )

                    for frame
                    in temporal[
                        "trajectory_frames"
                    ]
                ],

            "player_tracks_evaluated":
                int(
                    len(
                        temporal[
                            "summary_table"
                        ]
                    )
                ),

            "eligible_full_length_tracks":
                int(
                    len(
                        temporal[
                            "eligible_table"
                        ]
                    )
                ),

            "distance_rows":
                int(
                    len(
                        temporal[
                            "evidence_table"
                        ]
                    )
                ),

            "selected_track_id":
                selected_track_id,

            "selected_image_xy":
                list(
                    image_xy
                ),

            "selected_pitch_xy":
                list(
                    pitch_xy
                ),

            "assignment_score":
                assignment_score,

            "primary_candidate_rank":
                int(
                    winner[
                        "primary_candidate_rank"
                    ]
                ),

            "first_distance_px":
                float(
                    winner[
                        "first_distance_px"
                    ]
                ),

            "first_distance_normalized":
                float(
                    winner[
                        "first_distance_normalized"
                    ]
                ),

            "minimum_distance_px":
                float(
                    winner[
                        "minimum_distance_px"
                    ]
                ),

            "last_distance_px":
                float(
                    winner[
                        "last_distance_px"
                    ]
                ),

            "last_distance_normalized":
                float(
                    winner[
                        "last_distance_normalized"
                    ]
                ),

            "early_mean_distance_normalized":
                float(
                    winner[
                        "early_mean_distance_normalized"
                    ]
                ),

            "late_mean_distance_normalized":
                float(
                    winner[
                        "late_mean_distance_normalized"
                    ]
                ),

            "separation_gain_normalized":
                float(
                    winner[
                        "separation_gain_normalized"
                    ]
                ),

            "normalized_nondec_fraction":
                float(
                    winner[
                        "normalized_nondec_fraction"
                    ]
                ),

            "historical_shooter_id_used":
                False,

            "team_information_used":
                False,

            "pitch_projection_used_for_ranking":
                False,

            "pitch_projection_used_after_selection":
                True,

            "known_shot_coordinates_used":
                False,

            "reference_homography_used":
                False,

            "manual_shooter_assignment_used":
                False,

            "_distance_evidence":
                temporal[
                    "evidence_table"
                ],

            "_candidate_summary":
                temporal[
                    "summary_table"
                ],

            "_selected_evidence":
                selected_evidence,
        }


        return {

            "distance_evidence":
                temporal[
                    "evidence_table"
                ],

            "candidate_summary":
                temporal[
                    "summary_table"
                ],

            "eligible_candidates":
                temporal[
                    "eligible_table"
                ],

            "selected_evidence":
                selected_evidence,
        }


    return shooter_assignment_handler


def register_shooter_assignment_handler(
    pipeline,
):
    """Register canonical Stage-11 shooter assignment."""

    pipeline.register_stage_handler(

        "shooter_assignment",

        build_shooter_assignment_handler(),
    )


    return pipeline

# NOTEBOOK43_CANONICAL_TEMPORAL_SHOOTER_ASSIGNMENT_FIX1_DETECTION_JOIN

# NOTEBOOK43_CANONICAL_TEMPORAL_SHOOTER_ASSIGNMENT_FIX2_CANONICAL_SEPARATION_GAIN



# NOTEBOOK43_CANONICAL_RELATIVE_TEAM_ASSIGNMENT_V1

NOTEBOOK43_JERSEY_CLUSTER_COUNT = 3
NOTEBOOK43_MIN_VALID_JERSEY_PIXELS = 40
NOTEBOOK43_HUE_HISTOGRAM_BINS = 12


NOTEBOOK43_APPEARANCE_FEATURE_COLUMNS = [
    "jersey_saturation_mean",
    "jersey_saturation_std",
    "jersey_value_mean",
    "jersey_value_std",
    "jersey_neutral_fraction",
    "jersey_dark_fraction",
    "jersey_bright_fraction",
    "jersey_red_fraction",
    "jersey_blue_fraction",
    "jersey_yellow_fraction",
] + [
    f"jersey_hue_hist_{i:02d}"
    for i in range(NOTEBOOK43_HUE_HISTOGRAM_BINS)
]


def _notebook43_extract_jersey_features(frame, x1, y1, x2, y2):
    import cv2
    import numpy as np

    frame_height, frame_width = frame.shape[:2]

    box_width = max(
        1.0,
        float(x2 - x1),
    )

    box_height = max(
        1.0,
        float(y2 - y1),
    )

    torso_x1 = int(
        np.clip(
            x1 + 0.15 * box_width,
            0,
            frame_width - 1,
        )
    )

    torso_x2 = int(
        np.clip(
            x2 - 0.15 * box_width,
            torso_x1 + 1,
            frame_width,
        )
    )

    torso_y1 = int(
        np.clip(
            y1 + 0.12 * box_height,
            0,
            frame_height - 1,
        )
    )

    torso_y2 = int(
        np.clip(
            y1 + 0.58 * box_height,
            torso_y1 + 1,
            frame_height,
        )
    )

    crop = frame[
        torso_y1:torso_y2,
        torso_x1:torso_x2,
    ]

    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV,
    )

    hue = hsv[:, :, 0].astype(float)
    saturation = hsv[:, :, 1].astype(float)
    value = hsv[:, :, 2].astype(float)

    green_mask = (
        (hue >= 30)
        &
        (hue <= 100)
        &
        (saturation >= 35)
    )

    valid_mask = ~green_mask

    valid_pixel_count = int(
        valid_mask.sum()
    )

    if valid_pixel_count < NOTEBOOK43_MIN_VALID_JERSEY_PIXELS:
        valid_mask = np.ones_like(
            green_mask,
            dtype=bool,
        )

        valid_pixel_count = int(
            valid_mask.sum()
        )

    if valid_pixel_count < NOTEBOOK43_MIN_VALID_JERSEY_PIXELS:
        return None

    valid_hue = hue[valid_mask]
    valid_saturation = saturation[valid_mask]
    valid_value = value[valid_mask]

    saturated_hue = valid_hue[
        valid_saturation >= 45
    ]

    if len(saturated_hue) > 0:
        histogram, _ = np.histogram(
            saturated_hue,
            bins=NOTEBOOK43_HUE_HISTOGRAM_BINS,
            range=(0, 180),
        )

        histogram = (
            histogram.astype(float)
            /
            max(
                1.0,
                float(histogram.sum()),
            )
        )

    else:
        histogram = np.zeros(
            NOTEBOOK43_HUE_HISTOGRAM_BINS,
            dtype=float,
        )

    neutral_fraction = float(
        (valid_saturation < 45).mean()
    )

    dark_fraction = float(
        (valid_value < 70).mean()
    )

    bright_fraction = float(
        (valid_value > 180).mean()
    )

    red_fraction = float(
        (
            (
                (valid_hue <= 12)
                |
                (valid_hue >= 168)
            )
            &
            (valid_saturation >= 50)
        ).mean()
    )

    blue_fraction = float(
        (
            (valid_hue >= 90)
            &
            (valid_hue <= 135)
            &
            (valid_saturation >= 50)
        ).mean()
    )

    yellow_fraction = float(
        (
            (valid_hue >= 15)
            &
            (valid_hue <= 35)
            &
            (valid_saturation >= 50)
        ).mean()
    )

    result = {
        "valid_jersey_pixels":
            valid_pixel_count,

        "jersey_saturation_mean":
            float(valid_saturation.mean()),

        "jersey_saturation_std":
            float(valid_saturation.std()),

        "jersey_value_mean":
            float(valid_value.mean()),

        "jersey_value_std":
            float(valid_value.std()),

        "jersey_neutral_fraction":
            neutral_fraction,

        "jersey_dark_fraction":
            dark_fraction,

        "jersey_bright_fraction":
            bright_fraction,

        "jersey_red_fraction":
            red_fraction,

        "jersey_blue_fraction":
            blue_fraction,

        "jersey_yellow_fraction":
            yellow_fraction,
    }

    for index, histogram_value in enumerate(histogram):
        result[
            f"jersey_hue_hist_{index:02d}"
        ] = float(histogram_value)

    return result


def _notebook43_read_relative_team_frames(video_path, frame_indices):
    import cv2

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open clip for appearance extraction: {video_path}"
        )

    frame_lookup = {}

    try:
        for frame_idx in sorted(
            set(
                int(v)
                for v in frame_indices
            )
        ):
            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_idx,
            )

            ok, frame = capture.read()

            if not ok or frame is None:
                raise RuntimeError(
                    f"Could not read frame {frame_idx}."
                )

            frame_lookup[
                frame_idx
            ] = frame

    finally:
        capture.release()

    return frame_lookup


def _notebook43_build_fresh_appearance_rows(result):
    import pandas as pd

    rows = []

    for track in result.player_tracks.tracks:
        for observation in track.observations:
            rows.append(
                {
                    "track_id":
                        int(track.track_id),

                    "frame_idx":
                        int(observation.frame_idx),

                    # Public PlayerTrackObservation uses detection_idx.
                    "detection_id":
                        int(observation.detection_idx),

                    "confidence":
                        float(observation.confidence),

                    "x1":
                        float(observation.x1),

                    "y1":
                        float(observation.y1),

                    "x2":
                        float(observation.x2),

                    "y2":
                        float(observation.y2),
                }
            )

    observations = pd.DataFrame(
        rows
    ).sort_values(
        [
            "frame_idx",
            "detection_id",
            "track_id",
        ]
    ).reset_index(
        drop=True
    )

    frame_lookup = (
        _notebook43_read_relative_team_frames(
            result.video_path,
            observations[
                "frame_idx"
            ].tolist(),
        )
    )

    feature_rows = []

    for _, row in observations.iterrows():
        features = (
            _notebook43_extract_jersey_features(
                frame_lookup[
                    int(row["frame_idx"])
                ],
                float(row["x1"]),
                float(row["y1"]),
                float(row["x2"]),
                float(row["y2"]),
            )
        )

        if features is None:
            raise RuntimeError(
                "Canonical jersey feature extraction failed minimum-pixel contract."
            )

        feature_rows.append(
            {
                **row.to_dict(),
                **features,
            }
        )

    return pd.DataFrame(
        feature_rows
    ).sort_values(
        [
            "frame_idx",
            "detection_id",
            "track_id",
        ]
    ).reset_index(
        drop=True
    )


def _notebook43_cluster_relative_appearance(appearance_table):
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    feature_matrix = appearance_table[
        NOTEBOOK43_APPEARANCE_FEATURE_COLUMNS
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        feature_matrix
    ).all():
        raise RuntimeError(
            "Appearance feature matrix contains non-finite values."
        )

    scaler = StandardScaler()

    scaled = scaler.fit_transform(
        feature_matrix
    )

    model = KMeans(
        n_clusters=
            NOTEBOOK43_JERSEY_CLUSTER_COUNT,

        random_state=
            42,

        n_init=
            20,
    )

    labels = model.fit_predict(
        scaled
    )

    clustered = appearance_table.copy()

    clustered[
        "appearance_cluster"
    ] = labels.astype(int)

    return (
        clustered,
        scaled,
        np.asarray(
            model.cluster_centers_,
            dtype=float,
        ),
    )


def _notebook43_relative_track_summary(clustered):
    import pandas as pd

    summaries = []

    for track_id, group in clustered.groupby(
        "track_id",
        sort=True,
    ):
        counts = {
            cluster:
                int(
                    (
                        group[
                            "appearance_cluster"
                        ]
                        .astype(int)
                        ==
                        cluster
                    ).sum()
                )

            for cluster
            in range(3)
        }

        maximum = max(
            counts.values()
        )

        winners = [
            cluster
            for cluster, count
            in counts.items()
            if count == maximum
        ]

        tied = (
            len(winners) != 1
        )

        modal_cluster = (
            int(winners[0])
            if not tied
            else None
        )

        summaries.append(
            {
                "track_id":
                    int(track_id),

                "appearance_rows":
                    int(len(group)),

                "cluster_0_rows":
                    counts[0],

                "cluster_1_rows":
                    counts[1],

                "cluster_2_rows":
                    counts[2],

                "modal_cluster":
                    modal_cluster,

                "modal_rows":
                    int(maximum),

                "modal_fraction":
                    float(
                        maximum
                        /
                        len(group)
                    ),

                "modal_tied":
                    bool(tied),
            }
        )

    return pd.DataFrame(
        summaries
    ).sort_values(
        "track_id"
    ).reset_index(
        drop=True
    )


def build_relative_team_assignment_handler():
    import pandas as pd

    def handler(result):
        expected_prefix = list(
            CANONICAL_PIPELINE_STAGES[
                : _STAGE_INDEX[
                    "relative_team_assignment"
                ]
            ]
        )

        if (
            result.pipeline_status.completed_stages
            !=
            expected_prefix
        ):
            raise PipelineContractError(
                "relative_team_assignment requires completed shooter_assignment."
            )

        if not result.shooter.success:
            raise PipelineContractError(
                "A successful ShooterResult is required."
            )

        appearance = (
            _notebook43_build_fresh_appearance_rows(
                result
            )
        )

        clustered, scaled, centers = (
            _notebook43_cluster_relative_appearance(
                appearance
            )
        )

        summary = (
            _notebook43_relative_track_summary(
                clustered
            )
        )

        if bool(
            summary[
                "modal_tied"
            ].any()
        ):
            raise RuntimeError(
                "At least one track has a tied modal appearance cluster."
            )

        shooter_track_id = int(
            result.shooter.track_id
        )

        shooter_rows = summary[
            summary[
                "track_id"
            ]
            .astype(int)
            .eq(
                shooter_track_id
            )
        ]

        if len(shooter_rows) != 1:
            raise RuntimeError(
                "Shooter modal appearance cluster could not be resolved."
            )

        attacking_cluster = int(
            shooter_rows.iloc[
                0
            ][
                "modal_cluster"
            ]
        )

        all_clusters = {
            0,
            1,
            2,
        }

        remaining = sorted(
            all_clusters
            -
            {
                attacking_cluster
            }
        )

        remaining_counts = {
            cluster:
                int(
                    (
                        clustered[
                            "appearance_cluster"
                        ]
                        .astype(int)
                        ==
                        cluster
                    ).sum()
                )

            for cluster
            in remaining
        }

        defending_cluster = sorted(
            remaining_counts,
            key=lambda cluster:
                (
                    -
                    remaining_counts[
                        cluster
                    ],
                    cluster,
                ),
        )[0]

        other_cluster = next(
            iter(
                all_clusters
                -
                {
                    attacking_cluster,
                    defending_cluster,
                }
            )
        )

        cluster_to_team = {
            attacking_cluster:
                "ATTACKING",

            defending_cluster:
                "DEFENDING",

            int(other_cluster):
                "OTHER",
        }

        track_labels = {
            int(row["track_id"]):
                cluster_to_team[
                    int(
                        row[
                            "modal_cluster"
                        ]
                    )
                ]

            for _, row
            in summary.iterrows()
        }

        summary = summary.copy()

        summary[
            "relative_team"
        ] = [
            track_labels[
                int(track_id)
            ]

            for track_id
            in summary[
                "track_id"
            ]
        ]

        projected = pd.DataFrame(
            result.pitch_projection.projected_players
        )

        projected[
            "relative_team"
        ] = (
            projected[
                "track_id"
            ]
            .astype(int)
            .map(
                track_labels
            )
        )

        if projected[
            "relative_team"
        ].isna().any():
            raise RuntimeError(
                "Projected-player relative labels are incomplete."
            )

        result.relative_teams.success = True
        result.relative_teams.shooter_team_label = "ATTACKING"
        result.relative_teams.track_labels = track_labels
        result.relative_teams.failure_reason = None

        row_cluster_counts = {
            cluster:
                int(
                    (
                        clustered[
                            "appearance_cluster"
                        ]
                        .astype(int)
                        ==
                        cluster
                    ).sum()
                )

            for cluster
            in range(3)
        }

        result.metadata[
            "relative_team_assignment"
        ] = {
            "success":
                True,

            "method":
                "FRESH_UNSUPERVISED_JERSEY_APPEARANCE_CLUSTERING",

            "appearance_rows":
                int(len(appearance)),

            "feature_count":
                int(
                    len(
                        NOTEBOOK43_APPEARANCE_FEATURE_COLUMNS
                    )
                ),

            "appearance_cluster_counts":
                row_cluster_counts,

            "attacking_cluster":
                attacking_cluster,

            "defending_cluster":
                defending_cluster,

            "other_cluster":
                int(other_cluster),

            "cluster_to_relative_team":
                cluster_to_team,

            "track_labels":
                track_labels,

            "absolute_team_names_used":
                False,

            "match_specific_color_rules_used":
                False,

            "historical_team_labels_used":
                False,

            "shooter_identity_used_in_clustering":
                False,

            "shooter_identity_used_posthoc":
                True,

            "ball_information_used":
                False,

            "pitch_coordinates_used_in_clustering":
                False,

            "_appearance_table":
                appearance,

            "_clustered_table":
                clustered,

            "_track_summary":
                summary,

            "_projected_table":
                projected,

            "_scaled_features":
                scaled,

            "_cluster_centers_scaled":
                centers,
        }

        return {
            "appearance_table":
                appearance,

            "clustered_table":
                clustered,

            "track_summary":
                summary,

            "projected_table":
                projected,

            "scaled_features":
                scaled,

            "cluster_centers_scaled":
                centers,

            "cluster_to_relative_team":
                cluster_to_team,
        }

    return handler


def register_relative_team_assignment_handler(pipeline):
    pipeline.register_stage_handler(
        "relative_team_assignment",
        build_relative_team_assignment_handler(),
    )

    return pipeline



# NOTEBOOK43_CANONICAL_TARGET_GOAL_INFERENCE_V1
# ====================================================================================================
# CANONICAL TARGET-GOAL INFERENCE FROM FRESH BALL MOTION
# ====================================================================================================


def _notebook43_project_ball_trajectory_to_pitch(
    result,
):
    """
    Project CURRENT BallTrackResult trajectory through CURRENT validated
    PitchProjectionResult homography.

    No historical projected ball positions are accepted.
    """

    H = np.asarray(
        result.pitch_projection.homography,
        dtype=np.float64,
    )


    if H.shape != (3, 3):
        raise RuntimeError(
            "Target-goal inference requires a 3x3 homography."
        )


    if not np.isfinite(H).all():
        raise RuntimeError(
            "Target-goal homography contains non-finite values."
        )


    rows = []


    for point in sorted(
        result.ball_track.track,
        key=lambda item:
            int(item.frame_idx),
    ):
        image_vector = np.asarray(
            [
                float(point.center_x),
                float(point.center_y),
                1.0,
            ],
            dtype=np.float64,
        )


        pitch_h = H @ image_vector


        denominator = float(
            pitch_h[2]
        )


        if (
            not np.isfinite(denominator)
            or abs(denominator) < 1e-12
        ):
            raise RuntimeError(
                "Ball projection encountered invalid homogeneous denominator."
            )


        pitch_x = float(
            pitch_h[0]
            /
            denominator
        )

        pitch_y = float(
            pitch_h[1]
            /
            denominator
        )


        if not np.isfinite(
            [
                pitch_x,
                pitch_y,
            ]
        ).all():
            raise RuntimeError(
                "Ball pitch projection produced non-finite coordinates."
            )


        rows.append(
            {
                "frame_idx":
                    int(point.frame_idx),

                "image_x":
                    float(point.center_x),

                "image_y":
                    float(point.center_y),

                "trajectory_source":
                    str(point.trajectory_source),

                "is_raw_v3_detection":
                    bool(point.is_raw_v3_detection),

                "is_interpolated":
                    bool(point.is_interpolated),

                "projected_pitch_x":
                    pitch_x,

                "projected_pitch_y":
                    pitch_y,
            }
        )


    table = (
        pd.DataFrame(rows)
        .sort_values(
            "frame_idx"
        )
        .reset_index(
            drop=True
        )
    )


    if len(table) < 3:
        raise RuntimeError(
            "Target-goal inference requires at least three trajectory points."
        )


    return table


def _notebook43_ball_motion_summary(
    trajectory_table,
):
    """
    Compute three independent X-direction summaries:

      1. endpoint displacement
      2. median per-frame displacement
      3. linear pitch-X slope
    """

    frames = (
        trajectory_table[
            "frame_idx"
        ]
        .to_numpy(
            dtype=float
        )
    )


    pitch_x = (
        trajectory_table[
            "projected_pitch_x"
        ]
        .to_numpy(
            dtype=float
        )
    )


    if (
        len(frames) < 3
        or
        len(np.unique(frames)) != len(frames)
    ):
        raise RuntimeError(
            "Target-goal motion frames are invalid."
        )


    endpoint_delta_x = float(
        pitch_x[-1]
        -
        pitch_x[0]
    )


    frame_deltas = np.diff(
        frames
    )


    if bool(
        np.any(
            frame_deltas
            <=
            0.0
        )
    ):
        raise RuntimeError(
            "Target-goal trajectory frames are not strictly increasing."
        )


    per_frame_delta_x = (
        np.diff(
            pitch_x
        )
        /
        frame_deltas
    )


    median_delta_x_per_frame = float(
        np.median(
            per_frame_delta_x
        )
    )


    linear_slope = float(
        np.polyfit(
            frames,
            pitch_x,
            1,
        )[0]
    )


    direction_values = [
        endpoint_delta_x,
        median_delta_x_per_frame,
        linear_slope,
    ]


    epsilon = 1e-12


    direction_signs = [
        (
            1
            if value > epsilon
            else
            (
                -1
                if value < -epsilon
                else
                0
            )
        )

        for value
        in direction_values
    ]


    unambiguous = all(
        sign != 0
        for sign
        in direction_signs
    )


    summaries_agree = (
        unambiguous
        and
        len(
            set(
                direction_signs
            )
        )
        ==
        1
    )


    return {
        "first_pitch_x":
            float(
                pitch_x[0]
            ),

        "last_pitch_x":
            float(
                pitch_x[-1]
            ),

        "endpoint_delta_x":
            endpoint_delta_x,

        "median_delta_x_per_frame":
            median_delta_x_per_frame,

        "linear_pitch_x_slope":
            linear_slope,

        "direction_signs":
            direction_signs,

        "direction_unambiguous":
            unambiguous,

        "summaries_agree":
            summaries_agree,

        "direction_sign":
            (
                int(direction_signs[0])
                if summaries_agree
                else 0
            ),
    }


def _notebook43_resolve_horizon_aware_ball_motion_summary(
    trajectory_table,
    homography,
    production_summary,
):
    """
    Preserve the ordinary ball-motion summary contract and resolve direction
    only when projective-horizon crossing invalidates global pitch-X summaries.

    The fallback excludes only adjacent trajectory transitions whose homogeneous
    denominator changes sign. It requires at least two usable nonzero local
    per-frame pitch-X deltas and requires every usable nonzero sign to agree.

    No goal side, team identity, shooter position, or historical label is used.
    """

    if bool(
        production_summary["summaries_agree"]
    ):
        return production_summary

    H = np.asarray(
        homography,
        dtype=np.float64,
    )

    if H.shape != (3, 3):
        return production_summary

    if not np.isfinite(H).all():
        return production_summary

    table = (
        trajectory_table
        .sort_values(
            "frame_idx"
        )
        .reset_index(
            drop=True
        )
    )

    frames = (
        table[
            "frame_idx"
        ]
        .to_numpy(
            dtype=float
        )
    )

    image_x = (
        table[
            "image_x"
        ]
        .to_numpy(
            dtype=float
        )
    )

    image_y = (
        table[
            "image_y"
        ]
        .to_numpy(
            dtype=float
        )
    )

    pitch_x = (
        table[
            "projected_pitch_x"
        ]
        .to_numpy(
            dtype=float
        )
    )

    if len(frames) < 3:
        return production_summary

    denominators = (
        H[2, 0]
        *
        image_x
        +
        H[2, 1]
        *
        image_y
        +
        H[2, 2]
    )

    if not np.isfinite(
        denominators
    ).all():
        return production_summary

    frame_deltas = np.diff(
        frames
    )

    if bool(
        np.any(
            frame_deltas
            <=
            0.0
        )
    ):
        return production_summary

    pitch_deltas = np.diff(
        pitch_x
    )

    denominator_products = (
        denominators[:-1]
        *
        denominators[1:]
    )

    crossing_mask = (
        denominator_products
        <=
        0.0
    )

    if not bool(
        np.any(
            crossing_mask
        )
    ):
        return production_summary

    usable_mask = ~crossing_mask

    usable_per_frame_delta = (
        pitch_deltas[
            usable_mask
        ]
        /
        frame_deltas[
            usable_mask
        ]
    )

    epsilon = 1e-12

    usable_signs = [
        (
            1
            if value > epsilon
            else
            (
                -1
                if value < -epsilon
                else
                0
            )
        )
        for value
        in usable_per_frame_delta
    ]

    nonzero_signs = [
        int(sign)
        for sign
        in usable_signs
        if int(sign) != 0
    ]

    if len(
        nonzero_signs
    ) < 2:
        return production_summary

    unique_signs = set(
        nonzero_signs
    )

    if len(
        unique_signs
    ) != 1:
        return production_summary

    consensus_sign = int(
        nonzero_signs[0]
    )

    if consensus_sign not in (
        -1,
        1,
    ):
        return production_summary

    resolved_summary = dict(
        production_summary
    )

    resolved_summary["direction_signs"] = [
        consensus_sign,
        consensus_sign,
        consensus_sign,
    ]

    resolved_summary["direction_unambiguous"] = True
    resolved_summary["summaries_agree"] = True
    resolved_summary["direction_sign"] = consensus_sign

    return resolved_summary

def build_target_goal_inference_handler():
    """
    Build canonical Stage-13 target-goal inference.

    Decision hierarchy:

      PRIMARY:
          RAW V3 ball points only.

      REQUIRED:
          endpoint, median-frame-delta and linear slope agree.

      CORROBORATION:
          full trajectory including interpolation agrees with RAW direction.
    """

    def handler(result):
        expected_prefix = list(
            CANONICAL_PIPELINE_STAGES[
                : _STAGE_INDEX[
                    "target_goal_inference"
                ]
            ]
        )


        if (
            result.pipeline_status.completed_stages
            !=
            expected_prefix
        ):
            raise PipelineContractError(
                "target_goal_inference requires completed relative_team_assignment."
            )


        if not result.ball_track.success:
            raise PipelineContractError(
                "Target-goal inference requires successful BallTrackResult."
            )


        if not result.pitch_projection.success:
            raise PipelineContractError(
                "Target-goal inference requires successful PitchProjectionResult."
            )


        trajectory_table = (
            _notebook43_project_ball_trajectory_to_pitch(
                result
            )
        )


        raw_table = (
            trajectory_table[
                trajectory_table[
                    "is_raw_v3_detection"
                ]
                .astype(bool)
            ]
            .copy()
            .sort_values(
                "frame_idx"
            )
            .reset_index(
                drop=True
            )
        )


        if len(raw_table) < 3:
            raise RuntimeError(
                "Target-goal inference requires at least three RAW V3 points."
            )


        raw_summary = (
            _notebook43_ball_motion_summary(
                raw_table
            )
        )

        raw_summary = (
            _notebook43_resolve_horizon_aware_ball_motion_summary(
                trajectory_table=raw_table,
                homography=result.pitch_projection.homography,
                production_summary=raw_summary,
            )
        )


        full_summary = (
            _notebook43_ball_motion_summary(
                trajectory_table
            )
        )

        full_summary = (
            _notebook43_resolve_horizon_aware_ball_motion_summary(
                trajectory_table=trajectory_table,
                homography=result.pitch_projection.homography,
                production_summary=full_summary,
            )
        )


        if not bool(
            raw_summary[
                "summaries_agree"
            ]
        ):
            raise RuntimeError(
                "RAW V3 target-goal direction is ambiguous."
            )


        if not bool(
            full_summary[
                "summaries_agree"
            ]
        ):
            raise RuntimeError(
                "Full trajectory target-goal corroboration is ambiguous."
            )


        raw_direction_sign = int(
            raw_summary[
                "direction_sign"
            ]
        )


        full_direction_sign = int(
            full_summary[
                "direction_sign"
            ]
        )


        if (
            raw_direction_sign
            !=
            full_direction_sign
        ):
            raise RuntimeError(
                "Interpolated trajectory disagrees with RAW V3 direction."
            )


        if raw_direction_sign < 0:
            target_goal = "LEFT"
            goal_center_pitch_xy = (
                0.0,
                34.0,
            )

        elif raw_direction_sign > 0:
            target_goal = "RIGHT"
            goal_center_pitch_xy = (
                105.0,
                34.0,
            )

        else:
            raise RuntimeError(
                "Target-goal direction resolved to zero motion."
            )


        # Three independent RAW summaries agree AND the full trajectory
        # independently corroborates the same direction.
        confidence = 1.0


        result.target_goal.success = True
        result.target_goal.target_goal = target_goal
        result.target_goal.confidence = confidence
        result.target_goal.failure_reason = None


        result.metadata[
            "target_goal_inference"
        ] = {
            "success":
                True,

            "method":
                "RAW_V3_PROJECTED_BALL_MOTION_CONSENSUS",

            "primary_evidence":
                "RAW_V3_BALL_MOTION",

            "raw_point_count":
                int(
                    len(
                        raw_table
                    )
                ),

            "raw_frames":
                raw_table[
                    "frame_idx"
                ]
                .astype(int)
                .tolist(),

            "raw_first_pitch_x":
                float(
                    raw_summary[
                        "first_pitch_x"
                    ]
                ),

            "raw_last_pitch_x":
                float(
                    raw_summary[
                        "last_pitch_x"
                    ]
                ),

            "raw_endpoint_delta_x":
                float(
                    raw_summary[
                        "endpoint_delta_x"
                    ]
                ),

            "raw_median_delta_x_per_frame":
                float(
                    raw_summary[
                        "median_delta_x_per_frame"
                    ]
                ),

            "raw_linear_pitch_x_slope":
                float(
                    raw_summary[
                        "linear_pitch_x_slope"
                    ]
                ),

            "raw_direction_signs":
                list(
                    raw_summary[
                        "direction_signs"
                    ]
                ),

            "raw_summaries_agree":
                bool(
                    raw_summary[
                        "summaries_agree"
                    ]
                ),

            "full_point_count":
                int(
                    len(
                        trajectory_table
                    )
                ),

            "interpolated_point_count":
                int(
                    trajectory_table[
                        "is_interpolated"
                    ]
                    .astype(bool)
                    .sum()
                ),

            "full_frames":
                trajectory_table[
                    "frame_idx"
                ]
                .astype(int)
                .tolist(),

            "full_endpoint_delta_x":
                float(
                    full_summary[
                        "endpoint_delta_x"
                    ]
                ),

            "full_median_delta_x_per_frame":
                float(
                    full_summary[
                        "median_delta_x_per_frame"
                    ]
                ),

            "full_linear_pitch_x_slope":
                float(
                    full_summary[
                        "linear_pitch_x_slope"
                    ]
                ),

            "full_direction_signs":
                list(
                    full_summary[
                        "direction_signs"
                    ]
                ),

            "full_summaries_agree":
                bool(
                    full_summary[
                        "summaries_agree"
                    ]
                ),

            "full_direction_matches_raw":
                bool(
                    full_direction_sign
                    ==
                    raw_direction_sign
                ),

            "target_goal":
                target_goal,

            # Internal/derived geometry only.
            # Deliberately not added to TargetGoalResult public schema.
            "derived_goal_center_pitch_xy":
                list(
                    goal_center_pitch_xy
                ),

            "confidence":
                confidence,

            "relative_team_labels_used_for_direction":
                False,

            "shooter_goal_distance_used_for_direction":
                False,

            "shooter_position_used_for_direction":
                False,

            "defender_positions_used_for_direction":
                False,

            "historical_target_goal_used":
                False,

            "manual_target_goal_used":
                False,

            "reference_shot_coordinates_used":
                False,

            "interpolated_point_used_as_primary_evidence":
                False,

            "_trajectory_table":
                trajectory_table,
        }


        return {
            "trajectory_table":
                trajectory_table,

            "raw_table":
                raw_table,

            "raw_summary":
                raw_summary,

            "full_summary":
                full_summary,

            "target_goal":
                target_goal,

            "goal_center_pitch_xy":
                goal_center_pitch_xy,

            "confidence":
                confidence,
        }


    return handler


def register_target_goal_inference_handler(
    pipeline,
):
    pipeline.register_stage_handler(
        "target_goal_inference",
        build_target_goal_inference_handler(),
    )

    return pipeline



# NOTEBOOK43_CANONICAL_SHOT_GEOMETRY_V1
# ====================================================================================================
# CANONICAL SHOT GEOMETRY
# ====================================================================================================

NOTEBOOK43_PITCH_LENGTH_M = 105.0
NOTEBOOK43_PITCH_WIDTH_M = 68.0

NOTEBOOK43_GOAL_CENTER_Y_M = 34.0
NOTEBOOK43_GOAL_WIDTH_M = 7.32


def _notebook43_goal_geometry_for_side(
    target_goal,
):
    """
    Resolve canonical pitch-space goal geometry from the
    CURRENT TargetGoalResult side.
    """

    side = str(
        target_goal
    ).upper()


    if side == "LEFT":
        goal_x = 0.0

    elif side == "RIGHT":
        goal_x = NOTEBOOK43_PITCH_LENGTH_M

    else:
        raise RuntimeError(
            f"Unsupported target goal side: {target_goal}"
        )


    goal_center = (
        float(goal_x),
        float(NOTEBOOK43_GOAL_CENTER_Y_M),
    )


    half_goal_width = (
        NOTEBOOK43_GOAL_WIDTH_M
        /
        2.0
    )


    top_post = (
        float(goal_x),
        float(
            NOTEBOOK43_GOAL_CENTER_Y_M
            -
            half_goal_width
        ),
    )


    bottom_post = (
        float(goal_x),
        float(
            NOTEBOOK43_GOAL_CENTER_Y_M
            +
            half_goal_width
        ),
    )


    return {
        "goal_center_pitch_xy":
            goal_center,

        "goal_post_top_pitch_xy":
            top_post,

        "goal_post_bottom_pitch_xy":
            bottom_post,
    }


def _notebook43_compute_shot_geometry(
    shooter_pitch_xy,
    target_goal,
):
    """
    Compute distance to goal center and visible goal-mouth angle
    entirely from CURRENT shooter pitch coordinates and CURRENT
    target-goal side.
    """

    import numpy as np


    shooter = np.asarray(
        shooter_pitch_xy,
        dtype=np.float64,
    )


    if shooter.shape != (2,):
        raise RuntimeError(
            "Shooter pitch coordinate must contain exactly two values."
        )


    if not np.isfinite(
        shooter
    ).all():
        raise RuntimeError(
            "Shooter pitch coordinate contains non-finite values."
        )


    goal_geometry = (
        _notebook43_goal_geometry_for_side(
            target_goal
        )
    )


    goal_center = np.asarray(
        goal_geometry[
            "goal_center_pitch_xy"
        ],
        dtype=np.float64,
    )


    top_post = np.asarray(
        goal_geometry[
            "goal_post_top_pitch_xy"
        ],
        dtype=np.float64,
    )


    bottom_post = np.asarray(
        goal_geometry[
            "goal_post_bottom_pitch_xy"
        ],
        dtype=np.float64,
    )


    # --------------------------------------------------------------------------------------------
    # DISTANCE
    # --------------------------------------------------------------------------------------------

    distance_to_goal_m = float(
        np.linalg.norm(
            shooter
            -
            goal_center
        )
    )


    # --------------------------------------------------------------------------------------------
    # SHOT ANGLE
    #
    # Angle subtended by the two goalposts at the shooter.
    # --------------------------------------------------------------------------------------------

    vector_to_top = (
        top_post
        -
        shooter
    )


    vector_to_bottom = (
        bottom_post
        -
        shooter
    )


    top_norm = float(
        np.linalg.norm(
            vector_to_top
        )
    )


    bottom_norm = float(
        np.linalg.norm(
            vector_to_bottom
        )
    )


    if (
        top_norm <= 0.0
        or
        bottom_norm <= 0.0
    ):
        raise RuntimeError(
            "Shooter position coincides with a goalpost."
        )


    cosine_angle = float(
        np.dot(
            vector_to_top,
            vector_to_bottom,
        )
        /
        (
            top_norm
            *
            bottom_norm
        )
    )


    cosine_angle = float(
        np.clip(
            cosine_angle,
            -1.0,
            1.0,
        )
    )


    shot_angle_deg = float(
        np.degrees(
            np.arccos(
                cosine_angle
            )
        )
    )


    if not np.isfinite(
        [
            distance_to_goal_m,
            shot_angle_deg,
        ]
    ).all():
        raise RuntimeError(
            "Shot geometry produced non-finite values."
        )


    if distance_to_goal_m < 0.0:
        raise RuntimeError(
            "Distance to goal is invalid."
        )


    if (
        shot_angle_deg < 0.0
        or
        shot_angle_deg > 180.0
    ):
        raise RuntimeError(
            "Shot angle is outside valid bounds."
        )


    return {
        "shooter_pitch_xy":
            (
                float(
                    shooter[0]
                ),
                float(
                    shooter[1]
                ),
            ),

        "target_goal":
            str(
                target_goal
            ).upper(),

        "goal_center_pitch_xy":
            tuple(
                float(value)
                for value
                in goal_center
            ),

        "goal_post_top_pitch_xy":
            tuple(
                float(value)
                for value
                in top_post
            ),

        "goal_post_bottom_pitch_xy":
            tuple(
                float(value)
                for value
                in bottom_post
            ),

        "distance_to_goal_m":
            distance_to_goal_m,

        "shot_angle_deg":
            shot_angle_deg,
    }


def build_shot_geometry_handler():
    """
    Canonical Stage-14 shot geometry.

    Production inputs:
      - current ShooterResult.pitch_xy
      - current TargetGoalResult.target_goal

    No reference shot coordinates, historical shot geometry,
    team identities, defenders, or xG are used.
    """

    def handler(result):
        expected_prefix = list(
            CANONICAL_PIPELINE_STAGES[
                : _STAGE_INDEX[
                    "shot_geometry"
                ]
            ]
        )


        if (
            result.pipeline_status.completed_stages
            !=
            expected_prefix
        ):
            raise PipelineContractError(
                "shot_geometry requires completed target_goal_inference."
            )


        if not result.shooter.success:
            raise PipelineContractError(
                "Shot geometry requires successful ShooterResult."
            )


        if not result.target_goal.success:
            raise PipelineContractError(
                "Shot geometry requires successful TargetGoalResult."
            )


        if result.shooter.pitch_xy is None:
            raise RuntimeError(
                "ShooterResult.pitch_xy is missing."
            )


        if result.target_goal.target_goal is None:
            raise RuntimeError(
                "TargetGoalResult.target_goal is missing."
            )


        geometry = (
            _notebook43_compute_shot_geometry(
                shooter_pitch_xy=
                    result.shooter.pitch_xy,

                target_goal=
                    result.target_goal.target_goal,
            )
        )


        result.shot_geometry.success = True

        result.shot_geometry.shooter_pitch_xy = (
            geometry[
                "shooter_pitch_xy"
            ]
        )

        result.shot_geometry.distance_to_goal_m = float(
            geometry[
                "distance_to_goal_m"
            ]
        )

        result.shot_geometry.shot_angle_deg = float(
            geometry[
                "shot_angle_deg"
            ]
        )

        result.shot_geometry.failure_reason = None


        result.metadata[
            "shot_geometry"
        ] = {
            "success":
                True,

            "method":
                "PITCH_SPACE_GOAL_CENTER_DISTANCE_AND_GOALPOST_SUBTENDED_ANGLE",

            "shooter_pitch_xy":
                list(
                    geometry[
                        "shooter_pitch_xy"
                    ]
                ),

            "target_goal":
                geometry[
                    "target_goal"
                ],

            "goal_center_pitch_xy":
                list(
                    geometry[
                        "goal_center_pitch_xy"
                    ]
                ),

            "goal_post_top_pitch_xy":
                list(
                    geometry[
                        "goal_post_top_pitch_xy"
                    ]
                ),

            "goal_post_bottom_pitch_xy":
                list(
                    geometry[
                        "goal_post_bottom_pitch_xy"
                    ]
                ),

            "goal_width_m":
                float(
                    NOTEBOOK43_GOAL_WIDTH_M
                ),

            "distance_to_goal_m":
                float(
                    geometry[
                        "distance_to_goal_m"
                    ]
                ),

            "shot_angle_deg":
                float(
                    geometry[
                        "shot_angle_deg"
                    ]
                ),

            "current_shooter_result_used":
                True,

            "current_target_goal_result_used":
                True,

            "historical_shot_geometry_used":
                False,

            "reference_shot_coordinates_used":
                False,

            "manual_shooter_coordinates_used":
                False,

            "relative_team_labels_used":
                False,

            "defender_context_used":
                False,

            "xg_used":
                False,
        }


        return geometry


    return handler


def register_shot_geometry_handler(
    pipeline,
):
    pipeline.register_stage_handler(
        "shot_geometry",
        build_shot_geometry_handler(),
    )

    return pipeline



# NOTEBOOK43_CANONICAL_DEFENDER_CONTEXT_V1
# NOTEBOOK43_CANONICAL_DEFENDER_CONTEXT_FIX1_PUBLIC_SCHEMA
# ====================================================================================================
# CANONICAL DEFENDER CONTEXT
# ====================================================================================================


def _notebook43_defender_context_nearest_field(
    defender_context,
):
    """
    Resolve the frozen public DefenderContextResult nearest-distance field
    without altering the public dataclass schema.
    """

    for field_name in (
        "nearest_defender_m",
        "nearest_defender_distance_m",
        "nearest_defender_distance",
    ):
        if hasattr(
            defender_context,
            field_name,
        ):
            return field_name


    raise RuntimeError(
        "DefenderContextResult has no recognized nearest-defender distance field."
    )


# NOTEBOOK43_CANONICAL_DEFENDER_CONTEXT_FIX2_PROJECTED_PLAYER_MAPPING


def _notebook43_projected_player_value(
    player,
    key,
):
    """
    Read a canonical projected-player field from either:

      - the public dictionary representation, or
      - an object/dataclass representation.

    Dictionary representation is the current canonical public contract.
    """

    if isinstance(
        player,
        dict,
    ):

        if key not in player:

            raise RuntimeError(
                f"Projected-player dictionary is missing required key: {key}"
            )

        return player[
            key
        ]


    if hasattr(
        player,
        key,
    ):

        return getattr(
            player,
            key,
        )


    raise RuntimeError(
        f"Projected-player record has no field/key: {key}"
    )


def _notebook43_compute_defender_context(
    result,
):
    """
    Compute canonical defender context using CURRENT outputs only.

    Inputs:
      - ShotGeometryResult.shooter_pitch_xy
      - PitchProjectionResult.projected_players
      - RelativeTeamResult.track_labels

    Eligibility:
      - relative label == DEFENDING
      - current geometry-frame projected record exists
      - projected player is inside canonical pitch

    Distance:
      Euclidean pitch-space distance from shooter to defender.
    """

    import numpy as np
    import pandas as pd


    shooter_xy = np.asarray(
        result.shot_geometry.shooter_pitch_xy,
        dtype=np.float64,
    )


    if shooter_xy.shape != (2,):

        raise RuntimeError(
            "Defender context requires a two-value shooter pitch coordinate."
        )


    if not np.isfinite(
        shooter_xy
    ).all():

        raise RuntimeError(
            "Defender context shooter coordinate is non-finite."
        )


    raw_track_labels = (
        result.relative_teams.track_labels
    )


    if not isinstance(
        raw_track_labels,
        dict,
    ):

        raise RuntimeError(
            "Defender context requires RelativeTeamResult.track_labels."
        )


    track_labels = {
        int(track_id):
            str(label)

        for track_id, label
        in raw_track_labels.items()
    }


    defender_track_ids_all = sorted(
        int(track_id)

        for track_id, label
        in track_labels.items()

        if str(label)
        ==
        "DEFENDING"
    )


    projected_players = (
        result.pitch_projection.projected_players
    )


    if projected_players is None:

        raise RuntimeError(
            "Defender context requires projected players."
        )


    rows = []


    for player in projected_players:

        track_id = int(
            _notebook43_projected_player_value(
                player,
                "track_id",
            )
        )


        relative_team = (
            track_labels.get(
                track_id
            )
        )


        if relative_team != "DEFENDING":
            continue


        inside_pitch = bool(
            _notebook43_projected_player_value(
                player,
                "inside_pitch",
            )
        )


        if not inside_pitch:
            continue


        frame_number = int(
            _notebook43_projected_player_value(
                player,
                "frame_number",
            )
        )


        detection_id = int(
            _notebook43_projected_player_value(
                player,
                "detection_id",
            )
        )


        confidence = float(
            _notebook43_projected_player_value(
                player,
                "confidence",
            )
        )


        pitch_x = float(
            _notebook43_projected_player_value(
                player,
                "pitch_x",
            )
        )


        pitch_y = float(
            _notebook43_projected_player_value(
                player,
                "pitch_y",
            )
        )


        if not np.isfinite(
            [
                pitch_x,
                pitch_y,
            ]
        ).all():

            raise RuntimeError(
                "Projected defender contains non-finite pitch coordinates."
            )


        dx = float(
            pitch_x
            -
            shooter_xy[0]
        )


        dy = float(
            pitch_y
            -
            shooter_xy[1]
        )


        distance_to_shooter_m = float(
            np.hypot(
                dx,
                dy,
            )
        )


        rows.append(
            {
                "frame_number":
                    frame_number,

                "track_id":
                    track_id,

                "detection_id":
                    detection_id,

                "confidence":
                    confidence,

                "pitch_x":
                    pitch_x,

                "pitch_y":
                    pitch_y,

                "inside_pitch":
                    inside_pitch,

                "relative_team":
                    relative_team,

                "shooter_pitch_x":
                    float(
                        shooter_xy[0]
                    ),

                "shooter_pitch_y":
                    float(
                        shooter_xy[1]
                    ),

                "dx":
                    dx,

                "dy":
                    dy,

                "distance_to_shooter_m":
                    distance_to_shooter_m,
            }
        )


    defender_table = pd.DataFrame(
        rows
    )


    if len(
        defender_table
    ) == 0:

        raise RuntimeError(
            "No valid projected DEFENDING players are available."
        )


    defender_table = (
        defender_table
        .sort_values(
            [
                "distance_to_shooter_m",
                "track_id",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


    defender_table[
        "distance_rank"
    ] = (
        np.arange(
            len(
                defender_table
            ),
            dtype=int,
        )
        +
        1
    )


    nearest_row = (
        defender_table.iloc[
            0
        ]
    )


    nearest_defender_track_id = int(
        nearest_row[
            "track_id"
        ]
    )


    nearest_defender_distance_m = float(
        nearest_row[
            "distance_to_shooter_m"
        ]
    )


    defenders_within_3m = int(
        (
            defender_table[
                "distance_to_shooter_m"
            ]
            <=
            3.0
        )
        .sum()
    )


    defenders_within_5m = int(
        (
            defender_table[
                "distance_to_shooter_m"
            ]
            <=
            5.0
        )
        .sum()
    )


    defenders_within_10m = int(
        (
            defender_table[
                "distance_to_shooter_m"
            ]
            <=
            10.0
        )
        .sum()
    )


    return {
        "defender_track_ids_all":
            defender_track_ids_all,

        "defender_table":
            defender_table,

        "defenders_available":
            int(
                len(
                    defender_table
                )
            ),

        "nearest_defender_track_id":
            nearest_defender_track_id,

        "nearest_defender_distance_m":
            nearest_defender_distance_m,

        "defenders_within_3m":
            defenders_within_3m,

        "defenders_within_5m":
            defenders_within_5m,

        "defenders_within_10m":
            defenders_within_10m,
    }

def build_defender_context_handler():
    """
    Canonical Stage-15 defender context.
    """

    def handler(result):
        expected_prefix = list(
            CANONICAL_PIPELINE_STAGES[
                : _STAGE_INDEX[
                    "defender_context"
                ]
            ]
        )


        if (
            result.pipeline_status.completed_stages
            !=
            expected_prefix
        ):
            raise PipelineContractError(
                "defender_context requires completed shot_geometry."
            )


        if not result.pitch_projection.success:
            raise PipelineContractError(
                "Defender context requires successful PitchProjectionResult."
            )


        if not result.relative_teams.success:
            raise PipelineContractError(
                "Defender context requires successful RelativeTeamResult."
            )


        if not result.shot_geometry.success:
            raise PipelineContractError(
                "Defender context requires successful ShotGeometryResult."
            )


        context = (
            _notebook43_compute_defender_context(
                result
            )
        )


        nearest_field = (
            _notebook43_defender_context_nearest_field(
                result.defender_context
            )
        )


        result.defender_context.success = True


        setattr(
            result.defender_context,
            nearest_field,
            float(
                context[
                    "nearest_defender_distance_m"
                ]
            ),
        )


        result.defender_context.defenders_within_5m = int(
            context[
                "defenders_within_5m"
            ]
        )


        if hasattr(
            result.defender_context,
            "defender_track_ids",
        ):
            result.defender_context.defender_track_ids = [
                int(track_id)

                for track_id
                in context[
                    "defender_table"
                ][
                    "track_id"
                ]
                .astype(int)
                .sort_values()
                .tolist()
            ]


        result.defender_context.failure_reason = None


        result.metadata[
            "defender_context"
        ] = {
            "success":
                True,

            "method":
                "CURRENT_GEOMETRY_FRAME_RELATIVE_DEFENDER_EUCLIDEAN_DISTANCE",

            "shooter_pitch_xy":
                list(
                    result.shot_geometry
                    .shooter_pitch_xy
                ),

            "all_relative_defender_track_ids":
                list(
                    context[
                        "defender_track_ids_all"
                    ]
                ),

            "geometry_frame_defenders":
                int(
                    context[
                        "defenders_available"
                    ]
                ),

            "nearest_defender_track_id":
                int(
                    context[
                        "nearest_defender_track_id"
                    ]
                ),

            "nearest_defender_distance_m":
                float(
                    context[
                        "nearest_defender_distance_m"
                    ]
                ),

            "defenders_within_3m":
                int(
                    context[
                        "defenders_within_3m"
                    ]
                ),

            "defenders_within_5m":
                int(
                    context[
                        "defenders_within_5m"
                    ]
                ),

            "defenders_within_10m":
                int(
                    context[
                        "defenders_within_10m"
                    ]
                ),

            "inside_pitch_only":
                True,

            "current_pitch_projection_used":
                True,

            "current_relative_team_labels_used":
                True,

            "current_shot_geometry_used":
                True,

            "historical_defender_distances_used":
                False,

            "absolute_team_names_used":
                False,

            "reference_player_coordinates_used":
                False,

            "manual_defender_assignment_used":
                False,

            "_defender_table":
                context[
                    "defender_table"
                ],
        }


        return context


    return handler


def register_defender_context_handler(
    pipeline,
):
    pipeline.register_stage_handler(
        "defender_context",
        build_defender_context_handler(),
    )

    return pipeline



# NOTEBOOK43_CANONICAL_ENHANCED_XG_V1
# NOTEBOOK43_CANONICAL_ENHANCED_XG_FIX2_COMPLETE_PUBLIC_RESULT
# ====================================================================================================
# CANONICAL NOTEBOOK-24 ENHANCED xG
# ====================================================================================================

NOTEBOOK43_ENHANCED_XG_MODEL_FILENAME = (
    "notebook24_tracking_enhanced_xg_model.joblib"
)


NOTEBOOK43_ENHANCED_XG_FEATURES = [
    "distance_to_goal",
    "shot_angle",
    "nearest_defender_distance",
    "defenders_within_5m",
]


def _notebook43_xg_probability_field(
    xg_result,
):
    """
    Resolve the probability field from the frozen public XGResult schema.

    This adapter does not alter results.py.
    """

    for field_name in (
        "xg",
        "enhanced_xg",
        "xg_probability",
        "probability",
    ):
        if hasattr(
            xg_result,
            field_name,
        ):
            return field_name


    raise RuntimeError(
        "XGResult has no recognized public xG probability field."
    )


def _notebook43_enhanced_model_path():
    from pathlib import Path

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    return (
        project_root
        / "models"
        / NOTEBOOK43_ENHANCED_XG_MODEL_FILENAME
    )


def _notebook43_build_enhanced_xg_features(
    result,
):
    import numpy as np
    import pandas as pd


    if not result.shot_geometry.success:

        raise PipelineContractError(
            "Enhanced xG requires successful ShotGeometryResult."
        )


    if not result.defender_context.success:

        raise PipelineContractError(
            "Enhanced xG requires successful DefenderContextResult."
        )


    feature_row = {
        "distance_to_goal":
            float(
                result
                .shot_geometry
                .distance_to_goal_m
            ),

        "shot_angle":
            float(
                result
                .shot_geometry
                .shot_angle_deg
            ),

        "nearest_defender_distance":
            float(
                result
                .defender_context
                .nearest_defender_m
            ),

        "defenders_within_5m":
            int(
                result
                .defender_context
                .defenders_within_5m
            ),
    }


    feature_table = pd.DataFrame(
        [
            feature_row
        ],
        columns=
            NOTEBOOK43_ENHANCED_XG_FEATURES,
    )


    feature_matrix = (
        feature_table
        .to_numpy(
            dtype=float
        )
    )


    if feature_matrix.shape != (1, 4):

        raise RuntimeError(
            "Enhanced xG feature matrix must have shape (1, 4)."
        )


    if not np.isfinite(
        feature_matrix
    ).all():

        raise RuntimeError(
            "Enhanced xG features contain non-finite values."
        )


    return feature_table


def _notebook43_predict_enhanced_xg(
    result,
):
    import joblib
    import numpy as np


    model_path = (
        _notebook43_enhanced_model_path()
    )


    if not model_path.exists():

        raise RuntimeError(
            f"Enhanced xG model not found: {model_path}"
        )


    model = joblib.load(
        model_path
    )


    if not hasattr(
        model,
        "predict_proba",
    ):

        raise RuntimeError(
            "Enhanced xG model does not expose predict_proba()."
        )


    feature_table = (
        _notebook43_build_enhanced_xg_features(
            result
        )
    )


    if hasattr(
        model,
        "feature_names_in_",
    ):

        model_feature_names = [
            str(value)

            for value
            in model
            .feature_names_in_
            .tolist()
        ]


        if (
            model_feature_names
            !=
            NOTEBOOK43_ENHANCED_XG_FEATURES
        ):

            raise RuntimeError(
                "Enhanced xG model feature-name/order contract mismatch."
            )


    if not hasattr(
        model,
        "classes_",
    ):

        raise RuntimeError(
            "Enhanced xG model has no classes_ attribute."
        )


    classes = [
        int(value)

        for value
        in np.asarray(
            model.classes_
        ).tolist()
    ]


    if 1 not in classes:

        raise RuntimeError(
            "Enhanced xG model has no positive class 1."
        )


    positive_class_index = (
        classes.index(
            1
        )
    )


    probabilities = (
        model.predict_proba(
            feature_table
        )
    )


    if probabilities.shape[0] != 1:

        raise RuntimeError(
            "Enhanced xG model returned unexpected prediction shape."
        )


    xg_probability = float(
        probabilities[
            0,
            positive_class_index,
        ]
    )


    if not np.isfinite(
        xg_probability
    ):

        raise RuntimeError(
            "Enhanced xG probability is non-finite."
        )


    if not (
        0.0
        <=
        xg_probability
        <=
        1.0
    ):

        raise RuntimeError(
            "Enhanced xG probability is outside [0, 1]."
        )


    return {
        "feature_table":
            feature_table,

        "model_path":
            model_path,

        "model":
            model,

        "model_classes":
            classes,

        "positive_class_index":
            positive_class_index,

        "xg_probability":
            xg_probability,
    }


def build_enhanced_xg_handler():
    """
    Canonical Stage-16 enhanced xG.

    Production inputs:
      - current ShotGeometryResult
      - current DefenderContextResult
      - frozen Notebook-24 serialized enhanced xG model

    The frozen Notebook-42 prediction is regression-only and is
    never used to produce the result.
    """

    def handler(result):

        expected_prefix = list(
            CANONICAL_PIPELINE_STAGES[
                : _STAGE_INDEX[
                    "enhanced_xg"
                ]
            ]
        )


        if (
            result.pipeline_status.completed_stages
            !=
            expected_prefix
        ):

            raise PipelineContractError(
                "enhanced_xg requires completed defender_context."
            )


        prediction = (
            _notebook43_predict_enhanced_xg(
                result
            )
        )


        probability_field = (
            _notebook43_xg_probability_field(
                result.xg
            )
        )


        setattr(
            result.xg,
            probability_field,
            float(
                prediction[
                    "xg_probability"
                ]
            ),
        )


        # Complete the public XGResult contract.
        #
        # probability was produced by fresh predict_proba inference.
        # percent and feature_values are deterministic derivatives
        # of that same current production inference/input row.
        if hasattr(
            result.xg,
            "percent",
        ):

            result.xg.percent = float(
                prediction[
                    "xg_probability"
                ]
                *
                100.0
            )


        if hasattr(
            result.xg,
            "feature_values",
        ):

            raw_feature_values = (
                prediction[
                    "feature_table"
                ]
                .iloc[
                    0
                ]
                .to_dict()
            )


            result.xg.feature_values = {
                "distance_to_goal":
                    float(
                        raw_feature_values[
                            "distance_to_goal"
                        ]
                    ),

                "shot_angle":
                    float(
                        raw_feature_values[
                            "shot_angle"
                        ]
                    ),

                "nearest_defender_distance":
                    float(
                        raw_feature_values[
                            "nearest_defender_distance"
                        ]
                    ),

                "defenders_within_5m":
                    int(
                        raw_feature_values[
                            "defenders_within_5m"
                        ]
                    ),
            }


        result.xg.success = True

        result.xg.failure_reason = None


        # Populate optional informational fields only if they
        # already exist in the frozen public schema.
        if hasattr(
            result.xg,
            "model_name",
        ):

            result.xg.model_name = (
                NOTEBOOK43_ENHANCED_XG_MODEL_FILENAME
            )


        if hasattr(
            result.xg,
            "model_path",
        ):

            result.xg.model_path = str(
                prediction[
                    "model_path"
                ]
            )


        result.metadata[
            "enhanced_xg"
        ] = {
            "success":
                True,

            "method":
                "NOTEBOOK24_SERIALIZED_ENHANCED_MODEL_PREDICT_PROBA",

            "model_path":
                str(
                    prediction[
                        "model_path"
                    ]
                ),

            "model_filename":
                NOTEBOOK43_ENHANCED_XG_MODEL_FILENAME,

            "feature_names":
                list(
                    NOTEBOOK43_ENHANCED_XG_FEATURES
                ),

            "feature_values":
                {
                    str(key):
                        float(value)

                    for key, value
                    in prediction[
                        "feature_table"
                    ]
                    .iloc[
                        0
                    ]
                    .to_dict()
                    .items()
                },

            "model_classes":
                list(
                    prediction[
                        "model_classes"
                    ]
                ),

            "positive_class_index":
                int(
                    prediction[
                        "positive_class_index"
                    ]
                ),

            "xg_probability":
                float(
                    prediction[
                        "xg_probability"
                    ]
                ),

            "current_shot_geometry_used":
                True,

            "current_defender_context_used":
                True,

            "historical_xg_prediction_used":
                False,

            "notebook41_xg_used":
                False,

            "notebook42_xg_used_as_production_input":
                False,

            "training_performed":
                False,

            "_feature_table":
                prediction[
                    "feature_table"
                ],
        }


        return {
            "feature_table":
                prediction[
                    "feature_table"
                ],

            "model_path":
                prediction[
                    "model_path"
                ],

            "model_classes":
                prediction[
                    "model_classes"
                ],

            "positive_class_index":
                prediction[
                    "positive_class_index"
                ],

            "probability_field":
                probability_field,

            "xg_probability":
                prediction[
                    "xg_probability"
                ],
        }


    return handler


def register_enhanced_xg_handler(
    pipeline,
):

    pipeline.register_stage_handler(
        "enhanced_xg",
        build_enhanced_xg_handler(),
    )


    return pipeline



# NOTEBOOK43_CANONICAL_ANNOTATION_V1
# ====================================================================================================
# CANONICAL FULL-CLIP ANNOTATION
# ====================================================================================================


def _notebook43_annotation_value(
    value,
    key,
):
    """
    Read one field from either:
      - public dictionary records, or
      - dataclass/object records.
    """

    if isinstance(
        value,
        dict,
    ):

        if key not in value:

            raise RuntimeError(
                f"Annotation record missing key: {key}"
            )

        return value[
            key
        ]


    if hasattr(
        value,
        key,
    ):

        return getattr(
            value,
            key,
        )


    raise RuntimeError(
        f"Annotation record has no field/key: {key}"
    )


def _notebook43_player_observations(
    result,
):
    """
    Recover the current 107 tracked player observations.
    """

    tracks = (
        result.player_tracks.tracks
    )


    if isinstance(
        tracks,
        dict,
    ):

        track_values = list(
            tracks.values()
        )

    else:

        track_values = list(
            tracks
        )


    rows = []


    for track in track_values:

        track_id = int(
            _notebook43_annotation_value(
                track,
                "track_id",
            )
        )


        observations = (
            _notebook43_annotation_value(
                track,
                "observations",
            )
        )


        for observation in observations:

            frame_idx = int(
                _notebook43_annotation_value(
                    observation,
                    "frame_idx",
                )
            )


            detection_idx = int(
                _notebook43_annotation_value(
                    observation,
                    "detection_idx",
                )
            )


            rows.append(
                {
                    "track_id":
                        track_id,

                    "frame_idx":
                        frame_idx,

                    "detection_id":
                        detection_idx,

                    "x1":
                        float(
                            _notebook43_annotation_value(
                                observation,
                                "x1",
                            )
                        ),

                    "y1":
                        float(
                            _notebook43_annotation_value(
                                observation,
                                "y1",
                            )
                        ),

                    "x2":
                        float(
                            _notebook43_annotation_value(
                                observation,
                                "x2",
                            )
                        ),

                    "y2":
                        float(
                            _notebook43_annotation_value(
                                observation,
                                "y2",
                            )
                        ),

                    "confidence":
                        float(
                            _notebook43_annotation_value(
                                observation,
                                "confidence",
                            )
                        ),
                }
            )


    import pandas as pd


    table = pd.DataFrame(
        rows
    )


    if len(
        table
    ) == 0:

        raise RuntimeError(
            "Annotation requires current player-track observations."
        )


    return (
        table
        .sort_values(
            [
                "frame_idx",
                "track_id",
                "detection_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def _notebook43_ball_annotation_rows(
    result,
):
    """
    Recover current six-point canonical ball trajectory.
    """

    import pandas as pd


    rows = []


    for point in result.ball_track.track:

        rows.append(
            {
                "frame_idx":
                    int(
                        _notebook43_annotation_value(
                            point,
                            "frame_idx",
                        )
                    ),

                "center_x":
                    float(
                        _notebook43_annotation_value(
                            point,
                            "center_x",
                        )
                    ),

                "center_y":
                    float(
                        _notebook43_annotation_value(
                            point,
                            "center_y",
                        )
                    ),

                "trajectory_source":
                    str(
                        _notebook43_annotation_value(
                            point,
                            "trajectory_source",
                        )
                    ),

                "is_raw_v3_detection":
                    bool(
                        _notebook43_annotation_value(
                            point,
                            "is_raw_v3_detection",
                        )
                    ),

                "is_interpolated":
                    bool(
                        _notebook43_annotation_value(
                            point,
                            "is_interpolated",
                        )
                    ),
            }
        )


    table = pd.DataFrame(
        rows
    )


    if len(
        table
    ) == 0:

        raise RuntimeError(
            "Annotation requires current ball trajectory."
        )


    return (
        table
        .sort_values(
            "frame_idx"
        )
        .reset_index(
            drop=True
        )
    )


def _notebook43_draw_label(
    frame,
    text,
    x,
    y,
    color,
    scale=0.45,
    thickness=1,
):
    import cv2


    font = cv2.FONT_HERSHEY_SIMPLEX


    (
        text_width,
        text_height,
    ), baseline = cv2.getTextSize(
        str(
            text
        ),
        font,
        scale,
        thickness,
    )


    x = int(
        max(
            0,
            x,
        )
    )


    y = int(
        max(
            text_height
            +
            4,
            y,
        )
    )


    x2 = int(
        min(
            frame.shape[
                1
            ]
            -
            1,

            x
            +
            text_width
            +
            6,
        )
    )


    y1 = int(
        max(
            0,
            y
            -
            text_height
            -
            5,
        )
    )


    y2 = int(
        min(
            frame.shape[
                0
            ]
            -
            1,

            y
            +
            baseline
            +
            3,
        )
    )


    cv2.rectangle(
        frame,
        (
            x,
            y1,
        ),
        (
            x2,
            y2,
        ),
        (
            0,
            0,
            0,
        ),
        -1,
    )


    cv2.putText(
        frame,
        str(
            text
        ),
        (
            x
            +
            3,
            y
            -
            2,
        ),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _notebook43_draw_annotation_hud(
    frame,
    result,
):
    import cv2


    panel_x1 = 18
    panel_y1 = 18
    panel_x2 = 455
    panel_y2 = 218


    overlay = frame.copy()


    cv2.rectangle(
        overlay,
        (
            panel_x1,
            panel_y1,
        ),
        (
            panel_x2,
            panel_y2,
        ),
        (
            10,
            10,
            10,
        ),
        -1,
    )


    cv2.addWeighted(
        overlay,
        0.70,
        frame,
        0.30,
        0.0,
        frame,
    )


    probability = float(
        result.xg.probability
    )


    if (
        hasattr(
            result.xg,
            "percent",
        )
        and
        result.xg.percent
        is not None
    ):

        percent = float(
            result.xg.percent
        )

    else:

        percent = float(
            probability
            *
            100.0
        )


    lines = [
        (
            "SOCCER ML",
            0.62,
            2,
        ),

        (
            f"Shooter: Track {int(result.shooter.track_id)}",
            0.50,
            1,
        ),

        (
            f"Target goal: {result.target_goal.target_goal}",
            0.50,
            1,
        ),

        (
            f"Distance: {float(result.shot_geometry.distance_to_goal_m):.2f} m",
            0.50,
            1,
        ),

        (
            f"Shot angle: {float(result.shot_geometry.shot_angle_deg):.2f} deg",
            0.50,
            1,
        ),

        (
            f"Nearest defender: {float(result.defender_context.nearest_defender_m):.2f} m",
            0.50,
            1,
        ),

        (
            f"Defenders <= 5 m: {int(result.defender_context.defenders_within_5m)}",
            0.50,
            1,
        ),

        (
            f"Enhanced xG: {percent:.2f}%",
            0.58,
            2,
        ),
    ]


    y = 45


    for text, scale, thickness in lines:

        cv2.putText(
            frame,
            text,
            (
                34,
                y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (
                255,
                255,
                255,
            ),
            thickness,
            cv2.LINE_AA,
        )


        y += 23


def _notebook43_render_annotation(
    result,
    output_path,
):
    import cv2
    import numpy as np
    from pathlib import Path


    input_path = Path(
        result.video_path
    )


    output_path = Path(
        output_path
    )


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    if not input_path.exists():

        raise RuntimeError(
            f"Annotation input video does not exist: {input_path}"
        )


    player_table = (
        _notebook43_player_observations(
            result
        )
    )


    ball_table = (
        _notebook43_ball_annotation_rows(
            result
        )
    )


    track_labels = {
        int(track_id):
            str(label)

        for track_id, label
        in result.relative_teams
        .track_labels
        .items()
    }


    shooter_track_id = int(
        result.shooter.track_id
    )


    defender_metadata = (
        result.metadata.get(
            "defender_context",
            {},
        )
    )


    if (
        "nearest_defender_track_id"
        not in
        defender_metadata
    ):

        raise RuntimeError(
            "Annotation requires nearest_defender_track_id from defender-context metadata."
        )


    nearest_defender_track_id = int(
        defender_metadata[
            "nearest_defender_track_id"
        ]
    )


    players_by_frame = {
        int(frame_idx):
            group.copy()

        for frame_idx, group
        in player_table.groupby(
            "frame_idx",
            sort=True,
        )
    }


    balls_by_frame = {
        int(row.frame_idx):
            row

        for row
        in ball_table.itertuples(
            index=False
        )
    }


    capture = cv2.VideoCapture(
        str(
            input_path
        )
    )


    if not capture.isOpened():

        raise RuntimeError(
            f"Could not open annotation input video: {input_path}"
        )


    fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )


    width = int(
        round(
            capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )
    )


    height = int(
        round(
            capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )
    )


    if (
        not np.isfinite(
            fps
        )
        or
        fps <= 0.0
        or
        width <= 0
        or
        height <= 0
    ):

        capture.release()

        raise RuntimeError(
            "Annotation input video metadata is invalid."
        )


    if output_path.exists():

        output_path.unlink()


    writer = cv2.VideoWriter(
        str(
            output_path
        ),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (
            width,
            height,
        ),
    )


    if not writer.isOpened():

        capture.release()

        raise RuntimeError(
            f"Could not open annotation video writer: {output_path}"
        )


    team_colors = {
        "ATTACKING":
            (
                0,
                220,
                0,
            ),

        "DEFENDING":
            (
                0,
                80,
                255,
            ),

        "OTHER":
            (
                200,
                200,
                200,
            ),
    }


    coverage = {
        "frames_written":
            0,

        "player_boxes_drawn":
            0,

        "attacking_boxes_drawn":
            0,

        "defending_boxes_drawn":
            0,

        "other_boxes_drawn":
            0,

        "shooter_boxes_drawn":
            0,

        "nearest_defender_boxes_drawn":
            0,

        "ball_markers_drawn":
            0,

        "raw_ball_markers_drawn":
            0,

        "interpolated_ball_markers_drawn":
            0,

        "hud_frames_drawn":
            0,
    }


    frame_idx = 0


    try:

        while True:

            ok, frame = capture.read()


            if not ok:

                break


            if frame_idx in players_by_frame:

                frame_players = (
                    players_by_frame[
                        frame_idx
                    ]
                )


                for row in frame_players.itertuples(
                    index=False
                ):

                    track_id = int(
                        row.track_id
                    )


                    label = (
                        track_labels.get(
                            track_id,
                            "OTHER",
                        )
                    )


                    color = (
                        team_colors.get(
                            label,
                            team_colors[
                                "OTHER"
                            ],
                        )
                    )


                    x1 = int(
                        round(
                            row.x1
                        )
                    )

                    y1 = int(
                        round(
                            row.y1
                        )
                    )

                    x2 = int(
                        round(
                            row.x2
                        )
                    )

                    y2 = int(
                        round(
                            row.y2
                        )
                    )


                    x1 = max(
                        0,
                        min(
                            width - 1,
                            x1,
                        ),
                    )

                    x2 = max(
                        0,
                        min(
                            width - 1,
                            x2,
                        ),
                    )

                    y1 = max(
                        0,
                        min(
                            height - 1,
                            y1,
                        ),
                    )

                    y2 = max(
                        0,
                        min(
                            height - 1,
                            y2,
                        ),
                    )


                    box_thickness = (
                        2
                    )


                    if track_id == shooter_track_id:

                        box_thickness = 4

                        color = (
                            0,
                            255,
                            255,
                        )


                    elif (
                        track_id
                        ==
                        nearest_defender_track_id
                    ):

                        box_thickness = 4

                        color = (
                            255,
                            0,
                            255,
                        )


                    cv2.rectangle(
                        frame,
                        (
                            x1,
                            y1,
                        ),
                        (
                            x2,
                            y2,
                        ),
                        color,
                        box_thickness,
                    )


                    label_text = (
                        f"T{track_id} {label}"
                    )


                    if track_id == shooter_track_id:

                        label_text += (
                            " SHOOTER"
                        )


                    if (
                        track_id
                        ==
                        nearest_defender_track_id
                    ):

                        label_text += (
                            " NEAREST"
                        )


                    _notebook43_draw_label(
                        frame,
                        label_text,
                        x1,
                        y1 - 4,
                        color,
                        scale=0.40,
                        thickness=1,
                    )


                    coverage[
                        "player_boxes_drawn"
                    ] += 1


                    if label == "ATTACKING":

                        coverage[
                            "attacking_boxes_drawn"
                        ] += 1


                    elif label == "DEFENDING":

                        coverage[
                            "defending_boxes_drawn"
                        ] += 1


                    else:

                        coverage[
                            "other_boxes_drawn"
                        ] += 1


                    if (
                        track_id
                        ==
                        shooter_track_id
                    ):

                        coverage[
                            "shooter_boxes_drawn"
                        ] += 1


                    if (
                        track_id
                        ==
                        nearest_defender_track_id
                    ):

                        coverage[
                            "nearest_defender_boxes_drawn"
                        ] += 1


            if frame_idx in balls_by_frame:

                ball = (
                    balls_by_frame[
                        frame_idx
                    ]
                )


                center = (
                    int(
                        round(
                            ball.center_x
                        )
                    ),

                    int(
                        round(
                            ball.center_y
                        )
                    ),
                )


                if bool(
                    ball.is_interpolated
                ):

                    ball_color = (
                        255,
                        255,
                        0,
                    )

                    ball_text = (
                        "BALL INTERP"
                    )


                    coverage[
                        "interpolated_ball_markers_drawn"
                    ] += 1


                else:

                    ball_color = (
                        0,
                        165,
                        255,
                    )

                    ball_text = (
                        "BALL RAW"
                    )


                    coverage[
                        "raw_ball_markers_drawn"
                    ] += 1


                cv2.circle(
                    frame,
                    center,
                    7,
                    ball_color,
                    -1,
                    cv2.LINE_AA,
                )


                cv2.circle(
                    frame,
                    center,
                    11,
                    (
                        255,
                        255,
                        255,
                    ),
                    2,
                    cv2.LINE_AA,
                )


                _notebook43_draw_label(
                    frame,
                    ball_text,
                    center[0] + 12,
                    center[1] - 8,
                    ball_color,
                    scale=0.40,
                    thickness=1,
                )


                coverage[
                    "ball_markers_drawn"
                ] += 1


            _notebook43_draw_annotation_hud(
                frame,
                result,
            )


            coverage[
                "hud_frames_drawn"
            ] += 1


            writer.write(
                frame
            )


            coverage[
                "frames_written"
            ] += 1


            frame_idx += 1


    finally:

        capture.release()

        writer.release()


    if not output_path.exists():

        raise RuntimeError(
            "Annotation output was not created."
        )


    if output_path.stat().st_size <= 0:

        raise RuntimeError(
            "Annotation output is empty."
        )


    return {
        "output_path":
            output_path,

        "frame_count":
            int(
                coverage[
                    "frames_written"
                ]
            ),

        "fps":
            fps,

        "width":
            width,

        "height":
            height,

        "coverage":
            coverage,

        "player_table":
            player_table,

        "ball_table":
            ball_table,

        "nearest_defender_track_id":
            nearest_defender_track_id,
    }


def build_annotation_handler(
    output_path=None,
):
    """
    Canonical Stage-17 annotation.

    Analytical values are READ-ONLY.

    The stage renders the complete input clip using current
    pipeline results only.
    """

    def handler(result):

        expected_prefix = list(
            CANONICAL_PIPELINE_STAGES[
                : _STAGE_INDEX[
                    "annotation"
                ]
            ]
        )


        if (
            result.pipeline_status.completed_stages
            !=
            expected_prefix
        ):

            raise PipelineContractError(
                "annotation requires completed enhanced_xg."
            )


        required_components = [
            result.ball_track,
            result.player_tracks,
            result.pitch_projection,
            result.shooter,
            result.relative_teams,
            result.target_goal,
            result.shot_geometry,
            result.defender_context,
            result.xg,
        ]


        if not all(
            bool(
                component.success
            )

            for component
            in required_components
        ):

            raise PipelineContractError(
                "Annotation requires all analytical components to be successful."
            )


        from pathlib import Path


        if output_path is None:

            input_path = Path(
                result.video_path
            )


            final_output_path = (
                input_path.parent
                /
                (
                    input_path.stem
                    +
                    "_soccer_ml_annotated.mp4"
                )
            )


        else:

            final_output_path = Path(
                output_path
            )


        rendered = (
            _notebook43_render_annotation(
                result,
                final_output_path,
            )
        )


        result.annotated_clip.success = True

        result.annotated_clip.output_path = str(
            rendered[
                "output_path"
            ]
        )

        result.annotated_clip.frame_count = int(
            rendered[
                "frame_count"
            ]
        )

        result.annotated_clip.failure_reason = None


        result.metadata[
            "annotation"
        ] = {
            "success":
                True,

            "method":
                "CURRENT_PIPELINE_FULL_CLIP_OVERLAY",

            "output_path":
                str(
                    rendered[
                        "output_path"
                    ]
                ),

            "frame_count":
                int(
                    rendered[
                        "frame_count"
                    ]
                ),

            "fps":
                float(
                    rendered[
                        "fps"
                    ]
                ),

            "width":
                int(
                    rendered[
                        "width"
                    ]
                ),

            "height":
                int(
                    rendered[
                        "height"
                    ]
                ),

            "nearest_defender_track_id":
                int(
                    rendered[
                        "nearest_defender_track_id"
                    ]
                ),

            "coverage":
                {
                    str(key):
                        int(value)

                    for key, value
                    in rendered[
                        "coverage"
                    ].items()
                },

            "current_ball_track_used":
                True,

            "current_player_tracks_used":
                True,

            "current_relative_team_labels_used":
                True,

            "current_shooter_used":
                True,

            "current_defender_context_used":
                True,

            "current_target_goal_used":
                True,

            "current_shot_geometry_used":
                True,

            "current_xg_used":
                True,

            "historical_annotation_used":
                False,

            "notebook41_annotation_used":
                False,

            "notebook42_annotation_copied":
                False,

            "_player_table":
                rendered[
                    "player_table"
                ],

            "_ball_table":
                rendered[
                    "ball_table"
                ],
        }


        return rendered


    return handler


def register_annotation_handler(
    pipeline,
    output_path=None,
):

    pipeline.register_stage_handler(
        "annotation",
        build_annotation_handler(
            output_path=
                output_path,
        ),
    )


    return pipeline



# NOTEBOOK43_CANONICAL_OUTPUT_VALIDATION_V1
# ====================================================================================================
# FINAL CANONICAL OUTPUT VALIDATION + HIGH-LEVEL RUNNER
# ====================================================================================================


def _notebook43_video_contract(
    video_path,
):
    """
    Decode a video fully and return its observable media contract.
    """

    import cv2
    import numpy as np

    from pathlib import Path


    path = Path(
        video_path
    )


    if not path.exists():

        raise RuntimeError(
            f"Video does not exist: {path}"
        )


    capture = cv2.VideoCapture(
        str(
            path
        )
    )


    if not capture.isOpened():

        raise RuntimeError(
            f"Video is not decodable: {path}"
        )


    fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )


    width = int(
        round(
            capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )
    )


    height = int(
        round(
            capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )
    )


    header_frame_count = int(
        round(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )
    )


    decoded_frame_count = 0


    while True:

        ok, _ = capture.read()

        if not ok:
            break

        decoded_frame_count += 1


    capture.release()


    if (
        not np.isfinite(
            fps
        )
        or
        fps <= 0.0
        or
        width <= 0
        or
        height <= 0
        or
        decoded_frame_count <= 0
    ):

        raise RuntimeError(
            f"Invalid decoded video contract: {path}"
        )


    return {
        "path":
            str(
                path
            ),

        "header_frame_count":
            header_frame_count,

        "decoded_frame_count":
            decoded_frame_count,

        "fps":
            fps,

        "width":
            width,

        "height":
            height,
    }


def build_output_validation_handler():
    """
    Canonical Stage-18 output validation.

    This stage:
      - validates all public analytical components,
      - validates the complete annotated video,
      - does not recompute any prediction,
      - does not change any analytical component.
    """

    def handler(result):

        expected_prefix = list(
            CANONICAL_PIPELINE_STAGES[
                : _STAGE_INDEX[
                    "output_validation"
                ]
            ]
        )


        if (
            result.pipeline_status.completed_stages
            !=
            expected_prefix
        ):

            raise PipelineContractError(
                "output_validation requires completed annotation."
            )


        required_components = [
            result.ball_track,
            result.player_tracks,
            result.pitch_projection,
            result.shooter,
            result.relative_teams,
            result.target_goal,
            result.shot_geometry,
            result.defender_context,
            result.xg,
            result.annotated_clip,
        ]


        for component in required_components:

            if not bool(
                getattr(
                    component,
                    "success",
                    False,
                )
            ):

                raise PipelineContractError(
                    "output_validation requires every public component to succeed."
                )


            if (
                getattr(
                    component,
                    "failure_reason",
                    None,
                )
                is not None
            ):

                raise PipelineContractError(
                    "A successful public component contains a failure_reason."
                )


        if result.annotated_clip.output_path is None:

            raise RuntimeError(
                "AnnotationResult.output_path is missing."
            )


        input_contract = (
            _notebook43_video_contract(
                result.video_path
            )
        )


        output_contract = (
            _notebook43_video_contract(
                result.annotated_clip.output_path
            )
        )


        frame_count_match = (
            input_contract[
                "decoded_frame_count"
            ]
            ==
            output_contract[
                "decoded_frame_count"
            ]
        )


        annotation_count_match = (
            int(
                result
                .annotated_clip
                .frame_count
            )
            ==
            output_contract[
                "decoded_frame_count"
            ]
        )


        fps_match = bool(
            np.isclose(
                input_contract[
                    "fps"
                ],
                output_contract[
                    "fps"
                ],
                rtol=0.0,
                atol=5e-4,
            )
        )


        resolution_match = all(
            [
                input_contract[
                    "width"
                ]
                ==
                output_contract[
                    "width"
                ],

                input_contract[
                    "height"
                ]
                ==
                output_contract[
                    "height"
                ],
            ]
        )


        if not all(
            [
                frame_count_match,
                annotation_count_match,
                fps_match,
                resolution_match,
            ]
        ):

            raise RuntimeError(
                "Final annotated-video contract failed."
            )


        validation = {
            "all_public_components_success":
                True,

            "input_video_contract":
                input_contract,

            "output_video_contract":
                output_contract,

            "frame_count_match":
                frame_count_match,

            "annotation_frame_count_match":
                annotation_count_match,

            "fps_match":
                fps_match,

            "resolution_match":
                resolution_match,

            "final_video_contract_valid":
                True,
        }


        result.metadata[
            "output_validation"
        ] = {
            "success":
                True,

            **validation,

            "analytical_recomputation_performed":
                False,

            "historical_result_used":
                False,

            "reference_homography_used":
                False,

            "manual_shooter_assignment_used":
                False,
        }


        return validation


    return handler


def register_output_validation_handler(
    pipeline,
):

    pipeline.register_stage_handler(
        "output_validation",
        build_output_validation_handler(),
    )


    return pipeline


NOTEBOOK43_CANONICAL_STAGE_BUILDERS = {
    "input_validation":
        "build_input_validation_handler",

    "artifact_validation":
        "build_artifact_validation_handler",

    "video_metadata":
        "build_video_metadata_handler",

    "ball_detection":
        "build_ball_detection_handler",

    "ball_tracking":
        "build_ball_tracking_handler",

    "player_detection_tracking":
        "build_player_detection_tracking_handler",

    "pitch_line_detection":
        "build_pitch_line_detection_handler",

    "homography_search":
        "build_homography_search_handler",

    "homography_validation":
        "build_homography_validation_handler",

    "pitch_projection":
        "build_pitch_projection_handler",

    "shooter_assignment":
        "build_shooter_assignment_handler",

    "relative_team_assignment":
        "build_relative_team_assignment_handler",

    "target_goal_inference":
        "build_target_goal_inference_handler",

    "shot_geometry":
        "build_shot_geometry_handler",

    "defender_context":
        "build_defender_context_handler",

    "enhanced_xg":
        "build_enhanced_xg_handler",

    "annotation":
        "build_annotation_handler",

    "output_validation":
        "build_output_validation_handler",
}


def canonical_api_stage_plan():
    """
    Public read-only description of the reusable 18-stage API plan.
    """

    return [
        {
            "stage_index":
                int(
                    index
                ),

            "stage":
                str(
                    stage
                ),

            "builder":
                str(
                    NOTEBOOK43_CANONICAL_STAGE_BUILDERS[
                        stage
                    ]
                ),
        }

        for index, stage
        in enumerate(
            CANONICAL_PIPELINE_STAGES,
            start=1,
        )
    ]


def _notebook43_build_stage_handler(
    stage,
    annotated_output_path=None,
):
    """
    Resolve one already-certified canonical stage builder.
    """

    if (
        stage
        not in
        NOTEBOOK43_CANONICAL_STAGE_BUILDERS
    ):

        raise PipelineContractError(
            f"No canonical handler builder registered for stage: {stage}"
        )


    builder_name = (
        NOTEBOOK43_CANONICAL_STAGE_BUILDERS[
            stage
        ]
    )


    builder = globals().get(
        builder_name
    )


    if builder is None:

        raise PipelineContractError(
            f"Canonical builder is unavailable: {builder_name}"
        )


    if stage == "annotation":

        return builder(
            output_path=
                annotated_output_path
        )


    return builder()


def run_canonical_clip_pipeline(
    video_path,
    annotated_output_path=None,
):
    """
    Run the complete certified 18-stage Soccer ML pipeline.

    Parameters
    ----------
    video_path:
        Path to one local soccer clip.

    annotated_output_path:
        Optional destination for the annotated MP4.

    Returns
    -------
    SoccerClipAnalysisResult
        Fully populated public structured result.

    Notes
    -----
    This function uses the same packaged stage handlers certified
    independently throughout Notebook 43.
    """

    from pathlib import Path

    from .results import SoccerClipAnalysisResult


    normalized_video_path = Path(
        video_path
    )


    result = SoccerClipAnalysisResult(
        video_path=
            str(
                normalized_video_path
            )
    )


    for stage in CANONICAL_PIPELINE_STAGES:

        try:

            handler = (
                _notebook43_build_stage_handler(
                    stage,
                    annotated_output_path=
                        annotated_output_path,
                )
            )


            handler(
                result
            )


            mark_stage_complete(
                result.pipeline_status,
                stage,
            )


        except Exception as exc:

            result.pipeline_status.success = False

            result.pipeline_status.degraded = False

            result.pipeline_status.failed_stage = (
                stage
            )


            message = (
                f"{stage}: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )


            if (
                message
                not in
                result.pipeline_status.warnings
            ):

                result.pipeline_status.warnings.append(
                    message
                )


            raise


    result.pipeline_status.success = True

    result.pipeline_status.degraded = False

    result.pipeline_status.failed_stage = None


    if not validate_pipeline_status(
        result.pipeline_status
    ):

        raise PipelineContractError(
            "Final reusable API pipeline status is invalid."
        )


    return result



# NOTEBOOK43_CANONICAL_DEPLOYMENT_ADAPTERS_FIX2
# ====================================================================================================
# APPLICATION DEPLOYMENT ADAPTERS
# ====================================================================================================
#
# These constants describe serialized production artifacts.
#
# They are deployment configuration, not historical analytical output.
#
# ====================================================================================================


NOTEBOOK43_DEFAULT_V3_BALL_MODEL_RELATIVE_PATH = (
    "models/v3/best.pt"
)


NOTEBOOK43_DEFAULT_V3_BALL_MODEL_SHA256 = (
    "bf5e8b1d1227655dc7a6ac34e0850fd50327466febd1ffa27531fac0ce5a9d05"
)


NOTEBOOK43_DEFAULT_PLAYER_MODEL_RELATIVE_PATH = (
    "notebooks/yolo11n.pt"
)


NOTEBOOK43_DEFAULT_PLAYER_MODEL_SHA256 = (
    "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
)


NOTEBOOK43_DEFAULT_ENHANCED_XG_MODEL_RELATIVE_PATH = (
    "models/notebook24_tracking_enhanced_xg_model.joblib"
)


NOTEBOOK43_DEFAULT_ENHANCED_XG_MODEL_SHA256 = (
    "fd40d1dbf4c0bc753d5f2ddde53794befe69b607ceef00798c9f0754b7eb026c"
)


def _notebook43_project_root():
    from pathlib import Path

    return (
        Path(__file__)
        .resolve()
        .parents[2]
    )


def _notebook43_sha256_file(
    path,
):
    import hashlib


    digest = hashlib.sha256()


    with open(
        path,
        "rb",
    ) as file:

        while True:

            block = file.read(
                1024 * 1024
            )


            if not block:

                break


            digest.update(
                block
            )


    return digest.hexdigest()


def _notebook43_default_deployment_artifacts():
    from pathlib import Path


    project_root = (
        _notebook43_project_root()
    )


    return [
        {
            "artifact_name":
                "v3_ball_model",

            "relative_path":
                NOTEBOOK43_DEFAULT_V3_BALL_MODEL_RELATIVE_PATH,

            "path":
                project_root
                /
                NOTEBOOK43_DEFAULT_V3_BALL_MODEL_RELATIVE_PATH,

            "expected_sha256":
                NOTEBOOK43_DEFAULT_V3_BALL_MODEL_SHA256,
        },

        {
            "artifact_name":
                "player_detector",

            "relative_path":
                NOTEBOOK43_DEFAULT_PLAYER_MODEL_RELATIVE_PATH,

            "path":
                project_root
                /
                NOTEBOOK43_DEFAULT_PLAYER_MODEL_RELATIVE_PATH,

            "expected_sha256":
                NOTEBOOK43_DEFAULT_PLAYER_MODEL_SHA256,
        },

        {
            "artifact_name":
                "enhanced_xg_model",

            "relative_path":
                NOTEBOOK43_DEFAULT_ENHANCED_XG_MODEL_RELATIVE_PATH,

            "path":
                project_root
                /
                NOTEBOOK43_DEFAULT_ENHANCED_XG_MODEL_RELATIVE_PATH,

            "expected_sha256":
                NOTEBOOK43_DEFAULT_ENHANCED_XG_MODEL_SHA256,
        },
    ]


def build_default_artifact_validation_handler():
    """
    Zero-argument production adapter for Stage 2.

    Validates the three serialized artifacts required by the reusable
    clip pipeline:

      - frozen V3 ball detector
      - frozen player detector
      - frozen enhanced-xG model

    Input video validation remains Stage 1.
    """

    def handler(
        result,
    ):

        expected_prefix = list(
            CANONICAL_PIPELINE_STAGES[
                :
                _STAGE_INDEX[
                    "artifact_validation"
                ]
            ]
        )


        if (
            result.pipeline_status.completed_stages
            !=
            expected_prefix
        ):

            raise PipelineContractError(
                "artifact_validation requires completed input_validation."
            )


        artifact_records = {}


        for artifact in (
            _notebook43_default_deployment_artifacts()
        ):

            path = artifact[
                "path"
            ]


            exists = (
                path.exists()
                and
                path.is_file()
            )


            nonempty = (
                exists
                and
                path.stat().st_size
                >
                0
            )


            if not exists:

                raise RuntimeError(
                    f"Required production artifact is missing: {path}"
                )


            if not nonempty:

                raise RuntimeError(
                    f"Required production artifact is empty: {path}"
                )


            actual_sha256 = (
                _notebook43_sha256_file(
                    path
                )
            )


            sha256_exact = (
                actual_sha256
                ==
                artifact[
                    "expected_sha256"
                ]
            )


            if not sha256_exact:

                raise RuntimeError(
                    "Production artifact SHA256 mismatch: "
                    f"{artifact['artifact_name']}"
                )


            artifact_name = str(
                artifact[
                    "artifact_name"
                ]
            )


            artifact_records[
                artifact_name
            ] = {
                "artifact_name":
                    artifact_name,

                "relative_path":
                    artifact[
                        "relative_path"
                    ],

                "path":
                    str(
                        path
                    ),

                "size_bytes":
                    int(
                        path.stat().st_size
                    ),

                "expected_sha256":
                    artifact[
                        "expected_sha256"
                    ],

                "actual_sha256":
                    actual_sha256,

                "sha256":
                    actual_sha256,

                "sha256_exact":
                    True,
            }


        result.metadata[
            "artifact_validation"
        ] = {
            "success":
                True,

            "method":
                "CANONICAL_DEPLOYMENT_ARTIFACT_SHA256_VALIDATION",

            "project_root":
                str(
                    _notebook43_project_root()
                ),

            "artifact_count":
                int(
                    len(
                        artifact_records
                    )
                ),

            "artifacts":
                artifact_records,

            "all_required_artifacts_valid":
                True,

            "historical_result_artifact_used":
                False,

            "reference_homography_used":
                False,

            "reference_coordinates_used":
                False,
        }


        return {
            "project_root":
                _notebook43_project_root(),

            "artifacts":
                artifact_records,

            "all_required_artifacts_valid":
                True,
        }


    return handler


def build_default_ball_detection_handler():
    """
    Zero-argument production adapter for canonical frozen V3 ball inference.
    """

    project_root = (
        _notebook43_project_root()
    )


    model_path = (
        project_root
        /
        NOTEBOOK43_DEFAULT_V3_BALL_MODEL_RELATIVE_PATH
    )


    return build_ball_detection_handler(
        model_path=
            model_path,

        expected_sha256=
            NOTEBOOK43_DEFAULT_V3_BALL_MODEL_SHA256,
    )


def build_default_player_detection_tracking_handler():
    """
    Zero-argument production adapter for canonical player inference/tracking.
    """

    project_root = (
        _notebook43_project_root()
    )


    model_path = (
        project_root
        /
        NOTEBOOK43_DEFAULT_PLAYER_MODEL_RELATIVE_PATH
    )


    return build_player_detection_tracking_handler(
        model_path=
            model_path,

        expected_sha256=
            NOTEBOOK43_DEFAULT_PLAYER_MODEL_SHA256,
    )


# ----------------------------------------------------------------------------------------------------
# Preserve all original stage builders.
#
# Only application-level stage-plan routing changes for the three factories that require deployment
# configuration.
# ----------------------------------------------------------------------------------------------------

NOTEBOOK43_CANONICAL_STAGE_BUILDERS = dict(
    NOTEBOOK43_CANONICAL_STAGE_BUILDERS
)


NOTEBOOK43_CANONICAL_STAGE_BUILDERS[
    "artifact_validation"
] = (
    "build_default_artifact_validation_handler"
)


NOTEBOOK43_CANONICAL_STAGE_BUILDERS[
    "ball_detection"
] = (
    "build_default_ball_detection_handler"
)


NOTEBOOK43_CANONICAL_STAGE_BUILDERS[
    "player_detection_tracking"
] = (
    "build_default_player_detection_tracking_handler"
)


for _export_name in [
    "build_default_artifact_validation_handler",
    "build_default_ball_detection_handler",
    "build_default_player_detection_tracking_handler",
]:

    if (
        _export_name
        not in
        __all__
    ):

        __all__.append(
            _export_name
        )

