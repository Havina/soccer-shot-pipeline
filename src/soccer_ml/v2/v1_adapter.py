from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from .orchestration_contracts import (
    V2StageAction,
    stage_spec,
)

from .semantic_contracts import (
    CANONICAL_V2_STAGE_NAMES,
)


FROZEN_V1_SOURCE_HASHES = {
    "api.py":
        "479fe7726c9049d9e126a780104aaa65fc67eb69a41d1ff4e19dbddf378a50db",

    "clip_pipeline.py":
        "913dee8e35a0612d3d6b9ad5465689ac682d99f7d485710a0ce23bd69396f880",

    "results.py":
        "883fae0407008e7de04189cc0332e8e836213fd2f87ac24a814db958e0466db5",

    "shot_pipeline.py":
        "0d205673398baaa4c3c20c02e3dcbbe2fa4409f093f3a3723fd258f7278e75f5",
}


EXPECTED_V1_BUILDER_BY_STAGE = {
    "input_validation":
        "build_input_validation_handler",

    "artifact_validation":
        "build_default_artifact_validation_handler",

    "video_metadata":
        "build_video_metadata_handler",

    "ball_detection":
        "build_default_ball_detection_handler",

    "ball_tracking":
        "build_ball_tracking_handler",

    "player_detection_tracking":
        "build_default_player_detection_tracking_handler",

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


DIRECT_REUSE_STAGE_NAMES = (
    "input_validation",
    "video_metadata",
    "ball_detection",
    "ball_tracking",
    "player_detection_tracking",
    "relative_team_assignment",
    "shot_geometry",
    "defender_context",
    "enhanced_xg",
    "annotation",
)


WRAPPED_REUSE_STAGE_NAMES = (
    "artifact_validation",
    "output_validation",
)


REPLACEMENT_STAGE_NAMES = (
    "pitch_line_detection",
    "homography_search",
    "homography_validation",
    "pitch_projection",
    "shooter_assignment",
    "target_goal_inference",
)


POST_RESOLUTION_REUSE_STAGE_NAMES = (
    "relative_team_assignment",
    "shot_geometry",
    "defender_context",
    "enhanced_xg",
    "annotation",
    "output_validation",
)


class FrozenV1AdapterError(RuntimeError):
    """Base error for the versioned frozen-v1 adapter boundary."""


class FrozenV1IdentityError(FrozenV1AdapterError):
    """Frozen production-v1 source/runtime identity changed."""


class FrozenV1StageNotReusableError(FrozenV1AdapterError):
    """A v2 replacement stage was incorrectly requested from frozen v1."""


class FrozenV1StageModeError(FrozenV1AdapterError):
    """A direct/wrapped v1 stage was requested through the wrong adapter path."""


class FrozenV1StageContractError(FrozenV1AdapterError):
    """The frozen handler violated the versioned adapter execution contract."""


class V1ReuseMode(str, Enum):

    DIRECT = "DIRECT"

    WRAPPED = "WRAPPED"

    REPLACEMENT = "REPLACEMENT"


@dataclass(frozen=True)
class FrozenV1StageBinding:

    stage: str

    reuse_mode: V1ReuseMode

    v2_action: V2StageAction

    builder_name: str

    requires_resolved_shooter: bool

    supports_annotation_output_path: bool


def _sha256_file(
    path: Path,
) -> str:

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


def _load_frozen_v1_clip_pipeline():

    return importlib.import_module(
        "soccer_ml.clip_pipeline"
    )


def assert_frozen_v1_source_identity(
    module=None,
) -> dict[str, str]:

    if module is None:

        module = _load_frozen_v1_clip_pipeline()


    module_file = getattr(
        module,
        "__file__",
        None,
    )


    if not module_file:

        raise FrozenV1IdentityError(
            "soccer_ml.clip_pipeline has no __file__ identity."
        )


    clip_path = Path(
        module_file
    ).resolve()


    if clip_path.name != "clip_pipeline.py":

        raise FrozenV1IdentityError(
            "Unexpected frozen-v1 clip_pipeline module path."
        )


    package_root = clip_path.parent


    observed_hashes: dict[str, str] = {}


    for filename, expected_sha in FROZEN_V1_SOURCE_HASHES.items():

        path = (
            package_root
            /
            filename
        )


        if not path.exists():

            raise FrozenV1IdentityError(
                f"Frozen-v1 source file is missing: {filename}"
            )


        actual_sha = _sha256_file(
            path
        )


        observed_hashes[
            filename
        ] = actual_sha


        if actual_sha != expected_sha:

            raise FrozenV1IdentityError(
                f"Frozen-v1 source hash mismatch: {filename}"
            )


    runtime_stages = tuple(
        getattr(
            module,
            "CANONICAL_PIPELINE_STAGES",
            (),
        )
    )


    if runtime_stages != CANONICAL_V2_STAGE_NAMES:

        raise FrozenV1IdentityError(
            "Frozen-v1 canonical stage sequence differs from the frozen v2 contract."
        )


    runtime_builder_map = getattr(
        module,
        "NOTEBOOK43_CANONICAL_STAGE_BUILDERS",
        None,
    )


    if not isinstance(
        runtime_builder_map,
        dict,
    ):

        raise FrozenV1IdentityError(
            "Frozen-v1 canonical builder map is unavailable."
        )


    if runtime_builder_map != EXPECTED_V1_BUILDER_BY_STAGE:

        raise FrozenV1IdentityError(
            "Frozen-v1 canonical builder map differs from the recovered contract."
        )


    resolver = getattr(
        module,
        "_notebook43_build_stage_handler",
        None,
    )


    if not callable(
        resolver
    ):

        raise FrozenV1IdentityError(
            "Frozen-v1 canonical handler resolver is unavailable."
        )


    mark_complete = getattr(
        module,
        "mark_stage_complete",
        None,
    )


    if not callable(
        mark_complete
    ):

        raise FrozenV1IdentityError(
            "Frozen-v1 mark_stage_complete is unavailable."
        )


    return observed_hashes


def v1_stage_binding(
    stage: str,
) -> FrozenV1StageBinding:

    if not isinstance(
        stage,
        str,
    ):

        raise FrozenV1StageContractError(
            "stage must be a canonical stage-name string."
        )


    if stage not in CANONICAL_V2_STAGE_NAMES:

        raise FrozenV1StageContractError(
            f"Unknown canonical v2 stage: {stage!r}"
        )


    spec = stage_spec(
        stage
    )


    builder_name = EXPECTED_V1_BUILDER_BY_STAGE[
        stage
    ]


    if stage in DIRECT_REUSE_STAGE_NAMES:

        reuse_mode = V1ReuseMode.DIRECT


        if spec.action not in {
            V2StageAction.REUSE_V1_HANDLER,
            V2StageAction.REUSE_AFTER_RESOLVED,
        }:

            raise FrozenV1StageContractError(
                f"Direct reuse classification disagrees with frozen v2 action for {stage!r}."
            )


    elif stage in WRAPPED_REUSE_STAGE_NAMES:

        reuse_mode = V1ReuseMode.WRAPPED


        if spec.action not in {
            V2StageAction.WRAP_V1_PLUS_GEOMETRY_RUNTIME_READINESS,
            V2StageAction.WRAP_V1_PLUS_AUTHORITATIVE_V2_PROVENANCE,
        }:

            raise FrozenV1StageContractError(
                f"Wrapped reuse classification disagrees with frozen v2 action for {stage!r}."
            )


    elif stage in REPLACEMENT_STAGE_NAMES:

        reuse_mode = V1ReuseMode.REPLACEMENT


        if spec.action in {
            V2StageAction.REUSE_V1_HANDLER,
            V2StageAction.REUSE_AFTER_RESOLVED,
            V2StageAction.WRAP_V1_PLUS_GEOMETRY_RUNTIME_READINESS,
            V2StageAction.WRAP_V1_PLUS_AUTHORITATIVE_V2_PROVENANCE,
        }:

            raise FrozenV1StageContractError(
                f"Replacement classification disagrees with frozen v2 action for {stage!r}."
            )


    else:

        raise FrozenV1StageContractError(
            f"Stage has no frozen v1 reuse classification: {stage!r}"
        )


    return FrozenV1StageBinding(
        stage=stage,
        reuse_mode=reuse_mode,
        v2_action=spec.action,
        builder_name=builder_name,
        requires_resolved_shooter=bool(
            spec.requires_resolved_shooter
        ),
        supports_annotation_output_path=(
            stage
            ==
            "annotation"
        ),
    )


def _validate_annotation_output_path(
    binding: FrozenV1StageBinding,
    annotated_output_path,
):

    if (
        annotated_output_path is not None
        and
        not binding.supports_annotation_output_path
    ):

        raise FrozenV1StageContractError(
            "annotated_output_path may only be supplied to the annotation stage."
        )


def _validate_resolved_shooter(
    result,
):

    shooter = getattr(
        result,
        "shooter",
        None,
    )


    if shooter is None:

        raise FrozenV1StageContractError(
            "Post-resolution frozen-v1 reuse requires result.shooter."
        )


    if getattr(
        shooter,
        "success",
        None,
    ) is not True:

        raise FrozenV1StageContractError(
            "Post-resolution frozen-v1 reuse requires shooter.success=True."
        )


    track_id = getattr(
        shooter,
        "track_id",
        None,
    )


    if (
        not isinstance(
            track_id,
            int,
        )
        or
        isinstance(
            track_id,
            bool,
        )
        or
        track_id
        <
        0
    ):

        raise FrozenV1StageContractError(
            "Post-resolution frozen-v1 reuse requires a valid resolved shooter track_id."
        )


def _snapshot_orchestration_owned_status(
    result,
):

    pipeline_status = getattr(
        result,
        "pipeline_status",
        None,
    )


    if pipeline_status is None:

        raise FrozenV1StageContractError(
            "Frozen-v1 stage execution requires result.pipeline_status."
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

        raise FrozenV1StageContractError(
            "result.pipeline_status.completed_stages must be a list."
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


def _validate_orchestration_owned_status_unchanged(
    result,
    before_snapshot,
):

    after_snapshot = _snapshot_orchestration_owned_status(
        result
    )


    if after_snapshot != before_snapshot:

        raise FrozenV1StageContractError(
            "Frozen-v1 handler modified orchestration-owned stage completion/failure state."
        )


def _resolve_handler(
    stage: str,
    *,
    expected_mode: V1ReuseMode,
    annotated_output_path=None,
):

    binding = v1_stage_binding(
        stage
    )


    if binding.reuse_mode == V1ReuseMode.REPLACEMENT:

        raise FrozenV1StageNotReusableError(
            f"Stage {stage!r} is a required semantic-v2 replacement and may not use the frozen-v1 handler."
        )


    if binding.reuse_mode != expected_mode:

        raise FrozenV1StageModeError(
            f"Stage {stage!r} must use {binding.reuse_mode.value} reuse, not {expected_mode.value} reuse."
        )


    _validate_annotation_output_path(
        binding,
        annotated_output_path,
    )


    module = _load_frozen_v1_clip_pipeline()


    assert_frozen_v1_source_identity(
        module
    )


    runtime_builder_map = getattr(
        module,
        "NOTEBOOK43_CANONICAL_STAGE_BUILDERS",
    )


    if runtime_builder_map.get(
        stage
    ) != binding.builder_name:

        raise FrozenV1IdentityError(
            f"Runtime builder identity mismatch for stage {stage!r}."
        )


    resolver = getattr(
        module,
        "_notebook43_build_stage_handler",
    )


    handler = resolver(
        stage,
        annotated_output_path=
            annotated_output_path,
    )


    if not callable(
        handler
    ):

        raise FrozenV1StageContractError(
            f"Frozen-v1 resolver did not return a callable handler for {stage!r}."
        )


    return handler


def resolve_direct_v1_stage_handler(
    stage: str,
    *,
    annotated_output_path=None,
):

    return _resolve_handler(
        stage,
        expected_mode=V1ReuseMode.DIRECT,
        annotated_output_path=annotated_output_path,
    )


def resolve_wrapped_v1_inner_handler(
    stage: str,
    *,
    annotated_output_path=None,
):

    return _resolve_handler(
        stage,
        expected_mode=V1ReuseMode.WRAPPED,
        annotated_output_path=annotated_output_path,
    )


def _execute_handler_without_orchestration_mutation(
    *,
    result,
    stage: str,
    handler: Callable[[Any], None],
):

    binding = v1_stage_binding(
        stage
    )


    if (
        binding.requires_resolved_shooter
        or
        stage in POST_RESOLUTION_REUSE_STAGE_NAMES
    ):

        _validate_resolved_shooter(
            result
        )


    before_snapshot = _snapshot_orchestration_owned_status(
        result
    )


    # NOTEBOOK50 CELL-40-LOCAL DEFECT FIX (v1_adapter r2)
    # --------------------------------------------------------------------
    # The frozen-v1 orchestrator (clip_pipeline.py, SoccerClipPipeline.analyze)
    # calls ``handler(result)`` and discards the return value. Eleven of the
    # eighteen real frozen-v1 handlers (artifact_validation, homography_validation,
    # pitch_projection, shooter_assignment, relative_team_assignment,
    # target_goal_inference, shot_geometry, defender_context, enhanced_xg,
    # annotation, output_validation) return diagnostic payloads. The previous
    # ``return_value is not None -> FrozenV1StageContractError`` check was
    # derived from the ``StageHandler`` type alias, was only ever exercised
    # against synthetic handlers (Cell 13: real_frozen_v1_handler_executed=False),
    # and made every real end-to-end run fail at stage 2. The adapter now
    # mirrors the frozen-v1 orchestrator exactly: the return value is ignored.
    # Orchestration-owned state is still verified unchanged below.
    handler(
        result
    )


    _validate_orchestration_owned_status_unchanged(
        result,
        before_snapshot,
    )


def execute_direct_v1_stage(
    result,
    stage: str,
    *,
    annotated_output_path=None,
):

    handler = resolve_direct_v1_stage_handler(
        stage,
        annotated_output_path=annotated_output_path,
    )


    _execute_handler_without_orchestration_mutation(
        result=result,
        stage=stage,
        handler=handler,
    )


def execute_wrapped_v1_inner_stage(
    result,
    stage: str,
    *,
    annotated_output_path=None,
):

    handler = resolve_wrapped_v1_inner_handler(
        stage,
        annotated_output_path=annotated_output_path,
    )


    _execute_handler_without_orchestration_mutation(
        result=result,
        stage=stage,
        handler=handler,
    )
