from __future__ import annotations

from pathlib import Path

from .. import clip_pipeline as _clip_pipeline
from . import semantic_v2_pipeline as _assembly
from . import v1_adapter as _v1_adapter
from .semantic_contracts import (
    SemanticV2State,
    SemanticV2Status,
    SoccerClipAnalysisV2Result,
)


GEOMETRY_RUNTIME_READINESS_METADATA_KEY = (
    "semantic_v2_geometry_runtime_readiness"
)


class SemanticV2ExecutionError(
    RuntimeError
):
    pass


class SemanticV2ExecutionContractError(
    SemanticV2ExecutionError
):
    pass


def _complete_status():

    return SemanticV2Status(
        state=
            SemanticV2State.COMPLETE,

        completed_stage_count=
            18,

        blocked_stage=
            None,

        confirmation_required=
            False,

        failure_stage=
            None,

        failure_reason=
            None,
    )


def _confirmation_required_status():

    return SemanticV2Status(
        state=
            SemanticV2State.SHOOTER_CONFIRMATION_REQUIRED,

        completed_stage_count=
            10,

        blocked_stage=
            "shooter_assignment",

        confirmation_required=
            True,

        failure_stage=
            None,

        failure_reason=
            None,
    )


def _failed_status(
    *,
    completed_stage_count,
    failure_stage,
    failure_reason,
):

    return SemanticV2Status(
        state=
            SemanticV2State.FAILED,

        completed_stage_count=
            int(
                completed_stage_count
            ),

        blocked_stage=
            None,

        confirmation_required=
            False,

        failure_stage=
            failure_stage,

        failure_reason=
            failure_reason,
    )


def _validate_runtime_dependencies(
    *,
    geometry_inference_callable,
    frame_loader,
    png_encoder,
    video_sha256_provider,
    shooter_confirmation_track_id,
):

    required_callables = {
        "geometry_inference_callable":
            geometry_inference_callable,

        "frame_loader":
            frame_loader,

        "png_encoder":
            png_encoder,
    }

    for name, value in required_callables.items():

        if not callable(
            value
        ):

            raise SemanticV2ExecutionContractError(
                "{} must be callable.".format(
                    name
                )
            )

    if (
        video_sha256_provider
        is not None
        and
        not callable(
            video_sha256_provider
        )
    ):

        raise SemanticV2ExecutionContractError(
            "video_sha256_provider must be callable or None."
        )

    if shooter_confirmation_track_id is not None:

        if (
            isinstance(
                shooter_confirmation_track_id,
                bool,
            )
            or
            not isinstance(
                shooter_confirmation_track_id,
                int,
            )
            or
            shooter_confirmation_track_id
            <
            0
        ):

            raise SemanticV2ExecutionContractError(
                "shooter_confirmation_track_id must be a non-negative integer or None."
            )


def _build_stage_handler(
    route,
    *,
    geometry_inference_callable,
    frame_loader,
    png_encoder,
    video_sha256_provider,
    annotated_output_path,
    shooter_confirmation_track_id,
):

    builder = _assembly.resolve_stage_builder(
        route.stage
    )

    if route.stage == "pitch_line_detection":

        handler = builder(
            inference_callable=
                geometry_inference_callable,

            frame_loader=
                frame_loader,

            png_encoder=
                png_encoder,

            video_sha256_provider=
                video_sha256_provider,
        )

    elif route.stage == "shooter_assignment":

        handler = builder(
            shooter_confirmation_track_id=
                shooter_confirmation_track_id,
        )

    elif route.stage == "annotation":

        handler = builder(
            output_path=
                annotated_output_path,
        )

    else:

        handler = builder()

    if not callable(
        handler
    ):

        raise SemanticV2ExecutionContractError(
            "Stage builder did not return a callable handler: {!r}".format(
                route.stage
            )
        )

    return handler


def _execute_stage_handler(
    route,
    handler,
    result,
):

    completed_before = tuple(
        result
        .pipeline_status
        .completed_stages
    )

    failed_before = (
        result
        .pipeline_status
        .failed_stage
    )

    if route.route_kind in {
        "FROZEN_V1_DIRECT",
        "FROZEN_V1_WRAPPED_GEOMETRY_READINESS",
    }:

        returned = (
            _v1_adapter
            ._execute_handler_without_orchestration_mutation(
                result=
                    result,

                stage=
                    route.stage,

                handler=
                    handler,
            )
        )

    else:

        returned = handler(
            result
        )

        if (
            tuple(
                result
                .pipeline_status
                .completed_stages
            )
            !=
            completed_before
        ):

            raise SemanticV2ExecutionContractError(
                "Stage handler mutated completed_stages: {!r}".format(
                    route.stage
                )
            )

        if (
            result
            .pipeline_status
            .failed_stage
            !=
            failed_before
        ):

            raise SemanticV2ExecutionContractError(
                "Stage handler mutated failed_stage: {!r}".format(
                    route.stage
                )
            )

    if returned is not None:

        raise SemanticV2ExecutionContractError(
            "Stage handler returned non-None: {!r}".format(
                route.stage
            )
        )


def _record_geometry_runtime_readiness(
    result,
    *,
    geometry_inference_callable,
    frame_loader,
    png_encoder,
    video_sha256_provider,
):

    metadata = getattr(
        result,
        "metadata",
        None,
    )

    if not isinstance(
        metadata,
        dict,
    ):

        raise SemanticV2ExecutionContractError(
            "Result metadata must be a dict."
        )

    if GEOMETRY_RUNTIME_READINESS_METADATA_KEY in metadata:

        raise SemanticV2ExecutionContractError(
            "Semantic-v2 Geometry runtime readiness is already populated."
        )

    metadata[
        GEOMETRY_RUNTIME_READINESS_METADATA_KEY
    ] = {
        "success":
            True,

        "action":
            "WRAP_V1_PLUS_GEOMETRY_RUNTIME_READINESS",

        "inference_callable_ready":
            callable(
                geometry_inference_callable
            ),

        "frame_loader_ready":
            callable(
                frame_loader
            ),

        "png_encoder_ready":
            callable(
                png_encoder
            ),

        "video_sha256_provider_ready":
            (
                video_sha256_provider
                is None
                or
                callable(
                    video_sha256_provider
                )
            ),

        "network_call_performed":
            False,

        "geometry_inference_performed":
            False,
    }


def _shooter_resolved(
    result,
):

    decision = getattr(
        result,
        "shooter_v2",
        None,
    )

    shooter = getattr(
        result,
        "shooter",
        None,
    )

    if decision is None:

        return False

    resolved_track_id = getattr(
        decision,
        "resolved_track_id",
        None,
    )

    if (
        isinstance(
            resolved_track_id,
            bool,
        )
        or
        not isinstance(
            resolved_track_id,
            int,
        )
        or
        resolved_track_id
        <
        0
    ):

        return False

    if (
        shooter is None
        or
        getattr(
            shooter,
            "success",
            False,
        )
        is not True
    ):

        return False

    return bool(
        getattr(
            shooter,
            "track_id",
            None,
        )
        ==
        resolved_track_id
    )


def _record_failure(
    result,
    *,
    stage,
    exc,
):

    result.pipeline_status.success = False
    result.pipeline_status.degraded = False
    result.pipeline_status.failed_stage = stage

    message = (
        "{}: {}: {}".format(
            stage,
            type(
                exc
            ).__name__,
            str(
                exc
            ),
        )
    )

    if (
        message
        not in
        result
        .pipeline_status
        .warnings
    ):

        result.pipeline_status.warnings.append(
            message
        )

    result.semantic_v2_status = (
        _failed_status(
            completed_stage_count=
                len(
                    result
                    .pipeline_status
                    .completed_stages
                ),

            failure_stage=
                stage,

            failure_reason=
                str(
                    exc
                ),
        )
    )


def run_semantic_v2_clip_pipeline(
    video_path,
    *,
    geometry_inference_callable,
    frame_loader,
    png_encoder,
    video_sha256_provider=None,
    annotated_output_path=None,
    shooter_confirmation_track_id=None,
) -> SoccerClipAnalysisV2Result:

    _validate_runtime_dependencies(
        geometry_inference_callable=
            geometry_inference_callable,

        frame_loader=
            frame_loader,

        png_encoder=
            png_encoder,

        video_sha256_provider=
            video_sha256_provider,

        shooter_confirmation_track_id=
            shooter_confirmation_track_id,
    )

    _assembly.validate_stage_routes()

    result = SoccerClipAnalysisV2Result(
        video_path=
            str(
                Path(
                    video_path
                )
            ),

        geometry_v2=
            None,

        shooter_v2=
            None,

        semantic_v2_status=
            None,
    )

    for route in (
        _assembly
        .SEMANTIC_V2_STAGE_ROUTES
    ):

        stage = route.stage

        try:

            # Frozen Stage-18 provenance records Semantic-v2 terminal state.
            #
            # At this point pipeline_status must still contain exactly the
            # canonical Stage-1..17 completed prefix required by frozen-v1
            # output validation.
            if stage == "output_validation":

                if (
                    result
                    .pipeline_status
                    .completed_stages
                    !=
                    [
                        item.stage
                        for item
                        in
                        _assembly
                        .SEMANTIC_V2_STAGE_ROUTES[
                            :
                            17
                        ]
                    ]
                ):

                    raise SemanticV2ExecutionContractError(
                        "Stage-18 entry does not contain the exact 17-stage canonical prefix."
                    )

                result.semantic_v2_status = (
                    _complete_status()
                )

            handler = _build_stage_handler(
                route,

                geometry_inference_callable=
                    geometry_inference_callable,

                frame_loader=
                    frame_loader,

                png_encoder=
                    png_encoder,

                video_sha256_provider=
                    video_sha256_provider,

                annotated_output_path=
                    annotated_output_path,

                shooter_confirmation_track_id=
                    shooter_confirmation_track_id,
            )

            _execute_stage_handler(
                route,
                handler,
                result,
            )

            if stage == "artifact_validation":

                _record_geometry_runtime_readiness(
                    result,

                    geometry_inference_callable=
                        geometry_inference_callable,

                    frame_loader=
                        frame_loader,

                    png_encoder=
                        png_encoder,

                    video_sha256_provider=
                        video_sha256_provider,
                )

            if stage == "shooter_assignment":

                if not _shooter_resolved(
                    result
                ):

                    if (
                        result
                        .pipeline_status
                        .completed_stages
                        !=
                        [
                            item.stage
                            for item
                            in
                            _assembly
                            .SEMANTIC_V2_STAGE_ROUTES[
                                :
                                10
                            ]
                        ]
                    ):

                        raise SemanticV2ExecutionContractError(
                            "Unresolved shooter did not stop at the exact 10-stage prefix."
                        )

                    result.pipeline_status.success = False
                    result.pipeline_status.degraded = False
                    result.pipeline_status.failed_stage = None

                    result.semantic_v2_status = (
                        _confirmation_required_status()
                    )

                    return result

            _clip_pipeline.mark_stage_complete(
                result.pipeline_status,
                stage,
            )

        except Exception as exc:

            _record_failure(
                result,
                stage=
                    stage,
                exc=
                    exc,
            )

            raise

    if (
        result
        .pipeline_status
        .completed_stages
        !=
        [
            item.stage
            for item
            in
            _assembly
            .SEMANTIC_V2_STAGE_ROUTES
        ]
    ):

        raise SemanticV2ExecutionContractError(
            "Semantic-v2 execution did not complete the exact 18-stage canonical sequence."
        )

    result.pipeline_status.success = True
    result.pipeline_status.degraded = False
    result.pipeline_status.failed_stage = None

    if not _clip_pipeline.validate_pipeline_status(
        result.pipeline_status
    ):

        raise SemanticV2ExecutionContractError(
            "Final Semantic-v2 PipelineStatus is invalid."
        )

    result.semantic_v2_status = (
        _complete_status()
    )

    return result
