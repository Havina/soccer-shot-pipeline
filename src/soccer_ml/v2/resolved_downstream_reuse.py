from __future__ import annotations

from typing import Any, Callable

import soccer_ml.clip_pipeline as _clip_pipeline

from . import v1_adapter as _v1_adapter


RESOLVED_DOWNSTREAM_REUSE_STAGE_NAMES = (
    "relative_team_assignment",
    "shot_geometry",
    "defender_context",
    "enhanced_xg",
)


EXPECTED_FROZEN_V1_BUILDER_BY_STAGE = {
    "relative_team_assignment":
        "build_relative_team_assignment_handler",

    "shot_geometry":
        "build_shot_geometry_handler",

    "defender_context":
        "build_defender_context_handler",

    "enhanced_xg":
        "build_enhanced_xg_handler",
}


FROZEN_V1_EXECUTOR_NAME = (
    "_execute_handler_without_orchestration_mutation"
)


FROZEN_PRODUCTION_RESOLVER_NAME = (
    "_notebook43_build_stage_handler"
)


class ResolvedDownstreamReuseError(
    RuntimeError
):
    pass


class ResolvedDownstreamReuseContractError(
    ResolvedDownstreamReuseError
):
    pass


def _resolve_frozen_v1_executor():

    executor = getattr(
        _v1_adapter,
        FROZEN_V1_EXECUTOR_NAME,
        None,
    )


    if not callable(
        executor
    ):

        raise ResolvedDownstreamReuseContractError(
            "Frozen-v1 adapter executor is unavailable."
        )


    return executor


def _resolve_frozen_v1_handler(
    stage: str,
):

    runtime_builder_map = getattr(
        _clip_pipeline,
        "NOTEBOOK43_CANONICAL_STAGE_BUILDERS",
        None,
    )


    if not isinstance(
        runtime_builder_map,
        dict,
    ):

        raise ResolvedDownstreamReuseContractError(
            "Frozen production builder map is unavailable."
        )


    expected_builder = (
        EXPECTED_FROZEN_V1_BUILDER_BY_STAGE[
            stage
        ]
    )


    runtime_builder = runtime_builder_map.get(
        stage
    )


    if runtime_builder != expected_builder:

        raise ResolvedDownstreamReuseContractError(
            "Frozen production builder identity changed for stage {!r}: "
            "expected={!r}, observed={!r}".format(
                stage,
                expected_builder,
                runtime_builder,
            )
        )


    resolver = getattr(
        _clip_pipeline,
        FROZEN_PRODUCTION_RESOLVER_NAME,
        None,
    )


    if not callable(
        resolver
    ):

        raise ResolvedDownstreamReuseContractError(
            "Frozen production stage resolver is unavailable."
        )


    resolved_handler = resolver(
        stage=stage
    )


    if not callable(
        resolved_handler
    ):

        raise ResolvedDownstreamReuseContractError(
            "Frozen production stage resolver did not return a callable handler."
        )


    return resolved_handler


def build_resolved_downstream_v1_reuse_handler(
    stage: str,
) -> Callable[
    [
        Any,
    ],
    None,
]:

    if not isinstance(
        stage,
        str,
    ):

        raise ResolvedDownstreamReuseContractError(
            "Resolved downstream stage must be a string."
        )


    if stage not in RESOLVED_DOWNSTREAM_REUSE_STAGE_NAMES:

        raise ResolvedDownstreamReuseContractError(
            "Stage is not a supported resolved-downstream frozen-v1 reuse stage: {!r}".format(
                stage
            )
        )


    if stage not in _v1_adapter.POST_RESOLUTION_REUSE_STAGE_NAMES:

        raise ResolvedDownstreamReuseContractError(
            "Stage is not classified as post-resolution reuse by the frozen-v1 adapter."
        )


    if stage not in _v1_adapter.DIRECT_REUSE_STAGE_NAMES:

        raise ResolvedDownstreamReuseContractError(
            "Stage is not classified as direct frozen-v1 reuse."
        )


    binding = _v1_adapter.v1_stage_binding(
        stage
    )


    if binding.requires_resolved_shooter is not True:

        raise ResolvedDownstreamReuseContractError(
            "Resolved-downstream stage lost its resolved-shooter requirement."
        )


    expected_builder = (
        _v1_adapter
        .EXPECTED_V1_BUILDER_BY_STAGE[
            stage
        ]
    )


    if not isinstance(
        expected_builder,
        str,
    ) or not expected_builder:

        raise ResolvedDownstreamReuseContractError(
            "Frozen-v1 builder identity is unavailable."
        )


    if (
        EXPECTED_FROZEN_V1_BUILDER_BY_STAGE[
            stage
        ]
        !=
        expected_builder
    ):

        raise ResolvedDownstreamReuseContractError(
            "Frozen-v1 adapter builder identity disagrees with "
            "the resolved-downstream production contract."
        )


    def handler(
        result,
    ):

        executor = (
            _resolve_frozen_v1_executor()
        )


        def deferred_frozen_handler(
            resolved_result,
        ):

            frozen_handler = (
                _resolve_frozen_v1_handler(
                    stage
                )
            )


            return frozen_handler(
                resolved_result
            )


        return executor(
            result=result,
            stage=stage,
            handler=deferred_frozen_handler,
        )


    return handler


def build_relative_team_assignment_v2_reuse_handler():

    return build_resolved_downstream_v1_reuse_handler(
        "relative_team_assignment"
    )


def build_shot_geometry_v2_reuse_handler():

    return build_resolved_downstream_v1_reuse_handler(
        "shot_geometry"
    )


def build_defender_context_v2_reuse_handler():

    return build_resolved_downstream_v1_reuse_handler(
        "defender_context"
    )


def build_enhanced_xg_v2_reuse_handler():

    return build_resolved_downstream_v1_reuse_handler(
        "enhanced_xg"
    )
