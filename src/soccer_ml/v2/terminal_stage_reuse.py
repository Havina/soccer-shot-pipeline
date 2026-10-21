from __future__ import annotations

from typing import Any, Callable

import soccer_ml.clip_pipeline as _clip_pipeline

from . import v1_adapter as _v1_adapter


ANNOTATION_STAGE_NAME = "annotation"

OUTPUT_VALIDATION_STAGE_NAME = "output_validation"

OUTPUT_VALIDATION_ACTION = (
    "WRAP_V1_PLUS_AUTHORITATIVE_V2_PROVENANCE"
)

V2_OUTPUT_VALIDATION_PROVENANCE_KEY = (
    "semantic_v2_output_validation"
)

EXPECTED_FROZEN_V1_BUILDER_BY_STAGE = {
    "annotation":
        "build_annotation_handler",

    "output_validation":
        "build_output_validation_handler",
}

FROZEN_V1_EXECUTOR_NAME = (
    "_execute_handler_without_orchestration_mutation"
)

FROZEN_PRODUCTION_RESOLVER_NAME = (
    "_notebook43_build_stage_handler"
)


class TerminalStageReuseError(
    RuntimeError
):
    pass


class TerminalStageReuseContractError(
    TerminalStageReuseError
):
    pass


def _mode(
    value,
):

    return str(
        getattr(
            value,
            "value",
            value,
        )
    )


def _validate_terminal_binding(
    stage,
    expected_mode,
):

    if stage not in (
        ANNOTATION_STAGE_NAME,
        OUTPUT_VALIDATION_STAGE_NAME,
    ):

        raise TerminalStageReuseContractError(
            "Unsupported terminal stage: {!r}".format(
                stage
            )
        )

    if stage not in _v1_adapter.POST_RESOLUTION_REUSE_STAGE_NAMES:

        raise TerminalStageReuseContractError(
            "Terminal stage lost post-resolution classification."
        )

    binding = _v1_adapter.v1_stage_binding(
        stage
    )

    if (
        getattr(
            binding,
            "requires_resolved_shooter",
            False,
        )
        is not True
    ):

        raise TerminalStageReuseContractError(
            "Terminal stage lost resolved-shooter requirement."
        )

    if (
        _mode(
            getattr(
                binding,
                "reuse_mode",
                None,
            )
        )
        !=
        expected_mode
    ):

        raise TerminalStageReuseContractError(
            "Terminal stage reuse mode changed."
        )

    if (
        expected_mode
        ==
        "DIRECT"
        and
        stage
        not in
        _v1_adapter.DIRECT_REUSE_STAGE_NAMES
    ):

        raise TerminalStageReuseContractError(
            "Direct reuse classification changed."
        )

    if (
        expected_mode
        ==
        "WRAPPED"
        and
        stage
        not in
        _v1_adapter.WRAPPED_REUSE_STAGE_NAMES
    ):

        raise TerminalStageReuseContractError(
            "Wrapped reuse classification changed."
        )

    if (
        _v1_adapter
        .EXPECTED_V1_BUILDER_BY_STAGE[
            stage
        ]
        !=
        EXPECTED_FROZEN_V1_BUILDER_BY_STAGE[
            stage
        ]
    ):

        raise TerminalStageReuseContractError(
            "Adapter builder identity changed."
        )


def _resolve_executor():

    executor = getattr(
        _v1_adapter,
        FROZEN_V1_EXECUTOR_NAME,
        None,
    )

    if not callable(
        executor
    ):

        raise TerminalStageReuseContractError(
            "Frozen-v1 executor unavailable."
        )

    return executor


def _resolve_handler(
    stage,
    annotated_output_path=None,
):

    builder_map = getattr(
        _clip_pipeline,
        "NOTEBOOK43_CANONICAL_STAGE_BUILDERS",
        None,
    )

    if not isinstance(
        builder_map,
        dict,
    ):

        raise TerminalStageReuseContractError(
            "Frozen production builder map unavailable."
        )

    if (
        builder_map.get(
            stage
        )
        !=
        EXPECTED_FROZEN_V1_BUILDER_BY_STAGE[
            stage
        ]
    ):

        raise TerminalStageReuseContractError(
            "Frozen production builder identity changed."
        )

    resolver = getattr(
        _clip_pipeline,
        FROZEN_PRODUCTION_RESOLVER_NAME,
        None,
    )

    if not callable(
        resolver
    ):

        raise TerminalStageReuseContractError(
            "Frozen production resolver unavailable."
        )

    if stage == ANNOTATION_STAGE_NAME:

        handler = resolver(
            stage=stage,
            annotated_output_path=annotated_output_path,
        )

    else:

        if annotated_output_path is not None:

            raise TerminalStageReuseContractError(
                "annotated_output_path is annotation-only."
            )

        handler = resolver(
            stage=stage
        )

    if not callable(
        handler
    ):

        raise TerminalStageReuseContractError(
            "Frozen production resolver returned non-callable."
        )

    return handler


def _scalar(
    value,
):

    if value is None:
        return None

    value = getattr(
        value,
        "value",
        value,
    )

    return (
        value
        if isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
            ),
        )
        else str(
            value
        )
    )


def _build_v2_provenance(
    result,
    frozen_validation,
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

        raise TerminalStageReuseContractError(
            "result.metadata must be a dict."
        )

    if (
        V2_OUTPUT_VALIDATION_PROVENANCE_KEY
        in
        metadata
    ):

        raise TerminalStageReuseContractError(
            "Semantic-v2 provenance already populated."
        )

    if not isinstance(
        frozen_validation,
        dict,
    ):

        raise TerminalStageReuseContractError(
            "Frozen output validation mapping missing."
        )

    if (
        not frozen_validation.get(
            "all_public_components_success"
        )
        or
        not frozen_validation.get(
            "final_video_contract_valid"
        )
    ):

        raise TerminalStageReuseContractError(
            "Frozen output validation was not successful."
        )

    if (
        not isinstance(
            metadata.get(
                "output_validation"
            ),
            dict,
        )
        or
        not metadata[
            "output_validation"
        ].get(
            "success"
        )
    ):

        raise TerminalStageReuseContractError(
            "Frozen output-validation metadata missing."
        )

    geometry = getattr(
        result,
        "geometry_v2",
        None,
    )

    shooter_v2 = getattr(
        result,
        "shooter_v2",
        None,
    )

    status = getattr(
        result,
        "semantic_v2_status",
        None,
    )

    shooter = getattr(
        result,
        "shooter",
        None,
    )

    target = getattr(
        result,
        "target_goal",
        None,
    )

    if (
        geometry is None
        or
        shooter_v2 is None
        or
        status is None
    ):

        raise TerminalStageReuseContractError(
            "Authoritative Semantic-v2 state missing."
        )

    if (
        not getattr(
            shooter,
            "success",
            False,
        )
        or
        not getattr(
            target,
            "success",
            False,
        )
    ):

        raise TerminalStageReuseContractError(
            "Resolved shooter/target-goal state missing."
        )

    return {
        "success":
            True,

        "action":
            OUTPUT_VALIDATION_ACTION,

        "frozen_v1_validation_preserved":
            True,

        "frozen_v1_output_validation":
            dict(
                frozen_validation
            ),

        "geometry_v2_success":
            bool(
                getattr(
                    geometry,
                    "success",
                    False,
                )
            ),

        "geometry_v2_runtime_mode":
            _scalar(
                getattr(
                    geometry,
                    "runtime_mode",
                    None,
                )
            ),

        "geometry_v2_selected_frame":
            _scalar(
                getattr(
                    geometry,
                    "selected_frame",
                    getattr(
                        geometry,
                        "selected_frame_idx",
                        None,
                    ),
                )
            ),

        "geometry_v2_selected_variant":
            _scalar(
                getattr(
                    geometry,
                    "selected_variant",
                    None,
                )
            ),

        "shooter_v2_resolution":
            _scalar(
                getattr(
                    shooter_v2,
                    "resolution",
                    None,
                )
            ),

        "shooter_v2_resolved_track":
            _scalar(
                getattr(
                    shooter,
                    "track_id",
                    None,
                )
            ),

        "semantic_v2_state":
            _scalar(
                getattr(
                    status,
                    "state",
                    None,
                )
            ),

        "target_goal_v2_authoritative":
            True,

        "target_goal":
            _scalar(
                getattr(
                    target,
                    "target_goal",
                    None,
                )
            ),

        "legacy_manual_shooter_assignment_flag_authoritative":
            False,

        "analytical_recomputation_performed":
            False,

        "analytical_components_modified":
            False,
    }


def build_annotation_v2_reuse_handler(
    output_path=None,
) -> Callable[
    [
        Any,
    ],
    None,
]:

    _validate_terminal_binding(
        ANNOTATION_STAGE_NAME,
        "DIRECT",
    )

    def handler(
        result,
    ):

        executor = (
            _resolve_executor()
        )

        def deferred(
            resolved_result,
        ):

            return (
                _resolve_handler(
                    ANNOTATION_STAGE_NAME,
                    output_path,
                )(
                    resolved_result
                )
            )

        return executor(
            result=result,
            stage=ANNOTATION_STAGE_NAME,
            handler=deferred,
        )

    return handler


def build_output_validation_v2_wrapped_handler() -> Callable[
    [
        Any,
    ],
    None,
]:

    _validate_terminal_binding(
        OUTPUT_VALIDATION_STAGE_NAME,
        "WRAPPED",
    )

    def handler(
        result,
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

            raise TerminalStageReuseContractError(
                "result.metadata must be a dict."
            )

        if (
            V2_OUTPUT_VALIDATION_PROVENANCE_KEY
            in
            metadata
        ):

            raise TerminalStageReuseContractError(
                "Semantic-v2 provenance already populated."
            )

        box = {}

        def deferred(
            resolved_result,
        ):

            frozen_return = (
                _resolve_handler(
                    OUTPUT_VALIDATION_STAGE_NAME
                )(
                    resolved_result
                )
            )

            if not isinstance(
                frozen_return,
                dict,
            ):

                raise TerminalStageReuseContractError(
                    "Frozen output validation did not return a mapping."
                )

            box[
                "validation"
            ] = dict(
                frozen_return
            )

            return None

        _resolve_executor()(
            result=result,
            stage=OUTPUT_VALIDATION_STAGE_NAME,
            handler=deferred,
        )

        metadata[
            V2_OUTPUT_VALIDATION_PROVENANCE_KEY
        ] = _build_v2_provenance(
            result,
            box.get(
                "validation"
            ),
        )

        return None

    return handler
