from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Dict, Tuple

from . import orchestration_contracts as _orchestration
from . import v1_adapter as _v1_adapter


class SemanticV2PipelineAssemblyError(RuntimeError):
    pass


class SemanticV2PipelineAssemblyContractError(
    SemanticV2PipelineAssemblyError
):
    pass


@dataclass(
    frozen=True,
)
class SemanticV2StageRoute:

    index: int
    stage: str
    action: str
    route_kind: str
    module_name: str
    builder_name: str
    may_execute_before_shooter_resolution: bool
    requires_resolved_shooter: bool
    geometry_v2_stage: bool
    output_path_aware: bool


ROUTE_KINDS = (
    "FROZEN_V1_DIRECT",
    "FROZEN_V1_WRAPPED_GEOMETRY_READINESS",
    "V2_REPLACEMENT",
    "RESOLVED_V1_REUSE",
    "TERMINAL_ANNOTATION_REUSE",
    "TERMINAL_OUTPUT_VALIDATION_WRAPPED",
)


_RAW_ROUTE_ROWS = [{'index': 1,
  'stage': 'input_validation',
  'action': 'REUSE_V1_HANDLER',
  'route_kind': 'FROZEN_V1_DIRECT',
  'module_name': 'soccer_ml.clip_pipeline',
  'builder_name': 'build_input_validation_handler',
  'may_execute_before_shooter_resolution': True,
  'requires_resolved_shooter': False,
  'geometry_v2_stage': False,
  'output_path_aware': False},
 {'index': 2,
  'stage': 'artifact_validation',
  'action': 'WRAP_V1_PLUS_GEOMETRY_RUNTIME_READINESS',
  'route_kind': 'FROZEN_V1_WRAPPED_GEOMETRY_READINESS',
  'module_name': 'soccer_ml.clip_pipeline',
  'builder_name': 'build_default_artifact_validation_handler',
  'may_execute_before_shooter_resolution': True,
  'requires_resolved_shooter': False,
  'geometry_v2_stage': False,
  'output_path_aware': False},
 {'index': 3,
  'stage': 'video_metadata',
  'action': 'REUSE_V1_HANDLER',
  'route_kind': 'FROZEN_V1_DIRECT',
  'module_name': 'soccer_ml.clip_pipeline',
  'builder_name': 'build_video_metadata_handler',
  'may_execute_before_shooter_resolution': True,
  'requires_resolved_shooter': False,
  'geometry_v2_stage': False,
  'output_path_aware': False},
 {'index': 4,
  'stage': 'ball_detection',
  'action': 'REUSE_V1_HANDLER',
  'route_kind': 'FROZEN_V1_DIRECT',
  'module_name': 'soccer_ml.clip_pipeline',
  'builder_name': 'build_default_ball_detection_handler',
  'may_execute_before_shooter_resolution': True,
  'requires_resolved_shooter': False,
  'geometry_v2_stage': False,
  'output_path_aware': False},
 {'index': 5,
  'stage': 'ball_tracking',
  'action': 'REUSE_V1_HANDLER',
  'route_kind': 'FROZEN_V1_DIRECT',
  'module_name': 'soccer_ml.clip_pipeline',
  'builder_name': 'build_ball_tracking_handler',
  'may_execute_before_shooter_resolution': True,
  'requires_resolved_shooter': False,
  'geometry_v2_stage': False,
  'output_path_aware': False},
 {'index': 6,
  'stage': 'player_detection_tracking',
  'action': 'REUSE_V1_HANDLER',
  'route_kind': 'FROZEN_V1_DIRECT',
  'module_name': 'soccer_ml.clip_pipeline',
  'builder_name': 'build_default_player_detection_tracking_handler',
  'may_execute_before_shooter_resolution': True,
  'requires_resolved_shooter': False,
  'geometry_v2_stage': False,
  'output_path_aware': False},
 {'index': 7,
  'stage': 'pitch_line_detection',
  'action': 'REPLACE_V2_PNLCALIB_KEYPOINT_LINE_INFERENCE',
  'route_kind': 'V2_REPLACEMENT',
  'module_name': 'soccer_ml.v2.geometry_stage_handlers',
  'builder_name': 'build_geometry_v2_stage7_handler',
  'may_execute_before_shooter_resolution': True,
  'requires_resolved_shooter': False,
  'geometry_v2_stage': True,
  'output_path_aware': False},
 {'index': 8,
  'stage': 'homography_search',
  'action': 'REPLACE_V2_PNLCALIB_BASE_PNL_SELECTION',
  'route_kind': 'V2_REPLACEMENT',
  'module_name': 'soccer_ml.v2.geometry_stage8_selection',
  'builder_name': 'build_geometry_v2_stage8_selection_handler',
  'may_execute_before_shooter_resolution': True,
  'requires_resolved_shooter': False,
  'geometry_v2_stage': True,
  'output_path_aware': False},
 {'index': 9,
  'stage': 'homography_validation',
  'action': 'REPLACE_V2_REFERENCE_FREE_GEOMETRY_VALIDATION',
  'route_kind': 'V2_REPLACEMENT',
  'module_name': 'soccer_ml.v2.geometry_stage9_validation',
  'builder_name': 'build_geometry_v2_stage9_validation_handler',
  'may_execute_before_shooter_resolution': True,
  'requires_resolved_shooter': False,
  'geometry_v2_stage': True,
  'output_path_aware': False},
 {'index': 10,
  'stage': 'pitch_projection',
  'action': 'REPLACE_V2_PLAYER_PROJECTION',
  'route_kind': 'V2_REPLACEMENT',
  'module_name': 'soccer_ml.v2.geometry_stage10_projection',
  'builder_name': 'build_geometry_v2_stage10_projection_handler',
  'may_execute_before_shooter_resolution': True,
  'requires_resolved_shooter': False,
  'geometry_v2_stage': True,
  'output_path_aware': False},
 {'index': 11,
  'stage': 'shooter_assignment',
  'action': 'REPLACE_V2_AMBIGUITY_SAFE_SHOOTER_GATE',
  'route_kind': 'V2_REPLACEMENT',
  'module_name': 'soccer_ml.v2.shooter_stage11_assignment',
  'builder_name': 'build_shooter_v2_stage11_assignment_handler',
  'may_execute_before_shooter_resolution': True,
  'requires_resolved_shooter': False,
  'geometry_v2_stage': False,
  'output_path_aware': False},
 {'index': 12,
  'stage': 'relative_team_assignment',
  'action': 'REUSE_AFTER_RESOLVED',
  'route_kind': 'RESOLVED_V1_REUSE',
  'module_name': 'soccer_ml.v2.resolved_downstream_reuse',
  'builder_name': 'build_relative_team_assignment_v2_reuse_handler',
  'may_execute_before_shooter_resolution': False,
  'requires_resolved_shooter': True,
  'geometry_v2_stage': False,
  'output_path_aware': False},
 {'index': 13,
  'stage': 'target_goal_inference',
  'action': 'REPLACE_V2_IMAGE_SPACE_TARGET_GOAL',
  'route_kind': 'V2_REPLACEMENT',
  'module_name': 'soccer_ml.v2.target_goal_stage13_inference',
  'builder_name': 'build_target_goal_v2_stage13_inference_handler',
  'may_execute_before_shooter_resolution': False,
  'requires_resolved_shooter': True,
  'geometry_v2_stage': False,
  'output_path_aware': False},
 {'index': 14,
  'stage': 'shot_geometry',
  'action': 'REUSE_AFTER_RESOLVED',
  'route_kind': 'RESOLVED_V1_REUSE',
  'module_name': 'soccer_ml.v2.resolved_downstream_reuse',
  'builder_name': 'build_shot_geometry_v2_reuse_handler',
  'may_execute_before_shooter_resolution': False,
  'requires_resolved_shooter': True,
  'geometry_v2_stage': False,
  'output_path_aware': False},
 {'index': 15,
  'stage': 'defender_context',
  'action': 'REUSE_AFTER_RESOLVED',
  'route_kind': 'RESOLVED_V1_REUSE',
  'module_name': 'soccer_ml.v2.resolved_downstream_reuse',
  'builder_name': 'build_defender_context_v2_reuse_handler',
  'may_execute_before_shooter_resolution': False,
  'requires_resolved_shooter': True,
  'geometry_v2_stage': False,
  'output_path_aware': False},
 {'index': 16,
  'stage': 'enhanced_xg',
  'action': 'REUSE_AFTER_RESOLVED',
  'route_kind': 'RESOLVED_V1_REUSE',
  'module_name': 'soccer_ml.v2.resolved_downstream_reuse',
  'builder_name': 'build_enhanced_xg_v2_reuse_handler',
  'may_execute_before_shooter_resolution': False,
  'requires_resolved_shooter': True,
  'geometry_v2_stage': False,
  'output_path_aware': False},
 {'index': 17,
  'stage': 'annotation',
  'action': 'REUSE_AFTER_RESOLVED',
  'route_kind': 'TERMINAL_ANNOTATION_REUSE',
  'module_name': 'soccer_ml.v2.terminal_stage_reuse',
  'builder_name': 'build_annotation_v2_reuse_handler',
  'may_execute_before_shooter_resolution': False,
  'requires_resolved_shooter': True,
  'geometry_v2_stage': False,
  'output_path_aware': True},
 {'index': 18,
  'stage': 'output_validation',
  'action': 'WRAP_V1_PLUS_AUTHORITATIVE_V2_PROVENANCE',
  'route_kind': 'TERMINAL_OUTPUT_VALIDATION_WRAPPED',
  'module_name': 'soccer_ml.v2.terminal_stage_reuse',
  'builder_name': 'build_output_validation_v2_wrapped_handler',
  'may_execute_before_shooter_resolution': False,
  'requires_resolved_shooter': True,
  'geometry_v2_stage': False,
  'output_path_aware': False}]


SEMANTIC_V2_STAGE_ROUTES = tuple(
    SemanticV2StageRoute(
        **row
    )
    for row
    in _RAW_ROUTE_ROWS
)


_ROUTE_BY_STAGE = {
    route.stage:
        route
    for route
    in SEMANTIC_V2_STAGE_ROUTES
}


_ROUTE_BY_INDEX = {
    route.index:
        route
    for route
    in SEMANTIC_V2_STAGE_ROUTES
}


EXPECTED_RESOLVED_REUSE_BUILDERS = {
    "relative_team_assignment":
        "build_relative_team_assignment_v2_reuse_handler",

    "shot_geometry":
        "build_shot_geometry_v2_reuse_handler",

    "defender_context":
        "build_defender_context_v2_reuse_handler",

    "enhanced_xg":
        "build_enhanced_xg_v2_reuse_handler",
}


EXPECTED_TERMINAL_BUILDERS = {
    "annotation":
        "build_annotation_v2_reuse_handler",

    "output_validation":
        "build_output_validation_v2_wrapped_handler",
}


def _action_value(
    value,
) -> str:

    return str(
        getattr(
            value,
            "value",
            value,
        )
    )


def validate_stage_routes() -> bool:

    specs = tuple(
        _orchestration.V2_STAGE_PLAN
    )

    if len(
        specs
    ) != 18:

        raise SemanticV2PipelineAssemblyContractError(
            "Semantic-v2 stage plan must contain exactly 18 stages."
        )

    if len(
        SEMANTIC_V2_STAGE_ROUTES
    ) != 18:

        raise SemanticV2PipelineAssemblyContractError(
            "Semantic-v2 route table must contain exactly 18 stages."
        )

    if len(
        _ROUTE_BY_STAGE
    ) != 18:

        raise SemanticV2PipelineAssemblyContractError(
            "Semantic-v2 route stage names are not unique."
        )

    if len(
        _ROUTE_BY_INDEX
    ) != 18:

        raise SemanticV2PipelineAssemblyContractError(
            "Semantic-v2 route indexes are not unique."
        )

    for expected_index, (
        spec,
        route,
    ) in enumerate(
        zip(
            specs,
            SEMANTIC_V2_STAGE_ROUTES,
        ),
        start=1,
    ):

        if route.index != expected_index:

            raise SemanticV2PipelineAssemblyContractError(
                "Semantic-v2 route index changed."
            )

        if spec.index != route.index:

            raise SemanticV2PipelineAssemblyContractError(
                "Route index disagrees with frozen orchestration."
            )

        if spec.name != route.stage:

            raise SemanticV2PipelineAssemblyContractError(
                "Route stage disagrees with frozen orchestration."
            )

        if (
            _action_value(
                spec.action
            )
            !=
            route.action
        ):

            raise SemanticV2PipelineAssemblyContractError(
                "Route action disagrees with frozen orchestration."
            )

        if (
            bool(
                spec.may_execute_before_shooter_resolution
            )
            !=
            route.may_execute_before_shooter_resolution
        ):

            raise SemanticV2PipelineAssemblyContractError(
                "Route pre-resolution policy disagrees with frozen orchestration."
            )

        if (
            bool(
                spec.requires_resolved_shooter
            )
            !=
            route.requires_resolved_shooter
        ):

            raise SemanticV2PipelineAssemblyContractError(
                "Route shooter-resolution policy disagrees with frozen orchestration."
            )

        if route.route_kind not in ROUTE_KINDS:

            raise SemanticV2PipelineAssemblyContractError(
                "Unknown Semantic-v2 route kind."
            )

        if not isinstance(
            route.module_name,
            str,
        ) or not route.module_name:

            raise SemanticV2PipelineAssemblyContractError(
                "Route module identity is invalid."
            )

        if not isinstance(
            route.builder_name,
            str,
        ) or not route.builder_name:

            raise SemanticV2PipelineAssemblyContractError(
                "Route builder identity is invalid."
            )

        if route.route_kind in {
            "FROZEN_V1_DIRECT",
            "FROZEN_V1_WRAPPED_GEOMETRY_READINESS",
        }:

            expected_builder = (
                _v1_adapter
                .EXPECTED_V1_BUILDER_BY_STAGE[
                    route.stage
                ]
            )

            if route.builder_name != expected_builder:

                raise SemanticV2PipelineAssemblyContractError(
                    "Frozen-v1 route builder identity changed."
                )

        if route.route_kind == "RESOLVED_V1_REUSE":

            if (
                EXPECTED_RESOLVED_REUSE_BUILDERS.get(
                    route.stage
                )
                !=
                route.builder_name
            ):

                raise SemanticV2PipelineAssemblyContractError(
                    "Resolved-v1 reuse route builder identity changed."
                )

        if route.stage in EXPECTED_TERMINAL_BUILDERS:

            if (
                EXPECTED_TERMINAL_BUILDERS[
                    route.stage
                ]
                !=
                route.builder_name
            ):

                raise SemanticV2PipelineAssemblyContractError(
                    "Terminal route builder identity changed."
                )

    annotation_route = _ROUTE_BY_STAGE[
        "annotation"
    ]

    if annotation_route.output_path_aware is not True:

        raise SemanticV2PipelineAssemblyContractError(
            "Annotation route must remain output-path aware."
        )

    for route in SEMANTIC_V2_STAGE_ROUTES:

        if (
            route.stage
            !=
            "annotation"
            and
            route.output_path_aware
        ):

            raise SemanticV2PipelineAssemblyContractError(
                "Only annotation may be output-path aware."
            )

    return True


def stage_route(
    stage: str,
) -> SemanticV2StageRoute:

    if not isinstance(
        stage,
        str,
    ):

        raise SemanticV2PipelineAssemblyContractError(
            "Stage name must be a string."
        )

    try:

        return _ROUTE_BY_STAGE[
            stage
        ]

    except KeyError as exc:

        raise SemanticV2PipelineAssemblyContractError(
            "Unknown Semantic-v2 stage: {!r}".format(
                stage
            )
        ) from exc


def stage_route_by_index(
    index: int,
) -> SemanticV2StageRoute:

    if (
        isinstance(
            index,
            bool,
        )
        or
        not isinstance(
            index,
            int,
        )
    ):

        raise SemanticV2PipelineAssemblyContractError(
            "Stage index must be an integer."
        )

    try:

        return _ROUTE_BY_INDEX[
            index
        ]

    except KeyError as exc:

        raise SemanticV2PipelineAssemblyContractError(
            "Unknown Semantic-v2 stage index: {!r}".format(
                index
            )
        ) from exc


def resolve_stage_builder(
    stage: str,
) -> Callable:

    validate_stage_routes()

    route = stage_route(
        stage
    )

    module = import_module(
        route.module_name
    )

    builder = getattr(
        module,
        route.builder_name,
        None,
    )

    if not callable(
        builder
    ):

        raise SemanticV2PipelineAssemblyContractError(
            "Stage builder is unavailable or non-callable: "
            "{}.{}".format(
                route.module_name,
                route.builder_name,
            )
        )

    if (
        getattr(
            builder,
            "__module__",
            None,
        )
        !=
        route.module_name
    ):

        raise SemanticV2PipelineAssemblyContractError(
            "Stage builder runtime module identity changed."
        )

    return builder


def resolve_all_stage_builders() -> Dict[str, Callable]:

    validate_stage_routes()

    resolved = {}

    for route in SEMANTIC_V2_STAGE_ROUTES:

        resolved[
            route.stage
        ] = resolve_stage_builder(
            route.stage
        )

    return resolved


def build_semantic_v2_execution_plan() -> Tuple[
    Tuple[
        SemanticV2StageRoute,
        Callable,
    ],
    ...,
]:

    validate_stage_routes()

    return tuple(
        (
            route,
            resolve_stage_builder(
                route.stage
            ),
        )
        for route
        in SEMANTIC_V2_STAGE_ROUTES
    )


def assembly_manifest():

    validate_stage_routes()

    return tuple(
        {
            "index":
                route.index,

            "stage":
                route.stage,

            "action":
                route.action,

            "route_kind":
                route.route_kind,

            "module_name":
                route.module_name,

            "builder_name":
                route.builder_name,

            "may_execute_before_shooter_resolution":
                route.may_execute_before_shooter_resolution,

            "requires_resolved_shooter":
                route.requires_resolved_shooter,

            "geometry_v2_stage":
                route.geometry_v2_stage,

            "output_path_aware":
                route.output_path_aware,
        }
        for route
        in SEMANTIC_V2_STAGE_ROUTES
    )
