"""
Local goal-end overlay.

Frozen Stage 13 picks LEFT/RIGHT as ends of the 105 m pitch model using
image-space ball motion. A valid homography can still swap the two ends, so a
box strike is scored as a 90 m shot at the far goal.

This overlay does not edit frozen files. After Stage 13 writes
``result.target_goal``, we compare the shooter's pitch position to both goal
centres. If the frozen choice is the far end and the other end is a box-range
distance, we flip the public target so Stages 14–17 (shot geometry, defenders,
xG, annotation) use the near goal.

Results that use this overlay are not Cell-40 certified.
"""

from __future__ import annotations

import math
from typing import Any, Optional


NEAR_MAX_M = 35.0
FAR_MIN_M = 45.0

_installed = False
last_diagnostics: dict[str, Any] = {"installed": False, "flipped": False}


def install() -> None:
    global _installed, last_diagnostics
    if _installed:
        return
    import soccer_ml.v2.target_goal_stage13_inference as stage13

    original_builder = stage13.build_target_goal_v2_stage13_inference_handler

    def wrapped_builder():
        inner = original_builder()

        def handler(result):
            inner(result)
            apply_goal_end_overlay(result)
            return None

        return handler

    # Frozen assembly rejects a builder whose __module__ is not the route module.
    wrapped_builder.__module__ = original_builder.__module__
    wrapped_builder.__name__ = original_builder.__name__
    wrapped_builder.__qualname__ = getattr(original_builder, "__qualname__", original_builder.__name__)
    stage13.build_target_goal_v2_stage13_inference_handler = wrapped_builder
    _installed = True
    last_diagnostics = {"installed": True, "flipped": False}


def pitch_goal_distances(pitch_xy) -> Optional[dict[str, float]]:
    xy = _xy(pitch_xy)
    if xy is None:
        return None
    from soccer_ml.v2.target_goal_stage13_inference import (
        TARGET_GOAL_V2_GOAL_CENTER_Y_M,
        TARGET_GOAL_V2_PITCH_LENGTH_M,
    )

    left = (0.0, float(TARGET_GOAL_V2_GOAL_CENTER_Y_M))
    right = (float(TARGET_GOAL_V2_PITCH_LENGTH_M), float(TARGET_GOAL_V2_GOAL_CENTER_Y_M))
    return {
        "LEFT": _dist(xy, left),
        "RIGHT": _dist(xy, right),
    }


def apply_goal_end_overlay(result: Any) -> dict[str, Any]:
    global last_diagnostics
    distances = pitch_goal_distances(getattr(getattr(result, "shooter", None), "pitch_xy", None))
    chosen = str(getattr(getattr(result, "target_goal", None), "target_goal", "") or "")
    diag: dict[str, Any] = {
        "installed": True,
        "flipped": False,
        "original_target_goal": chosen or None,
        "resolved_target_goal": chosen or None,
        "distance_to_left_goal_m": None if distances is None else distances["LEFT"],
        "distance_to_right_goal_m": None if distances is None else distances["RIGHT"],
        "reason": "no_pitch_xy_or_target" if distances is None or chosen not in {"LEFT", "RIGHT"} else "kept_frozen_choice",
    }
    if distances is None or chosen not in {"LEFT", "RIGHT"}:
        last_diagnostics = diag
        return diag

    d_left, d_right = distances["LEFT"], distances["RIGHT"]
    d_chosen = distances[chosen]
    other = "RIGHT" if chosen == "LEFT" else "LEFT"
    d_other = distances[other]
    d_near = min(d_left, d_right)
    near_side = "LEFT" if d_left <= d_right else "RIGHT"

    should_flip = d_chosen >= FAR_MIN_M and d_other <= NEAR_MAX_M and other == near_side
    if should_flip:
        result.target_goal.target_goal = other
        try:
            result.target_goal.confidence = min(float(result.target_goal.confidence or 1.0), 0.7)
        except Exception:
            pass
        _patch_metadata(result, other, diag)
        diag.update(
            {
                "flipped": True,
                "resolved_target_goal": other,
                "reason": (
                    f"Frozen Stage 13 chose {chosen} at {d_chosen:.1f} m; "
                    f"the other end ({other}) is {d_other:.1f} m. Using the near goal. Not Cell-40 certified."
                ),
            }
        )
    last_diagnostics = diag
    return diag


def geometry_warnings(metrics_like: Any) -> list[str]:
    warnings: list[str] = []
    d = _opt_float(getattr(metrics_like, "distance_to_goal_m", None))
    angle = _opt_float(getattr(metrics_like, "shot_angle_deg", None))
    d_left = _opt_float(getattr(metrics_like, "distance_to_left_goal_m", None))
    d_right = _opt_float(getattr(metrics_like, "distance_to_right_goal_m", None))
    chosen = getattr(metrics_like, "target_goal", None)
    if d_left is not None and d_right is not None and chosen in {"LEFT", "RIGHT"}:
        d_chosen = d_left if chosen == "LEFT" else d_right
        d_other = d_right if chosen == "LEFT" else d_left
        if d_chosen >= FAR_MIN_M and d_other <= NEAR_MAX_M:
            warnings.append(
                f"Chosen goal is the far pitch end ({d_chosen:.1f} m); the other end is {d_other:.1f} m. "
                "This usually means the camera pose swapped the two goals."
            )
    if d is not None and d > 40:
        warnings.append(
            f"Distance {d:.1f} m is a long-range figure. Penalty-area shots are usually 8-25 m."
        )
    if angle is not None and angle < 8 and (d is None or d > 30):
        warnings.append(
            f"Shot angle {angle:.1f}° is very narrow, consistent with a far-goal measurement rather than a box strike."
        )
    return warnings


def _patch_metadata(result: Any, new_goal: str, diag: dict[str, Any]) -> None:
    from soccer_ml.v2.target_goal_stage13_inference import TARGET_GOAL_V2_METADATA_KEY

    metadata = getattr(result, "metadata", None)
    if not isinstance(metadata, dict):
        return
    row = metadata.get(TARGET_GOAL_V2_METADATA_KEY)
    if isinstance(row, dict):
        row["target_goal"] = new_goal
        row["local_goal_end_overlay"] = {
            "flipped": True,
            "original_target_goal": diag.get("original_target_goal"),
            "resolved_target_goal": new_goal,
        }


def _xy(value) -> Optional[tuple[float, float]]:
    if value is None:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except Exception:
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _opt_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None
