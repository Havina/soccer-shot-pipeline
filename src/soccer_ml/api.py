# NOTEBOOK43_APPLICATION_API_V1
"""
Application-facing Soccer ML clip API.

Primary entry point
-------------------
analyze_soccer_clip(video_path, annotated_output_path=None)

The implementation delegates to the canonical 18-stage
pipeline packaged and certified in soccer_ml.clip_pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
import math

import numpy as np

from .clip_pipeline import run_canonical_clip_pipeline
from .results import SoccerClipAnalysisResult


__all__ = [
    "analyze_soccer_clip",
    "soccer_clip_result_to_dict",
]


def _json_safe(value):
    """
    Convert public result data into JSON-safe Python primitives.
    """

    if is_dataclass(value):
        return _json_safe(
            asdict(value)
        )

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return _json_safe(
            value.tolist()
        )

    if isinstance(value, np.generic):
        return _json_safe(
            value.item()
        )

    if isinstance(value, float):

        if not math.isfinite(value):
            return None

        return float(value)

    if isinstance(value, dict):
        return {
            str(key):
                _json_safe(item)

            for key, item
            in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item)
            for item
            in value
        ]

    if isinstance(value, set):
        return [
            _json_safe(item)
            for item
            in sorted(
                value,
                key=str,
            )
        ]

    return value


def analyze_soccer_clip(
    video_path,
    annotated_output_path=None,
) -> SoccerClipAnalysisResult:
    """
    Analyze one local soccer clip with the canonical Soccer ML pipeline.

    Parameters
    ----------
    video_path:
        Local path to the input clip.

    annotated_output_path:
        Optional local output path for the annotated MP4.

    Returns
    -------
    SoccerClipAnalysisResult
        Structured result containing detections, tracking,
        projection, shooter, teams, target goal, geometry,
        defender context, xG, annotation and pipeline status.
    """

    return run_canonical_clip_pipeline(
        video_path=
            video_path,

        annotated_output_path=
            annotated_output_path,
    )


def soccer_clip_result_to_dict(
    result: SoccerClipAnalysisResult,
) -> dict:
    """
    JSON-safe representation suitable for a FastAPI response.
    """

    if not isinstance(
        result,
        SoccerClipAnalysisResult,
    ):
        raise TypeError(
            "result must be SoccerClipAnalysisResult"
        )

    return _json_safe(
        result
    )
