from __future__ import annotations

from typing import Callable, Optional

from .semantic_contracts import SoccerClipAnalysisV2Result
from .semantic_v2_execution import run_semantic_v2_clip_pipeline


__all__ = [
    "analyze_soccer_clip",
]


def analyze_soccer_clip(
    video_path,
    annotated_output_path=None,
    *,
    geometry_inference_callable: Callable,
    frame_loader: Callable,
    png_encoder: Callable,
    video_sha256_provider: Optional[Callable] = None,
    shooter_confirmation_track_id: Optional[int] = None,
) -> SoccerClipAnalysisV2Result:
    """
    Analyze one local soccer clip with the canonical Semantic-v2 Soccer ML pipeline.

    This is the versioned application-facing Semantic-v2 API.

    Parameters
    ----------
    video_path:
        Local path to the input soccer clip.

    annotated_output_path:
        Optional local path for the annotated MP4.

    geometry_inference_callable:
        Configured Geometry-v2 inference callable consumed by canonical Stage 7.

    frame_loader:
        Callable that loads one exact zero-based frame from the local video.

    png_encoder:
        Callable that encodes one frame as PNG bytes for Geometry-v2 inference.

    video_sha256_provider:
        Optional callable returning the source-video SHA-256 used by Geometry-v2 provenance.

    shooter_confirmation_track_id:
        Optional externally confirmed shooter track ID.

        When the first Semantic-v2 pass returns SHOOTER_CONFIRMATION_REQUIRED,
        the caller should invoke this public API again from the original video
        with the selected track ID. The frozen execution driver performs a
        stateless replay from Stage 1.

    Returns
    -------
    SoccerClipAnalysisV2Result
        Canonical Semantic-v2 result.

        The result may terminate as:

        - COMPLETE
        - SHOOTER_CONFIRMATION_REQUIRED

        Pipeline failures propagate as exceptions after the result state is
        internally marked FAILED by the frozen execution driver.

    Notes
    -----
    This API intentionally does not own deployment-specific Geometry-v2
    service configuration. The caller provides configured runtime adapters.

    The API performs no direct model inference, network requests, frame IO,
    stage construction, stage execution, orchestration mutation, or result
    reconstruction. Those responsibilities remain behind the frozen
    Semantic-v2 execution boundary.
    """

    return run_semantic_v2_clip_pipeline(
        video_path=
            video_path,

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
