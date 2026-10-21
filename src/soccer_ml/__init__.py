from .shot_pipeline import (
    ShotAnalysisResult,
    SoccerShotPipeline,
    analyze_shot,
    get_default_pipeline,
)

from .video_source import (
    VideoMetadata,
    FrameExtractionInfo,
    VideoFrameWindow,
    LocalVideoSource,
    read_video_metadata,
    extract_frame_by_number,
    extract_frame_by_timestamp,
    frame_to_seconds,
    seconds_to_frame,
)

__all__ = [
    # Shot-analysis pipeline
    "ShotAnalysisResult",
    "SoccerShotPipeline",
    "analyze_shot",
    "get_default_pipeline",

    # Video-ingestion layer
    "VideoMetadata",
    "FrameExtractionInfo",
    "VideoFrameWindow",
    "LocalVideoSource",
    "read_video_metadata",
    "extract_frame_by_number",
    "extract_frame_by_timestamp",
    "frame_to_seconds",
    "seconds_to_frame",
]


# NOTEBOOK 36 — RAW VIDEO CANDIDATE RANKING EXPORTS
from .raw_video_ranker import (
    CandidateRankingResult,
    RawVideoCandidateRanker,
    rank_video_candidates,
)

try:
    __all__
except NameError:
    __all__ = []

for _export_name in [
    "CandidateRankingResult",
    "RawVideoCandidateRanker",
    "rank_video_candidates",
]:
    if _export_name not in __all__:
        __all__.append(_export_name)

del _export_name
