
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
import time

import cv2
import joblib
import numpy as np
import pandas as pd


# ============================================================
# RESULT CONTAINER
# ============================================================

@dataclass
class CandidateRankingResult:
    """
    Result from raw-video candidate ranking.

    Attributes
    ----------
    video_path
        Source local video.

    metadata
        Video scan / pipeline metadata.

    signals
        One row per sampled frame.

    candidate_members
        Raw Stage-1 candidate samples with temporal event IDs.

    ranked_events
        One row per temporal event, ranked by shot-event probability.
    """

    video_path: Path
    metadata: dict
    signals: pd.DataFrame
    candidate_members: pd.DataFrame
    ranked_events: pd.DataFrame


# ============================================================
# VISUAL FEATURE HELPERS
# ============================================================

def _green_fraction(
    bgr_frame: np.ndarray,
) -> float:
    """
    Broad soccer-pitch green fraction.
    """

    hsv = cv2.cvtColor(
        bgr_frame,
        cv2.COLOR_BGR2HSV,
    )

    lower_green = np.array(
        [30, 35, 25],
        dtype=np.uint8,
    )

    upper_green = np.array(
        [100, 255, 255],
        dtype=np.uint8,
    )

    mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green,
    )

    return float(
        (mask > 0).mean()
    )


def _hsv_histogram(
    bgr_frame: np.ndarray,
) -> np.ndarray:
    """
    Normalized 2-D HSV histogram.
    """

    hsv = cv2.cvtColor(
        bgr_frame,
        cv2.COLOR_BGR2HSV,
    )

    histogram = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [30, 32],
        [0, 180, 0, 256],
    )

    histogram = cv2.normalize(
        histogram,
        histogram,
    ).flatten()

    return histogram


# ============================================================
# MAIN PIPELINE
# ============================================================

class RawVideoCandidateRanker:
    """
    High-recall soccer shot-candidate ranking pipeline.

    Pipeline
    --------
    raw video
        ->
    blind Stage-1 sampled-frame signals
        ->
    high-recall raw candidates
        ->
    multi-frame temporal events
        ->
    six event-level features
        ->
    Notebook-35 logistic event ranker

    Notes
    -----
    The bundled Notebook-35 model is exploratory and was trained
    from two matches. Ranking probabilities should therefore not
    be interpreted as production-calibrated shot probabilities.
    """

    def __init__(
        self,
        project_root: Optional[str | Path] = None,
        ranker_path: Optional[str | Path] = None,
        ranker_metadata_path: Optional[str | Path] = None,
        stage1_metadata_path: Optional[str | Path] = None,
        detector_config_path: Optional[str | Path] = None,
    ):

        if project_root is None:

            project_root = (
                Path(__file__)
                .resolve()
                .parents[2]
            )

        self.project_root = Path(
            project_root
        ).resolve()

        self.processed_dir = (
            self.project_root
            / "data"
            / "processed"
        )

        self.models_dir = (
            self.project_root
            / "models"
        )


        # ----------------------------------------------------
        # Artifact paths
        # ----------------------------------------------------

        self.ranker_path = Path(
            ranker_path
            if ranker_path is not None
            else (
                self.models_dir
                / "notebook35_event_ranker.joblib"
            )
        )


        self.ranker_metadata_path = Path(
            ranker_metadata_path
            if ranker_metadata_path is not None
            else (
                self.processed_dir
                / "notebook35_final_ranker_metadata.json"
            )
        )


        self.stage1_metadata_path = Path(
            stage1_metadata_path
            if stage1_metadata_path is not None
            else (
                self.processed_dir
                / "notebook35_stage1_signal_metadata.json"
            )
        )


        self.detector_config_path = Path(
            detector_config_path
            if detector_config_path is not None
            else (
                self.processed_dir
                / "notebook34_frozen_detector_config.json"
            )
        )


        required_paths = [
            self.ranker_path,
            self.ranker_metadata_path,
            self.stage1_metadata_path,
            self.detector_config_path,
        ]


        missing = [
            path
            for path in required_paths
            if not path.exists()
        ]


        if missing:

            raise FileNotFoundError(
                "Missing required candidate-ranking artifacts: "
                +
                ", ".join(
                    str(path)
                    for path in missing
                )
            )


        # ----------------------------------------------------
        # Load model / metadata
        # ----------------------------------------------------

        self.ranker = joblib.load(
            self.ranker_path
        )


        with open(
            self.ranker_metadata_path,
            "r",
            encoding="utf-8",
        ) as f:

            self.ranker_metadata = json.load(
                f
            )


        with open(
            self.stage1_metadata_path,
            "r",
            encoding="utf-8",
        ) as f:

            self.stage1_metadata = json.load(
                f
            )


        with open(
            self.detector_config_path,
            "r",
            encoding="utf-8",
        ) as f:

            self.detector_config = json.load(
                f
            )


        # ----------------------------------------------------
        # Frozen Stage-1 configuration
        # ----------------------------------------------------

        configuration = (
            self.stage1_metadata[
                "configuration"
            ]
        )


        self.sample_every_n_frames = int(
            configuration[
                "sample_every_n_frames"
            ]
        )


        self.feature_width = int(
            configuration[
                "feature_width"
            ]
        )


        self.feature_height = int(
            configuration[
                "feature_height"
            ]
        )


        self.past_window_seconds = float(
            configuration[
                "past_window_seconds"
            ]
        )


        self.future_window_seconds = float(
            configuration[
                "future_window_seconds"
            ]
        )


        self.min_pitch_green_fraction = float(
            configuration[
                "min_pitch_green_fraction"
            ]
        )


        self.stage1_score_quantile = float(
            configuration[
                "stage1_score_quantile"
            ]
        )


        self.event_cluster_gap_seconds = float(
            self.detector_config[
                "candidate_cluster_gap_seconds"
            ]
        )


        self.ranker_features = list(
            self.ranker_metadata[
                "features"
            ]
        )


    # ========================================================
    # BASIC INFORMATION
    # ========================================================

    def summary(
        self,
    ) -> dict:

        return {
            "project_root":
                str(
                    self.project_root
                ),

            "sample_every_n_frames":
                self.sample_every_n_frames,

            "feature_resolution":
                (
                    self.feature_width,
                    self.feature_height,
                ),

            "past_window_seconds":
                self.past_window_seconds,

            "future_window_seconds":
                self.future_window_seconds,

            "min_pitch_green_fraction":
                self.min_pitch_green_fraction,

            "stage1_score_quantile":
                self.stage1_score_quantile,

            "event_cluster_gap_seconds":
                self.event_cluster_gap_seconds,

            "ranker_features":
                self.ranker_features,

            "production_validated":
                bool(
                    self.ranker_metadata[
                        "production_validated"
                    ]
                ),
        }


    # ========================================================
    # RAW VIDEO -> STAGE-1 SIGNALS
    # ========================================================

    def extract_stage1_signals(
        self,
        video_path: str | Path,
        match_key: str = "video",
        progress_every_samples: int = 500,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Blindly scan a local video and construct Stage-1 signals.

        No labels, YOLO, homography, or xG information is required.
        """

        video_path = Path(
            video_path
        ).resolve()


        if not video_path.exists():

            raise FileNotFoundError(
                video_path
            )


        capture = cv2.VideoCapture(
            str(
                video_path
            )
        )


        if not capture.isOpened():

            raise RuntimeError(
                f"Could not open video: {video_path}"
            )


        fps = float(
            capture.get(
                cv2.CAP_PROP_FPS
            )
        )


        frame_count = int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )


        if (
            not np.isfinite(
                fps
            )
            or
            fps <= 0
        ):

            capture.release()

            raise RuntimeError(
                f"Invalid video FPS: {fps}"
            )


        samples_per_second = (
            fps
            /
            self.sample_every_n_frames
        )


        past_window_samples = int(
            round(
                self.past_window_seconds
                *
                samples_per_second
            )
        )


        future_window_samples = int(
            round(
                self.future_window_seconds
                *
                samples_per_second
            )
        )


        rows = []

        previous_gray = None
        previous_histogram = None

        decoded_frames = 0
        sampled_frames = 0

        start_time = time.time()


        while True:

            success, frame = (
                capture.read()
            )


            if not success:

                break


            frame_number = (
                decoded_frames
            )


            if (
                frame_number
                %
                self.sample_every_n_frames
                ==
                0
            ):

                small = cv2.resize(
                    frame,
                    (
                        self.feature_width,
                        self.feature_height,
                    ),
                    interpolation=
                        cv2.INTER_AREA,
                )


                gray = cv2.cvtColor(
                    small,
                    cv2.COLOR_BGR2GRAY,
                )


                mean_brightness = float(
                    gray.mean()
                )


                brightness_std = float(
                    gray.std()
                )


                green_fraction = (
                    _green_fraction(
                        small
                    )
                )


                edges = cv2.Canny(
                    gray,
                    80,
                    160,
                )


                edge_density = float(
                    (edges > 0).mean()
                )


                histogram = (
                    _hsv_histogram(
                        small
                    )
                )


                if previous_gray is None:

                    mean_abs_frame_change = (
                        np.nan
                    )

                    changed_pixel_fraction = (
                        np.nan
                    )

                else:

                    difference = cv2.absdiff(
                        gray,
                        previous_gray,
                    )


                    mean_abs_frame_change = float(
                        difference.mean()
                    )


                    changed_pixel_fraction = float(
                        (
                            difference
                            >=
                            20
                        ).mean()
                    )


                if previous_histogram is None:

                    histogram_correlation = (
                        np.nan
                    )

                    histogram_change = (
                        np.nan
                    )

                else:

                    histogram_correlation = float(
                        cv2.compareHist(
                            previous_histogram.astype(
                                np.float32
                            ),
                            histogram.astype(
                                np.float32
                            ),
                            cv2.HISTCMP_CORREL,
                        )
                    )


                    histogram_change = float(
                        1.0
                        -
                        histogram_correlation
                    )


                timestamp_seconds = float(
                    frame_number
                    /
                    fps
                )


                rows.append(
                    {
                        "match_key":
                            match_key,

                        "sample_index":
                            sampled_frames,

                        "frame_number":
                            frame_number,

                        "timestamp_seconds":
                            timestamp_seconds,

                        "mean_brightness":
                            mean_brightness,

                        "brightness_std":
                            brightness_std,

                        "green_fraction":
                            green_fraction,

                        "edge_density":
                            edge_density,

                        "mean_abs_frame_change":
                            mean_abs_frame_change,

                        "changed_pixel_fraction":
                            changed_pixel_fraction,

                        "histogram_correlation":
                            histogram_correlation,

                        "histogram_change":
                            histogram_change,
                    }
                )


                previous_gray = (
                    gray.copy()
                )


                previous_histogram = (
                    histogram.copy()
                )


                sampled_frames += 1


                if (
                    progress_every_samples
                    and
                    sampled_frames
                    %
                    progress_every_samples
                    ==
                    0
                ):

                    elapsed = (
                        time.time()
                        -
                        start_time
                    )


                    print(
                        f"{match_key}: "
                        f"{sampled_frames} samples | "
                        f"{timestamp_seconds:.1f}s video | "
                        f"{elapsed:.1f}s elapsed"
                    )


            decoded_frames += 1


        capture.release()


        signals = pd.DataFrame(
            rows
        )


        if signals.empty:

            raise RuntimeError(
                "No video samples were extracted."
            )


        # ----------------------------------------------------
        # Past pitch context
        # ----------------------------------------------------

        signals[
            "past_green_mean"
        ] = (
            signals[
                "green_fraction"
            ]
            .rolling(
                window=
                    past_window_samples,

                min_periods=1,
            )
            .mean()
        )


        # ----------------------------------------------------
        # Future context
        # ----------------------------------------------------

        green_values = (
            signals[
                "green_fraction"
            ]
            .to_numpy(
                dtype=float
            )
        )


        histogram_values = (
            signals[
                "histogram_change"
            ]
            .to_numpy(
                dtype=float
            )
        )


        frame_change_values = (
            signals[
                "mean_abs_frame_change"
            ]
            .to_numpy(
                dtype=float
            )
        )


        future_min_green = []
        future_max_hist_change = []
        future_max_frame_change = []


        n_samples = len(
            signals
        )


        for i in range(
            n_samples
        ):

            future_start = (
                i + 1
            )


            future_end = min(
                n_samples,
                i
                +
                future_window_samples
                +
                1,
            )


            if future_start >= future_end:

                future_min_green.append(
                    np.nan
                )

                future_max_hist_change.append(
                    np.nan
                )

                future_max_frame_change.append(
                    np.nan
                )

                continue


            future_min_green.append(
                float(
                    np.nanmin(
                        green_values[
                            future_start:
                            future_end
                        ]
                    )
                )
            )


            future_max_hist_change.append(
                float(
                    np.nanmax(
                        histogram_values[
                            future_start:
                            future_end
                        ]
                    )
                )
            )


            future_max_frame_change.append(
                float(
                    np.nanmax(
                        frame_change_values[
                            future_start:
                            future_end
                        ]
                    )
                )
            )


        signals[
            "future_min_green"
        ] = (
            future_min_green
        )


        signals[
            "future_max_hist_change"
        ] = (
            future_max_hist_change
        )


        signals[
            "future_max_frame_change"
        ] = (
            future_max_frame_change
        )


        signals[
            "future_green_drop"
        ] = (
            signals[
                "past_green_mean"
            ]
            -
            signals[
                "future_min_green"
            ]
        ).clip(
            lower=0.0
        )


        # ----------------------------------------------------
        # Within-video percentile ranks
        # ----------------------------------------------------

        rank_features = [
            "edge_density",
            "changed_pixel_fraction",
            "future_green_drop",
            "future_max_hist_change",
        ]


        for feature in rank_features:

            signals[
                f"{feature}_rank"
            ] = (
                signals[
                    feature
                ]
                .rank(
                    method="average",
                    pct=True,
                )
            )


        # ----------------------------------------------------
        # Frozen Notebook-34 Stage-1 score
        # ----------------------------------------------------

        signals[
            "candidate_score"
        ] = (

            0.40
            *
            signals[
                "edge_density_rank"
            ]

            +

            0.35
            *
            signals[
                "future_green_drop_rank"
            ]

            +

            0.15
            *
            signals[
                "future_max_hist_change_rank"
            ]

            +

            0.10
            *
            signals[
                "changed_pixel_fraction_rank"
            ]
        )


        signals[
            "pitch_view"
        ] = (
            signals[
                "past_green_mean"
            ]
            >=
            self.min_pitch_green_fraction
        )


        valid_score_mask = (
            signals[
                "candidate_score"
            ]
            .notna()
            &
            signals[
                "pitch_view"
            ]
        )


        if not valid_score_mask.any():

            raise RuntimeError(
                "No valid Stage-1 pitch-view scores were produced."
            )


        stage1_threshold = float(
            signals.loc[
                valid_score_mask,
                "candidate_score",
            ]
            .quantile(
                self.stage1_score_quantile
            )
        )


        signals[
            "raw_candidate"
        ] = (
            signals[
                "pitch_view"
            ]
            &
            (
                signals[
                    "candidate_score"
                ]
                >=
                stage1_threshold
            )
        )


        elapsed_seconds = float(
            time.time()
            -
            start_time
        )


        metadata = {
            "video_path":
                str(
                    video_path
                ),

            "match_key":
                match_key,

            "fps":
                fps,

            "frame_count":
                frame_count,

            "decoded_frames":
                decoded_frames,

            "sampled_frames":
                int(
                    len(
                        signals
                    )
                ),

            "sample_interval_seconds":
                float(
                    self.sample_every_n_frames
                    /
                    fps
                ),

            "stage1_score_threshold":
                stage1_threshold,

            "raw_candidates":
                int(
                    signals[
                        "raw_candidate"
                    ]
                    .sum()
                ),

            "elapsed_seconds":
                elapsed_seconds,
        }


        return (
            signals,
            metadata,
        )


    # ========================================================
    # STAGE-1 SIGNALS -> MULTI-FRAME EVENTS
    # ========================================================

    def assign_events(
        self,
        signal_table: pd.DataFrame,
        match_key: str = "video",
    ) -> pd.DataFrame:
        """
        Keep all raw candidate samples and assign temporal event IDs.
        """

        candidates = (
            signal_table[
                self._normalize_bool_series(
                    signal_table[
                        "raw_candidate"
                    ]
                )
            ]
            .copy()
            .sort_values(
                "timestamp_seconds"
            )
            .reset_index(
                drop=True
            )
        )


        if candidates.empty:

            candidates[
                "event_number"
            ] = pd.Series(
                dtype=int
            )

            candidates[
                "event_id"
            ] = pd.Series(
                dtype=str
            )

            return candidates


        event_numbers = []

        current_event = 0
        previous_time = None


        for candidate_time in (
            candidates[
                "timestamp_seconds"
            ]
            .astype(float)
        ):

            if previous_time is None:

                current_event += 1


            elif (
                candidate_time
                -
                previous_time
                >
                self.event_cluster_gap_seconds
            ):

                current_event += 1


            event_numbers.append(
                current_event
            )


            previous_time = (
                candidate_time
            )


        candidates[
            "event_number"
        ] = (
            event_numbers
        )


        candidates[
            "event_id"
        ] = (
            candidates[
                "event_number"
            ]
            .apply(
                lambda value:
                    (
                        f"{match_key}_event_"
                        f"{int(value):03d}"
                    )
            )
        )


        return candidates


    # ========================================================
    # MULTI-FRAME EVENTS -> FINAL SIX FEATURES
    # ========================================================

    def build_event_features(
        self,
        candidate_members: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Aggregate all samples in each temporal event.

        Produces the exact six-feature schema expected by the
        Notebook-35 serialized ranker.
        """

        if candidate_members.empty:

            columns = [
                "match_key",
                "event_id",
                "event_start_seconds",
                "event_end_seconds",
                "event_duration_seconds",
                "member_samples",
                "peak_candidate_time_seconds",
                "peak_candidate_score",
            ] + self.ranker_features

            return pd.DataFrame(
                columns=columns
            )


        event_rows = []


        for (
            match_key,
            event_id
        ), group in (
            candidate_members.groupby(
                [
                    "match_key",
                    "event_id",
                ],
                sort=False,
            )
        ):

            group = (
                group
                .sort_values(
                    "timestamp_seconds"
                )
                .reset_index(
                    drop=True
                )
            )


            start_time = float(
                group[
                    "timestamp_seconds"
                ]
                .min()
            )


            end_time = float(
                group[
                    "timestamp_seconds"
                ]
                .max()
            )


            peak_index = int(
                group[
                    "candidate_score"
                ]
                .idxmax()
            )


            peak_row = (
                group.loc[
                    peak_index
                ]
            )


            row = {
                "match_key":
                    match_key,

                "event_id":
                    event_id,

                "event_start_seconds":
                    start_time,

                "event_end_seconds":
                    end_time,

                "event_duration_seconds":
                    float(
                        end_time
                        -
                        start_time
                    ),

                "member_samples":
                    int(
                        len(
                            group
                        )
                    ),

                "peak_candidate_time_seconds":
                    float(
                        peak_row[
                            "timestamp_seconds"
                        ]
                    ),

                "peak_candidate_score":
                    float(
                        peak_row[
                            "candidate_score"
                        ]
                    ),

                # --------------------------------------------
                # Exact final Notebook-35 model features
                # --------------------------------------------

                "candidate_score_mean":
                    float(
                        group[
                            "candidate_score"
                        ]
                        .astype(float)
                        .mean()
                    ),

                "candidate_score_std":
                    float(
                        group[
                            "candidate_score"
                        ]
                        .astype(float)
                        .std(
                            ddof=0
                        )
                    ),

                "histogram_change_mean":
                    float(
                        group[
                            "histogram_change"
                        ]
                        .astype(float)
                        .mean()
                    ),

                "mean_abs_frame_change_mean":
                    float(
                        group[
                            "mean_abs_frame_change"
                        ]
                        .astype(float)
                        .mean()
                    ),

                "future_green_drop_mean":
                    float(
                        group[
                            "future_green_drop"
                        ]
                        .astype(float)
                        .mean()
                    ),

                "changed_pixel_fraction_mean":
                    float(
                        group[
                            "changed_pixel_fraction"
                        ]
                        .astype(float)
                        .mean()
                    ),
            }


            event_rows.append(
                row
            )


        events = (
            pd.DataFrame(
                event_rows
            )
            .sort_values(
                [
                    "match_key",
                    "event_start_seconds",
                ]
            )
            .reset_index(
                drop=True
            )
        )


        missing_features = [
            feature
            for feature in self.ranker_features
            if feature
            not in events.columns
        ]


        if missing_features:

            raise RuntimeError(
                "Missing ranker features: "
                +
                ", ".join(
                    missing_features
                )
            )


        if (
            events[
                self.ranker_features
            ]
            .isna()
            .any()
            .any()
        ):

            missing_counts = (
                events[
                    self.ranker_features
                ]
                .isna()
                .sum()
            )

            raise RuntimeError(
                "Missing event-feature values: "
                +
                str(
                    missing_counts[
                        missing_counts > 0
                    ]
                    .to_dict()
                )
            )


        return events


    # ========================================================
    # EVENT FEATURES -> RANKED EVENTS
    # ========================================================

    def rank_events(
        self,
        event_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply the serialized Notebook-35 event ranker.
        """

        if event_features.empty:

            result = (
                event_features.copy()
            )

            result[
                "ranking_probability"
            ] = pd.Series(
                dtype=float
            )

            result[
                "rank"
            ] = pd.Series(
                dtype=int
            )

            return result


        X = (
            event_features[
                self.ranker_features
            ]
            .astype(float)
        )


        probabilities = (
            self.ranker.predict_proba(
                X
            )[
                :,
                1
            ]
        )


        ranked = (
            event_features
            .copy()
        )


        ranked[
            "ranking_probability"
        ] = (
            probabilities
        )


        ranked = (
            ranked
            .sort_values(
                "ranking_probability",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )


        ranked[
            "rank"
        ] = np.arange(
            1,
            len(
                ranked
            )
            +
            1,
        )


        return ranked


    # ========================================================
    # SAVED SIGNAL TABLE -> RANKED EVENTS
    # ========================================================

    def rank_from_signals(
        self,
        signal_table: pd.DataFrame,
        match_key: str = "video",
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """
        Rank events from a previously extracted Stage-1 signal table.

        Useful for regression testing without decoding video again.
        """

        signals = (
            signal_table.copy()
        )


        signals[
            "match_key"
        ] = (
            match_key
        )


        members = (
            self.assign_events(
                signals,
                match_key=
                    match_key,
            )
        )


        features = (
            self.build_event_features(
                members
            )
        )


        ranked = (
            self.rank_events(
                features
            )
        )


        return (
            members,
            ranked,
        )


    # ========================================================
    # FULL RAW-VIDEO PIPELINE
    # ========================================================

    def analyze_video(
        self,
        video_path: str | Path,
        match_key: str = "video",
        progress_every_samples: int = 500,
    ) -> CandidateRankingResult:
        """
        Run the complete candidate-ranking pipeline on a local video.
        """

        signals, metadata = (
            self.extract_stage1_signals(
                video_path=
                    video_path,

                match_key=
                    match_key,

                progress_every_samples=
                    progress_every_samples,
            )
        )


        candidate_members = (
            self.assign_events(
                signals,
                match_key=
                    match_key,
            )
        )


        event_features = (
            self.build_event_features(
                candidate_members
            )
        )


        ranked_events = (
            self.rank_events(
                event_features
            )
        )


        metadata = {
            **metadata,

            "temporal_events":
                int(
                    len(
                        ranked_events
                    )
                ),

            "ranker_feature_count":
                int(
                    len(
                        self.ranker_features
                    )
                ),

            "production_validated":
                bool(
                    self.ranker_metadata[
                        "production_validated"
                    ]
                ),
        }


        return CandidateRankingResult(
            video_path=
                Path(
                    video_path
                ).resolve(),

            metadata=
                metadata,

            signals=
                signals,

            candidate_members=
                candidate_members,

            ranked_events=
                ranked_events,
        )


    # ========================================================
    # INTERNAL UTILITIES
    # ========================================================

    @staticmethod
    def _normalize_bool_series(
        series: pd.Series,
    ) -> pd.Series:

        if pd.api.types.is_bool_dtype(
            series
        ):

            return series.astype(
                bool
            )


        return (
            series
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                    "yes",
                ]
            )
        )


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def rank_video_candidates(
    video_path: str | Path,
    match_key: str = "video",
    project_root: Optional[str | Path] = None,
    progress_every_samples: int = 500,
) -> CandidateRankingResult:
    """
    Convenience wrapper for one-shot local-video ranking.
    """

    ranker = RawVideoCandidateRanker(
        project_root=
            project_root
    )


    return ranker.analyze_video(
        video_path=
            video_path,

        match_key=
            match_key,

        progress_every_samples=
            progress_every_samples,
    )
