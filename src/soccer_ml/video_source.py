
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd


# ============================================================
# DATA OBJECTS
# ============================================================

@dataclass
class VideoMetadata:

    path: str
    can_open: bool

    fps: Optional[float] = None
    frame_count: Optional[int] = None

    width: Optional[int] = None
    height: Optional[int] = None

    duration_seconds: Optional[float] = None
    duration_minutes: Optional[float] = None

    fourcc: Optional[str] = None

    first_frame_readable: bool = False


@dataclass
class FrameExtractionInfo:

    video_path: str
    success: bool

    requested_frame: Optional[int] = None
    actual_frame: Optional[int] = None

    requested_seconds: Optional[float] = None
    frame_timestamp_seconds: Optional[float] = None

    fps: Optional[float] = None

    width: Optional[int] = None
    height: Optional[int] = None

    seek_error_frames: Optional[int] = None


@dataclass
class VideoFrameWindow:

    video_path: str

    center_frame: int
    start_frame: int
    end_frame: int

    requested_frames: int
    decoded_frames: int

    fps: float

    center_timestamp_seconds: float
    start_timestamp_seconds: float
    end_timestamp_seconds: float

    complete: bool


# ============================================================
# BASIC HELPERS
# ============================================================

def decode_fourcc(
    fourcc_value
):
    """
    Convert OpenCV integer FOURCC value into readable text.
    """

    try:

        value = int(
            fourcc_value
        )

        chars = [
            chr(
                (value >> (8 * i))
                & 0xFF
            )
            for i in range(4)
        ]

        decoded = "".join(
            chars
        ).strip()

        return (
            decoded
            if decoded
            else None
        )

    except Exception:

        return None


def frame_to_seconds(
    frame_number,
    fps
):
    """
    Convert zero-based frame number to seconds.
    """

    frame_number = int(
        frame_number
    )

    fps = float(
        fps
    )


    if fps <= 0:

        raise ValueError(
            "FPS must be positive."
        )


    if frame_number < 0:

        raise ValueError(
            "Frame number cannot be negative."
        )


    return float(
        frame_number
        /
        fps
    )


def seconds_to_frame(
    timestamp_seconds,
    fps
):
    """
    Convert timestamp to nearest zero-based frame number.
    """

    timestamp_seconds = float(
        timestamp_seconds
    )

    fps = float(
        fps
    )


    if fps <= 0:

        raise ValueError(
            "FPS must be positive."
        )


    if timestamp_seconds < 0:

        raise ValueError(
            "Timestamp cannot be negative."
        )


    return int(
        round(
            timestamp_seconds
            *
            fps
        )
    )


# ============================================================
# VIDEO METADATA
# ============================================================

def read_video_metadata(
    video_path
):
    """
    Read local-video metadata using OpenCV.
    """

    video_path = Path(
        video_path
    ).resolve()


    if not video_path.exists():

        raise FileNotFoundError(
            f"Video file does not exist: {video_path}"
        )


    capture = cv2.VideoCapture(
        str(
            video_path
        )
    )


    can_open = bool(
        capture.isOpened()
    )


    if not can_open:

        capture.release()

        return VideoMetadata(
            path=
                str(
                    video_path
                ),

            can_open=
                False,
        )


    raw_fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    raw_frame_count = float(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    raw_width = float(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    raw_height = float(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    raw_fourcc = capture.get(
        cv2.CAP_PROP_FOURCC
    )


    fps = (
        raw_fps
        if (
            np.isfinite(
                raw_fps
            )
            and
            raw_fps > 0
        )
        else None
    )


    frame_count = (
        int(
            round(
                raw_frame_count
            )
        )
        if (
            np.isfinite(
                raw_frame_count
            )
            and
            raw_frame_count > 0
        )
        else None
    )


    width = (
        int(
            round(
                raw_width
            )
        )
        if (
            np.isfinite(
                raw_width
            )
            and
            raw_width > 0
        )
        else None
    )


    height = (
        int(
            round(
                raw_height
            )
        )
        if (
            np.isfinite(
                raw_height
            )
            and
            raw_height > 0
        )
        else None
    )


    if (
        fps is not None
        and
        frame_count is not None
    ):

        duration_seconds = (
            frame_count
            /
            fps
        )

        duration_minutes = (
            duration_seconds
            /
            60.0
        )

    else:

        duration_seconds = None
        duration_minutes = None


    # --------------------------------------------------------
    # Confirm actual frame decoding
    # --------------------------------------------------------
    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        0
    )

    success, first_frame = (
        capture.read()
    )


    first_frame_readable = bool(
        success
        and
        first_frame is not None
        and
        first_frame.size > 0
    )


    capture.release()


    return VideoMetadata(

        path=
            str(
                video_path
            ),

        can_open=
            True,

        fps=
            fps,

        frame_count=
            frame_count,

        width=
            width,

        height=
            height,

        duration_seconds=
            (
                float(
                    duration_seconds
                )
                if duration_seconds
                is not None
                else None
            ),

        duration_minutes=
            (
                float(
                    duration_minutes
                )
                if duration_minutes
                is not None
                else None
            ),

        fourcc=
            decode_fourcc(
                raw_fourcc
            ),

        first_frame_readable=
            first_frame_readable,
    )


# ============================================================
# RANDOM-ACCESS FRAME EXTRACTION
# ============================================================

def extract_frame_by_number(
    video_path,
    frame_number,
    metadata=None
):
    """
    Extract one exact frame by zero-based frame number.
    """

    video_path = Path(
        video_path
    ).resolve()

    frame_number = int(
        frame_number
    )


    if not video_path.exists():

        raise FileNotFoundError(
            f"Video file does not exist: {video_path}"
        )


    if metadata is None:

        metadata = read_video_metadata(
            video_path
        )


    if not metadata.can_open:

        raise RuntimeError(
            f"Video cannot be opened: {video_path}"
        )


    if metadata.fps is None:

        raise RuntimeError(
            "Video FPS is unavailable."
        )


    if frame_number < 0:

        raise ValueError(
            "Frame number cannot be negative."
        )


    if (
        metadata.frame_count is not None
        and
        frame_number
        >= metadata.frame_count
    ):

        raise ValueError(
            f"Frame {frame_number} is outside video range "
            f"0-{metadata.frame_count - 1}."
        )


    capture = cv2.VideoCapture(
        str(
            video_path
        )
    )


    if not capture.isOpened():

        capture.release()

        raise RuntimeError(
            f"OpenCV could not open video: {video_path}"
        )


    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number
    )


    success, frame = (
        capture.read()
    )


    if success:

        next_position = capture.get(
            cv2.CAP_PROP_POS_FRAMES
        )


        if np.isfinite(
            next_position
        ):

            actual_frame = int(
                round(
                    next_position
                )
            ) - 1

        else:

            actual_frame = frame_number

    else:

        actual_frame = None


    capture.release()


    frame_valid = bool(
        success
        and
        frame is not None
        and
        frame.size > 0
    )


    if not frame_valid:

        frame = None


    requested_seconds = (
        frame_to_seconds(
            frame_number,
            metadata.fps
        )
    )


    if actual_frame is not None:

        actual_seconds = (
            frame_to_seconds(
                actual_frame,
                metadata.fps
            )
        )

        seek_error_frames = int(
            actual_frame
            -
            frame_number
        )

    else:

        actual_seconds = None
        seek_error_frames = None


    info = FrameExtractionInfo(

        video_path=
            str(
                video_path
            ),

        success=
            frame_valid,

        requested_frame=
            frame_number,

        actual_frame=
            actual_frame,

        requested_seconds=
            requested_seconds,

        frame_timestamp_seconds=
            actual_seconds,

        fps=
            metadata.fps,

        width=
            metadata.width,

        height=
            metadata.height,

        seek_error_frames=
            seek_error_frames,
    )


    return (
        frame,
        info
    )


def extract_frame_by_timestamp(
    video_path,
    timestamp_seconds,
    metadata=None
):
    """
    Extract nearest frame for a video timestamp.
    """

    if metadata is None:

        metadata = read_video_metadata(
            video_path
        )


    if metadata.fps is None:

        raise RuntimeError(
            "Video FPS is unavailable."
        )


    timestamp_seconds = float(
        timestamp_seconds
    )


    frame_number = seconds_to_frame(
        timestamp_seconds,
        metadata.fps
    )


    if metadata.frame_count is not None:

        frame_number = min(
            frame_number,
            metadata.frame_count - 1
        )


    frame, info = (
        extract_frame_by_number(
            video_path,
            frame_number,
            metadata=metadata
        )
    )


    info.requested_seconds = (
        timestamp_seconds
    )


    return (
        frame,
        info
    )


# ============================================================
# LOCAL VIDEO SOURCE
# ============================================================

class LocalVideoSource:
    """
    Reusable local-video interface for Soccer ML.
    """

    def __init__(
        self,
        video_path
    ):

        self.path = Path(
            video_path
        ).resolve()


        if not self.path.exists():

            raise FileNotFoundError(
                f"Video file does not exist: {self.path}"
            )


        self.metadata = (
            read_video_metadata(
                self.path
            )
        )


        if not self.metadata.can_open:

            raise RuntimeError(
                f"Video cannot be opened: {self.path}"
            )


        if not self.metadata.first_frame_readable:

            raise RuntimeError(
                "Video opened but the first frame "
                "could not be decoded."
            )


    @property
    def fps(self):

        return self.metadata.fps


    @property
    def frame_count(self):

        return self.metadata.frame_count


    @property
    def width(self):

        return self.metadata.width


    @property
    def height(self):

        return self.metadata.height


    @property
    def duration_seconds(self):

        return self.metadata.duration_seconds


    @property
    def duration_minutes(self):

        return self.metadata.duration_minutes


    def frame_to_seconds(
        self,
        frame_number
    ):

        return frame_to_seconds(
            frame_number,
            self.fps
        )


    def seconds_to_frame(
        self,
        timestamp_seconds
    ):

        return seconds_to_frame(
            timestamp_seconds,
            self.fps
        )


    def extract_frame(
        self,
        frame_number
    ):

        return extract_frame_by_number(
            self.path,
            frame_number,
            metadata=self.metadata
        )


    def extract_timestamp(
        self,
        timestamp_seconds
    ):

        return extract_frame_by_timestamp(
            self.path,
            timestamp_seconds,
            metadata=self.metadata
        )


    def extract_frames(
        self,
        frame_numbers
    ):
        """
        Extract multiple arbitrary frames.
        """

        frames_by_number = {}

        audit_rows = []


        for frame_number in frame_numbers:

            frame_number = int(
                frame_number
            )


            try:

                frame, info = (
                    self.extract_frame(
                        frame_number
                    )
                )


                frames_by_number[
                    frame_number
                ] = frame


                audit_rows.append(
                    {
                        "requested_frame":
                            frame_number,

                        "actual_frame":
                            info.actual_frame,

                        "timestamp_seconds":
                            info.frame_timestamp_seconds,

                        "success":
                            info.success,

                        "seek_error_frames":
                            info.seek_error_frames,

                        "width":
                            (
                                frame.shape[1]
                                if frame is not None
                                else None
                            ),

                        "height":
                            (
                                frame.shape[0]
                                if frame is not None
                                else None
                            ),
                    }
                )


            except Exception as exc:

                frames_by_number[
                    frame_number
                ] = None


                audit_rows.append(
                    {
                        "requested_frame":
                            frame_number,

                        "actual_frame":
                            None,

                        "timestamp_seconds":
                            None,

                        "success":
                            False,

                        "seek_error_frames":
                            None,

                        "width":
                            None,

                        "height":
                            None,

                        "error":
                            str(
                                exc
                            ),
                    }
                )


        return (
            frames_by_number,
            pd.DataFrame(
                audit_rows
            )
        )


    def extract_window(
        self,
        center_frame,
        before_seconds=1.0,
        after_seconds=1.0
    ):
        """
        Extract contiguous temporal window efficiently.

        Opens the video once, seeks once, then decodes
        sequentially.
        """

        center_frame = int(
            center_frame
        )

        before_seconds = float(
            before_seconds
        )

        after_seconds = float(
            after_seconds
        )


        if before_seconds < 0:

            raise ValueError(
                "before_seconds cannot be negative."
            )


        if after_seconds < 0:

            raise ValueError(
                "after_seconds cannot be negative."
            )


        if center_frame < 0:

            raise ValueError(
                "center_frame cannot be negative."
            )


        if (
            self.frame_count is not None
            and
            center_frame >= self.frame_count
        ):

            raise ValueError(
                f"Center frame {center_frame} "
                "is outside the video."
            )


        before_frames = int(
            round(
                before_seconds
                *
                self.fps
            )
        )

        after_frames = int(
            round(
                after_seconds
                *
                self.fps
            )
        )


        requested_start = (
            center_frame
            -
            before_frames
        )

        requested_end = (
            center_frame
            +
            after_frames
        )


        start_frame = max(
            0,
            requested_start
        )


        if self.frame_count is not None:

            end_frame = min(
                requested_end,
                self.frame_count - 1
            )

        else:

            end_frame = (
                requested_end
            )


        expected_frames = list(
            range(
                start_frame,
                end_frame + 1
            )
        )


        capture = cv2.VideoCapture(
            str(
                self.path
            )
        )


        if not capture.isOpened():

            capture.release()

            raise RuntimeError(
                f"Could not open video: {self.path}"
            )


        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            start_frame
        )


        frames = {}

        audit_rows = []


        for expected_frame in expected_frames:

            success, frame = (
                capture.read()
            )


            if success:

                next_position = capture.get(
                    cv2.CAP_PROP_POS_FRAMES
                )


                if np.isfinite(
                    next_position
                ):

                    actual_frame = int(
                        round(
                            next_position
                        )
                    ) - 1

                else:

                    actual_frame = (
                        expected_frame
                    )


                valid = bool(
                    frame is not None
                    and
                    frame.size > 0
                )


                frames[
                    expected_frame
                ] = (
                    frame
                    if valid
                    else None
                )

            else:

                actual_frame = None
                valid = False

                frames[
                    expected_frame
                ] = None


            audit_rows.append(
                {
                    "expected_frame":
                        expected_frame,

                    "actual_frame":
                        actual_frame,

                    "timestamp_seconds":
                        self.frame_to_seconds(
                            expected_frame
                        ),

                    "success":
                        valid,

                    "seek_error_frames":
                        (
                            actual_frame
                            -
                            expected_frame

                            if actual_frame
                            is not None

                            else None
                        ),
                }
            )


            if not success:
                break


        capture.release()


        audit = pd.DataFrame(
            audit_rows
        )


        decoded_frames = int(
            audit[
                "success"
            ].sum()
        )


        requested_frames = len(
            expected_frames
        )


        complete = bool(
            len(audit)
            ==
            requested_frames
            and
            audit[
                "success"
            ].all()
        )


        info = VideoFrameWindow(

            video_path=
                str(
                    self.path
                ),

            center_frame=
                center_frame,

            start_frame=
                start_frame,

            end_frame=
                end_frame,

            requested_frames=
                requested_frames,

            decoded_frames=
                decoded_frames,

            fps=
                float(
                    self.fps
                ),

            center_timestamp_seconds=
                self.frame_to_seconds(
                    center_frame
                ),

            start_timestamp_seconds=
                self.frame_to_seconds(
                    start_frame
                ),

            end_timestamp_seconds=
                self.frame_to_seconds(
                    end_frame
                ),

            complete=
                complete,
        )


        return (
            frames,
            audit,
            info
        )


    def summary(
        self
    ):

        return {
            "path":
                str(
                    self.path
                ),

            "fps":
                self.fps,

            "frame_count":
                self.frame_count,

            "resolution":
                f"{self.width}x{self.height}",

            "duration_seconds":
                self.duration_seconds,

            "duration_minutes":
                self.duration_minutes,

            "fourcc":
                self.metadata.fourcc,
        }


__all__ = [
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
