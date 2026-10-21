"""
App-layer clip preflight and auto-trim.

This never feeds results into the frozen pipeline as inference evidence: it
only decides *which file* is passed to ``analyze_soccer_clip``. Camera-cut
trimming, resolution warnings and pitch-visibility notes live here so the
certified 18-stage stack still sees a single continuous shot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np


CUT_DELTA_MIN = 16.0
CUT_Z_MIN = 5.0
MIN_SEGMENT_SECONDS = 2.0
SAMPLE_SIZE = (64, 36)
GREEN_HUE = (30, 100)
GREEN_SAT_MIN = 35


def inspect_clip(video_path: Path) -> dict[str, Any]:
    import cv2

    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open uploaded clip: {path}")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration_s = frame_count / fps if fps else 0.0

        green_samples: list[float] = []
        deltas: list[tuple[int, float]] = []
        prev = None
        step = max(1, frame_count // 8) if frame_count else 1
        for idx in range(frame_count):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            small = cv2.resize(frame, SAMPLE_SIZE).astype(np.float32)
            if prev is not None:
                deltas.append((idx, float(np.abs(small - prev).mean())))
            prev = small
            if idx % step == 0:
                green_samples.append(_green_fraction(frame))
        cuts = _detect_cuts(deltas, frame_count, fps)
    finally:
        cap.release()

    green_fraction = float(np.mean(green_samples)) if green_samples else 0.0
    warnings = _warnings(width, height, duration_s, cuts, green_fraction)
    return {
        "path": str(path),
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "duration_s": round(duration_s, 3),
        "cut_frames": cuts,
        "green_fraction": round(green_fraction, 4),
        "warnings": warnings,
        "trimmed": False,
        "trim_start_frame": 0,
        "trim_end_frame": frame_count,
        "original_duration_s": round(duration_s, 3),
        "original_path": str(path),
    }


def maybe_trim(video_path: Path, dest_dir: Path, *, enabled: bool) -> dict[str, Any]:
    info = inspect_clip(video_path)
    if not enabled or not info["cut_frames"]:
        return info

    start, end = _best_segment(
        Path(info["original_path"]),
        info["frame_count"],
        info["fps"],
        info["cut_frames"],
    )
    if end - start < int(MIN_SEGMENT_SECONDS * info["fps"]):
        info["warnings"].append("Camera cuts were found but no segment was long enough to keep; using the original clip.")
        return info
    if start <= 1 and end >= info["frame_count"] - 1:
        return info

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{Path(video_path).stem}_trimmed.mp4"
    written = _write_span(Path(info["original_path"]), dest, start, end, info["fps"], info["width"], info["height"])
    if written < int(MIN_SEGMENT_SECONDS * info["fps"]):
        info["warnings"].append("Auto-trim produced too few frames; using the original clip.")
        dest.unlink(missing_ok=True)
        return info

    trimmed = inspect_clip(dest)
    trimmed.update(
        {
            "trimmed": True,
            "trim_start_frame": start,
            "trim_end_frame": start + written,
            "original_duration_s": info["duration_s"],
            "original_path": info["original_path"],
            "original_cut_frames": info["cut_frames"],
            "warnings": info["warnings"] + trimmed["warnings"],
            "path": str(dest),
        }
    )
    trimmed["warnings"] = list(dict.fromkeys(trimmed["warnings"]))
    trimmed["warnings"].insert(
        0,
        f"Auto-trimmed at the first camera cut: kept {trimmed['duration_s']:.1f}s "
        f"(frames {start}-{start + written - 1}) of the original {info['duration_s']:.1f}s.",
    )
    return trimmed


def _detect_cuts(deltas: list[tuple[int, float]], frame_count: int, fps: float) -> list[int]:
    if len(deltas) < 8:
        return []
    values = np.array([d for _, d in deltas], dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) + 1e-6
    z = (values - median) / mad
    cuts: list[int] = []
    min_gap = max(3, int(0.4 * fps))
    for i, ((idx, delta), zi) in enumerate(zip(deltas, z)):
        if idx <= 8 or idx >= frame_count - 8:
            continue
        if delta < CUT_DELTA_MIN or zi < CUT_Z_MIN:
            continue
        neighborhood = values[max(0, i - 2): i + 3]
        if delta + 1e-6 < float(neighborhood.max()):
            continue
        if not cuts or (idx - cuts[-1]) > min_gap:
            cuts.append(int(idx))
    return cuts


def _best_segment(path: Path, frame_count: int, fps: float, cuts: list[int]) -> tuple[int, int]:
    bounds = [0, *cuts, frame_count]
    best = (0, frame_count, -1.0)
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        length = end - start
        if length < int(MIN_SEGMENT_SECONDS * fps):
            continue
        mid = (start + end) // 2
        frame = _read_frame(path, mid)
        green = _green_fraction(frame) if frame is not None else 0.0
        score = length * (0.25 + green)
        if score > best[2]:
            best = (start, end, score)
    return int(best[0]), int(best[1])


def _write_span(src: Path, dest: Path, start: int, end: int, fps: float, width: int, height: int) -> int:
    import cv2

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return 0
    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (int(width), int(height)))
    written = 0
    try:
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        for _ in range(int(end - start)):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
            written += 1
    finally:
        cap.release()
        writer.release()
    return written


def _read_frame(path: Path, idx: int) -> Optional[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def _green_fraction(frame: np.ndarray) -> float:
    import cv2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[:, :, 0], hsv[:, :, 1]
    mask = (hue >= GREEN_HUE[0]) & (hue <= GREEN_HUE[1]) & (sat >= GREEN_SAT_MIN)
    return float(mask.mean())


def _warnings(width: int, height: int, duration_s: float, cuts: list[int], green_fraction: float) -> list[str]:
    warnings: list[str] = []
    if width and width < 720:
        warnings.append(f"Low resolution ({width}×{height}). 720p or higher tracks the ball and pitch lines more reliably.")
    if height > width:
        warnings.append("Vertical / cropped video. A standard landscape broadcast angle of the penalty area works best.")
    if duration_s < 2.5:
        warnings.append("Clip is very short; the ball may not be visible for four frames after the strike.")
    if duration_s > 12:
        warnings.append("Clip is long. A 3–6 second window around the shot (no replay, no celebration cut) is more reliable.")
    if cuts:
        warnings.append(f"Detected {len(cuts)} camera cut{'s' if len(cuts) != 1 else ''} inside the clip.")
    if green_fraction < 0.18:
        warnings.append("Little grass is visible. Pitch calibration needs the penalty area and goal in view.")
    return warnings
