"""
Candidate shooter preview (app layer only).

After the automatic pass stops at stage 11 with SHOOTER_CONFIRMATION_REQUIRED,
the frozen result already contains everything needed to *show* the user who
the candidates are: every player track's bounding boxes per frame, the ball
track, and — in ``metadata["semantic_v2_shooter_stage11"]`` — each candidate's
estimated contact frame. This module extracts a compact evidence record from
that result and renders preview images (frame with candidate boxes, and a
jersey crop per candidate) with OpenCV.

Nothing here feeds back into inference. The evidence is read-only data copied
out of the frozen result; the images exist solely so a human can pick the
correct track.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

PREVIEW_COLORS_BGR = [
    (80, 220, 100),   # green
    (255, 170, 60),   # blue-ish
    (90, 90, 255),    # red
    (0, 200, 255),    # yellow
    (255, 90, 230),   # magenta
    (255, 255, 120),  # cyan
]
BALL_COLOR_BGR = (0, 140, 255)


def _value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _iter_tracks(player_tracks: Any):
    raw = _value(player_tracks, "tracks")
    if isinstance(raw, Mapping):
        for track_id, observations in raw.items():
            yield int(track_id), observations
    elif isinstance(raw, (list, tuple)):
        for track in raw:
            yield int(_value(track, "track_id")), _value(track, "observations")


def extract_candidate_evidence(result: Any) -> Optional[dict[str, Any]]:
    """
    Build the compact JSON record used for previews. Returns None when the
    result has no candidate decision (e.g. pipeline failed before stage 11).
    """
    decision = _value(result, "shooter_v2")
    if decision is None:
        return None
    candidate_ids = [int(t) for t in (_value(decision, "candidate_track_ids") or ())]
    if not candidate_ids:
        return None

    wanted = set(candidate_ids)
    tracks: dict[str, list[dict[str, float]]] = {}
    try:
        for track_id, observations in _iter_tracks(_value(result, "player_tracks")):
            if track_id not in wanted or not isinstance(observations, (list, tuple)):
                continue
            rows = []
            for obs in observations:
                try:
                    rows.append({
                        "frame_idx": int(_value(obs, "frame_idx")),
                        "x1": float(_value(obs, "x1")), "y1": float(_value(obs, "y1")),
                        "x2": float(_value(obs, "x2")), "y2": float(_value(obs, "y2")),
                    })
                except (TypeError, ValueError):
                    continue
            rows.sort(key=lambda r: r["frame_idx"])
            tracks[str(track_id)] = rows
    except Exception:
        return None

    ball_rows = []
    raw_ball = _value(_value(result, "ball_track"), "track")
    if isinstance(raw_ball, (list, tuple)):
        for row in raw_ball:
            try:
                ball_rows.append({
                    "frame_idx": int(_value(row, "frame_idx")),
                    "center_x": float(_value(row, "center_x")),
                    "center_y": float(_value(row, "center_y")),
                })
            except (TypeError, ValueError):
                continue
        ball_rows.sort(key=lambda r: r["frame_idx"])

    contact_frames: dict[str, int] = {}
    stage11 = (_value(result, "metadata") or {}).get("semantic_v2_shooter_stage11", {})
    for row in stage11.get("ranked_candidate_features", []) or []:
        try:
            if int(row["track_id"]) in wanted:
                contact_frames[str(int(row["track_id"]))] = int(row["best_backprojection_frame"])
        except (KeyError, TypeError, ValueError):
            continue

    # Shot frame: median of candidate contact frames, else first tracked ball frame.
    if contact_frames:
        shot_frame = int(np.median(list(contact_frames.values())))
    elif ball_rows:
        shot_frame = ball_rows[0]["frame_idx"]
    else:
        shot_frame = None

    auto_scores = {str(int(k)): float(v) for k, v in dict(_value(decision, "auto_scores") or {}).items()}
    return {
        "candidate_track_ids": candidate_ids,
        "auto_scores": auto_scores,
        "contact_frames": contact_frames,
        "shot_frame_idx": shot_frame,
        "ball_start_frame_idx": ball_rows[0]["frame_idx"] if ball_rows else None,
        "tracks": tracks,
        "ball": ball_rows,
        "source": "frozen automatic-pass result (player_tracks, ball_track, stage-11 metadata); display only",
    }


# --------------------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------------------
def _nearest_observation(rows: list[dict[str, float]], frame_idx: int, max_gap: int = 15) -> Optional[dict[str, float]]:
    best = None
    for row in rows:
        gap = abs(row["frame_idx"] - frame_idx)
        if best is None or gap < best[0]:
            best = (gap, row)
    if best is None or best[0] > max_gap:
        return None
    return best[1]


def _read_frame(video_path: Path, frame_idx: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def _label(img: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (w, h), base = cv2.getTextSize(text, font, scale, thick)
    y = max(h + 6, y)
    cv2.rectangle(img, (x, y - h - 6), (x + w + 8, y + base), color, -1)
    cv2.putText(img, text, (x + 4, y - 2), font, scale, (15, 15, 15), thick, cv2.LINE_AA)


def render_preview(video_path: Path, evidence: Mapping[str, Any], frame_idx: Optional[int] = None) -> Optional[bytes]:
    """Full frame with every candidate boxed and labelled. Returns JPEG bytes."""
    frame_idx = evidence.get("shot_frame_idx") if frame_idx is None else int(frame_idx)
    if frame_idx is None:
        return None
    frame = _read_frame(video_path, frame_idx)
    if frame is None:
        return None
    img = frame.copy()
    for i, track_id in enumerate(evidence["candidate_track_ids"]):
        color = PREVIEW_COLORS_BGR[i % len(PREVIEW_COLORS_BGR)]
        obs = _nearest_observation(evidence["tracks"].get(str(track_id), []), frame_idx)
        if obs is None:
            continue
        x1, y1, x2, y2 = (int(round(obs[k])) for k in ("x1", "y1", "x2", "y2"))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        _label(img, f"Track {track_id}", x1, y1 - 4, color)
    ball = _nearest_observation(
        [{"frame_idx": b["frame_idx"], "x1": b["center_x"], "y1": b["center_y"], "x2": b["center_x"], "y2": b["center_y"]} for b in evidence.get("ball", [])],
        frame_idx, max_gap=3,
    )
    if ball is not None:
        cv2.circle(img, (int(round(ball["x1"])), int(round(ball["y1"]))), 9, BALL_COLOR_BGR, 3)
    _label(img, f"frame {frame_idx}", 8, 28, (230, 230, 230))
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return buf.tobytes() if ok else None


def render_crop(video_path: Path, evidence: Mapping[str, Any], track_id: int, frame_idx: Optional[int] = None,
                target_height: int = 260) -> Optional[bytes]:
    """Padded crop of one candidate at the shot frame (or its contact frame). JPEG bytes."""
    rows = evidence["tracks"].get(str(track_id), [])
    if not rows:
        return None
    if frame_idx is None:
        frame_idx = evidence.get("contact_frames", {}).get(str(track_id), evidence.get("shot_frame_idx"))
    if frame_idx is None:
        frame_idx = rows[len(rows) // 2]["frame_idx"]
    obs = _nearest_observation(rows, int(frame_idx), max_gap=10**9)
    frame = _read_frame(video_path, obs["frame_idx"])
    if frame is None:
        return None
    h, w = frame.shape[:2]
    bw, bh = obs["x2"] - obs["x1"], obs["y2"] - obs["y1"]
    pad_x, pad_y = bw * 0.35, bh * 0.15
    x1 = int(max(0, obs["x1"] - pad_x)); y1 = int(max(0, obs["y1"] - pad_y))
    x2 = int(min(w, obs["x2"] + pad_x)); y2 = int(min(h, obs["y2"] + pad_y))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    scale = target_height / max(1, crop.shape[0])
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), target_height), interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return buf.tobytes() if ok else None


def load_evidence(job_dir: Path) -> Optional[dict[str, Any]]:
    path = job_dir / "candidates.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
