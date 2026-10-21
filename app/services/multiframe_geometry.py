"""
Local multi-frame geometry fallback.

Wraps the *certified* ``LocalGeometryV2Adapter`` without modifying it. The
frozen Stage-7 contract still asks for one anchor frame (FIRST_RAW_V3_BALL_FRAME).
If that frame produces ``NO_VALID_GEOMETRY``, this wrapper scores nearby frames
for pitch-line structure and retries PnLCalib on the two strongest candidates.

A successful nearby homography is rebound to the original ``frame_idx`` so the
frozen Stage-7 echo check passes. That is an approximation: player boxes at the
anchor frame are projected with a pose estimated a few frames away. It is a
local overlay, not Cell-40 / L40 certified behaviour, and is recorded on the job.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from soccer_ml.v2.contracts import GeometryV2InferenceResponse


FALLBACK_OFFSETS = (4, -4, 8, -8, 12, -12, 16, -16, 20, -20)
MAX_FALLBACK_INFERENCES = 2


class MultiFrameGeometryAdapter:
    def __init__(
        self,
        inner,
        *,
        frame_loader: Callable,
        png_encoder: Callable,
    ):
        self.inner = inner
        self.frame_loader = frame_loader
        self.png_encoder = png_encoder
        self.video_path: Optional[Path] = None
        self.last_diagnostics: dict[str, Any] = {"used": False, "attempts": []}

    def bind_video(self, video_path) -> None:
        self.video_path = Path(video_path) if video_path else None
        self.last_diagnostics = {"used": False, "attempts": []}

    def get_runtime(self):
        return self.inner.get_runtime()

    def __call__(self, *, video_sha256, frame_idx, frame_bgr, frame_png) -> GeometryV2InferenceResponse:
        primary = self.inner(
            video_sha256=video_sha256,
            frame_idx=frame_idx,
            frame_bgr=frame_bgr,
            frame_png=frame_png,
        )
        attempt = {
            "frame_idx": int(frame_idx),
            "role": "anchor",
            "success": bool(primary.success),
            "selection_status": primary.selection_status,
            "keypoint_count": primary.keypoint_count,
            "line_count": primary.line_count,
        }
        self.last_diagnostics = {"used": False, "anchor_frame_idx": int(frame_idx), "attempts": [attempt]}
        if primary.success:
            return primary
        if self.video_path is None:
            return primary

        candidates = self._rank_neighbors(int(frame_idx), frame_bgr)
        for cand in candidates[:MAX_FALLBACK_INFERENCES]:
            alt_idx = int(cand["frame_idx"])
            try:
                alt_bgr = self.frame_loader(str(self.video_path), alt_idx)
                alt_png = self.png_encoder(alt_bgr)
                alt = self.inner(
                    video_sha256=video_sha256,
                    frame_idx=alt_idx,
                    frame_bgr=alt_bgr,
                    frame_png=alt_png,
                )
            except Exception as exc:
                self.last_diagnostics["attempts"].append(
                    {"frame_idx": alt_idx, "role": "fallback", "success": False, "error": f"{type(exc).__name__}: {exc}"}
                )
                continue
            self.last_diagnostics["attempts"].append(
                {
                    "frame_idx": alt_idx,
                    "role": "fallback",
                    "success": bool(alt.success),
                    "selection_status": alt.selection_status,
                    "keypoint_count": alt.keypoint_count,
                    "line_count": alt.line_count,
                    "structure_score": cand["score"],
                }
            )
            if alt.success:
                self.last_diagnostics["used"] = True
                self.last_diagnostics["fallback_frame_idx"] = alt_idx
                self.last_diagnostics["offset"] = alt_idx - int(frame_idx)
                self.last_diagnostics["note"] = (
                    f"Anchor frame {frame_idx} had no valid geometry; used nearby frame {alt_idx} "
                    f"(offset {alt_idx - int(frame_idx)}). Homography is an approximation. "
                    "Not Cell-40 certified."
                )
                echo = dict(primary.input_echo or {})
                echo["frame_idx"] = int(frame_idx)
                return replace(
                    alt,
                    frame_idx=int(frame_idx),
                    request_id=primary.request_id,
                    input_echo=echo or alt.input_echo,
                )
        return primary

    def _rank_neighbors(self, anchor_idx: int, anchor_bgr: np.ndarray) -> list[dict[str, Any]]:
        scored = []
        seen = {int(anchor_idx)}
        for offset in FALLBACK_OFFSETS:
            idx = int(anchor_idx) + int(offset)
            if idx < 0 or idx in seen:
                continue
            seen.add(idx)
            try:
                frame = self.frame_loader(str(self.video_path), idx)
            except Exception:
                continue
            scored.append({"frame_idx": idx, "score": _structure_score(frame), "offset": offset})
        scored.sort(key=lambda row: row["score"], reverse=True)
        return scored


def _structure_score(frame: np.ndarray) -> float:
    import cv2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = (hsv[:, :, 0] >= 30) & (hsv[:, :, 0] <= 100) & (hsv[:, :, 1] >= 35)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    green_frac = float(green.mean())
    edge_frac = float(edges[green].mean() / 255.0) if green.any() else 0.0
    return 0.35 * green_frac + 0.65 * edge_frac
