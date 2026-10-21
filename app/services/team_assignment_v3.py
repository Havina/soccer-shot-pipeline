"""
Local team-assignment overlay (v3).

The frozen V1 Stage-12 handler raises if *any* track has a tied modal jersey
cluster, including two-frame fragments at the edge of the picture. This overlay
monkey-patches only ``_notebook43_relative_track_summary`` at runtime:

* tracks with fewer than MIN_APPEARANCE_ROWS still receive a cluster, but they
  cannot fail the clip;
* remaining ties are broken by nearest cluster mean in appearance-feature space.

The frozen ``clip_pipeline.py`` file is not edited (its SHA-256 is unchanged).
Results that use this overlay are a local improvement, not Cell-40 certified.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


MIN_APPEARANCE_ROWS = 4

_installed = False
last_diagnostics: dict[str, Any] = {"installed": False, "ties_broken": 0, "short_tracks_ignored": 0}


def install() -> None:
    global _installed, last_diagnostics
    if _installed:
        return
    import soccer_ml.clip_pipeline as clip_pipeline

    original = clip_pipeline._notebook43_relative_track_summary
    feature_columns = list(clip_pipeline.NOTEBOOK43_APPEARANCE_FEATURE_COLUMNS)

    def patched(clustered):
        summary = original(clustered)
        return _resolve_ties(summary, clustered, feature_columns)

    clip_pipeline._notebook43_relative_track_summary = patched
    _installed = True
    last_diagnostics = {"installed": True, "ties_broken": 0, "short_tracks_ignored": 0}


def _resolve_ties(summary: pd.DataFrame, clustered: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    global last_diagnostics
    last_diagnostics = {"installed": True, "ties_broken": 0, "short_tracks_ignored": 0, "tracks": []}
    if summary is None or len(summary) == 0:
        return summary

    out = summary.copy()
    available = [c for c in feature_columns if c in clustered.columns]
    cluster_centers = _cluster_centers(clustered, available)

    for idx, row in out.iterrows():
        tied = bool(row.get("modal_tied"))
        rows = int(row.get("appearance_rows") or 0)
        track_id = int(row["track_id"])
        note = {"track_id": track_id, "appearance_rows": rows, "tied": tied}
        if not tied:
            last_diagnostics["tracks"].append(note)
            continue
        group = clustered[clustered["track_id"].astype(int) == track_id]
        chosen = _nearest_winner(group, cluster_centers, available)
        if chosen is None:
            last_diagnostics["tracks"].append({**note, "resolved": False})
            continue
        out.at[idx, "modal_cluster"] = int(chosen)
        if rows < MIN_APPEARANCE_ROWS:
            out.at[idx, "modal_tied"] = False
            last_diagnostics["short_tracks_ignored"] += 1
            note["action"] = "ignored_short_track"
        else:
            out.at[idx, "modal_tied"] = False
            last_diagnostics["ties_broken"] += 1
            note["action"] = "tie_broken_nearest_center"
        note["modal_cluster"] = int(chosen)
        last_diagnostics["tracks"].append(note)
    return out


def _cluster_centers(clustered: pd.DataFrame, columns: list[str]) -> dict[int, np.ndarray]:
    centers: dict[int, np.ndarray] = {}
    if not columns:
        return centers
    for cluster, group in clustered.groupby(clustered["appearance_cluster"].astype(int)):
        centers[int(cluster)] = group[columns].mean().to_numpy(dtype=float)
    return centers


def _nearest_winner(group: pd.DataFrame, centers: dict[int, np.ndarray], columns: list[str]) -> Optional[int]:
    if group is None or len(group) == 0:
        return None
    counts: dict[int, int] = {}
    for cluster in range(3):
        counts[cluster] = int((group["appearance_cluster"].astype(int) == cluster).sum())
    maximum = max(counts.values()) if counts else 0
    winners = [c for c, n in counts.items() if n == maximum and n > 0]
    if not winners:
        return None
    if len(winners) == 1 or not columns or not centers:
        return int(winners[0])
    track_mean = group[columns].mean().to_numpy(dtype=float)
    best, best_d = winners[0], float("inf")
    for cluster in winners:
        center = centers.get(cluster)
        if center is None:
            continue
        dist = float(np.linalg.norm(track_mean - center))
        if dist < best_d:
            best, best_d = cluster, dist
    return int(best)
