"""
Application configuration and frozen-source integrity policy.

The single source of truth for what the app is allowed to run is the
Notebook-50 Cell 41-LOCAL app-readiness freeze. Hashes are read from that
freeze at startup rather than duplicated here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

UPLOADS_DIR = Path(os.environ.get("SOCCER_ML_UPLOADS_DIR", APP_ROOT / "uploads")).resolve()
RESULTS_DIR = Path(os.environ.get("SOCCER_ML_RESULTS_DIR", APP_ROOT / "results")).resolve()

APP_READINESS_FREEZE = (
    PROJECT_ROOT / "outputs" / "notebook50_local" / "freeze"
    / "notebook50_local_semantic_v2_app_readiness_freeze.json"
)

FROZEN_V1_DIR = SRC_ROOT / "soccer_ml"
FROZEN_V2_DIR = SRC_ROOT / "soccer_ml" / "v2"

ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".m4v"}
MAX_UPLOAD_BYTES = int(os.environ.get("SOCCER_ML_MAX_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))

# The GPU runtime is single-tenant: exactly one pipeline execution at a time.
ANALYSIS_WORKERS = 1

# Set to "0" to skip the live geometry readiness probe at startup (health will
# still report the probe as not performed; analysis requests will trigger it).
PROBE_RUNTIME_AT_STARTUP = os.environ.get("SOCCER_ML_PROBE_AT_STARTUP", "1") != "0"

# Local overlays. These never edit frozen sources. Results that use them are
# not Cell-40 certified. Set any to "0" to restore the certified-only path.
AUTO_TRIM = os.environ.get("SOCCER_ML_AUTO_TRIM", "1") != "0"
MULTIFRAME_GEOMETRY = os.environ.get("SOCCER_ML_MULTIFRAME_GEOMETRY", "1") != "0"
TEAM_ASSIGNMENT_V3 = os.environ.get("SOCCER_ML_TEAM_ASSIGNMENT_V3", "1") != "0"
GOAL_END_V3 = os.environ.get("SOCCER_ML_GOAL_END_V3", "1") != "0"
