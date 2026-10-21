"""
Soccer ML Geometry-v2 -- LOCAL RTX-2060 PnLCalib worker.

Runs inside ``.venv-geometry-v2`` (Python 3.10.10 / Torch 2.0.0+cu117).

This is a NEW, LOCAL-ONLY worker.  It is NOT the frozen L40 worker
(``outputs/notebook50_local/server_handoff/extracted/geometry_v2_worker.py``,
SHA-256 2c3fc0b8...) and it does not claim that worker's certification.

What it reproduces, source-backed:

  * PnLCalib ``inference.py::inference()`` preprocessing and model execution
    (BGR->RGB, to_tensor, resize 540x960, SV_kp + SV_lines HRNet-w48 models,
    ``get_keypoints_from_heatmap_batch_maxpool[_l]``, ``coords_to_dict`` with
    the frozen thresholds, ``complete_keypoints``).
  * ``FramebyFrameCalib.heuristic_voting(refine_lines=False)``  -> BASE
    ``FramebyFrameCalib.heuristic_voting(refine_lines=True)``   -> PNL_REFINED
    (output dict shape matches the stored Notebook-37 ``base_result`` /
    ``pnl_result`` exactly: mode / use_ransac / rep_err / cam_params / calib_plane).
  * cam_params -> ground-plane homography using PnLCalib's projection
    (P = Q . R . [I | -t]) restricted to z = 0 and re-expressed in
    corner-origin pitch coordinates (0..105 m, 0..68 m).  This derivation was
    verified against all 8 stored Notebook-37 matrices (4 events x BASE/PNL)
    with max |delta| = 2.274e-13.
  * The frozen operational policy PNLCALIB_PNL_PRIMARY_BASE_FALLBACK_V1
    (kp 0.3434 / line 0.7867 / PNL primary / BASE fallback).
  * The historical runner CLI/manifest/result contract (one manifest row in,
    one legacy result row out) and the frozen Geometry-v2 wire protocol
    ``soccer-ml-geometry-v2/1`` for request manifests and inference responses.

Reference-free: the worker receives only an image and a frame index.  It never
receives shooter identity, target goal, xG, defender data, expected
homographies or any ground truth, and it rejects manifests carrying such keys.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------------------------------------------------------------------------------
# IDENTITIES
# ----------------------------------------------------------------------------------------------------

PROTOCOL_VERSION = "soccer-ml-geometry-v2/1"

RUNTIME_PROFILE = "PNLCALIB_NOTEBOOK37_LOCAL_RTX2060_RUNTIME_V1"

HISTORICAL_RUNTIME_PROFILE = "PNLCALIB_NOTEBOOK37_EXACT_RUNTIME_V1"

HISTORICAL_L40_RUNTIME_FINGERPRINT_SHA256 = (
    "715f9b64b964b9a2b8d8011d231d132819ccff540563aa34e9284ae94c976d5e"
)

POLICY_NAME = "PNLCALIB_PNL_PRIMARY_BASE_FALLBACK_V1"

POLICY_FINGERPRINT = "62e0431373c6a665cbe07561c178cf750dd78ea610b11678644a28d7a0d342fd"

GEOMETRY_FRAME_POLICY = "GEOMETRY_FRAME_POLICY_V2_0"

EXPECTED_REPO_HEAD = "8c87391d6f4ea40c5e4d65e61529916c7a49ce62"

EXPECTED_KP_SHA = "7ea78fa76aaf94976a8eca428d6e3c59697a93430cba1a4603e20284b61f5113"

EXPECTED_LINES_SHA = "d72f4ed71734a2e3df9fa084f666e9b8adaef21bf69bac8952d6d3f970ff7455"

EXPECTED_POLICY_FILE_SHA = "90e71b7ebd7d002b62e702d9af61b6cbaa24abc2833f0b19b576549d065a66d1"

EXPECTED_GPU_NAME_SUBSTRING = "RTX 2060"

KP_THRESHOLD = 0.3434

LINE_THRESHOLD = 0.7867

MODEL_INPUT_WIDTH = 960

MODEL_INPUT_HEIGHT = 540

PITCH_LENGTH_M = 105.0

PITCH_WIDTH_M = 68.0

HOMOGRAPHY_DERIVATION = {
    "pitch_origin": "CORNER_ORIGIN_X_0_105_Y_0_68",
    "method": (
        "P = Q @ (R @ [I | -t]) from PnLCalib cam_params; "
        "H_pitch_to_image = P[:, [0, 1, 3]] @ T where T translates corner-origin "
        "pitch metres to PnLCalib centred metres (x-52.5, y-34); "
        "normalised so H[2,2] == 1; H_image_to_pitch = inv(H_pitch_to_image) renormalised"
    ),
    "verified_against": "notebook37_pnlcalib_operational_predictions.json (4 events x BASE/PNL)",
    "verified_max_abs_delta": 2.274e-13,
}

EXPECTED_VERSIONS = {
    "python": "3.10.10",
    "torch": "2.0.0",
    "torch_cuda": "11.7",
    "torchvision": "0.15.0",
    "opencv": "4.7.0",
    "scipy": "1.10.1",
    "numpy": "1.23.5",
    "pyyaml": "6.0",
    "shapely": "2.0.7",
    "humanfriendly": "10.0",
    "coloredlogs": "15.0.1",
}


# ----------------------------------------------------------------------------------------------------
# PATHS (project-relative; this file lives at <ROOT>/local_geometry_v2/)
# ----------------------------------------------------------------------------------------------------

WORKER_PATH = Path(__file__).resolve()

LOCAL_GEOMETRY_ROOT = WORKER_PATH.parent

PROJECT_ROOT = LOCAL_GEOMETRY_ROOT.parent

PNLCALIB_REPO = (LOCAL_GEOMETRY_ROOT / "PnLCalib").resolve()

WEIGHT_KP = (PNLCALIB_REPO / "weights" / "SV_kp").resolve()

WEIGHT_LINES = (PNLCALIB_REPO / "weights" / "SV_lines").resolve()

CONFIG_KP = (PNLCALIB_REPO / "config" / "hrnetv2_w48.yaml").resolve()

CONFIG_LINES = (PNLCALIB_REPO / "config" / "hrnetv2_w48_l.yaml").resolve()

POLICY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "notebook37_gpu_return_bundle"
    / "frozen_policy"
    / "notebook37_pnlcalib_operational_policy.json"
).resolve()


# ----------------------------------------------------------------------------------------------------
# HASHING / SUBPROCESS HELPERS
# ----------------------------------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def semantic_sha256(value: Dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_text(command, cwd=None) -> Dict[str, Any]:
    result = subprocess.run(
        [str(item) for item in command],
        cwd=str(cwd) if cwd is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return {
        "returncode": int(result.returncode),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


# ----------------------------------------------------------------------------------------------------
# HOMOGRAPHY MATHS
# ----------------------------------------------------------------------------------------------------

def projection_from_cam_params(cam_params: Dict[str, Any]):
    """Identical to PnLCalib inference.py::projection_from_cam_params."""
    import numpy as np

    x_focal_length = cam_params["x_focal_length"]
    y_focal_length = cam_params["y_focal_length"]
    principal_point = np.array(cam_params["principal_point"], dtype=np.float64)
    position_meters = np.array(cam_params["position_meters"], dtype=np.float64)
    rotation = np.array(cam_params["rotation_matrix"], dtype=np.float64)

    It = np.eye(4)[:-1]
    It[:, -1] = -position_meters
    Q = np.array(
        [
            [x_focal_length, 0, principal_point[0]],
            [0, y_focal_length, principal_point[1]],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    return Q @ (rotation @ It)


def homographies_from_cam_params(cam_params: Dict[str, Any]) -> Tuple[List[List[float]], List[List[float]]]:
    """Return (H_image_to_pitch, H_pitch_to_image) in corner-origin pitch metres."""
    import numpy as np

    P = projection_from_cam_params(cam_params)
    T = np.array(
        [
            [1.0, 0.0, -PITCH_LENGTH_M / 2.0],
            [0.0, 1.0, -PITCH_WIDTH_M / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    H_pitch_to_image = P[:, [0, 1, 3]] @ T
    H_pitch_to_image = H_pitch_to_image / H_pitch_to_image[2, 2]
    H_image_to_pitch = np.linalg.inv(H_pitch_to_image)
    H_image_to_pitch = H_image_to_pitch / H_image_to_pitch[2, 2]
    return H_image_to_pitch.tolist(), H_pitch_to_image.tolist()


def validate_homography_matrix(value) -> Dict[str, Any]:
    """Mirrors the frozen policy's mathematical validity checks (and the stored *_validation rows)."""
    import numpy as np

    result = {
        "available": value is not None,
        "shape_valid": False,
        "finite": False,
        "full_rank": False,
        "invertible": False,
        "mathematically_valid": False,
        "determinant": None,
    }
    if value is None:
        return result
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except Exception:
        return result
    result["shape_valid"] = matrix.shape == (3, 3)
    if not result["shape_valid"]:
        return result
    result["finite"] = bool(np.isfinite(matrix).all())
    if not result["finite"]:
        return result
    result["full_rank"] = int(np.linalg.matrix_rank(matrix)) == 3
    try:
        np.linalg.inv(matrix)
        result["invertible"] = True
    except Exception:
        result["invertible"] = False
    result["determinant"] = float(np.linalg.det(matrix))
    result["mathematically_valid"] = bool(
        result["shape_valid"] and result["finite"] and result["full_rank"] and result["invertible"]
    )
    return result


# ----------------------------------------------------------------------------------------------------
# RUNTIME CONTRACT / READINESS
# ----------------------------------------------------------------------------------------------------

def _runtime_versions() -> Tuple[Dict[str, Any], List[str]]:
    failures: List[str] = []
    try:
        import cv2
        import coloredlogs
        import humanfriendly
        import matplotlib
        import numpy
        import scipy
        import shapely
        import torch
        import torchvision
        import yaml

        cudnn_version = torch.backends.cudnn.version()
        versions = {
            "python": sys.version.split()[0],
            "torch": str(torch.__version__).split("+")[0],
            "torch_full": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "torchvision": str(torchvision.__version__).split("+")[0],
            "torchvision_full": str(torchvision.__version__),
            "opencv": str(cv2.__version__),
            "scipy": str(scipy.__version__),
            "numpy": str(numpy.__version__),
            "pyyaml": str(yaml.__version__),
            "shapely": str(shapely.__version__),
            "humanfriendly": str(humanfriendly.__version__),
            "coloredlogs": str(coloredlogs.__version__),
            "matplotlib": str(matplotlib.__version__),
            "cudnn": int(cudnn_version) if cudnn_version is not None else None,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_name": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None,
            "gpu_capability": (
                list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
            ),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "platform": sys.platform,
        }
    except Exception as exc:
        versions = {"runtime_import_error": f"{type(exc).__name__}: {exc}"}
        failures.append("runtime import probe failed")
        return versions, failures

    for key, expected in EXPECTED_VERSIONS.items():
        if versions.get(key) != expected:
            failures.append(f"runtime mismatch for {key}: {versions.get(key)!r} != {expected!r}")

    if versions.get("cuda_available") is not True:
        failures.append("CUDA is not available")

    if EXPECTED_GPU_NAME_SUBSTRING not in str(versions.get("gpu_name")):
        failures.append(f"{EXPECTED_GPU_NAME_SUBSTRING} is not the active GPU")

    return versions, failures


def runtime_contract():
    failure_reasons: List[str] = []

    for path in (PNLCALIB_REPO, WEIGHT_KP, WEIGHT_LINES, CONFIG_KP, CONFIG_LINES, POLICY_PATH):
        if not path.exists():
            failure_reasons.append(f"missing required path: {path}")

    kp_sha = sha256_file(WEIGHT_KP) if WEIGHT_KP.exists() else None
    lines_sha = sha256_file(WEIGHT_LINES) if WEIGHT_LINES.exists() else None
    policy_file_sha = sha256_file(POLICY_PATH) if POLICY_PATH.exists() else None
    worker_sha = sha256_file(WORKER_PATH)

    if kp_sha != EXPECTED_KP_SHA:
        failure_reasons.append("SV_kp SHA-256 mismatch")
    if lines_sha != EXPECTED_LINES_SHA:
        failure_reasons.append("SV_lines SHA-256 mismatch")
    if policy_file_sha != EXPECTED_POLICY_FILE_SHA:
        failure_reasons.append("policy file SHA-256 mismatch")

    git_head = run_text(["git", "rev-parse", "HEAD"], cwd=PNLCALIB_REPO)
    repo_head = git_head["stdout"].strip() if git_head["returncode"] == 0 else None
    if repo_head != EXPECTED_REPO_HEAD:
        failure_reasons.append("PnLCalib repository HEAD mismatch")

    git_status = run_text(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=PNLCALIB_REPO,
    )
    tracked_source_clean = bool(git_status["returncode"] == 0 and git_status["stdout"].strip() == "")
    if not tracked_source_clean:
        failure_reasons.append("PnLCalib tracked source is not clean")

    runtime_versions, version_failures = _runtime_versions()
    failure_reasons.extend(version_failures)

    try:
        policy_payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        policy_payload = {}
        failure_reasons.append(f"policy JSON load failed: {type(exc).__name__}: {exc}")

    if policy_payload.get("policy_name") != POLICY_NAME:
        failure_reasons.append("policy_name mismatch")
    if policy_payload.get("policy_sha256") != POLICY_FINGERPRINT:
        failure_reasons.append("policy fingerprint mismatch")
    if float(policy_payload.get("kp_threshold", float("nan"))) != KP_THRESHOLD:
        failure_reasons.append("keypoint threshold mismatch")
    if float(policy_payload.get("line_threshold", float("nan"))) != LINE_THRESHOLD:
        failure_reasons.append("line threshold mismatch")
    if policy_payload.get("pnl_refinement_enabled") is not True:
        failure_reasons.append("PnL refinement policy mismatch")
    if policy_payload.get("primary_variant") != "PNL_REFINED":
        failure_reasons.append("primary variant mismatch")
    if policy_payload.get("fallback_variant") != "BASE":
        failure_reasons.append("fallback variant mismatch")

    policy_contract = {
        "policy_name": POLICY_NAME,
        "policy_fingerprint": POLICY_FINGERPRINT,
        "policy_file_sha256": policy_file_sha,
        "keypoint_threshold": KP_THRESHOLD,
        "line_threshold": LINE_THRESHOLD,
        "pnl_refinement_enabled": True,
        "primary_variant": "PNL_REFINED",
        "fallback_variant": "BASE",
    }

    weights = {
        "SV_kp": {
            "path": str(WEIGHT_KP),
            "size_bytes": int(WEIGHT_KP.stat().st_size) if WEIGHT_KP.exists() else None,
            "sha256": kp_sha,
        },
        "SV_lines": {
            "path": str(WEIGHT_LINES),
            "size_bytes": int(WEIGHT_LINES.stat().st_size) if WEIGHT_LINES.exists() else None,
            "sha256": lines_sha,
        },
    }

    runtime_manifest: Dict[str, Any] = {
        "runtime_profile": RUNTIME_PROFILE,
        "certification_lineage": {
            "historical_runtime_profile": HISTORICAL_RUNTIME_PROFILE,
            "historical_l40_runtime_fingerprint_sha256": HISTORICAL_L40_RUNTIME_FINGERPRINT_SHA256,
            "relationship": "LOCAL_RTX2060_REIMPLEMENTATION_OF_HISTORICAL_RUNNER_CONTRACT",
            "claims_l40_certification": False,
        },
        "repository": {
            "path": str(PNLCALIB_REPO),
            "head": repo_head,
            "tracked_source_clean": tracked_source_clean,
        },
        "worker": {
            "path": str(WORKER_PATH),
            "sha256": worker_sha,
        },
        "runtime": runtime_versions,
        "policy_fingerprint": POLICY_FINGERPRINT,
        "policy_file_sha256": policy_file_sha,
        "weight_hashes": {
            "SV_kp": kp_sha,
            "SV_lines": lines_sha,
        },
        "homography_derivation": HOMOGRAPHY_DERIVATION,
    }
    runtime_manifest["fingerprint_sha256"] = semantic_sha256(runtime_manifest)

    ready = failure_reasons == []
    failure_reason = None if ready else "; ".join(failure_reasons)
    return ready, runtime_manifest, policy_contract, weights, failure_reason


def readiness_response() -> Dict[str, Any]:
    ready, runtime_fingerprint, policy, weights, failure_reason = runtime_contract()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "ready": ready,
        "runtime_fingerprint": runtime_fingerprint,
        "policy": policy,
        "weights": weights,
        "failure_reason": failure_reason,
    }


# ----------------------------------------------------------------------------------------------------
# PNLCALIB ENGINE (loaded once per process)
# ----------------------------------------------------------------------------------------------------

class PnLCalibEngine:

    def __init__(self, device: str = "cuda:0"):
        import torch
        import torchvision.transforms as T
        import yaml

        repo = str(PNLCALIB_REPO)
        if repo not in sys.path:
            sys.path.insert(0, repo)

        from model.cls_hrnet import get_cls_net
        from model.cls_hrnet_l import get_cls_net as get_cls_net_l

        self.torch = torch
        self.device = device

        cfg = yaml.safe_load(CONFIG_KP.read_text(encoding="utf-8"))
        cfg_l = yaml.safe_load(CONFIG_LINES.read_text(encoding="utf-8"))

        loaded_state = torch.load(str(WEIGHT_KP), map_location=device)
        self.model = get_cls_net(cfg)
        self.model.load_state_dict(loaded_state)
        self.model.to(device)
        self.model.eval()

        loaded_state_l = torch.load(str(WEIGHT_LINES), map_location=device)
        self.model_l = get_cls_net_l(cfg_l)
        self.model_l.load_state_dict(loaded_state_l)
        self.model_l.to(device)
        self.model_l.eval()

        self.transform2 = T.Resize((MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH))

    def run_frame(self, frame_bgr) -> Dict[str, Any]:
        """
        Reproduces PnLCalib inference.py::inference() through cam.update(), then runs
        heuristic_voting twice (BASE / PNL_REFINED) on independent calibrator instances.
        """
        import cv2
        import torchvision.transforms.functional as f
        from PIL import Image

        from utils.utils_calib import FramebyFrameCalib
        from utils.utils_heatmap import (
            complete_keypoints,
            coords_to_dict,
            get_keypoints_from_heatmap_batch_maxpool,
            get_keypoints_from_heatmap_batch_maxpool_l,
        )

        torch = self.torch

        frame_height, frame_width = frame_bgr.shape[:2]

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = f.to_tensor(pil).float().unsqueeze(0)
        if tensor.size()[-1] != MODEL_INPUT_WIDTH:
            tensor = self.transform2(tensor)
        tensor = tensor.to(self.device)
        _, _, h, w = tensor.size()

        with torch.no_grad():
            heatmaps = self.model(tensor)
            heatmaps_l = self.model_l(tensor)

        kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:, :-1, :, :])
        line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:, :-1, :, :])
        kp_dict = coords_to_dict(kp_coords, threshold=KP_THRESHOLD)
        lines_dict = coords_to_dict(line_coords, threshold=LINE_THRESHOLD)

        raw_keypoints = copy.deepcopy(kp_dict[0])
        raw_lines = copy.deepcopy(lines_dict[0])

        completed_kp, completed_lines = complete_keypoints(
            copy.deepcopy(kp_dict[0]),
            copy.deepcopy(lines_dict[0]),
            w=w,
            h=h,
            normalize=True,
        )

        completed_kp_normalized = copy.deepcopy(completed_kp)
        completed_lines_normalized = copy.deepcopy(completed_lines)

        def run_variant(refine_lines: bool):
            cam = FramebyFrameCalib(iwidth=frame_width, iheight=frame_height, denormalize=True)
            cam.update(copy.deepcopy(completed_kp), copy.deepcopy(completed_lines))
            return cam.heuristic_voting(refine_lines=refine_lines)

        base_result = run_variant(refine_lines=False)
        pnl_result = run_variant(refine_lines=True)

        return {
            "frame_width": int(frame_width),
            "frame_height": int(frame_height),
            "model_input_width": int(w),
            "model_input_height": int(h),
            "raw_keypoints": raw_keypoints,
            "raw_lines": raw_lines,
            "completed_keypoints_normalized": completed_kp_normalized,
            "completed_lines_normalized": completed_lines_normalized,
            "base_result": base_result,
            "pnl_result": pnl_result,
        }


def _jsonable(value):
    import numpy as np

    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def legacy_row_from_engine_output(
    *,
    event_id: str,
    event_rank: int,
    frame_number: int,
    image_path: str,
    engine_output: Dict[str, Any],
    policy_file_sha: Optional[str],
) -> Dict[str, Any]:
    """Apply the frozen policy and emit a row in the historical Notebook-37 prediction schema."""

    base_result = engine_output["base_result"]
    pnl_result = engine_output["pnl_result"]

    base_H_i2p = base_H_p2i = pnl_H_i2p = pnl_H_p2i = None
    if base_result is not None:
        base_H_i2p, base_H_p2i = homographies_from_cam_params(base_result["cam_params"])
    if pnl_result is not None:
        pnl_H_i2p, pnl_H_p2i = homographies_from_cam_params(pnl_result["cam_params"])

    base_validation = validate_homography_matrix(base_H_i2p)
    pnl_validation = validate_homography_matrix(pnl_H_i2p)

    # Frozen selection rule (policy JSON "selection_rule"):
    if pnl_validation["mathematically_valid"]:
        selected_variant = "PNL_REFINED"
        selection_status = "PRIMARY_PNL_REFINED"
        selected_H_i2p, selected_H_p2i = pnl_H_i2p, pnl_H_p2i
    elif base_validation["mathematically_valid"]:
        selected_variant = "BASE"
        selection_status = "FALLBACK_BASE"
        selected_H_i2p, selected_H_p2i = base_H_i2p, base_H_p2i
    else:
        selected_variant = None
        selection_status = "NO_VALID_SOLUTION"
        selected_H_i2p = selected_H_p2i = None

    row = {
        "event_id": event_id,
        "event_rank": int(event_rank),
        "frame_number": int(frame_number),
        "image_path": str(image_path),
        "frame_width": engine_output["frame_width"],
        "frame_height": engine_output["frame_height"],
        "raw_keypoints": engine_output["raw_keypoints"],
        "raw_lines": engine_output["raw_lines"],
        "raw_keypoint_count": int(len(engine_output["raw_keypoints"])),
        "raw_line_count": int(len(engine_output["raw_lines"])),
        "completed_keypoints_normalized": engine_output["completed_keypoints_normalized"],
        "completed_lines_normalized": engine_output["completed_lines_normalized"],
        "base_result": base_result,
        "pnl_result": pnl_result,
        "base_error": None,
        "pnl_error": None,
        "selected_variant": selected_variant,
        "selection_status": selection_status,
        "base_validation": base_validation,
        "pnl_validation": pnl_validation,
        "base_H_image_to_pitch": base_H_i2p,
        "base_H_pitch_to_image": base_H_p2i,
        "pnl_H_image_to_pitch": pnl_H_i2p,
        "pnl_H_pitch_to_image": pnl_H_p2i,
        "selected_H_image_to_pitch": selected_H_i2p,
        "selected_H_pitch_to_image": selected_H_p2i,
        "policy_name": POLICY_NAME,
        "policy_sha256": POLICY_FINGERPRINT,
        "policy_file_sha256": policy_file_sha,
        "reference_used_for_inference": False,
        "reference_used_for_selection": False,
    }
    return _jsonable(row)


def run_legacy_manifest(manifest_path: Path, result_path: Path, engine: Optional[PnLCalibEngine] = None):
    """Historical runner contract: list of manifest rows in -> list of legacy result rows out."""
    import cv2

    manifest_rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_rows, list) or not manifest_rows:
        raise ValueError("legacy manifest must be a non-empty list")

    for row in manifest_rows:
        if row.get("reference_used_for_inference") is not False:
            raise ValueError("legacy manifest row must declare reference_used_for_inference = false")
        if row.get("reference_used_for_selection") is not False:
            raise ValueError("legacy manifest row must declare reference_used_for_selection = false")

    if engine is None:
        engine = PnLCalibEngine()

    policy_file_sha = sha256_file(POLICY_PATH) if POLICY_PATH.exists() else None

    results = []
    for row in manifest_rows:
        image_path = Path(row["image_path"])
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"OpenCV could not decode image: {image_path}")
        engine_output = engine.run_frame(frame)
        results.append(
            legacy_row_from_engine_output(
                event_id=str(row["event_id"]),
                event_rank=int(row.get("event_rank", 1)),
                frame_number=int(row["frame_number"]),
                image_path=str(image_path),
                engine_output=engine_output,
                policy_file_sha=policy_file_sha,
            )
        )

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return results


# ----------------------------------------------------------------------------------------------------
# GEOMETRY-v2 WIRE PROTOCOL (identical to the frozen worker's request/response contract)
# ----------------------------------------------------------------------------------------------------

REQUIRED_INFERENCE_MANIFEST_KEYS = {
    "protocol_version",
    "request_id",
    "video_sha256",
    "frame_idx",
    "frame_width",
    "frame_height",
    "frame_pixel_format",
    "frame_indexing",
    "frame_png_sha256",
    "frame_pixels_sha256",
    "geometry_frame_policy",
    "require_reference_free",
    "policy_fingerprint",
}

FORBIDDEN_MANIFEST_FIELDS = {
    "shooter_track_id",
    "shooter_identity",
    "expected_shot_distance",
    "expected_shot_angle",
    "xg",
    "nearest_defender",
    "player_inside_fraction",
    "reference_homography",
    "reference_pitch_coordinates",
    "ground_truth",
}


def load_and_validate_request(manifest_path: Path, frame_path: Path):
    import cv2
    import numpy as np

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("request manifest must be a JSON object")

    if set(manifest.keys()) != REQUIRED_INFERENCE_MANIFEST_KEYS:
        missing = sorted(REQUIRED_INFERENCE_MANIFEST_KEYS - set(manifest.keys()))
        extra = sorted(set(manifest.keys()) - REQUIRED_INFERENCE_MANIFEST_KEYS)
        raise ValueError(f"manifest key mismatch; missing={missing} extra={extra}")

    forbidden_hits = sorted(FORBIDDEN_MANIFEST_FIELDS.intersection(manifest.keys()))
    if forbidden_hits:
        raise ValueError(f"forbidden semantic/reference fields present: {forbidden_hits}")

    if manifest["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("protocol_version mismatch")
    if manifest["geometry_frame_policy"] != GEOMETRY_FRAME_POLICY:
        raise ValueError("geometry_frame_policy mismatch")
    if manifest["policy_fingerprint"] != POLICY_FINGERPRINT:
        raise ValueError("policy_fingerprint mismatch")
    if manifest["require_reference_free"] is not True:
        raise ValueError("require_reference_free must be true")
    if manifest["frame_indexing"] != "ZERO_BASED":
        raise ValueError("frame_indexing must be ZERO_BASED")
    if manifest["frame_pixel_format"] != "BGR8":
        raise ValueError("frame_pixel_format must be BGR8")
    if not isinstance(manifest["frame_idx"], int) or manifest["frame_idx"] < 0:
        raise ValueError("frame_idx must be a non-negative integer")

    encoded_bytes = frame_path.read_bytes()
    if sha256_bytes(encoded_bytes) != manifest["frame_png_sha256"]:
        raise ValueError("frame_png_sha256 mismatch")

    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("OpenCV could not decode frame PNG")

    height, width = frame.shape[:2]
    if width != manifest["frame_width"]:
        raise ValueError("frame_width mismatch")
    if height != manifest["frame_height"]:
        raise ValueError("frame_height mismatch")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("decoded frame is not BGR8")

    pixel_sha = sha256_bytes(np.ascontiguousarray(frame).tobytes())
    if pixel_sha != manifest["frame_pixels_sha256"]:
        raise ValueError("frame_pixels_sha256 mismatch")

    request_material = (
        str(manifest["protocol_version"])
        + ":"
        + str(manifest["video_sha256"])
        + ":"
        + str(manifest["frame_idx"])
        + ":"
        + str(manifest["frame_pixels_sha256"])
    ).encode("utf-8")
    if manifest["request_id"] != hashlib.sha256(request_material).hexdigest():
        raise ValueError("request_id mismatch")

    return manifest, frame


def input_echo(manifest):
    return {
        "video_sha256": manifest["video_sha256"],
        "frame_idx": manifest["frame_idx"],
        "frame_width": manifest["frame_width"],
        "frame_height": manifest["frame_height"],
        "frame_pixel_format": manifest["frame_pixel_format"],
        "frame_png_sha256": manifest["frame_png_sha256"],
        "frame_pixels_sha256": manifest["frame_pixels_sha256"],
    }


def failure_response(
    *,
    manifest,
    runtime_fingerprint,
    selection_status,
    failure_reason,
    keypoint_count=None,
    line_count=None,
    reference_free=None,
):
    is_dict = isinstance(manifest, dict)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": manifest.get("request_id") if is_dict else None,
        "success": False,
        "frame_idx": manifest.get("frame_idx") if is_dict else None,
        "selected_variant": None,
        "selection_status": selection_status,
        "homography_image_to_pitch": None,
        "homography_pitch_to_image": None,
        "keypoint_count": keypoint_count,
        "line_count": line_count,
        "reference_free": reference_free,
        "input_echo": (
            input_echo(manifest)
            if is_dict and REQUIRED_INFERENCE_MANIFEST_KEYS.issubset(set(manifest.keys()))
            else None
        ),
        "runtime_fingerprint": runtime_fingerprint,
        "failure_reason": failure_reason,
    }


def infer(manifest_path: Path, frame_path: Path, diagnostics_path: Optional[Path] = None):
    ready_payload = readiness_response()
    runtime_fingerprint = ready_payload["runtime_fingerprint"]

    if ready_payload["ready"] is not True:
        return failure_response(
            manifest={},
            runtime_fingerprint=runtime_fingerprint,
            selection_status="RUNTIME_FAILURE",
            failure_reason=ready_payload["failure_reason"],
        )

    try:
        manifest, frame = load_and_validate_request(manifest_path, frame_path)
    except Exception as exc:
        raw_manifest = {}
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                raw_manifest = loaded
        except Exception:
            pass
        return failure_response(
            manifest=raw_manifest,
            runtime_fingerprint=runtime_fingerprint,
            selection_status="INPUT_CONTRACT_FAILURE",
            failure_reason=f"{type(exc).__name__}: {exc}",
        )

    # Historical runner contract: exactly one legacy manifest row, reference-free.
    # Short names: the request directory already carries the request identity, and Windows
    # MAX_PATH (260) is easily exceeded with a 64-hex directory plus 64-hex filenames.
    output_parent = manifest_path.parent
    legacy_manifest_path = output_parent / "legacy_manifest.json"
    legacy_result_path = output_parent / "legacy_result.json"

    legacy_manifest = [
        {
            "event_id": "geometry_v2_" + manifest["request_id"][:16],
            "event_rank": 1,
            "frame_number": int(manifest["frame_idx"]),
            "image_path": str(frame_path.resolve()),
            "reference_used_for_inference": False,
            "reference_used_for_selection": False,
        }
    ]
    legacy_manifest_path.write_text(
        json.dumps(legacy_manifest, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    try:
        rows = run_legacy_manifest(legacy_manifest_path, legacy_result_path)
    except Exception as exc:
        return failure_response(
            manifest=manifest,
            runtime_fingerprint=runtime_fingerprint,
            selection_status="RUNTIME_FAILURE",
            failure_reason=f"local PnLCalib runner failed: {type(exc).__name__}: {exc}",
        )

    if len(rows) != 1:
        return failure_response(
            manifest=manifest,
            runtime_fingerprint=runtime_fingerprint,
            selection_status="RUNTIME_FAILURE",
            failure_reason="local runner result must contain exactly one row",
        )

    row = rows[0]
    raw_kp_count = row.get("raw_keypoint_count")
    raw_line_count = row.get("raw_line_count")

    reference_free = bool(
        row.get("reference_used_for_inference") is False
        and row.get("reference_used_for_selection") is False
    )
    if not reference_free:
        return failure_response(
            manifest=manifest,
            runtime_fingerprint=runtime_fingerprint,
            selection_status="RUNTIME_FAILURE",
            failure_reason="local runner violated reference-free contract",
            keypoint_count=raw_kp_count,
            line_count=raw_line_count,
            reference_free=False,
        )

    if row.get("frame_number") != manifest["frame_idx"]:
        return failure_response(
            manifest=manifest,
            runtime_fingerprint=runtime_fingerprint,
            selection_status="RUNTIME_FAILURE",
            failure_reason="local runner frame_number echo mismatch",
            keypoint_count=raw_kp_count,
            line_count=raw_line_count,
            reference_free=True,
        )

    selected_variant = row.get("selected_variant")
    legacy_selection_status = row.get("selection_status")

    if selected_variant == "PNL_REFINED":
        expected_selection_status = "PRIMARY_PNL_REFINED"
    elif selected_variant == "BASE":
        expected_selection_status = "FALLBACK_BASE"
    else:
        expected_selection_status = "NO_VALID_GEOMETRY"

    if selected_variant in {"PNL_REFINED", "BASE"} and legacy_selection_status != expected_selection_status:
        return failure_response(
            manifest=manifest,
            runtime_fingerprint=runtime_fingerprint,
            selection_status="RUNTIME_FAILURE",
            failure_reason=(
                "policy selection status mismatch: "
                f"{legacy_selection_status!r} != {expected_selection_status!r}"
            ),
            keypoint_count=raw_kp_count,
            line_count=raw_line_count,
            reference_free=True,
        )

    if selected_variant not in {"PNL_REFINED", "BASE"}:
        return failure_response(
            manifest=manifest,
            runtime_fingerprint=runtime_fingerprint,
            selection_status="NO_VALID_GEOMETRY",
            failure_reason="local PnLCalib runner produced no valid geometry",
            keypoint_count=raw_kp_count,
            line_count=raw_line_count,
            reference_free=True,
        )

    h_image_to_pitch = row.get("selected_H_image_to_pitch")
    h_pitch_to_image = row.get("selected_H_pitch_to_image")
    v1 = validate_homography_matrix(h_image_to_pitch)
    v2 = validate_homography_matrix(h_pitch_to_image)
    if not (v1["mathematically_valid"] and v2["mathematically_valid"]):
        return failure_response(
            manifest=manifest,
            runtime_fingerprint=runtime_fingerprint,
            selection_status="RUNTIME_FAILURE",
            failure_reason=f"selected homography validation failed; image_to_pitch={v1}; pitch_to_image={v2}",
            keypoint_count=raw_kp_count,
            line_count=raw_line_count,
            reference_free=True,
        )

    if diagnostics_path is not None:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(
            json.dumps(row, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": manifest["request_id"],
        "success": True,
        "frame_idx": manifest["frame_idx"],
        "selected_variant": selected_variant,
        "selection_status": expected_selection_status,
        "homography_image_to_pitch": h_image_to_pitch,
        "homography_pitch_to_image": h_pitch_to_image,
        "keypoint_count": int(raw_kp_count),
        "line_count": int(raw_line_count),
        "reference_free": True,
        "input_echo": input_echo(manifest),
        "runtime_fingerprint": runtime_fingerprint,
        "failure_reason": None,
    }


def emit(payload, output_path: Optional[Path] = None):
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text)


def main():
    parser = argparse.ArgumentParser(
        description="Soccer ML Geometry-v2 LOCAL RTX-2060 PnLCalib worker"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--runtime-json", action="store_true")
    mode.add_argument("--infer", action="store_true")
    mode.add_argument("--legacy", action="store_true",
                      help="historical runner contract: --manifest <legacy rows> --output <legacy result>")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--frame", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args()

    if args.runtime_json:
        emit(readiness_response())
        return

    if args.legacy:
        if args.manifest is None or args.output is None:
            parser.error("--manifest and --output are required with --legacy")
        ready = readiness_response()
        if ready["ready"] is not True:
            raise SystemExit(f"local runtime not ready: {ready['failure_reason']}")
        run_legacy_manifest(args.manifest.resolve(), args.output.resolve())
        print(json.dumps({"status": "OK", "output": str(args.output.resolve())}))
        return

    if args.manifest is None or args.frame is None or args.output is None:
        parser.error("--manifest, --frame and --output are required with --infer")

    payload = infer(
        args.manifest.resolve(),
        args.frame.resolve(),
        args.diagnostics.resolve() if args.diagnostics is not None else None,
    )
    emit(payload, args.output.resolve())


if __name__ == "__main__":
    main()
