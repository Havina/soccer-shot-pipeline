"""
LOCAL RTX-2060 Geometry-v2 adapter for the frozen Semantic-v2 pipeline.

This module is NEW and LOCAL-ONLY.  It lives outside the frozen ``soccer_ml.v2``
package and does not modify it.  It plugs into the public Semantic-v2 API through
the existing dependency-injection boundary::

    soccer_ml.v2.api.analyze_soccer_clip(
        video_path,
        geometry_inference_callable=LocalGeometryV2Adapter(...),
        frame_loader=default_frame_loader,
        png_encoder=default_png_encoder,
        video_sha256_provider=default_video_sha256_provider,
        ...
    )

Stage 7 of the canonical pipeline calls the adapter as::

    response = adapter(video_sha256=..., frame_idx=..., frame_bgr=..., frame_png=...)

and receives a ``soccer_ml.v2.contracts.GeometryV2InferenceResponse``.

How it works
------------
The frozen HTTP client (``soccer_ml.v2.geometry_client``) is hard-pinned to the
certified NVIDIA-L40 runtime fingerprint and is therefore unusable on the local
RTX 2060.  This adapter instead:

1. builds the exact frozen request manifest via the frozen, hardware-agnostic
   ``geometry_client.build_geometry_request_manifest`` (unchanged, re-used);
2. writes manifest + PNG into a per-request directory;
3. executes ``local_geometry_v2/local_geometry_v2_worker.py`` inside
   ``.venv-geometry-v2`` (Python 3.10.10 / Torch 2.0.0+cu117) as a fresh
   subprocess -- the same worker -> runner process boundary the certified L40
   service used;
4. validates the response against the frozen wire protocol with the same
   checks the frozen client applies, except that the runtime fingerprint is
   required to be the LOCAL RTX-2060 fingerprint (optionally pinned to the
   SHA recorded by the local certification freeze) rather than the L40 one;
5. returns ``GeometryV2InferenceResponse``.

Nothing here claims L40 certification.  The local certification artefact lives
under ``outputs/notebook50_local/local_geometry_v2/``.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from .v2.contracts import GeometryV2InferenceResponse, GeometryV2Readiness
from .v2.geometry_client import (
    FAILURE_SELECTION_STATUSES,
    INFERENCE_RESPONSE_KEYS,
    POLICY_FINGERPRINT,
    PROTOCOL_VERSION,
    READINESS_KEYS,
    REQUIRED_WEIGHT_HASHES,
    SUCCESS_SELECTIONS,
    GeometryV2ContractError,
    GeometryV2RuntimeNotReadyError,
    build_geometry_request_manifest,
    validate_homography_pair,
)


# ----------------------------------------------------------------------------------------------------
# IDENTITIES / PATHS
# ----------------------------------------------------------------------------------------------------

LOCAL_RUNTIME_PROFILE = "PNLCALIB_NOTEBOOK37_LOCAL_RTX2060_RUNTIME_V1"

LOCAL_GPU_NAME_SUBSTRING = "RTX 2060"

HISTORICAL_L40_RUNTIME_FINGERPRINT_SHA256 = (
    "715f9b64b964b9a2b8d8011d231d132819ccff540563aa34e9284ae94c976d5e"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GEOMETRY_VENV_PYTHON = (PROJECT_ROOT / ".venv-geometry-v2" / "Scripts" / "python.exe").resolve()

WORKER_PATH = (PROJECT_ROOT / "local_geometry_v2" / "local_geometry_v2_worker.py").resolve()

LOCAL_GEOMETRY_OUTPUT_ROOT = (PROJECT_ROOT / "outputs" / "notebook50_local" / "local_geometry_v2").resolve()

DEFAULT_REQUEST_ROOT = (LOCAL_GEOMETRY_OUTPUT_ROOT / "requests").resolve()

LOCAL_CERTIFICATION_PATH = (LOCAL_GEOMETRY_OUTPUT_ROOT / "local_geometry_v2_certification.json").resolve()

DEFAULT_TIMEOUT_SECONDS = 600.0


class LocalGeometryV2Error(RuntimeError):
    """Raised when the local geometry worker cannot be executed."""


# ----------------------------------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------------------------------

def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _semantic_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _expected_input_echo(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "video_sha256": manifest["video_sha256"],
        "frame_idx": manifest["frame_idx"],
        "frame_width": manifest["frame_width"],
        "frame_height": manifest["frame_height"],
        "frame_pixel_format": manifest["frame_pixel_format"],
        "frame_png_sha256": manifest["frame_png_sha256"],
        "frame_pixels_sha256": manifest["frame_pixels_sha256"],
    }


def read_local_certification(path: Optional[Path] = None) -> Mapping[str, Any]:
    certification_path = Path(path) if path is not None else LOCAL_CERTIFICATION_PATH
    if not certification_path.exists():
        raise LocalGeometryV2Error(
            f"Local Geometry-v2 certification freeze not found: {certification_path}"
        )
    payload = json.loads(certification_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise LocalGeometryV2Error("Local Geometry-v2 certification must be a JSON object.")
    return payload


def certified_runtime_fingerprint_sha256(path: Optional[Path] = None) -> str:
    payload = read_local_certification(path)
    if payload.get("LOCAL_GEOMETRY_V2_CERTIFIED") is not True:
        raise LocalGeometryV2Error("Local Geometry-v2 certification freeze is not certified.")
    value = payload.get("runtime_fingerprint_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise LocalGeometryV2Error("Local Geometry-v2 certification has no runtime fingerprint SHA.")
    return value


# ----------------------------------------------------------------------------------------------------
# LOCAL FINGERPRINT VALIDATION (mirrors the frozen client, but for the LOCAL profile)
# ----------------------------------------------------------------------------------------------------

def validate_local_runtime_fingerprint(
    runtime_fingerprint: Any,
    *,
    expected_fingerprint_sha256: Optional[str] = None,
) -> Mapping[str, Any]:

    if not isinstance(runtime_fingerprint, Mapping):
        raise GeometryV2ContractError("runtime_fingerprint must be an object.")

    reported_sha = runtime_fingerprint.get("fingerprint_sha256")
    if not isinstance(reported_sha, str) or len(reported_sha) != 64:
        raise GeometryV2ContractError("runtime_fingerprint.fingerprint_sha256 is missing.")

    if reported_sha == HISTORICAL_L40_RUNTIME_FINGERPRINT_SHA256:
        raise GeometryV2ContractError(
            "Local adapter received the historical L40 runtime fingerprint; "
            "this adapter only accepts the local RTX-2060 runtime."
        )

    if expected_fingerprint_sha256 is not None and reported_sha != expected_fingerprint_sha256:
        raise GeometryV2ContractError(
            "Local Geometry-v2 runtime fingerprint does not match the locally certified fingerprint."
        )

    material = dict(runtime_fingerprint)
    material.pop("fingerprint_sha256", None)
    if _semantic_sha256(material) != reported_sha:
        raise GeometryV2ContractError("Local Geometry-v2 runtime fingerprint self-check failed.")

    if runtime_fingerprint.get("runtime_profile") != LOCAL_RUNTIME_PROFILE:
        raise GeometryV2ContractError("Local Geometry-v2 runtime profile mismatch.")

    lineage = runtime_fingerprint.get("certification_lineage")
    if not isinstance(lineage, Mapping) or lineage.get("claims_l40_certification") is not False:
        raise GeometryV2ContractError("Local runtime must explicitly disclaim L40 certification.")

    repository = runtime_fingerprint.get("repository")
    if not isinstance(repository, Mapping):
        raise GeometryV2ContractError("runtime_fingerprint.repository is missing.")
    if repository.get("tracked_source_clean") is not True:
        raise GeometryV2ContractError("Local PnLCalib tracked source is not clean.")

    runtime = runtime_fingerprint.get("runtime")
    if not isinstance(runtime, Mapping):
        raise GeometryV2ContractError("runtime_fingerprint.runtime is missing.")
    if runtime.get("cuda_available") is not True:
        raise GeometryV2ContractError("Local Geometry-v2 CUDA runtime is not available.")
    if LOCAL_GPU_NAME_SUBSTRING not in str(runtime.get("gpu_name")):
        raise GeometryV2ContractError("Local Geometry-v2 runtime is not the local RTX-2060 runtime.")

    if runtime_fingerprint.get("policy_fingerprint") != POLICY_FINGERPRINT:
        raise GeometryV2ContractError("Runtime policy fingerprint mismatch.")

    weight_hashes = runtime_fingerprint.get("weight_hashes")
    if not isinstance(weight_hashes, Mapping):
        raise GeometryV2ContractError("runtime_fingerprint.weight_hashes is missing.")
    for name, expected_sha in REQUIRED_WEIGHT_HASHES.items():
        if weight_hashes.get(name) != expected_sha:
            raise GeometryV2ContractError(f"Runtime weight hash mismatch for {name}.")

    return runtime_fingerprint


def parse_local_readiness_payload(
    payload: Any,
    *,
    expected_fingerprint_sha256: Optional[str] = None,
) -> GeometryV2Readiness:

    if not isinstance(payload, Mapping):
        raise GeometryV2ContractError("Geometry-v2 readiness payload must be an object.")
    if set(payload.keys()) != READINESS_KEYS:
        raise GeometryV2ContractError("Geometry-v2 readiness response keys do not match the frozen contract.")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise GeometryV2ContractError("Geometry-v2 readiness protocol mismatch.")
    if not isinstance(payload.get("ready"), bool):
        raise GeometryV2ContractError("Geometry-v2 readiness 'ready' field must be boolean.")

    runtime_fingerprint = validate_local_runtime_fingerprint(
        payload.get("runtime_fingerprint"),
        expected_fingerprint_sha256=expected_fingerprint_sha256,
    )

    policy = payload.get("policy")
    if not isinstance(policy, Mapping) or policy.get("policy_fingerprint") != POLICY_FINGERPRINT:
        raise GeometryV2ContractError("Geometry-v2 readiness policy fingerprint mismatch.")

    weights = payload.get("weights")
    if not isinstance(weights, Mapping):
        raise GeometryV2ContractError("Geometry-v2 readiness weights are missing.")
    for name, expected_sha in REQUIRED_WEIGHT_HASHES.items():
        row = weights.get(name)
        if not isinstance(row, Mapping) or row.get("sha256") != expected_sha:
            raise GeometryV2ContractError(f"Geometry-v2 readiness weight hash mismatch for {name}.")

    failure_reason = payload.get("failure_reason")
    if payload["ready"] is True and failure_reason is not None:
        raise GeometryV2ContractError("Ready Geometry-v2 runtime cannot contain a failure reason.")
    if payload["ready"] is not True and (not isinstance(failure_reason, str) or not failure_reason):
        raise GeometryV2ContractError("Non-ready Geometry-v2 runtime must provide a failure reason.")

    return GeometryV2Readiness(
        protocol_version=PROTOCOL_VERSION,
        ready=bool(payload["ready"]),
        runtime_fingerprint=runtime_fingerprint,
        policy=policy,
        weights=weights,
        failure_reason=failure_reason,
    )


def parse_local_inference_payload(
    payload: Any,
    *,
    expected_manifest: Optional[Mapping[str, Any]] = None,
    expected_fingerprint_sha256: Optional[str] = None,
) -> GeometryV2InferenceResponse:

    if not isinstance(payload, Mapping):
        raise GeometryV2ContractError("Geometry-v2 inference payload must be an object.")
    if set(payload.keys()) != INFERENCE_RESPONSE_KEYS:
        raise GeometryV2ContractError("Geometry-v2 inference response keys do not match the frozen contract.")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise GeometryV2ContractError("Geometry-v2 inference protocol mismatch.")

    runtime_fingerprint = validate_local_runtime_fingerprint(
        payload.get("runtime_fingerprint"),
        expected_fingerprint_sha256=expected_fingerprint_sha256,
    )

    success = payload.get("success")
    if not isinstance(success, bool):
        raise GeometryV2ContractError("Geometry-v2 inference success field must be boolean.")

    selection_status = payload.get("selection_status")
    if not isinstance(selection_status, str):
        raise GeometryV2ContractError("Geometry-v2 inference selection_status must be a string.")

    if expected_manifest is not None:
        if payload.get("request_id") != expected_manifest.get("request_id"):
            raise GeometryV2ContractError("Geometry-v2 request_id echo mismatch.")
        if payload.get("frame_idx") != expected_manifest.get("frame_idx"):
            raise GeometryV2ContractError("Geometry-v2 frame_idx echo mismatch.")
        if success is True and payload.get("input_echo") != _expected_input_echo(expected_manifest):
            raise GeometryV2ContractError("Geometry-v2 input echo mismatch.")

    if success is True:
        selected_variant = payload.get("selected_variant")
        if selected_variant not in SUCCESS_SELECTIONS:
            raise GeometryV2ContractError(
                "Successful Geometry-v2 inference returned an unsupported selected variant."
            )
        if selection_status != SUCCESS_SELECTIONS[selected_variant]:
            raise GeometryV2ContractError("Geometry-v2 selected variant / selection status mismatch.")
        if payload.get("reference_free") is not True:
            raise GeometryV2ContractError("Successful Geometry-v2 inference must be reference-free.")
        if payload.get("failure_reason") is not None:
            raise GeometryV2ContractError("Successful Geometry-v2 inference cannot contain a failure reason.")

        for name in ("keypoint_count", "line_count"):
            value = payload.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise GeometryV2ContractError(f"Geometry-v2 {name} must be a non-negative integer.")

        validate_homography_pair(
            payload.get("homography_image_to_pitch"),
            payload.get("homography_pitch_to_image"),
        )
    else:
        if selection_status not in FAILURE_SELECTION_STATUSES:
            raise GeometryV2ContractError("Geometry-v2 failure returned an unsupported selection status.")
        if payload.get("selected_variant") is not None:
            raise GeometryV2ContractError("Failed Geometry-v2 inference cannot select a geometry variant.")
        if (
            payload.get("homography_image_to_pitch") is not None
            or payload.get("homography_pitch_to_image") is not None
        ):
            raise GeometryV2ContractError("Failed Geometry-v2 inference cannot contain selected homographies.")
        failure_reason = payload.get("failure_reason")
        if not isinstance(failure_reason, str) or not failure_reason:
            raise GeometryV2ContractError("Failed Geometry-v2 inference must provide a failure reason.")

    return GeometryV2InferenceResponse(
        protocol_version=PROTOCOL_VERSION,
        request_id=payload.get("request_id"),
        success=success,
        frame_idx=payload.get("frame_idx"),
        selected_variant=payload.get("selected_variant"),
        selection_status=selection_status,
        homography_image_to_pitch=payload.get("homography_image_to_pitch"),
        homography_pitch_to_image=payload.get("homography_pitch_to_image"),
        keypoint_count=payload.get("keypoint_count"),
        line_count=payload.get("line_count"),
        reference_free=payload.get("reference_free"),
        input_echo=payload.get("input_echo"),
        runtime_fingerprint=runtime_fingerprint,
        failure_reason=payload.get("failure_reason"),
    )


# ----------------------------------------------------------------------------------------------------
# ADAPTER
# ----------------------------------------------------------------------------------------------------

class LocalGeometryV2Adapter:
    """
    Stage-7 ``geometry_inference_callable`` backed by the local RTX-2060 PnLCalib worker.

    Parameters
    ----------
    expected_runtime_fingerprint_sha256:
        If given, every worker response must carry exactly this local runtime
        fingerprint SHA (normally the value frozen by the local certification).
    request_root:
        Directory under which per-request manifest/PNG/response files are written.
    python_executable / worker_path:
        Override only for tests; defaults are the project geometry venv and worker.
    """

    def __init__(
        self,
        *,
        expected_runtime_fingerprint_sha256: Optional[str] = None,
        request_root: Optional[Path] = None,
        python_executable: Optional[Path] = None,
        worker_path: Optional[Path] = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        check_readiness: bool = True,
    ):
        self.expected_runtime_fingerprint_sha256 = expected_runtime_fingerprint_sha256
        self.request_root = Path(request_root).resolve() if request_root is not None else DEFAULT_REQUEST_ROOT
        self.python_executable = Path(python_executable).resolve() if python_executable else GEOMETRY_VENV_PYTHON
        self.worker_path = Path(worker_path).resolve() if worker_path else WORKER_PATH
        self.timeout_seconds = float(timeout_seconds)
        self.check_readiness = bool(check_readiness)
        self.last_request_dir: Optional[Path] = None
        self.last_readiness: Optional[GeometryV2Readiness] = None
        self.call_count = 0

        if not self.python_executable.exists():
            raise LocalGeometryV2Error(f"Geometry venv python not found: {self.python_executable}")
        if not self.worker_path.exists():
            raise LocalGeometryV2Error(f"Local geometry worker not found: {self.worker_path}")

    # -- process plumbing ---------------------------------------------------------------------------

    def _run_worker(self, args: list[str], cwd: Optional[Path] = None) -> tuple[int, str, str]:
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)  # never leak the main-venv path into the geometry venv
        env["PYTHONIOENCODING"] = "utf-8"
        completed = subprocess.run(
            [str(self.python_executable), str(self.worker_path), *args],
            cwd=str(cwd) if cwd is not None else str(self.worker_path.parent),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout_seconds,
            check=False,
        )
        return int(completed.returncode), completed.stdout, completed.stderr

    @staticmethod
    def _last_json_line(stdout: str) -> Mapping[str, Any]:
        for line in reversed(stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                payload = json.loads(line)
                if isinstance(payload, Mapping):
                    return payload
        raise LocalGeometryV2Error("Local geometry worker produced no JSON payload.")

    # -- readiness ----------------------------------------------------------------------------------

    def get_runtime(self) -> GeometryV2Readiness:
        code, stdout, stderr = self._run_worker(["--runtime-json"])
        if code != 0:
            raise LocalGeometryV2Error(
                f"Local geometry worker readiness failed (rc={code}): {stderr[-4000:]}"
            )
        readiness = parse_local_readiness_payload(
            self._last_json_line(stdout),
            expected_fingerprint_sha256=self.expected_runtime_fingerprint_sha256,
        )
        self.last_readiness = readiness
        if not readiness.ready:
            raise GeometryV2RuntimeNotReadyError(readiness.failure_reason or "Local Geometry-v2 runtime is not ready.")
        return readiness

    # -- inference ----------------------------------------------------------------------------------

    def infer_frame(
        self,
        *,
        video_sha256: str,
        frame_idx: int,
        frame_bgr: np.ndarray,
        frame_png: bytes,
    ) -> GeometryV2InferenceResponse:

        if self.check_readiness and self.last_readiness is None:
            self.get_runtime()

        manifest = build_geometry_request_manifest(
            video_sha256=video_sha256,
            frame_idx=frame_idx,
            frame_bgr=frame_bgr,
            frame_png_bytes=frame_png,
        )

        # Request directory keyed by the request-id prefix (full id is inside manifest.json);
        # keeps absolute paths under the Windows MAX_PATH limit.
        request_dir = (self.request_root / manifest["request_id"][:24]).resolve()
        request_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = request_dir / "manifest.json"
        frame_path = request_dir / "frame.png"
        response_path = request_dir / "response.json"
        diagnostics_path = request_dir / "diagnostics.json"

        manifest_path.write_text(
            json.dumps(manifest, separators=(",", ":"), allow_nan=False),
            encoding="utf-8",
        )
        frame_path.write_bytes(frame_png)

        self.call_count += 1
        self.last_request_dir = request_dir

        code, stdout, stderr = self._run_worker(
            [
                "--infer",
                "--manifest", str(manifest_path),
                "--frame", str(frame_path),
                "--output", str(response_path),
                "--diagnostics", str(diagnostics_path),
            ]
        )
        (request_dir / "worker_stderr.log").write_text(stderr, encoding="utf-8")
        if code != 0:
            raise LocalGeometryV2Error(
                f"Local geometry worker inference failed (rc={code}): {stderr[-4000:]}"
            )
        if not response_path.exists():
            raise LocalGeometryV2Error("Local geometry worker returned without writing its response JSON.")

        payload = json.loads(response_path.read_text(encoding="utf-8"))
        return parse_local_inference_payload(
            payload,
            expected_manifest=manifest,
            expected_fingerprint_sha256=self.expected_runtime_fingerprint_sha256,
        )

    # Stage 7 calls the injected object with keyword arguments.
    def __call__(self, *, video_sha256, frame_idx, frame_bgr, frame_png) -> GeometryV2InferenceResponse:
        return self.infer_frame(
            video_sha256=video_sha256,
            frame_idx=frame_idx,
            frame_bgr=frame_bgr,
            frame_png=frame_png,
        )


# ----------------------------------------------------------------------------------------------------
# DEFAULT FRAME / PNG / SHA PROVIDERS (identical behaviour to the Cell-40 child helpers)
# ----------------------------------------------------------------------------------------------------

def default_frame_loader(video_path, frame_idx):
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for exact frame load: {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not decode exact frame {frame_idx} from {video_path}")
    return frame


def default_png_encoder(frame):
    import cv2

    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError("cv2.imencode('.png', frame) failed.")
    return bytes(encoded.tobytes())


def default_video_sha256_provider(video_path):
    return sha256_file(Path(video_path))


__all__ = [
    "LOCAL_RUNTIME_PROFILE",
    "LOCAL_CERTIFICATION_PATH",
    "LOCAL_GEOMETRY_OUTPUT_ROOT",
    "GEOMETRY_VENV_PYTHON",
    "WORKER_PATH",
    "LocalGeometryV2Adapter",
    "LocalGeometryV2Error",
    "certified_runtime_fingerprint_sha256",
    "default_frame_loader",
    "default_png_encoder",
    "default_video_sha256_provider",
    "parse_local_inference_payload",
    "parse_local_readiness_payload",
    "read_local_certification",
    "sha256_file",
    "validate_local_runtime_fingerprint",
]
