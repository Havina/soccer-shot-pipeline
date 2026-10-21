from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Mapping, Optional

import numpy as np

from .contracts import (
    GeometryV2InferenceResponse,
    GeometryV2Readiness,
)


PROTOCOL_VERSION = "soccer-ml-geometry-v2/1"

GEOMETRY_FRAME_POLICY = "GEOMETRY_FRAME_POLICY_V2_0"

POLICY_FINGERPRINT = "62e0431373c6a665cbe07561c178cf750dd78ea610b11678644a28d7a0d342fd"

REQUIRED_RUNTIME_FINGERPRINT_SHA256 = (
    "715f9b64b964b9a2b8d8011d231d132819ccff540563aa34e9284ae94c976d5e"
)

REQUIRED_WEIGHT_HASHES = {
    "SV_kp":
        "7ea78fa76aaf94976a8eca428d6e3c59697a93430cba1a4603e20284b61f5113",

    "SV_lines":
        "d72f4ed71734a2e3df9fa084f666e9b8adaef21bf69bac8952d6d3f970ff7455",
}

SERVICE_URL_ENV = "SOCCER_ML_GEOMETRY_V2_URL"

READINESS_ENDPOINT = "/v1/runtime"

INFERENCE_ENDPOINT = "/v1/geometry/infer"

DEFAULT_TIMEOUT_SECONDS = 60.0

MAX_ATTEMPTS = 1

INVERSE_TOLERANCE = 1e-8


REQUEST_MANIFEST_KEYS = (
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
)


READINESS_KEYS = {
    "protocol_version",
    "ready",
    "runtime_fingerprint",
    "policy",
    "weights",
    "failure_reason",
}


INFERENCE_RESPONSE_KEYS = {
    "protocol_version",
    "request_id",
    "success",
    "frame_idx",
    "selected_variant",
    "selection_status",
    "homography_image_to_pitch",
    "homography_pitch_to_image",
    "keypoint_count",
    "line_count",
    "reference_free",
    "input_echo",
    "runtime_fingerprint",
    "failure_reason",
}


SUCCESS_SELECTIONS = {
    "PNL_REFINED":
        "PRIMARY_PNL_REFINED",

    "BASE":
        "FALLBACK_BASE",
}


FAILURE_SELECTION_STATUSES = {
    "NO_VALID_GEOMETRY",
    "RUNTIME_FAILURE",
    "INPUT_CONTRACT_FAILURE",
}


class GeometryV2ClientError(RuntimeError):
    """Base error for the versioned Geometry-v2 client."""


class GeometryV2ContractError(GeometryV2ClientError):
    """Raised when a server/request payload violates the frozen protocol."""


class GeometryV2TransportError(GeometryV2ClientError):
    """Raised when the HTTP transport fails."""


class GeometryV2RuntimeNotReadyError(GeometryV2ClientError):
    """Raised when the frozen Geometry-v2 runtime is unavailable or mismatched."""


def _sha256_bytes(
    value: bytes,
) -> str:

    return hashlib.sha256(
        value
    ).hexdigest()


def _semantic_sha256(
    value: Mapping[str, Any],
) -> str:

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        allow_nan=False,
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        encoded
    ).hexdigest()


def _require_sha256_hex(
    value: Any,
    name: str,
):

    if not isinstance(
        value,
        str,
    ):

        raise GeometryV2ContractError(
            f"{name} must be a SHA-256 hex string."
        )


    if len(
        value
    ) != 64:

        raise GeometryV2ContractError(
            f"{name} must contain exactly 64 hex characters."
        )


    try:

        int(
            value,
            16,
        )

    except ValueError as exc:

        raise GeometryV2ContractError(
            f"{name} is not valid hexadecimal."
        ) from exc


def _validate_runtime_fingerprint(
    runtime_fingerprint: Any,
) -> Mapping[str, Any]:

    if not isinstance(
        runtime_fingerprint,
        Mapping,
    ):

        raise GeometryV2ContractError(
            "runtime_fingerprint must be an object."
        )


    reported_sha = runtime_fingerprint.get(
        "fingerprint_sha256"
    )


    if reported_sha != REQUIRED_RUNTIME_FINGERPRINT_SHA256:

        raise GeometryV2ContractError(
            "Geometry-v2 runtime fingerprint does not match the certified L40 runtime."
        )


    fingerprint_material = dict(
        runtime_fingerprint
    )


    fingerprint_material.pop(
        "fingerprint_sha256",
        None,
    )


    recomputed_sha = _semantic_sha256(
        fingerprint_material
    )


    if recomputed_sha != reported_sha:

        raise GeometryV2ContractError(
            "Geometry-v2 runtime fingerprint self-check failed."
        )


    repository = runtime_fingerprint.get(
        "repository"
    )


    if not isinstance(
        repository,
        Mapping,
    ):

        raise GeometryV2ContractError(
            "runtime_fingerprint.repository is missing."
        )


    if repository.get(
        "tracked_source_clean"
    ) is not True:

        raise GeometryV2ContractError(
            "Certified PnLCalib tracked source is not clean."
        )


    runtime = runtime_fingerprint.get(
        "runtime"
    )


    if not isinstance(
        runtime,
        Mapping,
    ):

        raise GeometryV2ContractError(
            "runtime_fingerprint.runtime is missing."
        )


    if runtime.get(
        "cuda_available"
    ) is not True:

        raise GeometryV2ContractError(
            "Certified Geometry-v2 CUDA runtime is not available."
        )


    if "L40" not in str(
        runtime.get(
            "gpu_name"
        )
    ):

        raise GeometryV2ContractError(
            "Geometry-v2 runtime is not the certified NVIDIA L40 runtime."
        )


    if runtime_fingerprint.get(
        "policy_fingerprint"
    ) != POLICY_FINGERPRINT:

        raise GeometryV2ContractError(
            "Runtime policy fingerprint mismatch."
        )


    weight_hashes = runtime_fingerprint.get(
        "weight_hashes"
    )


    if not isinstance(
        weight_hashes,
        Mapping,
    ):

        raise GeometryV2ContractError(
            "runtime_fingerprint.weight_hashes is missing."
        )


    for name, expected_sha in REQUIRED_WEIGHT_HASHES.items():

        if weight_hashes.get(
            name
        ) != expected_sha:

            raise GeometryV2ContractError(
                f"Runtime weight hash mismatch for {name}."
            )


    return runtime_fingerprint


def _normalize_homography(
    value: Any,
    name: str,
) -> np.ndarray:

    try:

        matrix = np.asarray(
            value,
            dtype=np.float64,
        )

    except Exception as exc:

        raise GeometryV2ContractError(
            f"{name} could not be converted to a numeric matrix."
        ) from exc


    if matrix.shape != (
        3,
        3,
    ):

        raise GeometryV2ContractError(
            f"{name} must be exactly 3x3."
        )


    if not np.isfinite(
        matrix
    ).all():

        raise GeometryV2ContractError(
            f"{name} contains non-finite values."
        )


    if int(
        np.linalg.matrix_rank(
            matrix
        )
    ) != 3:

        raise GeometryV2ContractError(
            f"{name} is not full rank."
        )


    try:

        np.linalg.inv(
            matrix
        )

    except np.linalg.LinAlgError as exc:

        raise GeometryV2ContractError(
            f"{name} is not invertible."
        ) from exc


    scale = float(
        matrix[
            2,
            2,
        ]
    )


    if abs(
        scale
    ) > 1e-12:

        matrix = (
            matrix
            /
            scale
        )


    return matrix


def validate_homography_pair(
    homography_image_to_pitch: Any,
    homography_pitch_to_image: Any,
) -> float:

    image_to_pitch = _normalize_homography(
        homography_image_to_pitch,
        "homography_image_to_pitch",
    )


    pitch_to_image = _normalize_homography(
        homography_pitch_to_image,
        "homography_pitch_to_image",
    )


    inverse = np.linalg.inv(
        image_to_pitch
    )


    inverse_scale = float(
        inverse[
            2,
            2,
        ]
    )


    if abs(
        inverse_scale
    ) > 1e-12:

        inverse = (
            inverse
            /
            inverse_scale
        )


    delta = float(
        np.max(
            np.abs(
                inverse
                -
                pitch_to_image
            )
        )
    )


    if delta > INVERSE_TOLERANCE:

        raise GeometryV2ContractError(
            "Geometry-v2 homography pair is not mutually inverse "
            f"within {INVERSE_TOLERANCE}; max_abs_delta={delta}."
        )


    return delta


def build_geometry_request_manifest(
    *,
    video_sha256: str,
    frame_idx: int,
    frame_bgr: np.ndarray,
    frame_png_bytes: bytes,
) -> dict[str, Any]:

    _require_sha256_hex(
        video_sha256,
        "video_sha256",
    )


    if not isinstance(
        frame_idx,
        int,
    ) or isinstance(
        frame_idx,
        bool,
    ) or frame_idx < 0:

        raise GeometryV2ContractError(
            "frame_idx must be a non-negative integer."
        )


    if not isinstance(
        frame_bgr,
        np.ndarray,
    ):

        raise GeometryV2ContractError(
            "frame_bgr must be a NumPy array."
        )


    if frame_bgr.dtype != np.uint8:

        raise GeometryV2ContractError(
            "frame_bgr must have dtype uint8."
        )


    if (
        frame_bgr.ndim != 3
        or
        frame_bgr.shape[
            2
        ]
        != 3
    ):

        raise GeometryV2ContractError(
            "frame_bgr must have shape (height, width, 3)."
        )


    if not isinstance(
        frame_png_bytes,
        bytes,
    ) or not frame_png_bytes:

        raise GeometryV2ContractError(
            "frame_png_bytes must be non-empty bytes."
        )


    frame_height = int(
        frame_bgr.shape[
            0
        ]
    )


    frame_width = int(
        frame_bgr.shape[
            1
        ]
    )


    pixel_bytes = np.ascontiguousarray(
        frame_bgr
    ).tobytes()


    frame_pixels_sha256 = _sha256_bytes(
        pixel_bytes
    )


    frame_png_sha256 = _sha256_bytes(
        frame_png_bytes
    )


    request_id_material = (
        PROTOCOL_VERSION
        + ":"
        + video_sha256
        + ":"
        + str(
            frame_idx
        )
        + ":"
        + frame_pixels_sha256
    ).encode(
        "utf-8"
    )


    request_id = _sha256_bytes(
        request_id_material
    )


    manifest = {
        "protocol_version":
            PROTOCOL_VERSION,

        "request_id":
            request_id,

        "video_sha256":
            video_sha256,

        "frame_idx":
            frame_idx,

        "frame_width":
            frame_width,

        "frame_height":
            frame_height,

        "frame_pixel_format":
            "BGR8",

        "frame_indexing":
            "ZERO_BASED",

        "frame_png_sha256":
            frame_png_sha256,

        "frame_pixels_sha256":
            frame_pixels_sha256,

        "geometry_frame_policy":
            GEOMETRY_FRAME_POLICY,

        "require_reference_free":
            True,

        "policy_fingerprint":
            POLICY_FINGERPRINT,
    }


    if tuple(
        manifest.keys()
    ) != REQUEST_MANIFEST_KEYS:

        raise GeometryV2ContractError(
            "Geometry-v2 request manifest key order changed."
        )


    return manifest


def parse_readiness_payload(
    payload: Any,
) -> GeometryV2Readiness:

    if not isinstance(
        payload,
        Mapping,
    ):

        raise GeometryV2ContractError(
            "Geometry-v2 readiness payload must be an object."
        )


    if set(
        payload.keys()
    ) != READINESS_KEYS:

        raise GeometryV2ContractError(
            "Geometry-v2 readiness response keys do not match the frozen contract."
        )


    if payload.get(
        "protocol_version"
    ) != PROTOCOL_VERSION:

        raise GeometryV2ContractError(
            "Geometry-v2 readiness protocol mismatch."
        )


    if not isinstance(
        payload.get(
            "ready"
        ),
        bool,
    ):

        raise GeometryV2ContractError(
            "Geometry-v2 readiness 'ready' field must be boolean."
        )


    runtime_fingerprint = _validate_runtime_fingerprint(
        payload.get(
            "runtime_fingerprint"
        )
    )


    policy = payload.get(
        "policy"
    )


    if not isinstance(
        policy,
        Mapping,
    ):

        raise GeometryV2ContractError(
            "Geometry-v2 readiness policy is missing."
        )


    if policy.get(
        "policy_fingerprint"
    ) != POLICY_FINGERPRINT:

        raise GeometryV2ContractError(
            "Geometry-v2 readiness policy fingerprint mismatch."
        )


    weights = payload.get(
        "weights"
    )


    if not isinstance(
        weights,
        Mapping,
    ):

        raise GeometryV2ContractError(
            "Geometry-v2 readiness weights are missing."
        )


    for name, expected_sha in REQUIRED_WEIGHT_HASHES.items():

        row = weights.get(
            name
        )


        if not isinstance(
            row,
            Mapping,
        ):

            raise GeometryV2ContractError(
                f"Geometry-v2 readiness weight entry missing for {name}."
            )


        if row.get(
            "sha256"
        ) != expected_sha:

            raise GeometryV2ContractError(
                f"Geometry-v2 readiness weight hash mismatch for {name}."
            )


    failure_reason = payload.get(
        "failure_reason"
    )


    if payload[
        "ready"
    ] is True:

        if failure_reason is not None:

            raise GeometryV2ContractError(
                "Ready Geometry-v2 runtime cannot contain a failure reason."
            )

    else:

        if not isinstance(
            failure_reason,
            str,
        ) or not failure_reason:

            raise GeometryV2ContractError(
                "Non-ready Geometry-v2 runtime must provide a failure reason."
            )


    return GeometryV2Readiness(
        protocol_version=PROTOCOL_VERSION,
        ready=bool(
            payload[
                "ready"
            ]
        ),
        runtime_fingerprint=runtime_fingerprint,
        policy=policy,
        weights=weights,
        failure_reason=failure_reason,
    )


def _expected_input_echo(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:

    return {
        "video_sha256":
            manifest[
                "video_sha256"
            ],

        "frame_idx":
            manifest[
                "frame_idx"
            ],

        "frame_width":
            manifest[
                "frame_width"
            ],

        "frame_height":
            manifest[
                "frame_height"
            ],

        "frame_pixel_format":
            manifest[
                "frame_pixel_format"
            ],

        "frame_png_sha256":
            manifest[
                "frame_png_sha256"
            ],

        "frame_pixels_sha256":
            manifest[
                "frame_pixels_sha256"
            ],
    }


def parse_inference_payload(
    payload: Any,
    *,
    expected_manifest: Optional[Mapping[str, Any]] = None,
) -> GeometryV2InferenceResponse:

    if not isinstance(
        payload,
        Mapping,
    ):

        raise GeometryV2ContractError(
            "Geometry-v2 inference payload must be an object."
        )


    if set(
        payload.keys()
    ) != INFERENCE_RESPONSE_KEYS:

        raise GeometryV2ContractError(
            "Geometry-v2 inference response keys do not match the frozen contract."
        )


    if payload.get(
        "protocol_version"
    ) != PROTOCOL_VERSION:

        raise GeometryV2ContractError(
            "Geometry-v2 inference protocol mismatch."
        )


    runtime_fingerprint = _validate_runtime_fingerprint(
        payload.get(
            "runtime_fingerprint"
        )
    )


    success = payload.get(
        "success"
    )


    if not isinstance(
        success,
        bool,
    ):

        raise GeometryV2ContractError(
            "Geometry-v2 inference success field must be boolean."
        )


    selection_status = payload.get(
        "selection_status"
    )


    if not isinstance(
        selection_status,
        str,
    ):

        raise GeometryV2ContractError(
            "Geometry-v2 inference selection_status must be a string."
        )


    if expected_manifest is not None:

        if payload.get(
            "request_id"
        ) != expected_manifest.get(
            "request_id"
        ):

            raise GeometryV2ContractError(
                "Geometry-v2 request_id echo mismatch."
            )


        if payload.get(
            "frame_idx"
        ) != expected_manifest.get(
            "frame_idx"
        ):

            raise GeometryV2ContractError(
                "Geometry-v2 frame_idx echo mismatch."
            )


        if (
            success is True
            and
            payload.get(
                "input_echo"
            )
            !=
            _expected_input_echo(
                expected_manifest
            )
        ):

            raise GeometryV2ContractError(
                "Geometry-v2 input echo mismatch."
            )


    if success is True:

        selected_variant = payload.get(
            "selected_variant"
        )


        if selected_variant not in SUCCESS_SELECTIONS:

            raise GeometryV2ContractError(
                "Successful Geometry-v2 inference returned an unsupported selected variant."
            )


        expected_status = SUCCESS_SELECTIONS[
            selected_variant
        ]


        if selection_status != expected_status:

            raise GeometryV2ContractError(
                "Geometry-v2 selected variant / selection status mismatch."
            )


        if payload.get(
            "reference_free"
        ) is not True:

            raise GeometryV2ContractError(
                "Successful Geometry-v2 inference must be reference-free."
            )


        if payload.get(
            "failure_reason"
        ) is not None:

            raise GeometryV2ContractError(
                "Successful Geometry-v2 inference cannot contain a failure reason."
            )


        keypoint_count = payload.get(
            "keypoint_count"
        )


        line_count = payload.get(
            "line_count"
        )


        if (
            not isinstance(
                keypoint_count,
                int,
            )
            or
            isinstance(
                keypoint_count,
                bool,
            )
            or
            keypoint_count < 0
        ):

            raise GeometryV2ContractError(
                "Geometry-v2 keypoint_count must be a non-negative integer."
            )


        if (
            not isinstance(
                line_count,
                int,
            )
            or
            isinstance(
                line_count,
                bool,
            )
            or
            line_count < 0
        ):

            raise GeometryV2ContractError(
                "Geometry-v2 line_count must be a non-negative integer."
            )


        validate_homography_pair(
            payload.get(
                "homography_image_to_pitch"
            ),
            payload.get(
                "homography_pitch_to_image"
            ),
        )


    else:

        if selection_status not in FAILURE_SELECTION_STATUSES:

            raise GeometryV2ContractError(
                "Geometry-v2 failure returned an unsupported selection status."
            )


        if payload.get(
            "selected_variant"
        ) is not None:

            raise GeometryV2ContractError(
                "Failed Geometry-v2 inference cannot select a geometry variant."
            )


        if (
            payload.get(
                "homography_image_to_pitch"
            )
            is not None
            or
            payload.get(
                "homography_pitch_to_image"
            )
            is not None
        ):

            raise GeometryV2ContractError(
                "Failed Geometry-v2 inference cannot contain selected homographies."
            )


        failure_reason = payload.get(
            "failure_reason"
        )


        if not isinstance(
            failure_reason,
            str,
        ) or not failure_reason:

            raise GeometryV2ContractError(
                "Failed Geometry-v2 inference must provide a failure reason."
            )


    return GeometryV2InferenceResponse(
        protocol_version=PROTOCOL_VERSION,
        request_id=payload.get(
            "request_id"
        ),
        success=success,
        frame_idx=payload.get(
            "frame_idx"
        ),
        selected_variant=payload.get(
            "selected_variant"
        ),
        selection_status=selection_status,
        homography_image_to_pitch=payload.get(
            "homography_image_to_pitch"
        ),
        homography_pitch_to_image=payload.get(
            "homography_pitch_to_image"
        ),
        keypoint_count=payload.get(
            "keypoint_count"
        ),
        line_count=payload.get(
            "line_count"
        ),
        reference_free=payload.get(
            "reference_free"
        ),
        input_echo=payload.get(
            "input_echo"
        ),
        runtime_fingerprint=runtime_fingerprint,
        failure_reason=payload.get(
            "failure_reason"
        ),
    )


def _validate_base_url(
    value: str,
) -> str:

    parsed = urllib.parse.urlparse(
        value
    )


    if parsed.scheme not in {
        "http",
        "https",
    }:

        raise GeometryV2ContractError(
            "Geometry-v2 URL must use http or https."
        )


    if not parsed.hostname:

        raise GeometryV2ContractError(
            "Geometry-v2 URL must include a hostname."
        )


    if parsed.query or parsed.fragment:

        raise GeometryV2ContractError(
            "Geometry-v2 base URL cannot include a query string or fragment."
        )


    if parsed.scheme == "http":

        hostname = parsed.hostname.lower()


        localhost_allowed = hostname in {
            "localhost",
            "127.0.0.1",
            "::1",
        }


        private_ip_allowed = False


        try:

            parsed_ip = ipaddress.ip_address(
                hostname
            )

            private_ip_allowed = bool(
                parsed_ip.is_private
                or
                parsed_ip.is_loopback
            )

        except ValueError:

            private_ip_allowed = False


        if not (
            localhost_allowed
            or
            private_ip_allowed
        ):

            raise GeometryV2ContractError(
                "Plain HTTP is permitted only for localhost/private-IP Geometry-v2 endpoints. "
                "Public endpoints must use HTTPS."
            )


    return value.rstrip(
        "/"
    )


def _decode_json_response(
    data: bytes,
) -> Mapping[str, Any]:

    try:

        payload = json.loads(
            data.decode(
                "utf-8"
            )
        )

    except Exception as exc:

        raise GeometryV2TransportError(
            "Geometry-v2 service returned invalid JSON."
        ) from exc


    if not isinstance(
        payload,
        Mapping,
    ):

        raise GeometryV2TransportError(
            "Geometry-v2 service response must be a JSON object."
        )


    return payload


def _multipart_body(
    *,
    manifest_bytes: bytes,
    frame_png_bytes: bytes,
    boundary: str,
) -> bytes:

    chunks: list[bytes] = []


    def add_text(
        value: str,
    ):

        chunks.append(
            value.encode(
                "utf-8"
            )
        )


    add_text(
        f"--{boundary}\r\n"
    )

    add_text(
        'Content-Disposition: form-data; name="manifest"; '
        'filename="manifest.json"\r\n'
    )

    add_text(
        "Content-Type: application/json\r\n\r\n"
    )

    chunks.append(
        manifest_bytes
    )

    add_text(
        "\r\n"
    )


    add_text(
        f"--{boundary}\r\n"
    )

    add_text(
        'Content-Disposition: form-data; name="frame"; '
        'filename="frame.png"\r\n'
    )

    add_text(
        "Content-Type: image/png\r\n\r\n"
    )

    chunks.append(
        frame_png_bytes
    )

    add_text(
        "\r\n"
    )

    add_text(
        f"--{boundary}--\r\n"
    )


    return b"".join(
        chunks
    )


class GeometryV2Client:

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ):

        resolved_url = (
            base_url
            if base_url is not None
            else os.environ.get(
                SERVICE_URL_ENV
            )
        )


        if not resolved_url:

            raise GeometryV2ContractError(
                f"Geometry-v2 service URL is required. "
                f"Set {SERVICE_URL_ENV} or pass base_url."
            )


        if (
            not isinstance(
                timeout_seconds,
                (int, float),
            )
            or
            isinstance(
                timeout_seconds,
                bool,
            )
            or
            float(
                timeout_seconds
            )
            <=
            0.0
        ):

            raise GeometryV2ContractError(
                "timeout_seconds must be positive."
            )


        self.base_url = _validate_base_url(
            str(
                resolved_url
            )
        )


        self.timeout_seconds = float(
            timeout_seconds
        )


        self.max_attempts = MAX_ATTEMPTS


    def get_runtime(
        self,
    ) -> GeometryV2Readiness:

        request = urllib.request.Request(
            url=(
                self.base_url
                +
                READINESS_ENDPOINT
            ),
            method="GET",
        )


        try:

            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:

                body = response.read()


                if int(
                    response.status
                ) != 200:

                    raise GeometryV2TransportError(
                        "Geometry-v2 readiness endpoint returned "
                        f"HTTP {response.status}."
                    )


        except urllib.error.HTTPError as exc:

            body = exc.read()

            raise GeometryV2TransportError(
                "Geometry-v2 readiness endpoint returned "
                f"HTTP {exc.code}: "
                f"{body.decode('utf-8', errors='replace')[:1000]}"
            ) from exc


        except urllib.error.URLError as exc:

            raise GeometryV2TransportError(
                f"Geometry-v2 readiness request failed: {exc}"
            ) from exc


        payload = _decode_json_response(
            body
        )


        result = parse_readiness_payload(
            payload
        )


        if not result.ready:

            raise GeometryV2RuntimeNotReadyError(
                result.failure_reason
                or
                "Geometry-v2 runtime is not ready."
            )


        return result


    def infer_frame(
        self,
        *,
        video_sha256: str,
        frame_idx: int,
        frame_bgr: np.ndarray,
        frame_png_bytes: bytes,
        check_readiness: bool = True,
    ) -> GeometryV2InferenceResponse:

        if check_readiness:

            self.get_runtime()


        manifest = build_geometry_request_manifest(
            video_sha256=video_sha256,
            frame_idx=frame_idx,
            frame_bgr=frame_bgr,
            frame_png_bytes=frame_png_bytes,
        )


        manifest_bytes = json.dumps(
            manifest,
            separators=(
                ",",
                ":",
            ),
            allow_nan=False,
        ).encode(
            "utf-8"
        )


        boundary = (
            "----SoccerMLGeometryV2"
            +
            manifest[
                "request_id"
            ][
                :24
            ]
        )


        body = _multipart_body(
            manifest_bytes=manifest_bytes,
            frame_png_bytes=frame_png_bytes,
            boundary=boundary,
        )


        request = urllib.request.Request(
            url=(
                self.base_url
                +
                INFERENCE_ENDPOINT
            ),
            data=body,
            method="POST",
            headers={
                "Content-Type":
                    (
                        "multipart/form-data; "
                        f"boundary={boundary}"
                    ),

                "Content-Length":
                    str(
                        len(
                            body
                        )
                    ),
            },
        )


        try:

            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:

                response_body = response.read()


                if int(
                    response.status
                ) != 200:

                    raise GeometryV2TransportError(
                        "Geometry-v2 inference endpoint returned "
                        f"HTTP {response.status}."
                    )


        except urllib.error.HTTPError as exc:

            error_body = exc.read()

            raise GeometryV2TransportError(
                "Geometry-v2 inference endpoint returned "
                f"HTTP {exc.code}: "
                f"{error_body.decode('utf-8', errors='replace')[:1000]}"
            ) from exc


        except urllib.error.URLError as exc:

            raise GeometryV2TransportError(
                f"Geometry-v2 inference request failed: {exc}"
            ) from exc


        payload = _decode_json_response(
            response_body
        )


        return parse_inference_payload(
            payload,
            expected_manifest=manifest,
        )
