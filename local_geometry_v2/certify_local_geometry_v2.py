"""
LOCAL RTX-2060 Geometry-v2 independent certification.

Run from the MAIN project venv (the one that runs Semantic-v2):

    .\\.venv\\Scripts\\python.exe local_geometry_v2\\certify_local_geometry_v2.py

Produces (never overwrites an existing freeze):

    outputs/notebook50_local/local_geometry_v2/local_geometry_v2_certification.json
    outputs/notebook50_local/local_geometry_v2/README_LOCAL_GEOMETRY_V2_CERTIFICATION.txt
    outputs/notebook50_local/local_geometry_v2/historical_frames/*.jpg
    outputs/notebook50_local/local_geometry_v2/historical_regression/*.json

Gates:
  * local runtime ready (versions / CUDA / RTX 2060 / clean PnLCalib HEAD / weights / policy)
  * worker + adapter source SHA-256 recorded
  * same-machine determinism (two independent worker processes agree bit-for-bit)
  * Notebook-37 historical regression on the 4 Palace-Arsenal events, classified honestly as
      BYTE_EXACT / NUMERICALLY_EQUIVALENT / SEMANTICALLY_EQUIVALENT / MISMATCH
  * Stage-7-shaped adapter call accepted by the frozen semantic contract
    (soccer_ml.v2.geometry_stage_handlers.inference_response_to_geometry_result)
  * reference-free enforcement (manifest with a forbidden semantic key is rejected)

The independent Cabo Verde clip is NOT touched here.
The historical L40 server certification artefacts are NOT modified here.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from soccer_ml import local_geometry_v2 as LG  # noqa: E402
from soccer_ml.v2.geometry_client import validate_homography_pair  # noqa: E402
from soccer_ml.v2.geometry_stage_handlers import inference_response_to_geometry_result  # noqa: E402

OUT_ROOT = LG.LOCAL_GEOMETRY_OUTPUT_ROOT
FRAMES_DIR = OUT_ROOT / "historical_frames"
REG_DIR = OUT_ROOT / "historical_regression"
CERT_PATH = LG.LOCAL_CERTIFICATION_PATH
README_PATH = OUT_ROOT / "README_LOCAL_GEOMETRY_V2_CERTIFICATION.txt"

PALACE_VIDEO = (
    ROOT / "data" / "raw" / "video" / "soccernet" / "england_epl" / "2014-2015"
    / "2015-02-21 - 18-00 Crystal Palace 1 - 2 Arsenal" / "1_720p.mkv"
)
STORED_PREDICTIONS = (
    ROOT / "data" / "processed" / "notebook37_gpu_return_bundle" / "operational_calibration"
    / "notebook37_pnlcalib_operational_predictions.json"
)
POLICY_PATH = (
    ROOT / "data" / "processed" / "notebook37_gpu_return_bundle" / "frozen_policy"
    / "notebook37_pnlcalib_operational_policy.json"
)
SERVER_WORKER = ROOT / "outputs" / "notebook50_local" / "server_handoff" / "extracted" / "geometry_v2_worker.py"
ADAPTER_PATH = SRC / "soccer_ml" / "local_geometry_v2.py"
WORKER_PATH = LG.WORKER_PATH

EXPECTED = {
    "repo_head": "8c87391d6f4ea40c5e4d65e61529916c7a49ce62",
    "SV_kp": "7ea78fa76aaf94976a8eca428d6e3c59697a93430cba1a4603e20284b61f5113",
    "SV_lines": "d72f4ed71734a2e3df9fa084f666e9b8adaef21bf69bac8952d6d3f970ff7455",
    "policy_file": "90e71b7ebd7d002b62e702d9af61b6cbaa24abc2833f0b19b576549d065a66d1",
    "policy_fingerprint": "62e0431373c6a665cbe07561c178cf750dd78ea610b11678644a28d7a0d342fd",
    "server_worker": "2c3fc0b8a40375a3a79c19e9a5d094cea2c78a7f8782d282264b9cc1ab2eb71f",
    "l40_fingerprint": "715f9b64b964b9a2b8d8011d231d132819ccff540563aa34e9284ae94c976d5e",
}

NUMERIC_EQUIVALENCE_MAX_PITCH_DISPLACEMENT_M = 0.05
NUMERIC_EQUIVALENCE_MAX_ABS_H_DELTA = 1e-4


def banner(text):
    print()
    print("-" * 100)
    print(text)
    print("-" * 100)


def geometry_python(code: str, timeout=900) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [str(LG.GEOMETRY_VENV_PYTHON), "-c", code],
        cwd=str(WORKER_PATH.parent),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def run_worker(args, timeout=1800) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [str(LG.GEOMETRY_VENV_PYTHON), str(WORKER_PATH), *args],
        cwd=str(WORKER_PATH.parent),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def main():
    print("=" * 100)
    print("LOCAL RTX-2060 GEOMETRY-v2 INDEPENDENT CERTIFICATION")
    print("=" * 100)

    if CERT_PATH.exists():
        raise RuntimeError(f"Refusing to overwrite existing local certification: {CERT_PATH}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    REG_DIR.mkdir(parents=True, exist_ok=True)

    gates = {}
    started = time.time()

    # ------------------------------------------------------------------------------------------------
    banner("1. SOURCE / ARTIFACT IDENTITIES")
    identities = {
        "worker_path": str(WORKER_PATH),
        "worker_sha256": LG.sha256_file(WORKER_PATH),
        "adapter_path": str(ADAPTER_PATH),
        "adapter_sha256": LG.sha256_file(ADAPTER_PATH),
        "policy_path": str(POLICY_PATH),
        "policy_file_sha256": LG.sha256_file(POLICY_PATH),
        "policy_fingerprint": json.loads(POLICY_PATH.read_text(encoding="utf-8")).get("policy_sha256"),
        "server_worker_path": str(SERVER_WORKER),
        "server_worker_sha256": LG.sha256_file(SERVER_WORKER) if SERVER_WORKER.exists() else None,
        "stored_predictions_path": str(STORED_PREDICTIONS),
        "stored_predictions_sha256": LG.sha256_file(STORED_PREDICTIONS),
        "palace_video_path": str(PALACE_VIDEO),
        "palace_video_exists": PALACE_VIDEO.exists(),
    }
    for k, v in identities.items():
        print(f"{k:28s}: {v}")
    gates["policy_file_sha_exact"] = identities["policy_file_sha256"] == EXPECTED["policy_file"]
    gates["policy_fingerprint_exact"] = identities["policy_fingerprint"] == EXPECTED["policy_fingerprint"]
    gates["server_worker_untouched"] = identities["server_worker_sha256"] == EXPECTED["server_worker"]
    gates["palace_video_present"] = identities["palace_video_exists"]

    # ------------------------------------------------------------------------------------------------
    banner("2. LOCAL RUNTIME READINESS (adapter -> worker, fresh process)")
    adapter = LG.LocalGeometryV2Adapter(request_root=OUT_ROOT / "certification_requests")
    readiness = adapter.get_runtime()
    fp = dict(readiness.runtime_fingerprint)
    runtime = dict(fp["runtime"])
    print(f"ready                       : {readiness.ready}")
    print(f"runtime_profile             : {fp['runtime_profile']}")
    print(f"fingerprint_sha256          : {fp['fingerprint_sha256']}")
    print(f"gpu_name                    : {runtime.get('gpu_name')}")
    print(f"python/torch/cuda/cudnn     : {runtime.get('python')} / {runtime.get('torch_full')} / {runtime.get('torch_cuda')} / {runtime.get('cudnn')}")
    print(f"repository.head             : {fp['repository']['head']}")
    print(f"repository.tracked_clean    : {fp['repository']['tracked_source_clean']}")
    print(f"weights                     : {fp['weight_hashes']}")
    gates["runtime_ready"] = readiness.ready is True
    gates["runtime_is_rtx2060"] = "RTX 2060" in str(runtime.get("gpu_name"))
    gates["runtime_cuda_available"] = runtime.get("cuda_available") is True
    gates["repo_head_exact"] = fp["repository"]["head"] == EXPECTED["repo_head"]
    gates["repo_tracked_clean"] = fp["repository"]["tracked_source_clean"] is True
    gates["weights_exact"] = (
        fp["weight_hashes"].get("SV_kp") == EXPECTED["SV_kp"]
        and fp["weight_hashes"].get("SV_lines") == EXPECTED["SV_lines"]
    )
    gates["fingerprint_not_l40"] = fp["fingerprint_sha256"] != EXPECTED["l40_fingerprint"]
    gates["fingerprint_worker_sha_matches_file"] = fp["worker"]["sha256"] == identities["worker_sha256"]
    gates["fingerprint_disclaims_l40"] = fp["certification_lineage"]["claims_l40_certification"] is False

    # ------------------------------------------------------------------------------------------------
    banner("3. HISTORICAL FRAME EXTRACTION (Notebook-37 method: CAP_PROP_POS_FRAMES seek, JPEG q97)")
    stored = json.loads(STORED_PREDICTIONS.read_text(encoding="utf-8"))
    events = [(r["event_id"], int(r["event_rank"]), int(r["frame_number"])) for r in stored]
    extract_code = (
        "import cv2, json, sys\n"
        f"video = {str(PALACE_VIDEO)!r}\n"
        f"events = {events!r}\n"
        f"out_dir = {str(FRAMES_DIR)!r}\n"
        "rows = []\n"
        "for event_id, rank, frame_number in events:\n"
        "    cap = cv2.VideoCapture(video)\n"
        "    assert cap.isOpened(), video\n"
        "    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))\n"
        "    ok, frame = cap.read(); cap.release()\n"
        "    assert ok and frame is not None, frame_number\n"
        "    path = out_dir + '/' + f'{event_id}_operational_frame_{frame_number}.jpg'\n"
        "    assert cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 97])\n"
        "    rows.append({'event_id': event_id, 'event_rank': rank, 'frame_number': frame_number,\n"
        "                 'image_path': path, 'frame_shape': list(frame.shape),\n"
        "                 'reference_used_for_inference': False, 'reference_used_for_selection': False})\n"
        "print(json.dumps({'cv2': cv2.__version__, 'rows': rows}))\n"
    )
    proc = geometry_python(extract_code)
    if proc.returncode != 0:
        raise RuntimeError(f"frame extraction failed: {proc.stderr[-3000:]}")
    extraction = json.loads(proc.stdout.strip().splitlines()[-1])
    manifest_rows = []
    frame_hashes = {}
    for row in extraction["rows"]:
        frame_hashes[row["event_id"]] = LG.sha256_file(row["image_path"])
        manifest_rows.append({k: row[k] for k in (
            "event_id", "event_rank", "frame_number", "image_path",
            "reference_used_for_inference", "reference_used_for_selection")})
        print(f"{row['event_id']} frame {row['frame_number']} shape={row['frame_shape']} sha256={frame_hashes[row['event_id']]}")
    print(f"extraction OpenCV           : {extraction['cv2']}")
    legacy_manifest = REG_DIR / "legacy_manifest.json"
    legacy_manifest.write_text(json.dumps(manifest_rows, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------------------------------------
    banner("4. LEGACY-CONTRACT RUNS (two independent processes -> determinism)")
    legacy_results = []
    timings = []
    for run_idx in (1, 2):
        out_path = REG_DIR / f"legacy_result_run{run_idx}.json"
        t0 = time.time()
        proc = run_worker(["--legacy", "--manifest", str(legacy_manifest), "--output", str(out_path)])
        timings.append(round(time.time() - t0, 2))
        (REG_DIR / f"legacy_run{run_idx}_stderr.log").write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(f"legacy run {run_idx} failed: {proc.stderr[-3000:]}")
        legacy_results.append(json.loads(out_path.read_text(encoding="utf-8")))
        print(f"run {run_idx}: {timings[-1]} s, rows={len(legacy_results[-1])}")
    determinism_exact = (
        json.dumps(legacy_results[0], sort_keys=True) == json.dumps(legacy_results[1], sort_keys=True)
    )
    print(f"same-machine determinism    : {determinism_exact}")
    gates["same_machine_determinism_exact"] = determinism_exact
    local_rows = legacy_results[0]

    # ------------------------------------------------------------------------------------------------
    banner("5. NOTEBOOK-37 HISTORICAL REGRESSION (L40 stored vs local RTX 2060)")
    H_KEYS = ("base_H_image_to_pitch", "base_H_pitch_to_image", "pnl_H_image_to_pitch",
              "pnl_H_pitch_to_image", "selected_H_image_to_pitch", "selected_H_pitch_to_image")
    per_event = []
    for s, o in zip(stored, local_rows):
        assert s["event_id"] == o["event_id"]
        s_kp = {str(k): v for k, v in s["raw_keypoints"].items()}
        o_kp = {str(k): v for k, v in o["raw_keypoints"].items()}
        s_ln = {str(k): v for k, v in s["raw_lines"].items()}
        o_ln = {str(k): v for k, v in o["raw_lines"].items()}
        same_kp_ids = sorted(s_kp, key=int) == sorted(o_kp, key=int)
        same_ln_ids = sorted(s_ln, key=int) == sorted(o_ln, key=int)
        coord_shift_px = {}
        max_dp = 0.0
        for k in set(s_kp) & set(o_kp):
            dx = abs(s_kp[k]["x"] - o_kp[k]["x"]); dy = abs(s_kp[k]["y"] - o_kp[k]["y"])
            if dx or dy:
                coord_shift_px[k] = {"dx_960": dx, "dy_540": dy}
            max_dp = max(max_dp, abs(s_kp[k]["p"] - o_kp[k]["p"]))
        h_delta = {}
        for k in H_KEYS:
            if s.get(k) is not None and o.get(k) is not None:
                h_delta[k] = float(np.max(np.abs(np.asarray(s[k], float) - np.asarray(o[k], float))))
        # pitch displacement (metres) of the detected keypoints' frame pixels under stored vs local selected H
        disp = None
        if s.get("selected_H_image_to_pitch") is not None and o.get("selected_H_image_to_pitch") is not None:
            Hs = np.asarray(s["selected_H_image_to_pitch"], float)
            Ho = np.asarray(o["selected_H_image_to_pitch"], float)
            fw, fh = o["frame_width"], o["frame_height"]
            pts = np.array([[v["x"] * fw / 960.0, v["y"] * fh / 540.0, 1.0] for v in o_kp.values()], float)
            ps = (Hs @ pts.T).T; ps = ps[:, :2] / ps[:, 2:3]
            po = (Ho @ pts.T).T; po = po[:, :2] / po[:, 2:3]
            d = np.linalg.norm(ps - po, axis=1)
            disp = {"mean_m": float(d.mean()), "max_m": float(d.max()), "n_points": int(len(d))}
        full_row_exact = all(
            json.dumps(s.get(k), sort_keys=True) == json.dumps(o.get(k), sort_keys=True)
            for k in ("raw_keypoints", "raw_lines", "base_result", "pnl_result", *H_KEYS,
                      "selected_variant", "selection_status")
        )
        semantic_equal = (
            s["selected_variant"] == o["selected_variant"]
            and s["selection_status"] == o["selection_status"]
            and len(s_kp) == o["raw_keypoint_count"]
            and len(s_ln) == o["raw_line_count"]
            and same_kp_ids and same_ln_ids
            and o["selected_H_image_to_pitch"] is not None
        )
        numeric_equal = (
            semantic_equal and disp is not None
            and disp["max_m"] <= NUMERIC_EQUIVALENCE_MAX_PITCH_DISPLACEMENT_M
            and max(h_delta.values()) <= NUMERIC_EQUIVALENCE_MAX_ABS_H_DELTA
        )
        if full_row_exact:
            cls = "BYTE_EXACT"
        elif numeric_equal:
            cls = "NUMERICALLY_EQUIVALENT"
        elif semantic_equal:
            cls = "SEMANTICALLY_EQUIVALENT"
        else:
            cls = "MISMATCH"
        row = {
            "event_id": s["event_id"],
            "frame_number": s["frame_number"],
            "input_frame_sha256": frame_hashes[s["event_id"]],
            "classification": cls,
            "stored_variant": s["selected_variant"], "local_variant": o["selected_variant"],
            "stored_status": s["selection_status"], "local_status": o["selection_status"],
            "stored_raw_keypoints": len(s_kp), "local_raw_keypoints": o["raw_keypoint_count"],
            "stored_raw_lines": len(s_ln), "local_raw_lines": o["raw_line_count"],
            "keypoint_ids_identical": same_kp_ids, "line_ids_identical": same_ln_ids,
            "keypoint_coordinates_identical": not coord_shift_px,
            "keypoint_coordinate_shifts_px_model_space": coord_shift_px,
            "max_abs_confidence_delta": max_dp,
            "max_abs_homography_delta": h_delta,
            "stored_rep_err": {"base": s["base_result"]["rep_err"], "pnl": s["pnl_result"]["rep_err"]},
            "local_rep_err": {"base": o["base_result"]["rep_err"], "pnl": o["pnl_result"]["rep_err"]},
            "selected_H_pitch_displacement_of_keypoints": disp,
        }
        per_event.append(row)
        print(f"{row['event_id']}: {cls:24s} variant={o['selected_variant']} kp={o['raw_keypoint_count']} "
              f"lines={o['raw_line_count']} coord_shifts={len(coord_shift_px)} "
              f"maxH|d|={max(h_delta.values()):.3e} pitch_disp max={disp['max_m']:.4f} m mean={disp['mean_m']:.4f} m")
    classes = [r["classification"] for r in per_event]
    gates["historical_all_semantically_equivalent_or_better"] = all(
        c in ("BYTE_EXACT", "NUMERICALLY_EQUIVALENT", "SEMANTICALLY_EQUIVALENT") for c in classes)
    gates["historical_all_pnl_refined"] = all(r["local_variant"] == "PNL_REFINED" for r in per_event)
    historical_summary = {
        "byte_exact_events": classes.count("BYTE_EXACT"),
        "numerically_equivalent_events": classes.count("NUMERICALLY_EQUIVALENT"),
        "semantically_equivalent_events": classes.count("SEMANTICALLY_EQUIVALENT"),
        "mismatch_events": classes.count("MISMATCH"),
        "numeric_equivalence_tolerances": {
            "max_pitch_displacement_m": NUMERIC_EQUIVALENCE_MAX_PITCH_DISPLACEMENT_M,
            "max_abs_homography_entry_delta": NUMERIC_EQUIVALENCE_MAX_ABS_H_DELTA,
        },
        "byte_exact_reproduction_achieved": classes.count("BYTE_EXACT") == len(classes),
        "caveat": (
            "Historical rows were produced on an NVIDIA L40 from JPEG frames whose bytes are not "
            "available locally; local frames are re-extracted from the same source video with the "
            "same Notebook-37 method. Residual differences stem from GPU floating-point differences "
            "(heatmap confidences ~1e-5) and/or frame re-encoding, which can move a keypoint argmax by "
            "one heatmap cell (2 px in 960x540 model space) and change the RANSAC solution."
        ),
    }
    print(json.dumps(historical_summary, indent=2))

    # ------------------------------------------------------------------------------------------------
    banner("6. STAGE-7-SHAPED ADAPTER CALL -> FROZEN SEMANTIC CONTRACT")
    import cv2
    probe_row = manifest_rows[0]
    frame_bgr = cv2.imread(probe_row["image_path"], cv2.IMREAD_COLOR)
    frame_png = LG.default_png_encoder(frame_bgr)
    pseudo_video_sha = LG.sha256_file(probe_row["image_path"])
    response = adapter(
        video_sha256=pseudo_video_sha,
        frame_idx=int(probe_row["frame_number"]),
        frame_bgr=frame_bgr,
        frame_png=frame_png,
    )
    geometry_result = inference_response_to_geometry_result(response)  # runs validate_geometry_v2_result
    inverse_delta = validate_homography_pair(response.homography_image_to_pitch, response.homography_pitch_to_image)
    print(f"response.success            : {response.success}")
    print(f"response.selected_variant   : {response.selected_variant} / {response.selection_status}")
    print(f"response.frame_idx          : {response.frame_idx}")
    print(f"keypoints / lines           : {response.keypoint_count} / {response.line_count}")
    print(f"reference_free              : {response.reference_free}")
    print(f"inverse pair max|delta|     : {inverse_delta:.3e}")
    print(f"GeometryV2Result.success    : {geometry_result.success} ({geometry_result.validation_status})")
    print(f"request dir                 : {adapter.last_request_dir}")
    gates["stage7_adapter_call_success"] = response.success is True
    gates["stage7_frame_echo_exact"] = response.frame_idx == int(probe_row["frame_number"])
    gates["stage7_semantic_contract_accepted"] = geometry_result.success is True
    gates["stage7_reference_free"] = response.reference_free is True
    gates["stage7_fingerprint_matches_readiness"] = (
        response.runtime_fingerprint["fingerprint_sha256"] == fp["fingerprint_sha256"]
    )
    adapter_probe = {
        "request_id": response.request_id,
        "selected_variant": response.selected_variant,
        "selection_status": response.selection_status,
        "keypoint_count": response.keypoint_count,
        "line_count": response.line_count,
        "inverse_pair_max_abs_delta": inverse_delta,
        "homography_image_to_pitch": response.homography_image_to_pitch,
        "homography_pitch_to_image": response.homography_pitch_to_image,
        "request_dir": str(adapter.last_request_dir),
    }

    # ------------------------------------------------------------------------------------------------
    banner("7. REFERENCE-FREE ENFORCEMENT (forbidden key must be rejected)")
    tamper_dir = OUT_ROOT / "certification_requests" / "_tamper_test"
    tamper_dir.mkdir(parents=True, exist_ok=True)
    good_manifest = json.loads((Path(adapter.last_request_dir) / "manifest.json").read_text(encoding="utf-8"))
    bad_manifest = dict(good_manifest)
    bad_manifest["reference_homography"] = response.homography_image_to_pitch
    (tamper_dir / "manifest.json").write_text(json.dumps(bad_manifest), encoding="utf-8")
    (tamper_dir / "frame.png").write_bytes(frame_png)
    proc = run_worker(["--infer", "--manifest", str(tamper_dir / "manifest.json"),
                       "--frame", str(tamper_dir / "frame.png"), "--output", str(tamper_dir / "response.json")])
    tamper_payload = json.loads((tamper_dir / "response.json").read_text(encoding="utf-8"))
    print(f"tampered manifest success   : {tamper_payload['success']}")
    print(f"tampered selection_status   : {tamper_payload['selection_status']}")
    print(f"tampered failure_reason     : {tamper_payload['failure_reason']}")
    gates["reference_free_enforced"] = (
        tamper_payload["success"] is False
        and tamper_payload["selection_status"] == "INPUT_CONTRACT_FAILURE"
        and "reference_homography" in str(tamper_payload["failure_reason"])
        and tamper_payload["homography_image_to_pitch"] is None
    )

    # ------------------------------------------------------------------------------------------------
    banner("8. CERTIFICATION DECISION")
    certified = all(bool(v) for v in gates.values())
    for k, v in gates.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    print()
    print(f"LOCAL_GEOMETRY_V2_CERTIFIED : {certified}")

    payload = {
        "LOCAL_GEOMETRY_V2_CERTIFIED": certified,
        "certification_kind": "LOCAL_RTX2060_RECERTIFICATION",
        "not_a_claim_of": "ORIGINAL_NVIDIA_L40_SERVER_CERTIFICATION",
        "historical_l40_runtime_fingerprint_sha256": EXPECTED["l40_fingerprint"],
        "runtime_profile": fp["runtime_profile"],
        "runtime_fingerprint_sha256": fp["fingerprint_sha256"],
        "runtime_fingerprint": fp,
        "readiness_policy": dict(readiness.policy),
        "readiness_weights": dict(readiness.weights),
        "identities": identities,
        "gates": gates,
        "historical_regression": {
            "events": per_event,
            "summary": historical_summary,
            "legacy_manifest": str(legacy_manifest),
            "legacy_results": [str(REG_DIR / "legacy_result_run1.json"), str(REG_DIR / "legacy_result_run2.json")],
            "run_seconds": timings,
            "frame_extraction_opencv": extraction["cv2"],
        },
        "stage7_adapter_probe": adapter_probe,
        "reference_free_tamper_test": tamper_payload,
        "cabo_verde_touched": False,
        "server_certification_modified": False,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    CERT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    README_PATH.write_text(
        "LOCAL RTX-2060 GEOMETRY-v2 CERTIFICATION\n"
        "=======================================\n\n"
        f"LOCAL_GEOMETRY_V2_CERTIFIED : {certified}\n"
        f"Runtime profile             : {fp['runtime_profile']}\n"
        f"Runtime fingerprint SHA-256 : {fp['fingerprint_sha256']}\n"
        f"GPU                         : {runtime.get('gpu_name')}\n"
        f"Worker SHA-256              : {identities['worker_sha256']}\n"
        f"Adapter SHA-256             : {identities['adapter_sha256']}\n"
        f"PnLCalib HEAD               : {fp['repository']['head']} (tracked clean={fp['repository']['tracked_source_clean']})\n"
        f"SV_kp / SV_lines            : {fp['weight_hashes']['SV_kp']} / {fp['weight_hashes']['SV_lines']}\n"
        f"Policy fingerprint          : {fp['policy_fingerprint']}\n\n"
        "This is a NEW local certification of a re-implemented PnLCalib runner on an RTX 2060.\n"
        "It is NOT the original NVIDIA L40 server certification (fingerprint 715f9b64...), which\n"
        "remains untouched under outputs/notebook50_local/server_handoff/.\n\n"
        "Historical Notebook-37 regression (4 Palace-Arsenal events):\n"
        f"  byte-exact              : {historical_summary['byte_exact_events']}\n"
        f"  numerically equivalent  : {historical_summary['numerically_equivalent_events']}\n"
        f"  semantically equivalent : {historical_summary['semantically_equivalent_events']}\n"
        f"  mismatch                : {historical_summary['mismatch_events']}\n"
        f"  tolerances              : {historical_summary['numeric_equivalence_tolerances']}\n\n"
        f"Certification JSON: {CERT_PATH}\n",
        encoding="utf-8",
    )
    print(f"\nWrote: {CERT_PATH}")
    print(f"Wrote: {README_PATH}")
    print(f"Elapsed: {payload['elapsed_seconds']} s")
    if not certified:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
