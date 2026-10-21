"""
NOTEBOOK 50 -- CELL 40 (LOCAL RTX-2060 VARIANT) -- DRIVER

Independent real Semantic-v2 public-API end-to-end regression on the Cabo Verde clip,
executed in a FRESH PROCESS, with Geometry-v2 served by the locally certified RTX-2060
adapter injected through the frozen public API's `geometry_inference_callable` boundary.

Run from the MAIN project venv:

    .\\.venv\\Scripts\\python.exe local_geometry_v2\\run_cell40_local_recertification.py

Preserves the original Cell-40 guards:
  * Cell-39 public-v2 API freeze identity
  * frozen production-v1 and frozen Semantic-v2 source SHA-256 + exact v2 file set
  * independent Cabo input identity
  * clean-output gate (never overwrites a prior real run)
  * child static safety gate (exactly two public API calls, no direct driver calls)
  * post-execution frozen-source guards
Adds:
  * local Geometry-v2 certification freeze must exist, be certified, and match the worker/adapter on disk

Writes:
    outputs/notebook50_local/v2_implementation/notebook50_cell40_local_semantic_v2_real_e2e_regression.json
"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PKG = SRC / "soccer_ml"
V2 = PKG / "v2"
FREEZE = ROOT / "outputs" / "notebook50_local" / "freeze"
AUDIT_ROOT = ROOT / "outputs" / "notebook50_local" / "v2_implementation"
E2E_ROOT = ROOT / "outputs" / "notebook50_local" / "e2e"
LOCAL_GEOM = ROOT / "outputs" / "notebook50_local" / "local_geometry_v2"

CELL39_AUDIT = AUDIT_ROOT / "notebook50_cell39_semantic_v2_public_api_adversarial_tests.json"
CELL39_FREEZE = FREEZE / "notebook50_semantic_v2_public_api_initial_freeze.json"
CELL39_README = FREEZE / "README_NOTEBOOK50_SEMANTIC_V2_PUBLIC_API_INITIAL_FREEZE.txt"
CABO_VIDEO = ROOT / "outputs" / "notebook44_local" / "input" / "caboverde.mp4"
LOCAL_CERT = LOCAL_GEOM / "local_geometry_v2_certification.json"
WORKER = ROOT / "local_geometry_v2" / "local_geometry_v2_worker.py"
ADAPTER = PKG / "local_geometry_v2.py"

CHILD_SCRIPT = E2E_ROOT / "cell40_local_semantic_v2_e2e_child.py"
CHILD_RESULT = E2E_ROOT / "cell40_local_semantic_v2_e2e_result.json"
ANNOTATED_OUTPUT = E2E_ROOT / "caboverde_semantic_v2_local_annotated.mp4"
PIPELINE_LOG = E2E_ROOT / "cell40_local_semantic_v2_pipeline.log"
CHILD_STDOUT = E2E_ROOT / "cell40_local_child_stdout.log"
CHILD_STDERR = E2E_ROOT / "cell40_local_child_stderr.log"
CELL40_AUDIT = AUDIT_ROOT / "notebook50_cell40_local_semantic_v2_real_e2e_regression.json"

EXPECTED_CELL39_AUDIT_SHA = "3339a9e409cf82906a7111129f0d0e6d1be7db27f844612c041011498bb888ee"
EXPECTED_CELL39_FREEZE_SHA = "7b5abe5afdc87b73361660b61d2781d01931918ad4ab46a6096858bae01991d5"
EXPECTED_CELL39_README_SHA = "e2fee2e74b88694f96cc2cec6f8011d3e748678900760e75842b13b03e7092a9"
EXPECTED_PUBLIC_API_SHA = "f06e0d74d8c3126d98a960b4059dee9d7b6c7cf3913d3d88c18802795a69e123"
EXPECTED_EXECUTION_SHA = "48345ebc6e607c096b1002b578c9f61ac3ee25c532bdb8398d4956861e85f9fa"
EXPECTED_CABO_SHA = "204337be5b5b5e381c07deaff70492b2885bc769d14997d68b1d8111e13dc840"

FROZEN_V1 = {
    PKG / "api.py": "479fe7726c9049d9e126a780104aaa65fc67eb69a41d1ff4e19dbddf378a50db",
    PKG / "clip_pipeline.py": "913dee8e35a0612d3d6b9ad5465689ac682d99f7d485710a0ce23bd69396f880",
    PKG / "results.py": "883fae0407008e7de04189cc0332e8e836213fd2f87ac24a814db958e0466db5",
    PKG / "shot_pipeline.py": "0d205673398baaa4c3c20c02e3dcbbe2fa4409f093f3a3723fd258f7278e75f5",
}
FROZEN_V2 = {
    "__init__.py": "e183bdad0ac692d637b0c18a028cecd99b291f3cffe818ce06e44813b28139f1",
    "contracts.py": "9311b5f78b778792ab6e5321d5bb2562c7e8e25cb18f0ff523e225f2cbae998f",
    "geometry_client.py": "d3d34dca650de08790308a110c8dd57dddc7543143ff420c232fb53bc5295e19",
    "semantic_contracts.py": "58bcbe5d91a7b9a4ddfd65290673e0ebe53b20174f9c058a6d0c13b19510f322",
    "orchestration_contracts.py": "966f27b1784ce366a3357c3dc5ba091c760fc31ac6b2cabeeeb17a644e58056a",
    # r2: Cell-39 baseline 37b9a0de... superseded by the documented return-value
    # contract defect fix (see V1_ADAPTER_DEFECT_FIX below). Only this one file
    # differs from the Cell-39 frozen Semantic-v2 set.
    "v1_adapter.py": "5505c35f8c0e92241fabd9cfa8da82b8bea608aad922bc88b00fa64ff66856b0",
    "geometry_stage_handlers.py": "ea44ae6a5a04886938ec0dac6f66c5cf6afe90e1d1bb333854aaede724e13f1c",
    "geometry_stage8_selection.py": "83f2e9bfe3f8cb7d4767a3d46e456c772ac17a8a69228f3eb600f080310a621e",
    # r2: Cell-39 baseline a0d5bae8... superseded by the documented projective-scale
    # inverse-consistency defect fix (see STAGE9_DEFECT_FIX below).
    "geometry_stage9_validation.py": "5e57319bc0712800c1af8daa46dc860c9aeacfe4eb6f63f37d8c6974aa18d3f7",
    "geometry_stage10_projection.py": "409746fc0ee94c6b1e6ef3ebb21ed6aa633a80678afe76852c998493171ae49e",
    "shooter_stage11_assignment.py": "922540ae5e5f177d3e251012099f3ca1d9896e894a13ad48d52226d76f3d96fe",
    "target_goal_stage13_inference.py": "451cea1a1380352535a7b9f23feb413e4332096de99b56b80186b0e1b568b9bd",
    "resolved_downstream_reuse.py": "9aeffc3d1a02a0eccc59ceea32efd42ed964a1df7e49d123521ef93dcfad3719",
    "terminal_stage_reuse.py": "caf251077be52643b4be5ee28422c7342e29a79827a95d0ac46f9457c7a14b96",
    "semantic_v2_pipeline.py": "73f29f97e6d1dd2133a7c2d8e935961af385db13efaa3970ea9de91582994b9e",
    "semantic_v2_execution.py": EXPECTED_EXECUTION_SHA,
    "api.py": EXPECTED_PUBLIC_API_SHA,
}
V1_ADAPTER_DEFECT_FIX = {
    "file": "src/soccer_ml/v2/v1_adapter.py",
    "cell39_frozen_sha256": "37b9a0dee6cd38642771eaf8f073bfa08ad4ec41bff3207bbf2e1ff960610ccc",
    "r2_sha256": FROZEN_V2["v1_adapter.py"],
    "preserved_original_copy": "outputs/notebook50_local/v2_implementation/superseded_sources/v1_adapter.py.cell39_frozen_37b9a0de.txt",
    "record": "outputs/notebook50_local/v2_implementation/notebook50_cell40_local_v1_adapter_return_contract_fix.json",
    "defect": "Adapter raised FrozenV1StageContractError when a real frozen-v1 StageHandler returned a non-None value; "
              "11/18 real frozen-v1 handlers return diagnostic payloads and the frozen-v1 orchestrator discards them.",
}
STAGE9_DEFECT_FIX = {
    "file": "src/soccer_ml/v2/geometry_stage9_validation.py",
    "cell39_frozen_sha256": "a0d5bae8294782ae3bbbb17ce932ba85218e41702aa4dae976a20c867bc4f023",
    "r2_sha256": FROZEN_V2["geometry_stage9_validation.py"],
    "preserved_original_copy": "outputs/notebook50_local/v2_implementation/superseded_sources/geometry_stage9_validation.py.cell39_frozen_a0d5bae8.txt",
    "record": "outputs/notebook50_local/v2_implementation/notebook50_cell40_local_stage9_inverse_consistency_fix.json",
    "defect": "Stage 9 compared raw H_i2p@H_p2i products against the identity; the frozen protocol/client/historical runner "
              "normalise both matrices to H[2,2]=1 so a valid pair multiplies to c*I. Products are now projectively "
              "normalised before comparison; tolerance 1e-8 unchanged.",
}
FROZEN_V2_DEVIATIONS_FROM_CELL39 = {"v1_adapter.py": V1_ADAPTER_DEFECT_FIX, "geometry_stage9_validation.py": STAGE9_DEFECT_FIX}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def banner(text):
    print()
    print("-" * 100)
    print(text)
    print("-" * 100)


def frozen_source_state():
    v1 = {str(p): (p.exists() and sha256_file(p) == s) for p, s in FROZEN_V1.items()}
    v2 = {f: ((V2 / f).exists() and sha256_file(V2 / f) == s) for f, s in FROZEN_V2.items()}
    files = sorted(str(p.relative_to(V2)) for p in V2.rglob("*.py") if "__pycache__" not in p.parts)
    return {
        "v1_exact": all(v1.values()), "v2_exact": all(v2.values()),
        "v2_file_set_exact": files == sorted(FROZEN_V2), "v1": v1, "v2": v2, "v2_files": files,
    }


def main():
    print("=" * 100)
    print("NOTEBOOK 50 -- CELL 40 (LOCAL RTX-2060 VARIANT)")
    print("LOCAL -- INDEPENDENT REAL SEMANTIC-v2 PUBLIC-API END-TO-END RECERTIFICATION")
    print("=" * 100)
    started = time.time()

    banner("CELL-39 PUBLIC-v2 API FREEZE PREFLIGHT")
    c39 = {
        "audit_exact": CELL39_AUDIT.exists() and sha256_file(CELL39_AUDIT) == EXPECTED_CELL39_AUDIT_SHA,
        "freeze_exact": CELL39_FREEZE.exists() and sha256_file(CELL39_FREEZE) == EXPECTED_CELL39_FREEZE_SHA,
        "readme_exact": CELL39_README.exists() and sha256_file(CELL39_README) == EXPECTED_CELL39_README_SHA,
    }
    payload39 = json.loads(CELL39_AUDIT.read_text(encoding="utf-8")) if CELL39_AUDIT.exists() else {}
    c39["cell39_valid"] = bool(payload39.get("validation", {}).get("cell39_valid", False))
    c39["public_api_frozen"] = bool(payload39.get("implementation_status", {}).get("public_v2_api_frozen", False))
    for k, v in c39.items():
        print(f"{k:22s}: {v}")
    if not all(c39.values()):
        raise RuntimeError("Cell 40 (local) did not begin from the exact successful Cell-39 freeze.")

    banner("FROZEN SOURCE + INDEPENDENT INPUT PRE-GUARDS")
    pre = frozen_source_state()
    cabo_sha = sha256_file(CABO_VIDEO) if CABO_VIDEO.exists() else None
    print(f"Frozen production v1 exact : {pre['v1_exact']}")
    print(f"Frozen Semantic-v2 exact   : {pre['v2_exact']}  (r2 pins; documented deviations from Cell-39 baseline:)")
    for fname, fix in FROZEN_V2_DEVIATIONS_FROM_CELL39.items():
        print(f"    {fname:32s} {fix['cell39_frozen_sha256'][:12]}... -> {fix['r2_sha256'][:12]}...")
    print(f"v2 source file set exact   : {pre['v2_file_set_exact']}")
    print(f"Independent Cabo SHA-256   : {cabo_sha}")
    print(f"Independent Cabo exact     : {cabo_sha == EXPECTED_CABO_SHA}")
    if not (pre["v1_exact"] and pre["v2_exact"] and pre["v2_file_set_exact"] and cabo_sha == EXPECTED_CABO_SHA):
        raise RuntimeError("Cell 40 (local) frozen-source or independent-input pre-guard failed.")

    banner("LOCAL GEOMETRY-v2 CERTIFICATION PRE-GUARD")
    if not LOCAL_CERT.exists():
        raise RuntimeError(f"Local Geometry-v2 certification missing: {LOCAL_CERT}")
    cert = json.loads(LOCAL_CERT.read_text(encoding="utf-8"))
    worker_sha = sha256_file(WORKER)
    adapter_sha = sha256_file(ADAPTER)
    cert_ok = {
        "certified": cert.get("LOCAL_GEOMETRY_V2_CERTIFIED") is True,
        "worker_sha_matches_cert": cert.get("identities", {}).get("worker_sha256") == worker_sha,
        "adapter_sha_matches_cert": cert.get("identities", {}).get("adapter_sha256") == adapter_sha,
        "not_l40_fingerprint": cert.get("runtime_fingerprint_sha256") != cert.get("historical_l40_runtime_fingerprint_sha256"),
        "historical_no_mismatch": cert.get("historical_regression", {}).get("summary", {}).get("mismatch_events") == 0,
    }
    for k, v in cert_ok.items():
        print(f"{k:26s}: {v}")
    print(f"pinned fingerprint        : {cert.get('runtime_fingerprint_sha256')}")
    print(f"certification SHA-256     : {sha256_file(LOCAL_CERT)}")
    if not all(cert_ok.values()):
        raise RuntimeError("Local Geometry-v2 certification pre-guard failed.")

    banner("CLEAN CELL-40-LOCAL EXECUTION OUTPUT GATE")
    partial = [p for p in (ANNOTATED_OUTPUT, CHILD_RESULT, PIPELINE_LOG, CHILD_STDOUT, CHILD_STDERR, CELL40_AUDIT) if p.exists()]
    print(f"Partial outputs : {[str(p) for p in partial]}")
    if partial:
        raise RuntimeError("Cell 40 (local) found pre-existing partial outputs. Do not overwrite a prior real run.")

    banner("CHILD STATIC SAFETY GATE")
    child_source = CHILD_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(child_source)
    api_calls = driver_calls = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else (node.func.id if isinstance(node.func, ast.Name) else None)
        if name == "analyze_soccer_clip":
            api_calls += 1
        if name == "run_semantic_v2_clip_pipeline":
            driver_calls += 1
    child_sha = sha256_file(CHILD_SCRIPT)
    print(f"Public-v2 API calls in AST    : {api_calls}")
    print(f"Direct execution-driver calls : {driver_calls}")
    print(f"Child SHA-256                 : {child_sha}")
    if not (api_calls == 2 and driver_calls == 0):
        raise RuntimeError("Cell 40 (local) child safety gate failed.")

    banner("REAL FRESH-PROCESS EXECUTION")
    print(f"Python executable : {sys.executable}")
    print("Call 1            : automatic ambiguity-safe pass")
    print("Call 2            : stateless replay with Track 16 confirmation")
    print("Geometry          : LocalGeometryV2Adapter (RTX 2060, locally certified)")
    t0 = time.time()
    run = subprocess.run([sys.executable, str(CHILD_SCRIPT)], cwd=str(ROOT), capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    elapsed_child = round(time.time() - t0, 1)
    CHILD_STDOUT.write_text(run.stdout, encoding="utf-8")
    CHILD_STDERR.write_text(run.stderr, encoding="utf-8")
    print(f"\nChild return code : {run.returncode}   ({elapsed_child} s)")
    print("\nCHILD STDOUT\n" + "-" * 100)
    print(run.stdout)
    if run.stderr.strip():
        print("\nCHILD STDERR\n" + "-" * 100)
        print(run.stderr[-6000:])
    if run.returncode != 0:
        raise RuntimeError(f"Cell 40 (local) child process failed. Pipeline log: {PIPELINE_LOG}")

    banner("REAL CHILD RESULT VALIDATION")
    if not CHILD_RESULT.exists():
        raise RuntimeError("Child returned successfully but did not write its result JSON.")
    res = json.loads(CHILD_RESULT.read_text(encoding="utf-8"))
    structural = bool(
        res["public_api_calls"] == 2
        and res["automatic_pass"]["semantic_state"] == "SHOOTER_CONFIRMATION_REQUIRED"
        and res["automatic_pass"]["completed_stage_count"] == 10
        and res["confirmed_pass"]["semantic_state"] == "COMPLETE"
        and res["confirmed_pass"]["completed_stage_count"] == 18
        and res["confirmed_pass"]["all_public_components_success"] is True
        and res["safety"]["public_api_calls_exactly_two"] is True
        and res["confirmed_pass"]["geometry"]["runtime_fingerprint_sha256"] == cert["runtime_fingerprint_sha256"]
    )
    reg = res["regression"]
    categorical_ok = not reg["categorical_mismatches"]
    numeric_class = reg["classification"]
    numeric_ok = numeric_class in ("STRICT_L40_TOLERANCES", "NUMERICALLY_CLOSE")
    cell40_local_pass = structural and categorical_ok and numeric_ok

    banner("POST-EXECUTION FROZEN SOURCE + INPUT GUARDS")
    post = frozen_source_state()
    post_ok = post["v1_exact"] and post["v2_exact"] and post["v2_file_set_exact"] and sha256_file(CABO_VIDEO) == EXPECTED_CABO_SHA
    post_ok = post_ok and sha256_file(WORKER) == worker_sha and sha256_file(ADAPTER) == adapter_sha
    print(f"Frozen v1/v2/file-set/Cabo/worker/adapter unchanged : {post_ok}")
    cell40_local_pass = cell40_local_pass and post_ok

    pub = res["confirmed_pass"]["public"]
    geo = res["confirmed_pass"]["geometry"]
    banner("INDEPENDENT CABO SEMANTIC-v2 LOCAL RECERTIFICATION RESULT")
    print(f"Structural E2E contract      : {structural}")
    print(f"Categorical regression       : {'EXACT' if categorical_ok else reg['categorical_mismatches']}")
    print(f"Numeric regression class     : {numeric_class}")
    for k, v in reg["numeric"].items():
        print(f"  {k:22s} observed={v['observed']:.12f} expected={v['expected']:.12f} |d|={v['abs_delta']:.3e} strict={v['within_strict']} close={v['within_close']}")
    print(f"Resolved shooter             : Track {pub['shooter_track_id']} ({res['confirmed_pass']['shooter_v2']['resolution']})")
    print(f"Target goal                  : {pub['target_goal']}")
    print(f"Geometry frame / variant     : {geo['selected_frame_idx']} / {geo['selected_variant']} (kp={geo['keypoint_count']}, lines={geo['line_count']})")
    print(f"Annotated video              : {res['confirmed_pass']['annotated_output']['path']}")
    print(f"Annotated video SHA-256      : {res['confirmed_pass']['annotated_output']['sha256']}")
    print(f"\nCELL40_LOCAL_PASS            : {cell40_local_pass}")

    audit = {
        "cell": "NOTEBOOK50_CELL40_LOCAL_RTX2060",
        "certification_kind": "LOCAL_RTX2060_RECERTIFICATION",
        "not_a_claim_of": "ORIGINAL_NVIDIA_L40_SERVER_CERTIFICATION",
        "CELL40_LOCAL_PASS": cell40_local_pass,
        "structural_contract_passed": structural,
        "categorical_regression_exact": categorical_ok,
        "numeric_regression_classification": numeric_class,
        "numeric_regression": reg["numeric"],
        "categorical_mismatches": reg["categorical_mismatches"],
        "preflight": {"cell39": c39, "frozen_source_pre": {k: pre[k] for k in ("v1_exact", "v2_exact", "v2_file_set_exact")},
                      "cabo_sha256": cabo_sha, "local_geometry_certification": cert_ok},
        "frozen_v2_pinned_hashes": FROZEN_V2,
        "frozen_v2_deviation_from_cell39_baseline": {"files_changed": sorted(FROZEN_V2_DEVIATIONS_FROM_CELL39),
                                                      "fixes": FROZEN_V2_DEVIATIONS_FROM_CELL39},
        "local_geometry_certification": {
            "path": str(LOCAL_CERT), "sha256": sha256_file(LOCAL_CERT),
            "runtime_fingerprint_sha256": cert["runtime_fingerprint_sha256"],
            "worker_sha256": worker_sha, "adapter_sha256": adapter_sha,
        },
        "child": {"path": str(CHILD_SCRIPT), "sha256": child_sha, "returncode": run.returncode,
                  "elapsed_seconds": elapsed_child, "python_executable": sys.executable,
                  "result_path": str(CHILD_RESULT), "result_sha256": sha256_file(CHILD_RESULT),
                  "stdout_log": str(CHILD_STDOUT), "stderr_log": str(CHILD_STDERR), "pipeline_log": str(PIPELINE_LOG)},
        "automatic_pass": {k: res["automatic_pass"][k] for k in ("semantic_state", "completed_stage_count", "blocked_stage",
                                                                "auto_status", "resolution", "candidate_track_ids",
                                                                "auto_top_track_id", "auto_second_track_id")},
        "confirmed_pass": {
            "semantic_state": res["confirmed_pass"]["semantic_state"],
            "completed_stage_count": res["confirmed_pass"]["completed_stage_count"],
            "public": pub, "geometry": geo, "shooter_v2": res["confirmed_pass"]["shooter_v2"],
            "annotated_output": res["confirmed_pass"]["annotated_output"],
            "output_validation": res["confirmed_pass"]["output_validation"],
        },
        "geometry_runtime": {k: res["geometry_runtime"][k] for k in ("provider", "runtime_profile", "runtime_fingerprint_sha256",
                                                                    "gpu_name", "worker_sha256", "adapter_call_count")},
        "post_execution_guards_passed": post_ok,
        "safety": res["safety"],
        "elapsed_seconds": round(time.time() - started, 1),
    }
    CELL40_AUDIT.write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
    print(f"\nWrote audit: {CELL40_AUDIT}")
    if not cell40_local_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
