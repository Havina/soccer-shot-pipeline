"""
NOTEBOOK 50 -- CELL 41 (LOCAL RTX-2060 VARIANT)
LOCAL SEMANTIC-v2 APPLICATION-READINESS AUDIT AND FREEZE

This cell does NOT run inference. It audits every artifact the application
layer will depend on, verifies that the certified local Cell-40 run is intact,
probes the local geometry runtime readiness (no frame inference), and writes a
NEW freeze:

    outputs/notebook50_local/freeze/notebook50_local_semantic_v2_app_readiness_freeze.json
    outputs/notebook50_local/freeze/README_NOTEBOOK50_LOCAL_SEMANTIC_V2_APP_READINESS_FREEZE.txt

Existing freeze artifacts are never overwritten. L40 server certification and
local RTX-2060 recertification are reported as separate facts.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PKG = ROOT / "src" / "soccer_ml"
V2 = PKG / "v2"
FREEZE_ROOT = ROOT / "outputs" / "notebook50_local" / "freeze"
AUDIT_ROOT = ROOT / "outputs" / "notebook50_local" / "v2_implementation"
E2E_ROOT = ROOT / "outputs" / "notebook50_local" / "e2e"
LOCAL_GEO_ROOT = ROOT / "outputs" / "notebook50_local" / "local_geometry_v2"

CELL39_FREEZE = FREEZE_ROOT / "notebook50_semantic_v2_public_api_initial_freeze.json"
CELL39_README = FREEZE_ROOT / "README_NOTEBOOK50_SEMANTIC_V2_PUBLIC_API_INITIAL_FREEZE.txt"
CELL39_AUDIT = AUDIT_ROOT / "notebook50_cell39_semantic_v2_public_api_adversarial_tests.json"
CELL40_LOCAL_AUDIT = AUDIT_ROOT / "notebook50_cell40_local_semantic_v2_real_e2e_regression.json"
CELL40_LOCAL_CHILD = E2E_ROOT / "cell40_local_semantic_v2_e2e_child.py"
CELL40_LOCAL_RESULT = E2E_ROOT / "cell40_local_semantic_v2_e2e_result.json"
CELL40_LOCAL_VIDEO = E2E_ROOT / "caboverde_semantic_v2_local_annotated.mp4"
LOCAL_CERT = LOCAL_GEO_ROOT / "local_geometry_v2_certification.json"
V1_ADAPTER_FIX_RECORD = AUDIT_ROOT / "notebook50_cell40_local_v1_adapter_return_contract_fix.json"
STAGE9_FIX_RECORD = AUDIT_ROOT / "notebook50_cell40_local_stage9_inverse_consistency_fix.json"
SUPERSEDED_DIR = AUDIT_ROOT / "superseded_sources"
CELL40_DRIVER = ROOT / "local_geometry_v2" / "run_cell40_local_recertification.py"
WORKER = ROOT / "local_geometry_v2" / "local_geometry_v2_worker.py"
ADAPTER = PKG / "local_geometry_v2.py"
GEOMETRY_VENV_PYTHON = ROOT / ".venv-geometry-v2" / "Scripts" / "python.exe"

OUT_FREEZE = FREEZE_ROOT / "notebook50_local_semantic_v2_app_readiness_freeze.json"
OUT_README = FREEZE_ROOT / "README_NOTEBOOK50_LOCAL_SEMANTIC_V2_APP_READINESS_FREEZE.txt"

EXPECTED_CELL39_AUDIT_SHA = "3339a9e409cf82906a7111129f0d0e6d1be7db27f844612c041011498bb888ee"
EXPECTED_CELL39_FREEZE_SHA = "7b5abe5afdc87b73361660b61d2781d01931918ad4ab46a6096858bae01991d5"
EXPECTED_CELL39_README_SHA = "e2fee2e74b88694f96cc2cec6f8011d3e748678900760e75842b13b03e7092a9"
EXPECTED_CABO_SHA = "204337be5b5b5e381c07deaff70492b2885bc769d14997d68b1d8111e13dc840"
HISTORICAL_L40_RUNTIME_FINGERPRINT = "715f9b64b964b9a2b8d8011d231d132819ccff540563aa34e9284ae94c976d5e"

# Frozen production v1 (unchanged since Notebook 43 / Cell 39)
FROZEN_V1 = {
    "api.py": "479fe7726c9049d9e126a780104aaa65fc67eb69a41d1ff4e19dbddf378a50db",
    "clip_pipeline.py": "913dee8e35a0612d3d6b9ad5465689ac682d99f7d485710a0ce23bd69396f880",
    "results.py": "883fae0407008e7de04189cc0332e8e836213fd2f87ac24a814db958e0466db5",
    "shot_pipeline.py": "0d205673398baaa4c3c20c02e3dcbbe2fa4409f093f3a3723fd258f7278e75f5",
}

# Cell-39 frozen Semantic-v2 baseline (historical, L40 era)
FROZEN_V2_CELL39_BASELINE = {
    "__init__.py": "e183bdad0ac692d637b0c18a028cecd99b291f3cffe818ce06e44813b28139f1",
    "contracts.py": "9311b5f78b778792ab6e5321d5bb2562c7e8e25cb18f0ff523e225f2cbae998f",
    "geometry_client.py": "d3d34dca650de08790308a110c8dd57dddc7543143ff420c232fb53bc5295e19",
    "semantic_contracts.py": "58bcbe5d91a7b9a4ddfd65290673e0ebe53b20174f9c058a6d0c13b19510f322",
    "orchestration_contracts.py": "966f27b1784ce366a3357c3dc5ba091c760fc31ac6b2cabeeeb17a644e58056a",
    "v1_adapter.py": "37b9a0dee6cd38642771eaf8f073bfa08ad4ec41bff3207bbf2e1ff960610ccc",
    "geometry_stage_handlers.py": "ea44ae6a5a04886938ec0dac6f66c5cf6afe90e1d1bb333854aaede724e13f1c",
    "geometry_stage8_selection.py": "83f2e9bfe3f8cb7d4767a3d46e456c772ac17a8a69228f3eb600f080310a621e",
    "geometry_stage9_validation.py": "a0d5bae8294782ae3bbbb17ce932ba85218e41702aa4dae976a20c867bc4f023",
    "geometry_stage10_projection.py": "409746fc0ee94c6b1e6ef3ebb21ed6aa633a80678afe76852c998493171ae49e",
    "shooter_stage11_assignment.py": "922540ae5e5f177d3e251012099f3ca1d9896e894a13ad48d52226d76f3d96fe",
    "target_goal_stage13_inference.py": "451cea1a1380352535a7b9f23feb413e4332096de99b56b80186b0e1b568b9bd",
    "resolved_downstream_reuse.py": "9aeffc3d1a02a0eccc59ceea32efd42ed964a1df7e49d123521ef93dcfad3719",
    "terminal_stage_reuse.py": "caf251077be52643b4be5ee28422c7342e29a79827a95d0ac46f9457c7a14b96",
    "semantic_v2_pipeline.py": "73f29f97e6d1dd2133a7c2d8e935961af385db13efaa3970ea9de91582994b9e",
    "semantic_v2_execution.py": "48345ebc6e607c096b1002b578c9f61ac3ee25c532bdb8398d4956861e85f9fa",
    "api.py": "f06e0d74d8c3126d98a960b4059dee9d7b6c7cf3913d3d88c18802795a69e123",
}

# Semantic-v2 r2: identical to the Cell-39 baseline except two documented defect fixes.
FROZEN_V2_R2 = dict(FROZEN_V2_CELL39_BASELINE)
FROZEN_V2_R2["v1_adapter.py"] = "5505c35f8c0e92241fabd9cfa8da82b8bea608aad922bc88b00fa64ff66856b0"
FROZEN_V2_R2["geometry_stage9_validation.py"] = "5e57319bc0712800c1af8daa46dc860c9aeacfe4eb6f63f37d8c6974aa18d3f7"
R2_DEVIATIONS = {
    "v1_adapter.py": {"record": V1_ADAPTER_FIX_RECORD, "preserved": SUPERSEDED_DIR / "v1_adapter.py.cell39_frozen_37b9a0de.txt"},
    "geometry_stage9_validation.py": {"record": STAGE9_FIX_RECORD, "preserved": SUPERSEDED_DIR / "geometry_stage9_validation.py.cell39_frozen_a0d5bae8.txt"},
}

EXPECTED_PUBLIC_API_PARAMS = ["video_path", "annotated_output_path", "geometry_inference_callable", "frame_loader",
                              "png_encoder", "video_sha256_provider", "shooter_confirmation_track_id"]


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


def show(checks):
    for k, v in checks.items():
        print(f"{k:48s}: {v}")


def main():
    print("=" * 100)
    print("NOTEBOOK 50 -- CELL 41 (LOCAL RTX-2060 VARIANT)")
    print("LOCAL SEMANTIC-v2 APPLICATION-READINESS AUDIT AND FREEZE")
    print("=" * 100)
    started = time.time()

    if OUT_FREEZE.exists() or OUT_README.exists():
        raise RuntimeError("Local app-readiness freeze already exists. Refusing to overwrite a freeze artifact.")

    # ------------------------------------------------------------------ 1. frozen sources
    banner("1. FROZEN SOURCE IDENTITY")
    v1_state = {f: sha256_file(PKG / f) for f in FROZEN_V1}
    v2_files = sorted(p.name for p in V2.glob("*.py"))
    v2_state = {f: sha256_file(V2 / f) for f in v2_files}
    src = {
        "frozen_v1_exact": all(v1_state[f] == s for f, s in FROZEN_V1.items()),
        "v2_file_set_exact": v2_files == sorted(FROZEN_V2_R2),
        "frozen_v2_r2_exact": all(v2_state.get(f) == s for f, s in FROZEN_V2_R2.items()),
        "v2_files_matching_cell39_baseline": sum(1 for f, s in FROZEN_V2_CELL39_BASELINE.items() if v2_state.get(f) == s),
        "v2_files_total": len(FROZEN_V2_CELL39_BASELINE),
    }
    deviations = {}
    for fname, meta in R2_DEVIATIONS.items():
        rec = json.loads(meta["record"].read_text(encoding="utf-8")) if meta["record"].exists() else {}
        deviations[fname] = {
            "cell39_frozen_sha256": FROZEN_V2_CELL39_BASELINE[fname],
            "r2_sha256": FROZEN_V2_R2[fname],
            "current_sha256": v2_state.get(fname),
            "current_is_r2": v2_state.get(fname) == FROZEN_V2_R2[fname],
            "fix_record": str(meta["record"]),
            "fix_record_exists": meta["record"].exists(),
            "fix_record_sha256": sha256_file(meta["record"]) if meta["record"].exists() else None,
            "fix_record_after_sha_matches_r2": rec.get("sha256_after_r2") == FROZEN_V2_R2[fname],
            "preserved_original": str(meta["preserved"]),
            "preserved_original_byte_exact_to_cell39": meta["preserved"].exists() and sha256_file(meta["preserved"]) == FROZEN_V2_CELL39_BASELINE[fname],
            "hardware_dependent": rec.get("symptom", {}).get("hardware_dependent"),
            "would_also_fail_on_l40": rec.get("symptom", {}).get("would_also_fail_on_l40"),
        }
    src["r2_deviations_fully_documented"] = all(
        d["current_is_r2"] and d["fix_record_exists"] and d["fix_record_after_sha_matches_r2"] and d["preserved_original_byte_exact_to_cell39"]
        for d in deviations.values()
    )
    show(src)
    for fname, d in deviations.items():
        print(f"  deviation {fname:32s} {d['cell39_frozen_sha256'][:12]}... -> {d['r2_sha256'][:12]}...  documented={d['fix_record_exists'] and d['preserved_original_byte_exact_to_cell39']}")

    # ------------------------------------------------------------------ 2. Cell-39 freeze intact
    banner("2. CELL-39 PUBLIC-v2 API FREEZE (L40-ERA, MUST BE INTACT)")
    c39 = {
        "cell39_audit_exact": CELL39_AUDIT.exists() and sha256_file(CELL39_AUDIT) == EXPECTED_CELL39_AUDIT_SHA,
        "cell39_freeze_exact": CELL39_FREEZE.exists() and sha256_file(CELL39_FREEZE) == EXPECTED_CELL39_FREEZE_SHA,
        "cell39_readme_exact": CELL39_README.exists() and sha256_file(CELL39_README) == EXPECTED_CELL39_README_SHA,
        "existing_freeze_artifact_count": len(list(FREEZE_ROOT.glob("*"))),
    }
    show(c39)

    # ------------------------------------------------------------------ 3. local geometry certification
    banner("3. LOCAL GEOMETRY-v2 CERTIFICATION (RTX 2060)")
    cert = json.loads(LOCAL_CERT.read_text(encoding="utf-8"))
    ids = cert["identities"]
    geo = {
        "certification_exists": True,
        "certification_sha256": sha256_file(LOCAL_CERT),
        "LOCAL_GEOMETRY_V2_CERTIFIED": cert.get("LOCAL_GEOMETRY_V2_CERTIFIED") is True,
        "runtime_fingerprint_sha256": cert.get("runtime_fingerprint_sha256"),
        "fingerprint_is_not_historical_l40": cert.get("runtime_fingerprint_sha256") != HISTORICAL_L40_RUNTIME_FINGERPRINT,
        "worker_sha_matches_cert": sha256_file(WORKER) == ids["worker_sha256"],
        "adapter_sha_matches_cert": sha256_file(ADAPTER) == ids["adapter_sha256"],
        "policy_file_sha_matches_cert": Path(ids["policy_path"]).exists() and sha256_file(ids["policy_path"]) == ids["policy_file_sha256"],
        "stored_predictions_sha_matches_cert": Path(ids["stored_predictions_path"]).exists() and sha256_file(ids["stored_predictions_path"]) == ids["stored_predictions_sha256"],
        "historical_regression_mismatch_events": cert["historical_regression"]["summary"]["mismatch_events"],
        "historical_regression_byte_exact": cert["historical_regression"]["summary"]["byte_exact_reproduction_achieved"],
        "historical_numerically_equivalent_events": cert["historical_regression"]["summary"]["numerically_equivalent_events"],
        "historical_semantically_equivalent_events": cert["historical_regression"]["summary"]["semantically_equivalent_events"],
        "geometry_venv_python_exists": GEOMETRY_VENV_PYTHON.exists(),
        "cabo_verde_touched_during_certification": cert.get("cabo_verde_touched"),
    }
    show(geo)

    # ------------------------------------------------------------------ 4. live readiness probe (no inference)
    banner("4. LIVE LOCAL RUNTIME READINESS PROBE (NO FRAME INFERENCE)")
    from soccer_ml import local_geometry_v2 as lg
    adapter = lg.LocalGeometryV2Adapter(expected_runtime_fingerprint_sha256=cert["runtime_fingerprint_sha256"])
    t0 = time.time()
    readiness = adapter.get_runtime()
    probe_fp = dict(readiness.runtime_fingerprint).get("fingerprint_sha256")
    probe = {
        "readiness_ready": bool(readiness.ready),
        "readiness_failure_reason": readiness.failure_reason,
        "readiness_protocol_version": readiness.protocol_version,
        "readiness_gpu_name": dict(readiness.runtime_fingerprint).get("gpu", {}).get("name") if isinstance(dict(readiness.runtime_fingerprint).get("gpu"), dict) else None,
        "readiness_fingerprint_sha256": probe_fp,
        "readiness_fingerprint_matches_cert": probe_fp == cert["runtime_fingerprint_sha256"],
        "certified_fingerprint_helper_matches": lg.certified_runtime_fingerprint_sha256() == cert["runtime_fingerprint_sha256"],
        "probe_seconds": round(time.time() - t0, 1),
    }
    show(probe)

    # ------------------------------------------------------------------ 5. Cell-40 local run intact
    banner("5. CELL-40 LOCAL REAL E2E RECERTIFICATION (MUST BE INTACT)")
    a40 = json.loads(CELL40_LOCAL_AUDIT.read_text(encoding="utf-8"))
    child_info = a40["child"]
    c40 = {
        "audit_exists": True,
        "audit_sha256": sha256_file(CELL40_LOCAL_AUDIT),
        "CELL40_LOCAL_PASS": a40.get("CELL40_LOCAL_PASS") is True,
        "structural_contract_passed": a40.get("structural_contract_passed") is True,
        "categorical_regression_exact": a40.get("categorical_regression_exact") is True,
        "numeric_regression_classification": a40.get("numeric_regression_classification"),
        "numeric_regression_strict_l40": a40.get("numeric_regression_classification") == "STRICT_L40_TOLERANCES",
        "child_sha_intact": sha256_file(CELL40_LOCAL_CHILD) == child_info["sha256"],
        "child_result_sha_intact": CELL40_LOCAL_RESULT.exists() and sha256_file(CELL40_LOCAL_RESULT) == child_info["result_sha256"],
        "annotated_video_sha_intact": CELL40_LOCAL_VIDEO.exists() and sha256_file(CELL40_LOCAL_VIDEO) == a40["confirmed_pass"]["annotated_output"]["sha256"],
        "annotated_video_sha256": a40["confirmed_pass"]["annotated_output"]["sha256"],
        "geometry_fingerprint_matches_cert": a40["geometry_runtime"]["runtime_fingerprint_sha256"] == cert["runtime_fingerprint_sha256"],
        "pinned_hashes_match_r2": a40.get("frozen_v2_pinned_hashes") == FROZEN_V2_R2,
        "cabo_sha_exact": a40["preflight"]["cabo_sha256"] == EXPECTED_CABO_SHA,
        "public_api_calls_exactly_two": a40["safety"]["public_api_calls_exactly_two"] is True,
        "auto_state": a40["automatic_pass"]["semantic_state"],
        "auto_candidates": a40["automatic_pass"]["candidate_track_ids"],
        "confirmed_state": a40["confirmed_pass"]["semantic_state"],
        "confirmed_stage_count": a40["confirmed_pass"]["completed_stage_count"],
        "cell40_driver_sha256": sha256_file(CELL40_DRIVER),
    }
    show(c40)

    # ------------------------------------------------------------------ 6. public API surface the app will call
    banner("6. PUBLIC SEMANTIC-v2 API SURFACE")
    from soccer_ml.v2 import api as public_api
    sig = inspect.signature(public_api.analyze_soccer_clip)
    params = list(sig.parameters)
    api = {
        "module": public_api.__name__,
        "function": "analyze_soccer_clip",
        "parameters": params,
        "parameters_exact": params == EXPECTED_PUBLIC_API_PARAMS,
        "runtime_dependencies_keyword_only": all(sig.parameters[p].kind is inspect.Parameter.KEYWORD_ONLY
                                                for p in ("geometry_inference_callable", "frame_loader", "png_encoder",
                                                          "video_sha256_provider", "shooter_confirmation_track_id")),
        "api_sha256_is_cell39_frozen": v2_state.get("api.py") == FROZEN_V2_CELL39_BASELINE["api.py"],
        "adapter_defaults_available": all(hasattr(lg, n) for n in ("default_frame_loader", "default_png_encoder", "default_video_sha256_provider")),
        "adapter_callable_signature_ok": list(inspect.signature(lg.LocalGeometryV2Adapter.__call__).parameters) == ["self", "video_sha256", "frame_idx", "frame_bgr", "frame_png"],
    }
    show(api)

    # ------------------------------------------------------------------ 7. decision
    banner("7. LOCAL APP-READINESS DECISION")
    required = {
        "frozen_v1_exact": src["frozen_v1_exact"],
        "v2_file_set_exact": src["v2_file_set_exact"],
        "frozen_v2_r2_exact": src["frozen_v2_r2_exact"],
        "r2_deviations_fully_documented": src["r2_deviations_fully_documented"],
        "cell39_freeze_intact": c39["cell39_audit_exact"] and c39["cell39_freeze_exact"] and c39["cell39_readme_exact"],
        "local_geometry_certified": geo["LOCAL_GEOMETRY_V2_CERTIFIED"] and geo["worker_sha_matches_cert"] and geo["adapter_sha_matches_cert"]
                                    and geo["policy_file_sha_matches_cert"] and geo["fingerprint_is_not_historical_l40"]
                                    and geo["historical_regression_mismatch_events"] == 0,
        "live_runtime_ready_and_pinned": probe["readiness_ready"] and probe["readiness_fingerprint_matches_cert"] and probe["certified_fingerprint_helper_matches"],
        "cell40_local_pass_intact": c40["CELL40_LOCAL_PASS"] and c40["child_sha_intact"] and c40["child_result_sha_intact"]
                                    and c40["annotated_video_sha_intact"] and c40["geometry_fingerprint_matches_cert"]
                                    and c40["pinned_hashes_match_r2"] and c40["cabo_sha_exact"],
        "two_call_confirmation_flow_proven": c40["auto_state"] == "SHOOTER_CONFIRMATION_REQUIRED" and c40["confirmed_state"] == "COMPLETE" and c40["confirmed_stage_count"] == 18,
        "public_api_surface_exact": api["parameters_exact"] and api["runtime_dependencies_keyword_only"] and api["api_sha256_is_cell39_frozen"]
                                    and api["adapter_defaults_available"] and api["adapter_callable_signature_ok"],
    }
    show(required)
    local_app_ready = all(required.values())
    print(f"\nLOCAL_SEMANTIC_V2_APP_READY : {local_app_ready}")

    freeze = {
        "notebook": 50,
        "cell": "41_LOCAL_RTX2060",
        "freeze_name": "LOCAL_SEMANTIC_V2_APP_READINESS",
        "status": "FROZEN" if local_app_ready else "NOT_READY",
        "LOCAL_SEMANTIC_V2_APP_READY": local_app_ready,
        "certification_kind": "LOCAL_RTX2060_RECERTIFICATION",
        "certification_separation": {
            "l40_server_certification": {
                "scope": "Notebook 50 Cells 1-39: Semantic-v2 contracts, adapters, stage handlers, orchestration, execution driver and public API were "
                         "certified at contract level with synthetic/boundary tests; the Geometry-v2 runtime was certified on an NVIDIA L40 "
                         "(runtime fingerprint 715f9b64...). The original Cell 40 (L40 real E2E) and Cell 41 were NEVER executed.",
                "historical_l40_runtime_fingerprint_sha256": HISTORICAL_L40_RUNTIME_FINGERPRINT,
                "l40_artifacts_modified": False,
                "claims_made_here_about_l40_results": "none",
            },
            "local_rtx2060_recertification": {
                "scope": "New local geometry adapter + worker certified independently (historical Notebook-37 regression: 2 numerically "
                         "equivalent, 2 semantically equivalent, 0 mismatches, byte-exact NOT achieved), then the real two-call Cabo Verde "
                         "E2E through the frozen public API (categorical regression EXACT, numeric regression NUMERICALLY_CLOSE, not within "
                         "strict L40 tolerances).",
                "runtime_fingerprint_sha256": cert["runtime_fingerprint_sha256"],
                "gpu": a40["geometry_runtime"]["gpu_name"],
            },
        },
        "frozen_source_identity": {
            "frozen_v1": {"hashes": FROZEN_V1, "exact": src["frozen_v1_exact"]},
            "frozen_v2_cell39_baseline": FROZEN_V2_CELL39_BASELINE,
            "frozen_v2_r2": FROZEN_V2_R2,
            "frozen_v2_r2_exact": src["frozen_v2_r2_exact"],
            "v2_file_set_exact": src["v2_file_set_exact"],
            "r2_deviations_from_cell39": deviations,
            "r2_deviation_summary": "Two hardware-independent defects in the Cell-39 frozen Semantic-v2 package were exposed by the first real "
                                    "end-to-end run and fixed with minimal, documented edits (originals preserved byte-exact): (1) v1_adapter "
                                    "rejected the non-None return values that 11/18 real frozen-v1 handlers produce; (2) Stage 9 compared raw "
                                    "homography products with the identity although the frozen protocol normalises both matrices to H[2,2]=1.",
        },
        "cell39_freeze": c39,
        "local_geometry_certification": {**geo, "path": str(LOCAL_CERT)},
        "live_runtime_probe": probe,
        "cell40_local": {**c40, "audit_path": str(CELL40_LOCAL_AUDIT), "numeric_regression": a40["numeric_regression"]},
        "public_api": api,
        "application_contract": {
            "entry_point": "soccer_ml.v2.api.analyze_soccer_clip",
            "geometry_inference_callable": "soccer_ml.local_geometry_v2.LocalGeometryV2Adapter(expected_runtime_fingerprint_sha256=<certified>)",
            "frame_loader": "soccer_ml.local_geometry_v2.default_frame_loader",
            "png_encoder": "soccer_ml.local_geometry_v2.default_png_encoder",
            "video_sha256_provider": "soccer_ml.local_geometry_v2.default_video_sha256_provider",
            "flow": [
                "Call 1: shooter_confirmation_track_id=None. If semantic_state == SHOOTER_CONFIRMATION_REQUIRED, surface candidate_track_ids to the user.",
                "Call 2: stateless replay of the same clip with shooter_confirmation_track_id=<user choice> and annotated_output_path set.",
            ],
            "app_startup_must_verify": ["frozen_v1 hashes", "frozen_v2_r2 hashes + exact file set", "local certification worker/adapter hashes",
                                        "live readiness fingerprint == certified fingerprint"],
            "forbidden": ["editing frozen ML sources", "reusing the L40 fingerprint", "feeding expected results as inference inputs",
                          "retrying the public API to force a different shooter"],
        },
        "required_checks": required,
        "execution_safety": {
            "frame_inference_performed": False,
            "pipeline_executed": False,
            "readiness_probe_performed": True,
            "model_weights_loaded_by_probe": True,
            "frozen_sources_modified_by_this_cell": False,
            "existing_freeze_artifacts_overwritten": False,
        },
        "elapsed_seconds": round(time.time() - started, 1),
        "next": "APPLICATION_INTEGRATION: FastAPI backend (app/) calling the frozen public API with the certified local adapter.",
    }
    OUT_FREEZE.write_text(json.dumps(freeze, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")

    readme = f"""NOTEBOOK 50 -- CELL 41 (LOCAL RTX-2060 VARIANT)
LOCAL SEMANTIC-v2 APPLICATION-READINESS FREEZE
{'=' * 78}

LOCAL_SEMANTIC_V2_APP_READY : {local_app_ready}
Certification kind          : LOCAL_RTX2060_RECERTIFICATION (not an NVIDIA L40 certification)
Freeze JSON                 : {OUT_FREEZE.name}
Freeze JSON SHA-256         : {sha256_file(OUT_FREEZE)}

WHAT WAS CERTIFIED WHERE
------------------------
* NVIDIA L40 (Notebook 50 Cells 1-39): contract-level certification of the Semantic-v2
  package and L40 geometry runtime (fingerprint {HISTORICAL_L40_RUNTIME_FINGERPRINT[:16]}...).
  The original Cell 40 / Cell 41 were never run on the L40. Nothing here claims L40 results.
* Local RTX 2060 (this machine):
  - Local Geometry-v2 adapter + worker certified independently
    (runtime fingerprint {cert['runtime_fingerprint_sha256']}).
    Historical Notebook-37 regression: {geo['historical_numerically_equivalent_events']} numerically equivalent,
    {geo['historical_semantically_equivalent_events']} semantically equivalent, {geo['historical_regression_mismatch_events']} mismatches,
    byte-exact reproduction: {geo['historical_regression_byte_exact']}.
  - Cell 40-LOCAL real Cabo Verde E2E through the frozen public API:
    auto -> {c40['auto_state']} (candidates {c40['auto_candidates']}),
    replay with Track 16 -> {c40['confirmed_state']} ({c40['confirmed_stage_count']} stages).
    Categorical regression EXACT; numeric regression {c40['numeric_regression_classification']}
    (distance |d|={a40['numeric_regression']['distance_to_goal_m']['abs_delta']:.3e} m, xG |d|={a40['numeric_regression']['xg_probability']['abs_delta']:.3e}).
    Annotated video SHA-256 {c40['annotated_video_sha256']}.

FROZEN SOURCE STATE
-------------------
* Frozen production v1 : exact ({src['frozen_v1_exact']}).
* Semantic-v2 r2       : {src['v2_files_matching_cell39_baseline']}/{src['v2_files_total']} files byte-identical to the Cell-39 freeze.
  Two documented, hardware-independent defect fixes (originals preserved byte-exact):
"""
    for fname, d in deviations.items():
        readme += f"    - {fname}: {d['cell39_frozen_sha256'][:16]}... -> {d['r2_sha256'][:16]}...\n      record: {Path(d['fix_record']).name}\n"
    readme += f"""
APPLICATION CONTRACT
--------------------
Entry point : soccer_ml.v2.api.analyze_soccer_clip(video_path, annotated_output_path=None, *,
              geometry_inference_callable, frame_loader, png_encoder, video_sha256_provider=None,
              shooter_confirmation_track_id=None)
Geometry    : soccer_ml.local_geometry_v2.LocalGeometryV2Adapter(expected_runtime_fingerprint_sha256=<certified>)
Flow        : call 1 (auto) -> if SHOOTER_CONFIRMATION_REQUIRED show candidates -> call 2 (stateless replay with confirmation).
Startup     : verify frozen v1 hashes, frozen v2 r2 hashes + file set, local certification hashes, live fingerprint.

Written {time.strftime('%Y-%m-%d %H:%M:%S')}. No frame inference or pipeline execution was performed by this cell.
"""
    OUT_README.write_text(readme, encoding="utf-8")
    print(f"\nWrote freeze : {OUT_FREEZE}")
    print(f"Wrote README : {OUT_README}")
    print(f"Freeze SHA   : {sha256_file(OUT_FREEZE)}")
    if not local_app_ready:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
