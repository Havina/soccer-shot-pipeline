NOTEBOOK 50 -- CELL 41 (LOCAL RTX-2060 VARIANT)
LOCAL SEMANTIC-v2 APPLICATION-READINESS FREEZE
==============================================================================

LOCAL_SEMANTIC_V2_APP_READY : True
Certification kind          : LOCAL_RTX2060_RECERTIFICATION (not an NVIDIA L40 certification)
Freeze JSON                 : notebook50_local_semantic_v2_app_readiness_freeze.json
Freeze JSON SHA-256         : 770a366176ab06c39d90e752232d7ca96ed7014fb263e2687dfcc7e4350920b3

WHAT WAS CERTIFIED WHERE
------------------------
* NVIDIA L40 (Notebook 50 Cells 1-39): contract-level certification of the Semantic-v2
  package and L40 geometry runtime (fingerprint 715f9b64b964b9a2...).
  The original Cell 40 / Cell 41 were never run on the L40. Nothing here claims L40 results.
* Local RTX 2060 (this machine):
  - Local Geometry-v2 adapter + worker certified independently
    (runtime fingerprint 64bcb6da6f438e1e47d50639d4f403e5686494043a3112beb808dc9ea92fb559).
    Historical Notebook-37 regression: 2 numerically equivalent,
    2 semantically equivalent, 0 mismatches,
    byte-exact reproduction: False.
  - Cell 40-LOCAL real Cabo Verde E2E through the frozen public API:
    auto -> SHOOTER_CONFIRMATION_REQUIRED (candidates [11, 16]),
    replay with Track 16 -> COMPLETE (18 stages).
    Categorical regression EXACT; numeric regression NUMERICALLY_CLOSE
    (distance |d|=4.692e-03 m, xG |d|=1.956e-05).
    Annotated video SHA-256 1ff8ca6202c84b49f12ad65592a068df91383189f807ca2f0ad132c47fec67e4.

FROZEN SOURCE STATE
-------------------
* Frozen production v1 : exact (True).
* Semantic-v2 r2       : 15/17 files byte-identical to the Cell-39 freeze.
  Two documented, hardware-independent defect fixes (originals preserved byte-exact):
    - v1_adapter.py: 37b9a0dee6cd3864... -> 5505c35f8c0e9224...
      record: notebook50_cell40_local_v1_adapter_return_contract_fix.json
    - geometry_stage9_validation.py: a0d5bae8294782ae... -> 5e57319bc0712800...
      record: notebook50_cell40_local_stage9_inverse_consistency_fix.json

APPLICATION CONTRACT
--------------------
Entry point : soccer_ml.v2.api.analyze_soccer_clip(video_path, annotated_output_path=None, *,
              geometry_inference_callable, frame_loader, png_encoder, video_sha256_provider=None,
              shooter_confirmation_track_id=None)
Geometry    : soccer_ml.local_geometry_v2.LocalGeometryV2Adapter(expected_runtime_fingerprint_sha256=<certified>)
Flow        : call 1 (auto) -> if SHOOTER_CONFIRMATION_REQUIRED show candidates -> call 2 (stateless replay with confirmation).
Startup     : verify frozen v1 hashes, frozen v2 r2 hashes + file set, local certification hashes, live fingerprint.

Written 2026-10-19 22:57:09. No frame inference or pipeline execution was performed by this cell.
