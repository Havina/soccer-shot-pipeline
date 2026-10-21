NOTEBOOK 50 — GEOMETRY-v2 STAGE-8 INITIAL FREEZE
===================================================

STATUS
------
INITIAL_BASELINE_FROZEN


SOURCE
------
geometry_stage8_selection.py

SHA-256
-------
83f2e9bfe3f8cb7d4767a3d46e456c772ac17a8a69228f3eb600f080310a621e


STAGE
-----
08 homography_search

Action:
REPLACE_V2_PNLCALIB_BASE_PNL_SELECTION


FROZEN SELECTION POLICY
-----------------------
Policy:
PNLCALIB_PNL_PRIMARY_BASE_FALLBACK_V1

Policy SHA-256:
62e0431373c6a665cbe07561c178cf750dd78ea610b11678644a28d7a0d342fd

Primary:
PNL_REFINED -> PRIMARY_PNL_REFINED

Fallback:
BASE -> FALLBACK_BASE


SELECTION SAFETY
----------------
Reference used for selection:
NO

Reference used for threshold tuning:
NO

Reference-based per-event switching:
NO

Expected-result selection:
NO

Temporal variant switching:
NO


EXECUTION
---------
Additional Geometry-v2 inference:
ZERO

Geometry service call:
NO

Network call:
NO

Frozen-v1 geometry fallback:
NO

Stage-7 GeometryV2Result replaced:
NO

Stage marks completion:
NO


ADVERSARIAL VALIDATION
----------------------
Tests passed:
73/73

All tests passed:
True


NEXT
----
Stage 9 will validate the selected homography reference-free.
It must consume the already-selected Stage-7/Stage-8 geometry and perform
ZERO additional Geometry-v2 inference.
