NOTEBOOK 50 — GEOMETRY-v2 STAGE-10 INITIAL FREEZE
====================================================

STATUS
------
INITIAL_BASELINE_FROZEN


SOURCE
------
geometry_stage10_projection.py

SHA-256
-------
409746fc0ee94c6b1e6ef3ebb21ed6aa633a80678afe76852c998493171ae49e


STAGE
-----
10 pitch_projection

Action:
REPLACE_V2_PLAYER_PROJECTION


PROJECTION
----------
Homography direction:
IMAGE_TO_PITCH

Pitch dimensions:
105.0 m x 68.0 m

Preferred player point:
FOOT_XY

Compatibility fallback:
BBOX_BOTTOM_CENTER

Detection identifier compatibility:
detection_idx / detection_id

Inside-pitch bounds:
0 <= x <= 105
0 <= y <= 68


SEMANTIC SAFETY
---------------
Stage-9 validated geometry required:
YES

Reference homography:
NO

Historical projected coordinates:
NO

Manual projection:
NO

Shooter identity:
NO

Relative-team assignment:
NO

Variant reselection:
NO

Temporal fallback:
NO

Frozen-v1 projection handler:
NO


EXECUTION
---------
Additional Geometry-v2 inference:
ZERO

Geometry service call:
NO

Network call:
NO

Geometry object replacement:
NO

Orchestration state mutation:
NO


PUBLIC RESULT
-------------
PitchProjectionResult

Validation status:
VALID


ADVERSARIAL VALIDATION
----------------------
Tests passed:
104/104

All tests passed:
True


GEOMETRY-v2 STATUS
------------------
Stages 7, 8, 9, and 10:
INITIAL_BASELINE_FROZEN


NEXT
----
Begin Shooter-v2 ambiguity-safe shooter-assignment boundary.
