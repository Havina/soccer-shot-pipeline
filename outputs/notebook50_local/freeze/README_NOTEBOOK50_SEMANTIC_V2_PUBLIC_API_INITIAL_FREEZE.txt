NOTEBOOK 50 — PUBLIC SEMANTIC-v2 API INITIAL FREEZE
================================================================================

STATUS
------
FROZEN

FROZEN SOURCE
-------------
src/soccer_ml/v2/api.py

SOURCE SHA-256
--------------
f06e0d74d8c3126d98a960b4059dee9d7b6c7cf3913d3d88c18802795a69e123

PUBLIC ENTRY POINT
------------------
soccer_ml.v2.api.analyze_soccer_clip(...)

RETURN TYPE
-----------
SoccerClipAnalysisV2Result

UPSTREAM EXECUTION DRIVER
-------------------------
src/soccer_ml/v2/semantic_v2_execution.py

EXECUTION DRIVER SHA-256
------------------------
48345ebc6e607c096b1002b578c9f61ac3ee25c532bdb8398d4956861e85f9fa

PUBLIC BOUNDARY
---------------
Exactly one Semantic-v2 execution-driver call per public API call.

The API forwards:

- video_path
- annotated_output_path
- geometry_inference_callable
- frame_loader
- png_encoder
- video_sha256_provider
- shooter_confirmation_track_id

The API does NOT:

- reconstruct the result
- call the frozen-v1 public API
- access stage builders directly
- access stage handlers directly
- mutate orchestration state
- execute runtime dependencies directly
- duplicate shooter-confirmation validation
- hardcode a Geometry service URL

CONFIRMATION
------------
None is preserved.
Track ID 0 is preserved.
Track ID 16 is preserved.
Validation authority remains in the frozen execution driver.

CERTIFICATION
-------------
Adversarial tests passed        : 64/64
All adversarial tests passed    : True
Runtime identities restored     : True
Source guards exact             : True
Public API source unchanged     : True

EXECUTION SAFETY
----------------
Real execution driver invoked   : NO
Real pipeline executed          : NO
Real stage builder invoked      : NO
Real stage handler invoked      : NO
Real video opened               : NO
Model loading performed         : NO
Geometry service called         : NO
Network call performed          : NO
GPU inference performed         : NO
Real xG inference performed     : NO
Annotation rendered             : NO

NEXT
----
Cell 40 — independent real Semantic-v2 end-to-end regression.
