NOTEBOOK 50 — SHOOTER-v2 STAGE-11 INITIAL FREEZE

STATUS
------
INITIAL_BASELINE_FROZEN

STAGE
-----
11 shooter_assignment

ACTION
------
REPLACE_V2_AMBIGUITY_SAFE_SHOOTER_GATE

SOURCE
------
C:\Users\hecto\OneDrive\Desktop\soccer-chance-analysis\src\soccer_ml\v2\shooter_stage11_assignment.py

FROZEN SOURCE SHA-256
---------------------
922540ae5e5f177d3e251012099f3ca1d9896e894a13ad48d52226d76f3d96fe

POLICY
------
SHOOTER_V2_TEMPORAL_LAUNCH_ORIGIN_CANDIDATE_V1
version 1.0.0

POLICY SEMANTIC SHA-256
-----------------------
2a87111ea4a01c365067ddf94150169ddc284b762eb3e11c26218ac7b5cf46bc

FROZEN THRESHOLD SEMANTICS
--------------------------
margin < 0.10 : AMBIGUOUS_TOP_TWO / UNRESOLVED
margin = 0.10 : AMBIGUOUS_TOP_TWO / UNRESOLVED
margin > 0.10 : AUTO_SELECTED / AUTO_SELECTION

CONFIRMATION
------------
USER_CONFIRMATION

VALID ADVERSARIAL REPAIR
------------------------
Stage-11 prepopulation guard checks:
semantic_v2_shooter_stage11

CABO REGRESSION
---------------
Candidates           : 18
Top Track            : 11
Top Score            : 0.985294117647059
Second Track         : 16
Second Score         : 0.938235294117647
Relative Margin      : 0.047761194029851
Required Margin      : 0.100000000000000
Automatic Result     : AMBIGUOUS_TOP_TWO / UNRESOLVED
External Confirmation: Track 16 / USER_CONFIRMATION

SAFETY
------
Human truth automatic scoring : NO
Human truth reranking          : NO
Confirmation as model evidence : NO
Team-label scoring             : NO
Target-goal scoring            : NO
xG scoring                     : NO
Reference homography scoring   : NO
Track-specific prior           : NO
New model inference            : NO
Geometry service call          : NO
Frozen-v1 shooter handler      : NO
Real pipeline execution        : NO

CERTIFICATION
-------------
Tests passed           : 151/151
Static source valid     : True
Feature table exact     : True
Frozen v1 exact         : True
Frozen v2 exact         : True
Stage-11 source stable  : True

FREEZE RULE
-----------
Do not modify this Stage-11 production baseline in place.
Any future production change requires an explicit new version
and full recertification.
