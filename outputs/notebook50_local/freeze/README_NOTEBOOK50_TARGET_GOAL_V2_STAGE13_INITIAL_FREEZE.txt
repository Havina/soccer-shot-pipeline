NOTEBOOK 50 — TARGET-GOAL-v2 STAGE-13 INITIAL FREEZE

STATUS
------
INITIAL_BASELINE_FROZEN

STAGE
-----
13 target_goal_inference

ACTION
------
REPLACE_V2_IMAGE_SPACE_TARGET_GOAL

SOURCE
------
C:\Users\hecto\OneDrive\Desktop\soccer-chance-analysis\src\soccer_ml\v2\target_goal_stage13_inference.py

FROZEN SOURCE SHA-256
---------------------
451cea1a1380352535a7b9f23feb413e4332096de99b56b80186b0e1b568b9bd

POLICY
------
IMAGE_SPACE_SEMANTIC_MOTION_V2

DECISION SPACE
--------------
IMAGE SPACE

GROUND-PLANE HOMOGRAPHY RULE
----------------------------
Airborne ball projected through homography : NO
Static goal centers projected pitch->image : YES

STATIC GOAL CENTERS
-------------------
LEFT  : (0.0, 34.0)
RIGHT : (105.0, 34.0)

MOTION MODELS
-------------
EARLY_LINEAR  : first 4 ball rows
FULL_ENDPOINT : first-to-last ball displacement

ORIGINS
-------
SHOOTER_FOOT
FIRST_BALL

DECISION RULE
-------------
All 4 comparisons must agree on the same goal.
The winning goal must be in the forward-motion hemisphere.
The opposite goal must be in the backward-motion hemisphere.

CABO REGRESSION
---------------
Target Goal          : LEFT
Confidence           : 1.0
LEFT goal image      : (364.45087545934945, 337.14591419804833)
RIGHT goal image     : (6277.083907344718, 1151.6478318592526)
Early velocity       : (1.026171875000036, -13.609783935546888)
Full endpoint motion : (1.379150390625, -98.92324829101562)
Comparison winners   : ['LEFT', 'LEFT', 'LEFT', 'LEFT']

CELL-28 ADVERSARIAL REPAIR
--------------------------
Production source defect : NO
Notebook test defect     : YES

The initial Cell-28 static test depended on exact whitespace around the
'airborne_ball_projected_through_homography' metadata entry.

The corrected certification uses Python AST and requires exactly one
dictionary entry with that key and literal value False.

The independent runtime metadata test already passed before this repair.

SAFETY
------
Manual target-goal evidence        : NO
Historical target-goal evidence    : NO
Shooter pitch location direction   : NO
Relative-team direction evidence   : NO
Defender direction evidence        : NO
Shot-geometry direction evidence   : NO
xG direction evidence              : NO
Reference homography               : NO
Geometry variant reselection       : NO
New model inference                : NO
Geometry service call              : NO
Frozen-v1 target-goal handler      : NO
Real pipeline execution            : NO

CERTIFICATION
-------------
Tests passed                    : 91/91
Static source valid             : True
AST airborne-ball safety valid  : True
Stage-13 source exact           : True
Frozen v1 exact                 : True
Frozen v2 upstream exact        : True
v2 file set exact               : True

FREEZE RULE
-----------
Do not modify this Stage-13 production baseline in place.
Any future production source change requires an explicit new version
and full recertification.
