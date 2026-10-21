NOTEBOOK 50 — GEOMETRY-v2 CLIENT INITIAL FREEZE
================================================

STATUS
------
INITIAL_BASELINE_FROZEN


FROZEN CLIENT FILES
-------------------
__init__.py
  e183bdad0ac692d637b0c18a028cecd99b291f3cffe818ce06e44813b28139f1

contracts.py
  9311b5f78b778792ab6e5321d5bb2562c7e8e25cb18f0ff523e225f2cbae998f

geometry_client.py
  d3d34dca650de08790308a110c8dd57dddc7543143ff420c232fb53bc5295e19


SERVER CONTRACT
---------------
Protocol:
  soccer-ml-geometry-v2/1

Geometry frame policy:
  GEOMETRY_FRAME_POLICY_V2_0

Policy fingerprint:
  62e0431373c6a665cbe07561c178cf750dd78ea610b11678644a28d7a0d342fd

Runtime fingerprint:
  715f9b64b964b9a2b8d8011d231d132819ccff540563aa34e9284ae94c976d5e

SV_kp:
  7ea78fa76aaf94976a8eca428d6e3c59697a93430cba1a4603e20284b61f5113

SV_lines:
  d72f4ed71734a2e3df9fa084f666e9b8adaef21bf69bac8952d6d3f970ff7455

Certified GPU:
  NVIDIA L40


CELL-7 CERTIFICATION
--------------------
Adversarial tests:
  44/44 passed

Network calls:
  0

Geometry inference:
  0

Frozen v1 modified:
  NO


IMPORTANT
---------
This freezes the INITIAL Geometry-v2 client boundary.

It does NOT mean the entire soccer_ml.v2 package is complete.

New versioned source files may still be added for semantic-v2 results,
pipeline orchestration, shooter ambiguity handling, target-goal logic,
annotation integration, and the public v2 API.

The three client files frozen here should remain unchanged unless a
deliberate versioned client repair and recertification is performed.

Production network exposure remains NOT CERTIFIED.

The full v2 public API remains NOT IMPLEMENTED / NOT RECERTIFIED.
