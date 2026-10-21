# Soccer shot analysis pipeline

Frozen 18-stage soccer clip pipeline: detect and track players and the ball, recover camera-to-pitch geometry, assign a shooter (with a human confirm when the top two tracks are ambiguous), then report distance, angle, nearby defenders, and a geometry-only expected-goals (xG) figure.

Frozen V1/V2 source hashes are checked when the app starts.

## What is in this repo

- `src/soccer_ml/` — frozen V1 clip pipeline and Semantic-v2 package
- `src/soccer_ml/local_geometry_v2.py` — local Geometry-v2 adapter (injected at the public API boundary)
- `local_geometry_v2/*.py` — local geometry worker scripts (the PnLCalib weight tree is not committed)
- `app/` — FastAPI backend and static UI
- `outputs/notebook50_local/freeze/` — Cell 41-LOCAL app-readiness freeze (JSON/txt only)
- `outputs/evaluation_corpus/catalog.json` — notes on clips that pass or fail

Notebooks, datasets, model weights, uploaded videos, and job results stay on the machine that ran them.

## How to run the app (local GPU)

You need an NVIDIA GPU, the main `.venv` (Python 3.13 + torch/CUDA), and `.venv-geometry-v2` (Python 3.10 + PnLCalib). From the repo root:

```powershell
.\.venv\Scripts\pip.exe install -r app\requirements-app.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000

## Clip requirements

Best chance of a complete run:

- 720p or higher, landscape
- about 3–6 seconds, **one** camera shot
- penalty area and goal visible around the strike
- solid, contrasting kits (stripes and green shirts are hard)

The app can auto-trim a celebration cut. It cannot invent a ball that the detector never sees.

## Local overlays (not Cell-40 certified)

These wrap the frozen stack at runtime; they do not edit frozen files:

| Overlay | Role |
| --- | --- |
| `auto_trim` | Drop frames after the first hard camera cut |
| `multiframe_geometry` | Retry PnLCalib on nearby frames if the anchor fails |
| `team_assignment_v3` | Break jersey-colour ties; ignore short tracks |
| `goal_end_v3` | If Stage 13 picks the far pitch end (≥45 m) and the other end is ≤35 m, flip before xG |

Disable any of them with `SOCCER_ML_AUTO_TRIM=0`, `SOCCER_ML_MULTIFRAME_GEOMETRY=0`, `SOCCER_ML_TEAM_ASSIGNMENT_V3=0`, `SOCCER_ML_GOAL_END_V3=0`.

## Honest limits

- xG is **distance + shot angle + nearby defenders**, not a model trained on thousands of shot outcomes.
- LEFT / RIGHT are ends of the 105 m pitch model (`x = 0` / `x = 105`), not “left of the TV”.
- Arbitrary broadcast highlights often fail at ball tracking or pitch calibration. That is a contract of the frozen stages, not a silent retry.
- Do not commit match footage. Broadcast video is copyrighted.

## Public API (frozen)

```python
soccer_ml.v2.api.analyze_soccer_clip(
    video_path,
    geometry_inference_callable=...,
    frame_loader=...,
    png_encoder=...,
    video_sha256_provider=...,
    shooter_confirmation_track_id=...,  # None, then the user's pick
)
```
