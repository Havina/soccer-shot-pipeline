"""
Soccer ML local backend (FastAPI).

Run from the project root with the main interpreter:

    .\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000

then open http://127.0.0.1:8000/ for the web UI (static files in ``app/static``)
or http://127.0.0.1:8000/docs for the raw API.

Flow
----
1. ``POST /analyze`` (multipart ``file``)      -> job created, automatic pass starts
2. ``GET  /jobs/{job_id}``                     -> poll; state becomes
       COMPLETE                      (shooter unambiguous, annotated video ready), or
       AWAITING_SHOOTER_CONFIRMATION (candidates listed under automatic.shooter.candidates)
3. ``POST /jobs/{job_id}/confirm`` {"shooter_track_id": N}
                                               -> stateless replay with the confirmed shooter
4. ``GET  /jobs/{job_id}/annotated.mp4``       -> annotated clip
   ``GET  /jobs/{job_id}/result.json``         -> full frozen-result snapshot(s)

Everything ML-related is delegated to ``app.services.soccer_analysis``; this
module only handles HTTP, uploads and file serving.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app import config
from app.schemas import ConfirmShooterRequest, ErrorResponse, HealthResponse, JobResponse, MatchContext
from app.services.soccer_analysis import IntegrityError, get_service


@asynccontextmanager
async def lifespan(_: FastAPI):
    service = get_service()
    if service.integrity.ok and config.PROBE_RUNTIME_AT_STARTUP:
        service.probe_runtime()
    yield


app = FastAPI(
    title="Soccer ML -- Semantic-v2 local backend",
    version="0.1.0",
    description=(
        "Local RTX-2060 deployment of the frozen Semantic-v2 public API. "
        "Certification kind: LOCAL_RTX2060_RECERTIFICATION (not an NVIDIA L40 certification)."
    ),
    lifespan=lifespan,
)


STATIC_DIR = config.APP_ROOT / "static"


class _NoCacheStaticFiles(StaticFiles):
    """Static assets are tiny and edited often during development; never let the browser cache them."""

    def file_response(self, *args, **kwargs):  # type: ignore[override]
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-store"
        return response


app.mount("/static", _NoCacheStaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def frontend():
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html", headers={"Cache-Control": "no-store"})


def _base_url(request: Request) -> str:
    return str(request.base_url).rstrip("/")


def _job_or_404(job_id: str):
    job = get_service().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job {job_id}")
    return job


@app.exception_handler(IntegrityError)
async def _integrity_handler(_: Request, exc: IntegrityError):
    return JSONResponse(status_code=503, content=ErrorResponse(detail=str(exc)).model_dump())


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    service = get_service()
    if not service.integrity.ok:
        status = "INTEGRITY_FAILURE"
    elif service.runtime.performed and not service.runtime.ready:
        status = "RUNTIME_NOT_READY"
    elif not service.runtime.performed:
        status = "READY_UNPROBED"
    else:
        status = "READY"
    return HealthResponse(
        status=status,
        can_analyze=service.can_analyze,
        integrity=service.integrity,
        runtime=service.runtime,
        active_jobs=service.active_job_count,
    )


@app.post("/health/probe", response_model=HealthResponse)
def probe() -> HealthResponse:
    service = get_service()
    if service.integrity.ok:
        service.probe_runtime()
    return health()


@app.post("/analyze", response_model=JobResponse, status_code=202)
async def analyze(request: Request, file: UploadFile = File(...)) -> JobResponse:
    service = get_service()
    if not service.integrity.ok:
        raise IntegrityError("; ".join(service.integrity.mismatches))
    if service.runtime.performed and not service.runtime.ready:
        raise HTTPException(status_code=503, detail=f"Local geometry runtime not ready: {service.runtime.failure_reason}")

    original_name = Path(file.filename or "upload.mp4").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in config.ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"Unsupported video type {suffix!r}; allowed: {sorted(config.ALLOWED_VIDEO_SUFFIXES)}")

    upload_dir = config.UPLOADS_DIR / Path(original_name).stem[:40]
    upload_dir.mkdir(parents=True, exist_ok=True)
    stored = upload_dir / f"{Path(original_name).stem[:40]}_{len(list(upload_dir.iterdir())):03d}{suffix}"

    written = 0
    with stored.open("wb") as handle:
        while chunk := await file.read(8 * 1024 * 1024):
            written += len(chunk)
            if written > config.MAX_UPLOAD_BYTES:
                handle.close()
                stored.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Upload exceeds the configured size limit.")
            handle.write(chunk)
    if written == 0:
        stored.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Empty upload.")

    job = service.create_job(original_name, stored)
    return job.to_response(_base_url(request))


@app.get("/jobs", response_model=list[JobResponse])
def list_jobs(request: Request) -> list[JobResponse]:
    base = _base_url(request)
    return [job.to_response(base) for job in get_service().list_jobs()]


@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, request: Request) -> JobResponse:
    return _job_or_404(job_id).to_response(_base_url(request))


@app.post("/jobs/{job_id}/confirm", response_model=JobResponse, status_code=202)
def confirm_shooter(job_id: str, body: ConfirmShooterRequest, request: Request) -> JobResponse:
    job = _job_or_404(job_id)
    try:
        get_service().request_confirmation(job, body.shooter_track_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return job.to_response(_base_url(request))


@app.get("/jobs/{job_id}/candidates/preview.jpg")
def candidates_preview(job_id: str, frame: int | None = None):
    """Shot-moment frame with every candidate shooter boxed and labelled (display only)."""
    job = _job_or_404(job_id)
    data = get_service().candidate_preview_jpeg(job, frame)
    if not data:
        raise HTTPException(status_code=404, detail="No candidate preview available for this job.")
    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


@app.get("/jobs/{job_id}/candidates/{track_id}.jpg")
def candidate_crop(job_id: str, track_id: int):
    """Padded crop of one candidate at its estimated contact frame (display only)."""
    job = _job_or_404(job_id)
    data = get_service().candidate_crop_jpeg(job, track_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"No crop available for track {track_id}.")
    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


@app.put("/jobs/{job_id}/context", response_model=JobResponse)
def set_match_context(job_id: str, body: MatchContext, request: Request) -> JobResponse:
    """Store team names/colours for display. Never used for inference."""
    job = _job_or_404(job_id)
    return get_service().set_match_context(job, body).to_response(_base_url(request))


@app.get("/jobs/{job_id}/annotated.mp4")
def annotated_video(job_id: str):
    job = _job_or_404(job_id)
    if job.annotated_path is None or not job.annotated_path.exists():
        raise HTTPException(status_code=404, detail="No annotated video for this job (yet).")
    return FileResponse(job.annotated_path, media_type="video/mp4", filename=f"{job.job_id}_annotated.mp4")


@app.get("/jobs/{job_id}/result.json")
def result_json(job_id: str):
    job = _job_or_404(job_id)
    payload = {"job": job.to_response().model_dump(mode="json")}
    for path in sorted(job.job_dir.glob("*.json")):
        if path.name == "job.json":
            continue
        try:
            payload[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
    return JSONResponse(payload)


@app.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: str):
    job = _job_or_404(job_id)
    try:
        get_service().delete_job(job)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse(status_code=204, content=None)
