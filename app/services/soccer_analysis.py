"""
Analysis service: the only place the application touches the frozen ML stack.

Responsibilities
----------------
* Verify at startup that the frozen production v1 sources, the Semantic-v2 r2
  sources and the local Geometry-v2 certification are byte-identical to what
  the Notebook-50 Cell 41-LOCAL app-readiness freeze pinned. If anything
  differs the service refuses to analyze.
* Hold one locally certified ``LocalGeometryV2Adapter`` pinned to the
  certified RTX-2060 runtime fingerprint.
* Run the certified two-call flow through the frozen public API:
    call 1  -> automatic pass (``shooter_confirmation_track_id=None``)
    call 2  -> stateless replay with the user's confirmed shooter and an
               annotated output path.
* Persist job state and JSON-safe result snapshots under ``app/results``.

The service never edits frozen sources, never reuses historical geometry as
inference input, never retries the public API to change a shooter, and never
claims NVIDIA L40 certification.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app import config
from app.schemas import (
    AnalysisSummary,
    ClipPreprocess,
    ConfirmationRecord,
    GeometrySummary,
    IntegrityReport,
    JobResponse,
    JobState,
    MatchContext,
    PublicMetrics,
    RuntimeProbe,
    SemanticState,
    ShooterCandidate,
    ShooterDecision,
)
from app.services import candidate_preview


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scalar(value: Any) -> Any:
    return getattr(value, "value", value)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    return repr(value)


def _component_snapshot(component: Any) -> Optional[dict[str, Any]]:
    if component is None:
        return None
    out: dict[str, Any] = {"type": type(component).__name__}
    if dataclasses.is_dataclass(component):
        for f in dataclasses.fields(component):
            value = getattr(component, f.name)
            if f.name in ("detections", "tracks", "projected_players", "trajectory", "track_labels", "feature_values"):
                try:
                    out[f.name + "_count"] = len(value)
                except Exception:
                    out[f.name + "_type"] = type(value).__name__
            else:
                out[f.name] = _jsonable(value)
    else:
        out["repr"] = repr(component)
    return out


def _opt_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def _opt_int(value: Any) -> Optional[int]:
    return None if value is None else int(value)


class _PipelineLogProxy(io.TextIOBase):
    """Long-lived text stream that forwards writes to the current job's log file (or drops them)."""

    def __init__(self) -> None:
        super().__init__()
        self._target: Optional[io.TextIOBase] = None

    @contextlib.contextmanager
    def targeting(self, target: io.TextIOBase):
        previous = self._target
        self._target = target
        try:
            yield
        finally:
            self._target = previous

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        target = self._target
        if target is not None and not target.closed:
            try:
                target.write(s)
            except ValueError:
                pass
        return len(s)

    def flush(self) -> None:
        target = self._target
        if target is not None and not target.closed:
            try:
                target.flush()
            except ValueError:
                pass

    def isatty(self) -> bool:
        return False


_PIPELINE_LOG = _PipelineLogProxy()


def friendly_error(exc: BaseException) -> str:
    """
    Keep the exact frozen-pipeline exception, but prefix the most common
    input-related failures with a plain-language explanation for the UI.
    """
    raw = f"{type(exc).__name__}: {exc}"
    text = str(exc)
    if "at least four canonical ball rows" in text:
        return (
            "The ball could not be tracked for at least four frames after the strike, so the shooter stage "
            "cannot fit the ball's launch trajectory. This usually means the ball is too small/far in a wide "
            "broadcast angle, blurred or hidden, the clip is cut before/after the strike, or there is a replay or "
            "camera cut inside the clip. Try a clip of roughly 3-6 seconds starting about one second before the "
            "shot, from the highest-resolution source you have, with the ball visible in flight.  "
            f"[{raw}]"
        )
    if "tied modal appearance cluster" in text or "modal appearance cluster could not be resolved" in text:
        return (
            "Team assignment failed: jersey colours could not be split cleanly into two teams. At least one tracked "
            "player's appearance was split evenly between colour clusters, so the pipeline refused to guess which team "
            "they belong to. This happens most with striped or two-tone kits, kits similar to the pitch or opponent, and "
            "very short tracking fragments at the edge of the frame. Choosing the other shooter candidate will not change "
            "this; try a tighter trim of the clip or a different clip.  "
            f"[{raw}]"
        )
    if "Stage 11 requires successful ball_track" in text or "Stage 11 requires successful player_tracks" in text:
        return "Ball or player tracking did not succeed on this clip, so the shooter could not be assessed.  " + f"[{raw}]"
    if "no valid geometry" in text.lower() or "GeometryV2" in type(exc).__name__ or "homography" in text.lower():
        return (
            "Pitch calibration failed on this clip: not enough pitch lines/keypoints were visible to recover the "
            "camera geometry. Clips showing the penalty area and goal from a standard broadcast angle work best.  "
            f"[{raw}]"
        )
    return raw


# --------------------------------------------------------------------------------------------------
# integrity
# --------------------------------------------------------------------------------------------------
class IntegrityError(RuntimeError):
    pass


def verify_integrity() -> tuple[IntegrityReport, dict[str, Any]]:
    """
    Compare the on-disk frozen sources and local certification against the
    Cell 41-LOCAL app-readiness freeze. Returns the report and the freeze.
    """
    mismatches: list[str] = []
    freeze_path = config.APP_READINESS_FREEZE
    if not freeze_path.exists():
        report = IntegrityReport(
            frozen_v1_exact=False, frozen_v2_r2_exact=False, v2_file_set_exact=False,
            local_geometry_certification_exact=False, app_readiness_freeze_ready=False,
            mismatches=[f"app-readiness freeze missing: {freeze_path}"],
        )
        return report, {}

    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    ready = freeze.get("LOCAL_SEMANTIC_V2_APP_READY") is True
    if not ready:
        mismatches.append("LOCAL_SEMANTIC_V2_APP_READY is not True in the app-readiness freeze")

    identity = freeze.get("frozen_source_identity", {})
    v1_hashes = identity.get("frozen_v1", {}).get("hashes", {})
    v2_hashes = identity.get("frozen_v2_r2", {})

    v1_ok = True
    for name, expected in v1_hashes.items():
        path = config.FROZEN_V1_DIR / name
        actual = sha256_file(path) if path.exists() else None
        if actual != expected:
            v1_ok = False
            mismatches.append(f"frozen v1 {name}: expected {expected[:12]}..., found {str(actual)[:12]}...")

    v2_ok = True
    for name, expected in v2_hashes.items():
        path = config.FROZEN_V2_DIR / name
        actual = sha256_file(path) if path.exists() else None
        if actual != expected:
            v2_ok = False
            mismatches.append(f"frozen v2 r2 {name}: expected {expected[:12]}..., found {str(actual)[:12]}...")

    on_disk = sorted(p.name for p in config.FROZEN_V2_DIR.glob("*.py"))
    file_set_ok = on_disk == sorted(v2_hashes)
    if not file_set_ok:
        mismatches.append(f"v2 file set differs from freeze: {sorted(set(on_disk) ^ set(v2_hashes))}")

    cert_section = freeze.get("local_geometry_certification", {})
    cert_path = Path(cert_section.get("path", ""))
    cert_ok = cert_path.exists() and sha256_file(cert_path) == cert_section.get("certification_sha256")
    if not cert_ok:
        mismatches.append("local Geometry-v2 certification JSON differs from the freeze")
    else:
        cert = json.loads(cert_path.read_text(encoding="utf-8"))
        ids = cert.get("identities", {})
        for label, key_path, key_sha in (
            ("worker", "worker_path", "worker_sha256"),
            ("adapter", "adapter_path", "adapter_sha256"),
            ("policy", "policy_path", "policy_file_sha256"),
        ):
            p = Path(ids.get(key_path, ""))
            if not (p.exists() and sha256_file(p) == ids.get(key_sha)):
                cert_ok = False
                mismatches.append(f"local geometry {label} differs from certification")

    report = IntegrityReport(
        frozen_v1_exact=v1_ok,
        frozen_v2_r2_exact=v2_ok,
        v2_file_set_exact=file_set_ok,
        local_geometry_certification_exact=cert_ok,
        app_readiness_freeze_sha256=sha256_file(freeze_path),
        app_readiness_freeze_ready=ready,
        certified_runtime_fingerprint_sha256=cert_section.get("runtime_fingerprint_sha256"),
        mismatches=mismatches,
    )
    return report, freeze


# --------------------------------------------------------------------------------------------------
# result -> schema
# --------------------------------------------------------------------------------------------------
def summarize_result(result: Any, annotated_path: Optional[Path]) -> AnalysisSummary:
    status = result.semantic_v2_status
    decision = result.shooter_v2
    geometry = result.geometry_v2

    auto_scores = {int(k): float(v) for k, v in dict(decision.auto_scores or {}).items()}
    candidates = [
        ShooterCandidate(
            track_id=int(t),
            auto_score=auto_scores.get(int(t)),
            is_auto_top=(decision.auto_top_track_id is not None and int(t) == int(decision.auto_top_track_id)),
        )
        for t in decision.candidate_track_ids
    ]

    fingerprint = dict(geometry.runtime_fingerprint or {})
    components = {
        name: bool(getattr(getattr(result, name, None), "success", False))
        for name in ("ball_track", "player_tracks", "pitch_projection", "shooter", "relative_teams",
                     "target_goal", "shot_geometry", "defender_context", "xg", "annotated_clip")
    }

    metrics = PublicMetrics()
    if components["shooter"]:
        metrics.shooter_track_id = _opt_int(result.shooter.track_id)
    if components["relative_teams"]:
        metrics.shooter_relative_team = result.relative_teams.shooter_team_label
    if components["target_goal"]:
        metrics.target_goal = _scalar(result.target_goal.target_goal)
        metrics.target_goal_confidence = _opt_float(result.target_goal.confidence)
    if components["shot_geometry"]:
        metrics.distance_to_goal_m = _opt_float(result.shot_geometry.distance_to_goal_m)
        metrics.shot_angle_deg = _opt_float(result.shot_geometry.shot_angle_deg)
    if components["defender_context"]:
        metrics.nearest_defender_m = _opt_float(result.defender_context.nearest_defender_m)
        metrics.defenders_within_5m = _opt_int(result.defender_context.defenders_within_5m)
        metrics.defender_track_ids = [int(t) for t in result.defender_context.defender_track_ids]
    if components["xg"]:
        metrics.xg_probability = _opt_float(result.xg.probability)
        metrics.xg_percent = _opt_float(result.xg.percent)

    from app.services.goal_end_overlay import geometry_warnings, pitch_goal_distances

    distances = pitch_goal_distances(getattr(getattr(result, "shooter", None), "pitch_xy", None))
    if distances is not None:
        metrics.distance_to_left_goal_m = distances["LEFT"]
        metrics.distance_to_right_goal_m = distances["RIGHT"]
    metrics.geometry_warnings = geometry_warnings(metrics)

    return AnalysisSummary(
        semantic_state=SemanticState(_scalar(status.state)),
        completed_stage_count=int(status.completed_stage_count),
        completed_stages=[str(s) for s in result.pipeline_status.completed_stages],
        blocked_stage=status.blocked_stage,
        failure_stage=status.failure_stage,
        failure_reason=status.failure_reason,
        shooter=ShooterDecision(
            auto_status=str(_scalar(decision.auto_status)),
            resolution=str(_scalar(decision.resolution)),
            resolved_track_id=_opt_int(decision.resolved_track_id),
            candidates=candidates,
            auto_margin_fraction=_opt_float(decision.auto_margin_fraction),
            required_margin_fraction=float(decision.required_margin_fraction),
            manual_confirmation_used=bool(decision.manual_confirmation_used),
            confirmation_track_id=_opt_int(decision.confirmation_track_id),
            confirmation_used_as_model_evidence=bool(decision.confirmation_used_as_model_evidence),
        ),
        geometry=GeometrySummary(
            success=bool(geometry.success),
            selected_frame_idx=_opt_int(geometry.selected_frame_idx),
            selected_variant=geometry.selected_variant,
            validation_status=str(geometry.validation_status),
            keypoint_count=_opt_int(geometry.keypoint_count),
            line_count=_opt_int(geometry.line_count),
            reference_free=geometry.reference_free,
            runtime_fingerprint_sha256=fingerprint.get("fingerprint_sha256"),
            runtime_profile=fingerprint.get("runtime_profile"),
        ),
        metrics=metrics,
        component_success=components,
        annotated_video_available=bool(annotated_path is not None and annotated_path.exists() and components["annotated_clip"]),
    )


def full_snapshot(result: Any) -> dict[str, Any]:
    return {
        "video_path": str(result.video_path),
        "pipeline_status": _jsonable(result.pipeline_status),
        "semantic_v2_status": _jsonable(result.semantic_v2_status),
        "geometry_v2": _jsonable(result.geometry_v2),
        "shooter_v2": _jsonable(result.shooter_v2),
        "shooter": _component_snapshot(result.shooter),
        "relative_teams": _component_snapshot(result.relative_teams),
        "target_goal": _component_snapshot(result.target_goal),
        "shot_geometry": _component_snapshot(result.shot_geometry),
        "defender_context": _component_snapshot(result.defender_context),
        "xg": _component_snapshot(result.xg),
        "annotated_clip": _component_snapshot(result.annotated_clip),
        "metadata": _jsonable(result.metadata),
    }


# --------------------------------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class Job:
    job_id: str
    original_filename: str
    video_path: Path
    job_dir: Path
    created_at: str = dataclasses.field(default_factory=_now)
    updated_at: str = dataclasses.field(default_factory=_now)
    state: JobState = JobState.QUEUED
    video_sha256: Optional[str] = None
    public_api_calls: int = 0
    error: Optional[str] = None
    automatic: Optional[AnalysisSummary] = None
    confirmed: Optional[AnalysisSummary] = None
    annotated_path: Optional[Path] = None
    shot_frame_idx: Optional[int] = None
    confirmations: list[ConfirmationRecord] = dataclasses.field(default_factory=list)
    match_context: Optional[MatchContext] = None
    preprocess: Optional[ClipPreprocess] = None

    @property
    def candidate_track_ids(self) -> list[int]:
        if self.automatic is None:
            return []
        return [c.track_id for c in self.automatic.shooter.candidates]

    @property
    def can_change_shooter(self) -> bool:
        """A different candidate may be chosen whenever the automatic pass asked for one and nothing is running."""
        return (
            self.automatic is not None
            and self.automatic.semantic_state == SemanticState.SHOOTER_CONFIRMATION_REQUIRED
            and len(self.candidate_track_ids) >= 2
            and self.state in (JobState.AWAITING_SHOOTER_CONFIRMATION, JobState.COMPLETE, JobState.FAILED)
        )

    def to_response(self, base_url: str = "") -> JobResponse:
        links = {"self": f"{base_url}/jobs/{self.job_id}", "result_json": f"{base_url}/jobs/{self.job_id}/result.json"}
        if self.can_change_shooter:
            links["confirm"] = f"{base_url}/jobs/{self.job_id}/confirm"
            links["candidates_preview"] = f"{base_url}/jobs/{self.job_id}/candidates/preview.jpg"
        if self.annotated_path is not None and self.annotated_path.exists():
            links["annotated_video"] = f"{base_url}/jobs/{self.job_id}/annotated.mp4"
        automatic = self.automatic
        if automatic is not None and self.can_change_shooter:
            automatic = automatic.model_copy(deep=True)
            for c in automatic.shooter.candidates:
                c.crop_url = f"{base_url}/jobs/{self.job_id}/candidates/{c.track_id}.jpg"
        return JobResponse(
            job_id=self.job_id, state=self.state, created_at=self.created_at, updated_at=self.updated_at,
            original_filename=self.original_filename, video_sha256=self.video_sha256,
            public_api_calls=self.public_api_calls, error=self.error,
            automatic=automatic, confirmed=self.confirmed, shot_frame_idx=self.shot_frame_idx,
            can_change_shooter=self.can_change_shooter, confirmations=list(self.confirmations),
            match_context=self.match_context, preprocess=self.preprocess, links=links,
        )

    def persist(self) -> None:
        self.updated_at = _now()
        payload = self.to_response().model_dump(mode="json")
        payload["video_path"] = str(self.video_path)
        payload["annotated_path"] = str(self.annotated_path) if self.annotated_path else None
        (self.job_dir / "job.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class AnalysisService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._executor = ThreadPoolExecutor(max_workers=config.ANALYSIS_WORKERS, thread_name_prefix="soccer-ml")
        self.integrity, self.freeze = verify_integrity()
        self.runtime = RuntimeProbe(performed=False)
        self._adapter = None
        self._public_api = None
        self._lg = None
        config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self._load_existing_jobs()

    # ---------------------------------------------------------------- readiness
    @property
    def can_analyze(self) -> bool:
        return self.integrity.ok and (not self.runtime.performed or self.runtime.ready)

    def _ensure_stack(self) -> None:
        if not self.integrity.ok:
            raise IntegrityError("Frozen-source integrity check failed: " + "; ".join(self.integrity.mismatches))
        if self._adapter is None:
            import importlib

            self._lg = importlib.import_module("soccer_ml.local_geometry_v2")
            self._public_api = importlib.import_module("soccer_ml.v2.api")
            pinned = self._lg.certified_runtime_fingerprint_sha256()
            if pinned != self.integrity.certified_runtime_fingerprint_sha256:
                raise IntegrityError("Certified fingerprint in certification JSON differs from the app-readiness freeze.")
            certified = self._lg.LocalGeometryV2Adapter(
                expected_runtime_fingerprint_sha256=pinned,
                request_root=config.RESULTS_DIR / "_geometry_requests",
            )
            if config.MULTIFRAME_GEOMETRY:
                from app.services.multiframe_geometry import MultiFrameGeometryAdapter

                self._adapter = MultiFrameGeometryAdapter(
                    certified,
                    frame_loader=self._lg.default_frame_loader,
                    png_encoder=self._lg.default_png_encoder,
                )
            else:
                self._adapter = certified
            if config.TEAM_ASSIGNMENT_V3:
                from app.services import team_assignment_v3

                team_assignment_v3.install()
            if config.GOAL_END_V3:
                from app.services import goal_end_overlay

                goal_end_overlay.install()

    def probe_runtime(self) -> RuntimeProbe:
        try:
            self._ensure_stack()
            readiness = self._adapter.get_runtime()
            fp = dict(readiness.runtime_fingerprint or {})
            gpu = fp.get("gpu") if isinstance(fp.get("gpu"), dict) else {}
            self.runtime = RuntimeProbe(
                performed=True,
                ready=bool(readiness.ready),
                fingerprint_sha256=fp.get("fingerprint_sha256"),
                fingerprint_matches_certification=fp.get("fingerprint_sha256") == self.integrity.certified_runtime_fingerprint_sha256,
                gpu_name=gpu.get("name") or fp.get("gpu_name"),
                failure_reason=readiness.failure_reason,
            )
        except Exception as exc:  # readiness failures are reported, not raised
            self.runtime = RuntimeProbe(performed=True, ready=False, failure_reason=f"{type(exc).__name__}: {exc}")
        return self.runtime

    # ---------------------------------------------------------------- job store
    def _load_existing_jobs(self) -> None:
        for job_file in config.RESULTS_DIR.glob("*/job.json"):
            try:
                payload = json.loads(job_file.read_text(encoding="utf-8"))
                job = Job(
                    job_id=payload["job_id"], original_filename=payload["original_filename"],
                    video_path=Path(payload["video_path"]), job_dir=job_file.parent,
                    created_at=payload["created_at"], updated_at=payload["updated_at"],
                    state=JobState(payload["state"]), video_sha256=payload.get("video_sha256"),
                    public_api_calls=int(payload.get("public_api_calls", 0)), error=payload.get("error"),
                    automatic=AnalysisSummary(**payload["automatic"]) if payload.get("automatic") else None,
                    confirmed=AnalysisSummary(**payload["confirmed"]) if payload.get("confirmed") else None,
                    annotated_path=Path(payload["annotated_path"]) if payload.get("annotated_path") else None,
                    shot_frame_idx=payload.get("shot_frame_idx"),
                    confirmations=[ConfirmationRecord(**c) for c in payload.get("confirmations", [])],
                    match_context=MatchContext(**payload["match_context"]) if payload.get("match_context") else None,
                    preprocess=ClipPreprocess(**payload["preprocess"]) if payload.get("preprocess") else None,
                )
                if job.state in (JobState.QUEUED, JobState.RUNNING, JobState.CONFIRMING):
                    job.state = JobState.FAILED
                    job.error = "Server restarted while the job was in progress."
                self._jobs[job.job_id] = job
            except Exception:
                continue

    def list_jobs(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    @property
    def active_job_count(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.state in (JobState.QUEUED, JobState.RUNNING, JobState.CONFIRMING))

    def create_job(self, original_filename: str, stored_video: Path) -> Job:
        job_id = uuid.uuid4().hex[:16]
        job_dir = config.RESULTS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        job = Job(job_id=job_id, original_filename=original_filename, video_path=stored_video, job_dir=job_dir)
        job.video_sha256 = sha256_file(stored_video)
        with self._lock:
            self._jobs[job_id] = job
        job.persist()
        self._executor.submit(self._run_automatic, job)
        return job

    def delete_job(self, job: Job) -> None:
        if job.state in (JobState.QUEUED, JobState.RUNNING, JobState.CONFIRMING):
            raise ValueError("Cannot delete a job while it is running.")
        with self._lock:
            self._jobs.pop(job.job_id, None)
        import shutil

        shutil.rmtree(job.job_dir, ignore_errors=True)

    def request_confirmation(self, job: Job, shooter_track_id: int) -> Job:
        if not job.can_change_shooter:
            raise ValueError(
                f"Job is in state {job.state.value} and cannot accept a shooter confirmation "
                "(either it is still running or the automatic pass never asked for one)."
            )
        if shooter_track_id not in job.candidate_track_ids:
            raise ValueError(f"Track {shooter_track_id} is not one of the automatic candidates {job.candidate_track_ids}.")
        job.state = JobState.CONFIRMING
        job.error = None
        job.persist()
        self._executor.submit(self._run_confirmed, job, shooter_track_id)
        return job

    def set_match_context(self, job: Job, context: MatchContext) -> Job:
        job.match_context = context
        job.persist()
        return job

    # ---------------------------------------------------------------- candidate previews
    def candidate_evidence(self, job: Job) -> Optional[dict[str, Any]]:
        return candidate_preview.load_evidence(job.job_dir)

    def candidate_preview_jpeg(self, job: Job, frame_idx: Optional[int] = None) -> Optional[bytes]:
        evidence = self.candidate_evidence(job)
        if evidence is None:
            return None
        cache = job.job_dir / "preview" / f"frame_{evidence.get('shot_frame_idx') if frame_idx is None else frame_idx}.jpg"
        if cache.exists():
            return cache.read_bytes()
        data = candidate_preview.render_preview(job.video_path, evidence, frame_idx)
        if data:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(data)
        return data

    def candidate_crop_jpeg(self, job: Job, track_id: int) -> Optional[bytes]:
        evidence = self.candidate_evidence(job)
        if evidence is None or track_id not in evidence.get("candidate_track_ids", []):
            return None
        cache = job.job_dir / "preview" / f"track_{track_id}.jpg"
        if cache.exists():
            return cache.read_bytes()
        data = candidate_preview.render_crop(job.video_path, evidence, track_id)
        if data:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(data)
        return data

    # ---------------------------------------------------------------- pipeline calls
    def _call_public_api(self, job: Job, *, shooter_track_id: Optional[int], annotated_output: Optional[Path]):
        self._ensure_stack()
        log_path = job.job_dir / ("pipeline_confirmed.log" if shooter_track_id is not None else "pipeline_automatic.log")
        import contextlib

        # Libraries such as ultralytics attach a logging.StreamHandler to whatever sys.stdout/stderr
        # is at import time. If that import happens during the first job, a per-job file would be
        # captured and later closed, producing "I/O operation on closed file" noise in every later
        # job. _PIPELINE_LOG is a single long-lived proxy whose target is swapped per job instead.
        bind = getattr(self._adapter, "bind_video", None)
        if callable(bind):
            bind(job.video_path)
        with log_path.open("w", encoding="utf-8") as log, _PIPELINE_LOG.targeting(log), \
                contextlib.redirect_stdout(_PIPELINE_LOG), contextlib.redirect_stderr(_PIPELINE_LOG):
            job.public_api_calls += 1
            return self._public_api.analyze_soccer_clip(
                str(job.video_path),
                annotated_output_path=str(annotated_output) if annotated_output else None,
                geometry_inference_callable=self._adapter,
                frame_loader=self._lg.default_frame_loader,
                png_encoder=self._lg.default_png_encoder,
                video_sha256_provider=self._lg.default_video_sha256_provider,
                shooter_confirmation_track_id=shooter_track_id,
            )

    def _active_overlays(self) -> list[str]:
        names = []
        if config.AUTO_TRIM:
            names.append("auto_trim")
        if config.MULTIFRAME_GEOMETRY:
            names.append("multiframe_geometry")
        if config.TEAM_ASSIGNMENT_V3:
            names.append("team_assignment_v3")
        if config.GOAL_END_V3:
            names.append("goal_end_v3")
        return names

    def _attach_overlay_diagnostics(self, job: Job) -> None:
        if job.preprocess is None:
            job.preprocess = ClipPreprocess(overlays=self._active_overlays())
        else:
            job.preprocess.overlays = self._active_overlays()
        fallback = getattr(self._adapter, "last_diagnostics", None) if self._adapter is not None else None
        if isinstance(fallback, dict):
            job.preprocess.geometry_fallback = fallback
        if config.TEAM_ASSIGNMENT_V3:
            from app.services import team_assignment_v3

            job.preprocess.team_assignment_v3 = dict(team_assignment_v3.last_diagnostics)
        if config.GOAL_END_V3:
            from app.services import goal_end_overlay

            job.preprocess.goal_end = dict(goal_end_overlay.last_diagnostics)

    def _run_automatic(self, job: Job) -> None:
        job.state = JobState.RUNNING
        job.persist()
        try:
            from app.services import clip_preprocess

            info = clip_preprocess.maybe_trim(job.video_path, job.job_dir / "preprocessed", enabled=config.AUTO_TRIM)
            if info.get("trimmed"):
                job.video_path = Path(info["path"])
                job.video_sha256 = sha256_file(job.video_path)
            job.preprocess = ClipPreprocess(
                width=info.get("width"), height=info.get("height"), fps=info.get("fps"),
                frame_count=info.get("frame_count"), duration_s=info.get("duration_s"),
                cut_frames=list(info.get("original_cut_frames") or info.get("cut_frames") or []),
                trimmed=bool(info.get("trimmed")),
                trim_start_frame=info.get("trim_start_frame"), trim_end_frame=info.get("trim_end_frame"),
                original_duration_s=info.get("original_duration_s"),
                green_fraction=info.get("green_fraction"),
                warnings=list(info.get("warnings") or []),
                overlays=self._active_overlays(),
            )
            (job.job_dir / "preprocess.json").write_text(json.dumps(info, indent=2, default=str) + "\n", encoding="utf-8")
            job.persist()
        except Exception as exc:
            job.preprocess = ClipPreprocess(
                overlays=self._active_overlays(),
                warnings=[f"Preflight failed ({type(exc).__name__}: {exc}); running the original clip."],
            )
            job.persist()
        # The automatic pass carries its own annotated output path. If the shooter is
        # unambiguous the pipeline reaches stage 17 and renders it in this single call;
        # if confirmation is required the pipeline stops at stage 11 and the file is
        # simply never written. No synthetic "confirmation" is ever injected by the app.
        annotated = job.job_dir / "annotated_automatic.mp4"
        try:
            result = self._call_public_api(job, shooter_track_id=None, annotated_output=annotated)
            (job.job_dir / "automatic_result.json").write_text(json.dumps(full_snapshot(result), indent=2, default=str) + "\n", encoding="utf-8")
            job.automatic = summarize_result(result, annotated if annotated.exists() else None)
            state = job.automatic.semantic_state
            if state == SemanticState.SHOOTER_CONFIRMATION_REQUIRED:
                evidence = candidate_preview.extract_candidate_evidence(result)
                if evidence is not None:
                    (job.job_dir / "candidates.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
                    job.shot_frame_idx = evidence.get("shot_frame_idx")
                    for c in job.automatic.shooter.candidates:
                        c.contact_frame_idx = evidence.get("contact_frames", {}).get(str(c.track_id))
                job.state = JobState.AWAITING_SHOOTER_CONFIRMATION
            elif state == SemanticState.COMPLETE:
                job.annotated_path = annotated if annotated.exists() else None
                job.state = JobState.COMPLETE
            else:
                job.state = JobState.FAILED
                job.error = job.automatic.failure_reason or f"Pipeline failed at stage {job.automatic.failure_stage}."
        except Exception as exc:
            job.state = JobState.FAILED
            job.error = friendly_error(exc)
        self._attach_overlay_diagnostics(job)
        job.persist()

    def _run_confirmed(self, job: Job, shooter_track_id: int, source: str = "USER_CONFIRMATION") -> None:
        job.state = JobState.CONFIRMING
        job.persist()
        # One annotated file per confirmed track so changing the shooter never overwrites a prior replay.
        annotated = job.job_dir / f"annotated_confirmed_track{shooter_track_id}.mp4"
        record = ConfirmationRecord(shooter_track_id=shooter_track_id, source=source, at=_now())
        try:
            result = self._call_public_api(job, shooter_track_id=shooter_track_id, annotated_output=annotated)
            (job.job_dir / f"confirmed_result_track{shooter_track_id}.json").write_text(
                json.dumps(full_snapshot(result), indent=2, default=str) + "\n", encoding="utf-8")
            job.annotated_path = annotated if annotated.exists() else None
            job.confirmed = summarize_result(result, job.annotated_path)
            if job.confirmed.semantic_state == SemanticState.COMPLETE:
                job.state = JobState.COMPLETE
                record.outcome = "COMPLETE"
            else:
                job.state = JobState.FAILED
                job.error = job.confirmed.failure_reason or f"Confirmed pass ended in {job.confirmed.semantic_state.value}."
                record.outcome = job.confirmed.semantic_state.value
        except Exception as exc:
            job.state = JobState.FAILED
            job.error = friendly_error(exc)
            record.outcome = f"ERROR: {type(exc).__name__}"
        job.confirmations.append(record)
        (job.job_dir / "confirmations.json").write_text(
            json.dumps([c.model_dump(mode="json") for c in job.confirmations], indent=2) + "\n", encoding="utf-8")
        self._attach_overlay_diagnostics(job)
        job.persist()


_service: Optional[AnalysisService] = None
_service_lock = threading.Lock()


def get_service() -> AnalysisService:
    global _service
    with _service_lock:
        if _service is None:
            _service = AnalysisService()
        return _service
