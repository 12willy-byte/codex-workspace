from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from aquaguard import __version__
from aquaguard.assembly import ConfiguredFrameAnalyzerFactory, VideoRuntimeAssembler
from aquaguard.audit import SQLiteEvaluationAuditRepository
from aquaguard.config import get_settings
from aquaguard.domain import EventStatus, RiskFeatures
from aquaguard.evidence import EvidenceRecorder, EventEvidenceService, FileEvidenceRepository
from aquaguard.protection import ValidatedProtectionAlarmGate
from aquaguard.risk import RiskEngine
from aquaguard.runtime import VideoRuntimeManager
from aquaguard.service import EventService
from aquaguard.world import CameraObservation, TemporalFusionCoordinator, WorldModelPipeline

settings = get_settings()
world_model = WorldModelPipeline()
evidence_service = EventEvidenceService(
    EvidenceRecorder(), FileEvidenceRepository(settings.evidence_directory)
)
evidence_service.recover()
video_manager: VideoRuntimeManager = VideoRuntimeAssembler(
    ConfiguredFrameAnalyzerFactory(),
    TemporalFusionCoordinator(world_model),
    evidence_service.recorder,
).build(settings.cameras)
service = EventService(
    RiskEngine(settings.risk_threshold, settings.confirmation_frames),
    settings.alarm_cooldown_seconds,
    evidence=evidence_service,
    evidence_pre_seconds=settings.evidence_pre_seconds,
    evidence_post_seconds=settings.evidence_post_seconds,
    alarm_gate=ValidatedProtectionAlarmGate(video_manager),
    audit_capacity=settings.evaluation_audit_capacity,
    audit_repository=(
        SQLiteEvaluationAuditRepository(
            settings.audit_database_path, settings.evaluation_audit_capacity
        )
        if settings.audit_database_path is not None
        else None
    ),
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    video_manager.start()
    try:
        yield
    finally:
        video_manager.stop()


app = FastAPI(title="AquaGuard AI", version=__version__, lifespan=lifespan)


class EvaluationRequest(BaseModel):
    camera_id: str
    track_id: str
    area: str
    features: RiskFeatures
    observed_at: float | None = Field(default=None, ge=0)


class StatusRequest(BaseModel):
    status: EventStatus


class ObservationRequest(BaseModel):
    camera_id: str
    track_id: str
    timestamp: float
    pool_x: float
    pool_y: float
    confidence: float = Field(default=1.0, ge=0, le=1)
    head_submerged: float = Field(default=0.0, ge=0, le=1)
    body_vertical: float = Field(default=0.0, ge=0, le=1)
    struggle: float = Field(default=0.0, ge=0, le=1)
    motion: float = Field(default=0.0, ge=0, le=1)
    occlusion: float = Field(default=0.0, ge=0, le=1)
    head_in_water_region: float = Field(default=0.0, ge=0, le=1)
    water_relation_confidence: float = Field(default=0.0, ge=0, le=1)
    wrist_motion: float = Field(default=0.0, ge=0, le=1)
    wrist_motion_confidence: float = Field(default=0.0, ge=0, le=1)


class WorldFrameRequest(BaseModel):
    observations: list[ObservationRequest]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/api/v1/video-runtimes")
def video_runtime_statuses() -> list[dict]:
    return video_manager.statuses()


@app.post("/api/v1/evaluations")
def evaluate(request: EvaluationRequest) -> dict:
    result = service.evaluate_decision(
        request.camera_id,
        request.track_id,
        request.area,
        request.features,
        observed_at=request.observed_at,
    )
    return {
        "assessment": result.assessment,
        "event": result.event,
        "alarm_eligible": result.alarm_eligible,
        "suppression_reason": result.suppression_reason,
    }


@app.get("/api/v1/events")
def list_events() -> list:
    return service.events


@app.get("/api/v1/evaluation-audits")
def list_evaluation_audits(
    limit: int | None = None,
    camera_id: str | None = None,
    suppression_reason: str | None = None,
) -> list:
    try:
        return service.list_audits(
            limit=limit,
            camera_id=camera_id,
            suppression_reason=suppression_reason,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.patch("/api/v1/events/{event_id}")
def update_event(event_id: str, request: StatusRequest):
    event = service.update_status(event_id, request.status)
    if event is None:
        raise HTTPException(status_code=404, detail="event not found")
    return event


@app.get("/api/v1/events/{event_id}/evidence")
def evidence_status(event_id: str) -> dict:
    status = evidence_service.status(event_id)
    if status["status"] == "missing":
        raise HTTPException(status_code=404, detail="evidence not found")
    return status


@app.post("/api/v1/world-model/frames")
def process_world_frame(request: WorldFrameRequest) -> dict:
    observations = [CameraObservation(**item.model_dump()) for item in request.observations]
    try:
        return world_model.process(observations)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
