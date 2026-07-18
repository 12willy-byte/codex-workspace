from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from aquaguard import __version__
from aquaguard.assembly import ConfiguredFrameAnalyzerFactory, VideoRuntimeAssembler
from aquaguard.config import get_settings
from aquaguard.domain import EventStatus, RiskFeatures
from aquaguard.evidence import EvidenceRecorder, EventEvidenceService, FileEvidenceRepository
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
    assessment, event = service.evaluate(
        request.camera_id,
        request.track_id,
        request.area,
        request.features,
        observed_at=request.observed_at,
    )
    return {"assessment": assessment, "event": event}


@app.get("/api/v1/events")
def list_events() -> list:
    return service.events


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
