from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from aquaguard import __version__
from aquaguard.assembly import ConfiguredFrameAnalyzerFactory, VideoRuntimeAssembler
from aquaguard.audit import SQLiteEvaluationAuditRepository
from aquaguard.auth import (
    AuthenticatedOperator,
    OperatorAuthenticator,
    Permission,
    is_authorized,
)
from aquaguard.config import get_settings
from aquaguard.consistency import EvidenceConsistencyService
from aquaguard.domain import EventStatus, RiskFeatures
from aquaguard.evidence import EvidenceRecorder, EventEvidenceService, FileEvidenceRepository
from aquaguard.events import SQLiteAlarmEventRepository
from aquaguard.protection import ValidatedProtectionAlarmGate
from aquaguard.remediation import (
    EvidenceRemediationService,
    RemediationAction,
    SQLiteRemediationRepository,
)
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
    event_repository=(
        SQLiteAlarmEventRepository(settings.event_database_path)
        if settings.event_database_path is not None
        else None
    ),
)
consistency_service = EvidenceConsistencyService(service.event_repository, evidence_service)
remediation_service = EvidenceRemediationService(
    consistency_service,
    evidence_service,
    (
        SQLiteRemediationRepository(settings.remediation_database_path)
        if settings.remediation_database_path is not None
        else None
    ),
)
operator_authenticator = OperatorAuthenticator(settings.operator_credentials)


def require_operator(
    authorization: str | None,
    permission: Permission,
) -> AuthenticatedOperator:
    operator = operator_authenticator.authenticate(authorization)
    if operator is None:
        raise HTTPException(
            status_code=401,
            detail="valid operator bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not is_authorized(operator, permission):
        raise HTTPException(status_code=403, detail="operator role is not permitted")
    return operator


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


class RemediationRequest(BaseModel):
    target_id: str = Field(min_length=1)
    action: RemediationAction
    reason: str = Field(min_length=1)


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
def video_runtime_statuses(
    authorization: str | None = Header(default=None),
) -> list[dict]:
    require_operator(authorization, Permission.VIEW_OPERATIONS)
    return video_manager.statuses()


@app.post("/api/v1/evaluations")
def evaluate(
    request: EvaluationRequest,
    authorization: str | None = Header(default=None),
) -> dict:
    require_operator(authorization, Permission.INGEST_OBSERVATIONS)
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
def list_events(authorization: str | None = Header(default=None)) -> list:
    require_operator(authorization, Permission.VIEW_OPERATIONS)
    return service.list_events()


@app.get("/api/v1/evaluation-audits")
def list_evaluation_audits(
    limit: int | None = None,
    camera_id: str | None = None,
    suppression_reason: str | None = None,
    authorization: str | None = Header(default=None),
) -> list:
    require_operator(authorization, Permission.VIEW_AUDIT)
    try:
        return service.list_audits(
            limit=limit,
            camera_id=camera_id,
            suppression_reason=suppression_reason,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.patch("/api/v1/events/{event_id}")
def update_event(
    event_id: str,
    request: StatusRequest,
    authorization: str | None = Header(default=None),
):
    require_operator(authorization, Permission.MANAGE_INCIDENTS)
    event = service.update_status(event_id, request.status)
    if event is None:
        raise HTTPException(status_code=404, detail="event not found")
    return event


@app.get("/api/v1/events/{event_id}/evidence")
def evidence_status(
    event_id: str,
    authorization: str | None = Header(default=None),
) -> dict:
    require_operator(authorization, Permission.VIEW_OPERATIONS)
    status = evidence_service.status(event_id)
    if status["status"] == "missing":
        raise HTTPException(status_code=404, detail="evidence not found")
    return status


@app.get("/api/v1/system/evidence-consistency")
def evidence_consistency(
    authorization: str | None = Header(default=None),
) -> dict:
    require_operator(
        authorization,
        Permission.VIEW_AUDIT,
    )
    return consistency_service.inspect().model_dump(mode="json")


@app.post("/api/v1/system/evidence-remediations")
def remediate_evidence(
    request: RemediationRequest,
    authorization: str | None = Header(default=None),
) -> dict:
    if not settings.remediation_enabled:
        raise HTTPException(status_code=503, detail="evidence remediation is disabled")
    operator = require_operator(
        authorization,
        Permission.REMEDIATE_EVIDENCE,
    )
    try:
        return remediation_service.execute(
            request.target_id,
            request.action,
            operator.username,
            request.reason,
        ).model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/v1/system/evidence-remediations")
def list_evidence_remediations(
    target_id: str | None = None,
    authorization: str | None = Header(default=None),
) -> list:
    require_operator(
        authorization,
        Permission.VIEW_AUDIT,
    )
    return remediation_service.list(target_id=target_id)


@app.post("/api/v1/world-model/frames")
def process_world_frame(
    request: WorldFrameRequest,
    authorization: str | None = Header(default=None),
) -> dict:
    require_operator(authorization, Permission.INGEST_OBSERVATIONS)
    observations = [CameraObservation(**item.model_dump()) for item in request.observations]
    try:
        return world_model.process(observations)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
