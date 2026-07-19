from contextlib import asynccontextmanager
from datetime import datetime

from math import ceil
from uuid import UUID

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from aquaguard import __version__
from aquaguard.assembly import ConfiguredFrameAnalyzerFactory, VideoRuntimeAssembler
from aquaguard.audit import SQLiteEvaluationAuditRepository
from aquaguard.auth import (
    AuthenticatedOperator,
    OperatorAuthenticator,
    OperatorCredential,
    OperatorRole,
    Permission,
    is_authorized,
)
from aquaguard.config import get_settings
from aquaguard.consistency import EvidenceConsistencyService
from aquaguard.credentials import (
    InMemoryOperatorCredentialRepository,
    OperatorCredentialService,
    SQLiteOperatorCredentialRepository,
)
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
from aquaguard.security import (
    AuthenticationFailureRateLimiter,
    InMemorySecurityAuditRepository,
    SecurityAudit,
    SecurityOutcome,
    SQLiteSecurityAuditRepository,
)
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
operator_authenticator = OperatorAuthenticator()
operator_credential_repository = (
    SQLiteOperatorCredentialRepository(settings.operator_credential_database_path)
    if settings.operator_credential_database_path is not None
    else InMemoryOperatorCredentialRepository()
)
if not operator_credential_repository.list():
    for configured_credential in settings.operator_credentials:
        operator_credential_repository.append(configured_credential)
operator_credential_service = OperatorCredentialService(
    operator_credential_repository, operator_authenticator
)
security_audit_repository = (
    SQLiteSecurityAuditRepository(
        settings.security_audit_database_path, settings.security_audit_capacity
    )
    if settings.security_audit_database_path is not None
    else InMemorySecurityAuditRepository(settings.security_audit_capacity)
)
auth_rate_limiter = AuthenticationFailureRateLimiter(
    settings.auth_failure_limit,
    settings.auth_failure_window_seconds,
    settings.auth_rate_limit_max_clients,
)


def require_operator(
    request: Request,
    authorization: str | None,
    permission: Permission,
) -> AuthenticatedOperator:
    client_host = request.client.host if request.client is not None else "unknown"
    retry_after = auth_rate_limiter.retry_after(client_host)
    if retry_after is not None:
        security_audit_repository.append(
            SecurityAudit(
                outcome=SecurityOutcome.RATE_LIMITED,
                permission=permission,
                path=request.url.path,
                client_host=client_host,
            )
        )
        raise HTTPException(
            status_code=429,
            detail="too many authentication failures",
            headers={"Retry-After": str(max(1, ceil(retry_after)))},
        )
    operator = operator_authenticator.authenticate(authorization)
    if operator is None:
        auth_rate_limiter.record_failure(client_host)
        security_audit_repository.append(
            SecurityAudit(
                outcome=SecurityOutcome.AUTHENTICATION_FAILED,
                permission=permission,
                path=request.url.path,
                client_host=client_host,
            )
        )
        raise HTTPException(
            status_code=401,
            detail="valid operator bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not is_authorized(operator, permission):
        security_audit_repository.append(
            SecurityAudit(
                outcome=SecurityOutcome.AUTHORIZATION_DENIED,
                permission=permission,
                path=request.url.path,
                client_host=client_host,
                operator=operator.username,
                role=operator.role,
            )
        )
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


class CredentialCreateRequest(BaseModel):
    username: str = Field(min_length=1)
    role: OperatorRole
    token_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expires_at: datetime | None = None

    @field_validator("expires_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("expires_at must include a timezone")
        return value


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
    request: Request,
    authorization: str | None = Header(default=None),
) -> list[dict]:
    require_operator(request, authorization, Permission.VIEW_OPERATIONS)
    return video_manager.statuses()


@app.post("/api/v1/evaluations")
def evaluate(
    payload: EvaluationRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict:
    require_operator(request, authorization, Permission.INGEST_OBSERVATIONS)
    result = service.evaluate_decision(
        payload.camera_id,
        payload.track_id,
        payload.area,
        payload.features,
        observed_at=payload.observed_at,
    )
    return {
        "assessment": result.assessment,
        "event": result.event,
        "alarm_eligible": result.alarm_eligible,
        "suppression_reason": result.suppression_reason,
    }


@app.get("/api/v1/events")
def list_events(request: Request, authorization: str | None = Header(default=None)) -> list:
    require_operator(request, authorization, Permission.VIEW_OPERATIONS)
    return service.list_events()


@app.get("/api/v1/evaluation-audits")
def list_evaluation_audits(
    request: Request,
    limit: int | None = None,
    camera_id: str | None = None,
    suppression_reason: str | None = None,
    authorization: str | None = Header(default=None),
) -> list:
    require_operator(request, authorization, Permission.VIEW_AUDIT)
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
    payload: StatusRequest,
    request: Request,
    authorization: str | None = Header(default=None),
):
    require_operator(request, authorization, Permission.MANAGE_INCIDENTS)
    event = service.update_status(event_id, payload.status)
    if event is None:
        raise HTTPException(status_code=404, detail="event not found")
    return event


@app.get("/api/v1/events/{event_id}/evidence")
def evidence_status(
    event_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict:
    require_operator(request, authorization, Permission.VIEW_OPERATIONS)
    status = evidence_service.status(event_id)
    if status["status"] == "missing":
        raise HTTPException(status_code=404, detail="evidence not found")
    return status


@app.get("/api/v1/system/evidence-consistency")
def evidence_consistency(
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict:
    require_operator(
        request,
        authorization,
        Permission.VIEW_AUDIT,
    )
    return consistency_service.inspect().model_dump(mode="json")


@app.post("/api/v1/system/evidence-remediations")
def remediate_evidence(
    payload: RemediationRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict:
    if not settings.remediation_enabled:
        raise HTTPException(status_code=503, detail="evidence remediation is disabled")
    operator = require_operator(
        request,
        authorization,
        Permission.REMEDIATE_EVIDENCE,
    )
    try:
        return remediation_service.execute(
            payload.target_id,
            payload.action,
            operator.username,
            payload.reason,
        ).model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/v1/system/evidence-remediations")
def list_evidence_remediations(
    request: Request,
    target_id: str | None = None,
    authorization: str | None = Header(default=None),
) -> list:
    require_operator(
        request,
        authorization,
        Permission.VIEW_AUDIT,
    )
    return remediation_service.list(target_id=target_id)


@app.get("/api/v1/security-audits")
def list_security_audits(
    request: Request,
    limit: int | None = None,
    outcome: SecurityOutcome | None = None,
    authorization: str | None = Header(default=None),
) -> list:
    require_operator(request, authorization, Permission.VIEW_SECURITY_AUDIT)
    try:
        return security_audit_repository.list(limit=limit, outcome=outcome)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/v1/operator-credentials")
def list_operator_credentials(
    request: Request,
    authorization: str | None = Header(default=None),
) -> list:
    require_operator(request, authorization, Permission.MANAGE_CREDENTIALS)
    return operator_credential_service.list()


@app.post("/api/v1/operator-credentials", status_code=201)
def create_operator_credential(
    payload: CredentialCreateRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict:
    operator = require_operator(request, authorization, Permission.MANAGE_CREDENTIALS)
    credential = OperatorCredential(**payload.model_dump())
    try:
        view = operator_credential_service.create(credential)
    except ValueError as exc:
        security_audit_repository.append(
            SecurityAudit(
                outcome=SecurityOutcome.CREDENTIAL_CHANGE_REJECTED,
                permission=Permission.MANAGE_CREDENTIALS,
                path=request.url.path,
                client_host=(request.client.host if request.client is not None else "unknown"),
                operator=operator.username,
                role=operator.role,
                target_id=str(credential.id),
            )
        )
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    security_audit_repository.append(
        SecurityAudit(
            outcome=SecurityOutcome.CREDENTIAL_CREATED,
            permission=Permission.MANAGE_CREDENTIALS,
            path=request.url.path,
            client_host=request.client.host if request.client is not None else "unknown",
            operator=operator.username,
            role=operator.role,
            target_id=str(view.id),
        )
    )
    return view.model_dump(mode="json")


@app.post("/api/v1/operator-credentials/{credential_id}/revoke")
def revoke_operator_credential(
    credential_id: UUID,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict:
    operator = require_operator(request, authorization, Permission.MANAGE_CREDENTIALS)
    try:
        view = operator_credential_service.revoke(credential_id, operator.credential_id)
    except ValueError as exc:
        security_audit_repository.append(
            SecurityAudit(
                outcome=SecurityOutcome.CREDENTIAL_CHANGE_REJECTED,
                permission=Permission.MANAGE_CREDENTIALS,
                path=request.url.path,
                client_host=(request.client.host if request.client is not None else "unknown"),
                operator=operator.username,
                role=operator.role,
                target_id=str(credential_id),
            )
        )
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if view is None:
        raise HTTPException(status_code=404, detail="credential not found")
    security_audit_repository.append(
        SecurityAudit(
            outcome=SecurityOutcome.CREDENTIAL_REVOKED,
            permission=Permission.MANAGE_CREDENTIALS,
            path=request.url.path,
            client_host=request.client.host if request.client is not None else "unknown",
            operator=operator.username,
            role=operator.role,
            target_id=str(view.id),
        )
    )
    return view.model_dump(mode="json")


@app.post("/api/v1/world-model/frames")
def process_world_frame(
    payload: WorldFrameRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict:
    require_operator(request, authorization, Permission.INGEST_OBSERVATIONS)
    observations = [CameraObservation(**item.model_dump()) for item in payload.observations]
    try:
        return world_model.process(observations)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
