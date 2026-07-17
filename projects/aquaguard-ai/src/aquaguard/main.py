from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from aquaguard import __version__
from aquaguard.config import get_settings
from aquaguard.domain import EventStatus, RiskFeatures
from aquaguard.risk import RiskEngine
from aquaguard.service import EventService

settings = get_settings()
service = EventService(RiskEngine(settings.risk_threshold, settings.confirmation_frames), settings.alarm_cooldown_seconds)
app = FastAPI(title="AquaGuard AI", version=__version__)


class EvaluationRequest(BaseModel):
    camera_id: str
    track_id: str
    area: str
    features: RiskFeatures


class StatusRequest(BaseModel):
    status: EventStatus


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/api/v1/evaluations")
def evaluate(request: EvaluationRequest) -> dict:
    assessment, event = service.evaluate(request.camera_id, request.track_id, request.area, request.features)
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
