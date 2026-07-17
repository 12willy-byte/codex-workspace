from datetime import datetime, timezone
from enum import IntEnum, StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class AlarmLevel(IntEnum):
    NORMAL = 0
    WATCH = 1
    WARNING = 2
    EMERGENCY = 3


class EventStatus(StrEnum):
    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class RiskFeatures(BaseModel):
    head_underwater: float = Field(ge=0, le=1)
    vertical_body: float = Field(ge=0, le=1)
    abnormal_motion: float = Field(ge=0, le=1)
    temporal_risk: float = Field(ge=0, le=1)


class RiskAssessment(BaseModel):
    score: float = Field(ge=0, le=100)
    level: AlarmLevel
    confirmed: bool


class AlarmEvent(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    camera_id: str
    track_id: str
    area: str
    assessment: RiskAssessment
    status: EventStatus = EventStatus.NEW
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
