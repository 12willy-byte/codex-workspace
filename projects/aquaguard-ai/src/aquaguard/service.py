from time import monotonic

from aquaguard.domain import AlarmEvent, EventStatus, RiskAssessment, RiskFeatures
from aquaguard.risk import RiskEngine


class EventService:
    def __init__(self, engine: RiskEngine, cooldown_seconds: float = 10) -> None:
        self.engine = engine
        self.cooldown_seconds = cooldown_seconds
        self.events: list[AlarmEvent] = []
        self._last_alarm: dict[str, float] = {}

    def evaluate(self, camera_id: str, track_id: str, area: str, features: RiskFeatures) -> tuple[RiskAssessment, AlarmEvent | None]:
        assessment = self.engine.assess(track_id, features)
        now = monotonic()
        last = self._last_alarm.get(track_id, float("-inf"))
        if not assessment.confirmed or now - last < self.cooldown_seconds:
            return assessment, None
        event = AlarmEvent(camera_id=camera_id, track_id=track_id, area=area, assessment=assessment)
        self.events.append(event)
        self._last_alarm[track_id] = now
        return assessment, event

    def update_status(self, event_id: str, status: EventStatus) -> AlarmEvent | None:
        for event in self.events:
            if str(event.id) == event_id:
                event.status = status
                return event
        return None
