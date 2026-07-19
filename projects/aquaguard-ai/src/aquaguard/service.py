from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import Protocol

from aquaguard.domain import AlarmEvent, EventStatus, RiskAssessment, RiskFeatures
from aquaguard.risk import RiskEngine
from aquaguard.protection import AlarmGate, AllowAllAlarmGate


class EvidenceRequester(Protocol):
    def request(
        self,
        event: AlarmEvent,
        trigger_timestamp: float,
        *,
        pre_seconds: float,
        post_seconds: float,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class EventEvaluation:
    assessment: RiskAssessment
    event: AlarmEvent | None
    alarm_eligible: bool
    suppression_reason: str | None


class EventService:
    def __init__(
        self,
        engine: RiskEngine,
        cooldown_seconds: float = 10,
        *,
        evidence: EvidenceRequester | None = None,
        evidence_pre_seconds: float = 30,
        evidence_post_seconds: float = 60,
        clock: Callable[[], float] = monotonic,
        alarm_gate: AlarmGate | None = None,
    ) -> None:
        self.engine = engine
        self.cooldown_seconds = cooldown_seconds
        self.evidence = evidence
        self.evidence_pre_seconds = evidence_pre_seconds
        self.evidence_post_seconds = evidence_post_seconds
        self.clock = clock
        self.alarm_gate = alarm_gate or AllowAllAlarmGate()
        self.events: list[AlarmEvent] = []
        self._last_alarm: dict[tuple[str, str], float] = {}

    def evaluate(
        self,
        camera_id: str,
        track_id: str,
        area: str,
        features: RiskFeatures,
        *,
        observed_at: float | None = None,
    ) -> tuple[RiskAssessment, AlarmEvent | None]:
        result = self.evaluate_decision(
            camera_id,
            track_id,
            area,
            features,
            observed_at=observed_at,
        )
        return result.assessment, result.event

    def evaluate_decision(
        self,
        camera_id: str,
        track_id: str,
        area: str,
        features: RiskFeatures,
        *,
        observed_at: float | None = None,
    ) -> EventEvaluation:
        assessment = self.engine.assess(f"{camera_id}:{track_id}", features)
        now = self.clock()
        alarm_key = (camera_id, track_id)
        last = self._last_alarm.get(alarm_key, float("-inf"))
        eligible, gate_reason = self.alarm_gate.check(camera_id)
        if not assessment.confirmed:
            return EventEvaluation(assessment, None, eligible, "risk_not_confirmed")
        if not eligible:
            return EventEvaluation(assessment, None, False, gate_reason)
        if now - last < self.cooldown_seconds:
            return EventEvaluation(assessment, None, True, "cooldown_active")
        event = AlarmEvent(camera_id=camera_id, track_id=track_id, area=area, assessment=assessment)
        if self.evidence is not None:
            self.evidence.request(
                event,
                now if observed_at is None else observed_at,
                pre_seconds=self.evidence_pre_seconds,
                post_seconds=self.evidence_post_seconds,
            )
        self.events.append(event)
        self._last_alarm[alarm_key] = now
        return EventEvaluation(assessment, event, True, None)

    def update_status(self, event_id: str, status: EventStatus) -> AlarmEvent | None:
        for event in self.events:
            if str(event.id) == event_id:
                event.status = status
                return event
        return None
