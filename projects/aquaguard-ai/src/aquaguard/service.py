from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from time import monotonic
from typing import Protocol

from aquaguard.domain import (
    AlarmEvent,
    EvaluationAudit,
    EventStatus,
    RiskAssessment,
    RiskFeatures,
)
from aquaguard.audit import EvaluationAuditRepository, InMemoryEvaluationAuditRepository
from aquaguard.events import AlarmEventRepository, InMemoryAlarmEventRepository
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
        audit_capacity: int = 1000,
        audit_repository: EvaluationAuditRepository | None = None,
        event_repository: AlarmEventRepository | None = None,
    ) -> None:
        if audit_capacity < 1:
            raise ValueError("audit_capacity must be positive")
        self.engine = engine
        self.cooldown_seconds = cooldown_seconds
        self.evidence = evidence
        self.evidence_pre_seconds = evidence_pre_seconds
        self.evidence_post_seconds = evidence_post_seconds
        self.clock = clock
        self.alarm_gate = alarm_gate or AllowAllAlarmGate()
        self.audit_repository = audit_repository or InMemoryEvaluationAuditRepository(
            audit_capacity
        )
        self.event_repository = event_repository or InMemoryAlarmEventRepository()
        self._last_alarm: dict[tuple[str, str], float] = {}
        self._lock = RLock()

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
        with self._lock:
            return self._evaluate_decision(
                camera_id,
                track_id,
                area,
                features,
                observed_at=observed_at,
            )

    def _evaluate_decision(
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
            return self._complete(
                camera_id,
                track_id,
                area,
                features,
                EventEvaluation(assessment, None, eligible, "risk_not_confirmed"),
            )
        if not eligible:
            return self._complete(
                camera_id,
                track_id,
                area,
                features,
                EventEvaluation(assessment, None, False, gate_reason),
            )
        if now - last < self.cooldown_seconds:
            return self._complete(
                camera_id,
                track_id,
                area,
                features,
                EventEvaluation(assessment, None, True, "cooldown_active"),
            )
        event = AlarmEvent(
            camera_id=camera_id,
            track_id=track_id,
            area=area,
            assessment=assessment,
        )
        if not self.event_repository.append_if_allowed(event, self.cooldown_seconds):
            return self._complete(
                camera_id,
                track_id,
                area,
                features,
                EventEvaluation(assessment, None, True, "cooldown_active"),
            )
        try:
            if self.evidence is not None:
                self.evidence.request(
                    event,
                    now if observed_at is None else observed_at,
                    pre_seconds=self.evidence_pre_seconds,
                    post_seconds=self.evidence_post_seconds,
                )
        except Exception:
            self.event_repository.delete(str(event.id))
            raise
        self._last_alarm[alarm_key] = now
        return self._complete(
            camera_id,
            track_id,
            area,
            features,
            EventEvaluation(assessment, event, True, None),
        )

    def _complete(
        self,
        camera_id: str,
        track_id: str,
        area: str,
        features: RiskFeatures,
        result: EventEvaluation,
    ) -> EventEvaluation:
        self.audit_repository.append(
            EvaluationAudit(
                camera_id=camera_id,
                track_id=track_id,
                area=area,
                features=features.model_copy(deep=True),
                assessment=result.assessment.model_copy(deep=True),
                alarm_eligible=result.alarm_eligible,
                suppression_reason=result.suppression_reason,
                event_id=result.event.id if result.event is not None else None,
            )
        )
        return result

    @property
    def audits(self) -> list[EvaluationAudit]:
        return self.audit_repository.list()

    def list_audits(
        self,
        *,
        limit: int | None = None,
        camera_id: str | None = None,
        suppression_reason: str | None = None,
    ) -> list[EvaluationAudit]:
        return self.audit_repository.list(
            limit=limit,
            camera_id=camera_id,
            suppression_reason=suppression_reason,
        )

    def update_status(self, event_id: str, status: EventStatus) -> AlarmEvent | None:
        with self._lock:
            return self.event_repository.update_status(event_id, status)

    def list_events(self) -> list[AlarmEvent]:
        with self._lock:
            return self.event_repository.list()

    @property
    def events(self) -> list[AlarmEvent]:
        return self.list_events()
