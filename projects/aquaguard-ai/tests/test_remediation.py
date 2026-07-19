import pytest
from uuid import uuid4

from aquaguard.consistency import EvidenceConsistencyService
from aquaguard.domain import AlarmEvent, AlarmLevel, RiskAssessment
from aquaguard.events import InMemoryAlarmEventRepository
from aquaguard.evidence import EvidenceRecorder, EventEvidenceService, FileEvidenceRepository
from aquaguard.remediation import (
    EvidenceRemediationService,
    RemediationAction,
    RemediationRecord,
    SQLiteRemediationRepository,
)


def registered_event(events: InMemoryAlarmEventRepository) -> AlarmEvent:
    event = AlarmEvent(
        camera_id="cam-a",
        track_id="track-1",
        area="pool",
        assessment=RiskAssessment(score=95, level=AlarmLevel.EMERGENCY, confirmed=True),
    )
    events.append_if_allowed(event, 0)
    return event


def service(tmp_path, repository=None):
    events = InMemoryAlarmEventRepository()
    evidence = EventEvidenceService(EvidenceRecorder(), FileEvidenceRepository(tmp_path))
    remediation = EvidenceRemediationService(
        EvidenceConsistencyService(events, evidence), evidence, repository
    )
    return events, evidence, remediation


def test_persist_ready_action_stores_evidence_and_audits_operator(tmp_path) -> None:
    events, evidence, remediation = service(tmp_path)
    event = registered_event(events)
    evidence.request(event, 10, pre_seconds=0, post_seconds=0)
    evidence.recorder.force_finalize(str(event.id))

    record = remediation.execute(
        str(event.id),
        RemediationAction.PERSIST_READY,
        "operator-1",
        "recover completed capture after restart",
    )

    assert record.previous_status == "ready"
    assert record.outcome_status == "stored"
    assert evidence.status(str(event.id))["integrity"] == "verified"
    assert remediation.list(target_id=str(event.id)) == [record]


def test_acknowledge_orphan_does_not_delete_evidence(tmp_path) -> None:
    _, evidence, remediation = service(tmp_path)
    evidence.recorder.request("orphan-1", "cam-a", 10, pre_seconds=1, post_seconds=1)

    record = remediation.execute(
        "orphan-1",
        RemediationAction.ACKNOWLEDGE_ORPHANED,
        "operator-1",
        "retain for incident review",
    )

    assert record.outcome_status == "orphaned"
    assert "orphan-1" in evidence.recorder.pending


def test_action_precondition_failure_does_not_create_audit(tmp_path) -> None:
    events, _, remediation = service(tmp_path)
    event = registered_event(events)

    with pytest.raises(ValueError, match="requires status ready, got missing"):
        remediation.execute(
            str(event.id),
            RemediationAction.PERSIST_READY,
            "operator-1",
            "invalid retry",
        )

    assert remediation.list() == []


def test_persistence_failure_is_recorded_without_hiding_error(tmp_path, monkeypatch) -> None:
    events, evidence, remediation = service(tmp_path)
    event = registered_event(events)
    evidence.request(event, 10, pre_seconds=0, post_seconds=0)
    evidence.recorder.force_finalize(str(event.id))

    def fail(_: str):
        raise RuntimeError("storage offline")

    monkeypatch.setattr(evidence, "force_persist", fail)

    with pytest.raises(RuntimeError, match="storage offline"):
        remediation.execute(
            str(event.id),
            RemediationAction.PERSIST_READY,
            "operator-1",
            "retry after restart",
        )

    records = remediation.list(target_id=str(event.id))
    assert len(records) == 1
    assert records[0].outcome_status == "failed"


def test_sqlite_remediation_repository_recovers_and_filters(tmp_path) -> None:
    path = tmp_path / "remediations.db"
    repository = SQLiteRemediationRepository(path)
    first = RemediationRecord(
        target_id="event-1",
        action=RemediationAction.ACKNOWLEDGE_MISSING,
        operator="operator-1",
        reason="camera outage confirmed",
        previous_status="missing",
        outcome_status="missing",
    )
    second = first.model_copy(update={"id": uuid4(), "target_id": "event-2"})
    repository.append(first)
    repository.append(second)

    recovered = SQLiteRemediationRepository(path)

    assert recovered.list(target_id="event-1") == [first]
    assert recovered.list() == [first, second]
