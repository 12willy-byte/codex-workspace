from concurrent.futures import ThreadPoolExecutor

import pytest

from aquaguard.domain import RiskFeatures
from aquaguard.evidence import EvidenceRecorder, EventEvidenceService, FileEvidenceRepository
from aquaguard.protection import ValidatedProtectionAlarmGate
from aquaguard.risk import RiskEngine
from aquaguard.service import EventService


def dangerous_features() -> RiskFeatures:
    return RiskFeatures(
        head_underwater=1,
        vertical_body=1,
        abnormal_motion=1,
        temporal_risk=1,
    )


def test_confirmed_event_creates_evidence_window_before_registration(tmp_path) -> None:
    evidence = EventEvidenceService(EvidenceRecorder(), FileEvidenceRepository(tmp_path))
    service = EventService(
        RiskEngine(confirmation_frames=1),
        cooldown_seconds=0,
        evidence=evidence,
        evidence_pre_seconds=4,
        evidence_post_seconds=6,
        clock=lambda: 100,
    )

    _, event = service.evaluate(
        "cam-a", "track-1", "deep-pool", dangerous_features(), observed_at=50
    )

    assert event is not None
    window = evidence.recorder.pending[str(event.id)]
    assert window.starts_at == 46
    assert window.ends_at == 56
    assert service.events == [event]


class FailingEvidenceRequester:
    def request(self, *args, **kwargs) -> object:
        raise RuntimeError("storage unavailable")


def test_evidence_failure_does_not_register_event_or_cooldown() -> None:
    service = EventService(
        RiskEngine(confirmation_frames=1),
        cooldown_seconds=60,
        evidence=FailingEvidenceRequester(),
        clock=lambda: 100,
    )

    with pytest.raises(RuntimeError, match="storage unavailable"):
        service.evaluate("cam-a", "track-1", "deep-pool", dangerous_features())

    assert service.events == []
    assert service._last_alarm == {}


def test_cooldown_is_isolated_by_camera_and_local_track_id() -> None:
    service = EventService(
        RiskEngine(confirmation_frames=1), cooldown_seconds=60, clock=lambda: 100
    )

    _, first = service.evaluate("cam-a", "track-1", "pool", dangerous_features())
    _, second = service.evaluate("cam-b", "track-1", "pool", dangerous_features())

    assert first is not None
    assert second is not None


def test_confirmation_frames_are_isolated_by_camera() -> None:
    service = EventService(
        RiskEngine(confirmation_frames=2), cooldown_seconds=0, clock=lambda: 100
    )

    _, first = service.evaluate("cam-a", "track-1", "pool", dangerous_features())
    _, second = service.evaluate("cam-b", "track-1", "pool", dangerous_features())

    assert first is None
    assert second is None


class ProtectionLevels:
    def __init__(self, levels: dict[str, str]) -> None:
        self.levels = levels

    def protection_level(self, camera_id: str) -> str:
        return self.levels.get(camera_id, "unconfigured")


def test_unvalidated_camera_cannot_register_alarm_or_evidence(tmp_path) -> None:
    evidence = EventEvidenceService(EvidenceRecorder(), FileEvidenceRepository(tmp_path))
    service = EventService(
        RiskEngine(confirmation_frames=1),
        evidence=evidence,
        alarm_gate=ValidatedProtectionAlarmGate(
            ProtectionLevels({"cam-a": "pose_baseline"})
        ),
    )

    result = service.evaluate_decision("cam-a", "track-1", "pool", dangerous_features())

    assert result.assessment.confirmed is True
    assert result.alarm_eligible is False
    assert result.suppression_reason == "camera_protection_level:pose_baseline"
    assert result.event is None
    assert service.events == []
    assert evidence.recorder.pending == {}


def test_validated_assistive_camera_can_register_alarm() -> None:
    service = EventService(
        RiskEngine(confirmation_frames=1),
        alarm_gate=ValidatedProtectionAlarmGate(
            ProtectionLevels({"cam-a": "validated_assistive_alerting"})
        ),
    )

    result = service.evaluate_decision("cam-a", "track-1", "pool", dangerous_features())

    assert result.alarm_eligible is True
    assert result.suppression_reason is None
    assert result.event is not None


def test_evaluation_audit_records_suppression_and_is_bounded() -> None:
    service = EventService(
        RiskEngine(confirmation_frames=1),
        alarm_gate=ValidatedProtectionAlarmGate(ProtectionLevels({})),
        audit_capacity=2,
    )

    for track_id in ("track-1", "track-2", "track-3"):
        service.evaluate_decision("cam-a", track_id, "pool", dangerous_features())

    assert [audit.track_id for audit in service.audits] == ["track-2", "track-3"]
    assert all(
        audit.suppression_reason == "camera_protection_level:unconfigured"
        for audit in service.audits
    )
    assert all(audit.event_id is None for audit in service.audits)


def test_evaluation_audit_links_created_event() -> None:
    service = EventService(RiskEngine(confirmation_frames=1))

    result = service.evaluate_decision("cam-a", "track-1", "pool", dangerous_features())

    assert result.event is not None
    assert service.audits[0].event_id == result.event.id


def test_evaluation_audit_is_a_snapshot_not_a_shared_reference() -> None:
    service = EventService(RiskEngine(confirmation_frames=1))
    features = dangerous_features()

    result = service.evaluate_decision("cam-a", "track-1", "pool", features)
    features.head_underwater = 0
    result.assessment.score = 0

    assert service.audits[0].features.head_underwater == 1
    assert service.audits[0].assessment.score == 100


def test_concurrent_confirmations_create_only_one_event() -> None:
    service = EventService(
        RiskEngine(confirmation_frames=2), cooldown_seconds=60, clock=lambda: 100
    )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda _: service.evaluate_decision(
                    "cam-a", "track-1", "pool", dangerous_features()
                ),
                range(20),
            )
        )

    assert sum(result.event is not None for result in results) == 1
    assert len(service.list_events()) == 1
    assert len(service.audits) == 20


def test_list_events_returns_detached_snapshots() -> None:
    service = EventService(RiskEngine(confirmation_frames=1))
    service.evaluate_decision("cam-a", "track-1", "pool", dangerous_features())

    events = service.list_events()
    events[0].area = "tampered"

    assert service.list_events()[0].area == "pool"
