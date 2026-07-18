import pytest

from aquaguard.domain import RiskFeatures
from aquaguard.evidence import EvidenceRecorder, EventEvidenceService, FileEvidenceRepository
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
