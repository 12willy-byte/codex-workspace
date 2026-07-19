from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from aquaguard.domain import (
    AlarmEvent,
    AlarmLevel,
    EventStatus,
    RiskAssessment,
)
from aquaguard.events import SQLiteAlarmEventRepository


def alarm_event(
    *,
    camera_id: str = "cam-a",
    track_id: str = "track-1",
    created_at: datetime | None = None,
) -> AlarmEvent:
    return AlarmEvent(
        camera_id=camera_id,
        track_id=track_id,
        area="deep-pool",
        assessment=RiskAssessment(score=95, level=AlarmLevel.EMERGENCY, confirmed=True),
        created_at=created_at or datetime.now(timezone.utc),
    )


def test_sqlite_event_repository_recovers_events_and_status(tmp_path) -> None:
    path = tmp_path / "events.db"
    repository = SQLiteAlarmEventRepository(path)
    event = alarm_event()

    assert repository.append_if_allowed(event, 60) is True
    updated = repository.update_status(str(event.id), EventStatus.ACKNOWLEDGED)

    assert updated is not None
    recovered = SQLiteAlarmEventRepository(path).list()
    assert len(recovered) == 1
    assert recovered[0].id == event.id
    assert recovered[0].status == EventStatus.ACKNOWLEDGED


def test_sqlite_event_repository_serializes_cross_instance_cooldown(tmp_path) -> None:
    path = tmp_path / "events.db"
    first = SQLiteAlarmEventRepository(path)
    second = SQLiteAlarmEventRepository(path)
    timestamp = datetime.now(timezone.utc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        accepted = list(
            executor.map(
                lambda pair: pair[0].append_if_allowed(pair[1], 60),
                (
                    (first, alarm_event(created_at=timestamp)),
                    (second, alarm_event(created_at=timestamp)),
                ),
            )
        )

    assert sorted(accepted) == [False, True]
    assert len(first.list()) == 1


def test_sqlite_event_repository_cooldown_is_per_camera_and_track(tmp_path) -> None:
    repository = SQLiteAlarmEventRepository(tmp_path / "events.db")
    timestamp = datetime.now(timezone.utc)

    assert repository.append_if_allowed(alarm_event(camera_id="cam-a", created_at=timestamp), 60)
    assert repository.append_if_allowed(alarm_event(camera_id="cam-b", created_at=timestamp), 60)
    assert repository.append_if_allowed(alarm_event(track_id="track-2", created_at=timestamp), 60)
    assert repository.append_if_allowed(
        alarm_event(created_at=timestamp + timedelta(seconds=60)), 60
    )
    assert len(repository.list()) == 4


def test_sqlite_event_repository_can_delete_failed_registration(tmp_path) -> None:
    repository = SQLiteAlarmEventRepository(tmp_path / "events.db")
    event = alarm_event()
    repository.append_if_allowed(event, 0)

    assert repository.delete(str(event.id)) is True
    assert repository.delete(str(event.id)) is False
    assert repository.list() == []
