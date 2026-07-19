import sqlite3
from pathlib import Path
from threading import Lock
from typing import Protocol

from aquaguard.domain import AlarmEvent, EventStatus


class AlarmEventRepository(Protocol):
    def append_if_allowed(self, event: AlarmEvent, cooldown_seconds: float) -> bool: ...

    def delete(self, event_id: str) -> bool: ...

    def update_status(self, event_id: str, status: EventStatus) -> AlarmEvent | None: ...

    def list(self) -> list[AlarmEvent]: ...


class InMemoryAlarmEventRepository:
    def __init__(self) -> None:
        self._events: list[AlarmEvent] = []
        self._lock = Lock()

    def append_if_allowed(self, event: AlarmEvent, cooldown_seconds: float) -> bool:
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must not be negative")
        with self._lock:
            latest = next(
                (
                    item
                    for item in reversed(self._events)
                    if item.camera_id == event.camera_id and item.track_id == event.track_id
                ),
                None,
            )
            if latest is not None:
                elapsed = (event.created_at - latest.created_at).total_seconds()
                if elapsed < cooldown_seconds:
                    return False
            self._events.append(event.model_copy(deep=True))
            return True

    def delete(self, event_id: str) -> bool:
        with self._lock:
            for index, event in enumerate(self._events):
                if str(event.id) == event_id:
                    del self._events[index]
                    return True
        return False

    def update_status(self, event_id: str, status: EventStatus) -> AlarmEvent | None:
        with self._lock:
            for event in self._events:
                if str(event.id) == event_id:
                    event.status = status
                    return event.model_copy(deep=True)
        return None

    def list(self) -> list[AlarmEvent]:
        with self._lock:
            return [event.model_copy(deep=True) for event in self._events]


class SQLiteAlarmEventRepository:
    def __init__(self, path: Path):
        self.path = path
        self._lock = Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS alarm_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    camera_id TEXT NOT NULL,
                    track_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS alarm_events_cooldown
                ON alarm_events (camera_id, track_id, created_at DESC)
                """
            )

    def append_if_allowed(self, event: AlarmEvent, cooldown_seconds: float) -> bool:
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must not be negative")
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """
                    SELECT created_at FROM alarm_events
                    WHERE camera_id = ? AND track_id = ?
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (event.camera_id, event.track_id),
                ).fetchone()
                created_at = event.created_at.timestamp()
                if row is not None and created_at - float(row[0]) < cooldown_seconds:
                    connection.rollback()
                    return False
                connection.execute(
                    """
                    INSERT INTO alarm_events
                        (event_id, camera_id, track_id, created_at, status, payload)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(event.id),
                        event.camera_id,
                        event.track_id,
                        created_at,
                        event.status.value,
                        event.model_dump_json(),
                    ),
                )
                connection.commit()
                return True
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    def delete(self, event_id: str) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute("DELETE FROM alarm_events WHERE event_id = ?", (event_id,))
            return cursor.rowcount > 0

    def update_status(self, event_id: str, status: EventStatus) -> AlarmEvent | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM alarm_events WHERE event_id = ?", (event_id,)
            ).fetchone()
            if row is None:
                return None
            event = AlarmEvent.model_validate_json(row[0])
            event.status = status
            connection.execute(
                """
                UPDATE alarm_events SET status = ?, payload = ? WHERE event_id = ?
                """,
                (status.value, event.model_dump_json(), event_id),
            )
            return event.model_copy(deep=True)

    def list(self) -> list[AlarmEvent]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM alarm_events ORDER BY sequence"
            ).fetchall()
        return [AlarmEvent.model_validate_json(row[0]) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5)
