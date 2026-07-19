import sqlite3
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Protocol

from aquaguard.domain import EvaluationAudit


class EvaluationAuditRepository(Protocol):
    def append(self, audit: EvaluationAudit) -> None: ...

    def list(
        self,
        *,
        limit: int | None = None,
        camera_id: str | None = None,
        suppression_reason: str | None = None,
    ) -> list[EvaluationAudit]: ...


class InMemoryEvaluationAuditRepository:
    def __init__(self, capacity: int = 1000):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._audits: deque[EvaluationAudit] = deque(maxlen=capacity)
        self._lock = Lock()

    def append(self, audit: EvaluationAudit) -> None:
        with self._lock:
            self._audits.append(audit.model_copy(deep=True))

    def list(
        self,
        *,
        limit: int | None = None,
        camera_id: str | None = None,
        suppression_reason: str | None = None,
    ) -> list[EvaluationAudit]:
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        with self._lock:
            items = [
                item.model_copy(deep=True)
                for item in self._audits
                if (camera_id is None or item.camera_id == camera_id)
                and (
                    suppression_reason is None
                    or item.suppression_reason == suppression_reason
                )
            ]
        return items[-limit:] if limit is not None else items


class SQLiteEvaluationAuditRepository:
    def __init__(self, path: Path, capacity: int = 1000):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.path = path
        self.capacity = capacity
        self._lock = Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS evaluation_audits (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    audit_id TEXT NOT NULL UNIQUE,
                    camera_id TEXT NOT NULL,
                    suppression_reason TEXT,
                    payload TEXT NOT NULL
                )
                """
            )

    def append(self, audit: EvaluationAudit) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO evaluation_audits
                    (audit_id, camera_id, suppression_reason, payload)
                VALUES (?, ?, ?, ?)
                """,
                (
                    str(audit.id),
                    audit.camera_id,
                    audit.suppression_reason,
                    audit.model_dump_json(),
                ),
            )
            connection.execute(
                """
                DELETE FROM evaluation_audits
                WHERE sequence NOT IN (
                    SELECT sequence FROM evaluation_audits
                    ORDER BY sequence DESC LIMIT ?
                )
                """,
                (self.capacity,),
            )

    def list(
        self,
        *,
        limit: int | None = None,
        camera_id: str | None = None,
        suppression_reason: str | None = None,
    ) -> list[EvaluationAudit]:
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        conditions = []
        parameters: list[object] = []
        if camera_id is not None:
            conditions.append("camera_id = ?")
            parameters.append(camera_id)
        if suppression_reason is not None:
            conditions.append("suppression_reason = ?")
            parameters.append(suppression_reason)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit_sql = "LIMIT ?" if limit is not None else ""
        if limit is not None:
            parameters.append(limit)
        query = f"""
            SELECT payload FROM evaluation_audits
            {where}
            ORDER BY sequence DESC
            {limit_sql}
        """
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [EvaluationAudit.model_validate_json(row[0]) for row in reversed(rows)]

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5)
