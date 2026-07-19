import sqlite3
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from threading import Lock
from typing import Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from aquaguard.consistency import EvidenceConsistencyService
from aquaguard.evidence.service import EventEvidenceService


class RemediationAction(StrEnum):
    PERSIST_READY = "persist_ready"
    ACKNOWLEDGE_MISSING = "acknowledge_missing"
    ACKNOWLEDGE_ORPHANED = "acknowledge_orphaned"
    ESCALATE_INTEGRITY_FAILURE = "escalate_integrity_failure"


class RemediationRecord(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    target_id: str
    action: RemediationAction
    operator: str
    reason: str
    previous_status: str
    outcome_status: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RemediationRepository(Protocol):
    def append(self, record: RemediationRecord) -> None: ...

    def replace(self, record: RemediationRecord) -> None: ...

    def list(self, *, target_id: str | None = None) -> list[RemediationRecord]: ...


class InMemoryRemediationRepository:
    def __init__(self) -> None:
        self._records: list[RemediationRecord] = []
        self._lock = Lock()

    def append(self, record: RemediationRecord) -> None:
        with self._lock:
            self._records.append(record.model_copy(deep=True))

    def replace(self, record: RemediationRecord) -> None:
        with self._lock:
            for index, existing in enumerate(self._records):
                if existing.id == record.id:
                    self._records[index] = record.model_copy(deep=True)
                    return
        raise KeyError(str(record.id))

    def list(self, *, target_id: str | None = None) -> list[RemediationRecord]:
        with self._lock:
            return [
                record.model_copy(deep=True)
                for record in self._records
                if target_id is None or record.target_id == target_id
            ]


class SQLiteRemediationRepository:
    def __init__(self, path: Path):
        self.path = path
        self._lock = Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS evidence_remediations (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT NOT NULL UNIQUE,
                    target_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS evidence_remediations_target
                ON evidence_remediations (target_id, sequence)
                """
            )

    def append(self, record: RemediationRecord) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO evidence_remediations
                    (record_id, target_id, action, payload)
                VALUES (?, ?, ?, ?)
                """,
                (
                    str(record.id),
                    record.target_id,
                    record.action.value,
                    record.model_dump_json(),
                ),
            )

    def replace(self, record: RemediationRecord) -> None:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE evidence_remediations
                SET target_id = ?, action = ?, payload = ?
                WHERE record_id = ?
                """,
                (
                    record.target_id,
                    record.action.value,
                    record.model_dump_json(),
                    str(record.id),
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(str(record.id))

    def list(self, *, target_id: str | None = None) -> list[RemediationRecord]:
        parameters: tuple[str, ...] = ()
        where = ""
        if target_id is not None:
            where = "WHERE target_id = ?"
            parameters = (target_id,)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT payload FROM evidence_remediations
                {where}
                ORDER BY sequence
                """,
                parameters,
            ).fetchall()
        return [RemediationRecord.model_validate_json(row[0]) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5)


class EvidenceRemediationService:
    """Perform explicit recovery actions without deleting or fabricating evidence."""

    def __init__(
        self,
        consistency: EvidenceConsistencyService,
        evidence: EventEvidenceService,
        repository: RemediationRepository | None = None,
    ) -> None:
        self.consistency = consistency
        self.evidence = evidence
        self.repository = repository or InMemoryRemediationRepository()
        self._lock = Lock()

    def execute(
        self,
        target_id: str,
        action: RemediationAction,
        operator: str,
        reason: str,
    ) -> RemediationRecord:
        if not target_id.strip() or not operator.strip() or not reason.strip():
            raise ValueError("target_id, operator and reason are required")
        with self._lock:
            previous = self._status(target_id)
            expected = {
                RemediationAction.PERSIST_READY: "ready",
                RemediationAction.ACKNOWLEDGE_MISSING: "missing",
                RemediationAction.ACKNOWLEDGE_ORPHANED: "orphaned",
                RemediationAction.ESCALATE_INTEGRITY_FAILURE: "integrity_failed",
            }[action]
            if previous != expected:
                raise ValueError(
                    f"action {action.value} requires status {expected}, got {previous}"
                )
            record = RemediationRecord(
                target_id=target_id,
                action=action,
                operator=operator.strip(),
                reason=reason.strip(),
                previous_status=previous,
                outcome_status="in_progress",
            )
            self.repository.append(record)
            try:
                outcome = previous
                if action is RemediationAction.PERSIST_READY:
                    self.evidence.force_persist(target_id)
                    outcome = self.evidence.status(target_id)["status"]
                    if outcome != "stored":
                        raise RuntimeError("evidence persistence did not produce stored evidence")
            except Exception:
                record.outcome_status = "failed"
                self.repository.replace(record)
                raise
            record.outcome_status = outcome
            self.repository.replace(record)
            return record.model_copy(deep=True)

    def list(self, *, target_id: str | None = None) -> list[RemediationRecord]:
        return self.repository.list(target_id=target_id)

    def _status(self, target_id: str) -> str:
        report = self.consistency.inspect()
        event_status = next(
            (link.evidence_status for link in report.events if link.event_id == target_id),
            None,
        )
        if event_status is not None:
            return event_status
        if target_id in report.orphaned_evidence_ids:
            return "orphaned"
        return "unknown"
