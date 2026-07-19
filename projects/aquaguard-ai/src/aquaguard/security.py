import sqlite3
from collections import OrderedDict, deque
from collections.abc import Callable
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from aquaguard.auth import OperatorRole, Permission


class SecurityOutcome(StrEnum):
    AUTHENTICATION_FAILED = "authentication_failed"
    AUTHORIZATION_DENIED = "authorization_denied"
    RATE_LIMITED = "rate_limited"
    CREDENTIAL_CREATED = "credential_created"
    CREDENTIAL_REVOKED = "credential_revoked"
    CREDENTIAL_CHANGE_REJECTED = "credential_change_rejected"


class SecurityAudit(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    outcome: SecurityOutcome
    permission: Permission
    path: str
    client_host: str
    operator: str | None = None
    role: OperatorRole | None = None
    target_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SecurityAuditRepository(Protocol):
    def append(self, audit: SecurityAudit) -> None: ...

    def list(
        self,
        *,
        limit: int | None = None,
        outcome: SecurityOutcome | None = None,
    ) -> list[SecurityAudit]: ...


class InMemorySecurityAuditRepository:
    def __init__(self, capacity: int = 1000) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._audits: deque[SecurityAudit] = deque(maxlen=capacity)
        self._lock = Lock()

    def append(self, audit: SecurityAudit) -> None:
        with self._lock:
            self._audits.append(audit.model_copy(deep=True))

    def list(
        self,
        *,
        limit: int | None = None,
        outcome: SecurityOutcome | None = None,
    ) -> list[SecurityAudit]:
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        with self._lock:
            items = [
                audit.model_copy(deep=True)
                for audit in self._audits
                if outcome is None or audit.outcome == outcome
            ]
        return items[-limit:] if limit is not None else items


class SQLiteSecurityAuditRepository:
    def __init__(self, path: Path, capacity: int = 1000) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.path = path
        self.capacity = capacity
        self._lock = Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS security_audits (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    audit_id TEXT NOT NULL UNIQUE,
                    outcome TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )

    def append(self, audit: SecurityAudit) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO security_audits (audit_id, outcome, payload)
                VALUES (?, ?, ?)
                """,
                (str(audit.id), audit.outcome.value, audit.model_dump_json()),
            )
            connection.execute(
                """
                DELETE FROM security_audits
                WHERE sequence NOT IN (
                    SELECT sequence FROM security_audits
                    ORDER BY sequence DESC LIMIT ?
                )
                """,
                (self.capacity,),
            )

    def list(
        self,
        *,
        limit: int | None = None,
        outcome: SecurityOutcome | None = None,
    ) -> list[SecurityAudit]:
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        conditions = "WHERE outcome = ?" if outcome is not None else ""
        parameters: list[object] = [outcome.value] if outcome is not None else []
        limit_sql = "LIMIT ?" if limit is not None else ""
        if limit is not None:
            parameters.append(limit)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT payload FROM security_audits
                {conditions}
                ORDER BY sequence DESC
                {limit_sql}
                """,
                parameters,
            ).fetchall()
        return [SecurityAudit.model_validate_json(row[0]) for row in reversed(rows)]

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5)


class AuthenticationFailureRateLimiter:
    def __init__(
        self,
        max_failures: int = 10,
        window_seconds: float = 60,
        max_clients: int = 10_000,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if max_failures < 1 or window_seconds <= 0 or max_clients < 1:
            raise ValueError("rate limit settings must be positive")
        self.max_failures = max_failures
        self.window_seconds = window_seconds
        self.max_clients = max_clients
        self.clock = clock
        self._failures: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = Lock()

    def retry_after(self, client_id: str) -> float | None:
        with self._lock:
            now = self.clock()
            failures = self._active_failures(client_id, now, create=False)
            if failures is None:
                return None
            if len(failures) < self.max_failures:
                return None
            return max(0.0, self.window_seconds - (now - failures[0]))

    def record_failure(self, client_id: str) -> None:
        with self._lock:
            now = self.clock()
            failures = self._active_failures(client_id, now, create=True)
            assert failures is not None
            failures.append(now)
            self._failures.move_to_end(client_id)
            while len(self._failures) > self.max_clients:
                self._failures.popitem(last=False)

    def _active_failures(self, client_id: str, now: float, *, create: bool) -> deque[float] | None:
        failures = self._failures.get(client_id)
        if failures is None:
            if not create:
                return None
            failures = deque()
            self._failures[client_id] = failures
        cutoff = now - self.window_seconds
        while failures and failures[0] <= cutoff:
            failures.popleft()
        if not failures and not create:
            del self._failures[client_id]
            return None
        return failures
