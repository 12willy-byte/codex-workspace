import sqlite3
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Protocol
from uuid import UUID

from pydantic import BaseModel

from aquaguard.auth import OperatorAuthenticator, OperatorCredential, OperatorRole


class OperatorCredentialView(BaseModel):
    id: UUID
    username: str
    role: OperatorRole
    expires_at: datetime | None
    revoked: bool

    @classmethod
    def from_credential(cls, credential: OperatorCredential) -> "OperatorCredentialView":
        return cls(
            id=credential.id,
            username=credential.username,
            role=credential.role,
            expires_at=credential.expires_at,
            revoked=credential.revoked,
        )


class OperatorCredentialRepository(Protocol):
    def append(self, credential: OperatorCredential) -> None: ...

    def revoke(self, credential_id: UUID) -> OperatorCredential | None: ...

    def list(self) -> list[OperatorCredential]: ...


class InMemoryOperatorCredentialRepository:
    def __init__(self, credentials: tuple[OperatorCredential, ...] = ()) -> None:
        self._credentials: list[OperatorCredential] = []
        self._lock = RLock()
        for credential in credentials:
            self.append(credential)

    def append(self, credential: OperatorCredential) -> None:
        with self._lock:
            self._validate_unique(credential)
            self._credentials.append(credential.model_copy(deep=True))

    def revoke(self, credential_id: UUID) -> OperatorCredential | None:
        with self._lock:
            for credential in self._credentials:
                if credential.id == credential_id:
                    credential.revoked = True
                    return credential.model_copy(deep=True)
        return None

    def list(self) -> list[OperatorCredential]:
        with self._lock:
            return [credential.model_copy(deep=True) for credential in self._credentials]

    def _validate_unique(self, credential: OperatorCredential) -> None:
        if any(item.id == credential.id for item in self._credentials):
            raise ValueError("credential id already exists")
        if any(item.token_sha256 == credential.token_sha256 for item in self._credentials):
            raise ValueError("credential token fingerprint already exists")


class SQLiteOperatorCredentialRepository:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS operator_credentials (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    credential_id TEXT NOT NULL UNIQUE,
                    token_sha256 TEXT NOT NULL UNIQUE,
                    payload TEXT NOT NULL
                )
                """
            )

    def append(self, credential: OperatorCredential) -> None:
        try:
            with self._lock, self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO operator_credentials
                        (credential_id, token_sha256, payload)
                    VALUES (?, ?, ?)
                    """,
                    (
                        str(credential.id),
                        credential.token_sha256,
                        credential.model_dump_json(),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("credential id or token fingerprint already exists") from exc

    def revoke(self, credential_id: UUID) -> OperatorCredential | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM operator_credentials WHERE credential_id = ?",
                (str(credential_id),),
            ).fetchone()
            if row is None:
                return None
            credential = OperatorCredential.model_validate_json(row[0])
            credential.revoked = True
            connection.execute(
                "UPDATE operator_credentials SET payload = ? WHERE credential_id = ?",
                (credential.model_dump_json(), str(credential_id)),
            )
            return credential.model_copy(deep=True)

    def list(self) -> list[OperatorCredential]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM operator_credentials ORDER BY sequence"
            ).fetchall()
        return [OperatorCredential.model_validate_json(row[0]) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5)


class OperatorCredentialService:
    def __init__(
        self,
        repository: OperatorCredentialRepository,
        authenticator: OperatorAuthenticator,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.repository = repository
        self.authenticator = authenticator
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = RLock()
        self._refresh()

    def create(self, credential: OperatorCredential) -> OperatorCredentialView:
        with self._lock:
            if credential.expires_at is not None and not self._is_active(credential):
                raise ValueError("new credential must expire in the future")
            self.repository.append(credential)
            self._refresh()
            return OperatorCredentialView.from_credential(credential)

    def revoke(
        self, credential_id: UUID, actor_credential_id: UUID
    ) -> OperatorCredentialView | None:
        with self._lock:
            credentials = self.repository.list()
            target = next((item for item in credentials if item.id == credential_id), None)
            if target is None:
                return None
            if target.id == actor_credential_id:
                raise ValueError("operators cannot revoke their current credential")
            if target.revoked:
                raise ValueError("credential is already revoked")
            if target.role == OperatorRole.ADMIN and self._is_active(target):
                active_admins = [
                    item
                    for item in credentials
                    if item.role == OperatorRole.ADMIN and self._is_active(item)
                ]
                if len(active_admins) <= 1:
                    raise ValueError("cannot revoke the last active admin credential")
            revoked = self.repository.revoke(credential_id)
            if revoked is None:
                return None
            self._refresh()
            return OperatorCredentialView.from_credential(revoked)

    def list(self) -> list[OperatorCredentialView]:
        return [
            OperatorCredentialView.from_credential(credential)
            for credential in self.repository.list()
        ]

    def _refresh(self) -> None:
        self.authenticator.credentials = tuple(self.repository.list())

    def _is_active(self, credential: OperatorCredential) -> bool:
        now = self.clock()
        if now.tzinfo is None:
            raise ValueError("credential clock must return a timezone-aware datetime")
        return not credential.revoked and (
            credential.expires_at is None or now < credential.expires_at
        )
