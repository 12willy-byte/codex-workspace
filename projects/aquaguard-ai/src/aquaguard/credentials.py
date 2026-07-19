from __future__ import annotations

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

    def seed_if_empty(self, credentials: tuple[OperatorCredential, ...]) -> bool: ...

    def version(self) -> int: ...

    def snapshot(self) -> tuple[int, list[OperatorCredential]]: ...

    def list(self) -> list[OperatorCredential]: ...


class InMemoryOperatorCredentialRepository:
    def __init__(self, credentials: tuple[OperatorCredential, ...] = ()) -> None:
        self._credentials: list[OperatorCredential] = []
        self._version = 0
        self._lock = RLock()
        for credential in credentials:
            self.append(credential)

    def append(self, credential: OperatorCredential) -> None:
        with self._lock:
            self._validate_unique(credential)
            self._credentials.append(credential.model_copy(deep=True))
            self._version += 1

    def revoke(self, credential_id: UUID) -> OperatorCredential | None:
        with self._lock:
            for credential in self._credentials:
                if credential.id == credential_id:
                    if credential.revoked:
                        return credential.model_copy(deep=True)
                    credential.revoked = True
                    self._version += 1
                    return credential.model_copy(deep=True)
        return None

    def seed_if_empty(self, credentials: tuple[OperatorCredential, ...]) -> bool:
        with self._lock:
            if self._credentials:
                return False
            staged: list[OperatorCredential] = []
            for credential in credentials:
                self._validate_unique_in(credential, staged)
                staged.append(credential.model_copy(deep=True))
            self._credentials.extend(staged)
            if staged:
                self._version += 1
            return True

    def version(self) -> int:
        with self._lock:
            return self._version

    def snapshot(self) -> tuple[int, list[OperatorCredential]]:
        with self._lock:
            return self._version, [
                credential.model_copy(deep=True) for credential in self._credentials
            ]

    def list(self) -> list[OperatorCredential]:
        return self.snapshot()[1]

    def _validate_unique(self, credential: OperatorCredential) -> None:
        self._validate_unique_in(credential, self._credentials)

    @staticmethod
    def _validate_unique_in(
        credential: OperatorCredential, credentials: list[OperatorCredential]
    ) -> None:
        if any(item.id == credential.id for item in credentials):
            raise ValueError("credential id already exists")
        if any(item.token_sha256 == credential.token_sha256 for item in credentials):
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
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS operator_credential_metadata (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    version INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO operator_credential_metadata (singleton, version)
                VALUES (1, 0)
                """
            )

    def append(self, credential: OperatorCredential) -> None:
        connection = self._connect()
        try:
            with self._lock:
                connection.execute("BEGIN IMMEDIATE")
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
                self._increment_version(connection)
                connection.commit()
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise ValueError("credential id or token fingerprint already exists") from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def revoke(self, credential_id: UUID) -> OperatorCredential | None:
        connection = self._connect()
        try:
            with self._lock:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT payload FROM operator_credentials WHERE credential_id = ?",
                    (str(credential_id),),
                ).fetchone()
                if row is None:
                    connection.rollback()
                    return None
                credential = OperatorCredential.model_validate_json(row[0])
                if credential.revoked:
                    connection.rollback()
                    return credential.model_copy(deep=True)
                credential.revoked = True
                connection.execute(
                    "UPDATE operator_credentials SET payload = ? WHERE credential_id = ?",
                    (credential.model_dump_json(), str(credential_id)),
                )
                self._increment_version(connection)
                connection.commit()
                return credential.model_copy(deep=True)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def seed_if_empty(self, credentials: tuple[OperatorCredential, ...]) -> bool:
        connection = self._connect()
        try:
            with self._lock:
                connection.execute("BEGIN IMMEDIATE")
                count = connection.execute("SELECT COUNT(*) FROM operator_credentials").fetchone()[
                    0
                ]
                if count:
                    connection.rollback()
                    return False
                for credential in credentials:
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
                if credentials:
                    self._increment_version(connection)
                connection.commit()
                return True
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise ValueError("credential id or token fingerprint already exists") from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def version(self) -> int:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT version FROM operator_credential_metadata WHERE singleton = 1"
            ).fetchone()
        if row is None:
            raise RuntimeError("credential metadata is missing")
        return int(row[0])

    def snapshot(self) -> tuple[int, list[OperatorCredential]]:
        connection = self._connect()
        try:
            with self._lock:
                connection.execute("BEGIN")
                version_row = connection.execute(
                    "SELECT version FROM operator_credential_metadata WHERE singleton = 1"
                ).fetchone()
                rows = connection.execute(
                    "SELECT payload FROM operator_credentials ORDER BY sequence"
                ).fetchall()
                connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        if version_row is None:
            raise RuntimeError("credential metadata is missing")
        return int(version_row[0]), [OperatorCredential.model_validate_json(row[0]) for row in rows]

    def list(self) -> list[OperatorCredential]:
        return self.snapshot()[1]

    @staticmethod
    def _increment_version(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            UPDATE operator_credential_metadata
            SET version = version + 1
            WHERE singleton = 1
            """
        )

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
        self._version = -1
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
        self.refresh_if_changed()
        return [
            OperatorCredentialView.from_credential(credential)
            for credential in self.repository.list()
        ]

    def refresh_if_changed(self) -> bool:
        with self._lock:
            if self.repository.version() == self._version:
                return False
            self._refresh()
            return True

    def _refresh(self) -> None:
        version, credentials = self.repository.snapshot()
        self.authenticator.credentials = tuple(credentials)
        self._version = version

    def _is_active(self, credential: OperatorCredential) -> bool:
        now = self.clock()
        if now.tzinfo is None:
            raise ValueError("credential clock must return a timezone-aware datetime")
        return not credential.revoked and (
            credential.expires_at is None or now < credential.expires_at
        )
