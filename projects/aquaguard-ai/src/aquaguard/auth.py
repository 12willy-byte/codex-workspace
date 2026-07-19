import hashlib
import hmac
from collections.abc import Callable
from datetime import datetime, timezone
from enum import StrEnum
from threading import RLock
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class OperatorRole(StrEnum):
    ADMIN = "admin"
    MAINTAINER = "maintainer"
    LIFEGUARD = "lifeguard"
    VIEWER = "viewer"
    SERVICE = "service"


class Permission(StrEnum):
    VIEW_OPERATIONS = "view_operations"
    MANAGE_INCIDENTS = "manage_incidents"
    VIEW_AUDIT = "view_audit"
    VIEW_SECURITY_AUDIT = "view_security_audit"
    REMEDIATE_EVIDENCE = "remediate_evidence"
    INGEST_OBSERVATIONS = "ingest_observations"
    MANAGE_CREDENTIALS = "manage_credentials"


ROLE_PERMISSIONS: dict[OperatorRole, frozenset[Permission]] = {
    OperatorRole.ADMIN: frozenset(Permission),
    OperatorRole.MAINTAINER: frozenset(
        {
            Permission.VIEW_OPERATIONS,
            Permission.MANAGE_INCIDENTS,
            Permission.VIEW_AUDIT,
            Permission.REMEDIATE_EVIDENCE,
        }
    ),
    OperatorRole.LIFEGUARD: frozenset({Permission.VIEW_OPERATIONS, Permission.MANAGE_INCIDENTS}),
    OperatorRole.VIEWER: frozenset({Permission.VIEW_OPERATIONS}),
    OperatorRole.SERVICE: frozenset({Permission.INGEST_OBSERVATIONS}),
}


class OperatorCredential(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    username: str = Field(min_length=1)
    role: OperatorRole
    token_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expires_at: datetime | None = None
    revoked: bool = False

    @field_validator("username")
    @classmethod
    def normalize_username(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("username must not be blank")
        return value

    @field_validator("expires_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("expires_at must include a timezone")
        return value


class AuthenticatedOperator(BaseModel):
    credential_id: UUID
    username: str
    role: OperatorRole


class OperatorAuthenticator:
    """Authenticate opaque bearer tokens against configured SHA-256 fingerprints."""

    def __init__(
        self,
        credentials: tuple[OperatorCredential, ...] = (),
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._lock = RLock()
        self._credentials: tuple[OperatorCredential, ...] = ()
        self.credentials = credentials
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    @property
    def credentials(self) -> tuple[OperatorCredential, ...]:
        with self._lock:
            return tuple(credential.model_copy(deep=True) for credential in self._credentials)

    @credentials.setter
    def credentials(self, credentials: tuple[OperatorCredential, ...]) -> None:
        fingerprints = [credential.token_sha256 for credential in credentials]
        ids = [credential.id for credential in credentials]
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError("operator token fingerprints must be unique")
        if len(ids) != len(set(ids)):
            raise ValueError("operator credential ids must be unique")
        with self._lock:
            self._credentials = tuple(
                credential.model_copy(deep=True) for credential in credentials
            )

    def authenticate(self, authorization: str | None) -> AuthenticatedOperator | None:
        if authorization is None or not authorization.startswith("Bearer "):
            return None
        token = authorization.removeprefix("Bearer ")
        if len(token) < 32:
            return None
        fingerprint = hashlib.sha256(token.encode()).hexdigest()
        now = self.clock()
        if now.tzinfo is None:
            raise ValueError("authentication clock must return a timezone-aware datetime")
        matched: OperatorCredential | None = None
        for credential in self.credentials:
            fingerprint_matches = hmac.compare_digest(fingerprint, credential.token_sha256)
            active = not credential.revoked and (
                credential.expires_at is None or now < credential.expires_at
            )
            if fingerprint_matches and active:
                matched = credential
        if matched is None:
            return None
        return AuthenticatedOperator(
            credential_id=matched.id,
            username=matched.username,
            role=matched.role,
        )


def is_authorized(operator: AuthenticatedOperator, permission: Permission) -> bool:
    return permission in ROLE_PERMISSIONS[operator.role]
