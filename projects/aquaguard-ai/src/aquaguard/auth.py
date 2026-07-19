import hashlib
import hmac
from enum import StrEnum

from pydantic import BaseModel, Field


class OperatorRole(StrEnum):
    ADMIN = "admin"
    MAINTAINER = "maintainer"
    LIFEGUARD = "lifeguard"
    VIEWER = "viewer"


class OperatorCredential(BaseModel):
    username: str = Field(min_length=1)
    role: OperatorRole
    token_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AuthenticatedOperator(BaseModel):
    username: str
    role: OperatorRole


class OperatorAuthenticator:
    """Authenticate opaque bearer tokens against configured SHA-256 fingerprints."""

    def __init__(self, credentials: tuple[OperatorCredential, ...] = ()) -> None:
        usernames = [credential.username for credential in credentials]
        if len(usernames) != len(set(usernames)):
            raise ValueError("operator usernames must be unique")
        self.credentials = credentials

    def authenticate(self, authorization: str | None) -> AuthenticatedOperator | None:
        if authorization is None or not authorization.startswith("Bearer "):
            return None
        token = authorization.removeprefix("Bearer ")
        if len(token) < 32:
            return None
        fingerprint = hashlib.sha256(token.encode()).hexdigest()
        matched: OperatorCredential | None = None
        for credential in self.credentials:
            if hmac.compare_digest(fingerprint, credential.token_sha256):
                matched = credential
        if matched is None:
            return None
        return AuthenticatedOperator(username=matched.username, role=matched.role)
