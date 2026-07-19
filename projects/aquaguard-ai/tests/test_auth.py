import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from aquaguard.auth import (
    AuthenticatedOperator,
    OperatorAuthenticator,
    OperatorCredential,
    OperatorRole,
    Permission,
    is_authorized,
)


TOKEN = "a-strong-local-operator-token-value-123456789"


def credential(role: OperatorRole = OperatorRole.MAINTAINER) -> OperatorCredential:
    return OperatorCredential(
        username="operator-1",
        role=role,
        token_sha256=hashlib.sha256(TOKEN.encode()).hexdigest(),
    )


def test_authenticator_accepts_fingerprinted_bearer_token() -> None:
    operator = OperatorAuthenticator((credential(),)).authenticate(f"Bearer {TOKEN}")

    assert operator is not None
    assert operator.username == "operator-1"
    assert operator.role == OperatorRole.MAINTAINER


@pytest.mark.parametrize(
    "authorization",
    (None, "Basic abc", "Bearer short", f"Bearer {TOKEN}x"),
)
def test_authenticator_rejects_invalid_credentials(authorization) -> None:
    assert OperatorAuthenticator((credential(),)).authenticate(authorization) is None


def test_authenticator_rejects_duplicate_token_fingerprints() -> None:
    with pytest.raises(ValueError, match="fingerprints must be unique"):
        OperatorAuthenticator((credential(), credential()))


def test_same_operator_can_hold_two_tokens_during_rotation() -> None:
    replacement = "replacement-local-operator-token-value-123456789"
    replacement_credential = credential().model_copy(
        update={"token_sha256": hashlib.sha256(replacement.encode()).hexdigest()}
    )
    authenticator = OperatorAuthenticator((credential(), replacement_credential))

    assert authenticator.authenticate(f"Bearer {TOKEN}") is not None
    assert authenticator.authenticate(f"Bearer {replacement}") is not None


def test_authenticator_rejects_revoked_and_expired_tokens() -> None:
    now = datetime(2026, 7, 19, tzinfo=timezone.utc)
    revoked = credential().model_copy(update={"revoked": True})
    expired = credential().model_copy(update={"expires_at": now - timedelta(seconds=1)})

    assert (
        OperatorAuthenticator((revoked,), clock=lambda: now).authenticate(f"Bearer {TOKEN}") is None
    )
    assert (
        OperatorAuthenticator((expired,), clock=lambda: now).authenticate(f"Bearer {TOKEN}") is None
    )


def test_credential_expiry_requires_timezone() -> None:
    with pytest.raises(ValueError, match="must include a timezone"):
        OperatorCredential(
            username="operator-1",
            role=OperatorRole.MAINTAINER,
            token_sha256=hashlib.sha256(TOKEN.encode()).hexdigest(),
            expires_at=datetime(2026, 7, 19),
        )


@pytest.mark.parametrize(
    ("role", "permission", "allowed"),
    (
        (OperatorRole.ADMIN, Permission.REMEDIATE_EVIDENCE, True),
        (OperatorRole.MAINTAINER, Permission.VIEW_AUDIT, True),
        (OperatorRole.LIFEGUARD, Permission.MANAGE_INCIDENTS, True),
        (OperatorRole.LIFEGUARD, Permission.VIEW_AUDIT, False),
        (OperatorRole.VIEWER, Permission.VIEW_OPERATIONS, True),
        (OperatorRole.VIEWER, Permission.MANAGE_INCIDENTS, False),
        (OperatorRole.SERVICE, Permission.INGEST_OBSERVATIONS, True),
        (OperatorRole.SERVICE, Permission.VIEW_OPERATIONS, False),
    ),
)
def test_role_permission_matrix(role, permission, allowed) -> None:
    operator = AuthenticatedOperator(credential_id=credential().id, username="test", role=role)

    assert is_authorized(operator, permission) is allowed
