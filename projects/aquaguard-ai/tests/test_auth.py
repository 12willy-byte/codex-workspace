import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from aquaguard.auth import OperatorAuthenticator, OperatorCredential, OperatorRole


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
