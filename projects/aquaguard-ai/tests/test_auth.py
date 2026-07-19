import hashlib

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


def test_authenticator_rejects_duplicate_operator_names() -> None:
    with pytest.raises(ValueError, match="usernames must be unique"):
        OperatorAuthenticator((credential(), credential()))
