import json

from aquaguard.auth import OperatorRole
from aquaguard.config import Settings


def test_operator_credentials_load_from_environment_json(monkeypatch) -> None:
    fingerprint = "a" * 64
    monkeypatch.setenv(
        "AQUAGUARD_OPERATOR_CREDENTIALS",
        json.dumps(
            [
                {
                    "username": "maintainer-1",
                    "role": "maintainer",
                    "token_sha256": fingerprint,
                    "expires_at": "2027-01-01T00:00:00Z",
                    "revoked": False,
                }
            ]
        ),
    )

    settings = Settings(_env_file=None)

    assert len(settings.operator_credentials) == 1
    assert settings.operator_credentials[0].username == "maintainer-1"
    assert settings.operator_credentials[0].role == OperatorRole.MAINTAINER
    assert settings.operator_credentials[0].token_sha256 == fingerprint
