import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from aquaguard.auth import OperatorAuthenticator, OperatorCredential, OperatorRole
from aquaguard.credentials import (
    InMemoryOperatorCredentialRepository,
    OperatorCredentialService,
    SQLiteOperatorCredentialRepository,
)


def credential(
    username: str,
    role: OperatorRole,
    token: str,
) -> OperatorCredential:
    return OperatorCredential(
        username=username,
        role=role,
        token_sha256=hashlib.sha256(token.encode()).hexdigest(),
    )


def test_sqlite_credential_repository_recovers_and_revokes(tmp_path) -> None:
    path = tmp_path / "credentials.db"
    repository = SQLiteOperatorCredentialRepository(path)
    stored = credential("viewer-1", OperatorRole.VIEWER, "viewer-token-value-12345678901234567890")
    repository.append(stored)

    recovered = SQLiteOperatorCredentialRepository(path)
    revoked = recovered.revoke_managed(
        stored.id, uuid4(), datetime(2026, 7, 19, tzinfo=timezone.utc)
    )

    assert revoked is not None
    assert revoked.revoked is True
    assert SQLiteOperatorCredentialRepository(path).list()[0].revoked is True


def test_sqlite_credential_version_changes_only_on_mutation(tmp_path) -> None:
    repository = SQLiteOperatorCredentialRepository(tmp_path / "credentials.db")
    stored = credential("viewer-1", OperatorRole.VIEWER, "viewer-token-value-12345678901234567890")
    now = datetime(2026, 7, 19, tzinfo=timezone.utc)
    actor_id = uuid4()

    assert repository.version() == 0
    repository.append(stored)
    assert repository.version() == 1
    repository.revoke_managed(stored.id, actor_id, now)
    assert repository.version() == 2
    with pytest.raises(ValueError, match="already revoked"):
        repository.revoke_managed(stored.id, actor_id, now)
    assert repository.version() == 2


def test_sqlite_seed_if_empty_is_atomic_across_repository_instances(tmp_path) -> None:
    path = tmp_path / "credentials.db"
    first = SQLiteOperatorCredentialRepository(path)
    second = SQLiteOperatorCredentialRepository(path)
    admin = credential("admin-1", OperatorRole.ADMIN, "admin-token-value-12345678901234567890")

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(lambda repository: repository.seed_if_empty((admin,)), (first, second))
        )

    assert sorted(results) == [False, True]
    assert first.version() == 1
    assert [item.id for item in first.list()] == [admin.id]


def test_sqlite_services_refresh_credentials_changed_by_another_process(tmp_path) -> None:
    path = tmp_path / "credentials.db"
    admin = credential("admin-1", OperatorRole.ADMIN, "admin-token-value-12345678901234567890")
    first_repository = SQLiteOperatorCredentialRepository(path)
    second_repository = SQLiteOperatorCredentialRepository(path)
    assert first_repository.seed_if_empty((admin,)) is True
    first_authenticator = OperatorAuthenticator()
    second_authenticator = OperatorAuthenticator()
    first_service = OperatorCredentialService(first_repository, first_authenticator)
    second_service = OperatorCredentialService(second_repository, second_authenticator)
    viewer_token = "viewer-token-value-12345678901234567890"
    viewer = credential("viewer-1", OperatorRole.VIEWER, viewer_token)

    first_service.create(viewer)
    assert second_authenticator.authenticate(f"Bearer {viewer_token}") is None
    assert second_service.refresh_if_changed() is True
    assert second_authenticator.authenticate(f"Bearer {viewer_token}") is not None
    assert second_service.refresh_if_changed() is False

    first_service.revoke(viewer.id, admin.id)
    assert second_service.refresh_if_changed() is True
    assert second_authenticator.authenticate(f"Bearer {viewer_token}") is None


def test_concurrent_sqlite_admin_revocation_preserves_one_active_admin(tmp_path) -> None:
    path = tmp_path / "credentials.db"
    now = datetime(2026, 7, 19, tzinfo=timezone.utc)
    first = credential("admin-1", OperatorRole.ADMIN, "admin-token-value-12345678901234567890")
    second = credential("admin-2", OperatorRole.ADMIN, "second-token-value-1234567890123456789")
    first_repository = SQLiteOperatorCredentialRepository(path)
    second_repository = SQLiteOperatorCredentialRepository(path)
    assert first_repository.seed_if_empty((first, second)) is True
    first_service = OperatorCredentialService(
        first_repository, OperatorAuthenticator(), clock=lambda: now
    )
    second_service = OperatorCredentialService(
        second_repository, OperatorAuthenticator(), clock=lambda: now
    )

    def revoke_other(service, target_id, actor_id):
        try:
            service.revoke(target_id, actor_id)
            return "revoked"
        except ValueError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(revoke_other, first_service, second.id, first.id),
            executor.submit(revoke_other, second_service, first.id, second.id),
        )
        results = [future.result() for future in futures]

    assert results.count("revoked") == 1
    assert results.count("cannot revoke the last active admin credential") == 1
    active_admins = [
        item
        for item in SQLiteOperatorCredentialRepository(path).list()
        if item.role == OperatorRole.ADMIN and not item.revoked
    ]
    assert len(active_admins) == 1


def test_credential_service_refreshes_authenticator_without_restart() -> None:
    admin = credential("admin-1", OperatorRole.ADMIN, "admin-token-value-12345678901234567890")
    viewer_token = "viewer-token-value-12345678901234567890"
    repository = InMemoryOperatorCredentialRepository((admin,))
    authenticator = OperatorAuthenticator()
    service = OperatorCredentialService(repository, authenticator)
    viewer = credential("viewer-1", OperatorRole.VIEWER, viewer_token)

    view = service.create(viewer)

    assert view.id == viewer.id
    assert "token" not in view.model_dump()
    assert authenticator.authenticate(f"Bearer {viewer_token}") is not None

    service.revoke(viewer.id, admin.id)

    assert authenticator.authenticate(f"Bearer {viewer_token}") is None


def test_credential_service_prevents_self_and_last_admin_revocation() -> None:
    now = datetime(2026, 7, 19, tzinfo=timezone.utc)
    first = credential("admin-1", OperatorRole.ADMIN, "admin-token-value-12345678901234567890")
    second = credential("admin-2", OperatorRole.ADMIN, "second-token-value-1234567890123456789")
    repository = InMemoryOperatorCredentialRepository((first, second))
    service = OperatorCredentialService(repository, OperatorAuthenticator(), clock=lambda: now)

    with pytest.raises(ValueError, match="cannot revoke their current"):
        service.revoke(first.id, first.id)

    service.revoke(second.id, first.id)

    with pytest.raises(ValueError, match="last active admin"):
        service.revoke(first.id, second.id)


def test_credential_repository_rejects_duplicate_fingerprint() -> None:
    first = credential("admin-1", OperatorRole.ADMIN, "admin-token-value-12345678901234567890")
    duplicate = first.model_copy(update={"id": uuid4()})
    repository = InMemoryOperatorCredentialRepository((first,))

    with pytest.raises(ValueError, match="fingerprint already exists"):
        repository.append(duplicate)


def test_credential_service_rejects_already_expired_credential() -> None:
    now = datetime(2026, 7, 19, tzinfo=timezone.utc)
    repository = InMemoryOperatorCredentialRepository()
    service = OperatorCredentialService(repository, OperatorAuthenticator(), clock=lambda: now)
    expired = credential(
        "viewer-1", OperatorRole.VIEWER, "viewer-token-value-12345678901234567890"
    ).model_copy(update={"expires_at": now})

    with pytest.raises(ValueError, match="expire in the future"):
        service.create(expired)

    assert repository.list() == []
