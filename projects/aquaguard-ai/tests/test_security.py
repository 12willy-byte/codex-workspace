from aquaguard.auth import OperatorRole, Permission
from aquaguard.security import (
    AuthenticationFailureRateLimiter,
    InMemorySecurityAuditRepository,
    SQLiteSecurityAuditRepository,
    SecurityAudit,
    SecurityOutcome,
)


def audit(
    outcome: SecurityOutcome = SecurityOutcome.AUTHENTICATION_FAILED,
) -> SecurityAudit:
    return SecurityAudit(
        outcome=outcome,
        permission=Permission.VIEW_OPERATIONS,
        path="/api/v1/events",
        client_host="127.0.0.1",
    )


def test_in_memory_security_audit_is_bounded_and_filtered() -> None:
    repository = InMemorySecurityAuditRepository(capacity=2)
    repository.append(audit())
    repository.append(audit(SecurityOutcome.AUTHORIZATION_DENIED))
    repository.append(audit(SecurityOutcome.RATE_LIMITED))

    assert [item.outcome for item in repository.list()] == [
        SecurityOutcome.AUTHORIZATION_DENIED,
        SecurityOutcome.RATE_LIMITED,
    ]
    assert repository.list(outcome=SecurityOutcome.RATE_LIMITED)[0].role is None


def test_sqlite_security_audit_recovers_filters_and_caps(tmp_path) -> None:
    path = tmp_path / "security.db"
    repository = SQLiteSecurityAuditRepository(path, capacity=2)
    repository.append(audit())
    repository.append(audit(SecurityOutcome.AUTHORIZATION_DENIED))
    repository.append(audit(SecurityOutcome.RATE_LIMITED))

    recovered = SQLiteSecurityAuditRepository(path, capacity=2)

    assert [item.outcome for item in recovered.list()] == [
        SecurityOutcome.AUTHORIZATION_DENIED,
        SecurityOutcome.RATE_LIMITED,
    ]
    assert recovered.list(outcome=SecurityOutcome.AUTHENTICATION_FAILED) == []


def test_authentication_failure_rate_limit_expires() -> None:
    now = 0.0
    limiter = AuthenticationFailureRateLimiter(
        max_failures=2,
        window_seconds=10,
        max_clients=5,
        clock=lambda: now,
    )

    limiter.record_failure("client-a")
    limiter.record_failure("client-a")

    assert limiter.retry_after("client-a") == 10
    now = 10.0
    assert limiter.retry_after("client-a") is None


def test_rate_limiter_bounds_failed_clients_without_tracking_clean_clients() -> None:
    limiter = AuthenticationFailureRateLimiter(max_clients=2)

    for index in range(100):
        assert limiter.retry_after(f"clean-{index}") is None
    for client in ("failed-1", "failed-2", "failed-3"):
        limiter.record_failure(client)

    assert len(limiter._failures) == 2
    assert "failed-1" not in limiter._failures


def test_security_audit_can_include_known_denied_operator() -> None:
    denied = audit(SecurityOutcome.AUTHORIZATION_DENIED).model_copy(
        update={"operator": "lifeguard-1", "role": OperatorRole.LIFEGUARD}
    )

    assert denied.operator == "lifeguard-1"
    assert denied.role == OperatorRole.LIFEGUARD
