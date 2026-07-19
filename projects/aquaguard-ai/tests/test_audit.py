from aquaguard.audit import SQLiteEvaluationAuditRepository
from aquaguard.domain import EvaluationAudit, RiskAssessment, RiskFeatures


def audit(camera_id: str, reason: str | None = None) -> EvaluationAudit:
    return EvaluationAudit(
        camera_id=camera_id,
        track_id="track-1",
        area="pool",
        features=RiskFeatures(
            head_underwater=0,
            vertical_body=0,
            abnormal_motion=0,
            temporal_risk=0,
        ),
        assessment=RiskAssessment(score=0, level=0, confirmed=False),
        alarm_eligible=False,
        suppression_reason=reason,
    )


def test_sqlite_audit_repository_recovers_after_restart(tmp_path) -> None:
    path = tmp_path / "audit.db"
    first = SQLiteEvaluationAuditRepository(path)
    item = audit("cam-a", "risk_not_confirmed")
    first.append(item)

    restarted = SQLiteEvaluationAuditRepository(path)

    assert restarted.list() == [item]


def test_sqlite_audit_repository_filters_and_enforces_capacity(tmp_path) -> None:
    repository = SQLiteEvaluationAuditRepository(tmp_path / "audit.db", capacity=2)
    repository.append(audit("cam-old", "old"))
    kept_a = audit("cam-a", "camera_protection_level:pose_baseline")
    kept_b = audit("cam-b", "risk_not_confirmed")
    repository.append(kept_a)
    repository.append(kept_b)

    assert repository.list() == [kept_a, kept_b]
    assert repository.list(camera_id="cam-a") == [kept_a]
    assert repository.list(suppression_reason="risk_not_confirmed") == [kept_b]
    assert repository.list(limit=1) == [kept_b]
