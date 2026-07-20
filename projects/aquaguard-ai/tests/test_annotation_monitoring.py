import json
from datetime import datetime, timedelta, timezone

import pytest

from aquaguard.annotation_monitoring_cli import main
from aquaguard.vision import (
    AnnotationAgreementReport,
    AnnotationBatchAdmissionReport,
    AnnotationBatchQualitySnapshot,
    ContinuousAnnotationQualityMonitor,
    ContinuousAnnotationQualityPolicy,
)

NOW = datetime(2026, 7, 19, 12, tzinfo=timezone.utc)
PROTOCOL_SHA256 = "a" * 64


def policy() -> ContinuousAnnotationQualityPolicy:
    return ContinuousAnnotationQualityPolicy(
        recent_window_batches=2,
        minimum_batches_before_evaluation=2,
        minimum_weighted_observed_agreement=0.8,
        minimum_weighted_cohen_kappa=0.6,
        maximum_weighted_uncertain_fraction=0.1,
        maximum_consecutive_denied_batches=1,
        minimum_alerts_for_retraining=2,
    )


def snapshot(
    batch_number: int,
    *,
    observed: float = 1,
    kappa: float | None = 1,
    uncertain_items: int = 0,
    admitted: bool = True,
    items: int = 10,
) -> AnnotationBatchQualitySnapshot:
    completed_at = NOW + timedelta(hours=batch_number)
    agreed_items = round(observed * items)
    observed = agreed_items / items
    expected_agreement = 1.0 if kappa is None else 0.5
    if kappa is not None:
        expected_agreement = (observed - kappa) / (1 - kappa) if kappa != 1 else 0.5
    agreement = AnnotationAgreementReport(
        reviewer_ids=("reviewer-a", "reviewer-b"),
        items=items,
        agreed_items=agreed_items,
        disputed_items=items - agreed_items,
        uncertain_items=uncertain_items,
        observed_agreement=observed,
        expected_agreement=expected_agreement,
        cohen_kappa=kappa,
    )
    admission = AnnotationBatchAdmissionReport(
        checked_at=completed_at,
        protocol_id="pool-risk",
        annotation_version="1.0.0",
        protocol_sha256=PROTOCOL_SHA256,
        reviewer_ids=agreement.reviewer_ids,
        admitted=admitted,
        denial_reasons=() if admitted else ("batch_quality_failed",),
        agreement=agreement,
    )
    return AnnotationBatchQualitySnapshot(
        batch_id=f"batch-{batch_number}",
        completed_at=completed_at,
        admission=admission,
    )


def test_monitor_reports_healthy_weighted_window_without_retraining() -> None:
    report = ContinuousAnnotationQualityMonitor().evaluate(
        [snapshot(1, items=5), snapshot(2, observed=0.9, kappa=0.8, items=15)],
        policy(),
        evaluated_at=NOW + timedelta(hours=3),
    )

    assert report.weighted_observed_agreement == pytest.approx(0.95)
    assert report.weighted_cohen_kappa == pytest.approx(0.85)
    assert report.alerts == ()
    assert report.retraining_required is False


def test_monitor_detects_degradation_and_triggers_retraining_review() -> None:
    history = [
        snapshot(1),
        snapshot(2),
        snapshot(3, observed=0.6, kappa=0.4, uncertain_items=3, admitted=False),
        snapshot(4, observed=0.6, kappa=0.4, uncertain_items=3, admitted=False),
    ]

    report = ContinuousAnnotationQualityMonitor().evaluate(
        history, policy(), evaluated_at=NOW + timedelta(hours=5)
    )

    assert report.observed_agreement_delta == pytest.approx(-0.4)
    assert report.cohen_kappa_delta == pytest.approx(-0.6)
    assert report.uncertain_fraction_delta == pytest.approx(0.3)
    assert report.consecutive_denied_batches == 2
    assert set(report.alerts) == {
        "observed_agreement_below_minimum",
        "cohen_kappa_below_minimum_or_unavailable",
        "uncertain_fraction_above_maximum",
        "consecutive_denied_batches_above_maximum",
    }
    assert report.retraining_required is True


def test_monitor_fails_closed_when_history_or_kappa_is_unavailable() -> None:
    insufficient = ContinuousAnnotationQualityMonitor().evaluate(
        [snapshot(1)], policy(), evaluated_at=NOW + timedelta(hours=2)
    )
    unavailable = ContinuousAnnotationQualityMonitor().evaluate(
        [snapshot(1), snapshot(2, kappa=None)],
        policy(),
        evaluated_at=NOW + timedelta(hours=3),
    )

    assert insufficient.alerts == ("insufficient_batch_history",)
    assert insufficient.retraining_required is False
    assert unavailable.weighted_cohen_kappa is None
    assert "cohen_kappa_below_minimum_or_unavailable" in unavailable.alerts


def test_monitor_rejects_mixed_or_misordered_history() -> None:
    first = snapshot(1)
    second = snapshot(2)
    other_reviewers = second.model_copy(
        update={
            "admission": second.admission.model_copy(
                update={"reviewer_ids": ("reviewer-a", "reviewer-c")}
            )
        }
    )

    with pytest.raises(ValueError, match="ascending"):
        ContinuousAnnotationQualityMonitor().evaluate(
            [second, first], policy(), evaluated_at=NOW + timedelta(hours=3)
        )
    with pytest.raises(ValueError, match="reviewer pair"):
        ContinuousAnnotationQualityMonitor().evaluate(
            [first, other_reviewers], policy(), evaluated_at=NOW + timedelta(hours=3)
        )


def test_monitoring_command_writes_report_and_returns_two_for_retraining(tmp_path) -> None:
    snapshots_path = tmp_path / "snapshots.jsonl"
    policy_path = tmp_path / "policy.json"
    report_path = tmp_path / "report.json"
    snapshots = [
        snapshot(1, observed=0.5, kappa=0.2, admitted=False),
        snapshot(2, observed=0.5, kappa=0.2, admitted=False),
    ]
    snapshots_path.write_text(
        "\n".join(item.model_dump_json() for item in snapshots) + "\n",
        encoding="utf-8",
    )
    policy_path.write_text(policy().model_dump_json(), encoding="utf-8")

    exit_code = main(
        [
            str(snapshots_path),
            str(policy_path),
            str(report_path),
            "--evaluated-at",
            (NOW + timedelta(hours=3)).isoformat(),
        ]
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert payload["retraining_required"] is True
    assert "observed_agreement_below_minimum" in payload["alerts"]
