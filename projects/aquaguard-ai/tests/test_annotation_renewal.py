import json
from datetime import datetime, timedelta, timezone

import pytest

from aquaguard.annotation_renewal_cli import main
from aquaguard.annotation_history_cli import main as history_main
from aquaguard.vision import (
    CalibrationGoldItem,
    CalibrationResponse,
    QualificationRenewalPolicy,
    QualificationHistoryVerifier,
    ReviewerCalibrationSet,
    ReviewerCalibrationSubmission,
    ReviewerQualificationEvaluator,
    ReviewerQualificationIssuer,
    ReviewerQualificationPolicy,
    ReviewerQualificationRenewalIssuer,
    ReviewerRetrainingCompletion,
    RiskJudgment,
)

NOW = datetime(2026, 7, 20, 12, tzinfo=timezone.utc)
PROTOCOL_SHA256 = "a" * 64


def calibration_set(version: str = "1.0.0") -> ReviewerCalibrationSet:
    return ReviewerCalibrationSet(
        calibration_set_id="reviewer-baseline",
        calibration_set_version=version,
        protocol_id="pool-risk",
        annotation_version="1.0.0",
        protocol_sha256=PROTOCOL_SHA256,
        items=(
            CalibrationGoldItem(item_id=f"{version}-danger", dangerous=True),
            CalibrationGoldItem(item_id=f"{version}-safe", dangerous=False),
        ),
    )


def submission(source: ReviewerCalibrationSet) -> ReviewerCalibrationSubmission:
    return ReviewerCalibrationSubmission(
        reviewer_id="reviewer-a",
        protocol_id=source.protocol_id,
        annotation_version=source.annotation_version,
        protocol_sha256=source.protocol_sha256,
        calibration_set_sha256=source.sha256(),
        responses=(
            CalibrationResponse(item_id=source.items[0].item_id, judgment=RiskJudgment.DANGEROUS),
            CalibrationResponse(item_id=source.items[1].item_id, judgment=RiskJudgment.SAFE),
        ),
    )


def qualification_policy() -> ReviewerQualificationPolicy:
    return ReviewerQualificationPolicy(
        minimum_items=2,
        minimum_dangerous_items=1,
        minimum_safe_items=1,
        minimum_accuracy=1,
        minimum_dangerous_recall=1,
        minimum_safe_specificity=1,
        maximum_uncertain_fraction=0,
    )


def initial_record():
    source = calibration_set()
    candidate = submission(source)
    policy = qualification_policy()
    report = ReviewerQualificationEvaluator().evaluate(source, candidate, policy)
    return ReviewerQualificationIssuer().issue(
        source,
        candidate,
        policy,
        report,
        qualified_at=NOW - timedelta(days=90),
        expires_at=NOW,
    )


def retraining(**updates) -> ReviewerRetrainingCompletion:
    values = {
        "completion_id": "retraining-2",
        "reviewer_id": "reviewer-a",
        "protocol_id": "pool-risk",
        "annotation_version": "1.0.0",
        "protocol_sha256": PROTOCOL_SHA256,
        "training_material_sha256": "b" * 64,
        "completion_evidence_sha256": "c" * 64,
        "completed_at": NOW + timedelta(hours=1),
        "assessor_id": "assessor-1",
    }
    values.update(updates)
    return ReviewerRetrainingCompletion(**values)


def test_renewal_appends_retraining_and_predecessor_evidence() -> None:
    predecessor = initial_record()
    source = calibration_set("2.0.0")
    candidate = submission(source)
    policy = qualification_policy()
    report = ReviewerQualificationEvaluator().evaluate(source, candidate, policy)
    completion = retraining()

    renewed = ReviewerQualificationRenewalIssuer().issue(
        predecessor,
        completion,
        source,
        candidate,
        policy,
        report,
        QualificationRenewalPolicy(allow_same_calibration_set=False),
        renewed_at=NOW + timedelta(hours=2),
        expires_at=NOW + timedelta(days=90),
    )

    assert renewed.predecessor_qualification_sha256 == predecessor.sha256()
    assert renewed.retraining_completion_sha256 == completion.sha256()
    assert renewed.calibration_set_sha256 == source.sha256()
    assert predecessor.predecessor_qualification_sha256 is None


def test_renewal_rejects_mismatched_or_stale_retraining() -> None:
    predecessor = initial_record()
    source = calibration_set("2.0.0")
    candidate = submission(source)
    policy = qualification_policy()
    report = ReviewerQualificationEvaluator().evaluate(source, candidate, policy)
    issuer = ReviewerQualificationRenewalIssuer()

    with pytest.raises(ValueError, match="renewal predecessor"):
        issuer.issue(
            predecessor.model_copy(update={"qualified": False}),
            retraining(),
            source,
            candidate,
            policy,
            report,
            QualificationRenewalPolicy(allow_same_calibration_set=False),
            renewed_at=NOW + timedelta(hours=2),
            expires_at=NOW + timedelta(days=90),
        )

    with pytest.raises(ValueError, match="match reviewer"):
        issuer.issue(
            predecessor,
            retraining(reviewer_id="reviewer-b"),
            source,
            candidate,
            policy,
            report,
            QualificationRenewalPolicy(allow_same_calibration_set=False),
            renewed_at=NOW + timedelta(hours=2),
            expires_at=NOW + timedelta(days=90),
        )
    with pytest.raises(ValueError, match="after the preceding"):
        issuer.issue(
            predecessor,
            retraining(completed_at=NOW - timedelta(days=100)),
            source,
            candidate,
            policy,
            report,
            QualificationRenewalPolicy(allow_same_calibration_set=False),
            renewed_at=NOW + timedelta(hours=2),
            expires_at=NOW + timedelta(days=90),
        )
    with pytest.raises(ValueError, match="cannot overlap"):
        issuer.issue(
            predecessor.model_copy(update={"expires_at": NOW + timedelta(days=1)}),
            retraining(),
            source,
            candidate,
            policy,
            report,
            QualificationRenewalPolicy(allow_same_calibration_set=False),
            renewed_at=NOW + timedelta(hours=2),
            expires_at=NOW + timedelta(days=90),
        )


def test_renewal_policy_can_forbid_reusing_calibration_set() -> None:
    predecessor = initial_record()
    source = calibration_set()
    candidate = submission(source)
    policy = qualification_policy()
    report = ReviewerQualificationEvaluator().evaluate(source, candidate, policy)

    with pytest.raises(ValueError, match="different calibration"):
        ReviewerQualificationRenewalIssuer().issue(
            predecessor,
            retraining(),
            source,
            candidate,
            policy,
            report,
            QualificationRenewalPolicy(allow_same_calibration_set=False),
            renewed_at=NOW + timedelta(hours=2),
            expires_at=NOW + timedelta(days=90),
        )


def test_retraining_requires_independent_assessor_and_timezone() -> None:
    with pytest.raises(ValueError, match="independent"):
        retraining(assessor_id="reviewer-a")
    with pytest.raises(ValueError, match="timezone"):
        retraining(completed_at=datetime(2026, 7, 20, 12))


def test_renewal_command_writes_recomputed_report_and_new_record(tmp_path) -> None:
    predecessor = initial_record()
    completion = retraining()
    source = calibration_set("2.0.0")
    candidate = submission(source)
    paths = {
        name: tmp_path / f"{name}.json"
        for name in (
            "predecessor",
            "retraining",
            "calibration",
            "submission",
            "qualification_policy",
            "renewal_policy",
            "report",
            "record",
        )
    }
    paths["predecessor"].write_text(predecessor.model_dump_json(), encoding="utf-8")
    paths["retraining"].write_text(completion.model_dump_json(), encoding="utf-8")
    source.save(paths["calibration"])
    paths["submission"].write_text(candidate.model_dump_json(), encoding="utf-8")
    paths["qualification_policy"].write_text(
        qualification_policy().model_dump_json(), encoding="utf-8"
    )
    paths["renewal_policy"].write_text(
        QualificationRenewalPolicy(allow_same_calibration_set=False).model_dump_json(),
        encoding="utf-8",
    )

    exit_code = main(
        [
            str(paths["predecessor"]),
            str(paths["retraining"]),
            str(paths["calibration"]),
            str(paths["submission"]),
            str(paths["qualification_policy"]),
            str(paths["renewal_policy"]),
            str(paths["report"]),
            str(paths["record"]),
            "--renewed-at",
            (NOW + timedelta(hours=2)).isoformat(),
            "--expires-at",
            (NOW + timedelta(days=90)).isoformat(),
        ]
    )

    record = json.loads(paths["record"].read_text(encoding="utf-8"))
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["qualified"] is True
    assert record["predecessor_qualification_sha256"] == predecessor.sha256()
    assert record["retraining_completion_sha256"] == completion.sha256()


def test_renewal_command_does_not_issue_record_after_failed_recalibration(tmp_path) -> None:
    predecessor = initial_record()
    completion = retraining()
    source = calibration_set("2.0.0")
    failed = submission(source).model_copy(
        update={
            "responses": (
                CalibrationResponse(
                    item_id=source.items[0].item_id, judgment=RiskJudgment.SAFE
                ),
                CalibrationResponse(
                    item_id=source.items[1].item_id, judgment=RiskJudgment.DANGEROUS
                ),
            )
        }
    )
    predecessor_path = tmp_path / "predecessor.json"
    retraining_path = tmp_path / "retraining.json"
    calibration_path = tmp_path / "calibration.json"
    submission_path = tmp_path / "submission.json"
    qualification_policy_path = tmp_path / "qualification-policy.json"
    renewal_policy_path = tmp_path / "renewal-policy.json"
    report_path = tmp_path / "report.json"
    record_path = tmp_path / "record.json"
    predecessor_path.write_text(predecessor.model_dump_json(), encoding="utf-8")
    retraining_path.write_text(completion.model_dump_json(), encoding="utf-8")
    source.save(calibration_path)
    submission_path.write_text(failed.model_dump_json(), encoding="utf-8")
    qualification_policy_path.write_text(
        qualification_policy().model_dump_json(), encoding="utf-8"
    )
    renewal_policy_path.write_text(
        QualificationRenewalPolicy(allow_same_calibration_set=False).model_dump_json(),
        encoding="utf-8",
    )

    exit_code = main(
        [
            str(predecessor_path),
            str(retraining_path),
            str(calibration_path),
            str(submission_path),
            str(qualification_policy_path),
            str(renewal_policy_path),
            str(report_path),
            str(record_path),
            "--renewed-at",
            (NOW + timedelta(hours=2)).isoformat(),
            "--expires-at",
            (NOW + timedelta(days=90)).isoformat(),
        ]
    )

    assert exit_code == 2
    assert json.loads(report_path.read_text(encoding="utf-8"))["qualified"] is False
    assert record_path.exists() is False


def renewed_chain():
    predecessor = initial_record()
    completion = retraining()
    source = calibration_set("2.0.0")
    candidate = submission(source)
    policy = qualification_policy()
    report = ReviewerQualificationEvaluator().evaluate(source, candidate, policy)
    renewed = ReviewerQualificationRenewalIssuer().issue(
        predecessor,
        completion,
        source,
        candidate,
        policy,
        report,
        QualificationRenewalPolicy(allow_same_calibration_set=False),
        renewed_at=NOW + timedelta(hours=2),
        expires_at=NOW + timedelta(days=90),
    )
    return predecessor, completion, renewed


def test_history_verifier_accepts_complete_append_only_chain() -> None:
    predecessor, completion, renewed = renewed_chain()

    report = QualificationHistoryVerifier().verify(
        [predecessor, renewed],
        [completion],
        checked_at=NOW + timedelta(days=1),
    )

    assert report.valid is True
    assert report.failures == ()
    assert report.active_qualification_sha256 == renewed.sha256()


def test_history_verifier_reports_broken_links_and_orphan_evidence() -> None:
    predecessor, completion, renewed = renewed_chain()
    broken = renewed.model_copy(update={"predecessor_qualification_sha256": "d" * 64})
    orphan = retraining(completion_id="orphan", completed_at=NOW + timedelta(hours=3))

    report = QualificationHistoryVerifier().verify(
        [predecessor, broken],
        [completion, orphan],
        checked_at=NOW + timedelta(days=1),
    )

    assert report.valid is False
    assert "predecessor_digest_mismatch:1" in report.failures
    assert f"orphan_retraining_evidence:{orphan.sha256()}" in report.failures


def test_history_verification_command_writes_failure_report(tmp_path) -> None:
    predecessor, completion, renewed = renewed_chain()
    broken = renewed.model_copy(update={"retraining_completion_sha256": "d" * 64})
    qualifications_path = tmp_path / "qualifications.jsonl"
    retraining_path = tmp_path / "retraining.jsonl"
    report_path = tmp_path / "history-report.json"
    qualifications_path.write_text(
        "\n".join(item.model_dump_json() for item in (predecessor, broken)) + "\n",
        encoding="utf-8",
    )
    retraining_path.write_text(completion.model_dump_json() + "\n", encoding="utf-8")

    exit_code = history_main(
        [
            str(qualifications_path),
            str(retraining_path),
            str(report_path),
            "--checked-at",
            (NOW + timedelta(days=1)).isoformat(),
        ]
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert payload["valid"] is False
    assert "missing_retraining_evidence:1" in payload["failures"]
