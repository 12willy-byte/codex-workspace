import json
from datetime import datetime, timedelta, timezone

import pytest

from aquaguard.annotation_quality_cli import agreement_main, qualify_main
from aquaguard.vision import (
    AnnotationReview,
    CalibrationGoldItem,
    CalibrationResponse,
    CohenAgreementReporter,
    ReviewerCalibrationSet,
    ReviewerCalibrationSubmission,
    ReviewerQualificationEvaluator,
    ReviewerQualificationPolicy,
    RiskJudgment,
)

PROTOCOL_SHA256 = "a" * 64


def calibration_set() -> ReviewerCalibrationSet:
    return ReviewerCalibrationSet(
        calibration_set_id="reviewer-baseline",
        calibration_set_version="1.0.0",
        protocol_id="pool-risk",
        annotation_version="1.0.0",
        protocol_sha256=PROTOCOL_SHA256,
        items=(
            CalibrationGoldItem(item_id="danger-1", dangerous=True),
            CalibrationGoldItem(item_id="danger-2", dangerous=True),
            CalibrationGoldItem(item_id="safe-1", dangerous=False),
            CalibrationGoldItem(item_id="safe-2", dangerous=False),
        ),
    )


def submission(
    source: ReviewerCalibrationSet,
    judgments: tuple[RiskJudgment, ...],
) -> ReviewerCalibrationSubmission:
    return ReviewerCalibrationSubmission(
        reviewer_id="reviewer-a",
        protocol_id=source.protocol_id,
        annotation_version=source.annotation_version,
        protocol_sha256=source.protocol_sha256,
        calibration_set_sha256=source.sha256(),
        responses=tuple(
            CalibrationResponse(item_id=item.item_id, judgment=judgment)
            for item, judgment in zip(source.items, judgments, strict=True)
        ),
    )


def policy() -> ReviewerQualificationPolicy:
    return ReviewerQualificationPolicy(
        minimum_items=4,
        minimum_dangerous_items=2,
        minimum_safe_items=2,
        minimum_accuracy=0.75,
        minimum_dangerous_recall=0.5,
        minimum_safe_specificity=1.0,
        maximum_uncertain_fraction=0.25,
    )


def test_calibration_set_digest_round_trip_and_qualification_metrics(tmp_path) -> None:
    source = calibration_set()
    path = tmp_path / "calibration.json"
    source.save(path)
    recovered = ReviewerCalibrationSet.load(path)
    candidate = submission(
        recovered,
        (
            RiskJudgment.DANGEROUS,
            RiskJudgment.UNCERTAIN,
            RiskJudgment.SAFE,
            RiskJudgment.SAFE,
        ),
    )

    report = ReviewerQualificationEvaluator().evaluate(recovered, candidate, policy())

    assert recovered.sha256() == source.sha256()
    assert report.correct_items == 3
    assert report.uncertain_items == 1
    assert report.accuracy == 0.75
    assert report.dangerous_recall == 0.5
    assert report.safe_specificity == 1
    assert report.uncertain_fraction == 0.25
    assert report.qualified is True
    assert report.failed_requirements == ()


def test_qualification_reports_each_failed_requirement() -> None:
    source = calibration_set()
    candidate = submission(
        source,
        (
            RiskJudgment.SAFE,
            RiskJudgment.UNCERTAIN,
            RiskJudgment.DANGEROUS,
            RiskJudgment.UNCERTAIN,
        ),
    )

    report = ReviewerQualificationEvaluator().evaluate(source, candidate, policy())

    assert report.qualified is False
    assert set(report.failed_requirements) == {
        "minimum_accuracy",
        "minimum_dangerous_recall",
        "minimum_safe_specificity",
        "maximum_uncertain_fraction",
    }


def test_qualification_rejects_incomplete_or_mismatched_submission() -> None:
    source = calibration_set()
    candidate = submission(
        source,
        (
            RiskJudgment.DANGEROUS,
            RiskJudgment.DANGEROUS,
            RiskJudgment.SAFE,
            RiskJudgment.SAFE,
        ),
    )

    with pytest.raises(ValueError, match="answer every item exactly once"):
        ReviewerQualificationEvaluator().evaluate(
            source,
            candidate.model_copy(update={"responses": candidate.responses[:-1]}),
            policy(),
        )

    with pytest.raises(ValueError, match="does not match protocol"):
        ReviewerQualificationEvaluator().evaluate(
            source,
            candidate.model_copy(update={"protocol_sha256": "b" * 64}),
            policy(),
        )


def annotation_review(
    reviewer_id: str,
    judgment: RiskJudgment,
    timestamp: float,
) -> AnnotationReview:
    return AnnotationReview(
        protocol_id="pool-risk",
        annotation_version="1.0.0",
        protocol_sha256=PROTOCOL_SHA256,
        recording_id="recording-1",
        venue_id="venue-a",
        session_id="session-1",
        timestamp=timestamp,
        track_id="person-1",
        reviewer_id=reviewer_id,
        judgment=judgment,
    )


def test_cohen_agreement_report_includes_uncertainty_and_chance_correction() -> None:
    pairs = (
        (RiskJudgment.SAFE, RiskJudgment.SAFE),
        (RiskJudgment.DANGEROUS, RiskJudgment.DANGEROUS),
        (RiskJudgment.SAFE, RiskJudgment.DANGEROUS),
        (RiskJudgment.UNCERTAIN, RiskJudgment.UNCERTAIN),
    )
    reviews = [
        annotation_review(reviewer_id, judgment, timestamp)
        for timestamp, pair in enumerate(pairs)
        for reviewer_id, judgment in zip(("reviewer-a", "reviewer-b"), pair, strict=True)
    ]

    report = CohenAgreementReporter().evaluate(reviews)

    assert report.items == 4
    assert report.agreed_items == 3
    assert report.disputed_items == 1
    assert report.uncertain_items == 1
    assert report.observed_agreement == 0.75
    assert report.expected_agreement == pytest.approx(0.3125)
    assert report.cohen_kappa == pytest.approx(7 / 11)


def test_cohen_agreement_requires_same_two_reviewers_on_every_item() -> None:
    reviews = [
        annotation_review("reviewer-a", RiskJudgment.SAFE, 0),
        annotation_review("reviewer-b", RiskJudgment.SAFE, 0),
        annotation_review("reviewer-a", RiskJudgment.SAFE, 1),
        annotation_review("reviewer-c", RiskJudgment.SAFE, 1),
    ]

    with pytest.raises(ValueError, match="same two reviewers"):
        CohenAgreementReporter().evaluate(reviews)


def test_agreement_report_rejects_internally_inconsistent_metrics() -> None:
    report = CohenAgreementReporter().evaluate(
        [
            annotation_review("reviewer-a", RiskJudgment.SAFE, 0),
            annotation_review("reviewer-b", RiskJudgment.SAFE, 0),
            annotation_review("reviewer-a", RiskJudgment.DANGEROUS, 1),
            annotation_review("reviewer-b", RiskJudgment.DANGEROUS, 1),
        ]
    )

    with pytest.raises(ValueError, match="observed agreement"):
        type(report)(**{**report.model_dump(), "observed_agreement": 0.5})


def test_reviewer_qualification_command_writes_report_and_signals_failure(tmp_path) -> None:
    source = calibration_set()
    calibration_path = tmp_path / "calibration.json"
    submission_path = tmp_path / "submission.json"
    policy_path = tmp_path / "policy.json"
    report_path = tmp_path / "qualification.json"
    source.save(calibration_path)
    failed_submission = submission(
        source,
        (
            RiskJudgment.SAFE,
            RiskJudgment.SAFE,
            RiskJudgment.DANGEROUS,
            RiskJudgment.DANGEROUS,
        ),
    )
    submission_path.write_text(failed_submission.model_dump_json(), encoding="utf-8")
    policy_path.write_text(policy().model_dump_json(), encoding="utf-8")

    exit_code = qualify_main(
        [str(calibration_path), str(submission_path), str(policy_path), str(report_path)]
    )

    report = json.loads(report_path.read_text())
    assert exit_code == 2
    assert report["qualified"] is False
    assert "minimum_dangerous_recall" in report["failed_requirements"]


def test_reviewer_qualification_command_can_issue_expiring_record(tmp_path) -> None:
    source = calibration_set()
    calibration_path = tmp_path / "calibration.json"
    submission_path = tmp_path / "submission.json"
    policy_path = tmp_path / "policy.json"
    report_path = tmp_path / "qualification.json"
    record_path = tmp_path / "record.json"
    source.save(calibration_path)
    passing = submission(
        source,
        (
            RiskJudgment.DANGEROUS,
            RiskJudgment.DANGEROUS,
            RiskJudgment.SAFE,
            RiskJudgment.SAFE,
        ),
    )
    submission_path.write_text(passing.model_dump_json(), encoding="utf-8")
    policy_path.write_text(policy().model_dump_json(), encoding="utf-8")
    qualified_at = datetime(2026, 7, 19, tzinfo=timezone.utc)

    exit_code = qualify_main(
        [
            str(calibration_path),
            str(submission_path),
            str(policy_path),
            str(report_path),
            "--record",
            str(record_path),
            "--qualified-at",
            qualified_at.isoformat(),
            "--expires-at",
            (qualified_at + timedelta(days=30)).isoformat(),
        ]
    )

    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert record["reviewer_id"] == "reviewer-a"
    assert record["qualified"] is True
    assert record["qualification_policy_sha256"] == policy().sha256()


def test_annotation_agreement_command_writes_kappa_report(tmp_path) -> None:
    reviews_path = tmp_path / "reviews.jsonl"
    report_path = tmp_path / "agreement.json"
    reviews = [
        annotation_review("reviewer-a", RiskJudgment.SAFE, 0),
        annotation_review("reviewer-b", RiskJudgment.SAFE, 0),
        annotation_review("reviewer-a", RiskJudgment.DANGEROUS, 1),
        annotation_review("reviewer-b", RiskJudgment.DANGEROUS, 1),
    ]
    reviews_path.write_text(
        "\n".join(item.model_dump_json() for item in reviews) + "\n", encoding="utf-8"
    )

    assert agreement_main([str(reviews_path), str(report_path)]) == 0

    report = json.loads(report_path.read_text())
    assert report["observed_agreement"] == 1
    assert report["cohen_kappa"] == 1
