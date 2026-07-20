import json
from datetime import datetime, timedelta, timezone

import pytest

from aquaguard.annotation_governance_cli import main
from aquaguard.vision import (
    AnnotationBatchAdmissionPolicy,
    AnnotationBatchAdmissionService,
    AnnotationReview,
    CalibrationGoldItem,
    CalibrationResponse,
    ReviewerCalibrationSet,
    ReviewerCalibrationSubmission,
    ReviewerQualificationEvaluator,
    ReviewerQualificationIssuer,
    ReviewerQualificationPolicy,
    ReviewerQualificationRecord,
    RiskJudgment,
)

PROTOCOL_SHA256 = "a" * 64
CALIBRATION_SHA256 = "b" * 64
POLICY_SHA256 = "c" * 64
NOW = datetime(2026, 7, 19, 12, tzinfo=timezone.utc)


def review(reviewer_id: str, judgment: RiskJudgment, timestamp: float) -> AnnotationReview:
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


def qualification(
    reviewer_id: str,
    *,
    qualified: bool = True,
    protocol_sha256: str = PROTOCOL_SHA256,
    qualified_at: datetime = NOW - timedelta(days=1),
    expires_at: datetime = NOW + timedelta(days=1),
) -> ReviewerQualificationRecord:
    return ReviewerQualificationRecord(
        reviewer_id=reviewer_id,
        protocol_id="pool-risk",
        annotation_version="1.0.0",
        protocol_sha256=protocol_sha256,
        calibration_set_sha256=CALIBRATION_SHA256,
        qualification_policy_sha256=POLICY_SHA256,
        qualification_report_sha256="e" * 64,
        qualified=qualified,
        qualified_at=qualified_at,
        expires_at=expires_at,
    )


def policy() -> AnnotationBatchAdmissionPolicy:
    return AnnotationBatchAdmissionPolicy(
        minimum_items=3,
        minimum_observed_agreement=0.8,
        minimum_cohen_kappa=0.5,
        maximum_uncertain_fraction=0,
    )


def matching_reviews() -> list[AnnotationReview]:
    judgments = (RiskJudgment.SAFE, RiskJudgment.DANGEROUS, RiskJudgment.SAFE)
    return [
        review(reviewer_id, judgment, timestamp)
        for timestamp, judgment in enumerate(judgments)
        for reviewer_id in ("reviewer-a", "reviewer-b")
    ]


def test_batch_is_admitted_only_with_current_matching_qualifications() -> None:
    report = AnnotationBatchAdmissionService().evaluate(
        matching_reviews(),
        [qualification("reviewer-a"), qualification("reviewer-b")],
        policy(),
        checked_at=NOW,
    )

    assert report.admitted is True
    assert report.denial_reasons == ()
    assert report.agreement.cohen_kappa == 1


@pytest.mark.parametrize(
    ("records", "reason"),
    [
        (
            [
                qualification("reviewer-a", expires_at=NOW),
                qualification("reviewer-b"),
            ],
            "qualification_inactive:reviewer-a",
        ),
        (
            [
                qualification("reviewer-a", protocol_sha256="d" * 64),
                qualification("reviewer-b"),
            ],
            "qualification_protocol_mismatch:reviewer-a",
        ),
        (
            [qualification("reviewer-a"), qualification("reviewer-b", qualified=False)],
            "reviewer_not_qualified:reviewer-b",
        ),
    ],
)
def test_batch_fails_closed_for_invalid_reviewer_lifecycle(records, reason) -> None:
    report = AnnotationBatchAdmissionService().evaluate(
        matching_reviews(), records, policy(), checked_at=NOW
    )

    assert report.admitted is False
    assert reason in report.denial_reasons


def test_batch_rejects_low_agreement_and_uncertainty() -> None:
    reviews = matching_reviews()
    reviews[-1] = review("reviewer-b", RiskJudgment.UNCERTAIN, 2)

    report = AnnotationBatchAdmissionService().evaluate(
        reviews,
        [qualification("reviewer-a"), qualification("reviewer-b")],
        policy(),
        checked_at=NOW,
    )

    assert report.admitted is False
    assert "minimum_observed_agreement" in report.denial_reasons
    assert "minimum_cohen_kappa" in report.denial_reasons
    assert "maximum_uncertain_fraction" in report.denial_reasons


def test_qualification_record_requires_aware_positive_window() -> None:
    with pytest.raises(ValueError, match="timezone"):
        qualification(
            "reviewer-a",
            qualified_at=datetime(2026, 7, 19),
            expires_at=datetime(2026, 7, 20),
        )
    with pytest.raises(ValueError, match="after qualification"):
        qualification("reviewer-a", qualified_at=NOW, expires_at=NOW)


def test_qualification_issuer_recomputes_and_binds_passing_evidence() -> None:
    source = ReviewerCalibrationSet(
        calibration_set_id="set-1",
        calibration_set_version="1.0.0",
        protocol_id="pool-risk",
        annotation_version="1.0.0",
        protocol_sha256=PROTOCOL_SHA256,
        items=(
            CalibrationGoldItem(item_id="danger", dangerous=True),
            CalibrationGoldItem(item_id="safe", dangerous=False),
        ),
    )
    submission = ReviewerCalibrationSubmission(
        reviewer_id="reviewer-a",
        protocol_id=source.protocol_id,
        annotation_version=source.annotation_version,
        protocol_sha256=source.protocol_sha256,
        calibration_set_sha256=source.sha256(),
        responses=(
            CalibrationResponse(item_id="danger", judgment=RiskJudgment.DANGEROUS),
            CalibrationResponse(item_id="safe", judgment=RiskJudgment.SAFE),
        ),
    )
    qualification_policy = ReviewerQualificationPolicy(
        minimum_items=2,
        minimum_dangerous_items=1,
        minimum_safe_items=1,
        minimum_accuracy=1,
        minimum_dangerous_recall=1,
        minimum_safe_specificity=1,
        maximum_uncertain_fraction=0,
    )
    report = ReviewerQualificationEvaluator().evaluate(
        source, submission, qualification_policy
    )

    record = ReviewerQualificationIssuer().issue(
        source,
        submission,
        qualification_policy,
        report,
        qualified_at=NOW,
        expires_at=NOW + timedelta(days=30),
    )

    assert record.calibration_set_sha256 == source.sha256()
    assert record.qualification_policy_sha256 == qualification_policy.sha256()
    assert record.qualification_report_sha256 == report.sha256()

    with pytest.raises(ValueError, match="does not match recomputed"):
        ReviewerQualificationIssuer().issue(
            source,
            submission,
            qualification_policy,
            report.model_copy(update={"accuracy": 0.5}),
            qualified_at=NOW,
            expires_at=NOW + timedelta(days=30),
        )


def test_admission_command_writes_denial_and_returns_two(tmp_path) -> None:
    reviews_path = tmp_path / "reviews.jsonl"
    qualifications_path = tmp_path / "qualifications.jsonl"
    policy_path = tmp_path / "policy.json"
    report_path = tmp_path / "report.json"
    reviews_path.write_text(
        "\n".join(item.model_dump_json() for item in matching_reviews()) + "\n",
        encoding="utf-8",
    )
    qualifications_path.write_text(
        "\n".join(
            item.model_dump_json()
            for item in (
                qualification("reviewer-a", expires_at=NOW),
                qualification("reviewer-b"),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    policy_path.write_text(policy().model_dump_json(), encoding="utf-8")

    exit_code = main(
        [
            str(reviews_path),
            str(qualifications_path),
            str(policy_path),
            str(report_path),
            "--checked-at",
            NOW.isoformat(),
        ]
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert payload["admitted"] is False
    assert "qualification_inactive:reviewer-a" in payload["denial_reasons"]
