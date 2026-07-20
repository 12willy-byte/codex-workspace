import argparse
from datetime import datetime
from pathlib import Path

from aquaguard.annotation_quality_cli import _write_json
from aquaguard.vision.annotation_governance import ReviewerQualificationRecord
from aquaguard.vision.annotation_quality import (
    ReviewerCalibrationSet,
    ReviewerCalibrationSubmission,
    ReviewerQualificationEvaluator,
    ReviewerQualificationPolicy,
)
from aquaguard.vision.annotation_renewal import (
    QualificationRenewalPolicy,
    ReviewerQualificationRenewalIssuer,
    ReviewerRetrainingCompletion,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aquaguard-renew-reviewer-qualification",
        description="Append a reviewer qualification after retraining and recalibration.",
    )
    parser.add_argument("predecessor", type=Path)
    parser.add_argument("retraining", type=Path)
    parser.add_argument("calibration_set", type=Path)
    parser.add_argument("submission", type=Path)
    parser.add_argument("qualification_policy", type=Path)
    parser.add_argument("renewal_policy", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("record", type=Path)
    parser.add_argument("--renewed-at", required=True, type=datetime.fromisoformat)
    parser.add_argument("--expires-at", required=True, type=datetime.fromisoformat)
    args = parser.parse_args(argv)

    predecessor = ReviewerQualificationRecord.model_validate_json(
        args.predecessor.read_text(encoding="utf-8")
    )
    retraining = ReviewerRetrainingCompletion.model_validate_json(
        args.retraining.read_text(encoding="utf-8")
    )
    calibration_set = ReviewerCalibrationSet.load(args.calibration_set)
    submission = ReviewerCalibrationSubmission.model_validate_json(
        args.submission.read_text(encoding="utf-8")
    )
    qualification_policy = ReviewerQualificationPolicy.model_validate_json(
        args.qualification_policy.read_text(encoding="utf-8")
    )
    renewal_policy = QualificationRenewalPolicy.model_validate_json(
        args.renewal_policy.read_text(encoding="utf-8")
    )
    report = ReviewerQualificationEvaluator().evaluate(
        calibration_set, submission, qualification_policy
    )
    _write_json(args.report, report.model_dump(mode="json"))
    if not report.qualified:
        return 2
    record = ReviewerQualificationRenewalIssuer().issue(
        predecessor,
        retraining,
        calibration_set,
        submission,
        qualification_policy,
        report,
        renewal_policy,
        renewed_at=args.renewed_at,
        expires_at=args.expires_at,
    )
    _write_json(args.record, record.model_dump(mode="json"))
    return 0
