import argparse
import json
from datetime import datetime
from pathlib import Path

from aquaguard.vision.annotation import AnnotationReview
from aquaguard.vision.annotation_quality import (
    CohenAgreementReporter,
    ReviewerCalibrationSet,
    ReviewerCalibrationSubmission,
    ReviewerQualificationEvaluator,
    ReviewerQualificationPolicy,
)
from aquaguard.vision.annotation_governance import ReviewerQualificationIssuer


def qualify_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aquaguard-qualify-reviewer",
        description="Evaluate a pseudonymous reviewer against a controlled calibration set.",
    )
    parser.add_argument("calibration_set", type=Path)
    parser.add_argument("submission", type=Path)
    parser.add_argument("policy", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--qualified-at", type=datetime.fromisoformat)
    parser.add_argument("--expires-at", type=datetime.fromisoformat)
    args = parser.parse_args(argv)
    lifecycle_values = (args.record, args.qualified_at, args.expires_at)
    if any(value is not None for value in lifecycle_values) and not all(
        value is not None for value in lifecycle_values
    ):
        parser.error("--record, --qualified-at, and --expires-at must be supplied together")
    calibration_set = ReviewerCalibrationSet.load(args.calibration_set)
    submission = ReviewerCalibrationSubmission.model_validate_json(
        args.submission.read_text(encoding="utf-8")
    )
    policy = ReviewerQualificationPolicy.model_validate_json(
        args.policy.read_text(encoding="utf-8")
    )
    report = ReviewerQualificationEvaluator().evaluate(calibration_set, submission, policy)
    _write_json(args.report, report.model_dump(mode="json"))
    if args.record is not None and report.qualified:
        record = ReviewerQualificationIssuer().issue(
            calibration_set,
            submission,
            policy,
            report,
            qualified_at=args.qualified_at,
            expires_at=args.expires_at,
        )
        _write_json(args.record, record.model_dump(mode="json"))
    return 0 if report.qualified else 2


def agreement_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aquaguard-annotation-agreement",
        description="Compute agreement and Cohen's kappa for the same two reviewers.",
    )
    parser.add_argument("reviews", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args(argv)
    reviews = _load_reviews(args.reviews)
    report = CohenAgreementReporter().evaluate(reviews)
    _write_json(args.report, report.model_dump(mode="json"))
    return 0


def _load_reviews(path: Path) -> list[AnnotationReview]:
    reviews: list[AnnotationReview] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            reviews.append(AnnotationReview.model_validate_json(raw_line))
        except ValueError as exc:
            raise ValueError(f"invalid {path.name} line {line_number}: {exc}") from exc
    return reviews


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
