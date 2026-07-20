import argparse
from datetime import datetime
from pathlib import Path

from aquaguard.annotation_quality_cli import _load_reviews, _write_json
from aquaguard.vision.annotation_governance import (
    AnnotationBatchAdmissionPolicy,
    AnnotationBatchAdmissionService,
    ReviewerQualificationRecord,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aquaguard-admit-annotation-batch",
        description="Fail closed unless reviewer qualifications and batch agreement are valid.",
    )
    parser.add_argument("reviews", type=Path)
    parser.add_argument("qualifications", type=Path)
    parser.add_argument("policy", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--checked-at", required=True, type=datetime.fromisoformat)
    args = parser.parse_args(argv)
    records = _load_qualifications(args.qualifications)
    policy = AnnotationBatchAdmissionPolicy.model_validate_json(
        args.policy.read_text(encoding="utf-8")
    )
    report = AnnotationBatchAdmissionService().evaluate(
        _load_reviews(args.reviews), records, policy, checked_at=args.checked_at
    )
    _write_json(args.report, report.model_dump(mode="json"))
    return 0 if report.admitted else 2


def _load_qualifications(path: Path) -> list[ReviewerQualificationRecord]:
    records: list[ReviewerQualificationRecord] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            records.append(ReviewerQualificationRecord.model_validate_json(raw_line))
        except ValueError as exc:
            raise ValueError(f"invalid {path.name} line {line_number}: {exc}") from exc
    return records
