import argparse
from datetime import datetime
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from aquaguard.annotation_quality_cli import _write_json
from aquaguard.vision.annotation_governance import ReviewerQualificationRecord
from aquaguard.vision.annotation_renewal import (
    QualificationHistoryVerifier,
    ReviewerRetrainingCompletion,
)

Artifact = TypeVar("Artifact", bound=BaseModel)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aquaguard-verify-qualification-history",
        description="Verify an append-only reviewer qualification and retraining chain.",
    )
    parser.add_argument("qualifications", type=Path)
    parser.add_argument("retraining_completions", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--checked-at", required=True, type=datetime.fromisoformat)
    args = parser.parse_args(argv)
    report = QualificationHistoryVerifier().verify(
        _load_jsonl(args.qualifications, ReviewerQualificationRecord),
        _load_jsonl(args.retraining_completions, ReviewerRetrainingCompletion),
        checked_at=args.checked_at,
    )
    _write_json(args.report, report.model_dump(mode="json"))
    return 0 if report.valid else 2


def _load_jsonl(path: Path, model: type[Artifact]) -> list[Artifact]:
    records: list[Artifact] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            records.append(model.model_validate_json(raw_line))
        except ValueError as exc:
            raise ValueError(f"invalid {path.name} line {line_number}: {exc}") from exc
    return records
