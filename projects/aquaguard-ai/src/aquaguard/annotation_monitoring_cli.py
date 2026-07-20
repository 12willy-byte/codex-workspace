import argparse
from datetime import datetime
from pathlib import Path

from aquaguard.annotation_quality_cli import _write_json
from aquaguard.vision.annotation_monitoring import (
    AnnotationBatchQualitySnapshot,
    ContinuousAnnotationQualityMonitor,
    ContinuousAnnotationQualityPolicy,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aquaguard-monitor-annotation-quality",
        description="Evaluate rolling annotation quality and retraining triggers.",
    )
    parser.add_argument("snapshots", type=Path)
    parser.add_argument("policy", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--evaluated-at", required=True, type=datetime.fromisoformat)
    args = parser.parse_args(argv)
    policy = ContinuousAnnotationQualityPolicy.model_validate_json(
        args.policy.read_text(encoding="utf-8")
    )
    report = ContinuousAnnotationQualityMonitor().evaluate(
        _load_snapshots(args.snapshots), policy, evaluated_at=args.evaluated_at
    )
    _write_json(args.report, report.model_dump(mode="json"))
    return 2 if report.retraining_required else 0


def _load_snapshots(path: Path) -> list[AnnotationBatchQualitySnapshot]:
    snapshots: list[AnnotationBatchQualitySnapshot] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            snapshots.append(AnnotationBatchQualitySnapshot.model_validate_json(raw_line))
        except ValueError as exc:
            raise ValueError(f"invalid {path.name} line {line_number}: {exc}") from exc
    return snapshots
