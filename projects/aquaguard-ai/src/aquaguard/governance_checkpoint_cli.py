import argparse
from pathlib import Path

from aquaguard.annotation_quality_cli import _write_json
from aquaguard.governance_audit import SQLiteGovernanceVerificationAuditRepository
from aquaguard.governance_checkpoint import (
    GovernanceAuditCheckpoint,
    GovernanceAuditCheckpointVerifier,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aquaguard-verify-governance-checkpoint",
        description="Detect audit rollback or divergence from an external checkpoint.",
    )
    parser.add_argument("audit_database", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args(argv)
    if not args.audit_database.is_file():
        raise ValueError("governance audit database must already exist")
    checkpoint = GovernanceAuditCheckpoint.model_validate_json(
        args.checkpoint.read_text(encoding="utf-8")
    )
    report = GovernanceAuditCheckpointVerifier().verify(
        SQLiteGovernanceVerificationAuditRepository(args.audit_database), checkpoint
    )
    _write_json(args.report, report.model_dump(mode="json"))
    return 0 if report.valid else 2
