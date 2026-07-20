import argparse
from pathlib import Path

from aquaguard.annotation_quality_cli import _write_json
from aquaguard.governance_audit import SQLiteGovernanceVerificationAuditRepository


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aquaguard-verify-governance-audit",
        description="Verify the internal hash chain of a governance signature audit database.",
    )
    parser.add_argument("audit_database", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args(argv)
    if not args.audit_database.is_file():
        raise ValueError("governance audit database must already exist")
    chain = SQLiteGovernanceVerificationAuditRepository(args.audit_database).verify_chain()
    _write_json(args.report, chain.model_dump(mode="json"))
    return 0 if chain.valid else 2
