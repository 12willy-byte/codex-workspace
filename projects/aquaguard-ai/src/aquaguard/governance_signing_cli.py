import argparse
from datetime import datetime
from pathlib import Path

from aquaguard.annotation_quality_cli import _write_json
from aquaguard.governance_audit import GovernanceVerificationRuntime
from aquaguard.governance_signing import (
    DetachedSignatureVerifier,
    Ed25519SignatureVerifier,
    GovernanceSignatureEnvelope,
)


def main(
    argv: list[str] | None = None,
    *,
    verifier: DetachedSignatureVerifier | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        prog="aquaguard-verify-governance-signature",
        description="Verify a detached governance artifact signature against a trusted key ring.",
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("envelope", type=Path)
    parser.add_argument("trust_store", type=Path)
    parser.add_argument("policy", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--audit-database", required=True, type=Path)
    parser.add_argument("--checked-at", required=True, type=datetime.fromisoformat)
    args = parser.parse_args(argv)
    envelope = GovernanceSignatureEnvelope.model_validate_json(
        args.envelope.read_text(encoding="utf-8")
    )
    runtime = GovernanceVerificationRuntime(
        trust_store_path=args.trust_store,
        policy_path=args.policy,
        audit_database_path=args.audit_database,
        verifiers=(verifier or Ed25519SignatureVerifier(),),
    )
    report = runtime.verifier.verify_and_record(
        args.artifact.read_bytes(),
        envelope,
        runtime.trust_store,
        runtime.policy,
        checked_at=args.checked_at,
        recorded_at=args.checked_at,
    )
    _write_json(args.report, report.model_dump(mode="json"))
    return 0 if report.valid else 2
