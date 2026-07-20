import argparse
from datetime import datetime
from pathlib import Path

from aquaguard.annotation_quality_cli import _write_json
from aquaguard.governance_signing import (
    DetachedSignatureVerifier,
    Ed25519SignatureVerifier,
    GovernanceSignatureEnvelope,
    GovernanceSignaturePolicy,
    GovernanceSignatureVerificationService,
    GovernanceTrustStore,
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
    parser.add_argument("--checked-at", required=True, type=datetime.fromisoformat)
    args = parser.parse_args(argv)
    envelope = GovernanceSignatureEnvelope.model_validate_json(
        args.envelope.read_text(encoding="utf-8")
    )
    trust_store = GovernanceTrustStore.model_validate_json(
        args.trust_store.read_text(encoding="utf-8")
    )
    policy = GovernanceSignaturePolicy.model_validate_json(
        args.policy.read_text(encoding="utf-8")
    )
    report = GovernanceSignatureVerificationService(
        (verifier or Ed25519SignatureVerifier(),)
    ).verify(
        args.artifact.read_bytes(),
        envelope,
        trust_store,
        policy,
        checked_at=args.checked_at,
    )
    _write_json(args.report, report.model_dump(mode="json"))
    return 0 if report.valid else 2
