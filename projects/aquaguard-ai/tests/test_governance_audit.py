import base64
import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from aquaguard.governance_audit import GovernanceVerificationRuntime
from aquaguard.governance_audit_cli import main as audit_main
from aquaguard.governance_signing import (
    GovernanceSignatureIssuer,
    GovernanceSignaturePolicy,
    GovernanceTrustStore,
    TrustedGovernancePublicKey,
)

NOW = datetime(2026, 7, 20, 12, tzinfo=timezone.utc)
PUBLIC_KEY = b"synthetic-governance-key"


class SyntheticSigner:
    algorithm = "ed25519"

    def sign(self, payload: bytes) -> bytes:
        return hashlib.sha256(PUBLIC_KEY + payload).digest()


class SyntheticVerifier:
    algorithm = "ed25519"

    def verify(self, public_key: bytes, payload: bytes, signature: bytes) -> bool:
        return signature == hashlib.sha256(public_key + payload).digest()


def trust_store() -> GovernanceTrustStore:
    return GovernanceTrustStore(
        keys=(
            TrustedGovernancePublicKey(
                key_id="board-key",
                key_version="1",
                signer_id="safety-board",
                algorithm="ed25519",
                public_key_base64=base64.b64encode(PUBLIC_KEY).decode("ascii"),
                valid_from=NOW - timedelta(days=1),
                valid_until=NOW + timedelta(days=30),
            ),
        )
    )


def assemble_runtime(tmp_path):
    trust_path = tmp_path / "trust-store.json"
    policy_path = tmp_path / "policy.json"
    audit_path = tmp_path / "audit.sqlite3"
    trust_path.write_text(trust_store().model_dump_json(), encoding="utf-8")
    policy = GovernanceSignaturePolicy(accept_pre_revocation_signatures=True)
    policy_path.write_text(policy.model_dump_json(), encoding="utf-8")
    runtime = GovernanceVerificationRuntime(
        trust_store_path=trust_path,
        policy_path=policy_path,
        audit_database_path=audit_path,
        verifiers=(SyntheticVerifier(),),
    )
    return runtime, audit_path


def envelope(artifact: bytes):
    key = trust_store().keys[0]
    return GovernanceSignatureIssuer().issue(
        artifact,
        artifact_type="reviewer_qualification",
        signer_id="safety-board",
        key=key,
        signed_at=NOW,
        signer=SyntheticSigner(),
    )


def test_runtime_requires_explicit_existing_trust_configuration(tmp_path) -> None:
    with pytest.raises(ValueError, match="trust store"):
        GovernanceVerificationRuntime(
            trust_store_path=tmp_path / "missing-trust.json",
            policy_path=tmp_path / "missing-policy.json",
            audit_database_path=tmp_path / "audit.sqlite3",
            verifiers=(SyntheticVerifier(),),
        )


def test_audited_verification_persists_hash_chain_across_restart(tmp_path) -> None:
    runtime, audit_path = assemble_runtime(tmp_path)
    artifact = b"qualification"
    signature = envelope(artifact)

    runtime.verifier.verify_and_record(
        artifact,
        signature,
        runtime.trust_store,
        runtime.policy,
        checked_at=NOW + timedelta(hours=1),
        recorded_at=NOW + timedelta(hours=1),
    )
    runtime.verifier.verify_and_record(
        artifact + b"tampered",
        signature,
        runtime.trust_store,
        runtime.policy,
        checked_at=NOW + timedelta(hours=2),
        recorded_at=NOW + timedelta(hours=2),
    )

    records = runtime.audit_repository.list()
    recovered = type(runtime.audit_repository)(audit_path)
    chain = recovered.verify_chain()
    assert len(records) == 2
    assert records[0].verification.valid is True
    assert records[1].verification.valid is False
    assert records[1].previous_entry_sha256 == records[0].entry_sha256
    assert chain.valid is True
    assert chain.head_sha256 == records[1].entry_sha256


def test_corrupt_audit_chain_is_detected_and_blocks_append(tmp_path) -> None:
    runtime, audit_path = assemble_runtime(tmp_path)
    artifact = b"qualification"
    signature = envelope(artifact)
    runtime.verifier.verify_and_record(
        artifact,
        signature,
        runtime.trust_store,
        runtime.policy,
        checked_at=NOW + timedelta(hours=1),
        recorded_at=NOW + timedelta(hours=1),
    )
    with sqlite3.connect(audit_path) as connection:
        connection.execute(
            "UPDATE governance_signature_audits SET payload = ? WHERE sequence = 1",
            ("{}",),
        )

    chain = runtime.audit_repository.verify_chain()
    assert chain.valid is False
    assert "invalid_payload:1" in chain.failures
    with pytest.raises(RuntimeError, match="refusing audit read"):
        runtime.audit_repository.list()
    with pytest.raises(RuntimeError, match="chain is corrupt"):
        runtime.verifier.verify_and_record(
            artifact,
            signature,
            runtime.trust_store,
            runtime.policy,
            checked_at=NOW + timedelta(hours=2),
            recorded_at=NOW + timedelta(hours=2),
        )


def test_concurrent_verifications_append_without_losing_audits(tmp_path) -> None:
    runtime, _ = assemble_runtime(tmp_path)
    artifact = b"qualification"
    signature = envelope(artifact)

    def verify_one(index: int) -> None:
        moment = NOW + timedelta(hours=1, seconds=index)
        runtime.verifier.verify_and_record(
            artifact,
            signature,
            runtime.trust_store,
            runtime.policy,
            checked_at=moment,
            recorded_at=moment,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(verify_one, range(8)))

    assert len(runtime.audit_repository.list()) == 8
    assert runtime.audit_repository.verify_chain().valid is True


def test_audit_recording_time_cannot_precede_verification(tmp_path) -> None:
    runtime, _ = assemble_runtime(tmp_path)
    artifact = b"qualification"
    with pytest.raises(ValueError, match="cannot precede"):
        runtime.verifier.verify_and_record(
            artifact,
            envelope(artifact),
            runtime.trust_store,
            runtime.policy,
            checked_at=NOW + timedelta(hours=2),
            recorded_at=NOW + timedelta(hours=1),
        )


def test_audit_chain_command_reports_integrity(tmp_path) -> None:
    runtime, audit_path = assemble_runtime(tmp_path)
    artifact = b"qualification"
    runtime.verifier.verify_and_record(
        artifact,
        envelope(artifact),
        runtime.trust_store,
        runtime.policy,
        checked_at=NOW + timedelta(hours=1),
        recorded_at=NOW + timedelta(hours=1),
    )
    report_path = tmp_path / "chain-report.json"

    assert audit_main([str(audit_path), str(report_path)]) == 0
