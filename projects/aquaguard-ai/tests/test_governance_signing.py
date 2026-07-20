import base64
import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest

from aquaguard.governance_signing import (
    GovernanceSignatureEnvelope,
    GovernanceSignatureIssuer,
    GovernanceSignaturePolicy,
    GovernanceSignatureVerificationService,
    GovernanceTrustStore,
    TrustedGovernancePublicKey,
)
from aquaguard.governance_signing_cli import main

NOW = datetime(2026, 7, 20, 12, tzinfo=timezone.utc)
PUBLIC_KEY = b"synthetic-public-key"


class SyntheticSigner:
    algorithm = "ed25519"

    def sign(self, payload: bytes) -> bytes:
        return hashlib.sha256(PUBLIC_KEY + payload).digest()


class SyntheticVerifier:
    algorithm = "ed25519"

    def verify(self, public_key: bytes, payload: bytes, signature: bytes) -> bool:
        return signature == hashlib.sha256(public_key + payload).digest()


def trusted_key(
    *,
    key_version: str = "1",
    valid_from: datetime = NOW - timedelta(days=1),
    valid_until: datetime = NOW + timedelta(days=30),
    revoked_at: datetime | None = None,
) -> TrustedGovernancePublicKey:
    return TrustedGovernancePublicKey(
        key_id="governance-key",
        key_version=key_version,
        signer_id="safety-board",
        algorithm="ed25519",
        public_key_base64=base64.b64encode(PUBLIC_KEY).decode("ascii"),
        valid_from=valid_from,
        valid_until=valid_until,
        revoked_at=revoked_at,
    )


def signed(artifact: bytes, key=None, signed_at: datetime = NOW):
    return GovernanceSignatureIssuer().issue(
        artifact,
        artifact_type="reviewer_qualification",
        signer_id="safety-board",
        key=key or trusted_key(),
        signed_at=signed_at,
        signer=SyntheticSigner(),
    )


def verify(artifact: bytes, envelope, key, *, accept_pre_revocation=True):
    return GovernanceSignatureVerificationService((SyntheticVerifier(),)).verify(
        artifact,
        envelope,
        GovernanceTrustStore(keys=(key,)),
        GovernanceSignaturePolicy(
            accept_pre_revocation_signatures=accept_pre_revocation
        ),
        checked_at=NOW + timedelta(days=2),
    )


def test_signature_envelope_binds_artifact_metadata_and_trusted_key() -> None:
    artifact = b'{"qualification":"approved"}\n'
    key = trusted_key()
    envelope = signed(artifact, key)

    report = verify(artifact, envelope, key)

    assert report.valid is True
    assert report.failures == ()
    assert envelope.artifact_sha256 == hashlib.sha256(artifact).hexdigest()


def test_signature_verification_rejects_artifact_and_envelope_tampering() -> None:
    artifact = b"governance artifact"
    key = trusted_key()
    envelope = signed(artifact, key)
    altered_metadata = envelope.model_copy(update={"artifact_type": "other"})

    altered_artifact = verify(artifact + b"!", envelope, key)
    altered_envelope = verify(artifact, altered_metadata, key)

    assert "artifact_digest_mismatch" in altered_artifact.failures
    assert "signature_invalid" in altered_envelope.failures


def test_key_rotation_resolves_exact_version_and_applies_revocation_policy() -> None:
    artifact = b"qualification"
    revoked_at = NOW + timedelta(hours=1)
    old_key = trusted_key(key_version="1", revoked_at=revoked_at)
    new_key = trusted_key(key_version="2", valid_from=revoked_at)
    envelope = signed(artifact, old_key, signed_at=NOW)
    store = GovernanceTrustStore(keys=(old_key, new_key))
    service = GovernanceSignatureVerificationService((SyntheticVerifier(),))

    preserved = service.verify(
        artifact,
        envelope,
        store,
        GovernanceSignaturePolicy(accept_pre_revocation_signatures=True),
        checked_at=NOW + timedelta(days=2),
    )
    strict = service.verify(
        artifact,
        envelope,
        store,
        GovernanceSignaturePolicy(accept_pre_revocation_signatures=False),
        checked_at=NOW + timedelta(days=2),
    )

    assert preserved.valid is True
    assert strict.valid is False
    assert "key_currently_revoked" in strict.failures


def test_scheduled_revocation_does_not_apply_before_effective_time() -> None:
    artifact = b"qualification"
    revoked_at = NOW + timedelta(days=5)
    key = trusted_key(revoked_at=revoked_at)
    envelope = signed(artifact, key, signed_at=NOW)

    report = GovernanceSignatureVerificationService((SyntheticVerifier(),)).verify(
        artifact,
        envelope,
        GovernanceTrustStore(keys=(key,)),
        GovernanceSignaturePolicy(accept_pre_revocation_signatures=False),
        checked_at=NOW + timedelta(days=1),
    )

    assert report.valid is True


def test_signature_service_reports_unknown_key_and_future_signature() -> None:
    artifact = b"qualification"
    envelope = signed(artifact).model_copy(
        update={"key_id": "unknown", "signed_at": NOW + timedelta(days=3)}
    )
    report = verify(artifact, envelope, trusted_key())

    assert report.valid is False
    assert set(report.failures) == {"signature_from_future", "trusted_key_not_found"}


def test_signature_models_reject_invalid_key_lifecycle_and_base64() -> None:
    with pytest.raises(ValueError, match="expiry"):
        trusted_key(valid_from=NOW, valid_until=NOW)
    with pytest.raises(ValueError, match="base64"):
        GovernanceSignatureEnvelope(
            artifact_type="qualification",
            artifact_sha256="a" * 64,
            signer_id="board",
            key_id="key",
            key_version="1",
            algorithm="ed25519",
            signed_at=NOW,
            signature_base64="not base64!",
        )


def test_signature_verification_command_writes_failure_report(tmp_path) -> None:
    artifact = b"qualification"
    key = trusted_key()
    envelope = signed(artifact, key)
    artifact_path = tmp_path / "artifact.json"
    envelope_path = tmp_path / "envelope.json"
    trust_store_path = tmp_path / "trust-store.json"
    policy_path = tmp_path / "policy.json"
    report_path = tmp_path / "report.json"
    artifact_path.write_bytes(artifact + b"tampered")
    envelope_path.write_text(envelope.model_dump_json(), encoding="utf-8")
    trust_store_path.write_text(
        GovernanceTrustStore(keys=(key,)).model_dump_json(), encoding="utf-8"
    )
    policy_path.write_text(
        GovernanceSignaturePolicy(
            accept_pre_revocation_signatures=True
        ).model_dump_json(),
        encoding="utf-8",
    )

    exit_code = main(
        [
            str(artifact_path),
            str(envelope_path),
            str(trust_store_path),
            str(policy_path),
            str(report_path),
            "--checked-at",
            (NOW + timedelta(days=1)).isoformat(),
        ],
        verifier=SyntheticVerifier(),
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert payload["valid"] is False
    assert "artifact_digest_mismatch" in payload["failures"]
