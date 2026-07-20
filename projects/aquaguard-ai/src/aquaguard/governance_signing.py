from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field, field_validator, model_validator

from aquaguard.vision.annotation_governance import _require_aware

SignatureAlgorithm = Literal["ed25519"]


def _decode_base64(value: str, *, field_name: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{field_name} must be canonical base64") from exc


class GovernanceSignatureEnvelope(BaseModel):
    model_config = {"frozen": True}

    schema_version: Literal["1.0"] = "1.0"
    artifact_type: str = Field(min_length=1)
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    signer_id: str = Field(min_length=1)
    key_id: str = Field(min_length=1)
    key_version: str = Field(min_length=1)
    algorithm: SignatureAlgorithm
    signed_at: datetime
    signature_base64: str = Field(min_length=1)

    @field_validator("artifact_type", "signer_id", "key_id", "key_version")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("signature identity fields must not be blank")
        return value

    @field_validator("signed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        return _require_aware(value)

    @field_validator("signature_base64")
    @classmethod
    def validate_signature_base64(cls, value: str) -> str:
        raw = _decode_base64(value, field_name="signature")
        canonical = base64.b64encode(raw).decode("ascii")
        if canonical != value:
            raise ValueError("signature must use canonical base64 encoding")
        return value

    def signing_bytes(self) -> bytes:
        payload = {
            "domain": "aquaguard-governance-signature-v1",
            "schema_version": self.schema_version,
            "artifact_type": self.artifact_type,
            "artifact_sha256": self.artifact_sha256,
            "signer_id": self.signer_id,
            "key_id": self.key_id,
            "key_version": self.key_version,
            "algorithm": self.algorithm,
            "signed_at": self.signed_at.isoformat(),
        }
        return (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode()

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.model_dump(mode="json"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


class TrustedGovernancePublicKey(BaseModel):
    model_config = {"frozen": True}

    key_id: str = Field(min_length=1)
    key_version: str = Field(min_length=1)
    signer_id: str = Field(min_length=1)
    algorithm: SignatureAlgorithm
    public_key_base64: str = Field(min_length=1)
    valid_from: datetime
    valid_until: datetime
    revoked_at: datetime | None = None

    @field_validator("key_id", "key_version", "signer_id")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("trusted key identity fields must not be blank")
        return value

    @field_validator("valid_from", "valid_until", "revoked_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _require_aware(value)

    @field_validator("public_key_base64")
    @classmethod
    def validate_public_key_base64(cls, value: str) -> str:
        raw = _decode_base64(value, field_name="public key")
        canonical = base64.b64encode(raw).decode("ascii")
        if canonical != value:
            raise ValueError("public key must use canonical base64 encoding")
        return value

    @model_validator(mode="after")
    def require_valid_lifecycle(self) -> TrustedGovernancePublicKey:
        if self.valid_until <= self.valid_from:
            raise ValueError("trusted key expiry must be after activation")
        if self.revoked_at is not None and self.revoked_at < self.valid_from:
            raise ValueError("trusted key cannot be revoked before activation")
        return self

    def public_key_bytes(self) -> bytes:
        return _decode_base64(self.public_key_base64, field_name="public key")


class GovernanceTrustStore(BaseModel):
    model_config = {"frozen": True}

    schema_version: Literal["1.0"] = "1.0"
    keys: tuple[TrustedGovernancePublicKey, ...] = Field(min_length=1)

    @field_validator("keys")
    @classmethod
    def require_unique_key_versions(
        cls, value: tuple[TrustedGovernancePublicKey, ...]
    ) -> tuple[TrustedGovernancePublicKey, ...]:
        identities = [(key.key_id, key.key_version) for key in value]
        if len(identities) != len(set(identities)):
            raise ValueError("trusted key id and version pairs must be unique")
        return value

    def resolve(self, key_id: str, key_version: str) -> TrustedGovernancePublicKey | None:
        return next(
            (
                key
                for key in self.keys
                if key.key_id == key_id and key.key_version == key_version
            ),
            None,
        )

    @classmethod
    def load(cls, path: Path) -> GovernanceTrustStore:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.model_dump(mode="json"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


class GovernanceSignaturePolicy(BaseModel):
    accept_pre_revocation_signatures: bool

    @classmethod
    def load(cls, path: Path) -> GovernanceSignaturePolicy:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.model_dump(mode="json"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


class DetachedSignatureVerifier(Protocol):
    algorithm: str

    def verify(self, public_key: bytes, payload: bytes, signature: bytes) -> bool: ...


class DetachedSignatureSigner(Protocol):
    algorithm: str

    def sign(self, payload: bytes) -> bytes: ...


class GovernanceSignatureVerificationReport(BaseModel):
    checked_at: datetime
    artifact_sha256: str
    key_id: str
    key_version: str
    signer_id: str
    valid: bool
    failures: tuple[str, ...]


class GovernanceSignatureIssuer:
    """Build a detached envelope while private key custody remains outside the model."""

    def issue(
        self,
        artifact: bytes,
        *,
        artifact_type: str,
        signer_id: str,
        key: TrustedGovernancePublicKey,
        signed_at: datetime,
        signer: DetachedSignatureSigner,
    ) -> GovernanceSignatureEnvelope:
        signed_at = _require_aware(signed_at)
        if signer.algorithm != key.algorithm:
            raise ValueError("signer algorithm does not match trusted key")
        if signer_id != key.signer_id:
            raise ValueError("signer identity does not match trusted key")
        if not key.valid_from <= signed_at < key.valid_until:
            raise ValueError("trusted key is not active at signing time")
        if key.revoked_at is not None and signed_at >= key.revoked_at:
            raise ValueError("trusted key is revoked at signing time")
        unsigned = GovernanceSignatureEnvelope(
            artifact_type=artifact_type,
            artifact_sha256=hashlib.sha256(artifact).hexdigest(),
            signer_id=signer_id,
            key_id=key.key_id,
            key_version=key.key_version,
            algorithm=key.algorithm,
            signed_at=signed_at,
            signature_base64=base64.b64encode(b"pending").decode("ascii"),
        )
        signature = signer.sign(unsigned.signing_bytes())
        return unsigned.model_copy(
            update={"signature_base64": base64.b64encode(signature).decode("ascii")}
        )


class GovernanceSignatureVerificationService:
    def __init__(self, verifiers: tuple[DetachedSignatureVerifier, ...]) -> None:
        self._verifiers = {verifier.algorithm: verifier for verifier in verifiers}
        if len(self._verifiers) != len(verifiers):
            raise ValueError("signature verifier algorithms must be unique")

    def verify(
        self,
        artifact: bytes,
        envelope: GovernanceSignatureEnvelope,
        trust_store: GovernanceTrustStore,
        policy: GovernanceSignaturePolicy,
        *,
        checked_at: datetime,
    ) -> GovernanceSignatureVerificationReport:
        checked_at = _require_aware(checked_at)
        failures: list[str] = []
        artifact_sha256 = hashlib.sha256(artifact).hexdigest()
        if artifact_sha256 != envelope.artifact_sha256:
            failures.append("artifact_digest_mismatch")
        if envelope.signed_at > checked_at:
            failures.append("signature_from_future")
        key = trust_store.resolve(envelope.key_id, envelope.key_version)
        if key is None:
            failures.append("trusted_key_not_found")
        else:
            if key.algorithm != envelope.algorithm:
                failures.append("signature_algorithm_mismatch")
            if key.signer_id != envelope.signer_id:
                failures.append("signer_identity_mismatch")
            if not key.valid_from <= envelope.signed_at < key.valid_until:
                failures.append("key_inactive_at_signing_time")
            if key.revoked_at is not None:
                if envelope.signed_at >= key.revoked_at:
                    failures.append("key_revoked_at_signing_time")
                elif (
                    key.revoked_at <= checked_at
                    and not policy.accept_pre_revocation_signatures
                ):
                    failures.append("key_currently_revoked")

            verifier = self._verifiers.get(envelope.algorithm)
            if verifier is None:
                failures.append("signature_verifier_unavailable")
            elif not verifier.verify(
                key.public_key_bytes(),
                envelope.signing_bytes(),
                _decode_base64(envelope.signature_base64, field_name="signature"),
            ):
                failures.append("signature_invalid")
        return GovernanceSignatureVerificationReport(
            checked_at=checked_at,
            artifact_sha256=artifact_sha256,
            key_id=envelope.key_id,
            key_version=envelope.key_version,
            signer_id=envelope.signer_id,
            valid=not failures,
            failures=tuple(failures),
        )


class Ed25519SignatureVerifier:
    algorithm = "ed25519"

    def verify(self, public_key: bytes, payload: bytes, signature: bytes) -> bool:
        try:
            from cryptography.exceptions import InvalidSignature
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        except ImportError as exc:
            raise RuntimeError(
                "Ed25519 verification requires the optional 'signing' dependencies"
            ) from exc
        try:
            Ed25519PublicKey.from_public_bytes(public_key).verify(signature, payload)
        except (InvalidSignature, ValueError):
            return False
        return True


def _canonical_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode()
