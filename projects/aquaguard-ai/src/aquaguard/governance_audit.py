from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

from aquaguard.governance_signing import (
    DetachedSignatureVerifier,
    GovernanceSignatureEnvelope,
    GovernanceSignaturePolicy,
    GovernanceSignatureVerificationReport,
    GovernanceSignatureVerificationService,
    GovernanceTrustStore,
    _canonical_bytes,
)
from aquaguard.vision.annotation_governance import _require_aware


class GovernanceVerificationAudit(BaseModel):
    model_config = {"frozen": True}

    audit_id: UUID = Field(default_factory=uuid4)
    recorded_at: datetime
    artifact_type: str
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    envelope_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trust_store_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verification: GovernanceSignatureVerificationReport
    previous_entry_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    entry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("recorded_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        return _require_aware(value)

    def computed_entry_sha256(self) -> str:
        payload = self.model_dump(mode="json", exclude={"entry_sha256"})
        return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


class GovernanceAuditChainReport(BaseModel):
    entries: int
    head_sha256: str | None
    valid: bool
    failures: tuple[str, ...]


class GovernanceVerificationAuditRepository(Protocol):
    def append_verification(
        self,
        *,
        recorded_at: datetime,
        artifact_type: str,
        artifact_sha256: str,
        envelope_sha256: str,
        trust_store_sha256: str,
        policy_sha256: str,
        verification: GovernanceSignatureVerificationReport,
    ) -> GovernanceVerificationAudit: ...

    def list(self) -> list[GovernanceVerificationAudit]: ...

    def verify_chain(self) -> GovernanceAuditChainReport: ...


class SQLiteGovernanceVerificationAuditRepository:
    """Append-only API with a hash chain; database administrators remain outside its trust model."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS governance_signature_audits (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    audit_id TEXT NOT NULL UNIQUE,
                    previous_entry_sha256 TEXT,
                    entry_sha256 TEXT NOT NULL UNIQUE,
                    payload TEXT NOT NULL
                )
                """
            )

    def append_verification(
        self,
        *,
        recorded_at: datetime,
        artifact_type: str,
        artifact_sha256: str,
        envelope_sha256: str,
        trust_store_sha256: str,
        policy_sha256: str,
        verification: GovernanceSignatureVerificationReport,
    ) -> GovernanceVerificationAudit:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = self._rows(connection)
            chain = self._verify_rows(rows)
            if not chain.valid:
                raise RuntimeError("governance audit chain is corrupt; refusing append")
            draft = GovernanceVerificationAudit(
                recorded_at=recorded_at,
                artifact_type=artifact_type,
                artifact_sha256=artifact_sha256,
                envelope_sha256=envelope_sha256,
                trust_store_sha256=trust_store_sha256,
                policy_sha256=policy_sha256,
                verification=verification,
                previous_entry_sha256=chain.head_sha256,
                entry_sha256="0" * 64,
            )
            audit = draft.model_copy(
                update={"entry_sha256": draft.computed_entry_sha256()}
            )
            connection.execute(
                """
                INSERT INTO governance_signature_audits
                    (audit_id, previous_entry_sha256, entry_sha256, payload)
                VALUES (?, ?, ?, ?)
                """,
                (
                    str(audit.audit_id),
                    audit.previous_entry_sha256,
                    audit.entry_sha256,
                    audit.model_dump_json(),
                ),
            )
            connection.commit()
            return audit

    def list(self) -> list[GovernanceVerificationAudit]:
        with self._connect() as connection:
            rows = self._rows(connection)
            if not self._verify_rows(rows).valid:
                raise RuntimeError("governance audit chain is corrupt; refusing audit read")
            return [GovernanceVerificationAudit.model_validate_json(row[4]) for row in rows]

    def verify_chain(self) -> GovernanceAuditChainReport:
        with self._connect() as connection:
            return self._verify_rows(self._rows(connection))

    @staticmethod
    def _verify_rows(rows: list[tuple]) -> GovernanceAuditChainReport:
        failures: list[str] = []
        previous: str | None = None
        for expected_sequence, row in enumerate(rows, 1):
            sequence, audit_id, stored_previous, stored_entry, payload = row
            if sequence != expected_sequence:
                failures.append(f"sequence_gap:{expected_sequence}")
            try:
                audit = GovernanceVerificationAudit.model_validate_json(payload)
            except ValueError:
                failures.append(f"invalid_payload:{sequence}")
                previous = stored_entry
                continue
            if str(audit.audit_id) != audit_id:
                failures.append(f"audit_id_mismatch:{sequence}")
            if audit.previous_entry_sha256 != stored_previous or stored_previous != previous:
                failures.append(f"predecessor_mismatch:{sequence}")
            if audit.entry_sha256 != stored_entry:
                failures.append(f"stored_digest_mismatch:{sequence}")
            if audit.computed_entry_sha256() != stored_entry:
                failures.append(f"payload_digest_mismatch:{sequence}")
            previous = stored_entry
        return GovernanceAuditChainReport(
            entries=len(rows),
            head_sha256=previous,
            valid=not failures,
            failures=tuple(failures),
        )

    def _rows(self, connection: sqlite3.Connection) -> list[tuple]:
        return connection.execute(
            """
            SELECT sequence, audit_id, previous_entry_sha256, entry_sha256, payload
            FROM governance_signature_audits ORDER BY sequence
            """
        ).fetchall()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5, isolation_level=None)


class AuditedGovernanceSignatureVerifier:
    def __init__(
        self,
        verification_service: GovernanceSignatureVerificationService,
        audit_repository: GovernanceVerificationAuditRepository,
    ) -> None:
        self._service = verification_service
        self._audits = audit_repository

    def verify_and_record(
        self,
        artifact: bytes,
        envelope: GovernanceSignatureEnvelope,
        trust_store: GovernanceTrustStore,
        policy: GovernanceSignaturePolicy,
        *,
        checked_at: datetime,
        recorded_at: datetime,
    ) -> GovernanceSignatureVerificationReport:
        recorded_at = _require_aware(recorded_at)
        checked_at = _require_aware(checked_at)
        if recorded_at < checked_at:
            raise ValueError("audit recording time cannot precede signature verification")
        report = self._service.verify(
            artifact, envelope, trust_store, policy, checked_at=checked_at
        )
        self._audits.append_verification(
            recorded_at=recorded_at,
            artifact_type=envelope.artifact_type,
            artifact_sha256=report.artifact_sha256,
            envelope_sha256=envelope.sha256(),
            trust_store_sha256=trust_store.sha256(),
            policy_sha256=policy.sha256(),
            verification=report,
        )
        return report


class GovernanceVerificationRuntime:
    """Explicitly assemble trusted roots, policy, verifier and durable audit storage."""

    def __init__(
        self,
        *,
        trust_store_path: Path,
        policy_path: Path,
        audit_database_path: Path,
        verifiers: tuple[DetachedSignatureVerifier, ...],
    ) -> None:
        if not trust_store_path.is_file():
            raise ValueError("governance trust store must be an existing explicit file")
        if not policy_path.is_file():
            raise ValueError("governance signature policy must be an existing explicit file")
        self.trust_store = GovernanceTrustStore.load(trust_store_path)
        self.policy = GovernanceSignaturePolicy.load(policy_path)
        self.audit_repository = SQLiteGovernanceVerificationAuditRepository(audit_database_path)
        self.verifier = AuditedGovernanceSignatureVerifier(
            GovernanceSignatureVerificationService(verifiers), self.audit_repository
        )
