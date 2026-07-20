from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel, Field, field_validator, model_validator

from aquaguard.governance_audit import GovernanceVerificationAuditRepository
from aquaguard.governance_signing import _canonical_bytes
from aquaguard.vision.annotation_governance import _require_aware


class GovernanceAuditCheckpoint(BaseModel):
    model_config = {"frozen": True}

    schema_version: Literal["1.0"] = "1.0"
    stream_id: str = Field(min_length=1)
    checkpoint_number: int = Field(ge=1)
    audit_entries: int = Field(ge=1)
    audit_head_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    previous_checkpoint_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    anchored_at: datetime

    @field_validator("stream_id")
    @classmethod
    def normalize_stream_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("checkpoint stream_id must not be blank")
        return value

    @field_validator("anchored_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        return _require_aware(value)

    @model_validator(mode="after")
    def require_initial_link_semantics(self) -> GovernanceAuditCheckpoint:
        if self.checkpoint_number == 1 and self.previous_checkpoint_sha256 is not None:
            raise ValueError("initial checkpoint cannot have a predecessor")
        if self.checkpoint_number > 1 and self.previous_checkpoint_sha256 is None:
            raise ValueError("later checkpoints require a predecessor digest")
        return self

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.model_dump(mode="json"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


class GovernanceCheckpointStore(Protocol):
    """Production implementations must provide externally controlled append-only storage."""

    def latest(self, stream_id: str) -> GovernanceAuditCheckpoint | None: ...

    def publish(
        self,
        checkpoint: GovernanceAuditCheckpoint,
        *,
        expected_previous_sha256: str | None,
    ) -> None: ...


class GovernanceCheckpointVerificationReport(BaseModel):
    stream_id: str
    checkpoint_sha256: str
    checkpoint_entries: int
    current_entries: int
    current_head_sha256: str | None
    valid: bool
    failures: tuple[str, ...]


class GovernanceAuditCheckpointVerifier:
    def verify(
        self,
        repository: GovernanceVerificationAuditRepository,
        checkpoint: GovernanceAuditCheckpoint,
    ) -> GovernanceCheckpointVerificationReport:
        chain = repository.verify_chain()
        failures = list(chain.failures)
        if chain.entries < checkpoint.audit_entries:
            failures.append("audit_rollback_or_truncation")
        else:
            anchored_digest = repository.entry_sha256_at(checkpoint.audit_entries)
            if anchored_digest != checkpoint.audit_head_sha256:
                failures.append("audit_history_diverged_from_checkpoint")
        return GovernanceCheckpointVerificationReport(
            stream_id=checkpoint.stream_id,
            checkpoint_sha256=checkpoint.sha256(),
            checkpoint_entries=checkpoint.audit_entries,
            current_entries=chain.entries,
            current_head_sha256=chain.head_sha256,
            valid=not failures,
            failures=tuple(failures),
        )


class GovernanceAuditCheckpointPublisher:
    def publish(
        self,
        repository: GovernanceVerificationAuditRepository,
        store: GovernanceCheckpointStore,
        *,
        stream_id: str,
        anchored_at: datetime,
    ) -> GovernanceAuditCheckpoint:
        anchored_at = _require_aware(anchored_at)
        stream_id = stream_id.strip()
        if not stream_id:
            raise ValueError("checkpoint stream_id must not be blank")
        chain = repository.verify_chain()
        if not chain.valid:
            raise RuntimeError("governance audit chain is corrupt; refusing checkpoint")
        if chain.entries == 0 or chain.head_sha256 is None:
            raise ValueError("cannot checkpoint an empty governance audit chain")

        previous = store.latest(stream_id)
        if previous is not None:
            verification = GovernanceAuditCheckpointVerifier().verify(repository, previous)
            if not verification.valid:
                raise RuntimeError("current audit does not contain the latest external checkpoint")
            if chain.entries == previous.audit_entries:
                if chain.head_sha256 != previous.audit_head_sha256:
                    raise RuntimeError("current audit head conflicts with external checkpoint")
                return previous
            if anchored_at <= previous.anchored_at:
                raise ValueError("new checkpoint time must follow the previous checkpoint")

        checkpoint = GovernanceAuditCheckpoint(
            stream_id=stream_id,
            checkpoint_number=1 if previous is None else previous.checkpoint_number + 1,
            audit_entries=chain.entries,
            audit_head_sha256=chain.head_sha256,
            previous_checkpoint_sha256=None if previous is None else previous.sha256(),
            anchored_at=anchored_at,
        )
        store.publish(
            checkpoint,
            expected_previous_sha256=None if previous is None else previous.sha256(),
        )
        return checkpoint
