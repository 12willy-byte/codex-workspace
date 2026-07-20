from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from aquaguard.vision.annotation_governance import (
    ReviewerQualificationRecord,
    _require_aware,
)
from aquaguard.vision.annotation_quality import (
    ReviewerCalibrationSet,
    ReviewerCalibrationSubmission,
    ReviewerQualificationEvaluator,
    ReviewerQualificationPolicy,
    ReviewerQualificationReport,
)


class ReviewerRetrainingCompletion(BaseModel):
    """Immutable evidence of completed retraining; not proof of qualification by itself."""

    model_config = {"frozen": True}

    schema_version: Literal["1.0"] = "1.0"
    completion_id: str = Field(min_length=1)
    reviewer_id: str = Field(min_length=1)
    protocol_id: str = Field(min_length=1)
    annotation_version: str = Field(min_length=1)
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_material_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    completion_evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    completed_at: datetime
    assessor_id: str = Field(min_length=1)

    @field_validator(
        "completion_id", "reviewer_id", "protocol_id", "annotation_version", "assessor_id"
    )
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("retraining identity fields must not be blank")
        return value

    @field_validator("completed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        return _require_aware(value)

    @model_validator(mode="after")
    def require_independent_assessor(self) -> ReviewerRetrainingCompletion:
        if self.assessor_id == self.reviewer_id:
            raise ValueError("retraining assessor must be independent from the reviewer")
        return self

    def canonical_bytes(self) -> bytes:
        serialized = json.dumps(
            self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True
        )
        return (serialized + "\n").encode()

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


class QualificationRenewalPolicy(BaseModel):
    allow_same_calibration_set: bool


class ReviewerQualificationRenewalIssuer:
    """Append a new qualification only after retraining and a fresh passing evaluation."""

    def issue(
        self,
        predecessor: ReviewerQualificationRecord,
        retraining: ReviewerRetrainingCompletion,
        calibration_set: ReviewerCalibrationSet,
        submission: ReviewerCalibrationSubmission,
        qualification_policy: ReviewerQualificationPolicy,
        qualification_report: ReviewerQualificationReport,
        renewal_policy: QualificationRenewalPolicy,
        *,
        renewed_at: datetime,
        expires_at: datetime,
    ) -> ReviewerQualificationRecord:
        renewed_at = _require_aware(renewed_at)
        expires_at = _require_aware(expires_at)
        if not predecessor.qualified:
            raise ValueError("failed qualification cannot be used as renewal predecessor")
        identity = (
            predecessor.reviewer_id,
            predecessor.protocol_id,
            predecessor.annotation_version,
            predecessor.protocol_sha256,
        )
        retraining_identity = (
            retraining.reviewer_id,
            retraining.protocol_id,
            retraining.annotation_version,
            retraining.protocol_sha256,
        )
        calibration_identity = (
            submission.reviewer_id,
            calibration_set.protocol_id,
            calibration_set.annotation_version,
            calibration_set.protocol_sha256,
        )
        if retraining_identity != identity or calibration_identity != identity:
            raise ValueError("renewal evidence must match reviewer and annotation protocol")
        if retraining.completed_at <= predecessor.qualified_at:
            raise ValueError("retraining must occur after the preceding qualification")
        if renewed_at < retraining.completed_at:
            raise ValueError("renewal cannot precede retraining completion")
        if renewed_at < predecessor.expires_at:
            raise ValueError("renewal cannot overlap the preceding qualification")
        if expires_at <= renewed_at:
            raise ValueError("renewal expiry must be after renewal time")
        if (
            not renewal_policy.allow_same_calibration_set
            and calibration_set.sha256() == predecessor.calibration_set_sha256
        ):
            raise ValueError("renewal policy requires a different calibration set")

        recomputed = ReviewerQualificationEvaluator().evaluate(
            calibration_set, submission, qualification_policy
        )
        if recomputed != qualification_report:
            raise ValueError("renewal report does not match recomputed evaluation")
        if not qualification_report.qualified:
            raise ValueError("failed requalification cannot renew a qualification")
        return ReviewerQualificationRecord(
            reviewer_id=submission.reviewer_id,
            protocol_id=calibration_set.protocol_id,
            annotation_version=calibration_set.annotation_version,
            protocol_sha256=calibration_set.protocol_sha256,
            calibration_set_sha256=calibration_set.sha256(),
            qualification_policy_sha256=qualification_policy.sha256(),
            qualification_report_sha256=qualification_report.sha256(),
            predecessor_qualification_sha256=predecessor.sha256(),
            retraining_completion_sha256=retraining.sha256(),
            qualified=True,
            qualified_at=renewed_at,
            expires_at=expires_at,
        )


class QualificationHistoryVerificationReport(BaseModel):
    checked_at: datetime
    reviewer_id: str | None
    protocol_id: str | None
    annotation_version: str | None
    protocol_sha256: str | None
    qualification_sha256: tuple[str, ...]
    retraining_sha256: tuple[str, ...]
    active_qualification_sha256: str | None
    valid: bool
    failures: tuple[str, ...]


class QualificationHistoryVerifier:
    """Verify an append-only qualification chain without mutating its artifacts."""

    def verify(
        self,
        qualifications: list[ReviewerQualificationRecord],
        retraining_completions: list[ReviewerRetrainingCompletion],
        *,
        checked_at: datetime,
    ) -> QualificationHistoryVerificationReport:
        checked_at = _require_aware(checked_at)
        qualification_hashes = tuple(record.sha256() for record in qualifications)
        retraining_hashes = tuple(record.sha256() for record in retraining_completions)
        failures: list[str] = []
        if not qualifications:
            failures.append("missing_qualification_history")
            return QualificationHistoryVerificationReport(
                checked_at=checked_at,
                reviewer_id=None,
                protocol_id=None,
                annotation_version=None,
                protocol_sha256=None,
                qualification_sha256=(),
                retraining_sha256=retraining_hashes,
                active_qualification_sha256=None,
                valid=False,
                failures=tuple(failures),
            )

        if len(qualification_hashes) != len(set(qualification_hashes)):
            failures.append("duplicate_qualification_artifact")
        if len(retraining_hashes) != len(set(retraining_hashes)):
            failures.append("duplicate_retraining_artifact")
        completion_ids = [item.completion_id for item in retraining_completions]
        if len(completion_ids) != len(set(completion_ids)):
            failures.append("duplicate_retraining_completion_id")

        first = qualifications[0]
        identity = (
            first.reviewer_id,
            first.protocol_id,
            first.annotation_version,
            first.protocol_sha256,
        )
        qualification_times = [record.qualified_at for record in qualifications]
        if qualification_times != sorted(qualification_times) or len(qualification_times) != len(
            set(qualification_times)
        ):
            failures.append("qualification_times_not_unique_ascending")
        if (
            first.predecessor_qualification_sha256 is not None
            or first.retraining_completion_sha256 is not None
        ):
            failures.append("initial_qualification_has_renewal_links")

        retraining_by_hash = {
            item.sha256(): item for item in retraining_completions
        }
        used_retraining_hashes: set[str] = set()
        for index, record in enumerate(qualifications):
            record_identity = (
                record.reviewer_id,
                record.protocol_id,
                record.annotation_version,
                record.protocol_sha256,
            )
            if record_identity != identity:
                failures.append(f"qualification_identity_mismatch:{index}")
            if not record.qualified:
                failures.append(f"qualification_not_passed:{index}")
            if index == 0:
                continue
            predecessor = qualifications[index - 1]
            if record.predecessor_qualification_sha256 != predecessor.sha256():
                failures.append(f"predecessor_digest_mismatch:{index}")
            retraining_hash = record.retraining_completion_sha256
            retraining = retraining_by_hash.get(retraining_hash or "")
            if retraining is None:
                failures.append(f"missing_retraining_evidence:{index}")
                continue
            used_retraining_hashes.add(retraining.sha256())
            retraining_identity = (
                retraining.reviewer_id,
                retraining.protocol_id,
                retraining.annotation_version,
                retraining.protocol_sha256,
            )
            if retraining_identity != identity:
                failures.append(f"retraining_identity_mismatch:{index}")
            if not predecessor.qualified_at < retraining.completed_at <= record.qualified_at:
                failures.append(f"retraining_time_outside_renewal_interval:{index}")

        for retraining_hash in retraining_hashes:
            if retraining_hash not in used_retraining_hashes:
                failures.append(f"orphan_retraining_evidence:{retraining_hash}")

        active = [
            record.sha256() for record in qualifications if record.is_active_at(checked_at)
        ]
        if len(active) > 1:
            failures.append("multiple_active_qualifications")
        return QualificationHistoryVerificationReport(
            checked_at=checked_at,
            reviewer_id=identity[0],
            protocol_id=identity[1],
            annotation_version=identity[2],
            protocol_sha256=identity[3],
            qualification_sha256=qualification_hashes,
            retraining_sha256=retraining_hashes,
            active_qualification_sha256=active[0] if len(active) == 1 else None,
            valid=not failures,
            failures=tuple(failures),
        )
