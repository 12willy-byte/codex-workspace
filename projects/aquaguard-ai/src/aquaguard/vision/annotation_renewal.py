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
