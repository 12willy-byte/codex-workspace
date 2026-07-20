from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from aquaguard.vision.annotation import AnnotationReview
from aquaguard.vision.annotation_quality import (
    AnnotationAgreementReport,
    CohenAgreementReporter,
    ReviewerCalibrationSet,
    ReviewerCalibrationSubmission,
    ReviewerQualificationEvaluator,
    ReviewerQualificationPolicy,
    ReviewerQualificationReport,
)


def _require_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("qualification timestamps must include a timezone")
    return value


class ReviewerQualificationRecord(BaseModel):
    """Immutable evidence that one reviewer qualified for one exact protocol."""

    model_config = {"frozen": True}

    schema_version: Literal["1.0"] = "1.0"
    reviewer_id: str = Field(min_length=1)
    protocol_id: str = Field(min_length=1)
    annotation_version: str = Field(min_length=1)
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    qualification_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    qualification_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    predecessor_qualification_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    retraining_completion_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    qualified: bool
    qualified_at: datetime
    expires_at: datetime

    @field_validator("reviewer_id", "protocol_id", "annotation_version")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("qualification identity fields must not be blank")
        return value

    @field_validator("qualified_at", "expires_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        return _require_aware(value)

    @model_validator(mode="after")
    def require_positive_validity_window(self) -> ReviewerQualificationRecord:
        if self.expires_at <= self.qualified_at:
            raise ValueError("qualification expiry must be after qualification time")
        renewal_links = (
            self.predecessor_qualification_sha256,
            self.retraining_completion_sha256,
        )
        if any(link is not None for link in renewal_links) and not all(
            link is not None for link in renewal_links
        ):
            raise ValueError("renewed qualification requires predecessor and retraining links")
        return self

    def is_active_at(self, checked_at: datetime) -> bool:
        checked_at = _require_aware(checked_at)
        return self.qualified and self.qualified_at <= checked_at < self.expires_at

    def canonical_bytes(self) -> bytes:
        serialized = json.dumps(
            self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True
        )
        return (serialized + "\n").encode()

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


class ReviewerQualificationIssuer:
    """Issue a lifecycle record only from a reproducible passing evaluation."""

    def issue(
        self,
        calibration_set: ReviewerCalibrationSet,
        submission: ReviewerCalibrationSubmission,
        policy: ReviewerQualificationPolicy,
        report: ReviewerQualificationReport,
        *,
        qualified_at: datetime,
        expires_at: datetime,
    ) -> ReviewerQualificationRecord:
        recomputed = ReviewerQualificationEvaluator().evaluate(calibration_set, submission, policy)
        if recomputed != report:
            raise ValueError("qualification report does not match recomputed evaluation")
        if not report.qualified:
            raise ValueError("failed qualification cannot be issued as a valid record")
        return ReviewerQualificationRecord(
            reviewer_id=submission.reviewer_id,
            protocol_id=calibration_set.protocol_id,
            annotation_version=calibration_set.annotation_version,
            protocol_sha256=calibration_set.protocol_sha256,
            calibration_set_sha256=calibration_set.sha256(),
            qualification_policy_sha256=policy.sha256(),
            qualification_report_sha256=report.sha256(),
            qualified=True,
            qualified_at=qualified_at,
            expires_at=expires_at,
        )


class AnnotationBatchAdmissionPolicy(BaseModel):
    """Project-owned thresholds; this repository intentionally supplies no defaults."""

    minimum_items: int = Field(ge=1)
    minimum_observed_agreement: float = Field(ge=0, le=1)
    minimum_cohen_kappa: float = Field(ge=-1, le=1)
    maximum_uncertain_fraction: float = Field(ge=0, le=1)


class AnnotationBatchAdmissionReport(BaseModel):
    checked_at: datetime
    protocol_id: str
    annotation_version: str
    protocol_sha256: str
    reviewer_ids: tuple[str, str]
    admitted: bool
    denial_reasons: tuple[str, ...]
    agreement: AnnotationAgreementReport

    @model_validator(mode="after")
    def require_consistent_decision(self) -> AnnotationBatchAdmissionReport:
        if self.reviewer_ids != self.agreement.reviewer_ids:
            raise ValueError("admission and agreement reviewer ids must match")
        if self.admitted == bool(self.denial_reasons):
            raise ValueError("admitted batches require no denials; denied batches require reasons")
        return self


class AnnotationBatchAdmissionService:
    """Fail closed unless both reviewers are current and batch quality passes."""

    def evaluate(
        self,
        reviews: list[AnnotationReview],
        qualifications: list[ReviewerQualificationRecord],
        policy: AnnotationBatchAdmissionPolicy,
        *,
        checked_at: datetime,
    ) -> AnnotationBatchAdmissionReport:
        checked_at = _require_aware(checked_at)
        agreement = CohenAgreementReporter().evaluate(reviews)
        protocol_references = {
            (review.protocol_id, review.annotation_version, review.protocol_sha256)
            for review in reviews
        }
        protocol_id, annotation_version, protocol_sha256 = next(iter(protocol_references))

        by_reviewer: dict[str, ReviewerQualificationRecord] = {}
        duplicate_reviewers: set[str] = set()
        for record in qualifications:
            if record.reviewer_id in by_reviewer:
                duplicate_reviewers.add(record.reviewer_id)
            by_reviewer[record.reviewer_id] = record

        reasons: list[str] = []
        expected_reviewers = set(agreement.reviewer_ids)
        if set(by_reviewer) != expected_reviewers:
            reasons.append("qualification_reviewer_set_mismatch")
        if duplicate_reviewers:
            reasons.append("duplicate_qualification_record")

        expected_protocol = (protocol_id, annotation_version, protocol_sha256)
        for reviewer_id in agreement.reviewer_ids:
            record = by_reviewer.get(reviewer_id)
            if record is None:
                reasons.append(f"missing_qualification:{reviewer_id}")
                continue
            record_protocol = (
                record.protocol_id,
                record.annotation_version,
                record.protocol_sha256,
            )
            if record_protocol != expected_protocol:
                reasons.append(f"qualification_protocol_mismatch:{reviewer_id}")
            if not record.qualified:
                reasons.append(f"reviewer_not_qualified:{reviewer_id}")
            elif not record.is_active_at(checked_at):
                reasons.append(f"qualification_inactive:{reviewer_id}")

        uncertain_fraction = agreement.uncertain_items / agreement.items
        checks = (
            (agreement.items >= policy.minimum_items, "minimum_items"),
            (
                agreement.observed_agreement >= policy.minimum_observed_agreement,
                "minimum_observed_agreement",
            ),
            (
                agreement.cohen_kappa is not None
                and agreement.cohen_kappa >= policy.minimum_cohen_kappa,
                "minimum_cohen_kappa",
            ),
            (
                uncertain_fraction <= policy.maximum_uncertain_fraction,
                "maximum_uncertain_fraction",
            ),
        )
        reasons.extend(name for passed, name in checks if not passed)
        return AnnotationBatchAdmissionReport(
            checked_at=checked_at,
            protocol_id=protocol_id,
            annotation_version=annotation_version,
            protocol_sha256=protocol_sha256,
            reviewer_ids=agreement.reviewer_ids,
            admitted=not reasons,
            denial_reasons=tuple(reasons),
            agreement=agreement,
        )
