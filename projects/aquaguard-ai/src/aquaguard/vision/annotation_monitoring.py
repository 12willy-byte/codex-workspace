from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from aquaguard.vision.annotation_governance import (
    AnnotationBatchAdmissionReport,
    _require_aware,
)


class AnnotationBatchQualitySnapshot(BaseModel):
    model_config = {"frozen": True}

    schema_version: Literal["1.0"] = "1.0"
    batch_id: str = Field(min_length=1)
    completed_at: datetime
    admission: AnnotationBatchAdmissionReport

    @field_validator("batch_id")
    @classmethod
    def normalize_batch_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("batch_id must not be blank")
        return value

    @field_validator("completed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        return _require_aware(value)

    @model_validator(mode="after")
    def require_check_not_after_completion(self) -> AnnotationBatchQualitySnapshot:
        if self.admission.checked_at > self.completed_at:
            raise ValueError("admission check cannot occur after batch completion")
        return self


class ContinuousAnnotationQualityPolicy(BaseModel):
    """Explicit governance values; no thresholds are implied by this repository."""

    recent_window_batches: int = Field(ge=1)
    minimum_batches_before_evaluation: int = Field(ge=1)
    minimum_weighted_observed_agreement: float = Field(ge=0, le=1)
    minimum_weighted_cohen_kappa: float = Field(ge=-1, le=1)
    maximum_weighted_uncertain_fraction: float = Field(ge=0, le=1)
    maximum_consecutive_denied_batches: int = Field(ge=0)
    minimum_alerts_for_retraining: int = Field(ge=1)

    @model_validator(mode="after")
    def require_enough_history_for_window(self) -> ContinuousAnnotationQualityPolicy:
        if self.minimum_batches_before_evaluation < self.recent_window_batches:
            raise ValueError("minimum batches must cover the recent window")
        return self


class AnnotationQualityTrendReport(BaseModel):
    evaluated_at: datetime
    protocol_id: str
    annotation_version: str
    protocol_sha256: str
    reviewer_ids: tuple[str, str]
    batches_seen: int
    recent_batches: int
    recent_items: int
    weighted_observed_agreement: float | None
    weighted_cohen_kappa: float | None
    weighted_uncertain_fraction: float | None
    observed_agreement_delta: float | None
    cohen_kappa_delta: float | None
    uncertain_fraction_delta: float | None
    consecutive_denied_batches: int
    alerts: tuple[str, ...]
    retraining_required: bool


class ContinuousAnnotationQualityMonitor:
    """Evaluate immutable batch reports without revoking credentials or training models."""

    def evaluate(
        self,
        snapshots: list[AnnotationBatchQualitySnapshot],
        policy: ContinuousAnnotationQualityPolicy,
        *,
        evaluated_at: datetime,
    ) -> AnnotationQualityTrendReport:
        evaluated_at = _require_aware(evaluated_at)
        if not snapshots:
            raise ValueError("at least one annotation batch snapshot is required")
        batch_ids = [snapshot.batch_id for snapshot in snapshots]
        if len(batch_ids) != len(set(batch_ids)):
            raise ValueError("annotation batch ids must be unique")
        completed = [snapshot.completed_at for snapshot in snapshots]
        if completed != sorted(completed) or len(completed) != len(set(completed)):
            raise ValueError("annotation batches must have unique ascending completion times")
        if completed[-1] > evaluated_at:
            raise ValueError("quality evaluation cannot precede a batch completion")

        references = {
            (
                snapshot.admission.protocol_id,
                snapshot.admission.annotation_version,
                snapshot.admission.protocol_sha256,
                snapshot.admission.reviewer_ids,
            )
            for snapshot in snapshots
        }
        if len(references) != 1:
            raise ValueError("quality trend requires one protocol and reviewer pair")
        protocol_id, annotation_version, protocol_sha256, reviewer_ids = next(iter(references))

        recent = snapshots[-policy.recent_window_batches :]
        previous = snapshots[
            max(0, len(snapshots) - 2 * policy.recent_window_batches) : -policy.recent_window_batches
        ]
        current_metrics = self._weighted_metrics(recent)
        previous_metrics = (
            self._weighted_metrics(previous)
            if len(previous) == policy.recent_window_batches
            else (None, None, None, 0)
        )
        observed, kappa, uncertain, recent_items = current_metrics
        previous_observed, previous_kappa, previous_uncertain, _ = previous_metrics

        consecutive_denied = 0
        for snapshot in reversed(snapshots):
            if snapshot.admission.admitted:
                break
            consecutive_denied += 1

        alerts: list[str] = []
        if len(snapshots) < policy.minimum_batches_before_evaluation:
            alerts.append("insufficient_batch_history")
        else:
            checks = (
                (
                    observed is not None
                    and observed >= policy.minimum_weighted_observed_agreement,
                    "observed_agreement_below_minimum",
                ),
                (
                    kappa is not None and kappa >= policy.minimum_weighted_cohen_kappa,
                    "cohen_kappa_below_minimum_or_unavailable",
                ),
                (
                    uncertain is not None
                    and uncertain <= policy.maximum_weighted_uncertain_fraction,
                    "uncertain_fraction_above_maximum",
                ),
                (
                    consecutive_denied <= policy.maximum_consecutive_denied_batches,
                    "consecutive_denied_batches_above_maximum",
                ),
            )
            alerts.extend(name for passed, name in checks if not passed)

        return AnnotationQualityTrendReport(
            evaluated_at=evaluated_at,
            protocol_id=protocol_id,
            annotation_version=annotation_version,
            protocol_sha256=protocol_sha256,
            reviewer_ids=reviewer_ids,
            batches_seen=len(snapshots),
            recent_batches=len(recent),
            recent_items=recent_items,
            weighted_observed_agreement=observed,
            weighted_cohen_kappa=kappa,
            weighted_uncertain_fraction=uncertain,
            observed_agreement_delta=self._delta(observed, previous_observed),
            cohen_kappa_delta=self._delta(kappa, previous_kappa),
            uncertain_fraction_delta=self._delta(uncertain, previous_uncertain),
            consecutive_denied_batches=consecutive_denied,
            alerts=tuple(alerts),
            retraining_required=(
                len(alerts) >= policy.minimum_alerts_for_retraining
                and "insufficient_batch_history" not in alerts
            ),
        )

    @staticmethod
    def _weighted_metrics(
        snapshots: list[AnnotationBatchQualitySnapshot],
    ) -> tuple[float | None, float | None, float | None, int]:
        total_items = sum(snapshot.admission.agreement.items for snapshot in snapshots)
        if total_items == 0:
            return None, None, None, 0
        observed = sum(
            snapshot.admission.agreement.observed_agreement
            * snapshot.admission.agreement.items
            for snapshot in snapshots
        ) / total_items
        uncertain = sum(
            snapshot.admission.agreement.uncertain_items for snapshot in snapshots
        ) / total_items
        if any(snapshot.admission.agreement.cohen_kappa is None for snapshot in snapshots):
            kappa = None
        else:
            kappa = sum(
                snapshot.admission.agreement.cohen_kappa
                * snapshot.admission.agreement.items
                for snapshot in snapshots
                if snapshot.admission.agreement.cohen_kappa is not None
            ) / total_items
        return observed, kappa, uncertain, total_items

    @staticmethod
    def _delta(current: float | None, previous: float | None) -> float | None:
        if current is None or previous is None:
            return None
        return current - previous
