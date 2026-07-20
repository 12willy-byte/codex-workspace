from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from aquaguard.vision.annotation import AnnotationReview, RiskJudgment


class CalibrationGoldItem(BaseModel):
    item_id: str = Field(min_length=1)
    dangerous: bool

    @field_validator("item_id")
    @classmethod
    def normalize_item_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("calibration item_id must not be blank")
        return value


class ReviewerCalibrationSet(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    calibration_set_id: str = Field(min_length=1)
    calibration_set_version: str = Field(min_length=1)
    protocol_id: str = Field(min_length=1)
    annotation_version: str = Field(min_length=1)
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    items: tuple[CalibrationGoldItem, ...] = Field(min_length=1)

    @field_validator(
        "calibration_set_id", "calibration_set_version", "protocol_id", "annotation_version"
    )
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("calibration set fields must not be blank")
        return value

    @field_validator("items")
    @classmethod
    def require_unique_items(
        cls, value: tuple[CalibrationGoldItem, ...]
    ) -> tuple[CalibrationGoldItem, ...]:
        item_ids = [item.item_id for item in value]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("calibration item ids must be unique")
        return value

    @classmethod
    def load(cls, path: Path) -> ReviewerCalibrationSet:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def canonical_bytes(self, *, pretty: bool = False) -> bytes:
        options = {"ensure_ascii": False, "sort_keys": True}
        if pretty:
            options["indent"] = 2
        serialized = json.dumps(self.model_dump(mode="json"), **options)
        return (serialized + "\n").encode()

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.canonical_bytes(pretty=True))


class CalibrationResponse(BaseModel):
    item_id: str = Field(min_length=1)
    judgment: RiskJudgment


class ReviewerCalibrationSubmission(BaseModel):
    reviewer_id: str = Field(min_length=1)
    protocol_id: str = Field(min_length=1)
    annotation_version: str = Field(min_length=1)
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    responses: tuple[CalibrationResponse, ...] = Field(min_length=1)


class ReviewerQualificationPolicy(BaseModel):
    minimum_items: int = Field(ge=1)
    minimum_dangerous_items: int = Field(ge=1)
    minimum_safe_items: int = Field(ge=1)
    minimum_accuracy: float = Field(ge=0, le=1)
    minimum_dangerous_recall: float = Field(ge=0, le=1)
    minimum_safe_specificity: float = Field(ge=0, le=1)
    maximum_uncertain_fraction: float = Field(ge=0, le=1)

    def canonical_bytes(self) -> bytes:
        serialized = json.dumps(
            self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True
        )
        return (serialized + "\n").encode()

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


class ReviewerQualificationReport(BaseModel):
    reviewer_id: str
    calibration_set_sha256: str
    total_items: int
    dangerous_items: int
    safe_items: int
    correct_items: int
    uncertain_items: int
    accuracy: float
    dangerous_recall: float
    safe_specificity: float
    uncertain_fraction: float
    qualified: bool
    failed_requirements: tuple[str, ...]

    def canonical_bytes(self) -> bytes:
        serialized = json.dumps(
            self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True
        )
        return (serialized + "\n").encode()

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


class ReviewerQualificationEvaluator:
    def evaluate(
        self,
        calibration_set: ReviewerCalibrationSet,
        submission: ReviewerCalibrationSubmission,
        policy: ReviewerQualificationPolicy,
    ) -> ReviewerQualificationReport:
        expected_reference = (
            calibration_set.protocol_id,
            calibration_set.annotation_version,
            calibration_set.protocol_sha256,
            calibration_set.sha256(),
        )
        submitted_reference = (
            submission.protocol_id,
            submission.annotation_version,
            submission.protocol_sha256,
            submission.calibration_set_sha256,
        )
        if submitted_reference != expected_reference:
            raise ValueError("calibration submission does not match protocol or calibration set")

        responses = {item.item_id: item.judgment for item in submission.responses}
        if len(responses) != len(submission.responses):
            raise ValueError("calibration response item ids must be unique")
        gold = {item.item_id: item.dangerous for item in calibration_set.items}
        if set(responses) != set(gold):
            raise ValueError("calibration submission must answer every item exactly once")

        total = len(gold)
        dangerous_items = sum(gold.values())
        safe_items = total - dangerous_items
        correct = sum(
            (judgment == RiskJudgment.DANGEROUS) == dangerous and judgment != RiskJudgment.UNCERTAIN
            for item_id, judgment in responses.items()
            for dangerous in (gold[item_id],)
        )
        dangerous_correct = sum(
            responses[item_id] == RiskJudgment.DANGEROUS
            for item_id, dangerous in gold.items()
            if dangerous
        )
        safe_correct = sum(
            responses[item_id] == RiskJudgment.SAFE
            for item_id, dangerous in gold.items()
            if not dangerous
        )
        uncertain = sum(value == RiskJudgment.UNCERTAIN for value in responses.values())

        accuracy = correct / total
        dangerous_recall = dangerous_correct / dangerous_items if dangerous_items else 0.0
        safe_specificity = safe_correct / safe_items if safe_items else 0.0
        uncertain_fraction = uncertain / total
        failed: list[str] = []
        checks = (
            (total >= policy.minimum_items, "minimum_items"),
            (dangerous_items >= policy.minimum_dangerous_items, "minimum_dangerous_items"),
            (safe_items >= policy.minimum_safe_items, "minimum_safe_items"),
            (accuracy >= policy.minimum_accuracy, "minimum_accuracy"),
            (
                dangerous_recall >= policy.minimum_dangerous_recall,
                "minimum_dangerous_recall",
            ),
            (
                safe_specificity >= policy.minimum_safe_specificity,
                "minimum_safe_specificity",
            ),
            (
                uncertain_fraction <= policy.maximum_uncertain_fraction,
                "maximum_uncertain_fraction",
            ),
        )
        failed.extend(name for passed, name in checks if not passed)
        return ReviewerQualificationReport(
            reviewer_id=submission.reviewer_id,
            calibration_set_sha256=calibration_set.sha256(),
            total_items=total,
            dangerous_items=dangerous_items,
            safe_items=safe_items,
            correct_items=correct,
            uncertain_items=uncertain,
            accuracy=accuracy,
            dangerous_recall=dangerous_recall,
            safe_specificity=safe_specificity,
            uncertain_fraction=uncertain_fraction,
            qualified=not failed,
            failed_requirements=tuple(failed),
        )


class AnnotationAgreementReport(BaseModel):
    reviewer_ids: tuple[str, str]
    items: int
    agreed_items: int
    disputed_items: int
    uncertain_items: int
    observed_agreement: float
    expected_agreement: float
    cohen_kappa: float | None


class CohenAgreementReporter:
    """Compute Cohen's kappa only when the same two reviewers cover every item."""

    def evaluate(self, reviews: list[AnnotationReview]) -> AnnotationAgreementReport:
        if not reviews:
            raise ValueError("at least one annotation review is required")
        protocol_references = {
            (item.protocol_id, item.annotation_version, item.protocol_sha256) for item in reviews
        }
        if len(protocol_references) != 1:
            raise ValueError("agreement report requires one annotation protocol")
        reviewer_ids = tuple(sorted({item.reviewer_id for item in reviews}))
        if len(reviewer_ids) != 2:
            raise ValueError("Cohen's kappa requires the same two reviewers")

        grouped: dict[tuple[str, float, str], dict[str, RiskJudgment]] = defaultdict(dict)
        for review in reviews:
            key = review.recording_id, review.timestamp, review.track_id
            if review.reviewer_id in grouped[key]:
                raise ValueError("reviewer submitted duplicate judgment for an item")
            grouped[key][review.reviewer_id] = review.judgment
        if any(set(item) != set(reviewer_ids) for item in grouped.values()):
            raise ValueError("both reviewers must cover every agreement item")

        first_counts: Counter[RiskJudgment] = Counter()
        second_counts: Counter[RiskJudgment] = Counter()
        agreed = uncertain = 0
        for judgments in grouped.values():
            first = judgments[reviewer_ids[0]]
            second = judgments[reviewer_ids[1]]
            first_counts[first] += 1
            second_counts[second] += 1
            agreed += first == second
            uncertain += RiskJudgment.UNCERTAIN in (first, second)
        total = len(grouped)
        observed = agreed / total
        expected = sum(
            (first_counts[category] / total) * (second_counts[category] / total)
            for category in RiskJudgment
        )
        kappa = None if expected == 1 else (observed - expected) / (1 - expected)
        return AnnotationAgreementReport(
            reviewer_ids=reviewer_ids,
            items=total,
            agreed_items=agreed,
            disputed_items=total - agreed,
            uncertain_items=uncertain,
            observed_agreement=observed,
            expected_agreement=expected,
            cohen_kappa=kappa,
        )
