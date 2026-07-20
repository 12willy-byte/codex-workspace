import hashlib

import pytest
from pydantic import ValidationError

from aquaguard.temporal_model_cli import main as temporal_model_main
from aquaguard.world import (
    TEMPORAL_FEATURE_SCHEMA_V1,
    CameraObservation,
    TemporalRiskModelManifest,
    ValidatedTemporalRiskPredictor,
    WorldModelPipeline,
)


class RecordingModel:
    def __init__(self, output: object) -> None:
        self.output = output
        self.sequences: list[tuple[tuple[float, ...], ...]] = []

    def predict(self, sequence: tuple[tuple[float, ...], ...]) -> object:
        self.sequences.append(sequence)
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


def model_and_manifest(tmp_path, *, validated: bool = False, **updates):
    model_path = tmp_path / "temporal-model.bin"
    model_path.write_bytes(b"synthetic temporal model fixture")
    values = {
        "model_version": "temporal-fixture-v1",
        "artifact_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "feature_schema": "aquaguard.temporal_features.v1",
        "minimum_samples": 2,
        "maximum_samples": 4,
        "minimum_duration_seconds": 0.5,
        "maximum_gap_seconds": 1.5,
        "validated_for_assistive_alerting": validated,
        "validation_bundle_sha256": "b" * 64 if validated else None,
        "validation_report_sha256": "c" * 64 if validated else None,
        "approval_artifact_sha256": "d" * 64 if validated else None,
    }
    values.update(updates)
    return model_path, TemporalRiskModelManifest(**values)


def observation(timestamp: float, *, severe: bool = False) -> CameraObservation:
    danger = 0.9 if severe else 0.1
    return CameraObservation(
        camera_id="cam-a",
        track_id="person-1",
        timestamp=timestamp,
        pool_x=5,
        pool_y=4,
        confidence=0.9,
        head_submerged=danger,
        body_vertical=danger,
        struggle=danger,
        motion=1 - danger,
        occlusion=0.1,
        head_in_water_region=1,
        water_relation_confidence=0.8,
        wrist_motion=0.2,
        wrist_motion_confidence=0.7,
    )


def high_risk_output() -> dict:
    return {
        "current_risk": 0.9,
        "future_risk": 0.95,
        "time_to_critical_seconds": 1.2,
        "uncertainty": 0.1,
        "reasons": ["learned_danger_pattern"],
    }


def test_validated_model_receives_fixed_feature_sequence_and_can_confirm(tmp_path) -> None:
    path, manifest = model_and_manifest(tmp_path, validated=True)
    model = RecordingModel(high_risk_output())
    predictor = ValidatedTemporalRiskPredictor(
        model, manifest, path, approval_verified=True
    )
    pipeline = WorldModelPipeline(predictor=predictor)

    first = pipeline.process([observation(0)])
    warning = pipeline.process([observation(1)])
    alarm = pipeline.process([observation(2)])

    assert first["tracks"][0]["forecast"]["uncertainty"] == 1
    assert first["tracks"][0]["forecast"]["reasons"] == (
        "temporal_model_insufficient_samples",
    )
    assert warning["tracks"][0]["decision"]["should_alarm"] is False
    assert alarm["tracks"][0]["decision"]["should_alarm"] is True
    assert alarm["tracks"][0]["forecast"]["predictor_kind"] == "learned_temporal"
    assert alarm["tracks"][0]["forecast"]["model_version"] == manifest.model_version
    assert len(model.sequences[-1][-1]) == len(TEMPORAL_FEATURE_SCHEMA_V1)
    assert model.sequences[-1][-1][0] == 2


def test_unvalidated_model_cannot_drive_prediction_only_alarm(tmp_path) -> None:
    path, manifest = model_and_manifest(tmp_path)
    model = RecordingModel(high_risk_output())
    pipeline = WorldModelPipeline(
        predictor=ValidatedTemporalRiskPredictor(model, manifest, path)
    )
    pipeline.process([observation(0)])
    result = pipeline.process([observation(1)])

    decision = result["tracks"][0]["decision"]
    assert decision["should_alarm"] is False
    assert decision["level"] == "observe"
    assert "unvalidated_learned_model" in decision["reasons"]


def test_manifest_validation_claim_without_verified_approval_is_suppressed(tmp_path) -> None:
    path, manifest = model_and_manifest(tmp_path, validated=True)
    model = RecordingModel(high_risk_output())
    pipeline = WorldModelPipeline(
        predictor=ValidatedTemporalRiskPredictor(model, manifest, path)
    )
    pipeline.process([observation(0)])
    result = pipeline.process([observation(1)])

    assert result["tracks"][0]["forecast"]["assistive_alerting_eligible"] is False
    assert result["tracks"][0]["decision"]["should_alarm"] is False


def test_independent_severe_signal_survives_unvalidated_model(tmp_path) -> None:
    path, manifest = model_and_manifest(tmp_path)
    model = RecordingModel(high_risk_output())
    pipeline = WorldModelPipeline(
        predictor=ValidatedTemporalRiskPredictor(model, manifest, path)
    )
    pipeline.process([observation(0, severe=True)])
    result = pipeline.process([observation(1, severe=True)])

    assert result["tracks"][0]["decision"]["should_alarm"] is True
    assert "independent_severe_signal" in result["tracks"][0]["decision"]["reasons"]


@pytest.mark.parametrize(
    ("timestamps", "reason"),
    [
        ((0,), "temporal_model_insufficient_samples"),
        ((0, 0), "temporal_model_non_increasing_timestamps"),
        ((0, 2), "temporal_model_sequence_gap"),
    ],
)
def test_invalid_temporal_sequences_fail_closed(tmp_path, timestamps, reason) -> None:
    path, manifest = model_and_manifest(tmp_path)
    model = RecordingModel(high_risk_output())
    predictor = ValidatedTemporalRiskPredictor(model, manifest, path)
    pipeline = WorldModelPipeline(predictor=predictor)

    result = None
    for timestamp in timestamps:
        result = pipeline.process([observation(timestamp)])

    assert result is not None
    assert result["tracks"][0]["forecast"]["uncertainty"] == 1
    assert reason in result["tracks"][0]["forecast"]["reasons"]
    assert model.sequences == []


@pytest.mark.parametrize(
    "output",
    [
        RuntimeError("backend failed"),
        {"current_risk": 2, "future_risk": 0, "uncertainty": 0},
        {"current_risk": 0, "future_risk": 0, "uncertainty": 0, "reasons": [""]},
    ],
)
def test_inference_errors_and_invalid_outputs_fail_closed(tmp_path, output) -> None:
    path, manifest = model_and_manifest(tmp_path, validated=True)
    model = RecordingModel(output)
    pipeline = WorldModelPipeline(
        predictor=ValidatedTemporalRiskPredictor(model, manifest, path)
    )
    pipeline.process([observation(0)])
    result = pipeline.process([observation(1)])

    forecast = result["tracks"][0]["forecast"]
    assert forecast["uncertainty"] == 1
    assert forecast["assistive_alerting_eligible"] is False
    assert forecast["reasons"] == ("temporal_model_inference_failed",)


def test_model_artifact_digest_and_manifest_approval_are_strict(tmp_path) -> None:
    path, manifest = model_and_manifest(tmp_path)
    path.write_bytes(b"replaced model")
    with pytest.raises(ValueError, match="SHA-256"):
        ValidatedTemporalRiskPredictor(RecordingModel(high_risk_output()), manifest, path)

    with pytest.raises(ValidationError, match="bundle, report, and approval"):
        model_and_manifest(tmp_path, validated=True, validation_bundle_sha256=None)
    with pytest.raises(ValidationError, match="cannot claim"):
        model_and_manifest(tmp_path, validation_bundle_sha256="c" * 64)
    clean_path, unvalidated = model_and_manifest(tmp_path)
    with pytest.raises(ValueError, match="without validation evidence"):
        ValidatedTemporalRiskPredictor(
            RecordingModel(high_risk_output()),
            unvalidated,
            clean_path,
            approval_verified=True,
        )


def test_model_manifest_round_trip_and_package_command(tmp_path) -> None:
    path, manifest = model_and_manifest(tmp_path, validated=True)
    manifest_path = tmp_path / "model-manifest.json"
    report_path = tmp_path / "model-report.json"
    manifest.save(manifest_path)

    assert TemporalRiskModelManifest.load(manifest_path) == manifest
    assert temporal_model_main([str(manifest_path), str(path), str(report_path)]) == 0
    report = report_path.read_text(encoding="utf-8")
    assert manifest.sha256() in report
    assert '"manifest_declares_assistive_validation": true' in report

    path.write_bytes(b"tampered")
    assert temporal_model_main([str(manifest_path), str(path), str(report_path)]) == 2
    assert "does not match manifest" in report_path.read_text(encoding="utf-8")
