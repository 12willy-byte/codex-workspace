import json

import pytest
from pydantic import ValidationError

from aquaguard.benchmark_cli import main as benchmark_main
from aquaguard.vision import (
    CameraCalibrationRecord,
    EvaluationProvenance,
    FrameRiskLabels,
    PixelObservationRecord,
    ReplayEvaluationBundle,
    ReplayEvaluationFrame,
    ReplayEvaluationResult,
    ReplayEvaluationRunner,
    TrackRiskLabel,
)


def observation(timestamp: float, danger: float) -> PixelObservationRecord:
    return PixelObservationRecord(
        camera_id="cam-a",
        local_track_id="local-1",
        global_track_id="person-1",
        timestamp=timestamp,
        anchor_x=50,
        anchor_y=40,
        confidence=0.9,
        head_submerged=danger,
        body_vertical=danger,
        struggle=danger,
        motion=1 - danger,
    )


def evaluation_frame(
    timestamp: float, danger: float, labelled_danger: bool
) -> ReplayEvaluationFrame:
    return ReplayEvaluationFrame(
        timestamp=timestamp,
        observations=(observation(timestamp, danger),),
        labels=FrameRiskLabels(
            timestamp=timestamp,
            tracks=(TrackRiskLabel(track_id="person-1", dangerous=labelled_danger),),
        ),
    )


def bundle() -> ReplayEvaluationBundle:
    return ReplayEvaluationBundle(
        name="controlled scenario 001",
        provenance=EvaluationProvenance(
            source_recording_sha256="a" * 64,
            model_version="scripted-observation-v1",
            configuration_version="risk-default-v1",
            annotation_version="review-protocol-v1",
        ),
        calibrations=(
            CameraCalibrationRecord(
                camera_id="cam-a",
                homography=(0.1, 0, 0, 0, 0.1, 0, 0, 0, 1),
            ),
        ),
        frames=(
            evaluation_frame(0, 0.2, False),
            evaluation_frame(1, 0.85, True),
            evaluation_frame(2, 0.9, True),
        ),
    )


def test_evaluation_bundle_round_trip_digest_and_replay(tmp_path) -> None:
    original = bundle()
    path = tmp_path / "evaluation.json"
    original.save(path)
    recovered = ReplayEvaluationBundle.load(path)

    assert recovered == original
    assert recovered.sha256() == original.sha256()

    result = ReplayEvaluationRunner().run(recovered)

    assert result.bundle_sha256 == original.sha256()
    assert result.provenance["model_version"] == "scripted-observation-v1"
    assert result.report.frames == 3
    assert result.report.true_positive == 2
    assert result.report.false_negative == 0
    assert result.report.dangerous_episodes == 1
    assert result.report.detected_episodes == 1
    assert result.report.mean_detection_latency_seconds == 0

    report_path = tmp_path / "report.json"
    result.save(report_path)
    stored = json.loads(report_path.read_text())
    assert stored["bundle_sha256"] == original.sha256()
    assert stored["report"]["recall"] == 1


def test_bundle_rejects_observations_without_calibration() -> None:
    original = bundle()

    with pytest.raises(ValidationError, match="missing camera calibrations: cam-a"):
        ReplayEvaluationBundle(
            name=original.name,
            provenance=original.provenance,
            calibrations=(),
            frames=original.frames,
        )


def test_frame_rejects_misaligned_observation_and_label_timestamps() -> None:
    labels = FrameRiskLabels(
        timestamp=1,
        tracks=(TrackRiskLabel(track_id="person-1", dangerous=False),),
    )

    with pytest.raises(ValidationError, match="frame and label timestamps must match"):
        ReplayEvaluationFrame(timestamp=0, observations=(observation(0, 0.1),), labels=labels)

    with pytest.raises(ValidationError, match="frame and observation timestamps must match"):
        ReplayEvaluationFrame(
            timestamp=0,
            observations=(observation(1, 0.1),),
            labels=labels.model_copy(update={"timestamp": 0}),
        )


def test_result_keeps_provenance_and_report_immutable_enough_for_export() -> None:
    result = ReplayEvaluationRunner().run(bundle())
    restored = ReplayEvaluationResult(
        bundle_sha256=result.bundle_sha256,
        provenance=dict(result.provenance),
        report=result.report,
    )

    assert restored.to_dict() == result.to_dict()


def test_benchmark_command_writes_traceable_report(tmp_path) -> None:
    bundle_path = tmp_path / "bundle.json"
    report_path = tmp_path / "report.json"
    original = bundle()
    original.save(bundle_path)

    assert benchmark_main([str(bundle_path), str(report_path)]) == 0

    stored = json.loads(report_path.read_text())
    assert stored["bundle_sha256"] == original.sha256()
    assert stored["provenance"]["annotation_version"] == "review-protocol-v1"
    assert stored["report"]["frames"] == 3
