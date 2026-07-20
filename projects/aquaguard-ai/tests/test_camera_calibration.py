import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from aquaguard.assembly import CameraRuntimeConfig
from aquaguard.camera_calibration_cli import main as calibration_main
from aquaguard.vision import (
    CalibrationCorrespondence,
    CameraCalibrationArtifact,
    CameraCalibrationVerifier,
)


NOW = datetime(2026, 7, 20, 12, tzinfo=timezone.utc)
IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def artifact(**updates) -> CameraCalibrationArtifact:
    values = {
        "camera_id": "cam-a",
        "calibration_version": "site-a-2026-07-20",
        "source_frame_sha256": "a" * 64,
        "frame_width": 100,
        "frame_height": 80,
        "homography": IDENTITY,
        "water_roi": ((10, 10), (90, 10), (90, 70), (10, 70)),
        "correspondences": (
            CalibrationCorrespondence(image_x=10, image_y=10, pool_x=10, pool_y=10),
            CalibrationCorrespondence(image_x=90, image_y=10, pool_x=90, pool_y=10),
            CalibrationCorrespondence(image_x=90, image_y=70, pool_x=90, pool_y=70),
            CalibrationCorrespondence(image_x=10, image_y=70, pool_x=10, pool_y=70),
        ),
        "maximum_reprojection_error_meters": 0.1,
        "calibrated_at": NOW,
        "calibrated_by": "technician-17",
    }
    values.update(updates)
    return CameraCalibrationArtifact(**values)


def test_calibration_round_trip_report_and_runtime_binding(tmp_path) -> None:
    source = artifact()
    path = tmp_path / "camera-calibration.json"
    source.save(path)

    recovered = CameraCalibrationArtifact.load(path)
    report = CameraCalibrationVerifier().verify(recovered)
    runtime = CameraRuntimeConfig.from_calibration(
        recovered,
        source="recording.mp4",
        enabled=True,
        analyzer="ultralytics_pose_tracking",
        model_path="pose.pt",
    )

    assert recovered == source
    assert report.valid is True
    assert report.control_points == 4
    assert report.water_roi_area_pixels == 4800
    assert report.max_reprojection_error_meters == 0
    assert runtime.camera_id == source.camera_id
    assert runtime.water_roi == source.water_roi
    assert runtime.calibration_sha256 == source.sha256()


def test_calibration_rejects_out_of_frame_roi_and_control_points() -> None:
    with pytest.raises(ValidationError, match="water ROI must stay within"):
        artifact(water_roi=((10, 10), (100, 10), (90, 70), (10, 70)))
    controls = list(artifact().correspondences)
    controls[0] = controls[0].model_copy(update={"image_x": -1})
    with pytest.raises(ValidationError, match="correspondence must stay within"):
        artifact(correspondences=tuple(controls))


def test_calibration_rejects_excessive_reprojection_error() -> None:
    controls = list(artifact().correspondences)
    controls[0] = controls[0].model_copy(update={"pool_x": 11})

    with pytest.raises(ValidationError, match="exceeds maximum reprojection"):
        artifact(correspondences=tuple(controls))


def test_calibration_rejects_duplicate_controls_and_naive_time() -> None:
    controls = artifact().correspondences
    with pytest.raises(ValidationError, match="image correspondences must be unique"):
        artifact(correspondences=controls[:3] + (controls[0],))
    with pytest.raises(ValidationError, match="timezone"):
        artifact(calibrated_at=NOW.replace(tzinfo=None))


def test_calibration_control_points_must_span_two_dimensions() -> None:
    controls = tuple(
        CalibrationCorrespondence(image_x=i, image_y=i, pool_x=i, pool_y=i)
        for i in (10, 20, 30, 40)
    )
    with pytest.raises(ValidationError, match="image correspondences must span"):
        artifact(correspondences=controls)


def test_calibration_command_writes_valid_and_invalid_reports(tmp_path) -> None:
    valid_path = tmp_path / "valid.json"
    valid_report = tmp_path / "valid-report.json"
    artifact().save(valid_path)
    assert calibration_main([str(valid_path), str(valid_report)]) == 0
    assert json.loads(valid_report.read_text())["valid"] is True

    invalid_path = tmp_path / "invalid.json"
    invalid_report = tmp_path / "invalid-report.json"
    invalid_path.write_text("{}", encoding="utf-8")
    assert calibration_main([str(invalid_path), str(invalid_report)]) == 2
    invalid = json.loads(invalid_report.read_text())
    assert invalid["valid"] is False
    assert invalid["failures"]
