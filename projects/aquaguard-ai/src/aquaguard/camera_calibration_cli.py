import argparse
import json
from pathlib import Path

from aquaguard.annotation_quality_cli import _write_json
from aquaguard.vision.camera_calibration import (
    CameraCalibrationArtifact,
    CameraCalibrationVerifier,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aquaguard-verify-camera-calibration",
        description="Validate a traceable per-camera pool calibration artifact.",
    )
    parser.add_argument("calibration", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args(argv)
    try:
        artifact = CameraCalibrationArtifact.load(args.calibration)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _write_json(args.report, {"valid": False, "failures": [str(exc)]})
        return 2
    report = CameraCalibrationVerifier().verify(artifact)
    _write_json(args.report, report.model_dump(mode="json"))
    return 0
