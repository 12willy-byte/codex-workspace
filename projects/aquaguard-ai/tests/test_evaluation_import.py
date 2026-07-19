import json

import pytest

from aquaguard.evaluation_import_cli import main as import_main
from aquaguard.vision import (
    CameraCalibrationRecord,
    EvaluationCalibrationFile,
    FrameRiskLabels,
    PixelObservationRecord,
    ReplayEvaluationBundle,
    ReplayEvaluationImporter,
    TrackRiskLabel,
    sha256_file,
)


def write_jsonl(path, records) -> None:
    path.write_text(
        "\n".join(json.dumps(record.model_dump(mode="json")) for record in records) + "\n",
        encoding="utf-8",
    )


def source_files(tmp_path):
    source = tmp_path / "recording.mp4"
    source.write_bytes(b"controlled recording fixture")
    observations = tmp_path / "observations.jsonl"
    labels = tmp_path / "labels.jsonl"
    calibration = tmp_path / "calibration.json"
    write_jsonl(
        observations,
        [
            PixelObservationRecord(
                camera_id="cam-a",
                local_track_id="local-1",
                global_track_id="person-1",
                timestamp=0,
                anchor_x=50,
                anchor_y=40,
            )
        ],
    )
    write_jsonl(
        labels,
        [
            FrameRiskLabels(
                timestamp=0,
                tracks=(TrackRiskLabel(track_id="person-1", dangerous=False),),
            ),
            FrameRiskLabels(
                timestamp=1,
                tracks=(TrackRiskLabel(track_id="person-1", dangerous=True),),
            ),
        ],
    )
    calibration.write_text(
        EvaluationCalibrationFile(
            calibrations=(
                CameraCalibrationRecord(
                    camera_id="cam-a",
                    homography=(0.1, 0, 0, 0, 0.1, 0, 0, 0, 1),
                ),
            )
        ).model_dump_json(),
        encoding="utf-8",
    )
    return source, observations, labels, calibration


def build(importer, files, **overrides):
    source, observations, labels, calibration = files
    arguments = {
        "name": "controlled import",
        "source_recording_path": source,
        "observations_path": observations,
        "labels_path": labels,
        "calibration_path": calibration,
        "model_version": "model-v1",
        "configuration_version": "config-v1",
        "annotation_version": "annotation-v1",
        "annotation_protocol_sha256": "a" * 64,
    }
    arguments.update(overrides)
    return importer.build(**arguments)


def test_importer_hashes_source_and_keeps_labelled_missed_detection_frame(tmp_path) -> None:
    files = source_files(tmp_path)
    bundle = build(
        ReplayEvaluationImporter(),
        files,
        expected_source_sha256=sha256_file(files[0]),
    )

    assert bundle.provenance.source_recording_sha256 == sha256_file(files[0])
    assert len(bundle.frames) == 2
    assert len(bundle.frames[0].observations) == 1
    assert bundle.frames[1].observations == ()
    assert bundle.frames[1].labels.tracks[0].dangerous is True


def test_importer_rejects_source_digest_mismatch(tmp_path) -> None:
    files = source_files(tmp_path)

    with pytest.raises(ValueError, match="does not match"):
        build(ReplayEvaluationImporter(), files, expected_source_sha256="0" * 64)


def test_importer_rejects_observations_without_frame_labels(tmp_path) -> None:
    files = source_files(tmp_path)
    observations = PixelObservationRecord(
        camera_id="cam-a",
        local_track_id="local-2",
        timestamp=2,
        anchor_x=10,
        anchor_y=10,
    )
    with files[1].open("a", encoding="utf-8") as destination:
        destination.write(observations.model_dump_json() + "\n")

    with pytest.raises(ValueError, match="unlabelled timestamps: 2.0"):
        build(ReplayEvaluationImporter(), files)


def test_importer_reports_invalid_jsonl_line(tmp_path) -> None:
    files = source_files(tmp_path)
    first_label = FrameRiskLabels(
        timestamp=0,
        tracks=(TrackRiskLabel(track_id="person-1", dangerous=False),),
    )
    files[2].write_text(first_label.model_dump_json() + "\nnot-json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid labels.jsonl line 2"):
        build(ReplayEvaluationImporter(), files)


def test_import_command_writes_validated_bundle(tmp_path) -> None:
    source, observations, labels, calibration = source_files(tmp_path)
    output = tmp_path / "bundle.json"

    result = import_main(
        [
            str(source),
            str(observations),
            str(labels),
            str(calibration),
            str(output),
            "--name",
            "controlled import",
            "--model-version",
            "model-v1",
            "--configuration-version",
            "config-v1",
            "--annotation-version",
            "annotation-v1",
            "--annotation-protocol-sha256",
            "a" * 64,
            "--expected-source-sha256",
            sha256_file(source),
        ]
    )

    assert result == 0
    recovered = ReplayEvaluationBundle.load(output)
    assert recovered.provenance.source_recording_sha256 == sha256_file(source)
