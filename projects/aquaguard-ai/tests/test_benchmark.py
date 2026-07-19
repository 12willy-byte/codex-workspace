import pytest
from pydantic import ValidationError

from aquaguard.vision import (
    BinaryRiskBenchmark,
    FrameRiskLabels,
    RiskBenchmarkManifest,
    TrackRiskLabel,
)


def frame(timestamp: float, dangerous: bool) -> FrameRiskLabels:
    return FrameRiskLabels(
        timestamp=timestamp,
        tracks=(TrackRiskLabel(track_id="person-1", dangerous=dangerous),),
    )


def result(alarmed: bool, track_id: str = "person-1") -> dict:
    return {
        "tracks": [
            {
                "track_id": track_id,
                "decision": {"should_alarm": alarmed},
            }
        ]
    }


def test_benchmark_reports_frame_metrics_episode_recall_and_latency() -> None:
    manifest = RiskBenchmarkManifest(
        name="scripted danger transition",
        source="synthetic/test-case-1",
        frames=(frame(0, False), frame(1, True), frame(2, True), frame(3, False)),
    )

    report = BinaryRiskBenchmark().evaluate(
        [result(False), result(False), result(True), result(False)], manifest
    )

    assert report.true_positive == 1
    assert report.false_positive == 0
    assert report.true_negative == 2
    assert report.false_negative == 1
    assert report.precision == 1
    assert report.recall == 0.5
    assert report.f1 == pytest.approx(2 / 3)
    assert report.dangerous_episodes == 1
    assert report.detected_episodes == 1
    assert report.missed_episodes == 0
    assert report.episode_recall == 1
    assert report.mean_detection_latency_seconds == 1
    assert report.max_detection_latency_seconds == 1


def test_benchmark_counts_missed_episode_and_unlabelled_alarm() -> None:
    manifest = RiskBenchmarkManifest(
        name="miss and phantom",
        source="synthetic/test-case-2",
        frames=(frame(0, True), frame(1, False)),
    )
    results = [
        result(False),
        {
            "tracks": [
                {"track_id": "person-1", "decision": {"should_alarm": False}},
                {"track_id": "phantom", "decision": {"should_alarm": True}},
            ]
        },
    ]

    report = BinaryRiskBenchmark().evaluate(results, manifest)

    assert report.false_negative == 1
    assert report.false_positive == 1
    assert report.dangerous_episodes == 1
    assert report.detected_episodes == 0
    assert report.missed_episodes == 1
    assert report.episode_recall == 0
    assert report.mean_detection_latency_seconds is None


def test_benchmark_does_not_invent_metrics_without_positive_examples() -> None:
    manifest = RiskBenchmarkManifest(
        name="negative only",
        source="synthetic/test-case-3",
        frames=(frame(0, False),),
    )

    report = BinaryRiskBenchmark().evaluate([result(False)], manifest)

    assert report.precision is None
    assert report.recall is None
    assert report.f1 is None
    assert report.episode_recall is None


def test_benchmark_requires_result_count_to_match_manifest() -> None:
    manifest = RiskBenchmarkManifest(
        name="length mismatch",
        source="synthetic/test-case-4",
        frames=(frame(0, False),),
    )

    with pytest.raises(ValueError, match="result count"):
        BinaryRiskBenchmark().evaluate([], manifest)


def test_manifest_rejects_duplicate_tracks_and_unordered_timestamps(tmp_path) -> None:
    duplicate = TrackRiskLabel(track_id="person-1", dangerous=False)
    with pytest.raises(ValidationError, match="must be unique"):
        FrameRiskLabels(timestamp=0, tracks=(duplicate, duplicate))

    with pytest.raises(ValidationError, match="unique and increasing"):
        RiskBenchmarkManifest(
            name="unordered",
            source="synthetic/test-case-5",
            frames=(frame(1, False), frame(0, False)),
        )

    manifest = RiskBenchmarkManifest(
        name="round trip",
        source="synthetic/test-case-6",
        frames=(frame(0, False),),
    )
    path = tmp_path / "manifest.json"
    manifest.save(path)

    assert RiskBenchmarkManifest.load(path) == manifest
