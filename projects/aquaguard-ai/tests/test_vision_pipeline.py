import pytest

from aquaguard.vision import (
    CalibratedObservationAdapter,
    HomographyProjector,
    ObservationReplay,
    PixelTrackObservation,
)
from aquaguard.world import WorldModelPipeline


def projector(camera_id: str, x_offset: float = 0.0) -> HomographyProjector:
    return HomographyProjector(camera_id, (0.1, 0, x_offset, 0, 0.1, 0, 0, 0, 1))


def pixel_observation(
    camera_id: str,
    timestamp: float,
    danger: float,
    *,
    global_track_id: str | None = "person-1",
) -> PixelTrackObservation:
    return PixelTrackObservation(
        camera_id=camera_id,
        local_track_id="local-1",
        global_track_id=global_track_id,
        timestamp=timestamp,
        anchor_x=50,
        anchor_y=40,
        confidence=0.9,
        head_submerged=danger,
        body_vertical=danger,
        struggle=danger,
        motion=1 - danger,
    )


def test_homography_projects_pixels_to_pool_coordinates() -> None:
    assert projector("cam-a").project(50, 40) == pytest.approx((5, 4))


def test_homography_rejects_singular_or_non_finite_matrix() -> None:
    with pytest.raises(ValueError, match="invertible"):
        HomographyProjector("cam-a", (1, 0, 0, 0, 0, 0, 0, 0, 1))
    with pytest.raises(ValueError, match="finite"):
        HomographyProjector("cam-a", (1, 0, 0, 0, 1, 0, 0, 0, float("nan")))


def test_homography_validation_is_scale_invariant() -> None:
    scaled = HomographyProjector("cam-a", tuple(value * 1e-6 for value in projector("x").matrix))

    assert scaled.project(50, 40) == pytest.approx((5, 4))


def test_two_calibrated_cameras_fuse_only_with_explicit_global_identity() -> None:
    adapter = CalibratedObservationAdapter(
        {"cam-a": projector("cam-a"), "cam-b": projector("cam-b")}
    )
    pipeline = WorldModelPipeline()
    fused = pipeline.process(
        adapter.convert(
            [pixel_observation("cam-a", 1, 0.2), pixel_observation("cam-b", 1, 0.2)]
        )
    )
    isolated = pipeline.process(
        adapter.convert(
            [
                pixel_observation("cam-a", 2, 0.2, global_track_id=None),
                pixel_observation("cam-b", 2, 0.2, global_track_id=None),
            ]
        )
    )
    assert len(fused["tracks"]) == 1
    assert len(isolated["tracks"]) == 2


def test_missing_calibration_is_rejected() -> None:
    adapter = CalibratedObservationAdapter({})
    with pytest.raises(ValueError, match="No calibration"):
        adapter.convert([pixel_observation("cam-a", 1, 0.2)])


def test_replay_drives_future_risk_and_safety_supervisor() -> None:
    frames = [
        [pixel_observation("cam-a", 0, 0.2)],
        [pixel_observation("cam-a", 1, 0.85)],
        [pixel_observation("cam-a", 2, 0.9)],
    ]
    replay = ObservationReplay(
        frames,
        CalibratedObservationAdapter({"cam-a": projector("cam-a")}),
        WorldModelPipeline(),
    )
    results = list(replay.run())
    assert results[-1]["tracks"][0]["decision"]["should_alarm"] is True
