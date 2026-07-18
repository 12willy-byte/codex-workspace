import pytest

from aquaguard.world import CameraObservation, TemporalFusionCoordinator, WorldModelPipeline


def observation(camera_id: str, timestamp: float, x: float) -> CameraObservation:
    return CameraObservation(
        camera_id=camera_id,
        track_id="person-global-1",
        timestamp=timestamp,
        pool_x=x,
        pool_y=4,
        confidence=1,
        motion=0.5,
    )


def test_coordinator_fuses_near_time_camera_frames() -> None:
    coordinator = TemporalFusionCoordinator(WorldModelPipeline(), max_skew_seconds=0.1)
    coordinator.process_frame("cam-a", 1.0, [observation("cam-a", 1.0, 4)])

    result = coordinator.process_frame("cam-b", 1.05, [observation("cam-b", 1.05, 6)])

    assert result["tracks"][0]["camera_ids"] == ("cam-a", "cam-b")
    assert result["tracks"][0]["position"] == {"x": 5.0, "y": 4.0}
    assert result["sync"]["camera_ids"] == ["cam-a", "cam-b"]


def test_coordinator_excludes_stale_camera_frames() -> None:
    coordinator = TemporalFusionCoordinator(WorldModelPipeline(), max_skew_seconds=0.1)
    coordinator.process_frame("cam-a", 1.0, [observation("cam-a", 1.0, 4)])

    result = coordinator.process_frame("cam-b", 1.2, [observation("cam-b", 1.2, 6)])

    assert result["tracks"][0]["camera_ids"] == ("cam-b",)
    assert result["sync"]["camera_ids"] == ["cam-b"]


def test_coordinator_rejects_out_of_order_frames_per_camera() -> None:
    coordinator = TemporalFusionCoordinator(WorldModelPipeline())
    coordinator.process_frame("cam-a", 2.0, [])

    with pytest.raises(ValueError, match="Out-of-order"):
        coordinator.process_frame("cam-a", 1.0, [])


def test_coordinator_rejects_cross_camera_observations() -> None:
    coordinator = TemporalFusionCoordinator(WorldModelPipeline())

    with pytest.raises(ValueError, match="supplied camera"):
        coordinator.process_frame("cam-a", 1.0, [observation("cam-b", 1.0, 4)])


def test_deferred_result_is_not_corrupted_by_external_mutation() -> None:
    coordinator = TemporalFusionCoordinator(WorldModelPipeline())
    result = coordinator.process_frame("cam-a", 2.0, [observation("cam-a", 2.0, 4)])
    result["tracks"].clear()

    deferred = coordinator.process_frame("cam-b", 1.0, [])

    assert len(deferred["tracks"]) == 1
