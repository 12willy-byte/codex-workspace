import pytest

from aquaguard.world import CameraObservation, PoolGeometry, WorldModelPipeline


def observation(
    camera: str,
    timestamp: float,
    *,
    danger: float,
    occlusion: float = 0.0,
    head_in_water_region: float = 0.0,
    water_relation_confidence: float = 0.0,
):
    return CameraObservation(
        camera_id=camera,
        track_id="person-1",
        timestamp=timestamp,
        pool_x=5.0,
        pool_y=4.0,
        confidence=0.9,
        head_submerged=danger,
        body_vertical=danger,
        struggle=danger,
        motion=1.0 - danger,
        occlusion=occlusion,
        head_in_water_region=head_in_water_region,
        water_relation_confidence=water_relation_confidence,
    )


def test_multicamera_fusion_populates_one_bev_track() -> None:
    pipeline = WorldModelPipeline(PoolGeometry(grid_width=5, grid_height=2))
    result = pipeline.process(
        [observation("cam-a", 1.0, danger=0.2), observation("cam-b", 1.0, danger=0.2)]
    )
    assert len(result["tracks"]) == 1
    assert result["tracks"][0]["camera_ids"] == ("cam-a", "cam-b")
    assert sum(sum(row) for row in result["bev"]["occupancy"]) == pytest.approx(0.9)


def test_world_output_exposes_auditable_water_relation_without_submersion_claim() -> None:
    result = WorldModelPipeline().process(
        [
            observation(
                "cam-a",
                1,
                danger=0,
                head_in_water_region=1,
                water_relation_confidence=0.8,
            )
        ]
    )

    features = result["tracks"][0]["features"]
    assert features["head_in_water_region"] == 1
    assert features["water_relation_confidence"] == 0.8
    assert features["head_submerged"] == 0


def test_future_risk_and_supervisor_escalate_continuous_danger() -> None:
    pipeline = WorldModelPipeline()
    pipeline.process([observation("cam-a", 0.0, danger=0.2)])
    warning = pipeline.process([observation("cam-a", 1.0, danger=0.85)])
    emergency = pipeline.process([observation("cam-a", 2.0, danger=0.9)])
    assert warning["tracks"][0]["forecast"]["future_risk"] >= warning["tracks"][0][
        "forecast"
    ]["current_risk"]
    assert emergency["tracks"][0]["decision"]["should_alarm"] is True


def test_high_uncertainty_suppresses_prediction_only_alarm() -> None:
    result = WorldModelPipeline().process(
        [observation("cam-a", 1.0, danger=0.75, occlusion=1.0)]
    )
    decision = result["tracks"][0]["decision"]
    assert decision["should_alarm"] is False
    assert "high_uncertainty" in decision["reasons"]


def test_observation_outside_pool_is_rejected() -> None:
    bad = CameraObservation("cam", "person", 0.0, 99.0, 2.0)
    with pytest.raises(ValueError, match="outside"):
        WorldModelPipeline().process([bad])
