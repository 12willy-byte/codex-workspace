from fastapi.testclient import TestClient

from aquaguard.main import app, evidence_service, service

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_video_runtime_status_api() -> None:
    response = client.get("/api/v1/video-runtimes")
    assert response.status_code == 200
    assert response.json() == []


def test_evaluation_validation() -> None:
    response = client.post("/api/v1/evaluations", json={"camera_id": "C01", "track_id": "T01", "area": "child-pool", "features": {"head_underwater": 2, "vertical_body": 0, "abnormal_motion": 0, "temporal_risk": 0}})
    assert response.status_code == 422


def test_world_model_api() -> None:
    response = client.post(
        "/api/v1/world-model/frames",
        json={
            "observations": [
                {
                    "camera_id": "C01",
                    "track_id": "T01",
                    "timestamp": 1.0,
                    "pool_x": 4.0,
                    "pool_y": 2.0,
                    "confidence": 0.9,
                    "motion": 0.8,
                }
            ]
        },
    )
    assert response.status_code == 200
    assert response.json()["tracks"][0]["track_id"] == "T01"
    assert "occupancy" in response.json()["bev"]


def test_evidence_status_api_reports_pending_and_missing() -> None:
    evidence_service.recorder.pending.clear()
    evidence_service.recorder.completed.clear()
    evidence_service.artifacts.clear()
    evidence_service.recorder.request("event-pending", "C01", 10, pre_seconds=2, post_seconds=3)

    response = client.get("/api/v1/events/event-pending/evidence")
    missing = client.get("/api/v1/events/unknown/evidence")

    assert response.status_code == 200
    assert response.json()["status"] == "pending"
    assert response.json()["starts_at"] == 8
    assert missing.status_code == 404


def test_confirmed_evaluation_is_suppressed_without_validated_camera() -> None:
    service.engine._consecutive.clear()
    service.events.clear()
    service._last_alarm.clear()
    evidence_service.recorder.pending.clear()
    evidence_service.recorder.completed.clear()
    track_id = "api-evidence-track"
    payload = {
        "camera_id": "C01",
        "track_id": track_id,
        "area": "child-pool",
        "observed_at": 100,
        "features": {
            "head_underwater": 1,
            "vertical_body": 1,
            "abnormal_motion": 1,
            "temporal_risk": 1,
        },
    }

    responses = [client.post("/api/v1/evaluations", json=payload) for _ in range(3)]
    result = responses[-1].json()

    assert all(response.status_code == 200 for response in responses)
    assert result["assessment"]["confirmed"] is True
    assert result["event"] is None
    assert result["alarm_eligible"] is False
    assert result["suppression_reason"] == "camera_protection_level:unconfigured"
    assert evidence_service.recorder.pending == {}

    audits = client.get("/api/v1/evaluation-audits")
    assert audits.status_code == 200
    assert audits.json()[-1]["track_id"] == track_id
    assert audits.json()[-1]["suppression_reason"] == result["suppression_reason"]
