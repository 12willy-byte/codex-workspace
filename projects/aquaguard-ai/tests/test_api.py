from fastapi.testclient import TestClient

from aquaguard.main import app
from aquaguard.main import evidence_service

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


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
