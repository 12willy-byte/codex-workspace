from fastapi.testclient import TestClient

from aquaguard.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_evaluation_validation() -> None:
    response = client.post("/api/v1/evaluations", json={"camera_id": "C01", "track_id": "T01", "area": "child-pool", "features": {"head_underwater": 2, "vertical_body": 0, "abnormal_motion": 0, "temporal_risk": 0}})
    assert response.status_code == 422
