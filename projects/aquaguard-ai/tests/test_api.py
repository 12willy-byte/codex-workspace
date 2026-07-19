import hashlib

from fastapi.testclient import TestClient

from aquaguard.auth import OperatorCredential, OperatorRole
from aquaguard.main import (
    app,
    evidence_service,
    operator_authenticator,
    service,
    settings,
)

client = TestClient(app)
ADMIN_TOKEN = "a-strong-local-admin-token-value-123456789"
operator_authenticator.credentials = (
    OperatorCredential(
        username="test-admin",
        role=OperatorRole.ADMIN,
        token_sha256=hashlib.sha256(ADMIN_TOKEN.encode()).hexdigest(),
    ),
)
client.headers["Authorization"] = f"Bearer {ADMIN_TOKEN}"


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_video_runtime_status_api() -> None:
    response = client.get("/api/v1/video-runtimes")
    assert response.status_code == 200
    assert response.json() == []


def test_operational_api_rejects_anonymous_requests() -> None:
    response = client.get("/api/v1/events", headers={"Authorization": ""})

    assert response.status_code == 401


def test_evaluation_validation() -> None:
    response = client.post(
        "/api/v1/evaluations",
        json={
            "camera_id": "C01",
            "track_id": "T01",
            "area": "child-pool",
            "features": {
                "head_underwater": 2,
                "vertical_body": 0,
                "abnormal_motion": 0,
                "temporal_risk": 0,
            },
        },
    )
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


def operator_headers(monkeypatch, role=OperatorRole.MAINTAINER) -> dict[str, str]:
    token = f"a-strong-local-{role.value}-token-value-123456789"
    credential = OperatorCredential(
        username=f"{role.value}-1",
        role=role,
        token_sha256=hashlib.sha256(token.encode()).hexdigest(),
    )
    monkeypatch.setattr(operator_authenticator, "credentials", (credential,))
    return {"Authorization": f"Bearer {token}"}


def test_evidence_consistency_api_requires_maintenance_role(monkeypatch) -> None:
    unauthenticated = client.get(
        "/api/v1/system/evidence-consistency", headers={"Authorization": ""}
    )
    response = client.get(
        "/api/v1/system/evidence-consistency",
        headers=operator_headers(monkeypatch),
    )

    assert unauthenticated.status_code == 401
    assert response.status_code == 200
    assert "events" in response.json()
    assert "orphaned_evidence_ids" in response.json()
    assert "counts" in response.json()


def test_evidence_remediation_api_is_disabled_without_explicit_configuration() -> None:
    response = client.post(
        "/api/v1/system/evidence-remediations",
        json={
            "target_id": "unknown-event",
            "action": "persist_ready",
            "operator": "operator-1",
            "reason": "test invalid request",
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "evidence remediation is disabled"


def test_enabled_evidence_remediation_api_requires_authentication(monkeypatch) -> None:
    monkeypatch.setattr(settings, "remediation_enabled", True)

    response = client.post(
        "/api/v1/system/evidence-remediations",
        headers={"Authorization": ""},
        json={
            "target_id": "unknown-event",
            "action": "persist_ready",
            "operator": "operator-1",
            "reason": "test invalid request",
        },
    )

    assert response.status_code == 401


def test_enabled_evidence_remediation_api_uses_authenticated_operator(monkeypatch) -> None:
    token = "a-strong-local-operator-token-value-123456789"
    credential = OperatorCredential(
        username="trusted-maintainer",
        role=OperatorRole.MAINTAINER,
        token_sha256=hashlib.sha256(token.encode()).hexdigest(),
    )
    monkeypatch.setattr(settings, "remediation_enabled", True)
    monkeypatch.setattr(operator_authenticator, "credentials", (credential,))

    response = client.post(
        "/api/v1/system/evidence-remediations",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "target_id": "unknown-event",
            "action": "persist_ready",
            "reason": "test invalid state",
        },
    )

    assert response.status_code == 409
    assert "requires status ready, got unknown" in response.json()["detail"]


def test_lifeguard_role_cannot_remediate_evidence(monkeypatch) -> None:
    token = "a-strong-local-lifeguard-token-value-123456789"
    credential = OperatorCredential(
        username="lifeguard-1",
        role=OperatorRole.LIFEGUARD,
        token_sha256=hashlib.sha256(token.encode()).hexdigest(),
    )
    monkeypatch.setattr(settings, "remediation_enabled", True)
    monkeypatch.setattr(operator_authenticator, "credentials", (credential,))

    response = client.post(
        "/api/v1/system/evidence-remediations",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "target_id": "unknown-event",
            "action": "persist_ready",
            "reason": "not permitted",
        },
    )

    assert response.status_code == 403


def test_lifeguard_can_view_operations_but_not_audits(monkeypatch) -> None:
    headers = operator_headers(monkeypatch, OperatorRole.LIFEGUARD)

    operations = client.get("/api/v1/video-runtimes", headers=headers)
    audits = client.get("/api/v1/evaluation-audits", headers=headers)

    assert operations.status_code == 200
    assert audits.status_code == 403


def test_service_role_can_ingest_but_not_read_incidents(monkeypatch) -> None:
    headers = operator_headers(monkeypatch, OperatorRole.SERVICE)
    payload = {
        "observations": [
            {
                "camera_id": "service-camera",
                "track_id": "service-track",
                "timestamp": 2.0,
                "pool_x": 1.0,
                "pool_y": 1.0,
            }
        ]
    }

    ingest = client.post("/api/v1/world-model/frames", headers=headers, json=payload)
    incidents = client.get("/api/v1/events", headers=headers)

    assert ingest.status_code == 200
    assert incidents.status_code == 403


def test_evidence_remediation_audit_api_requires_maintenance_role(monkeypatch) -> None:
    unauthenticated = client.get(
        "/api/v1/system/evidence-remediations", headers={"Authorization": ""}
    )
    response = client.get(
        "/api/v1/system/evidence-remediations",
        headers=operator_headers(monkeypatch),
    )

    assert unauthenticated.status_code == 401
    assert response.status_code == 200
    assert isinstance(response.json(), list)


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

    filtered = client.get(
        "/api/v1/evaluation-audits",
        params={"camera_id": "C01", "suppression_reason": result["suppression_reason"], "limit": 1},
    )
    invalid = client.get("/api/v1/evaluation-audits", params={"limit": 0})
    assert filtered.status_code == 200
    assert len(filtered.json()) == 1
    assert invalid.status_code == 422
