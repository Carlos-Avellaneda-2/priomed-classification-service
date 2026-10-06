from fastapi.testclient import TestClient

from priomed_classification.api import app


def test_health_and_classify():
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        r = client.post(
            "/classify",
            json={"referral_id": "r1", "text": "Refiere pensamientos de muerte", "structured_urgency": 0},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["priority"] == "HIGH" and body["source"] == "guardrail"
        assert body["requires_human_review"] is True


def test_rejects_invalid_urgency():
    with TestClient(app) as client:
        r = client.post("/classify", json={"referral_id": "r2", "text": "tos", "structured_urgency": 5})
        assert r.status_code == 422


def test_cors_allows_local_frontend_and_rejects_other_origins():
    preflight = {"Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"}
    with TestClient(app) as client:
        allowed = client.options("/classify", headers={"Origin": "http://localhost:5173", **preflight})
        assert allowed.status_code == 200
        assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"

        denied = client.options("/classify", headers={"Origin": "http://otro-origen.example", **preflight})
        assert "access-control-allow-origin" not in denied.headers
