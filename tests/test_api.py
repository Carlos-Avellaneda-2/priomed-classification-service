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
