"""FastAPI end-to-end test using TestClient (no network, no real server)."""


import pytest
from fastapi.testclient import TestClient

from z_jev import ZJevConfig, ZJevModel
from z_jev.serve import create_app


@pytest.fixture()
def client_with_ckpt(tmp_path):
    cfg = ZJevConfig()
    model = ZJevModel(cfg)
    path = tmp_path / "model.pt"
    model.save_checkpoint(str(path))
    app = create_app(str(path))
    with TestClient(app) as client:
        yield client


def test_healthz(client_with_ckpt):
    r = client_with_ckpt.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["checkpoint_loaded"]


def test_evaluate_endpoint_returns_jev_shape(client_with_ckpt):
    payload = {
        "state": "free lunch click urgent verify password",
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "is spam",
                "criteria": {"spam": "yes", "ham": "no"},
            },
            "risk": {
                "type": "score",
                "instructions": "risk level",
                "criteria": ["low", "med", "high"],
            },
            "is_urgent": {"type": "noul", "instructions": "urgent?"},
        },
    }
    r = client_with_ckpt.post("/v1/evaluate", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "answers" in body
    ans = body["answers"]
    assert set(ans.keys()) == {"category", "risk", "is_urgent"}
    assert ans["category"]["choice"] in {"spam", "ham"}
    s = sum(ans["category"]["probabilities"].values())
    assert abs(s - 1.0) < 1e-3
    assert ans["risk"]["legend"] == [1.0, 2.0, 3.0]
    assert "noul" in ans["is_urgent"] and "answer" in ans["is_urgent"]


def test_evaluate_returns_503_when_no_checkpoint():
    app = create_app(checkpoint_path=None)
    with TestClient(app) as client:
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["checkpoint_loaded"] is False
        # /readyz should be 503 until a checkpoint loads.
        r2 = client.get("/readyz")
        assert r2.status_code == 503
        assert r2.json()["status"] == "loading"
        r = client.post(
            "/v1/evaluate",
            json={
                "state": {"text": "x"},
                "questions": {
                    "is_urgent": {"type": "noul", "instructions": "urgent?"}
                },
            },
        )
        assert r.status_code == 503
        # Error envelope shape.
        body = r.json()
        assert body["error"]["code"] == "checkpoint_not_loaded"
        assert "request_id" in body["error"]
