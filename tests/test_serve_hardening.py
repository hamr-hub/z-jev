"""Tests for the production hardening in ``z_jev.serve``.

Covers:

* optional Bearer-token authentication (off / on / 401);
* structured JSON-lines logging (no full state, truncated preview);
* unified error envelope (``{"error": {"code","message","request_id"}}``)
  with ``X-Request-ID`` round-trip;
* request-size / question-count / option-count / state-length limits;
* split health endpoints (``/healthz`` always 200, ``/readyz`` reflects
  checkpoint load).
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from z_jev import ZJevConfig, ZJevModel
from z_jev.serve import create_app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client_open(tmp_path):
    cfg = ZJevConfig()
    model = ZJevModel(cfg)
    path = tmp_path / "model.pt"
    model.save_checkpoint(str(path))
    app = create_app(str(path))
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def client_with_auth(tmp_path):
    cfg = ZJevConfig()
    model = ZJevModel(cfg)
    path = tmp_path / "model.pt"
    model.save_checkpoint(str(path))
    app = create_app(str(path), api_key="sek-1234")
    with TestClient(app) as c:
        yield c


def _payload() -> dict:
    return {
        "state": "free lunch click urgent verify password now",
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "is spam",
                "criteria": {"spam": "yes", "ham": "no"},
            },
            "risk": {
                "type": "score",
                "instructions": "risk",
                "criteria": ["low", "mid", "high"],
            },
            "is_urgent": {"type": "noul", "instructions": "urgent?"},
        },
    }


# ---------------------------------------------------------------------------
# /healthz + /readyz
# ---------------------------------------------------------------------------


def test_healthz_always_ok(client_open):
    r = client_open.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"


def test_readyz_reports_checkpoint_loaded(client_open):
    r = client_open.get("/readyz")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_readyz_503_without_checkpoint():
    app = create_app(checkpoint_path=None)
    with TestClient(app) as c:
        r = c.get("/readyz")
        assert r.status_code == 503
        assert r.json()["status"] == "loading"


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_auth_open_by_default(client_open):
    r = client_open.post("/v1/evaluate", json=_payload())
    assert r.status_code == 200


def test_auth_required_when_api_key_set(client_with_auth):
    # No header -> 401 with error envelope.
    r = client_with_auth.post("/v1/evaluate", json=_payload())
    assert r.status_code == 401
    body = r.json()
    assert "error" in body and "code" in body["error"]
    assert body["error"]["code"] == "unauthorized"
    assert "request_id" in body["error"]


def test_auth_wrong_key_returns_401(client_with_auth):
    r = client_with_auth.post(
        "/v1/evaluate",
        headers={"Authorization": "Bearer wrong-key"},
        json=_payload(),
    )
    assert r.status_code == 401


def test_auth_correct_key_succeeds(client_with_auth):
    r = client_with_auth.post(
        "/v1/evaluate",
        headers={"Authorization": "Bearer sek-1234"},
        json=_payload(),
    )
    assert r.status_code == 200


def test_healthz_bypasses_auth(client_with_auth):
    # /healthz must always be open even when an API key is required.
    r = client_with_auth.get("/healthz")
    assert r.status_code == 200
    r = client_with_auth.get("/readyz")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Request id round-trip
# ---------------------------------------------------------------------------


def test_request_id_round_trip(client_open):
    custom = "abc-123"
    r = client_open.post(
        "/v1/evaluate",
        headers={"X-Request-ID": custom},
        json=_payload(),
    )
    assert r.headers.get("X-Request-ID") == custom


def test_request_id_generated_when_absent(client_open):
    r = client_open.post("/v1/evaluate", json=_payload())
    rid = r.headers.get("X-Request-ID")
    assert rid and len(rid) >= 8


# ---------------------------------------------------------------------------
# Error envelope on bad input
# ---------------------------------------------------------------------------


def test_error_envelope_on_missing_questions(client_open):
    r = client_open.post("/v1/evaluate", json={"state": "hi", "questions": {}})
    # Empty questions -> 422 from Pydantic (or our envelope).
    assert r.status_code in (200, 422)
    if r.status_code == 422:
        body = r.json()
        assert "error" in body and "request_id" in body["error"]
        assert body["error"]["code"] in ("validation_error", "empty_questions")


def test_error_envelope_on_unknown_question_type(client_open):
    payload = _payload()
    payload["questions"]["weird"] = {"type": "weird", "instructions": "x"}
    r = client_open.post("/v1/evaluate", json=payload)
    # The protocol parser raises ValueError; the server returns 422 via envelope.
    assert r.status_code == 422
    body = r.json()
    assert body["error"]["code"] in ("validation_error", "inference_error")


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------


def test_request_body_size_limit(monkeypatch, tmp_path):
    monkeypatch.setattr("z_jev.serve.MAX_REQUEST_BYTES", 200)
    cfg = ZJevConfig()
    model = ZJevModel(cfg)
    path = tmp_path / "model.pt"
    model.save_checkpoint(str(path))
    app = create_app(str(path))
    with TestClient(app) as c:
        payload = _payload()
        big_state = "x" * 500
        payload["state"] = big_state
        r = c.post("/v1/evaluate", json=payload)
        assert r.status_code == 422
        body = r.json()
        assert body["error"]["code"] in ("request_too_large", "state_too_long")


def test_question_count_limit(monkeypatch, tmp_path):
    monkeypatch.setattr("z_jev.serve.MAX_QUESTIONS", 2)
    cfg = ZJevConfig()
    model = ZJevModel(cfg)
    path = tmp_path / "model.pt"
    model.save_checkpoint(str(path))
    app = create_app(str(path))
    with TestClient(app) as c:
        payload = _payload()
        # 3 questions -> over the limit.
        r = c.post("/v1/evaluate", json=payload)
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "too_many_questions"


def test_option_count_limit(monkeypatch, tmp_path):
    monkeypatch.setattr("z_jev.serve.MAX_OPTIONS_PER_CHOICE", 1)
    cfg = ZJevConfig()
    model = ZJevModel(cfg)
    path = tmp_path / "model.pt"
    model.save_checkpoint(str(path))
    app = create_app(str(path))
    with TestClient(app) as c:
        payload = _payload()
        r = c.post("/v1/evaluate", json=payload)
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "too_many_options"


# ---------------------------------------------------------------------------
# Logging output
# ---------------------------------------------------------------------------


def test_logging_writes_json_line_with_truncated_state(client_open, capsys):
    payload = _payload()
    # Padding pushes the secret suffix past the LOG_STATE_CHARS=40 cutoff.
    secret_suffix = "SECRET-CONTENT-DO-NOT-LEAK"
    payload["state"] = ("a" * 50) + secret_suffix
    client_open.post("/v1/evaluate", json=payload)
    captured = capsys.readouterr()
    # Stdout should contain at least one JSON-line log entry.
    lines = [ln for ln in captured.out.splitlines() if ln.startswith("{")]
    assert lines, captured.out
    parsed = [json.loads(ln) for ln in lines]
    # The state preview must be truncated -- never the full string, and
    # in particular must not contain the secret suffix that lives past
    # the 40-char LOG_STATE_CHARS cutoff.
    for entry in parsed:
        preview = entry.get("state_preview", "")
        assert len(preview) <= 40, (preview, len(preview))
        if preview:
            assert secret_suffix not in preview
