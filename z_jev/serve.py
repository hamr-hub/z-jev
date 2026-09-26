"""FastAPI server exposing the Jev-compatible ``/v1/evaluate`` endpoint.

Run with ::

    uvicorn z_jev.serve:app --host 0.0.0.0 --port 8080

or via the project script ::

    z-jev-serve --checkpoint checkpoints/tiny/model.pt

The request / response payloads match the official Jev spec at
https://docs.typesafe.ai/primitives/ as closely as a local model allows.

Production hardening in this version:

* Optional Bearer-token authentication (``ZJEV_API_KEY`` env var).
* Structured JSON-lines request logging middleware (no full state text is
  logged to avoid leaking PII; the state is truncated to a configurable
  prefix length).
* Unified error envelope ``{"error": {"code","message","request_id"}}``
  for validation (422), missing-checkpoint (503), and uncaught (500)
  errors. Every response carries an ``X-Request-ID`` header that
  round-trips a client-supplied id when present.
* Configurable request size, question count, and per-question option
  count limits. Anything above the limit returns 422 with a code-tagged
  error envelope.
* Split health endpoints: ``GET /healthz`` is a cheap liveness probe
  (always 200 when the process is up); ``GET /readyz`` returns 503 until
  the checkpoint has loaded, which is what Kubernetes should consult.
* Pydantic / FastAPI OpenAPI metadata with tags + examples, while keeping
  the wire format backward-compatible with the Jev spec.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from z_jev import __version__
from z_jev.model import ZJevModel
from z_jev.protocol import DecisionsRequest, DecisionsResponse

# ---------------------------------------------------------------------------
# Config / limits (env-tunable)
# ---------------------------------------------------------------------------


def _get_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# Max bytes of the request body the server is willing to accept.
MAX_REQUEST_BYTES = _get_int_env("ZJEV_MAX_REQUEST_BYTES", 256 * 1024)
# Max number of questions per request.
MAX_QUESTIONS = _get_int_env("ZJEV_MAX_QUESTIONS", 32)
# Max number of options per Choice question.
MAX_OPTIONS_PER_CHOICE = _get_int_env("ZJEV_MAX_OPTIONS_PER_CHOICE", 16)
# Max number of levels per Score question.
MAX_LEVELS_PER_SCORE = _get_int_env("ZJEV_MAX_LEVELS_PER_SCORE", 8)
# Max state text length.
MAX_STATE_CHARS = _get_int_env("ZJEV_MAX_STATE_CHARS", 8192)
# Length of the state prefix that gets logged.
LOG_STATE_CHARS = _get_int_env("ZJEV_LOG_STATE_CHARS", 40)
# When set, requests must carry ``Authorization: Bearer <key>``.
API_KEY: str | None = os.environ.get("ZJEV_API_KEY") or None

logger = logging.getLogger("z_jev.serve")


# ---------------------------------------------------------------------------
# Pydantic schemas (kept intentionally flat so curl-friendly JSON works)
# ---------------------------------------------------------------------------


class StateModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def coerce(cls, value: Any) -> StateModel:
        """Accept either a plain string or a ``{text, metadata}`` object."""
        if isinstance(value, str):
            return cls(text=value)
        if isinstance(value, dict):
            return cls(**value)
        raise ValueError(f"state must be string or object, got {type(value).__name__}")


class ChoiceQ(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str = "choice"
    instructions: str = ""
    criteria: dict[str, str] = Field(default_factory=dict)


class ScoreQ(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str = "score"
    instructions: str = ""
    criteria: list[str] = Field(default_factory=list)
    legend: list[float] | None = None


class NoulQ(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str = "noul"
    instructions: str = ""
    criteria: dict[str, str] | None = None


QuestionUnion = ChoiceQ | ScoreQ | NoulQ


class EvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    state: Any
    questions: dict[str, dict[str, Any]]

    @field_validator("state", mode="before")
    @classmethod
    def _coerce_state(cls, v: Any) -> Any:
        return StateModel.coerce(v).model_dump()


class EvaluateResponse(BaseModel):
    answers: dict[str, Any]


# ---------------------------------------------------------------------------
# Error envelope helpers
# ---------------------------------------------------------------------------


def _error_envelope(
    code: str,
    message: str,
    request_id: str,
    http_status: int,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    payload = {"error": {"code": code, "message": message, "request_id": request_id}}
    return JSONResponse(status_code=http_status, content=payload, headers=headers or {})


# ---------------------------------------------------------------------------
# Middleware: structured JSON-lines logging + request id
# ---------------------------------------------------------------------------


async def _request_logging_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Log every request as a single JSON line on stdout.

    State text is **not** logged in full -- only a truncated prefix
    (length controlled by ``ZJEV_LOG_STATE_CHARS``). The log is parseable
    by ``jq`` / fluent-bit out of the box.
    """
    # Pick up a client-supplied request id when present so downstream
    # systems can correlate.
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    request.state.request_id = request_id
    t0 = time.time()
    response: Response | None = None
    status_code = 500
    n_questions = 0
    state_preview = ""
    try:
        # Body size limit -- bail early if the client overshoots.
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > MAX_REQUEST_BYTES:
                    response = _error_envelope(
                        "request_too_large",
                        f"Content-Length {cl} exceeds ZJEV_MAX_REQUEST_BYTES={MAX_REQUEST_BYTES}",
                        request_id,
                        status.HTTP_422_UNPROCESSABLE_ENTITY,
                        {"X-Request-ID": request_id},
                    )
                    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
                    return response
            except ValueError:
                pass

        response = await call_next(request)
        status_code = response.status_code
        # Try to extract a question count + state preview for the log line.
        # We only inspect this on POST /v1/evaluate, otherwise skip.
        if request.method == "POST" and request.url.path.endswith("/v1/evaluate"):
            try:
                # Re-parse body from the receive channel -- but doing that
                # consumes the stream. Instead, rely on FastAPI having
                # already parsed it; we pull it from request.state.
                body: dict[str, Any] | None = getattr(request.state, "parsed_body", None)
                if body is not None:
                    qmap = body.get("questions") or {}
                    n_questions = len(qmap)
                    state = body.get("state") or ""
                    if isinstance(state, dict):
                        state = state.get("text", "")
                    state_preview = str(state)[:LOG_STATE_CHARS]
            except Exception:  # noqa: BLE001 -- never let logging break a request
                pass
        return response
    finally:
        elapsed_ms = (time.time() - t0) * 1000.0
        try:
            sys.stdout.write(
                json.dumps(
                    {
                        "ts": time.time(),
                        "logger": "z_jev.serve",
                        "level": "INFO",
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status": status_code,
                        "elapsed_ms": round(elapsed_ms, 2),
                        "questions": n_questions,
                        "state_preview": state_preview,
                        "client": request.client.host if request.client else None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            sys.stdout.flush()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


async def _enforce_api_key(request: Request) -> None:
    """Reject requests when ``ZJEV_API_KEY`` is set but the header is bad.

    Health endpoints (``/healthz`` / ``/readyz``) are always exempt so a
    probe loop never deadlocks on auth.
    """
    if API_KEY is None:
        return
    auth = request.headers.get("authorization") or ""
    expected = f"Bearer {API_KEY}"
    if auth != expected:
        rid = getattr(request.state, "request_id", uuid.uuid4().hex)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "unauthorized", "message": "Invalid or missing API key",
                    "request_id": rid},
            headers={"WWW-Authenticate": "Bearer", "X-Request-ID": rid},
        )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    checkpoint_path: str | None = None,
    *,
    api_key: str | None = None,
) -> FastAPI:
    """Create the FastAPI app, loading ``checkpoint_path`` on startup.

    When ``api_key`` is provided, it overrides the ``ZJEV_API_KEY`` env
    var. Passing ``api_key=""`` disables auth entirely (useful for tests
    that need to exercise both modes).
    """
    effective_api_key: str | None
    if api_key is None:
        effective_api_key = API_KEY
    elif api_key == "":
        effective_api_key = None
    else:
        effective_api_key = api_key

    state_holder: dict[str, Any] = {
        "model": None,
        "checkpoint": None,
        "load_error": None,
        "loaded_at": None,
    }

    @asynccontextmanager
    async def _lifespan(_: FastAPI):
        ckpt = checkpoint_path or os.environ.get("ZJEV_CHECKPOINT")
        if ckpt and os.path.exists(ckpt):
            try:
                state_holder["model"] = ZJevModel.load_checkpoint(ckpt, hf_cfg_only=True)
                state_holder["checkpoint"] = ckpt
                state_holder["loaded_at"] = time.time()
            except Exception as exc:  # noqa: BLE001
                state_holder["load_error"] = repr(exc)
                logger.exception("Failed to load checkpoint %s", ckpt)
        try:
            yield
        finally:
            state_holder["model"] = None

    app = FastAPI(
        title="Z-Jev",
        version=__version__,
        description=(
            "Non-autoregressive decision heads on top of GLM-5 (or a tiny CPU "
            "replica). The `/v1/evaluate` endpoint implements the Jev "
            "decision primitives (`Choice`, `Score`, `Noul`) and is wire-"
            "compatible with the upstream Jev spec at "
            "https://docs.typesafe.ai/primitives/."
        ),
        openapi_tags=[
            {"name": "decisions", "description": "Jev primitive evaluation."},
            {"name": "health", "description": "Liveness and readiness probes."},
        ],
        lifespan=_lifespan,
    )
    app.middleware("http")(_request_logging_middleware)

    # -----------------------------------------------------------------------
    # Health endpoints
    # -----------------------------------------------------------------------

    @app.get(
        "/healthz",
        tags=["health"],
        summary="Liveness probe (process is up).",
        responses={
            200: {
                "description": "Process is alive.",
                "content": {"application/json": {"example": {"status": "ok"}}},
            }
        },
    )
    async def healthz() -> dict[str, Any]:
        """Always 200 when the Python process is up.

        Use this for liveness probes (restart the container on failure).
        Use ``/readyz`` for readiness (route traffic only after the
        checkpoint has loaded). The legacy ``checkpoint_loaded`` field is
        still present so older clients keep working.
        """
        return {
            "status": "ok",
            "version": __version__,
            "checkpoint_loaded": state_holder["model"] is not None,
        }

    @app.get(
        "/readyz",
        tags=["health"],
        summary="Readiness probe (checkpoint loaded).",
        responses={
            200: {"description": "Ready to serve traffic."},
            503: {"description": "Checkpoint not yet loaded."},
        },
    )
    async def readyz(request: Request) -> Response:
        rid = getattr(request.state, "request_id", uuid.uuid4().hex)
        body = {
            "status": "ready" if state_holder["model"] is not None else "loading",
            "checkpoint": state_holder["checkpoint"],
            "loaded_at": state_holder["loaded_at"],
            "version": __version__,
        }
        if state_holder["model"] is None:
            body["load_error"] = state_holder["load_error"]
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=body,
                headers={"X-Request-ID": rid},
            )
        return JSONResponse(content=body, headers={"X-Request-ID": rid})

    # -----------------------------------------------------------------------
    # Body-parsing dependency (used both for limit checks + request logging)
    # -----------------------------------------------------------------------

    async def _capture_body(request: Request) -> dict[str, Any]:
        body_bytes = await request.body()
        if len(body_bytes) > MAX_REQUEST_BYTES:
            rid = getattr(request.state, "request_id", uuid.uuid4().hex)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "request_too_large",
                    "message": (
                        f"Request body {len(body_bytes)} bytes exceeds "
                        f"ZJEV_MAX_REQUEST_BYTES={MAX_REQUEST_BYTES}"
                    ),
                    "request_id": rid,
                },
                headers={"X-Request-ID": rid},
            )
        try:
            parsed = json.loads(body_bytes or b"{}")
        except json.JSONDecodeError as exc:
            rid = getattr(request.state, "request_id", uuid.uuid4().hex)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "bad_json",
                    "message": str(exc),
                    "request_id": rid,
                },
                headers={"X-Request-ID": rid},
            ) from exc
        request.state.parsed_body = parsed
        return parsed

    async def _enforce_question_limits(
        parsed: dict[str, Any], request: Request
    ) -> JSONResponse | None:
        """Return a JSONResponse error envelope if any limit is exceeded."""
        rid = getattr(request.state, "request_id", uuid.uuid4().hex)
        qs = parsed.get("questions") or {}
        if not isinstance(qs, dict):
            qs = {f"q{i}": q for i, q in enumerate(qs)}
        if len(qs) > MAX_QUESTIONS:
            return _error_envelope(
                "too_many_questions",
                (
                    f"questions count {len(qs)} exceeds "
                    f"ZJEV_MAX_QUESTIONS={MAX_QUESTIONS}"
                ),
                rid,
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                {"X-Request-ID": rid},
            )
        for qid, q in qs.items():
            if not isinstance(q, dict):
                continue
            if q.get("type") == "choice":
                opts = q.get("criteria") or {}
                if isinstance(opts, dict) and len(opts) > MAX_OPTIONS_PER_CHOICE:
                    return _error_envelope(
                        "too_many_options",
                        (
                            f"question {qid!r} has {len(opts)} options, "
                            f"limit is {MAX_OPTIONS_PER_CHOICE}"
                        ),
                        rid,
                        status.HTTP_422_UNPROCESSABLE_ENTITY,
                        {"X-Request-ID": rid},
                    )
            elif q.get("type") == "score":
                levels = q.get("criteria") or []
                if isinstance(levels, list) and len(levels) > MAX_LEVELS_PER_SCORE:
                    return _error_envelope(
                        "too_many_levels",
                        (
                            f"question {qid!r} has {len(levels)} levels, "
                            f"limit is {MAX_LEVELS_PER_SCORE}"
                        ),
                        rid,
                        status.HTTP_422_UNPROCESSABLE_ENTITY,
                        {"X-Request-ID": rid},
                    )
        state = parsed.get("state")
        if isinstance(state, dict) and len(str(state.get("text", ""))) > MAX_STATE_CHARS:
            return _error_envelope(
                "state_too_long",
                (
                    f"state.text length exceeds "
                    f"ZJEV_MAX_STATE_CHARS={MAX_STATE_CHARS}"
                ),
                rid,
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                {"X-Request-ID": rid},
            )
        if isinstance(state, str) and len(state) > MAX_STATE_CHARS:
            return _error_envelope(
                "state_too_long",
                (
                    f"state length exceeds "
                    f"ZJEV_MAX_STATE_CHARS={MAX_STATE_CHARS}"
                ),
                rid,
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                {"X-Request-ID": rid},
            )
        return None

    # -----------------------------------------------------------------------
    # /v1/evaluate
    # -----------------------------------------------------------------------

    @app.post(
        "/v1/evaluate",
        response_model=EvaluateResponse,
        tags=["decisions"],
        summary="Evaluate one or more Jev primitives against a state.",
        description=(
            "Accepts a Jev-compatible request envelope and returns a "
            "parallel answer for every question in a single forward pass."
        ),
        responses={
            200: {
                "description": "All questions answered.",
                "content": {"application/json": {"example": {
                    "answers": {
                        "category": {
                            "choice": "spam",
                            "probabilities": {"spam": 0.83, "ham": 0.17},
                            "confidence": 0.66,
                        },
                        "risk": {
                            "score": 2.41,
                            "legend": [1.0, 2.0, 3.0],
                            "probabilities": {"1": 0.05, "2": 0.49, "3": 0.46},
                            "confidence": 0.03,
                        },
                        "is_urgent": {
                            "noul": 0.82,
                            "answer": "yes",
                            "probability": 0.82,
                            "confidence": 0.64,
                        },
                    },
                }}},
            },
            401: {"description": "Missing or invalid API key."},
            422: {"description": "Validation or limit error."},
            503: {"description": "Checkpoint not loaded."},
        },
    )
    async def evaluate(request: Request) -> JSONResponse:
        rid = getattr(request.state, "request_id", uuid.uuid4().hex)
        # Auth (skips health endpoints).
        if effective_api_key is not None:
            auth = request.headers.get("authorization") or ""
            if auth != f"Bearer {effective_api_key}":
                return _error_envelope(
                    "unauthorized",
                    "Invalid or missing API key",
                    rid,
                    status.HTTP_401_UNAUTHORIZED,
                    {"WWW-Authenticate": "Bearer", "X-Request-ID": rid},
                )
        # Parse + limit checks.
        parsed = await _capture_body(request)
        limit_resp = await _enforce_question_limits(parsed, request)
        if limit_resp is not None:
            return limit_resp
        # Empty questions is a 422, not a 500 (the model can't run a 0-Q batch).
        if not (parsed.get("questions") or {}):
            return _error_envelope(
                "empty_questions",
                "request must contain at least one question",
                rid,
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                {"X-Request-ID": rid},
            )
        # Checkpoint loaded?
        if state_holder["model"] is None:
            return _error_envelope(
                "checkpoint_not_loaded",
                "No checkpoint loaded. Start the server with --checkpoint PATH "
                "or set ZJEV_CHECKPOINT.",
                rid,
                status.HTTP_503_SERVICE_UNAVAILABLE,
                {"X-Request-ID": rid},
            )
        # Validate the structure via Pydantic so users get nice 422 errors.
        try:
            req = EvaluateRequest.model_validate(parsed)
        except RequestValidationError as exc:
            return _error_envelope(
                "validation_error",
                str(exc),
                rid,
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                {"X-Request-ID": rid},
            )
        # Run inference.
        try:
            model: ZJevModel = state_holder["model"]
            payload: dict[str, Any] = {
                "state": req.state if isinstance(req.state, dict)
                else {"text": str(req.state)},
                "questions": req.questions,
            }
            try:
                dreq = DecisionsRequest.from_dict(payload)
            except (ValueError, KeyError, TypeError) as exc:
                return _error_envelope(
                    "validation_error",
                    f"could not parse request: {exc}",
                    rid,
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                    {"X-Request-ID": rid},
                )
            dres: DecisionsResponse = model.evaluate(dreq)
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"answers": dres.to_dict()["answers"]},
                headers={"X-Request-ID": rid},
            )
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("inference failure")
            return _error_envelope(
                "inference_error",
                repr(exc),
                rid,
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                {"X-Request-ID": rid},
            )

    # -----------------------------------------------------------------------
    # Global exception handlers (catch anything that slips past the route)
    # -----------------------------------------------------------------------

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        rid = getattr(request.state, "request_id", uuid.uuid4().hex)
        return _error_envelope(
            "validation_error",
            str(exc),
            rid,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            {"X-Request-ID": rid},
        )

    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        rid = getattr(request.state, "request_id", uuid.uuid4().hex)
        logger.exception("unhandled exception")
        return _error_envelope(
            "internal_error",
            repr(exc),
            rid,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            {"X-Request-ID": rid},
        )

    return app


# Default app for ``uvicorn z_jev.serve:app``.
_DEFAULT_CKPT = os.environ.get("ZJEV_CHECKPOINT")
app = create_app(_DEFAULT_CKPT)


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - manual use
    parser = argparse.ArgumentParser(description="Serve a trained tiny Z-Jev model.")
    parser.add_argument("--checkpoint", help="Path to model.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--api-key",
        default=None,
        help="Override ZJEV_API_KEY for this process. Pass empty string to disable.",
    )
    args = parser.parse_args(argv)

    # Print startup banner so operators can confirm version + checkpoint
    # are what they expect before traffic lands.
    ckpt = args.checkpoint or os.environ.get("ZJEV_CHECKPOINT") or "<unset>"
    print(
        json.dumps(
            {
                "ts": time.time(),
                "logger": "z_jev.serve",
                "level": "INFO",
                "event": "startup",
                "version": __version__,
                "checkpoint": ckpt,
                "host": args.host,
                "port": args.port,
                "api_key_set": (args.api_key is not None)
                or bool(os.environ.get("ZJEV_API_KEY")),
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
        flush=True,
    )

    import uvicorn

    app_local = create_app(args.checkpoint, api_key=args.api_key) if (
        args.checkpoint or args.api_key
    ) else app
    uvicorn.run(app_local, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    main()
