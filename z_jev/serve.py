"""FastAPI server exposing the Jev-compatible ``/v1/evaluate`` endpoint.

Run with::

    uvicorn z_jev.serve:app --host 0.0.0.0 --port 8080

or via the project script::

    z-jev-serve --checkpoint checkpoints/tiny/model.pt

The request / response payloads match the official Jev spec at
https://docs.typesafe.ai/primitives/ as closely as a local model allows.
"""

from __future__ import annotations

import argparse
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from z_jev.model import ZJevModel
from z_jev.protocol import DecisionsRequest, DecisionsResponse

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
# App factory
# ---------------------------------------------------------------------------


def create_app(checkpoint_path: str | None = None) -> FastAPI:
    """Create the FastAPI app, loading ``checkpoint_path`` on startup."""
    model_holder: dict[str, Any] = {"model": None, "checkpoint": None}

    @asynccontextmanager
    async def _lifespan(_: FastAPI):
        ckpt = checkpoint_path or os.environ.get("ZJEV_CHECKPOINT")
        if ckpt and os.path.exists(ckpt):
            model_holder["model"] = ZJevModel.load_checkpoint(ckpt, hf_cfg_only=True)
            model_holder["checkpoint"] = ckpt
        try:
            yield
        finally:
            model_holder["model"] = None

    app = FastAPI(
        title="Z-Jev",
        version="0.1.0",
        description="Non-autoregressive decision heads on top of GLM-5 (or a tiny replica).",
        lifespan=_lifespan,
    )

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "checkpoint_loaded": model_holder["model"] is not None,
            "checkpoint": model_holder["checkpoint"],
        }

    @app.post("/v1/evaluate", response_model=EvaluateResponse)
    def evaluate(req: EvaluateRequest) -> EvaluateResponse:
        if model_holder["model"] is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "No checkpoint loaded. Start the server with --checkpoint "
                    "PATH or set the ZJEV_CHECKPOINT env var."
                ),
            )
        model: ZJevModel = model_holder["model"]
        payload: dict[str, Any] = {
            "state": req.state if isinstance(req.state, dict) else {"text": str(req.state)},
            "questions": req.questions,
        }
        dreq = DecisionsRequest.from_dict(payload)
        dres: DecisionsResponse = model.evaluate(dreq)
        return EvaluateResponse(answers=dres.to_dict()["answers"])

    return app


# When invoked via ``uvicorn z_jev.serve:app`` we need a default app.
_DEFAULT_CKPT = os.environ.get("ZJEV_CHECKPOINT")
app = create_app(_DEFAULT_CKPT)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - manual use
    parser = argparse.ArgumentParser(description="Serve a trained tiny Z-Jev model.")
    parser.add_argument("--checkpoint", help="Path to model.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)
    import uvicorn

    app_local = create_app(args.checkpoint) if args.checkpoint else app
    uvicorn.run(app_local, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    main()
