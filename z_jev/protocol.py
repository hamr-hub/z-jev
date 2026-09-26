"""Jev-compatible decision protocol.

The shapes here track the upstream Jev spec from
https://docs.typesafe.ai/primitives/ verbatim:

* **Choice** returns ``{choice, probabilities, confidence}``.
* **Score** returns ``{score, legend, probabilities, confidence}``.
* **Noul** returns ``{noul}`` (probability of "yes"). We also expose derived
  ``answer`` (string label), ``probability`` (alias of ``noul``) and
  ``confidence`` (peakness) so the model is convenient to consume from
  non-Jev code that asks for an explicit yes/no/uncertain verdict.

The request envelope is::

    {
        "state": "<text describing the situation>",
        "questions": { "<id>": { "type": ..., "instructions": ..., "criteria": ... } }
    }

Multiple questions are evaluated in **one** forward pass -- they never see
each other's answers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


@dataclass
class State:
    """The textual (or structured) situation the decision operates on."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"text": self.text}
        if self.metadata:
            out["metadata"] = self.metadata
        return out


@dataclass
class QuestionChoice:
    """Pick one option out of a set of named keys."""

    type: Literal["choice"] = "choice"
    question_id: str = ""
    instructions: str = ""
    # criteria maps option_key -> human description. Order of options in the
    # response follows the iteration order of this dict (Python 3.7+ preserves
    # insertion order).
    criteria: dict[str, str] = field(default_factory=dict)


@dataclass
class QuestionScore:
    """Rate something on an ordered scale.

    ``criteria`` is a list of level descriptions, ordered low to high. The
    ``legend`` echoed back in the response is ``[1, 2, ..., len(criteria)]``
    unless the user supplies ``legend``.
    """

    type: Literal["score"] = "score"
    question_id: str = ""
    instructions: str = ""
    criteria: list[str] = field(default_factory=list)
    legend: list[float] | None = None  # defaults to [1, 2, ..., n]


@dataclass
class QuestionNoul:
    """Yes / no / uncertain boolean judgment.

    ``criteria`` is optional; if supplied it is a dict ``{"yes": ..., "no": ...}``
    that clarifies what "yes" and "no" mean.
    """

    type: Literal["noul"] = "noul"
    question_id: str = ""
    instructions: str = ""
    criteria: dict[str, str] | None = None


# A question is one of the three primitive variants.
Question = QuestionChoice | QuestionScore | QuestionNoul


@dataclass
class DecisionsRequest:
    """A single evaluation request carrying one state and N parallel questions."""

    state: State
    questions: list[Question] = field(default_factory=list)

    # -- (de)serialisation -------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionsRequest:
        state_data = data.get("state", {})
        if isinstance(state_data, str):
            state = State(text=state_data)
        elif isinstance(state_data, dict):
            state = State(
                text=state_data.get("text", ""),
                metadata=state_data.get("metadata", {}) or {},
            )
        else:
            raise ValueError(f"Unsupported state payload: {type(state_data).__name__}")

        raw_qs = data.get("questions", {})
        if isinstance(raw_qs, dict):
            items = list(raw_qs.items())
        elif isinstance(raw_qs, list):
            items = [(q.get("question_id", f"q{i}"), q) for i, q in enumerate(raw_qs)]
        else:
            raise ValueError(f"questions must be dict or list, got {type(raw_qs)}")

        parsed: list[Question] = []
        for qid, q in items:
            qtype = q.get("type")
            if qtype == "choice":
                parsed.append(
                    QuestionChoice(
                        question_id=qid,
                        instructions=q.get("instructions", ""),
                        criteria=dict(q.get("criteria", {}) or {}),
                    )
                )
            elif qtype == "score":
                parsed.append(
                    QuestionScore(
                        question_id=qid,
                        instructions=q.get("instructions", ""),
                        criteria=list(q.get("criteria", []) or []),
                        legend=q.get("legend"),
                    )
                )
            elif qtype == "noul":
                parsed.append(
                    QuestionNoul(
                        question_id=qid,
                        instructions=q.get("instructions", ""),
                        criteria=q.get("criteria"),
                    )
                )
            else:
                raise ValueError(f"Unknown question type: {qtype!r}")
        return cls(state=state, questions=parsed)

    def to_dict(self) -> dict[str, Any]:
        qdict: dict[str, Any] = {}
        for q in self.questions:
            payload: dict[str, Any] = {"type": q.type, "instructions": q.instructions}
            if q.type == "choice":
                payload["criteria"] = q.criteria
            elif q.type == "score":
                payload["criteria"] = q.criteria
                if q.legend is not None:
                    payload["legend"] = q.legend
            elif q.type == "noul":
                if q.criteria is not None:
                    payload["criteria"] = q.criteria
            qdict[q.question_id] = payload
        return {"state": self.state.to_dict(), "questions": qdict}


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------


@dataclass
class AnswerChoice:
    """Choice primitive answer.

    * ``choice`` is the selected option key (string).
    * ``probabilities`` is a dict option_key -> probability, summing to 1.
    * ``confidence`` is the margin-calibrated certainty in [0, 1] -- see
      :mod:`z_jev.scorer` for the exact definition.
    """

    question_id: str
    choice: str
    probabilities: dict[str, float]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "choice": self.choice,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
        }


@dataclass
class AnswerScore:
    """Score primitive answer.

    * ``score`` is a probability-weighted position on the legend.
    * ``legend`` echoes the level positions (defaults to ``[1, 2, ..., n]``).
    * ``probabilities`` is a dict str(level) -> probability, summing to 1.
    * ``confidence`` is the margin-calibrated certainty in [0, 1].
    """

    question_id: str
    score: float
    legend: list[float]
    probabilities: dict[str, float]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "legend": self.legend,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
        }


# Decision thresholds for mapping a Noul probability to a discrete label.
NOUL_LOWER = 1.0 / 3.0  # below this -> "no"
NOUL_UPPER = 2.0 / 3.0  # above this -> "yes"


@dataclass
class AnswerNoul:
    """Noul primitive answer.

    Canonical field is ``noul`` (probability of "yes"). We also expose:

    * ``answer``: "yes" / "no" / "uncertain" derived from thresholds,
    * ``probability``: alias of ``noul`` for callers that prefer that name,
    * ``confidence``: peakness = ``1 - 2 * |noul - 0.5|``.
    """

    question_id: str
    noul: float
    answer: str  # "yes" / "no" / "uncertain"
    probability: float
    confidence: float

    @staticmethod
    def from_noul(question_id: str, p: float) -> AnswerNoul:
        p = float(min(1.0, max(0.0, p)))
        if p >= NOUL_UPPER:
            label = "yes"
        elif p <= NOUL_LOWER:
            label = "no"
        else:
            label = "uncertain"
        # Confidence = how decisive the probability is. Peaks at p in {0, 1},
        # drops to 0 at p == 0.5. Same shape as ``scorer.noul_confidence``.
        confidence = 2.0 * abs(p - 0.5)
        return AnswerNoul(
            question_id=question_id,
            noul=p,
            answer=label,
            probability=p,
            confidence=confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "noul": self.noul,
            "answer": self.answer,
            "probability": self.probability,
            "confidence": self.confidence,
        }


Answer = AnswerChoice | AnswerScore | AnswerNoul


@dataclass
class DecisionsResponse:
    """The full response: an ``answers`` dict keyed by question id."""

    answers: dict[str, Answer] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"answers": {qid: a.to_dict() for qid, a in self.answers.items()}}

    def probabilities_sum_to_one(self, tol: float = 1e-4) -> bool:
        """Sanity check used by tests."""
        for ans in self.answers.values():
            if isinstance(ans, (AnswerChoice, AnswerScore)):
                s = sum(ans.probabilities.values())
                if not math.isfinite(s) or abs(s - 1.0) > tol:
                    return False
            elif isinstance(ans, AnswerNoul):
                if not (0.0 - tol <= ans.noul <= 1.0 + tol):
                    return False
        return True
