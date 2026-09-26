"""Protocol (de)serialisation tests."""

import math

from z_jev.protocol import (
    NOUL_LOWER,
    NOUL_UPPER,
    AnswerNoul,
    DecisionsRequest,
    DecisionsResponse,
    State,
)


def test_request_roundtrip_dict_and_list_questions():
    raw = {
        "state": "this is a test message",
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "is it spam?",
                "criteria": {"spam": "yes", "ham": "no"},
            },
            "risk": {
                "type": "score",
                "instructions": "risk level",
                "criteria": ["low", "mid", "high"],
            },
            "is_urgent": {
                "type": "noul",
                "instructions": "urgent?",
            },
        },
    }
    req = DecisionsRequest.from_dict(raw)
    assert isinstance(req.state, State)
    assert req.state.text == "this is a test message"
    assert len(req.questions) == 3
    kinds = [q.type for q in req.questions]
    assert kinds == ["choice", "score", "noul"]
    # Round-trip
    rt = req.to_dict()
    assert rt["state"]["text"] == raw["state"]
    assert "category" in rt["questions"]


def test_request_accepts_list_of_questions():
    raw = {
        "state": "msg",
        "questions": [
            {"question_id": "q1", "type": "choice", "instructions": "x", "criteria": {"a": "A", "b": "B"}},
        ],
    }
    req = DecisionsRequest.from_dict(raw)
    assert req.questions[0].question_id == "q1"


def test_answer_noul_thresholds():
    a = AnswerNoul.from_noul("q", 0.9)
    assert a.answer == "yes"
    a = AnswerNoul.from_noul("q", 0.5)
    assert a.answer == "uncertain"
    a = AnswerNoul.from_noul("q", 0.1)
    assert a.answer == "no"
    # Confidence peaks at 0 and 1.
    assert math.isclose(AnswerNoul.from_noul("q", 0.0).confidence, 1.0)
    assert math.isclose(AnswerNoul.from_noul("q", 1.0).confidence, 1.0)
    assert math.isclose(AnswerNoul.from_noul("q", 0.5).confidence, 0.0)


def test_noul_thresholds_match_documented_constants():
    assert NOUL_UPPER == 2.0 / 3.0
    assert NOUL_LOWER == 1.0 / 3.0


def test_score_response_includes_legend_and_probabilities():
    raw = {
        "state": "x",
        "questions": {
            "score": {
                "type": "score",
                "instructions": "rate",
                "criteria": ["low", "high"],
                "legend": [1, 5],
            }
        },
    }
    req = DecisionsRequest.from_dict(raw)
    assert req.questions[0].legend == [1, 5]


def test_response_probabilities_sum_to_one_property():
    res = DecisionsResponse(answers={})
    assert res.probabilities_sum_to_one()
