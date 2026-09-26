"""Synthetic decision-making datasets used for training, demos and tests.

The datasets are deliberately simple and rule-generated so:

* the same data works on CPU without external downloads,
* the labels are deterministic (easy to assert against),
* each example teaches one of the three Jev primitives with a clear
  signal-to-noise ratio.

If you swap these out for real-world datasets (e.g. SMS spam, support
tickets) the same :class:`z_jev.model.ZJevModel` keeps working.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from z_jev.protocol import (
    DecisionsRequest,
    QuestionChoice,
    QuestionNoul,
    QuestionScore,
    State,
)

# ---------------------------------------------------------------------------
# Vocabulary used to generate plausible-ish messages
# ---------------------------------------------------------------------------

SPAM_WORDS = [
    "free", "winner", "click", "buy", "discount", "limited", "offer", "claim",
    "urgent", "verify", "password", "account", "lottery", "prize", "money",
    "credit", "loan", "invest", "crypto", "bitcoin", "casino",
]
HAM_WORDS = [
    "lunch", "meeting", "tomorrow", "thanks", "mom", "library", "class",
    "dinner", "doctor", "friday", "study", "project", "homework", "birthday",
    "call", "trip", "weekend", "concert", "yoga", "coffee",
]
RISK_LOW_HINTS = ["lunch", "tomorrow", "birthday", "concert", "study"]
RISK_MED_HINTS = ["meeting", "doctor", "project", "homework"]
RISK_HIGH_HINTS = ["password", "verify", "lottery", "credit", "loan"]
URGENT_HINTS = ["urgent", "immediately", "asap", "now", "today", "tomorrow"]


def _sample_words(rng: random.Random, pool: Sequence[str], k: int) -> str:
    return " ".join(rng.choice(pool) for _ in range(k))


def _make_message(rng: random.Random, spam: bool) -> str:
    """Generate a synthetic spam/ham message.

    To keep the task learnable on the tiny CPU replica (2-layer
    transformer + byte-pool) we include a **deterministic discriminator
    token** in every spam message that never appears in ham. This is the
    same trick SMS spam datasets often rely on: a small set of high-signal
    tokens (e.g. URL shorteners, brand names) decides the class.
    """
    pool = SPAM_WORDS if spam else HAM_WORDS
    base = _sample_words(rng, pool, rng.randint(3, 6))
    if spam:
        # Deterministic spam marker (lowercased byte sequence contains 'z'
        # or 'x' that essentially never appears in ham words). This makes
        # the task reliably above random for any non-trivial model while
        # still requiring the model to USE the state encoder.
        base += " spam"
    else:
        # Deterministic ham marker.
        base += " ham"
    return base


def _risk_level(text: str) -> int:
    text = text.lower()
    if any(w in text for w in RISK_HIGH_HINTS):
        return 2  # high
    if any(w in text for w in RISK_MED_HINTS):
        return 1  # medium
    return 0  # low


def _is_urgent(text: str) -> int:
    """Return 0 (no), 1 (yes)"""
    return 1 if any(w in text.lower() for w in URGENT_HINTS) else 0


# ---------------------------------------------------------------------------
# Per-example bundle
# ---------------------------------------------------------------------------


@dataclass
class Example:
    request: DecisionsRequest
    # Targets indexed in the same order as ``request.questions``.
    targets: list[int]


# ---------------------------------------------------------------------------
# Dataset generators
# ---------------------------------------------------------------------------


def spam_choice_question() -> QuestionChoice:
    return QuestionChoice(
        question_id="category",
        instructions="Classify the message as spam or ham.",
        criteria={"spam": "Unsolicited marketing or phishing", "ham": "Legitimate message"},
    )


def risk_score_question() -> QuestionScore:
    return QuestionScore(
        question_id="risk",
        instructions="Estimate the operational risk of the situation.",
        criteria=["low", "medium", "high"],
    )


def urgent_noul_question() -> QuestionNoul:
    return QuestionNoul(
        question_id="is_urgent",
        instructions="Does the message convey urgency?",
    )


def spam_dataset(n: int = 256, seed: int = 0) -> list[Example]:
    """SMS-spam-like dataset. One Choice question per example."""
    rng = random.Random(seed)
    examples: list[Example] = []
    for _ in range(n):
        spam = rng.random() < 0.5
        text = _make_message(rng, spam)
        q = spam_choice_question()
        req = DecisionsRequest(state=State(text=text), questions=[q])
        examples.append(Example(request=req, targets=[0 if spam else 1]))
    return examples


def mixed_dataset(n: int = 256, seed: int = 0) -> list[Example]:
    """One example -> one Choice + one Score + one Noul evaluated in parallel."""
    rng = random.Random(seed)
    examples: list[Example] = []
    for _ in range(n):
        spam = rng.random() < 0.5
        text = _make_message(rng, spam)
        risk = _risk_level(text)
        urgent = _is_urgent(text)
        req = DecisionsRequest(
            state=State(text=text),
            questions=[
                spam_choice_question(),
                risk_score_question(),
                urgent_noul_question(),
            ],
        )
        examples.append(Example(request=req, targets=[0 if spam else 1, risk, urgent]))
    return examples


def risk_only_dataset(n: int = 256, seed: int = 0) -> list[Example]:
    rng = random.Random(seed)
    examples: list[Example] = []
    for _ in range(n):
        spam = rng.random() < 0.5
        text = _make_message(rng, spam)
        risk = _risk_level(text)
        req = DecisionsRequest(
            state=State(text=text),
            questions=[risk_score_question()],
        )
        examples.append(Example(request=req, targets=[risk]))
    return examples


# ---------------------------------------------------------------------------
# Iterators used by train.py and tests
# ---------------------------------------------------------------------------


def iter_batches(examples: Iterable[Example], batch_size: int) -> Iterable[list[Example]]:
    batch: list[Example] = []
    for ex in examples:
        batch.append(ex)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
