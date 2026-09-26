"""Confidence and probability calibration utilities.

All confidence values in the protocol are defined here in one place so the
math can be inspected, tested, and re-tuned without scattering lambdas
through the head code.

**Confidence definition** (margin-calibrated top-1 probability)

Given a discrete probability distribution ``p_1, ..., p_K`` summing to 1:

    raw_top1 = max(p_i)
    uniform  = 1 / K
    confidence = max(0, (raw_top1 - uniform) / max(eps, 1 - uniform))

This goes from 0 (uniform distribution -- the model has no idea) to 1 (the
model is fully peaked on a single option). For K=2 this collapses to
``|p_yes - p_no|``, which is the natural certainty measure.

For Noul (binary yes/no), the same formula collapses to
``|p - (1-p)| = 2|p - 0.5|``, which is what
``AnswerNoul.from_noul`` uses.

**Temperature calibration** is applied at the *logits* level inside the head
(see ``z_jev.head``), so the same temperature rescales both probabilities
and confidence -- a more peaked model gets higher confidence automatically.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

_EPS = 1e-9


def softmax_with_temperature(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Numerically stable softmax with a temperature divisor.

    With ``temperature > 1`` the distribution flattens (less confident).
    With ``temperature < 1`` it sharpens (more confident).
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    t = float(temperature)
    z = [float(x) / t for x in logits]
    m = max(z)
    exps = [math.exp(x - m) for x in z]
    s = sum(exps)
    if s == 0:
        # Pathological (all -inf logits). Return uniform.
        n = len(z)
        return [1.0 / n] * n
    return [e / s for e in exps]


def confidence_from_probs(probs: Sequence[float]) -> float:
    """Margin-calibrated top-1 confidence in [0, 1].

    See module docstring for the formula.
    """
    n = len(probs)
    if n == 0:
        return 0.0
    top1 = max(probs)
    uniform = 1.0 / n
    margin = top1 - uniform
    if margin <= 0:
        return 0.0
    return margin / max(_EPS, 1.0 - uniform)


def noul_confidence(p: float) -> float:
    """Confidence for a binary yes/no probability.

    Equals ``2 * |p - 0.5|`` which is the same as ``|p - (1-p)|`` -- the
    difference between the "yes" and "no" probability masses. It ranges
    from 0 (fully uncertain at ``p == 0.5``) to 1 (fully certain at the
    extremes).
    """
    p = min(1.0, max(0.0, float(p)))
    return 2.0 * abs(p - 0.5)
