"""Confidence / softmax math tests."""

import math

import pytest

from z_jev.scorer import (
    confidence_from_probs,
    noul_confidence,
    softmax_with_temperature,
)


def test_softmax_sums_to_one():
    p = softmax_with_temperature([1.0, 2.0, 3.0, 4.0])
    assert math.isclose(sum(p), 1.0, abs_tol=1e-9)
    # Monotonic in the input.
    assert p[0] < p[1] < p[2] < p[3]


def test_softmax_temperature_sharpens_and_flattens():
    base = softmax_with_temperature([1.0, 2.0])
    sharp = softmax_with_temperature([1.0, 2.0], temperature=0.25)
    flat = softmax_with_temperature([1.0, 2.0], temperature=4.0)
    assert sharp[1] > base[1] > flat[1]
    assert sum(base) == pytest.approx(1.0)
    assert sum(sharp) == pytest.approx(1.0)
    assert sum(flat) == pytest.approx(1.0)


def test_softmax_temperature_must_be_positive():
    with pytest.raises(ValueError):
        softmax_with_temperature([0.0], temperature=0.0)


def test_confidence_zero_for_uniform():
    assert confidence_from_probs([0.25, 0.25, 0.25, 0.25]) == 0.0


def test_confidence_one_for_peaked():
    assert confidence_from_probs([1.0, 0.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_confidence_binary_equals_margin():
    # For K=2, confidence = |p1 - p2|.
    assert confidence_from_probs([0.9, 0.1]) == pytest.approx(0.8)


def test_noul_confidence_peaks_at_extremes():
    assert noul_confidence(0.5) == 0.0
    assert noul_confidence(0.0) == pytest.approx(1.0)
    assert noul_confidence(1.0) == pytest.approx(1.0)
    assert noul_confidence(0.75) == pytest.approx(0.5)
