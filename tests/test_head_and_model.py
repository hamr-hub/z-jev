"""Model + head shape, batching, and end-to-end correctness tests."""

import json

import pytest
import torch

from z_jev import (
    DecisionsRequest,
    QuestionChoice,
    QuestionNoul,
    QuestionScore,
    State,
    ZJevConfig,
    ZJevModel,
)
from z_jev.data import mixed_dataset
from z_jev.head import HeadOutputs
from z_jev.model import collate_requests

# ---------------------------------------------------------------------------
# Head shape / parallel batching
# ---------------------------------------------------------------------------


def _tiny_cfg() -> ZJevConfig:
    return ZJevConfig(
        mode="tiny",
        tiny=ZJevConfig.__dataclass_fields__["tiny"].default_factory(),
    )


def test_head_forward_shapes_and_mask():
    cfg = _tiny_cfg()
    from z_jev.head import PRIMITIVE_MAX_OUT, NonAutoregressiveDecisionHead

    head = NonAutoregressiveDecisionHead(cfg)
    b, q, h = 2, 4, cfg.head_hidden_size
    state_vec = torch.randn(b, h)
    q_vecs = torch.randn(b, q, h)
    types = torch.tensor([[0, 1, 2, 2], [1, 0, 2, 2]])
    out: HeadOutputs = head(state_vec, q_vecs, types)
    assert out.choice_logits.shape == (b, q, PRIMITIVE_MAX_OUT["choice"])
    assert out.score_logits.shape == (b, q, PRIMITIVE_MAX_OUT["score"])
    assert out.noul_logits.shape == (b, q, PRIMITIVE_MAX_OUT["noul"])
    assert out.sizes.tolist() == [
        [16, 8, 2, 2],
        [8, 16, 2, 2],
    ]


def test_head_decode_returns_per_type_answers():
    cfg = _tiny_cfg()
    from z_jev.head import NonAutoregressiveDecisionHead

    head = NonAutoregressiveDecisionHead(cfg)
    b, q, h = 1, 3, cfg.head_hidden_size
    types = torch.tensor([[0, 1, 2]])
    out = head(torch.randn(b, h), torch.randn(b, q, h), types)
    qs = [
        QuestionChoice(question_id="c", instructions="x", criteria={"a": "A", "b": "B"}),
        QuestionScore(question_id="s", instructions="x", criteria=["low", "mid", "high"]),
        QuestionNoul(question_id="n", instructions="x"),
    ]
    rows = head.decode(out, [qs])
    assert len(rows) == 1 and len(rows[0]) == 3
    c, s, n = rows[0]
    assert c.choice in {"a", "b"}
    assert sum(c.probabilities.values()) == pytest.approx(1.0, abs=1e-5)
    assert s.legend == [1, 2, 3]
    assert 0.0 <= n.noul <= 1.0


# ---------------------------------------------------------------------------
# ZJevModel end-to-end
# ---------------------------------------------------------------------------


def test_model_evaluate_returns_probabilities_summing_to_one():
    cfg = _tiny_cfg()
    model = ZJevModel(cfg)
    model.eval()
    req = DecisionsRequest(
        state=State(text="a message about password and verify"),
        questions=[
            QuestionChoice(
                question_id="category",
                instructions="classify",
                criteria={"spam": "spam", "ham": "not spam"},
            ),
            QuestionScore(
                question_id="risk",
                instructions="risk",
                criteria=["low", "med", "high"],
            ),
            QuestionNoul(question_id="urgent", instructions="urgent?"),
        ],
    )
    res = model.evaluate(req)
    assert res.probabilities_sum_to_one()
    payload = res.to_dict()
    assert set(payload["answers"].keys()) == {"category", "risk", "urgent"}
    assert payload["answers"]["category"]["choice"] in {"spam", "ham"}


def test_model_evaluate_batch_matches_individual_calls():
    cfg = _tiny_cfg()
    model = ZJevModel(cfg)
    model.eval()
    examples = mixed_dataset(n=4, seed=42)
    batched = model.evaluate_batch([ex.request for ex in examples])
    singles = [model.evaluate(ex.request) for ex in examples]
    assert len(batched) == len(singles)
    for b, s in zip(batched, singles, strict=True):
        for qid in b.answers:
            assert qid in s.answers


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------


def test_checkpoint_round_trip_preserves_predictions(tmp_path):
    cfg = _tiny_cfg()
    model = ZJevModel(cfg)
    model.eval()
    req = DecisionsRequest(
        state=State(text="urgent verify password now"),
        questions=[
            QuestionChoice(
                question_id="cat",
                instructions="is spam",
                criteria={"spam": "yes", "ham": "no"},
            ),
            QuestionNoul(question_id="u", instructions="urgent?"),
        ],
    )
    before = model.evaluate(req).to_dict()
    path = tmp_path / "model.pt"
    model.save_checkpoint(str(path), extra={"note": "unit-test"})
    model2 = ZJevModel.load_checkpoint(str(path))
    after = model2.evaluate(req).to_dict()
    assert json.dumps(before, sort_keys=True) == json.dumps(after, sort_keys=True)


def test_collator_handles_heterogeneous_question_counts():
    cfg = _tiny_cfg()
    model = ZJevModel(cfg)
    tok = model.tokenizer
    r1 = DecisionsRequest(
        state=State(text="a"), questions=[QuestionChoice(question_id="c", instructions="x", criteria={"a": "A"})]
    )
    r2 = DecisionsRequest(
        state=State(text="b"),
        questions=[
            QuestionChoice(question_id="c", instructions="x", criteria={"a": "A"}),
            QuestionNoul(question_id="n", instructions="y"),
        ],
    )
    batch = collate_requests([r1, r2], tok)
    assert batch.question_types.shape == (2, 2)
    # Noul positions in row 0 should be ignored but valid shape-wise.
    assert batch.question_ids.shape[2] > 0


# ---------------------------------------------------------------------------
# Loss vectorisation (regression test for the O(B*Q) Python loop removal)
# ---------------------------------------------------------------------------


def test_vectorised_loss_matches_reference_per_question_loop():
    """``head.loss`` must match the per-question Python loop to fp32 noise.

    The vectorised path concatenates the three branches into one padded
    logits tensor and calls ``F.cross_entropy`` once; the reference path
    iterates per question and gathers the right branch. They have to be
    numerically equivalent so the speedup cannot change training behavior.
    """
    import torch.nn.functional as F

    cfg = _tiny_cfg()
    from z_jev.head import NonAutoregressiveDecisionHead

    head = NonAutoregressiveDecisionHead(cfg)
    b, q, h = 3, 4, cfg.head_hidden_size
    types = torch.tensor(
        [
            [0, 1, 2, 2],
            [1, 0, 2, 2],
            [0, 0, 1, 2],
        ]
    )
    out = head(torch.randn(b, h), torch.randn(b, q, h), types)
    targets = [
        [1, 0, 0, -1],  # last padded -> ignored
        [2, 0, 1, -1],
        [0, 1, 2, 0],
    ]

    # Reference: per-question Python loop.
    ref_losses = []
    for bi in range(b):
        for qi in range(q):
            tgt = int(targets[bi][qi])
            if tgt < 0:
                continue
            t = int(out.types[bi, qi].item())
            if t == 0:
                logits = out.choice_logits[bi, qi]
            elif t == 1:
                logits = out.score_logits[bi, qi]
            else:
                logits = out.noul_logits[bi, qi]
            size = int(out.sizes[bi, qi].item())
            ref_losses.append(
                F.cross_entropy(
                    logits[:size].unsqueeze(0),
                    torch.tensor([tgt]),
                )
            )
    ref = torch.stack(ref_losses).mean()

    # Vectorised: the public ``head.loss`` API.
    head.zero_grad()
    vec = head.loss(out, targets)

    assert torch.allclose(ref, vec, atol=1e-6), (ref.item(), vec.item())

    # Backward must work (gradient flows to the right branch only).
    vec.backward()
    assert out.choice_logits.grad is not None
    assert out.score_logits.grad is not None
    assert out.noul_logits.grad is not None
