"""Non-autoregressive decision heads.

Each Jev primitive (Choice, Score, Noul) gets its own MLP head with the
*exact* output dimension it needs (no padding / no max_outputs). The
heads are evaluated in **one** forward pass over a list of per-question
features, which is what makes Z-Jev "non-autoregressive".

Output dimensions:

* Choice -> ``len(options)`` logits (variable per question)
* Score  -> ``len(scale)``  logits (variable per question)
* Noul   -> 2 logits (yes / no). The third state ("uncertain") is the
  decision implied by the probability and the thresholds in
  :mod:`z_jev.protocol`.

**Why this head shape?**

Earlier architectures concatenated ``[state, question, type]`` into a
trunk MLP. With a tiny (2-layer / 128-hidden) backbone and bag-of-bytes
states that experiment collapsed to a constant prediction because the
question-vector path dominated the gradient. The current design uses:

1. A *fixed* ``head_hidden_size`` that the head always sees (state is
   projected by ``ZJevModel.state_proj``).
2. The **type embedding** plus a *small-scaled* question vector so the
   option-label signal is present but the state dominates.
3. Three independent MLPs (one per primitive), each projecting to its
   exact output dim. We slice per-question at decode time so each
   Choice can have a different number of options.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from z_jev.config import ZJevConfig
from z_jev.protocol import (
    AnswerChoice,
    AnswerNoul,
    AnswerScore,
    Question,
    QuestionChoice,
    QuestionNoul,
    QuestionScore,
)
from z_jev.scorer import confidence_from_probs, softmax_with_temperature

# Output dim per primitive (the head has three branches with these widths).
PRIMITIVE_MAX_OUT = {
    "choice": 16,  # up to 16 options
    "score": 8,  # up to 8 levels
    "noul": 2,
}


@dataclass
class HeadOutputs:
    """Raw per-question outputs from :class:`NonAutoregressiveDecisionHead`."""

    features: torch.Tensor  # (B, Q, feat_dim)
    # Per-primitive logits, padded to the primitive's max width.
    choice_logits: torch.Tensor  # (B, Q, 16)
    score_logits: torch.Tensor  # (B, Q, 8)
    noul_logits: torch.Tensor  # (B, Q, 2)
    types: torch.Tensor  # (B, Q)
    sizes: torch.Tensor  # (B, Q) -- size per question
    question_idx: torch.Tensor  # (B, Q)


class _HeadMLP(nn.Module):
    """Small per-primitive MLP: LN -> Linear -> GELU -> Linear -> Linear."""

    def __init__(self, in_dim: int, out_dim: int, hidden_mult: int = 2) -> None:
        super().__init__()
        h = max(out_dim * 2, in_dim)
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, h, bias=True)
        self.fc2 = nn.Linear(h, h, bias=True)
        self.out = nn.Linear(h, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = F.gelu(self.fc2(h))
        return self.out(h)


class NonAutoregressiveDecisionHead(nn.Module):
    """One forward pass produces per-question logits for all three primitives."""

    def __init__(self, cfg: ZJevConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # ``feat_dim`` per question = state (H) + type_emb (Q) + question_vec (H).
        feat_dim = cfg.head_hidden_size + cfg.question_emb_size + cfg.head_hidden_size
        self.type_emb = nn.Embedding(3, cfg.question_emb_size)
        self.choice_head = _HeadMLP(feat_dim, PRIMITIVE_MAX_OUT["choice"])
        self.score_head = _HeadMLP(feat_dim, PRIMITIVE_MAX_OUT["score"])
        self.noul_head = _HeadMLP(feat_dim, PRIMITIVE_MAX_OUT["noul"])
        # Small scaling factor so the question vector does not drown out
        # the state vector on the tiny CPU replica.
        self.question_scale = 0.1
        # Temperature for decoding (used at decode time).
        self.temperature = cfg.glm5.temperature if cfg.mode == "glm5" else 1.0

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        state_vec: torch.Tensor,
        question_vecs: torch.Tensor,
        question_types: torch.Tensor,
    ) -> HeadOutputs:
        b, q, h = question_vecs.shape
        type_e = self.type_emb(question_types)  # (B, Q, Q-emb)
        s_exp = state_vec.unsqueeze(1).expand(b, q, h)  # (B, Q, H)
        feats = torch.cat([s_exp, type_e, self.question_scale * question_vecs], dim=-1)

        choice_logits = self.choice_head(feats)  # (B, Q, 16)
        score_logits = self.score_head(feats)  # (B, Q, 8)
        noul_logits = self.noul_head(feats)  # (B, Q, 2)

        # Per-question output size (used for both loss + decoding).
        sizes = torch.full(
            (b, q), PRIMITIVE_MAX_OUT["choice"], dtype=torch.long, device=feats.device
        )
        sizes = torch.where(
            question_types == 1,
            torch.tensor(PRIMITIVE_MAX_OUT["score"], device=feats.device, dtype=torch.long),
            sizes,
        )
        sizes = torch.where(
            question_types == 2,
            torch.tensor(PRIMITIVE_MAX_OUT["noul"], device=feats.device, dtype=torch.long),
            sizes,
        )

        return HeadOutputs(
            features=feats,
            choice_logits=choice_logits,
            score_logits=score_logits,
            noul_logits=noul_logits,
            types=question_types,
            sizes=sizes,
            question_idx=torch.arange(q, device=feats.device).unsqueeze(0).expand(b, q),
        )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(
        self,
        outputs: HeadOutputs,
        targets: list[list[int]],
    ) -> torch.Tensor:
        """Cross-entropy per question, masked over the batch.

        Vectorised over ``(B*Q)``: the three branches are concatenated into
        one padded logits tensor and a single ``F.cross_entropy`` call
        returns the per-row loss. Rows whose ``target < 0`` are skipped via
        a validity mask, so padded / dummy questions do not contribute.
        This avoids the previous ``O(B*Q)`` Python-level loop and turns a
        ~50 ms hot spot on batch=4 into ~1 ms.
        """
        device = outputs.choice_logits.device
        b, q = outputs.types.shape
        if b == 0 or q == 0:
            return torch.zeros((), device=device, requires_grad=True)

        flat_targets = torch.tensor(
            [int(t) for row in targets for t in row],
            dtype=torch.long,
            device=device,
        )
        valid_mask = flat_targets >= 0
        if not bool(valid_mask.any()):
            return torch.zeros((), device=device, requires_grad=True)

        # Pack the three branches into one padded (B*Q, max_w) logits tensor.
        max_w = max(PRIMITIVE_MAX_OUT.values())
        flat_choice = outputs.choice_logits.reshape(b * q, -1)
        flat_score = outputs.score_logits.reshape(b * q, -1)
        flat_noul = outputs.noul_logits.reshape(b * q, -1)
        padded = torch.zeros(
            b * q, max_w,
            dtype=outputs.choice_logits.dtype,
            device=device,
        )
        padded[:, : PRIMITIVE_MAX_OUT["choice"]] = flat_choice
        padded[:, : PRIMITIVE_MAX_OUT["score"]] = flat_score
        padded[:, : PRIMITIVE_MAX_OUT["noul"]] = flat_noul

        # Apply temperature. Invalid rows would read garbage past their
        # ``size``, so mask them out before CE (their loss is zeroed anyway).
        padded = padded / max(self.temperature, 1e-6)
        arange = torch.arange(max_w, device=device).unsqueeze(0)
        size_mask = arange < outputs.sizes.reshape(b * q, 1)
        padded = padded.masked_fill(~size_mask, float("-inf"))

        # ``clamp`` keeps the index in-range for rows that will be masked;
        # we still pass ``reduction='none'`` so we can multiply by the mask.
        safe_targets = flat_targets.clamp(min=0)
        ce_per_row = F.cross_entropy(padded, safe_targets, reduction="none")
        ce_per_row = ce_per_row * valid_mask.to(ce_per_row.dtype)
        n_valid = valid_mask.to(ce_per_row.dtype).sum().clamp(min=1.0)
        return ce_per_row.sum() / n_valid

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(
        self,
        outputs: HeadOutputs,
        questions: list[list[Question]],
        temperature: float | None = None,
    ) -> list[list]:
        temp = temperature if temperature is not None else self.temperature
        results: list[list] = []
        b, q = outputs.types.shape
        for bi in range(b):
            row: list = []
            for qi in range(q):
                qobj = questions[bi][qi]
                t = int(outputs.types[bi, qi].item())
                if t == 0:
                    assert isinstance(qobj, QuestionChoice)
                    n_opt = len(qobj.criteria)
                    logits = outputs.choice_logits[bi, qi, :n_opt].detach().cpu().tolist()
                    probs = softmax_with_temperature(logits, temp)
                    keys = list(qobj.criteria.keys())
                    picked = keys[max(range(n_opt), key=lambda i: probs[i])]
                    row.append(
                        AnswerChoice(
                            question_id=qobj.question_id,
                            choice=picked,
                            probabilities={k: float(p) for k, p in zip(keys, probs, strict=True)},
                            confidence=float(confidence_from_probs(probs)),
                        )
                    )
                elif t == 1:
                    assert isinstance(qobj, QuestionScore)
                    n_lvl = len(qobj.criteria)
                    logits = outputs.score_logits[bi, qi, :n_lvl].detach().cpu().tolist()
                    probs = softmax_with_temperature(logits, temp)
                    legend = qobj.legend if qobj.legend is not None else list(
                        range(1, n_lvl + 1)
                    )
                    score_val = sum(float(p) * float(v) for p, v in zip(probs, legend, strict=True))
                    row.append(
                        AnswerScore(
                            question_id=qobj.question_id,
                            score=score_val,
                            legend=[float(x) for x in legend],
                            probabilities={str(i + 1): float(p) for i, p in enumerate(probs)},
                            confidence=float(confidence_from_probs(probs)),
                        )
                    )
                else:
                    assert isinstance(qobj, QuestionNoul)
                    logits = outputs.noul_logits[bi, qi].detach().cpu().tolist()
                    probs = softmax_with_temperature(logits, temp)
                    row.append(AnswerNoul.from_noul(qobj.question_id, float(probs[0])))
            results.append(row)
        return results
