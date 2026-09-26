"""End-to-end :class:`ZJevModel` = backbone + decision heads.

Forward path:

1. ``state_ids`` -> :class:`backbone.GLM5Backbone` -> ``state_hidden`` (B, T, H)
2. Mean-pool ``state_hidden`` -> ``state_vec`` (B, H)
3. For every question in the request, build a textual prompt and run it
   through a *lightweight* :class:`QuestionEncoder` that re-uses the
   backbone's token embedding matrix -> ``question_vecs`` (B, Q, H)
4. Combine ``[state_vec, q_vec, type_emb]`` through
   :class:`head.NonAutoregressiveDecisionHead` -> per-question logits

Because every question is scored with one shared forward call, this is the
"non-autoregressive" half of the design: no token-by-token generation,
no language modelling head, no hallucination surface.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn

from z_jev.backbone import ByteTokenizer, GLM5Backbone
from z_jev.config import ZJevConfig
from z_jev.head import HeadOutputs, NonAutoregressiveDecisionHead
from z_jev.protocol import (
    Answer,
    DecisionsRequest,
    DecisionsResponse,
    Question,
    QuestionChoice,
    QuestionNoul,
    QuestionScore,
)

# ---------------------------------------------------------------------------
# Helpers: build question prompts that the encoder can tokenise
# ---------------------------------------------------------------------------


def _question_prompt(q: Question) -> str:
    """Concatenate the question metadata into a single text prompt.

    For Choice and Score we deliberately include only the **option keys**
    (not their descriptions). The descriptions would let the model shortcut
    to a constant answer regardless of the state, which collapses the
    state encoder to a no-op and prevents learning. The keys alone force
    the model to use the state to decide which option matches.
    """
    parts: list[str] = [q.instructions.strip()]
    if isinstance(q, QuestionChoice):
        for k in q.criteria:
            parts.append(str(k))
    elif isinstance(q, QuestionScore):
        for i, _v in enumerate(q.criteria):
            parts.append(f"level_{i + 1}")
    elif isinstance(q, QuestionNoul):
        if q.criteria:
            for k in q.criteria:
                parts.append(str(k))
    return " | ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Question encoder (shared across questions in a request)
# ---------------------------------------------------------------------------


class QuestionEncoder(nn.Module):
    """Mean-pool of token embeddings + small MLP -> per-question vector.

    Re-uses the backbone's token embedding matrix so the question encoder
    lives in the same vocabulary space without duplicating parameters.
    """

    def __init__(
        self,
        out_hidden_size: int,
        tok_emb: nn.Embedding,
        max_len: int = 96,
    ) -> None:
        super().__init__()
        self.out_hidden_size = out_hidden_size
        self.tok_emb = tok_emb
        self.max_len = max_len
        emb_dim = tok_emb.embedding_dim
        # Project from the embedding dim to whatever the head expects.
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, out_hidden_size),
            nn.GELU(),
            nn.LayerNorm(out_hidden_size),
        )

    def forward(self, q_ids: torch.Tensor) -> torch.Tensor:
        """``q_ids`` shape ``(B, Q, T)`` -> (B, Q, out_hidden_size)."""
        b, q, t = q_ids.shape
        emb = self.tok_emb(q_ids)  # (B, Q, T, emb_dim)
        # Mask out PAD tokens (id == 0) so they don't pull the mean around.
        pad_mask = (q_ids != ByteTokenizer.PAD).float().unsqueeze(-1)
        summed = (emb * pad_mask).sum(dim=2)
        counts = pad_mask.sum(dim=2).clamp(min=1.0)
        pooled = summed / counts  # (B, Q, emb_dim)
        return self.proj(pooled)


# ---------------------------------------------------------------------------
# ZJevModel
# ---------------------------------------------------------------------------


@dataclass
class ForwardBatch:
    """Padded tensors produced by :func:`collate_requests`."""

    state_ids: torch.Tensor  # (B, T)
    state_mask: torch.Tensor  # (B, T) -- not always needed; kept for HF
    question_ids: torch.Tensor  # (B, Q, Tq)
    question_types: torch.Tensor  # (B, Q) integer
    requests: list[DecisionsRequest]


def collate_requests(
    requests: Sequence[DecisionsRequest],
    tokenizer: ByteTokenizer,
    max_state_len: int = 128,
    max_question_len: int = 96,
) -> ForwardBatch:
    """Pad and batch a list of :class:`DecisionsRequest`.

    All requests in the batch must have the *same number of questions* --
    the protocol allows batching heterogeneous question sets per request,
    so callers pad with "dummy" noul questions if needed. We pad here to
    the max question count across the batch.
    """
    if not requests:
        raise ValueError("collate_requests: empty request list")
    n_q = max(len(r.questions) for r in requests)
    state_ids_list: list[list[int]] = []
    state_masks: list[list[int]] = []
    q_ids_padded: list[list[list[int]]] = []
    q_types: list[list[int]] = []
    for r in requests:
        sids = tokenizer.encode(r.state.text)[:max_state_len]
        if len(sids) < max_state_len:
            mask = [1] * len(sids) + [0] * (max_state_len - len(sids))
            sids = sids + [ByteTokenizer.PAD] * (max_state_len - len(sids))
        else:
            mask = [1] * max_state_len
        state_ids_list.append(sids)
        state_masks.append(mask)

        per_q_ids: list[list[int]] = []
        per_q_types: list[int] = []
        for q in r.questions:
            type_id = tokenizer.type_id(q.type)
            ids = [type_id] + tokenizer.encode(_question_prompt(q))[: max_question_len - 1]
            ids = ids + [ByteTokenizer.PAD] * (max_question_len - len(ids))
            per_q_ids.append(ids)
            per_q_types.append(0 if isinstance(q, QuestionChoice) else 1 if isinstance(q, QuestionScore) else 2)
        # Pad questions to n_q with PAD noul questions (type 2, empty prompt).
        while len(per_q_ids) < n_q:
            per_q_ids.append([ByteTokenizer.PAD] * max_question_len)
            per_q_types.append(2)  # noul -- harmless if logits ignored
        q_ids_padded.append(per_q_ids)
        q_types.append(per_q_types)

    return ForwardBatch(
        state_ids=torch.tensor(state_ids_list, dtype=torch.long),
        state_mask=torch.tensor(state_masks, dtype=torch.long),
        question_ids=torch.tensor(q_ids_padded, dtype=torch.long),
        question_types=torch.tensor(q_types, dtype=torch.long),
        requests=list(requests),
    )


class ZJevModel(nn.Module):
    """The end-to-end model: state encoder + question encoder + decision head."""

    def __init__(self, cfg: ZJevConfig, hf_cfg_only: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = GLM5Backbone(cfg, hf_cfg_only=hf_cfg_only)
        # TinyGLM exposes its token embedding; for the HF adapter we use a
        # fresh ByteTokenizer embedding so the question encoder stays
        # in a controlled space.
        if cfg.mode == "tiny":
            tok_emb = self.backbone.impl.tok_emb  # type: ignore[attr-defined]
            backbone_dim = self.backbone.impl.output_dim  # type: ignore[attr-defined]
        else:
            tok_emb = nn.Embedding(self.backbone.tokenizer.vocab_size, cfg.head_hidden_size)
            backbone_dim = cfg.head_hidden_size
        # Project the backbone output to the fixed ``head_hidden_size``.
        self.state_proj = nn.Sequential(
            nn.Linear(backbone_dim, cfg.head_hidden_size),
            nn.GELU(),
            nn.LayerNorm(cfg.head_hidden_size),
        )
        self.question_encoder = QuestionEncoder(cfg.head_hidden_size, tok_emb)
        self.head = NonAutoregressiveDecisionHead(cfg)
        self.tokenizer = self.backbone.tokenizer

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode_state(self, state_ids: torch.Tensor) -> torch.Tensor:
        """Encode the state text into a per-row feature vector (B, head_hidden_size).

        ``self.backbone(...)`` returns either ``(B, H)`` (when
        ``TinyGLMConfig.use_byte_pool`` is on -- the default for tiny) or
        ``(B, T, H)`` (when it is off). We normalise to ``(B, H)`` and then
        project to the head's expected hidden size.
        """
        out = self.backbone(state_ids)
        if out.dim() == 3:
            # Take the last non-pad position per row.
            pad_mask = (state_ids != ByteTokenizer.PAD)
            lengths = (pad_mask.sum(dim=1).clamp(min=1) - 1).long()
            bsz = out.shape[0]
            out = out[torch.arange(bsz, device=out.device), lengths]
        return self.state_proj(out)

    def forward(self, batch: ForwardBatch) -> HeadOutputs:
        state_vec = self.encode_state(batch.state_ids)  # (B, H_head)
        q_vecs = self.question_encoder(batch.question_ids)  # (B, Q, H_head)
        return self.head(state_vec, q_vecs, batch.question_types)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        request: DecisionsRequest,
        max_state_len: int = 128,
        max_question_len: int = 96,
        temperature: float | None = None,
    ) -> DecisionsResponse:
        self.eval()
        batch = collate_requests(
            [request], self.tokenizer, max_state_len, max_question_len
        )
        outputs = self.forward(batch)
        rows = self.head.decode(outputs, [request.questions], temperature=temperature)
        answers: dict[str, Answer] = {}
        for ans in rows[0]:
            answers[ans.question_id] = ans  # type: ignore[assignment]
        return DecisionsResponse(answers=answers)

    @torch.no_grad()
    def evaluate_batch(
        self,
        requests: Sequence[DecisionsRequest],
        max_state_len: int = 128,
        max_question_len: int = 96,
        temperature: float | None = None,
    ) -> list[DecisionsResponse]:
        self.eval()
        batch = collate_requests(requests, self.tokenizer, max_state_len, max_question_len)
        outputs = self.forward(batch)
        rows = self.head.decode(
            outputs, [r.questions for r in batch.requests], temperature=temperature
        )
        out: list[DecisionsResponse] = []
        for row in rows:
            answers = {a.question_id: a for a in row}  # type: ignore[union-attr]
            out.append(DecisionsResponse(answers=answers))
        return out

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str, extra: dict | None = None) -> None:
        payload = {
            "config": self.cfg.__dict__,
            "state_dict": self.state_dict(),
        }
        if extra:
            payload["extra"] = extra
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(cls, path: str, hf_cfg_only: bool = True) -> ZJevModel:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ZJevConfig(**payload["config"])
        model = cls(cfg, hf_cfg_only=hf_cfg_only)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model
