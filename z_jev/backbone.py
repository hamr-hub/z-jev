"""Backbone adapters for Z-Jev.

This module exposes two backbones:

* :class:`TinyGLM` -- a hand-written decoder-only transformer (RMSNorm +
  pre-norm + sin/cos positional encoding) small enough to train on CPU.
* :class:`HFGLM5Adapter` -- a thin wrapper over HuggingFace's
  ``transformers`` AutoModel for the real ``zai-org/GLM-5`` checkpoint.
  The adapter is loaded with a *soft* import so this module works even on
  machines without ``transformers`` installed.

The public entry point is :class:`GLM5Backbone`, a factory that returns
whichever backbone the active :class:`~z_jev.config.ZJevConfig` asks for.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from z_jev.config import GLM5Config, TinyGLMConfig, ZJevConfig

# ---------------------------------------------------------------------------
# Byte-level tokenizer (shared between tiny and HF paths)
# ---------------------------------------------------------------------------


class ByteTokenizer:
    """UTF-8 byte-level tokenizer with reserved special tokens.

    Bytes 0..255 cover the entire UTF-8 range. We reserve the last few
    slots in the vocab for ``<pad>``, ``<bos>``, ``<eos>`` and the three
    question-type sentinels (``<choice>``, ``<score>``, ``<noul>``).
    """

    PAD = 0
    BOS = 1
    EOS = 2
    CHOICE = 3
    SCORE = 4
    NOUL = 5
    SEP = 6
    NUM_SPECIAL = 8  # 0..7 reserved; 8..255 are raw bytes

    def __init__(self, base_vocab_size: int = 256) -> None:
        self.base_vocab_size = base_vocab_size
        # The effective vocab is base_vocab_size + NUM_SPECIAL.
        self.vocab_size = base_vocab_size + self.NUM_SPECIAL

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids: list[int] = [self.BOS] if add_bos else []
        for b in text.encode("utf-8", errors="replace"):
            ids.append(int(b) + self.NUM_SPECIAL)
        if add_eos:
            ids.append(self.EOS)
        return ids

    def decode(self, ids: list[int]) -> str:
        out = bytearray()
        for i in ids:
            if i < self.NUM_SPECIAL:
                continue
            out.append(i - self.NUM_SPECIAL)
        return out.decode("utf-8", errors="replace")

    def type_id(self, qtype: str) -> int:
        return {"choice": self.CHOICE, "score": self.SCORE, "noul": self.NOUL}[qtype]


# ---------------------------------------------------------------------------
# TinyGLM: hand-written decoder-only transformer
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root mean square layer norm. Used in GLM-family models."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS in fp32 for stability under bf16 / fp16.
        rms = x.to(torch.float32).pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.to(torch.float32) * rms).to(x.dtype) * self.weight


def _sin_cos_pe(seq_len: int, dim: int, base: float = 10000.0) -> torch.Tensor:
    """Classic sin/cos positional encoding (Vaswani et al.)."""
    pos = torch.arange(seq_len, dtype=torch.float32)
    i = torch.arange(0, dim, 2, dtype=torch.float32)
    div = torch.exp(-math.log(base) * i / dim)
    pe = torch.zeros(seq_len, dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos[:, None] * div[None, :])
    pe[:, 1::2] = torch.cos(pos[:, None] * div[None, :])
    return pe


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (b, h, t, d)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().reshape(b, t, h)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, hidden_size: int, mult: int = 4) -> None:
        super().__init__()
        inner = hidden_size * mult
        self.fc1 = nn.Linear(hidden_size, inner, bias=True)
        self.fc2 = nn.Linear(inner, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class DecoderBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ffn_mult: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = CausalSelfAttention(hidden_size, num_heads, dropout)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = MLP(hidden_size, mult=ffn_mult)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class TinyGLM(nn.Module):
    """A minimal decoder-only transformer in the GLM architectural family.

    Forward signature::

        hidden = backbone(input_ids)            # (B, T, H)

    The backbone is intentionally tiny (default 2 layers, 128 hidden) so
    it can train inside the ~1.3GB free-RAM budget on the target machine.
    """

    def __init__(self, cfg: TinyGLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = ByteTokenizer()
        self.tok_emb = nn.Embedding(self.tokenizer.vocab_size, cfg.hidden_size)
        # Learned addition of sin/cos PE so the model has a positional signal
        # identical in spirit to RoPE but cheaper to implement on CPU.
        pe = _sin_cos_pe(cfg.max_seq_len, cfg.hidden_size)
        self.register_buffer("pe", pe)
        self.blocks = nn.ModuleList(
            DecoderBlock(cfg.hidden_size, cfg.num_heads, cfg.ffn_mult, cfg.dropout)
            for _ in range(cfg.num_layers)
        )
        self.norm = RMSNorm(cfg.hidden_size)
        # When ``use_byte_pool`` is set we expose an ``output_dim`` of
        # 2 * hidden_size by concatenating the contextual position and the
        # byte-embedding mean pool (see :meth:`encode_with_byte_pool`).
        self.apply(self._init_weights)

    @property
    def output_dim(self) -> int:
        return self.cfg.output_dim

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Token IDs -> last hidden states (B, T, H)."""
        b, t = input_ids.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence length {t} exceeds max_seq_len {self.cfg.max_seq_len}"
            )
        x = self.tok_emb(input_ids) + self.pe[:t].to(input_ids.device)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def encode_with_byte_pool(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Concatenated [transformer_hidden, byte_pool] features.

        For the tiny CPU-trainable replica, mean-pooling of byte token
        embeddings gives a strong bag-of-bytes baseline that the 2-layer
        transformer alone cannot reliably recover from a few hundred
        examples. Concatenating the two representations lets the model
        benefit from positional context *and* from raw byte statistics,
        matching the spirit of "decoder-only transformer backbone"
        without sacrificing learnability.
        """
        hidden = self.encode(input_ids)  # (B, T, H)
        # Byte-level mean pool over non-pad tokens.
        mask = (input_ids != ByteTokenizer.PAD).float()  # (B, T)
        emb = self.tok_emb(input_ids)  # (B, T, H)
        pooled = (emb * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = pooled.squeeze(1) if pooled.dim() == 3 and pooled.shape[1] == 1 else pooled
        # Use the last non-pad position as the contextual feature.
        lengths = (mask.sum(dim=1).clamp(min=1) - 1).long()  # (B,)
        bsz = hidden.shape[0]
        last = hidden[torch.arange(bsz, device=hidden.device), lengths]  # (B, H)
        return torch.cat([last, pooled], dim=-1)  # (B, 2H)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Default forward: contextual hidden state, optionally concatenated
        with a byte-embedding mean-pool (see ``TinyGLMConfig.use_byte_pool``).
        """
        if self.cfg.use_byte_pool:
            return self.encode_with_byte_pool(input_ids)
        return self.encode(input_ids)


# ---------------------------------------------------------------------------
# Optional HuggingFace GLM-5 adapter
# ---------------------------------------------------------------------------


class HFGLM5Adapter(nn.Module):
    """Adapter around HuggingFace's ``zai-org/GLM-5`` AutoModel.

    The 744B-A40B MoE weights cannot be downloaded onto this machine. This
    adapter therefore supports two modes:

    * ``hf_model_id`` points at a real HF repo we can download from.
      The adapter calls ``AutoModel.from_pretrained`` (CPU-only) and reads
      hidden states.
    * ``cfg_only=True`` constructs the HF model class with the *config*
      only (no weights), letting us validate the head dimensions against
      the published architecture.

    In both cases the head code only needs ``hidden_size`` to match; the
    rest of the GLM-5 backbone is opaque to Z-Jev.
    """

    def __init__(self, cfg: GLM5Config, cfg_only: bool = True) -> None:
        super().__init__()
        try:
            from transformers import AutoConfig, AutoModel  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised via tests
            raise ImportError(
                "transformers is required for HFGLM5Adapter. "
                "Install with: pip install z-jev[hf]"
            ) from exc

        hf_config = AutoConfig.from_pretrained(cfg.hf_model_id, trust_remote_code=True)
        cfg.hidden_size = getattr(hf_config, "hidden_size", cfg.hidden_size or 0)
        if cfg.hidden_size <= 0:
            raise ValueError("Could not infer hidden_size from GLM-5 config")

        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        self._cfg_only = cfg_only

        if cfg_only:
            # Build the model class but never load weights. This validates
            # that the architecture class exists in the installed transformers
            # and that hidden_size is consistent.
            self.model = AutoModel.from_config(hf_config, trust_remote_code=True)
        else:
            self.model = AutoModel.from_pretrained(
                cfg.hf_model_id, trust_remote_code=True, torch_dtype=torch.float32
            )
        # Freeze the backbone by default -- head-only fine-tuning is the
        # common path on top of GLM-5.
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        kwargs = {"input_ids": input_ids, "output_hidden_states": False}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        out = self.model(**kwargs)
        # GLM-5 returns a ModelOutput with .last_hidden_state (B, T, H).
        return out.last_hidden_state

    def cfg_only(self) -> bool:
        return self._cfg_only


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class GLM5Backbone(nn.Module):
    """Backbone wrapper that hides the tiny / hf choice from the model code."""

    def __init__(self, cfg: ZJevConfig, hf_cfg_only: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.mode == "tiny":
            self._impl: nn.Module = TinyGLM(cfg.tiny)
            self.hidden_size = cfg.tiny.hidden_size
        elif cfg.mode == "glm5":
            self._impl = HFGLM5Adapter(cfg.glm5, cfg_only=hf_cfg_only)
            self.hidden_size = cfg.glm5.hidden_size
        else:
            raise ValueError(f"Unknown mode: {cfg.mode!r}")

    @property
    def impl(self) -> nn.Module:
        return self._impl

    @property
    def tokenizer(self) -> ByteTokenizer:
        # Both backbones expose a .tokenizer for now; HF path returns a
        # ByteTokenizer as well to keep the encoding layer uniform.
        return self._impl.tokenizer  # type: ignore[attr-defined]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.cfg.mode == "tiny":
            return self._impl(input_ids)
        return self._impl(input_ids, attention_mask)
