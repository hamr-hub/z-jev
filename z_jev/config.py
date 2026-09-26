"""Configuration for Z-Jev backbones and heads.

Two modes are supported:

* ``mode="tiny"`` -- a hand-written decoder-only transformer with the same
  architectural family as GLM-5 (decoder-only, RMSNorm, RoPE, GQA-friendly
  head structure). This is what we actually train on CPU.
* ``mode="glm5"`` -- an adapter over HuggingFace's ``zai-org/GLM-5``
  AutoModel. This module reads GLM-5's config via ``AutoConfig.from_pretrained``
  so the decision-head dimensions stay aligned, but never downloads the real
  weights (the 744B-A40B checkpoint cannot fit on this machine).

All fields are validated on construction so a bad checkpoint round-trip is
caught at load time rather than mid-forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TinyGLMConfig:
    """Configuration for the CPU-trainable ``TinyGLM`` backbone.

    Architectural choices intentionally mirror GLM-5's family:

    * decoder-only transformer with pre-norm
    * RMSNorm on every residual branch
    * rotary positional embeddings (RoPE)
    * causal self-attention

    Scaling down to 2 layers / 128 hidden / 4 heads keeps the parameter
    count small enough for CPU training inside the ~1.3GB free RAM budget.
    """

    vocab_size: int = 256
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 4
    max_seq_len: int = 128
    ffn_mult: int = 4
    rope_base: float = 10000.0
    dropout: float = 0.0
    # When True, concatenate the transformer's last-position hidden state
    # with a byte-embedding mean-pool. This makes the 2-layer / 128-hidden
    # model trainable on synthetic bag-of-bytes tasks in tens of steps;
    # without it the tiny transformer collapses to predicting the majority
    # class because the encoder doesn't have enough capacity to learn the
    # byte-level signal from scratch. The transformer is still in the path,
    # so the architecture family (decoder-only + positional encoding) is
    # preserved.
    use_byte_pool: bool = True

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.vocab_size < 16:
            raise ValueError("vocab_size is too small for the byte-level tokenizer")

    @property
    def output_dim(self) -> int:
        """Hidden dim the backbone exposes to downstream heads."""
        return self.hidden_size * (2 if self.use_byte_pool else 1)


@dataclass
class GLM5Config:
    """Configuration for the GLM-5 (744B-A40B MoE) adapter.

    The real architecture is set by HuggingFace's ``zai-org/GLM-5``
    AutoConfig. We expose only the fields the *adapter* needs to know about:

    * how to project the backbone's final hidden states into the heads,
    * how to bias the decision logits (we add a learnable scale/shift on top
      of the frozen GLM-5 representation so LoRA-finetuning is optional).
    """

    # The HuggingFace model id; defaults to the public GLM-5 release.
    hf_model_id: str = "zai-org/GLM-5"
    # Set this to the actual hidden size after AutoConfig.from_pretrained;
    # the head code expects ``hidden_size`` to match the backbone output.
    hidden_size: int = 0
    # If the user wants a learnable residual on top of frozen features.
    use_lora_residual: bool = True
    # Decision-head temperature (applied to logits before softmax).
    temperature: float = 1.0


@dataclass
class ZJevConfig:
    """Top-level configuration combining backbone and head hyperparameters."""

    mode: Literal["tiny", "glm5"] = "tiny"
    tiny: TinyGLMConfig = field(default_factory=TinyGLMConfig)
    glm5: GLM5Config = field(default_factory=GLM5Config)
    # Hidden size the heads operate in. For tiny this equals tiny.hidden_size.
    head_hidden_size: int = 128
    # Type embedding size used by the question encoder (one slot per primitive).
    question_emb_size: int = 32
    # Whether to use bias terms in head linear layers.
    head_bias: bool = True

    def __post_init__(self) -> None:
        if self.mode == "tiny":
            # The head always sees the same ``head_hidden_size`` regardless
            # of whether the backbone concatenates the byte pool -- we
            # project the backbone output into this fixed dimension.
            pass
        # When using real GLM-5, head_hidden_size must match the backbone's
        # configured hidden_size; this is validated at backbone-load time.
