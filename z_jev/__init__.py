"""Z-Jev: non-autoregressive decision heads on top of GLM-5.

This package implements the three Jev decision primitives (Choice, Score, Noul)
as a non-autoregressive decoder that runs parallel heads over a backbone's
hidden states. Two backbones are supported:

- ``tiny``: a hand-written ``TinyGLM`` decoder-only transformer (CPU-trainable).
- ``glm5``: an adapter over the HuggingFace ``zai-org/GLM-5`` AutoModel.

See ``README.md`` for the full protocol, math definitions, and the honest
hardware constraints this implementation actually runs under.
"""

from z_jev.backbone import GLM5Backbone, TinyGLM
from z_jev.config import GLM5Config, TinyGLMConfig, ZJevConfig
from z_jev.head import HeadOutputs, NonAutoregressiveDecisionHead
from z_jev.model import ZJevModel
from z_jev.protocol import (
    Answer,
    AnswerChoice,
    AnswerNoul,
    AnswerScore,
    DecisionsRequest,
    DecisionsResponse,
    Question,
    QuestionChoice,
    QuestionNoul,
    QuestionScore,
    State,
)

__version__ = "0.1.0"

__all__ = [
    "ZJevConfig",
    "TinyGLMConfig",
    "GLM5Config",
    "State",
    "QuestionChoice",
    "QuestionScore",
    "QuestionNoul",
    "Question",
    "AnswerChoice",
    "AnswerScore",
    "AnswerNoul",
    "Answer",
    "DecisionsRequest",
    "DecisionsResponse",
    "NonAutoregressiveDecisionHead",
    "HeadOutputs",
    "ZJevModel",
    "GLM5Backbone",
    "TinyGLM",
]
