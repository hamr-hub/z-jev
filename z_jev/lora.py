"""Lightweight, dependency-free LoRA implementation for Z-Jev.

The official HuggingFace ``peft`` package is a soft dependency: we only
import it when ``--backbone glm5`` is selected. For the tiny CPU mode we
ship our own minimal LoRA so the train / save / load / merge path can be
exercised end-to-end without ``peft`` installed.

The math is the canonical LoRA decomposition (Hu et al., 2021):

    W' = W + (alpha / r) * (B @ A)
    where W in R^{d_out x d_in}, A in R^{r x d_in}, B in R^{d_out x r}

``B`` is zero-initialized so that at step 0 the wrapped layer produces
exactly the same output as the original frozen layer. ``A`` is initialised
with a small Gaussian (Kaiming-style) so its gradients are non-zero from
the first backward pass.

Public surface:

* :class:`LoRALinear` -- drop-in replacement for :class:`nn.Linear` that
  keeps the original weight frozen and learns a low-rank delta.
* :func:`inject_lora` -- walks an :class:`nn.Module` and replaces all
  ``nn.Linear`` whose name matches a substring filter with LoRA-wrapped
  versions.
* :func:`merge_lora` / :func:`unmerge_lora` -- fold the LoRA delta back
  into the base weight (or restore the original state). The merged model
  is byte-identical to the un-wrapped base after a merge + zero-init.
* :func:`lora_state_dict` / :func:`load_lora_state_dict` -- save / load
  only the trainable LoRA parameters.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LoRAConfig:
    """Hyperparameters for a LoRA injection pass."""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    # Only wrap linear layers whose attribute name (e.g. ``"qkv"``, ``"fc1"``)
    # appears in this iterable. An empty iterable means "wrap nothing";
    # passing ``["qkv", "fc1", "fc2", "out", "proj"]`` covers the TinyGLM
    # attention + MLP modules. ``None`` (the default) means "wrap every
    # Linear whose output dim >= min_target_dim" -- see
    # :func:`inject_lora`.
    target_names: Iterable[str] | None = None
    min_target_dim: int = 32
    # When True, the base weight is kept frozen and only LoRA params learn.
    freeze_base: bool = True


class LoRALinear(nn.Module):
    """LoRA-wrapped :class:`nn.Linear`.

    The original ``weight`` and ``bias`` are kept as buffers (frozen); the
    trainable parameters are ``lora_A`` (r x in_features) and ``lora_B``
    (out_features x r). Forward::

        y = (W + scaling * B @ A) @ x + b
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        # Frozen base weight as a Parameter so the parent module can save
        # it in state_dicts but we explicitly set requires_grad=False.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, a=0.0, mode="fan_in", nonlinearity="linear")
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        # Trainable LoRA params.
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_normal_(self.lora_A, a=0.0, mode="fan_in", nonlinearity="linear")
        # B is zero-initialised so the initial output equals the base layer.
        nn.init.zeros_(self.lora_B)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Freeze the base.
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = torch.nn.functional.linear(x, self.weight, self.bias)
        lora = torch.nn.functional.linear(
            self.dropout(x),
            (self.scaling * self.lora_B) @ self.lora_A,
            None,
        )
        return base + lora

    # ------------------------------------------------------------------
    # Merge / unmerge utilities
    # ------------------------------------------------------------------

    def get_delta_weight(self) -> torch.Tensor:
        """Return the LoRA delta: ``scaling * B @ A``."""
        return self.scaling * (self.lora_B @ self.lora_A)

    def merge(self) -> None:
        """Fold the LoRA delta into ``self.weight`` (in-place)."""
        with torch.no_grad():
            if not getattr(self, "_lora_B_backup", None):
                # Snapshot lora_B so ``unmerge`` can restore it.
                self._lora_B_backup = self.lora_B.detach().clone()
            self.weight.add_(self.get_delta_weight())
            nn.init.zeros_(self.lora_B)

    def unmerge(self) -> None:
        """Subtract the delta from ``self.weight`` and restore the LoRA B."""
        with torch.no_grad():
            backup = getattr(self, "_lora_B_backup", None)
            if backup is None:
                # Nothing to restore -- this layer was never merged.
                return
            self.weight.sub_(self.scaling * (backup @ self.lora_A))
            self.lora_B.copy_(backup)
            self._lora_B_backup = None


def _linear_target_predicate(name: str, target_names: Iterable[str] | None) -> bool:
    if not target_names:
        return False
    return any(t in name for t in target_names)


def inject_lora(
    module: nn.Module,
    cfg: LoRAConfig,
    prefix: str = "",
) -> list[str]:
    """Walk ``module`` and wrap matching :class:`nn.Linear` layers in place.

    Returns the dotted attribute names that were wrapped (e.g.
    ``"blocks.0.attn.qkv"``). The base weights are frozen regardless of
    ``cfg.freeze_base`` so the caller never accidentally trains them.
    """
    wrapped: list[str] = []
    for name, child in module.named_children():
        full = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"
        if isinstance(child, nn.Linear):
            if not _linear_target_predicate(name, cfg.target_names):
                continue
            if child.out_features < cfg.min_target_dim:
                continue
            if child.in_features < cfg.min_target_dim:
                continue
            lora = LoRALinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                rank=cfg.rank,
                alpha=cfg.alpha,
                dropout=cfg.dropout,
            )
            with torch.no_grad():
                lora.weight.copy_(child.weight)
                if child.bias is not None and lora.bias is not None:
                    lora.bias.copy_(child.bias)
            setattr(module, name, lora)
            wrapped.append(full)
        else:
            wrapped.extend(inject_lora(child, cfg, full))
    return wrapped


def lora_parameters(module: nn.Module) -> list[nn.Parameter]:
    """Return only the trainable LoRA parameters of a wrapped module."""
    params: list[nn.Parameter] = []
    for m in module.modules():
        if isinstance(m, LoRALinear):
            params.append(m.lora_A)
            params.append(m.lora_B)
    return params


def lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Extract only LoRA params into a saveable dict."""
    out: dict[str, torch.Tensor] = {}
    for name, m in module.named_modules():
        if isinstance(m, LoRALinear):
            out[f"{name}.lora_A"] = m.lora_A.detach().clone()
            out[f"{name}.lora_B"] = m.lora_B.detach().clone()
            out[f"{name}.weight"] = m.weight.detach().clone()
            if m.bias is not None:
                out[f"{name}.bias"] = m.bias.detach().clone()
    return out


def load_lora_state_dict(module: nn.Module, state: dict[str, torch.Tensor]) -> int:
    """Restore LoRA params + base weights. Returns the number of tensors loaded."""
    loaded = 0
    for name, m in module.named_modules():
        if isinstance(m, LoRALinear):
            for key in ("lora_A", "lora_B", "weight", "bias"):
                full = f"{name}.{key}"
                if full not in state:
                    continue
                param = getattr(m, key, None)
                if param is None:
                    continue
                with torch.no_grad():
                    param.copy_(state[full])
                loaded += 1
    return loaded


def merge_lora(module: nn.Module) -> None:
    """Fold every LoRA delta into its base weight (in-place)."""
    for m in module.modules():
        if isinstance(m, LoRALinear):
            m.merge()


def unmerge_lora(module: nn.Module) -> None:
    """Undo :func:`merge_lora` so the module is back in LoRA form."""
    for m in module.modules():
        if isinstance(m, LoRALinear):
            m.unmerge()


def freeze_non_lora(module: nn.Module) -> None:
    """Disable grad for every parameter except LoRA params (in-place)."""
    for name, p in module.named_parameters():
        if ".lora_A" in name or ".lora_B" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False


def try_import_peft() -> tuple[bool, str]:
    """Soft-import peft. Returns (available, install_hint)."""
    try:
        import peft  # noqa: F401

        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, (
            "peft is required for --backbone glm5 LoRA training. "
            "Install with: pip install peft>=0.10 transformers>=4.40"
            f" (got: {exc.__class__.__name__}: {exc})"
        )


def try_import_transformers() -> tuple[bool, str]:
    """Soft-import transformers. Returns (available, install_hint)."""
    try:
        import transformers  # noqa: F401

        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, (
            "transformers is required for --backbone glm5 LoRA training. "
            "Install with: pip install transformers>=4.40"
            f" (got: {exc.__class__.__name__}: {exc})"
        )
