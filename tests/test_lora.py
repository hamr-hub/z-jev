"""Unit tests for the dependency-free LoRA implementation."""

from __future__ import annotations

import torch
import torch.nn as nn

from z_jev.lora import (
    LoRAConfig,
    LoRALinear,
    inject_lora,
    load_lora_state_dict,
    lora_parameters,
    lora_state_dict,
    merge_lora,
    try_import_peft,
    try_import_transformers,
    unmerge_lora,
)


def test_lora_zero_init_matches_base_forward():
    """At init (B=0) the LoRA-wrapped layer must be byte-identical to base."""
    torch.manual_seed(0)
    base = nn.Linear(8, 4)
    x = torch.randn(2, 8)
    y_base = base(x)

    lora = LoRALinear(8, 4, rank=4, alpha=16.0)
    with torch.no_grad():
        lora.weight.copy_(base.weight)
        if base.bias is not None and lora.bias is not None:
            lora.bias.copy_(base.bias)
    y_lora = lora(x)
    assert torch.allclose(y_base, y_lora, atol=1e-6)


def test_lora_merge_unmerge_round_trip():
    """merge preserves output; unmerge restores original behaviour."""
    torch.manual_seed(1)
    base = nn.Linear(8, 4)
    x = torch.randn(2, 8)

    lora = LoRALinear(8, 4, rank=4, alpha=16.0)
    with torch.no_grad():
        lora.weight.copy_(base.weight)
        if lora.bias is not None:
            lora.bias.copy_(base.bias)
        lora.lora_B.add_(torch.randn_like(lora.lora_B) * 0.5)

    y_pre = lora(x)
    merge_lora(lora)
    y_merged = lora(x)
    assert torch.allclose(y_pre, y_merged, atol=1e-5)
    unmerge_lora(lora)
    y_restored = lora(x)
    assert torch.allclose(y_pre, y_restored, atol=1e-5)


def test_inject_lora_wraps_only_named_targets():
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qkv = nn.Linear(16, 48)
            self.proj = nn.Linear(16, 16)
            self.fc1 = nn.Linear(16, 32)
            self.fc2 = nn.Linear(32, 16)
            self.unrelated = nn.Linear(8, 8)

    net = Net()
    wrapped = inject_lora(
        net,
        LoRAConfig(rank=4, target_names=("qkv", "proj", "fc1", "fc2"), min_target_dim=0),
    )
    assert "qkv" in wrapped and "proj" in wrapped and "fc1" in wrapped and "fc2" in wrapped
    assert "unrelated" not in wrapped
    # Unrelated must remain a plain Linear (untouched).
    assert isinstance(net.unrelated, nn.Linear)
    assert not isinstance(net.unrelated, LoRALinear)


def test_lora_state_dict_round_trip():
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qkv = nn.Linear(16, 48)

    torch.manual_seed(2)
    a = Net()
    inject_lora(a, LoRAConfig(rank=4, target_names=("qkv",), min_target_dim=0))
    x = torch.randn(2, 16)
    y_a = a.qkv(x)

    sd = lora_state_dict(a)
    b = Net()
    inject_lora(b, LoRAConfig(rank=4, target_names=("qkv",), min_target_dim=0))
    load_lora_state_dict(b, sd)
    y_b = b.qkv(x)
    assert torch.allclose(y_a, y_b, atol=1e-6)


def test_lora_parameters_lists_only_trainable():
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qkv = nn.Linear(16, 48)
            self.fc1 = nn.Linear(16, 32)

    torch.manual_seed(3)
    net = Net()
    inject_lora(net, LoRAConfig(rank=4, target_names=("qkv", "fc1"), min_target_dim=0))
    ps = lora_parameters(net)
    assert len(ps) == 4  # two A + two B
    for p in ps:
        assert p.requires_grad


def test_try_import_peft_returns_tuple():
    ok, hint = try_import_peft()
    assert isinstance(ok, bool)
    # When peft is not installed we get an install hint; when it is, hint is empty.
    if not ok:
        assert "pip install" in hint


def test_try_import_transformers_returns_tuple():
    ok, hint = try_import_transformers()
    assert isinstance(ok, bool)
    if not ok:
        assert "pip install" in hint
