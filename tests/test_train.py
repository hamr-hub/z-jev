"""End-to-end tiny training test: small budget, assert loss decreases."""

import torch

from z_jev import ZJevConfig, ZJevModel
from z_jev.data import spam_dataset
from z_jev.model import collate_requests
from z_jev.train import train as train_fn


def _build_args(tmp_path, **overrides):
    import argparse

    defaults = dict(
        dataset="spam",
        steps=120,
        n_train=256,
        n_val=64,
        batch_size=4,
        lr=5e-3,
        seed=0,
        max_state_len=96,
        hidden=128,
        layers=2,
        heads=4,
        ffn_mult=4,
        log_every=20,
        out=str(tmp_path / "ckpt"),
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_train_runs_and_loss_drops(tmp_path):
    args = _build_args(tmp_path)
    import os

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = train_fn(args)
    finally:
        os.chdir(cwd)
    history = result["history"]
    assert len(history) >= args.steps
    losses = [h["loss"] for h in history if h.get("loss") is not None]
    assert losses[-1] < losses[0], (losses[0], losses[-1])
    # Choice accuracy should beat random (50%).
    metrics = result["metrics"]
    assert metrics["accuracy"]["choice"] >= 0.5, metrics["accuracy"]
    # Checkpoint + metrics file exist.
    import json
    metrics_path = tmp_path / "ckpt" / "metrics.json"
    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text())
    assert "history" in payload
    assert "val_metrics" in payload


def test_train_uses_choice_targets_correctly(tmp_path):
    """Sanity: model should be able to overfit a tiny hand-crafted batch."""
    import os

    os.chdir(tmp_path)
    cfg = ZJevConfig()
    model = ZJevModel(cfg)
    examples = spam_dataset(n=16, seed=1)
    batch = collate_requests([ex.request for ex in examples], model.tokenizer)
    targets = [ex.targets for ex in examples]
    outputs = model(batch)
    loss_before = model.head.loss(outputs, targets).item()
    optim = torch.optim.AdamW(model.parameters(), lr=5e-3)
    for _ in range(30):
        outputs = model(batch)
        loss = model.head.loss(outputs, targets)
        optim.zero_grad()
        loss.backward()
        optim.step()
    assert loss.item() < loss_before
