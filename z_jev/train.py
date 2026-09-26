"""Training CLI for the tiny ``ZJevModel``.

Usage::

    python -m z_jev.train --dataset mixed --steps 200 --out checkpoints/tiny/ckpt.pt

Designed to run on CPU with at most a few hundred MB of RAM:

* ``batch_size`` defaults to 4,
* ``max_state_len`` defaults to 96,
* the tiny backbone is only 2 layers / 128 hidden.

Loss is cross-entropy per question (Noul is treated as a 2-way classifier;
"uncertain" comes from the probability threshold on top of that). On the
synthetic datasets this reaches above-random accuracy in tens of steps.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from z_jev.config import TinyGLMConfig, ZJevConfig
from z_jev.data import iter_batches, mixed_dataset, risk_only_dataset, spam_dataset
from z_jev.model import ZJevModel, collate_requests


def _build_dataset(name: str, n: int, seed: int):
    if name == "spam":
        return spam_dataset(n=n, seed=seed), "spam"
    if name == "mixed":
        return mixed_dataset(n=n, seed=seed), "mixed"
    if name == "risk":
        return risk_only_dataset(n=n, seed=seed), "risk"
    raise ValueError(f"Unknown dataset: {name}")


def evaluate(
    model: ZJevModel, examples, tokenizer, max_batches: int = 5, max_state_len: int = 128
) -> dict[str, float]:
    """Compute mean loss + accuracy on a subset of examples."""
    model.eval()
    correct = {0: 0, 1: 0, 2: 0}
    total = {0: 0, 1: 0, 2: 0}
    losses = []
    with torch.no_grad():
        for bi, batch in enumerate(iter_batches(examples, batch_size=4)):
            if bi >= max_batches:
                break
            fb = collate_requests(
                [ex.request for ex in batch], tokenizer, max_state_len=max_state_len
            )
            outputs = model(fb)
            targets = [ex.targets for ex in batch]
            loss = model.head.loss(outputs, targets)
            losses.append(loss.item())
            # Accuracy per primitive type
            for bi_b, ex in enumerate(batch):
                for qi, target in enumerate(ex.targets):
                    t = int(outputs.types[bi_b, qi].item())
                    if target < 0:
                        continue
                    if t == 0:
                        logits = outputs.choice_logits[bi_b, qi]
                    elif t == 1:
                        logits = outputs.score_logits[bi_b, qi]
                    else:
                        logits = outputs.noul_logits[bi_b, qi]
                    size = int(outputs.sizes[bi_b, qi].item())
                    logits = logits[:size]
                    pred = int(logits.argmax().item())
                    total[t] = total.get(t, 0) + 1
                    if pred == target:
                        correct[t] = correct.get(t, 0) + 1
    accs = {k: (correct[v] / max(1, total[v])) for k, v in [("choice", 0), ("score", 1), ("noul", 2)]}
    return {"loss": sum(losses) / max(1, len(losses)), "accuracy": accs}


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    cfg = ZJevConfig(
        mode="tiny",
        tiny=TinyGLMConfig(
            vocab_size=256,
            hidden_size=args.hidden,
            num_layers=args.layers,
            num_heads=args.heads,
            max_seq_len=max(128, args.max_state_len + 16),
            ffn_mult=args.ffn_mult,
        ),
    )
    model = ZJevModel(cfg)

    examples, ds_name = _build_dataset(args.dataset, args.n_train, args.seed)
    val_examples, _ = _build_dataset(args.dataset, args.n_val, args.seed + 1)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history: list[dict] = []
    print(f"[train] dataset={ds_name} n_train={len(examples)} n_val={len(val_examples)}")
    t0 = time.time()
    model.train()
    step = 0
    epoch = 0
    last_loss = None
    while step < args.steps:
        epoch += 1
        # Shuffle per epoch
        random.shuffle(examples)
        for batch in iter_batches(examples, batch_size=args.batch_size):
            if step >= args.steps:
                break
            fb = collate_requests(
                [ex.request for ex in batch],
                model.tokenizer,
                max_state_len=args.max_state_len,
            )
            outputs = model(fb)
            targets = [ex.targets for ex in batch]
            loss = model.head.loss(outputs, targets)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            last_loss = loss.item()
            history.append({"step": step, "loss": last_loss})
            if step % args.log_every == 0:
                print(
                    f"[train] step={step:4d} epoch={epoch} loss={last_loss:.4f} "
                    f"elapsed={time.time() - t0:.1f}s"
                )
            step += 1

    print(f"[train] final loss={last_loss:.4f}, evaluating on val...")
    metrics = evaluate(
        model, val_examples, model.tokenizer, max_batches=10, max_state_len=args.max_state_len
    )
    print(f"[val] {json.dumps(metrics)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "model.pt"
    extra = {
        "history": history,
        "val_metrics": metrics,
        "dataset": ds_name,
        "args": vars(args),
    }
    model.save_checkpoint(str(ckpt_path), extra=extra)
    print(f"[train] checkpoint saved to {ckpt_path}")

    # Also write a small JSON sidecar so users can inspect metrics without torch.
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(extra, f, indent=2)
    return {"history": history, "metrics": metrics, "checkpoint": str(ckpt_path)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a tiny Z-Jev model on CPU.")
    parser.add_argument("--dataset", choices=["spam", "mixed", "risk"], default="mixed")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--n-train", type=int, default=128)
    parser.add_argument("--n-val", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-state-len", type=int, default=96)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--out", type=str, default="checkpoints/tiny")
    args_ = parser.parse_args(argv)
    train(args_)


if __name__ == "__main__":  # pragma: no cover
    main()
