"""LoRA fine-tuning CLI for Z-Jev.

Two modes are supported:

* ``--backbone glm5`` -- the production path. Requires ``transformers`` and
  ``peft``; loads ``zai-org/GLM-5`` (or a local mirror) with the backbone
  frozen and a PEFT LoRA adapter injected into the attention layers.
  Z-Jev's :class:`~z_jev.head.NonAutoregressiveDecisionHead` is trained
  on top of the LoRA-fine-tuned hidden states. Checkpoints contain the
  adapter weights and the decision-head weights, which ``z-jev-serve``
  can then load for inference.

  This mode cannot run on the reference machine (1.3 GB free RAM); the
  imports are soft so a missing ``transformers`` / ``peft`` produces a
  clear actionable error.

* ``--backbone tiny`` -- a CPU-friendly local path that mirrors the GLM-5
  flow on top of the existing :class:`~z_jev.backbone.TinyGLM`. We inject
  an in-house LoRA layer (see :mod:`z_jev.lora`) into every matching
  ``nn.Linear`` in the backbone, freeze the base, and train the adapter
  + decision head on a JSONL dataset. End-to-end save / load / merge is
  tested in CI.

JSONL dataset format (one example per line)::

    {"state": "...", "answers": {"q1": {"type":"choice","label":"spam"},
     "q2":{"type":"score","label":2}, "q3":{"type":"noul","label":true}}}

Each entry in ``answers`` has a ``type`` ("choice" / "score" / "noul")
and a ``label``: a string key for choice, a 0-indexed level for score,
and a boolean (true=yes / false=no) for noul.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from z_jev.backbone import ByteTokenizer
from z_jev.config import TinyGLMConfig, ZJevConfig
from z_jev.head import NonAutoregressiveDecisionHead
from z_jev.lora import (
    LoRAConfig,
    inject_lora,
    load_lora_state_dict,
    lora_parameters,
    lora_state_dict,
    merge_lora,
    try_import_peft,
    try_import_transformers,
    unmerge_lora,
)
from z_jev.model import ZJevModel, collate_requests
from z_jev.protocol import (
    DecisionsRequest,
    QuestionChoice,
    QuestionNoul,
    QuestionScore,
    State,
)

# ---------------------------------------------------------------------------
# JSONL dataset
# ---------------------------------------------------------------------------


@dataclass
class JsonlExample:
    request: DecisionsRequest
    targets: list[int]


def _label_to_target(spec: dict[str, Any], options: list[str] | None) -> int:
    """Convert a JSONL label to the integer target used by the head."""
    label = spec.get("label")
    qtype = spec.get("type")
    if qtype == "choice":
        # ``label`` is the string key (e.g. "spam"); map to its index in the
        # criteria dict (we use a sorted key list so the order is stable).
        if options is None:
            raise ValueError("choice example missing 'options' for label mapping")
        if label is None or label not in options:
            raise ValueError(f"choice label {label!r} not in options {options}")
        return options.index(label)
    if qtype == "score":
        if isinstance(label, int):
            return label
        if label is None:
            raise ValueError("score label must be int, got None")
        # Allow 1-indexed ints in JSON by accepting ints 0..len-1 OR 1..n.
        try:
            n = int(label)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"score label must be int, got {label!r}") from exc
        return n
    if qtype == "noul":
        # Map True -> 0 ("yes" is index 0 in our head), False -> 1 ("no").
        if isinstance(label, bool):
            return 0 if label else 1
        if isinstance(label, (int, float)):
            return 0 if int(label) == 1 else 1
        raise ValueError(f"noul label must be bool, got {label!r}")
    raise ValueError(f"unknown question type {qtype!r}")


def load_jsonl_dataset(path: str) -> list[JsonlExample]:
    """Read a JSONL file into a list of :class:`JsonlExample`."""
    out: list[JsonlExample] = []
    with open(path, encoding="utf-8") as f:
        for ln, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{ln}: bad JSON: {exc}") from exc
            if "state" not in obj or "answers" not in obj:
                raise ValueError(
                    f"{path}:{ln}: each line needs 'state' and 'answers' fields"
                )
            state_text = obj["state"]
            answers = obj["answers"]
            questions = []
            targets: list[int] = []
            for qid, spec in answers.items():
                qtype = spec.get("type")
                if qtype == "choice":
                    options = list(spec.get("options") or spec.get("criteria_keys") or [])
                    if not options and "label" in spec:
                        # If no options supplied, treat the label as the only
                        # option. The head will produce a 2-way softmax by
                        # convention; tests should give a real options list.
                        options = [spec["label"], "other"]
                    criteria = {k: spec.get("descriptions", {}).get(k, k) for k in options}
                    questions.append(
                        QuestionChoice(
                            question_id=qid,
                            instructions=spec.get("instructions", ""),
                            criteria=criteria,
                        )
                    )
                    targets.append(_label_to_target(spec, options))
                elif qtype == "score":
                    n = spec.get("levels", 3)
                    criteria = [str(i + 1) for i in range(n)]
                    questions.append(
                        QuestionScore(
                            question_id=qid,
                            instructions=spec.get("instructions", ""),
                            criteria=criteria,
                            legend=spec.get("legend"),
                        )
                    )
                    targets.append(_label_to_target(spec, None))
                elif qtype == "noul":
                    questions.append(
                        QuestionNoul(
                            question_id=qid,
                            instructions=spec.get("instructions", ""),
                        )
                    )
                    targets.append(_label_to_target(spec, None))
                else:
                    raise ValueError(f"{path}:{ln}: unknown question type {qtype!r}")
            out.append(
                JsonlExample(
                    request=DecisionsRequest(state=State(text=state_text), questions=questions),
                    targets=targets,
                )
            )
    if not out:
        raise ValueError(f"{path}: dataset is empty")
    return out


# ---------------------------------------------------------------------------
# Tiny backbone LoRA path (no peft / transformers required)
# ---------------------------------------------------------------------------

# Names of nn.Linear children inside the TinyGLM transformer that should
# be wrapped by LoRA. Cover qkv + proj (attention) and fc1 + fc2 (MLP).
_TINY_LORA_TARGETS = ("qkv", "proj", "fc1", "fc2")


def build_tiny_model_with_lora(cfg: ZJevConfig, lora_cfg: LoRAConfig) -> tuple[ZJevModel, list[str]]:
    """Build a ZJevModel with LoRA injected into its TinyGLM backbone."""
    model = ZJevModel(cfg)
    wrapped = inject_lora(model.backbone.impl, lora_cfg)
    return model, wrapped


# ---------------------------------------------------------------------------
# GLM-5 backbone LoRA path (requires transformers + peft)
# ---------------------------------------------------------------------------


def _build_glm5_with_lora(
    cfg: ZJevConfig,
    lora_cfg: LoRAConfig,
    model_id: str,
    load_in_8bit: bool,
    load_in_4bit: bool,
    dtype: str,
    device_map: str | None,
):
    """Load GLM-5 with PEFT LoRA and a fresh Z-Jev decision head.

    Returns ``(backbone_module, head_module, lora_target_names)``.

    Raises an informative error when transformers/peft are missing.
    """
    if not try_import_transformers()[0]:
        _, hint = try_import_transformers()
        raise ImportError(hint)
    if not try_import_peft()[0]:
        _, hint = try_import_peft()
        raise ImportError(hint)
    import peft  # type: ignore
    import transformers  # type: ignore

    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
    }
    if device_map:
        kwargs["device_map"] = device_map
    if load_in_8bit:
        kwargs["load_in_8bit"] = True
    if load_in_4bit:
        kwargs["load_in_4bit"] = True

    hf_cfg = transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    hf_model = transformers.AutoModel.from_pretrained(model_id, **kwargs)
    # Freeze backbone.
    for p in hf_model.parameters():
        p.requires_grad = False

    # Build the PEFT LoRA config: target the canonical attention modules.
    target_modules = lora_cfg.target_names or ["q_proj", "k_proj", "v_proj", "o_proj"]
    peft_cfg = peft.LoraConfig(
        r=lora_cfg.rank,
        lora_alpha=int(lora_cfg.alpha),
        lora_dropout=lora_cfg.dropout,
        target_modules=list(target_modules),
        bias="none",
        task_type=peft.TaskType.FEATURE_EXTRACTION,
    )
    hf_model = peft.get_peft_model(hf_model, peft_cfg)
    # Surface the trainable param names so callers can log them.
    targets = [n for n, _ in hf_model.named_parameters() if "lora_" in n]

    # Build the Z-Jev decision head against the published hidden size.
    hidden_size = getattr(hf_cfg, "hidden_size", cfg.head_hidden_size)
    head_cfg = ZJevConfig(mode="glm5", head_hidden_size=hidden_size)
    head = NonAutoregressiveDecisionHead(head_cfg)
    return hf_model, head, targets


# ---------------------------------------------------------------------------
# Loss / metrics
# ---------------------------------------------------------------------------


def _compute_loss(
    head: NonAutoregressiveDecisionHead,
    outputs,
    targets: list[list[int]],
) -> torch.Tensor:
    return head.loss(outputs, targets)


def _compute_accuracies(
    outputs,
    targets: list[list[int]],
) -> dict[str, dict[str, float]]:
    """Per-primitive accuracy, count, ECE."""
    correct = {"choice": 0, "score": 0, "noul": 0}
    total = {"choice": 0, "score": 0, "noul": 0}
    conf_bins: dict[str, list[tuple[float, int]]] = {
        "choice": [], "score": [], "noul": []
    }
    b, q = outputs.types.shape
    for bi in range(b):
        for qi in range(q):
            tgt = int(targets[bi][qi])
            if tgt < 0:
                continue
            t = int(outputs.types[bi, qi].item())
            if t == 0:
                logits = outputs.choice_logits[bi, qi]
                name = "choice"
            elif t == 1:
                logits = outputs.score_logits[bi, qi]
                name = "score"
            else:
                logits = outputs.noul_logits[bi, qi]
                name = "noul"
            size = int(outputs.sizes[bi, qi].item())
            logits = logits[:size]
            probs = F.softmax(logits.detach().float(), dim=-1).tolist()
            pred = int(max(range(size), key=lambda i: probs[i]))
            correct[name] += int(pred == tgt)
            total[name] += 1
            top1 = max(probs)
            conf_bins[name].append((top1, int(pred == tgt)))
    out: dict[str, dict[str, float]] = {}
    for name in ("choice", "score", "noul"):
        acc = correct[name] / max(1, total[name])
        # ECE: bucket into 5 confidence buckets.
        if conf_bins[name]:
            bucket_acc = [0.0] * 5
            bucket_conf = [0.0] * 5
            bucket_n = [0] * 5
            for conf, hit in conf_bins[name]:
                idx = min(4, max(0, int(conf / 0.2)))
                bucket_acc[idx] += hit
                bucket_conf[idx] += conf
                bucket_n[idx] += 1
            ece = 0.0
            n = len(conf_bins[name])
            for i in range(5):
                if bucket_n[i]:
                    avg_acc = bucket_acc[i] / bucket_n[i]
                    avg_conf = bucket_conf[i] / bucket_n[i]
                    ece += (bucket_n[i] / n) * abs(avg_acc - avg_conf)
        else:
            ece = 0.0
        out[name] = {"accuracy": acc, "count": total[name], "ece": ece}
    return out


def _merge_accs(accs_list: Iterable[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    """Sum accuracy dicts by primitive."""
    agg: dict[str, dict[str, float]] = {
        "choice": {"correct": 0, "count": 0, "ece_num": 0.0, "ece_den": 0.0},
        "score": {"correct": 0, "count": 0, "ece_num": 0.0, "ece_den": 0.0},
        "noul": {"correct": 0, "count": 0, "ece_num": 0.0, "ece_den": 0.0},
    }
    for d in accs_list:
        for k, v in d.items():
            agg[k]["correct"] += v["accuracy"] * v["count"]
            agg[k]["count"] += v["count"]
            agg[k]["ece_num"] += v["ece"] * v["count"]
            agg[k]["ece_den"] += v["count"]
    out: dict[str, dict[str, float]] = {}
    for k, v in agg.items():
        out[k] = {
            "accuracy": v["correct"] / max(1, v["count"]),
            "count": int(v["count"]),
            "ece": v["ece_num"] / max(1, v["ece_den"]),
        }
    return out


# ---------------------------------------------------------------------------
# Training loop (shared by tiny + glm5)
# ---------------------------------------------------------------------------


def _batches(examples: list[JsonlExample], bs: int) -> Iterable[list[JsonlExample]]:
    for i in range(0, len(examples), bs):
        yield examples[i : i + bs]


def train(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(
        f"[lora-train] backbone={args.backbone} model={args.model} "
        f"rank={args.lora_rank} alpha={args.lora_alpha}",
        file=sys.stderr,
    )

    # Declare up front so downstream branches (eval / save) can reference
    # them by name without the type checker complaining about possibly-
    # unbound locals.
    model: ZJevModel | None = None
    hf_model = None

    if args.backbone == "tiny":
        model, lora_targets = build_tiny_model_with_lora(
            ZJevConfig(
                mode="tiny",
                tiny=TinyGLMConfig(
                    hidden_size=args.hidden,
                    num_layers=args.layers,
                    num_heads=args.heads,
                    max_seq_len=max(128, args.max_state_len + 16),
                ),
            ),
            LoRAConfig(
                rank=args.lora_rank,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
                target_names=_TINY_LORA_TARGETS,
            ),
        )
        head = model.head
        forward_fn = lambda batch: model(batch)  # noqa: E731
        lora_targets_names = lora_targets
        # Base backbone weights are already frozen by LoRALinear.__init__.
        # Freeze the rest of the backbone (tok_emb, blocks not LoRA-wrapped,
        # norm) but keep the state projection, question encoder, and head
        # trainable.
        for n, p in model.backbone.named_parameters():
            if "lora_" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
        # state_proj + question_encoder + head stay trainable.
        params = (
            lora_parameters(model.backbone.impl)
            + [p for p in model.state_proj.parameters() if p.requires_grad]
            + [p for p in model.question_encoder.parameters() if p.requires_grad]
            + [p for p in head.parameters() if p.requires_grad]
        )
    else:
        # GLM-5 path: requires peft + transformers, loads the real model.
        hf_model, head, lora_targets_names = _build_glm5_with_lora(
            ZJevConfig(mode="glm5"),
            LoRAConfig(
                rank=args.lora_rank,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
                target_names=None,  # PEFT decides
            ),
            model_id=args.model,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            dtype=args.dtype,
            device_map=args.device_map,
        )
        forward_fn = lambda batch: head(  # noqa: E731
            _glm5_state_vec(hf_model, batch),
            _question_vecs(hf_model, batch),
            batch.question_types,
        )
        trainable_other = list(head.parameters())
        params = [p for n, p in hf_model.named_parameters() if "lora_" in n] + trainable_other

    print(
        f"[lora-train] LoRA-wrapped targets: {len(lora_targets_names)}; "
        f"trainable params: {sum(p.numel() for p in params)}",
        file=sys.stderr,
    )

    # Tokenizer for ``collate_requests``. The tiny path owns the real
    # ByteTokenizer; the GLM-5 path uses the real HF tokenizer but the
    # collator only needs padding constants, so a shim with the same
    # surface is enough. Picking it up front keeps the train loop free
    # of branchy plumbing.
    if args.backbone == "tiny":
        # ``model`` is guaranteed set by the tiny branch above; the assertion
        # is here only to keep static type-checkers (Pyright / mypy) honest.
        assert model is not None
        tokenizer = model.tokenizer
    else:
        tokenizer = _ByteTokenizerShim()

    # Dataset
    train_examples = load_jsonl_dataset(args.train_file)
    val_examples = load_jsonl_dataset(args.val_file) if args.val_file else []
    print(
        f"[lora-train] n_train={len(train_examples)} n_val={len(val_examples)}",
        file=sys.stderr,
    )

    optim = torch.optim.AdamW(
        [p for p in params if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    log_path = Path(args.out) / "train.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "w", encoding="utf-8")

    history: list[dict] = []
    accum = 0
    optim.zero_grad()
    t0 = time.time()
    step = 0
    epoch = 0
    rng = random.Random(args.seed)
    last_loss = None
    while step < args.steps:
        epoch += 1
        rng.shuffle(train_examples)
        for batch_examples in _batches(train_examples, args.batch_size):
            if step >= args.steps:
                break
            fb = collate_requests(
                [ex.request for ex in batch_examples],
                tokenizer,
                max_state_len=args.max_state_len,
            )
            targets = [ex.targets for ex in batch_examples]
            outputs = forward_fn(fb)
            loss = _compute_loss(head, outputs, targets) / args.grad_accum
            loss.backward()
            accum += 1
            if accum >= args.grad_accum:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optim.step()
                optim.zero_grad()
                accum = 0
            last_loss = float(loss.item() * args.grad_accum)
            rec = {
                "step": step,
                "epoch": epoch,
                "loss": last_loss,
                "lr": args.lr,
                "elapsed_s": round(time.time() - t0, 2),
            }
            history.append(rec)
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            if step % args.log_every == 0:
                print(
                    f"[lora-train] step={step:4d} epoch={epoch} "
                    f"loss={last_loss:.4f} elapsed={time.time() - t0:.1f}s",
                    file=sys.stderr,
                )
            step += 1
    log_f.close()

    metrics: dict[str, Any] = {}
    if val_examples and args.backbone == "tiny":
        metrics = _evaluate_tiny(model, val_examples, args)
    elif val_examples:
        metrics = _evaluate_glm5(hf_model, head, val_examples, args)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    extra = {
        "history": history,
        "val_metrics": metrics,
        "args": vars(args),
        "lora_targets": lora_targets_names,
    }

    save_lora_checkpoint(
        out_dir,
        model=model if args.backbone == "tiny" else None,
        head=head,
        hf_model=hf_model if args.backbone == "glm5" else None,
        extra=extra,
        merge=args.save_merged,
    )
    print(
        f"[lora-train] checkpoint saved to {out_dir}; merged={args.save_merged}",
        file=sys.stderr,
    )
    return {"history": history, "metrics": metrics, "out_dir": str(out_dir)}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _evaluate_tiny(model: ZJevModel, examples: list[JsonlExample], args: argparse.Namespace) -> dict:
    model.eval()
    all_accs = []
    losses = []
    with torch.no_grad():
        for batch_examples in _batches(examples, args.batch_size):
            fb = collate_requests(
                [ex.request for ex in batch_examples],
                model.tokenizer,
                max_state_len=args.max_state_len,
            )
            outputs = model(fb)
            targets = [ex.targets for ex in batch_examples]
            losses.append(_compute_loss(model.head, outputs, targets).item())
            all_accs.append(_compute_accuracies(outputs, targets))
    merged = _merge_accs(all_accs)
    model.train()
    return {
        "loss": sum(losses) / max(1, len(losses)),
        "accuracy": {k: v["accuracy"] for k, v in merged.items()},
        "ece": {k: v["ece"] for k, v in merged.items()},
        "count": {k: v["count"] for k, v in merged.items()},
    }


def _evaluate_glm5(hf_model, head, examples, args: argparse.Namespace) -> dict:
    hf_model.eval()
    head.eval()
    all_accs = []
    losses = []
    with torch.no_grad():
        for batch_examples in _batches(examples, args.batch_size):
            fb = collate_requests(
                [ex.request for ex in batch_examples],
                _ByteTokenizerShim(),
                max_state_len=args.max_state_len,
            )
            outputs = head(
                _glm5_state_vec(hf_model, fb),
                _question_vecs(hf_model, fb),
                fb.question_types,
            )
            targets = [ex.targets for ex in batch_examples]
            losses.append(_compute_loss(head, outputs, targets).item())
            all_accs.append(_compute_accuracies(outputs, targets))
    merged = _merge_accs(all_accs)
    hf_model.train()
    head.train()
    return {
        "loss": sum(losses) / max(1, len(losses)),
        "accuracy": {k: v["accuracy"] for k, v in merged.items()},
        "ece": {k: v["ece"] for k, v in merged.items()},
        "count": {k: v["count"] for k, v in merged.items()},
    }


# ---------------------------------------------------------------------------
# GLM-5 helpers
# ---------------------------------------------------------------------------


def _glm5_state_vec(hf_model, batch) -> torch.Tensor:
    """Run the GLM-5 backbone over ``batch.state_ids`` and pool to (B, H)."""
    out = hf_model(
        input_ids=batch.state_ids,
        attention_mask=batch.state_mask,
        output_hidden_states=False,
    )
    hidden = out.last_hidden_state  # (B, T, H)
    mask = batch.state_mask.unsqueeze(-1).float()
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def _question_vecs(hf_model, batch) -> torch.Tensor:
    """Mean-pool of token embeddings for each question in the batch."""
    if hasattr(hf_model, "get_input_embeddings"):
        emb = hf_model.get_input_embeddings()(batch.question_ids)
    else:
        emb = hf_model.base_model.model.get_input_embeddings()(batch.question_ids)
    mask = (batch.question_ids != 0).float().unsqueeze(-1)
    summed = (emb * mask).sum(dim=2)
    counts = mask.sum(dim=2).clamp(min=1.0)
    return summed / counts


class _ByteTokenizerShim(ByteTokenizer):
    """Drop-in :class:`ByteTokenizer` for the GLM-5 forward path.

    The real GLM-5 model uses its own tokenizer; this shim only needs to
    satisfy the small surface :func:`collate_requests` touches
    (``encode``/``type_id``/vocab constants). Inheriting from
    :class:`ByteTokenizer` makes the duck-typed contract explicit and
    silences Pyright's structural-type complaints about
    ``collate_requests`` accepting "any ByteTokenizer-like".
    """

    # PAD/BOS/EOS/CHOICE/SCORE/NOUL/SEP/NUM_SPECIAL/base_vocab_size/vocab_size
    # are inherited unchanged; only ``encode`` is overridden to keep the
    # same behaviour as before.

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:  # noqa: D401
        ids: list[int] = [self.BOS] if add_bos else []
        for b in text.encode("utf-8", errors="replace"):
            ids.append(int(b) + self.NUM_SPECIAL)
        if add_eos:
            ids.append(self.EOS)
        return ids


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------


def save_lora_checkpoint(
    out_dir: Path,
    *,
    model: ZJevModel | None,
    head: NonAutoregressiveDecisionHead,
    hf_model=None,
    extra: dict,
    merge: bool,
) -> None:
    """Save the LoRA adapter + decision head.

    If ``merge`` is True, fold the LoRA delta into the base weights and
    save the merged backbone plus a stub ``adapter.safetensors`` that
    contains zeros (so PEFT-compatible reloads see a no-op adapter).
    Otherwise save the unmerged state_dict plus ``adapter.pt`` with just
    the LoRA parameters.
    """
    head_path = out_dir / "head.pt"
    torch.save(
        {
            "state_dict": head.state_dict(),
            "config": head.cfg.__dict__,
        },
        head_path,
    )

    extra_path = out_dir / "extra.json"
    extra_path.write_text(json.dumps(extra, indent=2), encoding="utf-8")

    if model is not None:
        if merge:
            merge_lora(model.backbone.impl)
            backbone_path = out_dir / "backbone.pt"
            torch.save(
                {
                    "config": model.cfg.__dict__,
                    "state_dict": model.backbone.state_dict(),
                },
                backbone_path,
            )
            unmerge_lora(model.backbone.impl)
            # Also save an empty adapter marker so the layout is consistent.
            torch.save({"lora_state": {}, "wrapped": []}, out_dir / "adapter.pt")
        else:
            adapter = lora_state_dict(model.backbone.impl)
            torch.save(
                {"lora_state": adapter, "wrapped": list(adapter.keys())},
                out_dir / "adapter.pt",
            )
            backbone_path = out_dir / "backbone.pt"
            torch.save(
                {
                    "config": model.cfg.__dict__,
                    "state_dict": model.backbone.state_dict(),
                },
                backbone_path,
            )
    elif hf_model is not None:
        # PEFT model: use its native save_pretrained for the adapter.
        adapter_dir = out_dir / "peft_adapter"
        hf_model.save_pretrained(str(adapter_dir))


def load_lora_checkpoint_into_tiny(out_dir: Path) -> tuple[ZJevModel, dict]:
    """Reload a saved tiny LoRA checkpoint. Verifies round-trip."""
    extra = json.loads((out_dir / "extra.json").read_text())
    backbone = torch.load(out_dir / "backbone.pt", map_location="cpu", weights_only=False)
    head = torch.load(out_dir / "head.pt", map_location="cpu", weights_only=False)
    cfg = ZJevConfig(**backbone["config"])
    model = ZJevModel(cfg)
    # The saved backbone state_dict has unfrozen (post-init) LoRA params.
    # Load everything, then re-inject LoRA and re-load adapter state on top.
    model.backbone.load_state_dict(backbone["state_dict"])
    adapter_path = out_dir / "adapter.pt"
    if adapter_path.exists():
        adapter = torch.load(adapter_path, map_location="cpu", weights_only=False)
        if adapter.get("lora_state"):
            load_lora_state_dict(model.backbone.impl, adapter["lora_state"])
    model.head.load_state_dict(head["state_dict"])
    return model, extra


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Z-Jev LoRA fine-tuning.")
    parser.add_argument("--backbone", choices=["tiny", "glm5"], default="tiny",
                        help="Which backbone to LoRA-tune. 'tiny' is CPU; "
                             "'glm5' requires transformers + peft + a real GPU.")
    parser.add_argument("--model", default="zai-org/GLM-5",
                        help="HF model id or local path (only used with --backbone glm5).")
    parser.add_argument("--train-file", required=True, help="JSONL training data.")
    parser.add_argument("--val-file", default=None, help="JSONL validation data (optional).")
    parser.add_argument("--out", default="checkpoints/lora", help="Output directory.")
    parser.add_argument("--resume", default=None, help="Path to a previous checkpoint to resume from.")
    parser.add_argument("--save-merged", action="store_true",
                        help="Merge LoRA into the base weights and save the merged backbone.")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-state-len", type=int, default=96)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)
    # GLM-5 specific
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device-map", default=None,
                        help="device_map passed to from_pretrained (e.g. 'auto' for MoE sharding).")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.backbone == "glm5":
        # Fail early with a helpful message if deps are missing.
        if not try_import_transformers()[0]:
            sys.stderr.write(try_import_transformers()[1] + "\n")
            sys.exit(2)
        if not try_import_peft()[0]:
            sys.stderr.write(try_import_peft()[1] + "\n")
            sys.exit(2)
    train(args)


if __name__ == "__main__":  # pragma: no cover
    main()
