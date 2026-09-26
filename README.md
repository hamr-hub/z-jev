# Z-Jev

> **Non-autoregressive decision heads on top of GLM-5** —
> the Jev `Choice` / `Score` / `Noul` primitives as a single forward pass.

[![CI](https://github.com/hamr-hub/z-jev/actions/workflows/ci.yml/badge.svg)](https://github.com/hamr-hub/z-jev/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Z-Jev reimplements the [TypeSafe AI "Jev"](https://docs.typesafe.ai/primitives/)
decision API as a **non-autoregressive decoder**: instead of asking a
language model to generate free-form text and post-parse it, every
question is scored in one parallel pass through a dedicated head that
sits on top of a backbone's hidden states.

Two backbones are supported:

| Mode | Weights | Trains on this machine? |
| --- | --- | --- |
| `tiny` | ~790 K params (CPU-only) | yes |
| `glm5` | `zai-org/GLM-5` 744B-A40B MoE | not here -- see *Real GLM-5* below |

The **honest scope**: GLM-5 is far too large to load or train on the
target machine (a Jetson Orin Nano with ~1.3 GB free RAM). Z-Jev ships a
byte-level decoder-only transformer (`TinyGLM`) that mirrors GLM-5's
architectural family and is what we actually train. The GLM-5 adapter is
provided as a code path that uses `transformers.AutoModel` /
`AutoConfig` for size alignment -- no weights are downloaded.

---

## Architecture

```
                       ┌───────────────────────────────┐
   state text ───────► │  backbone (TinyGLM / GLM-5)   │ ─► state_vec
                       └───────────────────────────────┘
                                                            │
   questions[] ──► question encoder (mean-pool tok_emb) ──►│
                                                            ▼
                                            ┌──────────────────────────┐
                                            │ NonAutoregressiveDecision│
                                            │ Head: per-primitive MLP  │
                                            └─────────────┬────────────┘
                                                          ▼
                                      {Choice, Score, Noul} answers
```

* **Non-autoregressive**: every question's logits come out in the same
  forward call -- no token-by-token decoding, no language modelling head,
  no hallucination surface.
* **Per-primitive output**: Choice returns `len(options)` logits, Score
  returns `len(scale)` logits, Noul returns 2 logits (`yes`/`no`).
  Probabilities are computed by softmax *over the question's own output
  size*, so they always sum to 1.
* **Calibrated confidence**: top-1 probability minus the uniform
  baseline, normalised by `(1 - 1/K)`. For Noul the formula collapses to
  `2 * |p - 0.5|`. See `z_jev/scorer.py` for the exact math.

---

## Jev protocol (verbatim field names)

The wire format matches the upstream Jev spec at
<https://docs.typesafe.ai/primitives/>. The bodies that the model
returns look exactly like the official examples.

### Choice

```json
{
  "choice": "spam",
  "probabilities": {"spam": 0.83, "ham": 0.17},
  "confidence": 0.66
}
```

### Score

```json
{
  "score": 2.41,
  "legend": [1, 2, 3],
  "probabilities": {"1": 0.05, "2": 0.49, "3": 0.46},
  "confidence": 0.03
}
```

### Noul

```json
{ "noul": 0.82, "answer": "yes", "probability": 0.82, "confidence": 0.64 }
```

> `noul` is the canonical field. `answer` / `probability` / `confidence`
> are derived for callers that prefer the yes/no/uncertain + certainty
> view; `answer` thresholds are `1/3` and `2/3`.

---

## Quick start

```bash
git clone <this repo>
cd z-jev
pip install -e .                 # installs the z_jev package
pip install -e .[dev]            # adds pytest + ruff

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q        # 24 tests, all CPU
python3 -m z_jev.train --out checkpoints/tiny      # ~30s on CPU
python3 -m z_jev.infer_cli \
    --checkpoint checkpoints/tiny/model.pt \
    --request examples/spam_request.json          # one-off decision
python3 -m z_jev.serve --checkpoint checkpoints/tiny/model.pt
# then:
curl -s -H 'Content-Type: application/json' \
     -X POST http://127.0.0.1:8080/v1/evaluate \
     --data @examples/support_ticket_request.json | jq
```

A one-shot smoke script does all of the above in a single
`scripts/smoke.sh` invocation.

---

## Project layout

```
z_jev/
  __init__.py        # public API
  config.py          # ZJevConfig + TinyGLMConfig + GLM5Config
  backbone.py        # TinyGLM (decoder-only transformer) + HF GLM-5 adapter
  head.py            # NonAutoregressiveDecisionHead (Choice / Score / Noul)
  model.py           # ZJevModel = backbone + heads; question encoder
  protocol.py        # Jev-compatible dataclasses + (de)serialisation
  scorer.py          # softmax + margin-calibrated confidence math
  data.py            # synthetic spam / risk / noul datasets
  train.py           # CLI trainer (CPU-friendly)
  infer_cli.py       # CLI single-request inference
serve.py              # FastAPI server: POST /v1/evaluate, GET /healthz
lora.py / lora_train.py  # LoRA math + frozen-GLM-5 adapter training CLI
tests/               # pytest suite (protocol/heads/training/API/LoRA/hardening)
examples/            # request JSON, sample output, train_sample.jsonl
docs/DEPLOYMENT.md   # bare metal / Docker / compose / nginx / systemd
Dockerfile, docker-compose.yml, Makefile
scripts/smoke.sh     # lint + tests + train + inference + API smoke
checkpoints/         # gitignored training output (keep .gitkeep)
```

---

## Real GLM-5 LoRA training (method, not exercised on this machine)

The production adaptation path is: **freeze the full GLM-5 backbone,
inject LoRA adapters into its linear projections, and train the adapters
together with the non-autoregressive decision head**. `z_jev/lora.py`
implements LoRA directly (`W' = W + (alpha/r) B A`, `B` zero-init so an
untrained adapter is mathematically identical to the base model); on a
multi-GPU host `z_jev/lora_train.py` can instead use `peft`.

Data is JSONL — one labelled decision packet per line:

```json
{"state": "free prize click now", "answers": {
  "category": {"type": "choice", "label": "spam"},
  "risk": {"type": "score", "label": 2},
  "is_urgent": {"type": "noul", "label": true}}}
```

A 20+ row sample lives at `examples/train_sample.jsonl`.

```bash
# On a cluster that can host 744B-A40B (BF16 weights ~1.4 TB; plan for
# 8x80GB-class GPUs minimum, more with full logit/head optimizer states;
# 4-bit/8bit quantization lowers the weight footprint substantially):
z-jev-lora-train --backbone glm5 --model zai-org/GLM-5 \
    --train-file data/train.jsonl --val-file data/val.jsonl \
    --load-in-8bit --lora-rank 16 --lora-alpha 32 \
    --grad-accum 8 --steps 5000 --out checkpoints/glm5-lora

# Same pipeline end-to-end on this machine, tiny backbone + built-in LoRA:
z-jev-lora-train --backbone tiny \
    --train-file examples/train_sample.jsonl \
    --steps 40 --out checkpoints/lora-tiny
```

Checkpoints save the LoRA adapter and decision head separately (plus an
optional `--save-merged` form); point `z-jev-serve --checkpoint` at the
output directory to deploy. The GLM-5 commands have **not** been run on
real hardware here — they are the prescribed method for downstream
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for deployment
  (bare metal, Docker, compose, nginx, systemd, env vars, rollback).
- [`docs/DECISION_TRAINING_VALIDATION.md`](docs/DECISION_TRAINING_VALIDATION.md)
  for the full decision flow, training process, metrics and validation.

### Docker quick start

```bash
docker compose up -d --build          # serves on :8000, checkpoint via volume
ZJEV_API_KEY=s3cret ...               # optional bearer-token auth
```

---

## Math: confidence

For a discrete distribution `p_1, ..., p_K` summing to 1:

```
raw_top1  = max(p_i)
uniform   = 1 / K
confidence = (raw_top1 - uniform) / max(eps, 1 - uniform)
```

This goes from 0 (uniform) to 1 (fully peaked). For K=2 it collapses to
`|p_1 - p_2|`; for Noul the implementation reuses
`2 * |p - 0.5|`. The temperature parameter on the logits rescales both
probabilities and confidence together, so a more confident model gets
higher confidence automatically.

---

## Known limitations

* The `tiny` replica trains on synthetic bag-of-bytes data. It is meant
  to demonstrate that the architecture + protocol round-trips end-to-end
  on CPU. Real GLM-5 fine-tuning would yield dramatically better
  accuracy; this repo intentionally does not download the real weights.
* The adapter never *downloads* GLM-5 weights; it only validates
  `AutoConfig` and the model class exist.
* Memory budget is ~1.3 GB free; training uses `batch_size=4`,
  `max_seq_len=128` and a hidden_size of 128. Larger values OOM in our
  test runs.
* CPU-only torch in this environment; CUDA is not exercised by tests.

---

## License

MIT -- see [LICENSE](LICENSE).