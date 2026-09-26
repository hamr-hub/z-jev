# Z-Jev

> **Non-autoregressive decision heads on top of GLM-5** —
> the Jev `Choice` / `Score` / `Noul` primitives as a single forward pass.

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
  serve.py           # FastAPI server: POST /v1/evaluate, GET /healthz
tests/               # pytest suite (24 tests, ~30s on CPU)
examples/            # example request JSON files + captured sample_output.json
scripts/smoke.sh     # lint + tests + train + inference + API smoke
checkpoints/         # gitignored training output (keep .gitkeep)
```

---

## Real GLM-5 (what would change with the real weights)

GLM-5 (744B-A40B MoE, BF16 ≈ 1.4 TB) cannot be loaded on this machine. The
adapter in `z_jev/backbone.py` is the only piece that ever touches the
HF model:

```python
from transformers import AutoConfig, AutoModel
cfg = AutoConfig.from_pretrained("zai-org/GLM-5", trust_remote_code=True)
model = AutoModel.from_pretrained(
    "zai-org/GLM-5", trust_remote_code=True, torch_dtype=torch.bfloat16
)
# Then attach the decision heads:
from z_jev import ZJevConfig, NonAutoregressiveDecisionHead
zj = ZJevConfig(mode="glm5", glm5={"hidden_size": cfg.hidden_size, ...})
head = NonAutoregressiveDecisionHead(zj)
state_vec = model(input_ids).last_hidden_state[:, -1, :]   # or pool
logits = head(state_vec, question_vecs, types)
```

On a machine that *can* host GLM-5, the recommended path is
**head-only fine-tuning**: freeze the backbone (LoRA optional), train
just the head + state projection for a few thousand steps on labelled
decisions. The state-projection layer in `ZJevModel` is what bridges
GLM-5's `hidden_size` to the head's fixed dim, so swapping backbones
does not require retraining the head.

The `transformers` package is a soft dependency: if it is not installed,
`GLM5Backbone(mode="glm5")` raises an informative `ImportError`
explaining how to install `z-jev[hf]`. The `tiny` mode is unaffected.

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