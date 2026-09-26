# Z-Jev 决策、训练与验证说明书

> 本文把 Z-Jev 的**决策过程、训练过程、指标与验证过程**一次讲清。
> 所有数学定义均与代码一一对应，可直接按图索骥到具体文件。
> 适用代码版本：phase 2（含 LoRA 与 API 加固）。

---

## 1. 总体定位

Z-Jev 是一个 **System 1（快思考）决策模型**：输入一个状态 `state` 和一组预定义问题 `questions`，**不生成任何文本**，在一次前向中并行返回每个问题的类型化决策及其概率分布。

与传统 "LLM + JSON mode + 解析重试" 的区别：

| 维度 | LLM 生成式做法 | Z-Jev |
| --- | --- | --- |
| 输出 | 逐 token 自回归文本 | 一次前向并行 logits |
| 幻觉/格式 | 可能跑偏、JSON 非法 | 输出空间被问题候选封闭，无自由文本 |
| 延迟 | 随输出长度增长 | 恒定（单前向） |
| 机器消费 | 需解析、需重试 | 直接读概率/标签 |

两个骨干：`tiny`（同构字节级 decoder-only，CPU 可训，本仓库真实跑的就是它）与 `glm5`（`zai-org/GLM-5` 744B-A40B，仅在多卡集群可用，代码接口对齐）。

---

## 2. 决策过程（推理数据流）

### 2.1 输入协议

```jsonc
{
  "state": "free prize click urgent verify password now",
  "questions": {
    "category":   {"type": "choice", "instructions": "...",
                    "criteria": {"spam": "...", "ham": "..."}},
    "risk":       {"type": "score", "instructions": "...",
                    "criteria": ["low", "medium", "high"]},
    "is_urgent":  {"type": "noul", "instructions": "..."}
  }
}
```

一个请求可含**多个问题**，问题类型在 `z_jev/protocol.py` 中定义为 `QuestionChoice / QuestionScore / QuestionNoul`。

### 2.2 逐步流程

```
state 文本 ──字节分词──► [backbone 一次前向] ──► state_vec（固定维度，所有问题共享）
问题文本+候选标签 ──embedding mean-pool──► question_vec（×0.1）
问题类型 ──查表──► type_emb
                  │
                  ▼
        拼接 [state_vec, type_emb, 0.1·question_vec]
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
  Choice MLP  Score MLP  Noul MLP     （同一次前向，并行）
   N logits    K logits    2 logits
        │         │         │
        ▼         ▼         ▼
  带温度 softmax（在问题自身候选内，概率和恒为 1）
        │         │         │
        ▼         ▼         ▼
  argmax 选项  概率加权期望  阈值判定 yes/no/uncertain
```

**关键点**

1. **骨干只算一次**：state 表征被所有问题复用，这是低延迟低成本的核心（`z_jev/model.py`）。
2. **问题向量乘 0.1**：早期版本问题向量梯度过强导致模型塌缩为恒定预测；缩小后保证 state 信号主导决策（`head.py` 的 `question_scale`）。
3. **非自回归**：三个独立 MLP 头在同一次前向给出全部 logits，没有语言模型头、没有解码循环（`head.py`）。

### 2.3 三类决策的产生规则

- **Choice**：对 N 个候选 logits softmax，`choice = argmax`，返回每项 `probabilities`。
- **Score**：返回**概率加权期望** `score = Σᵢ pᵢ · levelᵢ`（保留"不确定但偏高档"的信息，不硬选档），同时返回每档概率与 legend。
- **Noul**：取 `p = P(yes)`，按阈值判定：
  - `p ≥ NOUL_UPPER`（默认 0.7）→ `yes`
  - `p ≤ NOUL_LOWER`（默认 0.3）→ `no`
  - 中间 → `uncertain`
  - "不确定"不是第三个输出头，而是概率落入中间区间的结论（`protocol.py` 的 `AnswerNoul.from_noul`）。

### 2.4 概率与置信度

**带温度 softmax**（`scorer.py`）：

```
zᵢ = logitᵢ / τ ;   pᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)
```

τ>1 分布变平（更保守），τ<1 分布变尖锐。

**置信度 confidence**（边际校准，统一公式）：

```
top1 = max(pᵢ);  uniform = 1/K
confidence = max(0, (top1 − uniform) / (1 − uniform))
```

- 0 = 分布均匀（毫无把握）；1 = 分布完全集中。
- K=2 时退化为 `|p₁ − p₂|`；Noul 即 `2|p − 0.5|`。
- 温度在 logits 层调节，概率与 confidence 被一起校准。

### 2.5 输出示例

见 `examples/sample_output.json`。每个 answer 均带概率，Choice/Score/Noul 字段完整。

---

## 3. 训练过程

### 3.1 数据

- **合成数据**（tiny 路径，`z_jev/data.py`）：垃圾短信（spam/ham）、风险打分、紧急判定，按词袋模板生成，供 CPU 快速训练。
- **JSONL 数据**（LoRA/真实路径，`z_jev/lora_train.py`）：每行一个带标注的决策包：

```json
{"state": "free prize click now", "answers": {
  "category": {"type": "choice", "label": "spam"},
  "risk":     {"type": "score", "label": 2},
  "is_urgent":{"type": "noul", "label": true}}}
```

示例：`examples/train_sample.jsonl`（≥20 行）。

### 3.2 训练目标

每个问题对其对应头的输出做**交叉熵**（Score 同样按档位 CE；Noul 为二分类 CE），batch 内对有效标注问题取平均：

```
L = (1/M) Σ_(valid questions) CE(softmax(logits_q / τ), target_q)
```

无标注的问题（target < 0）被 mask，不计入。实现见 `head.py` 的 `loss()`。

### 3.3 两条训练路径

**A. 决策头训练（`z_jev/train.py`）**

- tiny 骨干 + 三个决策头，AdamW，CPU 友好（默认 batch=4、max_len=96、2 层/128 hidden）。
- 输出训练 history 与 checkpoint（模型权重 + config）。

**B. 冻结 GLM-5 + LoRA（`z_jev/lora_train.py`，生产路径）**

- 冻结全部骨干参数；向线性层注入 LoRA adapter，仅训练 adapter + 决策头。
- LoRA 数学（`z_jev/lora.py`）：

```
W′ = W + (α/r) · B A,   A ∈ R^(r×in), B ∈ R^(out×r)
B 零初始化 ⇒ 初始时 W′ = W（未训练 adapter 与基模型严格等价）
```

- `--backbone glm5`：transformers + peft 软依赖，支持 `--model`、`device_map`、`--load-in-8bit/4bit`、梯度累积、混合精度、断点续训；adapter 与决策头分别保存，可选 `--save-merged` 合并。
- `--backbone tiny`：用内置低秩实现，在本机 CPU 真实跑通同样的保存/加载/推理流程。

### 3.4 部署训练产物

`z-jev-serve --checkpoint <out>`（或环境变量 `ZJEV_CHECKPOINT`）加载 checkpoint 对外提供 `/v1/evaluate`。LoRA adapter 可合并后部署以消除额外算子。

---

## 4. 指标与验证过程

### 4.1 指标定义

- **Accuracy（按原语分别统计）**：argmax 预测档位与标签一致的比例，分为 `choice / score / noul`。
- **Loss**：验证集平均交叉熵。
- **Score 期望值**：除硬准确率外，概率加权期望反映评分校准质量。
- **ECE（Expected Calibration Error）**：LoRA 验证中按 confidence 分 5 桶，比较桶内平均准确率与平均置信度：

```
ECE = Σ_b (n_b / n) · |acc_b − conf_b|
```

ECE 越小代表置信度与真实命中率越一致（`lora_train.py` 的验证函数）。
- **概率合法性**：每个问题概率和必须为 1（softmax 定义保证，并有测试守护）。

### 4.2 验证分层

验证分三层，越往下越接近端到端：

**① 单元/特性测试（`tests/`）**，关键守护点：

- 协议：请求/响应序列化、概率和为 1、Noul 阈值常量。
- 头与模型：输出形状、概率和、**批量并行结果与逐问题单算一致**、checkpoint 往返预测一致、异构问题数 collate。
- LoRA：**零初始化时前向与基模型严格相等**、merge/unmerge 往返、只训练 adapter 参数、合并后推理字节一致、loss 下降、checkpoint 可重载。
- API：healthz/readyz、Jev 形状输出、503 无 checkpoint、401 鉴权、X-Request-ID、统一错误包、请求/问题/候选上限、日志截断。
- 训练：真实短训练中 **loss 下降** 且准确率高于随机基线。

**② smoke 端到端（`scripts/smoke.sh`）**：ruff → pytest → 真实训练 tiny → CLI 推理 → 启动服务并 **live curl** `/v1/evaluate`，一次请求覆盖三类原语。

**③ CI（`.github/workflows/ci.yml`）**：push/PR 上在干净环境复跑 lint、tests、smoke，并构建 Docker 镜像，保证方案对下游可复现。

### 4.3 已知验证边界（诚实声明）

- 744B 真实权重路径**未在真机执行**：glm5 模式的命令是既定方法，需多卡集群（BF16 权重约 1.4TB，建议 8×80GB 级 GPU 起步；4/8bit 量化可降 footprint）。
- tiny 训练基于合成词袋数据，用于证明架构与协议在 CPU 上端到端闭合，不代表真实业务精度。
- 当前所有实跑均为 CPU；CUDA 路径可能存在未暴露的 shape/dtype 问题。
- 校准为温度 + 边际公式；复杂分布可能需要 Platt/Dirichlet 再校准。

---

## 5. 文件索引

| 主题 | 文件 |
| --- | --- |
| 协议/数据类 | `z_jev/protocol.py` |
| 骨干（GLM-5 适配 + TinyGLM） | `z_jev/backbone.py` |
| 非自回归决策头 | `z_jev/head.py` |
| 概率/置信度/温度 | `z_jev/scorer.py` |
| 模型装配 | `z_jev/model.py` |
| 决策头训练 | `z_jev/train.py` |
| LoRA 数学 / 冻结训练 | `z_jev/lora.py`, `z_jev/lora_train.py` |
| 服务与加固 | `z_jev/serve.py` |
| 测试 | `tests/` |
| 端到端冒烟 | `scripts/smoke.sh` |
| 部署方法 | `docs/DEPLOYMENT.md` |
