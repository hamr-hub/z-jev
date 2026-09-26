# Z-Jev 项目研究报告

> 一份系统性梳理：项目定位、未来应用、可提升方向与实施路线图。
> 阅读对象：项目维护者、潜在贡献者、希望基于本项目二次开发的人。

---

## 1. 项目概述

### 1.1 一句话定位
Z-Jev 把 TypeSafe AI 的 [Jev 决策 API](https://docs.typesafe.ai/primitives/) 重新实现为**非自回归 (non-autoregressive) 解码器**：在 GLM-5 (744B-A40B MoE) 架构之上挂载专用决策头，一次前向并行输出带概率的类型化决策。

### 1.2 三大决策原语

| 原语 | 输入 | 输出 | 典型用途 |
|------|------|------|---------|
| **Choice** | 选项集 `{"spam": ..., "ham": ...}` | `{choice, probabilities, confidence}` | 分类、路由 |
| **Score** | 有序量表 `[low, med, high]` | `{score, legend, probabilities, confidence}` | 风险分级、情感打分 |
| **Noul** | 二元判定 | `{noul, answer, probability, confidence}` | yes/no/uncertain 判定 |

### 1.3 核心创新点

1. **非自回归**：每个问题的所有候选 logits 由专用决策头一次性并行输出；没有 token-by-token 解码、没有 LM head、没有幻觉表面。
2. **架构同构**：内置 `TinyGLM` 镜像 GLM-5 架构族 (decoder-only + RMSNorm + 位置编码)，可在 CPU 上真实训练验证。
3. **协议完整**：100% 兼容 Jev 上游 wire format (`{choice, probabilities, confidence}` 等字段名逐字一致)。
4. **校准置信度**：margin-calibrated top-1 公式 `(top1 - 1/K) / (1 - 1/K)`，Noul 退化为 `2|p - 0.5|`，数学定义清晰。
5. **可热插拔 backbone**：`mode="tiny" | "glm5"`，head 代码通用；切换 backbone 不需要重训 head。

### 1.4 工程现状

- 完整 Python 包 (8 个模块，约 60KB 源码)
- FastAPI 服务 (`POST /v1/evaluate` + `GET /healthz`)
- CLI 训练 (`python -m z_jev.train`)
- CLI 推理 (`python -m z_jev.infer_cli`)
- 24 个测试用例 (~30s CPU)
- 3 个合成数据集 (spam / risk / mixed)
- `transformers` 软依赖（无 HF 时仅 tiny 模式可用）
- 性能约束：~1.3 GB RAM，CPU only，无 GPU 集群

---

## 2. 未来应用方向

### 2.1 客服与支持系统（最有落地价值）
仓库里已经有一个 `support_ticket_request.json` 示例，未来可扩展为：
- **多问题并行评估**：一个工单同时给出"部门归属""紧急度""客户情绪""是否需升级"，由后端业务系统按决策自动分流。
- **多轮对话决策**：在对话每一轮评估"用户意图""情绪""是否需要转人工"，把对话状态机显式化。
- **SLA 风险预警**：实时评估在途工单 SLA 违约概率，提前触发升级流程。

### 2.2 内容审核与风控
- 短信 / 邮件 / 评论 spam 检测（已有合成数据集）。
- **多维度风险**：内容合规、隐私泄露、辱骂、欺诈、品牌声誉一次评估完。
- **分级审批**：自动决定是否需要人工复核，置信度低于阈值才转人工，节省 80%+ 人工成本。

### 2.3 业务流程自动化
- **申请审批**：信贷、保险理赔、招聘简历初筛、补助申请。
- **合同抽取**：合同类型、风险点、合规性、签约方风险一次评估。
- **医疗分诊**（需配套合规）：主诉 → 科室推荐 + 紧急程度。
- **法律文档**：案件分类、紧迫度、关联法条提示。

### 2.4 LLM 后处理与守卫（最有想象空间）
当前 LLM 应用的最大痛点是"自由文本 → 结构化字段"这一步极其脆弱（正则/JSON 解析易碎）。Z-Jev 可以：
- **结构化抽取校验**：LLM 生成 JSON → Z-Jev 逐字段校验置信度。
- **幻觉检测**：对 LLM 关键判断做置信度阈值检查，低置信度触发重生成。
- **一致性检查**：用 Noul 验证 LLM 不同问题的回答之间是否自洽。
- **Guardrail-as-a-service**：把 Z-Jev 部署为微服务，统一把关所有 LLM 输出。

### 2.5 实时生产决策
- **告警分级**：自动 P0/P1/P2，影响范围、SLA 风险一次评估完。
- **在线竞价策略选择**：实时决定出价、创意、人群定向。
- **运维事件自动派单**：日志/告警 → 责任人 + 优先级 + 是否需要值班长介入。

### 2.6 Agent / Tool 决策
- **Function calling 选择**：多候选工具中按 state 选最佳调用。
- **Plan step 选择**：多步骤任务中下一步选哪个 action。
- **Sub-agent 路由**：多 agent 系统中按 state 决定调用哪个 agent。
- **Tool call 参数验证**：参数类型 / 范围 / 业务约束的合规检查。

### 2.7 推荐与评价
- **多维评分**：商品评价同时给"性价比/质量/服务/物流"独立分数。
- **意图细粒度分类**：多轮对话中的复合意图。
- **A/B 实验实时分流**：按用户特征 + 内容 state 决定实验组。

### 2.8 多模态扩展（中期）
- 图文混合 spam 检测（广告图片 + 营销文案）。
- 音视频内容的合规 + 紧急度评估。
- 文档 OCR 后的多字段结构化抽取。

---

## 3. 可提升方向

### 3.1 架构层

| 改进项 | 当前 | 建议 | 收益 |
|--------|------|------|------|
| 位置编码 | sin/cos 绝对编码 | 真正的 **RoPE** (rotary) | 与 GLM-5 对齐，未来挂真实权重零成本 |
| 注意力 | 标准 MHA | **GQA** (grouped-query attn) | GLM-5 用 GQA；KV cache 减半 |
| 归一化 | RMSNorm | RMSNorm + **QK-norm** | 训练更稳，长序列不崩 |
| MoE | dense | **MoE routing** 占位 | 真实 GLM-5 是 744B-A40B MoE |
| Tokenizer | byte-level | **BPE/WordPiece** | 真实 GLM-5 用 BPE，迁移需对齐 vocab |
| Activation | GELU | **SwiGLU** | GLM-5 用 SwiGLU |
| 长序列 | max_len=128 硬限 | **YaRN/位置插值** | 真实决策场景常有数千 token 文档 |
| Flash Attn | 未用 | `scaled_dot_product_attention` 已 OK | 需确认 backend path |

### 3.2 训练层

| 改进项 | 建议 |
|--------|------|
| 真实数据 | 接入 SMS Spam Collection、AG News、IMDB、HelpSteer 等公开数据集 |
| 多任务加权 | Choice/Score/Noul loss 按频率加权，避免被高频任务主导 |
| 课程学习 | 从 2-选项 Choice 逐步到 16-选项 Choice |
| 标签平滑 | Noul 用 `[0.95, 0.05]` 而非硬标签，防过自信 |
| LoRA | 支持 HF backbone 的 LoRA fine-tuning（PEFT 集成） |
| 知识蒸馏 | 大模型 GLM-5 → tiny 蒸馏，提升 tiny 准确率 |
| 对抗训练 | FGSM/PGD 防御，提升鲁棒性 |
| 持续预训练 | 在 BPE tokenizer 上做 MLM 预训练 |
| 数据增广 | 同义改写、选项打乱、state 注入噪声 |
| 早停 + 检查点 | 按 val_loss 保存 best checkpoint |

### 3.3 API / 协议层

| 改进项 | 建议 |
|--------|------|
| 流式响应 | SSE/WebSocket 渐进式返回（问题 N 多时分批吐） |
| 异步批量 | `POST /v1/evaluate_batch` + 任务 ID 轮询 |
| 错误码 | 标准化 4xx/5xx + 机器可读 error code |
| 鉴权 | API Key / OAuth2 / JWT（避免裸奔） |
| 限流 | token bucket + sliding window |
| 可观测性 | Prometheus metrics + OpenTelemetry tracing |
| OpenAPI | 完整 schema 自动生成 |
| 决策追溯 | 返回每个 question 的中间向量/attention（用于解释与审计） |
| 多步决策 | decision chain（前一个决策影响后一个的 criteria） |
| 客户端 schema | 用 Pydantic / JSON Schema 校验 questions 合法性 |

### 3.4 评估层

| 改进项 | 建议 |
|--------|------|
| 校准指标 | **ECE** (Expected Calibration Error) + reliability diagram |
| Choice | accuracy / top-2 / macro-F1 / Cohen's κ |
| Score | MSE / MAE / Spearman / quadratic weighted κ |
| Noul | AUROC / F1 / Brier score / precision-recall |
| 鲁棒性 | 对抗扰动测试、OOD 检测、置信度-准确率一致性 |
| A/B 框架 | 在线分流 + 长期效果追踪 |
| Drift 检测 | 输入分布 / 置信度分布变化告警 |
| 主动学习 | 低置信度样本送人工标注 → retrain 闭环 |

### 3.5 部署层

| 改进项 | 建议 |
|--------|------|
| Docker | 多阶段构建（CPU / CUDA / torch+onnx） |
| ONNX | tiny 模型导出 ONNX，跨框架部署 |
| 量化 | INT8/INT4 量化，tiny 模型可压到 < 1MB |
| Triton | NVIDIA Triton Inference Server 集成 |
| KV cache | HF GLM-5 路径支持 prefix cache，长文档评估降本 |
| 投机解码 | tiny 模型预判 GLM-5 决策，命中则跳过 |
| gRPC | 高吞吐内部调用 |
| K8s | HPA / PDB / sidecar metrics |
| 多版本 | canary 灰度 + A/B 自动切流 |

### 3.6 决策原语扩展

| 新原语 | 描述 |
|--------|------|
| **Rank** | 对候选列表排序输出 |
| **Extract** | 抽取键值对（金额/日期/甲方/利率…） |
| **Compare** | 比较两个 state（A vs B 哪个更优） |
| **Cite** | 返回支持决策的 state span（可解释性） |
| **Cluster** | 把 state 归入 N 个 cluster |
| **Reason** | 带置信度的多步 CoT 输出 |
| **Decide** | 在多个候选动作中选一个并说明依据 |

### 3.7 工程化 / DX

| 改进项 | 建议 |
|--------|------|
| 类型检查 | `mypy --strict` |
| 格式化 | `ruff format` |
| CI/CD | GitHub Actions：lint + test + train + serve smoke |
| 文档 | Sphinx + Read the Docs |
| 监控 | Sentry / GlitchTip |
| 数据飞轮 | 主动学习闭环（详见 3.4） |
| Python SDK | `pip install z-jev-client` |
| JS/TS SDK | `npm install z-jev-client` |
| Jupyter | 内置教程 notebook |
| 实验平台 | W&B / MLflow 集成 |

### 3.8 测试层

| 改进项 | 建议 |
|--------|------|
| 属性测试 | `hypothesis` 库自动构造边界 |
| 模糊测试 | 随机 state/questions 输入 |
| 性能基准 | `pytest-benchmark` |
| 内存分析 | `memray` 抓内存峰值 |
| 覆盖率 | `pytest-cov` + codecov badge |
| E2E | 在有 GPU 的 CI runner 跑真实 GLM-5 |

### 3.9 代码层关键改进（短期可见收益）

读完后整理出的具体 issue：

1. **`head.py:loss()` 双重 Python 循环** — `for bi in range(b): for qi in range(q):` 是性能瓶颈，应当向量化（gather + scatter）。
2. **`head.py:_gather_logits()` 是 dead code** — 没人调用，可以删除。
3. **`head.py:outputs.sizes` 字段冗余** — sizes 总是等于 PRIMITIVE_MAX_OUT[type]，没有信息量，要么删掉要么真正按 question 动态 size。
4. **`model.py:collate_requests` 的 padding 浪费算力** — 用 PAD noul 填充缺失问题，但 decode 仍跑一遍 head；应该在 head.decode 时按 `valid_mask` 跳过。
5. **`protocol.py:QuestionScore.legend` 加权假设等距** — 当前用 legend 加权平均，但 legend 可能非等距（如 `[0, 5, 100]`），未做归一化。
6. **`protocol.py:NOUL_LOWER/UPPER` 硬编码** — 1/3 与 2/3 是阈值常量，应做成可配置（不同业务期望不同）。
7. **`model.py` 中 `max_question_len` 与 `QuestionEncoder.max_len` 硬编码不一致风险** — 96 重复出现在多处，应统一通过 config。
8. **`config.py` 缺校验** — `ZJevConfig(mode="glm5")` 时 `head_hidden_size` 应等于 `glm5.hidden_size`，但未校验。
9. **`backbone.py:HFGLM5Adapter` 没有验证 `trust_remote_code` 的安全性提示** — 真实 GLM-5 在 HF 上是 gated repo，缺少清晰引导。
10. **`train.py:evaluate()` 中 `max_batches` 是绝对值** — 应该按 n_val 比例采样，避免数据集大小变化时评估量失真。

### 3.10 文档层

| 改进项 | 建议 |
|--------|------|
| 架构图 | 更详细的 mermaid（含 GLM-5 路径与数据流） |
| 教程 | 真实数据集 fine-tune 完整教程 |
| FAQ | 如何选 tiny/GLM-5、内存预算、延迟基准 |
| 性能基准 | 不同 backbone 的吞吐量对比表 |
| 迁移指南 | 从传统规则系统 / LLM JSON 输出迁移到 Z-Jev |
| 协议 changelog | 版本演进记录 |
| 数学附录 | confidence、temperature、加权平均的严格数学定义 |

---

## 4. 实施路线图

### v0.2 — 工程化与稳定性（短期 1-2 周）
- 修 `head.py:loss()` 双重循环为向量化
- 修 `collate_requests` padding 浪费
- 删 `_gather_logits` dead code
- 加 `mypy --strict`
- 加 GitHub Actions CI（lint + test + train smoke）
- 加 ECE 等校准指标
- 加 Docker 镜像 + ONNX 导出
- 真实数据集支持（SMS Spam Collection）

### v0.3 — 真实 GLM-5 路径（中期 1-2 月）
- 真正的 RoPE 实现
- GQA 支持
- SwiGLU activation
- LoRA fine-tuning 脚本（PEFT 集成）
- HuggingFace Trainer 集成
- 模型卡（model card）
- 在有 GPU 的机器上跑通端到端 benchmark

### v0.4 — 决策原语扩展（中长期）
- 新增 Rank / Extract / Compare / Cite
- 协议向后兼容（schema versioning）
- 迁移学习工具链
- 业务场景模板库（客服/审核/合同…）

### v0.5 — 多模态 + Agent（长期）
- 图文混合输入
- Function calling 决策
- Plan step 选择
- 多 agent 路由
- 主动学习飞轮（生产数据回流 → 标注 → retrain）

### v1.0 — 生产级（远期）
- 高吞吐推理（Triton + KV cache + 投机解码）
- 完整可观测性（metrics + tracing + logging）
- 漂移检测 + 自动 retrain
- 多语言 SDK（Python / JS / Go / Rust）
- 完整文档站 + 教程 + 视频
- 协议 v2 演进

---

## 5. 风险与限制

- **GLM-5 真实权重无法在 1.3GB RAM 加载**：核心限制，只能在真实硬件上验证（仓库 README 已诚实声明）。
- **Tiny 训练在合成数据**：合成数据带"spam" / "ham"显式标记词，迁移到真实场景会失效。
- **校准公式简单**：复杂场景可能需要温度标定 (Platt scaling) 或 Dirichlet 校准。
- **协议锁定**：上游 Jev 协议若演进需跟进；建议协议版本号化便于兼容。
- **PyTorch 锁**：目前仅 PyTorch，ONNX/其它后端需扩展。
- **8GB RAM 内存预算脆弱**：实测 OOM 边界紧，max_len 增到 256 会爆。
- **CUDA 路径未覆盖**：所有测试仅 CPU，CUDA 路径可能存在 shape/dtype bug 未发现。

---

## 6. 总结

Z-Jev 当前的真正价值不在 tiny 模型本身（那是 demo），而在于：

1. **协议对齐** — 100% 兼容 TypeSafe Jev spec，便于接入更广泛的生态系统（OpenRouter、LiteLLM 等都支持 Jev）。
2. **架构对齐** — 与 GLM-5 同构，未来在真实硬件上可直接挂载，无需重写 head。
3. **应用空间** — 客服、内容审核、业务流程、Agent 决策、LLM 守卫均有广阔落地。
4. **可演进性** — 当前实现预留了清晰的扩展点（新原语、新 backbone、新数据集）。

**下一步最关键的两件事**：
1. 修 `head.py:loss()` 双重循环为向量化（性能立刻翻倍）
2. 在有 GPU 的机器上跑通真实 GLM-5 + 决策头的端到端验证（证明架构假设）

**长期最具想象力的方向**：
- LLM 输出守卫 / 结构化抽取校验（替代脆弱的 JSON 解析）
- Agent function-calling / plan-step 决策（与 agent 框架深度集成）
- 多模态决策（图文音视频统一接入）

---

*文档生成时间：2026-09-26*
*基于 commit：5012f5f*