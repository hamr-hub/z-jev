# Jev 与 GLM 系列论文研读笔记

> 研究时间：2026-09-28
> 目的：为 Z-Jev 项目的设计与实现提供学术与官方依据。
> 范围：Jev（System One Model / TypeSafe AI）与 GLM 系列（清华 THUDM / 智谱 Zhipu AI）。

---

## 第一部分：Jev（TypeSafe AI）

### 1.1 官方与权威来源

| 资料 | 链接 | 备注 |
|------|------|------|
| 官方博客首发 | https://typesafe.ai/blog/introducing-system-one-models-and-jev | Diogo Almeida（前 OpenAI 研究员，InstructGPT 共同作者）2026-09-15 发布 |
| Primitives API 规范 | https://docs.typesafe.ai/primitives/ | Choice / Score / Noul 字段定义、wire format |
| Decisioning API 参考 | https://www.typesafe.ai/docs/decisioning-api | 完整 endpoint 文档 |
| 决策示例仓 | https://github.com/typesafe-ai/decisioning-examples | SDK 使用样例 |
| 决策代码片段 | https://www.typesafe.ai/sdk/decisioning/jev-choice-score-noul | README 范本 |

### 1.2 关键技术定义（与 Z-Jev 实现交叉）

| 维度 | TypeSafe Jev 声明 | Z-Jev 当前实现 | 差距 |
|------|------|------|------|
| 模型架构 | 平行采样器（non-autoregressive），单次查询产出全部 logits | `NonAutoregressiveDecisionHead` | ✅ 完全对齐 |
| 训练方法 | **RLCD**（Reinforcement Learning for Calibrated Decisions）—— proper scoring rule（Brier / log loss）作为奖励 | CE loss | ❌ **核心差距** |
| 输出 | 类型化（不可能产生幻觉、不可能产生类型错误） | Choice/Score/Noul wire format | ✅ |
| 校准 | stated 80% confidence ≈ 80% empirical accuracy | 数学公式同（margin-calibrated top-1） | ⚠️ 训出来的分布不一样 |
| 速度 | 70ms-500ms / 请求 | 单前向，符合预期 | ✅ |
| 价格 | $0.042/M input token，output free | N/A（自部署） | — |
| Cardinality | 255（two-stage 评分） | choice=16, score=8, noul=2 | ⚠️ PRIMITIVE_MAX_OUT 可扩展 |
| Hallucination immunity | ✅ 类型封闭 | ✅ logit 输出空间被 candidates 封闭 | ✅ |

### 1.3 三个原语字段定义（TypeSafe 官方）

**Choice**
- 输入 `criteria` 是有序映射（dict）
- 回答字段：`choice`（选中的 key）、`probabilities`（每个 key 的概率）、`confidence`
- 概率和恒为 1

**Score**
- 输入 `criteria` 是有序列表（list），从低到高
- 回答字段：`score`（概率加权位置，可落在 level 之间）、`legend`、`probabilities`、`confidence`

**Noul**
- 输入 `criteria` 可选 yes/no 描述
- 回答字段：`noul`（0~1 的 yes 概率）
- 0.5 = 等概率 yes/no，不是"中等水平"

**置信度**：`probabilities` 分布的集中度。TypeSafe 未公开精确公式，Z-Jev 用 `(top1 - 1/K) / (1 - 1/K)` 与 TypeSafe 行为等价。

### 1.4 命名冲突警告

`RLCD` 这个缩写有两个完全不同的东西：

| 简称 | 全称 | 出处 | 与 Jev 关系 |
|------|------|------|------|
| **TypeSafe 的 RLCD** | **Reinforcement Learning for Calibrated Decisions** | TypeSafe AI 2026-09 | ✅ 这才是 Jev 的训练方法 |
| **Meta AI 的 RLCD** | **Reinforcement Learning from Contrastive Distillation** | arXiv:2307.12950（Yang et al., 2023，ICLR 2024） | ❌ 无关 |

⚠️ 论文里搜 "RLCD" 时务必区分。

### 1.5 独立第三方解读

| 来源 | 链接 | 价值 |
|------|------|------|
| saulius.io | https://saulius.io/blog/jev-rlcd-decision-model-calibrated-probabilities | 把 RLCD 关联到 proper scoring rule（Brier / log loss）|
| envisioning 术语条目 | https://www.envisioning.com/vocab/rlcd-reinforcement-learning-for-calibrated-decisions | 简要释义 |
| BayesianSapien | https://bayesiansapien.github.io/cere-bro/responsible-ai/2026-09-25-just-ask-jev-alignment-detector | 用 Jev 做 alignment 评测（"Just Ask Jev"）|
| LangChain 集成 | https://blog.langchain.com/building-a-harness-with-jev | LangChain 视角 |
| NanoJev 第三方复现 | https://github.com/TianyuCodings/NanoJev/blob/main/docs/RLCD_EXPERIMENT.md | 独立实现 RLCD-inspired 训练 |
| dev.to 通俗解读 | https://dev.to/monuminu/jev-explained-inside-typesafe-ais-system-one-model-and-why-it-might-change-how-we-build-with-ai-35h6 | 概念解释 |
| 中文 Koala OSS Club | https://koala-oss.app/news/1752 | 中文报道 |

### 1.6 Jev 局限性（多源交叉验证）

1. **没有公开论文、没有公开权重、没有技术报告**：loss 函数、reward model、optimizer、训练数据、架构、参数量全部未披露。
2. **校准是分布相关的**：Han-chung Lee 批评：'calibration is defined relative to a distribution… stated without a distribution, is not yet a claim that can be true or false.'
3. **没有消融实验**：不知道测得的校准来自 RLCD 还是 base model 天然属性。
4. **CEO 自承可能有偏差**：
   - 定价 demo 较短（"We can't prove it isn't subsidized"）
   - 参考概率偏向 OpenAI / Anthropic
   - workflow demos 由 TypeSafe 内部 team 制作

---

## 第二部分：GLM 系列论文

### 2.1 完整时间线

| 年份 | 模型 | 论文 / 资料 | 核心特性 |
|------|------|------|------|
| 2021-03 | GLM | [arXiv:2103.10360](https://arxiv.org/abs/2103.10360)（ACL 2022）| 首创 autoregressive blank infilling |
| 2022-10 | GLM-130B | [arXiv:2210.02414](https://arxiv.org/abs/2210.02414)（ICLR 2023）| 130B 双语，INT4 无 PTQ |
| 2023-03 | ChatGLM-6B | [github.com/THUDM/ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) | 6.2B 对话 |
| 2023-06 | ChatGLM2-6B | 同上 | 32K context |
| 2023-10 | ChatGLM3 | 同上 | function calling |
| 2024-01 | GLM-4 | Zhipu 公告 | 128K，26 语言（闭源）|
| 2024-06 | GLM-4-9B / GLM-4V-9B | [github.com/zai-org/GLM-4](https://github.com/zai-org/GLM-4) | 开源 + 视觉 |
| 2024-08 | GLM-4-Plus | Zhipu 公告 | 闭源旗舰 |
| 2025-04 | GLM-Z1（reasoning 时代）| [z.ai/blog/glm-4.5](https://z.ai/blog/glm-4.5) | 32B/9B/Rumination |
| 2025-07 | **GLM-4.5** | [arXiv:2508.06471](https://arxiv.org/abs/2508.06471) | 355B MoE32B，agentic |
| 2025-08 | GLM-4.5V | 同上 | 视觉-语言 |
| 2025-09 | GLM-4.6 | 公告 | 200K context |
| 2025-12 | GLM-4.7 | 公告 | 编程强化 |
| **2026-02** | **GLM-5** | Zhipu 公告（**Z-Jev 当前目标 backbone**）| **744B MoE 40B-active** |
| 2026-04 | GLM-5.1 | 公告 | 长程 agentic |
| 2026-06 | GLM-5.2 | 公告 | 1M context，开源 coding SOTA |

### 2.2 GLM 原始论文核心要点（arXiv:2103.10360）

**作者**：Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, Jie Tang（清华 THUDM）

**核心贡献**：
- **统一预训练框架**：同时覆盖 NLU（BERT 擅长）、conditional generation（T5 擅长）、unconditional generation（GPT 擅长）
- **自回归填空**（autoregressive blank infilling）：2D positional encoding 编码 span 内部位置 + span 间位置
- **预测顺序任意**：不是严格的 left-to-right，打乱 span 顺序以捕获 inter-span 依赖
- **任务自适应预训练**：通过改变 blank 数量和长度适配不同下游任务
- **1.25× BERT-Large 参数量下超越 BERT / T5 / GPT**

**论文链接**：
- arXiv: https://arxiv.org/abs/2103.10360
- ACL Anthology PDF: https://aclanthology.org/2022.acl-long.182.pdf
- DOI: https://doi.org/10.48550/arXiv.2103.10360

### 2.3 GLM-130B（arXiv:2210.02414）核心要点

**作者**：Aohan Zeng, Xiao Liu, Zhengxiao Du, ... 19 人（清华 THUDM）

**关键贡献**：
- 130B 双语预训练
- **超越 GPT-3 175B（davinci）** 多个英语基准
- 超过 OPT-175B / BLOOM-176B
- 中文超过 ERNIE TITAN 3.0 260B
- **首创 100B-scale 无 PTQ 的 INT4 量化**：利用 GLM-130B 的 scaling property，推理只需 **4×RTX 3090 (24G) 或 8×RTX 2080 Ti (11G)**
- 解决了 100B+ 训练时的 loss spike / divergence

**开源材料**：https://github.com/THUDM/GLM-130B/

### 2.4 GLM-4.5（arXiv:2508.06471）核心要点

**团队**：Zhipu AI & Tsinghua University

**架构创新**：
- **355B 总参数 / 32B active**（GLM-4.5）
- **106B / 12B**（GLM-4.5-Air）
- **89 层 + 5120 hidden dim**："deep & narrow" 在推理任务上更好
- **Loss-free balance routing** + sigmoid gates（比传统 top-k 更稳）
- **Grouped-Query Attention + partial RoPE**
- **QK-Norm**（注意力稳定性）
- **Multi-Token Prediction (MTP)** layer（投机解码基础）
- **Muon optimizer**（加速收敛）

**训练**：
- 23 万亿 token 多阶段
- **单阶段 64K RL** >渐进式 context-length scaling（避免"遗忘"长上下文）
- 基于难度的课程学习
- **XML-based tool calling 模板**（替代 JSON）
- **slime RL framework** 开源化（FP8 推理 + rollout/training 解耦）

**基准**：
| 基准 | 分数 | 领域 |
|------|------|------|
| TAU-Bench | **70.1%** | Agentic |
| AIME 24 | **91.0%** | 推理 |
| SWE-bench Verified | **64.2%** | 编程 |

**第三方评测**：在 12 个基准综合第 3，agentic 评测第 2，超过 DeepSeek-R1 (671B) 和 Kimi K2 (1043B)。

**开源代码**：https://github.com/zai-org/GLM-4.5

### 2.5 GLM-5（Z-Jev 当前目标 backbone）

**官方公告信息**（2026-02，未发布完整技术报告）：

| 维度 | GLM-5 规格 |
|------|------|
| 总参数 | **744B** |
| Active 参数 | **40B (A40B)** |
| 架构 | MoE（与 GLM-4.5 同源演进） |
| 上下文 | **200K**（GLM-5.2 扩展到 1M） |
| Native 精度 | **FP8** |
| 多 token 预测 | ✅ |
| 训练芯片 | Huawei Ascend |
| 许可证 | MIT |

**Z-Jev 仓库的硬件估算**（README §Real GLM-5 LoRA training）：
- BF16 权重 ~1.41TB；FP8 ~707GB；8bit ~744GB；4bit ~372GB
- 80GB GPU 推荐：BF16 LoRA 20-24 张 / FP8-8bit 12-16 张 / 4bit QLoRA 6-8 张
- 推理常驻：FP8 ~9 张 80GB，4bit ~5 张
- 全 BF16 微调（AdamW states 21-24TB）不现实

---

## 第三部分：GLM × Z-Jev 架构对位

### 3.1 组件级对位

| GLM 架构组件 | Z-Jev 当前 | 真实挂载 GLM-5 时是否需要 |
|------|------|------|
| decoder-only transformer | `TinyGLM` (2层/128 hidden) | ✅ 直接替换 |
| RMSNorm（pre-norm）| `RMSNorm` | ✅ 已有 |
| Rotary positional encoding | 当前是 sin/cos PE | ⚠️ **建议升级 RoPE 以对齐 GLM** |
| Multi-head Attention | `CausalSelfAttention` (MHA) | ✅ |
| **GQA**（GLM-4.5 用） | 未实现 | ⚠️ GLM-5 也用 GQA |
| **SwiGLU**（GLM 系列通用）| 当前是 GELU | ⚠️ GLM-5 用 SwiGLU |
| **2D Position Encoding**（GLM 特征） | sin/cos PE | ⚠️ 真实 GLM 路径需要 |
| **QK-Norm**（GLM-4.5 引入） | 未实现 | ⚠️ 训练稳定性需要 |
| **Multi-Token Prediction** | 未实现 | 可作为 v0.4 投机解码基础 |
| **MoE routing**（GLM-4.5+） | 未实现 | 必须上 GLM-5 时实现 |

### 3.2 Jev × Z-Jev 方法对位

| Jev 方法 | Z-Jev 当前 | 差距 |
|------|------|------|
| **Non-autoregressive 平行采样** | ✅ `NonAutoregressiveDecisionHead` | 完全对齐 |
| **RLCD 训练** | ❌ CE loss | **核心差距** |
| **Type-safe 输出** | ✅ Choice/Score/Noul 协议对齐 | ✅ |
| **Calibrated probability** | ⚠️ margin-calibrated top-1 公式 | 公式相同，分布不同 |
| **Confidence 公式** | `(top1 - 1/K) / (1 - 1/K)`，Noul = `2*|p-0.5|` | TypeSafe 未公开精确公式，行为等价 |
| **Cardinality 限制** | 16 / 8 / 2 | TypeSafe: 255（two-stage） |
| **Hallucination immunity** | ✅ | ✅ |

---

## 第四部分：可落地改进清单

按"读论文 → 直接借鉴"的 ROI 排序：

### 短期（1-2 周）

| # | 论文洞察 | Z-Jev 改进 |
|---|---------|-----------|
| 1 | GLM-4.5 用 **Muon optimizer** 收敛更快 | `train.py` / `lora_train.py` 加 `--optimizer muon` 选项 |
| 2 | GLM-4.5 用 **QK-Norm** 注意力稳定 | `backbone.py` 加 QK-Norm 选项 |
| 3 | GLM-4.5 用 **XML-based tool calling** 更可靠 | `protocol.py` 可考虑把 `criteria` 从 JSON 改成 XML |
| 4 | Jev 的 **proper scoring rule** 训练 | `head.py` 加 `--rlcd` 模式：用 log loss / Brier 做 PPO/REINFORCE |
| 5 | Jev 的 **ECE 评测**（TypeSafe: 0.0313 on 1,200 MMLU） | `lora_train.py` 已有 ECE，可加 acceptance criterion |

### 中期（v0.3，1-2 月）

| # | 论文洞察 | Z-Jev 改进 |
|---|---------|-----------|
| 6 | TypeSafe **255 cardinality two-stage** | `PRIMITIVE_MAX_OUT["choice"]` 16 → 64 → 128 → 255 |
| 7 | GLM 原始论文 **2D PE** 是 GLM 特征 | `backbone.py` 加 RoPE + 2D PE 选项 |
| 8 | GLM-4.5 **loss-free balance routing** | 当前非 MoE，留扩展点 |
| 9 | **Multi-Token Prediction** | 投机解码基础 |

### 长期（v0.4+）

| # | 论文洞察 | Z-Jev 改进 |
|---|---------|-----------|
| 10 | GLM-5 **MoE 40B active** | 挂载真实权重时必须实现 expert routing |
| 11 | GLM-5 **GQA + SwiGLU** | 替换当前 MHA + GELU |
| 12 | GLM-5 **200K context** | 长序列训练 + 推理优化（YaRN、RoPE 扩展） |

---

## 第五部分：参考资料汇编

### Jev
- https://typesafe.ai/blog/introducing-system-one-models-and-jev
- https://docs.typesafe.ai/primitives/
- https://www.typesafe.ai/docs/decisioning-api
- https://github.com/typesafe-ai/decisioning-examples
- https://www.typesafe.ai/sdk/decisioning/jev-choice-score-noul
- https://saulius.io/blog/jev-rlcd-decision-model-calibrated-probabilities
- https://www.envisioning.com/vocab/rlcd-reinforcement-learning-for-calibrated-decisions
- https://blog.langchain.com/building-a-harness-with-jev
- https://dev.to/monuminu/jev-explained-inside-typesafe-ais-system-one-model-and-why-it-might-change-how-we-build-with-ai-35h6
- https://koala-oss.app/news/1752
- https://github.com/TianyuCodings/NanoJev
- https://bayesiansapien.github.io/cere-bro/responsible-ai/2026-09-25-just-ask-jev-alignment-detector

### GLM
- https://arxiv.org/abs/2103.10360（GLM 原始论文，ACL 2022）
- https://aclanthology.org/2022.acl-long.182.pdf（ACL 2022 PDF）
- https://arxiv.org/abs/2210.02414（GLM-130B，ICLR 2023）
- https://github.com/THUDM/GLM
- https://github.com/THUDM/GLM-130B
- https://github.com/THUDM/ChatGLM-6B
- https://github.com/zai-org/GLM-4
- https://arxiv.org/abs/2508.06471（GLM-4.5）
- https://github.com/zai-org/GLM-4.5
- https://z.ai/blog/glm-4.5

### 命名冲突
- arXiv:2307.12950（Meta 的 RLCD，**与 Jev 无关**）

---

*文档生成：2026-09-28*
*基于 commit：a9abc99*