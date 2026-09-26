# Z-Jev 构建任务书（给 Claude Code）

## 目标
构建 **Z-Jev**：将 Jev（TypeSafe AI System One）的决策模型理论，移植到智谱 GLM-5 开源基座架构上——在 GLM-5 模型之上增加**非自回归决策头（non-autoregressive decision head）**，一次前向直接输出带概率的类型化决策，而非逐 token 生成文本。

## 硬性现实约束（必须在 README 如实说明，禁止夸大）
- GLM-5 真实权重为 744B-A40B MoE（BF16 约 1.4TB+），本机（Jetson Orin Nano, 8GB RAM, 无 GPU 集群）**无法加载或训练真实权重**。
- 因此工程必须做到两点且都真实可运行：
  1. **决策头与 GLM-5 架构接口对齐**：按 zai-org/GLM-5 的 HuggingFace 配置/模型类接口实现 adapter，能读取 GLM-5 的 hidden states 维度与 config（用 `transformers` 的 AutoConfig 接口；不实际下载权重），文档说明挂真实权重的方法。
  2. **tiny-glm 配置**：提供一个结构同构（同 GLM-5 架构家族：decoder-only transformer + 对应 norm/位置编码）的极小可训练配置（如 2 层、128 hidden、4 heads），在本机 CPU 上**真实完成训练→评估→推理**。

## Jev 决策理论（需自行联网调研核实，参考来源）
- 官方博客 https://typesafe.ai/blog/introducing-system-one-models-and-jev
- 原语文档 https://openrouter.ai/docs/guides/community/jev 、LiteLLM 集成文档
- 三种决策原语必须全部实现：
  - **Choice**：从候选选项中选一个，返回 {choice, probabilities(每项), confidence}
  - **Score**：有序量表打分，返回 {score(概率加权位置), probabilities(每档), confidence}
  - **Noul**（yes/no/uncertain 的布尔判定）：返回 {answer, probability, confidence}
- 输入协议：`state`（任意文本/结构化状态）+ `questions`（一个请求内多个问题，**并行一次前向批量求值**）。
- 非自回归：每个问题的各候选 logits 由决策头一次性并行输出，softmax 得概率分布；confidence 需有明确定义（如 top1 概率经温度/边际校准），并在文档中写清数学定义。
- System 1（快决策）定位：低延迟、无自由文本、无幻觉、机器可直接消费。

## 必须交付的工程结构（Python，可微调但需完整）
```
z_jev/
  __init__.py
  config.py          # ZJevConfig：含 GLM-5 744B 对齐配置与 tiny 配置
  backbone.py        # GLM5Backbone adapter：优先用 transformers AutoModel/AutoConfig
                     #   无网络/无权重时回退到内置同构 TinyGLM（PyTorch 手写 decoder-only）
  head.py            # NonAutoregressiveDecisionHead：Choice/Score/Noul 三个输出头
  model.py           # ZJevModel = backbone + heads；一次前向批量回答多问题
  protocol.py        # dataclasses：State, Question(Choice/Score/Noul), Answer, DecisionResponse
  scorer.py          # confidence / 校准
  serve.py           # FastAPI：POST /v1/evaluate（Jev 兼容）+ /healthz
  data.py            # 合成数据集（垃圾短信分类等），供 tiny 训练与 demo
  train.py            # CLI：训练 tiny 模型，loss=各问题类型 CE/有序回归，落盘 checkpoint
  infer_cli.py       # CLI 推理示例
tests/               # pytest：协议序列化、三个头形状/概率和=1、批量并行、
                     #   checkpoint 保存加载、端到端 tiny 训练 30 step 内 loss 下降、
                     #   API TestClient e2e
examples/            # 至少 2 个：垃圾短信 Choice、风控 Score + Noul
scripts/smoke.sh     # 一键：lint + tests + 训练小模型 + 跑一个推理 + 起服务打 curl
```

## "能跑通"的验收标准（全部要真实执行，禁止只写不跑）
1. `pip install -e .` 成功（依赖：torch CPU、transformers、fastapi、uvicorn、pydantic、pytest、httpx；写全 pyproject.toml，锁定最低版本，全部 CPU 可装）。
2. `pytest -q` 全绿，且包含一个真实训练测试（小数据、几十 step，断言 loss 下降 + Choice 准确率 > 随机基线）。
3. `python -m z_jev.train` 用 tiny 配置真实训练并保存 checkpoint 到 `checkpoints/tiny/`。
4. 用该 checkpoint 跑 CLI 推理，输出合法 Jev 风格 JSON（概率和为 1、字段齐全）。
5. 启动 FastAPI，**真实 curl** `POST /v1/evaluate`（一次请求含多个 Choice/Score/Noul 问题），返回正确 JSON。把真实命令输出保存到 `examples/sample_output.json`。
6. `ruff check` 干净。
7. README.md：中英双语或中文为主；必须包含——架构图（ASCII/mermaid）、Jev 三原语数学定义、与真实 GLM-5 权重对接方法（加载 zai-org/GLM-5 → 挂决策头 → LoRA/冻结训练的命令骨架）、诚实的规模限制声明、快速开始、协议示例、License。
8. LICENSE = MIT；.gitignore 忽略 checkpoints/（但保留 .gitkeep 或下载脚本）、__pycache__、.venv。
9. 提交 git：原子化、有意义的多个 commit；不要 commit 大文件。

## 工作方式
- 你在 `/mnt/ssd/codespace/ai/z-jev` 内自主完成：调研→设计→实现→反复运行修复→全部验收项通过。
- 内存只有 ~1.3GB 可用：tiny 模型/batch 必须小，训练中监控内存，严禁 OOM（batch 小、序列短，如 max_len 128）。
- torch 若本机已装则复用；先 `python -c "import torch"` 检查，不要盲目重装。transformers 装不上时，backbone 必须能在无 transformers 的情况下用内置 TinyGLM 跑通（做成软依赖，import 失败自动回退）。
- 完成后输出：仓库文件树、各验收命令的真实结果摘要、commit 列表、仍存在的限制。
