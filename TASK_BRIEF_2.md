# Z-Jev 工程化补全任务书（给 Claude Code，二期）

仓库：/mnt/ssd/codespace/ai/z-jev（已存在一期代码，先通读全部源码与 README）。
一期已完成：GLM-5 backbone adapter（可回退 TinyGLM）、非自回归决策头 Choice/Score/Noul、FastAPI、train/infer CLI、24 测试、smoke 脚本。
本期目标：**把它从 demo 补全为可工程应用（production-usable）的完整工程**。已有 `.github/workflows/ci.yml`（如不存在则按末尾规格创建）。

## 必须交付

### 1. 真实 GLM-5 权重 LoRA 训练脚本（B 项核心）
- 新增 `z_jev/lora_train.py`（CLI，入口注册到 pyproject：`z-jev-lora-train`）：
  - 用 transformers + peft 加载 `zai-org/GLM-5`（支持 `--model` 指定本地路径/MoE 设备映射 `device_map`、`--load-in-8bit/4bit`、CPU/dtype 参数），**冻结全部 backbone**。
  - 在 frozen hidden states 上挂一期的 `NonAutoregressiveDecisionHead`，LoRA adapter 注入 backbone 使 backbone 可少量适配决策任务；决策头全量训练。
  - 支持 JSONL 数据集（`--train-file/--val-file`），每行格式：`{"state": "...", "answers": {"q1": {"type":"choice","label":"spam"}, "q2":{"type":"score","label":2}, "q3":{"type":"noul","label":true}}}`。给出 `examples/train_sample.jsonl` 示例数据（≥20 行，可真实用于训练）。
  - 优化器、梯度累积、混合精度、checkpoint 保存（LoRA adapter + decision head 分别/合并保存）、val 指标（各类型 accuracy / ECE）、断点续训、训练日志（json lines）。
  - **关键约束**：该脚本在本机（无权重）无法真实跑通，所以必须：
    a) import transformers/peft 为软依赖，缺失时给出明确报错与安装指引；
    b) 提供 `--backbone tiny` 模式：用一期 TinyGLM 模拟 backbone 接口，在本机 CPU 用 jsonl 数据真实跑 LoRA 训练全流程（可用一个不依赖 peft 的简单低秩适配层实现 LoRA 数学：W' = W + (alpha/r) BA，B 零初始化），保存/加载/推理都要通；
    c) pytest 覆盖 tiny 模式的 LoRA 训练（短步数，断言 loss 下降、零初始化时初始输出等价原模型、adapter 合并后推理一致）。
- README 新增"真实 GLM-5 LoRA 训练"章节：多卡硬件要求（744B-A40B 实测量级估算）、完整命令、设备映射/量化选项、checkpoint 部署到 serve 的方法。禁止夸大，明确标注未在真机验证。

### 2. 容器化与部署
- `Dockerfile`：多阶段、slim 基础镜像（python:3.11-slim）、CPU torch（--index-url .../cpu）、非 root 用户、容器内端口 8000、HEALTHCHECK 调 /healthz、OCI labels、镜像大小尽量小。
- `docker-compose.yml`：服务 z-jev，环境变量 `ZJEV_CHECKPOINT`、`ZJEV_HOST/PORT`、可选 `ZJEV_API_KEY`、volume 挂载 ./checkpoints 与（未来）jsonl 数据；健康检查。
- `.dockerignore`。
- serve.py 增加 `main()` CLI 入口（z-jev-serve 已有则完善）：`--host/--port/--checkpoint`，uvicorn 启动，启动日志打印 checkpoint 与版本。
- 新增 `docs/DEPLOYMENT.md`：裸机 / Docker / compose / 反向代理（nginx 示例）/ systemd unit 示例 / 环境变量清单表 / 升级与回滚。

### 3. 生产级 API 加固
- 可选 API Key 鉴权：设置环境变量 `ZJEV_API_KEY` 时，除 /healthz 外全部要求 `Authorization: Bearer <key>`，未设置时保持开放（向后兼容）。pytest 覆盖两模式 + 401。
- 结构化请求日志（请求 id、问题数、耗时 ms、状态码；不要 log state 全文，截断到可配置长度，避免泄密）；中间件实现，stdout JSON lines。
- 明确错误协议：422 校验错误、503 无 checkpoint、500 兜底错误均返回统一 `{"error": {"code","message","request_id"}}`；所有响应带 `X-Request-ID`（接受客户端同名头）。
- 请求大小/问题数/候选数上限（可配置，带默认值），超限 422；pytest 覆盖。
- /healthz 区分 liveness 与 readiness（readiness 要求 checkpoint 加载完成，供 K8s 用）。
- OpenAPI 描述完善，tags、示例；保留 Jev 协议兼容。

### 4. 工程基建
- `Makefile`：install / lint / test / train / smoke / docker-build / docker-up / clean。
- `requirements.txt` 或 requirements/（cpu.txt 固定主要版本下限；README 保持 pip install -e 路径）。
- CI（`.github/workflows/ci.yml`，若已存在则按此补全）：单 job 即可——checkout、setup-python 3.11、装 CPU torch、`pip install -e .[dev]`、ruff、pytest、`bash scripts/smoke.sh`、校验 sample_output.json、**Dockerfile 语法/构建可用性**（用 `docker build` 不可行时至少 `python -c` 元数据检查；runner 有 docker，可加一个 docker build job，但注意控制时间，可放在第二个 job）。
- README 更新：徽章（CI）、目录结构刷新、Docker 快速开始、API Key 用法、链接到 docs/DEPLOYMENT.md 与 lora 章节；保留诚实的规模限制声明。
- 所有新增代码过 ruff；新功能全部有 pytest；总测试数应显著多于一期。

## 工作约束
- 内存仅 ~1.3GB 可用，任何真实训练必须 tiny/小 batch；严禁 OOM。
- torch（本机 ~/.pyenv python3.14 有 torch 2.11）已装可复用；transformers/peft 若本机没有，**不要硬装**（作为软依赖处理），先检查 `~/.pyenv/shims/python3 -c "import transformers,peft"`。
- 必须真实执行：ruff、全部 pytest、`bash scripts/smoke.sh`、tiny 模式 lora 训练一次并保存产物到 examples 或 checkpoints（checkpoints 不入库）、docker 若本机可用则 `docker build`（不可用则在汇报中说明原因）。
- 原子化多个 git commit（我不 push，由主控统一推送）。
- 完成后汇报：文件树、新增测试数与 pytest 结果、lora tiny 真实输出摘要、smoke 结果、docker 结果、限制。
