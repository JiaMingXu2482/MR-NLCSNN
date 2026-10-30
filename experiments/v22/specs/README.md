# V22-AE Hybrid 训练包（5 模型精简版）

**用途：** 交给本地 Agent，在现有 V20 工程和正式数据集上实现、审核并训练 V22 轻量化 CDC 减振器正向模型，同时验证 100 Hz 电流逆求解。

## 阅读顺序

1. `00_AGENT_TASK.md`：唯一执行主指令、停止条件和总体流程。
2. `01_MODEL_SPEC.md`：五个模型的公式、输入时序、参数预算和状态定义。
3. `02_TRAINING_PROTOCOL.md`：数据、训练、GPU 并行和评估口径。
4. `03_ABLATION_MATRIX.md`：五个模型、五个正式 run 和四个必要消融对照。
5. `04_ACCEPTANCE_CHECKLIST.md`：开训前、训练中和交付验收。
6. `run_config_template.json`、`historical_baselines.json`：机器可读配置和历史锚点。

## 本轮保留的五个模型

- `AGT`：A 骨干 + 单调 G + 全局 tau；
- `E0`：E 原始基准；
- `EG`：E + 单调 G；
- `EGT`：E + 单调 G + 全局 tau；
- `EGTA`：EGT + 显式加速度残差。

四个对照分别回答 A/E 骨干差异、G 的贡献、全局 tau 的贡献、显式加速度支路的贡献。

`E0` 仅作为前向基准，不具备当前候选电流逆求解通道；AGT、EG、EGT、EGTA 才进行单调性和 100 Hz 逆求解验收。

## 执行约束

- 所有模型统一 seed 7；共 5 个正式 run，每个 130 epochs。
- 先做 5 个模型各 2 epochs 的 smoke test。
- CUDA 强制启用，FP32；8 GB 显存默认五模型同时并行，依据五进程 smoke test 的实测显存余量按 4→3→2→1 自适应降级。
- Observer/NFL 只看 `I[t-1]`；当前候选电流只进 `G -> q -> current-gain`。
- 参数量硬上限 3,500；精度接近时优先小模型。
- `final_test` 全程封存；单 seed 只用于架构筛选，不代表跨 seed 稳健性。

## 本轮暂缓

不训练 AG、A0、EGTc、EGT2L，不增加额外 Bouc-Wen、摩擦或多尺度物理支路。若五模型结果显示明确需要，再设计第二轮针对性实验。

## 参考资料

- `reference_v20/`：V20 只读参考实现；Agent 必须核对本地最新因果实现，不能照抄旧版时序处理。
- `references/EXPERIMENT_LOG_v20(4).md`：V20 正式实验记录。
- `references/V21_CODE_REVIEW_AND_SMOKE_REPORT.md`：V21 代码审查记录。
- `references/VADM_NVADM_acceleration_reference.pdf`：用户提供的加速度相关减振器建模论文。

包内历史日志保留原始多 seed 实验记录；它们不是本轮 V22.2 执行要求。
