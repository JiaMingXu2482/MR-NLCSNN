# V22-AE Hybrid：本地 Agent 执行任务书

版本：V22.2（5 模型、单 seed 训练版）  
任务：在 V20 A/E 轻量 GRU+MLP 骨干上，加入单调电流增益和可选一阶动态时滞；利用加速度相关物理启发设计一个极轻量消融；完成正向精度、100 Hz 逆求解和参数量审查。  
**本文件是执行主指令。若本地源代码、manifest 或数据与本文不一致，先停止并报告，禁止自行猜测或覆盖。**

---

## 0. 目标与硬约束

1. 以 V20 的 **A raw_gru** 和 **E nfl_small_gru** 为骨干，不回到 V21 的 D/H/F/B 强物理分解。
2. 保留 GRU+MLP 的时序表达能力，显式分离：
   - `F_base`：Observer 历史和当前机械观测形成的基准力；
   - `DeltaF`：非负的电流调节幅值；
   - `G -> q`：当前候选电流的单调静态映射及可选时滞状态。
3. 当前候选电流不得进入 GRU、NFL、`F_base`、`DeltaF` 或 tau 条件头；只允许进入 `G -> q -> current-gain`。
4. Observer 的电流历史严格使用 `I[t-1]`，首样本为 0；不得使用 `I[t]` 或未来电流。
5. 参数量目标约 2,000–3,500，**硬上限 3,500**。超过必须停止并报告，不得私自缩小/扩大其他模块掩盖问题。
6. 用户选型硬约束：**误差接近时选择参数量更小的模型**。
7. 正向训练频率 1 kHz；逆求解验证频率 100 Hz。必须验证 sample-rate mismatch 和候选试算状态不变性。
8. GPU 训练必须使用 CUDA。目标 GPU 显存约 8 GB；**默认目标为 5 个训练进程同时并行（五个模型各一个进程）**，通过 smoke test 实测显存峰值后决定是否需要降到 4/3/2。
9. `final_test` 的 10 个文件全程封存，不得读取标签、计算指标、选 checkpoint 或调参。

## 1. 开始编码前的审计（必须先做）

先检查本地工程和数据，至少包括：

- `v20_models.py` / 当前实际使用的模型文件；
- `v20_data.py` / 当前实际使用的数据处理文件；
- `train_v20.py`、并行启动脚本；
- A、E 的历史 checkpoint、`run_config.json`、训练日志；
- `alignment_manifest.json`、`COMPLETE.json`；
- V20 的归一化、edge guard、pooled NRMSE、阶跃响应和回线指标代码。

历史记录：

- 数据目录：`D:\v20_data\Parsed_v19_F42_I42`
- manifest SHA256：`0766763e7fb77880ffa04f2d93bcf629bfeb817e51c571188eee1ea83262e2ff`
- train / validation / final_test：259 / 16 / 10 文件
- 历史训练工况数：SH/ST/RLF/SR/RHF = 186/30/18/17/8
- 历史验证工况数：SH/ST/RLF/SR/RHF = 8/3/3/1/1

若 hash、文件数量、split、`processed_sha256` 或时间采样与记录不一致：

1. 不要开始正式训练；
2. 输出实际值、期望值和差异；
3. 等用户确认后再继续；
4. 不允许自动重建 manifest、重切 split 或使用时间片随机划分。

建议新建独立目录：`D:\v22_ae_hybrid\`。不得覆盖 V20/V21 代码、checkpoint 或结果。

## 2. 本轮模型矩阵（仅保留 5 个关键模型）

统一使用 seed 7。正式训练 **5 个模型、5 个 run**，每个 130 epochs。

| 模型 | 结构 | 主要对照问题 | 预计参数 |
|---|---|---|---:|
| `AGT` | A raw GRU16×2 + MLP16 + 单调 G + 全局 tau | 与 EGT 对照 A/E 骨干；A 骨干能否保留精度 | 3,220 |
| `E0` | 原始 E：NFL31 + GRU8×2 + MLP16，无 G/tau | E 的轻量正模型基准 | 2,073 |
| `EG` | E + 单调 G，无显式 tau（q=g） | G 电流增益支路是否有收益 | 2,187 |
| `EGT` | EG + 全局可训练一阶 tau | 显式时滞是否改善阶跃及随机电流动态；主轻量候选 | 2,188 |
| `EGTA` | EGT + 显式加速度残差 `b_a*a_norm` | 加速度物理启发支路是否改善动态滞回 | 2,189 |

保留这五个模型的原因：

- `AGT vs EGT`：在相同 G/tau 电流调节结构下比较 A、E 两种骨干。
- `E0 vs EG`：比较 E 骨干增加单调电流增益支路前后的变化。
- `EG vs EGT`：单独识别全局 tau 的贡献。
- `EGT vs EGTA`：单独识别显式加速度残差支路的贡献。

重要边界：

- `E0` 是无当前候选电流通道的正模型基准，只用于前向精度参考，不作为 100 Hz 候选电流逆求解模型。
- `AGT/EG/EGT/EGTA` 具有当前候选电流通道，可进行单调性和逆求解验证。
- `AG`、`EGTc`、`EGT2L` 本轮不训练。工况条件化 tau 和两层 MLP 留到第二轮，只有在本轮结果明确显示有必要时才增加。
- 不再实现或启动 A0、AG、EGTc、EGT2L 等额外 run。

## 3. 训练执行顺序

### Phase 0：代码和数据审计

先完成第 1 节。数据和时序检查不通过，停止。

### Phase 1：实现与参数核算

- 实现上述 5 个模型及统一 forward/state API。
- 输出每个模型真实参数量 `sum(p.numel() for p in model.parameters())` 和各模块参数量。
- 所有模型必须 `<= 3500` 参数。
- 先完成因果、单调性、tau、状态复位、逆求解 trial 不修改状态等单元测试。

### Phase 2：GPU smoke test

- 5 个模型全部做 seed 7、2 epoch smoke test。
- 使用完整 train/validation 文件清单，不做 `max_train_files` 或 `max_val_files` 截断；smoke 只缩短 epoch，不改变数据分布。
- CUDA、显存、梯度、loss、checkpoint 保存/重载均通过后才能进入正式训练。
- 五个模型先同时执行 2 epoch smoke test；记录每个进程 `max_memory_allocated/reserved` 和 `nvidia-smi` 整卡显存峰值。
- 若 smoke test 全部通过且整卡峰值显存后仍至少保留约 1.0 GiB 可用，则正式训练五个模型全部并行。
- 若 OOM 或可用显存低于约 1.0 GiB，按 4→3→2→1 逐级降低并行度；每次重试保持模型、数据、序列长度、batch、优化器和训练参数完全不变。不得静默切 CPU 或裁短序列。

### Phase 3：正式训练

- 5 模型 × seed 7 = **5 个正式 runs**。
- 每个 run 130 epochs；每个模型使用同一个固定随机种子 7，模型之间独立初始化、独立目录。
- 本轮不运行 seed 17/27；不得把单 seed 结果表述为跨 seed 稳健或统计显著。
- 8 GB GPU 首选 5 个训练进程同时运行，每个模型一个独立进程；只有 smoke test 显示显存压力或发生 OOM 时才降低并行度。
- 每个 run 结束后进程退出释放 CUDA context；launcher 采用可配置并行 worker 队列。
- 不允许无依据地固定为 1 或 2 个进程。

### Phase 4：评估和报告

完成总体、分工况、阶跃、滞回、加速度残差、单调性、100 Hz 逆解和资源指标；按参数量优先规则选型。

## 4. 交付物

每个 run：

- `run_config.json`
- `metrics.jsonl`（每 epoch）
- `best.pth`、`last.pth`
- `summary.json`
- `per_file_metrics.csv`
- `per_condition_metrics.csv`
- `step_response_metrics_250N.csv`
- `step_response_metrics_50N.csv`
- `hysteresis_metrics.csv`
- `acceleration_residual_diagnostics.json`
- `monotonicity_report.json`
- `inverse_100hz_report.json`
- `resource_report.json`
- `figures/`（预测时序、力-速度云图、力-位移回线、阶跃响应、加速度着色 F-v 图）

总目录：

- `V22_FINAL_REPORT.md`
- `summary_all.csv` / `summary_all.json`
- `paired_file_differences.csv`（同一验证文件上的模型逐文件配对差值，仅作描述性分析）
- `model_parameter_counts.csv`
- `gpu_parallel_log/`（记录每秒 GPU 利用率、显存占用、温度、功耗、每进程峰值显存、并行度和吞吐量）
- `failed_or_missing_runs.csv`

## 5. 关键停止条件

以下任一项发生，立即停止相关 run 并报告，不得自行修补后继续：

- manifest 或数据 split 不匹配；
- final_test 被读取；
- Observer/NFL 使用 `I[t]` 或未来电流；
- 候选电流进入 `F_base`、`DeltaF`、GRU 或 tau 条件头；
- 固定状态下电流—力单调性违反超出数值容差；
- tau 越界、状态发散、NaN/Inf；
- 100 Hz trial 修改 q/GRU 状态；
- 参数量超过 3,500；
- CUDA OOM 后发生静默 CPU fallback、缩短序列或改变训练配置；
- 任何模型使用不同 loss、split、归一化或 epoch 预算。

## 6. 最终选型原则

先过因果性、单调性、100 Hz 逆解、最差文件和动态响应底线，再比较精度与参数量。

- pooled NRMSE 差异约 `0.5 percentage point` 以内，且逐文件配对误差没有显示大模型具有一致、实质性的收益时，优先选参数量更小者。
- `0.5 pp` 是工程选型参考带，不代表统计等价证明。
- 本轮只有 seed 7，不能估计随机种子方差；逐文件配对差值也不能替代多 seed 稳健性检验。最终结论应标注为单 seed 筛选结果。
- 若大模型只改善总体均值，却明显恶化 SR、零速、阶跃或最差文件，不得仅凭 pooled NRMSE 选它。
- 不打开 final_test 选型。最终模型锁定后，另行由用户决定是否做一次 final_test。
