# 给本地 Agent 的执行指令（V22.3，5 模型，GPU 最大并行）

请严格按同目录 `00_AGENT_TASK.md` 实现和训练 V22-AE Hybrid。先审计本地 V20 代码、manifest、数据 split 和严格因果输入，再实现并训练且只训练以下五个模型：`AGT`、`E0`、`EG`、`EGT`、`EGTA`。统一 `seed=7`，先让五个模型同时执行 2 epoch smoke test，再进行五个正式 run、每个 130 epochs。

四个消融问题：AGT vs EGT 比较 A/E 骨干；E0 vs EG 比较 G；EG vs EGT 比较全局 tau；EGT vs EGTA 比较显式加速度残差。E0 仅作前向基准，不参加候选电流逆求解。不要实现或启动 AG、A0、EGTc、EGT2L 等额外模型。

**GPU 并行：** CUDA FP32；默认 `--max-parallel 5`，五个模型各一个进程同时运行。五进程 smoke test 全部通过且整卡峰值后仍约有 1.0 GiB 可用显存，正式训练就保持五进程。若 OOM 或显存余量不足，按 4→3→2→1 逐级降低并发，不能切 CPU、截短序列、改变 batch/data/model/optimizer。每秒记录 GPU utilization、显存、温度、功耗；每个 run 记录 CUDA peak allocated/reserved、epoch 时间和总吞吐量。若 GPU 利用率低，先检查 DataLoader/I/O/CPU bottleneck，不要盲目增加重复训练进程。

当前候选电流只允许进入单调 G→q→电流增益，Observer/NFL 只允许 I[t-1]。参数量硬上限 3500；final_test 封存；误差接近优先小模型。单 seed 结果仅作架构筛选，不得声称跨 seed 稳健或统计显著。若数据、因果、参数量、单调性或 GPU OOM 冲突，先停止并报告；禁止自行改变 split、截短序列、切 CPU 或覆盖旧结果。
