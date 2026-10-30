# V22 验收清单

任一关键项失败，暂停后续正式训练并报告。

## A. 数据与因果

- [ ] manifest SHA256 与历史值匹配，或差异已报告并获确认。
- [ ] train 259 / validation 16 / final_test 10 核验通过。
- [ ] 文件级 split，无时间片泄漏。
- [ ] final_test 标签未读取。
- [ ] 时间戳 1 kHz 检查通过。
- [ ] normalization 只从 train 计算。
- [ ] observer 电流严格使用 `I[t-1]`，首拍 0。
- [ ] E 的全部 NFL 电流特征都基于 `I[t-1]`。
- [ ] 当前候选电流只进入 G→q→current-gain。
- [ ] 未来电流扰动不影响过去输出。

## B. 模型和参数

- [ ] AGT、E0、EG、EGT、EGTA 五个模型结构与规范一致。
- [ ] AGT 与 EGT 使用相同 G/tau 结构，仅骨干不同。
- [ ] 参数量来自真实 PyTorch 模型实例。
- [ ] 每个模型参数量 ≤ 3,500。
- [ ] G_pos 正权重、单调激活、端点归一化有效。
- [ ] 当前方向审计支持 `g=1-G_pos(i)`；否则先停止。
- [ ] DeltaF 使用 Softplus，非负。
- [ ] tau 在 1–50 ms 范围，GT 初始化约 11 ms。
- [ ] EGTA 的 `b_a` 是单一可训练标量，且不依赖候选电流。

## C. 状态和逆解

- [ ] 每个文件 GRU hidden 按规则 reset。
- [ ] 每个文件 q 初始化为首拍 g；文件间不传状态。
- [ ] tau 用真实 dt 精确离散。
- [ ] 100 Hz 用 dt=10 ms，不复用 1 ms alpha。
- [ ] 所有候选 trial 固定同一 q_prev 和 observer hidden。
- [ ] trial 是纯计算，不修改任何输入状态。
- [ ] 选定电流后只 commit 一次 q。
- [ ] 0–1.6 A 以外目标能返回饱和/不可实现。
- [ ] 零速附近非唯一逆解有明确处理。

## D. GPU 与训练完整性

- [ ] CUDA 可用；无 CPU fallback。
- [ ] GPU 型号、总显存、可用显存和软件版本已记录。
- [ ] GPU utilization、显存、温度、功耗、epoch 时间和总吞吐量已记录。
- [ ] FP32；无 AMP/FP16。
- [ ] 五模型并发 smoke test 已完成，记录整卡峰值显存和各进程峰值显存。
- [ ] 正式训练默认 5 个并行 GPU 进程；若显存余量不足，记录自适应降级后的实际并发度。
- [ ] OOM 时仅降低并发度并重跑失败 run，不截短序列、不改训练配置、不切 CPU。
- [ ] smoke 5/5 通过，2 epoch。
- [ ] 正式 5/5 runs 完成，每 run 130/130 epochs，全部 seed=7。
- [ ] 每个 run 的 seed、目录、manifest、代码 hash 对应。
- [ ] best/last/config/metrics/summary 文件齐全。

## E. 结果

- [ ] seed 7 pooled NRMSE 点估计及同一验证文件上的逐文件配对差值（描述性分析，不作显著性检验）。
- [ ] SH/ST/RLF/RHF/SR 分工况结果。
- [ ] 每文件指标、最差 5 文件。
- [ ] `|v|<=300 mm/s` 覆盖率和误差。
- [ ] `|v|<=5 mm/s` 零速误差。
- [ ] 250 N / 50 N 阶跃口径分开报告。
- [ ] 力-位移回线面积与压缩/复原支路误差。
- [ ] 加速度动态残差 PCC(a)、PCC(x) 诊断。
- [ ] 单调性报告和 100 Hz 逆求解报告。
- [ ] 参数量、权重大小、GPU 显存和推理耗时。
- [ ] Python 耗时未冒称 ECU WCET。
- [ ] 最终选型遵守“误差接近时选参数量小的模型”。
