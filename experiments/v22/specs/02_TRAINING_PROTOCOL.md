# V22 训练与评估协议

## 1. 数据与 split

严格沿用 V20 正式文件级 split：

- train：259 files；
- validation：16 files；
- final_test：10 files，封存。

历史 manifest SHA256：

`0766763e7fb77880ffa04f2d93bcf629bfeb817e51c571188eee1ea83262e2ff`

历史 train 工况：SH 186 / ST 30 / RLF 18 / SR 17 / RHF 8。  
历史 validation 工况：SH 8 / ST 3 / RLF 3 / SR 1 / RHF 1。

验证集是跨操作点泛化验证，历史记录指出 15/16 个完整工况四元组未在训练集中出现。报告时不要把它描述成普通随机验证。

任何 hash、文件数或 split 不符，停止并报告。不得重划数据或使用 `split_sh1of3`。

## 2. 预处理和严格因果输入

Observer 输入：

`obs[t] = [x[t], v[t], a[t], I_prev[t], T[t]]`

其中 `I_prev[0]=0`，`I_prev[t]=clip(abs(I[t-1])/1.6,0,1)`（按当前 V20/V21 已核验的数据约定）。

候选电流路径单独使用：

`i_candidate[t]=clip(abs(I_candidate[t])/1.6,0,1)`。

训练正模型时 `I_candidate[t]` 对应当前真实测量 `I[t]`。逆求解时替换为候选控制电流。**不得把当前 I[t] 重新塞回 observer/NFL。**

E 的 NFL31 必须完全由 `obs[t]` 计算，不能从另一个含 `I[t]` 的数组计算。所有含电流的 NFL 交互项也必须使用 `I_prev[t]`。

沿用 V20：

- `x_ref/x_scale` 仅从 train 计算；
- `v_scale=1000 mm/s`；
- `a_scale=50000`（沿用 V20 单位/归一化约定）；
- `i_scale=1.6 A`；
- `t_ref=40`、`t_scale=40`；
- force scale 从 train-only 统计计算，历史约 7664.3 N，必须核对实际值；
- `edge_guard`：长文件两端各 1000 点；短文件按 V20 原规则；
- 时间戳核验 1 kHz；
- 原始电流读取容差沿用已核验的 `|I|<=2.10 A`，超出即报告；控制候选范围仍为 0–1.6 A。

注意：包内旧版 `reference_v20/v20_data.py` 仅作参考，其中输入时序实现可能不是当前严格因果版。Agent 必须核对本地最新代码，并明确实现 `I_prev` shift；不能照抄后默认因果正确。

## 3. 公共训练配置

| 项目 | 配置 |
|---|---|
| epochs | 130 |
| seed | **7（本轮唯一 seed）** |
| optimizer | AdamW |
| learning rate | `3e-4` |
| weight decay | `1e-4` |
| scheduler | 5 epochs linear warmup + cosine，按完整 130 epochs 计算 |
| gradient clipping | 1.0 |
| loss | V20 plain force MSE，不增加工况/加速度辅助损失 |
| sequence | 每次一个完整文件，batch=1，文件间 reset |
| normalization | train-only |
| checkpoint | best 按 validation pooled NRMSE；同时保存 last |
| precision | FP32；本轮不使用 AMP/FP16 |
| final_test | 全程不读取 |

不得对某个模型单独改变 epoch、lr、loss weight、归一化、数据增强或序列截断。

## 4. 8 GB GPU 并行要求（首选 5 个进程）

本轮只有 5 个正式模型 run。目标是尽可能利用单张 8 GB GPU，而不是保守地固定为 2 个进程。

### 4.1 默认并行策略

1. 开训前打印 GPU 型号、显存总量/可用量、CUDA、PyTorch 版本，并检查是否有其他进程占用 GPU。
2. 强制 `device=cuda`；CUDA 不可用直接报错，不得静默 CPU fallback。
3. **目标并行度为 5：AGT、E0、EG、EGT、EGTA 五个模型各启动一个独立进程。**
4. 五个模型先同时执行 2 epoch smoke test，不能只用单模型峰值显存推测并发能力。
5. smoke 全部通过且整卡显存峰值后仍有约 1.0 GiB 可用空间，则正式训练 5 个模型全部并行。
6. 不要将所有 275 个文件常驻显存；每个进程按文件/批次加载，保持完整序列和原始数据划分。

### 4.2 自适应降级条件

- 如果 5 进程 smoke 出现 CUDA OOM，或整卡可用显存低于约 1.0 GiB，降低到 4 个进程重试。
- 若 4 进程仍 OOM/显存余量不足，再依次尝试 3、2、1。
- 降级只改变并发数量，不得改变模型、batch size、序列长度、数据、精度、优化器或训练轮数。
- 不允许静默切 CPU、截断时间序列、缩小 batch 或跳过验证。
- 若某次 OOM 导致进程退出，记录失败 run 和显存日志；修复并发度后只重跑失败 run，已完成的有效 checkpoint 不得覆盖。

### 4.3 GPU 监控与吞吐量优化

每 1 秒记录一次 `nvidia-smi` 指标：GPU utilization、memory used/free、temperature、power draw；每个训练进程记录 `torch.cuda.max_memory_allocated()` 和 `max_memory_reserved()`。

同时记录：
- 每个 run 的 epoch wall time；
- 五个 run 的总完成时间；
- GPU 平均/中位利用率；
- 整卡显存峰值和最低可用显存；
- 各并发度下的总吞吐量（epoch/min 或 samples/sec）。

如果五进程并发导致 GPU utilization 仍长期低于约 70%，先检查 CPU 数据加载、文件 I/O、同步点和小 batch kernel launch overhead；不要仅通过无限增加进程数解决。可在不改变训练数学定义的前提下优化 DataLoader、pin_memory、persistent_workers 和异步数据搬运。`num_workers` 是 CPU 数据加载 worker，不等同于 GPU 训练进程。

如果五进程时显存很充裕、GPU 利用率仍低，可以在确认每个 run 的数据和梯度更新顺序不变后，评估适度增加 DataLoader workers；不要复制同一个模型或重复训练来制造 GPU 利用率。

### 4.4 训练结束释放资源

每个 run 独立进程和目录。进程结束后释放 CUDA context。launcher 需能设置 `--max-parallel 5`，并支持按显存监控结果降到 4/3/2/1；不得硬编码成 2。

## 5. 训练阶段

### Smoke

- 5 个模型各做 seed 7、2 epochs；
- 完整 train/validation split，不截断文件数；
- 默认 5 个 GPU worker；按 smoke test 峰值显存自适应降级；
- 只检查能否训练、loss/gradient 有限、checkpoint 可重载、显存是否稳定；不能用 smoke 精度选型。

### Formal

- **5 runs：AGT、E0、EG、EGT、EGTA，各使用 seed 7；**
- 每 run 130 epochs；
- 每个模型独立初始化、独立输出目录，所有模型使用同一个固定 seed 7；
- 不启动 seed 17 或 seed 27；
- 五个模型默认同时并行；若 smoke test 显示显存不足，按 4→3→2→1 自适应降级。

## 6. 主要评价指标

### 总体精度

- validation pooled NRMSE：沿用 V20 的 `100*sqrt(sum(SSE)/sum(SST))`；
- train NRMSE、train-validation gap；
- 单 seed 7 的 pooled NRMSE 点估计；
- 同一验证文件上的模型逐文件配对差值（描述性分析，不作统计显著性检验）；
- 明确标注无法由单 seed 估计 seed 方差或训练随机性。

### 工况与尾部

- SH / ST / RLF / RHF / SR 分工况 pooled NRMSE；
- 每个 validation 文件 NRMSE 和最差 5 文件；
- `|v|<=300 mm/s` 应用范围误差与覆盖率；
- `|v|<=5 mm/s` 零速区 RMSE；
- 压缩/复原、上升/下降电流阶跃分层；
- 6.62 Hz 高频指标（沿用 V20 文件和口径）。

### 阶跃动态

沿用 V20 同一事件检测和速度漂移校正代码，分开报告：

- 主口径：`|DeltaF| >= 250 N`；
- 补充口径：`|DeltaF| >= 50 N`。

分别给实测/预测 10%→90% 响应时间、误差、事件数、上升/下降方向、压缩/复原方向和平台力幅值误差。不能混用两种阈值。

### 滞回

对可分割完整周期的 SH/SR 文件：

- measured/predicted force-displacement loop；
- 每周期 `∮F dx` 有符号面积和绝对面积；
- loop area relative error；
- 压缩支路、复原支路误差。

不完整周期标记为不可计算，不补造结果。

### 加速度残差诊断（不作为训练损失）

参照论文做法，在动态随机工况上按速度分箱估计 `F_sk(v)=E[F|v-bin]`，计算：

`DeltaF_dyn = F_measured - F_sk(v)`。

报告 `|PCC(DeltaF_dyn,a)|` 和 `|PCC(DeltaF_dyn,x)|`，并尽可能按电流/工况分层。明确说明这是数据诊断，不是模型预测误差，不可与论文不同数据集的 PCC 直接比较。对 EGTA，额外报告显式加速度项的贡献和参数 `b_a`。

### 单调性与逆求解

- 固定 observer/运动状态，扫候选电流 0–1.6 A，至少 101 点；
- 在压缩/复原、不同速度和历史状态下检查 `F(I_candidate)` 单调方向；
- 低速控制权不足样本单独统计，不用它们掩盖有效速度区的单调违反；
- 100 Hz 周期 `dt=0.01 s`，tau 按连续时间精确离散；
- 报告逆解力残差、电流范围、不可实现比例、迭代数、均值/P99 耗时；
- 每个 trial 固定相同 `q_prev` 和 observer hidden；只在选定电流后 commit 一次。

### 资源

- 实际参数量、各模块参数量、权重文件大小；
- batch=1 CPU 推理均值/P99；
- GPU 单文件峰值显存、训练 epoch 时间；
- 100 Hz 单步逆求解均值/P99；
- Python 时间不是 ECU WCET，不得这样宣称；
- 给出后续 C 实现所需的权重、归一化常量、状态变量和更新公式。

## 7. 100 Hz 逆求解

- GRU observer 仍按 1 kHz 可观测输入更新；100 Hz 控制电流在一个 10 ms 周期内 ZOH 保持。
- `alpha_100Hz=1-exp(-0.01/tau)`，不能复用 1 kHz 的 alpha。
- 对每个候选电流，固定 observer hidden 和 `q_prev` 计算候选力；trial 不得改变状态。
- 用二分法或经验证的有界单调求根方法；确认单调方向后再确定搜索比较方向。
- 选定实际电流后只提交一次 q。
- 若目标力不在 0–1.6 A 的可实现力范围，返回 saturation/infeasible 状态，不返回伪精确电流。
- 零速附近电流增益趋近于零，允许出现非唯一逆解，但必须显式报告。

## 8. 选型规则

硬约束：精度接近时优先参数量小的模型。

1. 先剔除因果、单调、数值稳定、100 Hz 状态管理或最差文件不合格者。
2. 对其余模型比较 seed 7 的 pooled NRMSE、逐文件配对差值、分工况、阶跃和回线指标；本轮不报告跨 seed mean ± SD。
3. 两模型差异约 0.5 percentage point 内，且逐文件对照没有显示大模型有一致、实质性的收益时，选小模型；该结论仅适用于本轮 seed 7 筛选。
4. 0.5 pp 是工程选型参考带，不代表统计等价。
5. 大模型若只改善 pooled 均值，却恶化 SR、零速、阶跃或最差文件，不得仅凭均值胜出。
6. 不打开 final_test 选型。

## 9. 历史结果的可比性

历史 V20 A：3105 参数，validation pooled NRMSE `11.07 ± 0.58%`。  
历史 V20 E：2073 参数，validation pooled NRMSE `11.53 ± 0.55%`。

V22 重跑 E0，并训练 AGT、EGT 两个相同 G/tau 结构的 A/E 骨干对照，统一使用严格 `I[t-1]` observer 因果约定。历史 V20 A/E 结果仅作为历史锚点；若原始训练版本使用同拍 `I[t]`，不能声称与 V22 是严格同口径配对比较。

历史 M3：约 11,049 参数、validation pooled NRMSE 约 9.29%；只作为高容量历史参考，不属于 V22 轻量模型矩阵，也不改变 3,500 参数硬上限。
