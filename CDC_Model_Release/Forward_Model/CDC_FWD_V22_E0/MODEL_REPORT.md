# MODEL_REPORT — CDC_FWD_V22_E0

**状态：FROZEN（已冻结）**　冻结日期：2026-10-30

| 项 | 值 |
|---|---|
| 正式名称 | `CDC_FWD_V22_E0` |
| 版本来源 | V22 AE-hybrid 主矩阵，模型 `E0` |
| 角色 | 正向阻尼力预测 |
| 参数量 | **2073**（gru 1416 + trunk 640 + head 17） |
| 骨干 | GRU(31,8,2) → concat(h, feat)=39 → Linear(39,16)+ReLU → Linear(16,1) |
| 采样率 | 1 kHz |
| Seed | 7 |
| 训练轮数 | 130（best checkpoint 存于 **ep 127**） |
| checkpoint 大小 | 11832 bytes |
| SHA256 | `dbfb844ff117161cc78c634e307c5215ff88abf769880691e4d72433179351b4` |

## 验证指标（独立进程复算，非引用历史）

| 指标 | 复算值 | 历史记录 | 差值 |
|---|---|---|---|
| Pooled NRMSE | **10.594998%** | 10.594997% | 0.00000010 pp |

分工况（NRMSE %）：SH 4.7411 / ST 11.5174 /
RLF 19.6818 / SR 34.9814 /
RHF 15.2199

**指标口径**：`100 * sqrt(sum(e²) / sum(y²))`，`sst = Σy²` **不中心化**，单位为牛顿，
使用 validation 边缘 guard 掩码。该口径与 V22 的 `evaluate()` 完全一致。

## 输入 / 输出

- **输入**：`u (B,T,5) = [x_mm, v_mm_s, a_mm_s2, I_A, T_C]`，1 kHz 原始单位。
- **内部电流移位**：模型内 `observer_input` 把电流列整体后移一拍，因此所有电流项实际用的是
  `I[t-1]`；`t=0` 处 `I[-1] = 0`。
- **输出**：归一化阻尼力；`F_N = output * f_scale`，其中 `f_scale = 7664.259766 N`。
- 符号约定与实测阻尼力一致（同号）。

## 归一化（取自 checkpoint 内嵌 norm，未重算）

| 通道 | 变换 | 参数 |
|---|---|---|
| x | (x − x_ref)/x_scale | x_ref = 279.134003 mm，x_scale = 63.970001 mm |
| v | v / v_scale | 1000.0 mm/s |
| a | a / a_scale | 50000.0 mm/s² |
| I | clip(\|I\| / i_scale, 0, 1) | 1.600 A |
| T | (T − t_ref)/t_scale | t_ref 40.0 °C，t_scale 40.0 °C |
| F | F / f_scale | 7664.259766 N |

## 状态与历史

- GRU 隐状态在**每个文件（每条独立记录）开始时清零**，文件之间不串接。
- **没有固定滑动窗口**：整条序列一次性前向，GRU 逐 1 ms 递推。
- 唯一的显式历史是电流的 1 拍延迟（`I[t-1]`）。

## 文件清单

| 文件 | 说明 |
|---|---|
| `best.pth` | 冻结权重（含 state_dict / norm / epoch / params） |
| `model_def.py` | 训练时的模型定义原件（逐字节复制，未改） |
| `model_config.json` | 结构、参数量、指标、来源 |
| `normalization.json` | 8 个归一化参数及用法 |
| `feature_spec.json` | 31 维特征逐项公式与时间索引 |
| `golden_vectors.npz` | **C 移植验收用**：200 个 1 ms 样本的输入与期望输出 |
| `golden_vectors_meta.json` | 上者的说明与预览 |
| `source_run_config.json` / `source_summary.json` | 原始训练产物（只读副本） |

## 已知限制

- 单 seed（7）训练，指标为验证集结果，不是跨 seed 统计量。
- `Pooled NRMSE 10.595%` 用的是不中心化口径；若换成中心化口径同一 checkpoint 为 11.376%。
  比较时**必须注明口径**。
- 本模型为数据驱动黑箱，不具备物理单调性保证（V22 的 G 通路版本已被证明无收益且不入版）。

## 冻结声明

本目录下的权重自冻结起不得被训练、微调或覆盖。
`model_def.py` 与 `feature_def` 仅供移植参考，**任何修改都会使 SHA256 失配并使本版本作废**。
