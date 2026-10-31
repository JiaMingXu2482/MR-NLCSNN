"""Phase 4 (part 2): trim golden meta, write per-model MODEL_REPORT.md."""
import hashlib
import json
import os

REL = r"D:/CDC_Model_Release"
FWD_DIR = os.path.join(REL, "Forward_Model", "CDC_FWD_V22_E0")
INV_DIR = os.path.join(REL, "Inverse_Model", "CDC_INV_V25_RI0")


def write(p, t):
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(t)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# ---- trim the two meta files: keep shape + a 3-row preview, data lives in .npz
for d in (FWD_DIR, INV_DIR):
    p = os.path.join(d, "golden_vectors_meta.json")
    m = json.load(open(p, encoding="utf-8"))
    prev = {}
    for k, v in list(m.items()):
        if isinstance(v, list) and v and isinstance(v[0], list):
            prev[k] = dict(npz_key=k, shape=[len(v), len(v[0])],
                           preview_first_3_rows=v[:3])
        else:
            prev[k] = v
    prev["data_file"] = "golden_vectors.npz (arrays stored there, not in this JSON)"
    write(p, json.dumps(prev, indent=2, ensure_ascii=False))

fwd_cfg = json.load(open(os.path.join(FWD_DIR, "model_config.json"), encoding="utf-8"))
inv_cfg = json.load(open(os.path.join(INV_DIR, "model_config.json"), encoding="utf-8"))
fwd_norm = json.load(open(os.path.join(FWD_DIR, "normalization.json"), encoding="utf-8"))
inv_norm = json.load(open(os.path.join(INV_DIR, "normalization.json"), encoding="utf-8"))
fwd_ck = json.load(open(os.path.join(REL, "audits", "phase2_fwd_recompute.json"),
                        encoding="utf-8"))
inv_ck = json.load(open(os.path.join(REL, "audits", "phase2_inv_recompute.json"),
                        encoding="utf-8"))

FWD_REPORT = f"""# MODEL_REPORT — CDC_FWD_V22_E0

**状态：FROZEN（已冻结）**　冻结日期：2026-10-30

| 项 | 值 |
|---|---|
| 正式名称 | `CDC_FWD_V22_E0` |
| 版本来源 | V22 AE-hybrid 主矩阵，模型 `E0` |
| 角色 | 正向阻尼力预测 |
| 参数量 | **{fwd_cfg['parameters']['total']}**（gru {fwd_cfg['parameters']['gru']} + trunk {fwd_cfg['parameters']['trunk']} + head {fwd_cfg['parameters']['head_force']}） |
| 骨干 | GRU(31,8,2) → concat(h, feat)=39 → Linear(39,16)+ReLU → Linear(16,1) |
| 采样率 | 1 kHz |
| Seed | 7 |
| 训练轮数 | 130（best checkpoint 存于 **ep {fwd_ck['epoch_in_checkpoint']}**） |
| checkpoint 大小 | {os.path.getsize(os.path.join(FWD_DIR, 'best.pth'))} bytes |
| SHA256 | `{sha(os.path.join(FWD_DIR, 'best.pth'))}` |

## 验证指标（独立进程复算，非引用历史）

| 指标 | 复算值 | 历史记录 | 差值 |
|---|---|---|---|
| Pooled NRMSE | **{fwd_ck['pooled_nrmse']:.6f}%** | {fwd_ck['recorded_pooled_nrmse']:.6f}% | {fwd_ck['abs_diff_pp']:.8f} pp |

分工况（NRMSE %）：SH {fwd_ck['per_condition']['SH']:.4f} / ST {fwd_ck['per_condition']['ST']:.4f} /
RLF {fwd_ck['per_condition']['RLF']:.4f} / SR {fwd_ck['per_condition']['SR']:.4f} /
RHF {fwd_ck['per_condition']['RHF']:.4f}

**指标口径**：`100 * sqrt(sum(e²) / sum(y²))`，`sst = Σy²` **不中心化**，单位为牛顿，
使用 validation 边缘 guard 掩码。该口径与 V22 的 `evaluate()` 完全一致。

## 输入 / 输出

- **输入**：`u (B,T,5) = [x_mm, v_mm_s, a_mm_s2, I_A, T_C]`，1 kHz 原始单位。
- **内部电流移位**：模型内 `observer_input` 把电流列整体后移一拍，因此所有电流项实际用的是
  `I[t-1]`；`t=0` 处 `I[-1] = 0`。
- **输出**：归一化阻尼力；`F_N = output * f_scale`，其中 `f_scale = {fwd_norm['values']['f_scale']:.6f} N`。
- 符号约定与实测阻尼力一致（同号）。

## 归一化（取自 checkpoint 内嵌 norm，未重算）

| 通道 | 变换 | 参数 |
|---|---|---|
| x | (x − x_ref)/x_scale | x_ref = {fwd_norm['values']['x_ref']:.6f} mm，x_scale = {fwd_norm['values']['x_scale']:.6f} mm |
| v | v / v_scale | {fwd_norm['values']['v_scale']:.1f} mm/s |
| a | a / a_scale | {fwd_norm['values']['a_scale']:.1f} mm/s² |
| I | clip(\\|I\\| / i_scale, 0, 1) | {fwd_norm['values']['i_scale']:.3f} A |
| T | (T − t_ref)/t_scale | t_ref {fwd_norm['values']['t_ref']:.1f} °C，t_scale {fwd_norm['values']['t_scale']:.1f} °C |
| F | F / f_scale | {fwd_norm['values']['f_scale']:.6f} N |

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
"""

INV_REPORT = f"""# MODEL_REPORT — CDC_INV_V25_RI0

**状态：FROZEN（已冻结）**　冻结日期：2026-10-30

| 项 | 值 |
|---|---|
| 正式名称 | `CDC_INV_V25_RI0` |
| 版本来源 | V25 残差式逆模型 |
| 角色 | 残差式逆模型（目标力 + 运动 → 电流增量） |
| 参数量 | **{inv_cfg['parameters']['total']}**（gru {inv_cfg['parameters']['gru']} + trunk {inv_cfg['parameters']['trunk']} + head {inv_cfg['parameters']['head']}） |
| 骨干 | GRU(31,8,2) → concat(h, x)=39 → Linear(39,16)+ReLU → **Linear(16,1) 无激活** |
| 采样率 | 100 Hz（控制周期 10 ms） |
| Seed | 7 |
| 训练轮数 | 130（best checkpoint 存于 **ep {inv_ck['epoch_in_checkpoint']}**） |
| warm-start | **否**，从头训练 |
| checkpoint 大小 | {os.path.getsize(os.path.join(INV_DIR, 'best.pth'))} bytes |
| SHA256 | `{sha(os.path.join(INV_DIR, 'best.pth'))}` |

## 核心公式

```
I_base[k] = clip(|I_meas_100Hz[k-1]|, 0, 1.6)        ← 上一拍【实测】电流
ΔI[k]     = f_θ(X_k)                                  ← 线性输出，可正可负
I_raw[k]  = I_base[k] + ΔI[k]
I_cmd[k]  = clip(I_raw[k], 0, 1.6)                    ← 部署输出
```

**绝不使用上一拍命令或上一拍预测值做基准。** 该定义是 V25 实验的成立前提。

## 验证指标（独立进程复算，非引用历史）

| 指标 | 复算值 | 历史记录 | 差值 |
|---|---|---|---|
| MAE | **{inv_ck['metrics']['mae']:.6f} A** | {inv_ck['recorded']['mae']:.6f} | {abs(inv_ck['metrics']['mae']-inv_ck['recorded']['mae']):.8f} |
| RMSE | **{inv_ck['metrics']['rmse']:.6f} A** | {inv_ck['recorded']['rmse']:.6f} | {abs(inv_ck['metrics']['rmse']-inv_ck['recorded']['rmse']):.8f} |
| P95 | **{inv_ck['metrics']['p95']:.6f} A** | {inv_ck['recorded']['p95']:.6f} | {abs(inv_ck['metrics']['p95']-inv_ck['recorded']['p95']):.8f} |
| P99 | **{inv_ck['metrics']['p99']:.6f} A** | {inv_ck['recorded']['p99']:.6f} | {abs(inv_ck['metrics']['p99']-inv_ck['recorded']['p99']):.8f} |
| pooled NRMSE | {inv_cfg['metrics']['pooled_nrmse_pct']}% | 同 | 0 |

分工况 MAE（A）：SH {inv_ck['per_condition']['SH']['mae']:.4f} /
ST {inv_ck['per_condition']['ST']['mae']:.4f} / RLF {inv_ck['per_condition']['RLF']['mae']:.4f} /
SR {inv_ck['per_condition']['SR']['mae']:.4f} / RHF {inv_ck['per_condition']['RHF']['mae']:.4f}

零电流专项（目标 < 0.05 A，{inv_ck['zero_current']['n']} 点）：
平均输出 **{inv_ck['zero_current']['pred_mean']:.4f} A**，
false turn-on(>0.05 A) **{100*inv_ck['zero_current']['false_turn_on_005']:.2f}%**。

限幅：raw 越界 **{100*inv_ck['raw_out_of_range']:.2f}%**，clip 触发 **{100*inv_ck['clip_activation']:.2f}%**。

## 输入 / 输出

- **输入**：`x (B,T,31)` = motion[20] + condition[11]，100 Hz，已归一化。
- **condition 第 0 槽** = `F_des = F_meas_decimated[k] / f_scale`（目标阻尼力，N）。
- **condition 第 1 槽** = `I_past = I_meas_decimated[k-1] / i_scale`（**有符号、未裁剪**）。
- **输出**：ΔI，单位 A，无界、可负。

### ⚠ 两个都叫“上一拍电流”但转换不同（C 移植必须区分）

| 用途 | 表达式 | 是否取绝对值 | 是否裁剪 |
|---|---|---|---|
| **残差基准** `I_base` | clip(\\|I[k-1]\\|, 0, 1.6) A | **是** | **是** |
| **特征第 1 槽** `I_past` | I[k-1] / 1.6（有符号） | **否** | **否** |

两者来自同一传感器采样点，但变换不同。**不得合并成一个量。**

## 预处理（必须逐位复现）

1. 61 抽头因果窗函数 sinc 低通，截止 40 Hz，Hamming 窗（系数见 `fir_61tap_40hz.json`）
2. 每 10 点抽取，从索引 0 对齐
3. 群时延 **30 ms**，对所有通道相同 → 输入与标签同步平移，训练时相互抵消

## 状态与历史

- GRU 隐状态与残差基准**在每个文件开始时复位**。
- 文件第一个控制拍没有同文件上一拍电流，训练与评估中**均排除**；部署上电首拍需另行定义
  （建议保持上一拍读数为 0，输出即等于 ΔI）。
- 无固定滑动窗口，逐 10 ms 递推。

## 文件清单

| 文件 | 说明 |
|---|---|
| `best.pth` | 冻结权重 |
| `model_def.py` | 训练时模型定义原件（逐字节复制） |
| `feature_def.py` | 特征构造原件（motion/cond/FIR），未改 |
| `model_config.json` | 结构、参数量、指标、来源 |
| `normalization.json` | 8 个归一化参数 |
| `feature_spec.json` | 31 维特征逐项公式 + 残差基准定义 |
| `fir_61tap_40hz.json` | 抗混叠滤波系数（C 移植必需） |
| `golden_vectors.npz` | **C 移植验收用**：100 个控制拍的输入/基准/期望输出 |
| `golden_vectors_meta.json` | 上者说明与预览 |
| `source_run_config.json` / `source_summary.json` | 原始训练产物只读副本 |

## 已知限制

- 单 seed(7)，130 轮结束时验证指标仍在缓慢改善 → **未收敛**，指标是预算下限。
- 8.05% 的样本 raw 输出越界并被裁剪，全部来自下界（基准接近 0 时预测为负）。
- 预测的电流变化量总变差为实测的 1.40 倍（抖动偏大）。
- SH/SR（电流恒定）上不如「保持上一拍」基线，仅在电流变化的 ST/RLF/RHF 上占优。

## 部署前提

**每个控制周期必须能读到上一拍实测电流。** 试验台已确认电流通道是实测线圈电流
（阶跃上升 10 ms、有过冲、18.31% 为负）；目标 ECU/HIL 是否提供同一路反馈需在集成阶段确认。
若只有上一拍命令，本模型不成立。

## 冻结声明

权重自冻结起不得训练、微调或覆盖。任何对 `model_def.py` / `feature_def.py` /
`fir_61tap_40hz.json` 的修改都会使 SHA256 失配并使本版本作废。
"""

write(os.path.join(FWD_DIR, "MODEL_REPORT.md"), FWD_REPORT)
write(os.path.join(INV_DIR, "MODEL_REPORT.md"), INV_REPORT)
print("  MODEL_REPORT.md written for both models")
print("  FWD report sha", sha(os.path.join(FWD_DIR, "MODEL_REPORT.md"))[:16])
print("  INV report sha", sha(os.path.join(INV_DIR, "MODEL_REPORT.md"))[:16])
