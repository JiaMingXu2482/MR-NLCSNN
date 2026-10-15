# v17 Simulink 正逆模型接口说明（MATLAB Function 版）

> 对应模型：`ForwardModel4_4.slx`（2026-10-15）
> 正模型：`v17_fwd_mf`（块内 wrapper）→ `v17_fwd_mf_impl.m`
> 逆模型：`v17_inv_mf`（块内 wrapper）→ `v17_inv_mf_impl.m`
> 每角一个块，共 8 个块。`v17_weights.mat` 须与 impl 文件同目录。

## 一、信号流（往返测试）

```
CarSim (vs_sf)
  └─ Demux → cmpD_L1/cmpRD_L1, cmpD_R1/cmpRD_R1, cmpD_L2/cmpRD_L2, cmpD_R2/cmpRD_R2
       │
       ├─→ 正模型块 (v17_fwd_mf, cid=1..4)
       │     输入：t, Temp_C=30, cmpD_*, cmpRD_*, I_cmd=[current]=0, cid
       │     输出：Fm_*  → Goto [Fm_L1] 等
       │
       └─→ 逆模型块 (v17_inv_mf, cid=1..4)
             输入：Fd_N=[Fm_*], t, Temp_C, cmpD_*, cmpRD_*, cid
             输出：I_* → [I_L1] 等；Flo/Fhi/F_clipped/F_ach → Scope
```

**往返逻辑**：正模型用 0A 电流算出阻尼力 `Fm_*` → 逆模型以 `Fm_*` 为目标反解电流 → 若正逆一致，`I_*` 应 ≈ 0A。

## 二、正模型块 `v17_fwd_mf`（4 个：L1/R1/L2/R2）

| 端口 | 信号源 | 物理含义 | 单位 | 符号约定 |
|---|---|---|---|---|
| `t` | [clock] | 仿真时间 | s | — |
| `Temp_C` | [Temper] | 油温（Constant 30） | °C | — |
| `cmpD` | [cmpD_L1] 等 | 减振器位移 | mm | CarSim：压缩为 + |
| `cmpRD` | [cmpRD_L1] 等 | 减振器速度 | mm/s | CarSim：压缩为 + |
| `I_cmd` | [current] | 电流指令（Constant 0） | A | [0, 1.6]，块内限幅 |
| `cid` | Constant | 角 ID | — | 1=L1, 2=L2, 3=R1, 4=R2 |
| `F`（输出） | → [Fm_L1] 等 | 减振器阻尼力 | N | CarSim：压缩为 + |

**注意**：
- 同一角的正逆模型块必须用**同一组** `cmpD`/`cmpRD` 和**同一个** `cid`。
- 输出 `F` 是 raw 网络力（无 TAU_X、无被动投影、无输出滤波），专供往返测试。
- **无 4 倍缩放**：本 MF 版输出即单角力，直接进逆模型（旧 Interpreted 版曾用 1/4 Gain，现已取消）。

## 三、逆模型块 `v17_inv_mf`（4 个：L1/R1/L2/R2）

| 端口 | 信号源 | 物理含义 | 单位 |
|---|---|---|---|
| `Fd_N` | [Fm_L1] 等 | 目标阻尼力（正模型输出） | N，CarSim 符号 |
| `t` | [clock] | 仿真时间（算 dt、限流） | s |
| `Temp_C` | [Temper] | 油温 | °C |
| `cmpD` | [cmpD_L1] 等 | 减振器位移（与正模型同源） | mm |
| `cmpRD` | [cmpRD_L1] 等 | 减振器速度（与正模型同源） | mm/s |
| `cid` | Constant | 角 ID（与正模型同角同值） | 1..4 |
| `I_cmd`（输出） | → [I_L1] 等 | 求解电流 | A，[0,1.6]，160A/s 限流 |
| `Flo`（输出） | → [Flo_L1] 等 | 可达力下包络 | N，CarSim 符号 |
| `Fhi`（输出） | → [Fhi_L1] 等 | 可达力上包络 | N，CarSim 符号 |
| `F_clipped`（输出） | → [Fcb_L1] 等 | 目标力经包络裁剪后 | N |
| `F_ach`（输出） | → [Fach_L1] 等 | 在 `I_cmd` 下模型预测力 | N |

**判定标准**（0A 往返测试）：
1. `I_L1` 等全程 ≈ 0A（允许 ±0.05A 数值噪声）
2. `F_ach` 应紧贴 `Fm_*`（同一 Scope 对比）
3. `Fm_*` 应落在 `[Flo, Fhi]` 内；若超出，`F_clipped` 为裁剪值，`F_ach` 跟踪 `F_clipped` 而非 `Fm_*`

## 四、状态与复位

- 每个 impl 用 `persistent` 存 4 角独立状态（GRU `h` 2×16、`ie_buf` 185×1、`t_prev` 等），`cid` 选槽。
- 换 impl 文件或改参数后，MATLAB 命令行执行：
  ```
  clear v17_fwd_mf_impl
  clear v17_inv_mf_impl
  ```
  （清 wrapper 没用，状态在 impl 里）
- 首次步进自动做 200 步 warmup（消除 t=0 尖峰），无需手动处理。

## 五、提速

- 当前为解释执行。如需提速，对每个 MATLAB Function 块右键 → C/C++ 代码 → 为此子系统构建，生成 MEX。
- `coder.load('v17_weights.mat')` 在编译时烘焙权重，运行时无文件 I/O。
