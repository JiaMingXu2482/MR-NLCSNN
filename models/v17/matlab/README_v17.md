# v17 减振器数字孪生 — PMPC 部署包

把 v17 训练最优权重（E330，holdout pooled NRMSE **16.80%**）移植为纯 MATLAB
函数，可直接放进 PMPC / Simulink / CarSim 仿真链路。**正模型 + 两种逆模型**
都在包里。

## 文件

| 文件 | 说明 |
|---|---|
| `v17_damper_init.m` | 一次性加载权重，返回 `net` 结构体（Simulink InitFcn 里调一次） |
| `v17_new_state.m` | 新建单角状态 `st`（GRU 隐状态 2×16 + 电流延迟缓冲） |
| `v17_damper_step.m` | **正模型**：物理量 → 阻尼力 F [N]，`%#codegen` 兼容 |
| `v17_inverse_bisection.m` | **逆模型（主）**：期望力 → 电流，二分 40 轮，`%#codegen` 兼容 |
| `v17_inverse_head.m` | **逆模型（快）**：训练好的逆头 MLP，单次前向，用前先看第 5 节说明 |
| `v17_weights.mat` | 全部权重 + 归一化参数（171 KB，float64） |
| `test_v17_port.m` | golden-vector 测试脚本（MATLAB 里跑） |
| `golden_vectors.mat` | Python 参考实现生成的标准答案 |
| `export_v17_matlab.py` | 生成 `v17_weights.mat` 的 Python 脚本（可复现） |
| `verify_v17_port.py` | numpy 镜像 vs PyTorch 参考的对照验证脚本 |
| `SIGN_CONVENTIONS.md` | 方向约定（v≥0=rebound 等，接线前先看） |

## 1. 模型档案

- 架构：`F = c·v·G_c(Ie) + (Σ_j s_j·tanh(v/σ_j))·G_s(Ie) + b(h,nfl)`，
  σ = [0.5, 1.0, 2.0, 4.0, 8.0]（5 基），v = v_norm×30
- 输入 `u`（归一化 6 维）：`[(x-279.1375)/63.9665, v/1000, a/50000,`
  `I/1.6, dI(恒为0), (T-40)/40]`，单位 x[mm] v[mm/s] a[mm/s²] I[A] T[°C]
- 35 维静态 NFL → GRU(35→16, 2 层) → 各 MLP 头（48→24→…）
- 电流延迟：`Ie = 1-|I|/1.6`，一阶延迟 a=0.032075（τ=30.67ms，
  KMAX=185）。**注意**：训练代码的卷积核方向是反的（最新样本权重最小），
  本移植**精确复刻**了训练行为，没有"纠正"它，数值与训练逐位一致
- 单调 G：`G(0)=0, G(1)=1`，严格单调递增，保证二分逆解唯一
- 力输出：`F = F_norm × 5728.556` [N]
- 参数量 20,998；GRU 保持 2×16（S32 部署瓶颈不在此处）

## 2. 快速验证（MATLAB）

```matlab
cd <本文件夹>
test_v17_port        % 四项应全部 PASS
```

| 检查 | 内容 | 通过标准 |
|---|---|---|
| [1] | 正模型 1ms 步进 ×2412 点 | 力误差 < 1e-6 N |
| [2] | 正模型宏步 dt=0.01（10 子步 ZOH） | 力误差 < 1e-6 N |
| [3] | 二分逆解 ×5 | 电流误差 < 1e-6 A |
| [4] | 逆头 ×5 | 电流误差 < 1e-9 A |

## 2.5 正弦激励快速检查（Simulink）

`v17_sine_check.m` 是一个可直接拖进 Simulink MATLAB Function 模块的函数。
注意：`net` 不能在模块内部加载（Simulink 编译不支持 `load`），必须先在
MATLAB 工作区加载一次，再作为参数传进模块：

```matlab
net = v17_damper_init();   % 工作区里跑一次，保持 net 变量别清掉
```

- 模块函数签名：`v17_sine_check(net, t, Amp_mm, Freq_Hz, x0_mm, Temp_C, reset)`
  - 在模块编辑器里点"编辑数据"（Ports and Data Manager），添加 `net`，
    范围选"参数"，值填 `net`
- 输入：`t`（Clock，[s]）；`Amp_mm` / `Freq_Hz` / `x0_mm`（如 279.1）/
  `Temp_C` / `reset`（可做成 Constant 模块）
- 输出：`F0`（0A 恒流力 [N]）、`F16`（1.6A 恒流力 [N]）、`x_mm`、`v_mms`
- 正弦位移/速度/加速度内部解析计算，不需要 Derivative 模块
- 两路（0A / 1.6A）各用一套独立 GRU+延迟状态（persistent），互不干扰
- 求解器用定步长，1kHz 最好，100Hz 也行（内部自动 10 子步）
- `reset>0.5` 清零两路状态；清零后留约 1s warmup

## 3. PMPC 接线（多速率，和 v15 同架构）
```
PMPC(100Hz) → F_des → [v17_inverse_bisection] → i_cmd ─┐
                                                        ▼
CarSim ← F_actual ← [v17_damper_step, 1kHz×10子步] ←───┘
```

每角（四角独立实例，各自一份 `st`）：

```matlab
% 初始化（InitFcn / pmpc_block.m 里一次）:
net = v17_damper_init();
st  = v17_new_state(net);   % 每角一份

% 控制器 100Hz 步:
[I_cmd, F_ach] = v17_inverse_bisection(net, F_des, x, v, a, T, st);
% 注意: 逆解用的是本步开始时的 st.h（和 forward 之前的状态一致）

% 植物 100Hz 宏步（内部 10×1ms 子步，输入 ZOH）:
[F, st] = v17_damper_step(net, x, v, a, I_cmd, 0, T, 0.01, st);
```

- `st` 用 Unit Delay 回送；四角各用一个 MATLAB Function 实例
- `reset` 后留约 **1s warmup**（训练时前 1000 点不计 loss，GRU 从零状态启动）
- 电流钳位 [0, 1.6]A 已在逆解内部处理
- `dI` 参数保留但强制为 0（训练时该通道恒置零，见 `train_v14.py` collate）

## 4. 逆解二分（主用）

`v17_inverse_bisection` 复刻 `model_v14.inverse_current`：

1. 电流通道 mask 为 0（待求量不泄漏进 nfl/GRU）
2. 9 点粗扫 + 把 `F_des` 裁到可达区间（半主动钳位）
3. `Ie∈[0,1]` 二分 40 轮；fast 路用 `[Ie,Ie]`（逆解不经过 CurrentLag，和训练一致）
4. `I_cmd = (1-Ie)·1.6`

`F_ach` 是模型在 `I_cmd` 下预测的力，可做诊断。

## 5. 逆头（快，但有 caveat）

`v17_inverse_head` 是联合训练出来的逆头（53→48→24→1，sigmoid 输出），
一次前向，适合实时预算紧张的场合。但注意：**训练时逆头的 GRU/nfl
特征里含的是真电流**（`train_epoch` 里 `inverse_predict(u, f)` 的 u 未 mask），
部署时我们 mask 了电流通道，两者有分布差。Holdout 上报的 inv_mae=0.0101A
也是在未 mask 协议下测的，偏乐观。**主用二分逆解**；逆头可做暖启动或
快速近似。

## 6. 精度（holdout，E330）

pooled NRMSE **16.80%**（v15 最优 18.77%，好 1.97pp；v14 最优 12.55%）

| 工况 | NRMSE |
|---|---|
| ST_step5（阶跃电流） | 9.58% |
| SH_10mm_1A | 9.66% |
| RHF_RMS10mm_4Hz | 16.71% |
| RLF_RMS10mm_2Hz | 17.84% |
| SH_30mm_0A | 18.18% |
| RLF_RMS10mm_4Hz | 20.00% |
| SR_RMS10mm_0A | 20.00% |
| RLF_RMS15mm_3Hz | 21.53% |

## 7. 已知事项

- 本减振器为**反特性**：0A 时阻尼最大，随电流增大而减小（台架 ST 文件实测：
  0A 时 rms|F|=658N，1.6A 时 175N）。模型用 `Ie=1-I/1.6`、G 关于 Ie 单调增，
  与此一致；看正弦对比时应看到 F0 > F16，不要当成 bug
- `F·v≥0` 不保证（v13 起用户取消该约束）；零速连续保持
- 训练/验证/部署三处 `dI` 通道恒为 0，不要接真实微分电流
- CurrentLag 反向核是训练代码的历史行为，本包原样复刻；
  如将来重训修正了核方向，需要重新导出
- 本包只含 CPU 纯 MATLAB 实现；S32 C 移植和 DWT 实测另行安排
