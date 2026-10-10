# v17 — 当前主推（训练中，PMPC/S32 部署目标）

- checkpoint: `nlcsnn_v17_best_E330_16p80pct.pth`（100KB）
- 当前最优：累计 E330，pooled NRMSE **16.80%**
  （train_loss 0.00310，mean 16.69%，max 21.53%，inv_mae 0.0101A）
- 状态：**训练进行中**（目标 E500，每 30 轮验证；本 checkpoint 会随训练更新）

## 架构（20,998 参数）

- GRU：2 层 × 16，**精确保留 v15 部署核心**（S32 算力约束）
- mlp_hidden：32 → **48**；g_hidden：16 → **24**（部署便宜的位置加容量）
- σ 基：3 → **5**（0.5, 1.0, 2.0, 4.0, 8.0）
- inv_head：53→48→24→1（训练专用，部署零成本）；dropout 0
- 35 维 static NFL（static_nfl_v12）
- CurrentLag：a=0.032075，τ=30.67ms，KMAX=185
  （训练代码卷积核方向与标准 EMA 相反——最新样本权重最小；
  部署包精确复刻以保数值一致，未自行"纠正"）
- 单调 G_c/G_s 保留，二分逆解成立；`F·v≥0` 不保证（v13 起）

## 训练来源

从 v15 E440（18.77%）经 `expand_v15_to_v17.py` 扩展热启动：
GRU 权重精确复制，头部分复制+新维小初始化，输出层新列零初始化。
E30 首验即 19.30%，E330 达 16.80%，已超 v15 最优 1.97pp。
部署算力约 v15 的 1.6 倍（约 2.3ms @48MHz，3ms 预算内）。

## PyTorch 使用

```python
import torch
ckpt = torch.load('nlcsnn_v17_best_E330_16p80pct.pth', map_location='cpu', weights_only=False)
sd = ckpt['state_dict']; te = ckpt['total_epoch']   # 330
# 模型类：训练代码 mr_nlcsnn/model_v14.py :: NLCSNN_v14
# 超参数：gru_hidden=16, gru_layers=2, mlp_hidden=48, g_hidden=24,
#         sigmas=[0.5, 1.0, 2.0, 4.0, 8.0]
```

## MATLAB / Simulink 使用（`matlab/` 目录）

纯 MATLAB 正逆模型包（PMPC 部署用），numpy 对 PyTorch 逐样本误差 ~3e-7：

```matlab
addpath('matlab');
net = v17_damper_init();        % 加载 v17_weights.mat（工作区跑一次）
st  = v17_new_state(net);       % 每角一份状态，四角各自维护

% 正模型（dt=0.01 时内部自动 10×1ms ZOH 子步）
[F, st] = v17_damper_step(net, x, v, a, I, 0, T, 0.01, st);

% 逆模型：二分法（主用，40 次，电流限 [0,1.6]A）
[I_cmd, F_ach] = v17_inverse_bisection(net, F_des, x, v, a, T, st);
```

- `v17_sine_check_sl.m`：Simulink 正弦验证（Interpreted MATLAB Function，
  Mux(6)→模块→Demux(4)，免代码生成）；另有 `v17_sine_check.m`（codegen 版，
  net 作参数传入）。
- 反特性：0A 阻尼最大，看 F0 > F16 是正常的。
- `dI` 接口保留但恒传 0（训练时该通道恒零）。
- reset 后留约 1s warmup。
- 详见 `matlab/README_v17.md`、`matlab/SIGN_CONVENTIONS.md`。
