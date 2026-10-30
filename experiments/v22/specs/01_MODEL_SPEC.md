# V22-AE Hybrid 模型结构规范（5 模型精简版）

## 1. 设计原则

V22 保留 V20 A/E 的 GRU+MLP 数据驱动表达能力，不沿用 V21 的 D/H/F/B 强物理元件拆分。本轮只验证三个增量因素：

1. 单调候选电流增益映射 G；
2. 电流增益通道内的全局一阶时滞 tau；
3. 一个极轻量的显式加速度残差项。

模型总式（EGT/AGT）：

\[
F_t=F_{base,t}+\hat v_t\,\Delta F_t\,q_t
\]

EGTA：

\[
F_t=F_{base,t}+\hat v_t\,\Delta F_t\,q_t+b_a\hat a_t
\]

其中 `F_base` 和 `DeltaF` 只依赖 observer 历史状态与当前机械观测，不得依赖当前候选电流。当前候选电流只进入 `G -> q -> current-gain`。

## 2. 五个保留模型

| 模型 | 骨干 | 电流 G | 全局 tau | 加速度残差 |
|---|---|---|---|---|
| `AGT` | A raw GRU16×2 + MLP16 | 有 | 有 | 无 |
| `E0` | E NFL31 + GRU8×2 + MLP16 | 无 | 无 | 无 |
| `EG` | E NFL31 + GRU8×2 + MLP16 | 有 | 无，`q=g` | 无 |
| `EGT` | E NFL31 + GRU8×2 + MLP16 | 有 | 有 | 无 |
| `EGTA` | E NFL31 + GRU8×2 + MLP16 | 有 | 有 | 有，单标量 `b_a` |

预计参数量（必须由实际 PyTorch 模型核算）：

- AGT：约 3,220；
- E0：约 2,073；
- EG：约 2,187；
- EGT：约 2,188；
- EGTA：约 2,189。

所有模型参数量必须不超过 3,500。

`E0` 是无候选电流支路的前向精度基准，不作为 100 Hz 当前候选电流逆求解模型。AGT、EG、EGT、EGTA 才进行候选电流单调性和逆求解验收。

## 3. A/E 两种骨干

### A 骨干（raw_gru）

- Observer 输入：5 维 `[x, v, a, I_prev, T]`；
- GRU：`input_size=5, hidden_size=16, num_layers=2, batch_first=True`；
- 融合输入：`[h_16, raw5]`，共 21 维；
- trunk：`Linear(21,16) -> ReLU`；
- AGT 使用 `head_base: Linear(16,1)` 和 `head_gain: Linear(16,1)`，gain 经 Softplus。

### E 骨干（nfl_small_gru）

- Observer 输入仍为 `[x,v,a,I_prev,T]`；
- NFL31 必须完全由严格因果 observer 输入计算，所有电流相关 NFL 特征只能使用 `I_prev`；
- GRU：`input_size=31, hidden_size=8, num_layers=2, batch_first=True`；
- 融合输入：`[h_8, NFL31]`，共 39 维；
- trunk：`Linear(39,16) -> ReLU`；
- E0 输出 `head_force: Linear(16,1)`；
- EG/EGT/EGTA 使用 `head_base: Linear(16,1)` 与 `head_gain: Linear(16,1)`，gain 经 Softplus。

E 的 NFL31 已经包含当前时刻的加速度及交互特征。EGTA 的意义不是首次输入加速度，而是把一个简单的显式加速度残差从数据驱动输出中拆出，检验它是否带来额外收益。

## 4. 单调电流增益 G

候选电流幅值：

\[
i_t=\operatorname{clip}(|I_{candidate,t}|/1.6,0,1)
\]

使用正权重单调网络 `G_pos: 1 -> 8 -> 8 -> 1`，中间使用单调 `tanh`，正权重参数化沿用 V20 `PositiveLinear` 思路，并用端点归一化约束：

\[
G_{pos}(0)=0,\quad G_{pos}(1)=1
\]

既有数据方向审查表明，在匹配运动工况中 F-v 斜率随电流幅值增加而下降。因此候选电流增益定义为：

\[
g_t=1-G_{pos}(i_t)
\]

即 `g_t` 随电流幅值单调不增。开训前必须在当前 manifest 上重做方向审计；若实际方向相反，停止并报告，不得静默翻转。

## 5. 全局一阶时滞 T

唯一顺序：

`I_candidate[t] -> abs/clip -> G_pos -> (1-G_pos) -> q state -> current-gain force`

### EG（无显式 tau）

\[
q_t=g_t
\]

### EGT / EGTA / AGT（全局可训练 tau）

\[
\alpha_t=1-\exp(-dt_t/\tau),\qquad
q_t=q_{t-1}+\alpha_t(g_t-q_{t-1})
\]

- tau 是一个全局正标量；
- 初始化 11 ms；
- 约束范围 1–50 ms；
- 使用有界 sigmoid 或等价稳定参数化；
- 1 kHz 训练 `dt=0.001 s`；100 Hz 逆解 `dt=0.01 s`，必须重新计算 alpha；
- 文件首拍初始化 `q[0]=g[0]`；文件间 reset。

本轮不训练工况条件化 tau。若全局 tau 已显示稳定收益，再在下一轮单独研究 tau 是否需要依赖工况状态。

## 6. EGTA 显式加速度残差

\[
F_{acc,t}=b_a\hat a_t,\qquad \hat a_t=a_t/50000
\]

- `b_a` 为一个全局可训练、有符号标量，建议初始化为 0；
- 不读取当前候选电流；
- 不增加 Bouc-Wen、摩擦、多尺度 tanh 或大参数 head；
- 这是论文启发的最小消融，不预设一定有效。

论文报告在其特定动态随机数据中，速度骨架残差与加速度相关性高于与位移的相关性。本数据必须重新计算对应 PCC，不能把论文中的相关系数直接当作本减振器的事实。

## 7. 严格因果输入与逆求解 API

Observer 输入：

`obs[t] = [x[t], v[t], a[t], I_prev[t], T[t]]`

其中：

- `I_prev[0]=0`；
- `I_prev[t]=clip(abs(I[t-1])/1.6,0,1)`；
- Observer/NFL 不允许读取 `I[t]` 或未来电流；
- 当前候选电流只进入 G/q/current-gain 路径。

实现接口至少包括：

- `observe(obs_t, hidden_state) -> z_t, hidden_next`；
- `current_map(I_candidate) -> g_candidate`；
- `trial_force(z_t, v_t, a_t, q_prev, I_candidate, dt) -> F_candidate, q_candidate`：纯函数，不修改任何传入状态；
- `commit_state(q_candidate)`：只有实际选定电流后调用一次。

100 Hz 二分试算中，所有候选电流都必须从同一个 `q_prev` 和同一个 observer 状态开始。不能在每次 trial 里更新 GRU hidden 或 q。

## 8. 单调性与符号约定

- 力、速度正负号沿用 V20 数据约定，不得凭直觉翻转；
- `v_hat=v/1000`，保持原始速度符号；
- `DeltaF=softplus(head_gain(z_t)) >= 0`；
- 固定非零速度和历史状态时，候选电流只通过 q 改变输出，预测力应沿预期方向单调；
- `|v|` 接近零时电流控制权趋近于零，逆解可能非唯一或目标不可实现，必须报告。

