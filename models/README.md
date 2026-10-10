# MR 减振器 NLCSNN 模型版本总览

磁流变减振器神经网络数字孪生（训练代号 NLCSNN），各版本最优 checkpoint 与说明。
力模型统一形式：`F = c·v·G_c(Ie) + (Σ_j s_j·tanh(v/σ_j))·G_s(Ie) + b(h,nfl)`，
其中 `Ie = 1 - |I|/1.6`，`G_c/G_s` 关于 `Ie` 严格单调增。

> **反特性注意**：本减振器 0A 时阻尼最大，随电流增大而减小（台架 ST 文件实测：
> 0A 时 rms|F|=658N，1.6A 时 175N）。`Ie` 的定义与此一致。

## 版本对比

| 版本 | 架构 | 参数量 | 最优 pooled NRMSE | 状态 | 用途 |
|------|------|-------|-------------------|------|------|
| [v14](v14/) | GRU 2×64 | 138,512 | **12.55%** | 停止 | 精度上限参考（不上车，太大） |
| [v15](v15/) | GRU 2×16 | 14,141 | 18.77% @E440 | 冻结 | S32 部署备选基线 |
| [v16](v16/) | GRU 2×24 | 26,232 | 22.75% @E210 | 已停止 | 证伪存档，不推荐 |
| [v17](v17/) | GRU 2×16（头加宽） | 20,998 | **16.80%** @E330 | **训练中** | 当前主推，PMPC/S32 部署目标 |

指标说明：pooled NRMSE 为全部 8 个 holdout 工况文件的 pooled 力误差；
`inv_mae` 为逆模型电流 MAE（训练用逆头，非部署二分）。

## 选型建议

- **要精度、离线用**：v14（12.55%），138k 参数，只跑上位机/NI。
- **要上 S32（48MHz，3ms 预算）**：v17（16.80%，约 2.3ms）；备选 v15（18.77%，更快）。
- **v16 不用**：精度不如 v15，参数量还大，GRU 2×24 在 48MHz 下超预算。

## PyTorch 加载（通用）

```python
import torch
ckpt = torch.load('models/v17/nlcsnn_v17_best_E330_16p80pct.pth',
                  map_location='cpu', weights_only=False)
state_dict = ckpt['state_dict']          # 载入模型
print(ckpt.get('total_epoch'))           # 累计训练轮数
# 模型类定义见训练代码 mr_nlcsnn/model_v14.py（NLCSNN_v14），按各版本 README 的
# 架构超参数实例化后 load_state_dict(state_dict)
```

各版本目录的 README 含：完整架构参数、最优成绩明细、MATLAB/Simulink 用法、
checkpoint 来源与训练记录。
