# v15 — 冻结基线（14k 参数，S32 备选）

- checkpoint: `nlcsnn_v15_best_E440_18p77pct.pth`（75KB）
- 最优成绩：累计 E440，pooled NRMSE **18.77%**
- 状态：训练已停止，checkpoint 冻结，作为 v17 的热启动来源和对比基线

## 架构

- GRU：2 层 × 16（部署核心，S32 48MHz 约 1.5ms）
- mlp_hidden=32，g_hidden=16
- σ 基：(0.5, 2.0, 8.0)
- 参数量：14,141（约 55KB）
- 35 维 static NFL（static_nfl_v12）
- CurrentLag：a=0.032075，τ=30.67ms，KMAX=185（训练代码卷积核方向与标准
  EMA 相反，部署时需精确复刻，见 v17 包）
- v13 起取消 `F·v≥0` 硬约束，零速连续保留

## 说明

v15 是第一个 S32 可部署版本（2×16 GRU 精确保留给 v17）。
欠拟合：train_loss 0.0055，为同数据 v14 的 2.2 倍，故启动 v17 加容量。

## PyTorch 使用

```python
import torch
ckpt = torch.load('nlcsnn_v15_best_E440_18p77pct.pth', map_location='cpu', weights_only=False)
sd = ckpt['state_dict']; te = ckpt['total_epoch']   # 440
# 模型类：训练代码 mr_nlcsnn/model_v14.py :: NLCSNN_v14
# 超参数：gru_hidden=16, gru_layers=2, mlp_hidden=32, g_hidden=16,
#         sigmas=[0.5, 2.0, 8.0]
```

## MATLAB 部署

v15 的纯 MATLAB 移植曾用于 PMPC（`nlcsnn_matlab/`），现已被 v17 包取代，
用法同 v17（见 `../v17/matlab/README_v17.md`），权重结构一致。
