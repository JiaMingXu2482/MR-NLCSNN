# v14 — 精度上限（138k 参数，不上车）

- checkpoint: `nlcsnn_v14_best_12p55pct.pth`（573KB）
- 最优成绩：pooled NRMSE **12.55%**，inv_mae 0.0139A
- 状态：训练已停止，仅作精度上限参考

## 架构

- GRU：2 层 × 64
- mlp_hidden=128，g_hidden=64
- σ 基：(0.5, 2.0, 8.0)
- 参数量：138,512
- 35 维 static NFL，力模型同总览
- CurrentLag：a=0.0583，τ=16.65ms，K=100

## 说明

v14 是容量最大的版本，精度最好（12.55%），但 138k 参数在 S32K144
48MHz 下远超 3ms 预算，只能跑上位机 Python 或 NI/PXI（Phar Lap 跑不了
PyTorch，需移植 LabVIEW/MATLAB，见 `v14_twin_package`）。

## PyTorch 使用

```python
import torch
ckpt = torch.load('nlcsnn_v14_best_12p55pct.pth', map_location='cpu', weights_only=False)
sd = ckpt['state_dict']
cfg = ckpt['model_cfg']
# {'gru_hidden': 64, 'gru_layers': 2, 'mlp_hidden': 128, 'g_hidden': 64,
#  'sigmas': [0.5, 2.0, 8.0], ...}
# 模型类：训练代码 mr_nlcsnn/model_v14.py :: NLCSNN_v14
```

checkpoint 内含 `norm_cfg`（归一化参数）、`holdout_stems`（验证文件列表），
`epoch` 为该 session 内轮数。
