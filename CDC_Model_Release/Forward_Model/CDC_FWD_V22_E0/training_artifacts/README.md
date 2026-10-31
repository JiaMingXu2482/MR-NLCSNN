# CDC_FWD_V22_E0 训练产物（只读归档）

本目录保存该模型训练时的逐轮日志与诊断产物，**仅供追溯，不是发布权重**。

- `metrics.jsonl`：**逐轮训练 log**（每行一个 epoch：train_loss、验证指标、耗时）
- `per_file_metrics.csv` / `per_condition_metrics.csv`：逐文件 / 分工况指标
- `figures/`：训练报告图
- 其他 JSON/CSV：各轮专项诊断（阶跃、回线、单调性、零电流等）

**未收录**：`last.pth`（最后一轮权重）、`best_by_val_mae.pth`（辅助选择权重）——
冻结版本只认上一级的 `best.pth`，避免与发布权重混淆。
原始 `run_config.json` / `summary.json` 已分别以 `source_run_config.json` /
`source_summary.json` 归档在模型目录下。
