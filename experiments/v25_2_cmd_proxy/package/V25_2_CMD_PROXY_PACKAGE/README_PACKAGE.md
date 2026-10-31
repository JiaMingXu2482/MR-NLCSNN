# V25.2-RI0-CMD 交付包

基于上一拍电流指令的残差逆模型（无电流反馈、无数字滤波器）。

## 先看哪个

| 需求 | 文件 |
|---|---|
| 完整报告（图片走相对路径） | `V25_2_FINAL_REPORT.md` |
| 单文件转发（图片 base64 内嵌，不依赖目录） | `V25_2_FINAL_REPORT_embedded.md` |
| 十二问速答 | 报告 §7 |
| 结论速览 | 报告 §0 |

## 目录

```
V25_2_CMD_PROXY_PACKAGE/
├── V25_2_FINAL_REPORT.md              最终报告（3 张图，相对路径）
├── V25_2_FINAL_REPORT_embedded.md     同一份报告，图片已内嵌
├── SHA256SUMS.txt                     全部文件 SHA256 + 字节数
├── figures/                           报告用图（3 张）
├── audits/                            Phase 0 七份审计 + smoke + Phase 4 复算
├── runs/V25_2_RI0_CMD_PROXY_seed7/    权重、指标、每文件/每工况 CSV
├── code/ (根目录)                     数据层 / 审计 / 训练 / 复算 / 出图脚本
├── configs/ task/ prompts/ scripts/   原始训练包内容
└── logs/                              phase0 / alias_edge / train 日志
```

## 关键数字

| 项 | 值 |
|---|---|
| 参数量 | 2073（与 V25 逐项相同） |
| 电流 MAE | 0.0388 A（Persistence 0.0445，−12.8%） |
| RMSE / P95 / P99 | 0.0912 / 0.1770 / 0.4325 A |
| 指令 vs 实测电流滞后 | **+1.3 ms**（V25 为 +38.7 ms） |
| 数字滤波附加延迟 | **0 ms**（FIR 已移除） |
| CPU 单步 P99 | 0.477 ms |
| final_test 读取 | 0 次 |

## 最重要的限制

所有指标都是**代理标签**结果。`I_act[k] ≈ I_cmd[k-1]` 是工程假设，
数据集中只有实测线圈电流、没有 ECU 指令记录，因此**真实指令精度未被验证**。
在此之前不得宣称指令跟踪精度已达标。
