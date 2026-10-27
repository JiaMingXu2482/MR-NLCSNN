# v19 A/B/D 本地训练交付

先看 [结果报告](results_summary_ABD/README.md)。

三个会话都位于 sessions 文件夹：
- v19_A_seed7_30epochs：本次追加完成的全输入 A，30轮。
- v19_B_seed7_30epochs：此前完成的 B，原文件直接复制。
- v19_D_seed7_30epochs：此前完成的 D，原文件直接复制。

每个会话包含配置及源码/数据哈希、完整30轮 metrics.jsonl、best.pth、latest.pth、最佳验证指标和PC CPU计时。
results_summary_ABD/code_snapshot 包含实际运行源码、runner差异、原补丁、G0、划分及检查记录。

固定同一协议：259训练文件、16验证文件，seed7冷启动，Stage1，30轮，RTX4060Ti FP32。最终测试10个CSV未打开，未推送GitHub。所有checkpoint的逆头均未训练。
run_config中的绝对路径为原运行位置，保留用于溯源。代码副本位于code_snapshot；在别的电脑使用时需配置源码和数据路径。
