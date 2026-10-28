# v19 四模型训练与诊断归档，2026-10-28

先读 [交接日志](../../docs/LOG_2026-10-28_v19_capacity_bias_periodic_work.md)。这是 **Stage 1 正向模型** 的完整结果快照，逆头尚未训练，未部署闭环，不要把 final.pth 理解为正逆联合训练完成。

## 阅读顺序

1. [训练结果与定义](training_delivery/README.md)、[机器可读汇总](training_delivery/comparison.json)。
2. [四组模型代码](code/model_v19_capacity.py)、[D 模型](code/model_v19_bd.py)、[实际训练器](code/train_capacity_four.py)、[实际运行的上游训练器](code/frozen_runtime/AA02Train/train_v19.py)。
3. [初始静态基准预处理](code/prepare_capacity_zero_data.py)、[参考基准审查记录](training_delivery/data_provenance/zero_reference_audit.json)。
4. [偏置审查](bias_audit/README.md)、[残差数值](bias_audit/residual_summary.json)、[仅训练拟合的参考基准探针](bias_audit/training_only_reference_probe.json)。
5. [周期功审查](periodic_work/README.md)、[全部1704组中的主扫描1680组](periodic_work/all_results.json)、[旧数值中心24组](periodic_legacy_center/all_results.json)。

## 路径与范围

- `training_delivery/sessions/{D,A,A_GRU,A_MLP}/`：最终权重、可恢复权重、300轮指标、运行参数、16文件预测及验证指标。每组均同起点规则/同采样序列，不继承旧D权重。
- `training_delivery/code_snapshot/`：原始字节快照及运行时源码记录。
- `code/`：可阅读的实验、数据处理、诊断和报告脚本；`frozen_runtime/`为实际使用的上游文件，不是远端主线的替代。
- `preprocessing_evidence/`：初始静态段审查、原划分、G0及Muse修复diff。G0对应旧F42/I42数据，新的静态参考数据哈希见新manifest，不能混用。
- `bias_audit/`：完整四模型三文件分支轨迹，另含全部随机类训练/验证冻结推理指标。
- `historical_v17_evidence/`：用于周期功比较的旧日志和确认位移中心的MAT；不是此次重跑v17。
- 旧本地绝对路径保留用于追溯。归档的README中的本地文字路径是历史出处，不要求其他电脑存在这些路径。
- 没有上传用户论文DOCX，也没有新增原始台架CSV或最终测试CSV。数据manifest保留文件名和哈希；最终测试未读取。训练重放仍需要相同哈希的248训练文件和验证文件。

## 不依赖本地D盘的冻结周期功复现

在本目录执行（Python3.11、PyTorch CUDA；实测版本torch2.11+cu128）：

```bash
python code/replay_periodic.py --out /new/path/periodic_main
python code/replay_periodic.py --out /new/path/periodic_legacy --legacy-center
```

`--out`必须不存在，避免覆盖。Windows可使用新建目录的绝对路径。加`--smoke`只运行D/0.5mm/8Hz/0A/30℃一个点用于检查环境，不能代替全量审计。入口先加载归档的上游依赖，再指定checkpoint和输出路径；不改网络、归一化、模型权重，也不启动训练。

其余原始训练/诊断脚本仍保留历史路径。阅读源码或重构路径时，应保留清单哈希、分组、loss、采样序列、初始条件和最终测试隔离；不应直接在云端按旧路径运行训练脚本。

## 完整性

`ARCHIVE_SHA256.json`记录本包每个文件的SHA256及字节数（不包含它自身）。`training_delivery/SHA256.json`保留原交付包校验，两个manifest的作用不同。权重中的source_hashes与训练时快照对应；不能拿修改后的远端主线替代它们而仍称完全复现。
