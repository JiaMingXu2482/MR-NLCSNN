# v20 时延分析与应用范围决策归档

先读 [决策日志](DECISION_LOG_2026-10-29.md)，再读 [统计口径](README.md) 和 [固定速度下 ΔI 趋势](delta_i_trend/README.md)。

本次分析不训练，不修改权重，不访问最终测试CSV。A/E最终选型尚未锁定；后续优先评估小位移、|v|≤0.3m/s。

- 原始分析代码保持字节一致；路径仍为运行机器上的绝对路径，迁移运行需配置数据/权重位置。
- `predictions/` 是33个非最终测试ST文件的A/E输出，float32 little-endian，记录了权重与数据出处；不含权重本体或训练CSV。
- `inference_source/` 是与执行时哈希一致的两份模型/数据预处理源码。
- `ARCHIVE_MANIFEST.json` 记录每份复制产物的SHA256；未复制旧SHA256清单和Python缓存。
- 完整重跑 analyze_delay.py/predict_st.py 需要相应数据及权重；已有事件JSON足以重画 ΔI 趋势，不需重新推理。
- 旧 v20 训练归档在 ../v20_training_20261029；本目录为新增独立分析，不覆盖旧结果。
