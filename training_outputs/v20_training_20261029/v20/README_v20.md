# v20 训练代码（本地手动运行版）

## 这版到底是什么

v20 正向模型消融，输入固定为 **[x, v, a, I, T]**，去 dI（逆模型不方便，2026-10-29 已定）。
**时延默认交给 GRU 隐状态去记**：没有任何 CurrentLag、TauNet、单独的 τ 参数，
也不给 τ 设训练目标。2026-10-29 的结论是单独训一个 τ 是错的——时延分布在
驱动器斜坡、阀动态、GRU 状态和系数里，力损失下不可辨识，单拎出来训只会
得到一个随定义而变的数。

这版先只做**正向**消融，逆模型不在本轮训。最终测试集（10 个文件）全程不打开，
只用来以后在消融胜出者锁定之后、由你决定时再验，绝不用来选模型。

## 消融模型：按你和 GPT 的聊天内容定

字母是 v20 的编号，**不是** v19 的输入消融 A/B/D，别混：

- **A raw_gru**：原始输入 → GRU(16×2) → MLP。GPT 的动态基线，测 NFL 有没有用。
- **B nfl_gru**：NFL(31维) → GRU(16×2) → MLP。GPT 的串行方案 A。
- **C nfl_mlp**：NFL → MLP，无 GRU。测没有记忆行不行。
- **D parallel**：原始输入 → GRU（时序支）+ 当前 NFL 支 → MLP 融合。
  GPT 最后倾向的并行结构。注意：GPT 的“GRU 先、再融合 NFL”方案，一旦按它自己
  的警告（NFL 只能由当前物理输入算，不能往隐状态上套 tanh）去实现，就会和
  并行结构算成同一个东西，所以本轮只训一次，不重复占 GPU。
- **E nfl_small_gru**：NFL → GRU(8×2) → MLP。GPT 的压缩方案，测能不能再压小。
- **F csb_nolag（可选，不在并行默认列表里）**：v19 式 c/s/b + 单调 G 力公式，
  但**去掉 CurrentLag**，q = 1 - 当前电流。单独跑它，才能回答“删掉 c/s/b
  是不是更好”，而不是凭感觉删。

所有 GPT 模型用同一套东西：同数据、同划分、同归一化、同 seed 7、同 30 轮、
同 AdamW（lr 3e-4、cosine、前 5 轮 warmup）、同全文件 batch、MLP 单隐层 16 个
ReLU 单元（GPT 建议的 8–16 起点）。F 例外，头保持 v19 的 48→24，否则它就
不是 v19 公式的对照了。

## 本地准备

1. 把 v19 的数据包解压到本地，训练代码要的是 **Parsed_v19_F42_I42** 这个
   目录，里面必须有 `alignment_manifest.json`、`COMPLETE.json` 和各工况 CSV。
   不要指到原始 Card 数据，也不要自己重做对齐。
2. Python 环境只要 torch（按你显卡装对应 CUDA 版即可），其余全是标准库。
   代码在 `v20_data.py` 里会校验 manifest 哈希、去重原始哈希、处理后文件哈希、
   1kHz 采样、电流幅值 ≤2.10A，任一项不对就报错停下，不会静默用错数据。
3. 把本目录（`v20_models.py`、`v20_data.py`、`train_v20.py`、
   `run_v20_parallel.py`）放在同一文件夹里运行，不要只拷一个文件。

## 运行命令

并行跑 A–E（默认，Windows cmd / PowerShell / Linux / macOS 通用，`^` 是
Windows cmd 的续行符，PowerShell 请用反引号或写成一行）：

```
python run_v20_parallel.py --data D:\你的路径\Parsed_v19_F42_I42 --output-root D:\v20_runs --epochs 30 --execute-training
```

不加 `--execute-training` 时只打印计划，不开训。输出目录已存在会报错，
绝不覆盖旧结果，请换新目录名。

单个模型、或单独跑 F：

```
python train_v20.py --data D:\你的路径\Parsed_v19_F42_I42 --model B --output D:\v20_runs\B --epochs 30 --execute-training
python train_v20.py --data D:\你的路径\Parsed_v19_F42_I42 --model F --output D:\v20_runs\F --epochs 30 --execute-training
```

并行占 GPU 或跑不动时，改串行（命令一样、一个一个跑）：

```
python run_v20_parallel.py --data D:\你的路径\Parsed_v19_F42_I42 --output-root D:\v20_runs --epochs 30 --sequential --execute-training
```

断点续训（本地关机、崩溃后接着跑，`--epochs` 是**总轮数**，不是追加轮数，
且模型、数据、seed 必须和原 session 一致，否则报错）：

```
python train_v20.py --resume D:\v20_runs\B --epochs 60 --execute-training
```

冒烟测试（只验证代码能跑通，**结果不能用来选模型**，输出目录名必须含 smoke）：

```
python train_v20.py --data D:\你的路径\Parsed_v19_F42_I42 --model B --output D:\v20_runs\smoke_B --epochs 2 --max-train-files 2 --max-val-files 2 --execute-training
```

## 结果怎么看

每个模型目录里有 `metrics.jsonl`（每轮）、`best.pth`（验证 pooled 最低那轮）、
`summary.json`（最佳轮 + 验证滞后）。并行跑完后根目录有 `summary_all.json`。

比较顺序，不要只看 pooled 一个数：

1. 验证 pooled NRMSE（主指标）；
2. 6.62Hz 高频 pooled；
3. 分工况 pooled（SH / ST / RLF / RHF / SR，看有没有某个工况系统性垮）；
4. 最差文件、验证最佳滞后（`validation_lag_ms`，只在最佳模型上算一次，
   不是训练信号）；
5. 参数量。

本轮是单 seed（7），先看方向，不下最终结论。胜出者以后要补多 seed 再定，
最终测试集更不能在本轮打开。

## 已知限制，老实写在前面

- 正向是 teacher-forced 历史（整文件 GRU 推进），不是闭环 rollout 精度；
  闭环要等逆模型出来、按 100Hz 指令保持 10ms 喂 1kHz 正模型那样验。
- 并行是多个进程共享一块 GPU，不是每模型独占显卡；小模型才敢这样写，
  大模型或显存不足时用 `--sequential`。
- 训练耗时不在此承诺：看第一轮日志里的秒数再估，不要按 v19 在别的机器上的
  数字去套本地。
