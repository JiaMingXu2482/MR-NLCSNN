# 10 ms 逆解：运动、上下文、求根与电流平滑分解

2026-10-29。沿用提交36793e6的A权重及全部四组合成工况。仅新增实验代码与结果，未训练、未改任何权重/标签、未读数据集CSV、未替换CarSim/Simulink。

## 结论

用户的核心判断成立：旧10 ms冻结逆解的10–43 N误差不能作为自洽测试通过。完整正模型在已知电流处的目标100%可达，冻结代理的“不可达”是代理范围失真。

实测结果不支持“只改v+a*t就解决大部分问题”：真实未来速度只把四组MAE从10.67/43.12/7.47/10.75 N降至7.72/34.33/5.65/8.53 N。保留GRU滚动同样重要。

真实未来运动+完整模型+9点插值达到0.528/1.105/0.283/0.237 N，但真实未来运动在线不可得。进一步离线自适应网格找到全部400窗的根，最大力残差小于6e-6 N；这是自洽参照通过，不是可部署逆解通过。

**常加速度预测+完整滚动仍有明显误差。** 独立窗口同历史下1.33/16.50/2.82/6.55 N；携带自己实际电流与状态后4.83/41.26/3.91/11.02 N。只看前者会高估连续运行表现。

正则和限速必须独立验收。lambda=10 N/A、增量0.2 A/10ms使电流更平滑，但连续混合多正弦力MAE升至85.11 N，不可接受。这是实验平滑参数，不是新增执行器物理限幅。

## 口径和自洽参照

- 用户表里的4.8%/8.9%/11.8%/11.7%对应RMSE/目标力RMS，不是MAE/RMS；本报告以NRMSE明确命名。
- 目标由同一A正模型、同一已知历史、真实未来运动、已知保持电流生成。所有逆解器均未获得真实待求电流作为初值或候选；它只用于目标生成和评分。
- FP64/CPU单线程用于降低求根数值噪声，原FP32权重数值不变；与原FP32目标最大差为 0.000292996 N。内部旧方案复算与先前10.67/43.12/7.47/10.75 N一致。
- 7项实现检查全部通过：非单调求根、真实电流处可达、完整序列与逐步提交一致、查询不改原状态/权重、平坦映射保留电流、限速生效。模型仍在候选副本上每1ms更新历史dI/GRU/CurrentLag。
- 四组各100窗，共400窗；没有弃掉高误差工况。完整结果、换向窗统计、灵敏度和饱和均在summary.json/windows.json。

## 第一阶段：相同已知历史，隔离近似误差

| Method / case | Force MAE N | I MAE A | TV / true TV | Saturation |
|---|---:|---:|---:|---:|
| old_frozen_window / multisine_low | 10.6674 | 0.2714 | 1.955 | 23.0% |
| old_frozen_window / multisine_mixed | 43.1180 | 0.4544 | 4.668 | 43.0% |
| old_frozen_window / sine_8Hz | 7.4719 | 0.5138 | 5.093 | 52.0% |
| old_frozen_window / sine_12Hz | 10.7489 | 0.5902 | 7.462 | 69.0% |
| true_v_frozen_csb / multisine_low | 7.7227 | 0.1987 | 2.086 | 14.0% |
| true_v_frozen_csb / multisine_mixed | 34.3324 | 0.3872 | 4.135 | 26.0% |
| true_v_frozen_csb / sine_8Hz | 5.6533 | 0.3902 | 5.885 | 29.0% |
| true_v_frozen_csb / sine_12Hz | 8.5313 | 0.4547 | 8.014 | 36.0% |
| ca_v_frozen_csb / multisine_low | 7.9488 | 0.1992 | 2.086 | 13.0% |
| ca_v_frozen_csb / multisine_mixed | 33.8490 | 0.3780 | 4.217 | 29.0% |
| ca_v_frozen_csb / sine_8Hz | 6.8327 | 0.4445 | 6.137 | 28.0% |
| ca_v_frozen_csb / sine_12Hz | 12.9175 | 0.5940 | 8.476 | 34.0% |
| full_true_motion_9 / multisine_low | 0.5280 | 0.0371 | 1.224 | 0.0% |
| full_true_motion_9 / multisine_mixed | 1.1053 | 0.0239 | 1.089 | 0.0% |
| full_true_motion_9 / sine_8Hz | 0.2828 | 0.0570 | 1.548 | 0.0% |
| full_true_motion_9 / sine_12Hz | 0.2366 | 0.0459 | 1.417 | 0.0% |
| full_ca_motion_9 / multisine_low | 1.3264 | 0.0748 | 1.594 | 2.0% |
| full_ca_motion_9 / multisine_mixed | 16.4993 | 0.2381 | 3.049 | 9.0% |
| full_ca_motion_9 / sine_8Hz | 2.8218 | 0.2868 | 2.551 | 2.0% |
| full_ca_motion_9 / sine_12Hz | 6.5544 | 0.3799 | 4.352 | 9.0% |


`true_v_frozen_csb`只提供真实未来速度，c/s/b仍固定。`ca_v_frozen_csb`用v0+a0*t且仍固定系数。`full_ca_motion_9`完整推进GRU与电流历史，输入x0+v0*t+0.5*a0*t²、v0+a0*t、a0（及当前温度），无需未来测量。9点插值不是精确求根；`*_refined`结果也全部保留。

窗口运动在t0..t0+9ms取样，电流EMA对应第1..10次更新，避免错一拍。独立窗口方法使用教师历史，仅用于误差来源隔离。

## 第二阶段：连续携带自己的实际电流与状态

所有方法从相同1秒预热结束状态出发，每个10ms窗只用起点测得的x/v/a；实际正模型在外生真实运动下推进10次，成为下一窗自己的历史。这里仍是外生运动模型回放，不是车辆闭环。

| Method / case | Force MAE N | I MAE A | TV / true TV | Saturation |
|---|---:|---:|---:|---:|
| own_history_full_ca9_lambda0 / multisine_low | 4.8268 | 0.2165 | 2.220 | 5.0% |
| own_history_full_ca9_lambda0 / multisine_mixed | 41.2646 | 0.3960 | 4.822 | 19.0% |
| own_history_full_ca9_lambda0 / sine_8Hz | 3.9141 | 0.3628 | 2.662 | 3.0% |
| own_history_full_ca9_lambda0 / sine_12Hz | 11.0216 | 0.4229 | 5.411 | 23.0% |
| own_history_full_ca9_lambda2 / multisine_low | 4.8711 | 0.2197 | 2.202 | 5.0% |
| own_history_full_ca9_lambda2 / multisine_mixed | 38.0154 | 0.3705 | 4.427 | 16.0% |
| own_history_full_ca9_lambda2 / sine_8Hz | 3.9109 | 0.3620 | 2.647 | 3.0% |
| own_history_full_ca9_lambda2 / sine_12Hz | 9.4481 | 0.3492 | 4.039 | 9.0% |
| own_history_full_ca9_lambda10 / multisine_low | 5.7000 | 0.2742 | 1.788 | 5.0% |
| own_history_full_ca9_lambda10 / multisine_mixed | 38.6392 | 0.3427 | 3.455 | 12.0% |
| own_history_full_ca9_lambda10 / sine_8Hz | 2.9632 | 0.2546 | 1.804 | 2.0% |
| own_history_full_ca9_lambda10 / sine_12Hz | 11.7665 | 0.3342 | 2.678 | 10.0% |
| own_history_full_ca9_lambda10_rate0p2 / multisine_low | 7.5249 | 0.2287 | 0.761 | 0.0% |
| own_history_full_ca9_lambda10_rate0p2 / multisine_mixed | 85.1071 | 0.4164 | 1.532 | 10.0% |
| own_history_full_ca9_lambda10_rate0p2 / sine_8Hz | 8.3102 | 0.5499 | 1.212 | 4.0% |
| own_history_full_ca9_lambda10_rate0p2 / sine_12Hz | 9.1647 | 0.2385 | 1.270 | 2.0% |


目标是绝对力误差(N)+lambda(N/A)*电流增量(A)。lambda=0/2/10在看到结果前固定，全数报告。0.2 A限速可能拦截首窗真实命令跳变；其四组初始跳变为 0.119, 0.387, 0.535, 0.584 A，首窗与余下99窗分别在rate_initial_transition.json，不能把首窗影响隐藏。此试验不以四组合成结果选择最终lambda。

## 第三阶段：离线oracle补查网格漏根和多根

这一补查由第一阶段失败触发，预先登记在ORACLE_SUPPLEMENT.md，与原PLAN.md分开记录。9点网格有43窗没发现括区间；只对这些窗口逐级33/129/513点加密，剩余6/2/0窗。24次二分只在发现的局部括区间内进行，不假设全局单调。

| Case | Force max residual N | I MAE A | TV ratio | Multi-root windows |
|---|---:|---:|---:|---:|
| multisine_low | 1.15e-06 | 0.0305 | 1.251 | 50/100 |
| multisine_mixed | 5.69e-06 | 0.0195 | 1.155 | 34/100 |
| sine_8Hz | 3.87e-07 | 0.0486 | 1.554 | 90/100 |
| sine_12Hz | 7.26e-07 | 0.0428 | 1.445 | 75/100 |


力残差在0.001 N数值等价带内时，选择已发现根中最接近上一实际电流的根，防止小数值差异把解切换到远根。这不是正则化力误差，也不保证找到所有根或全局最近根。8Hz的TV比1.554仍略高于1.5，因此不能宣称所有指标通过。

**相同窗均值不一定唯一确定电流。** 已找到同历史下的两个电流：8Hz第6窗，真实1.168625 A与另一解0.049520 A，均值力都约−118.790735 N，后者残差1.74e−8 N。两个解可有不同逐点波形和结束内部状态。此例既证明力误差能接近0，也说明电流恢复与分支连续性需要单独评价；不能把全滚动映射也当作严格单调。

## 验收与下一步

同意用户提出TV<=真实1.5倍、饱和<10%和力误差预算作为工程筛选，但它不替代oracle接近0的数值验收；7 N、10 N两档均报告，同时检查此前电流MAE<=0.1 A。

本次没有可宣布全面通过的在线方案。下一步重点是因果运动预测（尤其高频/换向时的加速度演化）和多根连续选择，而不是继续放宽力误差或靠更强限速掩盖不准确的力映射。新增网络/训练前应先有这两个环节的独立证据。

NI未实测。本次运行耗时是CPU/PyTorch FP64离线批处理，不能据此宣称9候选方案满足NI 10ms周期；计算量估算也不能替代目标机测量。

## 复现

权重沿用 `training_outputs/v19_capacity_four_20261028/training_delivery/sessions/A/final.pth`，SHA256 `3a407a77e8f0b9167282fdbfc77923dac23fcd0430fa498432e36e31f97423c7`。只需要PyTorch；报告绘图另需matplotlib。冻结模型和旧基线依赖放在frozen_dependencies/，无训练/CSV加载入口。

```sh
python run_motion_audit.py /path/to/A/final.pth --output replay_results
python refine_oracle.py /path/to/A/final.pth --results replay_results
```

拒绝覆盖已有运行结果。results/executed_source/保留实际产生首轮及oracle指标时的源码；根目录脚本仅增添便携路径/命令行参数，模型与实验逻辑一致。哈希见SHA256.json与各summary的source字段。
