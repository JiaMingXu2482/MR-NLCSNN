# v19 交接与本次测试记录

2026-10-27；基线4c9681ff，目标分支train/nlcsnn-v3-5pct。
**没有启动训练，没有调用optimizer.step，没有改旧模型权重，没有做部署替换。**
完整方案见[TRAIN_PLAN_v19_joint_fwd_inv.md](TRAIN_PLAN_v19_joint_fwd_inv.md)。

## 已完成

- 原始F/I列平移发生在滤波、分段、零点之前；不可覆盖输出，源/处理后哈希与逐行索引可追溯。
- 明确了原8文件为验证集、10个6.62Hz最终测试文件，以及重复工况/文件哈希隔离。
- 新v19骨架保留35维NFL和自由b，21,041参数；当前未知电流不进入骨架，保留已知历史。
- 正逆共用force_at；候选查询不推进状态，实际电流每拍提交一次；修正新候选CurrentLag。
- 分阶段冻结、完整逆loss、固定reference掩码/灵敏度、独立true/teacher两臂、验证回滚。
- 默认关闭训练的runner，以及1kHz observer/100Hz逆头自输出历史的只读rollout工具。
- 旧train_v18/model_v14等源码保留，便于复现审查；不会悄悄修旧源码后把旧指标当新指标。

## 测试范围与结果

12项PyTorch模型/协议测试通过：实际NFL等价、当前/未来电流隔离、步进/分块/整段一致、
EMA递推与旧卷积差异、同context逆解、单调条件映射、导数有限差分、低速空掩码、
冻结/参数预算、显式热启动映射、10ms保持与无真电流历史泄漏、纯评估权重不变。
另核对冻结骨架后逆loss只对逆头产生有限梯度，不更新参数。总计12个unittest方法。

4项数据测试通过：正负时移原始索引、不修改原表、哈希/工况隔离与冲突拒绝、
ST分段随原始电流移位而变化（输出长度不必旧长−42）、metrics最佳有限行取证。
这些也是4个unittest方法，部分含多条断言。

实际manifest只读检查301项：训练259、验证16、最终测试10、排除9、副本7。
只核对了清单和分组，未生成全部对齐数据。云端须重新核对来源哈希。
新/改动Python文件AST语法检查通过；runner用不存在的数据路径不加执行开关时退出提示“不训练”。
没有跑epoch，也没有验证optimizer执行、收敛、完整数据质量、预测精度、速度或车辆闭环。

旧CurrentLag数值证据：τ初始化11ms、输入160点单位阶跃，正确EMA首点约0.0868993，
上传旧实现首点约0.000215402。这证明实现不等价，**不证明它单独造成v18全部退化**。
旧骨架指定64/128/64、5基函数实例参数138,705；不能用它替代缺失历史checkpoint的参数实计数。

本机分开运行两组测试：PyTorch 2.11环境缺NumPy（警告不影响torch-only测试）；
NumPy/Pandas/SciPy测试用另一环境。Muse统一安装torch、numpy、pandas、scipy、tqdm后可运行：

```bash
python -m unittest discover -s tests -p 'test_v19_*.py' -v
python AA02Train/train_v19.py --data unused
```

## Muse 接手顺序（用户审完diff后）

1. 补原始v18/v17 metrics与配置：当前Git树没有目标会话metrics，36.24/16.27/0.464仍注明来源未复核。
   用audit_v18_metrics.py导出最佳行与SHA256，补逐文件/数据口径证据。不要把注释当运行配置。
2. 按统一时移网格准备新目录，原始数据不覆盖。例（不是本次已执行命令）：

   `python AA01DataProcess/prepare_v19_data.py --archive Raw_Card_Categorized_All --output Parsed_v19_F42_I42 --force-shift 42 --current-shift 42`

3. 完成G0并锁定manifest hash/时移/τ理由，审查原始传感器同步与零点变化。
4. 冷热30轮同预算短消融，选Stage1；true/teacher Stage2分开；Stage3默认不开。
   runner需要显式--execute-training、--g0-report、--output；Stage2/3另需前一阶段--checkpoint。
   Stage3最多30轮，Stage2可按计划30轮。输出目录必须新建，不覆盖旧会话。
   本次没有生成passed=true的G0文件，也没有替用户批准数据。
5. 核对逐文件覆盖率/误差及灵敏度网格。程序只硬拒绝零验证覆盖，不自动认定其它覆盖率合格。
6. 用audit_v19_rollout.py评估validation的head与bounded两条路线；两者都只用当前目标力，
   固定当前运动下1ms可逆不代表10ms保持后力仍准确。记录运动rollout也不等于CarSim闭环。
7. 参数/选择锁定后单独--evaluate-final-test；最终测试不挑checkpoint。
8. 周期功分量、符号/单位/滤波/初态检查通过后，再做CarSim和目标硬件验收。

新模型G作用于滞后有效电流、fast heads作用于已知历史，区别于v14的瞬时/滤波双旁路。
它是为正逆一致性设计的新候选，准确率可能提高也可能下降；Stage1是实质性门禁。
本次测试不能支持“训练必达5%”“一定无源”“已经能在S32/NI实时运行”等结论。
