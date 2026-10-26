# v19：正向-only → 冻结骨架训逆头 → 可选联合微调

2026-10-27 修订；**未训练、未验证精度、未批准部署**。用户审查 diff 后由 Muse 训练。
审查基线 4c9681ff；以实际上传的 train_v18.py 和 model_v3/v7/v8/v10/v11/v12/v14.py 为准。
LOG_2026-10-22_M208_channel_delay.md 保留作历史记录，其中“42ms 定案”等强表述不作为本次定论。

## 1. 结论和证据

**有条件开训**：先完成 G0，再分别验收正向质量、逆任务覆盖率、周期功和部署耗时。
代码一致性通过不代表精度通过；不再承诺“解析逆解一定能交付稳定闭环”。

历史报告：v18_20261015_221237 最优 E501 pooled 36.24%、逆 MAE 0.464A、loss_fwd 0.0273；
v17 E480 pooled 16.27%。算术为 **2.23 倍、恶化 19.97 个百分点**。
基线 Git 树没有该会话原始 metrics.csv、逐文件评估及运行配置；上述为用户/旧计划报告值，**未逐行核实**。
Muse 补两会话 CSV、run config、checkpoint hash、数据/划分/归一化 hash；
运行 AA02Train/audit_v18_metrics.py --v18 .../metrics.csv --v17 .../metrics.csv 导出实际最优行和来源哈希。
缺失的 metrics 不伪造，旧新数据口径是否可比也需复核。

| 项目 | 实际证据与处理 |
|---|---|
| 训练掩蔽 | AA02Train/train_v18.py:234–237 在重算 NFL/GRU 前清零 I、dI |
| 评估漏洞 | train_v18.py:331 把带真 I 的 u 交给 inverse_predict；训练评估分布不同，0.464A不是部署指标 |
| NFL 电流通道 | model_v12.py:40–80 含原始I、vI、I·tanh、I·v负·abs(v)、I·sign(v)、aI、I·abs(a)、Ix·slow等，必须在NFL前处理 |
| 原正逆映射 | model_v14.py:92–122,131–164 正向fast heads用瞬时/滤波电流，逆解用[Ie,Ie]及另一历史提取的骨架；只检查G单调不足以证明完整F(I)单调 |
| 下限更正 | model_v14.py:147–151已给逆解加入c_min、s_min；撤销基于本地旧导出包的“漏下限”判断 |
| CurrentLag | model_v10.py:43–56升序核直接用于conv1d交叉相关，当前样本乘最小核系数，不是注释的EMA；新候选改为显式状态递推 |
| 退化原因 | 梯度冲突、不可辨识监督、错位数据、滞后实现、规模/初始化均为候选解释，没有消融不能写定论 |
| 参数量 | 缺运行配置，不能确认历史v18为138,512。上传骨架指定64/128/64、5基函数实例为138,705，不是历史checkpoint计数 |

## 2. 状态定义与架构修订

新增 mr_nlcsnn/model_v19.py，architecture=nlcsnn_v19_strict_past_v1。旧模型/权重不覆盖。
这是架构语义改变，不能加载旧权重就假装同一模型，需重新验证精度。

- I_t：本拍待求、尚未施加的物理电流，0–1.6A，归一化除以1.6。
- 已知历史：截至t−1实际施加电流；若部署以命令代实测，须另评估驱动器偏差。
- GRU状态h：用当前运动/温度和上一拍已施加电流组成35维NFL更新，dI恒0，当前未知I和F_des不进骨架。
  不再清除全部已知历史。运动本身可能含电流的物理作用，不等于直接泄漏待求标签。
- CurrentLag状态j：上一拍有效物理电流；j_t=j_(t−1)+α(I_t−j_(t−1))，
  α=1−exp(−dt/τ)，dt=1ms。h、前电流、j逐拍携带。
- 教师I*：冻结context下，对同一force_at求得的有界条件二分解，是模型解，不是真实执行器保证解。
  不可达目标裁剪并标记；平坦零速时的约定电流不是真实唯一解。

prepare不提交电流状态；force_at/二分可以反复查询；commit(actual_applied_current)每拍只一次。
c/s fast heads仅用已知前电流及lag，候选电流只经同一lag→G映射进入力，固定context的F(I)单调。
自由b保留，**不保证无源、不加Fv投影**。G输入为硬度q=1−j，G(0)=0、G(1)=1；
不能写“物理电流1.6A时G=1”。reset h/前电流/lag为0，首1s实测历史burn-in不计分，
不宣称这个reset等于实际初始状态。

训练和普通验证统一strict-past实测历史，标明teacher-forced-history。
audit_v19_rollout.py在burn-in后仅用自身输出更新历史：逆头100Hz并保持电流，observer/正向1kHz。
这仍是记录运动下的rollout，不是车辆闭环。二分教师是1ms条件逆解，不是未来10ms力跟踪优化器；
10ms保持误差单独报告，部署前做CarSim闭环。

## 3. Stage 0：原始列对齐与G0

prepare_v19_data.py先平移原始列，再调用原parser做物理转换、滤波、分段、零点及导数重算。
正shift定义F_aligned[k]=F_raw[k+shift_F]，I同理，x/温度不移；取公共有效区间，无循环/补零/插值。
旧parser默认行为不变。原始档案不改、已有输出不覆盖。
逐行保存source_x/F/I_index（去表头后0-based数据行）；manifest保存源/输出SHA256、dt、
原长度、公共区间、分段/裁剪、零点前后值与变化。旧parser索引改称aligned_idx；
映射原x需加common_start，F/I再加对应shift。COMPLETE.json最后写，半成品禁止训练。
首尾1s训练/评估均排除。离线零相位滤波不能直接用作在线滤波。
excluded/duplicate仅保留manifest与原始档案，不生成训练CSV，避免已排除坏文件阻断整个导出。

**不要求“旧解析长度−42”**：ST/RLF/RHF电流触发分段会随移位变化，
运动分段、静态窗与物理估计零点也必须重新计算。允许长度变化，要求原始索引映射闭合。

预声明统一网格：(F,I)=(42,42)为中心，公共移位(40,40)/(44,44)，
相对同步(42,40)/(42,44)，保留(0,0)基线。
不逐文件挑最优；只在训练/验证文件选择，最终测试不调时移，G0记录并锁定。
**42ms仍是待核验假设**，不是硬件原因或F/I严格同步已证明。

G0记录：
1. F/x/I单位、极性、来源、静态窗/滤波，平移前后相位/周期功及F/I相对响应。
2. M208通道配置和可取得的同步校验证据。另一支MTS减振器只作定性参照，不当相位真值；
   ST电流边沿不要求与运动启动重合，不人为对齐到重合。
3. 对齐后ST重新估计有效τ；11ms只是初始化候选。旧卷积参数不能直接解释为物理τ，
   不能靠大τ补偿未修正标签错位。
4. 锁定划分；G0 JSON含passed、data_manifest_sha256、reviewed_by、evidence、
   chosen_shifts、tau_rationale。runner核对passed及hash；证据内容仍需人工审查。

## 4. 训练、验证、最终测试

AA02Train/v19_split.json给出可执行定义。原8个holdout改称validation用于选checkpoint；
新增SH_{5,10}mm_6.62Hz_0.4A_r{1,2}作高频验证哨兵。
以下10个原始文件从训练移入最终测试：

    SH_5mm_6.62Hz_0A_r1.csv       SH_5mm_6.62Hz_0A_r2.csv
    SH_5mm_6.62Hz_0.8A_r1.csv     SH_5mm_6.62Hz_0.8A_r2.csv
    SH_5mm_6.62Hz_1.6A.csv
    SH_10mm_6.62Hz_0A_r1.csv      SH_10mm_6.62Hz_0A_r2.csv
    SH_10mm_6.62Hz_0.8A_r1.csv    SH_10mm_6.62Hz_0.8A_r2.csv
    SH_10mm_6.62Hz_1.6A.csv

去_rN工况名与raw SHA256做传递分组，同组不跨split，哈希副本仅一份。
本地301项manifest只读检查：train259、validation16、final_test10、excluded9、duplicate7；
云端按其hash重验，不硬编码计数。验证集也纳入原工况的重复文件，故可能大于12。
这是相对v19重新训练的预留集；旧模型/前期时延调查可能见过，不能称全研究从未接触的盲测。
外部泛化须新采集。时移/架构/超参/checkpoint全部锁定后单独--evaluate-final-test，
测试后再调参须披露测试已消耗。归一化和灵敏度标尺只用训练集。

## 5. Stage 1：正向-only

GRU2×16、MLP48、G24、5基函数，实测计数21,041参数，自由b保留。
正向loss按文件p2p归一化平方误差，负力权重1.5、ST文件权重3，逆loss=0。
AdamW lr3e−4、weight_decay1e−4、clip1、seed7、5轮warmup后cosine。
runner每批一个完整文件，无随机块丢历史，无电流噪声增强；首尾1000点不计分。
不声称复现v18的batch12/chunk12000。内存不足若Muse加TBPTT，必须携带和正确detach状态。

冷启动为主候选，先做相同30轮预算、种子、数据、验证频率的冷热短消融。
热启动显式转移匹配GRU/slow heads并记录来源hash/tensor map，lag/G/fast/inverse新初始化。
不同语义不能仅因形状相同就转移；64宽v18不静默裁为16宽。
两臂共用训练集新归一化，验证集选择方向，不必两臂都训满500轮。
τ由对齐后ST复核，可--tau-ms指定并存model_cfg。正向通过工程审查再进入Stage2。

## 6. Stage 2：冻结骨架，只训逆头

只有inv_head可训练；相同初始逆头分别做true/teacher两臂，各30轮、lr1e−4，不默认1:1混合。
uI=I/1.6A，f=F/Fscale，velocity=uV×1000mm/s。
固定Stage1 reference的训练集首尾1s外、|v|≥10mm/s、固定每10点取1点，
计算s=|∂F/∂I|（N/A）；全训练样本p90为s_ref，存checkpoint。
不从验证/测试或每文件最大值重标定；s_ref≈0停止，前向不支持可靠逆监督。
autograd对完整force_at求导，含lag/G和量纲；用完整F(I)有限差分验证，不能只用G导数。

w=clip(s/s_ref,0,1)。eligible=有效窗且|v|≥10mm/s且w≥0.2，
且目标在reference的0–1.6A端点可达范围内、端点力差非平坦。
10mm/s和0.2是工程起点，不是普适物理定理。
固定reference报告{5,10,20}mm/s×{0.1,0.2,0.3}覆盖率；主结果固定(10,0.2)，不按最终测试挑阈值。
mask、w、teacher全部stop-gradient；Stage3也固定reference，禁止通过缩小灵敏度逃避loss。

每文件loss=sum(eligible*w*(I_pred_norm−target_norm)^2)/sum(eligible*w)，
对有eligible点文件等权平均，不叠加ST/负力权重；空文件剔出计数、整批空返回零及零覆盖。
true标签为清洁实测I/1.6；teacher为同reference的条件逆解。
低速/低灵敏度/不可达不监督唯一电流、不算可辨识MAE；
仍报告全样本MAE、力重构误差、饱和率、低速/不可达比例、逐文件覆盖率。
不能把不可达裁剪标签混成可达样本。平坦点约定电流不是唯一真值。
参考目标可辨识MAE≤0.15A，同时覆盖率不能坍缩；误差与覆盖率并列，
不能靠剔除困难点宣称提高。runner拒绝零验证覆盖，其余工程覆盖率/10ms rollout仍需审查。

## 7. Stage 3：可选，默认不运行

从选定Stage2开始，≤30轮、lr3e−5，L=L_forward+0.02L_inverse。
0.02是预声明起点，不是验证过的最优；两项归一化不同，需记录loss和梯度量级。
cycle只审计、不进loss，保留真实F监督，避免正逆共同拟合自身错误。
每轮验证，相对固定Stage1，pooled或0.4A高频验证NRMSE任一恶化>1pp，
终止joint并保留输入Stage2。1pp是风险预算，不是显著性界限；
多seed/逐文件仍需检查，不能让pooled掩盖关键工况退化。

## 8. 无源与部署审计

负周期功先核对F/x坐标和功符号、N/mm到SI的1e−3、dt、重复平移、滤波/导数相位、
完整周期边界、残余位移和偏置功F0·Δx、GRU/lag初始化及重复推进、
暂态/热漂移、训练覆盖/外推、比较时是否同一电流历史。
∮F dx与∫Fv dt互验；固定电流到稳态，报告逐周期功和积分容限。
短时Fv<0允许；有储能变化/未稳态时单周期负功不能直接认定采集错误。
确认稳态持续负净耗散后再分解c/s/b功贡献，不先改权重或截断力。
审计SH1.66/4.97/6.62Hz及OOD0.5mm、0A、30°C、2/8/12Hz，域内/OOD分开；
容限来自积分收敛与测量不确定性，不拍定某个J数。

参数量、精度、耗时独立验收。PC Python/MEX不能写成S32K14448MHz实测。
NI测四角正向1kHz+逆向100Hz、通信/调度总负载、最坏/分位耗时和超时数；
迁S32须目标板编译计时，未测标估算。本次没有精度/硬件/闭环通过结论。

## 9. 交接

- prepare_v19_data.py：原始列对齐、不可覆盖输出、可追溯split。
- audit_v18_metrics.py：历史取证；缺失数字不补造。
- train_v19.py：默认不训练，显式--execute-training、G0记录、新目录才运行；
  Stage2要求Stage1、Stage3要求Stage2 checkpoint，数据/归一化必须一致。
- audit_v19_rollout.py：记录运动自输出历史，1ms observer/10ms保持。
- tests/test_v19_model.py、test_v19_data.py：数值回归，不执行optimizer.step。

本次交源码、计划和测试；没有全量生成对齐数据、没有训练、没有改旧权重或替换CarSim。
Muse待用户审diff后补metrics、完成G0、消融及分阶段训练，再做周期功/覆盖率、
10ms rollout、独立最终测试、闭环和目标板验收。
