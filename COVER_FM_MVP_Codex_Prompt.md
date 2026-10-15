# COVER-FM：本地 Codex MVP 实验指令

## 目标与边界

你在我的本地科研仓库中工作：
https://github.com/Kevin11Kaikai/COVER-Selective-Sleep-Staging

请实现并实际运行一个有明确停止条件的 MVP，而不是只写计划或搭空框架。

唯一研究问题：对一个已经冻结的预训练睡眠模型及其五分类任务头，模型内部 embedding 是否在原 COVER 的 37 维概率/分歧特征之外，提供可用于识别错误的增量信息？这种信息能否改善未参与本轮拟合的受试者上的错误排序与校准后接受覆盖率？

不要预设正面结果。区分代码跑通、初步实证支持、论文创新性和临床安全；本任务只能完成前两项。5% 是实验接受错误率目标，不是临床安全阈值。不要声称 population guarantee、首次、SOTA 或实际人工工时节省。

## 1. 先审计，不破坏旧成果

- 遵守现有 AGENTS.md。检查 git status；保留用户未提交的改动。不自动 commit/push，不 reset/clean。
- 先阅读 README.md、REPRODUCIBILITY.md、Scripts/reproduce.py、src/stage1_analysis/risk_features.py、src/stage1_analysis/acceptance.py，以及 New_Objective_Idea/src/new_objective/ 内的损失、求解器和协议。
- 运行原始 reproduction 和已有测试，记录原始 PASS/FAIL，不把原有失败伪装成新代码通过。
- 新代码主要放在 experiments/cover_fm_mvp/，新结果放 outputs/cover_fm_mvp/。冻结论文文件、历史结果、原协议和原数值实现只读。使用独立实验环境，不升级破坏原环境。
- 仅检查当前工作区、已有配置指定目录，以及用户已提供的 DATA_ROOT/CHECKPOINT_ROOT；不要搜索整台电脑或读取无关私人文件。没有路径时生成明确配置项和 BLOCKERS.md，不杜撰数据。
- 用 nvidia-smi / torch 检测实际 GPU、显存、CUDA、可用 dtype 和磁盘；不要假定显卡型号或显存。
- 资源上限：一个预训练 backbone、一张 GPU、默认累计 GPU 运行预算 4 小时、新下载预算 10 GB；不使用付费 API、云 GPU，不从零预训练，不全参数微调，不做 LoRA、Agent、第二个数据集或网页 demo。先测吞吐；超预算时保留可恢复结果并报告，不自动扩张任务。

## 2. 模型和数据：只选一个可核验的组合

优先复用我本地此前 two-stage 实验中的 SleepMaMi 预训练权重与输入管线。注意：我已经做过包含 SleepMaMi 的多专家 cascade；本任务不是把它再次接到 Conformer 后面，而是把一个预训练 encoder 作为主分类器的表征来源，再研究其自身预测的错误。

参考上游：
https://github.com/KeondoPark/SleepMaMi
https://github.com/yang-ai-lab/OSF-Open-Sleep-FM

- 本地只有预测 CSV 不等于拥有 embedding；需要真实数据与真实 checkpoint 做前向，或有可核验来源的 embedding 缓存。
- 首选可复用的 SleepMaMi encoder；如果不满足条件，但本地已有通道兼容的 OSF 数据和 checkpoint，可在看任何效果之前改选 OSF。最多实现一个 adapter；两者均不具备条件时停止科学实验，输出阻塞项，不静默换模型或假数据。
- SleepMaMi 官方 inference.py 提供 extract_micro_features / extract_macro_micro_features 等接口。阅读实际实现与 checkpoint 配置，核验缺失模态、输入长度、归一化和融合模块的支持，不因为函数接受一个 modalities 参数就假定任意单模态一定合法。
- OSF 官方输入为 [B,12,1920]、64 Hz、30 秒，包含指定导联；不能把 Sleep-EDF 的 Fpz-Cz 重命名为 C3-A2，也不能用复制一个 EEG 通道伪造 12 个真实通道。未经支持的导联替代只可标为另一个探索实验，不纳入本 MVP 主结果。
- 现有 COVER 特征函数要求 N×5 概率。任务统一为 W/N1/N2/N3/REM 五类；OSF 发布示例中的四分类不能直接混用。这里训练新的五分类线性任务头，不冒充原论文四分类结果。
- 检查并记录 checkpoint 来源、版本/hash、预训练数据、任何下游微调数据、目标数据集是否被预训练/微调见过。将 overlap 标为 documented_nonoverlap、known_overlap 或 unknown，并写明证据粒度。unknown 不等于无泄漏；known_overlap/unknown 结果只能作相应限定的探索证据。
- 检查所有被使用的 encoder 可训练参数均已正确载入。上游 strict=False 不等于加载成功；逐项列出 missing/unexpected keys，只有明确无用的旧任务头及文档允许重新生成的 buffer 可例外。实际使用的 encoder 参数缺失、shape 不符或保留随机初始化时必须失败。
- 优先使用当前已有 Sleep-EDF-20 数据。必须核验 subject/record/epoch 身份、采样率、导联、单位、五类标签映射、时间对齐和预处理来源。不得强行凑成历史 42,308 条或复制历史主模型概率。
- 如果只能重建新的 EDF 预处理，单独版本化，不声称精确复现旧训练数据；不覆盖旧结果。如果输入不足，完成工程测试并明确科学实验 BLOCKED。

先将决定写入 PREFLIGHT.md 和 provenance.json，锁定模型与样本 manifest；禁止根据测试表现重新挑 backbone。

## 3. 冻结协议：按受试者分角色

在读取任何测试结果之前写出 protocol.yaml、splits.json 及其 hash。

默认：将 Sleep-EDF-20 的 20 个受试者按固定随机种子 20261015 分成 5 组，每组 4 人，同一人的所有夜晚始终同组。运行 5 个外层折。第 k 折：
- TEST = 第 k 组；
- CAL = 第 (k+1) mod 5 组；
- RISK_FIT = 第 (k+2) mod 5 组；
- BASE_TRAIN = 剩余两组。

每人在一个外层折中只作为 TEST 一次。本方案是小样本 development pilot，不是新的独立临床验证；历史上使用过 Sleep-EDF-20 的事实也必须披露。

角色用途严格如下：
1. 冻结外部预训练 encoder，eval() + inference_mode() 导出 embedding。缓存键包含数据、预处理、模型、代码版本及条件 hash。
2. 仅在 BASE_TRAIN 拟合 embedding 标准化和五分类线性任务头；固定超参数，不用 CAL/TEST 选模型。默认 L2 multinomial logistic regression，C=1，max_iter=2000。记录收敛情况。
3. 仅在 BASE_TRAIN 训练辅助 RF；优先复用已核验的 35 维信号特征代码，固定 RF 超参数。不能直接复用曾在本折 RISK_FIT/CAL/TEST 受试者上训练的历史 RF。特征/权重来源不足时阻塞主比较，不偷偷换一套并仍称原 COVER。
4. 主分类器和 RF 冻结后，在 RISK_FIT 得到真实的主模型错误标签 e=1[argmax(p)!=y]，训练风险评分器。
5. 仅 CAL 选择接受规则；随后规则、PCA、标准化、任务头、评分器全部锁定。
6. TEST 只做最终预测与指标汇总，不提供任何训练、调参或规则选择信息。

不把四万条相邻 epoch 当成四万个独立受试者，不随机打散 epoch 划分。

## 4. 最小对照：同一预测，只换错误评分器

对同一折、同一输入条件，所有方法使用完全相同的主分类器概率、argmax 标签及辅助 RF 输出；自动断言这一点。

主表包含五行：
A. MSP：1-max(p) 作为风险分数。
B. 简单 logit 基线：max-logit / 文献 p-norm-normalized max-logit 中选择一个固定实现，在协议里写清公式、方向及来源，不用 TEST 挑选。
C. COVER-37-BCE：复用 heterogeneous_features(p,q)，训练原 37 维 Linear-BCE。
D. COVER-37-FixedScale：在 C 上复用原 FixedScale refinement；不重新发明损失或依据新测试集搜索超参数。
E. COVER-37+Embedding-BCE：37 维特征拼接预训练 embedding 的 16 维 PCA 表示，用与 C 相同的线性 BCE 实现和同级正则化预算学习错误。

E 是唯一新候选。PCA、拼接后标准化仅在 RISK_FIT 拟合；维度固定为 min(16,可用维度)，不在测试上搜索。所有可监督的训练使用 subject-equal sample weighting；正则化、迭代数和初始化先固定。无法直接复用求解器时，写隔离 adapter 并验证与旧实现的数值一致性，不改原文件。

两个轻量控制：
- Embedding-only-BCE，检验只用表征是否已足够。
- 37+shuffled-embedding-BCE：分别在各 split 内、按预测类别分组打乱 embedding 与 epoch 的配对，不利用真实标签；用固定随机种子，重新拟合评分器并独立完成 CAL。此项为描述性负控制，不是正式因果证明。

主信息增量比较：E vs C。
完整方法价值比较：E vs D 及置信度基线。
只赢 MSP、不赢 C/D，不能写成新表征方法已成立。

## 5. 接受规则与指标

主规则统一为 calibration-locked global score threshold，不混用逐受试者 top-k 与全局阈值。可复用原规则接口，但不同规则必须独立命名；本主表必须所有方法一致。

在 CAL 上：对风险分数的固定 0..1、步长 0.01 的分位点及全接收/全拒绝候选，选择满足经验 subject-equal accepted risk <=0.05 的最大覆盖率阈值。使用同一固定 tie 规则；并列时按预注册顺序处理。不满足非空接受条件时返回全拒绝，风险标 NA，不伪装成 0% 安全。

按受试者 j 定义：
c_j = accepted_j / total_j；u_j = accepted_errors_j / total_j。
C_SE = mean_j(c_j)；R_SE = sum_j(u_j)/sum_j(c_j)。
另外报告 pooled risk。不要把 R_SE 写成各人 accepted error rate 的简单平均。

TEST 主输出：
- 全部样本的 accuracy、macro-F1；所有评分器应一致。
- CAL 锁定阈值对应的 TEST C_SE、R_SE、pooled risk、接受人数/epoch 数。
- subject-equal risk-coverage curve 和 AURC；使用预注册的逐人同覆盖率网格用于排序诊断，与部署的全局阈值分开标注。
- 预先固定的 30%、50%、70% 逐人匹配接受预算下的错误数/风险，排序只使用 score，不能看标签选预算。
- 高置信错误 e=1 且 max(p)>=0.9 的被拒绝比例与分母；分母为零时标 NA，不更改 0.9。
- 各真实睡眠阶段的覆盖率和接受错误，尤其 N1，避免仅靠拒绝整个难类取得表面收益。
- 每位受试者一行的结果、错误预测 AUROC/AUPRC；某子集无正/负类则标 NA。
- 真实 encoder 前向耗时、缓存大小、评分器额外耗时、显存峰值。缓存后的 CPU 速度不能当作端到端推理速度；不能把拒答直接换算为 GPU 计算节省。

不能从 TEST risk-coverage curve 反选一个恰好满足 5% 的点，再当成部署结果。原始 risk 超标就如实报告。

在所有预注册折完成后，对配对受试者记录做 2,000 次固定种子的 bootstrap，重算 ratio/差值，报告 E-C、E-D 的描述性区间。明确：这些是已训练模型/固定折下的条件性小样本不确定性，不包含全部训练与模型选择不确定性，也不是总体风险保证。不因区间跨零继续换 seed。

## 6. 只增加一个压力测试

在同一 TEST 受试者上对真实 EEG 通道加入固定 20 dB SNR 高斯噪声。使用仅由 epoch 身份与实验 seed 决定的随机数；相同输入送给所有方法。

噪声在预先规定的预处理位置加入，记录定义和实际 SNR；重新计算 embedding、主模型概率、RF 特征/概率及风险分数，不能使用 clean cache 冒充受扰动输出。分类器、评分器、标准化和 CAL 阈值一律不重新拟合。每个条件内部仍共享相同主预测。

只称其为 synthetic-noise stress test；不称跨医院/真实 domain-shift 验证。不增加第二种扰动，也不根据结果挑 SNR。

## 7. 交付与自检

至少交付：
- PREFLIGHT.md、provenance.json、protocol.yaml、splits.json；
- 可运行的 preflight / extract / fit / evaluate / report 入口和逐步恢复命令；
- per_epoch_predictions.parquet（数据许可允许范围内本地保存，不上传）、per_subject_results.csv、summary.csv；
- 独立的 risk_coverage.png、locked_operating_points.png、stage_coverage.png；
- RESULTS.md：先用不超过 300 字中文说明发现，再给表、区间、限制、失败记录；
- 所有执行日志、退出码、未完成步骤和 BLOCKERS.md（存在阻塞时）。

测试必须覆盖：
- 四种角色的 subject 不相交；同一人所有夜晚归组一致；
- 修改 TEST 标签不改变 checkpoint、PCA、风险 scorer、CAL 阈值或接受掩码；只允许评价指标变化；
- encoder 权重实际加载且冻结；真实/噪声缓存不混用；
- 五类概率行和为 1，类别顺序一致，embedding/epoch 身份严格对齐；
- 同一条件各方法的主预测完全一致；
- 全接收/全拒绝/无错误/无某类别的边界行为；
- 旧 COVER reproduction 和原测试未退化。

## 8. 预先规定如何下结论

ENGINEERING_PASS：真实 checkpoint、真实数据、无泄漏切分、训练、评价和原仓库回归检查完整跑通。

GO_SIGNAL（只是后续投入依据）：在 clean 的五折汇总上，E 相比 C 的 AURC 改善，并且相较 D 或预注册的主要强基线，在双方测试经验 R_SE 均不超过 5% 时获得至少 5 个百分点 C_SE 增益；报告所有折和描述性区间。5 pp 是我的工程投入门槛，不是论文或显著性的通用标准。压力测试结果另行报告，决定能否提出鲁棒性假设。

PARTIAL：COVER 迁移到 FM 有效，但 embedding 增量不足；或错误排序改善但锁定规则下风险超标。明确是哪一项成立，不把它们混为成功。

NO_SUPPORT / INCONCLUSIVE：候选没有超过原 COVER 强基线、效果只在少数人出现、覆盖率几乎为零、区间很宽或负控制同样有效。保留结果，不自动换 backbone、阈值、seed 或指标找正面结论。

BLOCKED：数据、通道、checkpoint、来源或环境阻止真实实验。合成数据只可用于单元测试；不能将 smoke test 当成研究发现。

现在先审计，写出最多 10 行执行计划，然后直接实现和运行到上述预算或明确阻塞点。不要停在只有设计文档的状态，也不要在没有真正运行时写 PASS。
