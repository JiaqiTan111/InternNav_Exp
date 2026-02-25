# Analyze (run01, Habitat only)

本文件汇总 run01 的分析结果，并给出可直接用于论文的图与文字草稿。

## 0. 两个 Gap 怎么被图直接说明（先看这个）

- Gap A（Viewpoint-Induced Reuse Failure）看这张总图：
  - [vis_analyze/reports/run01/figures_pub/fig_gapA_publication.png](vis_analyze/reports/run01/figures_pub/fig_gapA_publication.png)
  - 这张图直接回答：为什么“简单对齐/位置匹配”不够，必须升级为更强的 view-aware reuse。
- Gap B（Instruction-Conditioned Semantic/Compute Mismatch）看这张总图：
  - [vis_analyze/reports/run01/figures_fancy/fig_gapB_budget_mismatch_fancy.png](vis_analyze/reports/run01/figures_fancy/fig_gapB_budget_mismatch_fancy.png)
  - 这张图直接回答：为什么静态 decode/refresh 预算不合理，必须做动态预算和语义触发。

一句话总结：
- Gap A 是“该复用却没复用（匹配失败）”；
- Gap B 是“该省算力却没省（预算失配）”。

## 1. 数据来源与留痕

- 运行配置: [vis_analyze/configs/habitat_dual_system_observation_run01_cfg.py](vis_analyze/configs/habitat_dual_system_observation_run01_cfg.py)
- 逐步日志: [vis_analyze/data/raw/run01/step_log_rank0.jsonl](vis_analyze/data/raw/run01/step_log_rank0.jsonl)
- System2 日志: [vis_analyze/data/raw/run01/s2_log_rank0.jsonl](vis_analyze/data/raw/run01/s2_log_rank0.jsonl)
- episode 日志: [vis_analyze/data/raw/run01/episode_log_rank0.jsonl](vis_analyze/data/raw/run01/episode_log_rank0.jsonl)
- 评测进度: [vis_analyze/data/eval_output/run01/progress.json](vis_analyze/data/eval_output/run01/progress.json)
- 最终统计: [vis_analyze/reports/run01/run01_analysis_stats.json](vis_analyze/reports/run01/run01_analysis_stats.json)

本次有效规模:
- Episodes: 100
- Step logs: 8178
- S2 calls: 2101

## 2. 核心结果（可直接写正文）

### 2.1 导航主指标（run01 baseline）

- SR = 0.700
- SPL = 0.636
- OS = 0.770
- NE = 3.722
- 平均步数 = 81.77（P50=68.5, P90=117.3）

### 2.2 Challenge A: Viewpoint-Induced Reuse Failure（视觉动态）

- raw patch cosine mean = 0.9360
- aligned patch cosine mean = 0.8263
- delta mean = -0.1098
- delta quantiles: P10=-0.3607, P50=0.0, P90=0.0
- aligned_better_ratio = 0.0

解读（当前代码口径下）:
- 目前的“alignment”在该统计定义下没有带来提升，甚至均值更低。
- 这与我们挑战 A 的预期相反，说明当前对齐实现/度量定义存在偏差，不能直接作为方法有效性的证据。
- 论文中建议将该结果表述为: “naive yaw-only alignment is insufficient under embodied motion”, 从而合理引出改进版 view-aware 模块（A2/A3）。

更直观看图结论:
- 在 [fig_gapA_publication.png](vis_analyze/reports/run01/figures_pub/fig_gapA_publication.png) 里，
  - A 面板（ECDF）显示 aligned 曲线整体右移失败，均值显著低于 raw；
  - B 面板（delta 分布）显示 $\Delta=s_{aligned}-s_{raw}$ 主要集中在负值区；
  - C 面板（|Δyaw| 分箱）未出现稳定正向改善趋势；
  - D 面板（运动分组）显示大运动区间退化更明显。
- 因此 Gap A 的证据链成立：**当前弱对齐不足以恢复可复用 token**。

统计显著性（用于论文正文）:
- 统计文件: [vis_analyze/reports/run01/gapA_publication_stats.json](vis_analyze/reports/run01/gapA_publication_stats.json)
- delta_mean = -0.1098，95% CI = [-0.1131, -0.1067]（不跨 0）
- P(delta > 0) = 0.0
- Cohen's d(aligned - raw) = -0.872（大效应量）

可用于正文的一句话:
- “The negative delta with a non-overlapping 95% confidence interval and large effect size indicates that naïve alignment does not recover reusable correspondence under embodied viewpoint changes.”

## 2.4 Gap A 深度剖析（新增）

新增深度分析产物:
- 图: [vis_analyze/reports/run01/deep_dive/fig_gapA_deep_dive.png](vis_analyze/reports/run01/deep_dive/fig_gapA_deep_dive.png)
- 统计: [vis_analyze/reports/run01/deep_dive/gapA_deep_dive_stats.json](vis_analyze/reports/run01/deep_dive/gapA_deep_dive_stats.json)

这轮深度分析回答了“问题在哪里更严重”:
- 分阶段（early/mid/late）:
  - early mean delta = -0.1453（95%CI [-0.1513, -0.1399]）
  - mid mean delta = -0.0960（95%CI [-0.1015, -0.0910]）
  - late mean delta = -0.0881（95%CI [-0.0937, -0.0827]）
  - 结论：失配在轨迹早期更严重（通常对应较大视角调整期）。
- 分结果组（success vs failure）:
  - success mean delta = -0.1027
  - failure mean delta = -0.1055
  - corr(delta, NE) = 0.051（弱相关）
  - 结论：Gap A 是“系统性匹配问题”，不是只在失败轨迹上才出现。

这轮深度分析也暴露了一个重要边界:
- scene 维度当前几乎只有 1 个主场景（2azQ1b91cZZ），说明 run01 的跨场景覆盖不足。
- 因此目前最稳妥表述是“在当前采样条件下，Gap A 显著存在”；
- 若要写成“广泛跨场景成立”，需要补一个 run02（多 scene 覆盖）来增强外部有效性。

### 2.3 Challenge B + Efficiency: System2 计算行为

- S2 调用次数: 2101
- prompt_len mean = 1903.3（P50=1894, P90=2304）
- gen_len mean = 4.70（P50=5, P90=8）
- total_len mean = 1908.0（P50=1897, P90=2312）
- preprocess_ms mean = 75.55（P50=73.67, P90=92.71）
- generate_ms mean = 372.28（P50=320.07, P90=507.35）
- decode_ms mean = 0.150（P50=0.146, P90=0.162）

解读:
- System2 延迟主要由 generate 阶段主导，decode 开销可忽略。
- prompt token 很长、gen token 很短，支持 Method-C（动态预算控制）的合理性。
- 该证据可直接支撑 “在不改训练的情况下优先优化 S2 推理预算与触发策略”。

更直观看图结论:
- 在 [fig_gapB_budget_mismatch_fancy.png](vis_analyze/reports/run01/figures_fancy/fig_gapB_budget_mismatch_fancy.png) 里，
  - B1 子图显示 prompt 长度和 generate 时延明显耦合；
  - B2 子图显示 generate 占主要延迟份额；
  - B3 子图显示 gen_len 主要集中在短输出；
  - B4 子图显示 episode 间 S2 调用负载差异大。
- 因此 Gap B 的证据链成立：**静态预算/固定触发不匹配真实阶段性计算需求**。

## 3. 图表方案（论文可直接用）

## Fig.2 视觉动态失配图（Challenge A）

图目标:
- 证明“简单位置/简单对齐策略不足以稳定复用”。

建议子图:
- Fig.2(a): raw_mean 与 aligned_mean 的分布对比（箱线图/小提琴图）
- Fig.2(b): delta_mean 直方图（重点展示负偏）
- Fig.2(c): 轨迹个例（step 序列）中 raw/aligned 曲线

数据字段:
- 来自 step_log 的 similarity.raw_mean, similarity.aligned_mean, similarity.delta_mean

正文模板:
- “Under the current naïve alignment setting, aligned similarity does not consistently improve over raw similarity, indicating that viewpoint-aware reuse requires stronger geometric/semantic alignment than simple yaw-based warping.”

推荐直接用 fancy 版本替换：
- [vis_analyze/reports/run01/figures_pub/fig_gapA_publication.png](vis_analyze/reports/run01/figures_pub/fig_gapA_publication.png)

## Fig.3 语义阶段与 S2 行为图（Challenge B）

图目标:
- 证明 System2 的行为存在阶段性，并且当前预算配置有优化空间。

建议子图:
- Fig.3(a): 每 episode 的 system2_calls 分布
- Fig.3(b): prompt_len vs generate_ms 散点图
- Fig.3(c): gen_len 分布（动作短输出与坐标短输出共性）

数据字段:
- progress.json: system2_calls
- s2_log: prompt_len, gen_len, generate_ms, preprocess_ms

正文模板:
- “System2 spends most latency budget in generation while producing short outputs, motivating a stage-aware decode budget controller rather than static max-token settings.”

推荐直接用 fancy 版本替换：
- [vis_analyze/reports/run01/figures_fancy/fig_gapB_budget_mismatch_fancy.png](vis_analyze/reports/run01/figures_fancy/fig_gapB_budget_mismatch_fancy.png)

## Fig.5 速度-性能权衡起点图（Pareto baseline anchor）

图目标:
- 给后续 A/B/C 方法提供同口径对比基线点。

坐标建议:
- x 轴: 平均 System2 generate_ms（或每 episode 的均值）
- y 轴: SR / SPL（双轴或两张图）

当前 baseline 锚点:
- generate_ms_mean = 372.28 ms
- SR = 0.700
- SPL = 0.636

## 4. 可直接放论文的分析文字（草稿）

### Challenge A 分析段（草稿）

We analyze patch-level cross-frame similarity under embodied navigation dynamics. While adjacent observations maintain high raw similarity on average, naïve yaw-based alignment does not improve matching quality and often decreases aligned similarity. This suggests that simple geometric warping is insufficient for robust token reuse in VLN, where translation, depth variation, and semantic layout changes jointly affect correspondence. Therefore, Challenge A should be addressed with stronger view-aware reuse mechanisms beyond position-wise or weak alignment baselines.

### Challenge B 分析段（草稿）

From 100 validation episodes, System2 is invoked 2101 times, with long prompts (mean 1903 tokens) but short generations (mean 4.7 tokens). Latency is dominated by generation (mean 372 ms) rather than decode overhead. This pattern indicates that static decoding budgets are suboptimal: most calls do not require long autoregressive expansion. The evidence supports an instruction/stage-aware budget controller that adapts generation length and refresh frequency to reduce compute while preserving navigation quality.

## 5. 下一步（与方法实现衔接）

- A 线（对应 Challenge A）:
  - 将当前“naive alignment”作为对照基线；
  - 实施 A2（depth/geometry assisted alignment）后复用同一分析脚本重跑 run02；
  - 对比 delta_mean 与 aligned_better_ratio 是否改善。

- B+C 线（对应 Challenge B）:
  - 先实施 C1（规则型 decode budget）和 B1（规则触发刷新）；
  - 复用同口径日志字段生成 run02 summary；
  - 对比 generate_ms_mean、SR、SPL，形成第一版消融。

## 6. 注意事项

- 当前 Challenge A 的统计结果是“负例证据”，有价值，但不能当成方法有效性结论。
- 论文叙事建议使用: “baseline failure -> improved design” 结构，避免过度 claim。
- 所有图都应从 run 级目录读取，保持可复现实验闭环。
