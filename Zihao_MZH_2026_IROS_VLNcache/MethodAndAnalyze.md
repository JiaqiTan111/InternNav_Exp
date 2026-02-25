# MethodAndAnalyze (System2-Focused, Challenge-Driven)

> 目标：将 Introduction 中的两个挑战，组织为“挑战 -> 分析证据 -> 方法模块 -> 图表 -> 实验验证”的闭环框架，并与 InternVLN 的 System2 慢系统改造严格对应。

---

## 0. 与 Intro 的一一对应

### Challenge A: Viewpoint-Induced Reuse Failure（视角变化导致可复用 token 被低估）
- Intro 对应：静态物体在相邻帧像素位置显著漂移，位置匹配低估可复用 token。
- 结论：不能只做 position-wise matching，必须做 view-aware 对齐后再复用。

### Challenge B: Instruction-Conditioned Semantic Drift（语义相关性阶段漂移）
- Intro 对应：同一地标在不同导航阶段重要性变化，缓存会“语义过期”。
- 结论：需要 instruction-guided refresh，而不是固定周期刷新。

---

## 1. 分析设计（Analyze）：先证明问题，再提方法

> 原则：每个挑战都先给“可量化证据图”，再给“方法模块”。

## 1.1 Challenge A 分析：为什么静态场景缓存在 VLN 失效

### A-Analysis-1: Token 可复用性-位移耦合分析（主图）
- 统计对象：相邻帧中高相似区域 token。
- 横轴：几何/像素位移幅度（可由光流或特征匹配近似）。
- 纵轴：错误拒绝率（本可复用却被位置匹配拒绝的比例）。
- 期望现象：位移上升时，位置匹配的误拒绝显著上升。

### A-Analysis-2: “位置匹配 vs 视角对齐后匹配”对比（主图）
- 画两条曲线：
  - Baseline：position-wise matching
  - Ours-Align：view-aware alignment + matching
- 指标：可复用 token 召回率（Recall@Reusable）。
- 期望现象：Ours-Align 在中高位移段显著更稳。

### A-Analysis-3: 个例可视化（辅图）
- 三行展示：原图、位置匹配结果、对齐后匹配结果。
- 强调“看起来变了位置，但语义仍是同一静态背景”的案例。

---

## 1.2 Challenge B 分析：为什么语义缓存会过期

### B-Analysis-1: 指令相关性随时间变化（主图）
- 对每步输出“指令-区域相关性”热度（可基于 cross-attention/grounding score 近似）。
- 纵向时间轴展示同一 landmark 的权重变化。
- 期望现象：拐弯前后/经过地标前后，相关性显著变化。

### B-Analysis-2: 固定刷新 vs 语义触发刷新（主图）
- 对比策略：
  - Fixed refresh（每 K 步刷新）
  - Semantic refresh（相关性变化超过阈值才刷新）
- 指标：无效刷新率 + 过期缓存率。
- 期望现象：Semantic refresh 同时降低无效刷新与过期风险。

### B-Analysis-3: 阶段性行为分布图（辅图）
- 将轨迹粗分为探索/精定位阶段，统计每阶段输出类型（动作串 vs 坐标）和刷新频率。
- 期望现象：不同阶段所需缓存策略不同。

---

## 2. 方法设计（Method）：一个挑战一个模块，最后组合

> 主体方法名建议：**VLN-Cache-S2**（System2-only enhancement）。

## 2.1 Method-A: View-Aware Reuse Module（解决 Challenge A）

### 核心思路
- 在 System2 做 token 复用前，先做跨帧 view-aware 对齐。
- 对齐后再判定 token reuse，避免位置漂移引起的误拒绝。

### 可选实现路线（按风险递增）
- A1（稳妥）：特征级近邻匹配对齐（轻量，不依赖准确位姿）。
- A2（平衡）：深度/几何辅助对齐（利用 depth + 内参）。
- A3（创新）：显式可学习对齐头（代价高，收益潜力大）。

### System2 代码落点（InternVLN）
- `internnav/model/basemodel/internvla_n1/internvla_n1_policy.py`
  - `s2_step(...)`：插入“对齐 -> 匹配 -> 复用决策”逻辑。
- `internnav/agent/internvla_n1_agent.py`
  - `step(...)`：维护跨步缓存状态，管理复用窗口。

---

## 2.2 Method-B: Instruction-Guided Semantic Refresh（解决 Challenge B）

### 核心思路
- 将 refresh 从“固定周期触发”改为“语义变化触发”。
- 当 instruction-grounded saliency 大幅变化时，强制刷新相关缓存。

### 可选实现路线（按风险递增）
- B1（稳妥）：基于输出模式（动作/坐标）+ 步态阶段的规则触发。
- B2（平衡）：基于 cross-attention/匹配分数变化触发。
- B3（创新）：轻量分类器预测“是否语义漂移”。

### System2 代码落点（InternVLN）
- `internnav/agent/internvla_n1_agent.py`
  - `should_infer_s2(...)`：将“纯步数触发”扩展为“步数 + 语义漂移触发”。
- `internnav/model/basemodel/internvla_n1/internvla_n1_policy.py`
  - `s2_step(...)`：输出 saliency 信号或 drift score。

---

## 2.3 Method-C: Decode Budget Controller（System2配套加速，建议并入主方法）

### 核心思路
- 根据当前阶段/输出类型，动态设定 `max_new_tokens`、停止策略和验证策略。
- 动作输出用短窗口，坐标输出用固定格式窗口。

### 可选实现路线
- C1（稳妥）：规则型预算控制（无需训练）。
- C2（平衡）：加入轻量 draft 预测器（speculative-lite）。
- C3（创新）：draft+verify 的完整 speculative decoding。

### System2 代码落点（InternVLN）
- `internnav/model/basemodel/internvla_n1/internvla_n1_policy.py`
  - `generate(...)` 参数改为动态预算。
- `scripts/eval/configs/habitat_dual_system_cfg.py`
  - 增加预算策略配置项，避免硬编码。

---

## 3. 多方案组合池（给你“可选路线”，不是单一路径）

## 3.1 路线 S（稳妥）
- 组合：A1 + B1 + C1
- 优点：实现快、风险低、容易复现。
- 风险：创新性一般。
- 适合：先产出稳定主结果，快速形成论文主表。

## 3.2 路线 M（平衡，推荐主线）
- 组合：A2 + B2 + C1/C2
- 优点：挑战-方法对齐强、创新与可行性平衡。
- 风险：阈值较多，需要系统调参。
- 适合：IROS 正文主线。

## 3.3 路线 X（创新）
- 组合：A3 + B3 + C3
- 优点：方法新颖，潜在上限高。
- 风险：实现/复现实验成本高，进度压力大。
- 适合：有充足算力和时间时做附加实验。

---

## 4. 图表与论文章节映射（直接可写论文）

## 4.1 图表清单（建议）
1. **Fig.1 Challenge Overview**：A/B 两个挑战的失败现象概览。  
2. **Fig.2 Viewpoint Mismatch Analysis**：位移 vs 误拒绝率、对齐前后对比曲线。  
3. **Fig.3 Semantic Drift Analysis**：指令相关性时间轴 + refresh 触发点。  
4. **Fig.4 Method Pipeline**：A/B/C 三模块在 System2 中的串联。  
5. **Fig.5 Pareto Curve**：速度-性能折中（Latency vs SR/SPL）。

## 4.2 表格清单（建议）
1. **Tab.1 Main Results**：NE/OS/SR/SPL + 平均 System2 时延。  
2. **Tab.2 Ablation by Challenge**：Base、+A、+B、+A+B、+A+B+C。  
3. **Tab.3 Overhead Breakdown**：对齐、刷新、预算控制各自开销。

---

## 5. 实验设计：确保“环环紧扣”

## 5.1 必做消融（最小闭环）
- E0: Baseline（原 System2）
- E1: Baseline + Method-A（只解 Challenge A）
- E2: Baseline + Method-B（只解 Challenge B）
- E3: Baseline + A + B（双挑战联合）
- E4: Baseline + A + B + C（完整系统）

## 5.2 每个实验都要回答的问题
- 是否真的解决了对应挑战（分析指标）？
- 是否提高了效率（时延/token）？
- 是否保持了导航性能（SR/SPL/NE）？

## 5.3 关键评价指标
- 导航性能：SR, SPL, NE, OS。
- 效率性能：System2 每步时延、总生成 token、S2 调用频率。
- 机制指标：A 的复用召回率、B 的过期缓存率/无效刷新率。

---

## 6. 实现顺序建议（按论文节奏推进）

### 阶段 1（1-2 周）：稳妥打底
- 上 C1（预算控制）+ B1（规则触发）
- 产出首版主表和流程图，打通实验链路。

### 阶段 2（2-3 周）：挑战闭环
- 上 A2（视角对齐）+ B2（语义触发）
- 完成 Fig.2 / Fig.3 / Tab.2。

### 阶段 3（可选）：创新增强
- 选择 C2 或 C3（speculative 系）做附加提升。

---

## 7. 写作注意事项（防止 claim 过度）

- 主文只承诺“System2 推理优化”，不要写成“全系统重构”。
- 将 “layer-adaptive reuse” 放在扩展/未来工作（若当前未完整实现）。
- 每个 claim 至少绑定一个可复现实验结果与一个代码改动点。

---

## 8. 你可以直接复制到论文的方法叙事模板

### 模板句 1（挑战到方法）
“For Challenge A, we introduce a view-aware reuse module that aligns cross-frame representations before token reuse, mitigating viewpoint-induced mismatch.”

### 模板句 2（挑战到方法）
“For Challenge B, we design an instruction-guided semantic refresh mechanism that updates cached regions only when task relevance shifts, reducing stale-cache errors.”

### 模板句 3（系统落地）
“All modules are integrated into the System2 branch of InternVLN, leaving training setup unchanged and focusing on inference-time acceleration.”

---

## 9. 当前建议的主线结论

- 如果你要“可行性+创新平衡”，优先选 **路线 M: A2 + B2 + C1/C2**。
- 这样可以保证：
  1) 与 intro 两挑战严格对齐；
  2) 改动集中在 System2，工程可落地；
  3) 图和消融能形成完整证据链。
