# VLN-Cache 设计与使用说明

> 本文档面向当前仓库 `InternNav_Exp`，说明 VLN-Cache 的核心思路、代码结构、执行时序与运行方法。

---

## 1. 一句话概述

VLN-Cache 是一个**训练无关（training-free）**的跨帧视觉 token KV 复用机制：

- 每步推理前，先基于深度+位姿估计当前帧哪些视觉 patch 与上一帧对应且稳定；
- 对可复用 token，在 decoder layer 中将当前步新算的 KV 替换为上一帧缓存 KV；
- 从而减少有效重算量，降低推理开销。

---

## 2. 核心思想（你的主要 idea）

VLN 中相邻帧视觉内容高度冗余，但语言条件会动态变化。因此复用策略分两层：

1. **几何/视觉静态门控（可复用候选）**
   - 用深度和位姿把当前 patch 重投影到上一帧，找几何对应；
   - 再用局部邻域余弦相似度精修匹配，避免纯几何误配。

2. **语义刷新门控（必须重算）**
   - 对语言任务相关性高或变化大的 token 强制刷新；
   - 最终只复用“视觉稳定且语义不需刷新”的 token。

最终公式：

- `reuse = m_vis & (~m_sem)`

---

## 3. 代码结构与职责

### 3.1 `internnav/vln_cache/vln_cache_utils.py`

纯算法函数（无模型依赖）：

- `view_aligned_remap(...)`
  - 输入当前/上一帧 depth、pose、相机内参；
  - 输出 patch 粗匹配索引与几何有效 mask。
- `neighborhood_refinement(...)`
  - 在粗匹配邻域内做 cosine 最优匹配；
  - 输出 refined 索引与视觉静态门 `m_vis`。
- `compute_semantic_refresh_gate(...)`
  - 根据 cross-attn proxy 计算语义刷新门 `m_sem`。
- `build_reuse_mask(m_vis, m_sem)`
  - 输出最终复用 mask。
- `layer_adaptive_threshold(...)`
  - 层级阈值函数（当前主流程预留，未强依赖启用）。

### 3.2 `internnav/vln_cache/vln_cache_wrapper.py`

状态管理与 KV 实际替换：

- `VLNCacheConfig`
  - 复用阈值、几何参数、相机参数、patch 网格等配置。
- `_FrameState`
  - 保存每帧 depth/pose/patch_embeds/cross_attn/KV。
- `VLNCacheManager`
  - `on_new_frame(...)`：计算本帧复用 mask；
  - `_make_layer_hook(...)`：decoder 每层后处理，执行 KV splice；
  - `reset()`：每个 episode 清空状态；
  - `remove_hooks()`：注销 hooks。

### 3.3 `internnav/vln_cache/evaluator_integration.py`

Evaluator 与 cache 的桥接层：

- `VLNCacheHook.from_evaluator(...)`
  - 从 evaluator 读取内参、分辨率，初始化 manager；
- `set_frame(...)`
  - 在 `generate()` 前注入 depth/pose/inputs；
- `model.visual` forward hook
  - 捕获 patch embeddings，触发 `manager.on_new_frame(...)`。

### 3.4 `internnav/habitat_extensions/vln/habitat_vln_evaluator.py`

真实调用入口：

- 读取 `vln_cache_enabled` 开关；
- 每 episode `reset_episode()`；
- 每步 `generate()` 前 `set_frame(...)`；
- episode 结束把 `vln_cache` 统计写入 `progress.json`。

---

## 4. 执行时序（真实推理）

一次 episode 的 cache 时序如下：

1. **初始化阶段**
   - `vln_cache_enabled=True` 时创建 `VLNCacheHook`；
   - 给 decoder layers 注册 forward hook。

2. **每回合开始**
   - 调 `reset_episode()`，清空上一回合缓存。

3. **每步推理前**
   - evaluator 组装 `inputs` 后调用 `set_frame(depth, pos, quat, inputs)`。

4. **visual encoder 前向时**
   - visual hook 拿到 patch embeddings；
   - 调 `manager.on_new_frame(...)` 计算 `reuse_mask`。

5. **decoder 每层 forward 后**
   - hook 读取本层 DynamicCache 的 K/V；
   - 先存当前帧 vision token KV；
   - 对 `reuse_mask=True` 的 token，用上一帧 KV 替换当前 KV。

6. **最后一层结束**
   - 提交 `_curr -> _prev`；
   - 清空本步临时 mask。

---

## 5. 关键配置说明

在评测配置里打开：

```python
eval_settings = {
    ...
    "vln_cache_enabled": True,
}
```

仓库中可直接参考：

- `scripts/eval/configs/habitat_dual_system_vln_cache_cfg.py`

常见参数（`VLNCacheConfig`）：

- `occlusion_eps`：深度一致性容忍度
- `neighbor_k`：局部精修窗口半径
- `tau_v_base`：视觉静态相似度阈值
- `tau_s, delta_s`：语义刷新阈值
- `max_reuse_ratio`：复用比例上限（安全阈）

---

## 6. 如何运行

### 6.1 运行带 VLN-Cache 的评测

```bash
python scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_vln_cache_cfg.py
```

### 6.2 对比 baseline 与 VLN-Cache

```bash
python -m internnav.vln_cache.run_comparison \
  --baseline-config scripts/eval/configs/habitat_dual_system_cfg.py \
  --cache-config scripts/eval/configs/habitat_dual_system_vln_cache_cfg.py
```

### 6.3 运行模块测试（单测/集成 mock）

```bash
conda run -n mzh-habitataenv python -m internnav.vln_cache.test_vln_cache
```

> 说明：该测试文件以算法与 hook 逻辑验证为主，包含 mock 结构，不等价于完整 Habitat 端到端评测。

---

## 7. 输出与验证指标

每个 episode 的 `progress.json` 中会记录：

- `vln_cache.mean_reuse_ratio`
- `vln_cache.cache_overhead_ms_mean`（若有）

可据此评估：

- 复用是否真实发生；
- 复用开销是否可接受；
- 对成功率/SPL/NE 的影响。

---

## 8. 当前实现边界（已知）

1. 当 patch 数不一致（分辨率/视角分支差异）时，本步会安全跳过复用。
2. `layer_adaptive_threshold` 已实现函数，但主流程当前以固定阈值路径为主。
3. 默认参数与测试期望有过历史偏差时，需要统一测试断言与配置默认值。

---

## 9. 结论

VLN-Cache 在本仓库中是**真实接入到评测推理链路**的：

- 有配置开关、真实调用点、真实 KV 替换、真实统计落盘；
- 不是只在 README 或 mock 里“展示概念”。

如果后续要写论文/答辩材料，这套实现可用“几何对齐 + 语义刷新 + 层内 KV 替换”三段式来讲，逻辑非常清晰。