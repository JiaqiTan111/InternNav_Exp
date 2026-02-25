#!/usr/bin/env bash
#
# run_small_comparison.sh  —  Quick 5-episode baseline vs VLN-Cache comparison.
#
# Usage:
#   bash scripts/eval/run_small_comparison.sh          # 全部运行
#   bash scripts/eval/run_small_comparison.sh baseline  # 仅baseline
#   bash scripts/eval/run_small_comparison.sh cache     # 仅VLN-Cache
#   bash scripts/eval/run_small_comparison.sh compare   # 仅对比已有结果
#
set -e
cd "$(dirname "$0")/../.."

BASELINE_CFG="scripts/eval/configs/habitat_dual_system_small_cfg.py"
CACHE_CFG="scripts/eval/configs/habitat_dual_system_vln_cache_small_cfg.py"

BASELINE_OUT="./logs/habitat/test_dual_system_small"
CACHE_OUT="./logs/habitat/test_dual_system_vln_cache_small"

MODE="${1:-all}"

# ── 清除 __pycache__ 避免旧字节码 ──
find internnav/vln_cache -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

if [[ "$MODE" == "all" || "$MODE" == "baseline" ]]; then
    echo "═══════════════════════════════════════════════════════════════"
    echo " Running BASELINE (5 episodes) ..."
    echo "═══════════════════════════════════════════════════════════════"
    CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval.py --config "$BASELINE_CFG"
fi

if [[ "$MODE" == "all" || "$MODE" == "cache" ]]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo " Running VLN-CACHE (5 episodes) ..."
    echo "═══════════════════════════════════════════════════════════════"
    CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval.py --config "$CACHE_CFG"
fi

if [[ "$MODE" == "all" || "$MODE" == "compare" ]]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo " Comparing results ..."
    echo "═══════════════════════════════════════════════════════════════"
    python -m internnav.vln_cache.run_comparison \
        --baseline-output "$BASELINE_OUT" \
        --cache-output "$CACHE_OUT" \
        --compare-only \
        --output-json "./logs/habitat/vln_cache_small_comparison.json"
fi

echo ""
echo "Done.  结果: ${BASELINE_OUT} vs ${CACHE_OUT}"
