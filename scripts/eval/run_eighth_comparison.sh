#!/bin/bash
# Run 1/8-scale baseline and VLN-Cache comparison on GPU 1.
# Usage: bash scripts/eval/run_eighth_comparison.sh
#
# The two runs are sequential (same GPU).  Each takes ~hours depending
# on model speed.  progress.json is written incrementally so you can
# monitor with:
#   tail -f logs/habitat/baseline_eighth/progress.json
#   tail -f logs/habitat/vln_cache_eighth/progress.json

set -e
cd "$(dirname "$0")/../.."

# Activate conda environment
source /home/iflab-zzh-intern/miniconda3/etc/profile.d/conda.sh
conda activate mzh-habitataenv

export CUDA_VISIBLE_DEVICES=1

echo "===== [1/2] Running BASELINE (230 episodes) ====="
python scripts/eval/eval.py \
    --config scripts/eval/configs/habitat_dual_system_baseline_eighth_cfg.py

echo ""
echo "===== [2/2] Running VLN-Cache (230 episodes) ====="
python scripts/eval/eval.py \
    --config scripts/eval/configs/habitat_dual_system_vln_cache_eighth_cfg.py

echo ""
echo "===== Both runs finished. Generating comparison table... ====="
python scripts/eval/compare_results.py \
    --baseline logs/habitat/baseline_eighth/progress.json \
    --method   logs/habitat/vln_cache_eighth/progress.json

echo "Done."
