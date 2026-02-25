# vis_analyze_new

Logging + hook + offline analysis pipeline for two diagnostic gaps in VLN token reuse (no cache modification).

## Gap 1 (Pseudo-dynamic)
- Runtime logs: pose/quaternion, RGB/Depth (optional dump), patch token hook (optional dump), position-vs-aligned similarity.
- Offline analysis: `scripts/analyze_gap1.py`
- Fancy figure G1: `scripts/plot_gap1_fancy.py`

## Gap 2 (Semantic relevance shift)
- Runtime probe: language→vision saliency via `output_attentions=True` forward probe.
- Offline analysis: `scripts/analyze_gap2.py`
- Fancy figure G2: `scripts/plot_gap2_fancy.py`

## Quick run
1. Run evaluation with analysis flags enabled (see `scripts/eval/configs/habitat_*.py`).
2. Generate Gap1 report:
   - `python vis_analyze_new/scripts/analyze_gap1.py --input vis_analyze_new/data/raw/step_log_rank0.jsonl`
   - `python vis_analyze_new/scripts/plot_gap1_fancy.py --input vis_analyze_new/reports/gap1_timeseries.jsonl`
3. Generate Gap2 report:
   - `python vis_analyze_new/scripts/analyze_gap2.py --input vis_analyze_new/data/raw/step_log_rank0.jsonl`
   - `python vis_analyze_new/scripts/plot_gap2_fancy.py --input vis_analyze_new/reports/gap2_timeseries.jsonl`
