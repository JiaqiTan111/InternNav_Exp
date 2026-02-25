#!/usr/bin/env python3
"""
run_comparison.py  –  Run baseline vs VLN-Cache evaluations and compare results.

This script:
    1. Runs baseline evaluation (standard InternVLA-N1).
    2. Runs VLN-Cache evaluation (with KV-Cache reuse).
    3. Reads both progress.json files and prints a comparison table.

Usage:
    # Run both evaluations and compare:
    python -m internnav.vln_cache.run_comparison \
        --baseline-config scripts/eval/configs/habitat_dual_system_cfg.py \
        --cache-config scripts/eval/configs/habitat_dual_system_vln_cache_cfg.py

    # Compare existing results (skip running):
    python -m internnav.vln_cache.run_comparison \
        --baseline-output ./logs/habitat/test_dual_system \
        --cache-output ./logs/habitat/test_dual_system_vln_cache \
        --compare-only

    # Quick test (2 episodes):
    python -m internnav.vln_cache.run_comparison \
        --baseline-config scripts/eval/configs/habitat_dual_system_cfg.py \
        --cache-config scripts/eval/configs/habitat_dual_system_vln_cache_cfg.py \
        --max-episodes 2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Result parsing
# ──────────────────────────────────────────────────────────────────────

def load_progress(output_dir: str) -> list[dict]:
    """Load progress.json lines from an evaluation run."""
    path = os.path.join(output_dir, "progress.json")
    if not os.path.exists(path):
        print(f"[WARN] progress.json not found at {path}")
        return []
    results = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def aggregate_metrics(results: list[dict]) -> dict:
    """Compute mean metrics from a list of per-episode results."""
    if not results:
        return {}

    sucs = [r["success"] for r in results]
    spls = [r["spl"] for r in results]
    oss = [r.get("os", r.get("oracle_success", 0.0)) for r in results]
    nes = [r["ne"] for r in results]
    steps = [r.get("steps", 0) for r in results]

    agg = {
        "n_episodes": len(results),
        "SR": float(np.mean(sucs)),
        "SPL": float(np.mean(spls)),
        "OS": float(np.mean(oss)),
        "NE": float(np.mean(nes)),
        "Avg Steps": float(np.mean(steps)),
    }

    # nDTW if available
    ndtws = [r["ndtw"] for r in results if "ndtw" in r]
    if ndtws:
        agg["nDTW"] = float(np.mean(ndtws))

    # VLN-Cache specific stats
    cache_stats = [r["vln_cache"] for r in results if "vln_cache" in r]
    if cache_stats:
        mean_reuse = np.mean([s["mean_reuse_ratio"] for s in cache_stats])
        agg["Cache Reuse"] = float(mean_reuse)
        if any("cache_overhead_ms_mean" in s for s in cache_stats):
            overheads = [s["cache_overhead_ms_mean"] for s in cache_stats if "cache_overhead_ms_mean" in s]
            agg["Cache OH (ms)"] = float(np.mean(overheads))

    # S2 token stats
    s2_calls = [r.get("system2_calls", 0) for r in results]
    if any(c > 0 for c in s2_calls):
        agg["S2 Calls/ep"] = float(np.mean(s2_calls))
        prompt_means = [r["system2_prompt_tokens_mean"] for r in results if "system2_prompt_tokens_mean" in r]
        if prompt_means:
            agg["S2 Prompt Tok"] = float(np.mean(prompt_means))

    return agg


# ──────────────────────────────────────────────────────────────────────
# Comparison
# ──────────────────────────────────────────────────────────────────────

def print_comparison(baseline: dict, cache: dict):
    """Print a side-by-side comparison table."""
    print("\n" + "=" * 70)
    print("   VLN-Cache  vs  Baseline  Comparison")
    print("=" * 70)

    all_keys = list(dict.fromkeys(list(baseline.keys()) + list(cache.keys())))

    header = f"{'Metric':<20} {'Baseline':>12} {'VLN-Cache':>12} {'Δ':>10}"
    print(header)
    print("-" * 56)

    for key in all_keys:
        bv = baseline.get(key)
        cv = cache.get(key)

        if bv is None and cv is None:
            continue

        bv_str = f"{bv:.4f}" if isinstance(bv, float) else str(bv) if bv is not None else "—"
        cv_str = f"{cv:.4f}" if isinstance(cv, float) else str(cv) if cv is not None else "—"

        delta_str = ""
        if isinstance(bv, (int, float)) and isinstance(cv, (int, float)):
            delta = cv - bv
            sign = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta:.4f}"

        print(f"{key:<20} {bv_str:>12} {cv_str:>12} {delta_str:>10}")

    print("=" * 70)

    # Highlight key metrics
    if "SR" in baseline and "SR" in cache:
        sr_delta = cache["SR"] - baseline["SR"]
        spl_delta = cache.get("SPL", 0) - baseline.get("SPL", 0)
        ne_delta = cache.get("NE", 0) - baseline.get("NE", 0)
        print(f"\n  SR:  {cache['SR']:.1%} vs {baseline['SR']:.1%}  (Δ = {sr_delta:+.1%})")
        print(f"  SPL: {cache.get('SPL',0):.1%} vs {baseline.get('SPL',0):.1%}  (Δ = {spl_delta:+.1%})")
        print(f"  NE:  {cache.get('NE',0):.2f} vs {baseline.get('NE',0):.2f}  (Δ = {ne_delta:+.2f})")

        if "Cache Reuse" in cache:
            print(f"  Avg vision token reuse: {cache['Cache Reuse']:.1%}")

    print()


def save_comparison_json(baseline: dict, cache: dict, output_path: str):
    """Save comparison to a JSON file."""
    comparison = {
        "baseline": baseline,
        "vln_cache": cache,
        "delta": {
            k: cache.get(k, 0) - baseline.get(k, 0)
            for k in ["SR", "SPL", "OS", "NE", "nDTW"]
            if k in baseline or k in cache
        },
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison saved to {output_path}")


# ──────────────────────────────────────────────────────────────────────
# Evaluation runner
# ──────────────────────────────────────────────────────────────────────

def run_eval(config_path: str, max_episodes: int = 0) -> str:
    """Run an evaluation via the standard eval.py entry point.

    Returns the output directory path (read from config).
    """
    # Load config to find output_path
    import importlib.util
    spec = importlib.util.spec_from_file_location("cfg_mod", config_path)
    cfg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_mod)
    output_path = cfg_mod.eval_cfg.eval_settings.get("output_path", "./logs/habitat/unknown")

    # Override max_episodes if requested
    if max_episodes > 0:
        cfg_mod.eval_cfg.eval_settings["max_episodes"] = max_episodes

    print(f"\n{'='*60}")
    print(f"Running evaluation: {config_path}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "scripts/eval/eval.py",
        "--config", config_path,
    ]

    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    proc = subprocess.run(cmd, env=env, cwd=os.getcwd())
    if proc.returncode != 0:
        print(f"[ERROR] Evaluation failed with return code {proc.returncode}")

    return output_path


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run baseline vs VLN-Cache comparison"
    )
    parser.add_argument(
        "--baseline-config", type=str,
        default="scripts/eval/configs/habitat_dual_system_cfg.py",
        help="Config file for baseline evaluation",
    )
    parser.add_argument(
        "--cache-config", type=str,
        default="scripts/eval/configs/habitat_dual_system_vln_cache_cfg.py",
        help="Config file for VLN-Cache evaluation",
    )
    parser.add_argument(
        "--baseline-output", type=str, default=None,
        help="Path to existing baseline output (skip running)",
    )
    parser.add_argument(
        "--cache-output", type=str, default=None,
        help="Path to existing VLN-Cache output (skip running)",
    )
    parser.add_argument(
        "--compare-only", action="store_true",
        help="Only compare existing results (require --baseline-output and --cache-output)",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=0,
        help="Max episodes per run (0 = all)",
    )
    parser.add_argument(
        "--output-json", type=str,
        default="./logs/habitat/vln_cache_comparison.json",
        help="Path to save comparison JSON",
    )

    args = parser.parse_args()

    baseline_output = args.baseline_output
    cache_output = args.cache_output

    if not args.compare_only:
        # Run baseline
        if baseline_output is None:
            baseline_output = run_eval(args.baseline_config, args.max_episodes)
        else:
            print(f"Using existing baseline output: {baseline_output}")

        # Run VLN-Cache
        if cache_output is None:
            cache_output = run_eval(args.cache_config, args.max_episodes)
        else:
            print(f"Using existing VLN-Cache output: {cache_output}")
    else:
        if baseline_output is None or cache_output is None:
            parser.error("--compare-only requires --baseline-output and --cache-output")

    # Load and compare
    baseline_results = load_progress(baseline_output)
    cache_results = load_progress(cache_output)

    if not baseline_results:
        print(f"No baseline results found in {baseline_output}")
        return
    if not cache_results:
        print(f"No VLN-Cache results found in {cache_output}")
        return

    baseline_agg = aggregate_metrics(baseline_results)
    cache_agg = aggregate_metrics(cache_results)

    print_comparison(baseline_agg, cache_agg)
    save_comparison_json(baseline_agg, cache_agg, args.output_json)


if __name__ == "__main__":
    main()
