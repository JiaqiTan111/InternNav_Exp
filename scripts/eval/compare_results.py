#!/usr/bin/env python3
"""
compare_results.py  —  Parse two progress.json files and print a side-by-side
                       comparison table of all VLN + VLN-Cache metrics.

Usage:
    python scripts/eval/compare_results.py \
        --baseline logs/habitat/baseline_eighth/progress.json \
        --method   logs/habitat/vln_cache_eighth/progress.json
"""

import argparse
import json
import math
import sys
from pathlib import Path


def load_progress(path: str) -> list[dict]:
    """Load a progress.json file (one JSON object per line)."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def safe_mean(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else 0.0


def compute_metrics(records: list[dict]) -> dict:
    """Compute aggregate metrics from per-episode records."""
    n = len(records)
    if n == 0:
        return {}

    sucs = [r["success"] for r in records]
    spls = [r["spl"] for r in records]
    oss  = [r["os"] for r in records]
    nes  = [r["ne"] for r in records]
    tls  = [r.get("tl", None) for r in records]
    ones = [r.get("one", None) for r in records]
    strs = [r.get("str", None) for r in records]
    frs  = [r.get("fr", None) for r in records]
    hcrs = [r.get("hcr", None) for r in records]
    steps = [r.get("steps", None) for r in records]

    result = {
        "Episodes": n,
        "SR (%)": safe_mean(sucs) * 100,
        "SPL (%)": safe_mean(spls) * 100,
        "NE (m)": safe_mean(nes),
        "OSR (%)": safe_mean(oss) * 100,
        "ONE (m)": safe_mean(ones),
        "TL (m)": safe_mean(tls),
        "StR (%)": safe_mean(strs) * 100,
        "FR (%)": safe_mean(frs) * 100 if any(v is not None for v in frs) else 0.0,
        "HCR (%)": safe_mean(hcrs) * 100 if any(v is not None for v in hcrs) else 0.0,
        "Avg Steps": safe_mean(steps),
    }

    # VLN-Cache specific
    cache_records = [r for r in records if "vln_cache" in r]
    if cache_records:
        reuse_ratios = [r["vln_cache"].get("mean_reuse_ratio", 0.0) for r in cache_records]
        overheads = [r["vln_cache"].get("cache_overhead_ms_mean", 0.0) for r in cache_records]
        result["Cache Reuse (%)"] = safe_mean(reuse_ratios) * 100
        result["Cache OH (ms)"] = safe_mean(overheads)

    return result


def print_comparison(baseline: dict, method: dict):
    """Print a formatted comparison table."""
    all_keys = list(baseline.keys())
    for k in method:
        if k not in all_keys:
            all_keys.append(k)

    # Determine column widths
    col0_w = max(len(k) for k in all_keys) + 2
    col1_w = 12
    col2_w = 12
    col3_w = 12

    header = f"{'Metric':<{col0_w}} {'Baseline':>{col1_w}} {'VLN-Cache':>{col2_w}} {'Delta':>{col3_w}}"
    sep = "-" * len(header)

    print()
    print(sep)
    print(header)
    print(sep)

    for key in all_keys:
        bv = baseline.get(key, None)
        mv = method.get(key, None)

        if bv is None and mv is None:
            continue

        # Format values
        if key == "Episodes":
            bv_str = str(int(bv)) if bv is not None else "—"
            mv_str = str(int(mv)) if mv is not None else "—"
            delta_str = ""
        elif "(%)" in key or key.endswith("(%)"):
            bv_str = f"{bv:.1f}" if bv is not None else "—"
            mv_str = f"{mv:.1f}" if mv is not None else "—"
            if bv is not None and mv is not None:
                d = mv - bv
                delta_str = f"{d:+.1f}"
            else:
                delta_str = "—"
        elif "(m)" in key:
            bv_str = f"{bv:.2f}" if bv is not None else "—"
            mv_str = f"{mv:.2f}" if mv is not None else "—"
            if bv is not None and mv is not None:
                d = mv - bv
                delta_str = f"{d:+.2f}"
            else:
                delta_str = "—"
        elif "(ms)" in key:
            bv_str = f"{bv:.2f}" if bv is not None else "—"
            mv_str = f"{mv:.2f}" if mv is not None else "—"
            delta_str = ""
        else:
            bv_str = f"{bv:.1f}" if bv is not None else "—"
            mv_str = f"{mv:.1f}" if mv is not None else "—"
            if bv is not None and mv is not None:
                d = mv - bv
                delta_str = f"{d:+.1f}"
            else:
                delta_str = "—"

        print(f"{key:<{col0_w}} {bv_str:>{col1_w}} {mv_str:>{col2_w}} {delta_str:>{col3_w}}")

    print(sep)
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs VLN-Cache results")
    parser.add_argument("--baseline", required=True, help="Path to baseline progress.json")
    parser.add_argument("--method", required=True, help="Path to VLN-Cache progress.json")
    args = parser.parse_args()

    if not Path(args.baseline).exists():
        print(f"ERROR: Baseline file not found: {args.baseline}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.method).exists():
        print(f"ERROR: Method file not found: {args.method}", file=sys.stderr)
        sys.exit(1)

    baseline_records = load_progress(args.baseline)
    method_records = load_progress(args.method)

    print(f"\nLoaded {len(baseline_records)} baseline episodes, {len(method_records)} VLN-Cache episodes.")

    baseline_metrics = compute_metrics(baseline_records)
    method_metrics = compute_metrics(method_records)

    print_comparison(baseline_metrics, method_metrics)

    # Also output as JSON for programmatic use
    out = {
        "baseline": baseline_metrics,
        "vln_cache": method_metrics,
    }
    out_path = Path(args.method).parent / "comparison.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Comparison saved to {out_path}")


if __name__ == "__main__":
    main()
