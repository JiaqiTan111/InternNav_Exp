#!/usr/bin/env python3
"""Compare 100-episode baseline vs VLN-Cache results with efficiency metrics."""
import json, sys, os, numpy as np

def load_progress(path, max_episodes=None):
    """Load progress.json → list of per-episode dicts (deduplicated by episode_id)."""
    results = []
    seen_ids = set()
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            eid = d.get('episode_id')
            if eid not in seen_ids:
                seen_ids.add(eid)
                results.append(d)
    if max_episodes and len(results) > max_episodes:
        results = results[:max_episodes]
    return results

def agg(results, key, default=0.0):
    vals = [r.get(key, default) for r in results]
    return float(np.mean(vals)) if vals else default

def agg_cache(results, key, default=0.0):
    vals = [r['vln_cache'].get(key, default) for r in results if 'vln_cache' in r]
    return float(np.mean(vals)) if vals else default

def main():
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "logs/habitat/baseline_100"
    cache_dir = sys.argv[2] if len(sys.argv) > 2 else "logs/habitat/vln_cache_100"

    base_pj = os.path.join(base_dir, "progress.json")
    cache_pj = os.path.join(cache_dir, "progress.json")

    if not os.path.exists(base_pj):
        print(f"Baseline progress.json not found: {base_pj}"); return
    if not os.path.exists(cache_pj):
        print(f"VLN-Cache progress.json not found: {cache_pj}"); return

    base = load_progress(base_pj, max_episodes=100)
    cache = load_progress(cache_pj, max_episodes=100)
    print(f"Baseline: {len(base)} episodes | VLN-Cache: {len(cache)} episodes\n")

    # ── Navigation Metrics ──
    nav_metrics = [
        ("SR (%)",   "success",  100),
        ("SPL (%)",  "spl",      100),
        ("NE (m)",   "ne",        1),
        ("OSR (%)",  "os",       100),
        ("TL (m)",   "tl",        1),
        ("ONE (m)",  "one",       1),
        ("StR (%)",  "str",     100),
    ]
    print("=" * 62)
    print(f"{'Metric':<20} {'Baseline':>12} {'VLN-Cache':>12} {'Δ':>12}")
    print("-" * 62)
    for label, key, scale in nav_metrics:
        bv = agg(base, key) * scale
        cv = agg(cache, key) * scale
        delta = cv - bv
        sign = "+" if delta >= 0 else ""
        fmt = ".1f" if scale == 100 else ".2f"
        print(f"{label:<20} {bv:>12{fmt}} {cv:>12{fmt}} {sign}{delta:>11{fmt}}")

    # ── Efficiency Metrics ──
    print("\n" + "=" * 62)
    print(f"{'Efficiency':<20} {'Baseline':>12} {'VLN-Cache':>12} {'Δ':>12}")
    print("-" * 62)
    eff_metrics = [
        ("Latency (ms)",    "action_latency_ms", ".1f"),
        ("Ctrl Freq (Hz)",  "control_freq_hz",   ".2f"),
        ("Peak GPU (MB)",   "peak_gpu_mb",       ".0f"),
        ("Cache Reuse (%)", None,                ".1f"),  # special
        ("Cache OH (ms)",   None,                ".2f"),  # special
    ]
    for label, key, fmt in eff_metrics:
        if key:
            bv = agg(base, key)
            cv = agg(cache, key)
        elif "Reuse" in label:
            bv = 0.0
            cv = agg_cache(cache, 'mean_reuse_ratio') * 100
        elif "OH" in label:
            bv = 0.0
            cv = agg_cache(cache, 'cache_overhead_ms_mean')
        delta = cv - bv
        sign = "+" if delta >= 0 else ""
        print(f"{label:<20} {bv:>12{fmt}} {cv:>12{fmt}} {sign}{delta:>11{fmt}}")

    print("=" * 62)

    # ── Save JSON ──
    out = {
        "baseline_episodes": len(base),
        "cache_episodes": len(cache),
        "baseline": {m[0]: agg(base, m[1]) * m[2] for m in nav_metrics},
        "vln_cache": {m[0]: agg(cache, m[1]) * m[2] for m in nav_metrics},
    }
    for label, key, fmt in eff_metrics:
        if key:
            out["baseline"][label] = agg(base, key)
            out["vln_cache"][label] = agg(cache, key)
        elif "Reuse" in label:
            out["baseline"][label] = 0.0
            out["vln_cache"][label] = agg_cache(cache, 'mean_reuse_ratio') * 100
        elif "OH" in label:
            out["baseline"][label] = 0.0
            out["vln_cache"][label] = agg_cache(cache, 'cache_overhead_ms_mean')

    comp_path = os.path.join(cache_dir, "comparison_100.json")
    with open(comp_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {comp_path}")

if __name__ == "__main__":
    main()
