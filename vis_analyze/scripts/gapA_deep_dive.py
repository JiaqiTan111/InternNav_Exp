import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path):
    rows = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def quantile(v, p):
    v = sorted(v)
    if not v:
        return None
    i = (len(v) - 1) * p
    lo = int(i)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] * (hi - i) + v[hi] * (i - lo)


def bootstrap_ci_mean(vals, n_boot=1000, alpha=0.05, seed=42):
    vals = np.asarray(vals, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    means = vals[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=str, default='vis_analyze/data/raw/run01/step_log_rank0.jsonl')
    parser.add_argument('--episode', type=str, default='vis_analyze/data/raw/run01/episode_log_rank0.jsonl')
    parser.add_argument('--out_dir', type=str, default='vis_analyze/reports/run01/deep_dive')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    step_rows = load_jsonl(args.step)
    ep_rows = load_jsonl(args.episode)

    # 1) Overall stats on delta
    all_delta = []
    for r in step_rows:
        sim = r.get('similarity', {})
        if 'delta_mean' in sim:
            all_delta.append(float(sim['delta_mean']))

    overall = {
        'count': len(all_delta),
        'mean': float(np.mean(all_delta)) if all_delta else None,
        'p10': quantile(all_delta, 0.1),
        'p50': quantile(all_delta, 0.5),
        'p90': quantile(all_delta, 0.9),
        'ci95_mean': bootstrap_ci_mean(all_delta) if all_delta else None,
    }

    # 2) Phase analysis (early/mid/late by normalized step index in each episode)
    grouped = defaultdict(list)
    for r in step_rows:
        sim = r.get('similarity', {})
        if 'delta_mean' in sim:
            key = (r.get('scene_id'), r.get('episode_id'))
            grouped[key].append((int(r.get('step_id', 0)), float(sim['delta_mean'])))

    phase_vals = {'early': [], 'mid': [], 'late': []}
    for _, arr in grouped.items():
        arr.sort(key=lambda x: x[0])
        n = len(arr)
        if n < 3:
            continue
        for i, (_, d) in enumerate(arr):
            ratio = i / (n - 1)
            if ratio < 1 / 3:
                phase_vals['early'].append(d)
            elif ratio < 2 / 3:
                phase_vals['mid'].append(d)
            else:
                phase_vals['late'].append(d)

    phase_stats = {}
    for k, vals in phase_vals.items():
        if vals:
            phase_stats[k] = {
                'count': len(vals),
                'mean': float(np.mean(vals)),
                'p50': quantile(vals, 0.5),
                'ci95_mean': bootstrap_ci_mean(vals),
            }

    # 3) Scene-level heterogeneity (top scenes by sample count)
    scene_delta = defaultdict(list)
    for r in step_rows:
        sim = r.get('similarity', {})
        if 'delta_mean' in sim:
            scene_delta[r.get('scene_id')].append(float(sim['delta_mean']))

    scene_items = sorted(scene_delta.items(), key=lambda kv: len(kv[1]), reverse=True)
    top_scene_items = scene_items[:8]
    scene_stats = {
        s: {
            'count': len(v),
            'mean': float(np.mean(v)),
            'p50': quantile(v, 0.5),
            'ci95_mean': bootstrap_ci_mean(v),
        }
        for s, v in top_scene_items
    }

    # 4) Episode-level relation to success/NE
    ep_metrics = {(r.get('scene_id'), r.get('episode_id')): r for r in ep_rows}
    ep_delta_mean = {}
    for key, arr in grouped.items():
        ep_delta_mean[key] = float(np.mean([x[1] for x in arr]))

    success_delta = []
    fail_delta = []
    ne_pairs = []
    for key, d in ep_delta_mean.items():
        info = ep_metrics.get(key)
        if not info:
            continue
        suc = float(info.get('success', 0.0))
        ne = float(info.get('ne', 0.0))
        ne_pairs.append((d, ne))
        if suc > 0.5:
            success_delta.append(d)
        else:
            fail_delta.append(d)

    def corr(x, y):
        if len(x) < 3:
            return None
        x = np.asarray(x)
        y = np.asarray(y)
        if np.std(x) < 1e-9 or np.std(y) < 1e-9:
            return None
        return float(np.corrcoef(x, y)[0, 1])

    relation_stats = {
        'success_group': {
            'count': len(success_delta),
            'mean_delta': float(np.mean(success_delta)) if success_delta else None,
        },
        'failure_group': {
            'count': len(fail_delta),
            'mean_delta': float(np.mean(fail_delta)) if fail_delta else None,
        },
        'corr_delta_ne': corr([x for x, _ in ne_pairs], [y for _, y in ne_pairs]),
    }

    summary = {
        'overall_delta': overall,
        'phase_stats': phase_stats,
        'scene_stats_top8': scene_stats,
        'episode_relation': relation_stats,
    }

    (out_dir / 'gapA_deep_dive_stats.json').write_text(json.dumps(summary, indent=2))

    # ----- Figure -----
    plt.rcParams.update(
        {
            'font.size': 10,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'grid.alpha': 0.25,
            'grid.linestyle': '--',
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # panel 1: phase boxplot
    labels = ['early', 'mid', 'late']
    box_vals = [phase_vals[k] for k in labels if len(phase_vals[k]) > 0]
    box_labels = [k for k in labels if len(phase_vals[k]) > 0]
    axes[0, 0].boxplot(box_vals, labels=box_labels, patch_artist=True, boxprops=dict(facecolor='#72B7B2', alpha=0.7))
    axes[0, 0].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[0, 0].set_title('A) Delta by Trajectory Phase')
    axes[0, 0].set_ylabel('delta_mean')

    # panel 2: top scene means with CI
    scene_names = list(scene_stats.keys())
    means = [scene_stats[s]['mean'] for s in scene_names]
    ci_l = [scene_stats[s]['ci95_mean'][0] for s in scene_names]
    ci_h = [scene_stats[s]['ci95_mean'][1] for s in scene_names]
    x = np.arange(len(scene_names))
    yerr = [np.array(means) - np.array(ci_l), np.array(ci_h) - np.array(means)]
    axes[0, 1].bar(x, means, color='#4C78A8', alpha=0.85)
    axes[0, 1].errorbar(x, means, yerr=yerr, fmt='none', ecolor='black', elinewidth=1, capsize=3)
    axes[0, 1].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(scene_names, rotation=40, ha='right')
    axes[0, 1].set_title('B) Scene-wise Delta Mean (Top-8 by Samples)')

    # panel 3: success vs failure
    vals = []
    labels = []
    if success_delta:
        vals.append(success_delta)
        labels.append('success')
    if fail_delta:
        vals.append(fail_delta)
        labels.append('failure')
    axes[1, 0].boxplot(vals, labels=labels, patch_artist=True, boxprops=dict(facecolor='#F58518', alpha=0.75))
    axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1, 0].set_title('C) Episode Delta by Outcome')
    axes[1, 0].set_ylabel('episode mean delta')

    # panel 4: delta vs NE
    if ne_pairs:
        dx = np.array([p[0] for p in ne_pairs])
        dy = np.array([p[1] for p in ne_pairs])
        axes[1, 1].scatter(dx, dy, s=18, alpha=0.65, color='#E45756')
        if len(dx) > 1:
            coeff = np.polyfit(dx, dy, 1)
            xs = np.linspace(dx.min(), dx.max(), 100)
            axes[1, 1].plot(xs, coeff[0] * xs + coeff[1], color='black', linewidth=1.2)
        axes[1, 1].set_title('D) Episode Delta vs Navigation Error')
        axes[1, 1].set_xlabel('episode mean delta')
        axes[1, 1].set_ylabel('NE')
        if relation_stats['corr_delta_ne'] is not None:
            axes[1, 1].text(
                0.03,
                0.95,
                f"corr={relation_stats['corr_delta_ne']:.3f}",
                transform=axes[1, 1].transAxes,
                va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
            )

    fig.suptitle('Gap A Deep Dive: Where and When Matching Failure Happens', fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_dir / 'fig_gapA_deep_dive.png', dpi=300)
    plt.close(fig)

    print(f"Saved stats: {out_dir / 'gapA_deep_dive_stats.json'}")
    print(f"Saved figure: {out_dir / 'fig_gapA_deep_dive.png'}")


if __name__ == '__main__':
    main()
