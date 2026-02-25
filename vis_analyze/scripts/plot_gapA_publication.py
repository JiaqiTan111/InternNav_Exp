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


def set_style():
    plt.rcParams.update(
        {
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'legend.fontsize': 9,
            'figure.titlesize': 15,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'grid.alpha': 0.25,
            'grid.linestyle': '--',
        }
    )


def extract_samples(step_rows):
    grouped = defaultdict(list)
    for row in step_rows:
        grouped[(row.get('scene_id'), row.get('episode_id'))].append(row)

    samples = []
    for _, rows in grouped.items():
        rows.sort(key=lambda r: int(r.get('step_id', 0)))
        prev_yaw = None
        for row in rows:
            sim = row.get('similarity', {})
            if 'raw_mean' in sim and 'aligned_mean' in sim and 'delta_mean' in sim:
                yaw = row.get('yaw', None)
                dyaw = None
                if prev_yaw is not None and yaw is not None:
                    dyaw = abs(float(yaw) - float(prev_yaw))
                samples.append(
                    {
                        'raw': float(sim['raw_mean']),
                        'aligned': float(sim['aligned_mean']),
                        'delta': float(sim['delta_mean']),
                        'dyaw': dyaw,
                    }
                )
            prev_yaw = row.get('yaw', prev_yaw)
    return samples


def ecdf(values):
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def bootstrap_ci_mean(values, n_boot=1200, alpha=0.05, seed=7):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo = np.quantile(means, alpha / 2)
    hi = np.quantile(means, 1 - alpha / 2)
    return float(lo), float(hi)


def effect_size_cohens_d(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if pooled < 1e-9:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def bin_with_ci(x, y, n_bins=6):
    valid = ~np.isnan(x)
    x = x[valid]
    y = y[valid]
    if len(x) < n_bins:
        return [], [], [], []

    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) <= 2:
        return [], [], [], []

    centers, means, lows, highs = [], [], [], []
    for i in range(len(edges) - 1):
        l, r = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            m = (x >= l) & (x <= r)
        else:
            m = (x >= l) & (x < r)
        vals = y[m]
        if len(vals) < 5:
            continue
        mu = vals.mean()
        se = vals.std(ddof=1) / np.sqrt(len(vals))
        ci = 1.96 * se
        centers.append((l + r) / 2)
        means.append(mu)
        lows.append(mu - ci)
        highs.append(mu + ci)
    return centers, means, lows, highs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=str, default='vis_analyze/data/raw/run01/step_log_rank0.jsonl')
    parser.add_argument('--out_dir', type=str, default='vis_analyze/reports/run01/figures_pub')
    parser.add_argument('--stats_out', type=str, default='vis_analyze/reports/run01/gapA_publication_stats.json')
    args = parser.parse_args()

    set_style()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    rows = load_jsonl(args.step)
    samples = extract_samples(rows)
    if not samples:
        raise RuntimeError('No valid similarity samples found')

    raw = np.array([s['raw'] for s in samples], dtype=np.float64)
    aligned = np.array([s['aligned'] for s in samples], dtype=np.float64)
    delta = np.array([s['delta'] for s in samples], dtype=np.float64)
    dyaw = np.array([np.nan if s['dyaw'] is None else s['dyaw'] for s in samples], dtype=np.float64)

    delta_ci = bootstrap_ci_mean(delta)
    p_positive = float((delta > 0).mean())
    cohen_d = effect_size_cohens_d(aligned, raw)

    small_mask = (dyaw < np.nanquantile(dyaw, 0.33))
    large_mask = (dyaw > np.nanquantile(dyaw, 0.66))
    small_delta = delta[small_mask & ~np.isnan(dyaw)]
    large_delta = delta[large_mask & ~np.isnan(dyaw)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Panel A: ECDF comparison
    xr, yr = ecdf(raw)
    xa, ya = ecdf(aligned)
    axes[0, 0].plot(xr, yr, color='#4C78A8', linewidth=2, label='Raw')
    axes[0, 0].plot(xa, ya, color='#F58518', linewidth=2, label='Aligned (naive)')
    axes[0, 0].axvline(raw.mean(), color='#4C78A8', linestyle='--', linewidth=1)
    axes[0, 0].axvline(aligned.mean(), color='#F58518', linestyle='--', linewidth=1)
    axes[0, 0].set_title('A) ECDF of Patch Similarity')
    axes[0, 0].set_xlabel('Similarity')
    axes[0, 0].set_ylabel('ECDF')
    axes[0, 0].legend(frameon=True)
    axes[0, 0].grid(True)

    # Panel B: delta density + CI
    axes[0, 1].hist(delta, bins=70, density=True, alpha=0.85, color='#E45756')
    axes[0, 1].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[0, 1].axvline(delta.mean(), color='black', linewidth=1.4)
    axes[0, 1].axvspan(delta_ci[0], delta_ci[1], color='gray', alpha=0.25, label='95% CI of mean')
    axes[0, 1].set_title('B) Delta Distribution: aligned - raw')
    axes[0, 1].set_xlabel('Delta Similarity')
    axes[0, 1].legend(frameon=True)
    axes[0, 1].text(
        0.02,
        0.95,
        f"mean={delta.mean():.4f}\n95%CI=[{delta_ci[0]:.4f},{delta_ci[1]:.4f}]\nP(delta>0)={p_positive:.3f}",
        transform=axes[0, 1].transAxes,
        va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
    )

    # Panel C: binned trend with confidence
    c, m, lo, hi = bin_with_ci(dyaw, delta, n_bins=7)
    if len(c) > 0:
        axes[1, 0].plot(c, m, marker='o', color='#54A24B', linewidth=2)
        axes[1, 0].fill_between(c, lo, hi, color='#54A24B', alpha=0.2)
        axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1, 0].set_title('C) Delta vs |ΔYaw| (quantile bins, 95% CI)')
    axes[1, 0].set_xlabel('|ΔYaw|')
    axes[1, 0].set_ylabel('Mean Delta Similarity')
    axes[1, 0].grid(True)

    # Panel D: small vs large yaw effect
    vals = []
    labels = []
    if len(small_delta) > 0:
        vals.append(small_delta)
        labels.append('Small |ΔYaw|')
    if len(large_delta) > 0:
        vals.append(large_delta)
        labels.append('Large |ΔYaw|')
    if len(vals) >= 1:
        axes[1, 1].boxplot(vals, labels=labels, patch_artist=True,
                           boxprops=dict(facecolor='#72B7B2', alpha=0.6))
    axes[1, 1].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1, 1].set_title('D) Delta Under Motion Regimes')
    axes[1, 1].set_ylabel('Delta Similarity')
    axes[1, 1].text(
        0.03,
        0.95,
        f"Cohen's d(aligned-raw)={cohen_d:.3f}",
        transform=axes[1, 1].transAxes,
        va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
    )

    fig.suptitle('Gap A Diagnosis (Publication Figure): Viewpoint-Induced Matching Failure', fontweight='bold')
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_dir / 'fig_gapA_publication.png', dpi=300)
    plt.close(fig)

    stats = {
        'num_samples': int(len(delta)),
        'raw_mean': float(raw.mean()),
        'aligned_mean': float(aligned.mean()),
        'delta_mean': float(delta.mean()),
        'delta_mean_ci95': [float(delta_ci[0]), float(delta_ci[1])],
        'p_delta_positive': p_positive,
        'cohens_d_aligned_minus_raw': float(cohen_d),
        'small_yaw_delta_mean': float(small_delta.mean()) if len(small_delta) else None,
        'large_yaw_delta_mean': float(large_delta.mean()) if len(large_delta) else None,
    }

    stats_path = Path(args.stats_out)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2))

    print(f"Saved figure: {out_dir / 'fig_gapA_publication.png'}")
    print(f"Saved stats: {stats_path}")


if __name__ == '__main__':
    main()
