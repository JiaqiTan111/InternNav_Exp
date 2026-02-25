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
    available = set(plt.style.available)
    if 'seaborn-v0_8-whitegrid' in available:
        plt.style.use('seaborn-v0_8-whitegrid')
    elif 'seaborn-whitegrid' in available:
        plt.style.use('seaborn-whitegrid')
    else:
        plt.style.use('ggplot')


def build_step_with_yaw_delta(step_rows):
    grouped = defaultdict(list)
    for row in step_rows:
        key = (row.get('scene_id'), row.get('episode_id'))
        grouped[key].append(row)

    merged = []
    for _, rows in grouped.items():
        rows.sort(key=lambda r: int(r.get('step_id', 0)))
        prev_yaw = None
        for row in rows:
            sim = row.get('similarity', {})
            if 'raw_mean' in sim and 'aligned_mean' in sim and 'delta_mean' in sim:
                cur_yaw = row.get('yaw', None)
                yaw_delta_abs = None
                if prev_yaw is not None and cur_yaw is not None:
                    yaw_delta_abs = abs(float(cur_yaw) - float(prev_yaw))
                merged.append(
                    {
                        'raw': float(sim['raw_mean']),
                        'aligned': float(sim['aligned_mean']),
                        'delta': float(sim['delta_mean']),
                        'yaw_delta_abs': yaw_delta_abs,
                    }
                )
            prev_yaw = row.get('yaw', prev_yaw)
    return merged


def moving_avg(values, window=60):
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(np.array(values), kernel, mode='valid')


def plot_gap_a(step_rows, out_dir: Path):
    data = build_step_with_yaw_delta(step_rows)
    if not data:
        return

    raw = np.array([d['raw'] for d in data], dtype=np.float32)
    aligned = np.array([d['aligned'] for d in data], dtype=np.float32)
    delta = np.array([d['delta'] for d in data], dtype=np.float32)

    yaw = np.array([d['yaw_delta_abs'] if d['yaw_delta_abs'] is not None else np.nan for d in data], dtype=np.float32)
    valid_yaw = ~np.isnan(yaw)

    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    bins = np.linspace(min(raw.min(), aligned.min()), max(raw.max(), aligned.max()), 50)
    axes[0, 0].hist(raw, bins=bins, alpha=0.55, label='raw similarity', color='#4C78A8', density=True)
    axes[0, 0].hist(aligned, bins=bins, alpha=0.55, label='aligned similarity', color='#F58518', density=True)
    axes[0, 0].axvline(raw.mean(), color='#4C78A8', linestyle='--', linewidth=1.3)
    axes[0, 0].axvline(aligned.mean(), color='#F58518', linestyle='--', linewidth=1.3)
    axes[0, 0].set_title('A1. Similarity Distribution: Raw vs Naive-Align')
    axes[0, 0].set_xlabel('patch cosine similarity')
    axes[0, 0].legend(frameon=True)

    axes[0, 1].hist(delta, bins=60, color='#E45756', alpha=0.9)
    axes[0, 1].axvline(0.0, color='black', linestyle='--', linewidth=1.2)
    axes[0, 1].axvline(delta.mean(), color='#1f1f1f', linestyle='-', linewidth=1.2)
    ratio_better = float(np.mean(delta > 0))
    axes[0, 1].set_title('A2. Delta Histogram (aligned - raw)')
    axes[0, 1].set_xlabel('delta_mean')
    axes[0, 1].text(
        0.02,
        0.95,
        f'mean={delta.mean():.3f}\nP(aligned>raw)={ratio_better:.3f}',
        transform=axes[0, 1].transAxes,
        va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
    )

    if valid_yaw.any():
        yaw_valid = yaw[valid_yaw]
        delta_valid = delta[valid_yaw]
        q = np.quantile(yaw_valid, [0.0, 0.25, 0.5, 0.75, 1.0])
        q = np.unique(q)
        if len(q) >= 3:
            centers = []
            means = []
            for i in range(len(q) - 1):
                left, right = q[i], q[i + 1]
                if i == len(q) - 2:
                    mask = (yaw_valid >= left) & (yaw_valid <= right)
                else:
                    mask = (yaw_valid >= left) & (yaw_valid < right)
                if mask.sum() > 0:
                    centers.append((left + right) / 2)
                    means.append(float(delta_valid[mask].mean()))
            axes[1, 0].plot(centers, means, marker='o', color='#72B7B2', linewidth=2)
            axes[1, 0].axhline(0.0, color='black', linestyle='--', linewidth=1.0)
            axes[1, 0].set_title('A3. Delta vs |ΔYaw| (quantile bins)')
            axes[1, 0].set_xlabel('|Δyaw|')
            axes[1, 0].set_ylabel('mean(delta_mean)')
        else:
            axes[1, 0].text(0.1, 0.5, 'Not enough yaw variation for binning')
            axes[1, 0].set_axis_off()
    else:
        axes[1, 0].text(0.1, 0.5, 'No valid yaw deltas in logs')
        axes[1, 0].set_axis_off()

    idx = np.random.RandomState(7).choice(len(delta), size=min(600, len(delta)), replace=False)
    axes[1, 1].scatter(raw[idx], aligned[idx], s=9, alpha=0.35, color='#54A24B')
    mn = min(raw.min(), aligned.min())
    mx = max(raw.max(), aligned.max())
    axes[1, 1].plot([mn, mx], [mn, mx], '--', color='black', linewidth=1.0)
    axes[1, 1].set_title('A4. Pairwise Similarity Scatter')
    axes[1, 1].set_xlabel('raw_mean')
    axes[1, 1].set_ylabel('aligned_mean')

    fig.suptitle('Gap A Evidence: Naive Alignment Fails to Recover Reusable Tokens', fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_dir / 'fig_gapA_viewpoint_failure_fancy.png', dpi=260)
    plt.close(fig)


def plot_gap_b(progress_rows, s2_rows, out_dir: Path):
    set_style()

    prompt = np.array([float(r.get('prompt_len', 0.0)) for r in s2_rows], dtype=np.float32)
    gen_len = np.array([float(r.get('gen_len', 0.0)) for r in s2_rows], dtype=np.float32)
    gen_ms = np.array([float(r.get('generate_ms', 0.0)) for r in s2_rows], dtype=np.float32)
    prep_ms = np.array([float(r.get('preprocess_ms', 0.0)) for r in s2_rows], dtype=np.float32)
    decode_ms = np.array([float(r.get('decode_ms', 0.0)) for r in s2_rows], dtype=np.float32)

    s2_calls = np.array([float(r['system2_calls']) for r in progress_rows if 'system2_calls' in r], dtype=np.float32)
    steps = np.array([float(r.get('steps', 0.0)) for r in progress_rows], dtype=np.float32)
    success = np.array([float(r.get('success', 0.0)) for r in progress_rows], dtype=np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].hexbin(prompt, gen_ms, gridsize=34, cmap='viridis', mincnt=1)
    if len(prompt) > 1:
        coeff = np.polyfit(prompt, gen_ms, 1)
        x = np.linspace(prompt.min(), prompt.max(), 120)
        axes[0, 0].plot(x, coeff[0] * x + coeff[1], color='white', linewidth=2)
    axes[0, 0].set_title('B1. Prompt Length vs Generation Latency')
    axes[0, 0].set_xlabel('prompt_len')
    axes[0, 0].set_ylabel('generate_ms')

    parts = [prep_ms.mean(), gen_ms.mean(), decode_ms.mean()]
    labels = ['preprocess', 'generate', 'decode']
    colors = ['#4C78A8', '#E45756', '#72B7B2']
    axes[0, 1].bar(labels, parts, color=colors)
    axes[0, 1].set_title('B2. Mean Latency Breakdown per S2 Call')
    axes[0, 1].set_ylabel('milliseconds')
    total = sum(parts)
    axes[0, 1].text(
        0.05,
        0.95,
        f'generate share = {parts[1] / total:.2%}',
        transform=axes[0, 1].transAxes,
        va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
    )

    bins = np.arange(-0.5, max(10.5, gen_len.max() + 1), 1.0)
    axes[1, 0].hist(gen_len, bins=bins, color='#F58518', alpha=0.9)
    axes[1, 0].axvline(gen_len.mean(), color='black', linestyle='--', linewidth=1.2)
    axes[1, 0].set_title('B3. Generated Token Length Distribution')
    axes[1, 0].set_xlabel('gen_len')
    axes[1, 0].set_ylabel('count')

    if len(s2_calls) > 0:
        colors_pts = np.where(success[: len(s2_calls)] > 0.5, '#54A24B', '#E45756')
        axes[1, 1].scatter(steps[: len(s2_calls)], s2_calls, s=20, alpha=0.65, c=colors_pts)
        if len(steps) > 1:
            coeff2 = np.polyfit(steps[: len(s2_calls)], s2_calls, 1)
            x2 = np.linspace(steps.min(), steps.max(), 100)
            axes[1, 1].plot(x2, coeff2[0] * x2 + coeff2[1], color='black', linewidth=1.2)
        axes[1, 1].set_title('B4. S2 Calls vs Episode Steps')
        axes[1, 1].set_xlabel('episode steps')
        axes[1, 1].set_ylabel('system2_calls')
    else:
        axes[1, 1].text(0.1, 0.5, 'No system2_calls fields in progress log')
        axes[1, 1].set_axis_off()

    fig.suptitle('Gap B Evidence: Static Decode/Refresh Budget Mismatch', fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_dir / 'fig_gapB_budget_mismatch_fancy.png', dpi=260)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=str, default='vis_analyze/data/raw/run01/step_log_rank0.jsonl')
    parser.add_argument('--s2', type=str, default='vis_analyze/data/raw/run01/s2_log_rank0.jsonl')
    parser.add_argument('--progress', type=str, default='vis_analyze/data/eval_output/run01/progress.json')
    parser.add_argument('--out', type=str, default='vis_analyze/reports/run01/figures_fancy')
    args = parser.parse_args()

    out_dir = Path(args.out)
    ensure_dir(out_dir)

    step_rows = load_jsonl(args.step)
    s2_rows = load_jsonl(args.s2)
    progress_rows = load_jsonl(args.progress)

    plot_gap_a(step_rows, out_dir)
    plot_gap_b(progress_rows, s2_rows, out_dir)

    print(f'Saved fancy figures to: {out_dir}')


if __name__ == '__main__':
    main()
