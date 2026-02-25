import argparse
import json
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


def plot_fig2(step_rows, out_dir: Path):
    raw_vals, aligned_vals, delta_vals = [], [], []
    for row in step_rows:
        sim = row.get('similarity', {})
        if 'raw_mean' in sim and 'aligned_mean' in sim and 'delta_mean' in sim:
            raw_vals.append(float(sim['raw_mean']))
            aligned_vals.append(float(sim['aligned_mean']))
            delta_vals.append(float(sim['delta_mean']))

    if not delta_vals:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].boxplot([raw_vals, aligned_vals], labels=['raw_mean', 'aligned_mean'])
    axes[0].set_title('Fig2(a) Similarity Distribution')
    axes[0].set_ylabel('cosine similarity')

    axes[1].hist(delta_vals, bins=40, color='#1f77b4', alpha=0.85)
    axes[1].axvline(np.mean(delta_vals), color='red', linestyle='--', label=f"mean={np.mean(delta_vals):.3f}")
    axes[1].set_title('Fig2(b) delta_mean Histogram')
    axes[1].set_xlabel('aligned_mean - raw_mean')
    axes[1].legend()

    n = min(300, len(raw_vals))
    x = np.arange(n)
    axes[2].plot(x, np.array(raw_vals[:n]), label='raw_mean', linewidth=1.0)
    axes[2].plot(x, np.array(aligned_vals[:n]), label='aligned_mean', linewidth=1.0)
    axes[2].set_title('Fig2(c) Trajectory Slice (first 300 valid steps)')
    axes[2].set_xlabel('valid step index')
    axes[2].set_ylabel('cosine similarity')
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_dir / 'fig2_viewpoint_mismatch.png', dpi=220)
    plt.close(fig)


def plot_fig3(progress_rows, s2_rows, out_dir: Path):
    s2_calls = [float(r['system2_calls']) for r in progress_rows if 'system2_calls' in r]
    prompt = [float(r.get('prompt_len', 0)) for r in s2_rows]
    gen_ms = [float(r.get('generate_ms', 0)) for r in s2_rows]
    gen_len = [float(r.get('gen_len', 0)) for r in s2_rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    if s2_calls:
        axes[0].hist(s2_calls, bins=25, color='#2ca02c', alpha=0.85)
        axes[0].axvline(np.mean(s2_calls), color='red', linestyle='--', label=f"mean={np.mean(s2_calls):.2f}")
        axes[0].set_title('Fig3(a) System2 Calls per Episode')
        axes[0].set_xlabel('system2_calls')
        axes[0].legend()
    else:
        axes[0].set_title('Fig3(a) no system2_calls in progress log')

    axes[1].scatter(prompt, gen_ms, s=8, alpha=0.35, color='#ff7f0e')
    if prompt and gen_ms:
        coeff = np.polyfit(prompt, gen_ms, 1)
        x = np.linspace(min(prompt), max(prompt), 100)
        axes[1].plot(x, coeff[0] * x + coeff[1], color='black', linewidth=1.2, label='linear fit')
        axes[1].legend()
    axes[1].set_title('Fig3(b) Prompt Length vs Generate Latency')
    axes[1].set_xlabel('prompt_len')
    axes[1].set_ylabel('generate_ms')

    axes[2].hist(gen_len, bins=np.arange(-0.5, 15.5, 1.0), color='#9467bd', alpha=0.9)
    axes[2].set_title('Fig3(c) Generated Token Length Distribution')
    axes[2].set_xlabel('gen_len')

    fig.tight_layout()
    fig.savefig(out_dir / 'fig3_semantic_and_s2_behavior.png', dpi=220)
    plt.close(fig)


def plot_fig5(progress_rows, s2_rows, out_dir: Path):
    sr = np.mean([float(r.get('success', 0.0)) for r in progress_rows])
    spl = np.mean([float(r.get('spl', 0.0)) for r in progress_rows])
    gen_ms = np.mean([float(r.get('generate_ms', 0.0)) for r in s2_rows])

    fig, ax1 = plt.subplots(figsize=(7.2, 5.2))
    ax1.scatter([gen_ms], [sr], color='#d62728', s=80, label='SR anchor')
    ax1.set_xlabel('Mean System2 generate latency (ms)')
    ax1.set_ylabel('SR', color='#d62728')
    ax1.tick_params(axis='y', labelcolor='#d62728')
    ax1.set_title('Fig5 Baseline Anchor (run01)')
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.scatter([gen_ms], [spl], color='#1f77b4', s=80, label='SPL anchor')
    ax2.set_ylabel('SPL', color='#1f77b4')
    ax2.tick_params(axis='y', labelcolor='#1f77b4')

    ax1.annotate(f"SR={sr:.3f}", (gen_ms, sr), textcoords='offset points', xytext=(10, 10))
    ax2.annotate(f"SPL={spl:.3f}", (gen_ms, spl), textcoords='offset points', xytext=(10, -15))

    fig.tight_layout()
    fig.savefig(out_dir / 'fig5_pareto_anchor.png', dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=str, default='vis_analyze/data/raw/run01/step_log_rank0.jsonl')
    parser.add_argument('--s2', type=str, default='vis_analyze/data/raw/run01/s2_log_rank0.jsonl')
    parser.add_argument('--progress', type=str, default='vis_analyze/data/eval_output/run01/progress.json')
    parser.add_argument('--out', type=str, default='vis_analyze/reports/run01/figures')
    args = parser.parse_args()

    out_dir = Path(args.out)
    ensure_dir(out_dir)

    step_rows = load_jsonl(args.step)
    s2_rows = load_jsonl(args.s2)
    progress_rows = load_jsonl(args.progress)

    plot_fig2(step_rows, out_dir)
    plot_fig3(progress_rows, s2_rows, out_dir)
    plot_fig5(progress_rows, s2_rows, out_dir)

    print(f"Saved figures to: {out_dir}")


if __name__ == '__main__':
    main()
