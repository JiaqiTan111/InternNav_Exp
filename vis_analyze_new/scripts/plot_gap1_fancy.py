import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_jsonl(path: Path):
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def aggregate_by_step(rows, key):
    bucket = defaultdict(list)
    for row in rows:
        bucket[int(row["step_id"])].append(float(row.get(key, 0.0)))
    xs = sorted(bucket.keys())
    mean = np.array([np.mean(bucket[x]) for x in xs], dtype=np.float64)
    p10 = np.array([np.percentile(bucket[x], 10) for x in xs], dtype=np.float64)
    p90 = np.array([np.percentile(bucket[x], 90) for x in xs], dtype=np.float64)
    return np.array(xs), mean, p10, p90


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="vis_analyze_new/reports/gap1_timeseries.jsonl")
    parser.add_argument("--output", type=str, default="vis_analyze_new/reports/fig_g1_fancy.png")
    parser.add_argument("--pseudo-threshold", type=float, default=0.2)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input))
    if not rows:
        raise RuntimeError("No data rows found.")

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)

    x, yaw_m, yaw_l, yaw_h = aggregate_by_step(rows, "delta_yaw")
    _, trans_m, trans_l, trans_h = aggregate_by_step(rows, "delta_trans")
    _, pos_m, pos_l, pos_h = aggregate_by_step(rows, "sim_pos")
    _, align_m, align_l, align_h = aggregate_by_step(rows, "sim_align")
    _, pseudo_m, pseudo_l, pseudo_h = aggregate_by_step(rows, "pseudo_dynamic_rate")
    _, lost_m, lost_l, lost_h = aggregate_by_step(rows, "lost_reuse")

    turn_rate = defaultdict(list)
    for row in rows:
        turn_rate[int(row["step_id"])].append(1.0 if int(row.get("action", 0)) in (2, 3) else 0.0)
    turn_ratio = np.array([np.mean(turn_rate[int(t)]) for t in x], dtype=np.float64)

    axes[0].plot(x, yaw_m, color="#4C78A8", label="|Δyaw|")
    axes[0].fill_between(x, yaw_l, yaw_h, color="#4C78A8", alpha=0.2)
    axes[0].plot(x, trans_m, color="#72B7B2", label="|Δtrans|")
    axes[0].fill_between(x, trans_l, trans_h, color="#72B7B2", alpha=0.2)
    axes[0].set_ylabel("Motion")
    axes[0].legend(loc="upper right")
    axes[0].set_title("(a) Ego-motion")

    for idx, ratio in enumerate(turn_ratio):
        if ratio > 0.5:
            axes[0].axvspan(x[idx] - 0.5, x[idx] + 0.5, color="#BBBBBB", alpha=0.2)

    axes[1].plot(x, pos_m, color="#54A24B", label="Position-wise")
    axes[1].fill_between(x, pos_l, pos_h, color="#54A24B", alpha=0.2)
    axes[1].plot(x, align_m, color="#E45756", label="View-aligned")
    axes[1].fill_between(x, align_l, align_h, color="#E45756", alpha=0.2)
    axes[1].set_ylabel("Similarity")
    axes[1].legend(loc="lower right")
    axes[1].set_title("(b) Position-wise vs View-aligned Similarity")

    axes[2].plot(x, pseudo_m, color="#F58518", label="Pseudo-dynamic rate")
    axes[2].fill_between(x, pseudo_l, pseudo_h, color="#F58518", alpha=0.2)
    axes[2].axhline(args.pseudo_threshold, linestyle="--", color="black", linewidth=1)
    high = pseudo_m > args.pseudo_threshold
    if np.any(high):
        axes[2].fill_between(x, 0, 1, where=high, transform=axes[2].get_xaxis_transform(), alpha=0.12, color="#F58518")
    axes[2].set_ylabel("ρ_pseudo")
    axes[2].set_title("(c) Pseudo-dynamic Rate")

    axes[3].plot(x, lost_m, color="#B279A2", label="Δr = r_align - r_pos")
    axes[3].fill_between(x, lost_l, lost_h, color="#B279A2", alpha=0.2)
    axes[3].axhline(0.0, linestyle="--", color="black", linewidth=1)
    axes[3].set_ylabel("Lost-reuse")
    axes[3].set_xlabel("Step")
    axes[3].set_title("(d) Missed Reuse by Position-wise Matching")

    for ax in axes:
        ax.grid(alpha=0.25, linestyle="--")

    fig.suptitle("Figure G1: Viewpoint-induced Pseudo-dynamic in VLN", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
