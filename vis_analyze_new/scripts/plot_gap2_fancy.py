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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="vis_analyze_new/reports/gap2_timeseries.jsonl")
    parser.add_argument("--output", type=str, default="vis_analyze_new/reports/fig_g2_fancy.png")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input))
    if not rows:
        raise RuntimeError("No data rows found.")

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["scene_id"], row["episode_id"])].append(row)

    # Use the longest trajectory for clean visualization.
    key = max(grouped.keys(), key=lambda k: len(grouped[k]))
    values = sorted(grouped[key], key=lambda x: int(x["step_id"]))

    x = np.array([int(v["step_id"]) for v in values], dtype=np.int32)
    stage = np.array([int(v["stage_id"]) for v in values], dtype=np.int32)
    shift = np.array([float(v["shift"]) for v in values], dtype=np.float64)
    rel_lm = np.array([float(v["rel_lm"]) for v in values], dtype=np.float64)
    rel_bg = np.array([float(v.get("rel_bg", 0.0)) for v in values], dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)

    axes[0].step(x, stage, where="post", color="#4C78A8", linewidth=2)
    axes[0].set_ylabel("Stage ID")
    axes[0].set_title("(a) Instruction Stage (Clause Bands)")
    boundaries = np.where(np.diff(stage) != 0)[0]
    for b in boundaries:
        axes[0].axvline(x[b + 1], color="#4C78A8", linestyle="--", alpha=0.5)

    axes[1].plot(x, shift, color="#E45756", linewidth=2)
    axes[1].set_ylabel("Shift (JS)")
    axes[1].set_title("(b) Relevance Shift Curve")
    if len(shift) >= 3:
        peak_ids = np.argsort(-shift)[:3]
        for pid in peak_ids:
            axes[1].scatter(x[pid], shift[pid], color="black", s=28)

    axes[2].plot(x, rel_lm, color="#54A24B", linewidth=2, label="Landmark relevance")
    axes[2].plot(x, rel_bg, color="#B279A2", linewidth=1.8, label="Background relevance")
    axes[2].fill_between(x, rel_lm, rel_bg, where=(rel_lm >= rel_bg), alpha=0.12, color="#54A24B")
    axes[2].set_ylabel("Saliency")
    axes[2].set_xlabel("Step")
    axes[2].set_title("(c) Landmark Relevance Rise-and-Fall")
    axes[2].legend(loc="upper right")

    for ax in axes:
        ax.grid(alpha=0.25, linestyle="--")

    fig.suptitle("Figure G2: Stage-wise Semantic Relevance Shift in VLN", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
