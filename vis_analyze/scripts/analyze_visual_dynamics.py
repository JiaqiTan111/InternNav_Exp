import argparse
import json
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="vis_analyze/data/raw/step_log_rank0.jsonl")
    parser.add_argument("--output", type=str, default="vis_analyze/reports/visual_dynamics_summary.json")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    raw_mean = []
    aligned_mean = []
    delta_mean = []
    with open(args.input, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sim = row.get("similarity", {})
            if "raw_mean" in sim:
                raw_mean.append(sim["raw_mean"])
            if "aligned_mean" in sim:
                aligned_mean.append(sim["aligned_mean"])
            if "delta_mean" in sim:
                delta_mean.append(sim["delta_mean"])

    summary = {
        "num_steps": len(raw_mean),
        "raw_mean_avg": float(np.mean(raw_mean)) if raw_mean else None,
        "aligned_mean_avg": float(np.mean(aligned_mean)) if aligned_mean else None,
        "delta_mean_avg": float(np.mean(delta_mean)) if delta_mean else None,
    }

    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
