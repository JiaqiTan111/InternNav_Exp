import argparse
import json
from collections import defaultdict
from pathlib import Path

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
    parser.add_argument("--input", type=str, default="vis_analyze_new/data/raw/step_log_rank0.jsonl")
    parser.add_argument("--output", type=str, default="vis_analyze_new/reports/gap1_timeseries.jsonl")
    parser.add_argument("--summary", type=str, default="vis_analyze_new/reports/gap1_summary.json")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("scene_id"), row.get("episode_id"))].append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pseudo_all, lost_all, delta_all, valid_all = [], [], [], []
    mode_counts = defaultdict(int)
    n = 0
    with output_path.open("w") as f:
        for (scene_id, episode_id), values in grouped.items():
            values.sort(key=lambda x: int(x.get("step_id", 0)))
            for row in values:
                sim = row.get("similarity", {})
                if "raw_mean" not in sim:
                    continue
                out = {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "step_id": int(row.get("step_id", 0)),
                    "action": int(row.get("action", 0)),
                    "delta_yaw": float(row.get("motion", {}).get("delta_yaw", 0.0) or 0.0),
                    "delta_trans": float(row.get("motion", {}).get("delta_trans", 0.0) or 0.0),
                    "sim_pos": float(sim.get("raw_mean", 0.0)),
                    "sim_align": float(sim.get("aligned_mean", sim.get("raw_mean", 0.0))),
                    "pseudo_dynamic_rate": float(sim.get("pseudo_dynamic_rate", 0.0)),
                    "reuse_pos": float(sim.get("raw_reuse_frac", 0.0)),
                    "reuse_align": float(sim.get("aligned_reuse_frac", sim.get("raw_reuse_frac", 0.0))),
                    "lost_reuse": float(sim.get("lost_reuse", 0.0)),
                    "aligned_valid_frac": float(sim.get("aligned_valid_frac", 0.0)),
                    "aligned_mode": sim.get("aligned_mode", None),
                }
                f.write(json.dumps(out) + "\n")
                n += 1
                delta_all.append(out["sim_align"] - out["sim_pos"])
                pseudo_all.append(out["pseudo_dynamic_rate"])
                lost_all.append(out["lost_reuse"])
                valid_all.append(out["aligned_valid_frac"])
                mode_counts[str(out["aligned_mode"])] += 1

    summary = {
        "num_points": n,
        "sim_delta_mean": float(np.mean(delta_all)) if delta_all else None,
        "pseudo_dynamic_mean": float(np.mean(pseudo_all)) if pseudo_all else None,
        "lost_reuse_mean": float(np.mean(lost_all)) if lost_all else None,
        "aligned_valid_frac_mean": float(np.mean(valid_all)) if valid_all else None,
        "aligned_mode_counts": dict(mode_counts),
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
