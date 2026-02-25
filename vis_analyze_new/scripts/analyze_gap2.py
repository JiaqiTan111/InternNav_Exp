import argparse
import json
import re
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


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p.astype(np.float64), eps, None)
    q = np.clip(q.astype(np.float64), eps, None)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)))
    kl_qm = np.sum(q * (np.log(q) - np.log(m)))
    return float(0.5 * (kl_pm + kl_qm))


def resample_prob(v: np.ndarray, target_dim: int = 512, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    v = np.clip(v, eps, None)
    v = v / v.sum()
    if v.shape[0] == target_dim:
        return v

    src_x = np.linspace(0.0, 1.0, num=v.shape[0], dtype=np.float64)
    tgt_x = np.linspace(0.0, 1.0, num=target_dim, dtype=np.float64)
    out = np.interp(tgt_x, src_x, v)
    out = np.clip(out, eps, None)
    return out / out.sum()


def split_instruction_clauses(instruction: str):
    if not instruction:
        return ["stage_0"]
    clauses = [x.strip() for x in re.split(r",|;|\bthen\b|\band\b", instruction, flags=re.IGNORECASE) if x.strip()]
    return clauses if clauses else [instruction.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="vis_analyze_new/data/raw/step_log_rank0.jsonl")
    parser.add_argument("--output", type=str, default="vis_analyze_new/reports/gap2_timeseries.jsonl")
    parser.add_argument("--summary", type=str, default="vis_analyze_new/reports/gap2_summary.json")
    parser.add_argument("--topk", type=int, default=32)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input))
    grouped = defaultdict(list)
    for row in rows:
        if row.get("saliency_path"):
            grouped[(row.get("scene_id"), row.get("episode_id"))].append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shifts_all, rel_all = [], []
    n_points = 0
    with output_path.open("w") as fout:
        for (scene_id, episode_id), values in grouped.items():
            values.sort(key=lambda x: int(x.get("step_id", 0)))
            if len(values) < 2:
                continue

            vectors = []
            for row in values:
                root = Path(args.input).parent
                sal_path = root / row["saliency_path"]
                if not sal_path.exists():
                    sal_path = Path("vis_analyze_new/data/raw") / row["saliency_path"]
                sal = np.load(sal_path)["saliency"].astype(np.float64)
                sal = resample_prob(sal, target_dim=512)
                vectors.append(sal)

            shifts = [0.0]
            for i in range(1, len(vectors)):
                shifts.append(js_divergence(vectors[i], vectors[i - 1]))

            anchor = int(np.argmax(shifts))
            topk = min(args.topk, vectors[anchor].shape[0])
            lm_idx = np.argpartition(-vectors[anchor], kth=topk - 1)[:topk]
            bg_mask = np.ones(vectors[anchor].shape[0], dtype=bool)
            bg_mask[lm_idx] = False

            instruction = str(values[0].get("instruction", ""))
            clauses = split_instruction_clauses(instruction)

            for i, row in enumerate(values):
                rel_lm = float(vectors[i][lm_idx].mean())
                rel_bg = float(vectors[i][bg_mask].mean()) if np.any(bg_mask) else 0.0
                stage_id = int(min(len(clauses) - 1, (i * len(clauses)) // max(len(values), 1)))
                out = {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "step_id": int(row.get("step_id", i)),
                    "stage_id": stage_id,
                    "num_stages": len(clauses),
                    "shift": float(shifts[i]),
                    "rel_lm": rel_lm,
                    "rel_bg": rel_bg,
                }
                fout.write(json.dumps(out) + "\n")
                n_points += 1
                shifts_all.append(shifts[i])
                rel_all.append(rel_lm)

    summary = {
        "num_points": n_points,
        "shift_mean": float(np.mean(shifts_all)) if shifts_all else None,
        "shift_p90": float(np.percentile(shifts_all, 90)) if shifts_all else None,
        "rel_lm_mean": float(np.mean(rel_all)) if rel_all else None,
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
