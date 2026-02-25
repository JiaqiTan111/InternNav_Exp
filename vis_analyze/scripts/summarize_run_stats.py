import argparse
import json
from pathlib import Path


def quantiles(vals, probs=(0.1, 0.5, 0.9)):
    vals = sorted(vals)
    out = {}
    for p in probs:
        idx = (len(vals) - 1) * p
        lo = int(idx)
        hi = min(lo + 1, len(vals) - 1)
        out[f"p{int(p * 100)}"] = vals[lo] * (hi - idx) + vals[hi] * (idx - lo)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", type=str, required=True)
    parser.add_argument("--step", type=str, required=True)
    parser.add_argument("--s2", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    progress_rows = [json.loads(l) for l in Path(args.progress).read_text().splitlines() if l.strip()]
    step_rows = [json.loads(l) for l in Path(args.step).read_text().splitlines() if l.strip()]
    s2_rows = [json.loads(l) for l in Path(args.s2).read_text().splitlines() if l.strip()]

    summary = {}

    summary["episodes"] = {
        "count": len(progress_rows),
        "sr_mean": sum(r.get("success", 0.0) for r in progress_rows) / len(progress_rows),
        "spl_mean": sum(r.get("spl", 0.0) for r in progress_rows) / len(progress_rows),
        "os_mean": sum(r.get("os", 0.0) for r in progress_rows) / len(progress_rows),
        "ne_mean": sum(r.get("ne", 0.0) for r in progress_rows) / len(progress_rows),
        "steps_mean": sum(r.get("steps", 0.0) for r in progress_rows) / len(progress_rows),
    }

    for key in ["steps", "ne", "system2_calls", "system2_prompt_tokens_mean", "system2_gen_tokens_mean"]:
        values = [float(r[key]) for r in progress_rows if key in r]
        if values:
            summary["episodes"][f"{key}_q"] = quantiles(values, (0.5, 0.9))

    similarity_rows = []
    for row in step_rows:
        similarity = row.get("similarity", {})
        if "delta_mean" in similarity:
            similarity_rows.append(similarity)

    summary["visual_similarity"] = {"valid_steps": len(similarity_rows)}
    if similarity_rows:
        delta_values = [float(s["delta_mean"]) for s in similarity_rows]
        raw_values = [float(s["raw_mean"]) for s in similarity_rows]
        aligned_values = [float(s["aligned_mean"]) for s in similarity_rows]
        summary["visual_similarity"].update(
            {
                "raw_mean_avg": sum(raw_values) / len(raw_values),
                "aligned_mean_avg": sum(aligned_values) / len(aligned_values),
                "delta_mean_avg": sum(delta_values) / len(delta_values),
                "delta_q": quantiles(delta_values, (0.1, 0.5, 0.9)),
                "aligned_better_ratio": sum(1 for value in delta_values if value > 0) / len(delta_values),
            }
        )

    summary["s2_efficiency"] = {"count": len(s2_rows)}
    for key in ["prompt_len", "gen_len", "total_len", "preprocess_ms", "generate_ms", "decode_ms"]:
        values = [float(r.get(key, 0.0)) for r in s2_rows]
        summary["s2_efficiency"][key] = {
            "mean": sum(values) / len(values),
            "q": quantiles(values, (0.5, 0.9)),
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
