import argparse
import json
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="vis_analyze/data/raw/s2_log_rank0.jsonl")
    parser.add_argument("--output", type=str, default="vis_analyze/reports/s2_efficiency_summary.json")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    prompt_lens, gen_lens, total_lens = [], [], []
    preprocess_ms, generate_ms, decode_ms = [], [], []

    with open(args.input, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_lens.append(float(row.get("prompt_len", 0)))
            gen_lens.append(float(row.get("gen_len", 0)))
            total_lens.append(float(row.get("total_len", 0)))
            preprocess_ms.append(float(row.get("preprocess_ms", 0)))
            generate_ms.append(float(row.get("generate_ms", 0)))
            decode_ms.append(float(row.get("decode_ms", 0)))

    summary = {
        "num_s2_calls": len(prompt_lens),
        "prompt_tokens_mean": float(np.mean(prompt_lens)) if prompt_lens else None,
        "gen_tokens_mean": float(np.mean(gen_lens)) if gen_lens else None,
        "total_tokens_mean": float(np.mean(total_lens)) if total_lens else None,
        "preprocess_ms_mean": float(np.mean(preprocess_ms)) if preprocess_ms else None,
        "generate_ms_mean": float(np.mean(generate_ms)) if generate_ms else None,
        "decode_ms_mean": float(np.mean(decode_ms)) if decode_ms else None,
    }

    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
