import argparse
import json
import os
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="vis_analyze/data/raw", help="raw data output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.root, exist_ok=True)

    manifest_path = os.path.join(args.root, "manifest.json")
    if not os.path.exists(manifest_path):
        with open(manifest_path, "w") as f:
            json.dump(
                {
                    "created_at": datetime.now().isoformat(),
                    "purpose": "Observation and Motivation raw artifacts",
                    "files": [
                        "step_log_rank0.jsonl",
                        "s2_log_rank0.jsonl",
                        "episode_log_rank0.jsonl",
                    ],
                },
                f,
                indent=2,
            )

    for name in ["step_log_rank0.jsonl", "s2_log_rank0.jsonl", "episode_log_rank0.jsonl"]:
        path = os.path.join(args.root, name)
        if not os.path.exists(path):
            with open(path, "w"):
                pass

    print(f"Bootstrapped data files at: {args.root}")


if __name__ == "__main__":
    main()
