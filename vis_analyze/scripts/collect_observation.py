import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="vis_analyze/configs/habitat_dual_system_observation_cfg.py",
        help="evaluation config path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    eval_entry = repo_root / "scripts" / "eval" / "eval.py"
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path

    cmd = [sys.executable, str(eval_entry), "--config", str(config_path)]
    print("Running:", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    subprocess.run(cmd, check=True, env=env, cwd=str(repo_root))


if __name__ == "__main__":
    main()
