#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from datetime import datetime


def run(cmd):
    print(">", " ".join(cmd))
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser(description="Parallel self-play data generator (wrapper).")
    parser.add_argument("--total-games", type=int, default=200000, help="Total games to generate.")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2), help="Parallel workers.")
    parser.add_argument("--out-dir", default="logs/train_data", help="Output directory.")
    parser.add_argument("--output", default=None, help="Optional explicit output path.")
    parser.add_argument("--log-mode", default="train", choices=["train", "compact", "delta"], help="Simulator log mode.")
    parser.add_argument("--policy-human", default="heuristic_v1")
    parser.add_argument("--policy-ai", default="heuristic_v2")
    parser.add_argument("--node", default="node")
    parser.add_argument("--script", default="scripts/simulate-ai-vs-ai.mjs")
    parser.add_argument("--keep-shards", action="store_true")
    args = parser.parse_args()

    if args.total_games <= 0:
        raise RuntimeError("total-games must be > 0")
    if args.total_games % 2 != 0:
        raise RuntimeError("total-games must be even")
    if args.workers <= 0:
        raise RuntimeError("workers must be > 0")

    os.makedirs(args.out_dir, exist_ok=True)
    if args.output:
        out_path = args.output
    else:
        stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        out_path = os.path.join(args.out_dir, f"train-{stamp}.jsonl")

    cmd = [
        sys.executable,
        "scripts/parallel_simulate_ai_vs_ai.py",
        str(args.total_games),
        "--workers",
        str(args.workers),
        "--output",
        out_path,
        "--node",
        args.node,
        "--script",
        args.script,
    ]
    if args.keep_shards:
        cmd.append("--keep-shards")
    cmd.extend(
        [
            "--",
            f"--log-mode={args.log_mode}",
            f"--policy-human={args.policy_human}",
            f"--policy-ai={args.policy_ai}",
        ]
    )
    run(cmd)
    print(f"generated: {out_path}")


if __name__ == "__main__":
    main()
