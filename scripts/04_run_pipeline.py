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
    parser = argparse.ArgumentParser(description="Run 01->02->03 training pipeline.")
    parser.add_argument("--input", nargs="+", default=["logs/*.jsonl"], help="Input JSONL/glob for all stages.")
    parser.add_argument("--tag", default=None, help="Model/report tag suffix. Default: UTC timestamp.")
    parser.add_argument("--python", default=sys.executable or "python", help="Python executable path.")
    parser.add_argument("--policy-old", default=None, help="Optional old policy for stage 03 compare.")
    parser.add_argument("--value-old", default=None, help="Optional old value for stage 03 compare.")
    parser.add_argument("--skip-eval", action="store_true", help="Run only 01,02 and skip 03.")
    args = parser.parse_args()

    tag = args.tag or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    policy_new = f"models/policy-{tag}.json"
    value_new = f"models/value-{tag}.json"
    eval_out = f"logs/model-eval-{tag}.json"

    run([args.python, "scripts/01_train_policy.py", "--input", *args.input, "--output", policy_new])
    run([args.python, "scripts/02_train_value.py", "--input", *args.input, "--output", value_new])

    if not args.skip_eval:
        eval_cmd = [
            args.python,
            "scripts/03_evaluate.py",
            "--input",
            *args.input,
            "--policy-new",
            policy_new,
            "--output",
            eval_out,
        ]
        if args.policy_old:
            eval_cmd.extend(["--policy-old", args.policy_old])
        else:
            eval_cmd.extend(["--policy-old", policy_new])
        if args.value_old:
            eval_cmd.extend(["--value-old", args.value_old, "--value-new", value_new])
        run(eval_cmd)

    print(f"policy: {policy_new}")
    print(f"value:  {value_new}")
    if not args.skip_eval:
        print(f"eval:   {eval_out}")


if __name__ == "__main__":
    main()
