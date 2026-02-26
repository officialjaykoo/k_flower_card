#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Pipeline Stage: Optuna Tuning Wrapper (V7)
# Quick Read Map:
# 1) Define search space/constants
# 2) suggest_params + run_duel helpers
# 3) objective() scoring
# 4) main(): study run + artifact export

"""
scripts/optuna_v7.py - V7 Optuna tuner

Target:
  heuristic_v7_gold_digger (A) vs heuristic_v5 (B)

Notes:
  - This repository locks test duels to exactly 1000 games per trial.
  - Uses nonlinear objective with loss-cut penalties to target V5.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("optuna not installed. run: pip install optuna")
    sys.exit(1)

SELF_POLICY = "heuristic_v7_gold_digger"
OPPONENT_POLICY = "heuristic_v5"
DUEL_SCRIPT = "scripts/model_duel_worker.mjs"

GAMES = 1000
SEED = "optuna-v7"
MAX_STEPS = 600
TRIAL_TIMEOUT_SEC = 900

FLOAT_PARAMS = {
    # Gemini requested sharp-target ranges
    "matchBase": (12.0, 25.0),
    "comboBonus": (35.0, 60.0),
    "riskTolerance": (0.8, 2.0),
    "denialBonus": (15.0, 40.0),
    "goAggression": (0.2, 0.5),
    "antiPiBakBonus": (30.0, 100.0),
    # Requested focus parameter for V5 countering
    "comboBreakerBonus": (20.0, 80.0),
}

INT_PARAMS = {
    "lockProfitScore": (6, 12),
}


def suggest_params(trial):
    params = {}
    for name, (lo, hi) in FLOAT_PARAMS.items():
        params[name] = trial.suggest_float(name, lo, hi)
    for name, (lo, hi) in INT_PARAMS.items():
        params[name] = trial.suggest_int(name, lo, hi)
    return params


def parse_duel_json(stdout):
    lines = [line.strip() for line in str(stdout or "").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("empty duel worker output")
    return json.loads(lines[-1])


def run_duel(params, seed_suffix="", runtime=None):
    runtime = runtime or {}
    env = os.environ.copy()
    env["HEURISTIC_V7_PARAMS"] = json.dumps(params)

    seed = f"{runtime.get('seed', SEED)}|{seed_suffix}" if seed_suffix else runtime.get("seed", SEED)
    cmd = [
        "node",
        DUEL_SCRIPT,
        "--policy-a",
        SELF_POLICY,
        "--policy-b",
        runtime.get("opponent_policy", OPPONENT_POLICY),
        "--games",
        str(GAMES),
        "--seed",
        seed,
        "--max-steps",
        str(runtime.get("max_steps", MAX_STEPS)),
        "--first-turn-policy",
        "alternate",
        "--continuous-series",
        "1",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(60, runtime.get("trial_timeout_sec", TRIAL_TIMEOUT_SEC)),
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(f"duel failed: {result.stderr[:400]}")
        return parse_duel_json(result.stdout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"timeout ({runtime.get('trial_timeout_sec', TRIAL_TIMEOUT_SEC)}s)")


def objective(trial, runtime):
    params = suggest_params(trial)
    try:
        duel = run_duel(params, seed_suffix=f"t{trial.number}", runtime=runtime)
    except Exception as exc:
        print(f"  [trial {trial.number:3d}] ERROR: {exc}", flush=True)
        raise optuna.exceptions.TrialPruned()

    win_rate = float(duel.get("win_rate_a", 0.0))
    gold_delta = float(duel.get("mean_gold_delta_a", 0.0))
    go_fail_rate = float(duel.get("go_fail_rate_a", 0.0))
    score = (win_rate * 2.0) + (gold_delta / 1000.0)

    # Loss-cut: force minimum win-rate floor against V5.
    if win_rate < 0.45:
        score -= (0.45 - win_rate) * 10.0

    # Penalize unstable go behavior (high go fail).
    if go_fail_rate > 0.08:
        score -= (go_fail_rate - 0.08) * 5.0

    # Additional penalty when average gold result is negative.
    if gold_delta < 0:
        score += gold_delta / 500.0

    print(
        f"  [trial {trial.number:3d}] "
        f"win={win_rate:.4f}  gold={gold_delta:+.1f}  "
        f"go_fail={go_fail_rate:.4f}  score={score:.4f}",
        flush=True,
    )

    trial.set_user_attr("win_rate", win_rate)
    trial.set_user_attr("gold_delta", gold_delta)
    trial.set_user_attr("go_fail_rate", go_fail_rate)
    trial.set_user_attr("wins_a", duel.get("wins_a", 0))
    trial.set_user_attr("losses_a", duel.get("losses_a", 0))
    return score


def parse_args():
    parser = argparse.ArgumentParser(description="V7 Optuna tuner (V7 vs V5)")
    parser.add_argument("--trials", type=int, default=150)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--db", type=str, default="")
    parser.add_argument("--study", type=str, default="v7_tuning")
    parser.add_argument("--output", type=str, default="logs/optuna_v7_best.json")
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--seed", type=str, default=SEED)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--trial-timeout", type=int, default=TRIAL_TIMEOUT_SEC)
    parser.add_argument("--opponent-policy", type=str, default=OPPONENT_POLICY)
    parser.add_argument("--startup-trials", type=int, default=24)
    return parser.parse_args()


def main():
    args = parse_args()

    if not Path(DUEL_SCRIPT).exists():
        print(f"duel worker not found: {DUEL_SCRIPT}")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    runtime = {
        "opponent_policy": args.opponent_policy,
        "seed": args.seed or SEED,
        "max_steps": max(20, args.max_steps),
        "trial_timeout_sec": max(60, args.trial_timeout),
    }

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=max(1, args.startup_trials),
        multivariate=True,
        seed=42,
    )

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study,
        storage=args.db or None,
        sampler=sampler,
        load_if_exists=True,
    )

    print("=== V7 Optuna start ===")
    print(f"  target={SELF_POLICY} vs {runtime['opponent_policy']}")
    print(f"  trials={args.trials}  workers={max(1, args.workers)}  games/trial={GAMES}")
    print(f"  params={len(FLOAT_PARAMS) + len(INT_PARAMS)}  seed={runtime['seed']}")
    print(f"  db={args.db or 'memory'}")
    print("  objective=(win_rate*2.0) + (gold_delta/1000.0)")
    print("  penalties: win_rate<0.45, go_fail_rate>0.08, gold_delta<0\n")

    study.optimize(
        lambda trial: objective(trial, runtime),
        n_trials=max(1, args.trials),
        n_jobs=max(1, args.workers),
        timeout=args.timeout if args.timeout > 0 else None,
        show_progress_bar=False,
    )

    best = study.best_trial
    print("\n=== best result ===")
    print(f"  score:      {best.value:.4f}")
    print(f"  win_rate:   {best.user_attrs.get('win_rate', 0):.4f}")
    print(f"  gold_delta: {best.user_attrs.get('gold_delta', 0):+.1f}")
    print(f"  go_fail:    {best.user_attrs.get('go_fail_rate', 0):.4f}")
    print(f"  W/L:        {best.user_attrs.get('wins_a', 0)}/{best.user_attrs.get('losses_a', 0)}")

    payload = {
        "score": best.value,
        "win_rate": best.user_attrs.get("win_rate"),
        "gold_delta": best.user_attrs.get("gold_delta"),
        "go_fail_rate": best.user_attrs.get("go_fail_rate"),
        "params": best.params,
        "trial_number": best.number,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "self_policy": SELF_POLICY,
        "opponent": runtime["opponent_policy"],
        "seed": runtime["seed"],
        "games_per_trial": GAMES,
        "objective": {
            "base": "score=(win_rate*2.0)+(gold_delta/1000.0)",
            "win_rate_floor": 0.45,
            "go_fail_rate_cap": 0.08,
            "negative_gold_penalty_divisor": 500.0,
        },
    }
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(f"  saved: {args.output}")

    print("\n=== top 5 ===")
    top = sorted(study.trials, key=lambda t: t.value if t.value is not None else -math.inf, reverse=True)[:5]
    for t in top:
        if t.value is None:
            continue
        print(
            f"  trial {t.number:3d}  score={t.value:.4f}  "
            f"win={t.user_attrs.get('win_rate', 0):.4f}  "
            f"gold={t.user_attrs.get('gold_delta', 0):+.1f}  "
            f"go_fail={t.user_attrs.get('go_fail_rate', 0):.4f}"
        )


if __name__ == "__main__":
    main()
