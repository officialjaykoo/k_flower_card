#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Pipeline Stage: Optuna Tuning Wrapper (V6)
# Quick Read Map:
# 1) Define search space/constants
# 2) suggest_params + run_duel helpers
# 3) objective() scoring
# 4) main(): study run + artifact export

"""
scripts/optuna_v6.py - V6 Optuna tuner
target: H-V6 vs H-V5 (1000 games per trial)
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

GAMES = 1000
OPPONENT_POLICY = "H-V5"
SEED = "optuna-v6"
MAX_STEPS = 600
TRIAL_TIMEOUT_SEC = 600
DUEL_SCRIPT = "scripts/model_duel_worker.mjs"
WIN_RATE_WEIGHT = 0.65
GOLD_DELTA_WEIGHT = 0.35
GOLD_DELTA_SCALE = 500.0

FLOAT_PARAMS = {
    # profile and phase
    "trailingAttackBoost": (0.0, 0.35),
    "trailingTempoBoost": (0.0, 0.4),
    "leadingDefenseBoost": (0.0, 0.35),
    "leadingRiskBoost": (0.0, 0.4),
    "highPressureDefenseBoost": (0.0, 0.4),
    "highPressureRiskBoost": (0.0, 0.4),
    # card utility
    "noMatchBase": (-12.0, -3.0),
    "matchOneBase": (4.0, 10.0),
    "matchTwoBase": (8.0, 16.0),
    "matchThreeBase": (10.0, 20.0),
    "captureGainMul": (0.7, 1.8),
    "junkPiMul": (2.0, 8.0),
    "selfPiWindowMul": (0.8, 2.4),
    "oppPiWindowMul": (0.6, 2.2),
    "comboOpportunityMul": (1.5, 8.0),
    "blockBase": (2.0, 10.0),
    "blockUrgencyMul": (0.6, 2.6),
    "blockThreatMul": (0.8, 5.0),
    "feedRiskNoMatchMul": (1.5, 8.0),
    "feedRiskMatchMul": (0.2, 2.0),
    "dangerNoMatchMul": (0.6, 4.5),
    "dangerMatchMul": (0.1, 1.5),
    "releaseRiskMul": (2.0, 14.0),
    "pukRiskMul": (1.0, 6.0),
    # go model
    "goUpsideScoreMul": (0.03, 0.15),
    "goUpsidePiMul": (0.01, 0.08),
    "goUpsideSelfJokboMul": (0.12, 0.8),
    "goRiskPressureMul": (0.1, 0.9),
    "goRiskOneAwayMul": (0.004, 0.03),
    "goRiskOppJokboMul": (0.05, 0.6),
    "goRiskOppOneAwayMul": (0.02, 0.2),
    "goBaseThreshold": (-0.1, 0.2),
    "goThresholdLeadUp": (0.0, 0.2),
    "goThresholdTrailDown": (0.0, 0.2),
    "goThresholdPressureUp": (0.0, 0.2),
    # shaking
    "shakeImmediateMul": (0.7, 2.4),
    "shakeComboMul": (0.6, 2.0),
    "shakeRiskMul": (0.0, 1.3),
    "shakeThreshold": (0.3, 1.1),
    # rollout
    "rolloutCardWeight": (0.4, 1.4),
    "rolloutGoWeight": (0.6, 1.8),
}

INT_PARAMS = {
    "phaseEarlyDeck": (12, 18),
    "phaseLateDeck": (6, 10),
    "phaseEndDeck": (3, 5),
    "goMinPi": (6, 9),
    "goMinPiDesperate": (5, 8),
    "goMinPiSecondTrailingDelta": (0, 2),
    "goHardThreatDeckCut": (5, 9),
    "goHardOppFiveCut": (6, 9),
    "goHardOppScoreCut": (7, 10),
    "rolloutTopK": (2, 4),
    "rolloutMaxSteps": (18, 36),
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
    env["HEURISTIC_V6_PARAMS"] = json.dumps(params)
    seed = f"{runtime.get('seed', SEED)}|{seed_suffix}" if seed_suffix else runtime.get("seed", SEED)
    cmd = [
        "node",
        DUEL_SCRIPT,
        "--human",
        "H-V6",
        "--ai",
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

    win_rate = float(duel.get("win_rate_a", 0))
    gold_delta = float(duel.get("mean_gold_delta_a", 0))
    gold_norm = max(-1.0, min(1.0, gold_delta / GOLD_DELTA_SCALE))
    score = win_rate * WIN_RATE_WEIGHT + gold_norm * GOLD_DELTA_WEIGHT

    print(f"  [trial {trial.number:3d}] win={win_rate:.4f}  gold={gold_delta:+.1f}  score={score:.4f}", flush=True)
    trial.set_user_attr("win_rate", win_rate)
    trial.set_user_attr("gold_delta", gold_delta)
    trial.set_user_attr("wins_a", duel.get("wins_a", 0))
    trial.set_user_attr("losses_a", duel.get("losses_a", 0))
    return score


def parse_args():
    parser = argparse.ArgumentParser(description="V6 Optuna tuner")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--db", type=str, default="")
    parser.add_argument("--study", type=str, default="v6_tuning")
    parser.add_argument("--output", type=str, default="logs/optuna_v6_best.json")
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

    print("=== V6 Optuna start ===")
    print(f"  trials={args.trials}  opponent={runtime['opponent_policy']}  games/trial={GAMES}")
    print(f"  params={len(FLOAT_PARAMS) + len(INT_PARAMS)}  seed={runtime['seed']}")
    print(f"  db={args.db or 'memory'}")
    print(f"  objective=win_rate*{WIN_RATE_WEIGHT} + gold_delta*{GOLD_DELTA_WEIGHT}\n")

    study.optimize(
        lambda trial: objective(trial, runtime),
        n_trials=args.trials,
        n_jobs=args.workers,
        timeout=args.timeout if args.timeout > 0 else None,
        show_progress_bar=False,
    )

    best = study.best_trial
    print("\n=== best result ===")
    print(f"  score:      {best.value:.4f}")
    print(f"  win_rate:   {best.user_attrs.get('win_rate', 0):.4f}")
    print(f"  gold_delta: {best.user_attrs.get('gold_delta', 0):+.1f}")
    print(f"  W/L:        {best.user_attrs.get('wins_a', 0)}/{best.user_attrs.get('losses_a', 0)}")

    payload = {
        "score": best.value,
        "win_rate": best.user_attrs.get("win_rate"),
        "gold_delta": best.user_attrs.get("gold_delta"),
        "params": best.params,
        "trial_number": best.number,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "opponent": runtime["opponent_policy"],
        "seed": runtime["seed"],
        "games_per_trial": GAMES,
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
            f"win={t.user_attrs.get('win_rate', 0):.4f}  gold={t.user_attrs.get('gold_delta', 0):+.1f}"
        )


if __name__ == "__main__":
    main()
