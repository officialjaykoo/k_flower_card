#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Pipeline Stage: Optuna Tuning Wrapper (GPT)
# Quick Read Map:
# 1) Define search space/constants
# 2) suggest_params + run_duel helpers
# 3) objective() scoring
# 4) main(): study run + artifact export

"""
scripts/optuna_gpt.py - GPT Optuna tuner
target: H-GPT vs H-CL (1000 games per trial)

Integration:
  - Can consume optimizer_by_gpt output plan JSON.
  - Converts defense/attack recommendations into Optuna seed trials.
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
SELF_POLICY = "H-GPT"
OPPONENT_POLICY = "H-CL"
SEED = "optuna-gpt"
MAX_STEPS = 600
TRIAL_TIMEOUT_SEC = 600
DUEL_SCRIPT = "scripts/model_duel_worker_by_GPT.mjs"
WIN_RATE_WEIGHT = 0.35
GOLD_DELTA_WEIGHT = 0.65
GOLD_DELTA_SCALE = 500.0
GOLD_DELTA_MIN_REQUIRED = 1.0
GOLD_DELTA_FAIL_SCORE = -2.0
GO_GAMES_TARGET = 210
GO_COUNT_TARGET = 320
GO_GAMES_MIN_REQUIRED = 150
GO_GAMES_WEIGHT = 0.14
GO_COUNT_WEIGHT = 0.12
GO_FAIL_SOFT_CAP = 0.14
GO_FAIL_PENALTY_WEIGHT = 0.22

FLOAT_PARAMS = {
    # profile + card utility
    "trailingAttackBoost": (0.0, 0.35),
    "trailingTempoBoost": (0.0, 0.4),
    "leadingDefenseBoost": (0.0, 0.35),
    "leadingRiskBoost": (0.0, 0.4),
    "highPressureDefenseBoost": (0.0, 0.4),
    "highPressureRiskBoost": (0.0, 0.4),
    "desperateAttackBoost": (0.0, 0.3),
    "desperateTempoBoost": (0.0, 0.2),
    "desperateRiskDown": (0.0, 0.2),
    "noMatchBase": (-12.0, -3.0),
    "matchOneBase": (2.0, 10.0),
    "matchTwoBase": (6.0, 18.0),
    "matchThreeBase": (9.0, 22.0),
    "captureGainMul": (0.7, 1.8),
    "junkPiMul": (1.8, 5.2),
    "comboOpportunityMul": (1.2, 6.0),
    "blockBase": (2.0, 10.0),
    "blockUrgencyMul": (0.4, 2.4),
    "blockThreatMul": (0.4, 3.0),
    "feedRiskNoMatchMul": (1.0, 6.0),
    "feedRiskMatchMul": (0.5, 3.0),
    "dangerNoMatchMul": (0.6, 4.0),
    "dangerMatchMul": (0.2, 1.8),
    "releaseRiskMul": (0.8, 5.0),
    "releaseRiskOppScoreExpMul": (0.0, 0.8),
    "pukRiskMul": (0.4, 2.6),
    "selfPukRiskMul": (0.4, 2.4),
    "selfPukRiskExpMul": (0.0, 1.0),
    "selfPukRiskPatternBonus": (0.0, 0.8),
    "phaseEarlyPiMul": (0.8, 1.8),
    "phaseLatePiMul": (0.6, 1.4),
    "phaseEndPiMul": (0.4, 1.2),
    "phaseLateRiskMul": (0.8, 2.0),
    "phaseEndRiskMul": (0.8, 2.6),
    "phaseLateBlockMul": (0.8, 2.2),
    "phaseEndBlockMul": (1.0, 3.2),
    # go model
    "goUpsideScoreMul": (0.03, 0.15),
    "goUpsidePiMul": (0.01, 0.08),
    "goUpsideSelfJokboMul": (0.1, 1.0),
    "goUpsideOneAwayMul": (0.02, 0.4),
    "goRiskPressureMul": (0.1, 0.9),
    "goRiskOneAwayMul": (0.02, 0.35),
    "goRiskOppJokboMul": (0.08, 0.7),
    "goRiskOppOneAwayMul": (0.02, 0.3),
    "goHardRiskThreatMul": (0.1, 0.8),
    "goHardRiskOneAwayMul": (0.1, 0.8),
    "goHardRiskOppStopPenalty": (0.05, 0.8),
    "goHardRiskThreshold": (0.25, 1.2),
    "goHardJokboOneAwaySwingCut": (0.35, 0.95),
    "goBaseThreshold": (-0.1, 0.3),
    "goThresholdLeadUp": (0.0, 0.3),
    "goThresholdTrailDown": (0.0, 0.3),
    "goThresholdPressureUp": (0.0, 0.3),
    "goLiteThreatPenaltyMul": (0.0, 0.25),
    "goLiteOneAwayPenaltyMul": (0.0, 0.25),
    "goLiteLatePenalty": (0.0, 0.2),
    "goLiteOppCanStopPenalty": (0.0, 0.4),
    "goLiteSafeAttackBonus": (0.0, 0.3),
    "goLookaheadThresholdMul": (0.0, 0.3),
    "goSafeGoBonus": (0.0, 0.35),
    "goUtilityUpsideWeight": (0.6, 1.8),
    "goUtilityRiskWeight": (0.6, 1.8),
    "goUtilityStopWeight": (0.6, 1.8),
    "goUtilityThresholdWeight": (0.6, 1.8),
    "rankReleaseRiskPenaltyMul": (8.0, 40.0),
    # shaking
    "shakeImmediateMul": (0.5, 1.8),
    "shakeComboMul": (0.5, 1.8),
    "shakeRiskMul": (0.0, 1.5),
    "shakeThreshold": (0.3, 1.2),
    "shakeLeadThresholdUp": (0.0, 0.5),
    "shakePressureThresholdUp": (0.0, 0.5),
}

INT_PARAMS = {
    "phaseEarlyDeck": (11, 18),
    "phaseLateDeck": (5, 10),
    "phaseEndDeck": (3, 6),
    "goMinPi": (3, 8),
    "goMinPiDesperate": (3, 7),
    "goMinPiSecondTrailingDelta": (0, 3),
    "goHardThreatDeckCut": (4, 10),
    "goHardOppFiveCut": (5, 10),
    "goHardOppScoreCut": (7, 12),
    "goHardLateOneAwayCut": (40, 85),
    "goHardLateOneAwayDeckCut": (5, 12),
    "goHardGoCountCap": (2, 5),
    "goHardJokboOneAwayCut": (40, 85),
    "goHardJokboOneAwayDeckCut": (6, 14),
    "goHardJokboOneAwayCountCut": (1, 2),
}


def sanitize_file_part(text):
    raw = str(text or "").strip().lower()
    if not raw:
        return "run"
    out = []
    for ch in raw:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "run"


def as_float(value, fallback=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def suggest_params(trial):
    params = {}
    for name, (lo, hi) in FLOAT_PARAMS.items():
        params[name] = trial.suggest_float(name, lo, hi)
    for name, (lo, hi) in INT_PARAMS.items():
        params[name] = trial.suggest_int(name, lo, hi)
    return params


def clamp_suggested_params(raw):
    out = {}
    for name, value in dict(raw or {}).items():
        if name in FLOAT_PARAMS:
            lo, hi = FLOAT_PARAMS[name]
            v = float(value)
            out[name] = max(lo, min(hi, v))
        elif name in INT_PARAMS:
            lo, hi = INT_PARAMS[name]
            v = int(round(float(value)))
            out[name] = max(lo, min(hi, v))
    return out


def load_optimizer_seed_trials(path, top_k):
    with open(path, "r", encoding="utf-8") as file:
        plan = json.load(file)
    if not isinstance(plan, dict):
        raise RuntimeError(f"optimizer plan is not an object: {path}")
    if "plans" not in plan or not isinstance(plan["plans"], dict):
        raise RuntimeError(f"optimizer plan missing plans object: {path}")

    allowed = set(FLOAT_PARAMS.keys()) | set(INT_PARAMS.keys())
    candidates = plan.get("all_candidates")
    baseline = {}
    if isinstance(candidates, list):
        for row in candidates:
            if not isinstance(row, dict):
                continue
            name = row.get("param")
            if name not in allowed:
                continue
            if "current" not in row:
                continue
            try:
                baseline[name] = float(row["current"])
            except (TypeError, ValueError):
                continue

    def make_trial(direction):
        d = plan["plans"].get(direction)
        if not isinstance(d, dict):
            raise RuntimeError(f"optimizer plan missing direction section: {direction}")
        recs = d.get("recommendations")
        if not isinstance(recs, list):
            raise RuntimeError(f"optimizer plan recommendations invalid: {direction}")
        out = dict(baseline)
        applied = 0
        for rec in recs:
            if not isinstance(rec, dict):
                continue
            name = rec.get("param")
            if name not in allowed:
                continue
            if "suggested" not in rec:
                continue
            try:
                out[name] = float(rec["suggested"])
            except (TypeError, ValueError):
                continue
            applied += 1
            if applied >= top_k:
                break
        if applied <= 0 and not out:
            return None
        return clamp_suggested_params(out)

    seeds = []
    base = clamp_suggested_params(baseline)
    if base:
        seeds.append(base)
    for direction in ("defense", "attack"):
        trial = make_trial(direction)
        if trial:
            seeds.append(trial)

    merged = []
    seen = set()
    for row in seeds:
        key = json.dumps(row, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        merged.append(row)
    return merged


def parse_duel_json(stdout, result_out=""):
    lines = [line.strip() for line in str(stdout or "").splitlines() if line.strip()]
    if lines:
        try:
            return json.loads(lines[-1])
        except json.JSONDecodeError:
            pass

    result_path = Path(str(result_out or "").strip())
    if result_path.exists():
        raw = result_path.read_text(encoding="utf-8").strip()
        if raw:
            file_lines = [line.strip() for line in raw.splitlines() if line.strip()]
            if file_lines:
                try:
                    return json.loads(file_lines[-1])
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"invalid duel result JSON file: {result_path} ({exc})") from exc

    if lines:
        tail = lines[-1][:200]
        raise RuntimeError(f"duel worker stdout is not JSON (tail={tail!r})")
    raise RuntimeError("empty duel worker output and missing result file JSON")


def run_duel(params, seed_suffix="", runtime=None):
    runtime = runtime or {}
    env = os.environ.copy()
    env["HEURISTIC_GPT_PARAMS"] = json.dumps(params)
    seed = f"{runtime.get('seed', SEED)}|{seed_suffix}" if seed_suffix else runtime.get("seed", SEED)
    result_dir = runtime.get("result_dir", os.path.join("logs", "optuna"))
    os.makedirs(result_dir, exist_ok=True)
    seed_tag = sanitize_file_part(runtime.get("seed", SEED))
    trial_tag = sanitize_file_part(seed_suffix or "base")
    result_out = os.path.join(result_dir, f"{seed_tag}_{trial_tag}_result.json")
    cmd = [
        "node",
        DUEL_SCRIPT,
        "--human",
        SELF_POLICY,
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
        "--stdout-format",
        "json",
        "--result-out",
        result_out,
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
        return parse_duel_json(result.stdout, result_out=result_out)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"timeout ({runtime.get('trial_timeout_sec', TRIAL_TIMEOUT_SEC)}s)")


def objective(trial, runtime):
    params = suggest_params(trial)
    try:
        duel = run_duel(params, seed_suffix=f"t{trial.number}", runtime=runtime)
    except Exception as exc:
        print(f"  [trial {trial.number:3d}] ERROR: {exc}", flush=True)
        raise optuna.exceptions.TrialPruned()

    required_keys = [
        "win_rate_a",
        "mean_gold_delta_a",
        "go_count_a",
        "go_games_a",
        "go_fail_rate_a",
    ]
    missing = [k for k in required_keys if k not in duel]
    if missing:
        print(f"  [trial {trial.number:3d}] ERROR: duel output missing keys: {missing}", flush=True)
        raise optuna.exceptions.TrialPruned()

    win_rate = float(duel.get("win_rate_a", 0))
    gold_delta = float(duel.get("mean_gold_delta_a", 0))
    go_count = float(duel.get("go_count_a", 0))
    go_games = float(duel.get("go_games_a", 0))
    go_fail_rate = float(duel.get("go_fail_rate_a", 0))
    gold_norm = max(-1.0, min(1.0, gold_delta / GOLD_DELTA_SCALE))
    if gold_delta <= GOLD_DELTA_MIN_REQUIRED:
        # Hard requirement: non-positive (or near-zero) gold is unacceptable.
        score = GOLD_DELTA_FAIL_SCORE + win_rate * 0.05 + gold_norm * 0.05
    else:
        score = win_rate * WIN_RATE_WEIGHT + gold_norm * GOLD_DELTA_WEIGHT

    go_games_progress = max(0.0, min(1.0, go_games / max(1.0, GO_GAMES_TARGET)))
    go_count_progress = max(0.0, min(1.0, go_count / max(1.0, GO_COUNT_TARGET)))
    score += go_games_progress * GO_GAMES_WEIGHT
    score += go_count_progress * GO_COUNT_WEIGHT

    if go_games < GO_GAMES_MIN_REQUIRED:
        shortage = (GO_GAMES_MIN_REQUIRED - go_games) / max(1.0, GO_GAMES_MIN_REQUIRED)
        score -= shortage * GO_GAMES_WEIGHT
    if go_fail_rate > GO_FAIL_SOFT_CAP:
        score -= (go_fail_rate - GO_FAIL_SOFT_CAP) * GO_FAIL_PENALTY_WEIGHT * 4.0

    print(
        f"  [trial {trial.number:3d}] "
        f"win={win_rate:.4f}  gold={gold_delta:+.1f}  "
        f"go_games={go_games:.0f}  go_count={go_count:.0f}  go_fail={go_fail_rate:.4f}  score={score:.4f}",
        flush=True,
    )
    trial.set_user_attr("win_rate", win_rate)
    trial.set_user_attr("gold_delta", gold_delta)
    trial.set_user_attr("go_count", go_count)
    trial.set_user_attr("go_games", go_games)
    trial.set_user_attr("go_fail_rate", go_fail_rate)
    trial.set_user_attr("wins_a", duel.get("wins_a", 0))
    trial.set_user_attr("losses_a", duel.get("losses_a", 0))
    return score


def parse_args():
    parser = argparse.ArgumentParser(description="GPT Optuna tuner")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--db", type=str, default="")
    parser.add_argument("--study", type=str, default="gpt_tuning")
    parser.add_argument("--output", type=str, default="logs/optuna/optuna_gpt_best.json")
    parser.add_argument("--result-dir", type=str, default="")
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--seed", type=str, default=SEED)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--trial-timeout", type=int, default=TRIAL_TIMEOUT_SEC)
    parser.add_argument("--opponent-policy", type=str, default=OPPONENT_POLICY)
    parser.add_argument("--startup-trials", type=int, default=24)
    parser.add_argument("--optimizer-plan", type=str, default="")
    parser.add_argument("--optimizer-top-k", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    if not Path(DUEL_SCRIPT).exists():
        print(f"duel worker not found: {DUEL_SCRIPT}")
        sys.exit(1)
    if args.optimizer_top_k <= 0:
        print("--optimizer-top-k must be >= 1")
        sys.exit(1)
    if args.optimizer_plan and not Path(args.optimizer_plan).exists():
        print(f"optimizer plan not found: {args.optimizer_plan}")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    runtime = {
        "opponent_policy": args.opponent_policy,
        "seed": args.seed or SEED,
        "max_steps": max(20, args.max_steps),
        "trial_timeout_sec": max(60, args.trial_timeout),
        "result_dir": args.result_dir
        or os.path.join("logs", "optuna", sanitize_file_part(args.study or "gpt_tuning")),
    }
    os.makedirs(runtime["result_dir"], exist_ok=True)

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

    print("=== GPT Optuna start ===")
    print(f"  trials={args.trials}  opponent={runtime['opponent_policy']}  games/trial={GAMES}")
    print(f"  params={len(FLOAT_PARAMS) + len(INT_PARAMS)}  seed={runtime['seed']}")
    print(f"  duel_worker={DUEL_SCRIPT}")
    print(f"  db={args.db or 'memory'}")
    print(f"  trial_result_dir={runtime['result_dir']}")
    print(
        "  objective="
        f"base(win_rate*{WIN_RATE_WEIGHT} + gold_delta_norm*{GOLD_DELTA_WEIGHT}) "
        f"+ go_games*{GO_GAMES_WEIGHT} + go_count*{GO_COUNT_WEIGHT} "
        f"- over_fail_penalty (soft_cap={GO_FAIL_SOFT_CAP})"
    )
    print(
        "  go_target="
        f"games:{GO_GAMES_TARGET} (min:{GO_GAMES_MIN_REQUIRED}), "
        f"count:{GO_COUNT_TARGET}\n"
    )

    if args.optimizer_plan:
        seed_trials = load_optimizer_seed_trials(args.optimizer_plan, args.optimizer_top_k)
        for seed in seed_trials:
            study.enqueue_trial(seed)
        print(
            f"  optimizer_plan={args.optimizer_plan} "
            f"(top_k={args.optimizer_top_k}, enqueued={len(seed_trials)})"
        )
        print("")

    study.optimize(
        lambda trial: objective(trial, runtime),
        n_trials=args.trials,
        n_jobs=args.workers,
        timeout=args.timeout if args.timeout > 0 else None,
        show_progress_bar=False,
    )

    complete = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if not complete:
        raise RuntimeError(
            "no completed trials. check duel worker output/required duel keys and rerun."
        )

    best = study.best_trial
    best_win_rate = as_float(best.user_attrs.get("win_rate"), 0.0)
    best_gold_delta = as_float(best.user_attrs.get("gold_delta"), 0.0)
    best_go_games = as_float(best.user_attrs.get("go_games"), 0.0)
    best_go_count = as_float(best.user_attrs.get("go_count"), 0.0)
    best_go_fail = as_float(best.user_attrs.get("go_fail_rate"), 0.0)
    print("\n=== best result ===")
    print(f"  score:      {best.value:.4f}")
    print(f"  win_rate:   {best_win_rate:.4f}")
    print(f"  gold_delta: {best_gold_delta:+.1f}")
    print(f"  go_games:   {best_go_games:.0f}")
    print(f"  go_count:   {best_go_count:.0f}")
    print(f"  go_fail:    {best_go_fail:.4f}")
    print(f"  W/L:        {best.user_attrs.get('wins_a', 0)}/{best.user_attrs.get('losses_a', 0)}")

    payload = {
        "score": best.value,
        "win_rate": best_win_rate,
        "gold_delta": best_gold_delta,
        "go_games": best_go_games,
        "go_count": best_go_count,
        "go_fail_rate": best_go_fail,
        "params": best.params,
        "trial_number": best.number,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "opponent": runtime["opponent_policy"],
        "seed": runtime["seed"],
        "games_per_trial": GAMES,
        "objective": {
            "win_rate_weight": WIN_RATE_WEIGHT,
            "gold_delta_weight": GOLD_DELTA_WEIGHT,
            "gold_delta_scale": GOLD_DELTA_SCALE,
            "go_games_target": GO_GAMES_TARGET,
            "go_count_target": GO_COUNT_TARGET,
            "go_games_weight": GO_GAMES_WEIGHT,
            "go_count_weight": GO_COUNT_WEIGHT,
            "go_fail_soft_cap": GO_FAIL_SOFT_CAP,
            "go_fail_penalty_weight": GO_FAIL_PENALTY_WEIGHT,
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
        top_win_rate = as_float(t.user_attrs.get("win_rate"), 0.0)
        top_gold_delta = as_float(t.user_attrs.get("gold_delta"), 0.0)
        top_go_games = as_float(t.user_attrs.get("go_games"), 0.0)
        top_go_count = as_float(t.user_attrs.get("go_count"), 0.0)
        top_go_fail = as_float(t.user_attrs.get("go_fail_rate"), 0.0)
        print(
            f"  trial {t.number:3d}  score={t.value:.4f}  "
            f"win={top_win_rate:.4f}  "
            f"gold={top_gold_delta:+.1f}  "
            f"go_games={top_go_games:.0f}  "
            f"go_count={top_go_count:.0f}  "
            f"go_fail={top_go_fail:.4f}"
        )


if __name__ == "__main__":
    main()
