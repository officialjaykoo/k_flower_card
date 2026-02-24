#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/optuna_v5.py  –  V5 Optuna 튜닝
대상: heuristic_v5 vs heuristic_v4 (1000게임/trial)
튜닝 대상: 승률에 영향 큰 핵심 31개 파라미터만 (나머지는 DEFAULT_PARAMS trial0 값 고정)

사용법:
  python scripts/optuna_v5.py --trials 200 --workers 4
"""
import argparse, json, math, os, subprocess, sys, time
from pathlib import Path

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("optuna 없음. pip install optuna"); sys.exit(1)

# ── 상수 ──────────────────────────────────────────────────────────
GAMES             = 1000
OPPONENT_POLICY   = "heuristic_v4"
SEED              = "optuna-v5"
MAX_STEPS         = 600
TRIAL_TIMEOUT_SEC = 600
DUEL_SCRIPT       = "scripts/heuristic_duel_worker.mjs"
WIN_RATE_WEIGHT   = 0.65
GOLD_DELTA_WEIGHT = 0.35
GOLD_DELTA_SCALE  = 500.0

# ── 핵심 파라미터만 (승률 영향 큰 순) ─────────────────────────────
FLOAT_PARAMS = {
    # GO 로직 (가장 중요)
    "goBaseThreshold":         (0.38, 0.62),
    "goOppOneAwayGate":        (30.0, 60.0),
    "goScoreDiffBonus":        (0.02, 0.14),
    "goDeckLowBonus":          (0.02, 0.18),
    "goUnseeHighPiPenalty":    (0.02, 0.20),
    "goBigLeadJokboThresh":    (0.10, 0.70),
    "goBigLeadNextThresh":     (0.10, 0.70),
    "goOpp0JokboThresh":       (0.10, 0.80),
    "goOpp0NextThresh":        (0.10, 0.80),
    "goOpp12JokboThresh":      (0.10, 0.80),
    "goOpp12NextThresh":       (0.10, 0.80),
    "secondMoverGoGateShrink": (0.0,  12.0),
    # 카드 선택
    "piGainMul":               (3.0,  10.0),
    "feedRiskNoMatchMul":      (2.0,  10.0),
    "feedRiskMatchMul":        (0.5,   3.0),
    "comboBlockBase":          (8.0,  36.0),
    "mongBakFiveBonus":        (15.0, 60.0),
    "doublePiMatchBonus":      (8.0,  28.0),
    "matchPiGainMul":          (3.0,  10.0),
    "matchMongBakFiveBonus":   (15.0, 60.0),
    # shaking
    "shakingScoreThreshold":   (0.35, 0.85),
    "shakingImmediateGainMul": (0.8,   2.5),
    "shakingAheadPenalty":     (0.05,  0.50),
    # 후공
    "secondMoverPiBonus":      (0.0,   5.0),
    "secondMoverBlockBonus":   (0.0,  10.0),
}
INT_PARAMS = {
    "goOppScoreGateHigh":      (5,  8),
    "goOneAwayThreshOpp3":     (28, 60),
    "goOneAwayThreshOpp2":     (24, 55),
    "goOneAwayThreshOpp1":     (20, 55),
    "goOneAwayThreshOpp4Early":(18, 50),
    "goOneAwayThreshOpp4Late": (12, 45),
}

# ── 보조 함수 ─────────────────────────────────────────────────────
def suggest_params(trial):
    p = {}
    for name, (lo, hi) in FLOAT_PARAMS.items():
        p[name] = trial.suggest_float(name, lo, hi)
    for name, (lo, hi) in INT_PARAMS.items():
        p[name] = trial.suggest_int(name, lo, hi)
    return p

def parse_duel_json(stdout):
    lines = [l.strip() for l in str(stdout or "").splitlines() if l.strip()]
    if not lines: raise RuntimeError("duel worker stdout empty")
    return json.loads(lines[-1])

def run_duel(params, seed_suffix="", rt=None):
    rt = rt or {}
    env = os.environ.copy()
    env["HEURISTIC_V5_PARAMS"] = json.dumps(params)
    seed = f"{rt.get('seed', SEED)}|{seed_suffix}" if seed_suffix else rt.get('seed', SEED)
    cmd = [
        "node", DUEL_SCRIPT,
        "--policy-a", "heuristic_v5",
        "--policy-b", rt.get("opponent_policy", OPPONENT_POLICY),
        "--games", str(GAMES),
        "--seed", seed,
        "--max-steps", str(rt.get("max_steps", MAX_STEPS)),
        "--first-turn-policy", "alternate",
        "--continuous-series", "1",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=max(60, rt.get("trial_timeout_sec", TRIAL_TIMEOUT_SEC)), env=env)
        if r.returncode != 0:
            raise RuntimeError(f"duel error: {r.stderr[:400]}")
        return parse_duel_json(r.stdout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"timeout ({rt.get('trial_timeout_sec', TRIAL_TIMEOUT_SEC)}s)")

def objective(trial, rt):
    params = suggest_params(trial)
    try:
        result = run_duel(params, seed_suffix=f"t{trial.number}", rt=rt)
    except Exception as e:
        print(f"  [trial {trial.number:3d}] ERROR: {e}", flush=True)
        raise optuna.exceptions.TrialPruned()

    win_rate   = float(result.get("win_rate_a", 0))
    gold_delta = float(result.get("mean_gold_delta_a", 0))
    gold_norm  = max(-1.0, min(1.0, gold_delta / GOLD_DELTA_SCALE))
    score      = win_rate * WIN_RATE_WEIGHT + gold_norm * GOLD_DELTA_WEIGHT

    print(f"  [trial {trial.number:3d}] win={win_rate:.4f}  gold={gold_delta:+.1f}  score={score:.4f}", flush=True)
    trial.set_user_attr("win_rate",   win_rate)
    trial.set_user_attr("gold_delta", gold_delta)
    trial.set_user_attr("wins_a",     result.get("wins_a", 0))
    trial.set_user_attr("losses_a",   result.get("losses_a", 0))
    return score

# ── 메인 ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="V5 Optuna 튜닝")
    p.add_argument("--trials",          type=int, default=200)
    p.add_argument("--workers",         type=int, default=1)
    p.add_argument("--db",              type=str, default="")
    p.add_argument("--study",           type=str, default="v5_tuning")
    p.add_argument("--output",          type=str, default="logs/optuna_v5_best.json")
    p.add_argument("--timeout",         type=int, default=0)
    p.add_argument("--seed",            type=str, default=SEED)
    p.add_argument("--max-steps",       type=int, default=MAX_STEPS)
    p.add_argument("--trial-timeout",   type=int, default=TRIAL_TIMEOUT_SEC)
    p.add_argument("--opponent-policy", type=str, default=OPPONENT_POLICY)
    p.add_argument("--startup-trials",  type=int, default=20)
    return p.parse_args()

def main():
    args = parse_args()
    if not Path(DUEL_SCRIPT).exists():
        print(f"duel worker not found: {DUEL_SCRIPT}"); sys.exit(1)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    rt = {
        "opponent_policy":   args.opponent_policy,
        "seed":              args.seed or SEED,
        "max_steps":         max(20, args.max_steps),
        "trial_timeout_sec": max(60, args.trial_timeout),
    }
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=max(1, args.startup_trials), multivariate=True, seed=42,
    )
    study = optuna.create_study(
        direction="maximize", study_name=args.study,
        storage=args.db or None, sampler=sampler, load_if_exists=True,
    )

    print(f"=== V5 Optuna 튜닝 시작 ===")
    print(f"  trials={args.trials}  opponent={rt['opponent_policy']}  games/trial={GAMES}")
    print(f"  params={len(FLOAT_PARAMS)+len(INT_PARAMS)}개  seed={rt['seed']}")
    print(f"  DB: {args.db or '메모리 (비영속)'}")
    print(f"  목적: win_rate×{WIN_RATE_WEIGHT} + gold_delta×{GOLD_DELTA_WEIGHT}\n")

    study.optimize(
        lambda trial: objective(trial, rt),
        n_trials=args.trials,
        n_jobs=args.workers,
        timeout=args.timeout if args.timeout > 0 else None,
        show_progress_bar=False,
    )

    best = study.best_trial
    print(f"\n=== 최적 결과 ===")
    print(f"  Score:      {best.value:.4f}")
    print(f"  win_rate:   {best.user_attrs.get('win_rate','?'):.4f}")
    print(f"  gold_delta: {best.user_attrs.get('gold_delta','?'):+.1f}")
    print(f"  W/L:        {best.user_attrs.get('wins_a','?')}/{best.user_attrs.get('losses_a','?')}")

    output = {
        "score":           best.value,
        "win_rate":        best.user_attrs.get("win_rate"),
        "gold_delta":      best.user_attrs.get("gold_delta"),
        "params":          best.params,
        "trial_number":    best.number,
        "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
        "opponent":        rt["opponent_policy"],
        "seed":            rt["seed"],
        "games_per_trial": GAMES,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  최적 파라미터 저장: {args.output}")

    print("\n=== Top 5 trials ===")
    for t in sorted(study.trials, key=lambda t: t.value or -math.inf, reverse=True)[:5]:
        if t.value is None: continue
        print(f"  trial {t.number:3d}  score={t.value:.4f}  win={t.user_attrs.get('win_rate','?'):.4f}  gold={t.user_attrs.get('gold_delta','?'):+.1f}")

if __name__ == "__main__":
    main()