#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/optuna_v5.py
──────────────────────────────────────────────────────────────────
V5 휴리스틱 파라미터를 Optuna로 튜닝하는 스크립트.

목표 메트릭: win_rate_a (승률) + mean_gold_delta_a (골드델타) 가중합
대상:  heuristic_v5  vs  heuristic_v4 (기준선)
게임 수: 1000 (AGENTS.md 규칙 준수)

사용법:
  python scripts/optuna_v5.py --trials 9 --workers 4

의존:
  pip install optuna
  node  (heuristic_duel_worker.mjs 실행용)
──────────────────────────────────────────────────────────────────
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
    print("optuna 없음. pip install optuna 후 재실행")
    sys.exit(1)

# ── 상수 ──────────────────────────────────────────────────────────
GAMES = 1000
OPPONENT_POLICY = "heuristic_v4"
SEED = "optuna-v5"
MAX_STEPS = 600
TRIAL_TIMEOUT_SEC = 600
DUEL_SCRIPT = "scripts/heuristic_duel_worker.mjs"

# 목적함수 가중치
WIN_RATE_WEIGHT = 0.65
GOLD_DELTA_WEIGHT = 0.35
GOLD_DELTA_SCALE = 500.0   # 골드델타 정규화 기준값

# ── 파라미터 탐색 공간 정의 ────────────────────────────────────────
PARAM_SPACE = {
    # 카드 가치
    "kwangWeight":           (4.0, 14.0),
    "fiveWeight":            (2.0, 9.0),
    "ribbonWeight":          (2.0, 8.0),
    "piWeight":              (0.5, 2.0),
    "doublePiBonus":         (3.0, 12.0),
    # rankHandCards
    "matchOneBase":          (3.0, 10.0),
    "matchTwoBase":          (7.0, 15.0),
    "matchThreeBase":        (10.0, 20.0),
    "noMatchPenalty":        (1.0, 5.0),
    "highValueMatchBonus":   (2.0, 9.0),
    "feedRiskMul":           (2.0, 9.0),
    "feedRiskMatchMul":      (0.5, 2.5),
    "pukRiskHighMul":        (3.0, 8.0),
    "pukRiskNormalMul":      (2.0, 6.0),
    "blockingBonus":         (12.0, 30.0),
    "jokboBlockBonus":       (3.0, 12.0),
    "firstTurnPiPlanBonus":  (2.0, 9.0),
    "comboBaseBonus":        (1.5, 7.0),
    # shouldGo
    "goBaseThreshold":       (0.40, 0.70),
    "goOppOneAwayGate":      (25.0, 50.0),
    "goScoreDiffBonus":      (0.02, 0.12),
    "goDeckLowBonus":        (0.02, 0.12),
    "goUnseeHighPiPenalty":  (0.03, 0.15),
    # shouldBomb
    "bombImpactMinGain":     (0.0, 3.0),
    # decideShaking
    "shakingScoreThreshold": (0.40, 0.80),
    "shakingImmediateGainMul": (0.8, 2.0),
    "shakingComboGainMul":   (0.7, 1.8),
    "shakingTempoBonusMul":  (0.2, 0.9),
    "shakingAheadPenalty":   (0.05, 0.40),
    # chooseMatch
    "matchPiGainMul":        (2.0, 7.0),
    "matchKwangBonus":       (5.0, 16.0),
    "matchRibbonBonus":      (4.0, 12.0),
    "matchFiveBonus":        (2.0, 10.0),
    "matchDoublePiBonus":    (8.0, 22.0),
    "matchMongBakFiveBonus": (25.0, 55.0),
}

# ── 보조 함수 ─────────────────────────────────────────────────────

def suggest_params(trial):
    """Optuna trial에서 파라미터 샘플링"""
    p = {}
    for name, (lo, hi) in PARAM_SPACE.items():
        p[name] = trial.suggest_float(name, lo, hi)
    # 정수 파라미터
    p["goOppScoreGateLow"] = trial.suggest_int("goOppScoreGateLow", 2, 4)
    p["goOppScoreGateHigh"] = trial.suggest_int("goOppScoreGateHigh", 4, 7)
    return p


def parse_duel_json(stdout_text: str) -> dict:
    lines = [line.strip() for line in str(stdout_text or "").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("duel worker stdout is empty")
    # Worker is expected to print one JSON line; parse last non-empty line for safety.
    return json.loads(lines[-1])


def run_duel(
    params: dict,
    seed_suffix: str = "",
    opponent_policy: str = OPPONENT_POLICY,
    base_seed: str = SEED,
    max_steps: int = MAX_STEPS,
    trial_timeout_sec: int = TRIAL_TIMEOUT_SEC,
) -> dict:
    """
    heuristic_duel_worker.mjs를 실행하고 결과 JSON 반환.
    V5 파라미터는 환경변수로 주입 → worker에서 읽어서 HEURISTIC_V5_DEPS에 넘김.
    """
    env = os.environ.copy()
    env["HEURISTIC_V5_PARAMS"] = json.dumps(params)

    seed = f"{base_seed}|{seed_suffix}" if seed_suffix else base_seed
    cmd = [
        "node", DUEL_SCRIPT,
        "--policy-a", "heuristic_v5",
        "--policy-b", opponent_policy,
        "--games", str(GAMES),
        "--seed", seed,
        "--max-steps", str(max_steps),
        "--first-turn-policy", "alternate",
        "--continuous-series", "1",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(60, int(trial_timeout_sec)),
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "duel worker error:\n"
                f"cmd={' '.join(cmd)}\n"
                f"stderr={result.stderr[:600]}\n"
                f"stdout={result.stdout[:300]}"
            )
        return parse_duel_json(result.stdout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"duel worker timeout ({trial_timeout_sec}s)")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"JSON 파싱 실패: {e}\nstdout={result.stdout[:400]}")


def objective(trial, runtime) -> float:
    """Optuna 목적함수: 승률 + 골드델타 가중합 (최대화)"""
    params = suggest_params(trial)

    try:
        result = run_duel(
            params,
            seed_suffix=f"trial{trial.number}",
            opponent_policy=runtime["opponent_policy"],
            base_seed=runtime["seed"],
            max_steps=runtime["max_steps"],
            trial_timeout_sec=runtime["trial_timeout_sec"],
        )
    except Exception as e:
        print(f"  [trial {trial.number}] ERROR: {e}", flush=True)
        raise optuna.exceptions.TrialPruned()

    win_rate = float(result.get("win_rate_a", 0))
    gold_delta = float(result.get("mean_gold_delta_a", 0))

    # 정규화된 골드델타 (상한/하한 클램프)
    gold_norm = max(-1.0, min(1.0, gold_delta / GOLD_DELTA_SCALE))

    score = win_rate * WIN_RATE_WEIGHT + gold_norm * GOLD_DELTA_WEIGHT

    print(
        f"  [trial {trial.number:3d}] "
        f"win={win_rate:.4f}  gold_delta={gold_delta:+.1f}  score={score:.4f}",
        flush=True,
    )

    # 중간 속성 저장 (분석용)
    trial.set_user_attr("win_rate", win_rate)
    trial.set_user_attr("gold_delta", gold_delta)
    trial.set_user_attr("wins_a", result.get("wins_a", 0))
    trial.set_user_attr("losses_a", result.get("losses_a", 0))

    return score


# ── 메인 ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="V5 Optuna 튜닝")
    p.add_argument("--trials",   type=int, default=9,    help="트라이얼 수")
    p.add_argument("--workers",  type=int, default=1,    help="병렬 워커 수 (sqlite 사용 시)")
    p.add_argument("--db",       type=str, default="",   help="Optuna DB URL (예: sqlite:///logs/optuna_v5.db)")
    p.add_argument("--study",    type=str, default="v5_tuning", help="스터디 이름")
    p.add_argument("--output",   type=str, default="logs/optuna_v5_best.json", help="최적 파라미터 출력 경로")
    p.add_argument("--timeout",  type=int, default=0,    help="전체 튜닝 시간 제한(초, 0=무제한)")
    p.add_argument("--seed",     type=str, default=SEED, help="기본 시드")
    p.add_argument("--max-steps", type=int, default=MAX_STEPS, help="듀얼 max steps")
    p.add_argument("--trial-timeout", type=int, default=TRIAL_TIMEOUT_SEC, help="트라이얼당 타임아웃(초)")
    p.add_argument("--opponent-policy", type=str, default=OPPONENT_POLICY, help="상대 정책 (기본 heuristic_v4)")
    p.add_argument("--startup-trials", type=int, default=20, help="TPE startup trials")
    return p.parse_args()


def main():
    args = parse_args()

    duel_path = Path(DUEL_SCRIPT)
    if not duel_path.exists():
        print(f"duel worker not found: {duel_path}")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    runtime = {
        "opponent_policy": str(args.opponent_policy).strip().lower(),
        "seed": str(args.seed).strip() or SEED,
        "max_steps": max(20, int(args.max_steps)),
        "trial_timeout_sec": max(60, int(args.trial_timeout)),
    }

    storage = args.db if args.db else None

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=max(1, int(args.startup_trials)),   # 초기 랜덤 탐색
        multivariate=True,     # 파라미터 간 상관관계 고려
        seed=42,
    )

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study,
        storage=storage,
        sampler=sampler,
        load_if_exists=True,
    )

    print(f"=== V5 Optuna 튜닝 시작 ===")
    print(f"  trials={args.trials}  opponent={runtime['opponent_policy']}  games/trial={GAMES}")
    print(f"  seed={runtime['seed']}  max_steps={runtime['max_steps']}  trial_timeout={runtime['trial_timeout_sec']}s")
    print(f"  DB: {storage or '메모리 (비영속)'}")
    print(f"  목적: win_rate×{WIN_RATE_WEIGHT} + gold_delta×{GOLD_DELTA_WEIGHT}")
    print()

    study.optimize(
        lambda trial: objective(trial, runtime),
        n_trials=args.trials,
        n_jobs=args.workers,
        timeout=args.timeout if args.timeout > 0 else None,
        show_progress_bar=False,
    )

    best = study.best_trial
    print()
    print(f"=== 최적 결과 ===")
    print(f"  Score:      {best.value:.4f}")
    print(f"  win_rate:   {best.user_attrs.get('win_rate', '?'):.4f}")
    print(f"  gold_delta: {best.user_attrs.get('gold_delta', '?'):+.1f}")
    print(f"  W/L:        {best.user_attrs.get('wins_a','?')}/{best.user_attrs.get('losses_a','?')}")
    print()

    best_params = {**best.params}
    output = {
        "score": best.value,
        "win_rate": best.user_attrs.get("win_rate"),
        "gold_delta": best.user_attrs.get("gold_delta"),
        "params": best_params,
        "trial_number": best.number,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "opponent": runtime["opponent_policy"],
        "seed": runtime["seed"],
        "max_steps": runtime["max_steps"],
        "games_per_trial": GAMES,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  최적 파라미터 저장: {args.output}")

    # 상위 5개 출력
    print()
    print("=== Top 5 trials ===")
    trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value is not None else -math.inf, reverse=True)
    for t in trials_sorted[:5]:
        if t.value is None:
            continue
        print(
            f"  trial {t.number:3d}  score={t.value:.4f}"
            f"  win={t.user_attrs.get('win_rate', '?'):.4f}"
            f"  gold={t.user_attrs.get('gold_delta', '?'):+.1f}"
        )


if __name__ == "__main__":
    main()
