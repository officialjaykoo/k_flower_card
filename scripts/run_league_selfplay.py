#!/usr/bin/env python3
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import math
from datetime import datetime, timezone


def run(cmd):
    print(">", " ".join(cmd))
    cp = subprocess.run(cmd)
    if cp.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "focal" not in cfg or "opponents" not in cfg:
        raise RuntimeError("Config must contain focal and opponents.")
    if not cfg["opponents"]:
        raise RuntimeError("Config opponents must be non-empty.")
    return cfg


def weighted_pick(items, rng):
    ws = [max(0.0, float(i.get("weight", 1.0))) for i in items]
    total = sum(ws)
    if total <= 0:
        return rng.choice(items)
    t = rng.random() * total
    acc = 0.0
    for item, w in zip(items, ws):
        acc += w
        if t <= acc:
            return item
    return items[-1]


def build_actor(base):
    return {
        "name": base.get("name", "anon"),
        "policy": base.get("fallback_policy", "heuristic_v3"),
        "policy_model": base.get("policy_model", ""),
        "value_model": base.get("value_model", ""),
    }


def add_actor_args(cmd, role, actor):
    cmd.append(f"--policy-{role}={actor['policy']}")
    if actor["policy_model"]:
        cmd.append(f"--policy-model-{role}={actor['policy_model']}")
    if actor["value_model"]:
        cmd.append(f"--value-model-{role}={actor['value_model']}")

def scheduled_rate(base_rate, min_rate, decay_k, progress):
    b = max(0.0, min(1.0, float(base_rate)))
    m = max(0.0, min(1.0, float(min_rate)))
    if m > b:
        m = b
    k = max(0.0, float(decay_k))
    if k <= 0:
        return b
    p = max(0.0, min(1.0, float(progress)))
    decay = math.exp(-k * p)
    return m + (b - m) * decay


def main():
    if os.environ.get("NO_SIMULATION") == "1":
        raise RuntimeError("Simulation blocked: NO_SIMULATION=1")

    parser = argparse.ArgumentParser(description="League self-play generator with weighted opponent pool.")
    parser.add_argument("--config", required=True, help="League config JSON path.")
    parser.add_argument("--total-games", type=int, required=True, help="Total games to generate (even).")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers.")
    parser.add_argument("--output", required=True, help="Merged output JSONL.")
    parser.add_argument("--chunk-games", type=int, default=None, help="Games per matchup chunk (even).")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--keep-chunks", action="store_true")
    parser.add_argument("--explore-base", type=float, default=None, help="Base --train-explore-rate.")
    parser.add_argument(
        "--explore-comeback-base",
        type=float,
        default=None,
        help="Base --train-explore-rate-comeback.",
    )
    parser.add_argument("--explore-min", type=float, default=None, help="Floor for train explore rate.")
    parser.add_argument(
        "--explore-comeback-min",
        type=float,
        default=None,
        help="Floor for train comeback explore rate.",
    )
    parser.add_argument(
        "--explore-decay-k",
        type=float,
        default=None,
        help="Exponential decay coefficient for chunk-wise exploration scheduling.",
    )
    args = parser.parse_args()

    if args.total_games <= 0 or args.total_games % 2 != 0:
        raise RuntimeError("total-games must be positive even number.")
    if args.workers <= 0:
        raise RuntimeError("workers must be > 0")

    cfg = load_config(args.config)
    focal = build_actor(cfg["focal"])
    opponents = [build_actor(x) | {"weight": x.get("weight", 1.0)} for x in cfg["opponents"]]
    log_mode = cfg.get("log_mode", "train")
    model_only_when_possible = bool(cfg.get("model_only_when_possible", True))
    chunk_games = args.chunk_games or int(cfg.get("chunk_games", 20000))
    explore_base = (
        float(args.explore_base)
        if args.explore_base is not None
        else float(cfg.get("explore_base", 0.006))
    )
    explore_comeback_base = (
        float(args.explore_comeback_base)
        if args.explore_comeback_base is not None
        else float(cfg.get("explore_comeback_base", 0.012))
    )
    explore_min = (
        float(args.explore_min)
        if args.explore_min is not None
        else float(cfg.get("explore_min", 0.0))
    )
    explore_comeback_min = (
        float(args.explore_comeback_min)
        if args.explore_comeback_min is not None
        else float(cfg.get("explore_comeback_min", 0.0))
    )
    explore_decay_k = (
        float(args.explore_decay_k)
        if args.explore_decay_k is not None
        else float(cfg.get("explore_decay_k", 0.0))
    )
    if explore_base < 0 or explore_base > 1:
        raise RuntimeError("explore-base must be in [0,1].")
    if explore_comeback_base < 0 or explore_comeback_base > 1:
        raise RuntimeError("explore-comeback-base must be in [0,1].")
    if explore_min < 0 or explore_min > 1:
        raise RuntimeError("explore-min must be in [0,1].")
    if explore_comeback_min < 0 or explore_comeback_min > 1:
        raise RuntimeError("explore-comeback-min must be in [0,1].")
    if explore_decay_k < 0:
        raise RuntimeError("explore-decay-k must be >= 0.")
    if chunk_games <= 0 or chunk_games % 2 != 0:
        raise RuntimeError("chunk-games must be positive even number.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base = args.output[:-6] if args.output.endswith(".jsonl") else args.output
    chunk_dir = f"{base}.chunks-{stamp}"
    os.makedirs(chunk_dir, exist_ok=True)

    rng = random.Random(args.seed)
    remaining = args.total_games
    side_flip = False
    total_chunks = max(1, int(math.ceil(args.total_games / float(chunk_games))))
    chunk_infos = []
    aggregate = {"games": 0, "completed": 0, "winners": {"mySide": 0, "yourSide": 0, "draw": 0, "unknown": 0}, "matchups": {}}

    i = 0
    while remaining > 0:
        i += 1
        g = min(chunk_games, remaining)
        if g % 2 != 0:
            g -= 1
        if g <= 0:
            g = 2
        opp = weighted_pick(opponents, rng)
        my_side = focal if not side_flip else opp
        your_side = opp if not side_flip else focal
        side_flip = not side_flip

        progress = (i - 1) / max(1, total_chunks - 1)
        chunk_explore = scheduled_rate(explore_base, explore_min, explore_decay_k, progress)
        chunk_explore_comeback = scheduled_rate(
            explore_comeback_base, explore_comeback_min, explore_decay_k, progress
        )

        chunk_out = os.path.join(chunk_dir, f"chunk-{i:03d}.jsonl")
        cmd = [
            sys.executable,
            "scripts/run_parallel_selfplay.py",
            str(g),
            "--workers",
            str(args.workers),
            "--output",
            chunk_out,
            "--",
            f"--log-mode={log_mode}",
            f"--train-explore-rate={chunk_explore:.8f}",
            f"--train-explore-rate-comeback={chunk_explore_comeback:.8f}",
            "--train-explore-min=0",
            "--train-explore-comeback-min=0",
        ]
        add_actor_args(cmd, "my-side", my_side)
        add_actor_args(cmd, "your-side", your_side)
        if model_only_when_possible and (my_side["policy_model"] or my_side["value_model"]) and (your_side["policy_model"] or your_side["value_model"]):
            cmd.append("--model-only")
        run(cmd)

        rep_path = chunk_out[:-6] + "-report.json"
        with open(rep_path, "r", encoding="utf-8-sig") as f:
            rep = json.load(f)

        aggregate["games"] += int(rep.get("games", 0))
        aggregate["completed"] += int(rep.get("completed", 0))
        for k in aggregate["winners"].keys():
            aggregate["winners"][k] += int((rep.get("winners") or {}).get(k, 0))
        key = f"{my_side['name']}(MY)_vs_{your_side['name']}(YOUR)"
        aggregate["matchups"][key] = aggregate["matchups"].get(key, 0) + int(rep.get("games", 0))

        chunk_infos.append(
            {
                "chunk": i,
                "games": g,
                "output": chunk_out,
                "report": rep_path,
                "mySide": my_side["name"],
                "yourSide": your_side["name"],
                "exploreRate": chunk_explore,
                "exploreRateComeback": chunk_explore_comeback,
                "exploreProgress": progress,
            }
        )
        remaining -= g

    with open(args.output, "w", encoding="utf-8") as fout:
        for ci in chunk_infos:
            with open(ci["output"], "r", encoding="utf-8") as fin:
                shutil.copyfileobj(fin, fout)

    report_out = args.output[:-6] + "-report.json" if args.output.endswith(".jsonl") else args.output + "-report.json"
    report = {
        "logMode": "league_parallel",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": args.config,
        "games": aggregate["games"],
        "completed": aggregate["completed"],
        "winners": aggregate["winners"],
        "workers": args.workers,
        "chunks": chunk_infos,
        "matchups": aggregate["matchups"],
        "explore": {
            "base": explore_base,
            "comebackBase": explore_comeback_base,
            "min": explore_min,
            "comebackMin": explore_comeback_min,
            "decayK": explore_decay_k,
        },
    }
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if not args.keep_chunks:
        for ci in chunk_infos:
            for p in [ci["output"], ci["report"]]:
                if os.path.exists(p):
                    os.remove(p)
        if os.path.isdir(chunk_dir):
            try:
                os.rmdir(chunk_dir)
            except OSError:
                pass

    print(f"generated: {args.output}")
    print(f"report: {report_out}")


if __name__ == "__main__":
    main()
