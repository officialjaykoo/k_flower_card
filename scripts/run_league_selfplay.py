#!/usr/bin/env python3
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
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
        "policy": base.get("fallback_policy", "heuristic_v1"),
        "policy_model": base.get("policy_model", ""),
        "value_model": base.get("value_model", ""),
    }


def add_actor_args(cmd, role, actor):
    cmd.append(f"--policy-{role}={actor['policy']}")
    if actor["policy_model"]:
        cmd.append(f"--policy-model-{role}={actor['policy_model']}")
    if actor["value_model"]:
        cmd.append(f"--value-model-{role}={actor['value_model']}")


def main():
    parser = argparse.ArgumentParser(description="League self-play generator with weighted opponent pool.")
    parser.add_argument("--config", required=True, help="League config JSON path.")
    parser.add_argument("--total-games", type=int, required=True, help="Total games to generate (even).")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers.")
    parser.add_argument("--output", required=True, help="Merged output JSONL.")
    parser.add_argument("--chunk-games", type=int, default=None, help="Games per matchup chunk (even).")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--keep-chunks", action="store_true")
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
    chunk_infos = []
    aggregate = {"games": 0, "completed": 0, "winners": {"human": 0, "ai": 0, "draw": 0, "unknown": 0}, "matchups": {}}

    i = 0
    while remaining > 0:
        i += 1
        g = min(chunk_games, remaining)
        if g % 2 != 0:
            g -= 1
        if g <= 0:
            g = 2
        opp = weighted_pick(opponents, rng)
        human = focal if not side_flip else opp
        ai = opp if not side_flip else focal
        side_flip = not side_flip

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
        ]
        add_actor_args(cmd, "human", human)
        add_actor_args(cmd, "ai", ai)
        if model_only_when_possible and (human["policy_model"] or human["value_model"]) and (ai["policy_model"] or ai["value_model"]):
            cmd.append("--model-only")
        run(cmd)

        rep_path = chunk_out[:-6] + "-report.json"
        with open(rep_path, "r", encoding="utf-8-sig") as f:
            rep = json.load(f)

        aggregate["games"] += int(rep.get("games", 0))
        aggregate["completed"] += int(rep.get("completed", 0))
        for k in aggregate["winners"].keys():
            aggregate["winners"][k] += int((rep.get("winners") or {}).get(k, 0))
        key = f"{human['name']}(H)_vs_{ai['name']}(A)"
        aggregate["matchups"][key] = aggregate["matchups"].get(key, 0) + int(rep.get("games", 0))

        chunk_infos.append({"chunk": i, "games": g, "output": chunk_out, "report": rep_path, "human": human["name"], "ai": ai["name"]})
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

