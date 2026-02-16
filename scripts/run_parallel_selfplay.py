#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime


def split_games(total, workers):
    if total % 2 != 0:
        raise RuntimeError("games must be even because each worker run requires even games.")
    pair_total = total // 2
    base = pair_total // workers
    rem = pair_total % workers
    return [2 * (base + (1 if i < rem else 0)) for i in range(workers)]


def build_default_out():
    stamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    return os.path.join("logs", f"side-vs-side-parallel-{stamp}.jsonl")


def main():
    if os.environ.get("NO_SIMULATION") == "1":
        raise RuntimeError("Simulation blocked: NO_SIMULATION=1")

    parser = argparse.ArgumentParser(description="Run selfplay_simulator.mjs in parallel workers and merge outputs.")
    parser.add_argument("games", type=int, help="Total number of games.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers.")
    parser.add_argument("--output", default=None, help="Merged JSONL output path.")
    parser.add_argument("--node", default="node", help="Node executable.")
    parser.add_argument("--script", default="scripts/selfplay_simulator.mjs", help="Path to simulate script.")
    parser.add_argument("--keep-shards", action="store_true", help="Keep worker shard outputs.")
    args, passthrough = parser.parse_known_args()

    if args.games <= 0:
        raise RuntimeError("games must be > 0")
    if args.workers <= 0:
        raise RuntimeError("workers must be > 0")

    out_path = args.output or build_default_out()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    base = out_path[:-6] if out_path.endswith(".jsonl") else out_path
    shard_counts = split_games(args.games, args.workers)

    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    procs = []
    shard_paths = []
    for wi, n_games in enumerate(shard_counts):
        if n_games <= 0:
            continue
        shard_out = f"{base}.part{wi + 1}.jsonl"
        shard_paths.append(shard_out)
        cmd = [args.node, args.script, str(n_games), shard_out, *passthrough]
        print(">", " ".join(cmd))
        procs.append(subprocess.Popen(cmd))

    failed = False
    for p in procs:
        rc = p.wait()
        if rc != 0:
            failed = True
    if failed:
        raise RuntimeError("One or more worker processes failed.")

    completed = 0
    winners = {"mySide": 0, "yourSide": 0, "draw": 0, "unknown": 0}
    my_side_score_sum = 0.0
    your_side_score_sum = 0.0
    my_side_gold_sum = 0.0
    your_side_gold_sum = 0.0
    my_side_gold_delta_sum = 0.0
    first1000_my_side_gold_delta_sum = 0.0
    first1000_count = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for shard in shard_paths:
            with open(shard, "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if obj.get("completed"):
                        completed += 1
                    winner = obj.get("winner")
                    if winner in winners:
                        winners[winner] += 1
                    else:
                        winners["unknown"] += 1
                    score = obj.get("score") or {}
                    my_side_score_sum += float(score.get("mySide") or 0)
                    your_side_score_sum += float(score.get("yourSide") or 0)
                    gold = obj.get("gold") or {}
                    g_m = float(gold.get("mySide") or 0)
                    g_y = float(gold.get("yourSide") or 0)
                    my_side_gold_sum += g_m
                    your_side_gold_sum += g_y
                    my_side_gold_delta_sum += g_m - g_y
                    if first1000_count < 1000:
                        first1000_my_side_gold_delta_sum += g_m - g_y
                        first1000_count += 1

    report_path = out_path[:-6] + "-report.json" if out_path.endswith(".jsonl") else out_path + "-report.json"
    games = max(1, args.games)
    report = {
        "logMode": "merged_parallel",
        "games": args.games,
        "completed": completed,
        "winners": winners,
        "sideStats": {
            "mySideWinRate": winners["mySide"] / games,
            "yourSideWinRate": winners["yourSide"] / games,
            "drawRate": winners["draw"] / games,
            "averageScoreMySide": my_side_score_sum / games,
            "averageScoreYourSide": your_side_score_sum / games,
        },
        "economy": {
            "averageGoldMySide": my_side_gold_sum / games,
            "averageGoldYourSide": your_side_gold_sum / games,
            "averageGoldDeltaMySide": my_side_gold_delta_sum / games,
            "cumulativeGoldDeltaOver1000": (my_side_gold_delta_sum / games) * 1000,
            "cumulativeGoldDeltaMySideFirst1000": first1000_my_side_gold_delta_sum,
        },
        "primaryMetric": "averageGoldDeltaMySide",
        "workers": args.workers,
        "shards": shard_paths,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if not args.keep_shards:
        for shard in shard_paths:
            if os.path.exists(shard):
                os.remove(shard)
            shard_report = shard[:-6] + "-report.json" if shard.endswith(".jsonl") else shard + "-report.json"
            if os.path.exists(shard_report):
                os.remove(shard_report)

    print(f"merged: {out_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()

