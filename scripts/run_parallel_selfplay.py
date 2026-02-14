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
    return os.path.join("logs", f"ai-vs-ai-parallel-{stamp}.jsonl")


def main():
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
    winners = {"human": 0, "ai": 0, "draw": 0, "unknown": 0}
    first_wins = 0
    second_wins = 0
    draws = 0
    first_score_sum = 0.0
    second_score_sum = 0.0
    human_gold_sum = 0.0
    ai_gold_sum = 0.0
    human_gold_delta_sum = 0.0
    first_gold_sum = 0.0
    second_gold_sum = 0.0
    first1000_human_gold_delta_sum = 0.0
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
                    order = obj.get("winnerTurnOrder")
                    if order == "first":
                        first_wins += 1
                    elif order == "second":
                        second_wins += 1
                    else:
                        draws += 1
                    score = obj.get("score") or {}
                    first_score_sum += float(score.get("first") or 0)
                    second_score_sum += float(score.get("second") or 0)
                    gold = obj.get("gold") or {}
                    g_h = float(gold.get("human") or 0)
                    g_a = float(gold.get("ai") or 0)
                    g_f = float(gold.get("first") or 0)
                    g_s = float(gold.get("second") or 0)
                    human_gold_sum += g_h
                    ai_gold_sum += g_a
                    human_gold_delta_sum += g_h - g_a
                    first_gold_sum += g_f
                    second_gold_sum += g_s
                    if first1000_count < 1000:
                        first1000_human_gold_delta_sum += g_h - g_a
                        first1000_count += 1

    report_path = out_path[:-6] + "-report.json" if out_path.endswith(".jsonl") else out_path + "-report.json"
    games = max(1, args.games)
    report = {
        "logMode": "merged_parallel",
        "games": args.games,
        "completed": completed,
        "winners": winners,
        "turnOrder": {
            "firstWinRate": first_wins / games,
            "secondWinRate": second_wins / games,
            "drawRate": draws / games,
            "averageScoreFirst": first_score_sum / games,
            "averageScoreSecond": second_score_sum / games,
        },
        "economy": {
            "averageGoldHuman": human_gold_sum / games,
            "averageGoldAi": ai_gold_sum / games,
            "averageGoldDeltaHuman": human_gold_delta_sum / games,
            "averageGoldFirst": first_gold_sum / games,
            "averageGoldSecond": second_gold_sum / games,
            "averageGoldDeltaFirst": (first_gold_sum - second_gold_sum) / games,
            "cumulativeGoldDeltaOver1000": (human_gold_delta_sum / games) * 1000,
            "cumulativeGoldDeltaFirst1000": first1000_human_gold_delta_sum,
        },
        "primaryMetric": "averageGoldDeltaHuman",
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

