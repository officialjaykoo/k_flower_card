#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
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


def _trace_phase_code(phase):
    if isinstance(phase, (int, float)):
        return int(phase)
    if isinstance(phase, str):
        phase_map = {
            "playing": 1,
            "select-match": 2,
            "go-stop": 3,
            "president-choice": 4,
            "gukjin-choice": 5,
            "shaking-confirm": 6,
            "resolution": 7,
        }
        return int(phase_map.get(phase, 0))
    return 0


def _policy_context_key(trace):
    if not isinstance(trace, dict):
        return None
    cached = trace.get("ck")
    if isinstance(cached, str) and cached:
        return cached
    dc = trace.get("dc") or {}
    decision_type = str(trace.get("dt") or "?")
    order = str(trace.get("o") or "?")
    deck_bucket = int((dc.get("deckCount") or 0) // 3)
    hand_self = int(dc.get("handCountSelf") or 0)
    hand_opp = int(dc.get("handCountOpp") or 0)
    go_self = int(dc.get("goCountSelf") or 0)
    go_opp = int(dc.get("goCountOpp") or 0)
    carry = max(1, int(dc.get("carryOverMultiplier") or 1))
    shake_self = min(3, int(dc.get("shakeCountSelf") or 0))
    shake_opp = min(3, int(dc.get("shakeCountOpp") or 0))
    cands = int(trace.get("cc") or 0)
    phase = _trace_phase_code(dc.get("phase"))
    return "|".join(
        [
            f"dt={decision_type}",
            f"ph={phase}",
            f"o={order}",
            f"db={deck_bucket}",
            f"hs={hand_self}",
            f"ho={hand_opp}",
            f"gs={go_self}",
            f"go={go_opp}",
            f"cm={carry}",
            f"ss={shake_self}",
            f"so={shake_opp}",
            f"cc={cands}",
        ]
    )


def _minimal_policy_trace(trace):
    if not isinstance(trace, dict):
        return None
    cc = trace.get("cc")
    try:
        if cc is not None and int(cc) <= 1:
            return None
    except Exception:
        pass
    out = {}
    for k in ("a", "o", "dt", "cc", "at", "c", "s"):
        if k in trace:
            out[k] = trace.get(k)
    ck = _policy_context_key(trace)
    if isinstance(ck, str) and ck:
        out["ck"] = ck
    else:
        return None
    return out


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
    parser.add_argument(
        "--emit-train-splits",
        action="store_true",
        help="Write minimal per-shard training logs: *.train_policy.jsonl",
    )
    parser.add_argument(
        "--shard-only",
        action="store_true",
        help="Do not create merged output JSONL. Keep and use per-worker shard files.",
    )
    args, passthrough = parser.parse_known_args()

    if args.games <= 0:
        raise RuntimeError("games must be > 0")
    if args.workers <= 0:
        raise RuntimeError("workers must be > 0")
    if args.shard_only:
        args.keep_shards = True

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
    my_side_bankrupt_inflicted = 0
    my_side_bankrupt_suffered = 0
    your_side_bankrupt_inflicted = 0
    your_side_bankrupt_suffered = 0
    bankrupt_resets = 0
    first1000_my_side_gold_delta_sum = 0.0
    first1000_count = 0
    train_policy_shards = []

    merged_writer = None
    try:
        if not args.shard_only:
            merged_writer = open(out_path, "w", encoding="utf-8")

        for shard in shard_paths:
            policy_writer = None
            policy_out = None
            if args.emit_train_splits:
                base_shard = shard[:-6] if shard.endswith(".jsonl") else shard
                policy_out = f"{base_shard}.train_policy.jsonl"
                policy_writer = open(policy_out, "w", encoding="utf-8")
                train_policy_shards.append(policy_out)

            try:
                with open(shard, "r", encoding="utf-8") as fin:
                    for line in fin:
                        if merged_writer is not None:
                            merged_writer.write(line)
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
                        g_m = float(gold.get("mySide") if gold.get("mySide") is not None else (obj.get("finalGoldMy") or 0))
                        g_y = float(gold.get("yourSide") if gold.get("yourSide") is not None else (obj.get("finalGoldYour") or 0))
                        my_side_gold_sum += g_m
                        your_side_gold_sum += g_y
                        my_side_gold_delta_sum += g_m - g_y
                        my_bankrupt = 1 if g_m <= 0 else 0
                        your_bankrupt = 1 if g_y <= 0 else 0
                        my_side_bankrupt_inflicted += your_bankrupt
                        my_side_bankrupt_suffered += my_bankrupt
                        your_side_bankrupt_inflicted += my_bankrupt
                        your_side_bankrupt_suffered += your_bankrupt
                        if my_bankrupt or your_bankrupt:
                            bankrupt_resets += 1
                        if first1000_count < 1000:
                            first1000_my_side_gold_delta_sum += g_m - g_y
                            first1000_count += 1

                        if policy_writer is not None:
                            raw_traces = obj.get("decision_trace") or []
                            policy_traces = []
                            for tr in raw_traces:
                                m = _minimal_policy_trace(tr)
                                if m is not None:
                                    policy_traces.append(m)
                            if policy_traces:
                                policy_writer.write(
                                    json.dumps(
                                        {
                                            "firstAttackerSide": obj.get("firstAttackerSide"),
                                            "firstAttackerActor": obj.get("firstAttackerActor"),
                                            "initialGoldMy": obj.get("initialGoldMy"),
                                            "initialGoldYour": obj.get("initialGoldYour"),
                                            "finalGoldMy": obj.get("finalGoldMy"),
                                            "finalGoldYour": obj.get("finalGoldYour"),
                                            "goldDeltaMy": obj.get("goldDeltaMy"),
                                            "goldDeltaYour": obj.get("goldDeltaYour"),
                                            "goldDeltaMyRatio": obj.get("goldDeltaMyRatio"),
                                            "goldDeltaMyNorm": obj.get("goldDeltaMyNorm"),
                                            "decision_trace": policy_traces,
                                        },
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )

            finally:
                if policy_writer is not None:
                    policy_writer.close()
    finally:
        if merged_writer is not None:
            merged_writer.close()

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
        "bankrupt": {
            "mySideInflicted": int(my_side_bankrupt_inflicted),
            "mySideSuffered": int(my_side_bankrupt_suffered),
            "mySideDiff": int(my_side_bankrupt_inflicted - my_side_bankrupt_suffered),
            "yourSideInflicted": int(your_side_bankrupt_inflicted),
            "yourSideSuffered": int(your_side_bankrupt_suffered),
            "yourSideDiff": int(your_side_bankrupt_inflicted - your_side_bankrupt_suffered),
            "resets": int(bankrupt_resets),
            "mySideInflictedRate": my_side_bankrupt_inflicted / games,
            "mySideSufferedRate": my_side_bankrupt_suffered / games,
            "yourSideInflictedRate": your_side_bankrupt_inflicted / games,
            "yourSideSufferedRate": your_side_bankrupt_suffered / games,
        },
        "primaryMetric": "averageGoldDeltaMySide",
        "workers": args.workers,
        "shards": shard_paths,
    }
    if args.emit_train_splits:
        report["trainSplits"] = {
            "policyShards": train_policy_shards,
        }
    if args.shard_only:
        report["mergedOutput"] = None

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if not args.keep_shards:
        for shard in shard_paths:
            if os.path.exists(shard):
                os.remove(shard)
            shard_report = shard[:-6] + "-report.json" if shard.endswith(".jsonl") else shard + "-report.json"
            if os.path.exists(shard_report):
                os.remove(shard_report)

    if args.shard_only:
        print("merged: skipped (--shard-only)")
    else:
        print(f"merged: {out_path}")
    print(f"report: {report_path}")
    if args.emit_train_splits:
        print("train policy shards:")
        for p in train_policy_shards:
            print(f"- {p}")


if __name__ == "__main__":
    main()

