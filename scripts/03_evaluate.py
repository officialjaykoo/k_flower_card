#!/usr/bin/env python3
import argparse
import glob
import hashlib
import json
import math
import os
from datetime import datetime

SIDE_MY = "mySide"
SIDE_YOUR = "yourSide"


def expand_inputs(patterns):
    paths = []
    for pattern in patterns:
        matched = glob.glob(pattern)
        if matched:
            paths.extend(matched)
        elif os.path.isfile(pattern):
            paths.append(pattern)
    out = sorted(set(paths))
    if not out:
        raise FileNotFoundError("No input files matched.")
    return out


def action_alias(action):
    if not action:
        return None
    aliases = {
        "choose_go": "go",
        "choose_stop": "stop",
        "choose_kung_use": "kung_use",
        "choose_kung_pass": "kung_pass",
        "choose_president_stop": "president_stop",
        "choose_president_hold": "president_hold",
    }
    return aliases.get(action, action)


def extract_policy_sample(trace):
    sp = trace.get("sp") or {}
    cards = sp.get("cards")
    if cards:
        chosen = trace.get("c")
        if chosen in cards:
            return "play", cards, chosen
        return None
    board = sp.get("boardCardIds")
    if board:
        chosen = trace.get("s")
        if chosen in board:
            return "match", board, chosen
        return None
    options = sp.get("options")
    if options:
        chosen = action_alias(trace.get("at"))
        if chosen in options:
            return "option", options, chosen
        return None
    return None


def policy_context_key(trace, decision_type):
    dc = trace.get("dc") or {}
    sp = trace.get("sp") or {}
    deck_bucket = int((dc.get("deckCount") or 0) // 3)
    hand_self = dc.get("handCountSelf", 0)
    hand_opp = dc.get("handCountOpp", 0)
    go_self = dc.get("goCountSelf", 0)
    go_opp = dc.get("goCountOpp", 0)
    cands = len(
        sp.get("cards")
        or sp.get("boardCardIds")
        or sp.get("options")
        or []
    )
    return "|".join(
        [
            f"dt={decision_type}",
            f"ph={dc.get('phase','?')}",
            f"o={trace.get('o','?')}",
            f"db={deck_bucket}",
            f"hs={hand_self}",
            f"ho={hand_opp}",
            f"gs={go_self}",
            f"go={go_opp}",
            f"cc={cands}",
        ]
    )


def policy_prob(model, sample, choice):
    alpha = model.get("alpha", 1.0)
    dt = sample["decision_type"]
    candidates = sample["candidates"]
    ck = sample["context_key"]
    k = max(1, len(candidates))

    dt_context_counts = (model.get("context_counts") or {}).get(dt) or {}
    dt_context_totals = (model.get("context_totals") or {}).get(dt) or {}
    ctx_counts = dt_context_counts.get(ck)
    if ctx_counts:
        total = dt_context_totals.get(ck, 0)
        return (ctx_counts.get(choice, 0) + alpha) / (total + alpha * k)

    dt_global = (model.get("global_counts") or {}).get(dt) or {}
    total = sum(dt_global.get(c, 0) for c in candidates)
    return (dt_global.get(choice, 0) + alpha) / (total + alpha * k)


def policy_prob_raw(model, decision_type, candidates, context_key, choice):
    alpha = model.get("alpha", 1.0)
    k = max(1, len(candidates))

    dt_context_counts = (model.get("context_counts") or {}).get(decision_type) or {}
    dt_context_totals = (model.get("context_totals") or {}).get(decision_type) or {}
    ctx_counts = dt_context_counts.get(context_key)
    if ctx_counts:
        total = dt_context_totals.get(context_key, 0)
        return (ctx_counts.get(choice, 0) + alpha) / (total + alpha * k)

    dt_global = (model.get("global_counts") or {}).get(decision_type) or {}
    total = sum(dt_global.get(c, 0) for c in candidates)
    return (dt_global.get(choice, 0) + alpha) / (total + alpha * k)


def policy_top1(model, sample):
    best = None
    best_p = -1.0
    for c in sample["candidates"]:
        p = policy_prob(model, sample, c)
        if p > best_p:
            best = c
            best_p = p
    return best, best_p


def policy_top1_raw(model, decision_type, candidates, context_key):
    best = None
    best_p = -1.0
    for c in candidates:
        p = policy_prob_raw(model, decision_type, candidates, context_key, c)
        if p > best_p:
            best = c
            best_p = p
    return best, best_p


def stable_hash(token, dim):
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % dim


def value_sample(trace, decision_type, chosen, score, gold_per_point):
    actor = trace.get("a")
    if actor not in (SIDE_MY, SIDE_YOUR):
        return None
    dc = trace.get("dc") or {}
    sp = trace.get("sp") or {}
    self_score = score.get(actor)
    opp = SIDE_YOUR if actor == SIDE_MY else SIDE_MY
    opp_score = score.get(opp)
    if self_score is None or opp_score is None:
        return None
    y = (float(self_score) - float(opp_score)) * float(gold_per_point)
    tokens = [
        f"phase={dc.get('phase','?')}",
        f"order={trace.get('o','?')}",
        f"decision_type={decision_type}",
        f"action={chosen or '?'}",
        f"deck_bucket={int((dc.get('deckCount') or 0)//3)}",
        f"self_hand={int(dc.get('handCountSelf') or 0)}",
        f"opp_hand={int(dc.get('handCountOpp') or 0)}",
        f"self_go={int(dc.get('goCountSelf') or 0)}",
        f"opp_go={int(dc.get('goCountOpp') or 0)}",
    ]
    numeric = {
        "deck_count": float(dc.get("deckCount") or 0),
        "hand_self": float(dc.get("handCountSelf") or 0),
        "hand_opp": float(dc.get("handCountOpp") or 0),
        "go_self": float(dc.get("goCountSelf") or 0),
        "go_opp": float(dc.get("goCountOpp") or 0),
        "cand_count": float(len(sp.get("cards") or sp.get("boardCardIds") or sp.get("options") or [])),
        "immediate_reward": float(trace.get("ir") or 0),
    }
    return {"tokens": tokens, "numeric": numeric, "y": y}


def value_predict(model, sample):
    dim = int(model["dim"])
    w = model["weights"]
    b = float(model["bias"])
    scale = model.get("numeric_scale") or {}
    total = b
    for tok in sample["tokens"]:
        i = stable_hash(f"tok:{tok}", dim)
        total += w[i]
    for k, v in sample["numeric"].items():
        i = stable_hash(f"num:{k}", dim)
        total += w[i] * (v / max(1e-9, float(scale.get(k, 1.0))))
    return total


def main():
    parser = argparse.ArgumentParser(description="Evaluate old vs new policy/value models.")
    parser.add_argument("--input", nargs="+", default=["logs/*.jsonl"])
    parser.add_argument("--policy-old", required=True)
    parser.add_argument("--policy-new", required=True)
    parser.add_argument("--value-old", default=None)
    parser.add_argument("--value-new", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    input_paths = expand_inputs(args.input)
    with open(args.policy_old, "r", encoding="utf-8") as f:
        policy_old = json.load(f)
    with open(args.policy_new, "r", encoding="utf-8") as f:
        policy_new = json.load(f)

    value_old = None
    value_new = None
    if args.value_old and os.path.isfile(args.value_old):
        with open(args.value_old, "r", encoding="utf-8") as f:
            value_old = json.load(f)
    if args.value_new and os.path.isfile(args.value_new):
        with open(args.value_new, "r", encoding="utf-8") as f:
            value_new = json.load(f)

    total = 0
    old_correct = 0
    new_correct = 0
    old_nll = 0.0
    new_nll = 0.0
    duel_new = 0
    duel_old = 0
    duel_tie = 0

    v_total = 0
    v_old_abs = 0.0
    v_new_abs = 0.0
    v_old_sq = 0.0
    v_new_sq = 0.0
    v_duel_new = 0
    v_duel_old = 0
    v_duel_tie = 0

    for path in input_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line_raw in f:
                line_raw = line_raw.strip()
                if not line_raw:
                    continue
                line = json.loads(line_raw)
                decision_trace = line.get("decision_trace") or []
                score = line.get("score") or {}
                for trace in decision_trace:
                    s = extract_policy_sample(trace)
                    if s is None:
                        continue
                    dt, candidates, chosen = s
                    ck = policy_context_key(trace, dt)
                    total += 1
                    p_old = max(1e-12, policy_prob_raw(policy_old, dt, candidates, ck, chosen))
                    p_new = max(1e-12, policy_prob_raw(policy_new, dt, candidates, ck, chosen))
                    old_nll += -math.log(p_old)
                    new_nll += -math.log(p_new)
                    old_top, _ = policy_top1_raw(policy_old, dt, candidates, ck)
                    new_top, _ = policy_top1_raw(policy_new, dt, candidates, ck)
                    if old_top == chosen:
                        old_correct += 1
                    if new_top == chosen:
                        new_correct += 1
                    if p_new > p_old:
                        duel_new += 1
                    elif p_old > p_new:
                        duel_old += 1
                    else:
                        duel_tie += 1

                    if value_old and value_new:
                        gold_per_point = value_new.get("gold_per_point", 100.0)
                        vs = value_sample(trace, dt, chosen, score, gold_per_point)
                        if vs:
                            y = vs["y"]
                            pred_old = value_predict(value_old, vs)
                            pred_new = value_predict(value_new, vs)
                            e_old = abs(pred_old - y)
                            e_new = abs(pred_new - y)
                            v_total += 1
                            v_old_abs += e_old
                            v_new_abs += e_new
                            v_old_sq += (pred_old - y) ** 2
                            v_new_sq += (pred_new - y) ** 2
                            if e_new < e_old:
                                v_duel_new += 1
                            elif e_old < e_new:
                                v_duel_old += 1
                            else:
                                v_duel_tie += 1

    report = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "inputs": input_paths,
        "policy": {
            "samples": total,
            "old_top1_accuracy": old_correct / max(1, total),
            "new_top1_accuracy": new_correct / max(1, total),
            "old_nll_per_sample": old_nll / max(1, total),
            "new_nll_per_sample": new_nll / max(1, total),
            "duel": {
                "new_wins": duel_new,
                "old_wins": duel_old,
                "ties": duel_tie,
                "new_win_rate": duel_new / max(1, duel_new + duel_old + duel_tie),
            },
        },
        "value": None,
    }

    if value_old and value_new:
        report["value"] = {
            "samples": v_total,
            "old_mae": v_old_abs / max(1, v_total),
            "new_mae": v_new_abs / max(1, v_total),
            "old_rmse": math.sqrt(v_old_sq / max(1, v_total)),
            "new_rmse": math.sqrt(v_new_sq / max(1, v_total)),
            "duel": {
                "new_wins": v_duel_new,
                "old_wins": v_duel_old,
                "ties": v_duel_tie,
                "new_win_rate": v_duel_new / max(1, v_duel_new + v_duel_old + v_duel_tie),
            },
        }

    out = args.output
    if out is None:
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out = os.path.join("logs", f"model-eval-{stamp}.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"evaluation report -> {out}")
    print(json.dumps(report["policy"], ensure_ascii=False))
    if report["value"] is not None:
        print(json.dumps(report["value"], ensure_ascii=False))


if __name__ == "__main__":
    main()
