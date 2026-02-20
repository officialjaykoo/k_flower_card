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
        "choose_shaking_yes": "shaking_yes",
        "choose_shaking_no": "shaking_no",
        "choose_president_stop": "president_stop",
        "choose_president_hold": "president_hold",
    }
    return aliases.get(action, action)


def trace_order(trace):
    o = str(trace.get("o") or "").strip().lower()
    if o in ("first", "second"):
        return o
    a = str(trace.get("a") or "").strip()
    if a == SIDE_MY:
        return "first"
    if a == SIDE_YOUR:
        return "second"
    return "?"


def extract_policy_sample(trace):
    dt = trace.get("dt")
    compact = trace.get("ch")
    chosen_play = compact if dt == "play" else trace.get("c")
    chosen_match = compact if dt == "match" else trace.get("s")
    chosen_option = action_alias(compact if dt == "option" else trace.get("at"))
    if dt == "play" and chosen_play:
        return "play", [chosen_play], chosen_play
    if dt == "match" and chosen_match:
        return "match", [chosen_match], chosen_match
    if dt == "option":
        by_chosen = {
            "go": ["go", "stop"],
            "stop": ["go", "stop"],
            "president_stop": ["president_stop", "president_hold"],
            "president_hold": ["president_stop", "president_hold"],
            "five": ["five", "junk"],
            "junk": ["five", "junk"],
            "shaking_yes": ["shaking_yes", "shaking_no"],
            "shaking_no": ["shaking_yes", "shaking_no"],
        }
        if chosen_option in by_chosen:
            return "option", by_chosen[chosen_option], chosen_option

    sp = trace.get("sp") or {}
    cards = sp.get("cards")
    if cards:
        chosen = compact if compact is not None else trace.get("c")
        if chosen in cards:
            return "play", cards, chosen
        return None
    board = sp.get("boardCardIds")
    if board:
        chosen = compact if compact is not None else trace.get("s")
        if chosen in board:
            return "match", board, chosen
        return None
    options = sp.get("options")
    if options:
        chosen = action_alias(compact if compact is not None else trace.get("at"))
        if chosen in options:
            return "option", options, chosen
        return None
    return None


def policy_context_key(trace, decision_type):
    cached = trace.get("ck")
    if isinstance(cached, str) and cached:
        return cached
    dc = trace.get("dc") or {}
    sp = trace.get("sp") or {}
    deck_bucket = int((dc.get("deckCount") or 0) // 3)
    hand_self = dc.get("handCountSelf", 0)
    hand_opp = dc.get("handCountOpp", 0)
    go_self = dc.get("goCountSelf", 0)
    go_opp = dc.get("goCountOpp", 0)
    carry = max(1, int(dc.get("carryOverMultiplier") or 1))
    shake_self = min(3, int(dc.get("shakeCountSelf") or 0))
    shake_opp = min(3, int(dc.get("shakeCountOpp") or 0))
    phase = dc.get("phase")
    if isinstance(phase, str):
        phase = {
            "playing": 1,
            "select-match": 2,
            "go-stop": 3,
            "president-choice": 4,
            "gukjin-choice": 5,
            "shaking-confirm": 6,
            "resolution": 7,
        }.get(phase, 0)
    else:
        try:
            phase = int(phase)
        except Exception:
            phase = 0
    cands = int(trace.get("cc") or 0)
    if cands <= 0:
        cands = len(
            sp.get("cards")
            or sp.get("boardCardIds")
            or sp.get("options")
            or []
        )
    return "|".join(
        [
            f"dt={decision_type}",
            f"ph={phase}",
            f"o={trace_order(trace)}",
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


def _float_or_none(v):
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def value_target(line, actor, gold_per_point, target_mode):
    mode = str(target_mode or "score").strip().lower()
    if mode == "gold":
        if actor == SIDE_MY:
            direct = _float_or_none(line.get("goldDeltaMy"))
            if direct is not None:
                return direct
            final_gold = _float_or_none(line.get("finalGoldMy"))
            initial_gold = _float_or_none(line.get("initialGoldMy"))
        else:
            direct = _float_or_none(line.get("goldDeltaYour"))
            if direct is not None:
                return direct
            final_gold = _float_or_none(line.get("finalGoldYour"))
            initial_gold = _float_or_none(line.get("initialGoldYour"))
        if final_gold is not None and initial_gold is not None:
            return final_gold - initial_gold
        mirror_my = _float_or_none(line.get("goldDeltaMy"))
        if mirror_my is not None:
            return mirror_my if actor == SIDE_MY else -mirror_my

    score = line.get("score") or {}
    self_score = score.get(actor)
    opp = SIDE_YOUR if actor == SIDE_MY else SIDE_MY
    opp_score = score.get(opp)
    if self_score is None or opp_score is None:
        return None
    return (float(self_score) - float(opp_score)) * float(gold_per_point)


def value_sample(trace, decision_type, chosen, line, gold_per_point, target_mode):
    actor = trace.get("a")
    if actor not in (SIDE_MY, SIDE_YOUR):
        return None
    dc = trace.get("dc") or {}
    sp = trace.get("sp") or {}
    y = value_target(line, actor, gold_per_point, target_mode)
    if y is None:
        return None
    def _prob01(v):
        try:
            x = float(v or 0.0)
        except Exception:
            return 0.0
        if x > 1.0:
            x = x / 100.0
        return max(0.0, min(1.0, x))

    def _signed_proxy(v, abs_max=3.0):
        try:
            x = float(v or 0.0)
        except Exception:
            return 0.0
        if abs(x) > abs_max:
            x = x / 1000.0
        if x > abs_max:
            x = abs_max
        if x < -abs_max:
            x = -abs_max
        return x

    bak_risk_total = (
        float(1 if dc.get("piBakRisk") else 0)
        + float(1 if dc.get("gwangBakRisk") else 0)
        + float(1 if dc.get("mongBakRisk") else 0)
    )
    score_diff = float(dc.get("currentScoreSelf") or 0) - float(dc.get("currentScoreOpp") or 0)
    tokens = [
        f"phase={dc.get('phase','?')}",
        f"order={trace_order(trace)}",
        f"decision_type={decision_type}",
        f"action={chosen or '?'}",
        f"deck_bucket={int((dc.get('deckCount') or 0)//3)}",
        f"self_hand={int(dc.get('handCountSelf') or 0)}",
        f"opp_hand={int(dc.get('handCountOpp') or 0)}",
        f"self_go={int(dc.get('goCountSelf') or 0)}",
        f"opp_go={int(dc.get('goCountOpp') or 0)}",
        f"score_diff={int(score_diff)}",
        f"bak_risk={int(bak_risk_total)}",
        f"jp_self_sum={int(dc.get('jokboProgressSelfSum') or 0)}",
        f"jp_opp_sum={int(dc.get('jokboProgressOppSum') or 0)}",
        f"go_stop_delta_bucket={int(_signed_proxy(dc.get('goStopDeltaProxy')) * 2)}",
    ]
    numeric = {
        "deck_count": float(dc.get("deckCount") or 0),
        "hand_self": float(dc.get("handCountSelf") or 0),
        "hand_opp": float(dc.get("handCountOpp") or 0),
        "go_self": float(dc.get("goCountSelf") or 0),
        "go_opp": float(dc.get("goCountOpp") or 0),
        "score_diff": score_diff,
        "bak_risk_total": bak_risk_total,
        "jokbo_progress_self_sum": float(dc.get("jokboProgressSelfSum") or 0),
        "jokbo_progress_opp_sum": float(dc.get("jokboProgressOppSum") or 0),
        "jokbo_one_away_self_count": float(dc.get("jokboOneAwaySelfCount") or 0),
        "jokbo_one_away_opp_count": float(dc.get("jokboOneAwayOppCount") or 0),
        "self_jokbo_threat_prob": _prob01(dc.get("selfJokboThreatProb")),
        "self_jokbo_one_away_prob": _prob01(dc.get("selfJokboOneAwayProb")),
        "self_gwang_threat_prob": _prob01(dc.get("selfGwangThreatProb")),
        "opp_jokbo_threat_prob": _prob01(dc.get("oppJokboThreatProb")),
        "opp_jokbo_one_away_prob": _prob01(dc.get("oppJokboOneAwayProb")),
        "opp_gwang_threat_prob": _prob01(dc.get("oppGwangThreatProb")),
        "go_stop_delta_proxy": _signed_proxy(dc.get("goStopDeltaProxy")),
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

def go_choice_from_trace(trace):
    dt = str(trace.get("dt") or "").strip()
    if dt != "option":
        return None
    chosen = action_alias(trace.get("ch") if trace.get("ch") is not None else trace.get("at"))
    if chosen in ("go", "stop"):
        return chosen
    return None


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
    games = 0
    avg_gold_delta_sum = 0.0
    avg_gold_delta_sq_sum = 0.0
    go_declared_total = 0
    go_failed_total = 0
    go_declared_by_side = {SIDE_MY: 0, SIDE_YOUR: 0}
    go_failed_by_side = {SIDE_MY: 0, SIDE_YOUR: 0}

    for path in input_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line_raw in f:
                line_raw = line_raw.strip()
                if not line_raw:
                    continue
                line = json.loads(line_raw)
                decision_trace = line.get("decision_trace") or []
                games += 1
                gold_delta_my = _float_or_none(line.get("goldDeltaMy"))
                if gold_delta_my is None:
                    final_my = _float_or_none(line.get("finalGoldMy"))
                    init_my = _float_or_none(line.get("initialGoldMy"))
                    if final_my is not None and init_my is not None:
                        gold_delta_my = final_my - init_my
                if gold_delta_my is not None:
                    avg_gold_delta_sum += gold_delta_my
                    avg_gold_delta_sq_sum += gold_delta_my * gold_delta_my
                winner = str(line.get("winner") or "unknown")
                for trace in decision_trace:
                    go_choice = go_choice_from_trace(trace)
                    if go_choice != "go":
                        continue
                    actor = str(trace.get("a") or "")
                    if actor not in (SIDE_MY, SIDE_YOUR):
                        continue
                    go_declared_total += 1
                    go_declared_by_side[actor] += 1
                    lost = (winner == SIDE_YOUR) if actor == SIDE_MY else (winner == SIDE_MY)
                    if lost:
                        go_failed_total += 1
                        go_failed_by_side[actor] += 1

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
                        target_mode = value_new.get("target_mode", "score")
                        vs = value_sample(trace, dt, chosen, line, gold_per_point, target_mode)
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

    gold_delta_avg = avg_gold_delta_sum / max(1, games)
    if games > 1:
        var = (avg_gold_delta_sq_sum - ((avg_gold_delta_sum * avg_gold_delta_sum) / games)) / max(1, games - 1)
        var = max(0.0, var)
        gold_delta_std = math.sqrt(var)
        gold_delta_se = gold_delta_std / math.sqrt(games)
    else:
        gold_delta_std = 0.0
        gold_delta_se = 0.0
    gold_delta_ci95_low = gold_delta_avg - 1.96 * gold_delta_se
    gold_delta_ci95_high = gold_delta_avg + 1.96 * gold_delta_se
    go_fail_rate_total = go_failed_total / max(1, go_declared_total)
    go_fail_rate_my = go_failed_by_side[SIDE_MY] / max(1, go_declared_by_side[SIDE_MY])
    go_fail_rate_your = go_failed_by_side[SIDE_YOUR] / max(1, go_declared_by_side[SIDE_YOUR])

    gold_gate = gold_delta_ci95_low > 0
    risk_gate = go_fail_rate_total <= 0.08
    value_gate = True
    if value_old and value_new and v_total > 0:
        value_gate = (v_new_abs / max(1, v_total)) <= (v_old_abs / max(1, v_total))
    promotion_recommended = bool(gold_gate and risk_gate and value_gate)

    report = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "inputs": input_paths,
        "north_star": {
            "games": games,
            "avg_gold_delta_my": gold_delta_avg,
            "gold_delta_std": gold_delta_std,
            "gold_delta_se": gold_delta_se,
            "gold_delta_ci95_low": gold_delta_ci95_low,
            "gold_delta_ci95_high": gold_delta_ci95_high,
            "go_declared_total": go_declared_total,
            "go_failed_total": go_failed_total,
            "go_fail_rate_total": go_fail_rate_total,
            "go_fail_rate_by_side": {
                SIDE_MY: go_fail_rate_my,
                SIDE_YOUR: go_fail_rate_your,
            },
            "gates": {
                "gold_significant_positive": gold_gate,
                "go_fail_rate_safe": risk_gate,
                "value_not_worse": value_gate,
            },
            "promotion_recommended": promotion_recommended,
        },
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
