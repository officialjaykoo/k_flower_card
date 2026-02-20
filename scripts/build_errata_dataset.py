#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
from datetime import datetime, timezone

SIDE_MY = "mySide"
SIDE_YOUR = "yourSide"

DEFAULT_POLICY_FILTER_RULES = {
    "min_candidate_count": 2,
}

DEFAULT_VALUE_FILTER_RULES = {
    "min_candidate_count": 2,
    "keep_if_action_type_in": ["declare_bomb"],
    "keep_if_trigger_prefixes": ["specialEvent", "bomb", "riskShift"],
    "keep_if_trigger_names": ["comboThreatEnter", "terminalContext", "goStopOption", "shakingYesOption"],
}
TRIGGER_BIT = {
    "earlyTurnForced": 1 << 0,
    "goStopOption": 1 << 1,
    "shakingYesOption": 1 << 2,
    "optionTurnOther": 1 << 3,
    "deckEmpty": 1 << 4,
    "specialEvent": 1 << 5,
    "bombEvent": 1 << 6,
    "riskShift": 1 << 7,
    "comboThreatEnter": 1 << 8,
    "terminalContext": 1 << 9,
}


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


def _to_int_or_none(value):
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _as_str_list(value):
    if not isinstance(value, list):
        return []
    out = []
    for v in value:
        s = str(v or "").strip()
        if s:
            out.append(s)
    return out


def normalize_policy_filter_rules(raw):
    rules = dict(DEFAULT_POLICY_FILTER_RULES)
    if isinstance(raw, dict):
        min_cc = _to_int_or_none(raw.get("min_candidate_count"))
        if min_cc is not None:
            rules["min_candidate_count"] = max(1, min_cc)
    return rules


def normalize_value_filter_rules(raw):
    rules = dict(DEFAULT_VALUE_FILTER_RULES)
    if isinstance(raw, dict):
        min_cc = _to_int_or_none(raw.get("min_candidate_count"))
        if min_cc is not None:
            rules["min_candidate_count"] = max(1, min_cc)
        actions = _as_str_list(raw.get("keep_if_action_type_in"))
        if actions:
            rules["keep_if_action_type_in"] = actions
        prefixes = _as_str_list(raw.get("keep_if_trigger_prefixes"))
        if prefixes:
            rules["keep_if_trigger_prefixes"] = prefixes
        names = _as_str_list(raw.get("keep_if_trigger_names"))
        if names:
            rules["keep_if_trigger_names"] = names
    return rules


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_policy_filter_rules(path):
    p = str(path or "").strip()
    if not p or not os.path.exists(p):
        return dict(DEFAULT_POLICY_FILTER_RULES), None
    return normalize_policy_filter_rules(load_json(p)), os.path.abspath(p)


def load_value_filter_rules(path):
    p = str(path or "").strip()
    if not p or not os.path.exists(p):
        return dict(DEFAULT_VALUE_FILTER_RULES), None
    return normalize_value_filter_rules(load_json(p)), os.path.abspath(p)


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


def trigger_names(trace):
    tg = trace.get("tg")
    if isinstance(tg, list):
        return [str(x or "") for x in tg if str(x or "").strip()]
    tm = trace.get("tm")
    try:
        mask = int(tm or 0)
    except Exception:
        mask = 0
    if mask <= 0:
        return []
    out = []
    for name, bit in TRIGGER_BIT.items():
        if mask & int(bit):
            out.append(name)
    return out


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
    hand_self = int(dc.get("handCountSelf") or 0)
    hand_diff = int(dc.get("handCountDiff") or 0) if dc.get("handCountDiff") is not None else hand_self - int(dc.get("handCountOpp") or 0)
    go_self = dc.get("goCountSelf", 0)
    go_opp = dc.get("goCountOpp", 0)
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
        cands = len(sp.get("cards") or sp.get("boardCardIds") or sp.get("options") or [])
    return "|".join(
        [
            f"dt={decision_type}",
            f"ph={phase}",
            f"o={trace_order(trace)}",
            f"db={deck_bucket}",
            f"hs={hand_self}",
            f"hd={hand_diff}",
            f"gs={go_self}",
            f"go={go_opp}",
            f"ss={shake_self}",
            f"so={shake_opp}",
            f"cc={cands}",
        ]
    )


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


def stable_hash(token, dim):
    import hashlib

    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % dim


def _float_or_none(v):
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _dc_hand_opp(dc):
    if (dc or {}).get("handCountOpp") is not None:
        return int((dc or {}).get("handCountOpp") or 0)
    hand_self = int((dc or {}).get("handCountSelf") or 0)
    hand_diff = int((dc or {}).get("handCountDiff") or 0)
    return hand_self - hand_diff


def _dc_score_diff(dc):
    if (dc or {}).get("scoreDiff") is not None:
        return float((dc or {}).get("scoreDiff") or 0)
    return float((dc or {}).get("currentScoreSelf") or 0) - float((dc or {}).get("currentScoreOpp") or 0)


def target_gold(line, actor, gold_per_point, target_mode="gold"):
    mode = str(target_mode or "gold").strip().lower()

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
    point_diff = float(self_score) - float(opp_score)
    return point_diff * float(gold_per_point)


def choose_label(trace):
    dt = trace.get("dt")
    compact = trace.get("ch")
    if dt == "play":
        return (compact if compact is not None else trace.get("c")), "play"
    if dt == "match":
        return (compact if compact is not None else trace.get("s")), "match"
    if dt == "option":
        return action_alias(compact if compact is not None else trace.get("at")), "option"

    sp = trace.get("sp") or {}
    if sp.get("cards"):
        return (compact if compact is not None else trace.get("c")), "play"
    if sp.get("boardCardIds"):
        return (compact if compact is not None else trace.get("s")), "match"
    if sp.get("options"):
        return action_alias(compact if compact is not None else trace.get("at")), "option"
    return None, None


def value_sample(trace, decision_type, chosen, line, gold_per_point, target_mode):
    actor = trace.get("a")
    if actor not in (SIDE_MY, SIDE_YOUR):
        return None
    dc = trace.get("dc") or {}
    y = target_gold(line, actor, gold_per_point, target_mode)
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
    score_diff = _dc_score_diff(dc)
    hand_opp = _dc_hand_opp(dc)
    tokens = [
        f"phase={dc.get('phase', '?')}",
        f"order={trace_order(trace)}",
        f"decision_type={decision_type}",
        f"action={chosen or '?'}",
        f"deck_bucket={int((dc.get('deckCount') or 0)//3)}",
        f"self_hand={int(dc.get('handCountSelf') or 0)}",
        f"opp_hand={int(hand_opp)}",
        f"self_go={int(dc.get('goCountSelf') or 0)}",
        f"opp_go={int(dc.get('goCountOpp') or 0)}",
        f"is_first_attacker={1 if trace_order(trace) == 'first' else 0}",
        f"score_diff={int(score_diff)}",
        f"bak_risk={int(bak_risk_total)}",
        f"jp_self_sum={int(dc.get('jokboProgressSelfSum') or 0)}",
        f"jp_opp_sum={int(dc.get('jokboProgressOppSum') or 0)}",
        f"go_stop_delta_bucket={int(_signed_proxy(dc.get('goStopDeltaProxy')) * 2)}",
    ]
    numeric = {
        "deck_count": float(dc.get("deckCount") or 0),
        "hand_self": float(dc.get("handCountSelf") or 0),
        "hand_opp": float(hand_opp),
        "go_self": float(dc.get("goCountSelf") or 0),
        "go_opp": float(dc.get("goCountOpp") or 0),
        "score_diff": score_diff,
        "is_first_attacker": float(1 if trace_order(trace) == "first" else 0),
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
        "cand_count": float(trace.get("cc") or 0),
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


def policy_trace_pass(trace, rules):
    min_cc = int((rules or {}).get("min_candidate_count") or 1)
    cc = _to_int_or_none((trace or {}).get("cc"))
    if cc is None:
        return True
    return cc >= min_cc


def value_exception_trace(trace, rules):
    action_set = set(_as_str_list((rules or {}).get("keep_if_action_type_in")))
    prefix_list = _as_str_list((rules or {}).get("keep_if_trigger_prefixes"))
    name_set = set(_as_str_list((rules or {}).get("keep_if_trigger_names")))

    action_type = str(trace.get("at") or "")
    if not action_type:
        chosen = action_alias(trace.get("ch"))
        if chosen in ("go", "stop"):
            action_type = f"choose_{chosen}"
    if action_type in action_set:
        return True
    for raw in trigger_names(trace):
        name = str(raw or "")
        if not name:
            continue
        if name in name_set:
            return True
        if any(name.startswith(prefix) for prefix in prefix_list):
            return True
    return False


def value_trace_pass(trace, rules):
    min_cc = int((rules or {}).get("min_candidate_count") or 1)
    cc = _to_int_or_none((trace or {}).get("cc"))
    if cc is None:
        return True
    if cc >= min_cc:
        return True
    return value_exception_trace(trace, rules)


def percentile(values, q):
    if not values:
        return None
    vals = sorted(values)
    if q <= 0:
        return vals[0]
    if q >= 1:
        return vals[-1]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def compact_game_line(game, traces):
    keys = [
        "ver",
        "run",
        "game",
        "seed",
        "winner",
        "score",
        "firstAttackerActor",
        "initialGoldMy",
        "initialGoldYour",
        "finalGoldMy",
        "finalGoldYour",
        "goldDeltaMy",
        "policy",
    ]
    out = {}
    for k in keys:
        if k in game:
            out[k] = game[k]
    out["decision_trace"] = traces
    return out


def compute_thresholds(
    input_paths,
    policy_model,
    value_model,
    policy_rules,
    value_rules,
    policy_loss_quantile,
    value_error_min,
    value_error_quantile,
):
    policy_losses = []
    value_errors = []

    gold_per_point = value_model.get("gold_per_point", 100.0)
    target_mode = value_model.get("target_mode", "score")

    for path in input_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line_raw in f:
                line_raw = line_raw.strip()
                if not line_raw:
                    continue
                game = json.loads(line_raw)
                for trace in game.get("decision_trace") or []:
                    if policy_trace_pass(trace, policy_rules):
                        s = extract_policy_sample(trace)
                        if s is not None:
                            dt, candidates, chosen = s
                            ck = policy_context_key(trace, dt)
                            p = max(1e-12, policy_prob_raw(policy_model, dt, candidates, ck, chosen))
                            policy_losses.append(-math.log(p))

                    if value_trace_pass(trace, value_rules):
                        chosen, dt = choose_label(trace)
                        if chosen is None:
                            continue
                        vs = value_sample(trace, dt, chosen, game, gold_per_point, target_mode)
                        if not vs:
                            continue
                        pred = value_predict(value_model, vs)
                        value_errors.append(abs(pred - float(vs["y"])))

    if not policy_losses:
        raise RuntimeError("No policy losses computed; cannot build errata.")
    if not value_errors:
        raise RuntimeError("No value errors computed; cannot build errata.")

    policy_thr = percentile(policy_losses, policy_loss_quantile)
    value_thr = float(value_error_min)
    if value_error_quantile is not None:
        qv = percentile(value_errors, value_error_quantile)
        if qv is not None:
            value_thr = max(value_thr, float(qv))

    return {
        "policy_threshold": float(policy_thr),
        "value_threshold": float(value_thr),
        "policy_loss_count": int(len(policy_losses)),
        "value_error_count": int(len(value_errors)),
    }


def build_errata_files(
    input_paths,
    policy_model,
    value_model,
    policy_rules,
    value_rules,
    policy_threshold,
    value_threshold,
    output_policy,
    output_policy_os,
    output_value,
    output_value_os,
    policy_repeat,
    value_repeat,
):
    os.makedirs(os.path.dirname(output_policy) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_policy_os) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_value) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_value_os) or ".", exist_ok=True)

    policy_repeat = max(1, int(policy_repeat))
    value_repeat = max(1, int(value_repeat))

    gold_per_point = value_model.get("gold_per_point", 100.0)
    target_mode = value_model.get("target_mode", "score")

    stats = {
        "games_seen": 0,
        "policy_errata_games": 0,
        "value_errata_games": 0,
        "policy_errata_traces": 0,
        "value_errata_traces": 0,
        "policy_errata_lines_oversampled": 0,
        "value_errata_lines_oversampled": 0,
    }

    with open(output_policy, "w", encoding="utf-8") as f_pol, open(
        output_policy_os, "w", encoding="utf-8"
    ) as f_pol_os, open(output_value, "w", encoding="utf-8") as f_val, open(
        output_value_os, "w", encoding="utf-8"
    ) as f_val_os:
        for path in input_paths:
            with open(path, "r", encoding="utf-8") as f:
                for line_raw in f:
                    line_raw = line_raw.strip()
                    if not line_raw:
                        continue
                    game = json.loads(line_raw)
                    stats["games_seen"] += 1

                    policy_hit = []
                    value_hit = []

                    for trace in game.get("decision_trace") or []:
                        if policy_trace_pass(trace, policy_rules):
                            s = extract_policy_sample(trace)
                            if s is not None:
                                dt, candidates, chosen = s
                                ck = policy_context_key(trace, dt)
                                p = max(1e-12, policy_prob_raw(policy_model, dt, candidates, ck, chosen))
                                loss = -math.log(p)
                                if loss >= policy_threshold:
                                    policy_hit.append(trace)

                        if value_trace_pass(trace, value_rules):
                            chosen, dt = choose_label(trace)
                            if chosen is not None:
                                vs = value_sample(trace, dt, chosen, game, gold_per_point, target_mode)
                                if vs:
                                    pred = value_predict(value_model, vs)
                                    err = abs(pred - float(vs["y"]))
                                    if err >= value_threshold:
                                        value_hit.append(trace)

                    if policy_hit:
                        stats["policy_errata_games"] += 1
                        stats["policy_errata_traces"] += len(policy_hit)
                        line = compact_game_line(game, policy_hit)
                        blob = json.dumps(line, ensure_ascii=False, separators=(",", ":"))
                        f_pol.write(blob + "\n")
                        for _ in range(policy_repeat):
                            f_pol_os.write(blob + "\n")
                            stats["policy_errata_lines_oversampled"] += 1

                    if value_hit:
                        stats["value_errata_games"] += 1
                        stats["value_errata_traces"] += len(value_hit)
                        line = compact_game_line(game, value_hit)
                        blob = json.dumps(line, ensure_ascii=False, separators=(",", ":"))
                        f_val.write(blob + "\n")
                        for _ in range(value_repeat):
                            f_val_os.write(blob + "\n")
                            stats["value_errata_lines_oversampled"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Build errata datasets from selfplay logs and trained models.")
    parser.add_argument("--input", nargs="+", required=True, help="Input JSONL path(s) or glob(s).")
    parser.add_argument("--policy-model", required=True, help="Policy model JSON path.")
    parser.add_argument("--value-model", required=True, help="Value model JSON path.")
    parser.add_argument("--policy-filter-rules", default="configs/policy_filter_rules.json")
    parser.add_argument("--value-filter-rules", default="configs/value_filter_rules.json")

    parser.add_argument("--policy-loss-quantile", type=float, default=0.95, help="Top-loss quantile for policy errata.")
    parser.add_argument(
        "--value-error-min",
        type=float,
        default=10000.0,
        help="Minimum absolute value error threshold.",
    )
    parser.add_argument(
        "--value-error-quantile",
        type=float,
        default=None,
        help="Optional quantile threshold for value error; final threshold=max(min, quantile).",
    )

    parser.add_argument("--policy-repeat", type=int, default=3, help="Oversample repeat for policy errata lines.")
    parser.add_argument("--value-repeat", type=int, default=2, help="Oversample repeat for value errata lines.")

    parser.add_argument("--output-policy", default="logs/errata_policy_v3.jsonl")
    parser.add_argument("--output-policy-oversampled", default="logs/errata_policy_v3_os.jsonl")
    parser.add_argument("--output-value", default="logs/errata_value_v3.jsonl")
    parser.add_argument("--output-value-oversampled", default="logs/errata_value_v3_os.jsonl")
    parser.add_argument("--output-summary", default="logs/errata_v3_summary.json")
    args = parser.parse_args()

    if not (0.0 <= float(args.policy_loss_quantile) <= 1.0):
        raise RuntimeError("--policy-loss-quantile must be in [0,1].")
    if args.value_error_quantile is not None and not (0.0 <= float(args.value_error_quantile) <= 1.0):
        raise RuntimeError("--value-error-quantile must be in [0,1].")

    input_paths = expand_inputs(args.input)
    policy_model = load_json(args.policy_model)
    value_model = load_json(args.value_model)
    policy_rules, policy_rules_path = load_policy_filter_rules(args.policy_filter_rules)
    value_rules, value_rules_path = load_value_filter_rules(args.value_filter_rules)

    print("phase=threshold_scan")
    thresholds = compute_thresholds(
        input_paths,
        policy_model,
        value_model,
        policy_rules,
        value_rules,
        float(args.policy_loss_quantile),
        float(args.value_error_min),
        args.value_error_quantile,
    )
    print(
        json.dumps(
            {
                "policy_threshold": thresholds["policy_threshold"],
                "value_threshold": thresholds["value_threshold"],
                "policy_loss_count": thresholds["policy_loss_count"],
                "value_error_count": thresholds["value_error_count"],
            },
            ensure_ascii=False,
        )
    )

    print("phase=build_errata")
    stats = build_errata_files(
        input_paths,
        policy_model,
        value_model,
        policy_rules,
        value_rules,
        thresholds["policy_threshold"],
        thresholds["value_threshold"],
        args.output_policy,
        args.output_policy_oversampled,
        args.output_value,
        args.output_value_oversampled,
        int(args.policy_repeat),
        int(args.value_repeat),
    )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": input_paths,
        "policy_model": os.path.abspath(args.policy_model),
        "value_model": os.path.abspath(args.value_model),
        "policy_filter_rules_path": policy_rules_path,
        "value_filter_rules_path": value_rules_path,
        "policy_filter_rules": policy_rules,
        "value_filter_rules": value_rules,
        "policy_loss_quantile": float(args.policy_loss_quantile),
        "value_error_min": float(args.value_error_min),
        "value_error_quantile": args.value_error_quantile,
        "policy_threshold": thresholds["policy_threshold"],
        "value_threshold": thresholds["value_threshold"],
        "policy_repeat": int(args.policy_repeat),
        "value_repeat": int(args.value_repeat),
        "outputs": {
            "policy": args.output_policy,
            "policy_oversampled": args.output_policy_oversampled,
            "value": args.output_value,
            "value_oversampled": args.output_value_oversampled,
            "summary": args.output_summary,
        },
        "stats": stats,
    }
    os.makedirs(os.path.dirname(args.output_summary) or ".", exist_ok=True)
    with open(args.output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"wrote: {args.output_policy}")
    print(f"wrote: {args.output_policy_oversampled}")
    print(f"wrote: {args.output_value}")
    print(f"wrote: {args.output_value_oversampled}")
    print(f"wrote: {args.output_summary}")
    print(json.dumps(summary["stats"], ensure_ascii=False))


if __name__ == "__main__":
    main()
