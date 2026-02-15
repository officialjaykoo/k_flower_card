#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
import random
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
GO_STOP_ACTIONS = {"go": 0, "stop": 1}
NUMERIC_KEYS = [
    "opp_danger_level",
    "my_win_progress",
    "deck_end_ratio",
    "high_value_match",
    "jokbo_potential",
    "can_match",
    "immediate_reward",
    "pi_bak_risk",
    "gwang_bak_risk",
    "mong_bak_risk",
    "shake_multiplier_state",
    "gold_self",
    "gold_opp",
    "hand_self",
    "hand_opp",
]

# Feature-priority weights for score-critical states.
# Applied after per-feature normalization so importance differences remain.
NUMERIC_FEATURE_WEIGHTS = {
    "opp_danger_level": 2.2,
    "my_win_progress": 2.0,
    "deck_end_ratio": 1.2,
    "high_value_match": 1.8,
    "jokbo_potential": 1.5,
    "can_match": 1.6,
    "immediate_reward": 1.3,
    "pi_bak_risk": 2.0,
    "gwang_bak_risk": 2.2,
    "mong_bak_risk": 2.0,
    "shake_multiplier_state": 2.0,
}

_CATALOG_CACHE = {}
SIDE_MY = "mySide"
SIDE_YOUR = "yourSide"


def _opp_side(actor):
    return SIDE_YOUR if actor == SIDE_MY else SIDE_MY


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


def choose_label_and_candidates(trace):
    sp = trace.get("sp") or {}
    cards = sp.get("cards")
    if cards:
        chosen = trace.get("c")
        if chosen in cards:
            return str(chosen), "play", [str(x) for x in cards]
        return None, None, None
    board = sp.get("boardCardIds")
    if board:
        chosen = trace.get("s")
        if chosen in board:
            return str(chosen), "match", [str(x) for x in board]
        return None, None, None
    options = sp.get("options")
    if options:
        chosen = action_alias(trace.get("at"))
        cands = [str(x) for x in options]
        if chosen in cands:
            return str(chosen), "option", cands
        return None, None, None
    return None, None, None


def _nested_get(root, path):
    cur = root
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
        if cur is None:
            return None
    return cur


def _to_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)


def _extract_actor_stat(line, actor, names):
    opp = _opp_side(actor)
    roots = [
        line.get("result"),
        line.get("summary"),
        line.get("stats"),
        line.get("captured"),
    ]
    key_groups = [
        [actor],
        [f"{actor}_stats"],
        [f"{actor}Stats"],
        [f"{actor}_capture"],
        [f"{actor}Capture"],
        [opp],  # fallback if naming is inverted
    ]
    for root in roots:
        if not isinstance(root, dict):
            continue
        for keys in key_groups:
            for n in names:
                v = _nested_get(root, keys + [n])
                if v is not None:
                    return _to_float(v, 0.0)
        for n in names:
            v = root.get(n)
            if v is not None:
                return _to_float(v, 0.0)
    return 0.0


def _extract_first_number(line, names, default=0.0):
    roots = [line, line.get("result"), line.get("summary"), line.get("state"), line.get("initial")]
    for root in roots:
        if not isinstance(root, dict):
            continue
        for n in names:
            if n in root:
                return _to_float(root.get(n), default)
    return float(default)


def _load_catalog(path):
    if not path:
        return None
    if path in _CATALOG_CACHE:
        return _CATALOG_CACHE[path]
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            _CATALOG_CACHE[path] = data
            return data
    except Exception:
        pass
    _CATALOG_CACHE[path] = None
    return None


def _card_id_from_item(item):
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        cid = item.get("id")
        if isinstance(cid, str):
            return cid
    return None


def _extract_card_ids(items):
    out = []
    if not isinstance(items, list):
        return out
    for it in items:
        cid = _card_id_from_item(it)
        if cid:
            out.append(cid)
    return out


def _month_from_card_id(card_id):
    if not isinstance(card_id, str):
        return None

    # Legacy runtime IDs: "<month>-<globalIndex>" (e.g. "9-32")
    head = card_id.split("-", 1)[0]
    try:
        return int(head)
    except Exception:
        pass

    # New stable IDs: A0..L3 (months 1..12), M0..M1 (bonus month 13)
    if len(card_id) < 2:
        return None
    prefix = card_id[0].upper()
    tail = card_id[1:]
    if not tail.isdigit():
        return None
    idx = int(tail)

    if "A" <= prefix <= "L" and 0 <= idx <= 3:
        return ord(prefix) - ord("A") + 1
    if prefix == "M" and 0 <= idx <= 1:
        return 13
    return None


def _card_meta(line, card_id):
    meta = None
    catalog = _load_catalog(line.get("catalogPath"))
    if isinstance(catalog, dict):
        cand = catalog.get(card_id)
        if isinstance(cand, dict):
            meta = cand
    month = None
    category = None
    if isinstance(meta, dict):
        try:
            month = int(meta.get("month")) if meta.get("month") is not None else None
        except Exception:
            month = None
        cat = meta.get("category")
        if isinstance(cat, str):
            category = cat
    if month is None:
        month = _month_from_card_id(card_id)
    return month, category


def _category_alias(cat):
    if not isinstance(cat, str):
        return None
    c = cat.strip().lower()
    if c == "kwang":
        return "gwang"
    if c == "five":
        return "yeol"
    if c == "ribbon":
        return "ttie"
    if c == "junk":
        return "pee"
    return c


def extract_numeric(line, trace, actor):
    dc = trace.get("dc") or {}
    sp = trace.get("sp") or {}
    fv = trace.get("fv") or {}
    ev = fv.get("eventCounts") or {}
    opp = _opp_side(actor)

    shake_events = _to_float(ev.get("shaking"), 0.0)
    deck_count = float(dc.get("deckCount") or 0.0)
    init_deck_count = _extract_first_number(line, ["deckCount", "boardDeckCount"], default=22.0)
    if init_deck_count <= 0:
        init_deck_count = 22.0
    deck_end_ratio = max(0.0, min(1.0, 1.0 - (deck_count / init_deck_count)))

    hand_self_raw = float(dc.get("handCountSelf") or 0.0)
    hand_opp_raw = float(dc.get("handCountOpp") or 0.0)
    hand_self = max(0.0, min(1.0, hand_self_raw / 10.0))
    hand_opp = max(0.0, min(1.0, hand_opp_raw / 10.0))

    hand_ids = _extract_card_ids((dc.get("handCards") or [])) or _extract_card_ids(sp.get("cards") or [])
    board_ids = _extract_card_ids(dc.get("boardCards") or []) or _extract_card_ids(sp.get("boardCards") or [])
    if not board_ids and isinstance(sp.get("boardCardIds"), list):
        board_ids = _extract_card_ids(sp.get("boardCardIds"))
    hand_months = {m for m in (_month_from_card_id(c) for c in hand_ids) if m is not None}
    board_months = {m for m in (_month_from_card_id(c) for c in board_ids) if m is not None}
    can_match = 1.0 if hand_months and board_months and (hand_months & board_months) else 0.0

    hand_by_month = {}
    board_by_month = {}
    for cid in hand_ids:
        m, cat = _card_meta(line, cid)
        if m is None:
            continue
        hand_by_month.setdefault(m, set()).add(cat or "?")
    for cid in board_ids:
        m, cat = _card_meta(line, cid)
        if m is None:
            continue
        board_by_month.setdefault(m, set()).add(cat or "?")
    high_value_months = hand_months & board_months
    high_value_match = 0.0
    godori_months = {2, 4, 8}
    for m in high_value_months:
        hand_cats = hand_by_month.get(m, set())
        board_cats = board_by_month.get(m, set())
        if ("kwang" in hand_cats or "kwang" in board_cats or "five" in hand_cats or "five" in board_cats or m in godori_months):
            high_value_match = 1.0
            break

    pi_self = _extract_actor_stat(line, actor, ["pi", "piCount", "junk", "junkCount"])
    gwang_self = _extract_actor_stat(line, actor, ["gwang", "gwangCount", "bright", "brightCount"])
    ribbon_self = _extract_actor_stat(line, actor, ["dan", "danCount", "tti", "ribbon", "ribbonCount"])
    hongdan_self = _extract_actor_stat(line, actor, ["hongdan", "hongDan", "redDan"])
    cheongdan_self = _extract_actor_stat(line, actor, ["cheongdan", "cheongDan", "blueDan"])
    chodan_self = _extract_actor_stat(line, actor, ["chodan", "choDan"])
    godori_self = _extract_actor_stat(line, actor, ["godori", "goDori"])

    go_self = float(dc.get("goCountSelf") or 0.0)
    go_opp = float(dc.get("goCountOpp") or 0.0)
    pi_opp = _extract_actor_stat(line, opp, ["pi", "piCount", "junk", "junkCount"])
    gwang_opp = _extract_actor_stat(line, opp, ["gwang", "gwangCount", "bright", "brightCount"])

    score_self_now = _to_float(dc.get("currentScoreSelf"), 0.0)
    score_opp_now = _to_float(dc.get("currentScoreOpp"), 0.0)
    opp_danger_level = (
        0.35 * min(1.0, go_opp / 3.0)
        + 0.25 * min(1.0, max(0.0, 1.0 - hand_opp))
        + 0.20 * min(1.0, pi_opp / 10.0)
        + 0.20 * min(1.0, gwang_opp / 3.0)
    )
    opp_danger_level = max(opp_danger_level, min(1.0, score_opp_now / 7.0))
    opp_danger_level = max(0.0, min(1.0, opp_danger_level))

    my_win_progress = (
        0.30 * min(1.0, go_self / 3.0)
        + 0.25 * min(1.0, pi_self / 10.0)
        + 0.20 * min(1.0, gwang_self / 3.0)
        + 0.15 * can_match
        + 0.10 * min(1.0, max(0.0, 1.0 - hand_self))
    )
    my_win_progress = max(my_win_progress, min(1.0, score_self_now / 7.0))
    my_win_progress = max(0.0, min(1.0, my_win_progress))
    jp = trace.get("jp") or {}
    if int(jp.get("completed_now") or 0) == 1:
        jokbo_potential = 1.0
    else:
        jokbo_potential = max(
            min(1.0, max(hongdan_self, cheongdan_self, chodan_self) / 3.0),
            min(1.0, godori_self / 5.0),
            min(1.0, ribbon_self / 7.0),
        )
    gold_self_raw = float(dc.get("goldSelf") or 0.0)
    gold_opp_raw = float(dc.get("goldOpp") or 0.0)
    gold_self = max(0.0, min(2.0, math.log1p(max(0.0, gold_self_raw)) / 10.0))
    gold_opp = max(0.0, min(2.0, math.log1p(max(0.0, gold_opp_raw)) / 10.0))

    return {
        "opp_danger_level": opp_danger_level,
        "my_win_progress": my_win_progress,
        "deck_end_ratio": deck_end_ratio,
        "high_value_match": high_value_match,
        "jokbo_potential": jokbo_potential,
        "can_match": can_match,
        "immediate_reward": float(trace.get("ir") or 0.0),
        "pi_bak_risk": _to_float(dc.get("piBakRisk"), _extract_actor_stat(line, actor, ["piBakRisk", "pibakRisk"])),
        "gwang_bak_risk": _to_float(dc.get("gwangBakRisk"), _extract_actor_stat(line, actor, ["gwangBakRisk"])),
        "mong_bak_risk": _to_float(dc.get("mongBakRisk"), _extract_actor_stat(line, actor, ["mongBakRisk", "meongBakRisk"])),
        "shake_multiplier_state": max(0.0, shake_events),
        "gold_self": gold_self,
        "gold_opp": gold_opp,
        "hand_self": hand_self,
        "hand_opp": hand_opp,
    }


def extract_tokens(line, trace, decision_type, actor):
    dc = trace.get("dc") or {}
    score = line.get("score") or {}
    opp = _opp_side(actor)
    self_score = float(score.get(actor) or 0.0)
    opp_score = float(score.get(opp) or 0.0)
    score_diff = self_score - opp_score
    bak_escape = line.get("bakEscape") or {}
    loser = bak_escape.get("loser")
    escaped = bak_escape.get("escaped")
    actor_bak = _extract_actor_stat(line, actor, ["piBak", "pi", "gwangBak", "gwang", "mongBak"])
    tokens = [
        f"phase={dc.get('phase','?')}",
        f"order={trace.get('o','?')}",
        f"decision_type={decision_type}",
        f"deck_bucket={int((dc.get('deckCount') or 0)//3)}",
        f"self_hand={int(dc.get('handCountSelf') or 0)}",
        f"opp_hand={int(dc.get('handCountOpp') or 0)}",
        f"self_go={int(dc.get('goCountSelf') or 0)}",
        f"opp_go={int(dc.get('goCountOpp') or 0)}",
        f"actor={actor}",
        f"self_score_b={int(self_score // 2)}",
        f"opp_score_b={int(opp_score // 2)}",
        f"score_diff_b={int(score_diff // 2)}",
        f"is_nagari={1 if bool(line.get('nagari', False)) else 0}",
        f"is_bak_loser={1 if (loser == actor and escaped is False) else 0}",
        f"bak_signal={int(actor_bak)}",
    ]
    hand_ids = _extract_card_ids((dc.get("handCards") or [])) or _extract_card_ids((trace.get("sp") or {}).get("cards") or [])
    board_ids = _extract_card_ids((dc.get("boardCards") or [])) or _extract_card_ids((trace.get("sp") or {}).get("boardCards") or [])
    if not board_ids and isinstance((trace.get("sp") or {}).get("boardCardIds"), list):
        board_ids = _extract_card_ids((trace.get("sp") or {}).get("boardCardIds"))
    hand_info = []
    board_info = []
    for cid in hand_ids:
        m, cat = _card_meta(line, cid)
        hand_info.append((m, cat))
        if m is not None:
            tokens.append(f"hand_month={m}")
        if cat:
            tokens.append(f"hand_cat={cat}")
    for cid in board_ids:
        m, cat = _card_meta(line, cid)
        board_info.append((m, cat))
        if m is not None:
            tokens.append(f"board_month={m}")
        if cat:
            tokens.append(f"board_cat={cat}")
    cap = trace.get("cap") or {}
    for cid in _extract_card_ids(cap.get("hand") or []) + _extract_card_ids(cap.get("flip") or []):
        m, cat = _card_meta(line, cid)
        if m is not None:
            tokens.append(f"cap_self_month={m}")
        if cat:
            tokens.append(f"cap_self_cat={cat}")
    hand_months = {m for m in (_month_from_card_id(c) for c in hand_ids) if m is not None}
    board_months = {m for m in (_month_from_card_id(c) for c in board_ids) if m is not None}
    tokens.append(f"can_match={1 if hand_months and board_months and (hand_months & board_months) else 0}")

    # Semantic tokens for better generalization than raw card IDs.
    for m in sorted({m for m, _ in hand_info if m is not None}):
        tokens.append(f"hand_m{m}")
    for m in sorted({m for m, _ in board_info if m is not None}):
        tokens.append(f"board_m{m}")
    for cat_alias in sorted({_category_alias(cat) for _, cat in hand_info if _category_alias(cat)}):
        tokens.append(f"hand_{cat_alias}")
    for cat_alias in sorted({_category_alias(cat) for _, cat in board_info if _category_alias(cat)}):
        tokens.append(f"board_{cat_alias}")

    matchable_months = hand_months & board_months
    if matchable_months:
        tokens.append("can_match_any")
    else:
        tokens.append("can_match_none")
    matchable_board_aliases = set()
    for m, cat in board_info:
        if m in matchable_months:
            ca = _category_alias(cat)
            if ca:
                matchable_board_aliases.add(ca)
    for ca in sorted(matchable_board_aliases):
        tokens.append(f"can_match_{ca}")
    return tokens


def target_value(line, actor, value_scale):
    score = line.get("score") or {}
    self_score = score.get(actor)
    opp = _opp_side(actor)
    opp_score = score.get(opp)
    if self_score is None or opp_score is None:
        return None
    diff = float(self_score) - float(opp_score)

    # Reward shaping with available log fields (compact/train compatible).
    mult = 1.0
    if bool(line.get("nagari", False)):
        mult *= 0.85

    bak_escape = line.get("bakEscape") or {}
    loser = bak_escape.get("loser")
    escaped = bak_escape.get("escaped")
    if loser in (SIDE_MY, SIDE_YOUR) and escaped is False:
        mult *= 1.2

    go_decision = line.get("goDecision") or {}
    if int(go_decision.get("declared") or 0) > 0:
        mult *= 1.05
        if int(go_decision.get("success") or 0) > 0:
            mult *= 1.05

    ef = line.get("eventFrequency") or {}
    special_count = (
        int(ef.get("ppuk") or 0)
        + int(ef.get("ddadak") or 0)
        + int(ef.get("jjob") or 0)
        + int(ef.get("ssul") or 0)
        + int(ef.get("ttak") or 0)
    )
    mult *= 1.0 + min(8, special_count) * 0.03

    return math.tanh((diff * mult) / max(1e-9, float(value_scale)))


def build_cache(input_paths, cache_path, max_samples, value_scale):
    token_counter = {}
    action_set = set()
    raw = []
    go_stop_count = 0
    decision_type_counts = {}

    for path in input_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line_raw in f:
                line_raw = line_raw.strip()
                if not line_raw:
                    continue
                line = json.loads(line_raw)
                for trace in line.get("decision_trace") or []:
                    actor = trace.get("a")
                    if actor not in (SIDE_MY, SIDE_YOUR):
                        continue
                    chosen, decision_type, candidates = choose_label_and_candidates(trace)
                    if chosen is None:
                        continue
                    tv = target_value(line, actor, value_scale)
                    if tv is None:
                        continue
                    tokens = extract_tokens(line, trace, decision_type, actor)
                    numeric = extract_numeric(line, trace, actor)
                    decision_type_counts[decision_type] = decision_type_counts.get(decision_type, 0) + 1
                    if chosen in GO_STOP_ACTIONS:
                        go_stop_count += 1
                    raw.append(
                        {
                            "tokens": tokens,
                            "numeric": numeric,
                            "candidates": candidates,
                            "chosen": chosen,
                            "value_target": float(tv),
                            "is_go_stop": int(chosen in GO_STOP_ACTIONS),
                        }
                    )
                    for t in tokens:
                        token_counter[t] = token_counter.get(t, 0) + 1
                    action_set.update(candidates)
                    action_set.add(chosen)
                    if max_samples is not None and len(raw) >= max_samples:
                        break
                if max_samples is not None and len(raw) >= max_samples:
                    break
        if max_samples is not None and len(raw) >= max_samples:
            break

    if not raw:
        raise RuntimeError("No trainable samples found.")

    token_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for tok, _ in sorted(token_counter.items(), key=lambda x: (-x[1], x[0])):
        token_to_idx[tok] = len(token_to_idx)
    idx_to_action = sorted(action_set)
    action_to_idx = {a: i for i, a in enumerate(idx_to_action)}

    numeric_scale = {}
    for k in NUMERIC_KEYS:
        numeric_scale[k] = max(1.0, max(abs(r["numeric"].get(k, 0.0)) for r in raw))

    samples = []
    for r in raw:
        token_ids = [token_to_idx.get(t, token_to_idx[UNK_TOKEN]) for t in r["tokens"]]
        cand_ids = [action_to_idx[c] for c in r["candidates"] if c in action_to_idx]
        target_id = action_to_idx.get(r["chosen"])
        if target_id is None or target_id not in cand_ids:
            continue
        numeric = []
        for k in NUMERIC_KEYS:
            base = float(r["numeric"].get(k, 0.0)) / numeric_scale[k]
            w = float(NUMERIC_FEATURE_WEIGHTS.get(k, 1.0))
            numeric.append(base * w)
        go_stop_target = GO_STOP_ACTIONS.get(r["chosen"], -1)
        samples.append(
            {
                "token_ids": token_ids,
                "numeric": numeric,
                "cand_ids": cand_ids,
                "target_id": target_id,
                "value_target": float(r["value_target"]),
                "is_go_stop": int(r["is_go_stop"]),
                "go_stop_target": int(go_stop_target),
            }
        )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_files": input_paths,
        "value_scale": value_scale,
        "token_to_idx": token_to_idx,
        "idx_to_action": idx_to_action,
        "numeric_keys": NUMERIC_KEYS,
        "numeric_feature_weights": NUMERIC_FEATURE_WEIGHTS,
        "numeric_scale": numeric_scale,
        "samples": samples,
        "stats": {
            "go_stop_samples": int(go_stop_count),
            "decision_type_counts": decision_type_counts,
            "action_has_go": bool("go" in action_to_idx),
            "action_has_stop": bool("stop" in action_to_idx),
        },
    }
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    torch.save(payload, cache_path)
    return payload


def load_or_build_cache(args):
    cache_path = args.cache_pt or os.path.join("logs", "dual-cache.pt")
    if args.rebuild_cache or not os.path.exists(cache_path):
        paths = expand_inputs(args.input)
        return build_cache(paths, cache_path, args.max_samples, args.value_scale), cache_path
    data = torch.load(cache_path, map_location="cpu")
    return data, cache_path


class KFlowerDualNet(nn.Module):
    def __init__(self, vocab_size, num_numeric, action_size, emb_dim=256, hidden_dim=1024, dropout=0.15):
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab_size, emb_dim, mode="mean", include_last_offset=True)
        self.token_proj = nn.Linear(emb_dim, hidden_dim // 2)
        self.numeric_proj = nn.Linear(num_numeric, hidden_dim // 2)

        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(hidden_dim, action_size)
        self.go_stop_head = nn.Linear(hidden_dim, 2)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, token_ids_flat, offsets, numeric):
        token_mean = self.emb(token_ids_flat, offsets)
        t = F.relu(self.token_proj(token_mean))
        n = F.relu(self.numeric_proj(numeric))
        h = self.trunk(torch.cat([t, n], dim=1))
        policy_logits = self.policy_head(h)
        go_stop_logits = self.go_stop_head(h)
        value = torch.tanh(self.value_head(h)).squeeze(-1)
        return policy_logits, go_stop_logits, value


def build_batch(samples, batch_ids, action_size, device, go_stop_policy_weight, go_stop_value_weight):
    bsz = len(batch_ids)
    flat_tokens = []
    offsets = [0]
    numeric = torch.zeros((bsz, len(NUMERIC_KEYS)), dtype=torch.float32, device=device)
    cand_mask = torch.zeros((bsz, action_size), dtype=torch.bool, device=device)
    target = torch.zeros((bsz,), dtype=torch.long, device=device)
    value_t = torch.zeros((bsz,), dtype=torch.float32, device=device)
    gs_target = torch.full((bsz,), -1, dtype=torch.long, device=device)
    gs_policy_weight = torch.ones((bsz,), dtype=torch.float32, device=device)
    gs_value_weight = torch.ones((bsz,), dtype=torch.float32, device=device)

    for r, sid in enumerate(batch_ids):
        s = samples[sid]
        tids = s["token_ids"] if s["token_ids"] else [1]
        flat_tokens.extend(tids)
        offsets.append(offsets[-1] + len(tids))
        numeric[r] = torch.tensor(s["numeric"], dtype=torch.float32, device=device)
        for c in s["cand_ids"]:
            cand_mask[r, c] = True
        target[r] = int(s["target_id"])
        value_t[r] = float(s["value_target"])
        gs_target[r] = int(s["go_stop_target"])
        if int(s["is_go_stop"]) == 1:
            gs_policy_weight[r] = float(go_stop_policy_weight)
            gs_value_weight[r] = float(go_stop_value_weight)
    token_ids_flat = torch.tensor(flat_tokens, dtype=torch.long, device=device)
    offsets_t = torch.tensor(offsets, dtype=torch.long, device=device)
    return (
        token_ids_flat,
        offsets_t,
        numeric,
        cand_mask,
        target,
        value_t,
        gs_target,
        gs_policy_weight,
        gs_value_weight,
    )


def evaluate(model, samples, action_size, batch_size, device):
    model.eval()
    total = 0
    acc_n = 0
    gs_total = 0
    gs_acc_n = 0
    se = 0.0
    ae = 0.0
    with torch.no_grad():
        for st in range(0, len(samples), batch_size):
            ids = list(range(st, min(len(samples), st + batch_size)))
            token_ids_flat, offsets, numeric, cand_mask, target, value_t, gs_target, _, _ = build_batch(
                samples,
                ids,
                action_size,
                device,
                go_stop_policy_weight=1.0,
                go_stop_value_weight=1.0,
            )
            p_logits, gs_logits, value_p = model(token_ids_flat, offsets, numeric)
            mask_fill = torch.finfo(p_logits.dtype).min
            p_logits = p_logits.masked_fill(~cand_mask, mask_fill)
            pred = torch.argmax(p_logits, dim=1)
            acc_n += int(torch.sum(pred == target).item())
            valid_gs = gs_target >= 0
            if int(torch.sum(valid_gs).item()) > 0:
                gs_pred = torch.argmax(gs_logits[valid_gs], dim=1)
                gs_acc_n += int(torch.sum(gs_pred == gs_target[valid_gs]).item())
                gs_total += int(torch.sum(valid_gs).item())
            err = value_p - value_t
            se += float(torch.sum(err * err).item())
            ae += float(torch.sum(torch.abs(err)).item())
            total += len(ids)
    if total == 0:
        return {"policy_acc": 0.0, "go_stop_acc": 0.0, "value_rmse": 0.0, "value_mae": 0.0, "valid_loss": 999.0}
    mse = se / total
    rmse = math.sqrt(mse)
    mae = ae / total
    policy_acc = acc_n / total
    go_stop_acc = (gs_acc_n / gs_total) if gs_total > 0 else 0.0
    valid_loss = (1.0 - policy_acc) + rmse
    return {
        "policy_acc": policy_acc,
        "go_stop_acc": go_stop_acc,
        "value_rmse": rmse,
        "value_mae": mae,
        "valid_loss": valid_loss,
    }


def main():
    parser = argparse.ArgumentParser(description="Train DualNet with vocab+embedding, cache .pt, AMP, and early stopping.")
    parser.add_argument("--input", nargs="+", default=["logs/*.jsonl"])
    parser.add_argument("--cache-pt", default="logs/dual-cache.pt")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--output", default="models/dual-model.pt")
    parser.add_argument("--meta-output", default=None)
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--value-loss-weight", type=float, default=0.7)
    parser.add_argument("--value-loss-weight-start", type=float, default=0.1)
    parser.add_argument("--value-loss-weight-end", type=float, default=None)
    parser.add_argument("--go-stop-policy-weight", type=float, default=1.8)
    parser.add_argument("--go-stop-value-weight", type=float, default=1.3)
    parser.add_argument("--value-scale", type=float, default=10.0)
    parser.add_argument("--max-samples", type=int, default=400000)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP.")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = args.device
    use_amp = (device == "cuda") and (not args.no_amp)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    cache, cache_path = load_or_build_cache(args)
    samples = cache["samples"]
    if not samples:
        raise RuntimeError("No samples in cache.")
    action_size = len(cache["idx_to_action"])
    vocab_size = len(cache["token_to_idx"])
    cache_stats = cache.get("stats") or {}

    order = list(range(len(samples)))
    random.Random(args.seed).shuffle(order)
    split = int(len(order) * 0.9)
    train_ids = order[:split]
    valid_ids = order[split:] if split < len(order) else []
    train_samples = [samples[i] for i in train_ids]
    valid_samples = [samples[i] for i in valid_ids]

    print(
        json.dumps(
            {
                "dataset_stats": {
                    "samples_total": len(samples),
                    "go_stop_samples": int(cache_stats.get("go_stop_samples", 0)),
                    "go_stop_ratio": float(cache_stats.get("go_stop_samples", 0)) / max(1, len(samples)),
                    "action_has_go": bool(cache_stats.get("action_has_go", False)),
                    "action_has_stop": bool(cache_stats.get("action_has_stop", False)),
                    "decision_type_counts": cache_stats.get("decision_type_counts", {}),
                    "split_train": len(train_samples),
                    "split_valid": len(valid_samples),
                }
            },
            ensure_ascii=False,
        )
    )

    model = KFlowerDualNet(
        vocab_size=vocab_size,
        num_numeric=len(NUMERIC_KEYS),
        action_size=action_size,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs), eta_min=args.lr_min)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = {"loss": float("inf"), "state": None, "epoch": 0}
    bad_epochs = 0
    rng = random.Random(args.seed)
    value_w_end = args.value_loss_weight if args.value_loss_weight_end is None else args.value_loss_weight_end

    for ep in range(args.epochs):
        phase = 0.0 if args.epochs <= 1 else (ep / (args.epochs - 1))
        value_w = args.value_loss_weight_start + (value_w_end - args.value_loss_weight_start) * phase
        model.train()
        perm = list(range(len(train_samples)))
        rng.shuffle(perm)
        for st in range(0, len(perm), args.batch_size):
            ids = perm[st : st + args.batch_size]
            token_ids_flat, offsets, numeric, cand_mask, target, value_t, gs_target, gs_policy_weight, gs_value_weight = build_batch(
                train_samples,
                ids,
                action_size,
                device,
                go_stop_policy_weight=args.go_stop_policy_weight,
                go_stop_value_weight=args.go_stop_value_weight,
            )
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                p_logits, gs_logits, value_p = model(token_ids_flat, offsets, numeric)
                mask_fill = torch.finfo(p_logits.dtype).min
                p_logits = p_logits.masked_fill(~cand_mask, mask_fill)
                p_loss_raw = F.cross_entropy(p_logits, target, reduction="none")
                p_loss = torch.mean(p_loss_raw * gs_policy_weight)
                valid_gs = gs_target >= 0
                if int(torch.sum(valid_gs).item()) > 0:
                    gs_loss = F.cross_entropy(gs_logits[valid_gs], gs_target[valid_gs])
                else:
                    gs_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                v_loss_raw = (value_p - value_t) * (value_p - value_t)
                v_loss = torch.mean(v_loss_raw * gs_value_weight)
                loss = p_loss + value_w * v_loss + 0.3 * gs_loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        scheduler.step()

        metrics = evaluate(model, valid_samples, action_size, args.batch_size, device) if valid_samples else {
            "policy_acc": 0.0,
            "go_stop_acc": 0.0,
            "value_rmse": 0.0,
            "value_mae": 0.0,
            "valid_loss": 999.0,
        }
        print(
            f"epoch {ep + 1}/{args.epochs} "
            f"| valid_policy_acc {metrics['policy_acc']:.4f} "
            f"| valid_go_stop_acc {metrics['go_stop_acc']:.4f} "
            f"| valid_value_rmse {metrics['value_rmse']:.4f} "
            f"| value_w {value_w:.4f} "
            f"| lr {scheduler.get_last_lr()[0]:.6g}"
        )

        if metrics["valid_loss"] < best["loss"]:
            best["loss"] = metrics["valid_loss"]
            best["state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best["epoch"] = ep + 1
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stopping at epoch {ep + 1} (patience={args.patience})")
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    train_metrics = evaluate(model, train_samples, action_size, args.batch_size, device)
    valid_metrics = evaluate(model, valid_samples, action_size, args.batch_size, device)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "dual_net_embed_mlp_v1",
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "token_to_idx": cache["token_to_idx"],
        "idx_to_action": cache["idx_to_action"],
        "numeric_keys": cache["numeric_keys"],
        "numeric_feature_weights": cache.get("numeric_feature_weights", {}),
        "numeric_scale": cache["numeric_scale"],
        "value_scale": cache["value_scale"],
        "emb_dim": args.emb_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "go_stop_actions": GO_STOP_ACTIONS,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(payload, args.output)

    summary = {
        "cache_path": cache_path,
        "samples_total": len(samples),
        "samples_train": len(train_samples),
        "samples_valid": len(valid_samples),
        "actions": action_size,
        "vocab_size": vocab_size,
        "epochs_requested": args.epochs,
        "best_epoch": best["epoch"],
        "patience": args.patience,
        "amp_enabled": use_amp,
        "go_stop_policy_weight": args.go_stop_policy_weight,
        "go_stop_value_weight": args.go_stop_value_weight,
        "value_loss_weight_start": args.value_loss_weight_start,
        "value_loss_weight_end": value_w_end,
        "lr_start": args.lr,
        "lr_min": args.lr_min,
        "dataset_stats": {
            "go_stop_samples": int(cache_stats.get("go_stop_samples", 0)),
            "go_stop_ratio": float(cache_stats.get("go_stop_samples", 0)) / max(1, len(samples)),
            "action_has_go": bool(cache_stats.get("action_has_go", False)),
            "action_has_stop": bool(cache_stats.get("action_has_stop", False)),
            "decision_type_counts": cache_stats.get("decision_type_counts", {}),
        },
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
    }
    meta_out = args.meta_output or (args.output + ".json")
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"trained dual net -> {args.output}")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

