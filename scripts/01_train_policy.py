#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone

try:
    import lmdb  # type: ignore
except Exception:
    lmdb = None

DEFAULT_POLICY_FILTER_RULES_PATH = "configs/policy_filter_rules.json"
DEFAULT_ACTION_VOCAB_PATH = "configs/action_vocab_v1.json"
DEFAULT_POLICY_FILTER_RULES = {
    "min_candidate_count": 2,
}
DEFAULT_ACTION_VOCAB = {
    "format_version": "action_vocab_v1",
    "decision_types": ["play", "match", "option"],
    "option_actions": [
        "go",
        "stop",
        "shaking_yes",
        "shaking_no",
        "president_stop",
        "president_hold",
        "five",
        "junk",
    ],
}


def _float_or_none(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _score_diff_from_dc(dc):
    return float((dc or {}).get("sd") or 0)


def _bucket_hand_self(hand_self):
    hs = int(hand_self or 0)
    if hs >= 8:
        return "8p"
    if hs >= 5:
        return "5_7"
    if hs >= 2:
        return "2_4"
    return "0_1"


def _bucket_hand_diff(hand_diff):
    hd = int(hand_diff or 0)
    if hd <= -3:
        return "n3"
    if hd <= -1:
        return "n2_1"
    if hd == 0:
        return "z0"
    if hd <= 2:
        return "p1_2"
    return "p3"


def _bucket_score_diff(score_diff):
    sd = float(score_diff or 0)
    if sd <= -10:
        return "n10"
    if sd <= -3:
        return "n9_3"
    if sd < 3:
        return "z2"
    if sd < 10:
        return "p3_9"
    return "p10"


def _bucket_go_count(go_count):
    g = int(go_count or 0)
    if g <= 0:
        return "0"
    if g == 1:
        return "1"
    return "2p"


def _bucket_candidates(cands):
    c = int(cands or 0)
    if c <= 1:
        return "1"
    if c == 2:
        return "2"
    if c == 3:
        return "3"
    return "4p"


def _bucket_risk_total(total):
    t = max(0, int(total or 0))
    if t <= 0:
        return "0"
    if t == 1:
        return "1"
    return "2p"


def _to_percent_scale(value):
    try:
        x = float(value or 0.0)
    except Exception:
        return 0.0
    if x <= 1.0:
        x = x * 100.0
    if x < 0.0:
        return 0.0
    if x > 100.0:
        return 100.0
    return x


def _bucket_threat_percent(value):
    x = _to_percent_scale(value)
    if x >= 70.0:
        return "h"
    if x >= 35.0:
        return "m"
    return "l"


def _bucket_progress_delta(delta):
    d = int(delta or 0)
    if d <= -3:
        return "n3"
    if d <= -1:
        return "n2_1"
    if d == 0:
        return "z0"
    if d <= 2:
        return "p1_2"
    return "p3"


def _bucket_go_stop_signal(gsd_permille):
    x = int(gsd_permille or 0)
    if x <= -1800:
        return "n2"
    if x <= -600:
        return "n1"
    if x < 600:
        return "z0"
    if x < 1800:
        return "p1"
    return "p2"


def actor_gold_delta(game, actor):
    if actor not in ("mySide", "yourSide"):
        return None

    f_my = _float_or_none(game.get("finalGoldMy"))
    i_my = _float_or_none(game.get("initialGoldMy"))
    f_your = _float_or_none(game.get("finalGoldYour"))
    i_your = _float_or_none(game.get("initialGoldYour"))
    if None not in (f_my, i_my):
        delta_my = float(f_my) - float(i_my)
        return delta_my if actor == "mySide" else -delta_my
    if None not in (f_your, i_your):
        delta_your = float(f_your) - float(i_your)
        return delta_your if actor == "yourSide" else -delta_your

    return None


def sample_weight_from_gold_delta(delta_gold, weight_config):
    cfg = dict(weight_config or {})
    mode = str(cfg.get("mode") or "none").strip().lower()
    lo = float(cfg.get("clip_min") or 0.5)
    hi = float(cfg.get("clip_max") or 3.0)
    lo, hi = min(lo, hi), max(lo, hi)
    if mode == "none" or delta_gold is None:
        return 1.0

    coef = float(cfg.get("coef") or 0.0)
    scale = max(1e-9, float(cfg.get("scale") or 1.0))
    x = float(delta_gold)
    if mode == "gold_signed_sqrt":
        z = math.sqrt(abs(x) / scale)
        s = 0.0 if x == 0 else (1.0 if x > 0 else -1.0)
        w = 1.0 + coef * s * z
    else:
        # gold_tanh: robust default for outlier-heavy tails
        z = x / scale
        w = 1.0 + coef * math.tanh(z)
    return _clamp(float(w), lo, hi)


def _quantile_threshold(values_sorted, q):
    if not values_sorted:
        return None
    n = len(values_sorted)
    qq = _clamp(float(q), 0.0, 1.0)
    idx = int(math.floor((n - 1) * qq))
    idx = max(0, min(n - 1, idx))
    return float(values_sorted[idx])


def compute_elite_threshold(paths, top_percent):
    tp = float(top_percent or 0.0)
    if tp <= 0.0:
        return None
    vals = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                game = json.loads(line)
                d_my = actor_gold_delta(game, "mySide")
                d_your = actor_gold_delta(game, "yourSide")
                if d_my is not None and d_my > 0:
                    vals.append(float(d_my))
                if d_your is not None and d_your > 0:
                    vals.append(float(d_your))
    if not vals:
        return None
    vals.sort()
    q = 1.0 - (_clamp(tp, 0.0, 100.0) / 100.0)
    return _quantile_threshold(vals, q)


def action_balance_multiplier(decision_type, chosen, weight_config):
    cfg = dict(weight_config or {})
    if not bool(cfg.get("action_balance_enabled")):
        return 1.0

    dt = str(decision_type or "").strip().lower()
    ch = str(chosen or "").strip().lower()

    play_mult = float(cfg.get("play_weight") or 1.0)
    match_mult = float(cfg.get("match_weight") or 1.0)
    option_mult = float(cfg.get("option_weight") or 1.0)
    go_mult = float(cfg.get("go_extra_weight") or 1.0)
    stop_mult = float(cfg.get("stop_extra_weight") or 1.0)
    special_option_mult = float(cfg.get("special_option_extra_weight") or 1.0)

    if dt == "play":
        return max(0.0, play_mult)
    if dt == "match":
        return max(0.0, match_mult)
    if dt == "option":
        m = max(0.0, option_mult)
        if ch == "go":
            m *= max(0.0, go_mult)
        elif ch == "stop":
            m *= max(0.0, stop_mult)
        elif ch in ("shaking_yes", "shaking_no", "president_stop", "president_hold", "five", "junk"):
            m *= max(0.0, special_option_mult)
        return m
    return 1.0


def elite_multiplier(delta_actor, weight_config):
    cfg = dict(weight_config or {})
    if not bool(cfg.get("elite_enabled")):
        return 1.0
    threshold = _float_or_none(cfg.get("elite_threshold"))
    mult = float(cfg.get("elite_multiplier") or 1.0)
    if threshold is None:
        return 1.0
    if delta_actor is None:
        return 1.0
    x = float(delta_actor)
    if x > 0 and x >= threshold:
        return max(0.0, mult)
    return 1.0


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


def input_manifest(paths):
    manifest = []
    for p in paths:
        try:
            st = os.stat(p)
            manifest.append(
                {
                    "path": os.path.abspath(p),
                    "size": int(st.st_size),
                    "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
                }
            )
        except Exception:
            manifest.append({"path": os.path.abspath(p), "size": -1, "mtime_ns": -1})
    return manifest


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
    if a == "mySide":
        return "first"
    if a == "yourSide":
        return "second"
    return "?"


def _to_int_or_none(value):
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _normalize_policy_filter_rules(raw):
    rules = dict(DEFAULT_POLICY_FILTER_RULES)
    if isinstance(raw, dict):
        min_cc = _to_int_or_none(raw.get("min_candidate_count"))
        if min_cc is not None:
            rules["min_candidate_count"] = max(1, min_cc)
    return rules


def load_policy_filter_rules(path):
    p = str(path or "").strip()
    if not p:
        return dict(DEFAULT_POLICY_FILTER_RULES), None
    if not os.path.exists(p):
        return dict(DEFAULT_POLICY_FILTER_RULES), None
    with open(p, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)
    return _normalize_policy_filter_rules(raw), os.path.abspath(p)


def _normalize_action_vocab(raw):
    out = dict(DEFAULT_ACTION_VOCAB)
    if not isinstance(raw, dict):
        return out
    fmt = str(raw.get("format_version") or "").strip()
    if fmt:
        out["format_version"] = fmt
    dt = raw.get("decision_types")
    if isinstance(dt, list):
        vals = [str(x or "").strip().lower() for x in dt]
        vals = [x for x in vals if x]
        if vals:
            out["decision_types"] = sorted(set(vals))
    oa = raw.get("option_actions")
    if isinstance(oa, list):
        vals = [str(x or "").strip().lower() for x in oa]
        vals = [x for x in vals if x]
        if vals:
            out["option_actions"] = sorted(set(vals))
    return out


def load_action_vocab(path):
    p = str(path or "").strip()
    if not p:
        raise RuntimeError("--action-vocab path is required.")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Action vocab file not found: {p}")
    with open(p, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)
    return _normalize_action_vocab(raw), os.path.abspath(p)


def trace_legal_actions(trace, decision_type=None):
    dt = str(decision_type or trace.get("dt") or "").strip().lower()
    raw_la = trace.get("la")
    if not isinstance(raw_la, list):
        return []
    out = []
    seen = set()
    for item in raw_la:
        if dt == "option":
            v = action_alias(item)
            v = str(v or "").strip().lower()
        else:
            v = str(item or "").strip()
        if not v:
            continue
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def trace_passes_policy_filter(trace, filter_rules):
    min_cc = int((filter_rules or {}).get("min_candidate_count") or 1)
    legal = trace_legal_actions(trace)
    return len(legal) >= min_cc


def context_key(trace, decision_type, context_bucketing=True):
    cached = trace.get("ck")
    if (not context_bucketing) and isinstance(cached, str) and cached:
        return cached
    dc = trace.get("dc") or {}
    deck_bucket = int((dc.get("d") or 0) // 3)
    hand_self = int(dc.get("hs") or 0)
    hand_diff = int(dc.get("hd") or 0)
    go_self = int(dc.get("gs") or 0)
    go_opp = int(dc.get("go") or 0)
    shake_self = min(3, int(dc.get("ss") or 0))
    shake_opp = min(3, int(dc.get("so") or 0))
    score_diff = _score_diff_from_dc(dc)
    bak_risk_total = (
        (1 if int(dc.get("rp") or 0) else 0)
        + (1 if int(dc.get("rg") or 0) else 0)
        + (1 if int(dc.get("rm") or 0) else 0)
    )
    opp_threat = max(
        _to_percent_scale(dc.get("ojt")),
        _to_percent_scale(dc.get("ojo")),
        _to_percent_scale(dc.get("ogt")),
    )
    self_threat = max(
        _to_percent_scale(dc.get("sjt")),
        _to_percent_scale(dc.get("sjo")),
        _to_percent_scale(dc.get("sgt")),
    )
    progress_delta = int(dc.get("jps") or 0) - int(dc.get("jpo") or 0)
    go_stop_signal = int(dc.get("gsd") or 0)
    phase = dc.get("p")
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
    cands = len(trace_legal_actions(trace, decision_type))
    if cands < 0:
        cands = 0
    if context_bucketing:
        hs_token = _bucket_hand_self(hand_self)
        hd_token = _bucket_hand_diff(hand_diff)
        sd_token = _bucket_score_diff(score_diff)
        gs_token = _bucket_go_count(go_self)
        go_token = _bucket_go_count(go_opp)
        cc_token = _bucket_candidates(cands)
    else:
        hs_token = str(hand_self)
        hd_token = str(hand_diff)
        sd_token = str(int(math.floor(score_diff)))
        gs_token = str(go_self)
        go_token = str(go_opp)
        cc_token = str(cands)
    out = [
        f"dt={decision_type}",
        f"ph={phase}",
        f"o={trace_order(trace)}",
        f"db={deck_bucket}",
        f"hs={hs_token}",
        f"hd={hd_token}",
        f"sd={sd_token}",
        f"gs={gs_token}",
        f"go={go_token}",
        f"ss={shake_self}",
        f"so={shake_opp}",
        f"cc={cc_token}",
    ]
    if context_bucketing:
        out.extend(
            [
                f"br={_bucket_risk_total(bak_risk_total)}",
                f"ot={_bucket_threat_percent(opp_threat)}",
                f"st={_bucket_threat_percent(self_threat)}",
                f"jp={_bucket_progress_delta(progress_delta)}",
                f"gd={_bucket_go_stop_signal(go_stop_signal)}",
            ]
        )
    return "|".join(out)


def extract_sample(trace, action_vocab):
    dt = trace.get("dt")
    dt = str(dt or "").strip().lower()
    if dt not in set(action_vocab.get("decision_types") or []):
        return None
    legal = trace_legal_actions(trace, dt)
    if len(legal) < 1:
        return None

    chosen_compact = trace.get("ch")
    if dt == "option":
        chosen = str(action_alias(chosen_compact) or "").strip().lower()
        option_allowed = set(str(x).strip().lower() for x in (action_vocab.get("option_actions") or []))
        if not option_allowed:
            return None
        if any(c not in option_allowed for c in legal):
            return None
        if chosen not in option_allowed:
            return None
    else:
        chosen = str(chosen_compact or "").strip()
    if not chosen:
        return None
    if chosen not in legal:
        return None
    return dt, legal, chosen


def iter_samples(
    paths,
    max_samples=None,
    filter_rules=None,
    context_bucketing=True,
    weight_config=None,
    action_vocab=None,
):
    yielded = 0
    active_rules = _normalize_policy_filter_rules(filter_rules)
    weight_cfg = dict(weight_config or {})
    vocab = _normalize_action_vocab(action_vocab)
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                game = json.loads(line)
                delta_my = actor_gold_delta(game, "mySide")
                delta_your = actor_gold_delta(game, "yourSide")
                for trace in game.get("decision_trace") or []:
                    if not trace_passes_policy_filter(trace, active_rules):
                        continue
                    sample = extract_sample(trace, vocab)
                    if sample is None:
                        continue
                    decision_type, candidates, chosen = sample
                    actor = str(trace.get("a") or "")
                    delta_actor = delta_my if actor == "mySide" else (delta_your if actor == "yourSide" else None)
                    base_weight = sample_weight_from_gold_delta(delta_actor, weight_cfg)
                    balance_mult = action_balance_multiplier(decision_type, chosen, weight_cfg)
                    elite_mult = elite_multiplier(delta_actor, weight_cfg)
                    sample_weight = float(base_weight) * float(balance_mult) * float(elite_mult)
                    final_lo = float(weight_cfg.get("final_clip_min") or 0.01)
                    final_hi = float(weight_cfg.get("final_clip_max") or 8.0)
                    sample_weight = _clamp(sample_weight, min(final_lo, final_hi), max(final_lo, final_hi))
                    yield {
                        "decision_type": decision_type,
                        "candidates": candidates,
                        "chosen": chosen,
                        "context_key": context_key(trace, decision_type, context_bucketing=context_bucketing),
                        "weight": sample_weight,
                    }
                    yielded += 1
                    if max_samples is not None and yielded >= max_samples:
                        return


def _ensure_lmdb_available():
    if lmdb is None:
        raise RuntimeError(
            "cache-backend=lmdb requires python package 'lmdb'. Install with: pip install lmdb"
        )


def _cache_meta_path(cache_path):
    return f"{cache_path}.meta.json"


def _cache_data_exists(cache_path, backend):
    if backend == "lmdb":
        return os.path.isdir(cache_path)
    return os.path.exists(cache_path)


def _cache_matches(meta, config, manifest):
    if not isinstance(meta, dict):
        return False
    if meta.get("format_version") != config.get("format_version"):
        return False
    if meta.get("config") != config:
        return False
    if meta.get("input_manifest") != manifest:
        return False
    counts = meta.get("counts") or {}
    return int(counts.get("samples_total") or 0) > 0


def _sample_key(index):
    return f"{index:09d}".encode("ascii")


def build_sample_cache_jsonl(
    cache_path,
    input_paths,
    max_samples,
    filter_rules,
    context_bucketing,
    weight_config,
    action_vocab,
):
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    total = 0
    with open(cache_path, "w", encoding="utf-8") as out:
        for sample in iter_samples(
            input_paths,
            max_samples,
            filter_rules,
            context_bucketing=context_bucketing,
            weight_config=weight_config,
            action_vocab=action_vocab,
        ):
            out.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
            total += 1
    if total <= 0:
        raise RuntimeError("No trainable decision samples found.")
    return {"samples_total": int(total)}


def build_sample_cache_lmdb(
    cache_path,
    input_paths,
    max_samples,
    map_size_bytes,
    filter_rules,
    context_bucketing,
    weight_config,
    action_vocab,
):
    _ensure_lmdb_available()
    os.makedirs(cache_path, exist_ok=True)
    env = lmdb.open(
        cache_path,
        map_size=max(1024 * 1024, int(map_size_bytes)),
        subdir=True,
        create=True,
        lock=True,
        readahead=False,
        metasync=False,
        sync=False,
        map_async=True,
    )
    total = 0
    txn = env.begin(write=True)
    try:
        for sample in iter_samples(
            input_paths,
            max_samples,
            filter_rules,
            context_bucketing=context_bucketing,
            weight_config=weight_config,
            action_vocab=action_vocab,
        ):
            blob = json.dumps(sample, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            txn.put(_sample_key(total), blob)
            total += 1
            if total % 2000 == 0:
                txn.commit()
                txn = env.begin(write=True)
        txn.put(b"__count__", str(int(total)).encode("ascii"))
        txn.commit()
        env.sync()
    finally:
        env.close()
    if total <= 0:
        raise RuntimeError("No trainable decision samples found.")
    return {"samples_total": int(total)}


def build_sample_cache(
    cache_backend,
    cache_path,
    input_paths,
    max_samples,
    lmdb_map_size_bytes,
    filter_rules,
    context_bucketing,
    weight_config,
    action_vocab,
):
    if cache_backend == "lmdb":
        return build_sample_cache_lmdb(
            cache_path,
            input_paths,
            max_samples,
            lmdb_map_size_bytes,
            filter_rules,
            context_bucketing,
            weight_config,
            action_vocab,
        )
    return build_sample_cache_jsonl(
        cache_path,
        input_paths,
        max_samples,
        filter_rules,
        context_bucketing,
        weight_config,
        action_vocab,
    )


def iter_cached_samples_jsonl(cache_path):
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def iter_cached_samples_lmdb(cache_path):
    _ensure_lmdb_available()
    env = lmdb.open(
        cache_path,
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=256,
        subdir=True,
    )
    try:
        with env.begin(write=False) as txn:
            raw_count = txn.get(b"__count__")
            count = int(raw_count.decode("ascii")) if raw_count else 0
            for i in range(count):
                blob = txn.get(_sample_key(i))
                if not blob:
                    continue
                yield json.loads(bytes(blob).decode("utf-8"))
    finally:
        env.close()


def iter_cached_samples(cache_backend, cache_path):
    if cache_backend == "lmdb":
        return iter_cached_samples_lmdb(cache_path)
    return iter_cached_samples_jsonl(cache_path)


def train_model(samples, alpha):
    global_counts = defaultdict(lambda: defaultdict(float))
    context_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    context_totals = defaultdict(lambda: defaultdict(float))
    sample_count = 0

    for s in samples:
        dt = s["decision_type"]
        ck = s["context_key"]
        chosen = s["chosen"]
        w = float(s.get("weight", 1.0) or 1.0)
        if (not math.isfinite(w)) or w <= 0:
            w = 1.0
        global_counts[dt][chosen] += w
        context_counts[dt][ck][chosen] += w
        context_totals[dt][ck] += w
        sample_count += 1

    model = {
        "model_type": "policy_frequency_weighted_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "alpha": alpha,
        "global_counts": {k: dict(v) for k, v in global_counts.items()},
        "context_counts": {dt: {ck: dict(cc) for ck, cc in cks.items()} for dt, cks in context_counts.items()},
        "context_totals": {dt: dict(v) for dt, v in context_totals.items()},
        "samples_observed": int(sample_count),
    }
    return model


def prob_of_choice(model, sample, choice):
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


def predict_top1(model, sample):
    best_choice = None
    best_prob = -1.0
    for c in sample["candidates"]:
        p = prob_of_choice(model, sample, c)
        if p > best_prob:
            best_prob = p
            best_choice = c
    return best_choice, best_prob

def sample_entropy(model, sample):
    candidates = sample.get("candidates") or []
    if not candidates:
        return 0.0, 0.0
    probs = [max(1e-12, float(prob_of_choice(model, sample, c))) for c in candidates]
    z = sum(probs)
    if z <= 0:
        return 0.0, 0.0
    probs = [p / z for p in probs]
    ent = -sum(p * math.log(p) for p in probs)
    k = len(candidates)
    if k <= 1:
        return ent, 0.0
    ent_norm = ent / math.log(k)
    return ent, ent_norm


def _resolve_sample_cache_path(output_path, sample_cache_arg, backend):
    cache_arg = str(sample_cache_arg or "").strip()
    if cache_arg.lower() in ("", "none", "off", "no", "0"):
        return None
    if cache_arg.lower() == "auto":
        base, _ = os.path.splitext(output_path)
        if backend == "lmdb":
            return f"{base}.samples.cache.lmdb"
        return f"{base}.samples.cache.jsonl"
    return cache_arg


def main():
    parser = argparse.ArgumentParser(description="Train policy classifier from kibo JSONL.")
    parser.add_argument(
        "--input",
        nargs="+",
        default=["logs/*.jsonl"],
        help="Input JSONL path(s) or glob(s).",
    )
    parser.add_argument(
        "--output",
        default="models/policy-model.json",
        help="Output model JSON path.",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of training samples.")
    parser.add_argument(
        "--skip-train-metrics",
        action="store_true",
        help="Skip train-set accuracy/NLL pass for faster training.",
    )
    parser.add_argument(
        "--sample-cache",
        default="none",
        help="Policy sample cache path. Use 'auto' to place near output model, or 'none' to disable cache.",
    )
    parser.add_argument(
        "--cache-backend",
        choices=["jsonl", "lmdb"],
        default="jsonl",
        help="Sample cache backend: jsonl (default) or lmdb.",
    )
    parser.add_argument("--rebuild-cache", action="store_true", help="Force rebuilding sample cache.")
    parser.add_argument(
        "--lmdb-map-size-gb",
        type=float,
        default=2.0,
        help="LMDB map size in GB (used only when --cache-backend lmdb).",
    )
    parser.add_argument(
        "--filter-rules",
        default=DEFAULT_POLICY_FILTER_RULES_PATH,
        help=f"Policy filter rules JSON path (default: {DEFAULT_POLICY_FILTER_RULES_PATH}).",
    )
    parser.add_argument(
        "--action-vocab",
        default=DEFAULT_ACTION_VOCAB_PATH,
        help=f"Action vocab JSON path (default: {DEFAULT_ACTION_VOCAB_PATH}).",
    )
    parser.add_argument(
        "--context-bucketing",
        choices=["on", "off"],
        default="on",
        help="Use coarse context buckets for better generalization (default: on).",
    )
    parser.add_argument(
        "--sample-weight-mode",
        choices=["none", "gold_tanh", "gold_signed_sqrt"],
        default="gold_tanh",
        help="Sample weighting mode for policy learning (default: gold_tanh).",
    )
    parser.add_argument(
        "--sample-weight-coef",
        type=float,
        default=1.2,
        help="Weighting coefficient (default: 1.2).",
    )
    parser.add_argument(
        "--sample-weight-scale",
        type=float,
        default=250000.0,
        help="Weighting scale for gold delta normalization (default: 250000).",
    )
    parser.add_argument(
        "--sample-weight-min",
        type=float,
        default=0.5,
        help="Minimum clipped sample weight (default: 0.5).",
    )
    parser.add_argument(
        "--sample-weight-max",
        type=float,
        default=3.0,
        help="Maximum clipped sample weight (default: 3.0).",
    )
    parser.add_argument(
        "--sample-weight-final-min",
        type=float,
        default=0.05,
        help="Final clipped sample weight min after all multipliers (default: 0.05).",
    )
    parser.add_argument(
        "--sample-weight-final-max",
        type=float,
        default=8.0,
        help="Final clipped sample weight max after all multipliers (default: 8.0).",
    )
    parser.add_argument(
        "--elite-weighting",
        choices=["on", "off"],
        default="on",
        help="Boost top-profit games by actor gold delta percentile (default: on).",
    )
    parser.add_argument(
        "--elite-top-percent",
        type=float,
        default=20.0,
        help="Top percent threshold for elite game weighting (default: 20).",
    )
    parser.add_argument(
        "--elite-multiplier",
        type=float,
        default=2.2,
        help="Multiplier for elite positive-gold samples (default: 2.2).",
    )
    parser.add_argument(
        "--action-balance",
        choices=["on", "off"],
        default="on",
        help="Enable action imbalance reweighting by decision type (default: on).",
    )
    parser.add_argument("--play-weight", type=float, default=1.0, help="Weight for play decisions.")
    parser.add_argument("--match-weight", type=float, default=1.1, help="Weight for match decisions.")
    parser.add_argument("--option-weight", type=float, default=1.8, help="Base weight for option decisions.")
    parser.add_argument(
        "--go-extra-weight",
        type=float,
        default=1.6666667,
        help="Extra multiplier for go option (default: 1.6666667 => go weight 3.0 with option_weight=1.8).",
    )
    parser.add_argument(
        "--stop-extra-weight",
        type=float,
        default=1.25,
        help="Extra multiplier for stop option (default: 1.25 => stop weight 2.25 with option_weight=1.8).",
    )
    parser.add_argument(
        "--special-option-extra-weight",
        type=float,
        default=1.1,
        help="Extra multiplier for shaking/president/gukjin options (default: 1.1).",
    )
    args = parser.parse_args()

    if float(args.lmdb_map_size_gb) <= 0:
        raise RuntimeError("--lmdb-map-size-gb must be > 0.")
    if float(args.sample_weight_scale) <= 0:
        raise RuntimeError("--sample-weight-scale must be > 0.")
    if float(args.sample_weight_min) <= 0 or float(args.sample_weight_max) <= 0:
        raise RuntimeError("--sample-weight-min/max must be > 0.")
    if float(args.sample_weight_final_min) <= 0 or float(args.sample_weight_final_max) <= 0:
        raise RuntimeError("--sample-weight-final-min/max must be > 0.")
    if float(args.elite_top_percent) < 0 or float(args.elite_top_percent) > 100:
        raise RuntimeError("--elite-top-percent must be in [0, 100].")
    if float(args.elite_multiplier) <= 0:
        raise RuntimeError("--elite-multiplier must be > 0.")

    resolved_go_extra = float(args.go_extra_weight)
    resolved_stop_extra = float(args.stop_extra_weight)

    if float(args.play_weight) <= 0 or float(args.match_weight) <= 0 or float(args.option_weight) <= 0:
        raise RuntimeError("--play-weight/--match-weight/--option-weight must be > 0.")
    if resolved_go_extra <= 0 or resolved_stop_extra <= 0 or float(args.special_option_extra_weight) <= 0:
        raise RuntimeError("--go-extra-weight/--stop-extra-weight/--special-option-extra-weight must be > 0.")
    if args.cache_backend == "lmdb":
        _ensure_lmdb_available()

    input_paths = expand_inputs(args.input)
    manifest = input_manifest(input_paths)
    filter_rules, filter_rules_path = load_policy_filter_rules(args.filter_rules)
    action_vocab, action_vocab_path = load_action_vocab(args.action_vocab)
    context_bucketing = str(args.context_bucketing).lower() == "on"
    elite_enabled = str(args.elite_weighting).lower() == "on"
    elite_threshold = compute_elite_threshold(input_paths, args.elite_top_percent) if elite_enabled else None
    weight_config = {
        "mode": str(args.sample_weight_mode).lower(),
        "coef": float(args.sample_weight_coef),
        "scale": float(args.sample_weight_scale),
        "clip_min": min(float(args.sample_weight_min), float(args.sample_weight_max)),
        "clip_max": max(float(args.sample_weight_min), float(args.sample_weight_max)),
        "final_clip_min": min(float(args.sample_weight_final_min), float(args.sample_weight_final_max)),
        "final_clip_max": max(float(args.sample_weight_final_min), float(args.sample_weight_final_max)),
        "elite_enabled": elite_enabled,
        "elite_top_percent": float(args.elite_top_percent),
        "elite_multiplier": float(args.elite_multiplier),
        "elite_threshold": elite_threshold,
        "action_balance_enabled": str(args.action_balance).lower() == "on",
        "play_weight": float(args.play_weight),
        "match_weight": float(args.match_weight),
        "option_weight": float(args.option_weight),
        "go_extra_weight": resolved_go_extra,
        "stop_extra_weight": resolved_stop_extra,
        "special_option_extra_weight": float(args.special_option_extra_weight),
    }
    cache_path = _resolve_sample_cache_path(args.output, args.sample_cache, args.cache_backend)
    cache_config = {
        "format_version": "policy_sample_cache_v1",
        "cache_backend": args.cache_backend,
        "max_samples": args.max_samples,
        "filter_rules": filter_rules,
        "action_vocab": action_vocab,
        "context_bucketing": context_bucketing,
        "weight_config": weight_config,
    }

    cache_ready = False
    counts = None
    if cache_path and (not args.rebuild_cache):
        meta_path = _cache_meta_path(cache_path)
        if _cache_data_exists(cache_path, args.cache_backend) and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if _cache_matches(meta, cache_config, manifest):
                counts = meta.get("counts") or {}
                cache_ready = True
                print(f"cache: reusing {cache_path}")

    lmdb_map_size_bytes = int(float(args.lmdb_map_size_gb) * (1024**3))
    if cache_path and (not cache_ready):
        print(f"cache: building {cache_path}")
        counts = build_sample_cache(
            args.cache_backend,
            cache_path,
            input_paths,
            args.max_samples,
            lmdb_map_size_bytes,
            filter_rules,
            context_bucketing,
            weight_config,
            action_vocab,
        )
        meta = {
            "format_version": "policy_sample_cache_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": cache_config,
            "input_manifest": manifest,
            "counts": counts,
        }
        with open(_cache_meta_path(cache_path), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        cache_ready = True

    def sample_iter_factory():
        if cache_ready and cache_path:
            return iter_cached_samples(args.cache_backend, cache_path)
        return iter_samples(
            input_paths,
            args.max_samples,
            filter_rules,
            context_bucketing=context_bucketing,
            weight_config=weight_config,
            action_vocab=action_vocab,
        )

    model = train_model(sample_iter_factory(), alpha=args.alpha)
    model["context_key_mode"] = "bucketed_v3" if context_bucketing else "raw_v1"
    model["weighting"] = dict(weight_config)
    model["action_vocab"] = dict(action_vocab)

    total = 0
    correct = 0
    nll = 0.0
    entropy_sum = 0.0
    entropy_norm_sum = 0.0
    if args.skip_train_metrics:
        total = int(model.get("samples_observed") or 0)
        if total <= 0:
            raise RuntimeError("No trainable decision samples found.")
    else:
        for s in sample_iter_factory():
            total += 1
            pred, _ = predict_top1(model, s)
            if pred == s["chosen"]:
                correct += 1
            p = max(1e-12, prob_of_choice(model, s, s["chosen"]))
            nll += -math.log(p)
            ent, ent_norm = sample_entropy(model, s)
            entropy_sum += ent
            entropy_norm_sum += ent_norm
        if total <= 0:
            raise RuntimeError("No trainable decision samples found.")

    model["train_summary"] = {
        "samples": total,
        "accuracy_top1": None if args.skip_train_metrics else (correct / total),
        "nll_per_sample": None if args.skip_train_metrics else (nll / total),
        "entropy_avg": None if args.skip_train_metrics else (entropy_sum / total),
        "entropy_norm_avg": None if args.skip_train_metrics else (entropy_norm_sum / total),
        "input_files": input_paths,
        "max_samples": args.max_samples,
        "skip_train_metrics": bool(args.skip_train_metrics),
        "cache_path": cache_path,
        "cache_enabled": bool(cache_path),
        "cache_used": bool(cache_ready and cache_path),
        "cache_backend": args.cache_backend,
        "filter_rules": filter_rules,
        "filter_rules_path": filter_rules_path,
        "action_vocab_path": action_vocab_path,
        "context_bucketing": context_bucketing,
        "weighting": weight_config,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)

    print(f"trained policy model -> {args.output}")
    print(json.dumps(model["train_summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
