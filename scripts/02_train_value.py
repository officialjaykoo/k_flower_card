#!/usr/bin/env python3
import argparse
import array
import glob
import hashlib
import json
import math
import os
import random
import struct
import sys
from datetime import datetime, timezone

try:
    import torch  # type: ignore
except Exception:
    torch = None

try:
    import lmdb  # type: ignore
except Exception:
    lmdb = None

SIDE_MY = "mySide"
SIDE_YOUR = "yourSide"
DEFAULT_VALUE_FILTER_RULES_PATH = "configs/value_filter_rules.json"
DEFAULT_VALUE_FILTER_RULES = {
    "min_candidate_count": 2,
    "keep_if_action_type_in": ["declare_bomb"],
    "keep_if_trigger_prefixes": ["specialEvent", "bomb", "riskShift"],
    "keep_if_trigger_names": ["comboThreatEnter", "terminalContext", "goStopOption", "shakingYesOption"],
}
DEFAULT_VALUE_FILTER_BRANCH = "default"
VALUE_FILTER_BRANCH_RULES = {
    "default": {},
    "bak": {
        "min_candidate_count": 2,
        "keep_if_action_type_in": ["declare_bomb"],
        "keep_if_trigger_prefixes": ["specialEvent", "bomb", "riskShift"],
        "keep_if_trigger_names": ["comboThreatEnter", "terminalContext", "goStopOption", "shakingYesOption"],
    },
    "combo": {
        "min_candidate_count": 2,
        "keep_if_action_type_in": ["declare_bomb"],
        "keep_if_trigger_prefixes": ["specialEvent", "bomb", "riskShift", "combo"],
        "keep_if_trigger_names": [
            "gwangThreatShift",
            "comboProgressStep",
            "comboComplete",
            "comboThreatCleared",
            "comboHoldGain",
            "terminalContext",
            "goStopOption",
            "shakingYesOption",
        ],
    },
    "meta": {
        "min_candidate_count": 2,
        "keep_if_action_type_in": ["declare_bomb"],
        "keep_if_trigger_prefixes": ["specialEvent", "bomb", "riskShift", "meta"],
        "keep_if_trigger_names": [
            "terminalContext",
            "goStopOption",
            "shakingYesOption",
            "comboThreatEnter",
            "metaHighMultiplierGoStop",
            "metaTerminalGoStop",
            "metaHighMultiplierWindow",
            "metaTerminalWindow",
        ],
    },
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


def stable_hash(token, dim):
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % dim


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


def _dedup_keep_order(items):
    seen = set()
    out = []
    for v in items:
        s = str(v or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _normalize_filter_branch_name(name):
    raw = str(name or DEFAULT_VALUE_FILTER_BRANCH).strip().lower()
    if raw in ("", "base"):
        return "default"
    if raw not in ("default", "bak", "combo", "meta", "auto"):
        raise RuntimeError(f"Unsupported --filter-branch: {name}")
    return raw


def _merge_filter_rules(base_rules, branch_rules):
    base = _normalize_value_filter_rules(base_rules)
    if not isinstance(branch_rules, dict) or not branch_rules:
        return base
    branch = _normalize_value_filter_rules(branch_rules)
    merged = dict(base)
    if "min_candidate_count" in branch:
        try:
            merged["min_candidate_count"] = max(1, int(branch.get("min_candidate_count") or merged["min_candidate_count"]))
        except Exception:
            pass
    merged["keep_if_action_type_in"] = _dedup_keep_order(
        _as_str_list(base.get("keep_if_action_type_in")) + _as_str_list(branch.get("keep_if_action_type_in"))
    )
    merged["keep_if_trigger_prefixes"] = _dedup_keep_order(
        _as_str_list(base.get("keep_if_trigger_prefixes")) + _as_str_list(branch.get("keep_if_trigger_prefixes"))
    )
    merged["keep_if_trigger_names"] = _dedup_keep_order(
        _as_str_list(base.get("keep_if_trigger_names")) + _as_str_list(branch.get("keep_if_trigger_names"))
    )
    return _normalize_value_filter_rules(merged)


def _branch_for_line(line, requested_branch):
    branch = _normalize_filter_branch_name(requested_branch)
    if branch != "auto":
        return branch
    sim = str((line or {}).get("simulator") or "").strip().lower()
    if "bak_bundle" in sim:
        return "bak"
    if "combo_bundle" in sim:
        return "combo"
    if "meta_bundle" in sim:
        return "meta"
    return "default"


def _normalize_value_filter_rules(raw):
    rules = dict(DEFAULT_VALUE_FILTER_RULES)
    if isinstance(raw, dict):
        min_cc = _to_int_or_none(raw.get("min_candidate_count"))
        if min_cc is not None:
            rules["min_candidate_count"] = max(1, min_cc)
        keep_actions = _as_str_list(raw.get("keep_if_action_type_in"))
        if keep_actions:
            rules["keep_if_action_type_in"] = keep_actions
        keep_prefixes = _as_str_list(raw.get("keep_if_trigger_prefixes"))
        if keep_prefixes:
            rules["keep_if_trigger_prefixes"] = keep_prefixes
        keep_names = _as_str_list(raw.get("keep_if_trigger_names"))
        if keep_names:
            rules["keep_if_trigger_names"] = keep_names
    return rules


def load_value_filter_rules(path):
    p = str(path or "").strip()
    if not p:
        return dict(DEFAULT_VALUE_FILTER_RULES), None
    if not os.path.exists(p):
        return dict(DEFAULT_VALUE_FILTER_RULES), None
    with open(p, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)
    return _normalize_value_filter_rules(raw), os.path.abspath(p)


def choose_label(trace):
    dt = trace.get("dt")
    if dt == "play":
        return trace.get("c"), "play"
    if dt == "match":
        return trace.get("s"), "match"
    if dt == "option":
        return action_alias(trace.get("at")), "option"

    sp = trace.get("sp") or {}
    if sp.get("cards"):
        return trace.get("c"), "play"
    if sp.get("boardCardIds"):
        return trace.get("s"), "match"
    if sp.get("options"):
        return action_alias(trace.get("at")), "option"
    return None, None


def is_value_exception_trace(trace, filter_rules):
    if not isinstance(trace, dict):
        return False
    rules = _normalize_value_filter_rules(filter_rules)
    action_set = set(_as_str_list(rules.get("keep_if_action_type_in")))
    prefix_list = _as_str_list(rules.get("keep_if_trigger_prefixes"))
    name_set = set(_as_str_list(rules.get("keep_if_trigger_names")))

    action_type = str(trace.get("at") or "")
    if action_type in action_set:
        return True
    tg = trace.get("tg")
    if not isinstance(tg, list):
        return False
    for raw in tg:
        name = str(raw or "")
        if not name:
            continue
        if name in name_set:
            return True
        if any(name.startswith(prefix) for prefix in prefix_list):
            return True
    return False


def trace_passes_value_filter(trace, filter_rules):
    rules = _normalize_value_filter_rules(filter_rules)
    min_cc = int(rules.get("min_candidate_count") or 1)
    cc = _to_int_or_none((trace or {}).get("cc"))
    if cc is None:
        return True
    if cc >= min_cc:
        return True
    return is_value_exception_trace(trace, rules)


def extract_numeric(trace):
    dc = trace.get("dc") or {}
    sp = trace.get("sp") or {}
    cands = int(trace.get("cc") or 0)
    if cands <= 0:
        cands = len(sp.get("cards") or sp.get("boardCardIds") or sp.get("options") or [])
    order = str(trace.get("o") or "").strip().lower()
    if order in ("first", "second"):
        is_first = 1.0 if order == "first" else 0.0
    else:
        is_first = float(1 if int(dc.get("isFirstAttacker") or 0) else 0)
    return {
        "deck_count": float(dc.get("deckCount") or 0),
        "hand_self": float(dc.get("handCountSelf") or 0),
        "hand_opp": float(dc.get("handCountOpp") or 0),
        "go_self": float(dc.get("goCountSelf") or 0),
        "go_opp": float(dc.get("goCountOpp") or 0),
        "is_first_attacker": is_first,
        "cand_count": float(cands),
        "immediate_reward": float(trace.get("ir") or 0),
    }


def extract_tokens(trace, decision_type, action_label):
    dc = trace.get("dc") or {}
    order = str(trace.get("o") or "").strip().lower()
    if order not in ("first", "second"):
        order = "first" if int(dc.get("isFirstAttacker") or 0) else "second"
    out = [
        f"phase={dc.get('phase','?')}",
        f"order={order}",
        f"decision_type={decision_type}",
        f"action={action_label or '?'}",
        f"deck_bucket={int((dc.get('deckCount') or 0)//3)}",
        f"self_hand={int(dc.get('handCountSelf') or 0)}",
        f"opp_hand={int(dc.get('handCountOpp') or 0)}",
        f"self_go={int(dc.get('goCountSelf') or 0)}",
        f"opp_go={int(dc.get('goCountOpp') or 0)}",
        f"is_first_attacker={1 if order == 'first' else 0}",
    ]
    return out


def _float_or_none(v):
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


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


def iter_samples(
    paths,
    gold_per_point,
    target_mode="gold",
    max_samples=None,
    filter_rules=None,
    filter_branch=DEFAULT_VALUE_FILTER_BRANCH,
):
    yielded = 0
    base_rules = _normalize_value_filter_rules(filter_rules)
    branch_rule_cache = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line_raw in f:
                line_raw = line_raw.strip()
                if not line_raw:
                    continue
                line = json.loads(line_raw)
                line_branch = _branch_for_line(line, filter_branch)
                active_rules = branch_rule_cache.get(line_branch)
                if active_rules is None:
                    branch_rules = VALUE_FILTER_BRANCH_RULES.get(line_branch, {})
                    active_rules = _merge_filter_rules(base_rules, branch_rules)
                    branch_rule_cache[line_branch] = active_rules
                for trace in line.get("decision_trace") or []:
                    if not trace_passes_value_filter(trace, active_rules):
                        continue
                    actor = trace.get("a")
                    if actor not in (SIDE_MY, SIDE_YOUR):
                        continue
                    action_label, decision_type = choose_label(trace)
                    if action_label is None:
                        continue
                    y = target_gold(line, actor, gold_per_point, target_mode=target_mode)
                    if y is None:
                        continue
                    yield {
                        "tokens": extract_tokens(trace, decision_type, action_label),
                        "numeric": extract_numeric(trace),
                        "y": y,
                        "actor": actor,
                    }
                    yielded += 1
                    if max_samples is not None and yielded >= max_samples:
                        return


def sparse_features(sample, dim, numeric_scale):
    x = {}
    for tok in sample["tokens"]:
        idx = stable_hash(f"tok:{tok}", dim)
        x[idx] = x.get(idx, 0.0) + 1.0
    for k, v in sample["numeric"].items():
        idx = stable_hash(f"num:{k}", dim)
        scaled = v / max(1e-9, numeric_scale.get(k, 1.0))
        x[idx] = x.get(idx, 0.0) + scaled
    idxs = list(x.keys())
    vals = [x[i] for i in idxs]
    return idxs, vals


def split_is_valid(sample_index, valid_ratio, seed):
    if valid_ratio <= 0:
        return False
    if valid_ratio >= 1:
        return True
    digest = hashlib.md5(f"{seed}:{sample_index}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], byteorder="big", signed=False)
    return (bucket / 4294967296.0) < valid_ratio


def input_manifest(paths):
    out = []
    for p in paths:
        ap = os.path.abspath(p)
        st = os.stat(ap)
        out.append({"path": ap, "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)})
    return out


def scan_dataset_stats(
    paths,
    gold_per_point,
    target_mode,
    max_samples,
    valid_ratio,
    seed,
    numeric_keys,
    filter_rules,
    filter_branch,
):
    numeric_max_abs = {k: 1.0 for k in numeric_keys}
    total = 0
    train_count = 0
    valid_count = 0
    for s in iter_samples(
        paths, gold_per_point, target_mode, max_samples, filter_rules, filter_branch
    ):
        is_valid = split_is_valid(total, valid_ratio, seed)
        if is_valid:
            valid_count += 1
        else:
            train_count += 1
            for k in numeric_keys:
                v = abs(float(s["numeric"].get(k, 0.0)))
                if v > numeric_max_abs[k]:
                    numeric_max_abs[k] = v
        total += 1
    return {
        "samples_total": total,
        "samples_train": train_count,
        "samples_valid": valid_count,
        "numeric_scale": {k: max(1.0, float(v)) for k, v in numeric_max_abs.items()},
    }


def iter_sparse_source_samples(
    paths,
    gold_per_point,
    target_mode,
    max_samples,
    dim,
    numeric_scale,
    valid_ratio,
    seed,
    subset,
    filter_rules,
    filter_branch,
):
    for idx, sample in enumerate(
        iter_samples(paths, gold_per_point, target_mode, max_samples, filter_rules, filter_branch)
    ):
        is_valid = split_is_valid(idx, valid_ratio, seed)
        split = "valid" if is_valid else "train"
        if subset is not None and split != subset:
            continue
        idxs, vals = sparse_features(sample, dim, numeric_scale)
        yield idxs, vals, float(sample["y"])


def _cache_meta_path(cache_path):
    return f"{cache_path}.meta.json"


def _ensure_lmdb_available():
    if lmdb is None:
        raise RuntimeError(
            "cache-backend=lmdb requires python package 'lmdb'. Install with: pip install lmdb"
        )


def _cache_data_exists(cache_path, backend):
    if backend == "lmdb":
        return os.path.isdir(cache_path)
    return os.path.exists(cache_path)


def load_cache_meta(cache_path, backend):
    meta_path = _cache_meta_path(cache_path)
    if not _cache_data_exists(cache_path, backend) or not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def cache_matches(meta, config, manifest):
    if not isinstance(meta, dict):
        return False
    return meta.get("config") == config and meta.get("input_manifest") == manifest


def _pack_cached_record(split_code, y, idxs, vals):
    n = len(idxs)
    if n >= 65535:
        raise RuntimeError(f"Feature count too large for cache record: {n}")
    flag = 1 if split_code == "v" else 0
    header = struct.pack("<BfH", flag, float(y), n)

    a_idx = array.array("I", (int(i) for i in idxs))
    if sys.byteorder != "little":
        a_idx.byteswap()

    a_val = array.array("f", (float(v) for v in vals))
    if sys.byteorder != "little":
        a_val.byteswap()

    return header + a_idx.tobytes() + a_val.tobytes()


def _unpack_cached_record(blob):
    flag, y, n = struct.unpack_from("<BfH", blob, 0)
    off = 7
    idx_bytes = int(n) * 4
    val_bytes = int(n) * 4

    a_idx = array.array("I")
    if idx_bytes > 0:
        a_idx.frombytes(blob[off : off + idx_bytes])
    if sys.byteorder != "little":
        a_idx.byteswap()
    off += idx_bytes

    a_val = array.array("f")
    if val_bytes > 0:
        a_val.frombytes(blob[off : off + val_bytes])
    if sys.byteorder != "little":
        a_val.byteswap()

    split_code = "v" if int(flag) == 1 else "t"
    return split_code, float(y), list(a_idx), list(a_val)


def build_feature_cache_jsonl(
    cache_path,
    paths,
    gold_per_point,
    target_mode,
    max_samples,
    dim,
    numeric_scale,
    valid_ratio,
    seed,
    filter_rules,
    filter_branch,
):
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    counts = {"samples_total": 0, "samples_train": 0, "samples_valid": 0}
    with open(cache_path, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(
            iter_samples(paths, gold_per_point, target_mode, max_samples, filter_rules, filter_branch)
        ):
            is_valid = split_is_valid(idx, valid_ratio, seed)
            split_code = "v" if is_valid else "t"
            idxs, vals = sparse_features(sample, dim, numeric_scale)
            rec = {
                "s": split_code,
                "y": float(sample["y"]),
                "i": idxs,
                "v": vals,
            }
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts["samples_total"] += 1
            if is_valid:
                counts["samples_valid"] += 1
            else:
                counts["samples_train"] += 1
    return counts


def build_feature_cache_lmdb(
    cache_path,
    paths,
    gold_per_point,
    target_mode,
    max_samples,
    dim,
    numeric_scale,
    valid_ratio,
    seed,
    map_size_bytes,
    filter_rules,
    filter_branch,
):
    _ensure_lmdb_available()
    os.makedirs(cache_path, exist_ok=True)
    counts = {"samples_total": 0, "samples_train": 0, "samples_valid": 0}
    env = lmdb.open(
        cache_path,
        subdir=True,
        create=True,
        lock=True,
        readahead=False,
        meminit=False,
        map_size=int(map_size_bytes),
        max_dbs=1,
    )
    try:
        txn = env.begin(write=True)
        try:
            for idx, sample in enumerate(
                iter_samples(paths, gold_per_point, target_mode, max_samples, filter_rules, filter_branch)
            ):
                is_valid = split_is_valid(idx, valid_ratio, seed)
                split_code = "v" if is_valid else "t"
                idxs, vals = sparse_features(sample, dim, numeric_scale)
                val = _pack_cached_record(split_code, float(sample["y"]), idxs, vals)
                key = b"d" + int(idx).to_bytes(8, byteorder="big", signed=False)
                txn.put(key, val)

                counts["samples_total"] += 1
                if is_valid:
                    counts["samples_valid"] += 1
                else:
                    counts["samples_train"] += 1

                if counts["samples_total"] % 5000 == 0:
                    txn.commit()
                    txn = env.begin(write=True)
            txn.commit()
        except Exception:
            txn.abort()
            raise
    finally:
        env.sync()
        env.close()
    return counts


def build_feature_cache(
    cache_backend,
    cache_path,
    paths,
    gold_per_point,
    target_mode,
    max_samples,
    dim,
    numeric_scale,
    valid_ratio,
    seed,
    lmdb_map_size_bytes,
    filter_rules,
    filter_branch,
):
    if cache_backend == "lmdb":
        return build_feature_cache_lmdb(
            cache_path,
            paths,
            gold_per_point,
            target_mode,
            max_samples,
            dim,
            numeric_scale,
            valid_ratio,
            seed,
            lmdb_map_size_bytes,
            filter_rules,
            filter_branch,
        )
    return build_feature_cache_jsonl(
        cache_path,
        paths,
        gold_per_point,
        target_mode,
        max_samples,
        dim,
        numeric_scale,
        valid_ratio,
        seed,
        filter_rules,
        filter_branch,
    )


def iter_cached_sparse_samples_jsonl(cache_path, subset):
    want = None
    if subset == "train":
        want = "t"
    elif subset == "valid":
        want = "v"
    with open(cache_path, "r", encoding="utf-8") as f:
        for line_raw in f:
            line_raw = line_raw.strip()
            if not line_raw:
                continue
            try:
                rec = json.loads(line_raw)
            except Exception:
                continue
            if want is not None and rec.get("s") != want:
                continue
            idxs = rec.get("i") or []
            vals = rec.get("v") or []
            y = float(rec.get("y") or 0.0)
            yield idxs, vals, y


def iter_cached_sparse_samples_lmdb(cache_path, subset):
    _ensure_lmdb_available()
    want = None
    if subset == "train":
        want = "t"
    elif subset == "valid":
        want = "v"
    env = lmdb.open(
        cache_path,
        subdir=True,
        create=False,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
        max_dbs=1,
    )
    try:
        with env.begin(write=False, buffers=True) as txn:
            cursor = txn.cursor()
            for key, val in cursor:
                if not key or key[:1] != b"d":
                    continue
                split_code, y, idxs, vals = _unpack_cached_record(bytes(val))
                if want is not None and split_code != want:
                    continue
                yield idxs, vals, y
    finally:
        env.close()


def iter_cached_sparse_samples(cache_backend, cache_path, subset):
    if cache_backend == "lmdb":
        return iter_cached_sparse_samples_lmdb(cache_path, subset)
    return iter_cached_sparse_samples_jsonl(cache_path, subset)


def iter_with_shuffle_buffer(base_iter, buffer_size, seed):
    if buffer_size is None or int(buffer_size) <= 1:
        for item in base_iter:
            yield item
        return
    rng = random.Random(seed)
    buf = []
    cap = int(buffer_size)
    for item in base_iter:
        buf.append(item)
        if len(buf) >= cap:
            rng.shuffle(buf)
            while buf:
                yield buf.pop()
    if buf:
        rng.shuffle(buf)
        while buf:
            yield buf.pop()


def predict_sparse_cpu(weights, bias, idxs, vals):
    out = bias
    for i, v in zip(idxs, vals):
        out += weights[i] * v
    return out


def evaluate_cpu_streaming(weight, bias, sample_iter):
    n = 0
    se = 0.0
    ae = 0.0
    for idxs, vals, y in sample_iter:
        p = predict_sparse_cpu(weight, bias, idxs, vals)
        e = p - y
        se += e * e
        ae += abs(e)
        n += 1
    if n == 0:
        return 0.0, 0.0, 0.0
    mse = se / n
    return mse, math.sqrt(mse), ae / n


def train_cpu_streaming(
    train_iter_epoch_factory,
    train_eval_iter_factory,
    valid_iter_factory,
    dim,
    epochs,
    lr,
    l2,
):
    w = [0.0] * dim
    b = 0.0
    for epoch in range(epochs):
        for idxs, vals, y in train_iter_epoch_factory(epoch):
            p = predict_sparse_cpu(w, b, idxs, vals)
            err = p - y
            b -= lr * err
            for i, xv in zip(idxs, vals):
                grad = err * xv + l2 * w[i]
                w[i] -= lr * grad

    train_mse, train_rmse, train_mae = evaluate_cpu_streaming(w, b, train_eval_iter_factory())
    valid_mse, valid_rmse, valid_mae = evaluate_cpu_streaming(w, b, valid_iter_factory())
    return w, b, train_mse, train_rmse, train_mae, valid_mse, valid_rmse, valid_mae


def resolve_cuda_device():
    if torch is None:
        raise RuntimeError("GPU mode requires PyTorch, but it is not installed.")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU mode requires CUDA, but torch.cuda.is_available() is False.")
    return "cuda"


def build_batch_tensors_from_samples(batch_features, batch_y, device):
    rows = []
    cols = []
    vals = []
    for r, (idxs, vls) in enumerate(batch_features):
        if idxs:
            rows.extend([r] * len(idxs))
            cols.extend(idxs)
            vals.extend(vls)

    if rows:
        rows_t = torch.tensor(rows, dtype=torch.long, device=device)
        cols_t = torch.tensor(cols, dtype=torch.long, device=device)
        vals_t = torch.tensor(vals, dtype=torch.float32, device=device)
    else:
        rows_t = torch.empty((0,), dtype=torch.long, device=device)
        cols_t = torch.empty((0,), dtype=torch.long, device=device)
        vals_t = torch.empty((0,), dtype=torch.float32, device=device)
    y_t = torch.tensor(batch_y, dtype=torch.float32, device=device)
    return rows_t, cols_t, vals_t, y_t


def predict_batch_torch(w, b, rows, cols, vals, batch_size):
    pred = torch.full((batch_size,), float(b.item()), dtype=torch.float32, device=w.device)
    if rows.numel() > 0:
        contrib = w.index_select(0, cols) * vals
        pred.scatter_add_(0, rows, contrib)
    return pred


def evaluate_torch_streaming(sample_iter, w, b, batch_size, device):
    se = 0.0
    ae = 0.0
    n = 0
    batch_features = []
    batch_y = []
    with torch.no_grad():
        for idxs, vals, y in sample_iter:
            batch_features.append((idxs, vals))
            batch_y.append(float(y))
            if len(batch_features) < batch_size:
                continue
            rows, cols, vals_t, y_t = build_batch_tensors_from_samples(batch_features, batch_y, device)
            pred = predict_batch_torch(w, b, rows, cols, vals_t, len(batch_features))
            err = pred - y_t
            se += float(torch.sum(err * err).item())
            ae += float(torch.sum(torch.abs(err)).item())
            n += len(batch_features)
            batch_features.clear()
            batch_y.clear()
        if batch_features:
            rows, cols, vals_t, y_t = build_batch_tensors_from_samples(batch_features, batch_y, device)
            pred = predict_batch_torch(w, b, rows, cols, vals_t, len(batch_features))
            err = pred - y_t
            se += float(torch.sum(err * err).item())
            ae += float(torch.sum(torch.abs(err)).item())
            n += len(batch_features)
    if n <= 0:
        return 0.0, 0.0, 0.0
    mse = se / n
    return mse, math.sqrt(mse), ae / n


def train_torch_streaming(
    train_iter_epoch_factory,
    train_eval_iter_factory,
    valid_iter_factory,
    train_count,
    dim,
    epochs,
    lr,
    l2,
    seed,
    batch_size,
    progress_every,
    device,
):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True

    w = torch.zeros((dim,), dtype=torch.float32, device=device, requires_grad=True)
    b = torch.zeros((1,), dtype=torch.float32, device=device, requires_grad=True)

    opt = torch.optim.SGD([w, b], lr=lr, weight_decay=l2)

    for epoch in range(epochs):
        num_batches = (max(1, int(train_count)) + batch_size - 1) // batch_size
        bi = 0
        seen = 0
        batch_features = []
        batch_y = []
        for idxs, vals, y in train_iter_epoch_factory(epoch):
            batch_features.append((idxs, vals))
            batch_y.append(float(y))
            if len(batch_features) < batch_size:
                continue
            bi += 1
            seen += len(batch_features)
            rows, cols, vals_t, y_t = build_batch_tensors_from_samples(batch_features, batch_y, device)

            opt.zero_grad(set_to_none=True)
            pred = predict_batch_torch(w, b, rows, cols, vals_t, len(batch_features))
            loss = torch.mean((pred - y_t) ** 2)
            loss.backward()
            opt.step()
            batch_features.clear()
            batch_y.clear()

            if progress_every > 0 and (bi % progress_every == 0 or bi == num_batches):
                print(
                    f"epoch {epoch + 1}/{epochs} | batch {bi}/{num_batches} | "
                    f"samples {seen}/{int(train_count)} | loss {float(loss.item()):.4f}"
                )
        if batch_features:
            bi += 1
            seen += len(batch_features)
            rows, cols, vals_t, y_t = build_batch_tensors_from_samples(batch_features, batch_y, device)
            opt.zero_grad(set_to_none=True)
            pred = predict_batch_torch(w, b, rows, cols, vals_t, len(batch_features))
            loss = torch.mean((pred - y_t) ** 2)
            loss.backward()
            opt.step()
            if progress_every > 0:
                print(
                    f"epoch {epoch + 1}/{epochs} | batch {bi}/{num_batches} | "
                    f"samples {seen}/{int(train_count)} | loss {float(loss.item()):.4f}"
                )

    with torch.no_grad():
        train_mse, train_rmse, train_mae = evaluate_torch_streaming(
            train_eval_iter_factory(), w, b, batch_size, device
        )
        valid_mse, valid_rmse, valid_mae = evaluate_torch_streaming(
            valid_iter_factory(), w, b, batch_size, device
        )
        w_out = w.detach().cpu().tolist()
        b_out = float(b.detach().cpu().item())

    return w_out, b_out, train_mse, train_rmse, train_mae, valid_mse, valid_rmse, valid_mae


def main():
    parser = argparse.ArgumentParser(description="Train value regressor (expected gold).")
    parser.add_argument("--input", nargs="+", default=["logs/*.jsonl"])
    parser.add_argument("--output", default="models/value-model.json")
    parser.add_argument("--gold-per-point", type=float, default=100.0)
    parser.add_argument(
        "--target-mode",
        choices=["gold", "score"],
        default="gold",
        help="Value target mode: gold (default, uses gold delta) or score (point diff * gold-per-point).",
    )
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--progress-every", type=int, default=0, help="Print every N batches (torch backend). 0 disables.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of training samples.")
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio in [0,1], deterministic hash split.",
    )
    parser.add_argument(
        "--sample-cache",
        default="auto",
        help="Sparse sample cache path. Use 'auto' to place near output model, or 'none' to disable cache.",
    )
    parser.add_argument(
        "--cache-backend",
        choices=["jsonl", "lmdb"],
        default="jsonl",
        help="Sample cache backend: jsonl (default) or lmdb.",
    )
    parser.add_argument(
        "--lmdb-map-size-gb",
        type=float,
        default=8.0,
        help="LMDB map size in GB (used only when --cache-backend lmdb).",
    )
    parser.add_argument("--rebuild-cache", action="store_true", help="Force rebuilding sample cache.")
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=4096,
        help="Shuffle buffer size for streaming training (0/1 disables shuffle).",
    )
    parser.add_argument(
        "--filter-rules",
        default=DEFAULT_VALUE_FILTER_RULES_PATH,
        help=f"Value filter rules JSON path (default: {DEFAULT_VALUE_FILTER_RULES_PATH}).",
    )
    parser.add_argument(
        "--filter-branch",
        choices=["default", "bak", "combo", "meta", "auto"],
        default=DEFAULT_VALUE_FILTER_BRANCH,
        help="Value filter branch profile for cc<2 exception handling.",
    )
    args = parser.parse_args()

    if not (0.0 <= float(args.valid_ratio) <= 1.0):
        raise RuntimeError("--valid-ratio must be in [0, 1].")
    if float(args.lmdb_map_size_gb) <= 0:
        raise RuntimeError("--lmdb-map-size-gb must be > 0.")
    if args.cache_backend == "lmdb":
        _ensure_lmdb_available()

    input_paths = expand_inputs(args.input)
    filter_rules, filter_rules_path = load_value_filter_rules(args.filter_rules)
    filter_branch = _normalize_filter_branch_name(args.filter_branch)
    keys = [
        "deck_count",
        "hand_self",
        "hand_opp",
        "go_self",
        "go_opp",
        "is_first_attacker",
        "cand_count",
        "immediate_reward",
    ]

    cache_arg = str(args.sample_cache or "").strip()
    cache_disabled = cache_arg.lower() in ("", "none", "off", "no", "0")
    if cache_disabled:
        cache_path = None
    elif cache_arg.lower() == "auto":
        base = args.output[:-5] if str(args.output).lower().endswith(".json") else args.output
        if args.cache_backend == "lmdb":
            cache_path = f"{base}.samples.cache.lmdb"
        else:
            cache_path = f"{base}.samples.cache.jsonl"
    else:
        cache_path = cache_arg

    manifest = input_manifest(input_paths)
    lmdb_map_size_bytes = int(float(args.lmdb_map_size_gb) * (1024**3))
    cache_config = {
        "format_version": "value_sparse_cache_v1",
        "cache_backend": args.cache_backend,
        "dim": int(args.dim),
        "gold_per_point": float(args.gold_per_point),
        "target_mode": str(args.target_mode),
        "max_samples": args.max_samples,
        "valid_ratio": float(args.valid_ratio),
        "seed": int(args.seed),
        "lmdb_map_size_bytes": lmdb_map_size_bytes if args.cache_backend == "lmdb" else None,
        "filter_rules": filter_rules,
        "filter_branch": filter_branch,
        "filter_branch_rules": VALUE_FILTER_BRANCH_RULES if filter_branch == "auto" else VALUE_FILTER_BRANCH_RULES.get(filter_branch, {}),
    }
    numeric_scale = None
    counts = None
    cache_ready = False

    if cache_path and not args.rebuild_cache:
        meta = load_cache_meta(cache_path, args.cache_backend)
        if cache_matches(meta, cache_config, manifest):
            numeric_scale = meta.get("numeric_scale")
            counts = meta.get("counts")
            cache_ready = isinstance(numeric_scale, dict) and isinstance(counts, dict)
            if cache_ready:
                print(f"cache: reusing {cache_path}")

    if not cache_ready:
        stats = scan_dataset_stats(
            input_paths,
            args.gold_per_point,
            args.target_mode,
            args.max_samples,
            args.valid_ratio,
            args.seed,
            keys,
            filter_rules,
            filter_branch,
        )
        if int(stats["samples_total"]) <= 0:
            raise RuntimeError("No trainable value samples found.")
        numeric_scale = stats["numeric_scale"]
        counts = {
            "samples_total": int(stats["samples_total"]),
            "samples_train": int(stats["samples_train"]),
            "samples_valid": int(stats["samples_valid"]),
        }
        if cache_path:
            print(f"cache: building {cache_path}")
            built_counts = build_feature_cache(
                args.cache_backend,
                cache_path,
                input_paths,
                args.gold_per_point,
                args.target_mode,
                args.max_samples,
                args.dim,
                numeric_scale,
                args.valid_ratio,
                args.seed,
                lmdb_map_size_bytes,
                filter_rules,
                filter_branch,
            )
            counts = {
                "samples_total": int(built_counts["samples_total"]),
                "samples_train": int(built_counts["samples_train"]),
                "samples_valid": int(built_counts["samples_valid"]),
            }
            meta_out = {
                "format_version": "value_sparse_cache_v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "config": cache_config,
                "input_manifest": manifest,
                "numeric_scale": numeric_scale,
                "counts": counts,
            }
            with open(_cache_meta_path(cache_path), "w", encoding="utf-8") as f:
                json.dump(meta_out, f, ensure_ascii=False, separators=(",", ":"))
            cache_ready = True

    train_count = int((counts or {}).get("samples_train") or 0)
    valid_count = int((counts or {}).get("samples_valid") or 0)
    total_count = int((counts or {}).get("samples_total") or 0)
    if total_count <= 0 or train_count <= 0:
        raise RuntimeError("No trainable value samples found after split.")

    shuffle_buf = max(0, int(args.shuffle_buffer))

    def make_train_iter_for_epoch(epoch):
        shuffle_seed = int(args.seed) + 1000003 * (int(epoch) + 1)
        if cache_ready and cache_path:
            base = iter_cached_sparse_samples(args.cache_backend, cache_path, "train")
        else:
            base = iter_sparse_source_samples(
                input_paths,
                args.gold_per_point,
                args.target_mode,
                args.max_samples,
                args.dim,
                numeric_scale,
                args.valid_ratio,
                args.seed,
                "train",
                filter_rules,
                filter_branch,
            )
        return iter_with_shuffle_buffer(base, shuffle_buf, shuffle_seed)

    def make_train_eval_iter():
        if cache_ready and cache_path:
            return iter_cached_sparse_samples(args.cache_backend, cache_path, "train")
        return iter_sparse_source_samples(
            input_paths,
            args.gold_per_point,
            args.target_mode,
            args.max_samples,
            args.dim,
            numeric_scale,
            args.valid_ratio,
            args.seed,
            "train",
            filter_rules,
            filter_branch,
        )

    def make_valid_eval_iter():
        if cache_ready and cache_path:
            return iter_cached_sparse_samples(args.cache_backend, cache_path, "valid")
        return iter_sparse_source_samples(
            input_paths,
            args.gold_per_point,
            args.target_mode,
            args.max_samples,
            args.dim,
            numeric_scale,
            args.valid_ratio,
            args.seed,
            "valid",
            filter_rules,
            filter_branch,
        )

    if args.device == "cpu":
        backend_used = "cpu"
        actual_device = "cpu"
        print(f"training backend=cpu device=cpu samples={train_count}/{valid_count} (train/valid)")
        w, b, train_mse, train_rmse, train_mae, valid_mse, valid_rmse, valid_mae = train_cpu_streaming(
            make_train_iter_for_epoch,
            make_train_eval_iter,
            make_valid_eval_iter,
            args.dim,
            args.epochs,
            args.lr,
            args.l2,
        )
    else:
        backend_used = "torch"
        dev = resolve_cuda_device()
        actual_device = dev
        print(
            f"training backend=torch device={dev} batch_size={args.batch_size} "
            f"samples={train_count}/{valid_count} (train/valid)"
        )
        w, b, train_mse, train_rmse, train_mae, valid_mse, valid_rmse, valid_mae = train_torch_streaming(
            make_train_iter_for_epoch,
            make_train_eval_iter,
            make_valid_eval_iter,
            train_count,
            args.dim,
            args.epochs,
            args.lr,
            args.l2,
            args.seed,
            args.batch_size,
            args.progress_every,
            dev,
        )

    model = {
        "model_type": "value_linear_hash_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dim": args.dim,
        "weights": w,
        "bias": b,
        "numeric_scale": numeric_scale,
        "gold_per_point": args.gold_per_point,
        "target_mode": args.target_mode,
        "feature_spec": {
            "categorical_tokens": [
                "phase",
                "order",
                "decision_type",
                "action",
                "deck_bucket",
                "self_hand",
                "opp_hand",
                "self_go",
                "opp_go",
                "is_first_attacker",
            ],
            "numeric_keys": keys,
            "hash_dim": args.dim,
        },
        "train_summary": {
            "samples_total": total_count,
            "samples_train": train_count,
            "samples_valid": valid_count,
            "train_mse": train_mse,
            "train_rmse": train_rmse,
            "train_mae": train_mae,
            "valid_mse": valid_mse,
            "valid_rmse": valid_rmse,
            "valid_mae": valid_mae,
            "input_files": input_paths,
            "epochs": args.epochs,
            "lr": args.lr,
            "l2": args.l2,
            "backend": backend_used,
            "device": actual_device,
            "batch_size": args.batch_size,
            "max_samples": args.max_samples,
            "valid_ratio": args.valid_ratio,
            "target_mode": args.target_mode,
            "shuffle_buffer": shuffle_buf,
            "cache_path": cache_path,
            "cache_enabled": bool(cache_path),
            "cache_used": bool(cache_ready and cache_path),
            "cache_backend": args.cache_backend,
            "lmdb_map_size_gb": args.lmdb_map_size_gb if args.cache_backend == "lmdb" else None,
            "filter_rules": filter_rules,
            "filter_rules_path": filter_rules_path,
            "filter_branch": filter_branch,
            "filter_branch_rules": VALUE_FILTER_BRANCH_RULES if filter_branch == "auto" else VALUE_FILTER_BRANCH_RULES.get(filter_branch, {}),
        },
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False)

    print(f"trained value model -> {args.output}")
    print(json.dumps(model["train_summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
