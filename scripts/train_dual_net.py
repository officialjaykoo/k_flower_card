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

try:
    import lmdb  # type: ignore
except Exception:
    lmdb = None

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
    "hand_self",
    "hand_opp",
    "is_first_attacker",
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
    "is_first_attacker": 1.2,
}

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


def _ensure_lmdb_available():
    if lmdb is None:
        raise RuntimeError(
            "cache-backend=lmdb requires python package 'lmdb'. Install with: pip install lmdb"
        )


def _sample_key(index):
    return f"{index:09d}".encode("ascii")


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


def choose_label_and_candidates(trace):
    dt = str(trace.get("dt") or "")
    chosen_compact = trace.get("ch")
    if dt == "play" and chosen_compact is not None:
        chosen = str(chosen_compact)
        return chosen, "play", [chosen]
    if dt == "match" and chosen_compact is not None:
        chosen = str(chosen_compact)
        return chosen, "match", [chosen]
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
        chosen = action_alias(chosen_compact)
        if chosen in by_chosen:
            return str(chosen), "option", [str(x) for x in by_chosen[chosen]]
        phase = (trace.get("dc") or {}).get("p")
        by_phase = {
            "go-stop": ["go", "stop"],
            "president-choice": ["president_stop", "president_hold"],
            "gukjin-choice": ["five", "junk"],
            "shaking-confirm": ["shaking_yes", "shaking_no"],
            3: ["go", "stop"],
            4: ["president_stop", "president_hold"],
            5: ["five", "junk"],
            6: ["shaking_yes", "shaking_no"],
        }
        cands = by_phase.get(phase) or by_phase.get(str(phase))
        chosen = action_alias(chosen_compact)
        if cands and chosen in cands:
            return str(chosen), "option", [str(x) for x in cands]
        if chosen:
            return str(chosen), "option", [str(chosen)]
    return None, None, None


def _to_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)


def _prob01(v):
    out = _to_float(v, 0.0)
    if out > 1.0:
        out = out / 100.0
    if out < 0.0:
        return 0.0
    if out > 1.0:
        return 1.0
    return out


def extract_numeric(line, trace, actor):
    _ = line
    dc = trace.get("dc") or {}
    order = str(trace.get("o") or "").strip().lower()
    if order in ("first", "second"):
        is_first_attacker = 1.0 if order == "first" else 0.0
    else:
        is_first_attacker = 1.0 if actor == SIDE_MY else 0.0

    deck_count = float(dc.get("d") or 0.0)
    deck_end_ratio = max(0.0, min(1.0, 1.0 - (deck_count / 22.0)))

    hand_self_raw = float(dc.get("hs") or 0.0)
    hand_diff_raw = float(dc.get("hd") or 0.0)
    hand_opp_raw = hand_self_raw - hand_diff_raw
    hand_self = max(0.0, min(1.0, hand_self_raw / 10.0))
    hand_opp = max(0.0, min(1.0, hand_opp_raw / 10.0))

    go_self = float(dc.get("gs") or 0.0)
    go_opp = float(dc.get("go") or 0.0)
    score_diff = _to_float(dc.get("sd"), 0.0)
    score_self_now = max(0.0, score_diff)
    score_opp_now = max(0.0, -score_diff)

    opp_combo_threat = _prob01(dc.get("ojt"))
    opp_gwang_threat = _prob01(dc.get("ogt"))
    self_combo_threat = max(0.0, min(1.0, _to_float(dc.get("jps"), 0.0) / 8.0))
    self_gwang_threat = max(0.0, min(1.0, _to_float(dc.get("jas"), 0.0) / 3.0))

    can_match = 1.0 if abs(hand_diff_raw) <= 1 and hand_self_raw > 0 else 0.0
    high_value_match = 1.0 if (self_combo_threat >= 0.6 or self_gwang_threat >= 0.6) else 0.0
    jokbo_potential = max(self_combo_threat, self_gwang_threat)

    opp_danger_level = (
        0.30 * min(1.0, go_opp / 3.0)
        + 0.25 * min(1.0, max(0.0, 1.0 - hand_opp))
        + 0.25 * opp_combo_threat
        + 0.20 * opp_gwang_threat
    )
    opp_danger_level = max(opp_danger_level, min(1.0, score_opp_now / 7.0))
    opp_danger_level = max(0.0, min(1.0, opp_danger_level))

    my_win_progress = (
        0.30 * min(1.0, go_self / 3.0)
        + 0.25 * self_combo_threat
        + 0.20 * self_gwang_threat
        + 0.15 * can_match
        + 0.10 * min(1.0, max(0.0, 1.0 - hand_self))
    )
    my_win_progress = max(my_win_progress, min(1.0, score_self_now / 7.0))
    my_win_progress = max(0.0, min(1.0, my_win_progress))

    shake_multiplier_state = max(0.0, min(3.0, abs(_to_float(dc.get("gsd"), 0.0)) / 1000.0))
    return {
        "opp_danger_level": opp_danger_level,
        "my_win_progress": my_win_progress,
        "deck_end_ratio": deck_end_ratio,
        "high_value_match": high_value_match,
        "jokbo_potential": jokbo_potential,
        "can_match": can_match,
        "immediate_reward": float(trace.get("ir") or 0.0),
        "pi_bak_risk": _to_float(dc.get("rp"), 0.0),
        "gwang_bak_risk": _to_float(dc.get("rg"), 0.0),
        "mong_bak_risk": _to_float(dc.get("rm"), 0.0),
        "shake_multiplier_state": shake_multiplier_state,
        "hand_self": hand_self,
        "hand_opp": hand_opp,
        "is_first_attacker": is_first_attacker,
    }


def extract_tokens(line, trace, decision_type, actor):
    _ = line
    dc = trace.get("dc") or {}
    order = str(trace.get("o") or "").strip().lower()
    if order not in ("first", "second"):
        order = "first" if actor == SIDE_MY else "second"
    hand_self = int(dc.get("hs") or 0)
    hand_opp = int((dc.get("hs") or 0) - (dc.get("hd") or 0))
    score_diff = int(_to_float(dc.get("sd"), 0.0))
    actor_bak = int(_to_float(dc.get("rp"), 0.0) + _to_float(dc.get("rg"), 0.0) + _to_float(dc.get("rm"), 0.0))
    return [
        f"phase={dc.get('p', '?')}",
        f"order={order}",
        f"decision_type={decision_type}",
        f"deck_bucket={int((dc.get('d') or 0)//3)}",
        f"self_hand={hand_self}",
        f"opp_hand={hand_opp}",
        f"self_go={int(dc.get('gs') or 0)}",
        f"opp_go={int(dc.get('go') or 0)}",
        f"actor={actor}",
        f"score_diff_b={int(score_diff // 2)}",
        f"is_first_attacker={1 if order == 'first' else 0}",
        f"bak_signal={actor_bak}",
        f"jps={int(dc.get('jps') or 0)}",
        f"jpo={int(dc.get('jpo') or 0)}",
        f"jas={int(dc.get('jas') or 0)}",
        f"jao={int(dc.get('jao') or 0)}",
        f"ojt_b={int(_prob01(dc.get('ojt')) * 10)}",
        f"ogt_b={int(_prob01(dc.get('ogt')) * 10)}",
        f"gsd_b={int((_to_float(dc.get('gsd'), 0.0) / 1000.0) * 4)}",
    ]


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

    return math.tanh((diff * mult) / max(1e-9, float(value_scale)))


def iter_raw_records(input_paths, max_samples, value_scale):
    yielded = 0
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
                    yield {
                        "tokens": extract_tokens(line, trace, decision_type, actor),
                        "numeric": extract_numeric(line, trace, actor),
                        "candidates": candidates,
                        "chosen": chosen,
                        "value_target": float(tv),
                        "is_go_stop": int(chosen in GO_STOP_ACTIONS),
                        "decision_type": str(decision_type),
                    }
                    yielded += 1
                    if max_samples is not None and yielded >= max_samples:
                        return


def scan_cache_schema(input_paths, max_samples, value_scale):
    token_counter = {}
    action_set = set()
    numeric_absmax = {k: 1.0 for k in NUMERIC_KEYS}
    go_stop_count = 0
    decision_type_counts = {}
    total = 0

    for raw in iter_raw_records(input_paths, max_samples, value_scale):
        total += 1
        for tok in raw["tokens"]:
            token_counter[tok] = token_counter.get(tok, 0) + 1
        for c in raw["candidates"]:
            action_set.add(str(c))
        action_set.add(str(raw["chosen"]))
        for k in NUMERIC_KEYS:
            numeric_absmax[k] = max(numeric_absmax[k], abs(float(raw["numeric"].get(k, 0.0))))
        if int(raw["is_go_stop"]) == 1:
            go_stop_count += 1
        dt = str(raw.get("decision_type") or "?")
        decision_type_counts[dt] = decision_type_counts.get(dt, 0) + 1

    if total <= 0:
        raise RuntimeError("No trainable samples found.")

    token_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for tok, _ in sorted(token_counter.items(), key=lambda x: (-x[1], x[0])):
        token_to_idx[tok] = len(token_to_idx)
    idx_to_action = sorted(action_set)
    action_to_idx = {a: i for i, a in enumerate(idx_to_action)}
    numeric_scale = {k: max(1.0, float(numeric_absmax.get(k, 1.0))) for k in NUMERIC_KEYS}

    return {
        "token_to_idx": token_to_idx,
        "idx_to_action": idx_to_action,
        "action_to_idx": action_to_idx,
        "numeric_scale": numeric_scale,
        "stats": {
            "go_stop_samples": int(go_stop_count),
            "decision_type_counts": decision_type_counts,
            "action_has_go": bool("go" in action_to_idx),
            "action_has_stop": bool("stop" in action_to_idx),
            "raw_samples_total": int(total),
        },
    }


def encode_sample(raw, token_to_idx, action_to_idx, numeric_scale):
    token_ids = [token_to_idx.get(t, token_to_idx[UNK_TOKEN]) for t in raw["tokens"]]
    cand_ids = [action_to_idx[c] for c in raw["candidates"] if c in action_to_idx]
    target_id = action_to_idx.get(raw["chosen"])
    if target_id is None or target_id not in cand_ids:
        return None
    numeric = []
    for k in NUMERIC_KEYS:
        base = float(raw["numeric"].get(k, 0.0)) / float(numeric_scale.get(k, 1.0) or 1.0)
        w = float(NUMERIC_FEATURE_WEIGHTS.get(k, 1.0))
        numeric.append(base * w)
    go_stop_target = GO_STOP_ACTIONS.get(raw["chosen"], -1)
    return {
        "token_ids": token_ids,
        "numeric": numeric,
        "cand_ids": cand_ids,
        "target_id": int(target_id),
        "value_target": float(raw["value_target"]),
        "is_go_stop": int(raw["is_go_stop"]),
        "go_stop_target": int(go_stop_target),
    }


def build_cache_pt(input_paths, cache_path, max_samples, value_scale):
    schema = scan_cache_schema(input_paths, max_samples, value_scale)
    token_to_idx = schema["token_to_idx"]
    idx_to_action = schema["idx_to_action"]
    action_to_idx = schema["action_to_idx"]
    numeric_scale = schema["numeric_scale"]

    samples = []
    go_stop_final = 0
    for raw in iter_raw_records(input_paths, max_samples, value_scale):
        sample = encode_sample(raw, token_to_idx, action_to_idx, numeric_scale)
        if sample is None:
            continue
        samples.append(sample)
        go_stop_final += int(sample["is_go_stop"])
    if not samples:
        raise RuntimeError("No trainable samples found after encoding.")

    stats = dict(schema["stats"])
    stats["go_stop_samples"] = int(go_stop_final)

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
        "stats": stats,
    }
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    torch.save(payload, cache_path)
    return payload


def build_cache_lmdb(input_paths, cache_path, max_samples, value_scale, map_size_bytes):
    _ensure_lmdb_available()
    schema = scan_cache_schema(input_paths, max_samples, value_scale)
    token_to_idx = schema["token_to_idx"]
    idx_to_action = schema["idx_to_action"]
    action_to_idx = schema["action_to_idx"]
    numeric_scale = schema["numeric_scale"]

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
    written = 0
    go_stop_final = 0
    txn = env.begin(write=True)
    try:
        for raw in iter_raw_records(input_paths, max_samples, value_scale):
            sample = encode_sample(raw, token_to_idx, action_to_idx, numeric_scale)
            if sample is None:
                continue
            blob = json.dumps(sample, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            txn.put(_sample_key(written), blob)
            written += 1
            go_stop_final += int(sample["is_go_stop"])
            if written % 2000 == 0:
                txn.commit()
                txn = env.begin(write=True)
        if written <= 0:
            raise RuntimeError("No trainable samples found after encoding.")
        stats = dict(schema["stats"])
        stats["go_stop_samples"] = int(go_stop_final)
        meta = {
            "format_version": "dual_lmdb_cache_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_files": input_paths,
            "value_scale": value_scale,
            "token_to_idx": token_to_idx,
            "idx_to_action": idx_to_action,
            "numeric_keys": NUMERIC_KEYS,
            "numeric_feature_weights": NUMERIC_FEATURE_WEIGHTS,
            "numeric_scale": numeric_scale,
            "stats": stats,
            "samples_total": int(written),
        }
        txn.put(b"__count__", str(int(written)).encode("ascii"))
        txn.put(b"__meta__", json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        txn.commit()
        env.sync()
    finally:
        env.close()
    return meta


def build_cache(cache_backend, input_paths, cache_path, max_samples, value_scale, lmdb_map_size_bytes):
    if cache_backend == "lmdb":
        meta = build_cache_lmdb(input_paths, cache_path, max_samples, value_scale, lmdb_map_size_bytes)
        cache = dict(meta)
        cache["samples"] = LMDBSampleStore(cache_path)
        return cache
    return build_cache_pt(input_paths, cache_path, max_samples, value_scale)


class LMDBSampleStore:
    def __init__(self, path):
        _ensure_lmdb_available()
        self.path = path
        self.env = lmdb.open(
            path,
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=512,
            subdir=True,
        )
        self.txn = self.env.begin(write=False)
        meta_blob = self.txn.get(b"__meta__")
        if not meta_blob:
            self.close()
            raise RuntimeError(f"Invalid LMDB cache (missing __meta__): {path}")
        self.meta = json.loads(bytes(meta_blob).decode("utf-8"))
        raw_count = self.txn.get(b"__count__")
        if raw_count:
            self.count = int(raw_count.decode("ascii"))
        else:
            self.count = int(self.meta.get("samples_total") or 0)

    def __len__(self):
        return int(self.count)

    def get(self, index):
        idx = int(index)
        if idx < 0 or idx >= int(self.count):
            raise IndexError(f"Sample index out of range: {idx}")
        blob = self.txn.get(_sample_key(idx))
        if blob is None:
            raise IndexError(f"Missing sample in cache at index {idx}")
        return json.loads(bytes(blob).decode("utf-8"))

    def close(self):
        try:
            if hasattr(self, "txn") and self.txn is not None:
                self.txn.abort()
                self.txn = None
        except Exception:
            pass
        try:
            if hasattr(self, "env") and self.env is not None:
                self.env.close()
                self.env = None
        except Exception:
            pass


def load_or_build_cache(args):
    backend = str(args.cache_backend or "pt").strip().lower()
    if backend == "lmdb":
        cache_path = args.cache_lmdb or os.path.join("logs", "dual-cache.lmdb")
        cache_exists = os.path.isdir(cache_path)
    else:
        cache_path = args.cache_pt or os.path.join("logs", "dual-cache.pt")
        cache_exists = os.path.exists(cache_path)

    if args.rebuild_cache or (not cache_exists):
        paths = expand_inputs(args.input)
        lmdb_map_size_bytes = int(float(args.lmdb_map_size_gb) * (1024**3))
        return build_cache(backend, paths, cache_path, args.max_samples, args.value_scale, lmdb_map_size_bytes), cache_path

    if backend == "lmdb":
        store = LMDBSampleStore(cache_path)
        cache = dict(store.meta)
        cache["samples"] = store
        return cache, cache_path

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


def _sample_at(samples, sid):
    if isinstance(samples, list):
        return samples[sid]
    return samples.get(sid)


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
        s = _sample_at(samples, sid)
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


def evaluate(model, samples, sample_ids, action_size, batch_size, device):
    model.eval()
    total = 0
    acc_n = 0
    gs_total = 0
    gs_acc_n = 0
    se = 0.0
    ae = 0.0
    with torch.no_grad():
        for st in range(0, len(sample_ids), batch_size):
            ids = sample_ids[st : st + batch_size]
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
    parser.add_argument("--cache-backend", choices=["pt", "lmdb"], default="pt")
    parser.add_argument("--cache-pt", default="logs/dual-cache.pt")
    parser.add_argument("--cache-lmdb", default="logs/dual-cache.lmdb")
    parser.add_argument(
        "--lmdb-map-size-gb",
        type=float,
        default=8.0,
        help="LMDB map size in GB (used only when --cache-backend lmdb).",
    )
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
    if float(args.lmdb_map_size_gb) <= 0:
        raise RuntimeError("--lmdb-map-size-gb must be > 0.")
    if str(args.cache_backend or "pt").strip().lower() == "lmdb":
        _ensure_lmdb_available()
    device = args.device
    use_amp = (device == "cuda") and (not args.no_amp)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    cache, cache_path = load_or_build_cache(args)
    samples = cache["samples"]
    sample_count = len(samples)
    if sample_count <= 0:
        raise RuntimeError("No samples in cache.")
    cached_numeric_keys = list(cache.get("numeric_keys") or [])
    if cached_numeric_keys and cached_numeric_keys != NUMERIC_KEYS:
        raise RuntimeError(
            "Cache numeric_keys mismatch with current trainer. Rebuild cache with --rebuild-cache."
        )
    action_size = len(cache["idx_to_action"])
    vocab_size = len(cache["token_to_idx"])
    cache_stats = cache.get("stats") or {}

    order = list(range(sample_count))
    random.Random(args.seed).shuffle(order)
    split = int(len(order) * 0.9)
    train_ids = order[:split]
    valid_ids = order[split:] if split < len(order) else []

    print(
        json.dumps(
            {
                "dataset_stats": {
                    "samples_total": sample_count,
                    "go_stop_samples": int(cache_stats.get("go_stop_samples", 0)),
                    "go_stop_ratio": float(cache_stats.get("go_stop_samples", 0)) / max(1, sample_count),
                    "action_has_go": bool(cache_stats.get("action_has_go", False)),
                    "action_has_stop": bool(cache_stats.get("action_has_stop", False)),
                    "decision_type_counts": cache_stats.get("decision_type_counts", {}),
                    "split_train": len(train_ids),
                    "split_valid": len(valid_ids),
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
        perm = list(train_ids)
        rng.shuffle(perm)
        for st in range(0, len(perm), args.batch_size):
            ids = perm[st : st + args.batch_size]
            token_ids_flat, offsets, numeric, cand_mask, target, value_t, gs_target, gs_policy_weight, gs_value_weight = build_batch(
                samples,
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

        metrics = evaluate(model, samples, valid_ids, action_size, args.batch_size, device) if valid_ids else {
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

    train_metrics = evaluate(model, samples, train_ids, action_size, args.batch_size, device)
    valid_metrics = evaluate(model, samples, valid_ids, action_size, args.batch_size, device)

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
        "cache_backend": str(args.cache_backend or "pt").strip().lower(),
        "samples_total": sample_count,
        "samples_train": len(train_ids),
        "samples_valid": len(valid_ids),
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
            "go_stop_ratio": float(cache_stats.get("go_stop_samples", 0)) / max(1, sample_count),
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

    if isinstance(samples, LMDBSampleStore):
        samples.close()

    print(f"trained dual net -> {args.output}")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

