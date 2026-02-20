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
DEFAULT_POLICY_FILTER_RULES = {
    "min_candidate_count": 2,
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


def trace_passes_policy_filter(trace, filter_rules):
    min_cc = int((filter_rules or {}).get("min_candidate_count") or 1)
    cc = _to_int_or_none((trace or {}).get("cc"))
    if cc is None:
        return True
    return cc >= min_cc


def context_key(trace, decision_type):
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
        cands = len(sp.get("cards") or sp.get("boardCardIds") or sp.get("options") or [])
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


def extract_sample(trace):
    dt = trace.get("dt")
    chosen_compact = trace.get("ch")
    chosen_play = chosen_compact if dt == "play" else trace.get("c")
    chosen_match = chosen_compact if dt == "match" else trace.get("s")
    chosen_option = action_alias(chosen_compact if dt == "option" else trace.get("at"))
    if dt == "play" and chosen_play:
        return "play", [chosen_play], chosen_play
    if dt == "match" and chosen_match:
        return "match", [chosen_match], chosen_match
    dc = trace.get("dc") or {}
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
        phase = dc.get("phase")
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
        candidates = by_phase.get(phase) or by_phase.get(str(phase))
        chosen = chosen_option
        if candidates and chosen in candidates:
            return "option", candidates, chosen
        if chosen:
            return "option", [chosen], chosen

    sp = trace.get("sp") or {}
    candidates = sp.get("cards")
    if candidates:
        chosen = trace.get("c")
        if chosen in candidates:
            return "play", candidates, chosen
        return None
    candidates = sp.get("boardCardIds")
    if candidates:
        chosen = trace.get("s")
        if chosen in candidates:
            return "match", candidates, chosen
        return None
    candidates = sp.get("options")
    if candidates:
        chosen = action_alias(trace.get("ch") if trace.get("ch") is not None else trace.get("at"))
        if chosen in candidates:
            return "option", candidates, chosen
        return None
    return None


def iter_samples(paths, max_samples=None, filter_rules=None):
    yielded = 0
    active_rules = _normalize_policy_filter_rules(filter_rules)
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                game = json.loads(line)
                for trace in game.get("decision_trace") or []:
                    if not trace_passes_policy_filter(trace, active_rules):
                        continue
                    sample = extract_sample(trace)
                    if sample is None:
                        continue
                    decision_type, candidates, chosen = sample
                    yield {
                        "decision_type": decision_type,
                        "candidates": candidates,
                        "chosen": chosen,
                        "context_key": context_key(trace, decision_type),
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


def build_sample_cache_jsonl(cache_path, input_paths, max_samples, filter_rules):
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    total = 0
    with open(cache_path, "w", encoding="utf-8") as out:
        for sample in iter_samples(input_paths, max_samples, filter_rules):
            out.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
            total += 1
    if total <= 0:
        raise RuntimeError("No trainable decision samples found.")
    return {"samples_total": int(total)}


def build_sample_cache_lmdb(cache_path, input_paths, max_samples, map_size_bytes, filter_rules):
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
        for sample in iter_samples(input_paths, max_samples, filter_rules):
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
):
    if cache_backend == "lmdb":
        return build_sample_cache_lmdb(
            cache_path,
            input_paths,
            max_samples,
            lmdb_map_size_bytes,
            filter_rules,
        )
    return build_sample_cache_jsonl(cache_path, input_paths, max_samples, filter_rules)


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
    global_counts = defaultdict(lambda: defaultdict(int))
    context_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    context_totals = defaultdict(lambda: defaultdict(int))

    for s in samples:
        dt = s["decision_type"]
        ck = s["context_key"]
        chosen = s["chosen"]
        global_counts[dt][chosen] += 1
        context_counts[dt][ck][chosen] += 1
        context_totals[dt][ck] += 1

    model = {
        "model_type": "policy_frequency_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "alpha": alpha,
        "global_counts": {k: dict(v) for k, v in global_counts.items()},
        "context_counts": {dt: {ck: dict(cc) for ck, cc in cks.items()} for dt, cks in context_counts.items()},
        "context_totals": {dt: dict(v) for dt, v in context_totals.items()},
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
    parser.add_argument("--seed", type=int, default=7, help="Unused compatibility option.")
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
    args = parser.parse_args()

    if float(args.lmdb_map_size_gb) <= 0:
        raise RuntimeError("--lmdb-map-size-gb must be > 0.")
    if args.cache_backend == "lmdb":
        _ensure_lmdb_available()

    input_paths = expand_inputs(args.input)
    manifest = input_manifest(input_paths)
    filter_rules, filter_rules_path = load_policy_filter_rules(args.filter_rules)
    cache_path = _resolve_sample_cache_path(args.output, args.sample_cache, args.cache_backend)
    cache_config = {
        "format_version": "policy_sample_cache_v1",
        "cache_backend": args.cache_backend,
        "max_samples": args.max_samples,
        "filter_rules": filter_rules,
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
        return iter_samples(input_paths, args.max_samples, filter_rules)

    model = train_model(sample_iter_factory(), alpha=args.alpha)

    total = 0
    correct = 0
    nll = 0.0
    entropy_sum = 0.0
    entropy_norm_sum = 0.0
    if args.skip_train_metrics:
        total = sum(sum(v.values()) for v in (model.get("global_counts") or {}).values())
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
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)

    print(f"trained policy model -> {args.output}")
    print(json.dumps(model["train_summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
