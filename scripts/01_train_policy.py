#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
import random
from collections import defaultdict
from datetime import datetime, timezone


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


def context_key(trace, decision_type):
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


def extract_sample(trace):
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
        chosen = action_alias(trace.get("at"))
        if chosen in candidates:
            return "option", candidates, chosen
        return None
    return None


def iter_samples(paths):
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                game = json.loads(line)
                for trace in game.get("decision_trace") or []:
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
        "context_counts": {
            dt: {ck: dict(cc) for ck, cc in cks.items()}
            for dt, cks in context_counts.items()
        },
        "context_totals": {
            dt: dict(v) for dt, v in context_totals.items()
        },
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
    parser.add_argument("--seed", type=int, default=7, help="Shuffle seed.")
    args = parser.parse_args()

    input_paths = expand_inputs(args.input)
    samples = list(iter_samples(input_paths))
    if not samples:
        raise RuntimeError("No trainable decision samples found.")
    random.Random(args.seed).shuffle(samples)

    model = train_model(samples, alpha=args.alpha)

    correct = 0
    nll = 0.0
    for s in samples:
        pred, _ = predict_top1(model, s)
        if pred == s["chosen"]:
            correct += 1
        p = max(1e-12, prob_of_choice(model, s, s["chosen"]))
        nll += -math.log(p)

    model["train_summary"] = {
        "samples": len(samples),
        "accuracy_top1": correct / max(1, len(samples)),
        "nll_per_sample": nll / max(1, len(samples)),
        "input_files": input_paths,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)

    print(f"trained policy model -> {args.output}")
    print(json.dumps(model["train_summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
