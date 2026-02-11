#!/usr/bin/env python3
import argparse
import glob
import hashlib
import json
import math
import os
import random
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


def stable_hash(token, dim):
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % dim


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


def choose_label(trace):
    sp = trace.get("sp") or {}
    if sp.get("cards"):
        return trace.get("c"), "play"
    if sp.get("boardCardIds"):
        return trace.get("s"), "match"
    if sp.get("options"):
        return action_alias(trace.get("at")), "option"
    return None, None


def extract_numeric(trace):
    dc = trace.get("dc") or {}
    sp = trace.get("sp") or {}
    cands = len(sp.get("cards") or sp.get("boardCardIds") or sp.get("options") or [])
    return {
        "deck_count": float(dc.get("deckCount") or 0),
        "hand_self": float(dc.get("handCountSelf") or 0),
        "hand_opp": float(dc.get("handCountOpp") or 0),
        "go_self": float(dc.get("goCountSelf") or 0),
        "go_opp": float(dc.get("goCountOpp") or 0),
        "cand_count": float(cands),
        "immediate_reward": float(trace.get("ir") or 0),
    }


def extract_tokens(trace, decision_type, action_label):
    dc = trace.get("dc") or {}
    out = [
        f"phase={dc.get('phase','?')}",
        f"order={trace.get('o','?')}",
        f"decision_type={decision_type}",
        f"action={action_label or '?'}",
        f"deck_bucket={int((dc.get('deckCount') or 0)//3)}",
        f"self_hand={int(dc.get('handCountSelf') or 0)}",
        f"opp_hand={int(dc.get('handCountOpp') or 0)}",
        f"self_go={int(dc.get('goCountSelf') or 0)}",
        f"opp_go={int(dc.get('goCountOpp') or 0)}",
    ]
    return out


def target_gold(line, actor, gold_per_point):
    score = line.get("score") or {}
    self_score = score.get(actor)
    opp = "ai" if actor == "human" else "human"
    opp_score = score.get(opp)
    if self_score is None or opp_score is None:
        return None
    point_diff = float(self_score) - float(opp_score)
    return point_diff * float(gold_per_point)


def iter_samples(paths, gold_per_point):
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line_raw in f:
                line_raw = line_raw.strip()
                if not line_raw:
                    continue
                line = json.loads(line_raw)
                for trace in line.get("decision_trace") or []:
                    actor = trace.get("a")
                    if actor not in ("human", "ai"):
                        continue
                    action_label, decision_type = choose_label(trace)
                    if action_label is None:
                        continue
                    y = target_gold(line, actor, gold_per_point)
                    if y is None:
                        continue
                    yield {
                        "tokens": extract_tokens(trace, decision_type, action_label),
                        "numeric": extract_numeric(trace),
                        "y": y,
                        "actor": actor,
                    }


def vectorize(sample, dim, numeric_scale):
    x = {}
    for tok in sample["tokens"]:
        idx = stable_hash(f"tok:{tok}", dim)
        x[idx] = x.get(idx, 0.0) + 1.0
    for k, v in sample["numeric"].items():
        idx = stable_hash(f"num:{k}", dim)
        scaled = v / max(1e-9, numeric_scale.get(k, 1.0))
        x[idx] = x.get(idx, 0.0) + scaled
    return x


def predict(weights, bias, x):
    out = bias
    for i, v in x.items():
        out += weights[i] * v
    return out


def mse_rmse_mae(weights, bias, vectors, ys):
    n = len(vectors)
    if n == 0:
        return 0.0, 0.0, 0.0
    se = 0.0
    ae = 0.0
    for x, y in zip(vectors, ys):
        p = predict(weights, bias, x)
        e = p - y
        se += e * e
        ae += abs(e)
    mse = se / n
    return mse, math.sqrt(mse), ae / n


def main():
    parser = argparse.ArgumentParser(description="Train value regressor (expected gold).")
    parser.add_argument("--input", nargs="+", default=["logs/*.jsonl"])
    parser.add_argument("--output", default="models/value-model.json")
    parser.add_argument("--gold-per-point", type=float, default=100.0)
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    input_paths = expand_inputs(args.input)
    samples = list(iter_samples(input_paths, args.gold_per_point))
    if not samples:
        raise RuntimeError("No trainable value samples found.")

    random.Random(args.seed).shuffle(samples)
    split_idx = int(len(samples) * 0.9)
    train_samples = samples[:split_idx]
    valid_samples = samples[split_idx:] if split_idx < len(samples) else []

    numeric_scale = {}
    keys = ["deck_count", "hand_self", "hand_opp", "go_self", "go_opp", "cand_count", "immediate_reward"]
    for k in keys:
        max_abs = max(abs(s["numeric"].get(k, 0.0)) for s in train_samples) if train_samples else 1.0
        numeric_scale[k] = max(1.0, max_abs)

    train_vectors = [vectorize(s, args.dim, numeric_scale) for s in train_samples]
    train_y = [s["y"] for s in train_samples]
    valid_vectors = [vectorize(s, args.dim, numeric_scale) for s in valid_samples]
    valid_y = [s["y"] for s in valid_samples]

    w = [0.0] * args.dim
    b = 0.0
    rng = random.Random(args.seed)

    order = list(range(len(train_vectors)))
    for _ in range(args.epochs):
        rng.shuffle(order)
        for idx in order:
            x = train_vectors[idx]
            y = train_y[idx]
            p = predict(w, b, x)
            err = p - y
            b -= args.lr * err
            for i, xv in x.items():
                grad = err * xv + args.l2 * w[i]
                w[i] -= args.lr * grad

    train_mse, train_rmse, train_mae = mse_rmse_mae(w, b, train_vectors, train_y)
    valid_mse, valid_rmse, valid_mae = mse_rmse_mae(w, b, valid_vectors, valid_y)

    model = {
        "model_type": "value_linear_hash_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dim": args.dim,
        "weights": w,
        "bias": b,
        "numeric_scale": numeric_scale,
        "gold_per_point": args.gold_per_point,
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
            ],
            "numeric_keys": keys,
            "hash_dim": args.dim,
        },
        "train_summary": {
            "samples_total": len(samples),
            "samples_train": len(train_samples),
            "samples_valid": len(valid_samples),
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
        },
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False)

    print(f"trained value model -> {args.output}")
    print(json.dumps(model["train_summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
