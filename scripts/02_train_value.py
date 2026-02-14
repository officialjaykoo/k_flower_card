#!/usr/bin/env python3
import argparse
import glob
import hashlib
import json
import math
import os
import random
from datetime import datetime, timezone

try:
    import torch  # type: ignore
except Exception:
    torch = None


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


def iter_samples(paths, gold_per_point, max_samples=None):
    yielded = 0
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


def predict_sparse_cpu(weights, bias, idxs, vals):
    out = bias
    for i, v in zip(idxs, vals):
        out += weights[i] * v
    return out


def mse_rmse_mae_cpu(weights, bias, features, ys):
    n = len(features)
    if n == 0:
        return 0.0, 0.0, 0.0
    se = 0.0
    ae = 0.0
    for (idxs, vals), y in zip(features, ys):
        p = predict_sparse_cpu(weights, bias, idxs, vals)
        e = p - y
        se += e * e
        ae += abs(e)
    mse = se / n
    return mse, math.sqrt(mse), ae / n


def train_cpu(train_features, train_y, valid_features, valid_y, dim, epochs, lr, l2, seed):
    w = [0.0] * dim
    b = 0.0
    rng = random.Random(seed)
    order = list(range(len(train_features)))

    for _ in range(epochs):
        rng.shuffle(order)
        for idx in order:
            idxs, vals = train_features[idx]
            y = train_y[idx]
            p = predict_sparse_cpu(w, b, idxs, vals)
            err = p - y
            b -= lr * err
            for i, xv in zip(idxs, vals):
                grad = err * xv + l2 * w[i]
                w[i] -= lr * grad

    train_mse, train_rmse, train_mae = mse_rmse_mae_cpu(w, b, train_features, train_y)
    valid_mse, valid_rmse, valid_mae = mse_rmse_mae_cpu(w, b, valid_features, valid_y)
    return w, b, train_mse, train_rmse, train_mae, valid_mse, valid_rmse, valid_mae


def resolve_cuda_device():
    if torch is None:
        raise RuntimeError("GPU mode requires PyTorch, but it is not installed.")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU mode requires CUDA, but torch.cuda.is_available() is False.")
    return "cuda"


def build_batch_tensors(features, ys, batch_ids, device):
    rows = []
    cols = []
    vals = []
    yb = []
    for r, sid in enumerate(batch_ids):
        idxs, vls = features[sid]
        if idxs:
            rows.extend([r] * len(idxs))
            cols.extend(idxs)
            vals.extend(vls)
        yb.append(ys[sid])

    if rows:
        rows_t = torch.tensor(rows, dtype=torch.long, device=device)
        cols_t = torch.tensor(cols, dtype=torch.long, device=device)
        vals_t = torch.tensor(vals, dtype=torch.float32, device=device)
    else:
        rows_t = torch.empty((0,), dtype=torch.long, device=device)
        cols_t = torch.empty((0,), dtype=torch.long, device=device)
        vals_t = torch.empty((0,), dtype=torch.float32, device=device)
    y_t = torch.tensor(yb, dtype=torch.float32, device=device)
    return rows_t, cols_t, vals_t, y_t


def predict_batch_torch(w, b, rows, cols, vals, batch_size):
    pred = torch.full((batch_size,), float(b.item()), dtype=torch.float32, device=w.device)
    if rows.numel() > 0:
        contrib = w.index_select(0, cols) * vals
        pred.scatter_add_(0, rows, contrib)
    return pred


def evaluate_torch(features, ys, w, b, batch_size, device):
    n = len(features)
    if n == 0:
        return 0.0, 0.0, 0.0
    se = 0.0
    ae = 0.0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch_ids = list(range(start, min(n, start + batch_size)))
            rows, cols, vals, y_t = build_batch_tensors(features, ys, batch_ids, device)
            pred = predict_batch_torch(w, b, rows, cols, vals, len(batch_ids))
            err = pred - y_t
            se += float(torch.sum(err * err).item())
            ae += float(torch.sum(torch.abs(err)).item())
    mse = se / n
    return mse, math.sqrt(mse), ae / n


def train_torch(train_features, train_y, valid_features, valid_y, dim, epochs, lr, l2, seed, batch_size, progress_every, device):
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

    rng = random.Random(seed)
    order = list(range(len(train_features)))

    for epoch in range(epochs):
        rng.shuffle(order)
        num_batches = (len(order) + batch_size - 1) // batch_size
        for bi in range(num_batches):
            s = bi * batch_size
            e = min(len(order), s + batch_size)
            batch_ids = order[s:e]
            rows, cols, vals, y_t = build_batch_tensors(train_features, train_y, batch_ids, device)

            opt.zero_grad(set_to_none=True)
            pred = predict_batch_torch(w, b, rows, cols, vals, len(batch_ids))
            loss = torch.mean((pred - y_t) ** 2)
            loss.backward()
            opt.step()

            if progress_every > 0 and ((bi + 1) % progress_every == 0 or (bi + 1) == num_batches):
                print(
                    f"epoch {epoch + 1}/{epochs} | batch {bi + 1}/{num_batches} | "
                    f"samples {e}/{len(order)} | loss {float(loss.item()):.4f}"
                )

    with torch.no_grad():
        train_mse, train_rmse, train_mae = evaluate_torch(train_features, train_y, w, b, batch_size, device)
        valid_mse, valid_rmse, valid_mae = evaluate_torch(valid_features, valid_y, w, b, batch_size, device)
        w_out = w.detach().cpu().tolist()
        b_out = float(b.detach().cpu().item())

    return w_out, b_out, train_mse, train_rmse, train_mae, valid_mse, valid_rmse, valid_mae


def main():
    parser = argparse.ArgumentParser(description="Train value regressor (expected gold).")
    parser.add_argument("--input", nargs="+", default=["logs/*.jsonl"])
    parser.add_argument("--output", default="models/value-model.json")
    parser.add_argument("--gold-per-point", type=float, default=100.0)
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--progress-every", type=int, default=0, help="Print every N batches (torch backend). 0 disables.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of training samples.")
    args = parser.parse_args()

    input_paths = expand_inputs(args.input)
    samples = list(iter_samples(input_paths, args.gold_per_point, args.max_samples))
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

    train_features = [sparse_features(s, args.dim, numeric_scale) for s in train_samples]
    train_y = [s["y"] for s in train_samples]
    valid_features = [sparse_features(s, args.dim, numeric_scale) for s in valid_samples]
    valid_y = [s["y"] for s in valid_samples]

    if args.device == "cpu":
        backend_used = "cpu"
        actual_device = "cpu"
        print("training backend=cpu device=cpu")
        w, b, train_mse, train_rmse, train_mae, valid_mse, valid_rmse, valid_mae = train_cpu(
            train_features,
            train_y,
            valid_features,
            valid_y,
            args.dim,
            args.epochs,
            args.lr,
            args.l2,
            args.seed,
        )
    else:
        backend_used = "torch"
        dev = resolve_cuda_device()
        actual_device = dev
        print(f"training backend=torch device={dev} batch_size={args.batch_size}")
        w, b, train_mse, train_rmse, train_mae, valid_mse, valid_rmse, valid_mae = train_torch(
            train_features,
            train_y,
            valid_features,
            valid_y,
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
            "backend": backend_used,
            "device": actual_device,
            "batch_size": args.batch_size,
            "max_samples": args.max_samples,
        },
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False)

    print(f"trained value model -> {args.output}")
    print(json.dumps(model["train_summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
