#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:  # pragma: no cover - fail-fast import guard
    raise SystemExit(f"torch is required for train_iqn_go_stop_scalar.py: {exc}")


FORMAT_VERSION = "iqn_go_stop_scalar_v1"
GO_STOP_ONE_HOT = [1.0, 0.0, 0.0, 0.0]
DEFAULT_HIDDEN_SIZES = (128, 128)
ACTION_ORDER = ("go", "stop")
BASE_FEATURES = 16
INPUT_DIM = BASE_FEATURES + len(GO_STOP_ONE_HOT) + 10


def fail(message: str) -> None:
    raise RuntimeError(str(message))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a scalar go/stop baseline from IQN teacher JSONL."
    )
    parser.add_argument("--teacher-in", required=True, help="Path to stage-3 teacher JSONL")
    parser.add_argument("--out-dir", required=True, help="Directory for checkpoint and metrics")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-sizes", default="128,128", help="Comma-separated MLP sizes")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or explicit torch device")
    parser.add_argument("--limit-decisions", type=int, default=0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def parse_hidden_sizes(raw: str) -> Tuple[int, ...]:
    parts = [part.strip() for part in str(raw or "").split(",")]
    sizes: List[int] = []
    for part in parts:
        if not part:
            continue
        try:
            size = int(part)
        except Exception:
            fail(f"invalid hidden size: {part}")
        if size <= 0:
            fail(f"hidden size must be > 0, got={size}")
        sizes.append(size)
    if not sizes:
        return DEFAULT_HIDDEN_SIZES
    return tuple(sizes)


def resolve_device(raw: str) -> torch.device:
    token = str(raw or "auto").strip().lower()
    if token in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


def clip(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / len(values))


@dataclass(frozen=True)
class ActionSample:
    decision_id: str
    action: str
    feature: List[float]
    target: float
    teacher_choice: str
    mean_choice: str


@dataclass(frozen=True)
class DecisionTarget:
    decision_id: str
    teacher_choice: str
    mean_choice: str
    go_target: float
    stop_target: float


def estimate_stop_value(snapshot: Dict[str, Any]) -> float:
    self_score = as_float(snapshot.get("self_score_total", 0.0))
    self_multiplier = as_float(snapshot.get("self_multiplier", 1.0), 1.0)
    carry = as_float(snapshot.get("carry_over_multiplier", 1.0), 1.0)
    estimated_gold_k = self_score * self_multiplier * carry * 0.1
    return clip(estimated_gold_k, -12.0, 12.0)


def build_payload(snapshot: Dict[str, Any]) -> List[float]:
    initial_gold = as_float(snapshot.get("initial_gold_base", 100000.0), 100000.0)
    self_gold = as_float(snapshot.get("self_gold", initial_gold), initial_gold)
    opp_gold = as_float(snapshot.get("opp_gold", initial_gold), initial_gold)
    self_score = clip(as_float(snapshot.get("self_score_total", 0.0)) / 20.0, -2.0, 2.0)
    opp_score = clip(as_float(snapshot.get("opp_score_total", 0.0)) / 20.0, -2.0, 2.0)
    carry = as_float(snapshot.get("carry_over_multiplier", 1.0), 1.0)
    self_multiplier = as_float(snapshot.get("self_multiplier", 1.0), 1.0)
    self_captured = snapshot.get("self_captured", {}) if isinstance(snapshot.get("self_captured"), dict) else {}
    go_count = clip(as_float(self_captured.get("go_count", 0.0)) / 6.0, 0.0, 2.0)
    deck_remaining = clip(as_float(snapshot.get("deck_count", 0.0)) / 20.0, 0.0, 2.0)
    self_gold_delta = clip((self_gold - initial_gold) / 1000.0, -12.0, 12.0)
    opp_gold_delta = clip((opp_gold - initial_gold) / 1000.0, -12.0, 12.0)

    return [
        0.0,
        estimate_stop_value(snapshot),
        clip(carry / 8.0, 0.0, 2.0),
        clip((self_multiplier * carry) / 16.0, 0.0, 2.0),
        self_score,
        opp_score,
        self_gold_delta,
        opp_gold_delta,
        go_count,
        deck_remaining,
    ]


def build_action_feature(row: Dict[str, Any], action: str) -> List[float]:
    features_by_action = row.get("features16_by_action")
    expected_label = "16D"
    if not isinstance(features_by_action, dict):
        fail("teacher row missing features16_by_action object")
    base = features_by_action.get(action)
    if not isinstance(base, list) or len(base) < BASE_FEATURES:
        fail(f"teacher row has invalid {expected_label} features for action={action}")
    payload = build_payload(row.get("public_snapshot", {}))
    feature = [as_float(v) for v in base[:BASE_FEATURES]] + GO_STOP_ONE_HOT + payload
    if len(feature) != INPUT_DIM:
        fail(f"invalid scalar input dim built: got={len(feature)}, expected={INPUT_DIM}")
    return feature


def load_teacher_rows(path: str, limit_decisions: int) -> Tuple[List[ActionSample], Dict[str, DecisionTarget]]:
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        fail(f"teacher dataset not found: {abs_path}")

    action_samples: List[ActionSample] = []
    decision_targets: Dict[str, DecisionTarget] = {}

    with open(abs_path, "r", encoding="utf-8-sig") as handle:
        for line_index, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                fail(f"failed to parse JSONL at line {line_index}: {exc}")

            decision_id = str(row.get("decision_id", "")).strip()
            if not decision_id:
                fail(f"missing decision_id at line {line_index}")
            if limit_decisions > 0 and decision_id not in decision_targets and len(decision_targets) >= limit_decisions:
                continue

            stats = row.get("stats", {})
            if not isinstance(stats, dict):
                fail(f"missing stats object at line {line_index}")
            go_target = as_float(stats.get("go_mean"))
            stop_target = as_float(stats.get("stop_mean"))
            mean_choice = "go" if go_target > stop_target else "stop"
            teacher_choice = str(row.get("teacher_choice", "")).strip() or mean_choice

            decision_targets[decision_id] = DecisionTarget(
                decision_id=decision_id,
                teacher_choice=teacher_choice,
                mean_choice=mean_choice,
                go_target=go_target,
                stop_target=stop_target,
            )

            for action in ACTION_ORDER:
                target = go_target if action == "go" else stop_target
                action_samples.append(
                    ActionSample(
                        decision_id=decision_id,
                        action=action,
                        feature=build_action_feature(row, action),
                        target=target,
                        teacher_choice=teacher_choice,
                        mean_choice=mean_choice,
                    )
                )

    if not action_samples:
        fail(f"no usable teacher samples found in: {abs_path}")
    return action_samples, decision_targets


def split_decision_ids(decision_ids: Sequence[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    ids = list(decision_ids)
    if len(ids) <= 1:
        return ids[:], ids[:]
    rng = random.Random(seed)
    rng.shuffle(ids)
    ratio = min(0.5, max(0.05, float(val_ratio)))
    val_count = max(1, int(round(len(ids) * ratio)))
    if val_count >= len(ids):
        val_count = max(1, len(ids) // 2)
    val_ids = ids[:val_count]
    train_ids = ids[val_count:]
    if not train_ids:
        train_ids = ids[:]
    return train_ids, val_ids


def tensorize(samples: Sequence[ActionSample], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    features = torch.tensor([sample.feature for sample in samples], dtype=torch.float32, device=device)
    targets = torch.tensor([[sample.target] for sample in samples], dtype=torch.float32, device=device)
    if features.ndim != 2 or features.shape[1] != INPUT_DIM:
        fail(f"invalid tensorized features shape: {tuple(features.shape)}")
    return features, targets


class ScalarValueNet(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Sequence[int], dropout: float):
        super().__init__()
        dims = [int(input_dim)] + [int(v) for v in hidden_sizes]
        layers: List[nn.Module] = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(dims[idx + 1]))
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_dataloader(features: torch.Tensor, targets: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(features, targets)
    return DataLoader(dataset, batch_size=max(1, int(batch_size)), shuffle=shuffle)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    loss_fn: nn.Module,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    for batch_x, batch_y in loader:
        pred = model(batch_x)
        loss = loss_fn(pred, batch_y)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        batch_n = int(batch_x.shape[0])
        total_loss += float(loss.detach().item()) * batch_n
        total_count += batch_n
    if total_count <= 0:
        return 0.0
    return float(total_loss / total_count)


def predict_actions(
    model: nn.Module,
    samples: Sequence[ActionSample],
    device: torch.device,
    batch_size: int,
) -> List[float]:
    if not samples:
        return []
    model.eval()
    features, _targets = tensorize(samples, device)
    loader = build_dataloader(features, _targets, batch_size=batch_size, shuffle=False)
    preds: List[float] = []
    with torch.no_grad():
        for batch_x, _batch_y in loader:
            pred = model(batch_x).squeeze(-1).detach().cpu().tolist()
            preds.extend(float(v) for v in pred)
    return preds


def compute_sample_metrics(samples: Sequence[ActionSample], preds: Sequence[float]) -> Dict[str, float]:
    if len(samples) != len(preds):
        fail("sample/prediction length mismatch")
    errors = [float(preds[i]) - float(samples[i].target) for i in range(len(samples))]
    mae = mean([abs(err) for err in errors])
    rmse = math.sqrt(mean([err * err for err in errors])) if errors else 0.0
    return {
        "sample_count": float(len(samples)),
        "mae": float(mae),
        "rmse": float(rmse),
        "target_mean": float(mean([sample.target for sample in samples])),
        "pred_mean": float(mean(preds)),
    }


def compute_decision_metrics(
    samples: Sequence[ActionSample],
    preds: Sequence[float],
    decision_targets: Dict[str, DecisionTarget],
) -> Dict[str, float]:
    pairs: Dict[str, Dict[str, float]] = {}
    for sample, pred in zip(samples, preds):
        bucket = pairs.setdefault(sample.decision_id, {})
        bucket[sample.action] = float(pred)

    total = 0
    mean_hits = 0
    teacher_hits = 0
    margin_abs_errors: List[float] = []
    for decision_id, pred_pair in pairs.items():
        if "go" not in pred_pair or "stop" not in pred_pair:
            continue
        target = decision_targets.get(decision_id)
        if target is None:
            continue
        pred_choice = "go" if pred_pair["go"] > pred_pair["stop"] else "stop"
        pred_margin = pred_pair["go"] - pred_pair["stop"]
        true_margin = target.go_target - target.stop_target
        total += 1
        if pred_choice == target.mean_choice:
            mean_hits += 1
        if pred_choice == target.teacher_choice:
            teacher_hits += 1
        margin_abs_errors.append(abs(pred_margin - true_margin))

    if total <= 0:
        return {
            "decision_count": 0.0,
            "mean_choice_accuracy": 0.0,
            "teacher_choice_accuracy": 0.0,
            "margin_mae": 0.0,
        }
    return {
        "decision_count": float(total),
        "mean_choice_accuracy": float(mean_hits / total),
        "teacher_choice_accuracy": float(teacher_hits / total),
        "margin_mae": float(mean(margin_abs_errors)),
    }


def subset_samples(samples: Sequence[ActionSample], allowed_decision_ids: Iterable[str]) -> List[ActionSample]:
    allowed = set(allowed_decision_ids)
    return [sample for sample in samples if sample.decision_id in allowed]


def save_checkpoint(
    out_dir: str,
    model: ScalarValueNet,
    args: argparse.Namespace,
    metrics: Dict[str, Any],
    hidden_sizes: Sequence[int],
) -> str:
    payload = {
        "format_version": FORMAT_VERSION,
        "input_dim": INPUT_DIM,
        "hidden_sizes": [int(v) for v in hidden_sizes],
        "dropout": float(args.dropout),
        "feature_spec": {
            "base_features": BASE_FEATURES,
            "option_type_one_hot": 4,
            "option_payload": 10,
            "actions": list(ACTION_ORDER),
        },
        "target_spec": {
            "kind": "scalar_mean_return",
            "unit": "gold_delta_div_1000",
            "teacher_dataset": os.path.abspath(args.teacher_in),
        },
        "train_args": vars(args),
        "metrics": metrics,
        "model_state_dict": model.state_dict(),
    }
    out_path = os.path.join(out_dir, "model.pt")
    torch.save(payload, out_path)
    return out_path


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        fail("--epochs must be > 0")
    if args.batch_size <= 0:
        fail("--batch-size must be > 0")
    if args.learning_rate <= 0:
        fail("--learning-rate must be > 0")
    if args.weight_decay < 0:
        fail("--weight-decay must be >= 0")
    if not (0.0 <= float(args.dropout) < 1.0):
        fail("--dropout must be in [0,1)")

    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    device = resolve_device(args.device)
    ensure_dir(args.out_dir)

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    action_samples, decision_targets = load_teacher_rows(args.teacher_in, int(args.limit_decisions))
    decision_ids = list(decision_targets.keys())
    train_decision_ids, val_decision_ids = split_decision_ids(decision_ids, float(args.val_ratio), int(args.seed))
    train_samples = subset_samples(action_samples, train_decision_ids)
    val_samples = subset_samples(action_samples, val_decision_ids)
    if not val_samples:
        val_samples = train_samples[:]
        val_decision_ids = train_decision_ids[:]

    train_x, train_y = tensorize(train_samples, device)
    val_x, val_y = tensorize(val_samples, device)

    model = ScalarValueNet(INPUT_DIM, hidden_sizes, float(args.dropout)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    loss_fn = nn.SmoothL1Loss(beta=0.5)

    train_loader = build_dataloader(train_x, train_y, batch_size=int(args.batch_size), shuffle=True)
    val_loader = build_dataloader(val_x, val_y, batch_size=int(args.batch_size), shuffle=False)

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, int(args.epochs) + 1):
        train_loss = run_epoch(model, train_loader, optimizer, loss_fn)
        val_loss = run_epoch(model, val_loader, None, loss_fn)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})

        improved = val_loss < (best_val_loss - float(args.min_delta))
        if improved:
            best_val_loss = float(val_loss)
            best_epoch = int(epoch)
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1

        if epoch % max(1, int(args.log_every)) == 0:
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "best_val_loss": best_val_loss,
                        "best_epoch": best_epoch,
                        "stale_epochs": stale_epochs,
                    }
                )
            )

        if stale_epochs >= int(args.patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_preds = predict_actions(model, train_samples, device, int(args.batch_size))
    val_preds = predict_actions(model, val_samples, device, int(args.batch_size))
    train_sample_metrics = compute_sample_metrics(train_samples, train_preds)
    val_sample_metrics = compute_sample_metrics(val_samples, val_preds)
    train_decision_metrics = compute_decision_metrics(train_samples, train_preds, decision_targets)
    val_decision_metrics = compute_decision_metrics(val_samples, val_preds, decision_targets)

    metrics = {
        "format_version": FORMAT_VERSION,
        "teacher_in": os.path.abspath(args.teacher_in),
        "device": str(device),
        "input_dim": INPUT_DIM,
        "hidden_sizes": [int(v) for v in hidden_sizes],
        "dataset": {
            "decision_count_total": len(decision_targets),
            "sample_count_total": len(action_samples),
            "train_decision_count": len(set(train_decision_ids)),
            "val_decision_count": len(set(val_decision_ids)),
            "train_sample_count": len(train_samples),
            "val_sample_count": len(val_samples),
        },
        "training": {
            "epochs_requested": int(args.epochs),
            "epochs_ran": len(history),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "final_train_loss": history[-1]["train_loss"] if history else 0.0,
            "final_val_loss": history[-1]["val_loss"] if history else 0.0,
        },
        "train_sample_metrics": train_sample_metrics,
        "val_sample_metrics": val_sample_metrics,
        "train_decision_metrics": train_decision_metrics,
        "val_decision_metrics": val_decision_metrics,
        "history": history,
    }

    checkpoint_path = save_checkpoint(args.out_dir, model, args, metrics, hidden_sizes)
    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    split_path = os.path.join(args.out_dir, "split.json")
    with open(split_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "train_decision_ids": train_decision_ids,
                "val_decision_ids": val_decision_ids,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(
        json.dumps(
            {
                "out_dir": os.path.abspath(args.out_dir),
                "checkpoint": os.path.abspath(checkpoint_path),
                "metrics": os.path.abspath(metrics_path),
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "val_mean_choice_accuracy": val_decision_metrics.get("mean_choice_accuracy", 0.0),
                "val_teacher_choice_accuracy": val_decision_metrics.get("teacher_choice_accuracy", 0.0),
            }
        )
    )


if __name__ == "__main__":
    main()
