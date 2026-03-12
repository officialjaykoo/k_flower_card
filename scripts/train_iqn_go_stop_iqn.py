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
    raise SystemExit(f"torch is required for train_iqn_go_stop_iqn.py: {exc}")


FORMAT_VERSION = "iqn_go_stop_v1"
RUNTIME_FORMAT_VERSION = "iqn_go_stop_runtime_v1"
GO_STOP_ONE_HOT = [1.0, 0.0, 0.0, 0.0]
DEFAULT_HIDDEN_SIZES = (128, 128)
ACTION_ORDER = ("go", "stop")
BASE_FEATURES = 16
INPUT_DIM = BASE_FEATURES + len(GO_STOP_ONE_HOT) + 10


def fail(message: str) -> None:
    raise RuntimeError(str(message))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a go/stop IQN baseline from stage-3 teacher JSONL."
    )
    parser.add_argument("--teacher-in", required=True, help="Path to stage-3 teacher JSONL")
    parser.add_argument("--out-dir", required=True, help="Directory for checkpoint and metrics")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-sizes", default="128,128", help="Comma-separated MLP sizes")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or explicit torch device")
    parser.add_argument("--limit-decisions", type=int, default=0)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--num-cosines", type=int, default=64)
    parser.add_argument("--num-quantiles-train", type=int, default=32)
    parser.add_argument("--num-quantiles-eval", type=int, default=64)
    parser.add_argument("--kappa", type=float, default=1.0)
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


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    p = clip(float(q), 0.0, 1.0)
    idx = int(math.floor((len(xs) - 1) * p))
    return float(xs[idx])


def cvar(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    cutoff = quantile(values, q)
    tail = [float(v) for v in values if float(v) <= cutoff]
    if not tail:
        return cutoff
    return float(sum(tail) / len(tail))


@dataclass(frozen=True)
class ActionSample:
    decision_id: str
    action: str
    feature: List[float]
    target_returns: List[float]
    target_mean: float
    target_cvar10: float
    target_score: float
    teacher_choice: str
    mean_choice: str


@dataclass(frozen=True)
class DecisionTarget:
    decision_id: str
    teacher_choice: str
    mean_choice: str
    go_mean: float
    stop_mean: float
    go_cvar10: float
    stop_cvar10: float
    go_score: float
    stop_score: float


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
        fail(f"invalid IQN input dim built: got={len(feature)}, expected={INPUT_DIM}")
    return feature


def load_teacher_rows(path: str, limit_decisions: int) -> Tuple[List[ActionSample], Dict[str, DecisionTarget], int]:
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        fail(f"teacher dataset not found: {abs_path}")

    action_samples: List[ActionSample] = []
    decision_targets: Dict[str, DecisionTarget] = {}
    target_return_count = 0

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
            returns = row.get("returns", {})
            if not isinstance(stats, dict) or not isinstance(returns, dict):
                fail(f"missing stats/returns object at line {line_index}")

            go_returns = [as_float(v) for v in returns.get("go", [])]
            stop_returns = [as_float(v) for v in returns.get("stop", [])]
            if not go_returns or not stop_returns:
                fail(f"empty action returns at line {line_index}")
            if target_return_count <= 0:
                target_return_count = len(go_returns)
            if len(go_returns) != target_return_count or len(stop_returns) != target_return_count:
                fail(
                    "teacher dataset must have a fixed samples_per_action per row: "
                    f"line={line_index}, expected={target_return_count}, got_go={len(go_returns)}, got_stop={len(stop_returns)}"
                )

            go_mean = as_float(stats.get("go_mean"), mean(go_returns))
            stop_mean = as_float(stats.get("stop_mean"), mean(stop_returns))
            go_cvar10 = as_float(stats.get("go_cvar10"), cvar(go_returns, 0.1))
            stop_cvar10 = as_float(stats.get("stop_cvar10"), cvar(stop_returns, 0.1))
            go_score = as_float(stats.get("go_score"), 0.7 * go_mean + 0.3 * go_cvar10)
            stop_score = as_float(stats.get("stop_score"), 0.7 * stop_mean + 0.3 * stop_cvar10)
            mean_choice = "go" if go_mean > stop_mean else "stop"
            teacher_choice = str(row.get("teacher_choice", "")).strip() or ("go" if go_score > stop_score else "stop")

            decision_targets[decision_id] = DecisionTarget(
                decision_id=decision_id,
                teacher_choice=teacher_choice,
                mean_choice=mean_choice,
                go_mean=go_mean,
                stop_mean=stop_mean,
                go_cvar10=go_cvar10,
                stop_cvar10=stop_cvar10,
                go_score=go_score,
                stop_score=stop_score,
            )

            action_rows = {
                "go": (go_returns, go_mean, go_cvar10, go_score),
                "stop": (stop_returns, stop_mean, stop_cvar10, stop_score),
            }
            for action in ACTION_ORDER:
                target_values, target_mean, target_cvar10, target_score = action_rows[action]
                action_samples.append(
                    ActionSample(
                        decision_id=decision_id,
                        action=action,
                        feature=build_action_feature(row, action),
                        target_returns=list(target_values),
                        target_mean=target_mean,
                        target_cvar10=target_cvar10,
                        target_score=target_score,
                        teacher_choice=teacher_choice,
                        mean_choice=mean_choice,
                    )
                )

    if not action_samples:
        fail(f"no usable teacher samples found in: {abs_path}")
    return action_samples, decision_targets, target_return_count


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


def subset_samples(samples: Sequence[ActionSample], allowed_decision_ids: Iterable[str]) -> List[ActionSample]:
    allowed = set(allowed_decision_ids)
    return [sample for sample in samples if sample.decision_id in allowed]


def tensorize(
    samples: Sequence[ActionSample],
    target_return_count: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    features = torch.tensor([sample.feature for sample in samples], dtype=torch.float32, device=device)
    returns = torch.tensor([sample.target_returns for sample in samples], dtype=torch.float32, device=device)
    if features.ndim != 2 or features.shape[1] != INPUT_DIM:
        fail(f"invalid tensorized features shape: {tuple(features.shape)}")
    if returns.ndim != 2 or returns.shape[1] != target_return_count:
        fail(f"invalid tensorized returns shape: {tuple(returns.shape)}")
    return features, returns


class IQNValueNet(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Sequence[int], dropout: float, num_cosines: int):
        super().__init__()
        dims = [int(input_dim)] + [int(v) for v in hidden_sizes]
        encoder_layers: List[nn.Module] = []
        for idx in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.LayerNorm(dims[idx + 1]))
            if dropout > 0:
                encoder_layers.append(nn.Dropout(float(dropout)))
        self.encoder = nn.Sequential(*encoder_layers)
        self.latent_dim = dims[-1]
        self.num_cosines = int(num_cosines)
        self.tau_fc = nn.Sequential(
            nn.Linear(self.num_cosines, self.latent_dim),
            nn.ReLU(),
        )
        self.quantile_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, 1),
        )

    def forward(self, features: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            fail(f"IQN forward expects features [B,D], got={tuple(features.shape)}")
        if taus.ndim != 2:
            fail(f"IQN forward expects taus [B,N], got={tuple(taus.shape)}")
        batch_size = int(features.shape[0])
        if taus.shape[0] != batch_size:
            fail(f"IQN batch mismatch: features={tuple(features.shape)}, taus={tuple(taus.shape)}")

        state_embed = self.encoder(features)
        cosine_idx = torch.arange(1, self.num_cosines + 1, device=features.device, dtype=features.dtype).view(1, 1, -1)
        tau_embed = torch.cos(math.pi * taus.unsqueeze(-1) * cosine_idx)
        tau_embed = self.tau_fc(tau_embed)
        fused = state_embed.unsqueeze(1) * tau_embed
        quantiles = self.quantile_head(fused).squeeze(-1)
        return quantiles


def random_taus(batch_size: int, num_quantiles: int, device: torch.device) -> torch.Tensor:
    return torch.rand(batch_size, num_quantiles, dtype=torch.float32, device=device)


def fixed_taus(batch_size: int, num_quantiles: int, device: torch.device) -> torch.Tensor:
    base = torch.linspace(
        0.5 / num_quantiles,
        1.0 - 0.5 / num_quantiles,
        num_quantiles,
        dtype=torch.float32,
        device=device,
    )
    return base.unsqueeze(0).repeat(batch_size, 1)


def quantile_huber_loss(
    pred_quantiles: torch.Tensor,
    taus: torch.Tensor,
    target_returns: torch.Tensor,
    kappa: float,
) -> torch.Tensor:
    if pred_quantiles.ndim != 2 or taus.ndim != 2 or target_returns.ndim != 2:
        fail(
            "quantile_huber_loss expects 2D tensors: "
            f"pred={tuple(pred_quantiles.shape)}, taus={tuple(taus.shape)}, target={tuple(target_returns.shape)}"
        )
    delta = target_returns.unsqueeze(1) - pred_quantiles.unsqueeze(2)
    abs_delta = delta.abs()
    k = float(max(1e-6, kappa))
    huber = torch.where(abs_delta <= k, 0.5 * delta.pow(2), k * (abs_delta - 0.5 * k))
    weight = torch.abs(taus.unsqueeze(2) - (delta.detach() < 0).to(dtype=pred_quantiles.dtype))
    return (weight * huber / k).mean()


def build_dataloader(features: torch.Tensor, returns: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(features, returns)
    return DataLoader(dataset, batch_size=max(1, int(batch_size)), shuffle=shuffle)


def run_epoch(
    model: IQNValueNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    num_quantiles: int,
    kappa: float,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    for batch_x, batch_returns in loader:
        if training:
            taus = random_taus(int(batch_x.shape[0]), int(num_quantiles), device=device)
        else:
            taus = fixed_taus(int(batch_x.shape[0]), int(num_quantiles), device=device)
        pred = model(batch_x, taus)
        loss = quantile_huber_loss(pred, taus, batch_returns, kappa)
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


def predict_distribution(
    model: IQNValueNet,
    features: torch.Tensor,
    device: torch.device,
    num_quantiles_eval: int,
    batch_size: int,
) -> torch.Tensor:
    loader = DataLoader(features, batch_size=max(1, int(batch_size)), shuffle=False)
    outputs: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch_x in loader:
            taus = fixed_taus(int(batch_x.shape[0]), int(num_quantiles_eval), device=device)
            outputs.append(model(batch_x.to(device), taus).detach().cpu())
    return torch.cat(outputs, dim=0) if outputs else torch.empty((0, num_quantiles_eval), dtype=torch.float32)


def summarize_predicted_distribution(pred_quantiles: torch.Tensor) -> Tuple[List[float], List[float], List[float]]:
    if pred_quantiles.ndim != 2:
        fail(f"predicted quantiles must be 2D, got={tuple(pred_quantiles.shape)}")
    means = pred_quantiles.mean(dim=1).tolist()
    tail_count = max(1, int(math.ceil(pred_quantiles.shape[1] * 0.1)))
    cvar10 = pred_quantiles[:, :tail_count].mean(dim=1).tolist()
    scores = [0.7 * float(m) + 0.3 * float(c) for m, c in zip(means, cvar10)]
    return [float(v) for v in means], [float(v) for v in cvar10], [float(v) for v in scores]


def compute_sample_metrics(
    samples: Sequence[ActionSample],
    pred_means: Sequence[float],
    pred_cvar10: Sequence[float],
    pred_scores: Sequence[float],
) -> Dict[str, float]:
    if len(samples) != len(pred_means) or len(samples) != len(pred_cvar10) or len(samples) != len(pred_scores):
        fail("sample/prediction length mismatch")
    mean_errors = [float(pred_means[i]) - float(samples[i].target_mean) for i in range(len(samples))]
    cvar_errors = [float(pred_cvar10[i]) - float(samples[i].target_cvar10) for i in range(len(samples))]
    score_errors = [float(pred_scores[i]) - float(samples[i].target_score) for i in range(len(samples))]
    return {
        "sample_count": float(len(samples)),
        "mean_mae": float(mean([abs(v) for v in mean_errors])),
        "mean_rmse": float(math.sqrt(mean([v * v for v in mean_errors]))) if mean_errors else 0.0,
        "cvar10_mae": float(mean([abs(v) for v in cvar_errors])),
        "score_mae": float(mean([abs(v) for v in score_errors])),
        "target_mean_avg": float(mean([sample.target_mean for sample in samples])),
        "pred_mean_avg": float(mean(pred_means)),
        "target_cvar10_avg": float(mean([sample.target_cvar10 for sample in samples])),
        "pred_cvar10_avg": float(mean(pred_cvar10)),
    }


def compute_decision_metrics(
    samples: Sequence[ActionSample],
    pred_means: Sequence[float],
    pred_cvar10: Sequence[float],
    pred_scores: Sequence[float],
    decision_targets: Dict[str, DecisionTarget],
) -> Dict[str, float]:
    pairs: Dict[str, Dict[str, Dict[str, float]]] = {}
    for idx, sample in enumerate(samples):
        bucket = pairs.setdefault(sample.decision_id, {})
        bucket[sample.action] = {
            "mean": float(pred_means[idx]),
            "cvar10": float(pred_cvar10[idx]),
            "score": float(pred_scores[idx]),
        }

    total = 0
    mean_hits = 0
    score_hits = 0
    teacher_hits = 0
    mean_margin_abs_errors: List[float] = []
    score_margin_abs_errors: List[float] = []
    for decision_id, pair in pairs.items():
        if "go" not in pair or "stop" not in pair:
            continue
        target = decision_targets.get(decision_id)
        if target is None:
            continue
        pred_mean_choice = "go" if pair["go"]["mean"] > pair["stop"]["mean"] else "stop"
        pred_score_choice = "go" if pair["go"]["score"] > pair["stop"]["score"] else "stop"
        pred_mean_margin = pair["go"]["mean"] - pair["stop"]["mean"]
        pred_score_margin = pair["go"]["score"] - pair["stop"]["score"]
        true_mean_margin = target.go_mean - target.stop_mean
        true_score_margin = target.go_score - target.stop_score
        total += 1
        if pred_mean_choice == target.mean_choice:
            mean_hits += 1
        if pred_score_choice == ("go" if target.go_score > target.stop_score else "stop"):
            score_hits += 1
        if pred_score_choice == target.teacher_choice:
            teacher_hits += 1
        mean_margin_abs_errors.append(abs(pred_mean_margin - true_mean_margin))
        score_margin_abs_errors.append(abs(pred_score_margin - true_score_margin))

    if total <= 0:
        return {
            "decision_count": 0.0,
            "mean_choice_accuracy": 0.0,
            "score_choice_accuracy": 0.0,
            "teacher_choice_accuracy": 0.0,
            "mean_margin_mae": 0.0,
            "score_margin_mae": 0.0,
        }
    return {
        "decision_count": float(total),
        "mean_choice_accuracy": float(mean_hits / total),
        "score_choice_accuracy": float(score_hits / total),
        "teacher_choice_accuracy": float(teacher_hits / total),
        "mean_margin_mae": float(mean(mean_margin_abs_errors)),
        "score_margin_mae": float(mean(score_margin_abs_errors)),
    }


def save_checkpoint(
    out_dir: str,
    model: IQNValueNet,
    args: argparse.Namespace,
    metrics: Dict[str, Any],
    hidden_sizes: Sequence[int],
) -> str:
    payload = {
        "format_version": FORMAT_VERSION,
        "input_dim": INPUT_DIM,
        "hidden_sizes": [int(v) for v in hidden_sizes],
        "dropout": float(args.dropout),
        "num_cosines": int(args.num_cosines),
        "num_quantiles_train": int(args.num_quantiles_train),
        "num_quantiles_eval": int(args.num_quantiles_eval),
        "kappa": float(args.kappa),
        "feature_spec": {
            "base_features": BASE_FEATURES,
            "option_type_one_hot": 4,
            "option_payload": 10,
            "actions": list(ACTION_ORDER),
        },
        "target_spec": {
            "kind": "distributional_return",
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


def tensor_to_list(tensor: torch.Tensor) -> Any:
    return tensor.detach().cpu().tolist()


def export_linear(linear: nn.Linear) -> Dict[str, Any]:
    return {
        "weight": tensor_to_list(linear.weight),
        "bias": tensor_to_list(linear.bias),
    }


def export_layer_norm(layer_norm: nn.LayerNorm) -> Dict[str, Any]:
    return {
        "weight": tensor_to_list(layer_norm.weight),
        "bias": tensor_to_list(layer_norm.bias),
        "eps": float(layer_norm.eps),
    }


def export_runtime_json(
    out_dir: str,
    model: IQNValueNet,
    args: argparse.Namespace,
    metrics: Dict[str, Any],
    hidden_sizes: Sequence[int],
) -> str:
    encoder_blocks: List[Dict[str, Any]] = []
    encoder_modules = list(model.encoder.children())
    idx = 0
    while idx < len(encoder_modules):
        linear = encoder_modules[idx]
        relu = encoder_modules[idx + 1] if idx + 1 < len(encoder_modules) else None
        layer_norm = encoder_modules[idx + 2] if idx + 2 < len(encoder_modules) else None
        if not isinstance(linear, nn.Linear) or not isinstance(relu, nn.ReLU) or not isinstance(layer_norm, nn.LayerNorm):
            fail(f"unexpected encoder layout at module index {idx}")
        idx += 3
        if idx < len(encoder_modules) and isinstance(encoder_modules[idx], nn.Dropout):
            idx += 1
        encoder_blocks.append(
            {
                "linear": export_linear(linear),
                "layer_norm": export_layer_norm(layer_norm),
            }
        )

    tau_modules = list(model.tau_fc.children())
    if len(tau_modules) < 2 or not isinstance(tau_modules[0], nn.Linear) or not isinstance(tau_modules[1], nn.ReLU):
        fail("unexpected tau_fc layout")

    q_modules = list(model.quantile_head.children())
    if (
        len(q_modules) < 3
        or not isinstance(q_modules[0], nn.Linear)
        or not isinstance(q_modules[1], nn.ReLU)
        or not isinstance(q_modules[2], nn.Linear)
    ):
        fail("unexpected quantile_head layout")

    payload = {
        "format_version": RUNTIME_FORMAT_VERSION,
        "input_dim": INPUT_DIM,
        "hidden_sizes": [int(v) for v in hidden_sizes],
        "latent_dim": int(model.latent_dim),
        "num_cosines": int(args.num_cosines),
        "num_quantiles_eval": int(args.num_quantiles_eval),
        "cvar_alpha": 0.1,
        "score_weights": {
            "mean": 0.7,
            "cvar10": 0.3,
        },
        "feature_spec": {
            "base_features": BASE_FEATURES,
            "option_type_one_hot": 4,
            "option_payload": 10,
            "actions": list(ACTION_ORDER),
            "decision_type": "go_stop",
        },
        "target_spec": {
            "kind": "distributional_return",
            "unit": "gold_delta_div_1000",
            "teacher_dataset": os.path.abspath(args.teacher_in),
        },
        "metrics": {
            "best_val_loss": float(metrics.get("training", {}).get("best_val_loss", 0.0)),
            "val_score_choice_accuracy": float(
                metrics.get("val_decision_metrics", {}).get("score_choice_accuracy", 0.0)
            ),
            "val_teacher_choice_accuracy": float(
                metrics.get("val_decision_metrics", {}).get("teacher_choice_accuracy", 0.0)
            ),
        },
        "encoder_blocks": encoder_blocks,
        "tau_fc": {
            "linear": export_linear(tau_modules[0]),
        },
        "quantile_head": {
            "hidden": export_linear(q_modules[0]),
            "output": export_linear(q_modules[2]),
        },
    }
    out_path = os.path.join(out_dir, "runtime.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
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
    if int(args.num_quantiles_train) <= 0 or int(args.num_quantiles_eval) <= 0:
        fail("--num-quantiles-train/eval must be > 0")
    if int(args.num_cosines) <= 0:
        fail("--num-cosines must be > 0")
    if float(args.kappa) <= 0:
        fail("--kappa must be > 0")
    if not (0.0 <= float(args.dropout) < 1.0):
        fail("--dropout must be in [0,1)")

    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    device = resolve_device(args.device)
    ensure_dir(args.out_dir)

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    action_samples, decision_targets, target_return_count = load_teacher_rows(
        args.teacher_in,
        int(args.limit_decisions),
    )
    decision_ids = list(decision_targets.keys())
    train_decision_ids, val_decision_ids = split_decision_ids(decision_ids, float(args.val_ratio), int(args.seed))
    train_samples = subset_samples(action_samples, train_decision_ids)
    val_samples = subset_samples(action_samples, val_decision_ids)
    if not val_samples:
        val_samples = train_samples[:]
        val_decision_ids = train_decision_ids[:]

    train_x, train_returns = tensorize(train_samples, target_return_count, device)
    val_x, val_returns = tensorize(val_samples, target_return_count, device)

    model = IQNValueNet(INPUT_DIM, hidden_sizes, float(args.dropout), int(args.num_cosines)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    train_loader = build_dataloader(train_x, train_returns, batch_size=int(args.batch_size), shuffle=True)
    val_loader = build_dataloader(val_x, val_returns, batch_size=int(args.batch_size), shuffle=False)

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, int(args.epochs) + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            int(args.num_quantiles_train),
            float(args.kappa),
        )
        val_loss = run_epoch(
            model,
            val_loader,
            None,
            device,
            int(args.num_quantiles_eval),
            float(args.kappa),
        )
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

    train_pred_quantiles = predict_distribution(model, train_x, device, int(args.num_quantiles_eval), int(args.batch_size))
    val_pred_quantiles = predict_distribution(model, val_x, device, int(args.num_quantiles_eval), int(args.batch_size))

    train_pred_means, train_pred_cvar10, train_pred_scores = summarize_predicted_distribution(train_pred_quantiles)
    val_pred_means, val_pred_cvar10, val_pred_scores = summarize_predicted_distribution(val_pred_quantiles)

    train_sample_metrics = compute_sample_metrics(train_samples, train_pred_means, train_pred_cvar10, train_pred_scores)
    val_sample_metrics = compute_sample_metrics(val_samples, val_pred_means, val_pred_cvar10, val_pred_scores)
    train_decision_metrics = compute_decision_metrics(
        train_samples,
        train_pred_means,
        train_pred_cvar10,
        train_pred_scores,
        decision_targets,
    )
    val_decision_metrics = compute_decision_metrics(
        val_samples,
        val_pred_means,
        val_pred_cvar10,
        val_pred_scores,
        decision_targets,
    )

    metrics = {
        "format_version": FORMAT_VERSION,
        "teacher_in": os.path.abspath(args.teacher_in),
        "device": str(device),
        "input_dim": INPUT_DIM,
        "hidden_sizes": [int(v) for v in hidden_sizes],
        "num_cosines": int(args.num_cosines),
        "num_quantiles_train": int(args.num_quantiles_train),
        "num_quantiles_eval": int(args.num_quantiles_eval),
        "kappa": float(args.kappa),
        "dataset": {
            "decision_count_total": len(decision_targets),
            "sample_count_total": len(action_samples),
            "target_return_count": target_return_count,
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
    runtime_json_path = export_runtime_json(args.out_dir, model, args, metrics, hidden_sizes)
    metrics["runtime_json"] = os.path.abspath(runtime_json_path)
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
                "runtime_json": os.path.abspath(runtime_json_path),
                "metrics": os.path.abspath(metrics_path),
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "val_mean_choice_accuracy": val_decision_metrics.get("mean_choice_accuracy", 0.0),
                "val_score_choice_accuracy": val_decision_metrics.get("score_choice_accuracy", 0.0),
                "val_teacher_choice_accuracy": val_decision_metrics.get("teacher_choice_accuracy", 0.0),
            }
        )
    )


if __name__ == "__main__":
    main()
