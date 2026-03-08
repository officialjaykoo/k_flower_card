#!/usr/bin/env python3
from __future__ import annotations

"""
Pipeline Stage: 1/3 (neat_train.py -> neat_eval_worker.mjs -> heuristic_duel_worker.mjs)

Execution Flow Map:
1) parse_args()/main(): runtime bootstrap + NEAT train loop
2) LoggedParallelEvaluator: generation-level metrics and gate tracking
3) eval_function(): per-genome worker evaluation call

File Layout Map (top-down):
1) runtime defaults + normalization + env bridge helpers
2) eval function + parallel evaluator + CLI entrypoint
"""

import argparse
import copy
import contextlib
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import random
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional

try:
    import neat  # type: ignore
except Exception:
    neat = None


# =============================================================================
# Section 1. Runtime Schema + Primitive Coercion Helpers
# =============================================================================
ENV_PREFIX = "KFC_NEAT_"
REQUIRED_RUNTIME_KEYS = [
    "format_version",
    "generations",
    "eval_workers",
    "games_per_genome",
    "eval_timeout_sec",
    "max_eval_steps",
    "opponent_policy",
    "opponent_policy_mix",
    "opponent_genome",
    "switch_seats",
    "checkpoint_every",
    "eval_script",
    "seed",
    "fitness_gold_scale",
    "fitness_gold_neutral_delta",
    "fitness_win_weight",
    "fitness_gold_weight",
    "fitness_win_neutral_rate",
    "gate_mode",
    "gate_ema_window",
    "transition_ema_imitation",
    "transition_ema_win_rate",
    "transition_mean_gold_delta_min",
    "transition_best_fitness_min",
    "transition_streak",
    "failure_generation_min",
    "failure_ema_win_rate_max",
    "failure_imitation_max",
    "failure_slope_5_max",
    "failure_slope_metric",
]


def _to_int(value, default):
    try:
        return int(value)
    except Exception:
        return int(default)


def _to_float(value, default):
    try:
        return float(value)
    except Exception:
        return float(default)


def _to_optional_float(value, default=None):
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in ("", "none", "null"):
        return None
    try:
        return float(value)
    except Exception:
        return default


def _to_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    s = str(value).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return bool(default)


def _to_seed(value, default):
    if value is None:
        return str(default)
    s = str(value).strip()
    if not s:
        return str(default)
    return s


def _resolve_runtime_path(base_path: str, child_path: str) -> str:
    if os.path.isabs(child_path):
        return child_path
    base_dir = os.path.dirname(os.path.abspath(base_path)) if base_path else os.getcwd()
    return os.path.normpath(os.path.join(base_dir, child_path))


def _parse_weighted_policy_entry(item: object) -> Optional[Dict[str, Any]]:
    policy = ""
    weight = 0.0
    if isinstance(item, dict):
        policy = str(item.get("policy") or item.get("opponent_policy") or "").strip()
        weight = _to_float(item.get("weight"), 0.0)
    else:
        token = str(item or "").strip()
        if not token:
            return None
        if ":" in token:
            left, right = token.rsplit(":", 1)
            policy = str(left or "").strip()
            weight = _to_float(right, 0.0)
        else:
            policy = token
            weight = 1.0

    if not policy:
        return None
    if (not math.isfinite(weight)) or float(weight) <= 0.0:
        return None
    return {"policy": policy, "weight": float(weight)}


def _parse_opponent_policy_mix(raw_value: object) -> list:
    if raw_value is None:
        return []

    source = []
    if isinstance(raw_value, list):
        source = raw_value
    elif isinstance(raw_value, dict):
        source = [raw_value]
    else:
        text = str(raw_value).strip()
        if not text:
            return []
        if text.startswith("[") or text.startswith("{"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    source = parsed
                elif isinstance(parsed, dict):
                    source = [parsed]
                else:
                    source = []
            except Exception:
                source = []
        else:
            source = [x.strip() for x in text.split(",") if str(x).strip()]

    merged: Dict[str, float] = {}
    ordered_policies = []
    for item in source:
        parsed = _parse_weighted_policy_entry(item)
        if not parsed:
            continue
        policy = str(parsed["policy"]).strip()
        weight = float(parsed["weight"])
        if policy not in merged:
            ordered_policies.append(policy)
            merged[policy] = 0.0
        merged[policy] += weight

    out = []
    for policy in ordered_policies:
        weight = float(merged.get(policy, 0.0))
        if weight > 0.0:
            out.append({"policy": policy, "weight": weight})
    return out


def _load_runtime_config_recursive(path: str, cfg: dict, seen: set[str]) -> None:
    abs_path = os.path.abspath(path)
    if abs_path in seen:
        return
    seen.add(abs_path)
    if not os.path.exists(abs_path):
        return

    with open(abs_path, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        return

    extends_raw = raw.get("extends")
    extends_list = []
    if isinstance(extends_raw, str) and extends_raw.strip():
        extends_list = [extends_raw.strip()]
    elif isinstance(extends_raw, list):
        extends_list = [str(x).strip() for x in extends_raw if str(x).strip()]

    for child in extends_list:
        child_path = _resolve_runtime_path(abs_path, child)
        _load_runtime_config_recursive(child_path, cfg, seen)

    local = dict(raw)
    local.pop("extends", None)
    cfg.update(local)


def _required_value(cfg: dict, key: str):
    if key not in cfg:
        raise RuntimeError(f"runtime missing required key: {key}")
    return cfg.get(key)


def _required_int(cfg: dict, key: str, min_value: Optional[int] = None) -> int:
    raw = _required_value(cfg, key)
    try:
        value = int(raw)
    except Exception:
        raise RuntimeError(f"runtime key '{key}' must be integer")
    if min_value is not None and value < min_value:
        raise RuntimeError(f"runtime key '{key}' must be >= {int(min_value)}")
    return int(value)


def _required_float(cfg: dict, key: str) -> float:
    raw = _required_value(cfg, key)
    try:
        value = float(raw)
    except Exception:
        raise RuntimeError(f"runtime key '{key}' must be number")
    if not math.isfinite(value):
        raise RuntimeError(f"runtime key '{key}' must be finite")
    return float(value)


def _required_optional_float(cfg: dict, key: str) -> Optional[float]:
    raw = _required_value(cfg, key)
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in ("", "none", "null"):
        return None
    try:
        value = float(raw)
    except Exception:
        raise RuntimeError(f"runtime key '{key}' must be number or null")
    if not math.isfinite(value):
        raise RuntimeError(f"runtime key '{key}' must be finite when provided")
    return float(value)


def _required_bool(cfg: dict, key: str) -> bool:
    raw = _required_value(cfg, key)
    if isinstance(raw, bool):
        return raw
    s = str(raw).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    raise RuntimeError(f"runtime key '{key}' must be boolean")


# =============================================================================
# Section 2. Runtime Config Normalization
# =============================================================================
def _load_runtime_config(path: str) -> dict:
    cfg = {}
    if path:
        _load_runtime_config_recursive(path, cfg, set())
    return _normalize_runtime_values(cfg)


def _normalize_runtime_values(cfg: dict) -> dict:
    cfg = dict(cfg or {})
    missing = [k for k in REQUIRED_RUNTIME_KEYS if k not in cfg]
    if missing:
        raise RuntimeError(f"runtime missing required keys: {', '.join(missing)}")

    cfg["format_version"] = str(_required_value(cfg, "format_version") or "").strip()
    if not cfg["format_version"]:
        raise RuntimeError("runtime key 'format_version' must be non-empty")

    cfg["generations"] = _required_int(cfg, "generations", 1)
    cfg["eval_workers"] = _required_int(cfg, "eval_workers", 2)
    cfg["games_per_genome"] = _required_int(cfg, "games_per_genome", 1)
    cfg["eval_timeout_sec"] = _required_int(cfg, "eval_timeout_sec", 10)
    cfg["max_eval_steps"] = _required_int(cfg, "max_eval_steps", 50)

    cfg["opponent_policy"] = str(_required_value(cfg, "opponent_policy") or "").strip()
    cfg["opponent_policy_mix"] = _parse_opponent_policy_mix(_required_value(cfg, "opponent_policy_mix"))
    cfg["opponent_genome"] = str(_required_value(cfg, "opponent_genome") or "").strip()
    cfg["switch_seats"] = _required_bool(cfg, "switch_seats")

    cfg["checkpoint_every"] = _required_int(cfg, "checkpoint_every", 1)
    cfg["eval_script"] = str(_required_value(cfg, "eval_script") or "").strip()
    if not cfg["eval_script"]:
        raise RuntimeError("runtime key 'eval_script' must be non-empty")
    cfg["seed"] = str(_required_value(cfg, "seed") or "").strip()
    if not cfg["seed"]:
        raise RuntimeError("runtime key 'seed' must be non-empty")

    cfg["fitness_gold_scale"] = _required_float(cfg, "fitness_gold_scale")
    cfg["fitness_gold_neutral_delta"] = _required_float(cfg, "fitness_gold_neutral_delta")
    cfg["fitness_win_weight"] = _required_float(cfg, "fitness_win_weight")
    cfg["fitness_gold_weight"] = _required_float(cfg, "fitness_gold_weight")
    cfg["fitness_win_neutral_rate"] = _required_float(cfg, "fitness_win_neutral_rate")

    gate_mode = str(_required_value(cfg, "gate_mode") or "").strip().lower()
    if gate_mode not in ("win_rate_only", "hybrid"):
        raise RuntimeError("runtime key 'gate_mode' must be one of: win_rate_only, hybrid")
    cfg["gate_mode"] = gate_mode
    cfg["gate_ema_window"] = _required_int(cfg, "gate_ema_window", 2)
    cfg["transition_ema_imitation"] = _required_optional_float(cfg, "transition_ema_imitation")
    cfg["transition_ema_win_rate"] = _required_optional_float(cfg, "transition_ema_win_rate")
    cfg["transition_mean_gold_delta_min"] = _required_optional_float(cfg, "transition_mean_gold_delta_min")
    cfg["transition_best_fitness_min"] = _required_optional_float(cfg, "transition_best_fitness_min")
    cfg["transition_streak"] = _required_int(cfg, "transition_streak", 1)
    cfg["failure_generation_min"] = _required_int(cfg, "failure_generation_min", 1)
    cfg["failure_ema_win_rate_max"] = _required_optional_float(cfg, "failure_ema_win_rate_max")
    cfg["failure_imitation_max"] = _required_optional_float(cfg, "failure_imitation_max")
    cfg["failure_slope_5_max"] = _required_float(cfg, "failure_slope_5_max")
    failure_slope_metric = str(_required_value(cfg, "failure_slope_metric") or "").strip().lower()
    if failure_slope_metric not in ("win_rate", "imitation"):
        raise RuntimeError("runtime key 'failure_slope_metric' must be one of: win_rate, imitation")
    cfg["failure_slope_metric"] = failure_slope_metric

    if cfg["fitness_gold_scale"] <= 0:
        raise RuntimeError("runtime key 'fitness_gold_scale' must be > 0")
    if cfg["fitness_win_weight"] < 0:
        raise RuntimeError("runtime key 'fitness_win_weight' must be >= 0")
    if cfg["fitness_gold_weight"] < 0:
        raise RuntimeError("runtime key 'fitness_gold_weight' must be >= 0")
    if (cfg["fitness_win_weight"] + cfg["fitness_gold_weight"]) <= 0:
        raise RuntimeError("runtime keys 'fitness_win_weight' + 'fitness_gold_weight' must be > 0")
    if cfg["fitness_win_neutral_rate"] <= 0 or cfg["fitness_win_neutral_rate"] >= 1:
        raise RuntimeError("runtime key 'fitness_win_neutral_rate' must be in (0,1)")

    if "output_dir" in cfg:
        cfg["output_dir"] = str(cfg.get("output_dir") or "").strip()
        if not cfg["output_dir"]:
            raise RuntimeError("runtime key 'output_dir' must be non-empty")
    return cfg


# =============================================================================
# Section 3. Runtime <-> Environment Bridge
# =============================================================================
def _set_eval_env(runtime: dict, output_dir: str) -> None:
    os.environ[f"{ENV_PREFIX}FORMAT_VERSION"] = str(runtime["format_version"])
    os.environ[f"{ENV_PREFIX}GENERATIONS"] = str(int(runtime["generations"]))
    os.environ[f"{ENV_PREFIX}EVAL_WORKERS"] = str(int(runtime["eval_workers"]))
    os.environ[f"{ENV_PREFIX}CHECKPOINT_EVERY"] = str(int(runtime["checkpoint_every"]))
    os.environ[f"{ENV_PREFIX}EVAL_SCRIPT"] = os.path.abspath(str(runtime["eval_script"]))
    os.environ[f"{ENV_PREFIX}GAMES_PER_GENOME"] = str(int(runtime["games_per_genome"]))
    os.environ[f"{ENV_PREFIX}EVAL_TIMEOUT_SEC"] = str(int(runtime["eval_timeout_sec"]))
    os.environ[f"{ENV_PREFIX}MAX_EVAL_STEPS"] = str(int(runtime["max_eval_steps"]))
    os.environ[f"{ENV_PREFIX}OPPONENT_POLICY"] = str(runtime["opponent_policy"])
    os.environ[f"{ENV_PREFIX}OPPONENT_POLICY_MIX"] = json.dumps(
        runtime.get("opponent_policy_mix") or [],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    os.environ[f"{ENV_PREFIX}OPPONENT_GENOME"] = str(runtime.get("opponent_genome") or "")
    os.environ[f"{ENV_PREFIX}SWITCH_SEATS"] = "1" if bool(runtime["switch_seats"]) else "0"
    os.environ[f"{ENV_PREFIX}SEED"] = str(runtime["seed"])
    os.environ[f"{ENV_PREFIX}OUTPUT_DIR"] = os.path.abspath(output_dir)
    os.environ[f"{ENV_PREFIX}FITNESS_GOLD_SCALE"] = str(float(runtime["fitness_gold_scale"]))
    os.environ[f"{ENV_PREFIX}FITNESS_GOLD_NEUTRAL_DELTA"] = str(
        float(runtime["fitness_gold_neutral_delta"])
    )
    os.environ[f"{ENV_PREFIX}FITNESS_WIN_WEIGHT"] = str(float(runtime["fitness_win_weight"]))
    os.environ[f"{ENV_PREFIX}FITNESS_GOLD_WEIGHT"] = str(float(runtime["fitness_gold_weight"]))
    os.environ[f"{ENV_PREFIX}FITNESS_WIN_NEUTRAL_RATE"] = str(
        float(runtime["fitness_win_neutral_rate"])
    )
    os.environ[f"{ENV_PREFIX}GATE_MODE"] = str(runtime["gate_mode"])
    os.environ[f"{ENV_PREFIX}GATE_EMA_WINDOW"] = str(int(runtime["gate_ema_window"]))
    os.environ[f"{ENV_PREFIX}TRANSITION_EMA_IMITATION"] = str(runtime.get("transition_ema_imitation"))
    os.environ[f"{ENV_PREFIX}TRANSITION_EMA_WIN_RATE"] = str(runtime.get("transition_ema_win_rate"))
    os.environ[f"{ENV_PREFIX}TRANSITION_MEAN_GOLD_DELTA_MIN"] = str(
        runtime.get("transition_mean_gold_delta_min")
    )
    os.environ[f"{ENV_PREFIX}TRANSITION_BEST_FITNESS_MIN"] = str(
        runtime.get("transition_best_fitness_min")
    )
    os.environ[f"{ENV_PREFIX}TRANSITION_STREAK"] = str(int(runtime["transition_streak"]))
    os.environ[f"{ENV_PREFIX}FAILURE_GENERATION_MIN"] = str(int(runtime["failure_generation_min"]))
    os.environ[f"{ENV_PREFIX}FAILURE_EMA_WIN_RATE_MAX"] = str(runtime.get("failure_ema_win_rate_max"))
    os.environ[f"{ENV_PREFIX}FAILURE_IMITATION_MAX"] = str(runtime.get("failure_imitation_max"))
    os.environ[f"{ENV_PREFIX}FAILURE_SLOPE_5_MAX"] = str(float(runtime["failure_slope_5_max"]))
    os.environ[f"{ENV_PREFIX}FAILURE_SLOPE_METRIC"] = str(runtime["failure_slope_metric"])


def _runtime_from_env() -> Dict[str, object]:
    raw: Dict[str, object] = {
        "format_version": os.environ.get(f"{ENV_PREFIX}FORMAT_VERSION"),
        "generations": os.environ.get(f"{ENV_PREFIX}GENERATIONS"),
        "eval_workers": os.environ.get(f"{ENV_PREFIX}EVAL_WORKERS"),
        "eval_script": os.environ.get(f"{ENV_PREFIX}EVAL_SCRIPT") or "",
        "games_per_genome": os.environ.get(f"{ENV_PREFIX}GAMES_PER_GENOME"),
        "eval_timeout_sec": os.environ.get(f"{ENV_PREFIX}EVAL_TIMEOUT_SEC"),
        "max_eval_steps": os.environ.get(f"{ENV_PREFIX}MAX_EVAL_STEPS"),
        "opponent_policy": os.environ.get(f"{ENV_PREFIX}OPPONENT_POLICY"),
        "opponent_policy_mix": os.environ.get(f"{ENV_PREFIX}OPPONENT_POLICY_MIX"),
        "opponent_genome": os.environ.get(f"{ENV_PREFIX}OPPONENT_GENOME"),
        "switch_seats": os.environ.get(f"{ENV_PREFIX}SWITCH_SEATS"),
        "checkpoint_every": os.environ.get(f"{ENV_PREFIX}CHECKPOINT_EVERY"),
        "seed": os.environ.get(f"{ENV_PREFIX}SEED"),
        "output_dir": os.environ.get(f"{ENV_PREFIX}OUTPUT_DIR"),
        "fitness_gold_scale": os.environ.get(f"{ENV_PREFIX}FITNESS_GOLD_SCALE"),
        "fitness_gold_neutral_delta": os.environ.get(f"{ENV_PREFIX}FITNESS_GOLD_NEUTRAL_DELTA"),
        "fitness_win_weight": os.environ.get(f"{ENV_PREFIX}FITNESS_WIN_WEIGHT"),
        "fitness_gold_weight": os.environ.get(f"{ENV_PREFIX}FITNESS_GOLD_WEIGHT"),
        "fitness_win_neutral_rate": os.environ.get(f"{ENV_PREFIX}FITNESS_WIN_NEUTRAL_RATE"),
        "gate_mode": os.environ.get(f"{ENV_PREFIX}GATE_MODE"),
        "gate_ema_window": os.environ.get(f"{ENV_PREFIX}GATE_EMA_WINDOW"),
        "transition_ema_imitation": os.environ.get(f"{ENV_PREFIX}TRANSITION_EMA_IMITATION"),
        "transition_ema_win_rate": os.environ.get(f"{ENV_PREFIX}TRANSITION_EMA_WIN_RATE"),
        "transition_mean_gold_delta_min": os.environ.get(f"{ENV_PREFIX}TRANSITION_MEAN_GOLD_DELTA_MIN"),
        "transition_best_fitness_min": os.environ.get(f"{ENV_PREFIX}TRANSITION_BEST_FITNESS_MIN"),
        "transition_streak": os.environ.get(f"{ENV_PREFIX}TRANSITION_STREAK"),
        "failure_generation_min": os.environ.get(f"{ENV_PREFIX}FAILURE_GENERATION_MIN"),
        "failure_ema_win_rate_max": os.environ.get(f"{ENV_PREFIX}FAILURE_EMA_WIN_RATE_MAX"),
        "failure_imitation_max": os.environ.get(f"{ENV_PREFIX}FAILURE_IMITATION_MAX"),
        "failure_slope_5_max": os.environ.get(f"{ENV_PREFIX}FAILURE_SLOPE_5_MAX"),
        "failure_slope_metric": os.environ.get(f"{ENV_PREFIX}FAILURE_SLOPE_METRIC"),
    }
    return _normalize_runtime_values(raw)


_RUNTIME_FROM_ENV_CACHE: Optional[Dict[str, object]] = None


def _runtime_from_env_cached(force_reload: bool = False) -> Dict[str, object]:
    global _RUNTIME_FROM_ENV_CACHE
    if force_reload or _RUNTIME_FROM_ENV_CACHE is None:
        _RUNTIME_FROM_ENV_CACHE = _runtime_from_env()
    return _RUNTIME_FROM_ENV_CACHE


# =============================================================================
# Section 6. Logging + Numeric Utilities
# =============================================================================
def _append_eval_failure_log(output_dir: str, record: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "eval_failures.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")


def _export_neat_python_genome(genome, config) -> dict:
    gcfg = config.genome_config
    input_keys = [int(x) for x in gcfg.input_keys]
    output_keys = [int(x) for x in gcfg.output_keys]

    default_activation = str(getattr(gcfg, "activation_default", "tanh") or "tanh")
    default_aggregation = str(getattr(gcfg, "aggregation_default", "sum") or "sum")

    nodes = {}
    for node_key, node_gene in genome.nodes.items():
        nk = int(node_key)
        nodes[str(nk)] = {
            "node_id": nk,
            "activation": str(getattr(node_gene, "activation", default_activation) or default_activation),
            "aggregation": str(getattr(node_gene, "aggregation", default_aggregation) or default_aggregation),
            "bias": float(getattr(node_gene, "bias", 0.0) or 0.0),
            "response": float(getattr(node_gene, "response", 1.0) or 1.0),
        }

    connections = []
    for key, conn_gene in genome.connections.items():
        in_node = int(key[0])
        out_node = int(key[1])
        connections.append(
            {
                "in_node": in_node,
                "out_node": out_node,
                "weight": float(getattr(conn_gene, "weight", 0.0) or 0.0),
                "enabled": bool(getattr(conn_gene, "enabled", True)),
            }
        )

    return {
        "format_version": "neat_python_genome_v1",
        "input_keys": input_keys,
        "output_keys": output_keys,
        "nodes": nodes,
        "connections": connections,
    }


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_optional_float(value):
    v = _safe_float(value, float("nan"))
    return v if v == v else None


def _ema(previous, current, alpha):
    if previous is None:
        return float(current)
    a = float(alpha)
    return a * float(current) + (1.0 - a) * float(previous)


def _five_gen_slope(values):
    if len(values) < 5:
        return 0.0
    recent = values[-5:]
    return (float(recent[-1]) - float(recent[0])) / 4.0


def _stddev(values):
    vals = [float(v) for v in values]
    if len(vals) <= 1:
        return 0.0
    m = sum(vals) / len(vals)
    var = sum((x - m) * (x - m) for x in vals) / len(vals)
    return math.sqrt(max(0.0, var))


def _quantile(values, q):
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    vals.sort()
    idx = max(0, min(len(vals) - 1, int((len(vals) - 1) * float(q))))
    return vals[idx]


def _stable_unit_float(token: str) -> float:
    raw = str(token or "").encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return float(value) / float((1 << 64) - 1)


def _select_opponent_policy(runtime: dict, seed_text: str, generation: int, genome_key: int) -> str:
    fallback = str(runtime.get("opponent_policy") or "").strip()
    if fallback:
        return fallback
    mix = runtime.get("opponent_policy_mix") or []
    if not isinstance(mix, list) or len(mix) <= 0:
        return fallback

    weighted = []
    total = 0.0
    for item in mix:
        if not isinstance(item, dict):
            continue
        policy = str(item.get("policy") or "").strip()
        weight = _to_float(item.get("weight"), 0.0)
        if not policy:
            continue
        if (not math.isfinite(weight)) or weight <= 0.0:
            continue
        ww = float(weight)
        weighted.append((policy, ww))
        total += ww

    if total <= 0.0 or len(weighted) <= 0:
        return fallback

    needle = _stable_unit_float(
        f"{seed_text}|gen={int(generation)}|genome={int(genome_key)}|opponent_policy_mix"
    ) * total
    acc = 0.0
    for policy, weight in weighted:
        acc += weight
        if needle <= acc:
            return policy
    return weighted[-1][0]


# =============================================================================
# Section 7. Single Genome Evaluation Worker
# =============================================================================
def eval_function(genome, config, seed_override="", generation=-1, genome_key=-1):
    runtime = _runtime_from_env_cached()
    eval_script = str(runtime["eval_script"] or "")
    seed_text = str(seed_override or runtime["seed"])
    opponent_policy = str(runtime.get("opponent_policy") or "").strip()
    opponent_policy_mix = runtime.get("opponent_policy_mix") or []
    has_opponent_policy = bool(opponent_policy)
    has_opponent_policy_mix = isinstance(opponent_policy_mix, list) and len(opponent_policy_mix) > 0
    requires_genome_opponent = (
        str(opponent_policy).strip().lower() == "genome"
        or any(
            str((item or {}).get("policy") or "").strip().lower() == "genome"
            for item in (opponent_policy_mix if isinstance(opponent_policy_mix, list) else [])
            if isinstance(item, dict)
        )
    )
    opponent_genome = str(runtime.get("opponent_genome") or "").strip()
    failure_meta = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "generation": int(generation),
        "genome_key": int(genome_key),
        "seed_used": seed_text,
        "opponent_policy": opponent_policy,
        "opponent_policy_mix": opponent_policy_mix,
    }
    if not eval_script or not os.path.exists(eval_script):
        _append_eval_failure_log(
            str(runtime["output_dir"]),
            dict(failure_meta, reason="eval_script_missing", eval_script=eval_script),
        )
        return {"fitness": -1e9, "seed_used": seed_text, "eval_ok": False}

    payload = _export_neat_python_genome(genome, config)
    output_dir = str(runtime["output_dir"])

    if (not has_opponent_policy) and (not has_opponent_policy_mix):
        _append_eval_failure_log(
            output_dir,
            dict(failure_meta, reason="opponent_policy_missing"),
        )
        return {"fitness": -1e9, "seed_used": seed_text, "eval_ok": False}

    if requires_genome_opponent:
        if not opponent_genome:
            _append_eval_failure_log(
                output_dir,
                dict(failure_meta, reason="opponent_genome_missing", opponent_policy=opponent_policy),
            )
            return {"fitness": -1e9, "seed_used": seed_text, "eval_ok": False}
        opponent_genome = os.path.abspath(opponent_genome)
        if not os.path.exists(opponent_genome):
            _append_eval_failure_log(
                output_dir,
                dict(
                    failure_meta,
                    reason="opponent_genome_not_found",
                    opponent_policy=opponent_policy,
                    opponent_genome=opponent_genome,
                ),
            )
            return {"fitness": -1e9, "seed_used": seed_text, "eval_ok": False}

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix="_neat_python_genome.json",
            delete=False,
            encoding="utf-8",
        ) as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
            genome_path = f.name

        cmd = [
            "node",
            eval_script,
            "--genome",
            genome_path,
            "--games",
            str(int(runtime["games_per_genome"])),
            "--seed",
            seed_text,
            "--max-steps",
            str(int(runtime["max_eval_steps"])),
            "--switch-seats",
            "1" if bool(runtime["switch_seats"]) else "0",
            "--fitness-gold-scale",
            str(float(runtime["fitness_gold_scale"])),
            "--fitness-gold-neutral-delta",
            str(float(runtime["fitness_gold_neutral_delta"])),
            "--fitness-win-weight",
            str(float(runtime["fitness_win_weight"])),
            "--fitness-gold-weight",
            str(float(runtime["fitness_gold_weight"])),
            "--fitness-win-neutral-rate",
            str(float(runtime["fitness_win_neutral_rate"])),
        ]
        if has_opponent_policy:
            cmd.extend(["--opponent-policy", opponent_policy])
        elif has_opponent_policy_mix:
            cmd.extend(
                [
                    "--opponent-policy-mix",
                    json.dumps(opponent_policy_mix, ensure_ascii=False, separators=(",", ":")),
                ]
            )
        if requires_genome_opponent:
            cmd.extend(["--opponent-genome", opponent_genome])

        proc = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=max(10, int(runtime["eval_timeout_sec"])),
        )
        lines = [x.strip() for x in str(proc.stdout or "").splitlines() if x.strip()]
        if not lines:
            _append_eval_failure_log(
                output_dir,
                dict(failure_meta, reason="worker_empty_stdout", stderr=str(proc.stderr or "")),
            )
            return {"fitness": -1e9, "seed_used": seed_text, "eval_ok": False}
        summary = json.loads(lines[-1])
        if not isinstance(summary, dict):
            summary = {}
        summary["fitness"] = _safe_float(summary.get("fitness"), -1e9)
        summary["seed_used"] = seed_text
        summary["eval_ok"] = True
        return summary
    except Exception as exc:
        _append_eval_failure_log(
            output_dir,
            dict(
                failure_meta,
                reason="worker_exception",
                error=repr(exc),
                traceback=traceback.format_exc(),
            ),
        )
        return {"fitness": -1e9, "seed_used": seed_text, "eval_ok": False}
    finally:
        try:
            if "genome_path" in locals() and genome_path and os.path.exists(genome_path):
                os.remove(genome_path)
        except Exception:
            pass


# =============================================================================
# Section 8. Parallel Evaluator + Gate Tracking
# =============================================================================
class LoggedParallelEvaluator:
    def __init__(self, num_workers: int, output_dir: str, runtime: dict):
        self.num_workers = max(1, int(num_workers))
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.runtime_seed = str(runtime["seed"])
        self.pool = mp.Pool(processes=self.num_workers)

        self.eval_metrics_log = os.path.join(self.output_dir, "eval_metrics.ndjson")
        self.generation_metrics_log = os.path.join(self.output_dir, "generation_metrics.ndjson")
        self.gate_state_path = os.path.join(self.output_dir, "gate_state.json")

        self.generation = -1
        self.gate_mode = str(runtime["gate_mode"])
        self.ema_window = int(runtime["gate_ema_window"])
        self.ema_alpha = 2.0 / (float(self.ema_window) + 1.0)
        self.transition_ema_imitation = runtime.get("transition_ema_imitation")
        self.transition_ema_win_rate = runtime.get("transition_ema_win_rate")
        self.transition_mean_gold_delta_min = runtime.get("transition_mean_gold_delta_min")
        self.transition_best_fitness_min = runtime.get("transition_best_fitness_min")
        self.transition_streak = int(runtime["transition_streak"])
        self.failure_generation_min = int(runtime["failure_generation_min"])
        self.failure_ema_win_rate_max = runtime.get("failure_ema_win_rate_max")
        self.failure_imitation_max = runtime.get("failure_imitation_max")
        self.failure_slope_5_max = float(runtime["failure_slope_5_max"])
        self.failure_slope_metric = str(runtime["failure_slope_metric"])
        self.ema_imitation = None
        self.ema_win_rate = None
        self.gate_streak = 0
        self.transition_generation = None
        self.failure_generation = None
        self.best_imitation_history = []
        self.best_win_rate_history = []
        self.gate_state = {}

    def _append_lines(self, path: str, records):
        if not records:
            return
        with open(path, "a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False))
                f.write("\n")

    def _thresholds(self) -> Dict[str, Any]:
        return {
            "gate_mode": self.gate_mode,
            "ema_window": int(self.ema_window),
            "transition_ema_imitation": (
                float(self.transition_ema_imitation)
                if self.transition_ema_imitation is not None
                else None
            ),
            "transition_ema_win_rate": (
                float(self.transition_ema_win_rate)
                if self.transition_ema_win_rate is not None
                else None
            ),
            "transition_mean_gold_delta_min": (
                float(self.transition_mean_gold_delta_min)
                if self.transition_mean_gold_delta_min is not None
                else None
            ),
            "transition_best_fitness_min": (
                float(self.transition_best_fitness_min)
                if self.transition_best_fitness_min is not None
                else None
            ),
            "transition_streak": int(self.transition_streak),
            "failure_generation_min": int(self.failure_generation_min),
            "failure_ema_win_rate_max": (
                float(self.failure_ema_win_rate_max)
                if self.failure_ema_win_rate_max is not None
                else None
            ),
            "failure_imitation_max": (
                float(self.failure_imitation_max)
                if self.failure_imitation_max is not None
                else None
            ),
            "failure_slope_5_max": float(self.failure_slope_5_max),
            "failure_slope_metric": self.failure_slope_metric,
        }

    def _is_imitation_required(self) -> bool:
        return (
            self.gate_mode == "hybrid"
            or self.failure_slope_metric == "imitation"
            or self.failure_imitation_max is not None
        )

    def _is_valid_gate_record(self, record: dict) -> bool:
        if not bool(record.get("eval_ok")):
            return False
        if "win_rate" not in record:
            return False
        win_rate = _safe_float(record.get("win_rate"), float("nan"))
        if win_rate != win_rate:
            return False
        if self._is_imitation_required():
            if "imitation_weighted_score" not in record:
                return False
            imitation = _safe_float(record.get("imitation_weighted_score"), float("nan"))
            if imitation != imitation:
                return False
        return True

    def _update_gate(self, best_record: Optional[dict], valid_count: int, total_count: int):
        if best_record is None or valid_count <= 0:
            # invalid_generation: EMA is not updated and streak is forcibly reset
            self.gate_streak = 0
            self.gate_state = {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "generation": int(self.generation),
                "data_quality": "invalid_generation",
                "valid_record_count": int(valid_count),
                "total_record_count": int(total_count),
                "ema_window": int(self.ema_window),
                "ema_alpha": float(self.ema_alpha),
                "ema_imitation": float(self.ema_imitation) if self.ema_imitation is not None else None,
                "ema_win_rate": float(self.ema_win_rate) if self.ema_win_rate is not None else None,
                "gate_streak": int(self.gate_streak),
                "transition_ready": bool(self.transition_generation is not None),
                "transition_generation": self.transition_generation,
                "failure_triggered": bool(self.failure_generation is not None),
                "failure_generation": self.failure_generation,
                "latest_imitation": None,
                "latest_win_rate": None,
                "latest_mean_gold_delta": None,
                "latest_best_fitness": None,
                "latest_imitation_slope_5": None,
                "latest_win_rate_slope_5": None,
                "thresholds": self._thresholds(),
            }
            return

        imitation = _safe_float(best_record.get("imitation_weighted_score"), 0.0)
        win_rate = _safe_float(best_record.get("win_rate"), 0.0)
        mean_gold_delta = _safe_float(best_record.get("mean_gold_delta"), float("nan"))
        best_fitness = _safe_float(best_record.get("fitness"), -1e9)
        self.best_imitation_history.append(imitation)
        self.best_win_rate_history.append(win_rate)

        self.ema_imitation = _ema(self.ema_imitation, imitation, self.ema_alpha)
        self.ema_win_rate = _ema(self.ema_win_rate, win_rate, self.ema_alpha)

        transition_ok = True
        if self.transition_ema_win_rate is not None:
            transition_ok = transition_ok and (self.ema_win_rate >= self.transition_ema_win_rate)
        if self.transition_mean_gold_delta_min is not None:
            transition_ok = transition_ok and (
                mean_gold_delta == mean_gold_delta
                and mean_gold_delta > self.transition_mean_gold_delta_min
            )
        if self.gate_mode == "hybrid" and self.transition_ema_imitation is not None:
            transition_ok = transition_ok and (self.ema_imitation >= self.transition_ema_imitation)
        if self.transition_best_fitness_min is not None:
            transition_ok = transition_ok and (best_fitness >= self.transition_best_fitness_min)

        if transition_ok:
            self.gate_streak += 1
        else:
            self.gate_streak = 0

        if self.transition_generation is None and self.gate_streak >= self.transition_streak:
            self.transition_generation = self.generation

        imitation_slope_5 = _five_gen_slope(self.best_imitation_history)
        win_rate_slope_5 = _five_gen_slope(self.best_win_rate_history)
        slope_5 = win_rate_slope_5 if self.failure_slope_metric == "win_rate" else imitation_slope_5
        metric_checks = []
        if self.failure_ema_win_rate_max is not None:
            metric_checks.append(self.ema_win_rate < self.failure_ema_win_rate_max)
        if self.failure_imitation_max is not None:
            metric_checks.append(imitation < self.failure_imitation_max)
        metric_low = all(metric_checks) if metric_checks else False

        is_failure = (
            self.failure_generation is None
            and (self.generation + 1) >= self.failure_generation_min
            and metric_low
            and slope_5 < self.failure_slope_5_max
        )
        if is_failure:
            self.failure_generation = self.generation

        self.gate_state = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "generation": int(self.generation),
            "data_quality": "valid_generation",
            "valid_record_count": int(valid_count),
            "total_record_count": int(total_count),
            "ema_window": int(self.ema_window),
            "ema_alpha": float(self.ema_alpha),
            "ema_imitation": float(self.ema_imitation),
            "ema_win_rate": float(self.ema_win_rate),
            "gate_streak": int(self.gate_streak),
            "transition_ready": bool(self.transition_generation is not None),
            "transition_generation": self.transition_generation,
            "failure_triggered": bool(self.failure_generation is not None),
            "failure_generation": self.failure_generation,
            "latest_imitation": float(imitation),
            "latest_win_rate": float(win_rate),
            "latest_mean_gold_delta": float(mean_gold_delta) if mean_gold_delta == mean_gold_delta else None,
            "latest_best_fitness": float(best_fitness),
            "latest_imitation_slope_5": float(imitation_slope_5),
            "latest_win_rate_slope_5": float(win_rate_slope_5),
            "thresholds": self._thresholds(),
        }

    def evaluate(self, genomes, config):
        self.generation += 1
        seed_for_generation = f"{self.runtime_seed}|gen={self.generation}"
        jobs = []
        for genome_key, genome in genomes:
            job = self.pool.apply_async(
                eval_function,
                (genome, config, seed_for_generation, int(self.generation), int(genome_key)),
            )
            jobs.append((genome_key, genome, job))

        records = []
        for genome_key, genome, job in jobs:
            try:
                result = job.get()
            except Exception as exc:
                result = {
                    "fitness": -1e9,
                    "worker_exception": repr(exc),
                    "traceback": traceback.format_exc(),
                    "seed_used": seed_for_generation,
                    "eval_ok": False,
                }

            if isinstance(result, dict):
                fitness = _safe_float(result.get("fitness"), -1e9)
            else:
                fitness = _safe_float(result, -1e9)
                result = {"fitness": fitness, "seed_used": seed_for_generation}

            genome.fitness = float(fitness)
            record = dict(result)
            record["saved_at"] = datetime.now(timezone.utc).isoformat()
            record["generation"] = int(self.generation)
            record["genome_key"] = int(genome_key)
            record["seed_used"] = str(record.get("seed_used") or seed_for_generation)
            record["fitness"] = float(fitness)
            record["eval_ok"] = bool(record.get("eval_ok", False))
            node_count = len(getattr(genome, "nodes", {}) or {})
            conn_genes = getattr(genome, "connections", {}) or {}
            enabled_count = 0
            for conn_gene in conn_genes.values():
                if bool(getattr(conn_gene, "enabled", True)):
                    enabled_count += 1
            record["num_nodes"] = int(node_count)
            record["num_connections"] = int(enabled_count)
            record["num_connections_total"] = int(len(conn_genes))
            records.append(record)

        self._append_lines(self.eval_metrics_log, records)

        valid_records = [r for r in records if self._is_valid_gate_record(r)]

        if records:
            best_record = max(records, key=lambda r: _safe_float(r.get("fitness"), -1e9))
            fitness_values = [_safe_float(r.get("fitness"), -1e9) for r in records]
            valid_fitness_values = [_safe_float(r.get("fitness"), -1e9) for r in valid_records]
            valid_win_values = [_safe_float(r.get("win_rate"), 0.0) for r in valid_records]
            valid_imit_values = []
            valid_go_games_values = []
            valid_go_rate_values = []
            valid_go_fail_rate_values = []
            for r in valid_records:
                if "imitation_weighted_score" not in r:
                    pass
                else:
                    v = _safe_float(r.get("imitation_weighted_score"), float("nan"))
                    if v == v:
                        valid_imit_values.append(v)
                go_games = _safe_optional_float(r.get("go_games"))
                if go_games is not None:
                    valid_go_games_values.append(go_games)
                go_rate = _safe_optional_float(r.get("go_rate"))
                if go_rate is not None:
                    valid_go_rate_values.append(go_rate)
                go_fail_rate = _safe_optional_float(r.get("go_fail_rate"))
                if go_fail_rate is not None:
                    valid_go_fail_rate_values.append(go_fail_rate)
            valid_eval_ms = [
                max(0.0, _safe_float(r.get("eval_time_ms"), 0.0))
                for r in valid_records
                if r.get("eval_time_ms") is not None
            ]
            generation_record = {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "generation": int(self.generation),
                "seed_used": seed_for_generation,
                "population_size": len(records),
                "valid_record_count": len(valid_records),
                "invalid_record_count": max(0, len(records) - len(valid_records)),
                "data_quality": "valid_generation" if len(valid_records) > 0 else "invalid_generation",
                "best_genome_key": int(best_record.get("genome_key", -1)),
                "best_fitness": _safe_float(best_record.get("fitness"), -1e9),
                "mean_fitness": sum(fitness_values) / max(1, len(fitness_values)),
                "std_fitness": _stddev(valid_fitness_values),
                "mean_win_rate": (
                    sum(valid_win_values) / max(1, len(valid_win_values)) if len(valid_win_values) > 0 else None
                ),
                "mean_imitation_weighted_score": (
                    sum(valid_imit_values) / max(1, len(valid_imit_values))
                    if len(valid_imit_values) > 0
                    else None
                ),
                "mean_go_games": (
                    sum(valid_go_games_values) / max(1, len(valid_go_games_values))
                    if len(valid_go_games_values) > 0
                    else None
                ),
                "mean_go_rate": (
                    sum(valid_go_rate_values) / max(1, len(valid_go_rate_values))
                    if len(valid_go_rate_values) > 0
                    else None
                ),
                "mean_go_fail_rate": (
                    sum(valid_go_fail_rate_values) / max(1, len(valid_go_fail_rate_values))
                    if len(valid_go_fail_rate_values) > 0
                    else None
                ),
                "best_win_rate": (
                    _safe_float(best_record.get("win_rate"), 0.0)
                    if self._is_valid_gate_record(best_record)
                    else None
                ),
                "best_imitation_weighted_score": (
                    _safe_optional_float(best_record.get("imitation_weighted_score"))
                    if (
                        self._is_valid_gate_record(best_record)
                        and "imitation_weighted_score" in best_record
                    )
                    else None
                ),
                "best_genome_nodes": int(best_record.get("num_nodes", 0)),
                "best_genome_connections": int(best_record.get("num_connections", 0)),
                "mean_eval_time_ms": (
                    sum(valid_eval_ms) / max(1, len(valid_eval_ms)) if len(valid_eval_ms) > 0 else None
                ),
                "p90_eval_time_ms": (
                    _quantile(valid_eval_ms, 0.9) if len(valid_eval_ms) > 0 else None
                ),
            }
        else:
            generation_record = {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "generation": int(self.generation),
                "seed_used": seed_for_generation,
                "population_size": 0,
                "valid_record_count": 0,
                "invalid_record_count": 0,
                "data_quality": "invalid_generation",
                "best_genome_key": -1,
                "best_fitness": -1e9,
                "mean_fitness": -1e9,
                "std_fitness": 0.0,
                "mean_win_rate": None,
                "mean_imitation_weighted_score": None,
                "mean_go_games": None,
                "mean_go_rate": None,
                "mean_go_fail_rate": None,
                "best_win_rate": None,
                "best_imitation_weighted_score": None,
                "best_genome_nodes": 0,
                "best_genome_connections": 0,
                "mean_eval_time_ms": None,
                "p90_eval_time_ms": None,
            }
            best_record = None

        best_gate_record = (
            max(valid_records, key=lambda r: _safe_float(r.get("fitness"), -1e9))
            if len(valid_records) > 0
            else None
        )
        self._update_gate(best_gate_record, len(valid_records), len(records))
        generation_record["gate_state"] = dict(self.gate_state)
        self._append_lines(self.generation_metrics_log, [generation_record])
        with open(self.gate_state_path, "w", encoding="utf-8") as f:
            json.dump(self.gate_state, f, ensure_ascii=False, indent=2)

    def close(self):
        if self.pool is None:
            return
        self.pool.close()
        self.pool.join()
        self.pool = None

    def snapshot(self):
        return dict(self.gate_state)


# =============================================================================
# Section 9. CLI + Config Bootstrap
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="neat-python training runner for k_flower_card")
    parser.add_argument(
        "--config-feedforward",
        default="scripts/configs/neat_feedforward.ini",
        help="Path to neat-python config file",
    )
    parser.add_argument(
        "--runtime-config",
        default="scripts/configs/runtime_phase1.json",
        help="Path to runtime JSON (workers/games/checkpoint interval)",
    )
    parser.add_argument("--output-dir", default="logs/NEAT/neat_python", help="Output directory")
    parser.add_argument("--resume", default="", help="Checkpoint file path to resume")
    parser.add_argument(
        "--seed-genome",
        default="",
        help="Path to winner_genome.pkl used to reseed a fresh population for the next phase",
    )
    parser.add_argument(
        "--base-generation",
        type=int,
        default=0,
        help="Resume start generation offset for checkpoint naming",
    )
    parser.add_argument("--generations", type=int, default=0, help="Override generations")
    parser.add_argument("--workers", type=int, default=0, help="Override worker count")
    parser.add_argument("--games-per-genome", type=int, default=0, help="Override games per genome")
    parser.add_argument("--eval-timeout-sec", type=int, default=0, help="Override evaluation timeout seconds")
    parser.add_argument("--max-eval-steps", type=int, default=0, help="Override max game steps per evaluation")
    parser.add_argument("--opponent-policy", default="", help="Override opponent policy")
    parser.add_argument(
        "--opponent-genome",
        default="",
        help="Path to fixed opponent genome JSON (used when opponent-policy=genome)",
    )
    parser.add_argument("--checkpoint-every", type=int, default=0, help="Override checkpoint interval")
    parser.add_argument("--seed", default="", help="Override runtime seed")
    parser.add_argument(
        "--fitness-gold-scale",
        type=float,
        default=0.0,
        help="Override fitness gold scale denominator",
    )
    parser.add_argument(
        "--fitness-gold-neutral-delta",
        type=float,
        default=float("nan"),
        help="Override gold neutral baseline delta for gold_norm",
    )
    parser.add_argument(
        "--fitness-win-weight",
        type=float,
        default=float("nan"),
        help="Override win-rate fitness weight",
    )
    parser.add_argument(
        "--fitness-gold-weight",
        type=float,
        default=float("nan"),
        help="Override loss-rate fitness penalty weight",
    )
    parser.add_argument(
        "--fitness-win-neutral-rate",
        type=float,
        default=float("nan"),
        help="Override neutral baseline win rate for win_norm",
    )
    parser.add_argument(
        "--profile-name",
        default="",
        help="Optional profile label included in run_summary",
    )
    parser.add_argument(
        "--switch-seats",
        dest="switch_seats",
        action="store_true",
        default=None,
        help="Force seat switching by game index",
    )
    parser.add_argument(
        "--fixed-seats",
        dest="switch_seats",
        action="store_false",
        default=None,
        help="Force fixed seats (control actor always human side)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use synthetic fitness (no game simulation).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable per-generation stdout reporter logs.",
    )
    return parser.parse_args()


def _build_config(config_path: str):
    if neat is None:
        raise RuntimeError("neat-python is not installed. Install with: pip install neat-python")
    # neat-python loads config files using the platform-default text codec.
    # On Korean Windows this is often cp949, which fails on UTF-8 BOM.
    # Keep repository files as UTF-8 BOM, but pass a temporary BOM-stripped
    # copy to neat.Config for robust loading.
    with open(config_path, "r", encoding="utf-8-sig") as f:
        config_text = f.read()
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix="_neat_feedforward.ini",
        delete=False,
        encoding="utf-8",
    ) as f_tmp:
        f_tmp.write(config_text)
        temp_config_path = f_tmp.name
    try:
        return neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            temp_config_path,
        )
    finally:
        try:
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
        except Exception:
            pass


def _run_dry_eval(population, _config):
    for _, genome in population:
        genome.fitness = float(len(genome.connections)) * 0.001 + float(len(genome.nodes)) * 0.0001


def _build_seeded_population(cfg, seed_genome_path: str):
    if neat is None:
        raise RuntimeError("neat-python is not installed. Install with: pip install neat-python")

    seed_path = os.path.abspath(str(seed_genome_path or "").strip())
    if not seed_path:
        raise RuntimeError("seed genome path is empty")
    if not os.path.exists(seed_path):
        raise RuntimeError(f"seed genome not found: {seed_genome_path}")

    with open(seed_path, "rb") as f:
        seed_genome = pickle.load(f)

    if seed_genome is None or not hasattr(seed_genome, "nodes") or not hasattr(seed_genome, "connections"):
        raise RuntimeError(f"invalid seed genome pickle: {seed_genome_path}")

    population = neat.Population(cfg)
    population_size = max(1, int(cfg.pop_size))
    existing_keys = sorted(population.population.keys())
    if len(existing_keys) < population_size:
        raise RuntimeError("failed to initialize base population for seed expansion")

    tracker = population.reproduction.innovation_tracker
    cfg.genome_config.innovation_tracker = tracker
    max_innovation = int(getattr(tracker, "global_counter", 0) or 0)
    for conn_gene in (getattr(seed_genome, "connections", {}) or {}).values():
        innovation = getattr(conn_gene, "innovation", None)
        if innovation is None:
            continue
        try:
            max_innovation = max(max_innovation, int(innovation))
        except Exception:
            continue
    tracker.global_counter = int(max_innovation)
    tracker.reset_generation()

    seeded_population = {}
    population.reproduction.ancestors = {}
    seed_key = int(getattr(seed_genome, "key", 0) or 0)

    for idx, genome_key in enumerate(existing_keys[:population_size]):
        genome = copy.deepcopy(seed_genome)
        genome.key = int(genome_key)
        genome.fitness = None
        if idx > 0:
            genome.mutate(cfg.genome_config)
            if (idx % 4) == 0:
                genome.mutate(cfg.genome_config)
        seeded_population[int(genome_key)] = genome
        population.reproduction.ancestors[int(genome_key)] = (seed_key,)

    population.population = seeded_population
    population.generation = 0
    population.best_genome = None
    population.species = cfg.species_set_type(cfg.species_set_config, population.reporters)
    population.species.speciate(cfg, population.population, population.generation)
    return population


if neat is not None:

    class OffsetCheckpointer(neat.Checkpointer):
        def __init__(
            self,
            base_generation: int,
            start_generation: int,
            total_generations: int,
            generation_interval: int,
            filename_prefix: str,
        ):
            super().__init__(
                generation_interval=max(1, int(generation_interval)),
                filename_prefix=str(filename_prefix),
            )
            self.base_generation = int(base_generation)
            self._start_generation = int(start_generation)
            self.total_generations = max(1, int(total_generations))
            self.last_generation_checkpoint = int(start_generation)
            self._last_saved_display_generation = None

        def _display_from_next_generation(self, next_generation: int) -> int:
            return int(self.base_generation + (int(next_generation) - self._start_generation))

        def _save_checkpoint_with_display(
            self,
            config,
            population,
            species_set,
            state_generation: int,
            display_generation: int,
        ) -> None:
            filename = f"{self.filename_prefix}gen{int(display_generation)}"
            print(f"Saving checkpoint to {filename}", file=sys.stderr)
            with gzip.open(filename, "w", compresslevel=5) as f:
                data = (int(state_generation), config, population, species_set, random.getstate())
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            self._last_saved_display_generation = int(display_generation)

        def save_checkpoint(self, config, population, species_set, generation):
            display_generation = self._display_from_next_generation(int(generation))
            self._save_checkpoint_with_display(
                config=config,
                population=population,
                species_set=species_set,
                state_generation=int(generation),
                display_generation=display_generation,
            )

        def end_generation(self, config, population, species_set):
            checkpoint_due = False

            if self.time_interval_seconds is not None:
                dt = time.time() - self.last_time_checkpoint
                if dt >= self.time_interval_seconds:
                    checkpoint_due = True

            next_generation = self.current_generation + 1

            if (not checkpoint_due) and (self.generation_interval is not None):
                dg = next_generation - self.last_generation_checkpoint
                if dg >= self.generation_interval:
                    checkpoint_due = True

            if not checkpoint_due:
                return

            is_final_generation = (next_generation - self._start_generation) >= self.total_generations
            if self.base_generation > 0 and is_final_generation:
                return

            self.save_checkpoint(config, population, species_set, next_generation)
            self.last_generation_checkpoint = next_generation
            self.last_time_checkpoint = time.time()

        def save_final_checkpoint(self, config, population, species_set, state_generation: int) -> None:
            state_generation = int(state_generation)
            if state_generation <= self._start_generation:
                return

            if self.base_generation > 0:
                display_generation = self._display_from_next_generation(state_generation - 1)
            else:
                display_generation = self._display_from_next_generation(state_generation)

            if self._last_saved_display_generation == int(display_generation):
                return

            self._save_checkpoint_with_display(
                config=config,
                population=population,
                species_set=species_set,
                state_generation=state_generation,
                display_generation=display_generation,
            )


# =============================================================================
# Section 10. Entrypoint
# =============================================================================
def main() -> None:
    args = parse_args()
    run_started_wall = datetime.now(timezone.utc)
    run_started_perf = time.perf_counter()

    if neat is None:
        raise RuntimeError("neat-python is not installed. Install with: pip install neat-python")

    runtime = _load_runtime_config(args.runtime_config)
    applied_overrides = {}
    seed_genome_path = str(args.seed_genome or "").strip()
    if seed_genome_path:
        applied_overrides["seed_genome"] = seed_genome_path
    if args.resume and seed_genome_path:
        raise RuntimeError("--resume and --seed-genome are mutually exclusive")
    base_generation = max(0, int(args.base_generation))
    if base_generation != 0:
        applied_overrides["base_generation"] = int(base_generation)
    override_keys = []
    if args.generations > 0:
        runtime["generations"] = args.generations
        override_keys.append("generations")
    if args.workers > 0:
        runtime["eval_workers"] = args.workers
        override_keys.append("eval_workers")
    if args.games_per_genome > 0:
        runtime["games_per_genome"] = args.games_per_genome
        override_keys.append("games_per_genome")
    if args.eval_timeout_sec > 0:
        runtime["eval_timeout_sec"] = args.eval_timeout_sec
        override_keys.append("eval_timeout_sec")
    if args.max_eval_steps > 0:
        runtime["max_eval_steps"] = args.max_eval_steps
        override_keys.append("max_eval_steps")
    if args.opponent_policy.strip():
        runtime["opponent_policy"] = args.opponent_policy.strip()
        override_keys.append("opponent_policy")
    if args.opponent_genome.strip():
        runtime["opponent_genome"] = args.opponent_genome.strip()
        override_keys.append("opponent_genome")
    if args.checkpoint_every > 0:
        runtime["checkpoint_every"] = args.checkpoint_every
        override_keys.append("checkpoint_every")
    if str(args.seed).strip():
        runtime["seed"] = str(args.seed).strip()
        override_keys.append("seed")
    if args.switch_seats is not None:
        runtime["switch_seats"] = bool(args.switch_seats)
        override_keys.append("switch_seats")
    if args.fitness_gold_scale > 0:
        runtime["fitness_gold_scale"] = args.fitness_gold_scale
        override_keys.append("fitness_gold_scale")
    if args.fitness_gold_neutral_delta == args.fitness_gold_neutral_delta:
        runtime["fitness_gold_neutral_delta"] = args.fitness_gold_neutral_delta
        override_keys.append("fitness_gold_neutral_delta")
    if args.fitness_win_weight == args.fitness_win_weight:
        runtime["fitness_win_weight"] = args.fitness_win_weight
        override_keys.append("fitness_win_weight")
    if args.fitness_gold_weight == args.fitness_gold_weight:
        runtime["fitness_gold_weight"] = args.fitness_gold_weight
        override_keys.append("fitness_gold_weight")
    if args.fitness_win_neutral_rate == args.fitness_win_neutral_rate:
        runtime["fitness_win_neutral_rate"] = args.fitness_win_neutral_rate
        override_keys.append("fitness_win_neutral_rate")
    runtime = _normalize_runtime_values(runtime)
    for key in override_keys:
        applied_overrides[key] = runtime.get(key)

    opponent_policy = str(runtime.get("opponent_policy") or "").strip()
    opponent_policy_mix = runtime.get("opponent_policy_mix") or []
    has_policy = bool(opponent_policy)
    has_policy_mix = isinstance(opponent_policy_mix, list) and len(opponent_policy_mix) > 0
    if not has_policy and not has_policy_mix:
        raise RuntimeError("runtime must define opponent_policy or opponent_policy_mix")

    opponent_genome = str(runtime.get("opponent_genome") or "").strip()
    requires_genome_path = opponent_policy.lower() == "genome"
    if not has_policy:
        for item in opponent_policy_mix:
            if isinstance(item, dict) and str(item.get("policy") or "").strip().lower() == "genome":
                requires_genome_path = True
                break
    if requires_genome_path:
        if not opponent_genome:
            raise RuntimeError("opponent-policy=genome requires opponent_genome path")
        if not os.path.exists(opponent_genome):
            raise RuntimeError(f"opponent genome not found: {opponent_genome}")

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(args.output_dir, "checkpoints")
    models_dir = os.path.join(args.output_dir, "models")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    cfg = _build_config(args.config_feedforward)
    if len(cfg.genome_config.output_keys) != 1:
        raise RuntimeError("num_outputs must be 1 for candidate-scoring policy")

    _set_eval_env(runtime, args.output_dir)

    if args.resume:
        p = neat.Checkpointer.restore_checkpoint(args.resume)
    elif seed_genome_path:
        p = _build_seeded_population(cfg, seed_genome_path)
    else:
        p = neat.Population(cfg)
    start_generation = int(getattr(p, "generation", 0))

    # Keep training output quiet by default to reduce terminal I/O overhead.
    if bool(args.verbose):
        p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    prefix = os.path.join(checkpoints_dir, "neat-checkpoint-")
    checkpointer = OffsetCheckpointer(
        base_generation=int(base_generation),
        start_generation=int(start_generation),
        total_generations=int(runtime["generations"]),
        generation_interval=max(1, int(runtime["checkpoint_every"])),
        filename_prefix=prefix,
    )
    p.add_reporter(checkpointer)

    evaluator = None

    def _run_population(eval_callable):
        if bool(args.verbose):
            return p.run(eval_callable, int(runtime["generations"]))
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull):
                return p.run(eval_callable, int(runtime["generations"]))

    if args.dry_run:
        winner = _run_population(_run_dry_eval)
        checkpointer.save_final_checkpoint(p.config, p.population, p.species, p.generation)
        mode = "dry_run"
    else:
        evaluator = LoggedParallelEvaluator(
            num_workers=int(runtime["eval_workers"]),
            output_dir=args.output_dir,
            runtime=runtime,
        )
        try:
            winner = _run_population(evaluator.evaluate)
            checkpointer.save_final_checkpoint(p.config, p.population, p.species, p.generation)
            mode = "real_eval"
        finally:
            evaluator.close()

    winner_pkl = os.path.join(models_dir, "winner_genome.pkl")
    with open(winner_pkl, "wb") as f:
        pickle.dump(winner, f)

    winner_json_path = os.path.join(models_dir, "winner_genome.json")
    with open(winner_json_path, "w", encoding="utf-8") as f:
        json.dump(_export_neat_python_genome(winner, p.config), f, ensure_ascii=False, separators=(",", ":"))

    best_fitness = float(getattr(winner, "fitness", float("nan")) or 0.0)
    run_finished_wall = datetime.now(timezone.utc)
    run_elapsed_sec = max(0.0, time.perf_counter() - run_started_perf)
    run_summary = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "run_started_at": run_started_wall.isoformat(),
        "run_finished_at": run_finished_wall.isoformat(),
        "run_elapsed_sec": float(run_elapsed_sec),
        "mode": mode,
        "profile_name": str(args.profile_name or ""),
        "generations": int(runtime["generations"]),
        "workers": int(runtime["eval_workers"]),
        "games_per_genome": int(runtime["games_per_genome"]),
        "best_fitness": best_fitness,
        "applied_overrides": applied_overrides,
        "runtime_effective": runtime,
        "eval_failure_log": os.path.join(args.output_dir, "eval_failures.log"),
        "eval_metrics_log": os.path.join(args.output_dir, "eval_metrics.ndjson"),
        "generation_metrics_log": os.path.join(args.output_dir, "generation_metrics.ndjson"),
        "gate_state_path": os.path.join(args.output_dir, "gate_state.json"),
        "gate_state": evaluator.snapshot() if evaluator is not None else {},
        "winner_pickle": winner_pkl,
        "winner_json": winner_json_path,
        "seed_genome": seed_genome_path or None,
        "config_feedforward": os.path.abspath(args.config_feedforward),
        "runtime_config": os.path.abspath(args.runtime_config),
    }
    run_summary_path = os.path.join(args.output_dir, "run_summary.json")
    with open(run_summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
