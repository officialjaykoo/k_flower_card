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
import functools
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import random
import re
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


def _parse_early_stop_win_rate_cutoffs(raw_value: object) -> list:
    if raw_value is None:
        return []

    source = []
    if isinstance(raw_value, list):
        source = raw_value
    else:
        text = str(raw_value).strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception as exc:
            raise RuntimeError(f"early_stop_win_rate_cutoffs must be a JSON array: {exc}") from exc
        if not isinstance(parsed, list):
            raise RuntimeError("early_stop_win_rate_cutoffs must be a JSON array")
        source = parsed

    out = []
    seen_games = set()
    for idx, item in enumerate(source):
        if not isinstance(item, dict):
            raise RuntimeError("early_stop_win_rate_cutoffs items must be objects")
        try:
            games = int(item.get("games"))
        except Exception as exc:
            raise RuntimeError(f"early_stop_win_rate_cutoffs[{idx}].games must be integer") from exc
        try:
            max_win_rate = float(item.get("max_win_rate"))
        except Exception as exc:
            raise RuntimeError(f"early_stop_win_rate_cutoffs[{idx}].max_win_rate must be number") from exc
        if games < 1:
            raise RuntimeError(f"early_stop_win_rate_cutoffs[{idx}].games must be >= 1")
        if (not math.isfinite(max_win_rate)) or max_win_rate < 0.0 or max_win_rate > 1.0:
            raise RuntimeError(
                f"early_stop_win_rate_cutoffs[{idx}].max_win_rate must be finite and in [0,1]"
            )
        if games in seen_games:
            raise RuntimeError(f"duplicate early_stop_win_rate_cutoffs games value: {games}")
        seen_games.add(games)
        out.append({"games": int(games), "max_win_rate": float(max_win_rate)})

    out.sort(key=lambda item: int(item["games"]))
    return out


def _parse_early_stop_go_take_rate_cutoffs(raw_value: object) -> list:
    if raw_value is None:
        return []

    source = []
    if isinstance(raw_value, list):
        source = raw_value
    else:
        text = str(raw_value).strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception as exc:
            raise RuntimeError(f"early_stop_go_take_rate_cutoffs must be a JSON array: {exc}") from exc
        if not isinstance(parsed, list):
            raise RuntimeError("early_stop_go_take_rate_cutoffs must be a JSON array")
        source = parsed

    out = []
    seen_games = set()
    for idx, item in enumerate(source):
        if not isinstance(item, dict):
            raise RuntimeError("early_stop_go_take_rate_cutoffs items must be objects")
        try:
            games = int(item.get("games"))
        except Exception as exc:
            raise RuntimeError(f"early_stop_go_take_rate_cutoffs[{idx}].games must be integer") from exc
        try:
            min_go_opportunity_count = int(item.get("min_go_opportunity_count"))
        except Exception as exc:
            raise RuntimeError(
                f"early_stop_go_take_rate_cutoffs[{idx}].min_go_opportunity_count must be integer"
            ) from exc

        min_go_take_rate = _safe_optional_float(item.get("min_go_take_rate"))
        max_go_take_rate = _safe_optional_float(item.get("max_go_take_rate"))

        if games < 1:
            raise RuntimeError(f"early_stop_go_take_rate_cutoffs[{idx}].games must be >= 1")
        if min_go_opportunity_count < 0:
            raise RuntimeError(
                f"early_stop_go_take_rate_cutoffs[{idx}].min_go_opportunity_count must be >= 0"
            )
        if min_go_take_rate is None and max_go_take_rate is None:
            raise RuntimeError(
                f"early_stop_go_take_rate_cutoffs[{idx}] must set at least one of min_go_take_rate/max_go_take_rate"
            )
        for field_name, value in (
            ("min_go_take_rate", min_go_take_rate),
            ("max_go_take_rate", max_go_take_rate),
        ):
            if value is None:
                continue
            if (not math.isfinite(value)) or value < 0.0 or value > 1.0:
                raise RuntimeError(
                    f"early_stop_go_take_rate_cutoffs[{idx}].{field_name} must be finite and in [0,1] or null"
                )
        if (
            min_go_take_rate is not None
            and max_go_take_rate is not None
            and min_go_take_rate > max_go_take_rate
        ):
            raise RuntimeError(
                f"early_stop_go_take_rate_cutoffs[{idx}].min_go_take_rate must be <= max_go_take_rate"
            )
        if games in seen_games:
            raise RuntimeError(f"duplicate early_stop_go_take_rate_cutoffs games value: {games}")
        seen_games.add(games)
        out.append(
            {
                "games": int(games),
                "min_go_opportunity_count": int(min_go_opportunity_count),
                "min_go_take_rate": min_go_take_rate,
                "max_go_take_rate": max_go_take_rate,
            }
        )

    out.sort(key=lambda item: int(item["games"]))
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


def _normalize_control_policy_mode(raw_value: object) -> str:
    raw = str(raw_value or "").strip().lower()
    if raw in ("", "pure_model"):
        return "pure_model"
    raise RuntimeError("runtime key 'control_policy_mode' must be one of: pure_model")


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
    cfg["feature_profile"] = str(cfg.get("feature_profile") or "material10").strip().lower()
    if cfg["feature_profile"] not in ("hand7", "hand10", "material10"):
        raise RuntimeError("runtime key 'feature_profile' must be one of: hand7, hand10, material10")
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
    cfg["early_stop_win_rate_cutoffs"] = _parse_early_stop_win_rate_cutoffs(
        cfg.get("early_stop_win_rate_cutoffs")
    )
    cfg["early_stop_go_take_rate_cutoffs"] = _parse_early_stop_go_take_rate_cutoffs(
        cfg.get("early_stop_go_take_rate_cutoffs")
    )
    cfg["control_policy_mode"] = _normalize_control_policy_mode(cfg.get("control_policy_mode"))
    cfg["winner_playoff_topk"] = max(1, _to_int(cfg.get("winner_playoff_topk"), 5))
    cfg["winner_playoff_finalists"] = max(1, _to_int(cfg.get("winner_playoff_finalists"), 2))
    cfg["winner_playoff_stage1_games"] = max(
        1, _to_int(cfg.get("winner_playoff_stage1_games"), cfg["games_per_genome"])
    )
    cfg["winner_playoff_stage2_games"] = max(
        1, _to_int(cfg.get("winner_playoff_stage2_games"), cfg["games_per_genome"])
    )
    cfg["winner_playoff_win_rate_tie_threshold"] = max(
        0.0, _to_float(cfg.get("winner_playoff_win_rate_tie_threshold"), 0.01)
    )
    cfg["winner_playoff_mean_gold_delta_tie_threshold"] = max(
        0.0, _to_float(cfg.get("winner_playoff_mean_gold_delta_tie_threshold"), 100.0)
    )
    cfg["winner_playoff_go_opp_min_count"] = max(
        0, _to_int(cfg.get("winner_playoff_go_opp_min_count"), 100)
    )
    cfg["winner_playoff_go_take_rate_tie_threshold"] = max(
        0.0, _to_float(cfg.get("winner_playoff_go_take_rate_tie_threshold"), 0.02)
    )
    if cfg["winner_playoff_finalists"] > cfg["winner_playoff_topk"]:
        cfg["winner_playoff_finalists"] = int(cfg["winner_playoff_topk"])

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
    os.environ[f"{ENV_PREFIX}FEATURE_PROFILE"] = str(runtime.get("feature_profile") or "material10")
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
    os.environ[f"{ENV_PREFIX}EARLY_STOP_WIN_RATE_CUTOFFS"] = json.dumps(
        runtime.get("early_stop_win_rate_cutoffs") or [],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    os.environ[f"{ENV_PREFIX}EARLY_STOP_GO_TAKE_RATE_CUTOFFS"] = json.dumps(
        runtime.get("early_stop_go_take_rate_cutoffs") or [],
        ensure_ascii=False,
        separators=(",", ":"),
    )


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
        "feature_profile": os.environ.get(f"{ENV_PREFIX}FEATURE_PROFILE"),
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
        "early_stop_win_rate_cutoffs": os.environ.get(f"{ENV_PREFIX}EARLY_STOP_WIN_RATE_CUTOFFS"),
        "early_stop_go_take_rate_cutoffs": os.environ.get(f"{ENV_PREFIX}EARLY_STOP_GO_TAKE_RATE_CUTOFFS"),
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


def _resolve_eval_output_dir(runtime: dict) -> str:
    runtime_dir = str(runtime.get("output_dir") or "").strip()
    if runtime_dir:
        return os.path.abspath(runtime_dir)
    env_dir = str((_runtime_from_env_cached().get("output_dir") if _runtime_from_env_cached() else "") or "").strip()
    if env_dir:
        return os.path.abspath(env_dir)
    return os.path.abspath(os.getcwd())


def _export_neat_python_genome(genome, config, runtime: Optional[dict] = None) -> dict:
    gcfg = config.genome_config
    input_keys = [int(x) for x in gcfg.input_keys]
    output_keys = [int(x) for x in gcfg.output_keys]
    runtime_ctx = runtime or (_runtime_from_env_cached() if os.environ.get(f"{ENV_PREFIX}FORMAT_VERSION") else {})
    feature_profile = str((runtime_ctx or {}).get("feature_profile") or "material10").strip().lower()

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
        "feature_spec": {
            "profile": feature_profile,
            "base_features": len(input_keys),
        },
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


def _clamp_unit(value):
    x = _safe_float(value, 0.0)
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x


def _normalize_unit_range(value, min_value, max_value):
    x = _safe_float(value, 0.0)
    lo = _safe_float(min_value, 0.0)
    hi = _safe_float(max_value, 0.0)
    span = hi - lo
    if not math.isfinite(span) or abs(span) <= 1e-12:
        return 0.5
    return _clamp_unit((x - lo) / span)


def _selection_fitness_value(record: Optional[dict]) -> float:
    if not isinstance(record, dict):
        return -1e9
    return _safe_float(record.get("fitness"), -1e9)


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


def _training_candidate_sort_key(record: dict):
    return (
        _safe_float((record or {}).get("fitness"), -1e9),
        _safe_float((record or {}).get("win_rate"), -1.0),
        _safe_float((record or {}).get("mean_gold_delta"), -1e18),
    )


def _playoff_record_sort_key(record: dict):
    return (
        _safe_float((record or {}).get("win_rate"), -1.0),
        _safe_float((record or {}).get("mean_gold_delta"), -1e18),
        _safe_float((record or {}).get("fitness"), -1e9),
    )


def _compare_desc(a: float, b: float) -> int:
    if a > b:
        return 1
    if a < b:
        return -1
    return 0


def _compare_asc(a: float, b: float) -> int:
    if a < b:
        return 1
    if a > b:
        return -1
    return 0


def _playoff_record_compare(record_a: dict, record_b: dict, runtime: dict) -> int:
    record_a = dict(record_a or {})
    record_b = dict(record_b or {})
    win_tie_threshold = max(
        0.0, _safe_float(runtime.get("winner_playoff_win_rate_tie_threshold"), 0.01)
    )
    gold_tie_threshold = max(
        0.0, _safe_float(runtime.get("winner_playoff_mean_gold_delta_tie_threshold"), 100.0)
    )
    go_opp_min_count = max(
        0, int(_safe_float(runtime.get("winner_playoff_go_opp_min_count"), 100.0))
    )
    go_take_tie_threshold = max(
        0.0, _safe_float(runtime.get("winner_playoff_go_take_rate_tie_threshold"), 0.02)
    )

    win_a = _safe_float(record_a.get("win_rate"), -1.0)
    win_b = _safe_float(record_b.get("win_rate"), -1.0)
    if abs(win_a - win_b) > win_tie_threshold:
        return _compare_desc(win_a, win_b)

    gold_a = _safe_float(record_a.get("mean_gold_delta"), -1e18)
    gold_b = _safe_float(record_b.get("mean_gold_delta"), -1e18)
    if abs(gold_a - gold_b) > gold_tie_threshold:
        return _compare_desc(gold_a, gold_b)

    go_opp_a = max(0, int(_safe_float(record_a.get("go_opportunity_count"), 0.0)))
    go_opp_b = max(0, int(_safe_float(record_b.get("go_opportunity_count"), 0.0)))
    if go_opp_a >= go_opp_min_count and go_opp_b >= go_opp_min_count:
        go_take_a = _safe_float(record_a.get("go_take_rate"), 0.0)
        go_take_b = _safe_float(record_b.get("go_take_rate"), 0.0)
        if abs(go_take_a - go_take_b) > go_take_tie_threshold:
            return _compare_desc(go_take_a, go_take_b)

        go_count_a = max(0, int(_safe_float(record_a.get("go_count"), 0.0)))
        go_count_b = max(0, int(_safe_float(record_b.get("go_count"), 0.0)))
        if go_count_a > 0 and go_count_b > 0:
            go_fail_a = _safe_float(record_a.get("go_fail_rate"), 1.0)
            go_fail_b = _safe_float(record_b.get("go_fail_rate"), 1.0)
            if abs(go_fail_a - go_fail_b) > 1e-12:
                return _compare_asc(go_fail_a, go_fail_b)

    fit_a = _safe_float(record_a.get("fitness"), -1e9)
    fit_b = _safe_float(record_b.get("fitness"), -1e9)
    if abs(fit_a - fit_b) > 1e-12:
        return _compare_desc(fit_a, fit_b)

    key_a = int(_safe_float(record_a.get("genome_key"), -1))
    key_b = int(_safe_float(record_b.get("genome_key"), -1))
    return _compare_asc(key_a, key_b)


# =============================================================================
# Section 7. Single Genome Evaluation Worker
# =============================================================================
def _run_eval_worker_for_genome(
    genome,
    config,
    runtime: dict,
    seed_text: str,
    generation: int = -1,
    genome_key: int = -1,
    games_override: Optional[int] = None,
    early_stop_win_rate_cutoffs_override: Optional[list] = None,
    early_stop_go_take_rate_cutoffs_override: Optional[list] = None,
    context_label: str = "train_eval",
):
    eval_script = str(runtime["eval_script"] or "")
    seed_text = str(seed_text or runtime["seed"])
    output_dir = _resolve_eval_output_dir(runtime)
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
        "context": str(context_label or "train_eval"),
    }
    if not eval_script or not os.path.exists(eval_script):
        _append_eval_failure_log(
            output_dir,
            dict(failure_meta, reason="eval_script_missing", eval_script=eval_script),
        )
        return {"fitness": -1e9, "seed_used": seed_text, "eval_ok": False}

    payload = _export_neat_python_genome(genome, config, runtime)

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

        games_value = int(games_override) if games_override is not None else int(runtime["games_per_genome"])
        early_stop_win_rate_cutoffs = (
            runtime.get("early_stop_win_rate_cutoffs") or []
            if early_stop_win_rate_cutoffs_override is None
            else early_stop_win_rate_cutoffs_override
        )
        early_stop_go_take_rate_cutoffs = (
            runtime.get("early_stop_go_take_rate_cutoffs") or []
            if early_stop_go_take_rate_cutoffs_override is None
            else early_stop_go_take_rate_cutoffs_override
        )

        cmd = [
            "node",
            eval_script,
            "--genome",
            genome_path,
            "--games",
            str(int(games_value)),
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
            "--early-stop-win-rate-cutoffs",
            json.dumps(
                early_stop_win_rate_cutoffs,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "--early-stop-go-take-rate-cutoffs",
            json.dumps(
                early_stop_go_take_rate_cutoffs,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
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


def eval_function(genome, config, seed_override="", generation=-1, genome_key=-1):
    runtime = _runtime_from_env_cached()
    seed_text = str(seed_override or runtime["seed"])
    return _run_eval_worker_for_genome(
        genome=genome,
        config=config,
        runtime=runtime,
        seed_text=seed_text,
        generation=int(generation),
        genome_key=int(genome_key),
        context_label="train_eval",
    )


# =============================================================================
# Section 8. Parallel Evaluator + Gate Tracking
# =============================================================================
class LoggedParallelEvaluator:
    def __init__(
        self,
        num_workers: int,
        output_dir: str,
        runtime: dict,
        start_generation: int = 0,
        generation_display_offset: int = 0,
    ):
        self.num_workers = max(1, int(num_workers))
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.runtime_seed = str(runtime["seed"])
        self.pool = mp.Pool(processes=self.num_workers)

        self.eval_metrics_log = os.path.join(self.output_dir, "eval_metrics.ndjson")
        self.generation_metrics_log = os.path.join(self.output_dir, "generation_metrics.ndjson")
        self.gate_state_path = os.path.join(self.output_dir, "gate_state.json")

        self.generation = int(start_generation) - 1
        self.generation_display_offset = int(generation_display_offset)
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
        self.full_eval_games = int(runtime["games_per_genome"])
        self.ema_imitation = None
        self.ema_win_rate = None
        self.gate_streak = 0
        self.transition_generation = None
        self.failure_generation = None
        self.best_imitation_history = []
        self.best_win_rate_history = []
        self.gate_state = {}
        self.best_record_overall = None
        self.best_genome_overall = None
        self.playoff_topk = max(1, int(runtime.get("winner_playoff_topk", 5)))
        self.top_candidate_entries = []
        self.champion_records = {
            "fitness": None,
        }
        self.champion_genomes = {
            "fitness": None,
        }

    def _display_generation(self, generation: Optional[int] = None) -> int:
        current = self.generation if generation is None else int(generation)
        return int(self.generation_display_offset + current)

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

    def _champion_is_eligible(self, champion_name: str, record: dict) -> bool:
        if not self._is_valid_gate_record(record):
            return False
        if not self._is_full_eval_record(record):
            return False
        return True

    def _champion_sort_key(self, champion_name: str, record: dict):
        if not isinstance(record, dict):
            if champion_name == "fitness":
                return (-1e9,)
            return (-1e9,)
        tie_fitness = _selection_fitness_value(record)
        if champion_name == "fitness":
            return (tie_fitness,)
        return (tie_fitness,)

    def _update_champion_snapshot(self, champion_name: str, record: dict, genome) -> None:
        current_record = self.champion_records.get(champion_name)
        candidate_key = self._champion_sort_key(champion_name, record)
        current_key = self._champion_sort_key(champion_name, current_record)
        if current_record is not None and not (candidate_key > current_key):
            return
        self.champion_records[champion_name] = dict(record)
        self.champion_genomes[champion_name] = copy.deepcopy(genome)

    def _update_champion_snapshots(self, full_eval_records, genomes) -> None:
        if len(full_eval_records) <= 0:
            return
        genome_map = {int(genome_key): genome for genome_key, genome in genomes}
        for champion_name in ("fitness",):
            eligible = [r for r in full_eval_records if self._champion_is_eligible(champion_name, r)]
            if len(eligible) <= 0:
                continue
            candidate = max(eligible, key=lambda r: self._champion_sort_key(champion_name, r))
            genome_key = int(candidate.get("genome_key", -1))
            genome = genome_map.get(genome_key)
            if genome is None:
                continue
            self._update_champion_snapshot(champion_name, candidate, genome)

    def _update_top_candidate_snapshots(self, full_eval_records, genomes) -> None:
        if self.playoff_topk <= 0 or len(full_eval_records) <= 0:
            return
        genome_map = {int(genome_key): genome for genome_key, genome in genomes}
        merged = list(self.top_candidate_entries)
        existing_by_key = {
            int((entry.get("record") or {}).get("genome_key", -1)): idx
            for idx, entry in enumerate(merged)
            if isinstance(entry, dict)
        }
        for record in full_eval_records:
            genome_key = int(record.get("genome_key", -1))
            genome = genome_map.get(genome_key)
            if genome is None:
                continue
            entry = {
                "record": dict(record),
                "genome": copy.deepcopy(genome),
            }
            if genome_key in existing_by_key:
                merged[existing_by_key[genome_key]] = entry
            else:
                merged.append(entry)
        merged.sort(key=lambda entry: _training_candidate_sort_key(entry.get("record") or {}), reverse=True)
        self.top_candidate_entries = merged[: self.playoff_topk]

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

    def _is_full_eval_record(self, record: dict) -> bool:
        if not bool(record.get("eval_ok")):
            return False
        games = _safe_float(record.get("games"), 0.0)
        return int(games) >= int(self.full_eval_games)

    def _update_gate(self, best_record: Optional[dict], valid_count: int, total_count: int):
        if best_record is None or valid_count <= 0:
            # invalid_generation: EMA is not updated and streak is forcibly reset
            self.gate_streak = 0
            self.gate_state = {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "generation": self._display_generation(),
                "data_quality": "invalid_generation",
                "valid_record_count": int(valid_count),
                "total_record_count": int(total_count),
                "ema_window": int(self.ema_window),
                "ema_alpha": float(self.ema_alpha),
                "ema_imitation": float(self.ema_imitation) if self.ema_imitation is not None else None,
                "ema_win_rate": float(self.ema_win_rate) if self.ema_win_rate is not None else None,
                "gate_streak": int(self.gate_streak),
                "transition_ready": bool(self.transition_generation is not None),
                "transition_generation": (
                    self._display_generation(self.transition_generation)
                    if self.transition_generation is not None
                    else None
                ),
                "failure_triggered": bool(self.failure_generation is not None),
                "failure_generation": (
                    self._display_generation(self.failure_generation)
                    if self.failure_generation is not None
                    else None
                ),
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
            "generation": self._display_generation(),
            "data_quality": "valid_generation",
            "valid_record_count": int(valid_count),
            "total_record_count": int(total_count),
            "ema_window": int(self.ema_window),
            "ema_alpha": float(self.ema_alpha),
            "ema_imitation": float(self.ema_imitation),
            "ema_win_rate": float(self.ema_win_rate),
            "gate_streak": int(self.gate_streak),
            "transition_ready": bool(self.transition_generation is not None),
            "transition_generation": (
                self._display_generation(self.transition_generation)
                if self.transition_generation is not None
                else None
            ),
            "failure_triggered": bool(self.failure_generation is not None),
            "failure_generation": (
                self._display_generation(self.failure_generation)
                if self.failure_generation is not None
                else None
            ),
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
        display_generation = self._display_generation()
        seed_for_generation = f"{self.runtime_seed}|gen={display_generation}"
        jobs = []
        for genome_key, genome in genomes:
            job = self.pool.apply_async(
                eval_function,
                (genome, config, seed_for_generation, int(display_generation), int(genome_key)),
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
            record["generation"] = int(display_generation)
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
            setattr(
                genome,
                "_codex_eval_meta",
                {
                    "generation": int(display_generation),
                    "games": int(_safe_float(record.get("games"), 0.0)),
                    "win_rate": _safe_optional_float(record.get("win_rate")),
                    "mean_gold_delta": _safe_optional_float(record.get("mean_gold_delta")),
                    "go_take_rate": _safe_optional_float(record.get("go_take_rate")),
                    "go_fail_rate": _safe_optional_float(record.get("go_fail_rate")),
                    "full_eval_passed": (
                        int(_safe_float(record.get("games"), 0.0)) >= int(self.full_eval_games)
                    ),
                },
            )
            records.append(record)

        self._append_lines(self.eval_metrics_log, records)

        valid_records = [r for r in records if self._is_valid_gate_record(r)]
        full_eval_records = [r for r in valid_records if self._is_full_eval_record(r)]

        if records:
            selection_best_record = (
                max(full_eval_records, key=lambda r: _safe_float(r.get("fitness"), -1e9))
                if len(full_eval_records) > 0
                else None
            )
            best_record = selection_best_record
            self._update_champion_snapshots(full_eval_records, genomes)
            self._update_top_candidate_snapshots(full_eval_records, genomes)
            if self.champion_records.get("fitness") is not None:
                self.best_record_overall = dict(self.champion_records["fitness"])
                self.best_genome_overall = copy.deepcopy(self.champion_genomes["fitness"])
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
                "generation": int(display_generation),
                "seed_used": seed_for_generation,
                "population_size": len(records),
                "valid_record_count": len(valid_records),
                "full_eval_record_count": len(full_eval_records),
                "early_stop_record_count": max(0, len(valid_records) - len(full_eval_records)),
                "invalid_record_count": max(0, len(records) - len(valid_records)),
                "data_quality": "valid_generation" if len(valid_records) > 0 else "invalid_generation",
                "best_genome_key": int(best_record.get("genome_key", -1)) if best_record is not None else -1,
                "best_fitness": (
                    _safe_float(best_record.get("fitness"), -1e9)
                    if best_record is not None
                    else -1e9
                ),
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
                    if best_record is not None and self._is_valid_gate_record(best_record)
                    else None
                ),
                "best_imitation_weighted_score": (
                    _safe_optional_float(best_record.get("imitation_weighted_score"))
                    if (
                        best_record is not None
                        and self._is_valid_gate_record(best_record)
                        and "imitation_weighted_score" in best_record
                    )
                    else None
                ),
                "best_genome_nodes": int(best_record.get("num_nodes", 0)) if best_record is not None else 0,
                "best_genome_connections": (
                    int(best_record.get("num_connections", 0)) if best_record is not None else 0
                ),
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
                "generation": int(display_generation),
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

        best_gate_record = max(valid_records, key=lambda r: _safe_float(r.get("fitness"), -1e9)) if len(valid_records) > 0 else None
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

    def best_record_snapshot(self):
        if self.best_record_overall is None:
            return None
        return dict(self.best_record_overall)

    def best_genome_snapshot(self):
        if self.best_genome_overall is None:
            return None
        return copy.deepcopy(self.best_genome_overall)

    def champion_snapshots(self):
        snapshots = {}
        for name in ("fitness",):
            record = self.champion_records.get(name)
            genome = self.champion_genomes.get(name)
            snapshots[name] = {
                "record": (dict(record) if isinstance(record, dict) else None),
                "genome": (copy.deepcopy(genome) if genome is not None else None),
            }
        return snapshots

    def top_candidate_snapshots(self, limit: Optional[int] = None):
        effective_limit = max(0, int(limit)) if limit is not None else len(self.top_candidate_entries)
        out = []
        for entry in self.top_candidate_entries[:effective_limit]:
            out.append(
                {
                    "record": dict(entry.get("record") or {}),
                    "genome": copy.deepcopy(entry.get("genome")),
                }
            )
        return out


def _serialize_playoff_entry(training_record: dict, playoff_record: dict) -> dict:
    training_record = dict(training_record or {})
    playoff_record = dict(playoff_record or {})
    return {
        "generation": training_record.get("generation"),
        "genome_key": training_record.get("genome_key"),
        "training_fitness": training_record.get("fitness"),
        "training_win_rate": training_record.get("win_rate"),
        "training_mean_gold_delta": training_record.get("mean_gold_delta"),
        "playoff_fitness": playoff_record.get("fitness"),
        "playoff_win_rate": playoff_record.get("win_rate"),
        "playoff_mean_gold_delta": playoff_record.get("mean_gold_delta"),
        "playoff_go_take_rate": playoff_record.get("go_take_rate"),
        "playoff_go_fail_rate": playoff_record.get("go_fail_rate"),
        "playoff_seed_used": playoff_record.get("seed_used"),
        "eval_ok": playoff_record.get("eval_ok"),
    }


def _build_pooled_best_record(records: list[dict], runtime: dict) -> Optional[dict]:
    source_records = [dict(item or {}) for item in records if isinstance(item, dict)]
    source_records = [item for item in source_records if _safe_float(item.get("games"), 0.0) > 0.0]
    if not source_records:
        return None

    total_games = sum(max(0, int(_safe_float(item.get("games"), 0.0))) for item in source_records)
    if total_games <= 0:
        return None

    wins = sum(max(0, int(_safe_float(item.get("wins"), 0.0))) for item in source_records)
    losses = sum(max(0, int(_safe_float(item.get("losses"), 0.0))) for item in source_records)
    draws = sum(max(0, int(_safe_float(item.get("draws"), 0.0))) for item in source_records)
    requested_games = sum(max(0, int(_safe_float(item.get("requested_games"), item.get("games")))) for item in source_records)

    def _seat_block(record: dict, side: str) -> dict:
        return dict(((record.get("seat_breakdown") or {}).get(side) or {}))

    def _sum_seat_count(side: str, key: str) -> int:
        return sum(max(0, int(_safe_float(_seat_block(item, side).get(key), 0.0))) for item in source_records)

    def _seat_mean_gold_delta(side: str) -> float:
        games = _sum_seat_count(side, "games")
        if games <= 0:
            return 0.0
        total = 0.0
        for item in source_records:
            seat = _seat_block(item, side)
            seat_games = max(0, int(_safe_float(seat.get("games"), 0.0)))
            seat_mean = _safe_float(seat.get("mean_gold_delta"), 0.0)
            total += float(seat_games) * seat_mean
        return total / float(games)

    first_games = _sum_seat_count("first", "games")
    second_games = _sum_seat_count("second", "games")
    first_wins = _sum_seat_count("first", "wins")
    first_losses = _sum_seat_count("first", "losses")
    first_draws = _sum_seat_count("first", "draws")
    second_wins = _sum_seat_count("second", "wins")
    second_losses = _sum_seat_count("second", "losses")
    second_draws = _sum_seat_count("second", "draws")
    first_mean_gold_delta = _seat_mean_gold_delta("first")
    second_mean_gold_delta = _seat_mean_gold_delta("second")

    def _rate(numerator: float, denominator: float) -> float:
        return (float(numerator) / float(denominator)) if float(denominator) > 0.0 else 0.0

    win_rate = _rate(wins, total_games)
    loss_rate = _rate(losses, total_games)
    draw_rate = _rate(draws, total_games)
    mean_gold_delta = sum(
        float(max(0, int(_safe_float(item.get("games"), 0.0)))) * _safe_float(item.get("mean_gold_delta"), 0.0)
        for item in source_records
    ) / float(total_games)

    first_win_rate = _rate(first_wins, first_games)
    first_loss_rate = _rate(first_losses, first_games)
    first_draw_rate = _rate(first_draws, first_games)
    second_win_rate = _rate(second_wins, second_games)
    second_loss_rate = _rate(second_losses, second_games)
    second_draw_rate = _rate(second_draws, second_games)

    weighted_win_rate = (0.48 * first_win_rate) + (0.52 * second_win_rate)
    weighted_loss_rate = (0.48 * first_loss_rate) + (0.52 * second_loss_rate)
    weighted_draw_rate = (0.48 * first_draw_rate) + (0.52 * second_draw_rate)
    weighted_mean_gold_delta = (0.48 * first_mean_gold_delta) + (0.52 * second_mean_gold_delta)

    fitness_gold_scale = max(1e-9, _safe_float(runtime.get("fitness_gold_scale"), 1500.0))
    fitness_gold_neutral_delta = _safe_float(runtime.get("fitness_gold_neutral_delta"), 0.0)
    fitness_win_neutral_rate = _safe_float(runtime.get("fitness_win_neutral_rate"), 0.5)
    fitness_win_weight_raw = _safe_float(runtime.get("fitness_win_weight"), 0.9)
    fitness_gold_weight_raw = _safe_float(runtime.get("fitness_gold_weight"), 0.1)
    fitness_weight_sum = max(1e-9, fitness_win_weight_raw + fitness_gold_weight_raw)
    fitness_win_weight = fitness_win_weight_raw / fitness_weight_sum
    fitness_gold_weight = fitness_gold_weight_raw / fitness_weight_sum

    gold_norm = math.tanh((weighted_mean_gold_delta - fitness_gold_neutral_delta) / fitness_gold_scale)
    expected_result_raw = max(0.0, min(1.0, weighted_win_rate)) + (0.5 * max(0.0, min(1.0, weighted_draw_rate))) - max(0.0, min(1.0, weighted_loss_rate))
    expected_result = max(-1.0, min(1.0, expected_result_raw))
    neutral_expected_result = (2.0 * fitness_win_neutral_rate) - 1.0
    if expected_result >= neutral_expected_result:
        result_upper_span = max(1e-9, 1.0 - neutral_expected_result)
        result_norm = max(0.0, min(1.0, (expected_result - neutral_expected_result) / result_upper_span))
    else:
        result_lower_span = max(1e-9, neutral_expected_result + 1.0)
        result_norm = -max(0.0, min(1.0, (neutral_expected_result - expected_result) / result_lower_span))

    fitness = (fitness_gold_weight * gold_norm) + (fitness_win_weight * result_norm)

    go_opportunity_count = sum(max(0, int(_safe_float(item.get("go_opportunity_count"), 0.0))) for item in source_records)
    go_opportunity_games = sum(max(0, int(_safe_float(item.get("go_opportunity_games"), 0.0))) for item in source_records)
    go_count = sum(max(0, int(_safe_float(item.get("go_count"), 0.0))) for item in source_records)
    go_fail_count = sum(max(0, int(_safe_float(item.get("go_fail_count"), 0.0))) for item in source_records)
    go_games = sum(max(0, int(_safe_float(item.get("go_games"), 0.0))) for item in source_records)

    my_bankrupt_count = sum(
        max(0, int(_safe_float(((item.get("bankrupt") or {}).get("my_bankrupt_count")), 0.0)))
        for item in source_records
    )
    inflicted_bankrupt_count = sum(
        max(0, int(_safe_float(((item.get("bankrupt") or {}).get("my_inflicted_bankrupt_count")), 0.0)))
        for item in source_records
    )

    imitation_play_total = sum(max(0, int(_safe_float(item.get("imitation_play_total"), 0.0))) for item in source_records)
    imitation_play_matches = sum(max(0, int(_safe_float(item.get("imitation_play_matches"), 0.0))) for item in source_records)
    imitation_match_total = sum(max(0, int(_safe_float(item.get("imitation_match_total"), 0.0))) for item in source_records)
    imitation_match_matches = sum(max(0, int(_safe_float(item.get("imitation_match_matches"), 0.0))) for item in source_records)
    imitation_option_total = sum(max(0, int(_safe_float(item.get("imitation_option_total"), 0.0))) for item in source_records)
    imitation_option_matches = sum(max(0, int(_safe_float(item.get("imitation_option_matches"), 0.0))) for item in source_records)
    imitation_weight_play = _safe_float(source_records[0].get("imitation_weight_play"), 0.5)
    imitation_weight_match = _safe_float(source_records[0].get("imitation_weight_match"), 0.3)
    imitation_weight_option = _safe_float(source_records[0].get("imitation_weight_option"), 0.2)
    imitation_play_ratio = _rate(imitation_play_matches, imitation_play_total)
    imitation_match_ratio = _rate(imitation_match_matches, imitation_match_total)
    imitation_option_ratio = _rate(imitation_option_matches, imitation_option_total)
    imitation_weight_sum = imitation_weight_play + imitation_weight_match + imitation_weight_option
    if imitation_weight_sum > 0.0:
        imitation_weighted_score = (
            (imitation_weight_play * imitation_play_ratio)
            + (imitation_weight_match * imitation_match_ratio)
            + (imitation_weight_option * imitation_option_ratio)
        ) / imitation_weight_sum
    else:
        imitation_weighted_score = 0.0

    pooled_record = {
        "generation": source_records[0].get("generation"),
        "genome_key": source_records[0].get("genome_key"),
        "games": int(total_games),
        "requested_games": int(requested_games),
        "wins": int(wins),
        "losses": int(losses),
        "draws": int(draws),
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "draw_rate": draw_rate,
        "mean_gold_delta": mean_gold_delta,
        "go_opportunity_count": int(go_opportunity_count),
        "go_opportunity_games": int(go_opportunity_games),
        "go_opportunity_rate": _rate(go_opportunity_games, total_games),
        "go_count": int(go_count),
        "go_fail_count": int(go_fail_count),
        "go_fail_rate": _rate(go_fail_count, go_games),
        "go_games": int(go_games),
        "go_rate": _rate(go_count, total_games),
        "go_take_rate": _rate(go_count, go_opportunity_count),
        "bankrupt": {
            "my_bankrupt_count": int(my_bankrupt_count),
            "my_inflicted_bankrupt_count": int(inflicted_bankrupt_count),
        },
        "my_bankrupt_rate": _rate(my_bankrupt_count, total_games),
        "inflicted_bankrupt_rate": _rate(inflicted_bankrupt_count, total_games),
        "seat_breakdown": {
            "first": {
                "games": int(first_games),
                "wins": int(first_wins),
                "losses": int(first_losses),
                "draws": int(first_draws),
                "win_rate": first_win_rate,
                "loss_rate": first_loss_rate,
                "draw_rate": first_draw_rate,
                "mean_gold_delta": first_mean_gold_delta,
            },
            "second": {
                "games": int(second_games),
                "wins": int(second_wins),
                "losses": int(second_losses),
                "draws": int(second_draws),
                "win_rate": second_win_rate,
                "loss_rate": second_loss_rate,
                "draw_rate": second_draw_rate,
                "mean_gold_delta": second_mean_gold_delta,
            },
            "weighted": {
                "win_rate": weighted_win_rate,
                "loss_rate": weighted_loss_rate,
                "draw_rate": weighted_draw_rate,
                "mean_gold_delta": weighted_mean_gold_delta,
                "win_weights": {"first": 0.48, "second": 0.52},
                "gold_weights": {"first": 0.48, "second": 0.52},
            },
        },
        "fitness_gold_scale": fitness_gold_scale,
        "fitness_gold_neutral_delta": fitness_gold_neutral_delta,
        "fitness_win_neutral_rate": fitness_win_neutral_rate,
        "fitness_win_weight": fitness_win_weight_raw,
        "fitness_gold_weight": fitness_gold_weight_raw,
        "imitation_play_total": int(imitation_play_total),
        "imitation_play_matches": int(imitation_play_matches),
        "imitation_play_ratio": imitation_play_ratio,
        "imitation_match_total": int(imitation_match_total),
        "imitation_match_matches": int(imitation_match_matches),
        "imitation_match_ratio": imitation_match_ratio,
        "imitation_go_stop_total": int(imitation_option_total),
        "imitation_go_stop_matches": int(imitation_option_matches),
        "imitation_go_stop_ratio": imitation_option_ratio,
        "imitation_option_total": int(imitation_option_total),
        "imitation_option_matches": int(imitation_option_matches),
        "imitation_option_ratio": imitation_option_ratio,
        "imitation_weight_play": imitation_weight_play,
        "imitation_weight_match": imitation_weight_match,
        "imitation_weight_option": imitation_weight_option,
        "imitation_weighted_score": imitation_weighted_score,
        "fitness_components": {
            "gold_norm": gold_norm,
            "weighted_gold_delta": weighted_mean_gold_delta,
            "gold_neutral_delta": fitness_gold_neutral_delta,
            "win_norm": result_norm,
            "result_norm": result_norm,
            "result_expected": expected_result,
            "result_neutral": neutral_expected_result,
            "weighted_win_rate": weighted_win_rate,
            "weighted_draw_rate": weighted_draw_rate,
            "weighted_loss_rate": weighted_loss_rate,
            "win_neutral_rate": fitness_win_neutral_rate,
            "base_fitness": fitness,
            "bankrupt_rates": {
                "my": _rate(my_bankrupt_count, total_games),
                "inflicted": _rate(inflicted_bankrupt_count, total_games),
            },
            "bankrupt_counts": {
                "my": int(my_bankrupt_count),
                "inflicted": int(inflicted_bankrupt_count),
            },
            "weights": {
                "win": fitness_win_weight_raw,
                "gold": fitness_gold_weight_raw,
            },
        },
        "fitness": fitness,
        "seed_used": "pooled:" + ",".join(
            str(item.get("seed_used") or "").strip()
            for item in source_records
            if str(item.get("seed_used") or "").strip()
        ),
        "record_mode": "pooled_training_stage1_stage2",
        "record_sources": [
            {
                "games": int(_safe_float(item.get("games"), 0.0)),
                "seed_used": item.get("seed_used"),
            }
            for item in source_records
        ],
    }
    return pooled_record


def _population_candidate_snapshots(population, limit: int = 5) -> list[dict]:
    if population is None or not hasattr(population, "population"):
        return []
    entries = []
    for genome_key, genome in (population.population or {}).items():
        fitness = _safe_optional_float(getattr(genome, "fitness", None))
        if fitness is None:
            continue
        num_nodes, num_enabled_connections = genome.size()
        num_connections_total = len(getattr(genome, "connections", {}) or {})
        entries.append(
            {
                "record": {
                    "generation": int(getattr(population, "generation", -1)),
                    "genome_key": int(genome_key),
                    "fitness": float(fitness),
                    "num_nodes": int(num_nodes),
                    "num_connections": int(num_enabled_connections),
                    "num_connections_total": int(num_connections_total),
                },
                "genome": copy.deepcopy(genome),
            }
        )
    entries.sort(key=lambda entry: _training_candidate_sort_key(entry.get("record") or {}), reverse=True)
    return entries[: max(1, int(limit))]


def _run_winner_playoff(candidate_entries, config, runtime: dict) -> Optional[dict]:
    if not candidate_entries:
        return None

    stage1_topk = max(1, int(runtime.get("winner_playoff_topk", 5)))
    stage2_topk = max(1, int(runtime.get("winner_playoff_finalists", 2)))
    stage1_games = max(1, int(runtime.get("winner_playoff_stage1_games", runtime["games_per_genome"])))
    stage2_games = max(1, int(runtime.get("winner_playoff_stage2_games", runtime["games_per_genome"])))
    seed_base = str(runtime.get("seed") or "winner_playoff")

    stage1_seed = f"{seed_base}|winner_playoff_stage1"
    stage2_seed = f"{seed_base}|winner_playoff_stage2"
    stage1_candidates = list(candidate_entries[:stage1_topk])
    stage1_results = []
    for entry in stage1_candidates:
        training_record = dict(entry.get("record") or {})
        genome = entry.get("genome")
        if genome is None:
            continue
        playoff_record = _run_eval_worker_for_genome(
            genome=genome,
            config=config,
            runtime=runtime,
            seed_text=stage1_seed,
            generation=int(training_record.get("generation", -1)),
            genome_key=int(training_record.get("genome_key", -1)),
            games_override=stage1_games,
            early_stop_win_rate_cutoffs_override=[],
            early_stop_go_take_rate_cutoffs_override=[],
            context_label="winner_playoff_stage1",
        )
        playoff_record["generation"] = int(training_record.get("generation", -1))
        playoff_record["genome_key"] = int(training_record.get("genome_key", -1))
        playoff_record["num_nodes"] = int(training_record.get("num_nodes", 0))
        playoff_record["num_connections"] = int(training_record.get("num_connections", 0))
        playoff_record["num_connections_total"] = int(training_record.get("num_connections_total", 0))
        stage1_results.append(
            {
                "training_record": training_record,
                "stage1_record": playoff_record,
                "genome": copy.deepcopy(genome),
            }
        )

    if not stage1_results:
        return None

    stage1_results.sort(
        key=functools.cmp_to_key(
            lambda a, b: -_playoff_record_compare(
                a.get("stage1_record") or {},
                b.get("stage1_record") or {},
                runtime,
            )
        )
    )
    stage2_candidates = list(stage1_results[:stage2_topk])
    stage2_results = []
    for entry in stage2_candidates:
        training_record = dict(entry.get("training_record") or {})
        genome = entry.get("genome")
        if genome is None:
            continue
        playoff_record = _run_eval_worker_for_genome(
            genome=genome,
            config=config,
            runtime=runtime,
            seed_text=stage2_seed,
            generation=int(training_record.get("generation", -1)),
            genome_key=int(training_record.get("genome_key", -1)),
            games_override=stage2_games,
            early_stop_win_rate_cutoffs_override=[],
            early_stop_go_take_rate_cutoffs_override=[],
            context_label="winner_playoff_stage2",
        )
        playoff_record["generation"] = int(training_record.get("generation", -1))
        playoff_record["genome_key"] = int(training_record.get("genome_key", -1))
        playoff_record["num_nodes"] = int(training_record.get("num_nodes", 0))
        playoff_record["num_connections"] = int(training_record.get("num_connections", 0))
        playoff_record["num_connections_total"] = int(training_record.get("num_connections_total", 0))
        stage2_results.append(
            {
                "training_record": training_record,
                "stage1_record": dict(entry.get("stage1_record") or {}),
                "stage2_record": playoff_record,
                "genome": copy.deepcopy(genome),
            }
        )

    if not stage2_results:
        return None

    stage2_results.sort(
        key=functools.cmp_to_key(
            lambda a, b: -_playoff_record_compare(
                a.get("stage2_record") or {},
                b.get("stage2_record") or {},
                runtime,
            )
        )
    )
    winner_entry = stage2_results[0]
    return {
        "winner_genome": copy.deepcopy(winner_entry.get("genome")),
        "winner_record": dict(winner_entry.get("stage2_record") or {}),
        "winner_training_record": dict(winner_entry.get("training_record") or {}),
        "winner_stage1_record": dict(winner_entry.get("stage1_record") or {}),
        "winner_stage2_record": dict(winner_entry.get("stage2_record") or {}),
        "summary": {
            "mode": "topk_fresh_seed_playoff",
            "criteria": "stage1 top-K by training fitness, then playoff ranking: win_rate (tie<=threshold), mean_gold_delta (tie<=threshold), go_take_rate, go_fail_rate, fitness; repeat for stage2",
            "thresholds": {
                "win_rate_tie_threshold": float(runtime.get("winner_playoff_win_rate_tie_threshold", 0.01)),
                "mean_gold_delta_tie_threshold": float(
                    runtime.get("winner_playoff_mean_gold_delta_tie_threshold", 100.0)
                ),
                "go_opp_min_count": int(runtime.get("winner_playoff_go_opp_min_count", 100)),
                "go_take_rate_tie_threshold": float(
                    runtime.get("winner_playoff_go_take_rate_tie_threshold", 0.02)
                ),
            },
            "stage1": {
                "topk": int(stage1_topk),
                "games": int(stage1_games),
                "seed": stage1_seed,
                "results": [
                    _serialize_playoff_entry(entry.get("training_record") or {}, entry.get("stage1_record") or {})
                    for entry in stage1_results
                ],
            },
            "stage2": {
                "topk": int(stage2_topk),
                "games": int(stage2_games),
                "seed": stage2_seed,
                "results": [
                    _serialize_playoff_entry(entry.get("training_record") or {}, entry.get("stage2_record") or {})
                    for entry in stage2_results
                ],
            },
        },
    }


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
        "--seed-genome-count",
        type=int,
        default=0,
        help="How many genomes in the new population should be seeded from --seed-genome; 0 fills the whole population",
    )
    parser.add_argument(
        "--seed-genome-spec",
        dest="seed_genome_specs",
        action="append",
        nargs=2,
        metavar=("PATH", "COUNT"),
        default=[],
        help="May be repeated to seed multiple genome lineages into a fresh population",
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
        "--feature-profile",
        default="",
        help="Override feature profile (hand7, hand10, material10)",
    )
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


def _load_seed_genome(seed_genome_path: str):
    seed_path = os.path.abspath(str(seed_genome_path or "").strip())
    if not seed_path:
        raise RuntimeError("seed genome path is empty")
    if not os.path.exists(seed_path):
        raise RuntimeError(f"seed genome not found: {seed_genome_path}")

    with open(seed_path, "rb") as f:
        seed_genome = pickle.load(f)

    if seed_genome is None or not hasattr(seed_genome, "nodes") or not hasattr(seed_genome, "connections"):
        raise RuntimeError(f"invalid seed genome pickle: {seed_genome_path}")
    return seed_path, seed_genome


def _adapt_seed_genome_to_config(cfg, seed_genome, seed_path: str):
    allowed_input_keys = {int(x) for x in getattr(cfg.genome_config, "input_keys", [])}
    if not allowed_input_keys:
        return seed_genome

    pruned = copy.deepcopy(seed_genome)
    removed = 0

    for conn_key, conn_gene in list((getattr(pruned, "connections", {}) or {}).items()):
        try:
            in_node = int(conn_key[0])
        except Exception:
            try:
                in_node = int(getattr(conn_gene, "key", [0, 0])[0])
            except Exception:
                continue
        try:
            out_node = int(conn_key[1])
        except Exception:
            try:
                out_node = int(getattr(conn_gene, "key", [0, 0])[1])
            except Exception:
                continue
        if in_node < 0 and in_node not in allowed_input_keys:
            del pruned.connections[conn_key]
            removed += 1
            continue
        if out_node < 0:
            del pruned.connections[conn_key]
            removed += 1

    if removed <= 0:
        return seed_genome

    setattr(pruned, "_codex_pruned_seed_inputs", True)
    setattr(pruned, "_codex_pruned_seed_path", os.path.abspath(seed_path))
    setattr(pruned, "_codex_pruned_connection_count", int(removed))
    return pruned


def _seed_source_label_from_path(seed_path: str) -> str:
    raw_path = os.path.abspath(str(seed_path or "").strip())
    if not raw_path:
        return ""
    normalized = raw_path.replace("\\", "/")
    match = re.search(r"(?:^|/)(?:neat_)?phase(\d+)_seed(\d+)(?:/|$)", normalized, re.IGNORECASE)
    if match:
        return f"phase{int(match.group(1))}_seed{int(match.group(2))}"
    return os.path.splitext(os.path.basename(raw_path))[0]


def _normalize_seed_specs(cfg, seed_specs_raw: list) -> list[dict]:
    normalized = []
    for item in seed_specs_raw or []:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip()
        if not path:
            raise RuntimeError("seed genome spec path is empty")
        try:
            count = int(item.get("count") or 0)
        except Exception:
            raise RuntimeError(f"invalid seed genome count for: {path}")
        if count <= 0:
            raise RuntimeError(f"seed genome count must be > 0 for: {path}")
        seed_path, seed_genome = _load_seed_genome(path)
        seed_genome = _adapt_seed_genome_to_config(cfg, seed_genome, seed_path)
        normalized.append(
            {
                "path": os.path.abspath(path),
                "count": int(count),
                "genome": seed_genome,
                "source_label": _seed_source_label_from_path(seed_path),
            }
        )
    if not normalized:
        raise RuntimeError("seed genome specs are empty")
    return normalized


def _seed_population_from_specs(cfg, population, seed_specs: list[dict]):
    population_size = max(1, int(cfg.pop_size))
    existing_keys = sorted(population.population.keys())
    if len(existing_keys) < population_size:
        raise RuntimeError("failed to initialize base population for seed expansion")

    requested_total = sum(max(0, int(item["count"])) for item in seed_specs)
    if requested_total > population_size:
        raise RuntimeError(
            f"seed genome counts exceed population size: {requested_total} > {population_size}"
        )

    tracker = population.reproduction.innovation_tracker
    cfg.genome_config.innovation_tracker = tracker
    max_innovation = int(getattr(tracker, "global_counter", 0) or 0)
    for item in seed_specs:
        seed_genome = item["genome"]
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

    seeded_population = dict(population.population)
    population.reproduction.ancestors = {}
    bootstrap_source_by_key: dict[int, str] = {}
    offset = 0
    for item in seed_specs:
        seed_genome = item["genome"]
        seed_count = max(1, int(item["count"]))
        seed_key = int(getattr(seed_genome, "key", 0) or 0)
        source_label = str(item.get("source_label") or "").strip()
        block_keys = existing_keys[offset : offset + seed_count]
        if len(block_keys) != seed_count:
            raise RuntimeError("failed to allocate genome keys for seed expansion")
        for idx, genome_key in enumerate(block_keys):
            genome = copy.deepcopy(seed_genome)
            genome.key = int(genome_key)
            genome.fitness = None
            if idx > 0:
                genome.mutate(cfg.genome_config)
                if (idx % 4) == 0:
                    genome.mutate(cfg.genome_config)
            seeded_population[int(genome_key)] = genome
            population.reproduction.ancestors[int(genome_key)] = (seed_key,)
            if source_label:
                bootstrap_source_by_key[int(genome_key)] = source_label
        offset += seed_count

    for genome_key in existing_keys[offset:population_size]:
        genome = seeded_population.get(int(genome_key))
        if genome is None:
            continue
        genome.fitness = None
        population.reproduction.ancestors[int(genome_key)] = tuple()

    population.population = seeded_population
    setattr(population, "_codex_bootstrap_source_by_key", dict(bootstrap_source_by_key))
    population.generation = 0
    population.best_genome = None
    population.species = cfg.species_set_type(cfg.species_set_config, population.reporters)
    population.species.speciate(cfg, population.population, population.generation)
    return population


def _build_seeded_population(cfg, seed_genome_path: str, seed_genome_count: int = 0):
    if neat is None:
        raise RuntimeError("neat-python is not installed. Install with: pip install neat-python")

    population = neat.Population(cfg)
    population_size = max(1, int(cfg.pop_size))
    seed_count = int(seed_genome_count or 0)
    if seed_count <= 0:
        seed_count = population_size
    seed_count = max(1, min(population_size, int(seed_count)))
    seed_specs = _normalize_seed_specs(
        cfg,
        [
            {
                "path": seed_genome_path,
                "count": int(seed_count),
            }
        ]
    )
    return _seed_population_from_specs(cfg, population, seed_specs)


def _parse_checkpoint_display_generation(path: str) -> Optional[int]:
    name = os.path.basename(os.path.abspath(str(path or "").strip()))
    m = re.search(r"gen(\d+)$", name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _resolve_effective_base_generation(
    resume_path: str,
    start_generation: int,
    explicit_base_generation: int,
) -> int:
    explicit = max(0, int(explicit_base_generation))
    if explicit > 0:
        return explicit
    resume_raw = str(resume_path or "").strip()
    if not resume_raw:
        return 0
    display_generation = _parse_checkpoint_display_generation(resume_raw)
    if display_generation is None:
        return 0
    return max(0, int(display_generation) - int(start_generation))


def _restore_population_from_checkpoint(filename: str, new_config=None):
    with gzip.open(filename, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, tuple) or len(data) < 5:
        raise RuntimeError(f"Unsupported checkpoint payload: {filename}")

    generation, saved_config, population, species_set, rndstate = data[:5]
    extra = data[5] if len(data) >= 6 else None
    metadata: Dict[str, Any] = extra if isinstance(extra, dict) else {}

    random.setstate(rndstate)

    saved_innovation_tracker = None
    if hasattr(saved_config.genome_config, "innovation_tracker"):
        saved_innovation_tracker = saved_config.genome_config.innovation_tracker

    config = new_config if new_config is not None else saved_config
    restored_pop = neat.Population(config, (population, species_set, generation))

    if saved_innovation_tracker is not None:
        restored_pop.reproduction.innovation_tracker = saved_innovation_tracker
        config.genome_config.innovation_tracker = saved_innovation_tracker

    restored_best = metadata.get("best_genome")
    if restored_best is not None:
        restored_pop.best_genome = restored_best
    restored_best_display_generation = metadata.get("best_genome_display_generation")
    if restored_best_display_generation is not None:
        setattr(restored_pop, "best_genome_display_generation", int(restored_best_display_generation))

    return restored_pop


def _coerce_lineage_parent_tuple(raw_value: object) -> tuple[int, ...]:
    out = []
    if isinstance(raw_value, (list, tuple)):
        for item in raw_value:
            try:
                out.append(int(item))
            except Exception:
                continue
    return tuple(out)


def _load_lineage_state_from_path(path: str) -> Optional[dict]:
    target = os.path.abspath(str(path or "").strip())
    if not target or not os.path.exists(target):
        return None
    try:
        with open(target, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None

    def _coerce_int_map(src: object) -> dict[int, int]:
        out: dict[int, int] = {}
        if not isinstance(src, dict):
            return out
        for key, value in src.items():
            try:
                out[int(key)] = int(value)
            except Exception:
                continue
        return out

    def _coerce_str_map(src: object) -> dict[int, str]:
        out: dict[int, str] = {}
        if not isinstance(src, dict):
            return out
        for key, value in src.items():
            try:
                out[int(key)] = str(value or "").strip()
            except Exception:
                continue
        return out

    parents_by_key: dict[int, tuple[int, ...]] = {}
    if isinstance(raw.get("parents_by_key"), dict):
        for key, value in raw.get("parents_by_key").items():
            try:
                parents_by_key[int(key)] = _coerce_lineage_parent_tuple(value)
            except Exception:
                continue

    return {
        "saved_at": str(raw.get("saved_at") or ""),
        "source_path": target,
        "birth_generation_by_key": _coerce_int_map(raw.get("birth_generation_by_key")),
        "parents_by_key": parents_by_key,
        "origin_by_key": _coerce_str_map(raw.get("origin_by_key")),
        "last_seen_generation_by_key": _coerce_int_map(raw.get("last_seen_generation_by_key")),
        "bootstrap_source_by_key": _coerce_str_map(raw.get("bootstrap_source_by_key")),
    }


def _load_first_lineage_state(paths: list[str]) -> Optional[dict]:
    for path in paths or []:
        loaded = _load_lineage_state_from_path(path)
        if loaded is not None:
            return loaded
    return None


def _lineage_state_candidates(output_dir: str, resume_path: str = "") -> list[str]:
    candidates = [os.path.join(os.path.abspath(output_dir), "lineage_state.json")]
    resume_raw = str(resume_path or "").strip()
    if resume_raw:
        resume_output_dir = os.path.dirname(os.path.dirname(os.path.abspath(resume_raw)))
        resume_state = os.path.join(resume_output_dir, "lineage_state.json")
        if os.path.abspath(resume_state) != os.path.abspath(candidates[0]):
            candidates.append(resume_state)
    return candidates


def _build_winner_lineage_export(lineage_state: Optional[dict], winner_genome_key: int) -> Optional[dict]:
    if winner_genome_key <= 0:
        return None
    state = dict(lineage_state or {})
    birth_generation_by_key = dict(state.get("birth_generation_by_key") or {})
    parents_by_key = dict(state.get("parents_by_key") or {})
    origin_by_key = dict(state.get("origin_by_key") or {})
    last_seen_generation_by_key = dict(state.get("last_seen_generation_by_key") or {})
    bootstrap_source_by_key = dict(state.get("bootstrap_source_by_key") or {})

    queue = [int(winner_genome_key)]
    visited = set()
    nodes = []
    missing_parent_keys = set()
    while queue and len(visited) < 512:
        genome_key = int(queue.pop(0))
        if genome_key in visited:
            continue
        visited.add(genome_key)
        parent_keys = list(_coerce_lineage_parent_tuple(parents_by_key.get(genome_key)))
        nodes.append(
            {
                "genome_key": int(genome_key),
                "birth_generation": birth_generation_by_key.get(genome_key),
                "last_seen_generation": last_seen_generation_by_key.get(genome_key),
                "origin": origin_by_key.get(genome_key),
                "bootstrap_source": bootstrap_source_by_key.get(genome_key) or None,
                "parent_keys": parent_keys,
            }
        )
        for parent_key in parent_keys:
            if parent_key in visited:
                continue
            if (
                parent_key not in birth_generation_by_key
                and parent_key not in parents_by_key
                and parent_key not in origin_by_key
            ):
                missing_parent_keys.add(int(parent_key))
                continue
            queue.append(int(parent_key))

    nodes.sort(
        key=lambda item: (
            -int(item.get("birth_generation") if item.get("birth_generation") is not None else -1),
            int(item.get("genome_key", -1)),
        )
    )
    return {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "winner_genome_key": int(winner_genome_key),
        "node_count": len(nodes),
        "missing_parent_keys": sorted(missing_parent_keys),
        "winner_bootstrap_sources": sorted(
            {
                str(item.get("bootstrap_source") or "").strip()
                for item in nodes
                if str(item.get("bootstrap_source") or "").strip()
            }
        ),
        "nodes": nodes,
    }


if neat is not None:

    class OffsetCheckpointer(neat.Checkpointer):
        def __init__(
            self,
            base_generation: int,
            start_generation: int,
            total_generations: int,
            generation_interval: int,
            filename_prefix: str,
            initial_best_genome=None,
            initial_best_display_generation: Optional[int] = None,
        ):
            super().__init__(
                generation_interval=max(1, int(generation_interval)),
                filename_prefix=str(filename_prefix),
            )
            self.base_generation = int(base_generation)
            self.last_generation_checkpoint = int(start_generation)
            self._last_saved_display_generation = None
            self.best_genome_snapshot = copy.deepcopy(initial_best_genome)
            self.best_genome_display_generation = (
                int(initial_best_display_generation)
                if initial_best_display_generation is not None
                else None
            )

        def _display_from_generation(self, generation: int) -> int:
            return int(self.base_generation + int(generation))

        def post_evaluate(self, config, population, species, best_genome):
            if best_genome is None:
                return
            best_fitness = getattr(best_genome, "fitness", None)
            snapshot_fitness = getattr(self.best_genome_snapshot, "fitness", None)
            if (
                self.best_genome_snapshot is None
                or snapshot_fitness is None
                or (best_fitness is not None and float(best_fitness) > float(snapshot_fitness))
            ):
                self.best_genome_snapshot = copy.deepcopy(best_genome)
                current_generation = int(getattr(self, "current_generation", 0))
                self.best_genome_display_generation = self._display_from_generation(current_generation)

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
                metadata = {
                    "best_genome": copy.deepcopy(self.best_genome_snapshot),
                    "best_genome_display_generation": self.best_genome_display_generation,
                }
                data = (
                    int(state_generation),
                    config,
                    population,
                    species_set,
                    random.getstate(),
                    metadata,
                )
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            self._last_saved_display_generation = int(display_generation)

        def save_checkpoint(self, config, population, species_set, generation):
            display_generation = self._display_from_generation(max(0, int(generation) - 1))
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

            self.save_checkpoint(config, population, species_set, next_generation)
            self.last_generation_checkpoint = next_generation
            self.last_time_checkpoint = time.time()

        def save_final_checkpoint(self, config, population, species_set, state_generation: int) -> None:
            state_generation = int(state_generation)
            display_generation = self._display_from_generation(max(0, state_generation - 1))

            if self._last_saved_display_generation == int(display_generation):
                return

            self._save_checkpoint_with_display(
                config=config,
                population=population,
                species_set=species_set,
                state_generation=state_generation,
                display_generation=display_generation,
            )


    class LineageReporter(neat.reporting.BaseReporter):
        def __init__(
            self,
            output_dir: str,
            generation_display_offset: int = 0,
            state_seed: Optional[dict] = None,
        ):
            self.output_dir = os.path.abspath(output_dir)
            os.makedirs(self.output_dir, exist_ok=True)
            self.lineage_log = os.path.join(self.output_dir, "lineage.ndjson")
            self.lineage_state_path = os.path.join(self.output_dir, "lineage_state.json")
            self.generation_display_offset = int(generation_display_offset)
            self.current_generation = 0

            seed = dict(state_seed or {})
            self.birth_generation_by_key = dict(seed.get("birth_generation_by_key") or {})
            self.parents_by_key = {
                int(key): _coerce_lineage_parent_tuple(value)
                for key, value in dict(seed.get("parents_by_key") or {}).items()
            }
            self.origin_by_key = dict(seed.get("origin_by_key") or {})
            self.last_seen_generation_by_key = dict(seed.get("last_seen_generation_by_key") or {})
            self.bootstrap_source_by_key = dict(seed.get("bootstrap_source_by_key") or {})
            self._write_state()

        def _display_generation(self, generation: int) -> int:
            return int(self.generation_display_offset + int(generation))

        def _append_lines(self, path: str, records) -> None:
            if not records:
                return
            with open(path, "a", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False))
                    f.write("\n")

        def _write_state(self) -> None:
            payload = {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "birth_generation_by_key": {
                    str(key): int(value) for key, value in self.birth_generation_by_key.items()
                },
                "parents_by_key": {
                    str(key): [int(x) for x in value] for key, value in self.parents_by_key.items()
                },
                "origin_by_key": {
                    str(key): str(value or "") for key, value in self.origin_by_key.items()
                },
                "last_seen_generation_by_key": {
                    str(key): int(value) for key, value in self.last_seen_generation_by_key.items()
                },
                "bootstrap_source_by_key": {
                    str(key): str(value or "") for key, value in self.bootstrap_source_by_key.items()
                },
            }
            with open(self.lineage_state_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        def _species_membership(self, species_set) -> dict[int, int]:
            out: dict[int, int] = {}
            for species_id, species in (getattr(species_set, "species", {}) or {}).items():
                for genome_key in (getattr(species, "members", {}) or {}).keys():
                    try:
                        out[int(genome_key)] = int(species_id)
                    except Exception:
                        continue
            return out

        def start_generation(self, generation):
            self.current_generation = int(generation)

        def post_evaluate(self, config, population, species, best_genome):
            display_generation = self._display_generation(self.current_generation)
            ancestors_map = getattr(config, "_codex_lineage_ancestors", {}) or {}
            bootstrap_sources_map = getattr(config, "_codex_lineage_bootstrap_sources", {}) or {}
            species_by_key = self._species_membership(species)
            records = []
            for genome_key, genome in (population or {}).items():
                key = int(genome_key)
                parents = _coerce_lineage_parent_tuple(ancestors_map.get(key))
                if key not in self.parents_by_key:
                    self.parents_by_key[key] = parents
                elif parents and len(self.parents_by_key.get(key, tuple())) <= 0:
                    self.parents_by_key[key] = parents
                bootstrap_source = str(bootstrap_sources_map.get(key) or self.bootstrap_source_by_key.get(key) or "").strip()
                if (not bootstrap_source) and parents:
                    inherited_sources = sorted(
                        {
                            str(self.bootstrap_source_by_key.get(int(parent_key)) or "").strip()
                            for parent_key in parents
                            if str(self.bootstrap_source_by_key.get(int(parent_key)) or "").strip()
                        }
                    )
                    if len(inherited_sources) == 1:
                        bootstrap_source = inherited_sources[0]
                    elif len(inherited_sources) >= 2:
                        bootstrap_source = ",".join(inherited_sources)
                if bootstrap_source:
                    self.bootstrap_source_by_key[key] = bootstrap_source

                is_first_seen = key not in self.birth_generation_by_key
                if is_first_seen:
                    self.birth_generation_by_key[key] = int(display_generation)
                    if len(parents) >= 2:
                        self.origin_by_key[key] = "offspring"
                    elif len(parents) == 1:
                        self.origin_by_key[key] = "bootstrap_seed"
                    else:
                        self.origin_by_key[key] = "init"
                self.last_seen_generation_by_key[key] = int(display_generation)

                eval_meta = dict(getattr(genome, "_codex_eval_meta", {}) or {})
                origin = str(self.origin_by_key.get(key) or "")
                records.append(
                    {
                        "saved_at": datetime.now(timezone.utc).isoformat(),
                        "generation": int(display_generation),
                        "genome_key": int(key),
                        "birth_generation": int(self.birth_generation_by_key.get(key, display_generation)),
                        "lineage_event": (origin if is_first_seen else "carryover"),
                        "origin": origin,
                        "bootstrap_source": (bootstrap_source or None),
                        "parent_keys": [int(x) for x in self.parents_by_key.get(key, tuple())],
                        "species_id": species_by_key.get(key),
                        "fitness": _safe_float(getattr(genome, "fitness", None), -1e9),
                        "win_rate": _safe_optional_float(eval_meta.get("win_rate")),
                        "mean_gold_delta": _safe_optional_float(eval_meta.get("mean_gold_delta")),
                        "games": int(_safe_float(eval_meta.get("games"), 0.0)),
                        "full_eval_passed": bool(eval_meta.get("full_eval_passed", False)),
                        "go_take_rate": _safe_optional_float(eval_meta.get("go_take_rate")),
                        "go_fail_rate": _safe_optional_float(eval_meta.get("go_fail_rate")),
                    }
                )
            self._append_lines(self.lineage_log, records)
            self._write_state()

        def snapshot_state(self) -> dict:
            return {
                "birth_generation_by_key": dict(self.birth_generation_by_key),
                "parents_by_key": {
                    int(key): tuple(value) for key, value in self.parents_by_key.items()
                },
                "origin_by_key": dict(self.origin_by_key),
                "last_seen_generation_by_key": dict(self.last_seen_generation_by_key),
                "bootstrap_source_by_key": dict(self.bootstrap_source_by_key),
            }


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
    seed_genome_count = max(0, int(args.seed_genome_count or 0))
    seed_genome_specs = []
    for raw_item in args.seed_genome_specs or []:
        if not isinstance(raw_item, (list, tuple)) or len(raw_item) != 2:
            continue
        raw_path = str(raw_item[0] or "").strip()
        if not raw_path:
            raise RuntimeError("seed genome spec path is empty")
        try:
            raw_count = int(raw_item[1])
        except Exception:
            raise RuntimeError(f"invalid seed genome count for: {raw_path}")
        seed_genome_specs.append({"path": raw_path, "count": raw_count})
    if seed_genome_path:
        applied_overrides["seed_genome"] = seed_genome_path
        if seed_genome_count > 0:
            applied_overrides["seed_genome_count"] = int(seed_genome_count)
    if seed_genome_specs:
        applied_overrides["seed_genome_specs"] = [
            {"path": str(item["path"]), "count": int(item["count"])} for item in seed_genome_specs
        ]
    if seed_genome_path and seed_genome_specs:
        raise RuntimeError("--seed-genome and --seed-genome-spec are mutually exclusive")
    if args.resume and (seed_genome_path or seed_genome_specs):
        raise RuntimeError("--resume and seed genome bootstrap args are mutually exclusive")
    explicit_base_generation = max(0, int(args.base_generation))
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
    if str(args.feature_profile).strip():
        runtime["feature_profile"] = str(args.feature_profile).strip().lower()
        override_keys.append("feature_profile")
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
    if len(cfg.genome_config.output_keys) != 2:
        raise RuntimeError("num_outputs must be 2 for action_score + option_bias threshold policy")

    _set_eval_env(runtime, args.output_dir)
    runtime["output_dir"] = os.path.abspath(args.output_dir)

    if args.resume:
        p = _restore_population_from_checkpoint(args.resume)
    elif seed_genome_specs:
        p = _seed_population_from_specs(cfg, neat.Population(cfg), _normalize_seed_specs(cfg, seed_genome_specs))
    elif seed_genome_path:
        p = _build_seeded_population(cfg, seed_genome_path, seed_genome_count=seed_genome_count)
    else:
        p = neat.Population(cfg)
    lineage_state_seed = _load_first_lineage_state(_lineage_state_candidates(args.output_dir, args.resume))
    if lineage_state_seed is not None:
        restored_parent_map = {
            int(key): _coerce_lineage_parent_tuple(value)
            for key, value in dict(lineage_state_seed.get("parents_by_key") or {}).items()
        }
        current_parent_map = dict(getattr(p.reproduction, "ancestors", {}) or {})
        for key, value in current_parent_map.items():
            coerced = _coerce_lineage_parent_tuple(value)
            if int(key) not in restored_parent_map or len(coerced) > 0:
                restored_parent_map[int(key)] = coerced
        p.reproduction.ancestors = restored_parent_map
    setattr(p.config, "_codex_lineage_ancestors", p.reproduction.ancestors)
    restored_bootstrap_sources = {
        int(key): str(value or "").strip()
        for key, value in dict((lineage_state_seed or {}).get("bootstrap_source_by_key") or {}).items()
        if str(value or "").strip()
    }
    current_bootstrap_sources = {
        int(key): str(value or "").strip()
        for key, value in dict(getattr(p, "_codex_bootstrap_source_by_key", {}) or {}).items()
        if str(value or "").strip()
    }
    restored_bootstrap_sources.update(current_bootstrap_sources)
    setattr(p.config, "_codex_lineage_bootstrap_sources", restored_bootstrap_sources)
    start_generation = int(getattr(p, "generation", 0))
    base_generation = _resolve_effective_base_generation(
        resume_path=args.resume,
        start_generation=start_generation,
        explicit_base_generation=explicit_base_generation,
    )
    if base_generation != 0:
        applied_overrides["base_generation"] = int(base_generation)

    # Keep training output quiet by default to reduce terminal I/O overhead.
    if bool(args.verbose):
        p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    lineage_reporter = LineageReporter(
        output_dir=args.output_dir,
        generation_display_offset=int(base_generation),
        state_seed=lineage_state_seed,
    )
    p.add_reporter(lineage_reporter)

    prefix = os.path.join(checkpoints_dir, "neat-checkpoint-")
    checkpointer = OffsetCheckpointer(
        base_generation=int(base_generation),
        start_generation=int(start_generation),
        total_generations=int(runtime["generations"]),
        generation_interval=max(1, int(runtime["checkpoint_every"])),
        filename_prefix=prefix,
        initial_best_genome=getattr(p, "best_genome", None),
        initial_best_display_generation=getattr(p, "best_genome_display_generation", None),
    )
    p.add_reporter(checkpointer)

    evaluator = None

    def _run_population(eval_callable):
        if bool(args.verbose):
            return p.run(eval_callable, int(runtime["generations"]))
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull):
                return p.run(eval_callable, int(runtime["generations"]))

    skip_training_run = bool(args.resume) and int(start_generation) >= int(runtime["generations"])
    if skip_training_run:
        winner = getattr(p, "best_genome", None)
        mode = "resume_postprocess"
    elif args.dry_run:
        winner = _run_population(_run_dry_eval)
        checkpointer.save_final_checkpoint(p.config, p.population, p.species, p.generation)
        mode = "dry_run"
    else:
        evaluator = LoggedParallelEvaluator(
            num_workers=int(runtime["eval_workers"]),
            output_dir=args.output_dir,
            runtime=runtime,
            start_generation=int(start_generation),
            generation_display_offset=int(base_generation),
        )
        try:
            winner = _run_population(evaluator.evaluate)
            checkpointer.save_final_checkpoint(p.config, p.population, p.species, p.generation)
            mode = "real_eval"
        finally:
            evaluator.close()

    best_winner = None
    if evaluator is not None:
        best_winner = evaluator.best_genome_snapshot()
    if best_winner is None:
        checkpointer_best = getattr(checkpointer, "best_genome_snapshot", None)
        if checkpointer_best is not None:
            best_winner = copy.deepcopy(checkpointer_best)
    if best_winner is None:
        raise RuntimeError(
            "no full-evaluation-qualified best genome found; winner selection requires games_per_genome completion"
        )

    selection_best_record = evaluator.best_record_snapshot() if evaluator is not None else None
    winner_playoff = None
    playoff_candidates = []
    if evaluator is not None:
        playoff_candidates = evaluator.top_candidate_snapshots(limit=int(runtime.get("winner_playoff_topk", 5)))
    elif skip_training_run:
        playoff_candidates = _population_candidate_snapshots(p, limit=int(runtime.get("winner_playoff_topk", 5)))
    if len(playoff_candidates) > 0:
        winner_playoff = _run_winner_playoff(playoff_candidates, p.config, runtime)
    if winner_playoff is not None and winner_playoff.get("winner_genome") is not None:
        best_winner = copy.deepcopy(winner_playoff.get("winner_genome"))

    def _write_genome_exports(file_stem: str, genome_obj):
        pickle_path = os.path.join(models_dir, f"{file_stem}.pkl")
        json_path = os.path.join(models_dir, f"{file_stem}.json")
        with open(pickle_path, "wb") as f:
            pickle.dump(genome_obj, f)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                _export_neat_python_genome(genome_obj, p.config),
                f,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        return pickle_path, json_path

    winner_pkl, winner_json_path = _write_genome_exports("winner_genome", best_winner)

    best_record_stage2 = (
        dict(winner_playoff.get("winner_record") or {})
        if winner_playoff is not None and isinstance(winner_playoff.get("winner_record"), dict)
        else selection_best_record
    )
    best_record_pooled = None
    if winner_playoff is not None:
        best_record_pooled = _build_pooled_best_record(
            [
                dict(winner_playoff.get("winner_training_record") or {}),
                dict(winner_playoff.get("winner_stage1_record") or {}),
                dict(winner_playoff.get("winner_stage2_record") or {}),
            ],
            runtime,
        )
    best_record = best_record_pooled or best_record_stage2
    best_fitness = _safe_float((best_record or {}).get("fitness"), float("nan"))
    if best_fitness != best_fitness:
        best_fitness = float(getattr(best_winner, "fitness", float("nan")) or 0.0)
    winner_generation = (best_record or {}).get("generation")

    champion_exports = {
        "fitness": {
            "criteria": (
                "pooled training + top-K fresh-seed playoff winner"
                if winner_playoff is not None
                else "full_eval, highest selection fitness"
            ),
            "record": best_record,
            "record_stage2": best_record_stage2,
            "winner_pickle": winner_pkl,
            "winner_json": winner_json_path,
        }
    }
    lineage_state_snapshot = lineage_reporter.snapshot_state()
    winner_lineage_key = int(
        _safe_float((best_record or {}).get("genome_key"), getattr(best_winner, "key", -1) or -1)
    )
    winner_lineage_path = os.path.join(args.output_dir, "winner_lineage.json")
    winner_lineage_export = _build_winner_lineage_export(lineage_state_snapshot, winner_lineage_key)
    if winner_lineage_export is not None:
        with open(winner_lineage_path, "w", encoding="utf-8") as f:
            json.dump(winner_lineage_export, f, ensure_ascii=False, indent=2)
    else:
        winner_lineage_path = None

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
        "winner_generation": (int(winner_generation) if winner_generation is not None else None),
        "best_record": best_record,
        "best_record_stage2": best_record_stage2,
        "best_record_pooled": best_record_pooled,
        "winner_playoff": (winner_playoff.get("summary") if winner_playoff is not None else None),
        "champions": champion_exports,
        "applied_overrides": applied_overrides,
        "runtime_effective": runtime,
        "eval_failure_log": os.path.join(args.output_dir, "eval_failures.log"),
        "eval_metrics_log": os.path.join(args.output_dir, "eval_metrics.ndjson"),
        "generation_metrics_log": os.path.join(args.output_dir, "generation_metrics.ndjson"),
        "lineage_log": os.path.join(args.output_dir, "lineage.ndjson"),
        "lineage_state_path": os.path.join(args.output_dir, "lineage_state.json"),
        "winner_lineage_path": winner_lineage_path,
        "gate_state_path": os.path.join(args.output_dir, "gate_state.json"),
        "gate_state": evaluator.snapshot() if evaluator is not None else {},
        "winner_pickle": winner_pkl,
        "winner_json": winner_json_path,
        "seed_genome": seed_genome_path or None,
        "seed_genome_specs": seed_genome_specs or None,
        "config_feedforward": os.path.abspath(args.config_feedforward),
        "runtime_config": os.path.abspath(args.runtime_config),
    }
    run_summary_path = os.path.join(args.output_dir, "run_summary.json")
    with open(run_summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    best_record_summary = best_record or {}
    console_summary = {
        "output_dir": os.path.abspath(args.output_dir),
        "run_summary": run_summary_path,
        "winner_generation": run_summary.get("winner_generation"),
        "best_fitness": run_summary.get("best_fitness"),
        "win_rate": best_record_summary.get("win_rate"),
        "mean_gold_delta": best_record_summary.get("mean_gold_delta"),
        "go_take_rate": best_record_summary.get("go_take_rate"),
        "go_fail_rate": best_record_summary.get("go_fail_rate"),
        "winner_selection_mode": (
            "topk_fresh_seed_playoff" if winner_playoff is not None else "selection_fitness"
        ),
        "best_record_mode": (
            "pooled_training_stage1_stage2" if winner_playoff is not None else "selection_fitness"
        ),
        "winner_json": winner_json_path,
    }
    print(json.dumps(console_summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
