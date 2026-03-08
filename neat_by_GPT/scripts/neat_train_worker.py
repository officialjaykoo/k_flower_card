#!/usr/bin/env python3
from __future__ import annotations

"""
 Pipeline Stage: focus_cl (neat_by_GPT/scripts/neat_train_worker.py -> neat_by_GPT/scripts/neat_eval_worker.mjs -> neat_by_GPT/eval.ps1)

Execution Flow Map:
1) parse_args()/main(): runtime bootstrap and training orchestration
2) LoggedParallelEvaluator: generation metrics + gate tracking
3) eval_function(): per-genome node-worker evaluation call

File Layout Map (top-down):
1) runtime defaults + normalization + env bridge helpers
2) logging/numeric utilities
3) eval function + parallel evaluator + CLI entrypoint
"""

import argparse
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


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NEAT_GPT_ROOT = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(NEAT_GPT_ROOT)
DEFAULT_FEEDFORWARD_CONFIG = os.path.join(NEAT_GPT_ROOT, "configs", "neat_feedforward.ini")
DEFAULT_RUNTIME_CONFIG = os.path.join(NEAT_GPT_ROOT, "configs", "runtime_focus_cl_v1.json")


# =============================================================================
# Section 1. Runtime Defaults + Primitive Coercion Helpers
# =============================================================================
ENV_PREFIX = "KFC_NEAT_"
ALLOWED_FITNESS_PROFILES = {"phase1", "phase2", "phase3", "focus"}
RUNTIME_REQUIRED_KEYS = (
    "format_version",
    "seed",
    "eval_script",
    "teacher_policy",
    "fitness_profile",
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
    "selection_eval_games",
    "selection_top_k",
    "selection_opponent_policy",
    "selection_opponent_policy_mix",
    "selection_opponent_genome",
    "gate_mode",
    "gate_ema_window",
    "transition_ema_win_rate",
    "transition_mean_gold_delta_min",
    "transition_cvar10_gold_delta_min",
    "transition_catastrophic_loss_rate_max",
    "transition_best_fitness_min",
    "transition_streak",
    "failure_generation_min",
    "failure_ema_win_rate_max",
    "failure_mean_gold_delta_max",
    "failure_cvar10_gold_delta_max",
    "failure_catastrophic_loss_rate_min",
    "failure_slope_5_max",
    "failure_slope_metric",
)

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


def _sanitize_file_part(value: object, fallback: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        text = str(fallback or "").strip().lower()
    if not text:
        return "run"
    out = []
    for ch in text:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    sanitized = "".join(out).strip("_")
    return sanitized or "run"


def _parse_int_strict(name: str, value: object) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise RuntimeError(f"{name} must be an integer") from exc


def _parse_float_strict(name: str, value: object) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise RuntimeError(f"{name} must be a number") from exc
    if not math.isfinite(out):
        raise RuntimeError(f"{name} must be finite")
    return out


def _parse_optional_float_strict(name: str, value: object) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in ("", "none", "null"):
        return None
    return _parse_float_strict(name, value)


def _parse_bool_strict(name: str, value: object) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    raise RuntimeError(f"{name} must be boolean")


def _resolve_runtime_path(base_path: str, child_path: str) -> str:
    if os.path.isabs(child_path):
        return child_path
    base_dir = os.path.dirname(os.path.abspath(base_path)) if base_path else os.getcwd()
    return os.path.normpath(os.path.join(base_dir, child_path))


def _resolve_runtime_value_paths(base_path: str, raw_cfg: dict) -> dict:
    cfg = dict(raw_cfg or {})
    for key in ("eval_script", "opponent_genome", "selection_opponent_genome"):
        value = cfg.get(key)
        if not isinstance(value, str):
            continue
        text = value.strip()
        if not text:
            continue
        cfg[key] = _resolve_runtime_path(base_path, text)
    return cfg


def _resolve_output_dir(cli_output_dir: str, runtime_config_path: str, runtime: dict) -> str:
    explicit = str(cli_output_dir or "").strip()
    if explicit:
        return os.path.abspath(explicit)

    runtime_output_dir = str(runtime.get("output_dir") or "").strip()
    if runtime_output_dir:
        if os.path.isabs(runtime_output_dir):
            return os.path.abspath(runtime_output_dir)
        return os.path.abspath(_resolve_runtime_path(runtime_config_path, runtime_output_dir))

    runtime_name = os.path.splitext(os.path.basename(str(runtime_config_path or "")))[0] or "runtime"
    seed_tag = _sanitize_file_part(runtime.get("seed"), "run")
    return os.path.join(REPO_ROOT, "logs", "NEAT_GPT", f"{runtime_name}_seed{seed_tag}")


def _require_runtime_key(cfg: dict, key: str) -> object:
    if key not in cfg:
        raise RuntimeError(f"runtime missing required key: {key}")
    return cfg.get(key)


def _parse_weighted_policy_entry(item: object) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        raise RuntimeError("opponent_policy_mix items must be objects with policy and weight")
    policy = str(item.get("policy") or "").strip()
    weight = _parse_float_strict("opponent_policy_mix.weight", item.get("weight"))
    if not policy:
        raise RuntimeError("opponent_policy_mix item policy is required")
    if (not math.isfinite(weight)) or float(weight) <= 0.0:
        raise RuntimeError(f"invalid opponent_policy_mix weight for policy={policy}")
    return {"policy": policy, "weight": float(weight)}


def _parse_opponent_policy_mix(raw_value: object) -> list:
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
            raise RuntimeError(f"opponent_policy_mix must be a JSON array: {exc}") from exc
        if not isinstance(parsed, list):
            raise RuntimeError("opponent_policy_mix must be a JSON array")
        source = parsed

    merged: Dict[str, float] = {}
    ordered_policies = []
    for item in source:
        parsed = _parse_weighted_policy_entry(item)
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
        raise RuntimeError(f"runtime config not found: {abs_path}")

    with open(abs_path, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise RuntimeError(f"runtime config root must be an object: {abs_path}")

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
    local = _resolve_runtime_value_paths(abs_path, local)
    cfg.update(local)


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
    for key in RUNTIME_REQUIRED_KEYS:
        _require_runtime_key(cfg, key)

    cfg["format_version"] = str(_require_runtime_key(cfg, "format_version") or "").strip()
    if not cfg["format_version"]:
        raise RuntimeError("runtime format_version is required")
    cfg["generations"] = max(1, _parse_int_strict("runtime generations", _require_runtime_key(cfg, "generations")))
    cfg["eval_workers"] = max(2, _parse_int_strict("runtime eval_workers", _require_runtime_key(cfg, "eval_workers")))
    cfg["games_per_genome"] = max(1, _parse_int_strict("runtime games_per_genome", _require_runtime_key(cfg, "games_per_genome")))
    cfg["eval_timeout_sec"] = max(10, _parse_int_strict("runtime eval_timeout_sec", _require_runtime_key(cfg, "eval_timeout_sec")))
    cfg["max_eval_steps"] = max(50, _parse_int_strict("runtime max_eval_steps", _require_runtime_key(cfg, "max_eval_steps")))
    cfg["opponent_policy"] = str(_require_runtime_key(cfg, "opponent_policy") or "").strip()
    cfg["opponent_policy_mix"] = _parse_opponent_policy_mix(_require_runtime_key(cfg, "opponent_policy_mix"))
    cfg["opponent_genome"] = str(_require_runtime_key(cfg, "opponent_genome") or "").strip()
    cfg["switch_seats"] = _parse_bool_strict("runtime switch_seats", _require_runtime_key(cfg, "switch_seats"))
    cfg["checkpoint_every"] = max(1, _parse_int_strict("runtime checkpoint_every", _require_runtime_key(cfg, "checkpoint_every")))
    cfg["selection_eval_games"] = max(
        1, _parse_int_strict("runtime selection_eval_games", _require_runtime_key(cfg, "selection_eval_games"))
    )
    cfg["selection_top_k"] = max(
        1, _parse_int_strict("runtime selection_top_k", _require_runtime_key(cfg, "selection_top_k"))
    )
    cfg["selection_opponent_policy"] = str(_require_runtime_key(cfg, "selection_opponent_policy") or "").strip()
    cfg["selection_opponent_policy_mix"] = _parse_opponent_policy_mix(
        _require_runtime_key(cfg, "selection_opponent_policy_mix")
    )
    cfg["selection_opponent_genome"] = str(_require_runtime_key(cfg, "selection_opponent_genome") or "").strip()
    cfg["eval_script"] = str(_require_runtime_key(cfg, "eval_script") or "").strip()
    if not cfg["eval_script"]:
        raise RuntimeError("runtime eval_script is required")
    cfg["teacher_policy"] = str(cfg.get("teacher_policy") or "").strip()
    fitness_profile = str(cfg.get("fitness_profile") or "").strip().lower()
    if fitness_profile not in ALLOWED_FITNESS_PROFILES:
        allowed = ", ".join(sorted(ALLOWED_FITNESS_PROFILES))
        raise RuntimeError(f"runtime fitness_profile must be one of: {allowed}")
    cfg["fitness_profile"] = fitness_profile
    cfg["seed"] = _to_seed(_require_runtime_key(cfg, "seed"), "")
    if not cfg["seed"]:
        raise RuntimeError("runtime seed is required")
    gate_mode = str(_require_runtime_key(cfg, "gate_mode") or "").strip().lower()
    if gate_mode != "target_eval_multi_metric":
        raise RuntimeError("runtime gate_mode must be target_eval_multi_metric")
    cfg["gate_mode"] = "target_eval_multi_metric"
    cfg["gate_ema_window"] = max(2, _parse_int_strict("runtime gate_ema_window", _require_runtime_key(cfg, "gate_ema_window")))
    cfg["transition_ema_win_rate"] = _parse_optional_float_strict(
        "runtime transition_ema_win_rate", _require_runtime_key(cfg, "transition_ema_win_rate")
    )
    cfg["transition_mean_gold_delta_min"] = _parse_optional_float_strict(
        "runtime transition_mean_gold_delta_min", _require_runtime_key(cfg, "transition_mean_gold_delta_min")
    )
    cfg["transition_cvar10_gold_delta_min"] = _parse_optional_float_strict(
        "runtime transition_cvar10_gold_delta_min", _require_runtime_key(cfg, "transition_cvar10_gold_delta_min")
    )
    cfg["transition_catastrophic_loss_rate_max"] = _parse_optional_float_strict(
        "runtime transition_catastrophic_loss_rate_max",
        _require_runtime_key(cfg, "transition_catastrophic_loss_rate_max"),
    )
    cfg["transition_best_fitness_min"] = _parse_optional_float_strict(
        "runtime transition_best_fitness_min", _require_runtime_key(cfg, "transition_best_fitness_min")
    )
    cfg["transition_streak"] = max(
        1, _parse_int_strict("runtime transition_streak", _require_runtime_key(cfg, "transition_streak"))
    )
    cfg["failure_generation_min"] = max(
        1, _parse_int_strict("runtime failure_generation_min", _require_runtime_key(cfg, "failure_generation_min"))
    )
    cfg["failure_ema_win_rate_max"] = _parse_optional_float_strict(
        "runtime failure_ema_win_rate_max", _require_runtime_key(cfg, "failure_ema_win_rate_max")
    )
    cfg["failure_mean_gold_delta_max"] = _parse_optional_float_strict(
        "runtime failure_mean_gold_delta_max", _require_runtime_key(cfg, "failure_mean_gold_delta_max")
    )
    cfg["failure_cvar10_gold_delta_max"] = _parse_optional_float_strict(
        "runtime failure_cvar10_gold_delta_max", _require_runtime_key(cfg, "failure_cvar10_gold_delta_max")
    )
    cfg["failure_catastrophic_loss_rate_min"] = _parse_optional_float_strict(
        "runtime failure_catastrophic_loss_rate_min",
        _require_runtime_key(cfg, "failure_catastrophic_loss_rate_min"),
    )
    cfg["failure_slope_5_max"] = _parse_float_strict(
        "runtime failure_slope_5_max", _require_runtime_key(cfg, "failure_slope_5_max")
    )
    failure_slope_metric = str(
        _require_runtime_key(cfg, "failure_slope_metric") or ""
    ).strip().lower()
    if failure_slope_metric != "win_rate":
        raise RuntimeError("runtime failure_slope_metric must be win_rate for GPT line")
    cfg["failure_slope_metric"] = "win_rate"

    # GPT line cleanup: offline teacher dataset path is disabled.
    teacher_dataset_path = str(cfg.get("teacher_dataset_path") or "").strip()
    teacher_kibo_path = str(cfg.get("teacher_kibo_path") or "").strip()
    teacher_dataset_decisions = max(0, _parse_int_strict("runtime teacher_dataset_decisions", cfg.get("teacher_dataset_decisions") or 0))
    if teacher_dataset_path or teacher_kibo_path or teacher_dataset_decisions > 0:
        raise RuntimeError("teacher_dataset_* options are disabled in GPT line runtime")
    cfg["teacher_dataset_path"] = ""
    cfg["teacher_kibo_path"] = ""
    cfg["teacher_kibo_count_records"] = False
    cfg["teacher_dataset_actor"] = "all"
    cfg["teacher_dataset_decisions"] = 0
    cfg["teacher_dataset_cache_path"] = ""

    if "output_dir" in cfg:
        cfg["output_dir"] = str(cfg.get("output_dir") or "").strip()
    has_selection_policy = bool(cfg["selection_opponent_policy"])
    has_selection_mix = isinstance(cfg["selection_opponent_policy_mix"], list) and len(cfg["selection_opponent_policy_mix"]) > 0
    if not has_selection_policy and not has_selection_mix:
        raise RuntimeError("runtime must define selection_opponent_policy or selection_opponent_policy_mix")
    return cfg


def _normalize_eval_runtime_values(cfg: dict) -> dict:
    cfg = dict(cfg or {})
    required = (
        "eval_script",
        "games_per_genome",
        "eval_timeout_sec",
        "max_eval_steps",
        "opponent_policy",
        "opponent_policy_mix",
        "opponent_genome",
        "teacher_policy",
        "fitness_profile",
        "switch_seats",
        "seed",
        "output_dir",
    )
    for key in required:
        _require_runtime_key(cfg, key)

    cfg["eval_script"] = str(_require_runtime_key(cfg, "eval_script") or "").strip()
    if not cfg["eval_script"]:
        raise RuntimeError("env eval_script is required")
    cfg["games_per_genome"] = max(1, _parse_int_strict("env games_per_genome", _require_runtime_key(cfg, "games_per_genome")))
    cfg["eval_timeout_sec"] = max(10, _parse_int_strict("env eval_timeout_sec", _require_runtime_key(cfg, "eval_timeout_sec")))
    cfg["max_eval_steps"] = max(50, _parse_int_strict("env max_eval_steps", _require_runtime_key(cfg, "max_eval_steps")))
    cfg["opponent_policy"] = str(_require_runtime_key(cfg, "opponent_policy") or "").strip()
    cfg["opponent_policy_mix"] = _parse_opponent_policy_mix(_require_runtime_key(cfg, "opponent_policy_mix"))
    cfg["opponent_genome"] = str(_require_runtime_key(cfg, "opponent_genome") or "").strip()
    cfg["teacher_policy"] = str(cfg.get("teacher_policy") or "").strip()
    cfg["switch_seats"] = _parse_bool_strict("env switch_seats", _require_runtime_key(cfg, "switch_seats"))
    fitness_profile = str(_require_runtime_key(cfg, "fitness_profile") or "").strip().lower()
    if fitness_profile not in ALLOWED_FITNESS_PROFILES:
        allowed = ", ".join(sorted(ALLOWED_FITNESS_PROFILES))
        raise RuntimeError(f"env fitness_profile must be one of: {allowed}")
    cfg["fitness_profile"] = fitness_profile
    cfg["seed"] = _to_seed(_require_runtime_key(cfg, "seed"), "")
    if not cfg["seed"]:
        raise RuntimeError("env seed is required")
    cfg["output_dir"] = str(_require_runtime_key(cfg, "output_dir") or "").strip()
    if not cfg["output_dir"]:
        raise RuntimeError("env output_dir is required")
    return cfg


# =============================================================================
# Section 3. Runtime <-> Environment Bridge
# =============================================================================
def _set_eval_env(runtime: dict, output_dir: str) -> None:
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
    os.environ[f"{ENV_PREFIX}TEACHER_POLICY"] = str(runtime.get("teacher_policy") or "")
    os.environ[f"{ENV_PREFIX}FITNESS_PROFILE"] = str(runtime["fitness_profile"])
    os.environ[f"{ENV_PREFIX}SWITCH_SEATS"] = "1" if bool(runtime["switch_seats"]) else "0"
    os.environ[f"{ENV_PREFIX}SEED"] = str(runtime["seed"])
    os.environ[f"{ENV_PREFIX}OUTPUT_DIR"] = os.path.abspath(output_dir)


def _runtime_from_env() -> Dict[str, object]:
    raw: Dict[str, object] = {
        "eval_script": os.environ.get(f"{ENV_PREFIX}EVAL_SCRIPT") or "",
        "games_per_genome": os.environ.get(f"{ENV_PREFIX}GAMES_PER_GENOME"),
        "eval_timeout_sec": os.environ.get(f"{ENV_PREFIX}EVAL_TIMEOUT_SEC"),
        "max_eval_steps": os.environ.get(f"{ENV_PREFIX}MAX_EVAL_STEPS"),
        "opponent_policy": os.environ.get(f"{ENV_PREFIX}OPPONENT_POLICY"),
        "opponent_policy_mix": os.environ.get(f"{ENV_PREFIX}OPPONENT_POLICY_MIX"),
        "opponent_genome": os.environ.get(f"{ENV_PREFIX}OPPONENT_GENOME"),
        "teacher_policy": os.environ.get(f"{ENV_PREFIX}TEACHER_POLICY"),
        "fitness_profile": os.environ.get(f"{ENV_PREFIX}FITNESS_PROFILE"),
        "switch_seats": os.environ.get(f"{ENV_PREFIX}SWITCH_SEATS"),
        "seed": os.environ.get(f"{ENV_PREFIX}SEED"),
        "output_dir": os.environ.get(f"{ENV_PREFIX}OUTPUT_DIR") or os.getcwd(),
    }
    return _normalize_eval_runtime_values(raw)


_RUNTIME_FROM_ENV_CACHE: Optional[Dict[str, object]] = None


def _runtime_from_env_cached(force_reload: bool = False) -> Dict[str, object]:
    global _RUNTIME_FROM_ENV_CACHE
    if force_reload or _RUNTIME_FROM_ENV_CACHE is None:
        _RUNTIME_FROM_ENV_CACHE = _runtime_from_env()
    return _RUNTIME_FROM_ENV_CACHE


# =============================================================================
# Section 4. Logging + Numeric Utilities
# =============================================================================
def _append_eval_failure_log(output_dir: str, record: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "eval_failures.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")


def _copy_genome(genome):
    return pickle.loads(pickle.dumps(genome, protocol=pickle.HIGHEST_PROTOCOL))


def _evaluate_genome_payload(
    payload: dict,
    eval_runtime: dict,
    seed_text: str,
    generation: int,
    genome_key: int,
    evaluation_scope: str,
) -> dict:
    eval_script = str(eval_runtime["eval_script"] or "")
    output_dir = str(eval_runtime["output_dir"] or os.getcwd())
    opponent_policy = str(eval_runtime.get("opponent_policy") or "").strip()
    opponent_policy_mix = eval_runtime.get("opponent_policy_mix") or []
    opponent_genome = str(eval_runtime.get("opponent_genome") or "").strip()
    teacher_policy = str(eval_runtime.get("teacher_policy") or "").strip()
    has_policy = bool(opponent_policy)
    has_policy_mix = isinstance(opponent_policy_mix, list) and len(opponent_policy_mix) > 0
    failure_meta = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "generation": int(generation),
        "genome_key": int(genome_key),
        "seed_used": seed_text,
        "evaluation_scope": evaluation_scope,
        "opponent_policy": opponent_policy,
        "opponent_policy_mix": opponent_policy_mix,
        "teacher_policy": teacher_policy,
    }

    if not eval_script or not os.path.exists(eval_script):
        _append_eval_failure_log(
            output_dir,
            dict(failure_meta, reason="eval_script_missing", eval_script=eval_script),
        )
        return {"fitness": -1e9, "seed_used": seed_text, "eval_ok": False, "evaluation_scope": evaluation_scope}

    if not has_policy and not has_policy_mix:
        _append_eval_failure_log(
            output_dir,
            dict(failure_meta, reason="missing_opponent_policy_and_mix"),
        )
        return {"fitness": -1e9, "seed_used": seed_text, "eval_ok": False, "evaluation_scope": evaluation_scope}

    requires_genome_path = opponent_policy.lower() == "genome"
    if not has_policy:
        for item in opponent_policy_mix:
            if isinstance(item, dict) and str(item.get("policy") or "").strip().lower() == "genome":
                requires_genome_path = True
                break
    if requires_genome_path:
        if not opponent_genome:
            _append_eval_failure_log(
                output_dir,
                dict(failure_meta, reason="opponent_genome_missing"),
            )
            return {"fitness": -1e9, "seed_used": seed_text, "eval_ok": False, "evaluation_scope": evaluation_scope}
        opponent_genome = os.path.abspath(opponent_genome)
        if not os.path.exists(opponent_genome):
            _append_eval_failure_log(
                output_dir,
                dict(failure_meta, reason="opponent_genome_not_found", opponent_genome=opponent_genome),
            )
            return {"fitness": -1e9, "seed_used": seed_text, "eval_ok": False, "evaluation_scope": evaluation_scope}

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
            str(int(eval_runtime["games_per_genome"])),
            "--seed",
            seed_text,
            "--max-steps",
            str(int(eval_runtime["max_eval_steps"])),
            "--fitness-profile",
            str(eval_runtime["fitness_profile"]),
            "--first-turn-policy",
            "alternate" if bool(eval_runtime["switch_seats"]) else "fixed",
        ]
        if not bool(eval_runtime["switch_seats"]):
            cmd.extend(["--fixed-first-turn", "human"])
        if has_policy:
            cmd.extend(["--opponent-policy", opponent_policy])
        else:
            cmd.extend([
                "--opponent-policy-mix",
                json.dumps(opponent_policy_mix, ensure_ascii=False, separators=(",", ":")),
            ])
        if requires_genome_path:
            cmd.extend(["--opponent-genome", opponent_genome])
        if teacher_policy:
            cmd.extend(["--teacher-policy", teacher_policy])

        proc = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=max(10, int(eval_runtime["eval_timeout_sec"])),
        )
        lines = [x.strip() for x in str(proc.stdout or "").splitlines() if x.strip()]
        if not lines:
            _append_eval_failure_log(
                output_dir,
                dict(failure_meta, reason="worker_empty_stdout", stderr=str(proc.stderr or "")),
            )
            return {"fitness": -1e9, "seed_used": seed_text, "eval_ok": False, "evaluation_scope": evaluation_scope}
        summary = json.loads(lines[-1])
        if not isinstance(summary, dict):
            summary = {}
        summary["fitness"] = _safe_float(summary.get("fitness"), -1e9)
        summary["seed_used"] = seed_text
        summary["eval_ok"] = True
        summary["evaluation_scope"] = evaluation_scope
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
        return {"fitness": -1e9, "seed_used": seed_text, "eval_ok": False, "evaluation_scope": evaluation_scope}
    finally:
        try:
            if "genome_path" in locals() and genome_path and os.path.exists(genome_path):
                os.remove(genome_path)
        except Exception:
            pass


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


def _write_genome_artifacts(genome, config, pickle_path: str, json_path: str) -> None:
    with open(pickle_path, "wb") as f:
        pickle.dump(genome, f)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_export_neat_python_genome(genome, config), f, ensure_ascii=False, separators=(",", ":"))


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
# Section 5. Single Genome Evaluation Worker
# - Builds a strict node command per genome.
# - Returns fitness summary dict; failures are explicit and heavily penalized.
# =============================================================================
def eval_function(genome, config, seed_override="", generation=-1, genome_key=-1):
    # 7-1) Export genome payload and runtime options.
    # 7-2) Execute node evaluator and parse last-line JSON summary.
    runtime = _runtime_from_env_cached()
    seed_text = str(seed_override or runtime["seed"])
    payload = _export_neat_python_genome(genome, config)
    return _evaluate_genome_payload(
        payload=payload,
        eval_runtime=runtime,
        seed_text=seed_text,
        generation=int(generation),
        genome_key=int(genome_key),
        evaluation_scope="train_mix",
    )


# =============================================================================
# Section 6. Parallel Evaluator + Gate Tracking
# - Aggregates generation-level metrics from per-genome evaluation results.
# =============================================================================
class LoggedParallelEvaluator:
    def __init__(self, num_workers: int, output_dir: str, runtime: dict):
        self.num_workers = max(1, int(num_workers))
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.runtime_seed = str(runtime["seed"])
        self.pool = mp.Pool(processes=self.num_workers)

        self.eval_metrics_log = os.path.join(self.output_dir, "eval_metrics.ndjson")
        self.selection_eval_metrics_log = os.path.join(self.output_dir, "selection_eval_metrics.ndjson")
        self.generation_metrics_log = os.path.join(self.output_dir, "generation_metrics.ndjson")
        self.gate_state_path = os.path.join(self.output_dir, "gate_state.json")
        self.best_target_state_path = os.path.join(self.output_dir, "best_target_state.json")

        self.generation = -1
        self.gate_mode = str(runtime["gate_mode"])
        self.ema_window = int(runtime["gate_ema_window"])
        self.ema_alpha = 2.0 / (float(self.ema_window) + 1.0)
        self.transition_ema_win_rate = runtime.get("transition_ema_win_rate")
        self.transition_mean_gold_delta_min = runtime.get("transition_mean_gold_delta_min")
        self.transition_cvar10_gold_delta_min = runtime.get("transition_cvar10_gold_delta_min")
        self.transition_catastrophic_loss_rate_max = runtime.get("transition_catastrophic_loss_rate_max")
        self.transition_best_fitness_min = runtime.get("transition_best_fitness_min")
        self.transition_streak = int(runtime["transition_streak"])
        self.failure_generation_min = int(runtime["failure_generation_min"])
        self.failure_ema_win_rate_max = runtime.get("failure_ema_win_rate_max")
        self.failure_mean_gold_delta_max = runtime.get("failure_mean_gold_delta_max")
        self.failure_cvar10_gold_delta_max = runtime.get("failure_cvar10_gold_delta_max")
        self.failure_catastrophic_loss_rate_min = runtime.get("failure_catastrophic_loss_rate_min")
        self.failure_slope_5_max = float(runtime["failure_slope_5_max"])
        self.failure_slope_metric = str(runtime["failure_slope_metric"])
        self.selection_eval_games = int(runtime["selection_eval_games"])
        self.selection_top_k = int(runtime["selection_top_k"])
        self.selection_runtime = {
            "eval_script": str(runtime["eval_script"]),
            "games_per_genome": int(runtime["selection_eval_games"]),
            "eval_timeout_sec": int(runtime["eval_timeout_sec"]),
            "max_eval_steps": int(runtime["max_eval_steps"]),
            "opponent_policy": str(runtime["selection_opponent_policy"]),
            "opponent_policy_mix": list(runtime.get("selection_opponent_policy_mix") or []),
            "opponent_genome": str(runtime.get("selection_opponent_genome") or ""),
            "teacher_policy": str(runtime.get("teacher_policy") or ""),
            "fitness_profile": str(runtime["fitness_profile"]),
            "switch_seats": bool(runtime["switch_seats"]),
            "seed": str(runtime["seed"]),
            "output_dir": self.output_dir,
        }
        self.ema_win_rate = None
        self.gate_streak = 0
        self.transition_generation = None
        self.failure_generation = None
        self.best_win_rate_history = []
        self.gate_state = {}
        self.latest_train_best_record = None
        self.latest_train_best_genome = None
        self.latest_target_best_record = None
        self.best_target_record = None
        self.best_target_genome = None

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
            "selection_eval_games": int(self.selection_eval_games),
            "selection_top_k": int(self.selection_top_k),
            "ema_window": int(self.ema_window),
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
            "transition_cvar10_gold_delta_min": (
                float(self.transition_cvar10_gold_delta_min)
                if self.transition_cvar10_gold_delta_min is not None
                else None
            ),
            "transition_catastrophic_loss_rate_max": (
                float(self.transition_catastrophic_loss_rate_max)
                if self.transition_catastrophic_loss_rate_max is not None
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
            "failure_mean_gold_delta_max": (
                float(self.failure_mean_gold_delta_max)
                if self.failure_mean_gold_delta_max is not None
                else None
            ),
            "failure_cvar10_gold_delta_max": (
                float(self.failure_cvar10_gold_delta_max)
                if self.failure_cvar10_gold_delta_max is not None
                else None
            ),
            "failure_catastrophic_loss_rate_min": (
                float(self.failure_catastrophic_loss_rate_min)
                if self.failure_catastrophic_loss_rate_min is not None
                else None
            ),
            "failure_slope_5_max": float(self.failure_slope_5_max),
            "failure_slope_metric": self.failure_slope_metric,
        }

    def _is_valid_gate_record(self, record: dict) -> bool:
        if not bool(record.get("eval_ok")):
            return False
        required_metrics = ("win_rate", "mean_gold_delta", "cvar10_gold_delta", "catastrophic_loss_rate")
        for name in required_metrics:
            value = _safe_float(record.get(name), float("nan"))
            if value != value:
                return False
        return True

    def _record_brief(self, record: Optional[dict]) -> Optional[dict]:
        if record is None:
            return None
        return {
            "generation": int(record.get("generation", self.generation)),
            "genome_key": int(record.get("genome_key", -1)),
            "evaluation_scope": str(record.get("evaluation_scope") or ""),
            "selection_rank": (
                int(record.get("selection_rank"))
                if record.get("selection_rank") is not None
                else None
            ),
            "fitness": _safe_float(record.get("fitness"), -1e9),
            "win_rate": _safe_optional_float(record.get("win_rate")),
            "mean_gold_delta": _safe_optional_float(record.get("mean_gold_delta")),
            "p10_gold_delta": _safe_optional_float(record.get("p10_gold_delta")),
            "p50_gold_delta": _safe_optional_float(record.get("p50_gold_delta")),
            "cvar10_gold_delta": _safe_optional_float(record.get("cvar10_gold_delta")),
            "catastrophic_loss_rate": _safe_optional_float(record.get("catastrophic_loss_rate")),
            "go_games": _safe_optional_float(record.get("go_games")),
            "go_fail_rate": _safe_optional_float(record.get("go_fail_rate")),
        }

    def _target_sort_key(self, record: dict):
        return (
            _safe_float(record.get("mean_gold_delta"), -1e18),
            _safe_float(record.get("cvar10_gold_delta"), -1e18),
            -_safe_float(record.get("catastrophic_loss_rate"), 1e18),
            _safe_float(record.get("win_rate"), -1e18),
            _safe_float(record.get("fitness"), -1e18),
        )

    def _is_better_target_record(self, candidate: Optional[dict], incumbent: Optional[dict]) -> bool:
        if candidate is None:
            return False
        if incumbent is None:
            return True
        return self._target_sort_key(candidate) > self._target_sort_key(incumbent)

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
                "ema_win_rate": float(self.ema_win_rate) if self.ema_win_rate is not None else None,
                "gate_streak": int(self.gate_streak),
                "transition_ready": bool(self.transition_generation is not None),
                "transition_generation": self.transition_generation,
                "failure_triggered": bool(self.failure_generation is not None),
                "failure_generation": self.failure_generation,
                "latest_target_best": None,
                "latest_target_win_rate_slope_5": None,
                "best_target": self._record_brief(self.best_target_record),
                "thresholds": self._thresholds(),
            }
            return

        win_rate = _safe_float(best_record.get("win_rate"), 0.0)
        mean_gold_delta = _safe_float(best_record.get("mean_gold_delta"), float("nan"))
        cvar10_gold_delta = _safe_float(best_record.get("cvar10_gold_delta"), float("nan"))
        catastrophic_loss_rate = _safe_float(best_record.get("catastrophic_loss_rate"), float("nan"))
        best_fitness = _safe_float(best_record.get("fitness"), -1e9)
        self.best_win_rate_history.append(win_rate)

        self.ema_win_rate = _ema(self.ema_win_rate, win_rate, self.ema_alpha)

        transition_ok = True
        if self.transition_ema_win_rate is not None:
            transition_ok = transition_ok and (self.ema_win_rate >= self.transition_ema_win_rate)
        if self.transition_mean_gold_delta_min is not None:
            transition_ok = transition_ok and (
                mean_gold_delta == mean_gold_delta
                and mean_gold_delta >= self.transition_mean_gold_delta_min
            )
        if self.transition_cvar10_gold_delta_min is not None:
            transition_ok = transition_ok and (
                cvar10_gold_delta == cvar10_gold_delta
                and cvar10_gold_delta >= self.transition_cvar10_gold_delta_min
            )
        if self.transition_catastrophic_loss_rate_max is not None:
            transition_ok = transition_ok and (
                catastrophic_loss_rate == catastrophic_loss_rate
                and catastrophic_loss_rate <= self.transition_catastrophic_loss_rate_max
            )
        if self.transition_best_fitness_min is not None:
            transition_ok = transition_ok and (best_fitness >= self.transition_best_fitness_min)

        if transition_ok:
            self.gate_streak += 1
        else:
            self.gate_streak = 0

        if self.transition_generation is None and self.gate_streak >= self.transition_streak:
            self.transition_generation = self.generation

        win_rate_slope_5 = _five_gen_slope(self.best_win_rate_history)
        slope_5 = win_rate_slope_5
        metric_checks = []
        if self.failure_ema_win_rate_max is not None:
            metric_checks.append(self.ema_win_rate < self.failure_ema_win_rate_max)
        if self.failure_mean_gold_delta_max is not None:
            metric_checks.append(
                mean_gold_delta == mean_gold_delta
                and mean_gold_delta < self.failure_mean_gold_delta_max
            )
        if self.failure_cvar10_gold_delta_max is not None:
            metric_checks.append(
                cvar10_gold_delta == cvar10_gold_delta
                and cvar10_gold_delta < self.failure_cvar10_gold_delta_max
            )
        if self.failure_catastrophic_loss_rate_min is not None:
            metric_checks.append(
                catastrophic_loss_rate == catastrophic_loss_rate
                and catastrophic_loss_rate > self.failure_catastrophic_loss_rate_min
            )
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
            "ema_win_rate": float(self.ema_win_rate),
            "gate_streak": int(self.gate_streak),
            "transition_ready": bool(self.transition_generation is not None),
            "transition_generation": self.transition_generation,
            "failure_triggered": bool(self.failure_generation is not None),
            "failure_generation": self.failure_generation,
            "latest_target_best": self._record_brief(best_record),
            "latest_target_win_rate_slope_5": float(win_rate_slope_5),
            "best_target": self._record_brief(self.best_target_record),
            "thresholds": self._thresholds(),
        }

    def evaluate(self, genomes, config):
        self.generation += 1
        seed_for_generation = f"{self.runtime_seed}|gen={self.generation}"
        self.latest_train_best_record = None
        self.latest_train_best_genome = None
        self.latest_target_best_record = None
        jobs = []
        genome_lookup = {}
        for genome_key, genome in genomes:
            genome_lookup[int(genome_key)] = genome
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
        train_best_record = (
            max(records, key=lambda r: _safe_float(r.get("fitness"), -1e9))
            if len(records) > 0
            else None
        )
        self.latest_train_best_record = dict(train_best_record) if train_best_record is not None else None
        if train_best_record is not None:
            train_best_genome_key = int(train_best_record.get("genome_key", -1))
            if train_best_genome_key in genome_lookup:
                self.latest_train_best_genome = _copy_genome(genome_lookup[train_best_genome_key])

        selection_candidates = sorted(
            valid_records,
            key=lambda r: _safe_float(r.get("fitness"), -1e9),
            reverse=True,
        )[: self.selection_top_k]
        selection_records = []
        for rank, candidate in enumerate(selection_candidates, start=1):
            genome_key = int(candidate.get("genome_key", -1))
            genome = genome_lookup.get(genome_key)
            if genome is None:
                continue
            payload = _export_neat_python_genome(genome, config)
            selection_seed = f"{seed_for_generation}|selection|rank={rank}|genome={genome_key}"
            selection_result = _evaluate_genome_payload(
                payload=payload,
                eval_runtime=self.selection_runtime,
                seed_text=selection_seed,
                generation=int(self.generation),
                genome_key=genome_key,
                evaluation_scope="selection_target",
            )
            selection_record = dict(selection_result)
            selection_record["saved_at"] = datetime.now(timezone.utc).isoformat()
            selection_record["generation"] = int(self.generation)
            selection_record["genome_key"] = int(genome_key)
            selection_record["seed_used"] = str(selection_record.get("seed_used") or selection_seed)
            selection_record["eval_ok"] = bool(selection_record.get("eval_ok", False))
            selection_record["selection_rank"] = int(rank)
            selection_record["train_fitness"] = _safe_float(candidate.get("fitness"), -1e9)
            selection_records.append(selection_record)

        self._append_lines(self.selection_eval_metrics_log, selection_records)
        valid_selection_records = [r for r in selection_records if self._is_valid_gate_record(r)]
        latest_target_record = (
            max(valid_selection_records, key=self._target_sort_key)
            if len(valid_selection_records) > 0
            else None
        )
        self.latest_target_best_record = dict(latest_target_record) if latest_target_record is not None else None
        if latest_target_record is not None and self._is_better_target_record(latest_target_record, self.best_target_record):
            self.best_target_record = dict(latest_target_record)
            best_target_genome_key = int(latest_target_record.get("genome_key", -1))
            if best_target_genome_key in genome_lookup:
                self.best_target_genome = _copy_genome(genome_lookup[best_target_genome_key])

        best_target_state = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "generation": int(self.generation),
            "latest_train_best": self._record_brief(self.latest_train_best_record),
            "latest_target_best": self._record_brief(self.latest_target_best_record),
            "best_target": self._record_brief(self.best_target_record),
            "selection_candidate_count": int(len(selection_records)),
            "selection_valid_count": int(len(valid_selection_records)),
            "selection_eval_games": int(self.selection_eval_games),
            "selection_top_k": int(self.selection_top_k),
        }
        with open(self.best_target_state_path, "w", encoding="utf-8") as f:
            json.dump(best_target_state, f, ensure_ascii=False, indent=2)

        if records:
            fitness_values = [_safe_float(r.get("fitness"), -1e9) for r in records]
            valid_fitness_values = [_safe_float(r.get("fitness"), -1e9) for r in valid_records]
            valid_win_values = [_safe_float(r.get("win_rate"), 0.0) for r in valid_records]
            valid_go_games_values = []
            valid_go_rate_values = []
            valid_go_fail_rate_values = []
            valid_mean_gold_delta_values = []
            valid_p10_gold_delta_values = []
            valid_p50_gold_delta_values = []
            valid_cvar10_gold_delta_values = []
            valid_catastrophic_loss_rate_values = []
            valid_imitation_weighted_score_values = []
            valid_scenario_shard_weighted_score_values = []
            valid_gold_core_values = []
            valid_expected_result_values = []
            valid_bankrupt_rate_values = []
            valid_bankrupt_penalty_values = []
            for r in valid_records:
                mean_gold_delta = _safe_optional_float(r.get("mean_gold_delta"))
                if mean_gold_delta is not None:
                    valid_mean_gold_delta_values.append(mean_gold_delta)
                p10_gold_delta = _safe_optional_float(r.get("p10_gold_delta"))
                if p10_gold_delta is not None:
                    valid_p10_gold_delta_values.append(p10_gold_delta)
                p50_gold_delta = _safe_optional_float(r.get("p50_gold_delta"))
                if p50_gold_delta is not None:
                    valid_p50_gold_delta_values.append(p50_gold_delta)
                cvar10_gold_delta = _safe_optional_float(r.get("cvar10_gold_delta"))
                if cvar10_gold_delta is not None:
                    valid_cvar10_gold_delta_values.append(cvar10_gold_delta)
                catastrophic_loss_rate = _safe_optional_float(r.get("catastrophic_loss_rate"))
                if catastrophic_loss_rate is not None:
                    valid_catastrophic_loss_rate_values.append(catastrophic_loss_rate)
                imitation_weighted_score = _safe_optional_float(r.get("imitation_weighted_score"))
                if imitation_weighted_score is not None:
                    valid_imitation_weighted_score_values.append(imitation_weighted_score)
                scenario_shard_weighted_score = _safe_optional_float(r.get("scenario_shard_weighted_score"))
                if scenario_shard_weighted_score is not None:
                    valid_scenario_shard_weighted_score_values.append(scenario_shard_weighted_score)
                go_games = _safe_optional_float(r.get("go_games"))
                if go_games is not None:
                    valid_go_games_values.append(go_games)
                go_rate = _safe_optional_float(r.get("go_rate"))
                if go_rate is not None:
                    valid_go_rate_values.append(go_rate)
                go_fail_rate = _safe_optional_float(r.get("go_fail_rate"))
                if go_fail_rate is not None:
                    valid_go_fail_rate_values.append(go_fail_rate)
                fitness_components = r.get("fitness_components") or {}
                gold_core = _safe_optional_float(fitness_components.get("gold_core"))
                if gold_core is not None:
                    valid_gold_core_values.append(gold_core)
                expected_result = _safe_optional_float(fitness_components.get("expected_result"))
                if expected_result is not None:
                    valid_expected_result_values.append(expected_result)
                bankrupt_penalty = _safe_optional_float(fitness_components.get("bankrupt_penalty"))
                if bankrupt_penalty is not None:
                    valid_bankrupt_penalty_values.append(bankrupt_penalty)
                bankrupt_rate = _safe_optional_float(fitness_components.get("bankrupt_rate"))
                if bankrupt_rate is not None:
                    valid_bankrupt_rate_values.append(bankrupt_rate)
                elif isinstance(r.get("bankrupt"), dict):
                    bankrupt_count = _safe_optional_float((r.get("bankrupt") or {}).get("my_bankrupt_count"))
                    games_count = _safe_optional_float(r.get("games"))
                    if bankrupt_count is not None and games_count is not None and games_count > 0:
                        valid_bankrupt_rate_values.append(max(0.0, bankrupt_count / games_count))
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
                "best_genome_key": int(train_best_record.get("genome_key", -1)) if train_best_record is not None else -1,
                "best_fitness": _safe_float(train_best_record.get("fitness"), -1e9) if train_best_record is not None else -1e9,
                "mean_fitness": sum(fitness_values) / max(1, len(fitness_values)),
                "std_fitness": _stddev(valid_fitness_values),
                "mean_win_rate": (
                    sum(valid_win_values) / max(1, len(valid_win_values)) if len(valid_win_values) > 0 else None
                ),
                "mean_mean_gold_delta": (
                    sum(valid_mean_gold_delta_values) / max(1, len(valid_mean_gold_delta_values))
                    if len(valid_mean_gold_delta_values) > 0
                    else None
                ),
                "mean_p10_gold_delta": (
                    sum(valid_p10_gold_delta_values) / max(1, len(valid_p10_gold_delta_values))
                    if len(valid_p10_gold_delta_values) > 0
                    else None
                ),
                "mean_p50_gold_delta": (
                    sum(valid_p50_gold_delta_values) / max(1, len(valid_p50_gold_delta_values))
                    if len(valid_p50_gold_delta_values) > 0
                    else None
                ),
                "mean_cvar10_gold_delta": (
                    sum(valid_cvar10_gold_delta_values) / max(1, len(valid_cvar10_gold_delta_values))
                    if len(valid_cvar10_gold_delta_values) > 0
                    else None
                ),
                "mean_catastrophic_loss_rate": (
                    sum(valid_catastrophic_loss_rate_values) / max(1, len(valid_catastrophic_loss_rate_values))
                    if len(valid_catastrophic_loss_rate_values) > 0
                    else None
                ),
                "mean_imitation_weighted_score": (
                    sum(valid_imitation_weighted_score_values) / max(1, len(valid_imitation_weighted_score_values))
                    if len(valid_imitation_weighted_score_values) > 0
                    else None
                ),
                "mean_scenario_shard_weighted_score": (
                    sum(valid_scenario_shard_weighted_score_values) / max(1, len(valid_scenario_shard_weighted_score_values))
                    if len(valid_scenario_shard_weighted_score_values) > 0
                    else None
                ),
                "mean_gold_core": (
                    sum(valid_gold_core_values) / max(1, len(valid_gold_core_values))
                    if len(valid_gold_core_values) > 0
                    else None
                ),
                "mean_expected_result": (
                    sum(valid_expected_result_values) / max(1, len(valid_expected_result_values))
                    if len(valid_expected_result_values) > 0
                    else None
                ),
                "mean_bankrupt_rate": (
                    sum(valid_bankrupt_rate_values) / max(1, len(valid_bankrupt_rate_values))
                    if len(valid_bankrupt_rate_values) > 0
                    else None
                ),
                "mean_bankrupt_penalty": (
                    sum(valid_bankrupt_penalty_values) / max(1, len(valid_bankrupt_penalty_values))
                    if len(valid_bankrupt_penalty_values) > 0
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
                "selection_candidate_count": int(len(selection_records)),
                "selection_valid_count": int(len(valid_selection_records)),
                "selection_eval_games": int(self.selection_eval_games),
                "selection_top_k": int(self.selection_top_k),
                "latest_train_best": self._record_brief(train_best_record),
                "latest_target_best": self._record_brief(latest_target_record),
                "best_target": self._record_brief(self.best_target_record),
                "best_win_rate": (
                    _safe_float(train_best_record.get("win_rate"), 0.0)
                    if self._is_valid_gate_record(train_best_record)
                    else None
                ),
                "best_mean_gold_delta": (
                    _safe_optional_float(train_best_record.get("mean_gold_delta"))
                    if self._is_valid_gate_record(train_best_record)
                    else None
                ),
                "best_p10_gold_delta": (
                    _safe_optional_float(train_best_record.get("p10_gold_delta"))
                    if self._is_valid_gate_record(train_best_record)
                    else None
                ),
                "best_p50_gold_delta": (
                    _safe_optional_float(train_best_record.get("p50_gold_delta"))
                    if self._is_valid_gate_record(train_best_record)
                    else None
                ),
                "best_cvar10_gold_delta": (
                    _safe_optional_float(train_best_record.get("cvar10_gold_delta"))
                    if self._is_valid_gate_record(train_best_record)
                    else None
                ),
                "best_catastrophic_loss_rate": (
                    _safe_optional_float(train_best_record.get("catastrophic_loss_rate"))
                    if self._is_valid_gate_record(train_best_record)
                    else None
                ),
                "best_imitation_weighted_score": (
                    _safe_optional_float(train_best_record.get("imitation_weighted_score"))
                    if self._is_valid_gate_record(train_best_record)
                    else None
                ),
                "best_scenario_shard_weighted_score": (
                    _safe_optional_float(train_best_record.get("scenario_shard_weighted_score"))
                    if self._is_valid_gate_record(train_best_record)
                    else None
                ),
                "best_gold_core": (
                    _safe_optional_float((train_best_record.get("fitness_components") or {}).get("gold_core"))
                    if self._is_valid_gate_record(train_best_record)
                    else None
                ),
                "best_expected_result": (
                    _safe_optional_float((train_best_record.get("fitness_components") or {}).get("expected_result"))
                    if self._is_valid_gate_record(train_best_record)
                    else None
                ),
                "best_bankrupt_rate": (
                    _safe_optional_float((train_best_record.get("fitness_components") or {}).get("bankrupt_rate"))
                    if self._is_valid_gate_record(train_best_record)
                    else None
                ),
                "best_bankrupt_penalty": (
                    _safe_optional_float((train_best_record.get("fitness_components") or {}).get("bankrupt_penalty"))
                    if self._is_valid_gate_record(train_best_record)
                    else None
                ),
                "best_fitness_model": (
                    str(train_best_record.get("fitness_model") or "")
                    if self._is_valid_gate_record(train_best_record)
                    else None
                ),
                "best_fitness_profile": (
                    str(train_best_record.get("fitness_profile") or "")
                    if self._is_valid_gate_record(train_best_record)
                    else None
                ),
                "best_genome_nodes": int(train_best_record.get("num_nodes", 0)) if train_best_record is not None else 0,
                "best_genome_connections": int(train_best_record.get("num_connections", 0)) if train_best_record is not None else 0,
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
                "mean_mean_gold_delta": None,
                "mean_p10_gold_delta": None,
                "mean_p50_gold_delta": None,
                "mean_cvar10_gold_delta": None,
                "mean_catastrophic_loss_rate": None,
                "mean_imitation_weighted_score": None,
                "mean_scenario_shard_weighted_score": None,
                "mean_gold_core": None,
                "mean_expected_result": None,
                "mean_bankrupt_rate": None,
                "mean_bankrupt_penalty": None,
                "mean_go_games": None,
                "mean_go_rate": None,
                "mean_go_fail_rate": None,
                "selection_candidate_count": int(len(selection_records)),
                "selection_valid_count": int(len(valid_selection_records)),
                "selection_eval_games": int(self.selection_eval_games),
                "selection_top_k": int(self.selection_top_k),
                "latest_train_best": self._record_brief(train_best_record),
                "latest_target_best": self._record_brief(latest_target_record),
                "best_target": self._record_brief(self.best_target_record),
                "best_win_rate": None,
                "best_mean_gold_delta": None,
                "best_p10_gold_delta": None,
                "best_p50_gold_delta": None,
                "best_cvar10_gold_delta": None,
                "best_catastrophic_loss_rate": None,
                "best_imitation_weighted_score": None,
                "best_scenario_shard_weighted_score": None,
                "best_gold_core": None,
                "best_expected_result": None,
                "best_bankrupt_rate": None,
                "best_bankrupt_penalty": None,
                "best_fitness_model": None,
                "best_fitness_profile": None,
                "best_genome_nodes": 0,
                "best_genome_connections": 0,
                "mean_eval_time_ms": None,
                "p90_eval_time_ms": None,
            }
            train_best_record = None

        best_gate_record = latest_target_record
        self._update_gate(best_gate_record, len(valid_selection_records), len(selection_records))
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

    def latest_train_snapshot(self):
        return self._record_brief(self.latest_train_best_record)

    def latest_target_snapshot(self):
        return self._record_brief(self.latest_target_best_record)

    def best_target_snapshot(self):
        return self._record_brief(self.best_target_record)


# =============================================================================
# Section 7. CLI + Config Bootstrap
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="neat-python training runner for k_flower_card")
    parser.add_argument(
        "--config-feedforward",
        default=DEFAULT_FEEDFORWARD_CONFIG,
        help="Path to neat-python config file",
    )
    parser.add_argument(
        "--runtime-config",
        default=DEFAULT_RUNTIME_CONFIG,
        help="Path to runtime JSON (workers/games/checkpoint interval)",
    )
    parser.add_argument("--output-dir", default="", help="Output directory")
    parser.add_argument("--resume", default="", help="Checkpoint file path to resume")
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
# Section 8. Entrypoint
# - Loads runtime/config, prepares population, runs NEAT, writes summary artifacts.
# =============================================================================
def main() -> None:
    # 10-1) Resolve runtime config and apply CLI overrides.
    # 10-2) Run population with parallel evaluator, then export winner artifacts.
    args = parse_args()
    run_started_wall = datetime.now(timezone.utc)
    run_started_perf = time.perf_counter()

    if neat is None:
        raise RuntimeError("neat-python is not installed. Install with: pip install neat-python")

    runtime = _load_runtime_config(args.runtime_config)
    applied_overrides = {}
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
    runtime = _normalize_runtime_values(runtime)
    args.output_dir = _resolve_output_dir(args.output_dir, args.runtime_config, runtime)
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

    selection_policy = str(runtime.get("selection_opponent_policy") or "").strip()
    selection_policy_mix = runtime.get("selection_opponent_policy_mix") or []
    selection_opponent_genome = str(runtime.get("selection_opponent_genome") or "").strip()
    selection_requires_genome_path = selection_policy.lower() == "genome"
    if not selection_requires_genome_path:
        for item in selection_policy_mix:
            if isinstance(item, dict) and str(item.get("policy") or "").strip().lower() == "genome":
                selection_requires_genome_path = True
                break
    if selection_requires_genome_path:
        if not selection_opponent_genome:
            raise RuntimeError("selection_opponent_policy=genome requires selection_opponent_genome path")
        if not os.path.exists(selection_opponent_genome):
            raise RuntimeError(f"selection opponent genome not found: {selection_opponent_genome}")

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

    train_winner_pkl = os.path.join(models_dir, "winner_train_genome.pkl")
    train_winner_json_path = os.path.join(models_dir, "winner_train_genome.json")
    _write_genome_artifacts(winner, p.config, train_winner_pkl, train_winner_json_path)

    latest_train_pkl = os.path.join(models_dir, "winner_latest_train_genome.pkl")
    latest_train_json_path = os.path.join(models_dir, "winner_latest_train_genome.json")
    latest_train_genome = evaluator.latest_train_best_genome if evaluator is not None else None
    if latest_train_genome is not None:
        _write_genome_artifacts(latest_train_genome, p.config, latest_train_pkl, latest_train_json_path)
    else:
        latest_train_pkl = ""
        latest_train_json_path = ""

    canonical_winner = evaluator.best_target_genome if evaluator is not None and evaluator.best_target_genome is not None else winner
    canonical_basis = "best_vs_target" if evaluator is not None and evaluator.best_target_genome is not None else "winner_train_fallback"
    winner_pkl = os.path.join(models_dir, "winner_genome.pkl")
    winner_json_path = os.path.join(models_dir, "winner_genome.json")
    _write_genome_artifacts(canonical_winner, p.config, winner_pkl, winner_json_path)

    best_target_pkl = os.path.join(models_dir, "winner_best_target_genome.pkl")
    best_target_json_path = os.path.join(models_dir, "winner_best_target_genome.json")
    if evaluator is not None and evaluator.best_target_genome is not None:
        _write_genome_artifacts(evaluator.best_target_genome, p.config, best_target_pkl, best_target_json_path)
    else:
        best_target_pkl = ""
        best_target_json_path = ""

    train_best_fitness = float(getattr(winner, "fitness", float("nan")) or 0.0)
    best_fitness = (
        _safe_float((evaluator.best_target_record or {}).get("fitness"), train_best_fitness)
        if evaluator is not None
        else train_best_fitness
    )
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
        "train_best_fitness": train_best_fitness,
        "winner_model_basis": canonical_basis,
        "applied_overrides": applied_overrides,
        "runtime_effective": runtime,
        "eval_failure_log": os.path.join(args.output_dir, "eval_failures.log"),
        "eval_metrics_log": os.path.join(args.output_dir, "eval_metrics.ndjson"),
        "selection_eval_metrics_log": os.path.join(args.output_dir, "selection_eval_metrics.ndjson"),
        "generation_metrics_log": os.path.join(args.output_dir, "generation_metrics.ndjson"),
        "gate_state_path": os.path.join(args.output_dir, "gate_state.json"),
        "gate_state": evaluator.snapshot() if evaluator is not None else {},
        "latest_train_best": evaluator.latest_train_snapshot() if evaluator is not None else None,
        "latest_target_best": evaluator.latest_target_snapshot() if evaluator is not None else None,
        "best_target": evaluator.best_target_snapshot() if evaluator is not None else None,
        "best_target_state_path": (
            os.path.join(args.output_dir, "best_target_state.json")
            if evaluator is not None
            else ""
        ),
        "winner_pickle": winner_pkl,
        "winner_json": winner_json_path,
        "winner_train_pickle": train_winner_pkl,
        "winner_train_json": train_winner_json_path,
        "winner_latest_train_pickle": latest_train_pkl,
        "winner_latest_train_json": latest_train_json_path,
        "winner_best_target_pickle": best_target_pkl,
        "winner_best_target_json": best_target_json_path,
        "config_feedforward": os.path.abspath(args.config_feedforward),
        "runtime_config": os.path.abspath(args.runtime_config),
    }
    run_summary_path = os.path.join(args.output_dir, "run_summary.json")
    with open(run_summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
