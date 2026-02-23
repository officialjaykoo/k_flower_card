#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import math
import multiprocessing as mp
import os
import pickle
import subprocess
import tempfile
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional

try:
    import neat  # type: ignore
except Exception:
    neat = None


ENV_PREFIX = "KFC_NEAT_"
DEFAULT_RUNTIME = {
    "format_version": "neat_runtime_v1",
    "generations": 50,
    "eval_workers": 6,
    "games_per_genome": 40,
    "eval_timeout_sec": 360,
    "max_eval_steps": 600,
    "opponent_policy": "heuristic_v4",
    "switch_seats": True,
    "checkpoint_every": 50,
    "eval_script": "scripts/neat_eval_worker.mjs",
    "seed": 13,
    "fitness_gold_scale": 10000.0,
    "fitness_win_weight": 2.5,
    "fitness_loss_weight": 1.5,
    "fitness_draw_weight": 0.1,
    # Gate / transition controls (can be phase-specific via runtime config)
    "gate_mode": "win_rate_only",  # win_rate_only | hybrid
    "gate_ema_window": 5,
    "transition_ema_imitation": 0.60,
    "transition_ema_win_rate": 0.45,
    "transition_streak": 3,
    "failure_generation_min": 30,
    "failure_ema_win_rate_max": 0.30,
    "failure_imitation_max": None,
    "failure_slope_5_max": 0.005,
    "failure_slope_metric": "win_rate",  # win_rate | imitation
}


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


def _load_runtime_config(path: str) -> dict:
    cfg = dict(DEFAULT_RUNTIME)
    if path:
        _load_runtime_config_recursive(path, cfg, set())

    cfg["generations"] = max(1, _to_int(cfg.get("generations"), DEFAULT_RUNTIME["generations"]))
    cfg["eval_workers"] = max(2, _to_int(cfg.get("eval_workers"), DEFAULT_RUNTIME["eval_workers"]))
    cfg["games_per_genome"] = max(
        1, _to_int(cfg.get("games_per_genome"), DEFAULT_RUNTIME["games_per_genome"])
    )
    cfg["eval_timeout_sec"] = max(
        10, _to_int(cfg.get("eval_timeout_sec"), DEFAULT_RUNTIME["eval_timeout_sec"])
    )
    cfg["max_eval_steps"] = max(
        50, _to_int(cfg.get("max_eval_steps"), DEFAULT_RUNTIME["max_eval_steps"])
    )
    cfg["opponent_policy"] = str(cfg.get("opponent_policy") or DEFAULT_RUNTIME["opponent_policy"]).strip()
    cfg["switch_seats"] = _to_bool(cfg.get("switch_seats"), DEFAULT_RUNTIME["switch_seats"])
    cfg["checkpoint_every"] = max(
        1, _to_int(cfg.get("checkpoint_every"), DEFAULT_RUNTIME["checkpoint_every"])
    )
    cfg["eval_script"] = str(cfg.get("eval_script") or DEFAULT_RUNTIME["eval_script"]).strip()
    cfg["seed"] = _to_seed(cfg.get("seed"), DEFAULT_RUNTIME["seed"])
    cfg["fitness_gold_scale"] = max(
        1.0, _to_float(cfg.get("fitness_gold_scale"), DEFAULT_RUNTIME["fitness_gold_scale"])
    )
    cfg["fitness_win_weight"] = _to_float(
        cfg.get("fitness_win_weight"), DEFAULT_RUNTIME["fitness_win_weight"]
    )
    cfg["fitness_loss_weight"] = _to_float(
        cfg.get("fitness_loss_weight"), DEFAULT_RUNTIME["fitness_loss_weight"]
    )
    cfg["fitness_draw_weight"] = _to_float(
        cfg.get("fitness_draw_weight"), DEFAULT_RUNTIME["fitness_draw_weight"]
    )
    gate_mode = str(cfg.get("gate_mode") or DEFAULT_RUNTIME["gate_mode"]).strip().lower()
    if gate_mode not in ("win_rate_only", "hybrid"):
        gate_mode = str(DEFAULT_RUNTIME["gate_mode"])
    cfg["gate_mode"] = gate_mode
    cfg["gate_ema_window"] = max(2, _to_int(cfg.get("gate_ema_window"), DEFAULT_RUNTIME["gate_ema_window"]))
    cfg["transition_ema_imitation"] = _to_optional_float(
        cfg.get("transition_ema_imitation"), DEFAULT_RUNTIME["transition_ema_imitation"]
    )
    cfg["transition_ema_win_rate"] = _to_optional_float(
        cfg.get("transition_ema_win_rate"), DEFAULT_RUNTIME["transition_ema_win_rate"]
    )
    cfg["transition_streak"] = max(
        1, _to_int(cfg.get("transition_streak"), DEFAULT_RUNTIME["transition_streak"])
    )
    cfg["failure_generation_min"] = max(
        1, _to_int(cfg.get("failure_generation_min"), DEFAULT_RUNTIME["failure_generation_min"])
    )
    cfg["failure_ema_win_rate_max"] = _to_optional_float(
        cfg.get("failure_ema_win_rate_max"), DEFAULT_RUNTIME["failure_ema_win_rate_max"]
    )
    cfg["failure_imitation_max"] = _to_optional_float(
        cfg.get("failure_imitation_max"), DEFAULT_RUNTIME["failure_imitation_max"]
    )
    cfg["failure_slope_5_max"] = _to_float(
        cfg.get("failure_slope_5_max"), DEFAULT_RUNTIME["failure_slope_5_max"]
    )
    failure_slope_metric = str(
        cfg.get("failure_slope_metric") or DEFAULT_RUNTIME["failure_slope_metric"]
    ).strip().lower()
    if failure_slope_metric not in ("win_rate", "imitation"):
        failure_slope_metric = str(DEFAULT_RUNTIME["failure_slope_metric"])
    cfg["failure_slope_metric"] = failure_slope_metric
    return cfg


def _set_eval_env(runtime: dict, output_dir: str) -> None:
    os.environ[f"{ENV_PREFIX}EVAL_SCRIPT"] = os.path.abspath(str(runtime["eval_script"]))
    os.environ[f"{ENV_PREFIX}GAMES_PER_GENOME"] = str(int(runtime["games_per_genome"]))
    os.environ[f"{ENV_PREFIX}EVAL_TIMEOUT_SEC"] = str(int(runtime["eval_timeout_sec"]))
    os.environ[f"{ENV_PREFIX}MAX_EVAL_STEPS"] = str(int(runtime["max_eval_steps"]))
    os.environ[f"{ENV_PREFIX}OPPONENT_POLICY"] = str(runtime["opponent_policy"])
    os.environ[f"{ENV_PREFIX}SWITCH_SEATS"] = "1" if bool(runtime["switch_seats"]) else "0"
    os.environ[f"{ENV_PREFIX}SEED"] = str(runtime["seed"])
    os.environ[f"{ENV_PREFIX}OUTPUT_DIR"] = os.path.abspath(output_dir)
    os.environ[f"{ENV_PREFIX}FITNESS_GOLD_SCALE"] = str(float(runtime["fitness_gold_scale"]))
    os.environ[f"{ENV_PREFIX}FITNESS_WIN_WEIGHT"] = str(float(runtime["fitness_win_weight"]))
    os.environ[f"{ENV_PREFIX}FITNESS_LOSS_WEIGHT"] = str(float(runtime["fitness_loss_weight"]))
    os.environ[f"{ENV_PREFIX}FITNESS_DRAW_WEIGHT"] = str(float(runtime["fitness_draw_weight"]))


def _runtime_from_env() -> Dict[str, object]:
    out: Dict[str, object] = {}
    out["eval_script"] = os.environ.get(f"{ENV_PREFIX}EVAL_SCRIPT") or ""
    out["games_per_genome"] = _to_int(
        os.environ.get(f"{ENV_PREFIX}GAMES_PER_GENOME"),
        DEFAULT_RUNTIME["games_per_genome"],
    )
    out["eval_timeout_sec"] = _to_int(
        os.environ.get(f"{ENV_PREFIX}EVAL_TIMEOUT_SEC"),
        DEFAULT_RUNTIME["eval_timeout_sec"],
    )
    out["max_eval_steps"] = _to_int(
        os.environ.get(f"{ENV_PREFIX}MAX_EVAL_STEPS"),
        DEFAULT_RUNTIME["max_eval_steps"],
    )
    out["opponent_policy"] = (
        os.environ.get(f"{ENV_PREFIX}OPPONENT_POLICY") or DEFAULT_RUNTIME["opponent_policy"]
    )
    out["switch_seats"] = _to_bool(
        os.environ.get(f"{ENV_PREFIX}SWITCH_SEATS"),
        DEFAULT_RUNTIME["switch_seats"],
    )
    out["seed"] = _to_seed(os.environ.get(f"{ENV_PREFIX}SEED"), DEFAULT_RUNTIME["seed"])
    out["output_dir"] = os.environ.get(f"{ENV_PREFIX}OUTPUT_DIR") or os.getcwd()
    out["fitness_gold_scale"] = max(
        1.0,
        _to_float(
            os.environ.get(f"{ENV_PREFIX}FITNESS_GOLD_SCALE"),
            DEFAULT_RUNTIME["fitness_gold_scale"],
        ),
    )
    out["fitness_win_weight"] = _to_float(
        os.environ.get(f"{ENV_PREFIX}FITNESS_WIN_WEIGHT"),
        DEFAULT_RUNTIME["fitness_win_weight"],
    )
    out["fitness_loss_weight"] = _to_float(
        os.environ.get(f"{ENV_PREFIX}FITNESS_LOSS_WEIGHT"),
        DEFAULT_RUNTIME["fitness_loss_weight"],
    )
    out["fitness_draw_weight"] = _to_float(
        os.environ.get(f"{ENV_PREFIX}FITNESS_DRAW_WEIGHT"),
        DEFAULT_RUNTIME["fitness_draw_weight"],
    )
    return out


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


def eval_function(genome, config, seed_override="", generation=-1, genome_key=-1):
    runtime = _runtime_from_env()
    eval_script = str(runtime["eval_script"] or "")
    seed_text = str(seed_override or runtime["seed"])
    failure_meta = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "generation": int(generation),
        "genome_key": int(genome_key),
        "seed_used": seed_text,
    }
    if not eval_script or not os.path.exists(eval_script):
        _append_eval_failure_log(
            str(runtime.get("output_dir") or os.getcwd()),
            dict(failure_meta, reason="eval_script_missing", eval_script=eval_script),
        )
        return {"fitness": -1e9, "seed_used": seed_text, "eval_ok": False}

    payload = _export_neat_python_genome(genome, config)
    output_dir = str(runtime.get("output_dir") or os.getcwd())

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
            "--opponent-policy",
            str(runtime["opponent_policy"]),
            "--switch-seats",
            "1" if bool(runtime["switch_seats"]) else "0",
            "--fitness-gold-scale",
            str(float(runtime["fitness_gold_scale"])),
            "--fitness-win-weight",
            str(float(runtime["fitness_win_weight"])),
            "--fitness-loss-weight",
            str(float(runtime["fitness_loss_weight"])),
            "--fitness-draw-weight",
            str(float(runtime["fitness_draw_weight"])),
        ]

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


class LoggedParallelEvaluator:
    def __init__(self, num_workers: int, output_dir: str, runtime: dict):
        self.num_workers = max(1, int(num_workers))
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.runtime_seed = str(runtime.get("seed") or DEFAULT_RUNTIME["seed"])
        self.pool = mp.Pool(processes=self.num_workers)

        self.eval_metrics_log = os.path.join(self.output_dir, "eval_metrics.ndjson")
        self.generation_metrics_log = os.path.join(self.output_dir, "generation_metrics.ndjson")
        self.gate_state_path = os.path.join(self.output_dir, "gate_state.json")

        self.generation = -1
        self.gate_mode = str(runtime.get("gate_mode") or DEFAULT_RUNTIME["gate_mode"]).strip().lower()
        if self.gate_mode not in ("win_rate_only", "hybrid"):
            self.gate_mode = str(DEFAULT_RUNTIME["gate_mode"])
        self.ema_window = max(2, _to_int(runtime.get("gate_ema_window"), DEFAULT_RUNTIME["gate_ema_window"]))
        self.ema_alpha = 2.0 / (float(self.ema_window) + 1.0)
        self.transition_ema_imitation = _to_optional_float(
            runtime.get("transition_ema_imitation"),
            DEFAULT_RUNTIME["transition_ema_imitation"],
        )
        self.transition_ema_win_rate = _to_optional_float(
            runtime.get("transition_ema_win_rate"),
            DEFAULT_RUNTIME["transition_ema_win_rate"],
        )
        self.transition_streak = max(
            1, _to_int(runtime.get("transition_streak"), DEFAULT_RUNTIME["transition_streak"])
        )
        self.failure_generation_min = max(
            1,
            _to_int(runtime.get("failure_generation_min"), DEFAULT_RUNTIME["failure_generation_min"]),
        )
        self.failure_ema_win_rate_max = _to_optional_float(
            runtime.get("failure_ema_win_rate_max"),
            DEFAULT_RUNTIME["failure_ema_win_rate_max"],
        )
        self.failure_imitation_max = _to_optional_float(
            runtime.get("failure_imitation_max"),
            DEFAULT_RUNTIME["failure_imitation_max"],
        )
        self.failure_slope_5_max = _to_float(
            runtime.get("failure_slope_5_max"),
            DEFAULT_RUNTIME["failure_slope_5_max"],
        )
        self.failure_slope_metric = str(
            runtime.get("failure_slope_metric") or DEFAULT_RUNTIME["failure_slope_metric"]
        ).strip().lower()
        if self.failure_slope_metric not in ("win_rate", "imitation"):
            self.failure_slope_metric = str(DEFAULT_RUNTIME["failure_slope_metric"])
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
                "latest_imitation_slope_5": None,
                "latest_win_rate_slope_5": None,
                "thresholds": self._thresholds(),
            }
            return

        imitation = _safe_float(best_record.get("imitation_weighted_score"), 0.0)
        win_rate = _safe_float(best_record.get("win_rate"), 0.0)
        self.best_imitation_history.append(imitation)
        self.best_win_rate_history.append(win_rate)

        self.ema_imitation = _ema(self.ema_imitation, imitation, self.ema_alpha)
        self.ema_win_rate = _ema(self.ema_win_rate, win_rate, self.ema_alpha)

        transition_ok = True
        if self.transition_ema_win_rate is not None:
            transition_ok = transition_ok and (self.ema_win_rate >= self.transition_ema_win_rate)
        if self.gate_mode == "hybrid" and self.transition_ema_imitation is not None:
            transition_ok = transition_ok and (self.ema_imitation >= self.transition_ema_imitation)

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
            for r in valid_records:
                if "imitation_weighted_score" not in r:
                    continue
                v = _safe_float(r.get("imitation_weighted_score"), float("nan"))
                if v == v:
                    valid_imit_values.append(v)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="neat-python training runner for k_flower_card")
    parser.add_argument(
        "--config-feedforward",
        default="configs/neat_feedforward.ini",
        help="Path to neat-python config file",
    )
    parser.add_argument(
        "--runtime-config",
        default="configs/neat_runtime.json",
        help="Path to runtime JSON (workers/games/checkpoint interval)",
    )
    parser.add_argument("--output-dir", default="logs/neat_python", help="Output directory")
    parser.add_argument("--resume", default="", help="Checkpoint file path to resume")
    parser.add_argument("--generations", type=int, default=0, help="Override generations")
    parser.add_argument("--workers", type=int, default=0, help="Override worker count")
    parser.add_argument("--games-per-genome", type=int, default=0, help="Override games per genome")
    parser.add_argument("--eval-timeout-sec", type=int, default=0, help="Override evaluation timeout seconds")
    parser.add_argument("--max-eval-steps", type=int, default=0, help="Override max game steps per evaluation")
    parser.add_argument("--opponent-policy", default="", help="Override opponent policy")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="Override checkpoint interval")
    parser.add_argument("--seed", default="", help="Override runtime seed")
    parser.add_argument(
        "--fitness-gold-scale",
        type=float,
        default=0.0,
        help="Override fitness gold scale denominator",
    )
    parser.add_argument(
        "--fitness-win-weight",
        type=float,
        default=float("nan"),
        help="Override win-rate fitness weight",
    )
    parser.add_argument(
        "--fitness-loss-weight",
        type=float,
        default=float("nan"),
        help="Override loss-rate fitness penalty weight",
    )
    parser.add_argument(
        "--fitness-draw-weight",
        type=float,
        default=float("nan"),
        help="Override draw-rate fitness weight",
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


def main() -> None:
    args = parse_args()

    if neat is None:
        raise RuntimeError("neat-python is not installed. Install with: pip install neat-python")

    runtime = _load_runtime_config(args.runtime_config)
    applied_overrides = {}
    if args.generations > 0:
        runtime["generations"] = int(args.generations)
        applied_overrides["generations"] = int(args.generations)
    if args.workers > 0:
        runtime["eval_workers"] = max(2, int(args.workers))
        applied_overrides["eval_workers"] = int(runtime["eval_workers"])
    if args.games_per_genome > 0:
        runtime["games_per_genome"] = max(1, int(args.games_per_genome))
        applied_overrides["games_per_genome"] = int(runtime["games_per_genome"])
    if args.eval_timeout_sec > 0:
        runtime["eval_timeout_sec"] = max(10, int(args.eval_timeout_sec))
        applied_overrides["eval_timeout_sec"] = int(runtime["eval_timeout_sec"])
    if args.max_eval_steps > 0:
        runtime["max_eval_steps"] = max(50, int(args.max_eval_steps))
        applied_overrides["max_eval_steps"] = int(runtime["max_eval_steps"])
    if args.opponent_policy.strip():
        runtime["opponent_policy"] = args.opponent_policy.strip()
        applied_overrides["opponent_policy"] = str(runtime["opponent_policy"])
    if args.checkpoint_every > 0:
        runtime["checkpoint_every"] = max(1, int(args.checkpoint_every))
        applied_overrides["checkpoint_every"] = int(runtime["checkpoint_every"])
    if str(args.seed).strip():
        runtime["seed"] = str(args.seed).strip()
        applied_overrides["seed"] = str(runtime["seed"])
    if args.switch_seats is not None:
        runtime["switch_seats"] = bool(args.switch_seats)
        applied_overrides["switch_seats"] = bool(runtime["switch_seats"])
    if args.fitness_gold_scale > 0:
        runtime["fitness_gold_scale"] = max(1.0, float(args.fitness_gold_scale))
        applied_overrides["fitness_gold_scale"] = float(runtime["fitness_gold_scale"])
    if args.fitness_win_weight == args.fitness_win_weight:
        runtime["fitness_win_weight"] = float(args.fitness_win_weight)
        applied_overrides["fitness_win_weight"] = float(runtime["fitness_win_weight"])
    if args.fitness_loss_weight == args.fitness_loss_weight:
        runtime["fitness_loss_weight"] = float(args.fitness_loss_weight)
        applied_overrides["fitness_loss_weight"] = float(runtime["fitness_loss_weight"])
    if args.fitness_draw_weight == args.fitness_draw_weight:
        runtime["fitness_draw_weight"] = float(args.fitness_draw_weight)
        applied_overrides["fitness_draw_weight"] = float(runtime["fitness_draw_weight"])

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

    # Keep training output quiet by default to reduce terminal I/O overhead.
    if bool(args.verbose):
        p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    prefix = os.path.join(checkpoints_dir, "neat-checkpoint-")
    p.add_reporter(
        neat.Checkpointer(
            generation_interval=max(1, int(runtime["checkpoint_every"])),
            filename_prefix=prefix,
        )
    )

    evaluator = None

    def _run_population(eval_callable):
        if bool(args.verbose):
            return p.run(eval_callable, int(runtime["generations"]))
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull):
                return p.run(eval_callable, int(runtime["generations"]))

    if args.dry_run:
        winner = _run_population(_run_dry_eval)
        mode = "dry_run"
    else:
        evaluator = LoggedParallelEvaluator(
            num_workers=int(runtime["eval_workers"]),
            output_dir=args.output_dir,
            runtime=runtime,
        )
        try:
            winner = _run_population(evaluator.evaluate)
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
    run_summary = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
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
        "config_feedforward": os.path.abspath(args.config_feedforward),
        "runtime_config": os.path.abspath(args.runtime_config),
    }
    run_summary_path = os.path.join(args.output_dir, "run_summary.json")
    with open(run_summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
