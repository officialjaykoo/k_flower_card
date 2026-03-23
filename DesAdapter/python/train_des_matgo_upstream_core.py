#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import copy
from functools import cmp_to_key
import gzip
import json
import math
import os
import pickle
import queue
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
FORK_ROOT = REPO_ROOT / "Des-HyperNEAT-Python"
if str(FORK_ROOT) not in sys.path:
    sys.path.insert(0, str(FORK_ROOT))
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from deshyperneat import (
    BaseReporter,
    Checkpointer,
    Config,
    Deshyperneat,
    Developer,
    Genome,
    SearchConfig,
    StatisticsReporter,
    StdOutReporter,
    prepare_algorithm,
)
from local.matgo.upstream_core import (
    build_upstream_core_environment_description,
    build_upstream_core_genome_init_config,
    build_upstream_core_io_topology,
)
from local.matgo.ini import load_genome_config as load_des_genome_config


RUNTIME_FORMAT = "k_hyperneat_runtime_train_v1"
K_HYPERNEAT_MODEL_FORMAT = "k_hyperneat_executor_v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clamp01(x: float) -> float:
    value = float(x)
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


def _prepare_genome_for_pickle(genome: Any) -> Any:
    node_cache = getattr(genome, "_node_cppn_cache", None)
    if isinstance(node_cache, dict):
        node_cache.clear()
    elif node_cache is not None:
        setattr(genome, "_node_cppn_cache", {})

    link_cache = getattr(genome, "_link_cppn_cache", None)
    if isinstance(link_cache, dict):
        link_cache.clear()
    elif link_cache is not None:
        setattr(genome, "_link_cppn_cache", {})

    return genome


def _prepare_population_for_pickle(population: dict[int, Any]) -> dict[int, Any]:
    for genome in dict(population or {}).values():
        _prepare_genome_for_pickle(genome)
    return population


class PersistentNodeDuelWorker:
    def __init__(self, worker_id: int) -> None:
        self.worker_id = int(worker_id)
        self.process = subprocess.Popen(
            ["node", str(REPO_ROOT / "scripts" / "model_duel_server.mjs")],
            cwd=str(REPO_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def request(self, argv: list[str]) -> dict[str, Any]:
        if self.process.poll() is not None:
            raise RuntimeError(f"persistent duel worker exited early: {self.worker_id}")
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError(f"persistent duel worker pipes unavailable: {self.worker_id}")

        request_id = f"{self.worker_id}:{threading.get_ident()}:{time.time_ns()}"
        payload = json.dumps({"id": request_id, "argv": list(argv)}, ensure_ascii=False)
        self.process.stdin.write(payload + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"persistent duel worker returned no response: {self.worker_id}")
        response = json.loads(line)
        if str(response.get("id")) != request_id:
            raise RuntimeError(f"persistent duel worker response mismatch: {self.worker_id}")
        if not bool(response.get("ok", False)):
            error = response.get("error") or {}
            raise RuntimeError(str(error.get("message") or "persistent duel worker request failed"))
        return dict(response.get("summary") or {})

    def close(self) -> None:
        try:
            if self.process.stdin is not None and not self.process.stdin.closed:
                self.process.stdin.close()
        except Exception:
            pass
        try:
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=2)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass


class PersistentNodeDuelPool:
    def __init__(self, worker_count: int) -> None:
        count = max(1, int(worker_count))
        self._workers = [PersistentNodeDuelWorker(i) for i in range(count)]
        self._available: queue.Queue[PersistentNodeDuelWorker] = queue.Queue()
        for worker in self._workers:
            self._available.put(worker)

    def request(self, argv: list[str]) -> dict[str, Any]:
        worker = self._available.get()
        try:
            return worker.request(argv)
        finally:
            self._available.put(worker)

    def close(self) -> None:
        for worker in self._workers:
            worker.close()


def load_json(path_value: str) -> dict[str, Any]:
    full = (REPO_ROOT / path_value).resolve() if not os.path.isabs(path_value) else Path(path_value)
    with full.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(path_value: str) -> Path:
    value = str(path_value or "").strip()
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def normalize_runtime(raw: dict[str, Any], seed_text: str) -> dict[str, Any]:
    runtime = copy.deepcopy(raw)
    if str(runtime.get("format_version", "")).strip() != RUNTIME_FORMAT:
        raise RuntimeError(f"invalid runtime format: expected {RUNTIME_FORMAT}")
    runtime["phase"] = int(runtime.get("phase", 1) or 1)
    runtime["generations"] = int(runtime.get("generations", 10) or 10)
    runtime["checkpoint_every"] = int(runtime.get("checkpoint_every", 1) or 1)
    runtime["eval_workers"] = max(1, int(runtime.get("eval_workers", 6) or 6))
    runtime["persistent_node_worker"] = bool(runtime.get("persistent_node_worker", True))
    runtime["games_per_genome"] = int(runtime.get("games_per_genome", 8) or 8)
    runtime["max_eval_steps"] = int(runtime.get("max_eval_steps", 300) or 300)
    runtime["first_turn_policy"] = str(runtime.get("first_turn_policy", "alternate") or "alternate").strip().lower()
    runtime["continuous_series"] = bool(runtime.get("continuous_series", True))
    runtime["fitness_gold_scale"] = float(runtime.get("fitness_gold_scale", 1500.0) or 1500.0)
    runtime["fitness_gold_neutral_delta"] = float(runtime.get("fitness_gold_neutral_delta", 0.0) or 0.0)
    runtime["fitness_win_weight"] = float(runtime.get("fitness_win_weight", 0.95) or 0.95)
    runtime["fitness_gold_weight"] = float(runtime.get("fitness_gold_weight", 0.05) or 0.05)
    runtime["fitness_win_neutral_rate"] = float(runtime.get("fitness_win_neutral_rate", 0.5) or 0.5)
    runtime["winner_playoff_topk"] = max(1, int(runtime.get("winner_playoff_topk", 5) or 5))
    runtime["winner_playoff_finalists"] = max(1, int(runtime.get("winner_playoff_finalists", 2) or 2))
    runtime["winner_playoff_stage1_games"] = max(
        1, int(runtime.get("winner_playoff_stage1_games", runtime["games_per_genome"]) or runtime["games_per_genome"])
    )
    runtime["winner_playoff_stage2_games"] = max(
        1, int(runtime.get("winner_playoff_stage2_games", runtime["games_per_genome"]) or runtime["games_per_genome"])
    )
    runtime["winner_playoff_win_rate_tie_threshold"] = max(
        0.0, float(runtime.get("winner_playoff_win_rate_tie_threshold", 0.01) or 0.01)
    )
    runtime["winner_playoff_mean_gold_delta_tie_threshold"] = max(
        0.0, float(runtime.get("winner_playoff_mean_gold_delta_tie_threshold", 100.0) or 100.0)
    )
    runtime["winner_playoff_go_opp_min_count"] = max(
        0, int(runtime.get("winner_playoff_go_opp_min_count", 100) or 100)
    )
    runtime["winner_playoff_go_take_rate_tie_threshold"] = max(
        0.0, float(runtime.get("winner_playoff_go_take_rate_tie_threshold", 0.02) or 0.02)
    )
    if runtime["winner_playoff_finalists"] > runtime["winner_playoff_topk"]:
        runtime["winner_playoff_finalists"] = int(runtime["winner_playoff_topk"])
    runtime["seed"] = str(seed_text)
    early_stop_raw = list(runtime.get("early_stop_win_rate_cutoffs") or [])
    runtime["early_stop_win_rate_cutoffs"] = []
    for item in early_stop_raw:
        games = max(1, int(item.get("games", 0) or 0))
        max_win_rate = float(item.get("max_win_rate", 0.0) or 0.0)
        if games > 0:
            runtime["early_stop_win_rate_cutoffs"].append(
                {
                    "games": games,
                    "max_win_rate": max(0.0, min(1.0, max_win_rate)),
                }
            )
    runtime["early_stop_win_rate_cutoffs"].sort(key=lambda item: int(item["games"]))
    early_stop_go_take_raw = list(runtime.get("early_stop_go_take_rate_cutoffs") or [])
    runtime["early_stop_go_take_rate_cutoffs"] = []
    for item in early_stop_go_take_raw:
        games = max(1, int(item.get("games", 0) or 0))
        min_go_opportunity_count = max(0, int(item.get("min_go_opportunity_count", 0) or 0))
        min_go_take_rate = item.get("min_go_take_rate")
        max_go_take_rate = item.get("max_go_take_rate")
        normalized_item: dict[str, Any] = {
            "games": games,
            "min_go_opportunity_count": min_go_opportunity_count,
            "min_go_take_rate": None,
            "max_go_take_rate": None,
        }
        if min_go_take_rate is not None:
            normalized_item["min_go_take_rate"] = max(0.0, min(1.0, float(min_go_take_rate)))
        if max_go_take_rate is not None:
            normalized_item["max_go_take_rate"] = max(0.0, min(1.0, float(max_go_take_rate)))
        if normalized_item["min_go_take_rate"] is None and normalized_item["max_go_take_rate"] is None:
            continue
        runtime["early_stop_go_take_rate_cutoffs"].append(normalized_item)
    runtime["early_stop_go_take_rate_cutoffs"].sort(key=lambda item: int(item["games"]))
    mix = runtime.get("opponent_policy_mix") or []
    if not isinstance(mix, list):
        raise RuntimeError("opponent_policy_mix must be a list")
    runtime["opponent_policy_mix"] = [
        {
            "policy": str(item.get("policy") or "").strip(),
            "weight": float(item.get("weight") or 0.0),
        }
        for item in mix
        if str(item.get("policy") or "").strip() and float(item.get("weight") or 0.0) > 0.0
    ]
    runtime["opponent_policy"] = str(runtime.get("opponent_policy", "") or "").strip()
    if not runtime["opponent_policy"] and not runtime["opponent_policy_mix"]:
        raise RuntimeError("runtime must define opponent_policy or opponent_policy_mix")
    cppn_config = str(runtime.get("cppn_config") or "").strip()
    if not cppn_config:
        raise RuntimeError("runtime.cppn_config is required")
    runtime["cppn_config"] = str(resolve_path(cppn_config))
    runtime["genome_backend"] = str(runtime.get("genome_backend", "deshyperneat") or "deshyperneat").strip().lower()
    if runtime["genome_backend"] != "deshyperneat":
        raise RuntimeError("runtime.genome_backend must be 'deshyperneat'")
    runtime["des_hyperneat"] = normalize_des_hyperneat_runtime(runtime.get("des_hyperneat") or {})
    return runtime


def normalize_des_hyperneat_runtime(raw: dict[str, Any]) -> dict[str, Any]:
    search_default = SearchConfig()
    des_default = Config()
    raw = dict(raw) if isinstance(raw, dict) else {}
    raw_search = dict(raw.get("search") or {})
    raw_depth_overrides = dict(raw.get("depth_overrides") or {})
    raw_edge_outer_weights = dict(raw.get("edge_outer_weights") or {})
    raw_identity_mapping_edges = list(raw.get("identity_mapping_edges") or [])
    depth_overrides: dict[str, int] = {}
    for key, value in raw_depth_overrides.items():
        name = str(key or "").strip()
        if not name:
            continue
        depth_overrides[name] = max(0, int(value or 0))
    edge_outer_weights: dict[str, float] = {}
    for key, value in raw_edge_outer_weights.items():
        name = str(key or "").strip()
        if not name:
            continue
        edge_outer_weights[name] = float(value or 0.0)
    identity_mapping_edges: list[str] = []
    for item in raw_identity_mapping_edges:
        name = str(item or "").strip()
        if not name:
            continue
        identity_mapping_edges.append(name)
    return {
        "output_activation": str(raw.get("output_activation", des_default.output_activation) or des_default.output_activation).strip().lower(),
        "hidden_activation": str(raw.get("hidden_activation", des_default.hidden_activation) or des_default.hidden_activation).strip().lower(),
        "debug_prune_log": bool(raw.get("debug_prune_log", getattr(des_default, "debug_prune_log", False))),
        "static_substrate_depth": int(raw.get("static_substrate_depth", -1) or -1),
        "max_input_substrate_depth": max(0, int(raw.get("max_input_substrate_depth", 0) or 0)),
        "max_hidden_substrate_depth": max(0, int(raw.get("max_hidden_substrate_depth", 5) or 5)),
        "max_output_substrate_depth": max(0, int(raw.get("max_output_substrate_depth", 0) or 0)),
        "depth_overrides": depth_overrides,
        "edge_outer_weights": edge_outer_weights,
        "identity_mapping_edges": identity_mapping_edges,
        "search": {
            "initial_resolution": max(1, int(raw_search.get("initial_resolution", search_default.initial_resolution) or search_default.initial_resolution)),
            "max_resolution": max(1, int(raw_search.get("max_resolution", search_default.max_resolution) or search_default.max_resolution)),
            "iteration_level": max(0, int(raw_search.get("iteration_level", search_default.iteration_level) or search_default.iteration_level)),
            "division_threshold": max(0.0, float(raw_search.get("division_threshold", search_default.division_threshold) or search_default.division_threshold)),
            "variance_threshold": max(0.0, float(raw_search.get("variance_threshold", search_default.variance_threshold) or search_default.variance_threshold)),
            "band_threshold": max(0.0, float(raw_search.get("band_threshold", search_default.band_threshold) or search_default.band_threshold)),
            "max_weight": max(0.1, float(raw_search.get("max_weight", search_default.max_weight) or search_default.max_weight)),
            "weight_threshold": max(0.0, min(0.999999, float(raw_search.get("weight_threshold", search_default.weight_threshold) or search_default.weight_threshold))),
            "leo_enabled": bool(raw_search.get("leo_enabled", search_default.leo_enabled)),
            "leo_threshold": max(-1.0, min(1.0, float(raw_search.get("leo_threshold", search_default.leo_threshold) or search_default.leo_threshold))),
            "max_discoveries": max(0, int(raw_search.get("max_discoveries", search_default.max_discoveries) or search_default.max_discoveries)),
            "max_outgoing": max(0, int(raw_search.get("max_outgoing", search_default.max_outgoing) or search_default.max_outgoing)),
            "only_leaf_variance": bool(raw_search.get("only_leaf_variance", search_default.only_leaf_variance)),
            "median_variance": bool(raw_search.get("median_variance", search_default.median_variance)),
            "relative_variance": bool(raw_search.get("relative_variance", search_default.relative_variance)),
            "max_variance": bool(raw_search.get("max_variance", search_default.max_variance)),
        },
    }


def build_des_hyperneat_config(runtime: dict[str, Any]) -> Config:
    raw = dict(runtime.get("des_hyperneat") or {})
    raw_search = dict(raw.get("search") or {})
    return Config(
        search=SearchConfig(
            initial_resolution=int(raw_search["initial_resolution"]),
            max_resolution=int(raw_search["max_resolution"]),
            iteration_level=int(raw_search["iteration_level"]),
            division_threshold=float(raw_search["division_threshold"]),
            variance_threshold=float(raw_search["variance_threshold"]),
            band_threshold=float(raw_search["band_threshold"]),
            max_weight=float(raw_search["max_weight"]),
            weight_threshold=float(raw_search["weight_threshold"]),
            leo_enabled=bool(raw_search["leo_enabled"]),
            leo_threshold=float(raw_search["leo_threshold"]),
            max_discoveries=int(raw_search["max_discoveries"]),
            max_outgoing=int(raw_search["max_outgoing"]),
            only_leaf_variance=bool(raw_search["only_leaf_variance"]),
            median_variance=bool(raw_search["median_variance"]),
            relative_variance=bool(raw_search["relative_variance"]),
            max_variance=bool(raw_search["max_variance"]),
        ),
        output_activation=str(raw["output_activation"]),
        hidden_activation=str(raw["hidden_activation"]),
        debug_prune_log=bool(raw.get("debug_prune_log", False)),
    )


def _load_seed_genome(seed_genome_path: str) -> tuple[str, Any]:
    seed_path = os.path.abspath(str(seed_genome_path or "").strip())
    if not seed_path:
        raise RuntimeError("seed genome path is empty")
    if not os.path.exists(seed_path):
        raise RuntimeError(f"seed genome not found: {seed_genome_path}")
    with open(seed_path, "rb") as handle:
        seed_genome = pickle.load(handle)
    if seed_genome is None or not hasattr(seed_genome, "mutate") or not hasattr(seed_genome, "topology"):
        raise RuntimeError(f"invalid DES seed genome pickle: {seed_genome_path}")
    return seed_path, seed_genome


def _seed_source_label_from_path(seed_path: str) -> str:
    raw_path = os.path.abspath(str(seed_path or "").strip())
    if not raw_path:
        return ""
    normalized = raw_path.replace("\\", "/")
    match = re.search(r"(?:^|/)(?:des(?:_matgo)?_)?phase(\d+)_seed(\d+)(?:/|$)", normalized, re.IGNORECASE)
    if match:
        return f"phase{int(match.group(1))}_seed{int(match.group(2))}"
    return os.path.splitext(os.path.basename(raw_path))[0]


def _normalize_seed_specs(seed_specs_raw: list) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
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
        normalized.append(
            {
                "path": os.path.abspath(seed_path),
                "count": int(count),
                "genome": seed_genome,
                "source_label": _seed_source_label_from_path(seed_path),
            }
        )
    if not normalized:
        raise RuntimeError("seed genome specs are empty")
    return normalized


def _seed_population_from_specs(population, config, seed_specs: list[dict[str, Any]]):
    population_size = max(1, int(config.pop_size))
    existing_keys = sorted(population.population.keys())
    if len(existing_keys) < population_size:
        raise RuntimeError("failed to initialize base population for DES seed expansion")
    requested_total = sum(max(0, int(item["count"])) for item in seed_specs)
    if requested_total > population_size:
        raise RuntimeError(f"seed genome counts exceed population size: {requested_total} > {population_size}")

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
            raise RuntimeError("failed to allocate genome keys for DES seed expansion")
        for idx, genome_key in enumerate(block_keys):
            genome = copy.deepcopy(seed_genome)
            genome.key = int(genome_key)
            genome.fitness = None
            if idx > 0:
                genome.mutate(config.genome_config, population.state)
                if (idx % 4) == 0:
                    genome.mutate(config.genome_config, population.state)
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
    population.species.speciate(population.population, population.generation)
    population.reproduction.seed_next_key_from_population(population.population)
    return population


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


def playoff_record_compare(record_a: dict[str, Any], record_b: dict[str, Any], runtime: dict[str, Any]) -> int:
    win_tie_threshold = float(runtime["winner_playoff_win_rate_tie_threshold"])
    gold_tie_threshold = float(runtime["winner_playoff_mean_gold_delta_tie_threshold"])
    go_opp_min_count = int(runtime["winner_playoff_go_opp_min_count"])
    go_take_tie_threshold = float(runtime["winner_playoff_go_take_rate_tie_threshold"])

    win_a = float(record_a.get("win_rate", 0.0) or 0.0)
    win_b = float(record_b.get("win_rate", 0.0) or 0.0)
    if abs(win_a - win_b) > win_tie_threshold:
        return _compare_desc(win_a, win_b)

    gold_a = float(record_a.get("mean_gold_delta", -1e18) or -1e18)
    gold_b = float(record_b.get("mean_gold_delta", -1e18) or -1e18)
    if abs(gold_a - gold_b) > gold_tie_threshold:
        return _compare_desc(gold_a, gold_b)

    go_opp_a = max(0, int(record_a.get("go_opportunity_count", 0) or 0))
    go_opp_b = max(0, int(record_b.get("go_opportunity_count", 0) or 0))
    if go_opp_a >= go_opp_min_count and go_opp_b >= go_opp_min_count:
        go_take_a = float(record_a.get("go_take_rate", 0.0) or 0.0)
        go_take_b = float(record_b.get("go_take_rate", 0.0) or 0.0)
        if abs(go_take_a - go_take_b) > go_take_tie_threshold:
            return _compare_desc(go_take_a, go_take_b)

        go_count_a = max(0, int(record_a.get("go_count", 0) or 0))
        go_count_b = max(0, int(record_b.get("go_count", 0) or 0))
        if go_count_a > 0 and go_count_b > 0:
            go_fail_a = float(record_a.get("go_fail_rate", 1.0) or 1.0)
            go_fail_b = float(record_b.get("go_fail_rate", 1.0) or 1.0)
            if abs(go_fail_a - go_fail_b) > 1e-12:
                return _compare_asc(go_fail_a, go_fail_b)

    fit_a = float(record_a.get("fitness", -1e9) or -1e9)
    fit_b = float(record_b.get("fitness", -1e9) or -1e9)
    return _compare_desc(fit_a, fit_b)


def init_aggregate_summary() -> dict[str, Any]:
    return {
        "games": 0,
        "wins_a": 0.0,
        "wins_b": 0.0,
        "draws": 0.0,
        "gold_sum_a": 0.0,
        "go_count_a": 0.0,
        "go_games_a": 0.0,
        "go_fail_count_a": 0.0,
        "go_opportunity_count_a": 0.0,
        "requested_games": 0,
        "early_stop_triggered": False,
        "early_stop_reason": None,
    }


def finalize_aggregate_summary(aggregate: dict[str, Any]) -> dict[str, Any]:
    games_total = max(1.0, float(aggregate["games"]))
    go_count = float(aggregate["go_count_a"])
    go_opp_count = float(aggregate["go_opportunity_count_a"])
    return {
        "games": int(aggregate["games"]),
        "requested_games": int(aggregate["requested_games"]),
        "wins_a": float(aggregate["wins_a"]),
        "wins_b": float(aggregate["wins_b"]),
        "draws": float(aggregate["draws"]),
        "win_rate_a": float(aggregate["wins_a"]) / games_total,
        "win_rate_b": float(aggregate["wins_b"]) / games_total,
        "draw_rate": float(aggregate["draws"]) / games_total,
        "mean_gold_delta_a": float(aggregate["gold_sum_a"]) / games_total,
        "go_count_a": int(aggregate["go_count_a"]),
        "go_games_a": int(aggregate["go_games_a"]),
        "go_fail_count_a": int(aggregate["go_fail_count_a"]),
        "go_fail_rate_a": (float(aggregate["go_fail_count_a"]) / go_count) if go_count > 0.0 else 0.0,
        "go_opportunity_count_a": int(aggregate["go_opportunity_count_a"]),
        "go_take_rate_a": (go_count / go_opp_count) if go_opp_count > 0.0 else 0.0,
        "early_stop_triggered": bool(aggregate["early_stop_triggered"]),
        "early_stop_reason": aggregate["early_stop_reason"],
    }


def merge_duel_summary(aggregate: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    aggregate["games"] += int(summary.get("games", 0) or 0)
    aggregate["wins_a"] += float(summary.get("wins_a", 0) or 0)
    aggregate["wins_b"] += float(summary.get("wins_b", 0) or 0)
    aggregate["draws"] += float(summary.get("draws", 0) or 0)
    aggregate["gold_sum_a"] += float(summary.get("mean_gold_delta_a", 0.0) or 0.0) * float(summary.get("games", 0) or 0)
    aggregate["go_count_a"] += float(summary.get("go_count_a", 0) or 0)
    aggregate["go_games_a"] += float(summary.get("go_games_a", 0) or 0)
    aggregate["go_fail_count_a"] += float(summary.get("go_fail_count_a", 0) or 0)
    aggregate["go_opportunity_count_a"] += float(summary.get("go_opportunity_count_a", 0) or 0)
    return aggregate


def build_log_dir(phase: int, seed_text: str) -> Path:
    return REPO_ROOT / "DesAdapter" / "artifacts" / f"des_matgo_phase{phase}_seed{seed_text}"


def allocate_games(total_games: int, entries: list[dict[str, Any]]) -> list[int]:
    total_games = max(1, int(total_games))
    if not entries:
        return []
    total_weight = sum(float(item["weight"]) for item in entries)
    raw = [float(item["weight"]) * total_games / max(1e-9, total_weight) for item in entries]
    counts = [max(1, int(math.floor(value))) for value in raw]
    while sum(counts) > total_games:
        idx = max(range(len(counts)), key=lambda i: counts[i])
        if counts[idx] <= 1:
            break
        counts[idx] -= 1
    while sum(counts) < total_games:
        idx = max(range(len(raw)), key=lambda i: raw[i] - counts[i])
        counts[idx] += 1
    return counts


def compute_fitness_from_summary(summary: dict[str, Any], runtime: dict[str, Any]) -> tuple[float, dict[str, float]]:
    weighted_win_rate = float(summary.get("win_rate_a", 0.0) or 0.0)
    weighted_draw_rate = float(summary.get("draw_rate", 0.0) or 0.0)
    weighted_loss_rate = float(summary.get("win_rate_b", 0.0) or 0.0)
    weighted_mean_gold_delta = float(summary.get("mean_gold_delta_a", 0.0) or 0.0)

    fitness_gold_scale = float(runtime["fitness_gold_scale"])
    fitness_gold_neutral_delta = float(runtime["fitness_gold_neutral_delta"])
    fitness_win_weight = float(runtime["fitness_win_weight"])
    fitness_gold_weight = float(runtime["fitness_gold_weight"])
    fitness_win_neutral_rate = float(runtime["fitness_win_neutral_rate"])

    gold_norm = math.tanh((weighted_mean_gold_delta - fitness_gold_neutral_delta) / max(1e-9, fitness_gold_scale))
    expected_result_raw = (
        clamp01(weighted_win_rate) + (0.5 * clamp01(weighted_draw_rate)) - clamp01(weighted_loss_rate)
    )
    expected_result = max(-1.0, min(1.0, expected_result_raw))
    neutral_expected_result = (2.0 * fitness_win_neutral_rate) - 1.0
    if expected_result >= neutral_expected_result:
        result_upper_span = max(1e-9, 1.0 - neutral_expected_result)
        result_norm = clamp01((expected_result - neutral_expected_result) / result_upper_span)
    else:
        result_lower_span = max(1e-9, neutral_expected_result + 1.0)
        result_norm = -clamp01((neutral_expected_result - expected_result) / result_lower_span)

    fitness = (fitness_gold_weight * gold_norm) + (fitness_win_weight * result_norm)
    return fitness, {
        "gold_norm": gold_norm,
        "result_norm": result_norm,
        "expected_result": expected_result,
        "weighted_win_rate": weighted_win_rate,
        "weighted_draw_rate": weighted_draw_rate,
        "weighted_loss_rate": weighted_loss_rate,
        "weighted_mean_gold_delta": weighted_mean_gold_delta,
    }


class GenerationStateReporter(BaseReporter):
    def __init__(self, holder: dict[str, Any]):
        self.holder = holder

    def start_generation(self, generation):
        self.holder["generation"] = int(generation)


def _coerce_lineage_parent_tuple(raw_value: object) -> tuple[int, ...]:
    out = []
    if isinstance(raw_value, (list, tuple)):
        for item in raw_value:
            try:
                out.append(int(item))
            except Exception:
                continue
    return tuple(out)


def _load_lineage_state_from_path(path: str) -> dict[str, Any] | None:
    target = os.path.abspath(str(path or "").strip())
    if not target or not os.path.exists(target):
        return None
    try:
        raw = json.loads(Path(target).read_text(encoding="utf-8"))
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
                text = str(value or "").strip()
            except Exception:
                continue
            if text:
                out[int(key)] = text
        return out

    parents_by_key: dict[int, tuple[int, ...]] = {}
    if isinstance(raw.get("parents_by_key"), dict):
        for key, value in dict(raw.get("parents_by_key") or {}).items():
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


def _lineage_state_candidates(output_dir: Path, resume_path: str = "") -> list[str]:
    candidates = [str((Path(output_dir) / "lineage_state.json").resolve())]
    resume_raw = str(resume_path or "").strip()
    if resume_raw:
        resume_state = str((Path(resume_raw).resolve().parent.parent / "lineage_state.json").resolve())
        if os.path.abspath(resume_state) != os.path.abspath(candidates[0]):
            candidates.append(resume_state)
    return candidates


def _load_first_lineage_state(paths: list[str]) -> dict[str, Any] | None:
    for path in paths or []:
        loaded = _load_lineage_state_from_path(path)
        if loaded is not None:
            return loaded
    return None


def _build_winner_lineage_export(lineage_state: dict[str, Any] | None, winner_genome_key: int) -> dict[str, Any] | None:
    if winner_genome_key <= 0:
        return None
    state = dict(lineage_state or {})
    birth_generation_by_key = dict(state.get("birth_generation_by_key") or {})
    parents_by_key = dict(state.get("parents_by_key") or {})
    origin_by_key = dict(state.get("origin_by_key") or {})
    last_seen_generation_by_key = dict(state.get("last_seen_generation_by_key") or {})
    bootstrap_source_by_key = dict(state.get("bootstrap_source_by_key") or {})

    queue = [int(winner_genome_key)]
    visited: set[int] = set()
    nodes = []
    missing_parent_keys: set[int] = set()
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


class LineageReporter(BaseReporter):
    def __init__(self, output_dir: Path, state_seed: dict[str, Any] | None = None):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lineage_log = self.output_dir / "lineage.ndjson"
        self.lineage_state_path = self.output_dir / "lineage_state.json"
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

    def _append_lines(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        with self.lineage_log.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")

    def _write_state(self) -> None:
        payload = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "birth_generation_by_key": {str(key): int(value) for key, value in self.birth_generation_by_key.items()},
            "parents_by_key": {str(key): [int(x) for x in value] for key, value in self.parents_by_key.items()},
            "origin_by_key": {str(key): str(value or "") for key, value in self.origin_by_key.items()},
            "last_seen_generation_by_key": {
                str(key): int(value) for key, value in self.last_seen_generation_by_key.items()
            },
            "bootstrap_source_by_key": {
                str(key): str(value or "") for key, value in self.bootstrap_source_by_key.items()
            },
        }
        self.lineage_state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

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
        ancestors_map = getattr(config, "_codex_lineage_ancestors", {}) or {}
        bootstrap_sources_map = getattr(config, "_codex_lineage_bootstrap_sources", {}) or {}
        species_by_key = self._species_membership(species)
        records: list[dict[str, Any]] = []
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
                self.birth_generation_by_key[key] = int(self.current_generation)
                if len(parents) >= 2:
                    self.origin_by_key[key] = "offspring"
                elif len(parents) == 1:
                    self.origin_by_key[key] = "bootstrap_seed"
                else:
                    self.origin_by_key[key] = "init"
            self.last_seen_generation_by_key[key] = int(self.current_generation)

            eval_meta = dict(getattr(genome, "_codex_eval_meta", {}) or {})
            records.append(
                {
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "generation": int(self.current_generation),
                    "genome_key": int(key),
                    "birth_generation": int(self.birth_generation_by_key.get(key, self.current_generation)),
                    "lineage_event": (self.origin_by_key.get(key) if is_first_seen else "carryover"),
                    "origin": str(self.origin_by_key.get(key) or ""),
                    "bootstrap_source": (bootstrap_source or None),
                    "parent_keys": [int(x) for x in self.parents_by_key.get(key, tuple())],
                    "species_id": species_by_key.get(key),
                    "fitness": float(getattr(genome, "fitness", -1e9) or -1e9),
                    "win_rate": float(eval_meta.get("win_rate", 0.0) or 0.0),
                    "mean_gold_delta": float(eval_meta.get("mean_gold_delta", 0.0) or 0.0),
                    "games": int(eval_meta.get("games", 0) or 0),
                    "full_eval_passed": bool(eval_meta.get("full_eval_passed", False)),
                    "go_take_rate": float(eval_meta.get("go_take_rate", 0.0) or 0.0),
                    "go_fail_rate": float(eval_meta.get("go_fail_rate", 0.0) or 0.0),
                }
            )
        self._append_lines(records)
        self._write_state()


class QuietCheckpointer(Checkpointer):
    def save_checkpoint(self, config, population, species_set, generation):
        filename = f"{self.filename_prefix}{generation}"
        print(
            f"[K-HyperNEAT] checkpoint saved: gen={int(generation)} path={filename}",
            file=sys.stderr,
            flush=True,
        )
        reproduction = getattr(getattr(species_set, "population_state", None), "reproduction", None)
        _prepare_population_for_pickle(population)
        with gzip.open(filename, "w", compresslevel=5) as handle:
            data = (generation, config, population, species_set, reproduction, random.getstate())
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


@dataclass
class EvalRecord:
    generation: int
    genome_key: int
    fitness: float
    win_rate: float
    draw_rate: float
    loss_rate: float
    mean_gold_delta: float
    games: int
    requested_games: int
    go_count: int
    go_games: int
    go_fail_count: int
    go_fail_rate: float
    go_opportunity_count: int
    go_take_rate: float
    early_stop_triggered: bool
    early_stop_reason: str | None


class KHyperneatTrainer:
    def __init__(self, runtime: dict[str, Any], config: Any, out_dir: Path):
        self.runtime = runtime
        self.config = config
        self.out_dir = out_dir
        self.checkpoints_dir = out_dir / "checkpoints"
        self.models_dir = out_dir / "models"
        self.lineage_log_path = out_dir / "lineage.ndjson"
        self.lineage_state_path = out_dir / "lineage_state.json"
        self.winner_lineage_path = out_dir / "winner_lineage.json"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.generation_metrics_path = out_dir / "generation_metrics.ndjson"
        self.runtime_path = out_dir / "runtime_config.json"
        self.runtime_path.write_text(json.dumps(runtime, indent=2), encoding="utf-8")
        self.des_config = build_des_hyperneat_config(runtime)
        self.topology = build_upstream_core_io_topology()
        self.genome_backend = str(self.runtime.get("genome_backend", "deshyperneat") or "deshyperneat").strip().lower()
        self.generation_state: dict[str, Any] = {"generation": 0}
        self.current_generation_records: list[EvalRecord] = []
        self.all_records: list[EvalRecord] = []
        self.best_record: EvalRecord | None = None
        self.training_best_record: EvalRecord | None = None
        self.final_winner_record: EvalRecord | None = None
        self.best_genome_pickle: bytes | None = None
        self.genome_pickles: dict[int, bytes] = {}
        self.winner_playoff_summary: dict[str, Any] | None = None
        self.run_error: dict[str, Any] | None = None
        self.runtime_bundle_cache: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
        self.runtime_model_cache_lock = threading.Lock()
        self.persistent_node_worker = bool(self.runtime.get("persistent_node_worker", True))
        self.node_duel_pool: PersistentNodeDuelPool | None = None

    def artifact_paths(self, prefix: str) -> dict[str, Path]:
        stem = str(prefix or "").strip() or "winner"
        return {
            "runtime": self.models_dir / f"{stem}_runtime.json",
            "runtime_stats": self.models_dir / f"{stem}_runtime_stats.json",
            "cppn_genome": self.models_dir / f"{stem}_cppn_genome.pkl",
            "cppn_network": self.models_dir / f"{stem}_cppn_network.json",
        }

    def _record_to_dict(self, record: EvalRecord | None) -> dict[str, Any] | None:
        if record is None:
            return None
        return {
            "generation": int(record.generation),
            "genome_key": int(record.genome_key),
            "fitness": float(record.fitness),
            "win_rate": float(record.win_rate),
            "draw_rate": float(record.draw_rate),
            "loss_rate": float(record.loss_rate),
            "mean_gold_delta": float(record.mean_gold_delta),
            "games": int(record.games),
            "requested_games": int(record.requested_games),
            "go_count": int(record.go_count),
            "go_games": int(record.go_games),
            "go_fail_count": int(record.go_fail_count),
            "go_fail_rate": float(record.go_fail_rate),
            "go_opportunity_count": int(record.go_opportunity_count),
            "go_take_rate": float(record.go_take_rate),
            "early_stop_triggered": bool(record.early_stop_triggered),
            "early_stop_reason": record.early_stop_reason,
        }

    def _artifact_paths_to_dict(self, prefix: str) -> dict[str, str | None]:
        paths = self.artifact_paths(prefix)
        return {
            f"{prefix}_runtime": str(paths["runtime"].resolve()) if paths["runtime"].exists() else None,
            f"{prefix}_runtime_stats": str(paths["runtime_stats"].resolve()) if paths["runtime_stats"].exists() else None,
            f"{prefix}_cppn_network": str(paths["cppn_network"].resolve()) if paths["cppn_network"].exists() else None,
            f"{prefix}_cppn_genome": str(paths["cppn_genome"].resolve()) if paths["cppn_genome"].exists() else None,
        }

    def register_run_error(self, stage: str, exc: BaseException) -> None:
        self.run_error = {
            "stage": str(stage or "").strip() or "unknown",
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    def get_node_duel_pool(self) -> PersistentNodeDuelPool | None:
        if not self.persistent_node_worker:
            return None
        if self.node_duel_pool is None:
            self.node_duel_pool = PersistentNodeDuelPool(int(self.runtime["eval_workers"]))
        return self.node_duel_pool

    def close(self) -> None:
        if self.node_duel_pool is not None:
            self.node_duel_pool.close()
            self.node_duel_pool = None

    def _attach_eval_meta(self, genome: Any, record: EvalRecord) -> None:
        setattr(
            genome,
            "_codex_eval_meta",
            {
                "games": int(record.games),
                "requested_games": int(record.requested_games),
                "win_rate": float(record.win_rate),
                "mean_gold_delta": float(record.mean_gold_delta),
                "go_take_rate": float(record.go_take_rate),
                "go_fail_rate": float(record.go_fail_rate),
                "full_eval_passed": bool(record.games >= record.requested_games and not record.early_stop_triggered),
            },
        )

    def evaluate_genomes(self, genomes, config):
        generation = int(self.generation_state.get("generation", 0))
        genome_pairs = list(genomes)
        records: list[EvalRecord] = []
        worker_count = min(max(1, int(self.runtime["eval_workers"])), max(1, len(genome_pairs)))

        if worker_count <= 1:
            for genome_id, genome in genome_pairs:
                record = self.evaluate_genome(generation, genome_id, genome, config)
                genome.fitness = record.fitness
                self._attach_eval_meta(genome, record)
                records.append(record)
                self.all_records.append(record)
                _prepare_genome_for_pickle(genome)
                self.genome_pickles[int(genome_id)] = pickle.dumps(genome, protocol=pickle.HIGHEST_PROTOCOL)
                self._maybe_promote_best(genome_id, genome, config, record)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(self.evaluate_genome, generation, genome_id, genome, config): (genome_id, genome)
                    for genome_id, genome in genome_pairs
                }
                for future in concurrent.futures.as_completed(future_map):
                    genome_id, genome = future_map[future]
                    record = future.result()
                    genome.fitness = record.fitness
                    self._attach_eval_meta(genome, record)
                    records.append(record)
                    self.all_records.append(record)
                    _prepare_genome_for_pickle(genome)
                    self.genome_pickles[int(genome_id)] = pickle.dumps(genome, protocol=pickle.HIGHEST_PROTOCOL)
                    self._maybe_promote_best(genome_id, genome, config, record)

        self.current_generation_records = records
        self.write_generation_metrics(generation, records)

    def _maybe_promote_best(self, genome_id: int, genome: Any, config: Any, record: EvalRecord) -> None:
        if self.best_record is None or record.fitness > self.best_record.fitness:
            self.best_record = record
            self.training_best_record = record
            _prepare_genome_for_pickle(genome)
            self.best_genome_pickle = pickle.dumps(genome, protocol=pickle.HIGHEST_PROTOCOL)
            self.export_model_artifacts(genome_id, genome, config, record, prefix="training_best")

    def evaluate_genome(self, generation: int, genome_id: int, genome: Any, config: Any) -> EvalRecord:
        runtime_model = self.build_runtime_model(genome_id, genome, config)
        with tempfile.TemporaryDirectory(prefix="k_hyperneat_eval_", dir=str(self.out_dir)) as temp_dir:
            temp_path = Path(temp_dir) / f"genome_{genome_id}.json"
            temp_result = Path(temp_dir) / f"genome_{genome_id}_result.json"
            temp_path.write_text(json.dumps(runtime_model, indent=2), encoding="utf-8")
            summary = self.run_duel_eval(temp_path, temp_result)

        fitness, components = compute_fitness_from_summary(summary, self.runtime)
        return EvalRecord(
            generation=int(generation),
            genome_key=int(genome_id),
            fitness=float(fitness),
            win_rate=float(components["weighted_win_rate"]),
            draw_rate=float(components["weighted_draw_rate"]),
            loss_rate=float(components["weighted_loss_rate"]),
            mean_gold_delta=float(components["weighted_mean_gold_delta"]),
            games=int(summary.get("games", 0) or 0),
            requested_games=int(summary.get("requested_games", summary.get("games", 0)) or 0),
            go_count=int(summary.get("go_count_a", 0) or 0),
            go_games=int(summary.get("go_games_a", 0) or 0),
            go_fail_count=int(summary.get("go_fail_count_a", 0) or 0),
            go_fail_rate=float(summary.get("go_fail_rate_a", 0.0) or 0.0),
            go_opportunity_count=int(summary.get("go_opportunity_count_a", 0) or 0),
            go_take_rate=float(summary.get("go_take_rate_a", 0.0) or 0.0),
            early_stop_triggered=bool(summary.get("early_stop_triggered", False)),
            early_stop_reason=summary.get("early_stop_reason"),
        )

    def build_runtime_model(self, genome_id: int, genome: Any, config: Any) -> dict[str, Any]:
        runtime_model, _stats = self.build_runtime_bundle(genome_id, genome, config)
        return runtime_model

    def build_runtime_bundle(self, genome_id: int, genome: Any, config: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        cache_key = int(genome_id)
        with self.runtime_model_cache_lock:
            cached = self.runtime_bundle_cache.get(cache_key)
        if cached is not None:
            return cached

        developer = Developer.from_environment_description(
            build_upstream_core_environment_description(
                topology=self.topology,
                des_runtime=dict(self.runtime.get("des_hyperneat") or {}),
            ),
            config=self.des_config,
        )
        executor, stats = developer.develop(genome)
        runtime_model = executor.to_runtime_dict(self.topology, adapter_kind="matgo_minimal_v1")
        assert runtime_model["format_version"] == K_HYPERNEAT_MODEL_FORMAT
        runtime_stats = asdict(stats)
        with self.runtime_model_cache_lock:
            cached = self.runtime_bundle_cache.get(cache_key)
            if cached is None:
                self.runtime_bundle_cache[cache_key] = (runtime_model, runtime_stats)
                cached = self.runtime_bundle_cache[cache_key]
        return cached

    def run_duel_eval(self, runtime_path: Path, result_path: Path, requested_games: int | None = None, use_early_stop: bool = True) -> dict[str, Any]:
        total_games = max(1, int(requested_games or self.runtime["games_per_genome"]))
        win_rate_cutoffs = []
        go_take_rate_cutoffs = []
        if use_early_stop:
            win_rate_cutoffs = [
                dict(item)
                for item in list(self.runtime.get("early_stop_win_rate_cutoffs") or [])
                if int(item.get("games", 0) or 0) < total_games
            ]
            go_take_rate_cutoffs = [
                dict(item)
                for item in list(self.runtime.get("early_stop_go_take_rate_cutoffs") or [])
                if int(item.get("games", 0) or 0) < total_games
            ]

        aggregate = init_aggregate_summary()
        aggregate["requested_games"] = total_games
        completed_games = 0
        previous_cutoff = 0

        checkpoints = sorted(
            set([int(item["games"]) for item in win_rate_cutoffs] + [int(item["games"]) for item in go_take_rate_cutoffs] + [total_games])
        )
        for index, checkpoint_games in enumerate(checkpoints):
            segment_games = int(checkpoint_games) - int(previous_cutoff)
            previous_cutoff = int(checkpoint_games)
            if segment_games <= 0:
                continue
            segment_summary = self.run_duel_segment(
                runtime_path,
                segment_games,
                result_path.with_name(f"{result_path.stem}_segment{index}.json"),
            )
            merge_duel_summary(aggregate, segment_summary)
            completed_games = int(aggregate["games"])
            current = finalize_aggregate_summary(aggregate)
            matching_win_rate_cutoffs = [item for item in win_rate_cutoffs if int(item["games"]) == int(checkpoint_games)]
            for cutoff in matching_win_rate_cutoffs:
                if float(current["win_rate_a"]) <= float(cutoff["max_win_rate"]):
                    aggregate["early_stop_triggered"] = True
                    aggregate["early_stop_reason"] = "win_rate_cutoff"
                    break
            if bool(aggregate["early_stop_triggered"]):
                break
            matching_go_take_cutoffs = [item for item in go_take_rate_cutoffs if int(item["games"]) == int(checkpoint_games)]
            for cutoff in matching_go_take_cutoffs:
                min_go_opp = max(0, int(cutoff.get("min_go_opportunity_count", 0) or 0))
                go_opp_count = int(current.get("go_opportunity_count_a", 0) or 0)
                if go_opp_count < min_go_opp:
                    continue
                go_take_rate = float(current.get("go_take_rate_a", 0.0) or 0.0)
                min_go_take_rate = cutoff.get("min_go_take_rate")
                max_go_take_rate = cutoff.get("max_go_take_rate")
                if min_go_take_rate is not None and go_take_rate < float(min_go_take_rate):
                    aggregate["early_stop_triggered"] = True
                    aggregate["early_stop_reason"] = "go_take_rate_cutoff"
                    break
                if max_go_take_rate is not None and go_take_rate > float(max_go_take_rate):
                    aggregate["early_stop_triggered"] = True
                    aggregate["early_stop_reason"] = "go_take_rate_cutoff"
                    break
            if bool(aggregate["early_stop_triggered"]):
                break

        final_summary = finalize_aggregate_summary(aggregate)
        result_path.write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
        return final_summary

    def run_duel_segment(self, runtime_path: Path, segment_games: int, result_path: Path) -> dict[str, Any]:
        mix = list(self.runtime.get("opponent_policy_mix") or [])
        if mix:
            counts = allocate_games(int(segment_games), mix)
            aggregate = init_aggregate_summary()
            aggregate["requested_games"] = int(segment_games)
            for index, (entry, games) in enumerate(zip(mix, counts)):
                summary = self._run_single_duel(
                    runtime_path,
                    str(entry["policy"]),
                    games,
                    result_path.with_name(f"{result_path.stem}_{index}.json"),
                )
                merge_duel_summary(aggregate, summary)
            return finalize_aggregate_summary(aggregate)

        return self._run_single_duel(
            runtime_path,
            str(self.runtime.get("opponent_policy") or "").strip(),
            int(segment_games),
            result_path,
        )

    def _run_single_duel(self, runtime_path: Path, opponent_policy: str, games: int, result_path: Path) -> dict[str, Any]:
        argv = [
            "--human",
            str(runtime_path),
            "--ai",
            opponent_policy,
            "--games",
            str(max(1, int(games))),
            "--seed",
            f"{self.runtime['seed']}_{time.time_ns()}",
            "--max-steps",
            str(int(self.runtime["max_eval_steps"])),
            "--first-turn-policy",
            str(self.runtime["first_turn_policy"]),
            "--continuous-series",
            "1" if self.runtime["continuous_series"] else "2",
            "--stdout-format",
            "json",
        ]
        pool = self.get_node_duel_pool()
        if pool is not None:
            try:
                return pool.request(argv)
            except Exception:
                pass

        cmd = ["node", str(REPO_ROOT / "scripts" / "model_duel_worker.mjs"), *argv, "--result-out", str(result_path)]
        completed = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, check=True)
        stdout_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if not stdout_lines:
            raise RuntimeError("model_duel_worker produced no stdout")
        return json.loads(stdout_lines[-1])

    def write_generation_metrics(self, generation: int, records: list[EvalRecord]) -> None:
        if not records:
            return
        best = max(records, key=lambda item: item.fitness)
        payload = {
            "generation": int(generation),
            "timestamp_utc": utc_now_iso(),
            "population_size": len(records),
            "eval_workers": int(self.runtime["eval_workers"]),
            "best_genome_key": int(best.genome_key),
            "best_fitness": float(best.fitness),
            "best_win_rate": float(best.win_rate),
            "best_mean_gold_delta": float(best.mean_gold_delta),
            "full_eval_record_count": int(sum(1 for item in records if item.games >= item.requested_games and not item.early_stop_triggered)),
            "early_stop_record_count": int(sum(1 for item in records if item.early_stop_triggered)),
            "mean_fitness": float(sum(item.fitness for item in records) / max(1, len(records))),
            "mean_win_rate": float(sum(item.win_rate for item in records) / max(1, len(records))),
            "mean_gold_delta": float(sum(item.mean_gold_delta for item in records) / max(1, len(records))),
        }
        with self.generation_metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def run_winner_playoff(self, config: Any) -> dict[str, Any] | None:
        if not self.all_records:
            return None

        unique_records: list[EvalRecord] = []
        seen: set[int] = set()
        for record in sorted(self.all_records, key=lambda item: (item.fitness, item.win_rate, item.mean_gold_delta), reverse=True):
            if record.genome_key in seen:
                continue
            if record.genome_key not in self.genome_pickles:
                continue
            unique_records.append(record)
            seen.add(record.genome_key)
            if len(unique_records) >= int(self.runtime["winner_playoff_topk"]):
                break
        if not unique_records:
            return None

        stage1_entries = []
        for record in unique_records:
            genome = pickle.loads(self.genome_pickles[int(record.genome_key)])
            runtime_model = self.build_runtime_model(int(record.genome_key), genome, config)
            with tempfile.TemporaryDirectory(prefix="k_hyperneat_playoff_", dir=str(self.out_dir)) as temp_dir:
                temp_path = Path(temp_dir) / f"stage1_{record.genome_key}.json"
                temp_result = Path(temp_dir) / f"stage1_{record.genome_key}_result.json"
                temp_path.write_text(json.dumps(runtime_model, indent=2), encoding="utf-8")
                summary = self.run_duel_eval(
                    temp_path,
                    temp_result,
                    requested_games=int(self.runtime["winner_playoff_stage1_games"]),
                    use_early_stop=False,
                )
            fitness, _ = compute_fitness_from_summary(summary, self.runtime)
            stage1_record = {
                "generation": int(record.generation),
                "genome_key": int(record.genome_key),
                "fitness": float(fitness),
                "win_rate": float(summary["win_rate_a"]),
                "draw_rate": float(summary["draw_rate"]),
                "loss_rate": float(summary["win_rate_b"]),
                "mean_gold_delta": float(summary["mean_gold_delta_a"]),
                "games": int(summary["games"]),
                "go_count": int(summary.get("go_count_a", 0) or 0),
                "go_games": int(summary.get("go_games_a", 0) or 0),
                "go_fail_count": int(summary.get("go_fail_count_a", 0) or 0),
                "go_fail_rate": float(summary.get("go_fail_rate_a", 0.0) or 0.0),
                "go_opportunity_count": int(summary.get("go_opportunity_count_a", 0) or 0),
                "go_take_rate": float(summary.get("go_take_rate_a", 0.0) or 0.0),
            }
            stage1_entries.append({"training_record": record, "stage1_record": stage1_record, "genome": genome})

        stage1_entries.sort(
            key=cmp_to_key(
                lambda a, b: -playoff_record_compare(a["stage1_record"], b["stage1_record"], self.runtime)
            )
        )
        finalists = stage1_entries[: max(1, int(self.runtime["winner_playoff_finalists"]))]

        stage2_entries = []
        for entry in finalists:
            genome = entry["genome"]
            runtime_model = self.build_runtime_model(int(entry["training_record"].genome_key), genome, config)
            with tempfile.TemporaryDirectory(prefix="k_hyperneat_playoff_", dir=str(self.out_dir)) as temp_dir:
                temp_path = Path(temp_dir) / f"stage2_{entry['training_record'].genome_key}.json"
                temp_result = Path(temp_dir) / f"stage2_{entry['training_record'].genome_key}_result.json"
                temp_path.write_text(json.dumps(runtime_model, indent=2), encoding="utf-8")
                summary = self.run_duel_eval(
                    temp_path,
                    temp_result,
                    requested_games=int(self.runtime["winner_playoff_stage2_games"]),
                    use_early_stop=False,
                )
            fitness, _ = compute_fitness_from_summary(summary, self.runtime)
            stage2_record = {
                "generation": int(entry["training_record"].generation),
                "genome_key": int(entry["training_record"].genome_key),
                "fitness": float(fitness),
                "win_rate": float(summary["win_rate_a"]),
                "draw_rate": float(summary["draw_rate"]),
                "loss_rate": float(summary["win_rate_b"]),
                "mean_gold_delta": float(summary["mean_gold_delta_a"]),
                "games": int(summary["games"]),
                "go_count": int(summary.get("go_count_a", 0) or 0),
                "go_games": int(summary.get("go_games_a", 0) or 0),
                "go_fail_count": int(summary.get("go_fail_count_a", 0) or 0),
                "go_fail_rate": float(summary.get("go_fail_rate_a", 0.0) or 0.0),
                "go_opportunity_count": int(summary.get("go_opportunity_count_a", 0) or 0),
                "go_take_rate": float(summary.get("go_take_rate_a", 0.0) or 0.0),
            }
            stage2_entries.append(
                {
                    "training_record": entry["training_record"],
                    "stage1_record": entry["stage1_record"],
                    "stage2_record": stage2_record,
                    "genome": genome,
                }
            )

        stage2_entries.sort(
            key=cmp_to_key(
                lambda a, b: -playoff_record_compare(a["stage2_record"], b["stage2_record"], self.runtime)
            )
        )
        winner_entry = stage2_entries[0]
        winner_training = winner_entry["training_record"]
        winner_stage2 = winner_entry["stage2_record"]
        final_record = EvalRecord(
            generation=int(winner_stage2["generation"]),
            genome_key=int(winner_stage2["genome_key"]),
            fitness=float(winner_stage2["fitness"]),
            win_rate=float(winner_stage2["win_rate"]),
            draw_rate=float(winner_stage2["draw_rate"]),
            loss_rate=float(winner_stage2["loss_rate"]),
            mean_gold_delta=float(winner_stage2["mean_gold_delta"]),
            games=int(winner_stage2["games"]),
            requested_games=int(winner_stage2["games"]),
            go_count=int(winner_stage2["go_count"]),
            go_games=int(winner_stage2["go_games"]),
            go_fail_count=int(winner_stage2["go_fail_count"]),
            go_fail_rate=float(winner_stage2["go_fail_rate"]),
            go_opportunity_count=int(winner_stage2["go_opportunity_count"]),
            go_take_rate=float(winner_stage2["go_take_rate"]),
            early_stop_triggered=False,
            early_stop_reason=None,
        )
        self.export_model_artifacts(
            int(winner_training.genome_key),
            winner_entry["genome"],
            config,
            final_record,
            prefix="winner",
        )
        self.final_winner_record = final_record
        self.best_record = final_record
        self.winner_playoff_summary = {
            "mode": "topk_fresh_seed_playoff",
            "criteria": {
                "win_rate_tie_threshold": float(self.runtime["winner_playoff_win_rate_tie_threshold"]),
                "mean_gold_delta_tie_threshold": float(self.runtime["winner_playoff_mean_gold_delta_tie_threshold"]),
                "go_opp_min_count": int(self.runtime["winner_playoff_go_opp_min_count"]),
                "go_take_rate_tie_threshold": float(self.runtime["winner_playoff_go_take_rate_tie_threshold"]),
            },
            "stage1": [entry["stage1_record"] for entry in stage1_entries],
            "stage2": [entry["stage2_record"] for entry in stage2_entries],
            "winner_record": winner_stage2,
            "winner_training_record": {
                "generation": int(winner_training.generation),
                "genome_key": int(winner_training.genome_key),
                "fitness": float(winner_training.fitness),
                "win_rate": float(winner_training.win_rate),
                "mean_gold_delta": float(winner_training.mean_gold_delta),
                "games": int(winner_training.games),
            },
        }
        return self.winner_playoff_summary

    def export_model_artifacts(self, genome_id: int, genome: Any, config: Any, record: EvalRecord, prefix: str = "winner") -> None:
        runtime_model, runtime_stats = self.build_runtime_bundle(genome_id, genome, config)
        paths = self.artifact_paths(prefix)

        paths["runtime"].write_text(
            json.dumps(runtime_model, indent=2),
            encoding="utf-8",
        )
        paths["runtime_stats"].write_text(
            json.dumps(runtime_stats, indent=2),
            encoding="utf-8",
        )
        _prepare_genome_for_pickle(genome)
        with paths["cppn_genome"].open("wb") as handle:
            pickle.dump(genome, handle, protocol=pickle.HIGHEST_PROTOCOL)
        payload = genome.export_components()
        payload["metadata"] = {
            "generation": int(record.generation),
            "genome_id": int(genome_id),
            "fitness": float(record.fitness),
            "win_rate": float(record.win_rate),
            "mean_gold_delta": float(record.mean_gold_delta),
        }
        paths["cppn_network"].write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def promote_training_best_as_winner(self, config: Any) -> None:
        if self.training_best_record is None:
            return
        training_paths = self.artifact_paths("training_best")
        winner_paths = self.artifact_paths("winner")
        if self.best_genome_pickle is not None:
            genome = pickle.loads(self.best_genome_pickle)
            self.export_model_artifacts(
                int(self.training_best_record.genome_key),
                genome,
                config,
                self.training_best_record,
                prefix="winner",
            )
        else:
            for key, source in training_paths.items():
                target = winner_paths[key]
                if source.exists():
                    shutil.copyfile(source, target)
        self.final_winner_record = self.training_best_record
        self.best_record = self.training_best_record

    def write_run_summary(self, winner: Any | None, run_elapsed_sec: float) -> Path:
        summary_path = self.out_dir / "run_summary.json"
        winner_paths = self._artifact_paths_to_dict("winner")
        training_paths = self._artifact_paths_to_dict("training_best")
        winner_lineage = None
        if self.final_winner_record is not None and self.lineage_state_path.exists():
            lineage_state = _load_lineage_state_from_path(str(self.lineage_state_path))
            winner_lineage = _build_winner_lineage_export(lineage_state, int(self.final_winner_record.genome_key))
            if winner_lineage is not None:
                self.winner_lineage_path.write_text(
                    json.dumps(winner_lineage, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        payload = {
            "format_version": "k_hyperneat_run_summary_v1",
            "seed": self.runtime["seed"],
            "phase": int(self.runtime["phase"]),
            "completed_at_utc": utc_now_iso(),
            "run_elapsed_sec": float(run_elapsed_sec),
            "winner_present": (winner is not None) or (self.final_winner_record is not None),
            "run_error": self.run_error,
            "generation_metrics_log": str(self.generation_metrics_path.resolve()),
            "best_record": self._record_to_dict(self.best_record),
            "training_best_record": self._record_to_dict(self.training_best_record),
            "final_winner_record": self._record_to_dict(self.final_winner_record),
            "winner_playoff": self.winner_playoff_summary,
            "lineage_log": str(self.lineage_log_path.resolve()) if self.lineage_log_path.exists() else None,
            "lineage_state": str(self.lineage_state_path.resolve()) if self.lineage_state_path.exists() else None,
            "winner_lineage": str(self.winner_lineage_path.resolve()) if self.winner_lineage_path.exists() else None,
            "models": {**winner_paths, **training_paths},
        }
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return summary_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", default="DesAdapter/configs/runtime_phase1_upstream_core.json")
    parser.add_argument("--seed", required=True)
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--seed-genome",
        default="",
        help="Path to DES winner genome pickle used to reseed a fresh population",
    )
    parser.add_argument(
        "--seed-genome-count",
        type=int,
        default=0,
        help="How many genomes should be seeded from --seed-genome; 0 fills the whole population",
    )
    parser.add_argument(
        "--seed-genome-spec",
        dest="seed_genome_specs",
        action="append",
        nargs=2,
        metavar=("PATH", "COUNT"),
        default=[],
        help="May be repeated to seed multiple DES genome lineages into a fresh population",
    )
    parser.add_argument("--generations", type=int, default=0)
    parser.add_argument("--stdout-format", choices=("text", "json"), default="text")
    return parser


def main() -> None:
    started_at = time.perf_counter()
    args = build_arg_parser().parse_args()
    seed_genome_path = str(args.seed_genome or "").strip()
    seed_genome_count = max(0, int(args.seed_genome_count or 0))
    seed_genome_specs: list[dict[str, Any]] = []
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
    if seed_genome_path and seed_genome_specs:
        raise RuntimeError("--seed-genome and --seed-genome-spec are mutually exclusive")
    if args.resume and (seed_genome_path or seed_genome_specs):
        raise RuntimeError("--resume and seed genome bootstrap args are mutually exclusive")

    runtime_raw = load_json(args.runtime)
    runtime = normalize_runtime(runtime_raw, args.seed)
    if args.phase and args.phase > 0:
        runtime["phase"] = int(args.phase)
    if args.generations and args.generations > 0:
        runtime["generations"] = int(args.generations)

    out_dir = build_log_dir(int(runtime["phase"]), str(runtime["seed"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(str(runtime["seed"]))
    topology = build_upstream_core_io_topology()
    genome_config = load_des_genome_config(
        runtime["cppn_config"],
        topology=topology,
        des_runtime=dict(runtime.get("des_hyperneat") or {}),
    )
    genome_init_config = build_upstream_core_genome_init_config(
        topology=topology,
        des_runtime=dict(runtime.get("des_hyperneat") or {}),
    )
    environment_description = build_upstream_core_environment_description(
        topology=topology,
        des_runtime=dict(runtime.get("des_hyperneat") or {}),
    )
    setup = prepare_algorithm(
        ini_path=runtime["cppn_config"],
        algorithm=Deshyperneat,
        description=environment_description,
        genome_config=genome_config,
        genome_init_config=genome_init_config,
        resume=(str(resolve_path(args.resume)) if args.resume else None),
    )
    config = setup.config
    population = setup.population
    if not args.resume:
        if seed_genome_specs:
            population = _seed_population_from_specs(population, config, _normalize_seed_specs(seed_genome_specs))
        elif seed_genome_path:
            seed_count = int(seed_genome_count or 0)
            if seed_count <= 0:
                seed_count = max(1, int(config.pop_size))
            population = _seed_population_from_specs(
                population,
                config,
                _normalize_seed_specs(
                    [
                        {
                            "path": seed_genome_path,
                            "count": int(seed_count),
                        }
                    ]
                ),
            )

    lineage_state_seed = _load_first_lineage_state(_lineage_state_candidates(out_dir, args.resume))
    if lineage_state_seed is not None:
        restored_parent_map = {
            int(key): _coerce_lineage_parent_tuple(value)
            for key, value in dict(lineage_state_seed.get("parents_by_key") or {}).items()
        }
        current_parent_map = dict(getattr(population.reproduction, "ancestors", {}) or {})
        for key, value in current_parent_map.items():
            coerced = _coerce_lineage_parent_tuple(value)
            if int(key) not in restored_parent_map or len(coerced) > 0:
                restored_parent_map[int(key)] = coerced
        population.reproduction.ancestors = restored_parent_map
    setattr(config, "_codex_lineage_ancestors", population.reproduction.ancestors)
    setattr(population.config, "_codex_lineage_ancestors", population.reproduction.ancestors)
    restored_bootstrap_sources = {
        int(key): str(value or "").strip()
        for key, value in dict((lineage_state_seed or {}).get("bootstrap_source_by_key") or {}).items()
        if str(value or "").strip()
    }
    current_bootstrap_sources = {
        int(key): str(value or "").strip()
        for key, value in dict(getattr(population, "_codex_bootstrap_source_by_key", {}) or {}).items()
        if str(value or "").strip()
    }
    restored_bootstrap_sources.update(current_bootstrap_sources)
    setattr(config, "_codex_lineage_bootstrap_sources", restored_bootstrap_sources)
    setattr(population.config, "_codex_lineage_bootstrap_sources", restored_bootstrap_sources)

    trainer = KHyperneatTrainer(runtime, config, out_dir)
    population.add_reporter(GenerationStateReporter(trainer.generation_state))
    population.add_reporter(LineageReporter(out_dir, state_seed=lineage_state_seed))
    if args.stdout_format == "text":
        population.add_reporter(StdOutReporter(False))
    population.add_reporter(StatisticsReporter())
    population.add_reporter(
        QuietCheckpointer(
            generation_interval=max(1, int(runtime["checkpoint_every"])),
            filename_prefix=str((trainer.checkpoints_dir / "des-matgo-checkpoint-gen").resolve()),
        )
    )

    winner = None
    run_exception: BaseException | None = None
    try:
        winner = population.run(trainer.evaluate_genomes, int(runtime["generations"]))
        trainer.run_winner_playoff(config)
        if trainer.final_winner_record is None:
            trainer.promote_training_best_as_winner(config)
    except BaseException as exc:
        trainer.register_run_error("training_or_playoff", exc)
        trainer.promote_training_best_as_winner(config)
        run_exception = exc
    run_elapsed_sec = round(time.perf_counter() - started_at, 3)
    summary_path = trainer.write_run_summary(winner, run_elapsed_sec)
    trainer.close()

    if args.stdout_format == "json":
        print(
            json.dumps(
                {
                    "run_summary": str(summary_path.resolve()),
                    "run_error": trainer.run_error,
                    "best_record": (
                        {
                            "generation": int(trainer.best_record.generation),
                            "genome_key": int(trainer.best_record.genome_key),
                            "fitness": float(trainer.best_record.fitness),
                            "win_rate": float(trainer.best_record.win_rate),
                            "mean_gold_delta": float(trainer.best_record.mean_gold_delta),
                            "games": int(trainer.best_record.games),
                        }
                        if trainer.best_record
                        else None
                    ),
                    "run_elapsed_sec": float(run_elapsed_sec),
                },
                ensure_ascii=False,
            )
        )
    if run_exception is not None:
        raise run_exception


if __name__ == "__main__":
    main()
