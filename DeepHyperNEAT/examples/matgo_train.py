from __future__ import annotations

import argparse
import copy
import functools
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import pickle
import random
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deep_hyperneat import genome as genome_module
from deep_hyperneat.matgo_runtime import (
    MATGO_OUTPUT_COUNT,
    decode_matgo_substrate,
    evaluate_substrate,
    load_runtime_settings,
    merge_runtime_settings,
    save_runtime_model,
)
from deep_hyperneat.population import Population
from deep_hyperneat.phenomes import FeedForwardCPPN as CPPN


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "matgo_train_phase1.json"
TRAIN_CONFIG_FORMAT = "deep_hyperneat_matgo_train_v1"
CHECKPOINT_FORMAT = "deep_hyperneat_matgo_checkpoint_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("seed")
    parser.add_argument("--resume", default="")
    parser.add_argument("--bootstrap-seed-genome", default="")
    parser.add_argument("--bootstrap-seed-genome-count", type=int, default=0)
    return parser.parse_args()


def _resolve_path(path_value: str, *, config_dir: Path) -> Path:
    raw = str(path_value or "").strip()
    path = Path(raw)
    if path.is_absolute():
        return path
    candidate = (config_dir / path).resolve()
    if candidate.exists():
        return candidate
    return (REPO_ROOT / path).resolve()


def _resolve_optional_path(path_value: str, *, config_dir: Path) -> Path | None:
    raw = str(path_value or "").strip()
    if not raw:
        return None
    return _resolve_path(raw, config_dir=config_dir)


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _runtime_opponent_summary(runtime_settings: dict[str, object]) -> dict[str, object]:
    opponent_policy = str(runtime_settings.get("opponent_policy") or "").strip()
    opponent_policy_mix = []
    for item in list(runtime_settings.get("opponent_policy_mix") or []):
        if not isinstance(item, dict):
            continue
        policy = str(item.get("policy") or "").strip()
        weight = _safe_float(item.get("weight"), 0.0)
        if not policy or weight <= 0.0:
            continue
        opponent_policy_mix.append({"policy": policy, "weight": weight})
    return {
        "opponent_policy": (opponent_policy or None),
        "opponent_policy_mix": opponent_policy_mix,
    }


def _append_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    if not records:
        return
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def _coerce_lineage_parent_tuple(raw_value: object) -> tuple[int, ...]:
    out: list[int] = []
    if isinstance(raw_value, (list, tuple)):
        for item in raw_value:
            try:
                out.append(int(item))
            except Exception:
                continue
    return tuple(out)


def _species_membership(species_set) -> dict[int, int]:
    out: dict[int, int] = {}
    species_map = getattr(species_set, "species", {}) or {}
    for species_id, species in species_map.items():
        for genome_key in (getattr(species, "members", {}) or {}).keys():
            try:
                out[int(genome_key)] = int(species_id)
            except Exception:
                continue
    return out


def _collect_lineage_state(population: Population) -> dict[str, object]:
    reproduction = getattr(population, "reproduction", None)
    ancestors = dict(getattr(reproduction, "ancestors", {}) or {})
    birth_generation = dict(getattr(reproduction, "birth_generation", {}) or {})
    origin_by_key = dict(getattr(reproduction, "origin_by_key", {}) or {})
    last_seen_generation = dict(getattr(population, "_codex_lineage_last_seen_generation_by_key", {}) or {})
    bootstrap_source_by_key = dict(getattr(population, "_codex_bootstrap_source_by_key", {}) or {})
    return {
        "birth_generation_by_key": {
            int(key): int(value) for key, value in birth_generation.items()
        },
        "parents_by_key": {
            int(key): _coerce_lineage_parent_tuple(value) for key, value in ancestors.items()
        },
        "origin_by_key": {
            int(key): str(value or "").strip() for key, value in origin_by_key.items()
        },
        "last_seen_generation_by_key": {
            int(key): int(value) for key, value in last_seen_generation.items()
        },
        "bootstrap_source_by_key": {
            int(key): str(value or "").strip() for key, value in bootstrap_source_by_key.items()
        },
    }


def _write_lineage_state(path: Path, population: Population) -> dict[str, object]:
    state = _collect_lineage_state(population)
    payload = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "birth_generation_by_key": {
            str(key): int(value) for key, value in dict(state["birth_generation_by_key"]).items()
        },
        "parents_by_key": {
            str(key): [int(x) for x in value] for key, value in dict(state["parents_by_key"]).items()
        },
        "origin_by_key": {
            str(key): str(value or "") for key, value in dict(state["origin_by_key"]).items()
        },
        "last_seen_generation_by_key": {
            str(key): int(value) for key, value in dict(state["last_seen_generation_by_key"]).items()
        },
        "bootstrap_source_by_key": {
            str(key): str(value or "") for key, value in dict(state["bootstrap_source_by_key"]).items()
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return state


def _append_lineage_records(path: Path, state_path: Path, population: Population, generation: int) -> dict[str, object]:
    display_generation = int(generation)
    reproduction = population.reproduction
    ancestors = dict(getattr(reproduction, "ancestors", {}) or {})
    birth_generation = dict(getattr(reproduction, "birth_generation", {}) or {})
    origin_by_key = dict(getattr(reproduction, "origin_by_key", {}) or {})
    last_seen_generation = dict(getattr(population, "_codex_lineage_last_seen_generation_by_key", {}) or {})
    bootstrap_source_by_key = dict(getattr(population, "_codex_bootstrap_source_by_key", {}) or {})
    species_by_key = _species_membership(population.species)
    records: list[dict[str, object]] = []

    for genome_key, genome in (population.population or {}).items():
        key = int(genome_key)
        parents = _coerce_lineage_parent_tuple(ancestors.get(key))
        if key not in birth_generation:
            birth_generation[key] = display_generation
        origin = str(origin_by_key.get(key) or "").strip()
        if not origin:
            origin = "offspring" if parents else "init"
            origin_by_key[key] = origin

        bootstrap_source = str(bootstrap_source_by_key.get(key) or "").strip()
        if (not bootstrap_source) and parents:
            inherited_sources = sorted(
                {
                    str(bootstrap_source_by_key.get(int(parent_key)) or "").strip()
                    for parent_key in parents
                    if str(bootstrap_source_by_key.get(int(parent_key)) or "").strip()
                }
            )
            if len(inherited_sources) == 1:
                bootstrap_source = inherited_sources[0]
            elif len(inherited_sources) >= 2:
                bootstrap_source = ",".join(inherited_sources)
        if bootstrap_source:
            bootstrap_source_by_key[key] = bootstrap_source

        is_first_seen = key not in last_seen_generation
        last_seen_generation[key] = display_generation
        eval_meta = dict(getattr(genome, "_codex_eval_meta", {}) or {})
        records.append(
            {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "generation": int(display_generation),
                "genome_key": int(key),
                "birth_generation": int(birth_generation.get(key, display_generation)),
                "lineage_event": (origin if is_first_seen else "carryover"),
                "origin": origin,
                "bootstrap_source": (bootstrap_source or None),
                "parent_keys": [int(x) for x in parents],
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

    reproduction.birth_generation = birth_generation
    reproduction.origin_by_key = origin_by_key
    population._codex_lineage_last_seen_generation_by_key = last_seen_generation
    population._codex_bootstrap_source_by_key = bootstrap_source_by_key
    _append_jsonl(path, records)
    return _write_lineage_state(state_path, population)


def _build_winner_lineage_export(lineage_state: dict[str, object], winner_genome_key: int) -> dict[str, object] | None:
    if int(winner_genome_key) <= 0:
        return None
    birth_generation_by_key = dict(lineage_state.get("birth_generation_by_key") or {})
    parents_by_key = dict(lineage_state.get("parents_by_key") or {})
    origin_by_key = dict(lineage_state.get("origin_by_key") or {})
    last_seen_generation_by_key = dict(lineage_state.get("last_seen_generation_by_key") or {})
    bootstrap_source_by_key = dict(lineage_state.get("bootstrap_source_by_key") or {})

    queue = [int(winner_genome_key)]
    visited: set[int] = set()
    nodes: list[dict[str, object]] = []
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


def _checkpoint_path(checkpoints_dir: Path, generation: int) -> Path:
    return checkpoints_dir / f"matgo-checkpoint-gen{int(generation)}.pkl.gz"


def _save_training_checkpoint(
    path: Path,
    population: Population,
    task_state: dict[str, object],
    runtime_settings: dict[str, object],
) -> Path:
    payload = {
        "format_version": CHECKPOINT_FORMAT,
        "generation": int(getattr(population, "current_gen", 0)),
        "population": population,
        "task_state": task_state,
        "random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "runtime_settings": dict(runtime_settings),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def _load_training_checkpoint(path: Path) -> dict[str, object]:
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or str(payload.get("format_version") or "").strip() != CHECKPOINT_FORMAT:
        raise RuntimeError(f"invalid checkpoint payload: {path}")
    random_state = payload.get("random_state")
    if random_state is not None:
        random.setstate(random_state)
    numpy_random_state = payload.get("numpy_random_state")
    if numpy_random_state is not None:
        np.random.set_state(numpy_random_state)
    return payload


def _training_candidate_sort_key(record: dict[str, object]) -> tuple[float, float, float]:
    return (
        _safe_float(record.get("fitness"), -1e9),
        _safe_float(record.get("win_rate"), -1.0),
        _safe_float(record.get("mean_gold_delta"), -1e18),
    )


def _playoff_record_sort_key(record: dict[str, object]) -> tuple[float, float, float]:
    return (
        _safe_float(record.get("win_rate"), -1.0),
        _safe_float(record.get("mean_gold_delta"), -1e18),
        _safe_float(record.get("fitness"), -1e9),
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


def _playoff_record_compare(record_a: dict[str, object], record_b: dict[str, object], runtime: dict[str, object]) -> int:
    win_tie_threshold = max(
        0.0, _safe_float(runtime.get("winner_playoff_win_rate_tie_threshold"), 0.01)
    )
    gold_tie_threshold = max(
        0.0, _safe_float(runtime.get("winner_playoff_mean_gold_delta_tie_threshold"), 100.0)
    )
    go_opp_min_count = max(0, int(_safe_float(runtime.get("winner_playoff_go_opp_min_count"), 100.0)))
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


def _serialize_playoff_entry(training_record: dict[str, object], playoff_record: dict[str, object]) -> dict[str, object]:
    return {
        "genome_key": int(_safe_float(training_record.get("genome_key"), -1)),
        "generation": int(_safe_float(training_record.get("generation"), -1)),
        "training_fitness": _safe_float(training_record.get("fitness"), -1e9),
        "training_win_rate": _safe_float(training_record.get("win_rate"), 0.0),
        "training_mean_gold_delta": _safe_float(training_record.get("mean_gold_delta"), 0.0),
        "playoff_fitness": _safe_float(playoff_record.get("fitness"), -1e9),
        "playoff_win_rate": _safe_float(playoff_record.get("win_rate"), 0.0),
        "playoff_mean_gold_delta": _safe_float(playoff_record.get("mean_gold_delta"), 0.0),
        "playoff_go_take_rate": _safe_float(playoff_record.get("go_take_rate"), 0.0),
        "playoff_go_fail_rate": _safe_float(playoff_record.get("go_fail_rate"), 0.0),
        "playoff_games": int(_safe_float(playoff_record.get("games"), 0.0)),
        "playoff_early_stop_triggered": bool(playoff_record.get("early_stop_triggered", False)),
        "seed_used": playoff_record.get("seed_used"),
    }


def _load_seed_genome(path: Path):
    if not path.exists():
        raise RuntimeError(f"bootstrap seed genome not found: {path}")
    with path.open("rb") as handle:
        genome = pickle.load(handle)
    if genome is None or not hasattr(genome, "nodes") or not hasattr(genome, "connections"):
        raise RuntimeError(f"invalid bootstrap seed genome pickle: {path}")
    return genome


def _seed_population(
    population: Population,
    seed_genome,
    seed_count: int,
    *,
    bootstrap_source: str = "",
) -> Population:
    existing_keys = sorted(int(key) for key in population.population.keys())
    if not existing_keys:
        return population

    population_size = len(existing_keys)
    use_seed_count = int(seed_count or 0)
    if use_seed_count <= 0:
        use_seed_count = population_size
    use_seed_count = max(1, min(population_size, use_seed_count))

    seeded_population = dict(population.population)
    ancestors = dict(getattr(population.reproduction, "ancestors", {}) or {})
    birth_generation = dict(getattr(population.reproduction, "birth_generation", {}) or {})
    origin_by_key = dict(getattr(population.reproduction, "origin_by_key", {}) or {})
    bootstrap_source_by_key = dict(getattr(population, "_codex_bootstrap_source_by_key", {}) or {})
    seed_parent_key = int(getattr(seed_genome, "key", 0) or 0)
    for idx, genome_key in enumerate(existing_keys[:use_seed_count]):
        genome = copy.deepcopy(seed_genome)
        genome.key = int(genome_key)
        genome.fitness = None
        if idx > 0:
            genome.mutate()
            if (idx % 4) == 0:
                genome.mutate()
        seeded_population[int(genome_key)] = genome
        ancestors[int(genome_key)] = ((seed_parent_key,) if seed_parent_key > 0 else tuple())
        birth_generation[int(genome_key)] = 0
        origin_by_key[int(genome_key)] = "bootstrap_seed"
        if bootstrap_source:
            bootstrap_source_by_key[int(genome_key)] = str(bootstrap_source)

    for genome_key in existing_keys[use_seed_count:]:
        genome = seeded_population.get(int(genome_key))
        if genome is not None:
            genome.fitness = None
        ancestors.setdefault(int(genome_key), tuple())
        birth_generation.setdefault(int(genome_key), 0)
        origin_by_key.setdefault(int(genome_key), "init")

    population.population = seeded_population
    population.best_genome = None
    population.species.species = {}
    population.species.genome_to_species = {}
    population.reproduction.ancestors = ancestors
    population.reproduction.birth_generation = birth_generation
    population.reproduction.origin_by_key = origin_by_key
    population._codex_bootstrap_source_by_key = bootstrap_source_by_key
    population._codex_lineage_last_seen_generation_by_key = {}
    population.species.speciate(population.population, 0)
    return population


def _evaluate_genome(
    genome,
    runtime_settings: dict[str, object],
    *,
    games: int,
    seed: str,
    sub_input_dims: list[int],
    sub_hidden_dims: list[int],
    sub_output_dims: int,
):
    cppn = CPPN.create(genome)
    hidden_node_count = max(1, int(sub_hidden_dims[1] if len(sub_hidden_dims) > 1 else sub_hidden_dims[0]))
    substrate = decode_matgo_substrate(cppn, hidden_node_count=hidden_node_count)
    return evaluate_substrate(
        substrate,
        REPO_ROOT,
        runtime_settings,
        games=games,
        seed=seed,
    )


def _build_eval_record(
    generation: int,
    genome_key: int,
    genome,
    fitness: float,
    summary: dict[str, object],
    requested_games: int,
    seed_used: str,
) -> dict[str, object]:
    conn_genes = getattr(genome, "connections", {}) or {}
    enabled_connections = 0
    for conn_gene in conn_genes.values():
        if bool(getattr(conn_gene, "enabled", True)):
            enabled_connections += 1
    games_played = max(0, int(_safe_float(summary.get("games"), 0.0)))
    return {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "generation": int(generation),
        "genome_key": int(genome_key),
        "seed_used": str(seed_used),
        "fitness": float(fitness),
        "games": int(games_played),
        "requested_games": int(requested_games),
        "eval_ok": True,
        "full_eval_passed": bool(games_played >= int(requested_games)),
        "win_rate": _safe_float(summary.get("win_rate_a"), 0.0),
        "loss_rate": _safe_float(summary.get("win_rate_b"), 0.0),
        "draw_rate": _safe_float(summary.get("draw_rate"), 0.0),
        "mean_gold_delta": _safe_float(summary.get("mean_gold_delta_a"), 0.0),
        "go_count": max(0, int(_safe_float(summary.get("go_count_a"), 0.0))),
        "go_games": max(0, int(_safe_float(summary.get("go_games_a"), 0.0))),
        "go_fail_rate": _safe_float(summary.get("go_fail_rate_a"), 0.0),
        "go_opportunity_count": max(0, int(_safe_float(summary.get("go_opportunity_count_a"), 0.0))),
        "go_take_rate": _safe_float(summary.get("go_take_rate_a"), 0.0),
        "early_stop_triggered": bool(summary.get("early_stop_triggered", False)),
        "early_stop_reason": summary.get("early_stop_reason"),
        "num_nodes": int(len(getattr(genome, "nodes", {}) or {})),
        "num_connections": int(enabled_connections),
        "num_connections_total": int(len(conn_genes)),
    }


def _evaluate_candidate_entry(
    entry: dict[str, object],
    runtime_settings: dict[str, object],
    *,
    games: int,
    seed: str,
    sub_input_dims: list[int],
    sub_hidden_dims: list[int],
    sub_output_dims: int,
) -> dict[str, object] | None:
    genome = entry.get("genome")
    if genome is None:
        return None
    training_record = dict(entry.get("record") or {})
    fitness, summary, meta = _evaluate_genome(
        genome,
        runtime_settings,
        games=int(games),
        seed=str(seed),
        sub_input_dims=sub_input_dims,
        sub_hidden_dims=sub_hidden_dims,
        sub_output_dims=sub_output_dims,
    )
    record = _build_eval_record(
        int(_safe_float(training_record.get("generation"), -1)),
        int(_safe_float(training_record.get("genome_key"), getattr(genome, "key", -1))),
        genome,
        float(fitness),
        summary,
        int(games),
        str(seed),
    )
    return {
        "training_record": training_record,
        "record": record,
        "genome": copy.deepcopy(genome),
        "summary": dict(summary or {}),
        "meta": dict(meta or {}),
    }


def _run_winner_playoff(
    candidate_entries: list[dict[str, object]],
    runtime_settings: dict[str, object],
    *,
    seed_tag: str,
    sub_input_dims: list[int],
    sub_hidden_dims: list[int],
    sub_output_dims: int,
) -> dict[str, object] | None:
    if not candidate_entries:
        return None

    stage1_topk = max(1, int(runtime_settings.get("winner_playoff_topk", 5) or 5))
    stage2_topk = max(1, int(runtime_settings.get("winner_playoff_finalists", 2) or 2))
    stage1_games = max(
        1,
        int(runtime_settings.get("winner_playoff_stage1_games", runtime_settings.get("games_per_genome", 1)) or 1),
    )
    stage2_games = max(
        1,
        int(runtime_settings.get("winner_playoff_stage2_games", runtime_settings.get("games_per_genome", 1)) or 1),
    )
    stage1_seed = f"{seed_tag}|winner_playoff_stage1"
    stage2_seed = f"{seed_tag}|winner_playoff_stage2"

    stage1_results: list[dict[str, object]] = []
    for entry in list(candidate_entries[:stage1_topk]):
        result = _evaluate_candidate_entry(
            entry,
            runtime_settings,
            games=stage1_games,
            seed=stage1_seed,
            sub_input_dims=sub_input_dims,
            sub_hidden_dims=sub_hidden_dims,
            sub_output_dims=sub_output_dims,
        )
        if result is not None:
            stage1_results.append(result)

    if not stage1_results:
        return None

    stage1_results.sort(
        key=functools.cmp_to_key(
            lambda a, b: -_playoff_record_compare(
                dict(a.get("record") or {}),
                dict(b.get("record") or {}),
                runtime_settings,
            )
        )
    )

    stage2_results: list[dict[str, object]] = []
    for entry in list(stage1_results[:stage2_topk]):
        result = _evaluate_candidate_entry(
            {
                "record": dict(entry.get("training_record") or {}),
                "genome": entry.get("genome"),
            },
            runtime_settings,
            games=stage2_games,
            seed=stage2_seed,
            sub_input_dims=sub_input_dims,
            sub_hidden_dims=sub_hidden_dims,
            sub_output_dims=sub_output_dims,
        )
        if result is None:
            continue
        result["stage1_record"] = dict(entry.get("record") or {})
        stage2_results.append(result)

    if not stage2_results:
        return None

    stage2_results.sort(
        key=functools.cmp_to_key(
            lambda a, b: -_playoff_record_compare(
                dict(a.get("record") or {}),
                dict(b.get("record") or {}),
                runtime_settings,
            )
        )
    )

    winner_entry = stage2_results[0]
    return {
        "winner_genome": copy.deepcopy(winner_entry.get("genome")),
        "winner_record": dict(winner_entry.get("record") or {}),
        "winner_summary": dict(winner_entry.get("summary") or {}),
        "winner_meta": dict(winner_entry.get("meta") or {}),
        "winner_training_record": dict(winner_entry.get("training_record") or {}),
        "winner_stage1_record": dict(winner_entry.get("stage1_record") or {}),
        "winner_stage2_record": dict(winner_entry.get("record") or {}),
        "summary": {
            "mode": "topk_fresh_seed_playoff",
            "criteria": "stage1 top-K by training fitness, then playoff ranking: win_rate, mean_gold_delta, go_take_rate, go_fail_rate, fitness; repeat for stage2",
            "thresholds": {
                "win_rate_tie_threshold": float(runtime_settings.get("winner_playoff_win_rate_tie_threshold", 0.01)),
                "mean_gold_delta_tie_threshold": float(
                    runtime_settings.get("winner_playoff_mean_gold_delta_tie_threshold", 100.0)
                ),
                "go_opp_min_count": int(runtime_settings.get("winner_playoff_go_opp_min_count", 100)),
                "go_take_rate_tie_threshold": float(
                    runtime_settings.get("winner_playoff_go_take_rate_tie_threshold", 0.02)
                ),
            },
            "stage1": {
                "topk": int(stage1_topk),
                "games": int(stage1_games),
                "seed": stage1_seed,
                "results": [
                    _serialize_playoff_entry(
                        dict(entry.get("training_record") or {}),
                        dict(entry.get("record") or {}),
                    )
                    for entry in stage1_results
                ],
            },
            "stage2": {
                "topk": int(stage2_topk),
                "games": int(stage2_games),
                "seed": stage2_seed,
                "results": [
                    _serialize_playoff_entry(
                        dict(entry.get("training_record") or {}),
                        dict(entry.get("record") or {}),
                    )
                    for entry in stage2_results
                ],
            },
        },
    }


def load_train_config(path_value: str | Path) -> dict[str, object]:
    config_path = Path(path_value).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if str(payload.get("format_version") or "").strip() != TRAIN_CONFIG_FORMAT:
        raise RuntimeError(f"invalid train config format: expected {TRAIN_CONFIG_FORMAT}")

    config_dir = config_path.parent
    runtime_path = _resolve_path(str(payload.get("runtime") or ""), config_dir=config_dir)
    out_path = _resolve_path(str(payload.get("out") or ""), config_dir=config_dir)
    runtime_override = dict(payload.get("runtime_override") or {})
    runtime_override["opponent_policy"] = str(payload.get("opponent_policy") or "").strip()
    runtime_override["opponent_policy_mix"] = list(payload.get("opponent_policy_mix") or [])
    return {
        "config_path": config_path,
        "runtime": runtime_path,
        "games_per_genome": max(1, int(payload.get("games_per_genome", 8) or 8)),
        "generations": max(1, int(payload.get("generations", 5) or 5)),
        "population_size": max(2, int(payload.get("population_size", 24) or 24)),
        "elitism": max(1, int(payload.get("elitism", 2) or 2)),
        "checkpoint_every": max(1, int(payload.get("checkpoint_every", 1) or 1)),
        "goal_fitness": float(payload.get("goal_fitness", 0.55) or 0.55),
        "sheet_width": max(1, int(payload.get("sheet_width", 26) or 26)),
        "species_threshold": float(payload.get("species_threshold", 3.5) or 3.5),
        "species_reproduction_threshold": float(payload.get("species_reproduction_threshold", 0.2) or 0.2),
        "candidate_pool_size": max(1, int(payload.get("candidate_pool_size", 20) or 20)),
        "mutation": dict(payload.get("mutation") or {}),
        "opponent_policy": str(payload.get("opponent_policy") or "").strip(),
        "opponent_policy_mix": list(payload.get("opponent_policy_mix") or []),
        "runtime_override": runtime_override,
        "seed_tag": str(payload.get("seed_tag") or "deep_hyperneat_matgo").strip() or "deep_hyperneat_matgo",
        "out": out_path,
    }


def _seed_to_int(seed_text: str) -> int:
    digest = hashlib.sha256(str(seed_text).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def bootstrap_train_run(
    seed_text: str,
    train_config: dict[str, object],
    runtime_settings: dict[str, object],
    *,
    bootstrap_seed_genome: Path | None,
    bootstrap_seed_genome_count: int,
    resume_path: Path | None,
) -> dict[str, object]:
    seed_label = str(seed_text or "").strip()
    if not seed_label:
        raise RuntimeError("seed is required")

    seed_value = _seed_to_int(seed_label)
    random.seed(seed_value)
    np.random.seed(seed_value % (2**32))

    runtime_settings["seed"] = seed_label

    base_out = Path(train_config["out"]).resolve()
    run_dir = base_out.parent / f"seed_{seed_label}"
    run_dir.mkdir(parents=True, exist_ok=True)

    base_name = base_out.stem
    suffix = base_out.suffix or ".json"
    train_config["seed_tag"] = seed_label
    train_config["run_dir"] = run_dir
    train_config["out"] = run_dir / f"{base_name}{suffix}"
    train_config["winner_genome_out"] = run_dir / "winner_genome.pkl"
    train_config["winner_eval_out"] = run_dir / "winner_eval.json"
    train_config["winner_playoff_out"] = run_dir / "winner_playoff.json"
    train_config["run_summary_out"] = run_dir / "run_summary.json"
    train_config["eval_metrics_out"] = run_dir / "eval_metrics.ndjson"
    train_config["generation_metrics_out"] = run_dir / "generation_metrics.ndjson"
    train_config["lineage_out"] = run_dir / "lineage.ndjson"
    train_config["lineage_state_out"] = run_dir / "lineage_state.json"
    train_config["winner_lineage_out"] = run_dir / "winner_lineage.json"
    train_config["checkpoints_dir"] = run_dir / "checkpoints"
    Path(train_config["checkpoints_dir"]).mkdir(parents=True, exist_ok=True)

    effective_config = {
        "format_version": TRAIN_CONFIG_FORMAT,
        "seed": seed_label,
        "runtime": str(Path(train_config["runtime"]).resolve()),
        "games_per_genome": int(train_config["games_per_genome"]),
        "generations": int(train_config["generations"]),
        "population_size": int(train_config["population_size"]),
        "elitism": int(train_config["elitism"]),
        "checkpoint_every": int(train_config["checkpoint_every"]),
        "goal_fitness": float(train_config["goal_fitness"]),
        "sheet_width": int(train_config["sheet_width"]),
        "species_threshold": float(train_config["species_threshold"]),
        "species_reproduction_threshold": float(train_config["species_reproduction_threshold"]),
        "candidate_pool_size": int(train_config["candidate_pool_size"]),
        "mutation": dict(train_config["mutation"]),
        "opponent_policy": str(train_config["opponent_policy"]),
        "opponent_policy_mix": list(train_config["opponent_policy_mix"]),
        "runtime_override": dict(train_config["runtime_override"]),
        "runtime_settings": dict(runtime_settings),
        "seed_tag": str(train_config["seed_tag"]),
        "out": str(Path(train_config["out"]).resolve()),
        "bootstrap_seed_genome": (
            str(Path(bootstrap_seed_genome).resolve())
            if bootstrap_seed_genome is not None
            else None
        ),
        "bootstrap_seed_genome_count": int(bootstrap_seed_genome_count),
        "resume_path": (str(Path(resume_path).resolve()) if resume_path is not None else None),
    }
    (run_dir / "effective_train_config.json").write_text(
        json.dumps(effective_config, indent=2),
        encoding="utf-8",
    )
    return train_config


def main() -> None:
    args = parse_args()
    train_config = load_train_config(DEFAULT_CONFIG_PATH)
    resume_path = _resolve_optional_path(
        str(args.resume or ""),
        config_dir=Path(DEFAULT_CONFIG_PATH).resolve().parent,
    )
    if resume_path is not None and str(args.bootstrap_seed_genome or "").strip():
        raise RuntimeError("--resume and --bootstrap-seed-genome are mutually exclusive")
    runtime_settings = merge_runtime_settings(
        load_runtime_settings(train_config["runtime"]),
        train_config["runtime_override"],
    )
    runtime_settings["games_per_genome"] = int(train_config["games_per_genome"])
    bootstrap_seed_genome = _resolve_optional_path(
        str(args.bootstrap_seed_genome or ""),
        config_dir=Path(DEFAULT_CONFIG_PATH).resolve().parent,
    )
    bootstrap_seed_genome_count = max(0, int(args.bootstrap_seed_genome_count or 0))
    train_config = bootstrap_train_run(
        str(args.seed),
        train_config,
        runtime_settings,
        bootstrap_seed_genome=bootstrap_seed_genome,
        bootstrap_seed_genome_count=bootstrap_seed_genome_count,
        resume_path=resume_path,
    )
    genome_module.configure_mutation_parameters(train_config["mutation"])

    run_dir = Path(train_config["run_dir"])
    eval_metrics_path = Path(train_config["eval_metrics_out"])
    generation_metrics_path = Path(train_config["generation_metrics_out"])
    lineage_path = Path(train_config["lineage_out"])
    lineage_state_path = Path(train_config["lineage_state_out"])
    checkpoints_dir = Path(train_config["checkpoints_dir"])

    sub_hidden_dims = [1, int(train_config["sheet_width"])]
    sub_output_dims = MATGO_OUTPUT_COUNT

    candidate_limit = max(1, int(train_config["candidate_pool_size"]))
    task_state: dict[str, object]
    if resume_path is not None:
        checkpoint_payload = _load_training_checkpoint(Path(resume_path))
        population = checkpoint_payload.get("population")
        if population is None or not isinstance(population, Population):
            raise RuntimeError(f"invalid checkpoint population payload: {resume_path}")
        task_state = dict(checkpoint_payload.get("task_state") or {})
        task_state.setdefault("generation", int(getattr(population, "current_gen", 0)))
        task_state.setdefault("candidate_entries", [])
        setattr(population, "_codex_bootstrap_source_by_key", dict(getattr(population, "_codex_bootstrap_source_by_key", {}) or {}))
        setattr(
            population,
            "_codex_lineage_last_seen_generation_by_key",
            dict(getattr(population, "_codex_lineage_last_seen_generation_by_key", {}) or {}),
        )
        _write_lineage_state(lineage_state_path, population)
    else:
        population = Population(
            0,
            int(train_config["population_size"]),
            int(train_config["elitism"]),
            species_threshold=float(train_config["species_threshold"]),
            species_reproduction_threshold=float(train_config["species_reproduction_threshold"]),
        )
        task_state = {
            "generation": 0,
            "candidate_entries": [],
        }
        setattr(population, "_codex_bootstrap_source_by_key", {})
        setattr(population, "_codex_lineage_last_seen_generation_by_key", {})
        if bootstrap_seed_genome is not None:
            seed_genome = _load_seed_genome(Path(bootstrap_seed_genome))
            population = _seed_population(
                population,
                seed_genome,
                int(bootstrap_seed_genome_count),
                bootstrap_source=str(Path(bootstrap_seed_genome).resolve()),
            )
        _write_lineage_state(lineage_state_path, population)

    checkpoint_state: dict[str, object] = {
        "last_saved_generation": None,
        "last_saved_path": None,
    }

    def matgo_task(genomes) -> None:
        generation = int(task_state["generation"])
        records: list[dict[str, object]] = []
        full_eval_records: list[dict[str, object]] = []
        for genome_key, genome in genomes:
            eval_seed = f"{train_config['seed_tag']}|gen={generation}|genome={genome_key}"
            fitness, summary, _meta = _evaluate_genome(
                genome,
                runtime_settings,
                games=int(train_config["games_per_genome"]),
                seed=eval_seed,
                sub_input_dims=[],
                sub_hidden_dims=sub_hidden_dims,
                sub_output_dims=sub_output_dims,
            )
            genome.fitness = float(fitness)
            record = _build_eval_record(
                generation,
                int(genome_key),
                genome,
                float(fitness),
                summary,
                int(train_config["games_per_genome"]),
                eval_seed,
            )
            setattr(
                genome,
                "_codex_eval_meta",
                {
                    "fitness": float(fitness),
                    "win_rate": _safe_float(record.get("win_rate"), 0.0),
                    "mean_gold_delta": _safe_float(record.get("mean_gold_delta"), 0.0),
                    "games": int(record.get("games", 0) or 0),
                    "full_eval_passed": bool(record.get("full_eval_passed", False)),
                    "go_take_rate": _safe_optional_float(record.get("go_take_rate")),
                    "go_fail_rate": _safe_optional_float(record.get("go_fail_rate")),
                },
            )
            records.append(record)
            if bool(record["full_eval_passed"]):
                full_eval_records.append(record)
                candidate_entries = list(task_state["candidate_entries"])
                candidate_entries.append(
                    {
                        "record": dict(record),
                        "genome": copy.deepcopy(genome),
                    }
                )
                candidate_entries.sort(
                    key=lambda entry: _training_candidate_sort_key(dict(entry.get("record") or {})),
                    reverse=True,
                )
                task_state["candidate_entries"] = candidate_entries[:candidate_limit]

        _append_jsonl(eval_metrics_path, records)

        best_record = None
        if full_eval_records:
            best_record = max(full_eval_records, key=lambda item: _safe_float(item.get("fitness"), -1e9))
        elif records:
            best_record = max(records, key=lambda item: _safe_float(item.get("fitness"), -1e9))

        fitness_values = [_safe_float(item.get("fitness"), -1e9) for item in records]
        generation_record = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "generation": generation,
            "population_size": len(records),
            "full_eval_record_count": len(full_eval_records),
            "early_stop_record_count": max(0, len(records) - len(full_eval_records)),
            "best_genome_key": int(_safe_float((best_record or {}).get("genome_key"), -1)),
            "best_fitness": _safe_float((best_record or {}).get("fitness"), -1e9),
            "best_win_rate": _safe_float((best_record or {}).get("win_rate"), 0.0) if best_record else None,
            "best_mean_gold_delta": (
                _safe_float((best_record or {}).get("mean_gold_delta"), 0.0) if best_record else None
            ),
            "mean_fitness": (sum(fitness_values) / len(fitness_values)) if fitness_values else -1e9,
        }
        _append_jsonl(generation_metrics_path, [generation_record])
        _append_lineage_records(lineage_path, lineage_state_path, population, generation)
        task_state["generation"] = generation + 1

    def checkpoint_generation(population_obj: Population, reached_goal: bool) -> None:
        current_generation = int(getattr(population_obj, "current_gen", 0))
        checkpoint_every = max(1, int(train_config["checkpoint_every"]))
        if current_generation <= 0:
            return
        if (current_generation % checkpoint_every) != 0 and not reached_goal:
            return
        if checkpoint_state.get("last_saved_generation") == current_generation:
            return
        checkpoint_path = _save_training_checkpoint(
            _checkpoint_path(checkpoints_dir, current_generation),
            population_obj,
            task_state,
            runtime_settings,
        )
        checkpoint_state["last_saved_generation"] = current_generation
        checkpoint_state["last_saved_path"] = checkpoint_path
        print(f"Saving checkpoint to {checkpoint_path}")

    start_summary = {
        "mode": ("resume" if resume_path is not None else "train"),
        "seed": str(args.seed),
        "run_dir": str(run_dir.resolve()),
        "games_per_genome": int(train_config["games_per_genome"]),
        "generations": int(train_config["generations"]),
        "population_size": int(train_config["population_size"]),
        "checkpoint_every": int(train_config["checkpoint_every"]),
    }
    start_summary.update(_runtime_opponent_summary(runtime_settings))
    if resume_path is not None:
        start_summary["resume_path"] = str(Path(resume_path).resolve())
    print(json.dumps(start_summary, ensure_ascii=False))

    winner_genome = population.run(
        matgo_task,
        float(train_config["goal_fitness"]),
        int(train_config["generations"]),
        generation_callback=checkpoint_generation,
        reset_generation=(resume_path is None),
        emit_builtin_reports=False,
    )

    if int(getattr(population, "current_gen", 0)) > 0:
        checkpoint_generation(population, True)

    playoff_runtime_settings = dict(runtime_settings)
    playoff_runtime_settings["early_stop_win_rate_cutoffs"] = []
    playoff_runtime_settings["early_stop_go_take_rate_cutoffs"] = []
    candidate_entries = list(task_state["candidate_entries"])
    candidate_entries.sort(
        key=lambda entry: _training_candidate_sort_key(dict(entry.get("record") or {})),
        reverse=True,
    )

    winner_playoff = _run_winner_playoff(
        candidate_entries,
        playoff_runtime_settings,
        seed_tag=str(train_config["seed_tag"]),
        sub_input_dims=[],
        sub_hidden_dims=sub_hidden_dims,
        sub_output_dims=sub_output_dims,
    )

    selected_winner = winner_genome
    if winner_playoff is not None and winner_playoff.get("winner_genome") is not None:
        selected_winner = winner_playoff["winner_genome"]
    elif candidate_entries:
        selected_winner = candidate_entries[0]["genome"]

    final_eval_games = int(train_config["games_per_genome"])
    if winner_playoff is not None:
        final_eval_games = max(
            final_eval_games,
            int(playoff_runtime_settings.get("winner_playoff_stage2_games", final_eval_games) or final_eval_games),
        )

    winner_fitness, winner_summary, winner_meta = _evaluate_genome(
        selected_winner,
        playoff_runtime_settings,
        games=final_eval_games,
        seed=f"{train_config['seed_tag']}|winner",
        sub_input_dims=[],
        sub_hidden_dims=sub_hidden_dims,
        sub_output_dims=sub_output_dims,
    )

    save_runtime_model(winner_meta["runtime_model"], train_config["out"])
    with Path(train_config["winner_genome_out"]).open("wb") as handle:
        pickle.dump(selected_winner, handle)
    Path(train_config["winner_eval_out"]).write_text(
        json.dumps(
            {
                "fitness": float(winner_fitness),
                "summary": winner_summary,
                "seed_used": f"{train_config['seed_tag']}|winner",
                "games": int(final_eval_games),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if winner_playoff is not None:
        Path(train_config["winner_playoff_out"]).write_text(
            json.dumps(winner_playoff["summary"], indent=2),
            encoding="utf-8",
        )

    lineage_state = _write_lineage_state(lineage_state_path, population)
    winner_lineage = _build_winner_lineage_export(lineage_state, int(getattr(selected_winner, "key", -1)))
    if winner_lineage is not None:
        Path(train_config["winner_lineage_out"]).write_text(
            json.dumps(winner_lineage, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    winner_generation = None
    if winner_playoff is not None:
        winner_generation = winner_playoff.get("winner_training_record", {}).get("generation")
        if winner_generation is None:
            winner_generation = winner_playoff.get("winner_record", {}).get("generation")
    if winner_generation is None:
        for entry in candidate_entries:
            record = dict(entry.get("record") or {})
            if int(_safe_float(record.get("genome_key"), -1)) == int(getattr(selected_winner, "key", -1)):
                winner_generation = record.get("generation")
                break

    run_summary = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "seed": str(args.seed),
        "config_path": str(Path(train_config["config_path"]).resolve()),
        "run_dir": str(run_dir.resolve()),
        "runtime_model_path": str(Path(train_config["out"]).resolve()),
        "winner_genome_path": str(Path(train_config["winner_genome_out"]).resolve()),
        "winner_eval_path": str(Path(train_config["winner_eval_out"]).resolve()),
        "winner_playoff_path": (
            str(Path(train_config["winner_playoff_out"]).resolve())
            if winner_playoff is not None
            else None
        ),
        "eval_metrics_path": str(eval_metrics_path.resolve()),
        "generation_metrics_path": str(generation_metrics_path.resolve()),
        "lineage_path": str(lineage_path.resolve()),
        "lineage_state_path": str(lineage_state_path.resolve()),
        "winner_lineage_path": (
            str(Path(train_config["winner_lineage_out"]).resolve())
            if winner_lineage is not None
            else None
        ),
        "winner_generation": (int(winner_generation) if winner_generation is not None else None),
        "checkpoints_dir": str(checkpoints_dir.resolve()),
        "last_checkpoint_path": (
            str(Path(checkpoint_state["last_saved_path"]).resolve())
            if checkpoint_state.get("last_saved_path") is not None
            else None
        ),
        "resume_path": (str(Path(resume_path).resolve()) if resume_path is not None else None),
        "bootstrap_seed_genome": (
            str(Path(bootstrap_seed_genome).resolve())
            if bootstrap_seed_genome is not None
            else None
        ),
        "bootstrap_seed_genome_count": int(bootstrap_seed_genome_count),
        "winner_fitness": float(winner_fitness),
        "winner_win_rate": _safe_float(winner_summary.get("win_rate_a"), 0.0),
        "winner_mean_gold_delta": _safe_float(winner_summary.get("mean_gold_delta_a"), 0.0),
        "winner_games": int(_safe_float(winner_summary.get("games"), 0.0)),
        "winner_playoff": (winner_playoff["summary"] if winner_playoff is not None else None),
        "runtime_effective": runtime_settings,
    }
    Path(train_config["run_summary_out"]).write_text(
        json.dumps(run_summary, indent=2),
        encoding="utf-8",
    )

    console_summary = {
        "run_dir": str(run_dir.resolve()),
        "run_summary": str(Path(train_config["run_summary_out"]).resolve()),
        "winner_generation": (int(winner_generation) if winner_generation is not None else None),
        "best_fitness": float(winner_fitness),
        "win_rate": _safe_float(winner_summary.get("win_rate_a"), 0.0),
        "mean_gold_delta": _safe_float(winner_summary.get("mean_gold_delta_a"), 0.0),
        "go_take_rate": _safe_optional_float(winner_summary.get("go_take_rate_a")),
        "go_fail_rate": _safe_optional_float(winner_summary.get("go_fail_rate_a")),
        "winner_selection_mode": (
            "topk_fresh_seed_playoff" if winner_playoff is not None else "selection_fitness"
        ),
        "winner_runtime": str(Path(train_config["out"]).resolve()),
        "winner_genome": str(Path(train_config["winner_genome_out"]).resolve()),
    }
    print(json.dumps(console_summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
