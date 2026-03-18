#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import gzip
import json
import math
import os
import pickle
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
CORE_ROOT = REPO_ROOT / "experiments" / "k_hyperneat_py"
VENDOR_DIR = REPO_ROOT / "vendor"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

import neat  # type: ignore
from neat.export import export_network_json  # type: ignore

sys.path.insert(0, str(EXPERIMENT_ROOT))

from k_hyperneat import DesDeveloper, DesHyperneatConfig, NeatPythonCppnAdapter, compile_executor
from local.matgo_topology import build_minimal_matgo_topology


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
    runtime["games_per_genome"] = int(runtime.get("games_per_genome", 8) or 8)
    runtime["max_eval_steps"] = int(runtime.get("max_eval_steps", 300) or 300)
    runtime["first_turn_policy"] = str(runtime.get("first_turn_policy", "alternate") or "alternate").strip().lower()
    runtime["continuous_series"] = bool(runtime.get("continuous_series", True))
    runtime["fitness_gold_scale"] = float(runtime.get("fitness_gold_scale", 1500.0) or 1500.0)
    runtime["fitness_gold_neutral_delta"] = float(runtime.get("fitness_gold_neutral_delta", 0.0) or 0.0)
    runtime["fitness_win_weight"] = float(runtime.get("fitness_win_weight", 0.95) or 0.95)
    runtime["fitness_gold_weight"] = float(runtime.get("fitness_gold_weight", 0.05) or 0.05)
    runtime["fitness_win_neutral_rate"] = float(runtime.get("fitness_win_neutral_rate", 0.5) or 0.5)
    runtime["seed"] = str(seed_text)
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
    return runtime


def build_log_dir(phase: int, seed_text: str) -> Path:
    return REPO_ROOT / "logs" / "K-HyperNEAT" / f"k_hyperneat_phase{phase}_seed{seed_text}"


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


class SingleCppnMatgoGenome:
    def __init__(self, cppn_network: Any, topology: Any):
        self._cppn = NeatPythonCppnAdapter(cppn_network)
        self._topology = topology

    def topology(self):
        return self._topology

    def get_node_cppn(self, substrate):
        return self._cppn

    def get_link_cppn(self, source, target):
        return self._cppn

    def get_depth(self, substrate):
        return 0


class GenerationStateReporter(neat.reporting.BaseReporter):
    def __init__(self, holder: dict[str, Any]):
        self.holder = holder

    def start_generation(self, generation):
        self.holder["generation"] = int(generation)


class QuietCheckpointer(neat.Checkpointer):
    def save_checkpoint(self, config, population, species_set, generation):
        filename = f"{self.filename_prefix}{generation}"
        with gzip.open(filename, "w", compresslevel=5) as handle:
            data = (generation, config, population, species_set, random.getstate())
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


class KHyperneatTrainer:
    def __init__(self, runtime: dict[str, Any], config: Any, out_dir: Path):
        self.runtime = runtime
        self.config = config
        self.out_dir = out_dir
        self.checkpoints_dir = out_dir / "checkpoints"
        self.models_dir = out_dir / "models"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.generation_metrics_path = out_dir / "generation_metrics.ndjson"
        self.runtime_path = out_dir / "runtime_config.json"
        self.runtime_path.write_text(json.dumps(runtime, indent=2), encoding="utf-8")
        self.des_config = DesHyperneatConfig()
        self.topology = build_minimal_matgo_topology()
        self.generation_state: dict[str, Any] = {"generation": 0}
        self.current_generation_records: list[EvalRecord] = []
        self.best_record: EvalRecord | None = None
        self.best_genome_pickle: bytes | None = None

    def evaluate_genomes(self, genomes, config):
        generation = int(self.generation_state.get("generation", 0))
        genome_pairs = list(genomes)
        records: list[EvalRecord] = []
        worker_count = min(max(1, int(self.runtime["eval_workers"])), max(1, len(genome_pairs)))

        if worker_count <= 1:
            for genome_id, genome in genome_pairs:
                record = self.evaluate_genome(generation, genome_id, genome, config)
                genome.fitness = record.fitness
                records.append(record)
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
                    records.append(record)
                    self._maybe_promote_best(genome_id, genome, config, record)

        self.current_generation_records = records
        self.write_generation_metrics(generation, records)

    def _maybe_promote_best(self, genome_id: int, genome: Any, config: Any, record: EvalRecord) -> None:
        if self.best_record is None or record.fitness > self.best_record.fitness:
            self.best_record = record
            self.best_genome_pickle = pickle.dumps(genome, protocol=pickle.HIGHEST_PROTOCOL)
            self.export_winner_artifacts(genome_id, genome, config, record)

    def evaluate_genome(self, generation: int, genome_id: int, genome: Any, config: Any) -> EvalRecord:
        runtime_model = self.build_runtime_model(genome, config)
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
        )

    def build_runtime_model(self, genome: Any, config: Any) -> dict[str, Any]:
        cppn_network = neat.nn.FeedForwardNetwork.create(genome, config)
        developer = DesDeveloper(self.des_config)
        developed = developer.develop(SingleCppnMatgoGenome(cppn_network, self.topology))
        executor = compile_executor(developed, self.topology, self.des_config)
        runtime_model = executor.to_runtime_dict(self.topology, adapter_kind="matgo_minimal_v1")
        assert runtime_model["format_version"] == K_HYPERNEAT_MODEL_FORMAT
        return runtime_model

    def run_duel_eval(self, runtime_path: Path, result_path: Path) -> dict[str, Any]:
        mix = list(self.runtime.get("opponent_policy_mix") or [])
        if mix:
            counts = allocate_games(int(self.runtime["games_per_genome"]), mix)
            aggregate = {
                "games": 0,
                "wins_a": 0.0,
                "wins_b": 0.0,
                "draws": 0.0,
                "gold_sum_a": 0.0,
            }
            for index, (entry, games) in enumerate(zip(mix, counts)):
                summary = self._run_single_duel(
                    runtime_path,
                    str(entry["policy"]),
                    games,
                    result_path.with_name(f"{result_path.stem}_{index}.json"),
                )
                aggregate["games"] += int(summary.get("games", 0) or 0)
                aggregate["wins_a"] += float(summary.get("wins_a", 0) or 0)
                aggregate["wins_b"] += float(summary.get("wins_b", 0) or 0)
                aggregate["draws"] += float(summary.get("draws", 0) or 0)
                aggregate["gold_sum_a"] += float(summary.get("mean_gold_delta_a", 0) or 0.0) * float(summary.get("games", 0) or 0)
            games_total = max(1.0, float(aggregate["games"]))
            return {
                "games": int(aggregate["games"]),
                "win_rate_a": aggregate["wins_a"] / games_total,
                "win_rate_b": aggregate["wins_b"] / games_total,
                "draw_rate": aggregate["draws"] / games_total,
                "mean_gold_delta_a": aggregate["gold_sum_a"] / games_total,
            }

        return self._run_single_duel(
            runtime_path,
            str(self.runtime.get("opponent_policy") or "").strip(),
            int(self.runtime["games_per_genome"]),
            result_path,
        )

    def _run_single_duel(self, runtime_path: Path, opponent_policy: str, games: int, result_path: Path) -> dict[str, Any]:
        cmd = [
            "node",
            str(REPO_ROOT / "scripts" / "model_duel_worker.mjs"),
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
            "--result-out",
            str(result_path),
        ]
        completed = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
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
            "mean_fitness": float(sum(item.fitness for item in records) / max(1, len(records))),
            "mean_win_rate": float(sum(item.win_rate for item in records) / max(1, len(records))),
            "mean_gold_delta": float(sum(item.mean_gold_delta for item in records) / max(1, len(records))),
        }
        with self.generation_metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def export_winner_artifacts(self, genome_id: int, genome: Any, config: Any, record: EvalRecord) -> None:
        cppn_network = neat.nn.FeedForwardNetwork.create(genome, config)
        runtime_model = self.build_runtime_model(genome, config)

        (self.models_dir / "winner_runtime.json").write_text(
            json.dumps(runtime_model, indent=2),
            encoding="utf-8",
        )
        with (self.models_dir / "winner_cppn_genome.pkl").open("wb") as handle:
            pickle.dump(genome, handle, protocol=pickle.HIGHEST_PROTOCOL)
        export_network_json(
            cppn_network,
            filepath=str(self.models_dir / "winner_cppn_network.json"),
            metadata={
                "generation": int(record.generation),
                "genome_id": int(genome_id),
                "fitness": float(record.fitness),
                "win_rate": float(record.win_rate),
                "mean_gold_delta": float(record.mean_gold_delta),
            },
        )

    def write_run_summary(self, winner: Any | None, run_elapsed_sec: float) -> Path:
        summary_path = self.out_dir / "run_summary.json"
        payload = {
            "format_version": "k_hyperneat_run_summary_v1",
            "seed": self.runtime["seed"],
            "phase": int(self.runtime["phase"]),
            "completed_at_utc": utc_now_iso(),
            "run_elapsed_sec": float(run_elapsed_sec),
            "winner_present": winner is not None,
            "generation_metrics_log": str(self.generation_metrics_path.resolve()),
            "best_record": (
                {
                    "generation": int(self.best_record.generation),
                    "genome_key": int(self.best_record.genome_key),
                    "fitness": float(self.best_record.fitness),
                    "win_rate": float(self.best_record.win_rate),
                    "draw_rate": float(self.best_record.draw_rate),
                    "loss_rate": float(self.best_record.loss_rate),
                    "mean_gold_delta": float(self.best_record.mean_gold_delta),
                    "games": int(self.best_record.games),
                }
                if self.best_record
                else None
            ),
            "models": {
                "winner_runtime": str((self.models_dir / "winner_runtime.json").resolve()) if (self.models_dir / "winner_runtime.json").exists() else None,
                "winner_cppn_network": str((self.models_dir / "winner_cppn_network.json").resolve()) if (self.models_dir / "winner_cppn_network.json").exists() else None,
                "winner_cppn_genome": str((self.models_dir / "winner_cppn_genome.pkl").resolve()) if (self.models_dir / "winner_cppn_genome.pkl").exists() else None,
            },
        }
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return summary_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", default="experiments/k_hyperneat_matgo/configs/runtime_phase1.json")
    parser.add_argument("--seed", required=True)
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--generations", type=int, default=0)
    parser.add_argument("--stdout-format", choices=("text", "json"), default="text")
    return parser


def main() -> None:
    started_at = time.perf_counter()
    args = build_arg_parser().parse_args()
    runtime_raw = load_json(args.runtime)
    runtime = normalize_runtime(runtime_raw, args.seed)
    if args.phase and args.phase > 0:
        runtime["phase"] = int(args.phase)
    if args.generations and args.generations > 0:
        runtime["generations"] = int(args.generations)

    out_dir = build_log_dir(int(runtime["phase"]), str(runtime["seed"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(str(runtime["seed"]))
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        runtime["cppn_config"],
    )

    if args.resume:
        population = neat.Checkpointer.restore_checkpoint(str(resolve_path(args.resume)), config)
    else:
        population = neat.Population(config)

    trainer = KHyperneatTrainer(runtime, config, out_dir)
    population.add_reporter(GenerationStateReporter(trainer.generation_state))
    if args.stdout_format == "text":
        population.add_reporter(neat.StdOutReporter(False))
    population.add_reporter(neat.StatisticsReporter())
    checkpointer_class = neat.Checkpointer if args.stdout_format == "text" else QuietCheckpointer
    population.add_reporter(
        checkpointer_class(
            generation_interval=max(1, int(runtime["checkpoint_every"])),
            filename_prefix=str((trainer.checkpoints_dir / "k-hyperneat-checkpoint-gen").resolve()),
        )
    )

    winner = population.run(trainer.evaluate_genomes, int(runtime["generations"]))
    run_elapsed_sec = round(time.perf_counter() - started_at, 3)
    summary_path = trainer.write_run_summary(winner, run_elapsed_sec)

    if args.stdout_format == "json":
        print(
            json.dumps(
                {
                    "run_summary": str(summary_path.resolve()),
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


if __name__ == "__main__":
    main()
