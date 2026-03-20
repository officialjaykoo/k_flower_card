from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable

from .reporting import ReporterSet
from .state import State


@dataclass
class EvolutionConfig:
    genome_type: type
    genome_config: Any
    pop_size: int
    fitness_threshold: float
    no_fitness_termination: bool
    reset_on_extinction: bool
    compatibility_threshold: float
    compatibility_threshold_min: float
    compatibility_threshold_max: float
    compatibility_threshold_step: float
    target_species_count: int
    max_stagnation: int
    species_elitism: int
    species_fitness_func: str
    elitism: int
    survival_threshold: float
    min_species_size: int


@dataclass
class SpeciesRecord:
    key: int
    created: int
    representative: Any
    members: dict[int, Any] = field(default_factory=dict)
    best_fitness: float = float("-inf")
    last_improved: int = 0
    fitness: float = float("-inf")
    adjusted_fitness: float = float("-inf")


class SpeciesSet:
    def __init__(self, config: EvolutionConfig) -> None:
        self.config = config
        self.species: dict[int, SpeciesRecord] = {}
        self.genome_to_species: dict[int, int] = {}
        self._next_species_key = 1

    def speciate(self, population: dict[int, Any], generation: int) -> None:
        previous = list(self.species.values())
        self.species = {}
        self.genome_to_species = {}

        representatives: list[tuple[int, Any]] = []
        for record in previous:
            representatives.append((int(record.key), record.representative))

        for genome_key, genome in sorted(population.items()):
            assigned_species_key: int | None = None
            for species_key, representative in representatives:
                distance = float(genome.distance(representative, self.config.genome_config))
                if distance <= float(self.config.compatibility_threshold):
                    assigned_species_key = int(species_key)
                    break
            if assigned_species_key is None:
                assigned_species_key = int(self._next_species_key)
                self._next_species_key += 1
                representatives.append((assigned_species_key, genome))
            record = self.species.get(assigned_species_key)
            if record is None:
                prior = next((item for item in previous if int(item.key) == assigned_species_key), None)
                record = SpeciesRecord(
                    key=int(assigned_species_key),
                    created=(int(prior.created) if prior is not None else int(generation)),
                    representative=genome,
                    members={},
                    best_fitness=(float(prior.best_fitness) if prior is not None else float("-inf")),
                    last_improved=(int(prior.last_improved) if prior is not None else int(generation)),
                )
                self.species[assigned_species_key] = record
            record.members[int(genome_key)] = genome
            self.genome_to_species[int(genome_key)] = int(assigned_species_key)

        for record in self.species.values():
            ranked = sorted(
                record.members.values(),
                key=lambda genome: float(getattr(genome, "fitness", float("-inf")) or float("-inf")),
                reverse=True,
            )
            if ranked:
                record.representative = ranked[0]
                member_fitness = [float(getattr(genome, "fitness", 0.0) or 0.0) for genome in ranked]
                if str(self.config.species_fitness_func).strip().lower() == "mean":
                    record.fitness = float(sum(member_fitness) / max(1, len(ranked)))
                else:
                    record.fitness = float(getattr(ranked[0], "fitness", float("-inf")) or float("-inf"))
                record.adjusted_fitness = float(sum(member_fitness) / max(1, len(ranked)))
                if record.fitness > record.best_fitness:
                    record.best_fitness = record.fitness
                    record.last_improved = int(generation)
        self._adjust_compatibility_threshold()

    def _adjust_compatibility_threshold(self) -> None:
        target = max(1, int(self.config.target_species_count))
        step = max(0.0, float(self.config.compatibility_threshold_step))
        if step <= 0.0:
            return
        current_count = int(len(self.species))
        threshold = float(self.config.compatibility_threshold)
        if current_count > target:
            threshold += step
        elif current_count < target:
            threshold -= step
        self.config.compatibility_threshold = max(
            float(self.config.compatibility_threshold_min),
            min(float(self.config.compatibility_threshold_max), threshold),
        )

    def surviving_species(self, generation: int) -> dict[int, SpeciesRecord]:
        ranked = sorted(self.species.values(), key=lambda item: float(item.fitness), reverse=True)
        protected = {int(item.key) for item in ranked[: max(0, int(self.config.species_elitism))]}
        surviving: dict[int, SpeciesRecord] = {}
        for record in ranked:
            stagnant = (int(generation) - int(record.last_improved)) >= int(self.config.max_stagnation)
            if stagnant and int(record.key) not in protected:
                continue
            surviving[int(record.key)] = record
        return surviving


class Reproduction:
    def __init__(self, config: EvolutionConfig, state: State) -> None:
        self.config = config
        self.state = state
        self.ancestors: dict[int, tuple[int, ...]] = {}
        self._next_genome_key = 1

    def seed_next_key_from_population(self, population: dict[int, Any]) -> None:
        highest = max([0, *[int(key) for key in population.keys()]])
        self._next_genome_key = max(int(self._next_genome_key), highest + 1)

    def create_new(self) -> dict[int, Any]:
        population: dict[int, Any] = {}
        for _ in range(int(self.config.pop_size)):
            genome_key = int(self._next_genome_key)
            self._next_genome_key += 1
            genome = self.config.genome_type(genome_key)
            genome.configure_new(self.config.genome_config, self.state)
            population[genome_key] = genome
            self.ancestors[genome_key] = tuple()
        return population

    def reproduce(self, species_set: SpeciesSet, generation: int) -> dict[int, Any]:
        surviving = species_set.surviving_species(generation)
        if not surviving:
            if self.config.reset_on_extinction:
                return self.create_new()
            raise RuntimeError("all species went extinct")

        species_items = list(surviving.values())
        raw_fitness = [
            float(
                item.adjusted_fitness if math.isfinite(item.adjusted_fitness) else item.fitness if math.isfinite(item.fitness) else 0.0
            )
            for item in species_items
        ]
        minimum = min(raw_fitness) if raw_fitness else 0.0
        adjusted = [max(0.0, value - minimum + 1e-6) for value in raw_fitness]
        if sum(adjusted) <= 0.0:
            adjusted = [1.0 for _ in species_items]

        spawn_counts = self._compute_spawn_counts(adjusted, int(self.config.pop_size))
        new_population: dict[int, Any] = {}
        new_ancestors: dict[int, tuple[int, ...]] = {}

        for species_record, spawn_count in zip(species_items, spawn_counts):
            ranked = sorted(
                species_record.members.items(),
                key=lambda item: float(getattr(item[1], "fitness", float("-inf")) or float("-inf")),
                reverse=True,
            )
            if not ranked:
                continue
            elite_count = min(int(self.config.elitism), int(spawn_count), len(ranked))
            for genome_key, genome in ranked[:elite_count]:
                clone = copy.deepcopy(genome)
                clone.key = int(genome_key)
                new_population[int(genome_key)] = clone
                new_ancestors[int(genome_key)] = tuple(self.ancestors.get(int(genome_key), tuple()))

            remaining = int(spawn_count) - int(elite_count)
            if remaining <= 0:
                continue
            parent_pool_size = max(1, int(math.ceil(float(self.config.survival_threshold) * len(ranked))))
            parent_pool = ranked[:parent_pool_size]
            if len(parent_pool) < int(self.config.min_species_size):
                parent_pool = ranked[: min(len(ranked), max(parent_pool_size, int(self.config.min_species_size)))]

            for _ in range(remaining):
                parent_a_key, parent_a = random.choice(parent_pool)
                parent_b_key, parent_b = random.choice(parent_pool)
                child_key = int(self._next_genome_key)
                self._next_genome_key += 1
                child = self.config.genome_type(child_key)
                child.configure_crossover(parent_a, parent_b, self.config.genome_config, self.state)
                child.mutate(self.config.genome_config, self.state)
                new_population[child_key] = child
                new_ancestors[child_key] = (int(parent_a_key), int(parent_b_key))

        while len(new_population) < int(self.config.pop_size):
            species_record = random.choice(species_items)
            ranked = sorted(
                species_record.members.items(),
                key=lambda item: float(getattr(item[1], "fitness", float("-inf")) or float("-inf")),
                reverse=True,
            )
            parent_a_key, parent_a = random.choice(ranked[: max(1, len(ranked))])
            parent_b_key, parent_b = random.choice(ranked[: max(1, len(ranked))])
            child_key = int(self._next_genome_key)
            self._next_genome_key += 1
            child = self.config.genome_type(child_key)
            child.configure_crossover(parent_a, parent_b, self.config.genome_config, self.state)
            child.mutate(self.config.genome_config, self.state)
            new_population[child_key] = child
            new_ancestors[child_key] = (int(parent_a_key), int(parent_b_key))

        if len(new_population) > int(self.config.pop_size):
            ranked_all = sorted(
                new_population.items(),
                key=lambda item: float(getattr(item[1], "fitness", float("-inf")) or float("-inf")),
                reverse=True,
            )
            trimmed = dict(ranked_all[: int(self.config.pop_size)])
            new_population = trimmed
            new_ancestors = {int(key): tuple(new_ancestors.get(int(key), tuple())) for key in new_population.keys()}

        self.ancestors = new_ancestors
        return new_population

    @staticmethod
    def _compute_spawn_counts(weights: list[float], total: int) -> list[int]:
        if not weights:
            return []
        total_weight = sum(weights)
        if total_weight <= 0.0:
            base = total // len(weights)
            counts = [base for _ in weights]
            for index in range(total - sum(counts)):
                counts[index % len(counts)] += 1
            return counts
        raw = [(float(weight) / total_weight) * float(total) for weight in weights]
        counts = [int(math.floor(value)) for value in raw]
        while sum(counts) < int(total):
            index = max(range(len(raw)), key=lambda i: raw[i] - counts[i])
            counts[index] += 1
        while sum(counts) > int(total):
            index = max(range(len(counts)), key=lambda i: counts[i])
            if counts[index] <= 0:
                break
            counts[index] -= 1
        return counts


class Population:
    def __init__(self, config: EvolutionConfig) -> None:
        self.config = config
        self.reporters = ReporterSet()
        self.state = State()
        self.species = SpeciesSet(config)
        self.reproduction = Reproduction(config, self.state)
        self.population = self.reproduction.create_new()
        self.reproduction.seed_next_key_from_population(self.population)
        self.generation = 0
        self.best_genome: Any | None = None

    def bind_state(self, state: State) -> None:
        self.state = state
        self.reproduction.state = self.state

    def add_reporter(self, reporter: Any) -> None:
        self.reporters.add(reporter)

    def best(self):
        return self.best_genome

    def get_stats(self) -> dict[str, Any]:
        return {
            "generation": int(self.generation),
            "population": int(len(self.population)),
            "species": int(len(getattr(self.species, "species", {}) or {})),
            "best_fitness": None
            if self.best_genome is None or getattr(self.best_genome, "fitness", None) is None
            else float(getattr(self.best_genome, "fitness")),
        }

    def run(self, fitness_function: Callable[[list[tuple[int, Any]], EvolutionConfig], None], generations: int):
        best_seen = self.best_genome
        for generation in range(int(generations)):
            self.generation = int(generation)
            self.reporters.start_generation(int(generation))
            fitness_function(list(self.population.items()), self.config)
            self.species.speciate(self.population, int(generation))
            setattr(self.species, "generation", int(generation))
            setattr(self.species, "population_state", self)
            generation_best = max(
                self.population.values(),
                key=lambda genome: float(getattr(genome, "fitness", float("-inf")) or float("-inf")),
            )
            if best_seen is None or float(getattr(generation_best, "fitness", float("-inf")) or float("-inf")) > float(
                getattr(best_seen, "fitness", float("-inf")) or float("-inf")
            ):
                best_seen = copy.deepcopy(generation_best)
            self.best_genome = best_seen
            self.reporters.post_evaluate(self.config, self.population, self.species, generation_best)
            self.reporters.end_generation(self.config, self.population, self.species)
            if generation < int(generations) - 1:
                self.population = self.reproduction.reproduce(self.species, int(generation))
                self.reproduction.seed_next_key_from_population(self.population)
        return best_seen


__all__ = [
    "EvolutionConfig",
    "Population",
    "Reproduction",
    "SpeciesRecord",
    "SpeciesSet",
]
