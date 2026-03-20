from __future__ import annotations

import gzip
import pickle
import random
from typing import Any


class BaseReporter:
    def start_generation(self, generation: int) -> None:
        return None

    def post_evaluate(self, config, population: dict[int, Any], species, best_genome: Any) -> None:
        return None

    def end_generation(self, config, population: dict[int, Any], species) -> None:
        return None


class ReporterSet:
    def __init__(self) -> None:
        self._reporters: list[Any] = []

    def add(self, reporter: Any) -> None:
        self._reporters.append(reporter)

    def start_generation(self, generation: int) -> None:
        for reporter in self._reporters:
            fn = getattr(reporter, "start_generation", None)
            if callable(fn):
                fn(generation)

    def post_evaluate(self, config, population: dict[int, Any], species, best_genome: Any) -> None:
        for reporter in self._reporters:
            fn = getattr(reporter, "post_evaluate", None)
            if callable(fn):
                fn(config, population, species, best_genome)

    def end_generation(self, config, population: dict[int, Any], species) -> None:
        for reporter in self._reporters:
            fn = getattr(reporter, "end_generation", None)
            if callable(fn):
                fn(config, population, species)


class StdOutReporter(BaseReporter):
    def __init__(self, show_species_detail: bool = False) -> None:
        self.show_species_detail = bool(show_species_detail)

    def start_generation(self, generation: int) -> None:
        print(f"[StandaloneEvolution] generation {int(generation)}", flush=True)


class StatisticsReporter(BaseReporter):
    def __init__(self) -> None:
        self.best_fitness_history: list[float] = []

    def post_evaluate(self, config, population: dict[int, Any], species, best_genome: Any) -> None:
        self.best_fitness_history.append(float(getattr(best_genome, "fitness", float("-inf")) or float("-inf")))


class Checkpointer(BaseReporter):
    def __init__(self, generation_interval: int = 1, filename_prefix: str = "checkpoint-") -> None:
        self.generation_interval = max(1, int(generation_interval))
        self.filename_prefix = str(filename_prefix)
        self._last_generation = -1

    def end_generation(self, config, population: dict[int, Any], species) -> None:
        generation = getattr(species, "generation", None)
        if generation is None:
            generation = max([self._last_generation + 1, 0])
        if int(generation) % int(self.generation_interval) != 0:
            return
        self.save_checkpoint(config, population, species, int(generation))
        self._last_generation = int(generation)

    def save_checkpoint(self, config, population: dict[int, Any], species, generation: int) -> None:
        filename = f"{self.filename_prefix}{int(generation)}"
        reproduction = getattr(getattr(species, "population_state", None), "reproduction", None)
        with gzip.open(filename, "wb", compresslevel=5) as handle:
            payload = (
                int(generation),
                config,
                population,
                species,
                reproduction,
                getattr(getattr(species, "population_state", None), "state", None),
                random.getstate(),
            )
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def restore_checkpoint(filename: str, config=None):
        from .population import Population, Reproduction

        with gzip.open(str(filename), "rb") as handle:
            payload = pickle.load(handle)
        if len(payload) == 5:
            generation, saved_config, population_data, species_set, random_state = payload
            reproduction = None
            state = None
        elif len(payload) == 6:
            generation, saved_config, population_data, species_set, reproduction, random_state = payload
            state = None
        else:
            generation, saved_config, population_data, species_set, reproduction, state, random_state = payload
        runtime_config = config or saved_config
        population = Population(runtime_config)
        population.population = population_data
        population.species = species_set
        population.generation = int(generation)
        if state is not None:
            population.bind_state(state)
        if reproduction is None:
            population.reproduction = Reproduction(runtime_config, population.state)
        else:
            population.reproduction = reproduction
            population.reproduction.state = population.state
        population.reproduction.seed_next_key_from_population(population.population)
        setattr(population.species, "generation", int(generation))
        setattr(population.species, "population_state", population)
        random.setstate(random_state)
        return population


__all__ = [
    "BaseReporter",
    "Checkpointer",
    "ReporterSet",
    "StatisticsReporter",
    "StdOutReporter",
]
