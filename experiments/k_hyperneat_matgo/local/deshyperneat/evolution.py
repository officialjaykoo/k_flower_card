from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .algorithm import Algorithm
from .environment import EnvironmentDescription
from .population import EvolutionConfig, Population
from .reporting import BaseReporter, Checkpointer, ReporterSet, StatisticsReporter, StdOutReporter


@dataclass
class EvolutionSetup:
    config: EvolutionConfig
    population: Population


def load_evolution_config(ini_path: str | Path, *, genome_type: type, genome_config: Any) -> EvolutionConfig:
    parser = configparser.ConfigParser()
    parser.read(str(ini_path), encoding="utf-8")
    if not parser.has_section("Evolution"):
        raise RuntimeError("missing [Evolution] section")
    if not parser.has_section("Speciation"):
        raise RuntimeError("missing [Speciation] section")
    if not parser.has_section("Stagnation"):
        raise RuntimeError("missing [Stagnation] section")
    if not parser.has_section("Reproduction"):
        raise RuntimeError("missing [Reproduction] section")
    evolution_section = dict(parser.items("Evolution"))
    species_section = dict(parser.items("Speciation"))
    stagnation_section = dict(parser.items("Stagnation"))
    reproduction_section = dict(parser.items("Reproduction"))
    return EvolutionConfig(
        genome_type=genome_type,
        genome_config=genome_config,
        pop_size=int(evolution_section.get("pop_size", 64) or 64),
        fitness_threshold=float(evolution_section.get("fitness_threshold", 0.0) or 0.0),
        no_fitness_termination=str(evolution_section.get("no_fitness_termination", "true")).strip().lower()
        in ("1", "true", "yes", "on"),
        reset_on_extinction=str(evolution_section.get("reset_on_extinction", "false")).strip().lower()
        in ("1", "true", "yes", "on"),
        compatibility_threshold=float(species_section.get("compatibility_threshold", 3.0) or 3.0),
        compatibility_threshold_min=float(species_section.get("compatibility_threshold_min", 0.5) or 0.5),
        compatibility_threshold_max=float(species_section.get("compatibility_threshold_max", 10.0) or 10.0),
        compatibility_threshold_step=float(species_section.get("compatibility_threshold_step", 0.25) or 0.25),
        target_species_count=int(species_section.get("target_species_count", 6) or 6),
        max_stagnation=int(stagnation_section.get("max_stagnation", 20) or 20),
        species_elitism=int(stagnation_section.get("species_elitism", 2) or 2),
        species_fitness_func=str(stagnation_section.get("species_fitness_func", "max") or "max"),
        elitism=int(reproduction_section.get("elitism", 2) or 2),
        survival_threshold=float(reproduction_section.get("survival_threshold", 0.25) or 0.25),
        min_species_size=int(reproduction_section.get("min_species_size", 2) or 2),
    )


def initialize_population(
    *,
    config: EvolutionConfig,
    resume: str | Path | None = None,
) -> Population:
    if resume is None:
        return Population(config)
    return Checkpointer.restore_checkpoint(str(resume), config)


def prepare_population(
    *,
    ini_path: str | Path,
    genome_type: type,
    genome_config: Any,
    resume: str | Path | None = None,
) -> EvolutionSetup:
    config = load_evolution_config(
        ini_path,
        genome_type=genome_type,
        genome_config=genome_config,
    )
    population = initialize_population(config=config, resume=resume)
    return EvolutionSetup(config=config, population=population)


def prepare_algorithm(
    *,
    ini_path: str | Path,
    algorithm: type[Algorithm],
    description: EnvironmentDescription | None = None,
    genome_config: Any | None = None,
    resume: str | Path | None = None,
) -> EvolutionSetup:
    resolved_description = description or EnvironmentDescription()
    resolved_genome_config = genome_config
    if resolved_genome_config is None:
        resolved_genome_config = algorithm.genome_config(resolved_description)
    init_config = algorithm.genome_init_config(resolved_description)
    if isinstance(init_config, dict):
        for key, value in dict(init_config).items():
            setattr(resolved_genome_config, str(key), value)
    return prepare_population(
        ini_path=ini_path,
        genome_type=algorithm.Genome,
        genome_config=resolved_genome_config,
        resume=resume,
    )


def attach_reporters(population: Population, reporters: list[Any] | None = None) -> Population:
    for reporter in list(reporters or []):
        population.add_reporter(reporter)
    return population


def evolve(
    *,
    ini_path: str | Path,
    genome_type: type,
    genome_config: Any,
    fitness_function: Callable[[list[tuple[int, Any]], EvolutionConfig], None],
    generations: int,
    reporters: list[Any] | None = None,
    resume: str | Path | None = None,
) -> tuple[Any, Population]:
    setup = prepare_population(
        ini_path=ini_path,
        genome_type=genome_type,
        genome_config=genome_config,
        resume=resume,
    )
    population = attach_reporters(setup.population, reporters)
    best = population.run(fitness_function, int(generations))
    return best, population


def evolve_algorithm(
    *,
    ini_path: str | Path,
    algorithm: type[Algorithm],
    description: EnvironmentDescription | None = None,
    genome_config: Any | None = None,
    fitness_function: Callable[[list[tuple[int, Any]], EvolutionConfig], None],
    generations: int,
    resume: str | Path | None = None,
    reporters: list[Any] | None = None,
) -> tuple[Any, Population]:
    setup = prepare_algorithm(
        ini_path=ini_path,
        algorithm=algorithm,
        description=description,
        genome_config=genome_config,
        resume=resume,
    )
    population = attach_reporters(setup.population, reporters)
    best = population.run(fitness_function, int(generations))
    return best, population


__all__ = [
    "Algorithm",
    "BaseReporter",
    "Checkpointer",
    "EnvironmentDescription",
    "EvolutionConfig",
    "EvolutionSetup",
    "Population",
    "ReporterSet",
    "StatisticsReporter",
    "StdOutReporter",
    "attach_reporters",
    "evolve",
    "evolve_algorithm",
    "initialize_population",
    "load_evolution_config",
    "prepare_algorithm",
    "prepare_population",
]
