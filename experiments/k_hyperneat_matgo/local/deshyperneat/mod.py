from __future__ import annotations

from typing import Any

from .conf import GenomeConfig, MethodConfig
from .developer import Developer, topology_init_config
from .environment import EnvironmentDescription
from .evolution import evolve_algorithm, prepare_algorithm
from .executor import Executor, compile_executor
from .genome import Genome
from .log import Logger


class Deshyperneat:
    Config = MethodConfig
    Genome = Genome
    Developer = Developer
    Executor = Executor
    Logger = Logger
    EnvironmentDescription = EnvironmentDescription

    @staticmethod
    def genome_config(_: Any | None = None) -> GenomeConfig:
        return Genome.parse_config({})

    @staticmethod
    def genome_init_config(description: EnvironmentDescription | None = None) -> dict[str, Any]:
        if description is None:
            return {}
        return topology_init_config(description)

    @staticmethod
    def prepare(
        *,
        ini_path,
        description: EnvironmentDescription | None = None,
        genome_config: Any | None = None,
        resume=None,
    ):
        return prepare_algorithm(
            ini_path=ini_path,
            algorithm=Deshyperneat,
            description=description,
            genome_config=genome_config,
            resume=resume,
        )

    @staticmethod
    def evolve(
        *,
        ini_path,
        description: EnvironmentDescription | None = None,
        genome_config: Any | None = None,
        fitness_function,
        generations: int,
        resume=None,
        reporters=None,
    ):
        return evolve_algorithm(
            ini_path=ini_path,
            algorithm=Deshyperneat,
            description=description,
            genome_config=genome_config,
            fitness_function=fitness_function,
            generations=generations,
            resume=resume,
            reporters=reporters,
        )


def deshyperneat(*args, **kwargs):
    if not args and not kwargs:
        return Deshyperneat
    if args:
        raise TypeError("deshyperneat() only accepts keyword arguments")
    call_kwargs = dict(kwargs)
    call_kwargs["algorithm"] = Deshyperneat
    return evolve_algorithm(**call_kwargs)


__all__ = [
    "Deshyperneat",
    "Developer",
    "EnvironmentDescription",
    "Executor",
    "Genome",
    "Logger",
    "compile_executor",
    "deshyperneat",
    "evolve_algorithm",
    "prepare_algorithm",
    "topology_init_config",
]
