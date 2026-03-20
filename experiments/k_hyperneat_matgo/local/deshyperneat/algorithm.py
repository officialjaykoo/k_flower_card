from __future__ import annotations

from typing import Any, Protocol

from .environment import EnvironmentDescription


class Algorithm(Protocol):
    Config: type
    Genome: type
    Developer: type

    @staticmethod
    def genome_config(description: EnvironmentDescription | None = None) -> Any: ...

    @staticmethod
    def genome_init_config(description: EnvironmentDescription | None = None) -> Any: ...


__all__ = ["Algorithm"]
