from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .conf import DESHYPERNEAT
from .developer import Developer
from .figure import save_fig_to_file


@dataclass
class Logger:
    default_logger: Any | None = None
    developer: Any | None = None
    log_interval: int = 10

    @classmethod
    def new(cls, description: Any, config: Any) -> "Logger":
        return cls(
            default_logger=None,
            developer=Developer(description) if isinstance(description, object) else None,
            log_interval=10,
        )

    def log(self, iteration: int, population: Any, stats: Any) -> None:
        if not getattr(DESHYPERNEAT, "log_visualizations", False):
            return None
        if int(iteration) % int(self.log_interval) != 0:
            return None
        if self.developer is None:
            return None
        best_fn = getattr(population, "best", None)
        if not callable(best_fn):
            return None
        best = best_fn()
        if best is None:
            return None
        genome = getattr(best, "genome", None)
        if genome is None:
            return None
        connections_fn = getattr(self.developer, "connections", None)
        if not callable(connections_fn):
            return None
        save_fig_to_file(connections_fn(genome), "g.json", 1.0, 4.0)

    def close(self) -> None:
        return None
