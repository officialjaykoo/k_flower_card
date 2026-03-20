from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchConfig:
    initial_resolution: int = 2
    max_resolution: int = 4
    iteration_level: int = 1
    division_threshold: float = 0.3
    variance_threshold: float = 0.03
    band_threshold: float = 0.2
    max_weight: float = 2.0
    weight_threshold: float = 0.05
    leo_enabled: bool = False
    leo_threshold: float = 0.0
    max_discoveries: int = 0
    max_outgoing: int = 6
    only_leaf_variance: bool = False
    median_variance: bool = False
    relative_variance: bool = False
    max_variance: bool = False


@dataclass(frozen=True)
class Config:
    search: SearchConfig = SearchConfig()
    output_activation: str = "tanh"
    hidden_activation: str = "tanh"
    debug_prune_log: bool = False


__all__ = ["Config", "SearchConfig"]
