from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class EnvironmentDescription:
    inputs: int = 0
    outputs: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class Environment(Protocol):
    def description(self) -> EnvironmentDescription: ...


__all__ = ["Environment", "EnvironmentDescription"]
