from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float

    def as_cppn_input(self, other: "Point2D", bias: float = 1.0) -> list[float]:
        return [self.x, self.y, other.x, other.y, bias]
