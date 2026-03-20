from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .coordinates import Point2D


class CppnModel(Protocol):
    def activate(self, values: Sequence[float]) -> Sequence[float]:
        ...


@dataclass
class CppnNetworkAdapter:
    network: object

    def activate(self, values: Sequence[float]) -> Sequence[float]:
        fn = getattr(self.network, "activate", None)
        if not callable(fn):
            raise TypeError("wrapped network does not expose activate(values)")
        result = fn(list(values))
        if isinstance(result, (list, tuple)):
            return list(result)
        return [float(result)]


def threshold_weight(raw_weight: float, threshold: float, max_weight: float) -> float:
    if abs(raw_weight) <= threshold:
        return 0.0
    if raw_weight > 0:
        scaled = (raw_weight - threshold) / max(1e-9, 1.0 - threshold)
    else:
        scaled = (raw_weight + threshold) / max(1e-9, 1.0 - threshold)
    return scaled * max_weight


def _query_outputs(
    cppn: CppnModel,
    source: Point2D,
    target: Point2D,
) -> list[float]:
    output = cppn.activate(source.as_cppn_input(target))
    return [float(value) for value in output] if output else [0.0]


def query_cppn_link(
    cppn: CppnModel,
    source: Point2D,
    target: Point2D,
    *,
    max_weight: float = 5.0,
    threshold: float = 0.2,
    leo_enabled: bool = False,
    leo_threshold: float = 0.0,
) -> float:
    outputs = _query_outputs(cppn, source, target)
    raw_weight = float(outputs[0]) if outputs else 0.0
    if leo_enabled:
        if len(outputs) < 2:
            raise ValueError("LEO requires a CPPN with at least 2 outputs")
        expression_raw = float(outputs[1])
        if expression_raw <= leo_threshold:
            return 0.0
        return max(-1.0, min(1.0, raw_weight)) * max_weight
    return threshold_weight(raw_weight, threshold=threshold, max_weight=max_weight)


def query_cppn_weight(
    cppn: CppnModel,
    source: Point2D,
    target: Point2D,
    *,
    max_weight: float = 5.0,
    threshold: float = 0.2,
) -> float:
    return query_cppn_link(
        cppn,
        source,
        target,
        max_weight=max_weight,
        threshold=threshold,
        leo_enabled=False,
        leo_threshold=0.0,
    )
