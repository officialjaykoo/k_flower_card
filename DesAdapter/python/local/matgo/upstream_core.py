from __future__ import annotations

from deshyperneat import EnvironmentDescription, Point2D, SubstrateRef, SubstrateSpec, SubstrateTopology
from src.cppn.developer import topology_init_config


UPSTREAM_CORE_ADAPTER_MODE = "upstream_core"
UPSTREAM_CORE_INPUT_COUNT = 20
UPSTREAM_CORE_OUTPUT_COUNT = 6


def _row(count: int, y: float, x_start: float = -1.0, x_end: float = 1.0) -> list[Point2D]:
    if count <= 1:
        return [Point2D(0.0, y)]
    step = (x_end - x_start) / float(count - 1)
    return [Point2D(x_start + (step * index), y) for index in range(count)]


def is_upstream_core_adapter_mode(runtime: dict[str, object] | None = None) -> bool:
    mode = ""
    if isinstance(runtime, dict):
        mode = str(runtime.get("adapter_mode", "") or "").strip().lower()
    return mode == UPSTREAM_CORE_ADAPTER_MODE


def build_upstream_core_io_topology() -> SubstrateTopology:
    return SubstrateTopology(
        inputs=[
            SubstrateSpec(
                SubstrateRef.input(0),
                seed_points=_row(UPSTREAM_CORE_INPUT_COUNT, y=1.0),
                depth=0,
            )
        ],
        outputs=[
            SubstrateSpec(
                SubstrateRef.output(0),
                seed_points=_row(UPSTREAM_CORE_OUTPUT_COUNT, y=-1.0),
                depth=0,
            )
        ],
        hidden=[],
        links=[],
    )


def build_upstream_core_topology() -> SubstrateTopology:
    return build_upstream_core_io_topology()


def build_upstream_core_environment_description(
    *,
    topology: SubstrateTopology,
    des_runtime: dict[str, object],
) -> EnvironmentDescription:
    del topology, des_runtime
    return EnvironmentDescription(
        inputs=UPSTREAM_CORE_INPUT_COUNT,
        outputs=UPSTREAM_CORE_OUTPUT_COUNT,
    )


def build_upstream_core_genome_init_config(
    *,
    topology: SubstrateTopology,
    des_runtime: dict[str, object],
) -> dict[str, int]:
    description = build_upstream_core_environment_description(topology=topology, des_runtime=des_runtime)
    resolved = topology_init_config(description)
    return {
        "inputs": int(resolved.get("inputs", 0) or 0),
        "outputs": int(resolved.get("outputs", 0) or 0),
    }
