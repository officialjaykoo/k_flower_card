from __future__ import annotations

import configparser
from pathlib import Path

from deshyperneat import (
    ControlSnapshot,
    EnvironmentDescription,
    Genome,
    GenomeConfig,
    SeededGenomeInitConfig,
    SubstrateTopology,
    TopologyGraph,
)


def build_environment_description(
    *,
    topology: SubstrateTopology,
    des_runtime: dict[str, object],
) -> EnvironmentDescription:
    _ = des_runtime
    return EnvironmentDescription(
        inputs=sum(len(spec.seed_points) for spec in list(topology.inputs or [])),
        outputs=sum(len(spec.seed_points) for spec in list(topology.outputs or [])),
    )


def build_control_snapshot(
    *,
    topology: SubstrateTopology,
    des_runtime: dict[str, object],
) -> ControlSnapshot:
    _ = topology
    runtime = dict(des_runtime or {})
    return ControlSnapshot(
        enable_identity_mapping=bool(runtime.get("enable_identity_mapping", True)),
        static_substrate_depth=int(runtime.get("static_substrate_depth", -1) or -1),
        max_input_substrate_depth=max(0, int(runtime.get("max_input_substrate_depth", 0) or 0)),
        max_hidden_substrate_depth=max(0, int(runtime.get("max_hidden_substrate_depth", 5) or 5)),
        max_output_substrate_depth=max(0, int(runtime.get("max_output_substrate_depth", 0) or 0)),
    )


def build_seed_graph(
    *,
    topology: SubstrateTopology,
) -> TopologyGraph:
    return TopologyGraph.from_substrate_topology(topology)


def build_matgo_genome_init_config(
    *,
    topology: SubstrateTopology,
    des_runtime: dict[str, object],
) -> SeededGenomeInitConfig:
    return SeededGenomeInitConfig(
        seed_graph=build_seed_graph(topology=topology),
        control_snapshot=build_control_snapshot(topology=topology, des_runtime=des_runtime),
    )


def load_genome_config(
    ini_path: str | Path,
    *,
    topology: SubstrateTopology,
    des_runtime: dict[str, object],
) -> GenomeConfig:
    parser = configparser.ConfigParser()
    parser.read(str(ini_path), encoding="utf-8")
    if not parser.has_section("TopologyGenome"):
        raise RuntimeError("missing [TopologyGenome] section")
    if not parser.has_section("CppnGenome"):
        raise RuntimeError("missing [CppnGenome] section")
    param_dict = {}
    param_dict.update({str(key): value for key, value in parser.items("TopologyGenome")})
    param_dict.update({str(key): value for key, value in parser.items("CppnGenome")})
    return Genome.parse_config(param_dict)
