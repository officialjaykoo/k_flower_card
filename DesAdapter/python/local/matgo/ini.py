from __future__ import annotations

import configparser
from pathlib import Path

from deshyperneat.environment import EnvironmentDescription
from deshyperneat.substrate import SubstrateTopology
from deshyperneat.topology_graph import TopologyGraph

from .controls import MatgoTopologyControl
from deshyperneat.conf import ControlSnapshot
from deshyperneat.genome import Genome
from deshyperneat.conf import GenomeConfig


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
    topology_control = MatgoTopologyControl.from_runtime(topology, dict(des_runtime or {}))
    return ControlSnapshot(
        static_substrate_depth=int(topology_control.static_substrate_depth),
        max_input_substrate_depth=int(topology_control.max_input_substrate_depth),
        max_hidden_substrate_depth=int(topology_control.max_hidden_substrate_depth),
        max_output_substrate_depth=int(topology_control.max_output_substrate_depth),
        node_depths={
            str(kind): int(control.depth)
            for kind, control in sorted(topology_control.node_controls.items())
        },
        edge_outer_weights={
            str(key): float(control.outer_weight)
            for key, control in sorted(topology_control.edge_controls.items())
        },
        identity_mapping_edges={
            str(key)
            for key, control in sorted(topology_control.edge_controls.items())
            if bool(control.allow_identity_mapping)
        },
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
) -> dict[str, object]:
    return {
        "seed_graph": build_seed_graph(topology=topology),
        "control_snapshot": build_control_snapshot(topology=topology, des_runtime=des_runtime),
    }


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
    config = Genome.parse_config(param_dict)
    init_config = build_matgo_genome_init_config(
        topology=topology,
        des_runtime=des_runtime,
    )
    config.seed_graph = init_config["seed_graph"]
    config.control_snapshot = init_config["control_snapshot"]
    return config
