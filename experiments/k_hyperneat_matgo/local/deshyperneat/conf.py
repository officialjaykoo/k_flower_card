from __future__ import annotations

from dataclasses import dataclass, field

from .cppn_genome import GenomeConfig as CppnGenomeConfig
from .topology_graph import TopologyGraph


@dataclass
class MethodConfig:
    single_cppn_state: bool = False
    input_config: str = "line"
    output_config: str = "line"
    mutate_node_depth_probability: float = 0.10
    mutate_all_components: bool = True
    log_visualizations: bool = False
    max_input_substrate_depth: int = 0
    max_output_substrate_depth: int = 0
    max_hidden_substrate_depth: int = 5
    enable_identity_mapping: bool = True
    static_substrate_depth: int = -1
    node_depths: dict[str, int] = field(default_factory=dict)
    edge_outer_weights: dict[str, float] = field(default_factory=dict)
    identity_mapping_edges: set[str] = field(default_factory=set)


@dataclass
class TopologyConfig:
    add_node_probability: float = 0.03
    add_link_probability: float = 0.20
    initial_link_weight_size: float = 0.10
    mutate_link_weight_probability: float = 0.90
    mutate_link_weight_size: float = 0.30
    remove_node_probability: float = 0.008
    remove_link_probability: float = 0.08
    only_hidden_node_distance: bool = True
    link_distance_weight: float = 0.50
    mutate_only_one_link: bool = False


@dataclass
class GenomeConfig:
    cppn_genome_type: type
    cppn_genome_config: CppnGenomeConfig
    topology_mutation: TopologyConfig = field(default_factory=TopologyConfig)
    mutate_node_depth_probability: float = 0.10
    mutate_all_components: bool = True
    edge_gate_init_enabled_rate: float = 1.0
    outer_weight_init_jitter: float = 0.10
    outer_weight_replace_rate: float = 0.05
    outer_weight_min_value: float = -2.0
    outer_weight_max_value: float = 2.0
    seed_graph: TopologyGraph | None = None
    control_snapshot: MethodConfig | None = None

    @property
    def cppn(self) -> CppnGenomeConfig:
        return self.cppn_genome_config

    @property
    def topology(self) -> TopologyConfig:
        return self.topology_mutation

    def save(self, handle) -> None:
        self.cppn_genome_config.save(handle)


DESHYPERNEAT = MethodConfig()

ControlSnapshot = MethodConfig
MutationConfig = TopologyConfig
