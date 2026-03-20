from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any

from .substrate import DevelopmentEdge, SubstrateRef, SubstrateSpec, SubstrateTopology

from .cppn import NetworkAdapter
from .cppn_genome import Genome as CppnGenome
from .cppn_genome import GenomeConfig as CppnGenomeConfig
from .cppn_genome import Network as CppnNetwork
from .conf import ControlSnapshot
from . import link as des_link
from . import node as des_node
from .conf import GenomeConfig
from .conf import MutationConfig
from .link import edge_key
from .link import export_link
from .link import Link
from .node import export_node
from .node import Node
from .state import State
from .topology_graph import (
    TopologyGraph,
    TopologyGraphLink,
    TopologyGraphNode,
)


def _kind_to_ref(kind: str) -> SubstrateRef:
    return SubstrateRef(kind=str(kind), index=0)


@dataclass
class NeatGenomeStats:
    hidden_nodes: int
    links: int


@dataclass
class DESGenomeStats:
    topology: NeatGenomeStats
    input_node_cppns: NeatGenomeStats
    hidden_node_cppns: NeatGenomeStats
    output_node_cppns: NeatGenomeStats
    link_cppns: NeatGenomeStats


@dataclass
class DesNeatGenome:
    inputs: dict[SubstrateRef, Node]
    hidden_nodes: dict[SubstrateRef, Node]
    outputs: dict[SubstrateRef, Node]
    links: dict[tuple[SubstrateRef, SubstrateRef], Link]

    def get_stats(self) -> NeatGenomeStats:
        return NeatGenomeStats(
            hidden_nodes=int(len(self.hidden_nodes)),
            links=int(len(self.links)),
        )


class Genome(
):
    @classmethod
    def parse_config(cls, param_dict):
        cppn_genome_type = CppnGenome
        cppn_genome_config = cppn_genome_type.parse_config(param_dict)
        mutate_node_depth_probability = float(param_dict.get("mutate_node_depth_probability", 0.10) or 0.10)
        mutate_all_components = str(param_dict.get("mutate_all_components", "true")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        topology_mutation = MutationConfig(
            add_node_probability=float(param_dict.get("layout_add_node_probability", 0.03) or 0.03),
            add_link_probability=float(param_dict.get("layout_add_link_probability", 0.20) or 0.20),
            initial_link_weight_size=float(param_dict.get("layout_initial_link_weight_size", 0.10) or 0.10),
            mutate_link_weight_probability=float(
                param_dict.get("layout_mutate_link_weight_probability", 0.90) or 0.90
            ),
            mutate_link_weight_size=float(param_dict.get("layout_mutate_link_weight_size", 0.30) or 0.30),
            remove_node_probability=float(param_dict.get("layout_remove_node_probability", 0.008) or 0.008),
            remove_link_probability=float(param_dict.get("layout_remove_link_probability", 0.08) or 0.08),
            only_hidden_node_distance=str(
                param_dict.get("layout_only_hidden_node_distance", "true")
            ).strip().lower()
            in ("1", "true", "yes", "on"),
            link_distance_weight=float(param_dict.get("layout_link_distance_weight", 0.50) or 0.50),
            mutate_only_one_link=str(param_dict.get("layout_mutate_only_one_link", "false")).strip().lower()
            in ("1", "true", "yes", "on"),
        )
        return GenomeConfig(
            cppn_genome_type=cppn_genome_type,
            cppn_genome_config=cppn_genome_config,
            topology_mutation=topology_mutation,
            mutate_node_depth_probability=mutate_node_depth_probability,
            mutate_all_components=mutate_all_components,
            edge_gate_init_enabled_rate=float(param_dict.get("edge_gate_init_enabled_rate", 1.0) or 1.0),
            outer_weight_init_jitter=float(param_dict.get("outer_weight_init_jitter", 0.10) or 0.10),
            outer_weight_replace_rate=float(param_dict.get("outer_weight_replace_rate", 0.05) or 0.05),
            outer_weight_min_value=float(param_dict.get("outer_weight_min_value", -2.0) or -2.0),
            outer_weight_max_value=float(param_dict.get("outer_weight_max_value", 2.0) or 2.0),
        )

    @classmethod
    def write_config(cls, handle, config):
        config.save(handle)

    def __init__(self, key):
        self.key = int(key)
        self.fitness = None
        self.neat = DesNeatGenome(inputs={}, hidden_nodes={}, outputs={}, links={})
        self._control_snapshot: ControlSnapshot | None = None
        self._cppn_genome_config: CppnGenomeConfig | None = None
        self._state: State | None = None
        self._nodes: dict[str, Node] = {}
        self._links: dict[str, Link] = {}
        self._node_cppn_cache: dict[str, Any] = {}
        self._link_cppn_cache: dict[str, Any] = {}
        self._topology_graph_cache: TopologyGraph | None = None
        self._topology_event_serial = 0
        self._dynamic_component_serial = 0
        self._node_innovation_serial = 0
        self._link_innovation_serial = 0
        self._last_topology_events: list[str] = []

    @staticmethod
    def export_cppn_component(genome: Any) -> dict[str, Any]:
        node_payload: dict[str, Any] = {}
        for key, gene in sorted((getattr(genome, "nodes", {}) or {}).items()):
            node_payload[str(key)] = {
                "bias": float(getattr(gene, "bias", 0.0)),
                "response": float(getattr(gene, "response", 1.0)),
                "activation": str(getattr(gene, "activation", "")),
                "aggregation": str(getattr(gene, "aggregation", "")),
            }
        connection_payload: dict[str, Any] = {}
        for key, gene in sorted((getattr(genome, "connections", {}) or {}).items()):
            connection_payload[f"{key[0]}->{key[1]}"] = {
                "weight": float(getattr(gene, "weight", 0.0)),
                "enabled": bool(getattr(gene, "enabled", True)),
            }
        return {
            "key": int(getattr(genome, "key", 0) or 0),
            "fitness": None if getattr(genome, "fitness", None) is None else float(getattr(genome, "fitness")),
            "nodes": node_payload,
            "connections": connection_payload,
        }

    def init_desgenome(self) -> None:
        return None

    def export_components(self) -> dict[str, Any]:
        graph = self.topology_graph()
        state_snapshot = self._export_component_state_snapshot(
            node_state_keys=[str(gene.state_key) for gene in self._iter_all_node_genes()],
            link_state_keys=[str(gene.state_key) for gene in self._links.values()],
        )
        return {
            "format_version": "deshyperneat_genome_v1",
            "genome_key": int(self.key),
            "topology_state": {
                "hidden_enabled_count": int(sum(1 for gene in self._iter_node_genes("hidden") if gene.enabled)),
                "hidden_total_count": int(self._node_gene_count("hidden")),
                "edge_enabled_count": int(sum(1 for gene in self._links.values() if gene.enabled)),
                "edge_total_count": int(len(self._links)),
            },
            "topology_innovations": {
                "all_node_innovations": sorted(self._topology_node_genes().keys()),
                "all_link_innovations": sorted(self._topology_link_genes().keys()),
                "active_node_innovations": sorted(self._active_node_genes().keys()),
                "active_link_innovations": sorted(self._active_link_genes().keys()),
            },
            "topology_graph": graph.stats(),
            "dynamic_hidden_specs": sorted(
                str(gene.ref.kind) for gene in self._iter_node_genes("hidden") if not bool(gene.seed)
            ),
            "last_topology_events": list(self._last_topology_events),
            "cppn_state": state_snapshot,
            "input_nodes": {
                kind: export_node(gene, self.export_cppn_component) for kind, gene in sorted(self._node_gene_map("input").items())
            },
            "hidden_nodes": {
                kind: export_node(gene, self.export_cppn_component) for kind, gene in sorted(self._node_gene_map("hidden").items())
            },
            "output_nodes": {
                kind: export_node(gene, self.export_cppn_component) for kind, gene in sorted(self._node_gene_map("output").items())
            },
            "links": {key: export_link(gene, self.export_cppn_component) for key, gene in sorted(self._links.items())},
        }

    def get_neat(self):
        self._sync_neat_view()
        return self.neat

    def topology(self) -> SubstrateTopology:
        if self._control_snapshot is None:
            raise RuntimeError("Genome topology is not initialized")
        return self.topology_graph().to_substrate_topology()

    def topology_graph(self) -> TopologyGraph:
        cached = self._topology_graph_cache
        if cached is not None:
            return cached
        cached = self._build_topology_graph()
        self._topology_graph_cache = cached
        return cached

    def get_node_cppn(self, substrate: SubstrateRef):
        kind = str(substrate.kind)
        cached = self._node_cppn_cache.get(kind)
        if cached is not None:
            return cached
        gene = self._get_node_gene(kind)
        network = CppnNetwork.create(gene.cppn, self._require_cppn_genome_config())
        cached = NetworkAdapter(network)
        self._node_cppn_cache[kind] = cached
        return cached

    def get_link_cppn(self, source: SubstrateRef, target: SubstrateRef):
        key = edge_key(source, target)
        cached = self._link_cppn_cache.get(key)
        if cached is not None:
            return cached
        gene = self._links[key]
        network = CppnNetwork.create(gene.cppn, self._require_cppn_genome_config())
        cached = NetworkAdapter(network)
        self._link_cppn_cache[key] = cached
        return cached

    def get_depth(self, substrate: SubstrateRef):
        gene = self._get_node_gene(str(substrate.kind))
        if not gene.enabled:
            return 0
        return int(gene.depth)

    def get_link_outer_weight(self, source: SubstrateRef, target: SubstrateRef):
        return float(self._links[edge_key(source, target)].outer_weight)

    def get_link_identity_mapping(self, source: SubstrateRef, target: SubstrateRef):
        return bool(self._links[edge_key(source, target)].identity_mapping_enabled)

    def is_substrate_enabled(self, substrate: SubstrateRef) -> bool:
        gene = self._get_node_gene(str(substrate.kind))
        return bool(gene.enabled)

    def is_link_enabled(self, source: SubstrateRef, target: SubstrateRef) -> bool:
        gene = self._links[edge_key(source, target)]
        return bool(gene.enabled) and self.is_substrate_enabled(source) and self.is_substrate_enabled(target)

    def configure_new(self, config: GenomeConfig, state: State | None = None):
        seed_graph = self._init_common(config, state)
        genome_type = config.cppn_genome_type
        seed_nodes = sorted(seed_graph.nodes.items(), key=lambda item: (item[1].role, item[0]))
        for index, (_, seed_node) in enumerate(seed_nodes):
            spec = seed_node.spec
            cppn = genome_type(self._node_component_key(index))
            cppn.configure_new(config.cppn_genome_config)
            gene = des_node.new(
                ref=spec.ref,
                spec=spec,
                role=str(seed_node.role),
                seed=True,
                cppn=cppn,
                innovation=self._node_innovation(index),
                depth=int(self._depth_for_ref(spec.ref)),
                enabled=self._init_node_enabled(spec.ref, config),
                state_key=str(spec.ref.kind),
                origin="new",
                split_from_link_key=None,
            )
            self._assign_node_gene(gene)
            self._register_component_state(
                state_key=gene.state_key,
                component_kind="node",
            )
        seed_links = sorted(seed_graph.links.items())
        for index, (_, seed_link) in enumerate(seed_links):
            edge = seed_link.edge
            key = edge_key(edge.source, edge.target)
            identity_mapping_enabled = bool(self._allow_identity_mapping(edge.source, edge.target))
            cppn = self._create_link_cppn(
                component_key=self._link_component_key(index),
                config=config,
                identity_mapping_enabled=identity_mapping_enabled,
            )
            self._links[key] = des_link.new(
                edge=edge,
                seed=True,
                cppn=cppn,
                innovation=self._link_innovation(index),
                depth=self._initial_link_depth(edge),
                enabled=self._init_edge_enabled(config),
                outer_weight=self._init_outer_weight(float(self._edge_outer_weight(edge.source, edge.target)), config),
                identity_mapping_enabled=identity_mapping_enabled,
                state_key=key,
                state_redirect_key=None,
                origin="identity" if identity_mapping_enabled else "new",
                cloned_from_key=None,
                split_hidden_kind=None,
            )
            self._register_component_state(
                state_key=self._links[key].state_key,
                redirect_key=self._links[key].state_redirect_key,
                component_kind="link",
            )
        self._sync_neat_view()

    def configure_crossover(
        self,
        genome1,
        genome2,
        config: GenomeConfig,
        state: State | None = None,
    ):
        self._init_common(config, state)
        genome_type = config.cppn_genome_type
        dominant_parent, recessive_parent = self._select_crossover_parents(genome1, genome2)
        base_node_keys = self._base_node_keys_from_parents(dominant_parent, recessive_parent)
        for index, kind in enumerate(base_node_keys):
            parent_a = dominant_parent._nodes.get(kind)
            parent_b = recessive_parent._nodes.get(kind)
            template = parent_a if parent_a is not None else parent_b
            cppn = genome_type(self._node_component_key(index))
            if parent_a is not None and parent_b is not None:
                self._prime_component_fitness(parent_a.cppn, dominant_parent)
                self._prime_component_fitness(parent_b.cppn, recessive_parent)
                cppn.configure_crossover(parent_a.cppn, parent_b.cppn, config.cppn_genome_config)
                gene = des_node.crossover(
                    parent_a,
                    parent_b,
                    cppn,
                    fitness=float(getattr(dominant_parent, "fitness", 0.0) or 0.0),
                    other_fitness=float(getattr(recessive_parent, "fitness", 0.0) or 0.0),
                )
                gene.seed = True
                gene.innovation = int(parent_a.innovation)
                gene.state_key = f"{kind}@g{self.key}"
                gene.origin = "crossover"
                gene.split_from_link_key = None
            else:
                gene = des_node.new(
                    ref=template.ref,
                    spec=template.spec,
                    role=str(template.role),
                    seed=True,
                    cppn=copy.deepcopy(template.cppn),
                    innovation=int(template.innovation),
                    depth=int(template.depth),
                    enabled=bool(template.enabled),
                    state_key=f"{kind}@g{self.key}",
                    origin="crossover_clone",
                    split_from_link_key=None,
                )
            self._assign_node_gene(gene)
            self._register_component_state(
                state_key=gene.state_key,
                component_kind="node",
            )
        base_link_keys = self._base_link_keys_from_parents(dominant_parent, recessive_parent)
        for index, key in enumerate(base_link_keys):
            parent_a = dominant_parent._links.get(key)
            parent_b = recessive_parent._links.get(key)
            template = parent_a if parent_a is not None else parent_b
            cppn = genome_type(self._link_component_key(index))
            if parent_a is not None and parent_b is not None:
                self._prime_component_fitness(parent_a.cppn, dominant_parent)
                self._prime_component_fitness(parent_b.cppn, recessive_parent)
                cppn.configure_crossover(parent_a.cppn, parent_b.cppn, config.cppn_genome_config)
                inherited_state_key = random.choice(
                    [
                        dominant_parent._resolve_component_state_key(key),
                        recessive_parent._resolve_component_state_key(key),
                    ]
                )
                child_state_key = f"{key}@g{self.key}"
                gene = des_link.crossover(
                    parent_a,
                    parent_b,
                    cppn,
                    fitness=float(getattr(dominant_parent, "fitness", 0.0) or 0.0),
                    other_fitness=float(getattr(recessive_parent, "fitness", 0.0) or 0.0),
                )
                gene.seed = True
                gene.innovation = int(parent_a.innovation)
                gene.state_key = child_state_key
                gene.state_redirect_key = inherited_state_key
                gene.origin = "crossover"
                gene.cloned_from_key = inherited_state_key
                gene.split_hidden_kind = None
                self._links[key] = gene
            else:
                inherited_state_key = (
                    dominant_parent._resolve_component_state_key(key)
                    if parent_a is not None
                    else recessive_parent._resolve_component_state_key(key)
                )
                self._links[key] = des_link.new(
                    edge=template.edge,
                    seed=True,
                    cppn=copy.deepcopy(template.cppn),
                    innovation=int(template.innovation),
                    depth=int(template.depth),
                    enabled=bool(template.enabled),
                    outer_weight=float(template.outer_weight),
                    identity_mapping_enabled=bool(template.identity_mapping_enabled),
                    state_key=f"{key}@g{self.key}",
                    state_redirect_key=inherited_state_key,
                    origin="crossover_clone",
                    cloned_from_key=inherited_state_key,
                    split_hidden_kind=None,
                )
            self._register_component_state(
                state_key=self._links[key].state_key,
                redirect_key=self._links[key].state_redirect_key,
                component_kind="link",
            )
        self._crossover_dynamic_structure(dominant_parent, recessive_parent, config)
        self._refresh_innovation_serials()
        self._repair_topology_state()
        self._sync_neat_view()

    def mutate(self, config: GenomeConfig, state: State | None = None):
        if self._control_snapshot is None:
            raise RuntimeError("Genome topology is not initialized")
        if state is not None:
            self._state = state
        self._node_cppn_cache.clear()
        self._link_cppn_cache.clear()
        self._topology_graph_cache = None
        self._last_topology_events = []

        self._mutate_topology_graph(config)
        self._repair_topology_state()

        node_genes = list(self._iter_all_node_genes())
        link_genes = list(self._links.values())
        mutate_all = bool(config.mutate_all_components)
        node_mut_prob = 1.0 if mutate_all else (3.0 / max(1, self._node_gene_count("hidden")))
        link_mut_prob = 1.0 if mutate_all else (3.0 / max(1, len(link_genes)))

        for gene in node_genes:
            if mutate_all or random.random() < node_mut_prob:
                gene.cppn.mutate(config.cppn_genome_config)

        for gene in link_genes:
            if mutate_all or random.random() < link_mut_prob:
                gene.cppn.mutate(config.cppn_genome_config)
            gene.outer_weight = self._mutate_outer_weight(gene.outer_weight, config)

        if random.random() < float(config.mutate_node_depth_probability):
            self._mutate_single_node_depth()

        self._node_cppn_cache.clear()
        self._link_cppn_cache.clear()
        self._topology_graph_cache = None
        self._sync_neat_view()

    def distance(self, other, config: GenomeConfig):
        topology_cfg = config.topology_mutation
        self_node_genes = self._topology_node_genes(hidden_only=bool(topology_cfg.only_hidden_node_distance))
        other_node_genes = other._topology_node_genes(hidden_only=bool(topology_cfg.only_hidden_node_distance))
        node_distance = 0.0
        node_count = 0
        for innovation in sorted(set(self_node_genes) | set(other_node_genes)):
            gene = self_node_genes.get(innovation)
            other_gene = other_node_genes.get(innovation)
            if gene is None or other_gene is None:
                node_distance += 1.0
                node_count += 1
                continue
            node_distance += des_node.distance(
                gene,
                other_gene,
                float(gene.cppn.distance(other_gene.cppn, config.cppn_genome_config)),
            )
            node_count += 1

        self_link_genes = self._topology_link_genes()
        other_link_genes = other._topology_link_genes()
        link_distance = 0.0
        link_count = 0
        for innovation in sorted(set(self_link_genes) | set(other_link_genes)):
            gene = self_link_genes.get(innovation)
            other_gene = other_link_genes.get(innovation)
            if gene is None or other_gene is None:
                link_distance += 1.0
                link_count += 1
                continue
            link_distance += des_link.distance(
                gene,
                other_gene,
                float(gene.cppn.distance(other_gene.cppn, config.cppn_genome_config)),
            )
            link_count += 1

        node_norm = node_distance / max(1, node_count)
        link_norm = link_distance / max(1, link_count)
        link_weight = max(0.0, min(1.0, float(topology_cfg.link_distance_weight)))
        return ((1.0 - link_weight) * node_norm) + (link_weight * link_norm)

    def size(self):
        hidden_nodes = 0
        links = 0
        for gene in self._iter_all_node_genes():
            hidden_nodes += int(len(getattr(gene.cppn, "nodes", {}) or {}))
            links += int(len(getattr(gene.cppn, "connections", {}) or {}))
        for gene in self._links.values():
            hidden_nodes += int(len(getattr(gene.cppn, "nodes", {}) or {}))
            links += int(len(getattr(gene.cppn, "connections", {}) or {}))
        return hidden_nodes, links

    def get_stats(self) -> DESGenomeStats:
        self._sync_neat_view()
        return DESGenomeStats(
            topology=self.neat.get_stats(),
            input_node_cppns=self._accumulate_cppn_stats(self._iter_node_genes("input")),
            hidden_node_cppns=self._accumulate_cppn_stats(self._iter_node_genes("hidden")),
            output_node_cppns=self._accumulate_cppn_stats(self._iter_node_genes("output")),
            link_cppns=self._accumulate_cppn_stats(self._links.values()),
        )

    def _init_common(
        self,
        config: GenomeConfig,
        state: State | None = None,
    ) -> TopologyGraph:
        seed_graph, control_snapshot = self._require_layout_config(config)
        self._control_snapshot = control_snapshot
        self._cppn_genome_config = config.cppn_genome_config
        self._state = (state if state is not None else State())
        self.neat = DesNeatGenome(inputs={}, hidden_nodes={}, outputs={}, links={})
        self._nodes = {}
        self._links = {}
        self._node_cppn_cache = {}
        self._link_cppn_cache = {}
        self._topology_graph_cache = None
        self._topology_event_serial = 0
        self._dynamic_component_serial = 0
        self._node_innovation_serial = self._node_innovation(max(0, len(seed_graph.nodes) - 1))
        self._link_innovation_serial = self._link_innovation(max(0, len(seed_graph.links) - 1))
        self._last_topology_events = []
        return seed_graph

    def _sync_neat_view(self) -> None:
        self.neat = DesNeatGenome(
            inputs={gene.ref: gene for gene in self._iter_node_genes("input")},
            hidden_nodes={gene.ref: gene for gene in self._iter_node_genes("hidden")},
            outputs={gene.ref: gene for gene in self._iter_node_genes("output")},
            links={(gene.edge.source, gene.edge.target): gene for gene in self._links.values()},
        )

    @staticmethod
    def _cppn_stats(cppn: Any) -> NeatGenomeStats:
        return NeatGenomeStats(
            hidden_nodes=int(len(getattr(cppn, "nodes", {}) or {})),
            links=int(len(getattr(cppn, "connections", {}) or {})),
        )

    def _accumulate_cppn_stats(self, genes) -> NeatGenomeStats:
        values = []
        for gene in genes:
            values.append(self._cppn_stats(gene.cppn))
        if not values:
            return NeatGenomeStats(hidden_nodes=0, links=0)
        return NeatGenomeStats(
            hidden_nodes=int(sum(item.hidden_nodes for item in values) / len(values)),
            links=int(sum(item.links for item in values) / len(values)),
        )

    def _assign_node_gene(self, gene: Node) -> None:
        kind = str(gene.ref.kind)
        self._nodes[kind] = gene

    def _node_items(self, role: str | None = None):
        for kind, gene in self._nodes.items():
            if role is not None and gene.role != role:
                continue
            yield kind, gene

    def _iter_node_genes(self, role: str | None = None):
        for _, gene in self._node_items(role):
            yield gene

    def _node_gene_map(self, role: str | None = None) -> dict[str, Node]:
        return {kind: gene for kind, gene in self._node_items(role)}

    def _node_gene_count(self, role: str | None = None) -> int:
        return sum(1 for _ in self._node_items(role))

    def _get_hidden_gene(self, kind: str) -> Node | None:
        gene = self._nodes.get(str(kind))
        if gene is None or gene.role != "hidden":
            return None
        return gene

    def _get_node_gene(self, kind: str) -> Node:
        return self._nodes[kind]

    def _iter_all_node_genes(self):
        yield from self._nodes.values()

    @staticmethod
    def _base_node_keys_from_parents(
        dominant_parent: "Genome",
        recessive_parent: "Genome",
    ) -> list[str]:
        dominant_graph = dominant_parent.topology_graph()
        recessive_graph = recessive_parent.topology_graph()
        kinds = set(dominant_graph.seed_node_keys()) | set(recessive_graph.seed_node_keys())
        return sorted(
            kinds,
            key=lambda kind: int(
                (dominant_parent._nodes.get(kind) or recessive_parent._nodes[kind]).innovation
            ),
        )

    @staticmethod
    def _base_link_keys_from_parents(
        dominant_parent: "Genome",
        recessive_parent: "Genome",
    ) -> list[str]:
        dominant_graph = dominant_parent.topology_graph()
        recessive_graph = recessive_parent.topology_graph()
        keys = set(dominant_graph.seed_link_keys()) | set(recessive_graph.seed_link_keys())
        return sorted(
            keys,
            key=lambda key: int(
                (dominant_parent._links.get(key) or recessive_parent._links[key]).innovation
            ),
        )

    @staticmethod
    def _inherit_node_enabled(
        parent_a: Node,
        parent_b: Node,
        *,
        prefer_parent_a: bool = False,
    ) -> bool:
        if parent_a.enabled and parent_b.enabled:
            return True
        if (not parent_a.enabled) and (not parent_b.enabled):
            return False
        if random.random() < 0.75:
            return False
        if prefer_parent_a:
            return bool(parent_a.enabled if random.random() < 0.75 else parent_b.enabled)
        return bool(parent_a.enabled if random.random() < 0.5 else parent_b.enabled)

    @staticmethod
    def _inherit_link_enabled(
        parent_a: Link,
        parent_b: Link,
        *,
        prefer_parent_a: bool = False,
    ) -> bool:
        if parent_a.enabled and parent_b.enabled:
            return True
        if (not parent_a.enabled) and (not parent_b.enabled):
            return False
        if random.random() < 0.75:
            return False
        if prefer_parent_a:
            return bool(parent_a.enabled if random.random() < 0.75 else parent_b.enabled)
        return bool(parent_a.enabled if random.random() < 0.5 else parent_b.enabled)

    @staticmethod
    def _select_crossover_parents(genome1, genome2):
        fitness1 = float(getattr(genome1, "fitness", 0.0) or 0.0)
        fitness2 = float(getattr(genome2, "fitness", 0.0) or 0.0)
        if fitness2 > fitness1:
            return genome2, genome1
        if fitness1 > fitness2:
            return genome1, genome2
        key1 = int(getattr(genome1, "key", 0) or 0)
        key2 = int(getattr(genome2, "key", 0) or 0)
        if key2 < key1:
            return genome2, genome1
        return genome1, genome2

    def _build_topology_graph(self) -> TopologyGraph:
        nodes = {
            kind: TopologyGraphNode(
                ref=gene.ref,
                spec=copy.deepcopy(gene.spec),
                role=str(gene.role),
                seed=bool(gene.seed),
                innovation=int(gene.innovation),
                depth=int(gene.depth),
                enabled=bool(gene.enabled),
                state_key=str(gene.state_key),
                origin=str(gene.origin),
            )
            for kind, gene in self._nodes.items()
        }
        links = {
            key: TopologyGraphLink(
                edge=gene.edge,
                seed=bool(gene.seed),
                innovation=int(gene.innovation),
                depth=int(gene.depth),
                enabled=bool(gene.enabled),
                outer_weight=float(gene.outer_weight),
                identity_mapping_enabled=bool(gene.identity_mapping_enabled),
                state_key=str(gene.state_key),
                state_redirect_key=None if gene.state_redirect_key is None else str(gene.state_redirect_key),
                origin=str(gene.origin),
                cloned_from_key=None if gene.cloned_from_key is None else str(gene.cloned_from_key),
            )
            for key, gene in self._links.items()
        }
        return TopologyGraph(nodes=nodes, links=links)

    def _crossover_dynamic_structure(self, dominant_parent, recessive_parent, config) -> None:
        dominant_graph = dominant_parent.topology_graph()
        recessive_graph = recessive_parent.topology_graph()
        dynamic_hidden_kinds = sorted(
            set(dominant_graph.dynamic_node_keys("hidden")) | set(recessive_graph.dynamic_node_keys("hidden"))
        )
        genome_type = config.cppn_genome_type
        for kind in dynamic_hidden_kinds:
            parent_a = dominant_parent._get_hidden_gene(kind)
            parent_b = recessive_parent._get_hidden_gene(kind)
            if parent_a is None:
                continue
            spec = copy.deepcopy((parent_a if parent_a is not None else parent_b).spec)
            if parent_a is not None and parent_b is not None:
                cppn = genome_type(self._next_dynamic_node_component_key())
                self._prime_component_fitness(parent_a.cppn, dominant_parent)
                self._prime_component_fitness(parent_b.cppn, recessive_parent)
                cppn.configure_crossover(parent_a.cppn, parent_b.cppn, config.cppn_genome_config)
                gene = Node(
                    ref=spec.ref,
                    spec=copy.deepcopy(spec),
                    role="hidden",
                    seed=False,
                    cppn=cppn,
                    innovation=int(parent_a.innovation),
                    depth=int(parent_a.depth if random.random() < 0.75 else parent_b.depth),
                    enabled=self._inherit_node_enabled(parent_a, parent_b, prefer_parent_a=True),
                    state_key=f"{kind}@g{self.key}",
                    origin="crossover",
                    split_from_link_key=parent_a.split_from_link_key or parent_b.split_from_link_key,
                )
            else:
                parent = parent_a
                gene = Node(
                    ref=spec.ref,
                    spec=copy.deepcopy(spec),
                    role="hidden",
                    seed=False,
                    cppn=copy.deepcopy(parent.cppn),
                    innovation=int(parent.innovation),
                    depth=int(parent.depth),
                    enabled=bool(parent.enabled),
                    state_key=f"{kind}@g{self.key}",
                    origin="crossover_clone",
                    split_from_link_key=parent.split_from_link_key,
                )
            self._assign_node_gene(gene)
            self._register_component_state(
                state_key=gene.state_key,
                component_kind="node",
            )

        dynamic_link_keys = sorted(set(dominant_graph.dynamic_link_keys()) | set(recessive_graph.dynamic_link_keys()))
        for key in dynamic_link_keys:
            parent_a = dominant_parent._links.get(key)
            parent_b = recessive_parent._links.get(key)
            if parent_a is None:
                continue
            if parent_a is not None and parent_b is not None:
                cppn = genome_type(self._next_dynamic_link_component_key())
                self._prime_component_fitness(parent_a.cppn, dominant_parent)
                self._prime_component_fitness(parent_b.cppn, recessive_parent)
                cppn.configure_crossover(parent_a.cppn, parent_b.cppn, config.cppn_genome_config)
                inherited_state_key = random.choice(
                    [
                        dominant_parent._resolve_component_state_key(parent_a.state_key),
                        recessive_parent._resolve_component_state_key(parent_b.state_key),
                    ]
                )
                gene = Link(
                    edge=copy.deepcopy(parent_a.edge),
                    seed=False,
                    cppn=cppn,
                    innovation=int(parent_a.innovation),
                    depth=int(parent_a.depth if random.random() < 0.75 else parent_b.depth),
                    enabled=self._inherit_link_enabled(parent_a, parent_b, prefer_parent_a=True),
                    outer_weight=float(parent_a.outer_weight if random.random() < 0.75 else parent_b.outer_weight),
                    identity_mapping_enabled=bool(parent_a.identity_mapping_enabled or parent_b.identity_mapping_enabled),
                    state_key=f"{key}@g{self.key}",
                    state_redirect_key=inherited_state_key,
                    origin="crossover",
                    cloned_from_key=inherited_state_key,
                    split_hidden_kind=parent_a.split_hidden_kind or parent_b.split_hidden_kind,
                )
            else:
                parent = parent_a
                inherited_state_key = dominant_parent._resolve_component_state_key(parent.state_key)
                gene = Link(
                    edge=copy.deepcopy(parent.edge),
                    seed=False,
                    cppn=copy.deepcopy(parent.cppn),
                    innovation=int(parent.innovation),
                    depth=int(parent.depth),
                    enabled=bool(parent.enabled),
                    outer_weight=float(parent.outer_weight),
                    identity_mapping_enabled=bool(parent.identity_mapping_enabled),
                    state_key=f"{key}@g{self.key}",
                    state_redirect_key=inherited_state_key,
                    origin="crossover_clone",
                    cloned_from_key=inherited_state_key,
                    split_hidden_kind=parent.split_hidden_kind,
                )
            self._links[key] = gene
            self._register_component_state(
                state_key=gene.state_key,
                redirect_key=gene.state_redirect_key,
                component_kind="link",
            )

    def _refresh_innovation_serials(self) -> None:
        all_node_innovations = [int(gene.innovation) for gene in self._iter_all_node_genes()]
        all_link_innovations = [int(gene.innovation) for gene in self._links.values()]
        if all_node_innovations:
            self._node_innovation_serial = max(all_node_innovations)
        if all_link_innovations:
            self._link_innovation_serial = max(all_link_innovations)

    def _next_node_state_key(self, kind: str) -> str:
        self._topology_event_serial += 1
        return f"{kind}@g{self.key}:n{self._topology_event_serial}"

    def _next_link_state_key(self, edge) -> str:
        self._topology_event_serial += 1
        return f"{edge_key(edge.source, edge.target)}@g{self.key}:l{self._topology_event_serial}"

    def _next_split_hidden_kind(self, source, target) -> str:
        self._dynamic_component_serial += 1
        return f"hidden_split_{self.key}_{self._dynamic_component_serial}_{source.kind}_to_{target.kind}"

    def _next_dynamic_node_component_key(self) -> int:
        self._dynamic_component_serial += 1
        return 50_000 + int(self._dynamic_component_serial)

    def _next_dynamic_link_component_key(self) -> int:
        self._dynamic_component_serial += 1
        return 60_000 + int(self._dynamic_component_serial)

    def _next_node_innovation(self) -> int:
        self._node_innovation_serial += 1
        return int(self._node_innovation_serial)

    def _next_link_innovation(self) -> int:
        self._link_innovation_serial += 1
        return int(self._link_innovation_serial)

    @staticmethod
    def _node_innovation(index: int) -> int:
        return 100_000 + int(index)

    @staticmethod
    def _link_innovation(index: int) -> int:
        return 200_000 + int(index)

    def _clamp_outer_weight(self, value: float, config) -> float:
        return max(float(config.outer_weight_min_value), min(float(config.outer_weight_max_value), float(value)))

    def _prime_component_fitness(self, component, owner_genome) -> None:
        if getattr(component, "fitness", None) is not None:
            return
        component.fitness = float(getattr(owner_genome, "fitness", 0.0) or 0.0)

    def _require_layout_config(self, config):
        seed_graph = getattr(config, "seed_graph", None)
        control_snapshot = getattr(config, "control_snapshot", None)
        if control_snapshot is None or seed_graph is None:
            raise RuntimeError("GenomeConfig is missing seed_graph/control_snapshot")
        return seed_graph, control_snapshot

    def _require_control_snapshot(self):
        if self._control_snapshot is None:
            raise RuntimeError("Genome control snapshot is not initialized")
        return self._control_snapshot

    def _depth_for_ref(self, substrate) -> int:
        control = self._require_control_snapshot()
        kind = str(substrate.kind)
        if control.static_substrate_depth >= 0:
            if kind.startswith("hidden_"):
                return int(control.static_substrate_depth)
            return 0
        explicit = control.node_depths.get(kind)
        if explicit is not None:
            return max(0, int(explicit))
        if kind.startswith("input_"):
            return min(1, int(control.max_input_substrate_depth))
        if kind.startswith("hidden_"):
            return min(1, int(control.max_hidden_substrate_depth))
        if kind.startswith("output_"):
            return min(1, int(control.max_output_substrate_depth))
        return 0

    def _edge_outer_weight(self, source, target) -> float:
        control = self._require_control_snapshot()
        return float(control.edge_outer_weights.get(edge_key(source, target), 1.0))

    def _allow_identity_mapping(self, source, target) -> bool:
        control = self._require_control_snapshot()
        return bool(edge_key(source, target) in control.identity_mapping_edges)

    def _require_cppn_genome_config(self):
        if self._cppn_genome_config is None:
            raise RuntimeError("Genome cppn genome config is not initialized")
        return self._cppn_genome_config

    def _require_state(self) -> State:
        if self._state is None:
            raise RuntimeError("Genome state is not initialized")
        return self._state

    def _resolve_component_state_key(self, state_key: str) -> str:
        return self._require_state().resolve_component_state_key(state_key)

    def _register_component_state(
        self,
        *,
        state_key: str,
        component_kind: str,
        redirect_key: str | None = None,
    ) -> str:
        return self._require_state().register_component_state(
            state_key=state_key,
            component_kind=component_kind,
            redirect_key=redirect_key,
        )

    def _export_component_state_snapshot(
        self,
        *,
        node_state_keys: list[str],
        link_state_keys: list[str],
    ) -> dict[str, dict[str, str]]:
        return self._require_state().export_component_snapshot(
            node_state_keys=node_state_keys,
            link_state_keys=link_state_keys,
        )

    @staticmethod
    def _node_component_key(index: int) -> int:
        return 30_000 + int(index)

    @staticmethod
    def _link_component_key(index: int) -> int:
        return 40_000 + int(index)

    @staticmethod
    def _is_hidden_kind(kind: str) -> bool:
        return str(kind or "").startswith("hidden_")

    @staticmethod
    def _is_context_hidden_kind(kind: str) -> bool:
        value = str(kind or "")
        return value.startswith("hidden_") and not value.startswith("hidden_group_")

    def _node_gene_distance(self, gene, other_gene, config) -> float:
        neat_distance = 0.0
        if bool(gene.enabled) != bool(other_gene.enabled):
            neat_distance += 1.0
        if bool(gene.seed) != bool(other_gene.seed):
            neat_distance += 0.5
        if bool(gene.split_from_link_key) != bool(other_gene.split_from_link_key):
            neat_distance += 0.25
        cppn_distance = float(gene.cppn.distance(other_gene.cppn, config.cppn_genome_config))
        depth_distance = math.tanh(abs(int(gene.depth) - int(other_gene.depth)))
        return float(neat_distance + (0.8 * cppn_distance) + (0.2 * depth_distance))

    def _link_gene_distance(self, gene, other_gene, config) -> float:
        neat_distance = abs(float(gene.outer_weight) - float(other_gene.outer_weight))
        if bool(gene.enabled) != bool(other_gene.enabled):
            neat_distance += 1.0
        if bool(gene.identity_mapping_enabled) != bool(other_gene.identity_mapping_enabled):
            neat_distance += 0.5
        if bool(gene.seed) != bool(other_gene.seed):
            neat_distance += 0.5
        cppn_distance = float(gene.cppn.distance(other_gene.cppn, config.cppn_genome_config))
        depth_distance = math.tanh(abs(int(gene.depth) - int(other_gene.depth)))
        return float((0.5 * neat_distance) + (0.4 * cppn_distance) + (0.1 * depth_distance))

    def _active_node_genes(self) -> dict[int, object]:
        graph = self.topology_graph()
        enabled_refs = {node.ref for node in graph.all_nodes() if node.enabled}
        result = {}
        for gene in self._iter_all_node_genes():
            if gene.ref not in enabled_refs:
                continue
            result[int(gene.innovation)] = gene
        return result

    def _active_link_genes(self) -> dict[int, object]:
        graph = self.topology_graph()
        enabled_pairs = {(link.edge.source, link.edge.target) for link in graph.enabled_links()}
        result = {}
        for gene in self._links.values():
            if (gene.edge.source, gene.edge.target) not in enabled_pairs:
                continue
            result[int(gene.innovation)] = gene
        return result

    def _topology_node_genes(self, *, hidden_only: bool = False) -> dict[int, object]:
        result = {}
        for gene in self._iter_all_node_genes():
            if hidden_only and not str(gene.ref.kind).startswith("hidden_"):
                continue
            result[int(gene.innovation)] = gene
        return result

    def _topology_link_genes(self) -> dict[int, object]:
        return {int(gene.innovation): gene for gene in self._links.values()}

    def _initial_link_depth(self, edge: DevelopmentEdge) -> int:
        if str(edge.source.kind).startswith("hidden_") or str(edge.target.kind).startswith("hidden_"):
            return 1
        return 0

    def _create_link_cppn(self, *, component_key: int, config, identity_mapping_enabled: bool):
        cppn = config.cppn_genome_type(int(component_key))
        cppn.configure_new(config.cppn_genome_config)
        if identity_mapping_enabled:
            self._seed_identity_link_cppn(cppn, config.cppn_genome_config)
        return cppn

    def _seed_identity_link_cppn(self, cppn, genome_config) -> None:
        connections = getattr(cppn, "connections", {})
        self._set_connection_weight(cppn, -1, 0, random.gauss(1.0, 0.15))
        self._set_connection_weight(cppn, 1, 0, random.gauss(-1.0, 0.15))
        self._set_connection_weight(cppn, -3, 0, random.gauss(1.0, 0.15))
        self._set_connection_weight(cppn, 3, 0, random.gauss(-1.0, 0.15))
        if len(getattr(genome_config, "output_keys", [])) > 1:
            self._set_connection_weight(cppn, -1, 1, random.gauss(1.0, 0.10))
            self._set_connection_weight(cppn, 1, 1, random.gauss(-1.0, 0.10))
            self._set_connection_weight(cppn, -3, 1, random.gauss(1.0, 0.10))
            self._set_connection_weight(cppn, 3, 1, random.gauss(-1.0, 0.10))
        for connection in connections.values():
            if not getattr(connection, "enabled", True):
                connection.enabled = True

    @staticmethod
    def _set_connection_weight(cppn, source_key: int, target_key: int, weight: float) -> None:
        connection = getattr(cppn, "connections", {}).get((int(source_key), int(target_key)))
        if connection is None:
            return
        connection.weight = float(weight)
        connection.enabled = True

    def _mutate_node_depth(self, substrate, current_depth: int) -> int:
        kind = str(substrate.kind)
        control = self._require_control_snapshot()
        if control.static_substrate_depth >= 0:
            return int(self._depth_for_ref(substrate))
        if kind.startswith("input_"):
            limit = int(control.max_input_substrate_depth)
        elif kind.startswith("hidden_"):
            limit = int(control.max_hidden_substrate_depth)
        elif kind.startswith("output_"):
            limit = int(control.max_output_substrate_depth)
        else:
            limit = 0
        if limit <= 0:
            return 0
        if current_depth <= 0:
            return 1
        if current_depth >= limit:
            return max(0, current_depth - 1)
        return current_depth + (1 if random.random() < 0.5 else -1)

    def _mutate_single_node_depth(self) -> None:
        node_genes = list(self._iter_all_node_genes())
        if not node_genes:
            return
        gene = random.choice(node_genes)
        gene.depth = self._mutate_node_depth(gene.ref, gene.depth)

    def _init_node_enabled(self, substrate, config) -> bool:
        return True

    def _init_edge_enabled(self, config) -> bool:
        return random.random() < float(config.edge_gate_init_enabled_rate)

    def _init_outer_weight(self, base: float, config) -> float:
        jitter = random.gauss(0.0, max(0.0, float(config.topology_mutation.initial_link_weight_size)))
        return self._clamp_outer_weight(base + jitter, config)

    def _mutate_outer_weight(self, current: float, config) -> float:
        value = float(current)
        if random.random() < float(config.outer_weight_replace_rate):
            return self._init_outer_weight(0.0, config)
        if random.random() < float(config.topology_mutation.mutate_link_weight_probability):
            value += random.gauss(0.0, max(0.0, float(config.topology_mutation.mutate_link_weight_size)))
        return self._clamp_outer_weight(value, config)

    def _enable_hidden_substrate(self, kind: str) -> None:
        gene = self._get_hidden_gene(kind)
        if gene is None:
            return
        gene.enabled = True
        if gene.depth <= 0:
            gene.depth = max(1, int(self._depth_for_ref(type(gene.ref)(kind=kind, index=0))))

    def _incident_edge_keys(self, kind: str) -> list[str]:
        result: list[str] = []
        for key, gene in self._links.items():
            if str(gene.edge.source.kind) == kind or str(gene.edge.target.kind) == kind:
                result.append(key)
        return result

    def _can_enable_edge_gene(self, gene: Link) -> bool:
        return self.is_substrate_enabled(gene.edge.source) and self.is_substrate_enabled(gene.edge.target)

    def _create_split_hidden_gene(self, config, source, target, split_from_link_key: str) -> Node:
        kind = self._next_split_hidden_kind(source, target)
        ref = type(source)(kind=kind, index=0)
        cppn = config.cppn_genome_type(self._next_dynamic_node_component_key())
        cppn.configure_new(config.cppn_genome_config)
        return Node(
            ref=ref,
            spec=self._make_dynamic_hidden_spec(ref),
            role="hidden",
            seed=False,
            cppn=cppn,
            innovation=self._next_node_innovation(),
            depth=max(1, int(self._depth_for_ref(ref))),
            enabled=True,
            state_key=self._next_node_state_key(kind),
            origin="graph_split_add_hidden",
            split_from_link_key=split_from_link_key,
        )

    def _create_split_link_gene(
        self,
        *,
        parent_gene: Link,
        source,
        target,
        inherited_state_key: str | None,
        origin: str,
        split_hidden_kind: str | None,
    ) -> Link:
        edge = DevelopmentEdge(
            source,
            target,
            base_weight=1.0,
            allow_identity_mapping=bool(parent_gene.identity_mapping_enabled),
        )
        cppn = copy.deepcopy(parent_gene.cppn)
        return Link(
            edge=edge,
            seed=False,
            cppn=cppn,
            innovation=self._next_link_innovation(),
            depth=max(1, int(parent_gene.depth)),
            enabled=True,
            outer_weight=float(parent_gene.outer_weight),
            identity_mapping_enabled=bool(parent_gene.identity_mapping_enabled),
            state_key=self._next_link_state_key(edge),
            state_redirect_key=inherited_state_key,
            origin=origin,
            cloned_from_key=inherited_state_key,
            split_hidden_kind=split_hidden_kind,
        )

    def _create_dynamic_link_gene(self, *, config, source, target, origin: str) -> Link:
        edge = DevelopmentEdge(
            source,
            target,
            base_weight=1.0,
            allow_identity_mapping=bool(self._allow_identity_mapping(source, target)),
        )
        cppn = config.cppn_genome_type(self._next_dynamic_link_component_key())
        cppn.configure_new(config.cppn_genome_config)
        if edge.allow_identity_mapping:
            self._seed_identity_link_cppn(cppn, config.cppn_genome_config)
        return Link(
            edge=edge,
            seed=False,
            cppn=cppn,
            innovation=self._next_link_innovation(),
            depth=max(1, self._initial_link_depth(edge)),
            enabled=True,
            outer_weight=self._init_outer_weight(self._edge_outer_weight(source, target), config),
            identity_mapping_enabled=bool(edge.allow_identity_mapping),
            state_key=self._next_link_state_key(edge),
            state_redirect_key=None,
            origin=origin,
            cloned_from_key=None,
            split_hidden_kind=None,
        )

    @staticmethod
    def _make_dynamic_hidden_spec(ref):
        return SubstrateSpec(ref=ref, seed_points=[], depth=0)

    def _mutate_topology_graph(self, config) -> None:
        topology_graph = self._build_topology_graph()
        topology_cfg = config.topology_mutation
        events_applied = 0
        if random.random() < float(topology_cfg.add_node_probability):
            if self._graph_split_enabled_link(topology_graph, config):
                events_applied += 1
        if random.random() < float(topology_cfg.remove_node_probability):
            if self._graph_collapse_split_hidden(topology_graph, config):
                events_applied += 1
        if random.random() < float(topology_cfg.add_link_probability):
            if self._graph_add_link(topology_graph, config):
                events_applied += 1
                if bool(topology_cfg.mutate_only_one_link):
                    self._topology_graph_cache = None
                    return
        if random.random() < float(topology_cfg.remove_link_probability):
            if self._graph_remove_link(topology_graph):
                events_applied += 1
        if events_applied > 0:
            self._topology_graph_cache = None

    def _graph_split_enabled_link(self, topology_graph, config) -> bool:
        candidates = topology_graph.candidate_split_hidden_events()
        if not candidates:
            return False
        candidates.sort(
            key=lambda item: (
                0 if self._is_context_hidden_kind(item.edge.source.kind) or self._is_context_hidden_kind(item.edge.target.kind) else 1,
                str(item.edge.source.kind),
                str(item.edge.target.kind),
            )
        )
        selected = random.choice(candidates[: min(8, len(candidates))])
        direct_key = f"{selected.edge.source.kind}->{selected.edge.target.kind}"
        direct_gene = self._links.get(direct_key)
        if direct_gene is None:
            return False
        split_hidden = self._create_split_hidden_gene(
            config=config,
            source=selected.edge.source,
            target=selected.edge.target,
            split_from_link_key=direct_key,
        )
        self._assign_node_gene(split_hidden)
        self._register_component_state(
            state_key=split_hidden.state_key,
            component_kind="node",
        )
        prior_state_key = self._resolve_component_state_key(direct_gene.state_key)
        self._delete_link_gene(direct_key, reason="graph_split_delete")
        incoming_gene = self._create_split_link_gene(
            parent_gene=direct_gene,
            source=selected.edge.source,
            target=split_hidden.ref,
            inherited_state_key=prior_state_key,
            origin="graph_split_add_link",
            split_hidden_kind=str(split_hidden.ref.kind),
        )
        outgoing_gene = self._create_split_link_gene(
            parent_gene=direct_gene,
            source=split_hidden.ref,
            target=selected.edge.target,
            inherited_state_key=prior_state_key,
            origin="graph_split_add_link",
            split_hidden_kind=str(split_hidden.ref.kind),
        )
        self._links[f"{incoming_gene.edge.source.kind}->{incoming_gene.edge.target.kind}"] = incoming_gene
        self._links[f"{outgoing_gene.edge.source.kind}->{outgoing_gene.edge.target.kind}"] = outgoing_gene
        self._register_component_state(
            state_key=incoming_gene.state_key,
            redirect_key=incoming_gene.state_redirect_key,
            component_kind="link",
        )
        self._register_component_state(
            state_key=outgoing_gene.state_key,
            redirect_key=outgoing_gene.state_redirect_key,
            component_kind="link",
        )
        self._record_topology_event(
            f"split_enabled_link:{selected.edge.source.kind}->{selected.edge.target.kind}:{split_hidden.ref.kind}"
        )
        return True

    def _graph_collapse_split_hidden(self, topology_graph, config) -> bool:
        candidates = []
        incoming_enabled = topology_graph.incoming_enabled_links()
        outgoing_enabled = topology_graph.outgoing_enabled_links()
        for gene in self._iter_node_genes("hidden"):
            if gene.seed or not gene.enabled or not gene.split_from_link_key:
                continue
            hidden_incoming = incoming_enabled.get(gene.ref, [])
            hidden_outgoing = outgoing_enabled.get(gene.ref, [])
            if len(hidden_incoming) != 1 or len(hidden_outgoing) != 1:
                continue
            if gene.split_from_link_key not in self._links:
                continue
            candidates.append((gene, hidden_incoming[0], hidden_outgoing[0]))
        if not candidates:
            return False
        candidates.sort(
            key=lambda item: (
                0 if self._is_context_hidden_kind(item[0].ref.kind) else 1,
                str(item[0].ref.kind),
            )
        )
        gene, in_link, out_link = random.choice(candidates[: min(6, len(candidates))])
        in_gene = self._links[f"{in_link.edge.source.kind}->{in_link.edge.target.kind}"]
        out_gene = self._links[f"{out_link.edge.source.kind}->{out_link.edge.target.kind}"]
        prior_state_key = self._resolve_component_state_key(in_gene.state_key)
        self._delete_link_gene(f"{in_gene.edge.source.kind}->{in_gene.edge.target.kind}", reason="graph_collapse_delete")
        self._delete_link_gene(f"{out_gene.edge.source.kind}->{out_gene.edge.target.kind}", reason="graph_collapse_delete")
        self._delete_hidden_gene(str(gene.ref.kind), reason="graph_collapse_hidden")
        parent_gene = self._links[gene.split_from_link_key]
        self._enable_link_gene(parent_gene, origin="graph_collapse_add_link", inherited_state_key=prior_state_key)
        self._record_topology_event(
            f"collapse_split_hidden:{gene.ref.kind}:{parent_gene.edge.source.kind}->{parent_gene.edge.target.kind}"
        )
        return True

    def _graph_add_link(self, topology_graph, config) -> bool:
        existing_candidates = topology_graph.candidate_add_links()
        existing_candidates.sort(
            key=lambda link: (
                0 if link.identity_mapping_enabled else 1,
                -abs(float(link.outer_weight)),
                str(link.edge.source.kind),
                str(link.edge.target.kind),
            )
        )
        new_candidates = self._graph_new_link_candidates(topology_graph)
        if new_candidates and (not existing_candidates or random.random() < 0.60):
            source, target = random.choice(new_candidates[: min(12, len(new_candidates))])
            new_gene = self._create_dynamic_link_gene(
                config=config,
                source=source,
                target=target,
                origin="graph_new_link",
            )
            key = f"{source.kind}->{target.kind}"
            self._links[key] = new_gene
            self._register_component_state(
                state_key=new_gene.state_key,
                redirect_key=new_gene.state_redirect_key,
                component_kind="link",
            )
            self._record_topology_event(f"new_link:{source.kind}->{target.kind}")
            return True
        if not existing_candidates:
            return False
        selected = random.choice(existing_candidates[: min(8, len(existing_candidates))])
        gene = self._links[f"{selected.edge.source.kind}->{selected.edge.target.kind}"]
        self._enable_link_gene(gene, origin="graph_add_link")
        self._record_topology_event(f"add_link:{selected.edge.source.kind}->{selected.edge.target.kind}")
        return True

    def _graph_remove_link(self, topology_graph) -> bool:
        candidates = topology_graph.candidate_remove_links()
        if not candidates:
            return False
        candidates.sort(
            key=lambda link: (
                1 if link.identity_mapping_enabled else 0,
                abs(float(link.outer_weight)),
                str(link.edge.source.kind),
                str(link.edge.target.kind),
            )
        )
        selected = random.choice(candidates[: min(8, len(candidates))])
        selected_key = f"{selected.edge.source.kind}->{selected.edge.target.kind}"
        self._delete_link_gene(selected_key, reason="graph_remove_link")
        self._record_topology_event(f"remove_link:{selected.edge.source.kind}->{selected.edge.target.kind}")
        return True

    def _enable_link_gene(self, gene: Link, *, origin: str, inherited_state_key: str | None = None) -> None:
        if not self._can_enable_edge_gene(gene):
            return
        prior_state_key = inherited_state_key or self._resolve_component_state_key(gene.state_key)
        gene.enabled = True
        gene.state_key = self._next_link_state_key(gene.edge)
        gene.state_redirect_key = prior_state_key
        gene.cloned_from_key = prior_state_key
        gene.origin = origin
        self._register_component_state(
            state_key=gene.state_key,
            redirect_key=gene.state_redirect_key,
            component_kind="link",
        )

    def _repair_topology_state(self) -> None:
        links_to_delete: list[str] = []
        for key, gene in self._links.items():
            if not self.is_substrate_enabled(gene.edge.source) or not self.is_substrate_enabled(gene.edge.target):
                links_to_delete.append(key)
        for key in links_to_delete:
            self._delete_link_gene(key, reason="graph_repair_delete")
        topology_graph = self._build_topology_graph()
        orphan_hidden_refs = topology_graph.orphan_hidden_refs()
        for ref in orphan_hidden_refs:
            kind = str(ref.kind)
            hidden_gene = self._get_hidden_gene(kind)
            if hidden_gene is not None:
                self._delete_hidden_gene(kind, reason="graph_repair_orphan")
        topology_graph = self._build_topology_graph()
        reachable_from_inputs, reachable_to_outputs = topology_graph.reachable_refs()
        for gene in list(self._iter_node_genes("hidden")):
            if not gene.enabled:
                continue
            if gene.ref not in reachable_from_inputs or gene.ref not in reachable_to_outputs:
                self._delete_hidden_gene(str(gene.ref.kind), reason="graph_repair_unreachable")
        context_enabled = [
            gene for gene in self._iter_node_genes("hidden")
            if self._is_context_hidden_kind(gene.ref.kind) and gene.enabled
        ]
        if not context_enabled:
            if self._get_hidden_gene("hidden_play_context") is not None:
                self._enable_hidden_substrate("hidden_play_context")
        self._repair_hidden_connectivity()

    def _repair_hidden_connectivity(self) -> None:
        topology_graph = self._build_topology_graph()
        incoming = topology_graph.incoming_enabled_links()
        outgoing = topology_graph.outgoing_enabled_links()
        for gene in list(self._iter_node_genes("hidden")):
            if not gene.enabled:
                continue
            ref = gene.ref
            if incoming.get(ref) and outgoing.get(ref):
                continue
            disabled_incident = [
                edge_gene
                for edge_gene in self._links.values()
                if not edge_gene.enabled
                and self._can_enable_edge_gene(edge_gene)
                and (edge_gene.edge.source == ref or edge_gene.edge.target == ref)
            ]
            disabled_incident.sort(
                key=lambda item: (
                    0 if item.identity_mapping_enabled else 1,
                    -abs(float(item.outer_weight)),
                    str(item.edge.source.kind),
                    str(item.edge.target.kind),
                )
            )
            has_incoming = bool(incoming.get(ref))
            has_outgoing = bool(outgoing.get(ref))
            for edge_gene in disabled_incident:
                if (not has_incoming) and edge_gene.edge.target == ref:
                    self._enable_link_gene(edge_gene, origin="graph_repair")
                    has_incoming = True
                elif (not has_outgoing) and edge_gene.edge.source == ref:
                    self._enable_link_gene(edge_gene, origin="graph_repair")
                    has_outgoing = True
                if has_incoming and has_outgoing:
                    break

    def _record_topology_event(self, event: str) -> None:
        self._last_topology_events.append(str(event))
        if len(self._last_topology_events) > 16:
            del self._last_topology_events[:-16]

    def _delete_link_gene(self, key: str, *, reason: str) -> None:
        gene = self._links.pop(key, None)
        if gene is None:
            return
        gene.origin = reason

    def _delete_hidden_gene(self, kind: str, *, reason: str) -> None:
        gene = self._get_hidden_gene(kind)
        if gene is None:
            return
        for edge_key_value in list(self._incident_edge_keys(kind)):
            edge_gene = self._links.get(edge_key_value)
            if edge_gene is None:
                continue
            self._delete_link_gene(edge_key_value, reason=reason)
        gene.origin = reason
        self._nodes.pop(kind, None)

    def _graph_new_link_candidates(self, topology_graph) -> list[tuple]:
        enabled_nodes = [node for node in topology_graph.all_nodes() if node.enabled]
        existing_pairs = {(gene.edge.source, gene.edge.target) for gene in self._links.values()}
        candidates = []
        for source_node in enabled_nodes:
            for target_node in enabled_nodes:
                if source_node.ref == target_node.ref:
                    continue
                if not self._can_create_dynamic_link(source_node.ref, target_node.ref):
                    continue
                pair = (source_node.ref, target_node.ref)
                if pair in existing_pairs:
                    continue
                candidates.append(pair)
        candidates.sort(key=lambda pair: (str(pair[0].kind), str(pair[1].kind)))
        return candidates

    def _can_create_dynamic_link(self, source, target) -> bool:
        source_stage = self._topology_stage(source)
        target_stage = self._topology_stage(target)
        if source_stage >= target_stage:
            return False
        if str(source.kind).startswith("output_"):
            return False
        if str(target.kind).startswith("input_"):
            return False
        return True

    @staticmethod
    def _topology_stage(ref) -> int:
        kind = str(ref.kind)
        if kind.startswith("input_"):
            return 0
        if kind.startswith("hidden_group_"):
            return 1
        if kind.startswith("hidden_"):
            return 2
        if kind.startswith("output_"):
            return 3
        return 2
__all__ = ["DESGenomeStats", "DesNeatGenome", "Genome", "NeatGenomeStats"]
