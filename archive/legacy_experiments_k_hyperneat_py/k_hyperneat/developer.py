from __future__ import annotations

"""Phenotype developer for DES-style substrate expansion.

This module takes a genome-provided substrate topology plus node/link CPPNs and
builds a concrete phenotype graph through ordered node growth, link search, and
reachable-path pruning.
"""

from collections import Counter
from dataclasses import dataclass
import sys
from typing import Protocol

from .config import DesHyperneatConfig
from .coordinates import Point2D
from .cppn import CppnModel, query_cppn_link
from .network import PhenotypeGraph
from .search import DesSearchEngine, SearchConnection, SearchResult
from .substrate import (
    DevelopmentEdge,
    OrderedSubstrateEdge,
    OrderedSubstrateNode,
    SubstrateRef,
    SubstrateSpec,
    SubstrateTopology,
)


class DesGenomeProtocol(Protocol):
    def topology(self) -> SubstrateTopology:
        ...

    def get_node_cppn(self, substrate: SubstrateRef) -> CppnModel:
        ...

    def get_link_cppn(self, source: SubstrateRef, target: SubstrateRef) -> CppnModel:
        ...

    def get_depth(self, substrate: SubstrateRef) -> int:
        ...

    def get_link_outer_weight(self, source: SubstrateRef, target: SubstrateRef) -> float:
        ...

    def get_link_identity_mapping(self, source: SubstrateRef, target: SubstrateRef) -> bool:
        ...

    def is_substrate_enabled(self, substrate: SubstrateRef) -> bool:
        ...

    def is_link_enabled(self, source: SubstrateRef, target: SubstrateRef) -> bool:
        ...


@dataclass
class DesDeveloper:
    config: DesHyperneatConfig

    def __post_init__(self) -> None:
        self.search_engine = DesSearchEngine(self.config.search)

    def develop(self, genome: DesGenomeProtocol) -> PhenotypeGraph:
        topology = genome.topology()
        topology.validate()
        genome_key = self._genome_debug_key(genome)

        graph = PhenotypeGraph()
        spec_by_ref = topology.spec_by_ref()
        output_refs = {spec.ref for spec in topology.outputs}
        points_by_substrate = self._initialize_points(topology, graph)

        for action in topology.topological_actions():
            if isinstance(action, OrderedSubstrateNode):
                spec = spec_by_ref[action.ref]
                if not self._is_substrate_enabled(genome, spec.ref):
                    continue
                self._develop_node_substrate(
                    spec=spec,
                    genome=genome,
                    graph=graph,
                    points_by_substrate=points_by_substrate,
                    reverse=spec.reverse_search or spec.ref in output_refs,
                    genome_key=genome_key,
                )
                continue
            if isinstance(action, OrderedSubstrateEdge):
                edge = action.edge
                if not self._is_link_enabled(genome, edge):
                    continue
                self._develop_link_search(
                    edge=edge,
                    target_spec=spec_by_ref[edge.target],
                    genome=genome,
                    graph=graph,
                    points_by_substrate=points_by_substrate,
                    target_is_output=edge.target in output_refs,
                    genome_key=genome_key,
                )

        if self.config.debug_prune_log:
            self._log_prune_state("before", graph, topology, genome_key)
        graph.prune_reachable(
            self._terminal_node_ids(graph, topology.inputs),
            self._terminal_node_ids(graph, topology.outputs),
            preserve_terminal_nodes=True,
        )
        if self.config.debug_prune_log:
            self._log_prune_state("after", graph, topology, genome_key)

        return graph

    def _initialize_points(
        self,
        topology: SubstrateTopology,
        graph: PhenotypeGraph,
    ) -> dict[SubstrateRef, set[Point2D]]:
        points_by_substrate: dict[SubstrateRef, set[Point2D]] = {}
        for spec in topology.all_specs():
            points_by_substrate[spec.ref] = set()
            self._register_points(graph, spec.ref, points_by_substrate[spec.ref], spec.seed_points)
        return points_by_substrate

    def _is_substrate_enabled(self, genome: DesGenomeProtocol, substrate: SubstrateRef) -> bool:
        checker = getattr(genome, "is_substrate_enabled", None)
        if callable(checker):
            try:
                return bool(checker(substrate))
            except Exception:
                return True
        return True

    def _is_link_enabled(self, genome: DesGenomeProtocol, edge: DevelopmentEdge) -> bool:
        if not self._is_substrate_enabled(genome, edge.source):
            return False
        if not self._is_substrate_enabled(genome, edge.target):
            return False
        checker = getattr(genome, "is_link_enabled", None)
        if callable(checker):
            try:
                return bool(checker(edge.source, edge.target))
            except Exception:
                return True
        return True

    def _register_points(
        self,
        graph: PhenotypeGraph,
        substrate: SubstrateRef,
        known_points: set[Point2D],
        points: list[Point2D],
    ) -> None:
        for point in points:
            if point in known_points:
                continue
            known_points.add(point)
            graph.ensure_node(substrate, point)

    def _depth_for(self, spec: SubstrateSpec, genome: DesGenomeProtocol) -> int:
        return max(int(spec.depth), int(genome.get_depth(spec.ref)))

    def _develop_node_substrate(
        self,
        *,
        spec: SubstrateSpec,
        genome: DesGenomeProtocol,
        graph: PhenotypeGraph,
        points_by_substrate: dict[SubstrateRef, set[Point2D]],
        reverse: bool,
        genome_key: int | None,
    ) -> None:
        depth = self._depth_for(spec, genome)
        seed_points = sorted(points_by_substrate.get(spec.ref, set()), key=lambda point: (point.y, point.x))
        if depth <= 0 or not seed_points:
            return

        search_result = self.search_engine.explore_substrate(
            seed_points=seed_points,
            cppn=genome.get_node_cppn(spec.ref),
            output_points=[],
            blocked_points=[],
            depth=depth,
            reverse=reverse,
            allow_target_links=False,
        )
        self._register_points(graph, spec.ref, points_by_substrate[spec.ref], search_result.discovered_points)
        self._add_internal_connections(graph, spec.ref, search_result.connections)

    def _develop_link_search(
        self,
        *,
        edge: DevelopmentEdge,
        target_spec: SubstrateSpec,
        genome: DesGenomeProtocol,
        graph: PhenotypeGraph,
        points_by_substrate: dict[SubstrateRef, set[Point2D]],
        target_is_output: bool,
        genome_key: int | None,
    ) -> None:
        source_points = sorted(points_by_substrate.get(edge.source, set()), key=lambda point: (point.y, point.x))
        if not source_points:
            return

        cppn = genome.get_link_cppn(edge.source, edge.target)
        outer_weight = float(genome.get_link_outer_weight(edge.source, edge.target))
        identity_mapping_enabled = bool(genome.get_link_identity_mapping(edge.source, edge.target))
        if target_is_output:
            search_result = self._search_output_edge(
                source_points=source_points,
                output_seed_points=target_spec.seed_points,
                cppn=cppn,
                target_depth=self._depth_for(target_spec, genome),
                identity_mapping_enabled=identity_mapping_enabled,
                genome_key=genome_key,
            )
        else:
            search_result = self.search_engine.explore_substrate(
                seed_points=source_points,
                cppn=cppn,
                output_points=[],
                blocked_points=[],
                depth=1,
                reverse=False,
                allow_target_links=True,
            )
            if not search_result.connections and edge.allow_identity_mapping and target_spec.seed_points:
                search_result = SearchResult(
                    discovered_points=[],
                    connections=self._aligned_seed_connections(
                        source_points=source_points,
                        target_points=target_spec.seed_points,
                        cppn=cppn,
                    ),
                )
            if not search_result.connections and target_spec.seed_points:
                # NOTE: This fallback is a practical guardrail, not canonical
                # ES-HyperNEAT. When quadtree search finds no connections, seed
                # points are linked directly so development does not collapse
                # into an empty phenotype. Long-term, LEO is the cleaner fix.
                search_result = SearchResult(
                    discovered_points=[],
                    connections=self._direct_seed_connections(
                        source_points=source_points,
                        target_points=target_spec.seed_points,
                        cppn=cppn,
                    ),
                )

        if self.config.debug_prune_log:
            print(
                    {
                    "genome_key": genome_key,
                    "k_hyperneat_edge_stage": "link_search",
                    "source": edge.source.kind,
                    "target": edge.target.kind,
                    "target_is_output": bool(target_is_output),
                    "source_points": len(source_points),
                    "target_seed_points": len(target_spec.seed_points),
                    "discovered_points": len(search_result.discovered_points),
                    "connections": len(search_result.connections),
                    "edge_base_weight": float(edge.base_weight),
                    "outer_weight": outer_weight,
                    "identity_mapping_enabled": identity_mapping_enabled,
                },
                file=sys.stderr,
                flush=True,
            )
        self._register_points(graph, edge.target, points_by_substrate[edge.target], search_result.discovered_points)
        self._add_cross_substrate_connections(graph, edge, search_result.connections, outer_weight)

    def _search_output_edge(
        self,
        *,
        source_points: list[Point2D],
        output_seed_points: list[Point2D],
        cppn: CppnModel,
        target_depth: int,
        identity_mapping_enabled: bool = False,
        genome_key: int | None = None,
    ) -> SearchResult:
        if target_depth <= 0:
            identity_used = False
            fallback_used = False
            result = SearchResult(discovered_points=[], connections=[])
            if identity_mapping_enabled:
                identity_used = True
                result = SearchResult(
                    discovered_points=[],
                    connections=self._aligned_seed_connections(
                        source_points=source_points,
                        target_points=output_seed_points,
                        cppn=cppn,
                    ),
                )
            if not result.connections:
                result = SearchResult(
                    discovered_points=[],
                    connections=self._direct_seed_connections(
                        source_points=source_points,
                        target_points=output_seed_points,
                        cppn=cppn,
                    ),
                )
            if not result.connections and output_seed_points:
                fallback_used = True
                if identity_mapping_enabled:
                    identity_used = True
                    result = SearchResult(
                        discovered_points=[],
                        connections=self._aligned_seed_connections(
                            source_points=source_points,
                            target_points=output_seed_points,
                            cppn=cppn,
                            force_expression=False,
                            force_threshold=0.0,
                        ),
                    )
                if not result.connections:
                    result = SearchResult(
                        discovered_points=[],
                        connections=self._direct_seed_connections(
                            source_points=source_points,
                            target_points=output_seed_points,
                            cppn=cppn,
                            force_expression=False,
                            force_threshold=0.0,
                        ),
                    )
            if self.config.debug_prune_log:
                print(
                    {
                        "genome_key": genome_key,
                        "k_hyperneat_output_edge_mode": "direct_seed",
                        "identity_mapping_enabled": identity_mapping_enabled,
                        "identity_used": identity_used,
                        "fallback_used": fallback_used,
                        "source_points": len(source_points),
                        "output_seed_points": len(output_seed_points),
                        "connections": len(result.connections),
                    },
                    file=sys.stderr,
                    flush=True,
                )
            return result

        reverse_result = self.search_engine.explore_substrate(
            seed_points=list(output_seed_points),
            cppn=cppn,
            output_points=[],
            blocked_points=[],
            depth=1,
            reverse=True,
            allow_target_links=True,
        )

        forward_result = self.search_engine.explore_substrate(
            seed_points=source_points,
            cppn=cppn,
            output_points=[],
            blocked_points=[],
            depth=1,
            reverse=False,
            allow_target_links=True,
        )
        output_seed_set = set(output_seed_points)
        merged_discovered = list(reverse_result.discovered_points)
        merged_connections = list(reverse_result.connections)
        merged_discovered.extend(
            point for point in forward_result.discovered_points if point not in output_seed_set
        )
        merged_connections.extend(
            connection
            for connection in forward_result.connections
            if connection.target not in output_seed_set
        )
        result = SearchResult(
            discovered_points=self._unique_points(merged_discovered),
            connections=self._unique_connections(merged_connections),
        )
        identity_used = False
        if not result.connections and output_seed_points and identity_mapping_enabled:
            identity_used = True
            result = SearchResult(
                discovered_points=[],
                connections=self._aligned_seed_connections(
                    source_points=source_points,
                    target_points=output_seed_points,
                    cppn=cppn,
                ),
            )
        fallback_used = False
        if not result.connections and output_seed_points:
            fallback_used = True
            if identity_mapping_enabled:
                identity_used = True
                result = SearchResult(
                    discovered_points=[],
                    connections=self._aligned_seed_connections(
                        source_points=source_points,
                        target_points=output_seed_points,
                        cppn=cppn,
                        force_expression=False,
                        force_threshold=0.0,
                    ),
                )
            if not result.connections:
                result = SearchResult(
                    discovered_points=[],
                    connections=self._direct_seed_connections(
                        source_points=source_points,
                        target_points=output_seed_points,
                        cppn=cppn,
                        force_expression=False,
                        force_threshold=0.0,
                    ),
                )
        if self.config.debug_prune_log:
            print(
                    {
                        "genome_key": genome_key,
                        "k_hyperneat_output_edge_mode": "reverse_forward_merge",
                        "source_points": len(source_points),
                        "output_seed_points": len(output_seed_points),
                        "target_depth": int(target_depth),
                        "identity_mapping_enabled": identity_mapping_enabled,
                        "identity_used": identity_used,
                        "reverse_discovered_points": len(reverse_result.discovered_points),
                        "reverse_connections": len(reverse_result.connections),
                        "forward_discovered_points": len(forward_result.discovered_points),
                        "forward_connections": len(forward_result.connections),
                        "fallback_used": fallback_used,
                        "merged_discovered_points": len(result.discovered_points),
                        "merged_connections": len(result.connections),
                    },
                file=sys.stderr,
                flush=True,
            )
        return result

    def _aligned_seed_connections(
        self,
        *,
        source_points: list[Point2D],
        target_points: list[Point2D],
        cppn: CppnModel,
        force_expression: bool | None = None,
        force_threshold: float | None = None,
    ) -> list[SearchConnection]:
        if not source_points or not target_points:
            return []
        candidates = sorted(target_points, key=lambda point: (point.x, point.y))
        connections: list[SearchConnection] = []
        for source in sorted(source_points, key=lambda point: (point.x, point.y)):
            target = min(
                candidates,
                key=lambda point: (abs(point.x - source.x), abs(point.y - source.y), point.x, point.y),
            )
            leo_enabled = self.config.search.leo_enabled if force_expression is None else force_expression
            threshold = self.config.search.weight_threshold if force_threshold is None else force_threshold
            final_weight = query_cppn_link(
                cppn,
                source,
                target,
                max_weight=self.config.search.max_weight,
                threshold=threshold,
                leo_enabled=leo_enabled,
                leo_threshold=self.config.search.leo_threshold,
            )
            if final_weight == 0.0:
                continue
            connections.append(
                SearchConnection(
                    source=source,
                    target=target,
                    weight=final_weight,
                )
            )
        return self._unique_connections(connections)

    def _direct_seed_connections(
        self,
        *,
        source_points: list[Point2D],
        target_points: list[Point2D],
        cppn: CppnModel,
        force_expression: bool | None = None,
        force_threshold: float | None = None,
    ) -> list[SearchConnection]:
        connections: list[SearchConnection] = []
        for source in source_points:
            source_connections: list[SearchConnection] = []
            for target in target_points:
                leo_enabled = self.config.search.leo_enabled if force_expression is None else force_expression
                threshold = self.config.search.weight_threshold if force_threshold is None else force_threshold
                final_weight = query_cppn_link(
                    cppn,
                    source,
                    target,
                    max_weight=self.config.search.max_weight,
                    threshold=threshold,
                    leo_enabled=leo_enabled,
                    leo_threshold=self.config.search.leo_threshold,
                )
                if final_weight == 0.0:
                    continue
                source_connections.append(
                    SearchConnection(
                        source=source,
                        target=target,
                        weight=final_weight,
                    )
                )
            if self.config.search.max_outgoing > 0 and len(source_connections) > self.config.search.max_outgoing:
                source_connections.sort(key=lambda item: abs(item.weight), reverse=True)
                source_connections = source_connections[: self.config.search.max_outgoing]
            connections.extend(source_connections)
        return self._unique_connections(connections)

    def _terminal_node_ids(
        self,
        graph: PhenotypeGraph,
        specs: list[SubstrateSpec],
    ) -> list[int]:
        terminal_ids: list[int] = []
        for spec in specs:
            for point in spec.seed_points:
                terminal_ids.append(graph.ensure_node(spec.ref, point))
        return terminal_ids

    def _unique_points(self, points: list[Point2D]) -> list[Point2D]:
        unique: dict[tuple[float, float], Point2D] = {}
        for point in points:
            unique[(point.x, point.y)] = point
        return sorted(unique.values(), key=lambda point: (point.y, point.x))

    def _unique_connections(self, connections: list[SearchConnection]) -> list[SearchConnection]:
        unique: dict[tuple[float, float, float, float], SearchConnection] = {}
        for connection in connections:
            unique[(
                connection.source.x,
                connection.source.y,
                connection.target.x,
                connection.target.y,
            )] = connection
        return list(unique.values())

    def _add_internal_connections(
        self,
        graph: PhenotypeGraph,
        substrate: SubstrateRef,
        connections: list[SearchConnection],
    ) -> None:
        for connection in connections:
            source_id = graph.ensure_node(substrate, connection.source)
            target_id = graph.ensure_node(substrate, connection.target)
            graph.add_edge(source_id, target_id, connection.weight)

    def _add_cross_substrate_connections(
        self,
        graph: PhenotypeGraph,
        edge: DevelopmentEdge,
        connections: list[SearchConnection],
        outer_weight: float,
    ) -> None:
        for connection in connections:
            final_weight = float(connection.weight) * float(edge.base_weight) * float(outer_weight)
            if final_weight == 0.0:
                continue
            source_id = graph.ensure_node(edge.source, connection.source)
            target_id = graph.ensure_node(edge.target, connection.target)
            graph.add_edge(source_id, target_id, final_weight)

    def _log_prune_state(
        self,
        stage: str,
        graph: PhenotypeGraph,
        topology: SubstrateTopology,
        genome_key: int | None,
    ) -> None:
        input_count = sum(len(spec.seed_points) for spec in topology.inputs)
        output_count = sum(len(spec.seed_points) for spec in topology.outputs)
        hidden_count = max(0, len(graph.nodes) - input_count - output_count)
        substrate_counts = Counter(node.substrate.key() for node in graph.nodes)
        print(
            {
                "genome_key": genome_key,
                "k_hyperneat_prune_stage": str(stage),
                "nodes": len(graph.nodes),
                "edges": len(graph.edges),
                "hidden_nodes": hidden_count,
                "by_substrate": dict(sorted(substrate_counts.items())),
            },
            file=sys.stderr,
            flush=True,
        )

    def _genome_debug_key(self, genome: DesGenomeProtocol) -> int | None:
        raw = getattr(genome, "genome_key", None)
        if raw is None:
            raw = getattr(genome, "key", None)
        if callable(raw):
            try:
                raw = raw()
            except Exception:
                raw = None
        try:
            return None if raw is None else int(raw)
        except Exception:
            return None
