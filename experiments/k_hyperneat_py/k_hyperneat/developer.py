from __future__ import annotations

"""Phenotype developer for DES-style substrate expansion.

This module takes a genome-provided substrate topology plus node/link CPPNs and
builds a concrete phenotype graph through ordered node growth, link search, and
reachable-path pruning.
"""

from dataclasses import dataclass
from typing import Protocol

from .config import DesHyperneatConfig
from .coordinates import Point2D
from .cppn import CppnModel, threshold_weight
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


@dataclass
class DesDeveloper:
    config: DesHyperneatConfig

    def __post_init__(self) -> None:
        self.search_engine = DesSearchEngine(self.config.search)

    def develop(self, genome: DesGenomeProtocol) -> PhenotypeGraph:
        topology = genome.topology()
        topology.validate()

        graph = PhenotypeGraph()
        spec_by_ref = topology.spec_by_ref()
        output_refs = {spec.ref for spec in topology.outputs}
        points_by_substrate = self._initialize_points(topology, graph)

        for action in topology.topological_actions():
            if isinstance(action, OrderedSubstrateNode):
                spec = spec_by_ref[action.ref]
                self._develop_node_substrate(
                    spec=spec,
                    genome=genome,
                    graph=graph,
                    points_by_substrate=points_by_substrate,
                    reverse=spec.reverse_search or spec.ref in output_refs,
                )
                continue
            if isinstance(action, OrderedSubstrateEdge):
                edge = action.edge
                self._develop_link_search(
                    edge=edge,
                    target_spec=spec_by_ref[edge.target],
                    genome=genome,
                    graph=graph,
                    points_by_substrate=points_by_substrate,
                    target_is_output=edge.target in output_refs,
                )

        graph.prune_reachable(
            self._terminal_node_ids(graph, topology.inputs),
            self._terminal_node_ids(graph, topology.outputs),
            preserve_terminal_nodes=True,
        )

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
    ) -> None:
        source_points = sorted(points_by_substrate.get(edge.source, set()), key=lambda point: (point.y, point.x))
        if not source_points:
            return

        cppn = genome.get_link_cppn(edge.source, edge.target)
        if target_is_output:
            search_result = self._search_output_edge(
                source_points=source_points,
                output_seed_points=target_spec.seed_points,
                cppn=cppn,
                target_depth=self._depth_for(target_spec, genome),
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

        self._register_points(graph, edge.target, points_by_substrate[edge.target], search_result.discovered_points)
        self._add_cross_substrate_connections(graph, edge, search_result.connections)

    def _search_output_edge(
        self,
        *,
        source_points: list[Point2D],
        output_seed_points: list[Point2D],
        cppn: CppnModel,
        target_depth: int,
    ) -> SearchResult:
        if target_depth <= 0:
            return SearchResult(
                discovered_points=[],
                connections=self._direct_seed_connections(
                    source_points=source_points,
                    target_points=output_seed_points,
                    cppn=cppn,
                ),
            )

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
        return SearchResult(
            discovered_points=self._unique_points(merged_discovered),
            connections=self._unique_connections(merged_connections),
        )

    def _direct_seed_connections(
        self,
        *,
        source_points: list[Point2D],
        target_points: list[Point2D],
        cppn: CppnModel,
    ) -> list[SearchConnection]:
        connections: list[SearchConnection] = []
        for source in source_points:
            for target in target_points:
                output = cppn.activate(source.as_cppn_input(target))
                raw_weight = float(output[0]) if output else 0.0
                final_weight = threshold_weight(
                    raw_weight,
                    threshold=self.config.search.weight_threshold,
                    max_weight=self.config.search.max_weight,
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
    ) -> None:
        for connection in connections:
            source_id = graph.ensure_node(edge.source, connection.source)
            target_id = graph.ensure_node(edge.target, connection.target)
            graph.add_edge(source_id, target_id, connection.weight)
