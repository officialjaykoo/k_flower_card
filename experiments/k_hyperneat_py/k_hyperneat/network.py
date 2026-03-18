from __future__ import annotations

from dataclasses import dataclass, field

from .coordinates import Point2D
from .substrate import SubstrateRef


@dataclass(frozen=True)
class PhenotypeNode:
    node_id: int
    substrate: SubstrateRef
    point: Point2D


@dataclass(frozen=True)
class PhenotypeEdge:
    source_id: int
    target_id: int
    weight: float


@dataclass
class PhenotypeGraph:
    nodes: list[PhenotypeNode] = field(default_factory=list)
    edges: list[PhenotypeEdge] = field(default_factory=list)
    _node_lookup: dict[tuple[str, int, float, float], int] = field(default_factory=dict)
    _edge_lookup: dict[tuple[int, int], int] = field(default_factory=dict)

    def ensure_node(self, substrate: SubstrateRef, point: Point2D) -> int:
        key = (substrate.kind, substrate.index, point.x, point.y)
        node_id = self._node_lookup.get(key)
        if node_id is not None:
            return node_id
        node_id = len(self.nodes)
        self.nodes.append(PhenotypeNode(node_id=node_id, substrate=substrate, point=point))
        self._node_lookup[key] = node_id
        return node_id

    def add_edge(self, source_id: int, target_id: int, weight: float) -> None:
        if source_id == target_id:
            return
        key = (source_id, target_id)
        edge_index = self._edge_lookup.get(key)
        edge = PhenotypeEdge(source_id=source_id, target_id=target_id, weight=float(weight))
        if edge_index is not None:
            self.edges[edge_index] = edge
            return
        self._edge_lookup[key] = len(self.edges)
        self.edges.append(edge)

    def node_ids_for_refs(self, refs: set[SubstrateRef]) -> list[int]:
        return [node.node_id for node in self.nodes if node.substrate in refs]

    def prune_reachable(
        self,
        input_ids: list[int],
        output_ids: list[int],
        *,
        preserve_terminal_nodes: bool = True,
    ) -> set[int]:
        if not self.nodes:
            return set()

        forward_adj: dict[int, list[int]] = {}
        reverse_adj: dict[int, list[int]] = {}
        for edge in self.edges:
            forward_adj.setdefault(edge.source_id, []).append(edge.target_id)
            reverse_adj.setdefault(edge.target_id, []).append(edge.source_id)

        forward = self._reachable(set(input_ids), forward_adj)
        backward = self._reachable(set(output_ids), reverse_adj)
        keep = forward & backward
        if preserve_terminal_nodes:
            keep.update(input_ids)
            keep.update(output_ids)

        removed = {node.node_id for node in self.nodes if node.node_id not in keep}
        if not removed:
            return removed

        new_nodes: list[PhenotypeNode] = []
        new_lookup: dict[tuple[str, int, float, float], int] = {}
        node_map: dict[int, int] = {}
        for node in self.nodes:
            if node.node_id not in keep:
                continue
            new_id = len(new_nodes)
            new_nodes.append(
                PhenotypeNode(
                    node_id=new_id,
                    substrate=node.substrate,
                    point=node.point,
                )
            )
            new_lookup[(node.substrate.kind, node.substrate.index, node.point.x, node.point.y)] = new_id
            node_map[node.node_id] = new_id

        new_edges: list[PhenotypeEdge] = []
        new_edge_lookup: dict[tuple[int, int], int] = {}
        for edge in self.edges:
            if edge.source_id not in keep or edge.target_id not in keep:
                continue
            source_id = node_map[edge.source_id]
            target_id = node_map[edge.target_id]
            key = (source_id, target_id)
            if key in new_edge_lookup:
                new_edges[new_edge_lookup[key]] = PhenotypeEdge(source_id, target_id, edge.weight)
                continue
            new_edge_lookup[key] = len(new_edges)
            new_edges.append(PhenotypeEdge(source_id, target_id, edge.weight))

        self.nodes = new_nodes
        self.edges = new_edges
        self._node_lookup = new_lookup
        self._edge_lookup = new_edge_lookup
        return removed

    @staticmethod
    def _reachable(start_nodes: set[int], adjacency: dict[int, list[int]]) -> set[int]:
        visited = set(start_nodes)
        stack = list(start_nodes)
        while stack:
            node_id = stack.pop()
            for next_id in adjacency.get(node_id, []):
                if next_id in visited:
                    continue
                visited.add(next_id)
                stack.append(next_id)
        return visited
