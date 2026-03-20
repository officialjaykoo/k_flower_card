from __future__ import annotations

from dataclasses import dataclass, field

from .coordinates import Point2D


@dataclass(frozen=True)
class SubstrateRef:
    kind: str
    index: int

    def key(self) -> str:
        return f"{self.kind}:{self.index}"


@dataclass(frozen=True)
class DevelopmentEdge:
    source: SubstrateRef
    target: SubstrateRef
    base_weight: float = 1.0
    allow_identity_mapping: bool = False


@dataclass(frozen=True)
class OrderedSubstrateNode:
    ref: SubstrateRef


@dataclass(frozen=True)
class OrderedSubstrateEdge:
    edge: DevelopmentEdge


@dataclass
class SubstrateSpec:
    ref: SubstrateRef
    seed_points: list[Point2D] = field(default_factory=list)
    depth: int = 0
    reverse_search: bool = False
    allow_identity_links: bool = False


@dataclass
class SubstrateTopology:
    inputs: list[SubstrateSpec]
    outputs: list[SubstrateSpec]
    hidden: list[SubstrateSpec]
    links: list[DevelopmentEdge]

    def all_specs(self) -> list[SubstrateSpec]:
        return [*self.inputs, *self.hidden, *self.outputs]

    def spec_by_ref(self) -> dict[SubstrateRef, SubstrateSpec]:
        return {spec.ref: spec for spec in self.all_specs()}

    def outgoing_edges(self) -> dict[SubstrateRef, list[DevelopmentEdge]]:
        outgoing = {spec.ref: [] for spec in self.all_specs()}
        for edge in self.links:
            outgoing[edge.source].append(edge)
        for edges in outgoing.values():
            edges.sort(
                key=lambda edge: (
                    edge.target.kind,
                    edge.target.index,
                    edge.source.kind,
                    edge.source.index,
                )
            )
        return outgoing

    def topological_actions(self) -> list[OrderedSubstrateNode | OrderedSubstrateEdge]:
        incoming = {spec.ref: 0 for spec in self.all_specs()}
        outgoing = self.outgoing_edges()
        for edge in self.links:
            incoming[edge.target] = incoming.get(edge.target, 0) + 1

        ready = sorted(
            [ref for ref, count in incoming.items() if count == 0],
            key=lambda ref: (ref.kind, ref.index),
        )
        processed: set[SubstrateRef] = set()
        actions: list[OrderedSubstrateNode | OrderedSubstrateEdge] = []

        while ready:
            ref = ready.pop(0)
            if ref in processed:
                continue
            processed.add(ref)
            actions.append(OrderedSubstrateNode(ref))
            for edge in outgoing.get(ref, []):
                actions.append(OrderedSubstrateEdge(edge))
                incoming[edge.target] -= 1
                if incoming[edge.target] == 0:
                    ready.append(edge.target)
                    ready.sort(key=lambda item: (item.kind, item.index))

        if len(processed) != len(incoming):
            raise ValueError("substrate topology contains a cycle")
        return actions

    def validate(self) -> None:
        refs = [spec.ref for spec in self.all_specs()]
        if len(set(refs)) != len(refs):
            raise ValueError("duplicate substrate refs detected")
        known = set(refs)
        for edge in self.links:
            if edge.source not in known:
                raise ValueError(f"unknown source substrate: {edge.source}")
            if edge.target not in known:
                raise ValueError(f"unknown target substrate: {edge.target}")
