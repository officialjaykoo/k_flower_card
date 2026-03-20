from __future__ import annotations

from dataclasses import dataclass

from .substrate import DevelopmentEdge, SubstrateRef, SubstrateSpec, SubstrateTopology
from .link import edge_key


@dataclass(frozen=True)
class TopologyGraphNode:
    ref: SubstrateRef
    spec: SubstrateSpec
    role: str
    seed: bool
    innovation: int
    depth: int
    enabled: bool
    state_key: str
    origin: str


@dataclass(frozen=True)
class TopologyGraphLink:
    edge: DevelopmentEdge
    seed: bool
    innovation: int
    depth: int
    enabled: bool
    outer_weight: float
    identity_mapping_enabled: bool
    state_key: str
    state_redirect_key: str | None
    origin: str
    cloned_from_key: str | None


@dataclass(frozen=True)
class TopologyGraph:
    nodes: dict[str, TopologyGraphNode]
    links: dict[str, TopologyGraphLink]

    @classmethod
    def from_substrate_topology(cls, topology: SubstrateTopology) -> "TopologyGraph":
        nodes: dict[str, TopologyGraphNode] = {}
        for spec in topology.inputs:
            nodes[str(spec.ref.kind)] = TopologyGraphNode(
                ref=spec.ref,
                spec=spec,
                role="input",
                seed=True,
                innovation=0,
                depth=int(spec.depth),
                enabled=True,
                state_key=str(spec.ref.kind),
                origin="seed",
            )
        for spec in topology.hidden:
            nodes[str(spec.ref.kind)] = TopologyGraphNode(
                ref=spec.ref,
                spec=spec,
                role="hidden",
                seed=True,
                innovation=0,
                depth=int(spec.depth),
                enabled=True,
                state_key=str(spec.ref.kind),
                origin="seed",
            )
        for spec in topology.outputs:
            nodes[str(spec.ref.kind)] = TopologyGraphNode(
                ref=spec.ref,
                spec=spec,
                role="output",
                seed=True,
                innovation=0,
                depth=int(spec.depth),
                enabled=True,
                state_key=str(spec.ref.kind),
                origin="seed",
            )
        links: dict[str, TopologyGraphLink] = {}
        for edge in topology.links:
            key = edge_key(edge.source, edge.target)
            links[key] = TopologyGraphLink(
                edge=edge,
                seed=True,
                innovation=0,
                depth=0,
                enabled=True,
                outer_weight=float(edge.base_weight),
                identity_mapping_enabled=bool(edge.allow_identity_mapping),
                state_key=key,
                state_redirect_key=None,
                origin="seed",
                cloned_from_key=None,
            )
        return cls(nodes=nodes, links=links)

    def node_items(self, role: str | None = None):
        for kind, node in self.nodes.items():
            if role is not None and node.role != role:
                continue
            yield kind, node

    def nodes_by_role(self, role: str) -> dict[str, TopologyGraphNode]:
        return {kind: node for kind, node in self.node_items(role)}

    def all_nodes(self) -> list[TopologyGraphNode]:
        return list(self.nodes.values())

    def enabled_hidden_nodes(self) -> list[TopologyGraphNode]:
        return [node for _, node in self.node_items("hidden") if node.enabled]

    def disabled_hidden_nodes(self) -> list[TopologyGraphNode]:
        return [node for _, node in self.node_items("hidden") if not node.enabled]

    def seed_node_keys(self, role: str | None = None) -> list[str]:
        return [kind for kind, node in self.node_items(role) if node.seed]

    def dynamic_node_keys(self, role: str | None = None) -> list[str]:
        return [kind for kind, node in self.node_items(role) if not node.seed]

    def seed_link_keys(self) -> list[str]:
        return [key for key, link in self.links.items() if link.seed]

    def dynamic_link_keys(self) -> list[str]:
        return [key for key, link in self.links.items() if not link.seed]

    def enabled_links(self) -> list[TopologyGraphLink]:
        enabled_refs = {node.ref for node in self.all_nodes() if node.enabled}
        links: list[TopologyGraphLink] = []
        for link in self.links.values():
            if not link.enabled:
                continue
            if link.edge.source not in enabled_refs or link.edge.target not in enabled_refs:
                continue
            links.append(link)
        return links

    def disabled_links(self) -> list[TopologyGraphLink]:
        return [link for link in self.links.values() if not link.enabled]

    def incoming_enabled_links(self) -> dict[SubstrateRef, list[TopologyGraphLink]]:
        incoming = {node.ref: [] for node in self.all_nodes() if node.enabled}
        for link in self.enabled_links():
            incoming.setdefault(link.edge.target, []).append(link)
        return incoming

    def outgoing_enabled_links(self) -> dict[SubstrateRef, list[TopologyGraphLink]]:
        outgoing = {node.ref: [] for node in self.all_nodes() if node.enabled}
        for link in self.enabled_links():
            outgoing.setdefault(link.edge.source, []).append(link)
        return outgoing

    def orphan_hidden_refs(self) -> set[SubstrateRef]:
        incoming = self.incoming_enabled_links()
        outgoing = self.outgoing_enabled_links()
        result: set[SubstrateRef] = set()
        for node in self.enabled_hidden_nodes():
            if not incoming.get(node.ref) or not outgoing.get(node.ref):
                result.add(node.ref)
        return result

    def reachable_refs(self) -> tuple[set[SubstrateRef], set[SubstrateRef]]:
        enabled_links = self.enabled_links()
        forward: dict[SubstrateRef, list[SubstrateRef]] = {}
        reverse: dict[SubstrateRef, list[SubstrateRef]] = {}
        for link in enabled_links:
            forward.setdefault(link.edge.source, []).append(link.edge.target)
            reverse.setdefault(link.edge.target, []).append(link.edge.source)

        from_inputs = {node.ref for _, node in self.node_items("input") if node.enabled}
        queue = list(from_inputs)
        while queue:
            ref = queue.pop(0)
            for target in forward.get(ref, []):
                if target in from_inputs:
                    continue
                from_inputs.add(target)
                queue.append(target)

        to_outputs = {node.ref for _, node in self.node_items("output") if node.enabled}
        queue = list(to_outputs)
        while queue:
            ref = queue.pop(0)
            for source in reverse.get(ref, []):
                if source in to_outputs:
                    continue
                to_outputs.add(source)
                queue.append(source)

        return from_inputs, to_outputs

    def candidate_add_links(self) -> list[TopologyGraphLink]:
        enabled_refs = {node.ref for node in self.all_nodes() if node.enabled}
        return [
            link
            for link in self.disabled_links()
            if link.edge.source in enabled_refs and link.edge.target in enabled_refs
        ]

    def candidate_remove_links(self) -> list[TopologyGraphLink]:
        incoming = self.incoming_enabled_links()
        outgoing = self.outgoing_enabled_links()
        result: list[TopologyGraphLink] = []
        for link in self.enabled_links():
            source_outgoing = len(outgoing.get(link.edge.source, []))
            target_incoming = len(incoming.get(link.edge.target, []))
            if source_outgoing > 1 and target_incoming > 1:
                result.append(link)
        return result

    def candidate_split_hidden_events(self) -> list[TopologyGraphLink]:
        return [
            link
            for link in self.enabled_links()
            if link.edge.source.kind.startswith("hidden_") or link.edge.target.kind.startswith("hidden_")
        ]

    def stats(self) -> dict[str, int]:
        input_nodes = self.nodes_by_role("input")
        hidden_nodes = self.nodes_by_role("hidden")
        output_nodes = self.nodes_by_role("output")
        return {
            "input_nodes": len(input_nodes),
            "hidden_nodes": len(hidden_nodes),
            "output_nodes": len(output_nodes),
            "hidden_enabled": sum(1 for node in hidden_nodes.values() if node.enabled),
            "seed_nodes": sum(1 for node in self.nodes.values() if node.seed),
            "dynamic_nodes": sum(1 for node in self.nodes.values() if not node.seed),
            "link_total": len(self.links),
            "link_enabled": len(self.enabled_links()),
            "seed_links": sum(1 for link in self.links.values() if link.seed),
            "dynamic_links": sum(1 for link in self.links.values() if not link.seed),
            "candidate_add_link": len(self.candidate_add_links()),
            "candidate_remove_link": len(self.candidate_remove_links()),
            "candidate_split_hidden": len(self.candidate_split_hidden_events()),
        }

    def to_substrate_topology(self) -> SubstrateTopology:
        inputs = [node.spec for _, node in self.node_items("input") if node.enabled]
        outputs = [node.spec for _, node in self.node_items("output") if node.enabled]
        hidden = [node.spec for _, node in self.node_items("hidden") if node.enabled]
        enabled_refs = {spec.ref for spec in [*inputs, *hidden, *outputs]}
        links = [
            link.edge
            for link in self.enabled_links()
            if link.edge.source in enabled_refs and link.edge.target in enabled_refs
        ]
        return SubstrateTopology(inputs=inputs, hidden=hidden, outputs=outputs, links=links)
