from __future__ import annotations

from dataclasses import dataclass
import math

from .config import Config
from .network import Graph
from .substrate import SubstrateSpec, SubstrateTopology


def _activate(name: str, value: float) -> float:
    activation = str(name or "identity").strip().lower()
    x = float(value)
    if activation in {"identity", "linear", "none"}:
        return x
    if activation == "sigmoid":
        return 1.0 / (1.0 + math.exp(-x))
    if activation == "relu":
        return max(0.0, x)
    if activation == "step":
        return 1.0 if x > 0.0 else 0.0
    if activation == "gaussian":
        return math.exp(-((2.5 * x) ** 2))
    if activation == "offsetgaussian":
        return 2.0 * math.exp(-((2.5 * x) ** 2)) - 1.0
    if activation == "sine":
        return math.sin(2.0 * x)
    if activation == "cos":
        return math.cos(2.0 * x)
    if activation == "square":
        return x * x
    if activation == "abs":
        return abs(x)
    if activation == "exp":
        return math.exp(min(1.0, x))
    return math.tanh(x)


@dataclass(frozen=True)
class LinkAction:
    source_id: int
    target_id: int
    weight: float


@dataclass(frozen=True)
class ActivationAction:
    node_id: int
    bias: float
    activation: str


@dataclass
class Executor:
    node_count: int
    input_node_ids: list[int]
    output_node_ids: list[int]
    actions: list[ActivationAction | LinkAction]

    def run(self, inputs: list[float]) -> list[float]:
        values = [0.0] * self.node_count
        for index, node_id in enumerate(self.input_node_ids):
            values[node_id] = float(inputs[index] if index < len(inputs) else 0.0)
        for action in self.actions:
            if isinstance(action, LinkAction):
                values[action.target_id] += values[action.source_id] * action.weight
            else:
                values[action.node_id] = _activate(
                    action.activation,
                    values[action.node_id] + action.bias,
                )
        return [
            values[node_id] if math.isfinite(values[node_id]) else 0.0
            for node_id in self.output_node_ids
        ]

    def to_runtime_dict(
        self,
        topology: SubstrateTopology | None = None,
        *,
        adapter_kind: str | None = None,
    ) -> dict:
        payload = {
            "format_version": "k_hyperneat_executor_v1",
            "node_count": self.node_count,
            "input_node_ids": list(self.input_node_ids),
            "output_node_ids": list(self.output_node_ids),
            "actions": [
                (
                    {
                        "kind": "activation",
                        "node_id": action.node_id,
                        "bias": action.bias,
                        "activation": action.activation,
                    }
                    if isinstance(action, ActivationAction)
                    else {
                        "kind": "link",
                        "source_id": action.source_id,
                        "target_id": action.target_id,
                        "weight": action.weight,
                    }
                )
                for action in self.actions
            ],
        }
        if topology is not None:
            payload["adapter"] = {
                "kind": str(adapter_kind or "generic_substrate_v1"),
                "inputs": _serialize_specs(topology.inputs),
                "outputs": _serialize_specs(topology.outputs),
            }
        return payload


def compile_executor(
    graph: Graph,
    topology: SubstrateTopology,
    config: Config,
) -> Executor:
    input_node_ids = _seed_node_ids(graph, topology.inputs)
    output_node_ids = _seed_node_ids(graph, topology.outputs)
    input_node_set = set(input_node_ids)
    output_node_set = set(output_node_ids)

    outgoing: dict[int, list[tuple[int, float]]] = {node.node_id: [] for node in graph.nodes}
    indegree: dict[int, int] = {node.node_id: 0 for node in graph.nodes}
    for edge in graph.edges:
        outgoing.setdefault(edge.source_id, []).append((edge.target_id, edge.weight))
        indegree[edge.target_id] = indegree.get(edge.target_id, 0) + 1
        indegree.setdefault(edge.source_id, 0)

    for targets in outgoing.values():
        targets.sort(key=lambda item: item[0])

    ready = sorted(node_id for node_id, count in indegree.items() if count == 0)
    actions: list[ActivationAction | LinkAction] = []
    processed: list[int] = []
    while ready:
        node_id = ready.pop(0)
        processed.append(node_id)
        activation = (
            "identity"
            if node_id in input_node_set
            else config.output_activation
            if node_id in output_node_set
            else config.hidden_activation
        )
        actions.append(ActivationAction(node_id=node_id, bias=0.0, activation=activation))
        for target_id, weight in outgoing.get(node_id, []):
            actions.append(LinkAction(source_id=node_id, target_id=target_id, weight=weight))
            indegree[target_id] -= 1
            if indegree[target_id] == 0:
                ready.append(target_id)
                ready.sort()

    if len(processed) != len(graph.nodes):
        raise ValueError("phenotype graph contains a cycle or unreachable ordering state")

    return Executor(
        node_count=len(graph.nodes),
        input_node_ids=input_node_ids,
        output_node_ids=output_node_ids,
        actions=actions,
    )


def _seed_node_ids(graph: Graph, specs: list[SubstrateSpec]) -> list[int]:
    node_ids: list[int] = []
    for spec in specs:
        for point in spec.seed_points:
            node_ids.append(graph.ensure_node(spec.ref, point))
    return node_ids


def _serialize_specs(specs: list[SubstrateSpec]) -> list[dict]:
    return [
        {
            "kind": spec.ref.kind,
            "index": spec.ref.index,
            "slot_count": len(spec.seed_points),
        }
        for spec in specs
    ]
__all__ = ["ActivationAction", "Executor", "LinkAction", "compile_executor"]
