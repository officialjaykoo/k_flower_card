import json
from pathlib import Path
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parent
FORK_ROOT = EXPERIMENT_ROOT.parents[1] / "Des-HyperNEAT-Python"
if str(FORK_ROOT) not in sys.path:
    sys.path.insert(0, str(FORK_ROOT))
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from deshyperneat import Config, Graph, Point2D, SubstrateRef, compile_executor
from local.matgo.topology import build_minimal_matgo_topology


def main():
    cfg = Config()
    topology = build_minimal_matgo_topology()
    graph = Graph()
    hidden_id = graph.ensure_node(SubstrateRef("hidden", 0), Point2D(0.0, 0.0))

    input_nodes = []
    output_nodes = []
    for spec in topology.inputs:
        for point in spec.seed_points:
            input_nodes.append(graph.ensure_node(spec.ref, point))
    for spec in topology.outputs:
        for point in spec.seed_points:
            output_nodes.append(graph.ensure_node(spec.ref, point))

    for index, node_id in enumerate(input_nodes):
        graph.add_edge(node_id, hidden_id, 0.05 * (index + 1))
    for index, node_id in enumerate(output_nodes):
        graph.add_edge(hidden_id, node_id, 0.02 * (index + 1))

    runtime = compile_executor(graph, topology, cfg).to_runtime_dict(
        topology,
        adapter_kind="matgo_minimal_v1",
    )

    out_path = Path(__file__).with_name("smoke_runtime.json")
    out_path.write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    print(
        {
            "out": str(out_path),
            "nodes": runtime["node_count"],
            "actions": len(runtime["actions"]),
            "inputs": len(runtime.get("input_node_ids", [])),
            "outputs": len(runtime.get("output_node_ids", [])),
        }
    )


if __name__ == "__main__":
    main()
