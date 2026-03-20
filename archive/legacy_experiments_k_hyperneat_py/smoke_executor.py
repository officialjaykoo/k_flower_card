from k_hyperneat import (
    DesHyperneatConfig,
    Point2D,
    PhenotypeGraph,
    SubstrateRef,
    SubstrateSpec,
    SubstrateTopology,
    compile_executor,
)


def main():
    cfg = DesHyperneatConfig()
    topology = SubstrateTopology(
        inputs=[SubstrateSpec(SubstrateRef("input", 0), seed_points=[Point2D(-0.8, 0.8)])],
        hidden=[],
        outputs=[SubstrateSpec(SubstrateRef("output", 0), seed_points=[Point2D(0.8, -0.8)])],
        links=[],
    )
    graph = PhenotypeGraph()
    input_id = graph.ensure_node(SubstrateRef("input", 0), Point2D(-0.8, 0.8))
    hidden_id = graph.ensure_node(SubstrateRef("hidden", 0), Point2D(0.0, 0.0))
    output_id = graph.ensure_node(SubstrateRef("output", 0), Point2D(0.8, -0.8))
    graph.add_edge(input_id, hidden_id, 1.5)
    graph.add_edge(hidden_id, output_id, 2.0)
    executor = compile_executor(graph, topology, cfg)
    outputs = executor.run([1.0])
    print(
        {
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "executor_actions": len(executor.actions),
            "outputs": [round(value, 6) for value in outputs],
        }
    )


if __name__ == "__main__":
    main()
