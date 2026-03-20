from k_hyperneat import (
    DesDeveloper,
    DesHyperneatConfig,
    DevelopmentEdge,
    LinkAction,
    Point2D,
    SearchConfig,
    SubstrateRef,
    SubstrateSpec,
    SubstrateTopology,
    compile_executor,
)


class GradientCppn:
    def activate(self, values):
        sx, sy, tx, ty, bias = values
        weight = (0.35 * sx) - (0.2 * sy) + (0.45 * tx) + (0.3 * ty) + (0.1 * bias)
        expression = 0.8 - abs(tx - sx) - (0.15 * abs(ty - sy))
        weight = max(-1.0, min(1.0, weight))
        expression = max(-1.0, min(1.0, expression))
        return [weight, expression]


class GenericGenome:
    def __init__(self):
        self._cppn = GradientCppn()
        self._input = SubstrateRef("input", 0)
        self._hidden = SubstrateRef("hidden", 0)
        self._output = SubstrateRef("output", 0)
        self._topology = SubstrateTopology(
            inputs=[
                SubstrateSpec(self._input, seed_points=[Point2D(-0.9, 0.9), Point2D(-0.3, 0.9)]),
            ],
            hidden=[
                SubstrateSpec(self._hidden, seed_points=[Point2D(-0.5, 0.0), Point2D(0.5, 0.0)]),
            ],
            outputs=[
                SubstrateSpec(self._output, seed_points=[Point2D(0.0, -0.9)]),
            ],
            links=[
                DevelopmentEdge(self._input, self._hidden),
                DevelopmentEdge(self._hidden, self._output),
            ],
        )

    def topology(self):
        return self._topology

    def get_node_cppn(self, substrate):
        _ = substrate
        return self._cppn

    def get_link_cppn(self, source, target):
        _ = source
        _ = target
        return self._cppn

    def get_depth(self, substrate):
        if substrate == self._hidden:
            return 1
        return 0

    def get_link_outer_weight(self, source, target):
        _ = source
        _ = target
        return 1.0

    def get_link_identity_mapping(self, source, target):
        _ = source
        _ = target
        return False


def main():
    config = DesHyperneatConfig(
        search=SearchConfig(
            leo_enabled=True,
            leo_threshold=0.0,
        )
    )
    genome = GenericGenome()
    graph = DesDeveloper(config).develop(genome)
    executor = compile_executor(graph, genome.topology(), config)
    link_count = sum(1 for action in executor.actions if isinstance(action, LinkAction))
    hidden_count = sum(1 for node in graph.nodes if node.substrate.kind.startswith("hidden"))
    print(
        {
            "leo_enabled": config.search.leo_enabled,
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "hidden_nodes": hidden_count,
            "non_activation_actions": link_count,
        }
    )


if __name__ == "__main__":
    main()
