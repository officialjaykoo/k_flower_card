from k_hyperneat import (
    DesDeveloper,
    DesHyperneatConfig,
    DevelopmentEdge,
    Point2D,
    SubstrateRef,
    SubstrateSpec,
    SubstrateTopology,
)


class GradientCppn:
    def activate(self, values):
        sx, sy, tx, ty, bias = values
        raw = (0.35 * sx) - (0.2 * sy) + (0.45 * tx) + (0.3 * ty) + (0.1 * bias)
        if raw <= -1.0:
            return [-1.0]
        if raw >= 1.0:
            return [1.0]
        return [raw]


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


def main():
    graph = DesDeveloper(DesHyperneatConfig()).develop(GenericGenome())
    print({"nodes": len(graph.nodes), "edges": len(graph.edges)})


if __name__ == "__main__":
    main()
