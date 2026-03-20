from pathlib import Path
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parent
FORK_ROOT = EXPERIMENT_ROOT.parents[1] / "Des-HyperNEAT-Python"
if str(FORK_ROOT) not in sys.path:
    sys.path.insert(0, str(FORK_ROOT))
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from deshyperneat import Config, Developer
from local.matgo.topology import build_minimal_matgo_topology


class GradientCppn:
    def activate(self, values):
        sx, sy, tx, ty, bias = values
        raw = (0.35 * sx) - (0.25 * sy) + (0.55 * tx) + (0.45 * ty) + (0.1 * bias)
        if raw <= -1.0:
            return [-1.0]
        if raw >= 1.0:
            return [1.0]
        return [raw]


class StaticGenome:
    def __init__(self):
        self._topology = build_minimal_matgo_topology()
        self._cppn = GradientCppn()

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
        _ = substrate
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
    graph = Developer(Config()).develop(StaticGenome())
    print({"nodes": len(graph.nodes), "edges": len(graph.edges)})


if __name__ == "__main__":
    main()
