from pathlib import Path
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parent
FORK_ROOT = EXPERIMENT_ROOT.parents[1] / "Des-HyperNEAT-Python"
if str(FORK_ROOT) not in sys.path:
    sys.path.insert(0, str(FORK_ROOT))
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from deshyperneat import Config, Developer, SearchConfig
from local.matgo.controls import MatgoTopologyControl
from local.matgo.topology import build_minimal_matgo_topology
from local.matgo.ini import load_genome_config
from deshyperneat.genome import Genome


def main():
    topology = build_minimal_matgo_topology()
    control = MatgoTopologyControl.from_runtime(
        topology,
        {
            "max_hidden_substrate_depth": 5,
            "depth_overrides": {
                "hidden_play_context": 2,
                "hidden_match_context": 2,
                "hidden_option_context": 1,
            },
        },
    )
    config = load_genome_config(
        FORK_ROOT / "configs" / "deshyperneat_cppn.ini",
        topology=topology,
        des_runtime={
            "static_substrate_depth": control.static_substrate_depth,
            "max_input_substrate_depth": control.max_input_substrate_depth,
            "max_hidden_substrate_depth": control.max_hidden_substrate_depth,
            "max_output_substrate_depth": control.max_output_substrate_depth,
            "depth_overrides": {
                str(kind): int(node_control.depth)
                for kind, node_control in sorted(control.node_controls.items())
            },
            "edge_outer_weights": {
                str(key): float(edge_control.outer_weight)
                for key, edge_control in sorted(control.edge_controls.items())
            },
            "identity_mapping_edges": [
                str(key)
                for key, edge_control in sorted(control.edge_controls.items())
                if bool(edge_control.allow_identity_mapping)
            ],
        },
    )

    genome = Genome(1)
    genome.configure_new(config)
    active_topology = genome.topology()
    topology_graph = genome.topology_graph()
    before_stats = topology_graph.stats()
    event_history = []
    for _ in range(16):
        genome.mutate(config)
        events = list(genome.export_components().get("last_topology_events") or [])
        if events:
            event_history = events
            break
    active_topology_after = genome.topology()
    topology_graph_after = genome.topology_graph()
    exported = genome.export_components()
    state_snapshot = dict(exported.get("cppn_state") or {})

    graph = Developer(
        Config(
            search=SearchConfig(
                leo_enabled=True,
                leo_threshold=0.0,
            )
        )
    ).develop(genome)
    print(
        {
            "input_node_genes": genome._node_gene_count("input"),
            "hidden_node_genes": genome._node_gene_count("hidden"),
            "output_node_genes": genome._node_gene_count("output"),
            "link_genes": len(genome._links),
            "hidden_enabled": sum(1 for gene in genome._iter_node_genes("hidden") if gene.enabled),
            "link_enabled": sum(1 for gene in genome._links.values() if gene.enabled),
            "identity_link_genes": sum(1 for gene in genome._links.values() if gene.identity_mapping_enabled),
            "active_hidden_specs": len(active_topology.hidden),
            "active_links": len(active_topology.links),
            "graph_hidden_enabled": before_stats["hidden_enabled"],
            "graph_link_enabled": before_stats["link_enabled"],
            "graph_hidden_enabled_after_mutate": topology_graph_after.stats()["hidden_enabled"],
            "graph_link_enabled_after_mutate": topology_graph_after.stats()["link_enabled"],
            "active_hidden_specs_after_mutate": len(active_topology_after.hidden),
            "active_links_after_mutate": len(active_topology_after.links),
            "last_topology_events": event_history,
            "all_node_innovations": len(exported.get("topology_innovations", {}).get("all_node_innovations") or []),
            "all_link_innovations": len(exported.get("topology_innovations", {}).get("all_link_innovations") or []),
            "active_node_innovations": len(exported.get("topology_innovations", {}).get("active_node_innovations") or []),
            "active_link_innovations": len(exported.get("topology_innovations", {}).get("active_link_innovations") or []),
            "node_component_states": len(state_snapshot.get("node_component_states") or {}),
            "link_component_states": len(state_snapshot.get("link_component_states") or {}),
            "component_state_redirects": len(state_snapshot.get("component_state_redirects") or {}),
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
        }
    )


if __name__ == "__main__":
    main()
