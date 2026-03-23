from __future__ import annotations

from dataclasses import dataclass, field

from deshyperneat import SubstrateRef, SubstrateTopology
from src.deshyperneat.link import edge_key


@dataclass(frozen=True)
class MatgoQueryControl:
    weight_scale: float = 1.0
    weight_bias: float = 0.0
    expression_scale: float = 1.0
    expression_bias: float = 0.0


@dataclass(frozen=True)
class MatgoNodeControl:
    kind: str
    depth: int
    query: MatgoQueryControl = MatgoQueryControl()


@dataclass(frozen=True)
class MatgoEdgeControl:
    key: str
    outer_weight: float = 1.0
    allow_identity_mapping: bool = False
    query: MatgoQueryControl = MatgoQueryControl()


@dataclass
class MatgoTopologyControl:
    static_substrate_depth: int = -1
    max_input_substrate_depth: int = 0
    max_hidden_substrate_depth: int = 5
    max_output_substrate_depth: int = 0
    node_controls: dict[str, MatgoNodeControl] = field(default_factory=dict)
    edge_controls: dict[str, MatgoEdgeControl] = field(default_factory=dict)

    @classmethod
    def from_runtime(cls, topology: SubstrateTopology, des_runtime: dict[str, object]) -> "MatgoTopologyControl":
        raw_runtime = dict(des_runtime or {})
        raw_depth_overrides = dict(raw_runtime.get("depth_overrides") or {})
        raw_edge_outer_weights = dict(raw_runtime.get("edge_outer_weights") or {})
        raw_identity_mapping_edges = {
            str(item or "").strip()
            for item in list(raw_runtime.get("identity_mapping_edges") or [])
            if str(item or "").strip()
        }
        raw_node_query_controls = dict(raw_runtime.get("node_query_controls") or {})
        raw_edge_query_controls = dict(raw_runtime.get("edge_query_controls") or {})
        node_controls = {
            str(kind): MatgoNodeControl(
                kind=str(kind),
                depth=max(0, int(depth or 0)),
                query=_query_control_from_raw(
                    raw_node_query_controls.get(str(kind)) or {},
                    _default_node_query_control(str(kind)),
                ),
            )
            for kind, depth in raw_depth_overrides.items()
            if str(kind or "").strip()
        }
        for spec in topology.hidden:
            kind = str(spec.ref.kind)
            if kind in node_controls:
                continue
            node_controls[kind] = MatgoNodeControl(
                kind=kind,
                depth=max(0, int(spec.depth or 0)),
                query=_query_control_from_raw(
                    raw_node_query_controls.get(kind) or {},
                    _default_node_query_control(kind),
                ),
            )
        edge_controls: dict[str, MatgoEdgeControl] = {}
        for edge in topology.links:
            key = edge_key(edge.source, edge.target)
            source_pattern = f"{edge.source.kind}->*"
            target_pattern = f"*->{edge.target.kind}"
            outer_weight = float(
                raw_edge_outer_weights.get(
                    key,
                    raw_edge_outer_weights.get(
                        source_pattern,
                        raw_edge_outer_weights.get(target_pattern, 1.0),
                    ),
                )
                or 1.0
            )
            allow_identity_mapping = bool(
                edge.allow_identity_mapping
                or key in raw_identity_mapping_edges
                or source_pattern in raw_identity_mapping_edges
                or target_pattern in raw_identity_mapping_edges
            )
            edge_controls[key] = MatgoEdgeControl(
                key=key,
                outer_weight=outer_weight,
                allow_identity_mapping=allow_identity_mapping,
                query=_query_control_from_raw(
                    raw_edge_query_controls.get(key)
                    or raw_edge_query_controls.get(source_pattern)
                    or raw_edge_query_controls.get(target_pattern)
                    or {},
                    _default_edge_query_control(edge.source.kind, edge.target.kind),
                ),
            )
        return cls(
            static_substrate_depth=int(raw_runtime.get("static_substrate_depth", -1) or -1),
            max_input_substrate_depth=max(0, int(raw_runtime.get("max_input_substrate_depth", 0) or 0)),
            max_hidden_substrate_depth=max(0, int(raw_runtime.get("max_hidden_substrate_depth", 5) or 5)),
            max_output_substrate_depth=max(0, int(raw_runtime.get("max_output_substrate_depth", 0) or 0)),
            node_controls=node_controls,
            edge_controls=edge_controls,
        )

    def depth_for(self, substrate: SubstrateRef) -> int:
        kind = str(substrate.kind)
        if self.static_substrate_depth >= 0:
            if kind.startswith("hidden_"):
                return int(self.static_substrate_depth)
            return 0
        if kind in self.node_controls:
            return max(0, int(self.node_controls[kind].depth))
        if kind.startswith("input_"):
            return min(1, self.max_input_substrate_depth)
        if kind.startswith("hidden_"):
            return min(1, self.max_hidden_substrate_depth)
        if kind.startswith("output_"):
            return min(1, self.max_output_substrate_depth)
        return 0

    def node_control_for(self, substrate: SubstrateRef) -> MatgoNodeControl:
        kind = str(substrate.kind)
        control = self.node_controls.get(kind)
        if control is not None:
            return control
        return MatgoNodeControl(kind=kind, depth=self.depth_for(substrate))

    def edge_control_for(self, source: SubstrateRef, target: SubstrateRef) -> MatgoEdgeControl:
        key = edge_key(source, target)
        control = self.edge_controls.get(key)
        if control is not None:
            return control
        return MatgoEdgeControl(key=key)

    def outer_weight_for(self, source: SubstrateRef, target: SubstrateRef) -> float:
        return float(self.edge_control_for(source, target).outer_weight)

    def allow_identity_mapping_for(self, source: SubstrateRef, target: SubstrateRef) -> bool:
        return bool(self.edge_control_for(source, target).allow_identity_mapping)


def _default_node_query_control(kind: str) -> MatgoQueryControl:
    name = str(kind or "")
    if name.startswith("hidden_group_hand_"):
        return MatgoQueryControl(expression_bias=0.08)
    if name.startswith("hidden_group_board_"):
        return MatgoQueryControl(expression_bias=0.10)
    if name.startswith("hidden_group_captured_"):
        return MatgoQueryControl(expression_bias=0.10)
    if name == "hidden_group_rule_context":
        return MatgoQueryControl(expression_bias=0.18)
    if name == "hidden_group_public_context":
        return MatgoQueryControl(expression_bias=0.12)
    if name == "hidden_group_ownership_context":
        return MatgoQueryControl(expression_bias=0.06)
    if name == "hidden_play_context":
        return MatgoQueryControl(expression_bias=0.22)
    if name == "hidden_match_context":
        return MatgoQueryControl(expression_bias=0.24)
    if name == "hidden_option_context":
        return MatgoQueryControl(expression_bias=0.16)
    return MatgoQueryControl()


def _default_edge_query_control(source_kind: str, target_kind: str) -> MatgoQueryControl:
    source_name = str(source_kind or "")
    target_name = str(target_kind or "")
    if source_name.startswith("input_") and target_name.startswith("hidden_group_"):
        return MatgoQueryControl(expression_bias=0.06)
    if source_name.startswith("hidden_group_") and target_name.startswith("hidden_"):
        return MatgoQueryControl(expression_bias=0.12)
    if source_name.startswith("hidden_") and target_name.startswith("output_"):
        return MatgoQueryControl(expression_bias=0.18)
    return MatgoQueryControl()


def _query_control_from_raw(raw: object, default: MatgoQueryControl) -> MatgoQueryControl:
    payload = dict(raw or {}) if isinstance(raw, dict) else {}
    return MatgoQueryControl(
        weight_scale=float(payload.get("weight_scale", default.weight_scale) or default.weight_scale),
        weight_bias=float(payload.get("weight_bias", default.weight_bias) or default.weight_bias),
        expression_scale=float(payload.get("expression_scale", default.expression_scale) or default.expression_scale),
        expression_bias=float(payload.get("expression_bias", default.expression_bias) or default.expression_bias),
    )
