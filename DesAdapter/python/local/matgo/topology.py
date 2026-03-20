from __future__ import annotations

from dataclasses import dataclass

from deshyperneat.coordinates import Point2D
from deshyperneat.substrate import DevelopmentEdge, SubstrateRef, SubstrateSpec, SubstrateTopology


CARD_INPUT_FEATURES = (
    "month",
    "kwang",
    "five",
    "ribbon",
    "junk",
    "pi",
    "bonus",
    "red_ribbons",
    "blue_ribbons",
    "plain_ribbons",
    "five_birds",
)

RULE_SCORE_FEATURES = (
    "self_go_bonus",
    "self_go_multiplier",
    "self_shaking_multiplier",
    "self_bomb_multiplier",
    "self_bak_multiplier",
    "self_pi_bak",
    "self_gwang_bak",
    "self_mong_bak",
    "self_five_base",
    "self_ribbon_base",
    "self_junk_base",
    "self_red_ribbons",
    "self_blue_ribbons",
    "self_plain_ribbons",
    "self_five_birds",
    "self_kwang",
    "opp_go_bonus",
    "opp_go_multiplier",
    "opp_shaking_multiplier",
    "opp_bomb_multiplier",
    "opp_bak_multiplier",
    "opp_pi_bak",
    "opp_gwang_bak",
    "opp_mong_bak",
    "opp_five_base",
    "opp_ribbon_base",
    "opp_junk_base",
    "opp_red_ribbons",
    "opp_blue_ribbons",
    "opp_plain_ribbons",
    "opp_five_birds",
    "opp_kwang",
    "self_go_legal",
    "self_stop_legal",
    "self_go_ready",
    "self_auto_stop_ready",
    "self_failed_go",
    "self_gukjin_mode_junk",
    "self_gukjin_locked",
    "self_pending_gukjin_choice",
    "self_gukjin_junk_better",
    "self_pending_president",
    "self_pending_president_month",
    "self_president_hold",
    "self_president_hold_month",
    "self_president_x4_ready",
    "self_dokbak_risk",
    "self_ppuk",
    "self_jjob",
    "self_jabbeok",
    "self_pansseul",
    "self_ppuk_active",
    "self_ppuk_streak",
    "self_held_bonus_cards",
    "opp_go_legal",
    "opp_stop_legal",
    "opp_go_ready",
    "opp_auto_stop_ready",
    "opp_failed_go",
    "opp_gukjin_mode_junk",
    "opp_gukjin_locked",
    "opp_pending_gukjin_choice",
    "opp_gukjin_junk_better",
    "opp_pending_president",
    "opp_pending_president_month",
    "opp_president_hold",
    "opp_president_hold_month",
    "opp_president_x4_ready",
    "opp_dokbak_risk",
    "opp_ppuk",
    "opp_jjob",
    "opp_jabbeok",
    "opp_pansseul",
    "opp_ppuk_active",
    "opp_ppuk_streak",
    "opp_held_bonus_cards",
    "state_carry_over_multiplier",
    "state_next_carry_over_multiplier",
    "state_last_nagari",
    "state_last_dokbak",
    "state_pending_steal",
    "state_pending_bonus_flips",
)

RULE_GROUP_FEATURES = (
    "score_progress",
    "score_base",
    "multiplier",
    "bak_flags",
    "go_state",
    "choice_state",
    "event_state",
    "round_state",
)

TYPE_GROUP_FEATURES = (
    "kwang",
    "five",
    "ribbon",
    "pi_value",
)

COMBO_GROUP_FEATURES = (
    "red_ribbons",
    "blue_ribbons",
    "plain_ribbons",
    "five_birds",
    "kwang_combo",
)

OWNERSHIP_GROUP_FEATURES = (
    "hand",
    "board",
    "captured_self",
    "captured_opp",
)

PUBLIC_GROUP_FEATURES = (
    "turn_phase",
    "deck_pressure",
    "board_pressure",
    "self_score",
    "opp_score",
    "option_window",
)

MONTH_XS = tuple(-1.0 + (2.0 * index / 11.0) for index in range(12))
TYPE_XS = (-0.75, -0.25, 0.25, 0.75)
COMBO_XS = (-0.8, -0.4, 0.0, 0.4, 0.8)
OWNERSHIP_XS = (-0.75, -0.25, 0.25, 0.75)
PUBLIC_XS = (-0.9, -0.54, -0.18, 0.18, 0.54, 0.9)
RULE_GROUP_XS = (-0.9, -0.64, -0.38, -0.12, 0.12, 0.38, 0.64, 0.9)
OPTION_XS = (-0.54, -0.18, 0.18, 0.54)
ACTION_X = (0.0,)

TYPE_FEATURES = frozenset({"kwang", "five", "ribbon", "pi_value"})
COMBO_FEATURES = frozenset({"red_ribbons", "blue_ribbons", "plain_ribbons", "five_birds"})


@dataclass(frozen=True)
class MatgoLayoutConfig:
    hand_slots: int = 10
    board_slots: int = 12
    match_slots: int = 12
    captured_self_slots: int = 12
    captured_opp_slots: int = 12
    option_slots: int = 4


def _row(count: int, y: float, x_start: float = -1.0, x_end: float = 1.0) -> list[Point2D]:
    if count <= 1:
        return [Point2D(0.0, y)]
    step = (x_end - x_start) / float(count - 1)
    return [Point2D(x_start + (step * index), y) for index in range(count)]


def _points_from_xs(xs: tuple[float, ...] | list[float], y: float) -> list[Point2D]:
    return [Point2D(float(x), float(y)) for x in xs]


def _feature_specs(zone: str, slot_count: int, y_top: float, y_bottom: float) -> list[SubstrateSpec]:
    if slot_count <= 0:
        return []
    if len(CARD_INPUT_FEATURES) <= 1:
        return [SubstrateSpec(SubstrateRef(f"input_{zone}_{CARD_INPUT_FEATURES[0]}", 0), seed_points=_row(slot_count, y=y_top))]
    step = (y_top - y_bottom) / float(len(CARD_INPUT_FEATURES) - 1)
    specs: list[SubstrateSpec] = []
    for index, feature in enumerate(CARD_INPUT_FEATURES):
        specs.append(
            SubstrateSpec(
                SubstrateRef(f"input_{zone}_{feature}", 0),
                seed_points=_row(slot_count, y=(y_top - (step * index))),
            )
        )
    return specs


def _semantic_bin_specs(zone: str, *, month_y: float, type_y: float, combo_y: float) -> list[SubstrateSpec]:
    return [
        SubstrateSpec(
            SubstrateRef(f"input_{zone}_month_bin", 0),
            seed_points=_points_from_xs(MONTH_XS, month_y),
        ),
        SubstrateSpec(
            SubstrateRef(f"input_{zone}_type_bin", 0),
            seed_points=_points_from_xs(TYPE_XS, type_y),
        ),
        SubstrateSpec(
            SubstrateRef(f"input_{zone}_combo_bin", 0),
            seed_points=_points_from_xs(COMBO_XS, combo_y),
        ),
    ]


def _rule_score_specs(y: float, x_start: float = -1.0, x_end: float = 1.0) -> list[SubstrateSpec]:
    del x_start, x_end
    return [
        SubstrateSpec(
            SubstrateRef("input_rule_score", 0),
            seed_points=_rule_score_points(y),
        )
    ]


def _rule_group_for_feature(feature_name: str) -> str:
    name = str(feature_name or "")
    if name in ("self_go_bonus", "opp_go_bonus"):
        return "score_progress"
    if any(token in name for token in ("_go_multiplier", "_shaking_multiplier", "_bomb_multiplier", "_bak_multiplier")):
        return "multiplier"
    if name.startswith("state_carry_over_multiplier") or name.startswith("state_next_carry_over_multiplier"):
        return "multiplier"
    if name.endswith("_pi_bak") or name.endswith("_gwang_bak") or name.endswith("_mong_bak") or name == "state_last_dokbak":
        return "bak_flags"
    if (
        "_go_legal" in name
        or "_stop_legal" in name
        or "_go_ready" in name
        or "_auto_stop_ready" in name
        or name.endswith("failed_go")
    ):
        return "go_state"
    if "gukjin" in name or "president" in name:
        return "choice_state"
    if any(token in name for token in ("ppuk", "jjob", "jabbeok", "pansseul", "held_bonus_cards")):
        return "event_state"
    if name.startswith("state_"):
        return "round_state"
    return "score_base"


def _rule_score_points(y: float) -> list[Point2D]:
    group_order = {name: index for index, name in enumerate(RULE_GROUP_FEATURES)}
    grouped: dict[str, list[str]] = {name: [] for name in RULE_GROUP_FEATURES}
    for feature_name in RULE_SCORE_FEATURES:
        grouped[_rule_group_for_feature(feature_name)].append(feature_name)

    points: list[Point2D] = []
    for feature_name in RULE_SCORE_FEATURES:
        group_name = _rule_group_for_feature(feature_name)
        members = grouped[group_name]
        group_index = group_order[group_name]
        member_index = members.index(feature_name)
        member_count = max(1, len(members))
        if member_count <= 1:
            y_offset = 0.0
        else:
            y_offset = ((member_index / float(member_count - 1)) - 0.5) * 0.18
        points.append(Point2D(RULE_GROUP_XS[group_index], y + y_offset))
    return points


def _zone_group_specs(zone: str, *, month_y: float, type_y: float, combo_y: float) -> list[SubstrateSpec]:
    del month_y, type_y, combo_y
    return [
        SubstrateSpec(
            SubstrateRef(f"hidden_group_{zone}_month", 0),
            seed_points=[],
            depth=0,
        ),
        SubstrateSpec(
            SubstrateRef(f"hidden_group_{zone}_type", 0),
            seed_points=[],
            depth=0,
        ),
        SubstrateSpec(
            SubstrateRef(f"hidden_group_{zone}_combo", 0),
            seed_points=[],
            depth=0,
        ),
    ]


def _group_specs(cfg: MatgoLayoutConfig) -> list[SubstrateSpec]:
    del cfg
    return [
        *_zone_group_specs("hand", month_y=-0.14, type_y=-0.18, combo_y=-0.22),
        SubstrateSpec(
            SubstrateRef("hidden_group_board_month", 0),
            seed_points=[],
            depth=0,
        ),
        *_zone_group_specs("captured_self", month_y=-0.34, type_y=-0.38, combo_y=-0.42),
        *_zone_group_specs("captured_opp", month_y=-0.34, type_y=-0.38, combo_y=-0.42),
        SubstrateSpec(
            SubstrateRef("hidden_group_rule_context", 0),
            seed_points=[],
            depth=0,
        ),
        SubstrateSpec(
            SubstrateRef("hidden_group_ownership_context", 0),
            seed_points=[],
            depth=0,
        ),
    ]


def _hidden_specs(cfg: MatgoLayoutConfig) -> list[SubstrateSpec]:
    del cfg
    return [
        SubstrateSpec(
            SubstrateRef("hidden_play_context", 0),
            seed_points=[],
            depth=1,
        ),
        SubstrateSpec(
            SubstrateRef("hidden_match_context", 0),
            seed_points=[],
            depth=1,
        ),
        SubstrateSpec(
            SubstrateRef("hidden_option_context", 0),
            seed_points=[],
            depth=1,
        ),
    ]


def _group_targets_for_input(input_kind: str) -> list[SubstrateRef]:
    kind = str(input_kind or "")
    if kind == "input_rule_score":
        return [SubstrateRef("hidden_group_rule_context", 0)]
    if kind == "input_hand":
        return [
            SubstrateRef("hidden_group_hand_month", 0),
            SubstrateRef("hidden_group_hand_type", 0),
            SubstrateRef("hidden_group_hand_combo", 0),
            SubstrateRef("hidden_group_ownership_context", 0),
        ]
    if kind.startswith("input_focus_"):
        return []
    if kind == "input_board_month_bin":
        return [
            SubstrateRef("hidden_group_board_month", 0),
            SubstrateRef("hidden_group_ownership_context", 0),
        ]
    for zone_name in ("captured_self", "captured_opp"):
        prefix = f"input_{zone_name}_"
        if not kind.startswith(prefix):
            continue
        suffix = kind[len(prefix) :]
        refs = [SubstrateRef("hidden_group_ownership_context", 0)]
        if suffix == "month" or suffix == "month_bin":
            refs.insert(0, SubstrateRef(f"hidden_group_{zone_name}_month", 0))
            return refs
        if suffix == "type_bin" or suffix in TYPE_FEATURES:
            refs.insert(0, SubstrateRef(f"hidden_group_{zone_name}_type", 0))
            return refs
        if suffix == "combo_bin" or suffix in COMBO_FEATURES:
            refs.insert(0, SubstrateRef(f"hidden_group_{zone_name}_combo", 0))
            return refs
    return []


def _hidden_targets_for_input(input_kind: str) -> list[SubstrateRef]:
    kind = str(input_kind or "")
    if kind.startswith("input_focus_"):
        return [
            SubstrateRef("hidden_play_context", 0),
            SubstrateRef("hidden_match_context", 0),
            SubstrateRef("hidden_option_context", 0),
        ]
    return []


def _hidden_targets_for_group(group_kind: str) -> list[SubstrateRef]:
    kind = str(group_kind or "")
    play_hidden = SubstrateRef("hidden_play_context", 0)
    match_hidden = SubstrateRef("hidden_match_context", 0)
    option_hidden = SubstrateRef("hidden_option_context", 0)

    if kind == "hidden_group_hand_month" or kind == "hidden_group_hand_type":
        return [play_hidden, option_hidden]
    if kind == "hidden_group_hand_combo":
        return [play_hidden, option_hidden]
    if kind == "hidden_group_board_month":
        return [play_hidden, match_hidden, option_hidden]
    if (
        kind == "hidden_group_captured_self_month"
        or kind == "hidden_group_captured_opp_month"
        or kind == "hidden_group_captured_self_type"
        or kind == "hidden_group_captured_opp_type"
        or kind == "hidden_group_captured_self_combo"
        or kind == "hidden_group_captured_opp_combo"
    ):
        return [play_hidden, match_hidden, option_hidden]
    if kind == "hidden_group_rule_context":
        return [play_hidden, match_hidden, option_hidden]
    if kind == "hidden_group_ownership_context":
        return [play_hidden, match_hidden, option_hidden]
    return [play_hidden, match_hidden, option_hidden]


def _input_to_group_weight(source_kind: str, target_kind: str) -> float:
    if target_kind == "hidden_group_ownership_context":
        return 0.55
    if source_kind == "input_rule_score" and target_kind == "hidden_group_rule_context":
        return 1.25
    if source_kind == "input_hand":
        return 1.05
    if source_kind.startswith("input_board_"):
        return 1.00
    if source_kind.startswith("input_captured_self_") or source_kind.startswith("input_captured_opp_"):
        return 0.95
    return 1.0


def _group_to_hidden_weight(source_kind: str, target_kind: str) -> float:
    if source_kind == "hidden_group_rule_context":
        if target_kind == "hidden_option_context":
            return 1.30
        if target_kind == "hidden_play_context":
            return 0.85
        return 0.75
    if source_kind == "hidden_group_ownership_context":
        return 0.45
    if source_kind.startswith("hidden_group_hand_"):
        if target_kind == "hidden_play_context":
            return 1.15
        if target_kind == "hidden_option_context":
            return 1.00
        return 0.60
    if source_kind.startswith("hidden_group_board_"):
        if target_kind == "hidden_match_context":
            return 1.15
        if target_kind == "hidden_play_context":
            return 0.95
        return 0.85
    if source_kind.startswith("hidden_group_captured_self_") or source_kind.startswith("hidden_group_captured_opp_"):
        if target_kind == "hidden_match_context":
            return 1.00
        if target_kind == "hidden_play_context":
            return 0.95
        return 0.80
    return 1.0


def _hidden_to_output_weight(source_kind: str, target_kind: str) -> float:
    if source_kind == "hidden_play_context" and target_kind == "output_play":
        return 1.10
    if source_kind == "hidden_match_context" and target_kind == "output_match":
        return 1.10
    if source_kind == "hidden_option_context" and target_kind == "output_option":
        return 1.20
    return 1.0


def _allow_identity_mapping(source_kind: str, target_kind: str) -> bool:
    if source_kind == "hidden_play_context" and target_kind == "output_play":
        return True
    if source_kind == "hidden_match_context" and target_kind == "output_match":
        return True
    if source_kind == "hidden_option_context" and target_kind == "output_option":
        return True
    if source_kind.endswith("_month_bin") and target_kind.endswith("_month"):
        return True
    if source_kind.endswith("_type_bin") and target_kind.endswith("_type"):
        return True
    if source_kind.endswith("_combo_bin") and target_kind.endswith("_combo"):
        return True
    return False


def _edge_weight(source_kind: str, target_kind: str) -> float:
    if source_kind.startswith("input_") and target_kind.startswith("hidden_group_"):
        return _input_to_group_weight(source_kind, target_kind)
    if source_kind.startswith("hidden_group_") and target_kind.startswith("hidden_"):
        return _group_to_hidden_weight(source_kind, target_kind)
    if source_kind.startswith("hidden_") and target_kind.startswith("output_"):
        return _hidden_to_output_weight(source_kind, target_kind)
    return 1.0


def _weighted_edge(source: SubstrateRef, target: SubstrateRef) -> DevelopmentEdge:
    return DevelopmentEdge(
        source,
        target,
        base_weight=_edge_weight(source.kind, target.kind),
        allow_identity_mapping=_allow_identity_mapping(source.kind, target.kind),
    )


def build_minimal_matgo_topology(config: MatgoLayoutConfig | None = None) -> SubstrateTopology:
    cfg = config or MatgoLayoutConfig()

    group_specs = _group_specs(cfg)
    play_hidden_ref = SubstrateRef("hidden_play_context", 0)
    match_hidden_ref = SubstrateRef("hidden_match_context", 0)
    option_hidden_ref = SubstrateRef("hidden_option_context", 0)
    play_ref = SubstrateRef("output_play", 0)
    match_ref = SubstrateRef("output_match", 0)
    option_ref = SubstrateRef("output_option", 0)

    inputs = [
        SubstrateSpec(SubstrateRef("input_hand", 0), seed_points=_row(cfg.hand_slots, y=0.76)),
    ]
    inputs.extend(_rule_score_specs(y=0.88, x_start=-0.95, x_end=0.95))
    inputs.extend(_feature_specs("focus", 1, y_top=0.68, y_bottom=0.48))
    inputs.append(
        SubstrateSpec(
            SubstrateRef("input_board_month_bin", 0),
            seed_points=_points_from_xs(MONTH_XS, 0.46),
        )
    )
    inputs.extend(_semantic_bin_specs("captured_self", month_y=0.16, type_y=0.10, combo_y=0.04))
    inputs.extend(_semantic_bin_specs("captured_opp", month_y=-0.04, type_y=-0.10, combo_y=-0.16))
    hidden = [*group_specs, *_hidden_specs(cfg)]
    outputs = [
        SubstrateSpec(play_ref, seed_points=_points_from_xs(ACTION_X, -0.84)),
        SubstrateSpec(match_ref, seed_points=_points_from_xs(ACTION_X, -0.92)),
        SubstrateSpec(option_ref, seed_points=_points_from_xs(OPTION_XS[: cfg.option_slots], -1.00)),
    ]

    links = []
    for input_spec in inputs:
        for group_ref in _group_targets_for_input(input_spec.ref.kind):
            links.append(_weighted_edge(input_spec.ref, group_ref))
        for hidden_ref in _hidden_targets_for_input(input_spec.ref.kind):
            links.append(_weighted_edge(input_spec.ref, hidden_ref))

    for group_spec in group_specs:
        for hidden_ref in _hidden_targets_for_group(group_spec.ref.kind):
            links.append(_weighted_edge(group_spec.ref, hidden_ref))

    links.extend(
        [
            _weighted_edge(play_hidden_ref, play_ref),
            _weighted_edge(match_hidden_ref, match_ref),
            _weighted_edge(option_hidden_ref, option_ref),
        ]
    )

    return SubstrateTopology(
        inputs=inputs,
        outputs=outputs,
        hidden=hidden,
        links=links,
    )
