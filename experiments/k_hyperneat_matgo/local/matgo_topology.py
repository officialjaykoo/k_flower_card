from __future__ import annotations

from dataclasses import dataclass

from k_hyperneat.coordinates import Point2D
from k_hyperneat.substrate import DevelopmentEdge, SubstrateRef, SubstrateSpec, SubstrateTopology


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


def build_minimal_matgo_topology(config: MatgoLayoutConfig | None = None) -> SubstrateTopology:
    cfg = config or MatgoLayoutConfig()

    public_ref = SubstrateRef("input_public", 0)
    hand_ref = SubstrateRef("input_hand", 0)
    board_ref = SubstrateRef("input_board", 0)
    captured_self_ref = SubstrateRef("input_captured_self", 0)
    captured_opp_ref = SubstrateRef("input_captured_opp", 0)
    play_ref = SubstrateRef("output_play", 0)
    match_ref = SubstrateRef("output_match", 0)
    option_ref = SubstrateRef("output_option", 0)

    inputs = [
        SubstrateSpec(public_ref, seed_points=_row(6, y=1.0, x_start=-0.9, x_end=0.9)),
        SubstrateSpec(hand_ref, seed_points=_row(cfg.hand_slots, y=0.5)),
        SubstrateSpec(board_ref, seed_points=_row(cfg.board_slots, y=0.15)),
        SubstrateSpec(captured_self_ref, seed_points=_row(cfg.captured_self_slots, y=-0.2)),
        SubstrateSpec(captured_opp_ref, seed_points=_row(cfg.captured_opp_slots, y=-0.55)),
    ]
    outputs = [
        SubstrateSpec(play_ref, seed_points=_row(cfg.hand_slots, y=-0.85)),
        SubstrateSpec(match_ref, seed_points=_row(cfg.match_slots, y=-0.93)),
        SubstrateSpec(option_ref, seed_points=_row(cfg.option_slots, y=-1.0, x_start=-0.3, x_end=0.3)),
    ]

    links = [
        DevelopmentEdge(public_ref, play_ref),
        DevelopmentEdge(hand_ref, play_ref),
        DevelopmentEdge(board_ref, play_ref),
        DevelopmentEdge(captured_self_ref, play_ref),
        DevelopmentEdge(captured_opp_ref, play_ref),
        DevelopmentEdge(public_ref, match_ref),
        DevelopmentEdge(hand_ref, match_ref),
        DevelopmentEdge(board_ref, match_ref),
        DevelopmentEdge(captured_self_ref, match_ref),
        DevelopmentEdge(captured_opp_ref, match_ref),
        DevelopmentEdge(public_ref, option_ref),
        DevelopmentEdge(hand_ref, option_ref),
        DevelopmentEdge(board_ref, option_ref),
        DevelopmentEdge(captured_self_ref, option_ref),
        DevelopmentEdge(captured_opp_ref, option_ref),
    ]

    return SubstrateTopology(
        inputs=inputs,
        outputs=outputs,
        hidden=[],
        links=links,
    )
