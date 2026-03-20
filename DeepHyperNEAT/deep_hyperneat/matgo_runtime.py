from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from deep_hyperneat.decode import create_substrate


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

MONTH_SLOT_COUNT = 13
MONTH_XS = tuple(-1.0 + (2.0 * index / float(MONTH_SLOT_COUNT - 1)) for index in range(MONTH_SLOT_COUNT))
TYPE_XS = (-0.85, -0.5, -0.15, 0.15, 0.5, 0.85)
COMBO_XS = (-0.8, -0.4, 0.0, 0.4, 0.8)
PUBLIC_XS = (-0.9, -0.54, -0.18, 0.18, 0.54, 0.9)
RULE_GROUP_XS = (-0.9, -0.64, -0.38, -0.12, 0.12, 0.38, 0.64, 0.9)
OPTION_XS = (-0.54, -0.18, 0.18, 0.54)

HAND_SLOT_COUNT = 10
BOARD_SLOT_COUNT = MONTH_SLOT_COUNT
CAPTURED_SLOT_COUNT = MONTH_SLOT_COUNT

MATGO_INPUT_SPECS = (
    {"kind": "input_public", "index": 0, "slot_count": 6},
    {"kind": "input_rule_score", "index": 0, "slot_count": len(RULE_SCORE_FEATURES)},
    *tuple({"kind": f"input_hand_{feature}", "index": 0, "slot_count": HAND_SLOT_COUNT} for feature in CARD_INPUT_FEATURES),
    {"kind": "input_board_month_bin", "index": 0, "slot_count": BOARD_SLOT_COUNT},
    {"kind": "input_board_type_bin", "index": 0, "slot_count": 6},
    {"kind": "input_board_combo_bin", "index": 0, "slot_count": 5},
    {"kind": "input_captured_self_month_bin", "index": 0, "slot_count": CAPTURED_SLOT_COUNT},
    {"kind": "input_captured_self_type_bin", "index": 0, "slot_count": 6},
    {"kind": "input_captured_self_combo_bin", "index": 0, "slot_count": 5},
    {"kind": "input_captured_opp_month_bin", "index": 0, "slot_count": CAPTURED_SLOT_COUNT},
    {"kind": "input_captured_opp_type_bin", "index": 0, "slot_count": 6},
    {"kind": "input_captured_opp_combo_bin", "index": 0, "slot_count": 5},
)

MATGO_OUTPUT_SPECS = (
    {"kind": "output_play", "index": 0, "slot_count": 10},
    {"kind": "output_match", "index": 0, "slot_count": BOARD_SLOT_COUNT},
    {"kind": "output_option", "index": 0, "slot_count": 4},
)

MATGO_INPUT_COUNT = sum(int(item["slot_count"]) for item in MATGO_INPUT_SPECS)
MATGO_OUTPUT_COUNT = sum(int(item["slot_count"]) for item in MATGO_OUTPUT_SPECS)
MATGO_RUNTIME_FORMAT = "k_hyperneat_executor_v1"


def clamp01(x: float) -> float:
    value = float(x)
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


def _row_points(count: int, y: float, x_start: float = -1.0, x_end: float = 1.0) -> list[tuple[float, float]]:
    if count <= 1:
        return [(0.0, float(y))]
    step = (float(x_end) - float(x_start)) / float(count - 1)
    return [(float(x_start + (step * index)), float(y)) for index in range(count)]


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


def _rule_score_points(y: float) -> list[tuple[float, float]]:
    group_order = {name: index for index, name in enumerate(RULE_GROUP_FEATURES)}
    grouped: dict[str, list[str]] = {name: [] for name in RULE_GROUP_FEATURES}
    for feature_name in RULE_SCORE_FEATURES:
        grouped[_rule_group_for_feature(feature_name)].append(feature_name)

    points: list[tuple[float, float]] = []
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
        points.append((float(RULE_GROUP_XS[group_index]), float(y + y_offset)))
    return points


def _input_points_for_spec(spec: dict[str, Any]) -> list[tuple[float, float]]:
    kind = str(spec.get("kind") or "")
    slot_count = max(0, int(spec.get("slot_count", 0) or 0))
    if kind == "input_public":
        return [(float(x), 1.0) for x in PUBLIC_XS[:slot_count]]
    if kind == "input_rule_score":
        return _rule_score_points(0.88)[:slot_count]
    if kind.startswith("input_hand_"):
        feature_name = kind[len("input_hand_") :]
        feature_index = CARD_INPUT_FEATURES.index(feature_name)
        if len(CARD_INPUT_FEATURES) <= 1:
            y = 0.76
        else:
            step = (0.76 - 0.56) / float(len(CARD_INPUT_FEATURES) - 1)
            y = 0.76 - (step * feature_index)
        return _row_points(slot_count, y)
    if kind == "input_board_month_bin":
        return [(float(x), 0.46) for x in MONTH_XS[:slot_count]]
    if kind == "input_board_type_bin":
        return [(float(x), 0.40) for x in TYPE_XS[:slot_count]]
    if kind == "input_board_combo_bin":
        return [(float(x), 0.34) for x in COMBO_XS[:slot_count]]
    if kind == "input_captured_self_month_bin":
        return [(float(x), 0.16) for x in MONTH_XS[:slot_count]]
    if kind == "input_captured_self_type_bin":
        return [(float(x), 0.10) for x in TYPE_XS[:slot_count]]
    if kind == "input_captured_self_combo_bin":
        return [(float(x), 0.04) for x in COMBO_XS[:slot_count]]
    if kind == "input_captured_opp_month_bin":
        return [(float(x), -0.04) for x in MONTH_XS[:slot_count]]
    if kind == "input_captured_opp_type_bin":
        return [(float(x), -0.10) for x in TYPE_XS[:slot_count]]
    if kind == "input_captured_opp_combo_bin":
        return [(float(x), -0.16) for x in COMBO_XS[:slot_count]]
    return _row_points(slot_count, 0.0)


def _output_points_for_spec(spec: dict[str, Any]) -> list[tuple[float, float]]:
    kind = str(spec.get("kind") or "")
    slot_count = max(0, int(spec.get("slot_count", 0) or 0))
    if kind == "output_play":
        return _row_points(slot_count, -0.84)
    if kind == "output_match":
        return [(float(x), -0.92) for x in MONTH_XS[:slot_count]]
    if kind == "output_option":
        return [(float(x), -1.00) for x in OPTION_XS[:slot_count]]
    return _row_points(slot_count, -0.90)


def _distribute_counts(total: int, ratios: tuple[int, ...]) -> list[int]:
    total = max(len(ratios), int(total))
    ratio_sum = max(1.0, float(sum(ratios)))
    raw = [float(ratio) * float(total) / ratio_sum for ratio in ratios]
    counts = [max(1, int(math.floor(value))) for value in raw]
    while sum(counts) > total:
        index = max(range(len(counts)), key=lambda i: counts[i])
        if counts[index] <= 1:
            break
        counts[index] -= 1
    while sum(counts) < total:
        index = max(range(len(raw)), key=lambda i: raw[i] - counts[i])
        counts[index] += 1
    return counts


def _build_hidden_points(total_hidden_nodes: int) -> list[tuple[float, float]]:
    play_count, match_count, option_count = _distribute_counts(int(total_hidden_nodes), (10, 12, 4))
    return (
        _row_points(play_count, -0.42)
        + _row_points(match_count, -0.56)
        + _row_points(option_count, -0.70, x_start=-0.7, x_end=0.7)
    )


def decode_matgo_substrate(cppn: Any, hidden_node_count: int = 26) -> Any:
    input_layer: list[tuple[float, float]] = []
    for spec in MATGO_INPUT_SPECS:
        input_layer.extend(_input_points_for_spec(spec))

    output_layer: list[tuple[float, float]] = []
    for spec in MATGO_OUTPUT_SPECS:
        output_layer.extend(_output_points_for_spec(spec))

    if len(input_layer) != MATGO_INPUT_COUNT:
        raise RuntimeError(f"matgo grouped decode expected {MATGO_INPUT_COUNT} input points, got {len(input_layer)}")
    if len(output_layer) != MATGO_OUTPUT_COUNT:
        raise RuntimeError(f"matgo grouped decode expected {MATGO_OUTPUT_COUNT} output points, got {len(output_layer)}")

    hidden_points = _build_hidden_points(max(1, int(hidden_node_count)))

    connection_mappings = [cppn.nodes[x].cppn_tuple for x in cppn.output_nodes if cppn.nodes[x].cppn_tuple[0] != (1, 1)]
    hidden_sheets = {cppn.nodes[node].cppn_tuple[0] for node in cppn.output_nodes}
    substrate = {sheet_id: list(hidden_points) for sheet_id in hidden_sheets}
    substrate[(1, 0)] = input_layer
    substrate[(0, 0)] = output_layer
    substrate[(1, 1)] = [(0.0, 0.0)]
    cppn_idx_dict = {cppn.nodes[idx].cppn_tuple: idx for idx in cppn.output_nodes}
    return create_substrate(cppn, substrate, connection_mappings, cppn_idx_dict)


def _normalize_early_stop_win_rate_cutoffs(raw: Any) -> list[dict[str, float]]:
    cutoffs: list[dict[str, float]] = []
    source = list(raw or [])
    for item in source:
        if not isinstance(item, dict):
            continue
        games = max(1, int(item.get("games", 0) or 0))
        max_win_rate = float(item.get("max_win_rate", 0.0) or 0.0)
        if not (0.0 <= max_win_rate <= 1.0):
            continue
        cutoffs.append({"games": games, "max_win_rate": max_win_rate})
    cutoffs.sort(key=lambda item: int(item["games"]))
    seen_games: set[int] = set()
    unique_cutoffs: list[dict[str, float]] = []
    for item in cutoffs:
        games = int(item["games"])
        if games in seen_games:
            continue
        seen_games.add(games)
        unique_cutoffs.append(item)
    return unique_cutoffs


def _normalize_early_stop_go_take_rate_cutoffs(raw: Any) -> list[dict[str, Any]]:
    cutoffs: list[dict[str, Any]] = []
    source = list(raw or [])
    for item in source:
        if not isinstance(item, dict):
            continue
        games = max(1, int(item.get("games", 0) or 0))
        min_go_opp_count = max(0, int(item.get("min_go_opportunity_count", 0) or 0))
        min_go_take_rate = item.get("min_go_take_rate")
        max_go_take_rate = item.get("max_go_take_rate")
        min_rate = None if min_go_take_rate is None else float(min_go_take_rate)
        max_rate = None if max_go_take_rate is None else float(max_go_take_rate)
        if min_rate is None and max_rate is None:
            continue
        valid = True
        for value in (min_rate, max_rate):
            if value is None:
                continue
            if value < 0.0 or value > 1.0:
                valid = False
                break
        if (not valid) or (min_rate is not None and max_rate is not None and min_rate > max_rate):
            continue
        cutoffs.append(
            {
                "games": games,
                "min_go_opportunity_count": min_go_opp_count,
                "min_go_take_rate": min_rate,
                "max_go_take_rate": max_rate,
            }
        )
    cutoffs.sort(key=lambda item: int(item["games"]))
    seen_games: set[int] = set()
    unique_cutoffs: list[dict[str, Any]] = []
    for item in cutoffs:
        games = int(item["games"])
        if games in seen_games:
            continue
        seen_games.add(games)
        unique_cutoffs.append(item)
    return unique_cutoffs


def _normalize_runtime_settings(raw: dict[str, Any]) -> dict[str, Any]:
    runtime = dict(raw or {})
    mix = runtime.get("opponent_policy_mix") or []
    if not isinstance(mix, list):
        mix = []
    runtime["opponent_policy_mix"] = [
        {
            "policy": str(item.get("policy") or "").strip(),
            "weight": float(item.get("weight") or 0.0),
        }
        for item in mix
        if str(item.get("policy") or "").strip() and float(item.get("weight") or 0.0) > 0.0
    ]
    runtime["opponent_policy"] = str(runtime.get("opponent_policy") or "").strip()
    runtime["games_per_genome"] = max(1, int(runtime.get("games_per_genome", 8) or 8))
    runtime["max_eval_steps"] = max(20, int(runtime.get("max_eval_steps", 300) or 300))
    runtime["first_turn_policy"] = str(runtime.get("first_turn_policy", "alternate") or "alternate").strip().lower()
    runtime["continuous_series"] = bool(runtime.get("continuous_series", True))
    runtime["fitness_gold_scale"] = float(runtime.get("fitness_gold_scale", 1500.0) or 1500.0)
    runtime["fitness_gold_neutral_delta"] = float(runtime.get("fitness_gold_neutral_delta", 0.0) or 0.0)
    runtime["fitness_win_weight"] = float(runtime.get("fitness_win_weight", 0.85) or 0.85)
    runtime["fitness_gold_weight"] = float(runtime.get("fitness_gold_weight", 0.15) or 0.15)
    runtime["fitness_win_neutral_rate"] = float(runtime.get("fitness_win_neutral_rate", 0.5) or 0.5)
    runtime["early_stop_win_rate_cutoffs"] = _normalize_early_stop_win_rate_cutoffs(
        runtime.get("early_stop_win_rate_cutoffs")
    )
    runtime["early_stop_go_take_rate_cutoffs"] = _normalize_early_stop_go_take_rate_cutoffs(
        runtime.get("early_stop_go_take_rate_cutoffs")
    )
    runtime["winner_playoff_topk"] = max(1, int(runtime.get("winner_playoff_topk", 5) or 5))
    runtime["winner_playoff_finalists"] = max(1, int(runtime.get("winner_playoff_finalists", 2) or 2))
    runtime["winner_playoff_stage1_games"] = max(
        1, int(runtime.get("winner_playoff_stage1_games", runtime["games_per_genome"]) or runtime["games_per_genome"])
    )
    runtime["winner_playoff_stage2_games"] = max(
        1, int(runtime.get("winner_playoff_stage2_games", runtime["games_per_genome"]) or runtime["games_per_genome"])
    )
    runtime["winner_playoff_win_rate_tie_threshold"] = max(
        0.0, float(runtime.get("winner_playoff_win_rate_tie_threshold", 0.01) or 0.01)
    )
    runtime["winner_playoff_mean_gold_delta_tie_threshold"] = max(
        0.0, float(runtime.get("winner_playoff_mean_gold_delta_tie_threshold", 100.0) or 100.0)
    )
    runtime["winner_playoff_go_opp_min_count"] = max(
        0, int(runtime.get("winner_playoff_go_opp_min_count", 100) or 100)
    )
    runtime["winner_playoff_go_take_rate_tie_threshold"] = max(
        0.0, float(runtime.get("winner_playoff_go_take_rate_tie_threshold", 0.02) or 0.02)
    )
    runtime["eval_pass_win_rate_min"] = float(runtime.get("eval_pass_win_rate_min", 0.45) or 0.45)
    runtime["eval_pass_mean_gold_delta_min"] = float(runtime.get("eval_pass_mean_gold_delta_min", 0.0) or 0.0)
    return runtime


def load_runtime_settings(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _normalize_runtime_settings(payload)


def merge_runtime_settings(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(base or {})
    for key, value in dict(override or {}).items():
        if value is None:
            continue
        merged[key] = value
    return _normalize_runtime_settings(merged)


def _activation_name(function: Any) -> str:
    name = str(getattr(function, "__name__", "") or "").strip().lower()
    if name in {"linear_activation", "identity_activation"}:
        return "identity"
    if name == "relu_activation":
        return "relu"
    if name == "sigmoid_activation":
        return "sigmoid"
    if name == "tanh_activation":
        return "tanh"
    if name == "sin_activation":
        return "sine"
    if name == "cos_activation":
        return "cos"
    if name in {"gauss_activation", "sharp_gauss_activation", "sharp_gauss_mu_2_activation"}:
        return "gaussian"
    return "tanh"


def substrate_to_runtime_model(substrate: Any) -> dict[str, Any]:
    input_nodes = [int(node_id) for node_id in list(getattr(substrate, "input_nodes", []) or [])]
    output_nodes = [int(node_id) for node_id in list(getattr(substrate, "output_nodes", []) or [])]
    bias_nodes = [int(node_id) for node_id in list(getattr(substrate, "bias_node", []) or [])]
    bias_source = bias_nodes[0] if bias_nodes else None
    node_evals = list(getattr(substrate, "node_evals", []) or [])

    if len(input_nodes) != MATGO_INPUT_COUNT:
        raise RuntimeError(f"matgo runtime expects {MATGO_INPUT_COUNT} inputs, got {len(input_nodes)}")
    if len(output_nodes) != MATGO_OUTPUT_COUNT:
        raise RuntimeError(f"matgo runtime expects {MATGO_OUTPUT_COUNT} outputs, got {len(output_nodes)}")

    node_ids: set[int] = {0}
    activation_by_id: dict[int, tuple[str, float]] = {0: ("identity", 1.0)}
    indegree: dict[int, int] = {0: 0}
    outgoing: dict[int, list[tuple[int, float]]] = {0: []}

    for node_id in input_nodes:
        mapped = int(node_id) + 1
        node_ids.add(mapped)
        activation_by_id[mapped] = ("identity", 0.0)
        indegree.setdefault(mapped, 0)
        outgoing.setdefault(mapped, [])

    for target, act_func, _agg_func, links in node_evals:
        mapped_target = int(target) + 1
        node_ids.add(mapped_target)
        activation_by_id[mapped_target] = (_activation_name(act_func), 0.0)
        indegree.setdefault(mapped_target, 0)
        outgoing.setdefault(mapped_target, [])
        for source, weight in links:
            if bias_source is not None and int(source) == int(bias_source):
                mapped_source = 0
            else:
                mapped_source = int(source) + 1
                node_ids.add(mapped_source)
                indegree.setdefault(mapped_source, 0)
                outgoing.setdefault(mapped_source, [])
                activation_by_id.setdefault(mapped_source, ("identity", 0.0))
            outgoing[mapped_source].append((mapped_target, float(weight)))
            indegree[mapped_target] = int(indegree.get(mapped_target, 0)) + 1

    for source_id in list(outgoing.keys()):
        outgoing[source_id].sort(key=lambda item: (item[0], item[1]))

    ready = sorted(node_id for node_id in node_ids if int(indegree.get(node_id, 0)) == 0)
    processed: list[int] = []
    actions: list[dict[str, Any]] = []
    while ready:
        node_id = ready.pop(0)
        processed.append(node_id)
        activation_name, bias = activation_by_id.get(node_id, ("identity", 0.0))
        actions.append(
            {
                "kind": "activation",
                "node_id": int(node_id),
                "bias": float(bias),
                "activation": str(activation_name),
            }
        )
        for target_id, weight in outgoing.get(node_id, []):
            actions.append(
                {
                    "kind": "link",
                    "source_id": int(node_id),
                    "target_id": int(target_id),
                    "weight": float(weight),
                }
            )
            indegree[target_id] = int(indegree.get(target_id, 0)) - 1
            if int(indegree[target_id]) == 0:
                ready.append(int(target_id))
                ready.sort()

    if len(processed) != len(node_ids):
        raise RuntimeError("unable to topologically order DeepHyperNEAT substrate for matgo runtime export")

    runtime = {
        "format_version": MATGO_RUNTIME_FORMAT,
        "node_count": max(node_ids) + 1,
        "input_node_ids": [int(node_id) + 1 for node_id in input_nodes],
        "output_node_ids": [int(node_id) + 1 for node_id in output_nodes],
        "actions": actions,
        "adapter": {
            "kind": "matgo_minimal_v1",
            "inputs": [dict(item) for item in MATGO_INPUT_SPECS],
            "outputs": [dict(item) for item in MATGO_OUTPUT_SPECS],
        },
    }
    return runtime


def save_runtime_model(runtime_model: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(runtime_model, indent=2), encoding="utf-8")
    return target


def allocate_games(total_games: int, entries: list[dict[str, Any]]) -> list[int]:
    total_games = max(1, int(total_games))
    if not entries:
        return []
    total_weight = sum(float(item["weight"]) for item in entries)
    raw = [float(item["weight"]) * total_games / max(1e-9, total_weight) for item in entries]
    counts = [max(1, int(math.floor(value))) for value in raw]
    while sum(counts) > total_games:
        index = max(range(len(counts)), key=lambda i: counts[i])
        if counts[index] <= 1:
            break
        counts[index] -= 1
    while sum(counts) < total_games:
        index = max(range(len(raw)), key=lambda i: raw[i] - counts[i])
        counts[index] += 1
    return counts


def _collect_eval_checkpoints(runtime_settings: dict[str, Any], requested_games: int) -> list[int]:
    checkpoints = {max(1, int(requested_games))}
    for item in runtime_settings.get("early_stop_win_rate_cutoffs") or []:
        games = max(1, int(item.get("games", 0) or 0))
        if games <= requested_games:
            checkpoints.add(games)
    for item in runtime_settings.get("early_stop_go_take_rate_cutoffs") or []:
        games = max(1, int(item.get("games", 0) or 0))
        if games <= requested_games:
            checkpoints.add(games)
    return sorted(checkpoints)


def _check_early_stop(summary: dict[str, Any], runtime_settings: dict[str, Any], checkpoint_games: int) -> dict[str, Any] | None:
    current_games = max(0, int(summary.get("games", 0) or 0))
    if current_games < checkpoint_games:
        return None

    for cutoff in runtime_settings.get("early_stop_win_rate_cutoffs") or []:
        cutoff_games = max(1, int(cutoff.get("games", 0) or 0))
        if cutoff_games != checkpoint_games:
            continue
        current_win_rate = float(summary.get("win_rate_a", 0.0) or 0.0)
        if current_win_rate <= float(cutoff["max_win_rate"]):
            return {
                "reason": "win_rate_cutoff",
                "cutoff_games": cutoff_games,
                "max_win_rate": float(cutoff["max_win_rate"]),
                "observed_win_rate": current_win_rate,
            }

    for cutoff in runtime_settings.get("early_stop_go_take_rate_cutoffs") or []:
        cutoff_games = max(1, int(cutoff.get("games", 0) or 0))
        if cutoff_games != checkpoint_games:
            continue
        go_opp_count = max(0, int(summary.get("go_opportunity_count_a", 0) or 0))
        min_go_opp_count = max(0, int(cutoff.get("min_go_opportunity_count", 0) or 0))
        if go_opp_count < min_go_opp_count:
            continue
        current_go_take_rate = float(summary.get("go_take_rate_a", 0.0) or 0.0)
        min_rate = cutoff.get("min_go_take_rate")
        max_rate = cutoff.get("max_go_take_rate")
        too_low = min_rate is not None and current_go_take_rate <= float(min_rate)
        too_high = max_rate is not None and current_go_take_rate >= float(max_rate)
        if too_low or too_high:
            return {
                "reason": "go_take_rate_cutoff",
                "cutoff_games": cutoff_games,
                "min_go_opportunity_count": min_go_opp_count,
                "min_go_take_rate": None if min_rate is None else float(min_rate),
                "max_go_take_rate": None if max_rate is None else float(max_rate),
                "observed_go_take_rate": current_go_take_rate,
                "observed_go_opportunity_count": go_opp_count,
            }

    return None


def _finalize_eval_summary(
    summary: dict[str, Any],
    runtime_settings: dict[str, Any],
    opponent_policy_counts: dict[str, int],
    early_stop: dict[str, Any] | None,
) -> dict[str, Any]:
    result = dict(summary or {})
    result["opponent_policy_counts"] = {
        str(key): int(value) for key, value in sorted(opponent_policy_counts.items()) if int(value) > 0
    }
    result["early_stop_win_rate_cutoffs"] = [
        dict(item) for item in (runtime_settings.get("early_stop_win_rate_cutoffs") or [])
    ]
    result["early_stop_go_take_rate_cutoffs"] = [
        dict(item) for item in (runtime_settings.get("early_stop_go_take_rate_cutoffs") or [])
    ]
    result["early_stop_triggered"] = bool(early_stop)
    result["early_stop_reason"] = None if early_stop is None else str(early_stop.get("reason") or "")
    result["early_stop_cutoff_games"] = None if early_stop is None else int(early_stop.get("cutoff_games", 0) or 0)
    result["early_stop_cutoff_max_win_rate"] = None if early_stop is None else early_stop.get("max_win_rate")
    result["early_stop_observed_win_rate"] = None if early_stop is None else early_stop.get("observed_win_rate")
    result["early_stop_cutoff_min_go_opportunity_count"] = (
        None if early_stop is None else early_stop.get("min_go_opportunity_count")
    )
    result["early_stop_cutoff_min_go_take_rate"] = None if early_stop is None else early_stop.get("min_go_take_rate")
    result["early_stop_cutoff_max_go_take_rate"] = None if early_stop is None else early_stop.get("max_go_take_rate")
    result["early_stop_observed_go_take_rate"] = None if early_stop is None else early_stop.get("observed_go_take_rate")
    result["early_stop_observed_go_opportunity_count"] = (
        None if early_stop is None else early_stop.get("observed_go_opportunity_count")
    )
    return result


def init_aggregate_summary() -> dict[str, Any]:
    return {
        "games": 0,
        "wins_a": 0.0,
        "wins_b": 0.0,
        "draws": 0.0,
        "gold_sum_a": 0.0,
        "go_count_a": 0.0,
        "go_games_a": 0.0,
        "go_fail_count_a": 0.0,
        "go_opportunity_count_a": 0.0,
        "requested_games": 0,
        "early_stop_triggered": False,
        "early_stop_reason": None,
    }


def merge_duel_summary(aggregate: dict[str, Any], summary: dict[str, Any]) -> None:
    games = int(summary.get("games", 0) or 0)
    aggregate["games"] += games
    aggregate["wins_a"] += float(summary.get("wins_a", 0) or 0)
    aggregate["wins_b"] += float(summary.get("wins_b", 0) or 0)
    aggregate["draws"] += float(summary.get("draws", 0) or 0)
    aggregate["gold_sum_a"] += float(summary.get("mean_gold_delta_a", 0.0) or 0.0) * float(games)
    aggregate["go_count_a"] += float(summary.get("go_count_a", 0) or 0)
    aggregate["go_games_a"] += float(summary.get("go_games_a", 0) or 0)
    aggregate["go_fail_count_a"] += float(summary.get("go_fail_count_a", 0) or 0)
    aggregate["go_opportunity_count_a"] += float(summary.get("go_opportunity_count_a", 0) or 0)


def finalize_aggregate_summary(aggregate: dict[str, Any]) -> dict[str, Any]:
    games_total = max(1.0, float(aggregate["games"]))
    go_count = float(aggregate["go_count_a"])
    go_opp_count = float(aggregate["go_opportunity_count_a"])
    return {
        "games": int(aggregate["games"]),
        "requested_games": int(aggregate["requested_games"]),
        "wins_a": float(aggregate["wins_a"]),
        "wins_b": float(aggregate["wins_b"]),
        "draws": float(aggregate["draws"]),
        "win_rate_a": float(aggregate["wins_a"]) / games_total,
        "win_rate_b": float(aggregate["wins_b"]) / games_total,
        "draw_rate": float(aggregate["draws"]) / games_total,
        "mean_gold_delta_a": float(aggregate["gold_sum_a"]) / games_total,
        "go_count_a": int(aggregate["go_count_a"]),
        "go_games_a": int(aggregate["go_games_a"]),
        "go_fail_count_a": int(aggregate["go_fail_count_a"]),
        "go_fail_rate_a": (float(aggregate["go_fail_count_a"]) / go_count) if go_count > 0.0 else 0.0,
        "go_opportunity_count_a": int(aggregate["go_opportunity_count_a"]),
        "go_take_rate_a": (go_count / go_opp_count) if go_opp_count > 0.0 else 0.0,
        "early_stop_triggered": bool(aggregate["early_stop_triggered"]),
        "early_stop_reason": aggregate["early_stop_reason"],
    }


def _run_single_duel(
    repo_root: str | Path,
    runtime_path: str | Path,
    opponent_policy: str,
    *,
    games: int,
    max_steps: int,
    first_turn_policy: str,
    continuous_series: bool,
    seed: str,
) -> dict[str, Any]:
    root = Path(repo_root)
    argv = [
        "node",
        str(root / "scripts" / "model_duel_worker.mjs"),
        "--human",
        str(Path(runtime_path).resolve()),
        "--ai",
        str(opponent_policy),
        "--games",
        str(max(1, int(games))),
        "--seed",
        str(seed),
        "--max-steps",
        str(max(20, int(max_steps))),
        "--first-turn-policy",
        str(first_turn_policy or "alternate"),
        "--continuous-series",
        "1" if continuous_series else "2",
        "--stdout-format",
        "json",
    ]
    completed = subprocess.run(
        argv,
        cwd=str(root),
        capture_output=True,
        text=True,
        check=True,
    )
    stdout_lines = [line.strip() for line in str(completed.stdout or "").splitlines() if line.strip()]
    if not stdout_lines:
        raise RuntimeError("model_duel_worker produced no stdout")
    return dict(json.loads(stdout_lines[-1]))


def run_duel_eval(
    repo_root: str | Path,
    runtime_model: dict[str, Any],
    runtime_settings: dict[str, Any],
    *,
    games: int | None = None,
    seed: str = "deep_hyperneat_matgo",
) -> dict[str, Any]:
    settings = _normalize_runtime_settings(runtime_settings)
    requested_games = max(1, int(games or settings["games_per_genome"]))
    checkpoints = _collect_eval_checkpoints(settings, requested_games)
    with tempfile.TemporaryDirectory(prefix="deep_hyperneat_matgo_") as temp_dir:
        runtime_path = Path(temp_dir) / "runtime.json"
        runtime_path.write_text(json.dumps(runtime_model, indent=2), encoding="utf-8")

        mix = list(settings.get("opponent_policy_mix") or [])
        aggregate = init_aggregate_summary()
        aggregate["requested_games"] = requested_games
        opponent_policy_counts: dict[str, int] = {}
        if mix:
            previous_counts = [0 for _ in mix]
            for checkpoint in checkpoints:
                counts = allocate_games(checkpoint, mix)
                for index, (entry, count, previous_count) in enumerate(zip(mix, counts, previous_counts)):
                    delta = max(0, int(count) - int(previous_count))
                    if delta <= 0:
                        continue
                    policy = str(entry["policy"])
                    summary = _run_single_duel(
                        repo_root,
                        runtime_path,
                        policy,
                        games=delta,
                        max_steps=int(settings["max_eval_steps"]),
                        first_turn_policy=str(settings["first_turn_policy"]),
                        continuous_series=bool(settings["continuous_series"]),
                        seed=f"{seed}_{checkpoint}_{index}",
                    )
                    merge_duel_summary(aggregate, summary)
                    opponent_policy_counts[policy] = int(opponent_policy_counts.get(policy, 0)) + int(delta)
                previous_counts = counts
                current_summary = finalize_aggregate_summary(aggregate)
                early_stop = _check_early_stop(current_summary, settings, checkpoint)
                if early_stop is not None:
                    return _finalize_eval_summary(current_summary, settings, opponent_policy_counts, early_stop)
            return _finalize_eval_summary(
                finalize_aggregate_summary(aggregate),
                settings,
                opponent_policy_counts,
                None,
            )

        opponent_policy = str(settings.get("opponent_policy") or "").strip()
        if not opponent_policy:
            raise RuntimeError("runtime settings require opponent_policy or opponent_policy_mix")
        previous_games = 0
        for checkpoint in checkpoints:
            delta = max(0, int(checkpoint) - int(previous_games))
            if delta > 0:
                summary = _run_single_duel(
                    repo_root,
                    runtime_path,
                    opponent_policy,
                    games=delta,
                    max_steps=int(settings["max_eval_steps"]),
                    first_turn_policy=str(settings["first_turn_policy"]),
                    continuous_series=bool(settings["continuous_series"]),
                    seed=f"{seed}_{checkpoint}",
                )
                merge_duel_summary(aggregate, summary)
                opponent_policy_counts[opponent_policy] = int(opponent_policy_counts.get(opponent_policy, 0)) + int(delta)
            previous_games = checkpoint
            current_summary = finalize_aggregate_summary(aggregate)
            early_stop = _check_early_stop(current_summary, settings, checkpoint)
            if early_stop is not None:
                return _finalize_eval_summary(current_summary, settings, opponent_policy_counts, early_stop)
        return _finalize_eval_summary(
            finalize_aggregate_summary(aggregate),
            settings,
            opponent_policy_counts,
            None,
        )


def compute_fitness_from_summary(summary: dict[str, Any], runtime_settings: dict[str, Any]) -> tuple[float, dict[str, float]]:
    settings = _normalize_runtime_settings(runtime_settings)
    weighted_win_rate = float(summary.get("win_rate_a", 0.0) or 0.0)
    weighted_draw_rate = float(summary.get("draw_rate", 0.0) or 0.0)
    weighted_loss_rate = float(summary.get("win_rate_b", 0.0) or 0.0)
    weighted_mean_gold_delta = float(summary.get("mean_gold_delta_a", 0.0) or 0.0)

    gold_norm = math.tanh(
        (weighted_mean_gold_delta - float(settings["fitness_gold_neutral_delta"]))
        / max(1e-9, float(settings["fitness_gold_scale"]))
    )
    expected_result_raw = (
        clamp01(weighted_win_rate) + (0.5 * clamp01(weighted_draw_rate)) - clamp01(weighted_loss_rate)
    )
    expected_result = max(-1.0, min(1.0, expected_result_raw))
    neutral_expected_result = (2.0 * float(settings["fitness_win_neutral_rate"])) - 1.0
    if expected_result >= neutral_expected_result:
        result_upper_span = max(1e-9, 1.0 - neutral_expected_result)
        result_norm = clamp01((expected_result - neutral_expected_result) / result_upper_span)
    else:
        result_lower_span = max(1e-9, neutral_expected_result + 1.0)
        result_norm = -clamp01((neutral_expected_result - expected_result) / result_lower_span)

    fitness = (float(settings["fitness_gold_weight"]) * gold_norm) + (
        float(settings["fitness_win_weight"]) * result_norm
    )
    return fitness, {
        "gold_norm": gold_norm,
        "result_norm": result_norm,
        "expected_result": expected_result,
        "weighted_win_rate": weighted_win_rate,
        "weighted_draw_rate": weighted_draw_rate,
        "weighted_loss_rate": weighted_loss_rate,
        "weighted_mean_gold_delta": weighted_mean_gold_delta,
    }


def evaluate_substrate(
    substrate: Any,
    repo_root: str | Path,
    runtime_settings: dict[str, Any],
    *,
    games: int | None = None,
    seed: str = "deep_hyperneat_matgo",
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    runtime_model = substrate_to_runtime_model(substrate)
    summary = run_duel_eval(repo_root, runtime_model, runtime_settings, games=games, seed=seed)
    fitness, components = compute_fitness_from_summary(summary, runtime_settings)
    return float(fitness), summary, {
        "runtime_model": runtime_model,
        "fitness_components": components,
    }
