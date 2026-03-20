from __future__ import annotations

from .models import EngineState


def get_action_player_key(state: EngineState) -> str | None:
    if state.phase == "playing":
        return state.current_turn
    if state.phase == "go-stop":
        return state.pending_go_stop
    if state.phase == "select-match":
        return str(state.pending_match.get("playerKey")) if state.pending_match else None
    if state.phase == "president-choice":
        return str(state.pending_president.get("playerKey")) if state.pending_president else None
    if state.phase == "shaking-confirm":
        return str(state.pending_shaking_confirm.get("playerKey")) if state.pending_shaking_confirm else None
    if state.phase == "gukjin-choice":
        return str(state.pending_gukjin_choice.get("playerKey")) if state.pending_gukjin_choice else None
    return None
