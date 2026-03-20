from __future__ import annotations

from datetime import datetime
from typing import Callable

from .captures import push_captured
from .economy import STARTING_GOLD
from .models import Card, PlayerState


def decide_first_turn(human_card: Card, ai_card: Card, rng: Callable[[], float]) -> dict[str, str]:
    hour = datetime.now().hour
    is_night = hour >= 18 or hour < 6

    winner_key = "human"
    reason = ""
    if human_card.month == ai_card.month:
        winner_key = "human" if rng() < 0.5 else "ai"
        reason = "same month random"
    elif is_night:
        winner_key = "human" if human_card.month < ai_card.month else "ai"
        reason = "night rule: lower month starts"
    else:
        winner_key = "human" if human_card.month > ai_card.month else "ai"
        reason = "day rule: higher month starts"

    winner_label = "Player" if winner_key == "human" else "AI"
    log = (
        f"Starter decided [{'night' if is_night else 'day'}]: "
        f"Player {human_card.month} vs AI {ai_card.month} -> {winner_label} ({reason})"
    )
    return {"winnerKey": winner_key, "log": log}


def empty_player(label: str) -> PlayerState:
    return PlayerState(label=label, gold=STARTING_GOLD)


def normalize_opening_hands(players: dict[str, PlayerState], remain: list[Card], init_log: list[str]) -> list[Card]:
    _ = (players, init_log)
    return list(remain)


def normalize_opening_board(board: list[Card], remain: list[Card], first_player: PlayerState, init_log: list[str]) -> dict[str, list[Card]]:
    next_board = list(board)
    next_remain = list(remain)
    changed = True
    while changed:
        changed = False
        bonus_cards = [card for card in next_board if card.bonus and card.bonus.get("stealPi")]
        if not bonus_cards:
            break
        changed = True
        for card in bonus_cards:
            push_captured(first_player.captured, card)
            init_log.append(f"Opening adjust: {first_player.label} captures board bonus card {card.name}")
        next_board = [card for card in next_board if not (card.bonus and card.bonus.get("stealPi"))]
        while len(next_board) < 8 and next_remain:
            next_board.append(next_remain.pop(0))
    return {"board": next_board, "remain": next_remain}


def find_president_month(cards: list[Card]) -> int | None:
    counts: dict[int, int] = {}
    for card in cards:
        if not card:
            continue
        if getattr(card, "pass_card", False):
            continue
        if card.month < 1 or card.month > 12:
            continue
        counts[card.month] = counts.get(card.month, 0) + 1
        if counts[card.month] >= 4:
            return card.month
    return None
