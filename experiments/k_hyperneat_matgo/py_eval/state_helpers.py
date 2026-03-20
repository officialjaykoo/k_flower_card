from __future__ import annotations

from typing import Callable

from .models import Card, CapturedCards


def create_seeded_rng(seed_text: str = "") -> Callable[[], float]:
    h = 1779033703 ^ len(seed_text)
    for char in seed_text:
        h = ((h ^ ord(char)) * 3432918353) & 0xFFFFFFFF
        h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
    h = ((h ^ (h >> 16)) * 2246822507) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 3266489909) & 0xFFFFFFFF
    h ^= h >> 16
    h &= 0xFFFFFFFF

    def seeded_random() -> float:
        nonlocal h
        t = (h + 0x6D2B79F5) & 0xFFFFFFFF
        h = t
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return float(((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0)

    return seeded_random


def unique_cards_by_id(cards: list[Card] | None = None) -> list[Card]:
    seen: set[str] = set()
    result: list[Card] = []
    for card in cards or []:
        if not card or not card.id or card.id in seen:
            continue
        seen.add(card.id)
        result.append(card)
    return result


def strip_captured_cards_by_ids(captured: CapturedCards, card_id_set: set[str]) -> CapturedCards:
    return CapturedCards(
        kwang=[card for card in captured.kwang if card.id not in card_id_set],
        five=[card for card in captured.five if card.id not in card_id_set],
        ribbon=[card for card in captured.ribbon if card.id not in card_id_set],
        junk=[card for card in captured.junk if card.id not in card_id_set],
    )


def bonus_steal_count(cards: list[Card] | None = None) -> int:
    return sum(int((card.bonus or {}).get("stealPi", 0)) for card in cards or [])


def apply_ppuk_board_stay_rule(
    *,
    is_last_hand_turn: bool = False,
    match_events: list[dict[str, object]] | None = None,
    board: list[Card] | None = None,
    captured: CapturedCards,
    newly_captured: list[Card] | None = None,
    rollback_cards: list[Card] | None = None,
    held_bonus_cards: list[Card] | None = None,
) -> dict[str, object]:
    ppuk_triggered = (
        not is_last_hand_turn
        and any(evt.get("source") == "flip" and evt.get("eventTag") == "PPUK" for evt in (match_events or []))
    )
    if not ppuk_triggered:
        return {
            "board": list(board or []),
            "captured": captured,
            "newlyCaptured": list(newly_captured or []),
            "heldBonusCards": list(held_bonus_cards or []),
        }

    stay_cards = unique_cards_by_id([*(rollback_cards or []), *(held_bonus_cards or [])])
    stay_id_set = {card.id for card in stay_cards}
    return {
        "board": unique_cards_by_id([*(board or []), *stay_cards]),
        "captured": strip_captured_cards_by_ids(captured, stay_id_set),
        "newlyCaptured": [card for card in (newly_captured or []) if card.id not in stay_id_set],
        "heldBonusCards": [],
    }
