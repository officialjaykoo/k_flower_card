from __future__ import annotations

from .cards import build_deck
from .models import Card


COMBO_KEYS = ("redRibbons", "blueRibbons", "plainRibbons", "fiveBirds")


def unique_cards(cards: list[Card] | None = None) -> list[Card]:
    seen: set[str] = set()
    out: list[Card] = []
    for card in cards or []:
        if not card:
            continue
        key = card.id or f"{card.month}:{card.category}:{card.name}"
        if key in seen:
            continue
        seen.add(key)
        out.append(card)
    return out


def has_combo_tag(card: Card | None, tag: str) -> bool:
    return bool(card and tag in card.combo_tags)


def count_combo_tag(cards: list[Card] | None, tag: str) -> int:
    return sum(1 for card in unique_cards(cards) if has_combo_tag(card, tag))


def _build_combo_month_map() -> dict[str, tuple[int, ...]]:
    raw: dict[str, set[int]] = {tag: set() for tag in COMBO_KEYS}
    for card in build_deck():
        for tag in card.combo_tags:
            if tag in raw:
                raw[tag].add(card.month)
    return {tag: tuple(sorted(raw[tag])) for tag in COMBO_KEYS}


COMBO_MONTHS = _build_combo_month_map()
COMBO_MONTH_SETS = {tag: set(months) for tag, months in COMBO_MONTHS.items()}


def missing_combo_months(cards: list[Card] | None, tag: str) -> list[int]:
    months = COMBO_MONTHS.get(tag, ())
    own_months = {card.month for card in unique_cards(cards) if has_combo_tag(card, tag)}
    return [month for month in months if month not in own_months]
