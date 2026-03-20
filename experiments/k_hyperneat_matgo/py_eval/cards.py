from __future__ import annotations

import random

from .models import Card


DEFAULT_CARD_THEME = "original"
CARD_THEMES = ("original", "k-flower")


def normalize_card_theme(theme: str | None) -> str:
    return theme if theme in CARD_THEMES else DEFAULT_CARD_THEME


def build_card_asset_path(card_id: str, theme: str = DEFAULT_CARD_THEME) -> str:
    return f"/cards/{normalize_card_theme(theme)}/{card_id}.svg"


def build_card_ui_asset_path(filename: str, theme: str = DEFAULT_CARD_THEME) -> str:
    return f"/cards/{normalize_card_theme(theme)}/{filename}"


_CARD_CATALOG: tuple[dict[str, object], ...] = (
    {"card_id": "A0", "month": 1, "category": "kwang", "name": "Pine Bright"},
    {"card_id": "A1", "month": 1, "category": "ribbon", "name": "Poetry Ribbon", "combo_tags": ("redRibbons",)},
    {"card_id": "A2", "month": 1, "category": "junk", "name": "Pine Junk A", "pi_value": 1},
    {"card_id": "A3", "month": 1, "category": "junk", "name": "Pine Junk B", "pi_value": 1},
    {"card_id": "B0", "month": 2, "category": "five", "name": "Bush Warbler", "combo_tags": ("fiveBirds",)},
    {"card_id": "B1", "month": 2, "category": "ribbon", "name": "Poetry Ribbon", "combo_tags": ("redRibbons",)},
    {"card_id": "B2", "month": 2, "category": "junk", "name": "Plum Junk A", "pi_value": 1},
    {"card_id": "B3", "month": 2, "category": "junk", "name": "Plum Junk B", "pi_value": 1},
    {"card_id": "C0", "month": 3, "category": "kwang", "name": "Cherry Bright"},
    {"card_id": "C1", "month": 3, "category": "ribbon", "name": "Poetry Ribbon", "combo_tags": ("redRibbons",)},
    {"card_id": "C2", "month": 3, "category": "junk", "name": "Cherry Junk A", "pi_value": 1},
    {"card_id": "C3", "month": 3, "category": "junk", "name": "Cherry Junk B", "pi_value": 1},
    {"card_id": "D0", "month": 4, "category": "five", "name": "Cuckoo", "combo_tags": ("fiveBirds",)},
    {"card_id": "D1", "month": 4, "category": "ribbon", "name": "Plant Ribbon", "combo_tags": ("plainRibbons",)},
    {"card_id": "D2", "month": 4, "category": "junk", "name": "Wisteria Junk A", "pi_value": 1},
    {"card_id": "D3", "month": 4, "category": "junk", "name": "Wisteria Junk B", "pi_value": 1},
    {"card_id": "E0", "month": 5, "category": "five", "name": "Bridge"},
    {"card_id": "E1", "month": 5, "category": "ribbon", "name": "Plant Ribbon", "combo_tags": ("plainRibbons",)},
    {"card_id": "E2", "month": 5, "category": "junk", "name": "Iris Junk A", "pi_value": 1},
    {"card_id": "E3", "month": 5, "category": "junk", "name": "Iris Junk B", "pi_value": 1},
    {"card_id": "F0", "month": 6, "category": "five", "name": "Butterflies"},
    {"card_id": "F1", "month": 6, "category": "ribbon", "name": "Blue Ribbon", "combo_tags": ("blueRibbons",)},
    {"card_id": "F2", "month": 6, "category": "junk", "name": "Peony Junk A", "pi_value": 1},
    {"card_id": "F3", "month": 6, "category": "junk", "name": "Peony Junk B", "pi_value": 1},
    {"card_id": "G0", "month": 7, "category": "five", "name": "Boar"},
    {"card_id": "G1", "month": 7, "category": "ribbon", "name": "Plant Ribbon", "combo_tags": ("plainRibbons",)},
    {"card_id": "G2", "month": 7, "category": "junk", "name": "Clover Junk A", "pi_value": 1},
    {"card_id": "G3", "month": 7, "category": "junk", "name": "Clover Junk B", "pi_value": 1},
    {"card_id": "H0", "month": 8, "category": "kwang", "name": "Moon Bright"},
    {"card_id": "H1", "month": 8, "category": "five", "name": "Geese", "combo_tags": ("fiveBirds",)},
    {"card_id": "H2", "month": 8, "category": "junk", "name": "Pampas Junk A", "pi_value": 1},
    {"card_id": "H3", "month": 8, "category": "junk", "name": "Pampas Junk B", "pi_value": 1},
    {"card_id": "I0", "month": 9, "category": "five", "name": "Sake Cup"},
    {"card_id": "I1", "month": 9, "category": "ribbon", "name": "Blue Ribbon", "combo_tags": ("blueRibbons",)},
    {"card_id": "I2", "month": 9, "category": "junk", "name": "Chrysanthemum Junk A", "pi_value": 1},
    {"card_id": "I3", "month": 9, "category": "junk", "name": "Chrysanthemum Junk B", "pi_value": 1},
    {"card_id": "J0", "month": 10, "category": "five", "name": "Deer"},
    {"card_id": "J1", "month": 10, "category": "ribbon", "name": "Maple Ribbon", "combo_tags": ("blueRibbons",)},
    {"card_id": "J2", "month": 10, "category": "junk", "name": "Maple Junk A", "pi_value": 1},
    {"card_id": "J3", "month": 10, "category": "junk", "name": "Maple Junk B", "pi_value": 1},
    {"card_id": "K0", "month": 11, "category": "kwang", "name": "Willow Bright"},
    {"card_id": "K1", "month": 11, "category": "junk", "name": "Willow Double Junk", "pi_value": 2},
    {"card_id": "K2", "month": 11, "category": "junk", "name": "Willow Junk A", "pi_value": 1},
    {"card_id": "K3", "month": 11, "category": "junk", "name": "Willow Junk B", "pi_value": 1},
    {"card_id": "L0", "month": 12, "category": "kwang", "name": "Paulownia Bright"},
    {"card_id": "L1", "month": 12, "category": "five", "name": "Rain Five"},
    {"card_id": "L2", "month": 12, "category": "ribbon", "name": "Paulownia Ribbon"},
    {"card_id": "L3", "month": 12, "category": "junk", "name": "Paulownia Junk", "pi_value": 2},
    {"card_id": "M0", "month": 13, "category": "junk", "name": "Bonus Double", "pi_value": 2, "bonus": {"stealPi": 1}},
    {"card_id": "M1", "month": 13, "category": "junk", "name": "Bonus Triple", "pi_value": 3, "bonus": {"stealPi": 1}},
)


def build_deck(theme: str = DEFAULT_CARD_THEME) -> list[Card]:
    card_theme = normalize_card_theme(theme)
    return [
        Card(
            id=str(card["card_id"]),
            month=int(card["month"]),
            category=str(card["category"]),
            name=str(card["name"]),
            combo_tags=tuple(card.get("combo_tags", ())),  # type: ignore[arg-type]
            pi_value=int(card.get("pi_value", 0) or 0),
            bonus=dict(card.get("bonus", {})) or None,  # type: ignore[arg-type]
            asset=build_card_asset_path(str(card["card_id"]), card_theme),
        )
        for card in _CARD_CATALOG
    ]


def shuffle(deck: list[Card], rng: random.Random | None = None) -> list[Card]:
    source = list(deck)
    (rng or random.Random()).shuffle(source)
    return source
