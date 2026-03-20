from __future__ import annotations

from .models import Card, CapturedCards, PlayerState
from .scoring import get_gukjin_mode, is_gukjin_card, pi_value


CAPTURE_ZONES = ("kwang", "five", "ribbon", "junk")


def category_key(card: Card) -> str:
    if card.category == "kwang":
        return "kwang"
    if card.category == "five":
        return "five"
    if card.category == "ribbon":
        return "ribbon"
    return "junk"


def has_captured_card_id(captured: CapturedCards, card_id: str) -> bool:
    return any(any(card.id == card_id for card in getattr(captured, zone)) for zone in CAPTURE_ZONES)


def push_captured(captured: CapturedCards, card: Card | None) -> None:
    if not card or has_captured_card_id(captured, card.id):
        return
    getattr(captured, category_key(card)).append(card)


def best_match_card(cards: list[Card]) -> Card | None:
    ranked = sorted(cards, key=lambda card: (-pi_value(card), card.id))
    return ranked[0] if ranked else None


def needs_choice(cards: list[Card] | None) -> bool:
    if not cards or len(cards) < 2:
        return False
    return cards[0].category != cards[1].category


def steal_pi_from_opponent(players: dict[str, PlayerState], taker_key: str, count: int) -> dict[str, object]:
    giver_key = "ai" if taker_key == "human" else "human"
    taker = players[taker_key]
    giver = players[giver_key]
    steal_log: list[str] = []
    remaining = int(count)

    while remaining > 0:
        junk_only_candidates = [
            {
                "idx": idx,
                "card": card,
                "value": pi_value(card),
                "is_gukjin": bool(getattr(card, "gukjin_transformed", False) or (is_gukjin_card(card) and card.category == "junk")),
                "source": "junk",
            }
            for idx, card in enumerate(giver.captured.junk)
        ]
        candidates = list(junk_only_candidates)

        if get_gukjin_mode(giver) == "junk":
            gukjin_idx = next(
                (idx for idx, card in enumerate(giver.captured.five) if is_gukjin_card(card) and not getattr(card, "gukjin_transformed", False)),
                -1,
            )
            if gukjin_idx >= 0:
                candidates.append(
                    {
                        "idx": gukjin_idx,
                        "card": giver.captured.five[gukjin_idx],
                        "value": 2,
                        "is_gukjin": True,
                        "source": "gukjin",
                    }
                )

        if not candidates:
            break

        def steal_rank(item: dict[str, object]) -> int:
            value = int(item["value"])
            is_gukjin = bool(item["is_gukjin"])
            if value == 1:
                return 1
            if value == 2 and not is_gukjin:
                return 2
            if is_gukjin:
                return 3
            if value >= 3:
                return 4
            return 9

        candidates.sort(key=lambda item: (steal_rank(item), -int(item["idx"]), str(item["card"].id)))  # type: ignore[index]
        pick = candidates[0]
        if pick["source"] == "gukjin":
            gukjin_card = giver.captured.five.pop(int(pick["idx"]))
            stolen = Card(
                id=gukjin_card.id,
                month=gukjin_card.month,
                category="junk",
                name=f"{gukjin_card.name} (Gukjin Pi)",
                combo_tags=gukjin_card.combo_tags,
                pi_value=2,
                bonus=gukjin_card.bonus,
                asset=gukjin_card.asset,
                gukjin_transformed=True,
            )
        else:
            stolen = giver.captured.junk.pop(int(pick["idx"]))

        taker.captured.junk.append(stolen)
        steal_log.append(f"{taker.label}: stole 1 pi from {giver.label} ({stolen.name}, value {pi_value(stolen)})")
        remaining -= 1

    return {"updatedPlayers": {**players, taker_key: taker, giver_key: giver}, "stealLog": steal_log}
