from __future__ import annotations

from .combos import count_combo_tag, unique_cards
from .models import Card, PlayerState
from .rules import RULE_SETS, RuleSet, normalize_rule_key


NON_BRIGHT_KWANG_ID = "L0"
GUKJIN_CARD_ID = "I0"


def calculate_score(player: PlayerState, opponent: PlayerState, rule_key: str | None = None) -> dict[str, object]:
    rules = RULE_SETS[normalize_rule_key(rule_key)]
    base_info = calculate_base_score(player, rules)
    go_bonus = int(player.go_count or 0)
    point_total = int(base_info["base"]) + go_bonus

    payout_multiplier = 1
    if player.go_count >= 3:
        payout_multiplier *= 2 ** (player.go_count - 2)
    if player.events.shaking > 0:
        payout_multiplier *= 2 ** player.events.shaking
    if player.events.bomb > 0:
        payout_multiplier *= 2 ** player.events.bomb

    bak = detect_bak(player, opponent, rule_key)
    payout_multiplier *= int(bak["multiplier"])

    breakdown = dict(base_info["breakdown"])
    breakdown["goBonus"] = go_bonus
    return {
        "base": point_total,
        "multiplier": payout_multiplier,
        "total": point_total,
        "payoutTotal": point_total * payout_multiplier,
        "bak": bak,
        "breakdown": breakdown,
    }


def calculate_base_score(player: PlayerState, rules: RuleSet | None = None) -> dict[str, object]:
    active_rules = rules or RULE_SETS["A"]
    kwang = unique_cards(player.captured.kwang)
    ribbons = unique_cards(player.captured.ribbon)
    five_cards = scoring_five_cards(player)
    five_count = len(five_cards)
    ribbon_count = len(ribbons)
    junk_count = scoring_pi_count(player)

    kwang_base = kwang_base_score(kwang)
    five_base = five_count - 4 if five_count >= 5 else 0
    ribbon_base = ribbon_count - 4 if ribbon_count >= 5 else 0
    junk_base = junk_count - 9 if junk_count >= 10 else 0
    ribbon_set_bonus = ribbon_bonus(ribbons)
    five_set_bonus = five_bonus(five_cards)
    base = kwang_base + five_base + ribbon_base + junk_base + ribbon_set_bonus + five_set_bonus

    return {
        "base": base,
        "breakdown": {
            "kwangBase": kwang_base,
            "fiveBase": five_base,
            "ribbonBase": ribbon_base,
            "junkBase": junk_base,
            "ribbonSetBonus": ribbon_set_bonus,
            "fiveSetBonus": five_set_bonus,
            "piCount": junk_count,
            "gukjinMode": get_gukjin_mode(player),
            "goMinScore": active_rules.go_min_score,
        },
    }


def ribbon_bonus(ribbons: list[Card]) -> int:
    bonus = 0
    if count_combo_tag(ribbons, "redRibbons") >= 3:
        bonus += 3
    if count_combo_tag(ribbons, "blueRibbons") >= 3:
        bonus += 3
    if count_combo_tag(ribbons, "plainRibbons") >= 3:
        bonus += 3
    return bonus


def five_bonus(fives: list[Card]) -> int:
    return 5 if count_combo_tag(fives, "fiveBirds") >= 3 else 0


def kwang_base_score(kwang_cards: list[Card]) -> int:
    count = len(kwang_cards)
    if count < 3:
        return 0
    if count == 3:
        return 2 if any(card.id == NON_BRIGHT_KWANG_ID for card in kwang_cards) else 3
    if count == 4:
        return 4
    return 15


def detect_bak(player: PlayerState, opponent: PlayerState, rule_key: str | None = None) -> dict[str, object]:
    rules = RULE_SETS[normalize_rule_key(rule_key)]
    opponent_pi = scoring_pi_count(opponent)
    player_pi = scoring_pi_count(player)
    player_five_count = len(scoring_five_cards(player))
    opponent_five_count = len(scoring_five_cards(opponent))

    gwang = len(unique_cards(player.captured.kwang)) >= 3 and len(unique_cards(opponent.captured.kwang)) == 0
    pi = opponent_pi >= 1 and opponent_pi <= 7 and player_pi >= 10
    mong_bak = opponent_five_count == 0 and player_five_count >= 7

    multiplier = 1
    if gwang:
        multiplier *= rules.bak_multipliers["gwang"]
    if pi:
        multiplier *= rules.bak_multipliers["pi"]
    if mong_bak:
        multiplier *= rules.bak_multipliers["mongBak"]
    return {"gwang": gwang, "pi": pi, "mongBak": mong_bak, "multiplier": multiplier}


def pi_value(card: Card | None) -> int:
    if not card:
        return 1
    explicit = int(card.pi_value or 0)
    return explicit if explicit > 0 else 1


def sum_pi_values(cards: list[Card] | None = None) -> int:
    return sum(pi_value(card) for card in unique_cards(cards))


def is_gukjin_card(card: Card | None) -> bool:
    return bool(card and card.id == GUKJIN_CARD_ID)


def get_gukjin_mode(player: PlayerState) -> str:
    return "junk" if player.gukjin_mode == "junk" else "five"


def scoring_five_cards(player: PlayerState) -> list[Card]:
    fives = unique_cards(player.captured.five)
    if get_gukjin_mode(player) == "five":
        return fives
    return [card for card in fives if not is_gukjin_card(card)]


def scoring_pi_count(player: PlayerState) -> int:
    base_pi = sum_pi_values(player.captured.junk)
    if get_gukjin_mode(player) != "junk":
        return base_pi
    has_gukjin = any(is_gukjin_card(card) for card in unique_cards(player.captured.five))
    return base_pi + 2 if has_gukjin else base_pi
