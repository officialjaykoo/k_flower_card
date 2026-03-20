from __future__ import annotations


STARTING_GOLD = 100_000
POINT_GOLD_UNIT = 100


def points_to_gold(points: int | float) -> int:
    return max(0, int(float(points or 0) * POINT_GOLD_UNIT))


def steal_gold_from_opponent(players: dict[str, dict[str, object]], taker_key: str, amount: int) -> dict[str, object]:
    giver_key = "ai" if taker_key == "human" else "human"
    taker = dict(players[taker_key])
    giver = dict(players[giver_key])
    available = max(0, int(giver.get("gold", 0) or 0))
    paid = max(0, min(int(amount or 0), available))

    taker["gold"] = int(taker.get("gold", 0) or 0) + paid
    giver["gold"] = available - paid

    logs = [f"{taker.get('label', taker_key)}: gains {paid} gold (from {giver.get('label', giver_key)})"]
    if paid < amount:
        logs.append(f"{giver.get('label', giver_key)}: could not fully pay ({amount - paid} unpaid)")
    if int(giver["gold"]) <= 0:
        logs.append(f"{giver.get('label', giver_key)}: bankrupt (0 gold). Next game starts with {STARTING_GOLD} gold.")

    return {
        "updatedPlayers": {**players, taker_key: taker, giver_key: giver},
        "log": logs,
    }


def settle_round_gold(players: dict[str, dict[str, object]], winner_key: str, score_point: int | float) -> dict[str, object]:
    if winner_key not in {"human", "ai"}:
        return {"players": players, "log": [], "paid": 0, "requested": 0}

    requested = points_to_gold(score_point)
    before = int(players[winner_key].get("gold", 0) or 0)
    result = steal_gold_from_opponent(players, winner_key, requested)
    after = int(result["updatedPlayers"][winner_key].get("gold", 0) or 0)
    paid = max(0, after - before)
    return {**result, "paid": paid, "requested": requested}
