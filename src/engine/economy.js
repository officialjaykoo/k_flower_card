/* ============================================================================
 * Gold economy helpers
 * - points -> gold conversion
 * - winner takes gold from opponent with bankruptcy guard
 * ========================================================================== */

export const STARTING_GOLD = 1_000_000;
export const POINT_GOLD_UNIT = 100;

export function pointsToGold(points) {
  return Math.max(0, Math.floor((points || 0) * POINT_GOLD_UNIT));
}

/* Transfer up to `amount` from opponent to taker. */
export function stealGoldFromOpponent(players, takerKey, amount) {
  const giverKey = takerKey === "human" ? "ai" : "human";
  const taker = { ...players[takerKey] };
  const giver = { ...players[giverKey] };
  const available = Math.max(0, giver.gold || 0);
  const paid = Math.max(0, Math.min(amount, available));

  taker.gold = (taker.gold || 0) + paid;
  giver.gold = available - paid;

  const logs = [`${taker.label}: gains ${paid} gold (from ${giver.label})`];
  if (paid < amount) logs.push(`${giver.label}: could not fully pay (${amount - paid} unpaid)`);
  if (giver.gold <= 0) {
    logs.push(`${giver.label}: bankrupt (0 gold). Next game starts with ${STARTING_GOLD} gold.`);
  }

  return {
    updatedPlayers: { ...players, [takerKey]: taker, [giverKey]: giver },
    log: logs
  };
}

/* Round-end settlement wrapper with paid/requested bookkeeping. */
export function settleRoundGold(players, winnerKey, scorePoint) {
  if (winnerKey !== "human" && winnerKey !== "ai") {
    return { players, log: [], paid: 0, requested: 0 };
  }

  const requested = pointsToGold(scorePoint);
  const before = players[winnerKey].gold || 0;
  const result = stealGoldFromOpponent(players, winnerKey, requested);
  const after = result.updatedPlayers[winnerKey].gold || 0;
  const paid = Math.max(0, after - before);

  return { ...result, paid, requested };
}
