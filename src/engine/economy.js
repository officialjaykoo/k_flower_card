export const STARTING_GOLD = 1_000_000;
export const POINT_GOLD_UNIT = 100;

export function pointsToGold(points) {
  return Math.max(0, Math.floor((points || 0) * POINT_GOLD_UNIT));
}

export function stealGoldFromOpponent(players, takerKey, amount) {
  const giverKey = takerKey === "human" ? "ai" : "human";
  const taker = { ...players[takerKey] };
  const giver = { ...players[giverKey] };
  const available = Math.max(0, giver.gold || 0);
  const paid = Math.max(0, Math.min(amount, available));
  taker.gold = (taker.gold || 0) + paid;
  giver.gold = available - paid;
  const logs = [`${taker.label}: 골드 ${paid} 획득 (상대 ${giver.label})`];
  if (paid < amount) logs.push(`${giver.label}: 잔액 부족으로 ${amount - paid}골드 미지급`);
  if (giver.gold <= 0) {
    logs.push(`${giver.label}: 파산(0골드). 다음 게임 시작 시 ${STARTING_GOLD}골드로 재시작`);
  }
  return {
    updatedPlayers: { ...players, [takerKey]: taker, [giverKey]: giver },
    log: logs
  };
}

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
