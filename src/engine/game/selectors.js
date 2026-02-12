import { groupByMonth } from "../../cards.js";

export function getDeclarableShakingMonths(state, playerKey) {
  if (state.phase !== "playing" || state.currentTurn !== playerKey) return [];
  const player = state.players[playerKey];
  const declared = new Set(player.shakingDeclaredMonths || []);
  const counts = {};
  player.hand.forEach((c) => {
    if (c.month <= 12) counts[c.month] = (counts[c.month] || 0) + 1;
  });
  return Object.entries(counts)
    .map(([m, count]) => ({ month: Number(m), count }))
    .filter((x) => x.count >= 3 && !declared.has(x.month))
    .filter((x) => state.board.every((b) => b.month !== x.month))
    .map((x) => x.month);
}

export function getDeclarableBombMonths(state, playerKey) {
  if (state.phase !== "playing" || state.currentTurn !== playerKey) return [];
  const player = state.players[playerKey];
  const counts = {};
  player.hand.forEach((c) => {
    if (c.month <= 12) counts[c.month] = (counts[c.month] || 0) + 1;
  });
  return Object.entries(counts)
    .map(([m, count]) => ({ month: Number(m), count }))
    .filter((x) => x.count >= 3)
    .filter((x) => state.board.filter((b) => b.month === x.month).length === 1)
    .map((x) => x.month);
}

export function getShakingReveal(state, now) {
  if (state.actionReveal && state.actionReveal.expiresAt > now) return state.actionReveal;
  if (!state.shakingReveal) return null;
  if (state.shakingReveal.expiresAt <= now) return null;
  return state.shakingReveal;
}

export function estimateRemaining(state) {
  const seen = [
    ...state.board,
    ...state.players.human.hand,
    ...state.players.ai.hand,
    ...state.players.human.captured.kwang,
    ...state.players.human.captured.five,
    ...state.players.human.captured.ribbon,
    ...state.players.human.captured.junk,
    ...state.players.ai.captured.kwang,
    ...state.players.ai.captured.five,
    ...state.players.ai.captured.ribbon,
    ...state.players.ai.captured.junk
  ];

  const groupedSeen = groupByMonth(seen);
  const result = {};
  for (let m = 1; m <= 12; m += 1) {
    const seenCount = groupedSeen[m]?.length ?? 0;
    result[m] = 4 - seenCount;
  }
  return result;
}
