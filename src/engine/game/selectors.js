export function getDeclarableShakingMonths(state, playerKey) {
  if (state.phase !== "playing" || state.currentTurn !== playerKey) return [];
  const player = state.players[playerKey];
  const declared = new Set(player.shakingDeclaredMonths || []);
  const counts = {};
  player.hand.forEach((c) => {
    if (!c || c.passCard) return;
    if (c.month < 1 || c.month > 12) return;
    counts[c.month] = (counts[c.month] || 0) + 1;
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
    if (!c || c.passCard) return;
    if (c.month < 1 || c.month > 12) return;
    counts[c.month] = (counts[c.month] || 0) + 1;
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
