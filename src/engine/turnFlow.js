export function clearExpiredReveal(state, now = Date.now()) {
  if (!state.shakingReveal) return state;
  if (state.shakingReveal.expiresAt > now) return state;
  return { ...state, shakingReveal: null };
}

export function ensurePassCardFor(state, playerKey) {
  const player = state.players[playerKey];
  if (!player || player.hand.length > 0 || state.deck.length === 0) return state;
  const passCard = {
    id: `pass-${playerKey}-${state.log.length}`,
    month: 0,
    category: "junk",
    name: "Pass",
    passCard: true,
    asset: ""
  };
  const nextPlayers = {
    ...state.players,
    [playerKey]: { ...player, hand: [passCard] }
  };
  return {
    ...state,
    players: nextPlayers,
    log: state.log.concat(`${player.label}: 패스 카드 자동 생성`)
  };
}
