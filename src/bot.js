import {
  playTurn,
  getDeclarableShakingMonths,
  declareShaking,
  getDeclarableBombMonths,
  declareBomb,
  chooseGo,
  chooseStop,
  chooseKungUse,
  chooseKungPass,
  choosePresidentStop,
  choosePresidentHold,
  chooseGukjinMode,
  chooseMatch
} from "./gameEngine.js";

export function botChooseCard(state, playerKey) {
  const player = state.players[playerKey];
  if (!player || player.hand.length === 0) return null;
  return pickRandom(player.hand)?.id ?? null;
}

export function botPlay(state, playerKey) {
  if (state.phase === "kung-choice" && state.pendingKung?.playerKey === playerKey) {
    return Math.random() < 0.5 ? chooseKungUse(state, playerKey) : chooseKungPass(state, playerKey);
  }

  if (state.phase === "gukjin-choice" && state.pendingGukjinChoice?.playerKey === playerKey) {
    return chooseGukjinMode(state, playerKey, Math.random() < 0.5 ? "five" : "junk");
  }

  if (state.phase === "president-choice" && state.pendingPresident?.playerKey === playerKey) {
    return Math.random() < 0.5
      ? choosePresidentStop(state, playerKey)
      : choosePresidentHold(state, playerKey);
  }

  if (state.phase === "select-match" && state.pendingMatch?.playerKey === playerKey) {
    const option = pickRandom(
      state.board.filter((c) => (state.pendingMatch?.boardCardIds || []).includes(c.id))
    );
    return option ? chooseMatch(state, option.id) : state;
  }

  if (state.phase === "go-stop" && state.pendingGoStop === playerKey) {
    return Math.random() < 0.5 ? chooseGo(state, playerKey) : chooseStop(state, playerKey);
  }

  if (state.phase === "playing" && state.currentTurn === playerKey) {
    const bombMonths = getDeclarableBombMonths(state, playerKey);
    if (bombMonths.length > 0 && Math.random() < 0.5) {
      return declareBomb(state, playerKey, pickRandom(bombMonths));
    }
    const shakingMonths = getDeclarableShakingMonths(state, playerKey);
    if (shakingMonths.length > 0 && Math.random() < 0.5) {
      return declareShaking(state, playerKey, pickRandom(shakingMonths));
    }
    const cardId = botChooseCard(state, playerKey);
    if (!cardId) return state;
    return playTurn(state, cardId);
  }

  return state;
}

function pickRandom(arr) {
  if (!arr || arr.length === 0) return null;
  return arr[Math.floor(Math.random() * arr.length)];
}
