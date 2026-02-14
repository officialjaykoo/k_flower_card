import {
  playTurn,
  getDeclarableShakingMonths,
  declareShaking,
  getDeclarableBombMonths,
  declareBomb,
  chooseGo,
  chooseStop,
  choosePresidentStop,
  choosePresidentHold,
  chooseGukjinMode,
  chooseMatch
} from "./gameEngine.js";

const POLICY_RANDOM = "random";
const POLICY_RANDOM_V2 = "random_v2";
const POLICY_HEURISTIC_V1 = "heuristic_v1";
const POLICY_HEURISTIC_V2 = "heuristic_v2";

export const BOT_POLICIES = [
  POLICY_RANDOM,
  POLICY_RANDOM_V2,
  POLICY_HEURISTIC_V1,
  POLICY_HEURISTIC_V2
];

export function botChooseCard(state, playerKey, policy = POLICY_RANDOM) {
  const player = state.players[playerKey];
  if (!player || player.hand.length === 0) return null;
  if (isHeuristicPolicy(policy) || policy === POLICY_RANDOM_V2) {
    const ranked = rankHandCards(state, playerKey);
    if (ranked.length > 0) {
      if (isHeuristicPolicy(policy)) return ranked[0].card.id;
      const top = ranked.slice(0, Math.min(3, ranked.length));
      return pickRandom(top)?.card?.id ?? null;
    }
  }
  return pickRandom(player.hand)?.id ?? null;
}

export function botPlay(state, playerKey, options = {}) {
  const policy = normalizePolicy(options.policy);
  if (policy === POLICY_RANDOM) return botPlayRandom(state, playerKey);
  return botPlaySmart(state, playerKey, policy);
}

function botPlayRandom(state, playerKey) {
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
    const cardId = botChooseCard(state, playerKey, POLICY_RANDOM);
    if (!cardId) return state;
    return playTurn(state, cardId);
  }

  return state;
}

function botPlaySmart(state, playerKey, policy) {
  if (state.phase === "gukjin-choice" && state.pendingGukjinChoice?.playerKey === playerKey) {
    return chooseGukjinMode(state, playerKey, chooseGukjinHeuristic(state, playerKey, policy));
  }

  if (state.phase === "president-choice" && state.pendingPresident?.playerKey === playerKey) {
    const shouldStop = shouldPresidentStop(state, playerKey, policy);
    return shouldStop ? choosePresidentStop(state, playerKey) : choosePresidentHold(state, playerKey);
  }

  if (state.phase === "select-match" && state.pendingMatch?.playerKey === playerKey) {
    const choiceId = chooseMatchHeuristic(state);
    return choiceId ? chooseMatch(state, choiceId) : state;
  }

  if (state.phase === "go-stop" && state.pendingGoStop === playerKey) {
    return shouldGo(state, playerKey, policy) ? chooseGo(state, playerKey) : chooseStop(state, playerKey);
  }

  if (state.phase === "playing" && state.currentTurn === playerKey) {
    const bombMonths = getDeclarableBombMonths(state, playerKey);
    if (bombMonths.length > 0 && shouldBomb(state, playerKey, bombMonths, policy)) {
      return declareBomb(state, playerKey, selectBestMonth(state, bombMonths));
    }
    const shakingMonths = getDeclarableShakingMonths(state, playerKey);
    if (shakingMonths.length > 0 && shouldShaking(state, playerKey, shakingMonths, policy)) {
      return declareShaking(state, playerKey, selectBestMonth(state, shakingMonths));
    }
    const cardId = botChooseCard(state, playerKey, policy);
    if (!cardId) return state;
    return playTurn(state, cardId);
  }

  return state;
}

function normalizePolicy(policy) {
  const raw = String(policy || POLICY_RANDOM).trim().toLowerCase();
  if (raw === "heuristic_v1" || raw === "heuristic" || raw === "smart") return POLICY_HEURISTIC_V1;
  if (raw === "heuristic_v2" || raw === "smart_v2") return POLICY_HEURISTIC_V2;
  if (
    raw === "random_v2" ||
    raw === "random_plus" ||
    raw === "random-plus" ||
    raw === "plus" ||
    raw === "semi_random"
  ) {
    return POLICY_RANDOM_V2;
  }
  return POLICY_RANDOM;
}

function isHeuristicPolicy(policy) {
  return policy === POLICY_HEURISTIC_V1 || policy === POLICY_HEURISTIC_V2;
}

function cardCaptureValue(card) {
  if (!card) return 0;
  if (card.category === "kwang") return 6;
  if (card.category === "five") return 4;
  if (card.category === "ribbon") return 2;
  if (card.category === "junk") {
    if (card.tripleJunk) return 3;
    if (card.doubleJunk) return 2;
    return 1;
  }
  if (card.bonus?.stealPi) return 3 + Number(card.bonus.stealPi || 0);
  return 0;
}

function boardMatchesByMonth(state) {
  const map = new Map();
  for (const card of state.board || []) {
    const month = card?.month;
    if (month == null) continue;
    const list = map.get(month) || [];
    list.push(card);
    map.set(month, list);
  }
  return map;
}

function rankHandCards(state, playerKey) {
  const player = state.players?.[playerKey];
  if (!player?.hand?.length) return [];
  const byMonth = boardMatchesByMonth(state);
  const ranked = player.hand.map((card) => {
    const matches = byMonth.get(card.month) || [];
    const captureGain = matches.reduce((sum, c) => sum + cardCaptureValue(c), 0);
    const selfValue = cardCaptureValue(card);
    let score = 0;
    if (matches.length === 0) score = -2 - selfValue;
    else if (matches.length === 1) score = 6 + captureGain - selfValue * 0.2;
    else if (matches.length === 2) score = 9 + captureGain - selfValue * 0.1;
    else score = 12 + captureGain;
    return { card, score, matches: matches.length };
  });
  ranked.sort((a, b) => b.score - a.score);
  return ranked;
}

function chooseMatchHeuristic(state) {
  const ids = state.pendingMatch?.boardCardIds || [];
  if (!ids.length) return null;
  const candidates = (state.board || []).filter((c) => ids.includes(c.id));
  if (!candidates.length) return null;
  candidates.sort((a, b) => cardCaptureValue(b) - cardCaptureValue(a));
  return candidates[0].id;
}

function chooseGukjinHeuristic(state, playerKey, policy) {
  const fiveCount = state.players?.[playerKey]?.captured?.five?.length || 0;
  if (policy === POLICY_HEURISTIC_V2) return fiveCount >= 3 ? "five" : "junk";
  return fiveCount >= 4 ? "five" : "junk";
}

function shouldPresidentStop(state, playerKey, policy) {
  if (isHeuristicPolicy(policy)) return false;
  const deckCount = state.deck?.length || 0;
  return deckCount <= 8 && Math.random() < 0.6;
}

function shouldGo(state, playerKey, policy) {
  const player = state.players?.[playerKey];
  const goCount = player?.goCount || 0;
  const deckCount = state.deck?.length || 0;
  if (policy === POLICY_HEURISTIC_V2) {
    if (goCount >= 3) return false;
    return deckCount > 6;
  }
  if (policy === POLICY_HEURISTIC_V1) {
    if (goCount >= 2) return false;
    return deckCount > 8;
  }
  if (goCount >= 2) return false;
  if (deckCount <= 8) return Math.random() < 0.2;
  return Math.random() < 0.7;
}

function monthBoardGain(state, month) {
  const cards = (state.board || []).filter((c) => c.month === month);
  return cards.reduce((sum, c) => sum + cardCaptureValue(c), 0);
}

function selectBestMonth(state, months) {
  if (!months?.length) return null;
  let best = months[0];
  let bestScore = monthBoardGain(state, best);
  for (const m of months.slice(1)) {
    const score = monthBoardGain(state, m);
    if (score > bestScore) {
      best = m;
      bestScore = score;
    }
  }
  return best;
}

function shouldBomb(state, playerKey, bombMonths, policy) {
  const bestGain = monthBoardGain(state, selectBestMonth(state, bombMonths));
  if (policy === POLICY_HEURISTIC_V2) return bestGain >= 1;
  if (policy === POLICY_HEURISTIC_V1) return bestGain >= 2;
  return bestGain >= 3 || Math.random() < 0.25;
}

function shouldShaking(state, playerKey, shakingMonths, policy) {
  const deckCount = state.deck?.length || 0;
  if (policy === POLICY_HEURISTIC_V2) return deckCount > 8;
  if (policy === POLICY_HEURISTIC_V1) return deckCount > 10;
  return deckCount > 12 && Math.random() < 0.4;
}

function pickRandom(arr) {
  if (!arr || arr.length === 0) return null;
  return arr[Math.floor(Math.random() * arr.length)];
}
