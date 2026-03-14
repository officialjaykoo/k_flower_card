import {
  clamp01,
  comboProgress,
  countKnownMonthCards,
  currentScoreTotal,
  cardCaptureValue,
  capturedCountByCategory,
  hasCardId,
  junkPiValue,
  otherPlayerKey
} from "./heuristicUtils.js";
import {
  blockingMonthsAgainst,
  estimateJokboExpectedPotential,
  estimateOpponentImmediateGainIfDiscard,
  matchableMonthCountForPlayer,
  nextTurnThreatScore,
  opponentThreatScore,
  estimateMonthCaptureChance
} from "./heuristicAnalysis.js";

export function cloneGameState(state) {
  return state ? structuredClone(state) : state;
}

export function makeHiddenCard(prefix, index) {
  return {
    id: `${prefix}_${index}`,
    month: 0,
    category: "hidden"
  };
}

export function hiddenCardCopies(prefix, count) {
  return Array.from({ length: Math.max(0, Number(count || 0)) }, (_, idx) => makeHiddenCard(prefix, idx));
}

export function collectPublicShakingRevealIds(state, targetPlayerKey) {
  const ids = new Set();
  const liveReveal = state?.shakingReveal;
  if (liveReveal?.playerKey === targetPlayerKey) {
    for (const card of liveReveal.cards || []) {
      const id = String(card?.id || "");
      if (id) ids.add(id);
    }
  }
  for (const entry of state?.kibo || []) {
    if (entry?.type !== "shaking_declare" || entry?.playerKey !== targetPlayerKey) continue;
    for (const card of entry?.revealCards || []) {
      const id = String(card?.id || "");
      if (id) ids.add(id);
    }
  }
  return ids;
}

export function getPublicKnownOpponentHandCards(state, observerKey) {
  const opp = otherPlayerKey(observerKey);
  const revealIds = collectPublicShakingRevealIds(state, opp);
  const oppHand = state?.players?.[opp]?.hand || [];
  return oppHand.filter((card) => revealIds.has(String(card?.id || "")));
}

export function maskPendingMatchContextForObserver(state, observerKey, publicState) {
  if (!publicState?.pendingMatch?.context) return;
  const actorKey = publicState?.pendingMatch?.playerKey;
  if (!actorKey) return;
  if (actorKey === observerKey) return;
  publicState.pendingMatch.context = {
    ...publicState.pendingMatch.context,
    deck: []
  };
}

export function createPublicState(state, observerKey) {
  const pub = cloneGameState(state);
  const opp = otherPlayerKey(observerKey);
  const oppHandLen = state?.players?.[opp]?.hand?.length || 0;
  const deckLen = state?.deck?.length || 0;
  const knownOppCards = getPublicKnownOpponentHandCards(state, observerKey).map((c) => ({ ...c }));
  const hiddenOppCount = Math.max(0, oppHandLen - knownOppCards.length);

  if (pub?.players?.[opp]) {
    pub.players[opp].hand = knownOppCards.concat(hiddenCardCopies(`opp_${opp}`, hiddenOppCount));
  }
  pub.deck = hiddenCardCopies("deck", deckLen);
  maskPendingMatchContextForObserver(state, observerKey, pub);
  return pub;
}

export function knownMonthCountForObserver(state, observerKey, month) {
  let count = 0;
  for (const c of state.board || []) if (c?.month === month) count += 1;
  const selfPlayer = state.players?.[observerKey];
  for (const c of selfPlayer?.hand || []) if (c?.month === month) count += 1;
  for (const c of getPublicKnownOpponentHandCards(state, observerKey)) if (c?.month === month) count += 1;
  for (const key of ["human", "ai"]) {
    const cap = state.players?.[key]?.captured || {};
    for (const cat of ["kwang", "five", "ribbon", "junk"]) {
      for (const c of cap[cat] || []) if (c?.month === month) count += 1;
    }
  }
  return count;
}

export function visibleHandForObserver(state, playerKey, observerKey) {
  if (playerKey === observerKey) return state?.players?.[playerKey]?.hand || [];
  return getPublicKnownOpponentHandCards(state, observerKey);
}

export function estimateMonthCaptureChanceForObserver(state, actorKey, month, requiredCategory, observerKey) {
  const hand = visibleHandForObserver(state, actorKey, observerKey);
  const board = state.board || [];
  const deckCount = state.deck?.length || 0;
  const drawFactor = deckCount <= 5 ? 0.68 : deckCount <= 8 ? 0.82 : 1.0;
  const handHasAny = hand.some((c) => c?.month === month);
  const handHasRequired = hand.some((c) => c?.month === month && c?.category === requiredCategory);
  const boardHasAny = board.some((c) => c?.month === month);
  const boardRequiredCount = board.filter((c) => c?.month === month && c?.category === requiredCategory).length;
  let chance = 0.12 * drawFactor;

  if (handHasRequired) chance = Math.max(chance, boardHasAny ? 0.64 : 0.5);
  if (boardRequiredCount > 0) chance = Math.max(chance, handHasAny ? 0.78 : 0.42);
  if (handHasAny) chance = Math.max(chance, 0.28);
  if (boardRequiredCount >= 2 && handHasAny) chance = Math.max(chance, 0.86);
  return Math.max(0, Math.min(0.92, chance));
}

export function estimateJokboExpectedPotentialForObserver(state, actorKey, blockerKey, observerKey) {
  void observerKey;
  const base = estimateJokboExpectedPotential(state, actorKey, blockerKey);
  return {
    total: Number(base?.total || 0),
    nearCompleteCount: Number(base?.nearCompleteCount || 0),
    oneAwayCount: Number(base?.oneAwayCount || 0)
  };
}

export function shakingImmediateGainScoreForObserver(state, playerKey, month, observerKey) {
  const monthCards = visibleHandForObserver(state, playerKey, observerKey).filter((c) => c?.month === month);
  const deckCount = state?.deck?.length || 0;
  const known = knownMonthCountForObserver(state, observerKey, month);
  const unseen = Math.max(0, 4 - known);
  const hasHighCard = monthCards.some((c) => c.category === "kwang" || c.category === "five");
  const piPayload = monthCards.reduce((sum, c) => sum + junkPiValue(c), 0);
  const flipMatchChance = deckCount > 0 ? Math.min(1, unseen / deckCount) : 0;
  let score = flipMatchChance * (2.1 + piPayload * 0.35 + (hasHighCard ? 0.8 : 0));
  if (monthCards.some((c) => c.category === "junk" && junkPiValue(c) >= 2)) score += 0.25;
  if (unseen === 0) score -= 0.7;
  return score;
}

export function hiddenPoolStatsForObserver(state, observerKey) {
  const opp = otherPlayerKey(observerKey);
  const oppHandSize = state?.players?.[opp]?.hand?.length || 0;
  const knownOppHandSize = getPublicKnownOpponentHandCards(state, observerKey).length;
  const hiddenOppHandSize = Math.max(0, oppHandSize - knownOppHandSize);
  const deckCount = state?.deck?.length || 0;
  return {
    oppHandSize,
    knownOppHandSize,
    hiddenOppHandSize,
    deckCount,
    hiddenPoolSize: hiddenOppHandSize + deckCount
  };
}

export function opponentMonthHoldProbPublic(state, observerKey, month) {
  const knownOppMonthCount = getPublicKnownOpponentHandCards(state, observerKey).filter((c) => c?.month === month).length;
  if (knownOppMonthCount > 0) return 1;
  const known = knownMonthCountForObserver(state, observerKey, month);
  const unseenMonthCards = Math.max(0, 4 - known);
  const { hiddenOppHandSize, hiddenPoolSize } = hiddenPoolStatsForObserver(state, observerKey);
  if (hiddenOppHandSize <= 0 || hiddenPoolSize <= 0 || unseenMonthCards <= 0) return 0;
  const perCardMiss = Math.max(0, 1 - unseenMonthCards / hiddenPoolSize);
  const missAll = Math.pow(perCardMiss, hiddenOppHandSize);
  return clamp01(1 - missAll);
}

export function matchableMonthCountForPlayerPublic(state, playerKey, observerKey) {
  if (playerKey === observerKey) return matchableMonthCountForPlayer(state, playerKey);
  const boardMonths = [...new Set((state.board || []).map((c) => c?.month).filter((m) => Number.isInteger(m)))];
  let expected = 0;
  for (const month of boardMonths) expected += opponentMonthHoldProbPublic(state, observerKey, month);
  return expected;
}

export function nextTurnThreatScorePublic(state, defenderKey, observerKey) {
  const attacker = otherPlayerKey(defenderKey);
  if (attacker === observerKey) return nextTurnThreatScore(state, defenderKey);

  const attackerCombos = comboProgress(state.players?.[attacker]);
  const monthToCards = new Map();
  for (const c of state.board || []) {
    if (!Number.isInteger(c?.month)) continue;
    const arr = monthToCards.get(c.month) || [];
    arr.push(c);
    monthToCards.set(c.month, arr);
  }

  let score = 0;
  for (const [month, cards] of monthToCards.entries()) {
    const matchProb = opponentMonthHoldProbPublic(state, observerKey, month);
    if (matchProb <= 0) continue;
    let local = 0;
    if (cards.some((c) => c?.category === "kwang")) local += 0.3;
    if (cards.some((c) => c?.category === "five")) local += 0.22;
    if (cards.some((c) => c?.category === "junk")) local += 0.1;
    if (attackerCombos.fiveBirds >= 2) local += 0.24;
    if (attackerCombos.redRibbons >= 2) local += 0.18;
    if (attackerCombos.blueRibbons >= 2) local += 0.18;
    if (attackerCombos.plainRibbons >= 2) local += 0.18;
    score += local * matchProb;
  }
  return Math.max(0, Math.min(1, score));
}

export function opponentThreatScorePublic(state, playerKey, observerKey) {
  const opp = otherPlayerKey(playerKey);
  const oppPlayer = state.players?.[opp];
  const oppScore = currentScoreTotal(state, opp);
  const oppPi = capturedCountByCategory(oppPlayer, "junk");
  const oppGwang = capturedCountByCategory(oppPlayer, "kwang");
  const p = comboProgress(oppPlayer);
  const nextThreat = nextTurnThreatScorePublic(state, playerKey, observerKey);
  let score = 0;
  score += Math.min(1.0, oppScore / 7.0) * 0.55;
  score += Math.min(1.0, oppPi / 10.0) * 0.15;
  score += Math.min(1.0, oppGwang / 3.0) * 0.1;
  score += p.redRibbons >= 2 || p.blueRibbons >= 2 || p.plainRibbons >= 2 ? 0.12 : 0;
  score += p.fiveBirds >= 2 ? 0.12 : 0;
  score += nextThreat * 0.28;
  return Math.max(0, Math.min(1, score));
}

export function boardHighValueThreatForPlayerPublic(state, playerKey, observerKey) {
  if (playerKey === observerKey) return false;
  const blockerKey = otherPlayerKey(playerKey);
  const blockMonths = blockingMonthsAgainst(state.players?.[playerKey], state.players?.[blockerKey]);
  for (const c of state.board || []) {
    const highValue = c?.category === "kwang" || c?.category === "five" || blockMonths.has(c?.month);
    if (!highValue || !Number.isInteger(c?.month)) continue;
    const matchProb = opponentMonthHoldProbPublic(state, observerKey, c.month);
    if (matchProb >= 0.33) return true;
  }
  return false;
}

export function estimateOpponentImmediateGainIfDiscardPublic(state, playerKey, month, observerKey) {
  const opp = otherPlayerKey(playerKey);
  if (opp === observerKey) return estimateOpponentImmediateGainIfDiscard(state, playerKey, month);
  const boardMonthCards = (state.board || []).filter((c) => c?.month === month);
  const target = boardMonthCards.length ? Math.max(...boardMonthCards.map((b) => cardCaptureValue(b))) : 0;
  const matchProb = opponentMonthHoldProbPublic(state, observerKey, month);
  let expected = matchProb * (0.45 + target * 0.38 + 0.25);
  if (boardMonthCards.some((c) => c?.category === "kwang" || c?.category === "five")) expected += matchProb * 0.12;
  return Math.max(0, Math.min(3.6, expected));
}

export function createObserverSafeHeuristicDeps(baseDeps, observerKey, extraOverrides = null) {
  return Object.freeze({
    ...baseDeps,
    countKnownMonthCards: (state, month) => knownMonthCountForObserver(state, observerKey, month),
    estimateJokboExpectedPotential: (state, actorKey, blockerKey) =>
      estimateJokboExpectedPotentialForObserver(state, actorKey, blockerKey, observerKey),
    estimateOpponentImmediateGainIfDiscard: (state, playerKey, month) =>
      estimateOpponentImmediateGainIfDiscardPublic(state, playerKey, month, observerKey),
    estimateOpponentJokboExpectedPotential: (state, playerKey) =>
      estimateJokboExpectedPotentialForObserver(state, otherPlayerKey(playerKey), playerKey, observerKey),
    matchableMonthCountForPlayer: (state, playerKey) =>
      matchableMonthCountForPlayerPublic(state, playerKey, observerKey),
    nextTurnThreatScore: (state, playerKey) => nextTurnThreatScorePublic(state, playerKey, observerKey),
    opponentThreatScore: (state, playerKey) => opponentThreatScorePublic(state, playerKey, observerKey),
    shakingImmediateGainScore: (state, playerKey, month) =>
      shakingImmediateGainScoreForObserver(state, playerKey, month, observerKey),
    ...(extraOverrides || {})
  });
}

export function createHeuristicGPTFairDeps(baseDeps, observerKey) {
  return createObserverSafeHeuristicDeps(baseDeps, observerKey, {
    estimateOpponentJokboExpectedPotential: (state, playerKey) =>
      estimateJokboExpectedPotentialForObserver(state, otherPlayerKey(playerKey), playerKey, observerKey),
    estimateOpponentImmediateGainIfDiscard: (state, playerKey, month) =>
      estimateOpponentImmediateGainIfDiscardPublic(state, playerKey, month, observerKey),
    matchableMonthCountForPlayer: (state, playerKey) =>
      matchableMonthCountForPlayerPublic(state, playerKey, observerKey),
    nextTurnThreatScore: (state, playerKey) => nextTurnThreatScorePublic(state, playerKey, observerKey),
    opponentThreatScore: (state, playerKey) => opponentThreatScorePublic(state, playerKey, observerKey)
  });
}
