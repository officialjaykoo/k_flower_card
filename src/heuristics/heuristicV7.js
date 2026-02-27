// heuristicV7.js - V7 "The Negotiator"

/* ============================================================================
 * Heuristic V7 (Negotiator)
 * - Adversarial card ranking with opponent counter-value penalty
 * - Phase-adaptive scoring (early resource gain / late denial and safety)
 * - Conservative GO gate when opponent threat is meaningful
 * ========================================================================== */

const GUKJIN_CARD_ID = "I0";

export const DEFAULT_PARAMS = {
  // Phase boundaries
  earlyPhaseLimit: 15,
  latePhaseLimit: 7,

  // Core utility
  baseUtility: 10.0,
  opponentOpportunityCost: 1.5,
  synergyWeight: 12.0,
  matchDenyBonus: 6.0,
  noMatchPenalty: 100,

  // GO/STOP
  stopLeadThreshold: 5,
  threatMultiplier: 2.0,

  // Bomb/Shaking/President fallbacks
  bombMinMonthGain: 0.5,
  shakingAllowThreshold: 1.4,
  presidentStopDiff: 2
};

function safeNum(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function getOtherPlayerKey(state, playerKey, deps) {
  if (typeof deps?.otherPlayerKey === "function") return deps.otherPlayerKey(playerKey);
  if (playerKey === "human") return "ai";
  if (playerKey === "ai") return "human";
  const keys = Object.keys(state?.players || {});
  if (keys.length === 2) return keys.find((k) => k !== playerKey) || null;
  return null;
}

function getDeckCount(state) {
  return safeNum(state?.deck?.length, safeNum(state?.deckCount, 0));
}

function getBoardMatchesByMonth(state, deps) {
  if (typeof deps?.boardMatchesByMonth === "function") return deps.boardMatchesByMonth(state);
  const out = new Map();
  for (const card of state?.board || []) {
    if (!out.has(card.month)) out.set(card.month, []);
    out.get(card.month).push(card);
  }
  return out;
}

function getPiCount(state, playerKey, category, deps) {
  const viaDeps = safeNum(deps?.capturedCountByCategory?.(state?.players?.[playerKey], category), NaN);
  if (Number.isFinite(viaDeps)) return viaDeps;
  return safeNum(state?.players?.[playerKey]?.captured?.[category]?.length, 0);
}

function getContext(state, playerKey, deps) {
  const raw = deps?.analyzeGameContext?.(state, playerKey) || {};
  const oppKey = getOtherPlayerKey(state, playerKey, deps);
  return {
    myScore: safeNum(raw?.myScore, safeNum(state?.players?.[playerKey]?.score, 0)),
    oppScore: safeNum(raw?.oppScore, safeNum(state?.players?.[oppKey]?.score, 0)),
    selfPi: safeNum(raw?.selfPi, getPiCount(state, playerKey, "junk", deps)),
    oppPi: safeNum(raw?.oppPi, getPiCount(state, oppKey, "junk", deps))
  };
}

function captureUtility(card, params, deps) {
  if (!card) return 0;
  let value = safeNum(params.baseUtility, 10);
  const id = String(card.id || "");
  if (card.category === "kwang" || id.includes("K") || id === GUKJIN_CARD_ID) value += 15;
  if (card.category === "ribbon" || card.category === "five") value += 8;
  if (safeNum(card?.bonus?.stealPi) > 0) value += 10;
  if (card.category === "junk") value += safeNum(deps?.junkPiValue?.(card), safeNum(card?.piValue, 1)) * 2;
  return value;
}

function comboPotentialScore(state, playerKey, month, deps, params) {
  const ownCombo = safeNum(deps?.ownComboOpportunityScore?.(state, playerKey, month), NaN);
  if (Number.isFinite(ownCombo)) return ownCombo * safeNum(params.synergyWeight, 12);
  const legacyCombo = safeNum(deps?.analyzeComboPotential?.(state, playerKey, month), 0);
  return legacyCombo * safeNum(params.synergyWeight, 12);
}

function estimateOpponentCounterValue(state, playerKey, month, deps) {
  const sameMonthOnBoard = (state?.board || []).filter((c) => c?.month === month).length;
  const immediateGain = safeNum(deps?.estimateOpponentImmediateGainIfDiscard?.(state, playerKey, month), 0);
  return sameMonthOnBoard * 10 + immediateGain * 4;
}

function deniedMonthSet(state, playerKey, deps) {
  const oppKey = getOtherPlayerKey(state, playerKey, deps);
  if (!oppKey || typeof deps?.blockingMonthsAgainst !== "function") return new Set();
  return deps.blockingMonthsAgainst(state?.players?.[oppKey], state?.players?.[playerKey]) || new Set();
}

function inferCurrentScoreV7(state, playerKey, options = {}) {
  const fromOption = safeNum(options.currentScore, NaN);
  if (Number.isFinite(fromOption)) return Math.max(0, fromOption);

  const fromPlayerScore = safeNum(state?.players?.[playerKey]?.score, NaN);
  if (Number.isFinite(fromPlayerScore)) return Math.max(0, fromPlayerScore);

  const fromPlayerTotal = safeNum(state?.players?.[playerKey]?.totalScore, NaN);
  if (Number.isFinite(fromPlayerTotal)) return Math.max(0, fromPlayerTotal);

  const fromRootScore = safeNum(state?.score?.[playerKey], NaN);
  if (Number.isFinite(fromRootScore)) return Math.max(0, fromRootScore);

  return NaN;
}

function inferMultiplierV7(state, playerKey) {
  const player = state?.players?.[playerKey] || {};
  let multiplier = Math.max(1, safeNum(state?.multiplier, 1));

  const carry = Math.max(1, safeNum(state?.carryOverMultiplier, 1));
  multiplier *= carry;

  const goCount = Math.max(0, Math.floor(safeNum(player?.goCount, 0)));
  if (goCount === 3) multiplier *= 2;
  if (goCount === 4) multiplier *= 4;
  if (goCount >= 5) multiplier *= 8;

  const shakingCount = Math.max(0, Math.floor(safeNum(player?.shakingCount, 0)));
  if (shakingCount > 0) multiplier *= 2;

  if (player?.isBomb === true || player?.bombDeclared === true) multiplier *= 2;
  return Math.max(1, multiplier);
}

export function canBankruptOpponentByStopV7(state, playerKey, options = {}) {
  if (!state?.players || !state.players[playerKey]) return false;
  const oppKey = getOtherPlayerKey(state, playerKey, null);
  if (!oppKey) return false;

  const oppGold = safeNum(state?.players?.[oppKey]?.gold, 0);
  if (oppGold <= 0) return true;

  if (state?.phase === "go-stop" && typeof options.fallbackExact === "function") {
    const exact = !!options.fallbackExact(state, playerKey);
    if (exact) return true;
  }

  const score = inferCurrentScoreV7(state, playerKey, options);
  if (!Number.isFinite(score) || score <= 0) {
    if (typeof options.fallbackExact === "function") return !!options.fallbackExact(state, playerKey);
    return false;
  }

  const betPerScore = Math.max(1, safeNum(state?.betAmount, safeNum(options.defaultBetAmount, 100)));
  const multiplier = inferMultiplierV7(state, playerKey);
  const safetyMargin = Math.min(1, Math.max(0.5, safeNum(options.safetyMargin, 0.9)));
  const expectedDamage = score * betPerScore * multiplier;

  return expectedDamage >= oppGold * safetyMargin;
}

export function rankHandCardsV7(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const player = state?.players?.[playerKey];
  if (!player?.hand?.length) return [];

  const P = { ...DEFAULT_PARAMS, ...params };
  const deckCount = getDeckCount(state);
  const isEarly = deckCount > P.earlyPhaseLimit;
  const boardByMonth = getBoardMatchesByMonth(state, deps);
  const deniedMonths = deniedMonthSet(state, playerKey, deps);

  const ranked = player.hand.map((card) => {
    const matches = boardByMonth.get(card.month) || [];
    if (matches.length === 0) return { card, score: -Math.abs(P.noMatchPenalty), matches: 0 };

    const myGain = matches.reduce((acc, m) => acc + captureUtility(m, P, deps), 0);
    const synergy = comboPotentialScore(state, playerKey, card.month, deps, P);
    const oppCounter = estimateOpponentCounterValue(state, playerKey, card.month, deps);

    let score = isEarly
      ? (myGain + synergy) - oppCounter * 0.5
      : (myGain * 0.8) - oppCounter * P.opponentOpportunityCost;

    if (deniedMonths.has(card.month)) score += P.matchDenyBonus;
    return { card, score, matches: matches.length };
  });

  ranked.sort((a, b) => b.score - a.score);
  return ranked;
}

export function chooseMatchHeuristicV7(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const ids = state?.pendingMatch?.boardCardIds || [];
  if (!ids.length) return null;

  const P = { ...DEFAULT_PARAMS, ...params };
  const deckCount = getDeckCount(state);
  const isEarly = deckCount > P.earlyPhaseLimit;
  const deniedMonths = deniedMonthSet(state, playerKey, deps);
  const candidates = (state?.board || []).filter((c) => ids.includes(c?.id));
  if (!candidates.length) return null;

  let bestId = null;
  let bestScore = -Infinity;

  for (const card of candidates) {
    const myGain = captureUtility(card, P, deps);
    const oppCounter = estimateOpponentCounterValue(state, playerKey, card.month, deps);

    let score = isEarly
      ? myGain - oppCounter * 0.4
      : (myGain * 0.85) - oppCounter * P.opponentOpportunityCost;

    if (deniedMonths.has(card.month)) score += P.matchDenyBonus;
    if (card.id === GUKJIN_CARD_ID) score += 8;

    if (score > bestScore) {
      bestScore = score;
      bestId = card.id;
    }
  }

  return bestId;
}

export function shouldGoV7(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const P = { ...DEFAULT_PARAMS, ...params };
  const ctx = getContext(state, playerKey, deps);
  const deckCount = getDeckCount(state);
  const scoreDiff = ctx.myScore - ctx.oppScore;
  const oppThreat = safeNum(
    deps?.opponentThreatScore?.(state, playerKey),
    safeNum(deps?.analyzeOpponentThreat?.(state, playerKey), 0)
  );

  if (deps?.canBankruptOpponentByStop?.(state, playerKey)) return false;

  if (ctx.myScore >= 7 && oppThreat * P.threatMultiplier > scoreDiff) return false;

  if (ctx.oppPi < 5 && ctx.selfPi >= 10 && deckCount > 5) return true;

  // Late game lock-in: avoid greed GO when a win is already secured.
  if (deckCount <= P.latePhaseLimit && ctx.myScore >= 7 && ctx.oppScore < 7) return false;

  return ctx.myScore > ctx.oppScore + P.stopLeadThreshold;
}

export function selectBombMonthV7(state, _playerKey, bombMonths, deps) {
  if (!bombMonths?.length) return null;
  return bombMonths.reduce(
    (best, month) =>
      safeNum(deps?.monthBoardGain?.(state, month)) > safeNum(deps?.monthBoardGain?.(state, best))
        ? month
        : best,
    bombMonths[0]
  );
}

export function shouldBombV7(state, playerKey, bombMonths, deps, params = DEFAULT_PARAMS) {
  if (!bombMonths?.length) return false;
  const P = { ...DEFAULT_PARAMS, ...params };
  const bestMonth = selectBombMonthV7(state, playerKey, bombMonths, deps);
  if (bestMonth == null) return false;

  const impact = deps?.isHighImpactBomb?.(state, playerKey, bestMonth);
  if (impact?.highImpact) return true;

  const gain = safeNum(deps?.monthBoardGain?.(state, bestMonth), 0);
  return gain >= safeNum(P.bombMinMonthGain, 0.5);
}

export function decideShakingV7(state, playerKey, shakingMonths, deps, params = DEFAULT_PARAMS) {
  if (!shakingMonths?.length) return { allow: false, month: null, score: -Infinity };
  const P = { ...DEFAULT_PARAMS, ...params };

  let bestMonth = null;
  let bestScore = -Infinity;
  let bestHighImpact = false;

  for (const month of shakingMonths) {
    const immediate = safeNum(deps?.shakingImmediateGainScore?.(state, playerKey, month), 0);
    const combo = safeNum(deps?.ownComboOpportunityScore?.(state, playerKey, month), 0);
    const impact = deps?.isHighImpactShaking?.(state, playerKey, month);
    const score = immediate + combo * 0.35 + (impact?.highImpact ? 0.4 : 0);

    if (score > bestScore) {
      bestScore = score;
      bestMonth = month;
      bestHighImpact = !!impact?.highImpact;
    }
  }

  const allow = bestHighImpact || bestScore >= P.shakingAllowThreshold;
  return { allow, month: bestMonth, score: bestScore, highImpact: bestHighImpact };
}

export function shouldPresidentStopV7(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const P = { ...DEFAULT_PARAMS, ...params };
  const ctx = getContext(state, playerKey, deps);
  const diff = ctx.myScore - ctx.oppScore;
  const carry = safeNum(state?.carryOverMultiplier, 1);
  if (carry >= 2) return false;
  return diff >= safeNum(P.presidentStopDiff, 2);
}

export function chooseGukjinHeuristicV7(state, playerKey, deps, _params = DEFAULT_PARAMS) {
  const oppKey = getOtherPlayerKey(state, playerKey, deps);
  if (!oppKey) return "junk";

  const oppPiCount = getPiCount(state, oppKey, "junk", deps);
  if (oppPiCount >= 5 && oppPiCount <= 8) return "junk";

  const myFiveCount = getPiCount(state, playerKey, "five", deps);
  if (myFiveCount >= 4) return "five";

  return "junk";
}

export {
  rankHandCardsV7 as rankHandCards,
  shouldGoV7 as shouldGo,
  chooseGukjinHeuristicV7 as chooseGukjin,
  chooseMatchHeuristicV7 as chooseMatch
};
