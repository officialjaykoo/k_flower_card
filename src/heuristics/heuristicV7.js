import {
  computePrudentGoThresholdV7,
  evaluateGoldPotentialV7,
  isOpponentImminentV7
} from "./heuristicV7_math.js";

// heuristicV7.js - V7.2 (Predatory Counter)

const GUKJIN_CARD_ID = "I0";

export const DEFAULT_PARAMS = {
  greedMul: 1.8,
  shakingGreed: 4.0,
  bombGreed: 4.0,
  goAggression: 0.3,
  lowThreatForceGo: 0.2,
  trailingRiskAppetite: 0.25,
  leadingRiskAppetite: 0.5,
  trailingThreatBuffer: 0.1,
  lockProfitScore: 8,

  noMatchBase: -5.0,
  matchBase: 15.0,
  doublePiBonus: 22.0,
  kwangBonus: 25.0,
  comboBonus: 35.0,
  denialBonus: 15.0,
  comboBreakerBonus: 40.0,
  bonusCardSteal: 30.0,
  piPressureBonus: 12.0,
  piLeakPenalty: 10.0,
  antiPiBakBonus: 50.0,
  antiPiBakNoPiPenalty: 14.0,

  riskTolerance: 1.2,
  pukOpportunity: 3.0
};

function safeNum(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function otherKey(state, playerKey) {
  if (playerKey === "human") return "ai";
  if (playerKey === "ai") return "human";
  const keys = Object.keys(state?.players || {});
  if (keys.length === 2) return keys.find((k) => k !== playerKey) || null;
  return null;
}

function piValue(card, deps) {
  if (!card) return 0;
  if (card.id === GUKJIN_CARD_ID) return 2;
  if (card.category !== "junk") return 0;
  return safeNum(deps.junkPiValue?.(card), 1);
}

function isDoublePi(card, deps) {
  return piValue(card, deps) >= 2;
}

function monthUrgencyScore(oppProgress, month) {
  return safeNum(oppProgress?.monthUrgency?.get?.(month), 0);
}

function normalizeComboTargets(raw) {
  const ids = new Set();
  const months = new Set();
  let imminent = false;

  const ingest = (value) => {
    if (value == null) return;
    if (typeof value === "string") {
      ids.add(value);
      return;
    }
    if (Number.isInteger(value)) {
      months.add(value);
      return;
    }
    if (Array.isArray(value)) {
      for (const v of value) ingest(v);
      return;
    }
    if (typeof value === "object") {
      if (typeof value.id === "string") ids.add(value.id);
      if (Number.isInteger(value.month)) months.add(value.month);
      if (value.imminent) imminent = true;
      if (Array.isArray(value.ids)) for (const id of value.ids) ingest(id);
      if (Array.isArray(value.months)) for (const month of value.months) ingest(month);
      if (Array.isArray(value.cards)) for (const card of value.cards) ingest(card);
    }
  };

  ingest(raw);
  return { ids, months, imminent };
}

function resolveComboBreakerTargets(state, playerKey, deps, oppProgress) {
  const oppKey = otherKey(state, playerKey);
  const rawMissing = oppKey ? deps.getMissingComboCards?.(state, oppKey) : null;
  const normalized = normalizeComboTargets(rawMissing);
  const monthUrgency = oppProgress?.monthUrgency;
  if (monthUrgency && typeof monthUrgency.entries === "function") {
    for (const [month, urgency] of monthUrgency.entries()) {
      if (safeNum(urgency) >= 20) normalized.months.add(month);
      if (safeNum(urgency) >= 28) normalized.imminent = true;
    }
  }
  return normalized;
}

function opponentPiCount(state, playerKey, deps) {
  const oppKey = otherKey(state, playerKey);
  if (!oppKey) return 0;
  const viaDeps = safeNum(deps.capturedCountByCategory?.(state?.players?.[oppKey], "junk"), NaN);
  if (Number.isFinite(viaDeps)) return viaDeps;
  return safeNum(state?.players?.[oppKey]?.captured?.junk?.length, 0);
}

function selfPiCount(state, playerKey, deps) {
  const viaDeps = safeNum(deps.capturedCountByCategory?.(state?.players?.[playerKey], "junk"), NaN);
  if (Number.isFinite(viaDeps)) return viaDeps;
  return safeNum(state?.players?.[playerKey]?.captured?.junk?.length, 0);
}

export function rankHandCardsV7(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const player = state.players?.[playerKey];
  if (!player?.hand?.length) return [];

  const P = { ...DEFAULT_PARAMS, ...params };
  const boardByMonth =
    typeof deps.boardMatchesByMonth === "function" ? deps.boardMatchesByMonth(state) : new Map();
  const oppProgress = deps.checkOpponentJokboProgress?.(state, playerKey) || null;
  const comboBreaker = resolveComboBreakerTargets(state, playerKey, deps, oppProgress);
  const oppPi = opponentPiCount(state, playerKey, deps);
  const myPi = selfPiCount(state, playerKey, deps);
  const deckCount = safeNum(state?.deck?.length, 0);
  const antiPiBakMode = myPi < 7 && deckCount < 10;
  const piPressureWindow = oppPi >= 5 && oppPi <= 7;

  const ranked = player.hand.map((card) => {
    const matches = boardByMonth.get(card.month) || [];
    const allCaptured = [card, ...matches];
    const isBonusCard = safeNum(card?.bonus?.stealPi) > 0;

    let score = matches.length > 0 ? P.matchBase : P.noMatchBase;

    // Denial: prioritize months opponent is close to completing.
    const urgency = monthUrgencyScore(oppProgress, card.month);
    if (urgency >= 20) score += P.denialBonus;
    if (comboBreaker.ids.has(card.id) || comboBreaker.months.has(card.month)) {
      score += P.comboBreakerBonus;
    }
    if (comboBreaker.imminent && comboBreaker.months.has(card.month)) {
      score += P.comboBreakerBonus * 0.35;
    }

    const piGain = allCaptured.reduce((sum, c) => sum + piValue(c, deps), 0);
    score += piGain * 10 * P.greedMul;

    if (matches.length > 0 && allCaptured.some((c) => isDoublePi(c, deps))) {
      score += P.doublePiBonus;
    }

    if (isBonusCard) score += P.bonusCardSteal;
    if (piPressureWindow && (piGain > 0 || isBonusCard)) score += P.piPressureBonus;
    if (piPressureWindow && matches.length === 0 && piValue(card, deps) > 0) {
      score -= P.piLeakPenalty * Math.max(1, piValue(card, deps));
    }
    if (antiPiBakMode) {
      if (piGain > 0 || isBonusCard) {
        score += P.antiPiBakBonus + piGain * 5.0;
      } else {
        score -= P.antiPiBakNoPiPenalty;
      }
    }

    for (const c of allCaptured) {
      if (c.category === "kwang") score += P.kwangBonus;
      if (c.category === "five" || c.category === "ribbon") score += P.comboBonus;
    }

    score += evaluateGoldPotentialV7(state, playerKey, card) * 0.35;

    const danger = safeNum(
      deps.estimateDangerMonthRisk?.(state, playerKey, card.month, new Map(), new Map(), new Map())
    );
    const feedRisk = safeNum(deps.estimateOpponentImmediateGainIfDiscard?.(state, playerKey, card.month));
    score -= (danger + feedRisk * 0.8) * P.riskTolerance;

    return { card, score, matches: matches.length };
  });

  ranked.sort((a, b) => b.score - a.score);
  return ranked;
}

export function chooseMatchHeuristicV7(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const ids = state.pendingMatch?.boardCardIds || [];
  if (!ids.length) return null;

  const P = { ...DEFAULT_PARAMS, ...params };
  const oppProgress = deps.checkOpponentJokboProgress?.(state, playerKey) || null;
  const comboBreaker = resolveComboBreakerTargets(state, playerKey, deps, oppProgress);
  const oppPi = opponentPiCount(state, playerKey, deps);
  const myPi = selfPiCount(state, playerKey, deps);
  const deckCount = safeNum(state?.deck?.length, 0);
  const antiPiBakMode = myPi < 7 && deckCount < 10;
  const piPressureWindow = oppPi >= 5 && oppPi <= 7;
  let bestId = null;
  let bestScore = -Infinity;

  for (const card of (state.board || []).filter((c) => ids.includes(c.id))) {
    let score = safeNum(deps.cardCaptureValue(card)) * 2.0;
    const pi = piValue(card, deps);

    score += pi * 15 * P.greedMul;
    if (card.category === "kwang") score += P.kwangBonus;
    if (card.category === "five" || card.category === "ribbon") score += P.comboBonus;
    if (isDoublePi(card, deps)) score += P.doublePiBonus;

    const urgency = monthUrgencyScore(oppProgress, card.month);
    if (urgency >= 20) score += P.denialBonus;
    if (comboBreaker.ids.has(card.id) || comboBreaker.months.has(card.month)) {
      score += P.comboBreakerBonus;
    }
    if (piPressureWindow && (card.category === "junk" || safeNum(card?.bonus?.stealPi) > 0)) {
      score += P.piPressureBonus;
    }
    if (antiPiBakMode) {
      const pi = piValue(card, deps);
      if (pi > 0 || safeNum(card?.bonus?.stealPi) > 0) score += P.antiPiBakBonus + pi * 4.0;
      else score -= P.antiPiBakNoPiPenalty;
    }

    if (score > bestScore) {
      bestScore = score;
      bestId = card.id;
    }
  }

  return bestId;
}

export function shouldGoV7(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const P = { ...DEFAULT_PARAMS, ...params };
  const oppKey = otherKey(state, playerKey);
  if (!oppKey) return false;

  const willBankruptNow = !!deps.canBankruptOpponentByStop?.(state, playerKey);
  const goldAnalysis = deps.analyzeGoldPotential?.(state, playerKey) || {
    willBankrupt: willBankruptNow
  };
  if (goldAnalysis.willBankrupt) return false;

  const ctx = deps.analyzeGameContext?.(state, playerKey) || {};
  const myScore = safeNum(ctx.myScore, safeNum(state?.players?.[playerKey]?.score, 0));
  const oppCurrentScore = safeNum(ctx.oppScore, safeNum(state?.players?.[oppKey]?.score, 0));
  const oppThreat = safeNum(deps.opponentThreatScore?.(state, playerKey));
  const deckCount = safeNum(state?.deck?.length, 0);
  const myGold = safeNum(state?.players?.[playerKey]?.gold, 0);
  const oppGold = safeNum(state?.players?.[oppKey]?.gold, 0);
  const trailingGold = myGold < oppGold;

  const oppJokbo = deps.checkOpponentJokboProgress?.(state, playerKey) || null;
  const oppImminent = isOpponentImminentV7(state, playerKey, oppJokbo);

  const prudentThreshold = computePrudentGoThresholdV7({
    base: P.goAggression,
    myScore,
    deckCount
  });
  const riskAppetite = trailingGold ? P.trailingRiskAppetite : P.leadingRiskAppetite;
  // Lower appetite value means "take more volatility" in this model.
  const dynamicThreshold = Math.max(0.2, Math.min(0.65, 0.75 - safeNum(riskAppetite, 0.4)));
  let threatStopThreshold = trailingGold
    ? Math.max(prudentThreshold, dynamicThreshold) + P.trailingThreatBuffer
    : Math.min(prudentThreshold, dynamicThreshold);
  threatStopThreshold = Math.max(0.2, Math.min(0.72, threatStopThreshold));

  if (oppThreat < P.lowThreatForceGo) return true;

  if (!trailingGold && myScore >= P.lockProfitScore && (oppThreat >= 0.16 || oppCurrentScore > 0 || oppImminent)) {
    return false;
  }
  if (oppImminent && myScore >= 4) return false;
  if (deckCount <= 4 && myScore >= 5 && oppThreat >= 0.2) return false;

  if (oppThreat > threatStopThreshold && myScore >= 4) return false;
  if (!trailingGold && oppCurrentScore > 0 && myScore >= 6) return false;

  return true;
}

export function selectBombMonthV7(state, _playerKey, bombMonths, deps) {
  if (!bombMonths?.length) return null;
  return bombMonths.reduce(
    (best, m) => (safeNum(deps.monthBoardGain(state, m)) > safeNum(deps.monthBoardGain(state, best)) ? m : best),
    bombMonths[0]
  );
}

export function shouldBombV7(_state, _playerKey, bombMonths, _deps, _params = DEFAULT_PARAMS) {
  return bombMonths?.length > 0;
}

export function decideShakingV7(state, playerKey, shakingMonths, deps, _params = DEFAULT_PARAMS) {
  if (!shakingMonths?.length) return { allow: false, month: null, score: -Infinity };

  let bestMonth = shakingMonths[0];
  let bestScore = safeNum(deps.shakingImmediateGainScore(state, playerKey, bestMonth));

  for (const month of shakingMonths.slice(1)) {
    const score = safeNum(deps.shakingImmediateGainScore(state, playerKey, month));
    if (score > bestScore) {
      bestScore = score;
      bestMonth = month;
    }
  }

  return { allow: true, month: bestMonth, score: bestScore, highImpact: true };
}

export function shouldPresidentStopV7(state, playerKey, deps, _params = DEFAULT_PARAMS) {
  const ctx = deps.analyzeGameContext(state, playerKey);
  const diff = safeNum(ctx.myScore) - safeNum(ctx.oppScore);
  return diff >= 4;
}

export function chooseGukjinHeuristicV7(state, playerKey, deps, _params = DEFAULT_PARAMS) {
  const oppKey = otherKey(state, playerKey);
  if (!oppKey) return "junk";

  const oppPiCount = safeNum(
    deps.capturedCountByCategory?.(state?.players?.[oppKey], "junk"),
    safeNum(state?.players?.[oppKey]?.captured?.junk?.length, 0)
  );
  // If opponent is in Pi-Bak window, keep gukjin as junk to pressure sub-10 pi finish.
  if (oppPiCount >= 5 && oppPiCount <= 7) return "junk";

  const myFiveCount = safeNum(
    deps.capturedCountByCategory?.(state?.players?.[playerKey], "five"),
    safeNum(state?.players?.[playerKey]?.captured?.five?.length, 0)
  );
  // Convert to five only when own five stack is already meaningful.
  if (myFiveCount >= 4) return "five";

  return "junk";
}
