// heuristicAntiCL.js - Anti-CL focused heuristic policy
/* ============================================================================
 * Intent:
 * - absorb 1~5 strategy notes into a single, maintainable rule engine
 * - keep fail-fast behavior and avoid hidden-information cheating
 * - provide the same exported decision surface as other heuristics
 * ========================================================================== */

const GUKJIN_CARD_ID = "I0";
const HIGH_VALUE_MONTHS = Object.freeze([1, 3, 8, 11, 12]);
const SELF_TRACK_CARD_IDS = Object.freeze({
  godori: new Set(["B2", "D2", "H2"]),
  cheongDan: new Set(["F2", "I2", "J2"]),
  hongDan: new Set(["A2", "B2", "C2"]),
  choDan: new Set(["D2", "E2", "G2"])
});

export const DEFAULT_PARAMS = {
  // phase/context
  phaseEarlyDeckCut: 35,
  phaseEndDeckCut: 10,
  phaseDrawDeckCut: 4,
  phaseLeadLockScoreDiff: 5,
  phaseReversalScoreDiff: -7,
  bonusDefaultCount: 2,

  // ranking base
  noMatchBase: -5.2,
  matchOneBase: 4.0,
  matchTwoBase: 8.6,
  matchThreeBase: 11.5,
  captureGainMul: 1.0,
  piMul: 4.6,
  categoryKwangBonus: 2.4,
  categoryFiveBonus: 1.6,
  categoryRibbonBonus: 1.1,
  highValueMonthBonus: 0.7,
  comboOpportunityMul: 3.1,
  blockUrgencyMul: 6.5,
  bonusCardPriorityBonus: 2.2,
  jokboNearFinishBonus: 4.5,
  jokboBuildBonus: 1.0,
  oppJokboBlockBonus: 2.1,
  noMatchFeedRiskMul: 1.2,
  matchFeedRiskMul: 0.5,
  blockHardUrgencyCut: 1.0,
  blockHardUrgencyMul: 10.0,
  blockCriticalBonus: 60.0,

  // anti-CL tactical weights
  holdingPenalty: 4.8,
  dangerMonthPenalty: 3.1,
  trapBonus: 1.9,
  secondOrderTrapPenalty: 1.3,
  puckDangerWeight: -45.0,
  oppJokboNearPenalty: -35.0,
  reverseTrapPenalty: 2.0,
  preemptiveDefenseMul: 1.8,
  preemptiveDangerCut: 0.62,
  preemptiveDangerMonthProbCut: 0.3,
  preemptiveDangerMonthPenalty: 1.8,
  lookAheadTrapBonus: 1.6,

  // GO/STOP
  goMinPi: 6,
  goMinPiDesperate: 5,
  goEmergencyDangerCut: 1.0,
  goEmergencyOppPiCut: 9,
  goEmergencyMyPiCut: 5,
  goEmergencyLeadMin: 0,
  goHardThreatCut: 0.92,
  goHardDangerCut: 0.72,
  goHardStopOppThreatCut: 0.85,
  goHardStopVeryLateDeckCut: 4,
  goHardStopVeryLateOppThreatCut: 0.6,
  goHardStopLowPiCut: 3,
  goHardStopLowPiLeadMax: 0,
  goHardStopLowPiThreatCut: 0.6,
  goHardStopLowComboCut: 0.4,
  goHardStopLowComboThreatCut: 0.7,
  goHardNoPotentialStop: 1,
  goHardNoPotentialDeckCut: 6,
  goHardLateLeadLockEnable: 1,
  goHardLateLeadLockDeckCut: 4,
  goHardLateLeadLockLeadCut: 7,
  // Dynamic GO decision (expected value + adaptive threshold)
  goDecisionScoreWeight: 0.1,
  goDecisionComboWeight: 0.26,
  goDecisionPiWeight: 0.05,
  goDecisionThreatWeight: 0.45,
  goDecisionDeckWeight: 0.2,
  goDecisionDeckRiskBase: 0.2,
  goDecisionDeckRiskMid: 0.5,
  goDecisionDeckRiskHigh: 1.0,
  goDecisionDeckRiskMidCut: 20,
  goDecisionDeckRiskHighCut: 10,
  goDecisionLateDeckCut: 6,
  goDecisionThresholdBase: 0.0,
  goDecisionThresholdLowDeckUp: 0.2,
  goDecisionThresholdVeryLateDeckUp: 0.12,
  goDecisionThresholdLeadDown: 0.2,
  goDecisionOppGoCountCut: 1,
  goDecisionRuleDeckCut: 10,
  goDecisionRuleThreatCut: 0.6,
  goDecisionRuleLeadMax: 2,
  goDecisionRuleLateHighThreatThresholdUp: 0.25,
  goDecisionRuleOppScoreThresholdUp: 0.3,
  goDecisionRuleOppGoThresholdUp: 0.15,
  goDecisionLeadAggressiveCut: 7,
  goDecisionOppScoreDangerCut: 6,
  goDecisionOppThreatMapA: 0.9,
  goDecisionOppThreatMapB: 0.0,
  goDecisionOppThreatSigmoidK: 6.0,
  goDecisionOppThreatSigmoidX0: 0.55,
  goDecisionComboSigmoidK: 1.2,
  goDecisionComboSigmoidX0: 1.0,
  goDecisionMargin: 0.18,
  goDecisionAmbiguousLogEnabled: 0,
  goDecisionAmbiguousLogSampleRate: 0.2,
  goDecisionRule2ThreatCut: 0.55,
  goDecisionRule2LeadMax: 3,
  goDecisionRule2ThresholdUp: 0.15,
  goDecisionRule3DeckCut: 6,
  goDecisionRule3ThreatCut: 0.45,
  goDecisionRule3ThresholdUp: 0.1,
  goDecisionRule4ComboCut: 0.8,
  goDecisionRule4ThreatCut: 0.4,
  goDecisionRule4ThresholdDown: 0.12,
  goDecisionRule5LeadCut: 5,
  goDecisionRule5ThreatCut: 0.5,
  goDecisionRule5ThresholdDown: 0.1,
  goDecisionAggRewardComboMul: 0.35,
  goDecisionAggRewardLeadMul: 0.12,
  goDecisionAggRewardPiMul: 0.05,
  goDecisionAggRiskThreatMul: 0.4,
  goDecisionAggRiskDeckMul: 0.2,
  goDecisionAggRiskDenomMin: 0.01,
  goDecisionAggRatioCut: 2.5,
  goDecisionAggRatioThresholdDown: 0.08,
  goDecisionAggComboCut: 0.8,
  goDecisionAggComboThresholdDown: 0.07,
  goDecisionAggLeadCut: 6,
  goDecisionAggLeadThresholdDown: 0.07,
  goDecisionAggThreatLowCut: 0.25,
  goDecisionAggThreatThresholdDown: 0.05,
  goDecisionLogEnabled: 0,
  goDecisionLogSampleRate: 0.05,

  // bomb / shaking / president / gukjin
  bombBoardGainMin: 0.8,
  bombTrailEnableScoreDiff: -1,
  bombOppPiWindowLow: 6,
  bombOppPiWindowHigh: 7,
  bombHighThreatBlock: 0.9,
  shakingStopLossDangerCut: 0.8,
  shakingStopLossEnableTrail: 1,
  shakingOppKwangCut: 2,
  shakingHighRiskMonth: 12,
  shakingHighRiskProbCut: 0.5,
  presidentStopLead: 1,
  presidentStopThreatCut: 0.65,
  presidentCarryStopMax: 1,
  presidentLateDeckCut: 7,
  gukjinFiveCut: 4
};

// ============================================================================
// Section 1) Small utility helpers
// ============================================================================
function safeNum(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function otherPlayerKey(playerKey) {
  return playerKey === "human" ? "ai" : "human";
}

function asList(v) {
  return Array.isArray(v) ? v : [];
}

// Build a deduplicated captured-all list when only category buckets are provided.
function getCapturedAllCards(player) {
  const all = asList(player?.captured?.all);
  if (all.length > 0) return all;

  const merged = [
    ...asList(player?.captured?.junk),
    ...asList(player?.captured?.five),
    ...asList(player?.captured?.ribbon),
    ...asList(player?.captured?.animal),
    ...asList(player?.captured?.kwang)
  ];
  const seen = new Set();
  const out = [];
  for (const card of merged) {
    const id = card?.id;
    if (!id || seen.has(id)) continue;
    seen.add(id);
    out.push(card);
  }
  return out;
}

// Convert card id prefix (A..L) to month (1..12) as a fallback.
function monthFromCardId(cardId) {
  const id = typeof cardId === "string" ? cardId.trim().toUpperCase() : "";
  if (id.length <= 0) return 0;
  const c = id.charCodeAt(0);
  if (c < 65 || c > 76) return 0; // A..L -> 1..12
  return c - 64;
}

// ============================================================================
// Section 2) Jokbo self-tracking and fallback model
// ============================================================================
// Urgency scale:
// - 1.5: missing card is on board (emergency)
// - 0.7: missing card is unknown/deck (warning)
// - 0.05: missing card is in my hand (safe)
function calculateJokboUrgency(state, opponent, myHand, targetSet) {
  const captured = getCapturedAllCards(opponent).filter((c) => targetSet.has(c?.id));
  if (captured.length < 2) {
    return {
      urgency: 0.1,
      missingId: null,
      missingMonth: 0,
      onBoard: false,
      inMyHand: false
    };
  }

  const capturedIds = new Set(captured.map((c) => c?.id).filter(Boolean));
  const missingId = [...targetSet].find((id) => !capturedIds.has(id)) || null;
  if (!missingId) {
    return {
      urgency: 0.1,
      missingId: null,
      missingMonth: 0,
      onBoard: false,
      inMyHand: false
    };
  }

  const board = asList(state?.board);
  const hand = asList(myHand);
  const boardCard = board.find((c) => c?.id === missingId);
  const handCard = hand.find((c) => c?.id === missingId);
  const onBoard = Boolean(boardCard);
  const inMyHand = Boolean(handCard);
  const missingMonth = safeNum(boardCard?.month || handCard?.month, monthFromCardId(missingId));

  let urgency = 0.7;
  if (onBoard) urgency = 1.5;
  else if (inMyHand) urgency = 0.05;

  return { urgency, missingId, missingMonth, onBoard, inMyHand };
}

// Visible-only fallback tracker for opponent jokbo progress.
function selfTrackJokbo(player) {
  const captured = getCapturedAllCards(player);
  const results = {
    godori: 0,
    cheongDan: 0,
    hongDan: 0,
    choDan: 0,
    kwang: 0,
    threatScore: 0
  };

  for (const card of captured) {
    const id = card?.id;
    if (SELF_TRACK_CARD_IDS.godori.has(id)) results.godori += 1;
    if (SELF_TRACK_CARD_IDS.cheongDan.has(id)) results.cheongDan += 1;
    if (SELF_TRACK_CARD_IDS.hongDan.has(id)) results.hongDan += 1;
    if (SELF_TRACK_CARD_IDS.choDan.has(id)) results.choDan += 1;
    if (card?.category === "kwang") results.kwang += 1;
  }

  let threatScore = 0;
  if (results.godori >= 2) threatScore += 0.8;
  if (results.cheongDan >= 2 || results.hongDan >= 2 || results.choDan >= 2) threatScore += 0.7;
  if (results.kwang >= 2) threatScore += 0.6;
  results.threatScore = threatScore;
  return results;
}

// Normalize external jokboProgress format into stable internal fields.
function normalizeJokboProgress(raw) {
  const out = {};
  if (!raw || typeof raw !== "object") return out;
  for (const key of Object.keys(raw)) {
    const entry = raw[key] || {};
    const count = Math.max(0, safeNum(entry?.count));
    const neededCount = Math.max(1, safeNum(entry?.neededCount, 3));
    const targetMonths = [...new Set(asList(entry?.targetMonths).map((m) => safeNum(m)).filter((m) => m > 0))];
    out[key] = { count, neededCount, targetMonths };
  }
  return out;
}

// Fallback jokbo progress when the engine does not provide jokboProgress.
function buildSelfTrackedJokboProgress(player) {
  const tracked = selfTrackJokbo(player);
  return {
    godori: { count: tracked.godori, neededCount: 3, targetMonths: [2, 4, 8] },
    cheongDan: { count: tracked.cheongDan, neededCount: 3, targetMonths: [6, 9, 10] },
    hongDan: { count: tracked.hongDan, neededCount: 3, targetMonths: [1, 2, 3] },
    choDan: { count: tracked.choDan, neededCount: 3, targetMonths: [4, 5, 7] },
    kwang: { count: tracked.kwang, neededCount: 3, targetMonths: [1, 3, 8, 11, 12] }
  };
}

// External jokboProgress takes priority; fallback tracker is used otherwise.
function getEffectiveJokboProgress(player) {
  const raw = player?.jokboProgress;
  if (raw && typeof raw === "object" && Object.keys(raw).length > 0) {
    return normalizeJokboProgress(raw);
  }
  return buildSelfTrackedJokboProgress(player);
}

// ============================================================================
// Section 3) Derived card/jokbo helpers
// ============================================================================
function piValue(card, deps) {
  if (!card) return 0;
  if (card.id === GUKJIN_CARD_ID) return 2;
  if (card.category !== "junk") return 0;
  return safeNum(deps?.junkPiValue?.(card), safeNum(card?.piValue, 1));
}

function countComboNear(player) {
  const j = getEffectiveJokboProgress(player);
  let near = 0;
  for (const k of Object.keys(j)) {
    const x = j[k] || {};
    const count = safeNum(x.count);
    const need = safeNum(x.neededCount, 3);
    if (need > 0 && count >= need - 1) near += 1;
  }
  return near;
}

function getJokboTargetPressure(player, month) {
  const m = safeNum(month);
  const jokbo = getEffectiveJokboProgress(player);
  let nearFinish = 0;
  let build = 0;
  for (const key of Object.keys(jokbo)) {
    const j = jokbo[key] || {};
    const targets = asList(j?.targetMonths);
    if (!targets.includes(m)) continue;
    const count = safeNum(j?.count);
    const need = safeNum(j?.neededCount, 3);
    if (count >= need - 1) nearFinish += 1;
    else if (count >= need - 2) build += 1;
  }
  return { nearFinish, build };
}

function collectSynergyMonths(player) {
  const out = new Set();
  const jokbo = getEffectiveJokboProgress(player);
  for (const key of Object.keys(jokbo)) {
    const j = jokbo[key] || {};
    const count = safeNum(j?.count);
    const need = safeNum(j?.neededCount, 3);
    if (count >= need - 2) {
      for (const month of asList(j?.targetMonths)) out.add(safeNum(month));
    }
  }
  return out;
}

// Collect public/visible cards used by probability inference.
function gatherVisibleCards(state, playerKey) {
  const me = state?.players?.[playerKey];
  const opp = state?.players?.[otherPlayerKey(playerKey)];
  return [
    ...asList(state?.board),
    ...asList(me?.hand),
    ...asList(state?.discarded),
    ...asList(me?.captured?.all),
    ...asList(opp?.captured?.all),
    ...asList(me?.captured?.junk),
    ...asList(me?.captured?.five),
    ...asList(me?.captured?.ribbon),
    ...asList(me?.captured?.kwang),
    ...asList(opp?.captured?.junk),
    ...asList(opp?.captured?.five),
    ...asList(opp?.captured?.ribbon),
    ...asList(opp?.captured?.kwang)
  ].filter(Boolean);
}

function buildVisibleMonthCount(cards) {
  const out = {};
  for (const c of asList(cards)) {
    const month = safeNum(c?.month);
    if (month <= 0) continue;
    out[month] = safeNum(out[month], 0) + 1;
  }
  return out;
}

// Total card count may include bonus cards in some modes.
function getDynamicTotalCards(state, P = DEFAULT_PARAMS) {
  const base = 48;
  const bonus = safeNum(
    state?.bonusCardCount ?? state?.config?.bonusCount,
    safeNum(P?.bonusDefaultCount, 2)
  );
  return Math.max(base, base + bonus);
}

// ============================================================================
// Section 4) Inference core
// Outputs:
// - month probability and danger maps
// - inferred holding/safe month hints
// - jokbo urgency target month
// ============================================================================
function buildInference(state, playerKey, P = DEFAULT_PARAMS) {
  const me = state?.players?.[playerKey];
  const oppKey = otherPlayerKey(playerKey);
  const opp = state?.players?.[oppKey];
  const visible = gatherVisibleCards(state, playerKey);
  const visibleMonthCount = buildVisibleMonthCount(visible);
  const totalCards = getDynamicTotalCards(state, P);
  const unknownTotal = Math.max(1, totalCards - visible.length);
  const history = asList(state?.actionHistory);

  const probs = {};
  const holdings = {};
  const safeMonths = new Set();
  const dangerLevel = {};
  const dangerCards = new Set();

  for (let month = 1; month <= 12; month += 1) {
    const knownInMonth = safeNum(visibleMonthCount[month], 0);
    const hiddenInMonth = clamp(4 - knownInMonth, 0, 4);
    probs[month] = clamp(hiddenInMonth / unknownTotal, 0, 1);
  }

  for (const action of history.slice(-16)) {
    if (action?.player !== oppKey) continue;
    if (action?.type === "play" && action?.hadMatchOnBoard && !action?.selectedMatch) {
      holdings[safeNum(action?.cardPlayed?.month)] = true;
    }
    if (action?.type === "puck" || action?.isPuck) {
      holdings[safeNum(action?.month)] = true;
    }
    const played = action?.cardPlayed;
    if (played && (played.category === "kwang" || safeNum(played?.piValue) >= 2)) {
      holdings[safeNum(played?.month)] = true;
    }
  }

  // Latest no-match month from opponent is relatively safer to discard.
  const lastOppAction = [...history].reverse().find((a) => a?.player === oppKey && a?.cardPlayed);
  if (lastOppAction && !lastOppAction?.hadMatchOnBoard) {
    const m = safeNum(lastOppAction?.cardPlayed?.month);
    if (m > 0) safeMonths.add(m);
  }

  const oppJokbo = getEffectiveJokboProgress(opp);
  const trackedOpp = selfTrackJokbo(opp);
  const jokboUrgencyByMonth = {};
  const jokboUrgencyReasons = {};
  let topDangerMonth = 0;
  let topJokboUrgency = 0;
  for (let month = 1; month <= 12; month += 1) {
    let risk = 0;
    for (const key of Object.keys(oppJokbo)) {
      const j = oppJokbo[key] || {};
      const targets = asList(j?.targetMonths);
      if (!targets.includes(month)) continue;
      const count = safeNum(j?.count);
      const need = safeNum(j?.neededCount, 3);
      risk += count >= need - 1 ? 0.55 : 0.2;
      if (count >= need - 1) dangerCards.add(month);
    }
    dangerLevel[month] = clamp(risk, 0, 1);
  }

  for (const [name, targetSet] of Object.entries(SELF_TRACK_CARD_IDS)) {
    const urgencyInfo = calculateJokboUrgency(state, opp, me?.hand, targetSet);
    const month = safeNum(urgencyInfo?.missingMonth);
    if (month <= 0) continue;

    const urgency = safeNum(urgencyInfo?.urgency);
    jokboUrgencyByMonth[month] = Math.max(safeNum(jokboUrgencyByMonth[month], 0), urgency);
    if (!jokboUrgencyReasons[month]) jokboUrgencyReasons[month] = [];
    jokboUrgencyReasons[month].push(name);

    const urgencyDanger = clamp(urgency / 1.5, 0, 1);
    dangerLevel[month] = Math.max(safeNum(dangerLevel[month], 0), urgencyDanger);
    if (urgency >= 1.0) dangerCards.add(month);

    if (urgency > topJokboUrgency) {
      topJokboUrgency = urgency;
      topDangerMonth = month;
    }
  }

  const dangerFromMap = Object.values(dangerLevel).reduce((a, b) => Math.max(a, safeNum(b)), 0);
  const trackedThreat = clamp(safeNum(trackedOpp?.threatScore) / 2.1, 0, 1);
  const maxDanger = Math.max(dangerFromMap, trackedThreat);

  return {
    probs,
    holdings,
    safeMonths,
    dangerLevel,
    dangerCards,
    visibleMonthCount,
    maxDanger,
    trackedOppThreat: trackedThreat,
    jokboUrgencyByMonth,
    jokboUrgencyReasons,
    topDangerMonth,
    topJokboUrgency,
    oppNearComboCount: countComboNear(opp),
    selfNearComboCount: countComboNear(me)
  };
}

// ============================================================================
// Section 5) Card-level scoring components
// ============================================================================
// One-step tactical look-ahead:
// - trap opportunity
// - feed risk
function evaluateLookAheadScore(state, playerKey, card, matches, inference, P) {
  const month = safeNum(card?.month);
  let score = 0;

  // Trap opportunity when opponent likely holds this month.
  if (matches.length === 0 && inference?.holdings?.[month]) {
    const knownInMonth = safeNum(inference?.visibleMonthCount?.[month], 0);
    const hiddenInMonth = Math.max(0, 4 - knownInMonth);
    if (hiddenInMonth >= 2) score += safeNum(P.lookAheadTrapBonus, 1.6);
  }

  // Single feed penalty path for consistency.
  if (matches.length === 0 && safeNum(inference?.probs?.[month]) > 0.4) {
    const puckDangerWeight = safeNum(P?.puckDangerWeight, 0);
    if (puckDangerWeight !== 0) score += puckDangerWeight;
  }
  return score;
}

function calculatePreemptiveRisk(inference, month, P) {
  const m = safeNum(month);
  if (!inference?.dangerCards?.has(m)) return 0;
  if (safeNum(inference?.probs?.[m]) < safeNum(P.preemptiveDangerMonthProbCut, 0.3)) return 0;
  return safeNum(P.preemptiveDangerMonthPenalty, 1.8);
}

function classifyPhase(deckCount, scoreDiff, P) {
  if (deckCount > safeNum(P.phaseEarlyDeckCut, 35)) return "early";
  if (scoreDiff >= safeNum(P.phaseLeadLockScoreDiff, 5) && deckCount <= safeNum(P.phaseEndDeckCut, 10)) return "lead_end";
  if (scoreDiff <= safeNum(P.phaseReversalScoreDiff, -7)) return "reversal";
  return "mid";
}

function phaseMultipliers(phase) {
  if (phase === "early") return { attack: 1.05, defense: 1.0, risk: 0.95 };
  if (phase === "lead_end") return { attack: 0.9, defense: 1.25, risk: 1.2 };
  if (phase === "reversal") return { attack: 1.3, defense: 0.85, risk: 0.9 };
  return { attack: 1.0, defense: 1.0, risk: 1.0 };
}

function detectOpponentReverseTrap(state, playerKey, card) {
  const oppKey = otherPlayerKey(playerKey);
  const history = asList(state?.actionHistory);
  const lastOppAction = [...history].reverse().find((a) => a?.player === oppKey);
  if (!lastOppAction) return 0;
  const played = lastOppAction?.cardPlayed;
  if (!played) return 0;
  if (played?.category === "kwang" && !lastOppAction?.hadMatchOnBoard && safeNum(card?.month) === safeNum(played?.month)) {
    return 1;
  }
  return 0;
}

// ============================================================================
// Section 6) Context + full card evaluation
// ============================================================================
function preemptiveDefenseWeight(baseDefense, inference, P) {
  if (safeNum(inference?.maxDanger) >= safeNum(P.preemptiveDangerCut, 0.62)) {
    return baseDefense * safeNum(P.preemptiveDefenseMul, 1.8);
  }
  return baseDefense;
}

function buildContext(state, playerKey, deps, P, inference) {
  const me = state?.players?.[playerKey];
  const opp = state?.players?.[otherPlayerKey(playerKey)];
  const deckCount = safeNum(state?.deckCount, asList(state?.deck).length);
  const scoreDiff = safeNum(me?.score) - safeNum(opp?.score);
  const phase = classifyPhase(deckCount, scoreDiff, P);
  const m = phaseMultipliers(phase);

  const oppGo = safeNum(opp?.goCount);
  const defenseBoost = oppGo > 0 ? 1.3 : 1.0;

  return {
    deckCount,
    scoreDiff,
    phase,
    attackMul: m.attack,
    defenseMul: preemptiveDefenseWeight(m.defense * defenseBoost, inference, P),
    riskMul: m.risk,
    selfPi: safeNum(deps?.capturedCountByCategory?.(me, "junk"), asList(me?.captured?.junk).length),
    oppPi: safeNum(deps?.capturedCountByCategory?.(opp, "junk"), asList(opp?.captured?.junk).length),
    lead: scoreDiff,
    trailing: scoreDiff < 0,
    leading: scoreDiff > 0,
    oppGoCount: oppGo,
    synergyMonths: collectSynergyMonths(me),
    second: typeof state?.startingTurnKey === "string" ? state.startingTurnKey !== playerKey : false
  };
}

function evaluateCard(state, playerKey, card, deps, P, inference, ctx) {
  const me = state?.players?.[playerKey];
  const opp = state?.players?.[otherPlayerKey(playerKey)];
  const month = safeNum(card?.month);
  const matches = asList(state?.board).filter((c) => safeNum(c?.month) === month);
  const captured = [card, ...matches];
  const captureGain = captured.reduce((sum, c) => sum + safeNum(deps?.cardCaptureValue?.(c), 0), 0);
  const piGain = captured.reduce((sum, c) => sum + piValue(c, deps), 0);
  const ownJokbo = getJokboTargetPressure(me, month);
  const oppJokbo = getJokboTargetPressure(opp, month);

  let score = matches.length === 0
    ? safeNum(P.noMatchBase, -5.2)
    : matches.length === 1
      ? safeNum(P.matchOneBase, 4.0)
      : matches.length === 2
        ? safeNum(P.matchTwoBase, 8.6)
        : safeNum(P.matchThreeBase, 11.5);

  score += captureGain * safeNum(P.captureGainMul, 1.0);
  score += piGain * safeNum(P.piMul, 4.6);
  if (card?.category === "kwang") score += safeNum(P.categoryKwangBonus, 2.4);
  if (card?.category === "five") score += safeNum(P.categoryFiveBonus, 1.6);
  if (card?.category === "ribbon") score += safeNum(P.categoryRibbonBonus, 1.1);
  score += safeNum(deps?.ownComboOpportunityScore?.(state, playerKey, month), 0) * safeNum(P.comboOpportunityMul, 3.1);
  if (HIGH_VALUE_MONTHS.includes(month)) score += safeNum(P.highValueMonthBonus, 0.7);
  if (card?.isBonus || month === 0) score += safeNum(P.bonusCardPriorityBonus, 2.2);
  if (card?.id === GUKJIN_CARD_ID) score += 0.8;
  score += ownJokbo.nearFinish * safeNum(P.jokboNearFinishBonus, 4.5);
  score += ownJokbo.build * safeNum(P.jokboBuildBonus, 1.0);
  score += oppJokbo.nearFinish * safeNum(P.oppJokboBlockBonus, 2.1);
  if (matches.length === 0 && oppJokbo.nearFinish > 0) {
    score += oppJokbo.nearFinish * safeNum(P.oppJokboNearPenalty, -35.0);
  }
  if (ctx?.synergyMonths?.has(month)) score += 0.8;

  const blockMonths = new Set(asList(deps?.blockingMonthsAgainst?.(opp, me)));
  const blockUrgency = deps?.blockingUrgencyByMonth?.(opp, me) || new Map();
  if (blockMonths.has(month)) {
    score += Math.max(0, safeNum(blockUrgency?.get?.(month), 0)) * safeNum(P.blockUrgencyMul, 6.5);
  }
  const jokboUrgency = safeNum(inference?.jokboUrgencyByMonth?.[month], 0);
  if (jokboUrgency > safeNum(P.blockHardUrgencyCut, 1.0)) {
    score += safeNum(P.blockUrgencyMul, 6.5) * safeNum(P.blockHardUrgencyMul, 10.0) * jokboUrgency;
  }
  if (
    safeNum(inference?.topJokboUrgency, 0) > safeNum(P.blockHardUrgencyCut, 1.0) &&
    month === safeNum(inference?.topDangerMonth)
  ) {
    score += safeNum(P.blockCriticalBonus, 60.0);
  }

  if (inference?.holdings?.[month]) score -= safeNum(P.holdingPenalty, 4.8);
  if (inference?.dangerCards?.has(month)) score -= safeNum(P.dangerMonthPenalty, 3.1);
  score -= safeNum(deps?.estimateOpponentImmediateGainIfDiscard?.(state, playerKey, month), 0) *
    (matches.length === 0 ? safeNum(P.noMatchFeedRiskMul, 1.2) : safeNum(P.matchFeedRiskMul, 0.5));
  score += evaluateLookAheadScore(state, playerKey, card, matches, inference, P);
  score -= calculatePreemptiveRisk(inference, month, P);

  if (matches.length === 0 && inference?.holdings?.[month]) {
    const knownInMonth = safeNum(inference?.visibleMonthCount?.[month], 0);
    const hiddenInMonth = Math.max(0, 4 - knownInMonth);
    if (hiddenInMonth >= 2) score += safeNum(P.trapBonus, 1.9);
  }
  if (matches.length === 2 && inference?.holdings?.[month]) {
    score -= safeNum(P.secondOrderTrapPenalty, 1.3);
  }
  if (detectOpponentReverseTrap(state, playerKey, card) > 0) {
    score -= safeNum(P.reverseTrapPenalty, 2.0);
  }

  score *= safeNum(ctx.attackMul, 1.0);
  if (ctx.trailing && matches.length === 0) score -= 0.6 * safeNum(ctx.defenseMul, 1.0);
  if (ctx.leading && matches.length === 0) score -= 0.2 * safeNum(ctx.riskMul, 1.0);

  return {
    card,
    score,
    matches: matches.length,
    piGain,
    captureGain
  };
}

// Stable rank comparator used by play-card selection.
function compareRank(a, b) {
  if (b.score !== a.score) return b.score - a.score;
  if (b.matches !== a.matches) return b.matches - a.matches;
  if (b.piGain !== a.piGain) return b.piGain - a.piGain;
  return b.captureGain - a.captureGain;
}

// ============================================================================
// Section 7) GO/STOP model
// ============================================================================
function extractGoCore(state, playerKey, deps, P, inference, ctx) {
  const me = state?.players?.[playerKey];
  const opp = state?.players?.[otherPlayerKey(playerKey)];
  const deckCount = safeNum(state?.deckCount, asList(state?.deck).length);

  const myScore = safeNum(me?.score);
  const oppScore = safeNum(opp?.score);
  const lead = myScore - oppScore;
  const goCount = safeNum(me?.goCount);
  const oppGoCount = safeNum(ctx?.oppGoCount, 0);
  const carry = Math.max(0, myScore - 6);
  const selfPi = safeNum(ctx.selfPi);
  const oppPi = safeNum(ctx.oppPi);
  const selfComboNear = countComboNear(me);
  const oppComboNear = safeNum(inference?.oppNearComboCount);
  const threat = clamp(
    safeNum(inference?.maxDanger) * 0.65 + clamp(oppComboNear / 3, 0, 1) * 0.35,
    0,
    1.4
  );
  const oneAwayProb = clamp(
    safeNum(inference?.probs?.[HIGH_VALUE_MONTHS[0]]) * 0.2 +
    safeNum(inference?.probs?.[HIGH_VALUE_MONTHS[1]]) * 0.2 +
    safeNum(inference?.probs?.[HIGH_VALUE_MONTHS[2]]) * 0.2 +
    safeNum(inference?.probs?.[HIGH_VALUE_MONTHS[3]]) * 0.2 +
    safeNum(inference?.probs?.[HIGH_VALUE_MONTHS[4]]) * 0.2,
    0,
    1
  );

  const selfCanStop = myScore >= 7;
  const oppCanStop = oppScore >= 7;
  const desperate = lead <= -6;
  const minPi = desperate ? safeNum(P.goMinPiDesperate, 5) : safeNum(P.goMinPi, 6);

  return {
    myScore,
    oppScore,
    lead,
    selfPi,
    oppPi,
    selfComboNear,
    oppComboNear,
    goCount,
    oppGoCount,
    carry,
    deckCount,
    threat,
    oneAwayProb,
    selfCanStop,
    oppCanStop,
    desperate,
    minPi
  };
}

function computeLinearAndSigmoidThreat(inference, P) {
  const linearThreat = clamp(
    safeNum(inference?.maxDanger, 0) * safeNum(P.goDecisionOppThreatMapA, 0.9) +
    safeNum(P.goDecisionOppThreatMapB, 0),
    0,
    1
  );
  const oppThreat = clamp(
    sigmoid(
      safeNum(P.goDecisionOppThreatSigmoidK, 6.0) *
      (linearThreat - safeNum(P.goDecisionOppThreatSigmoidX0, 0.55))
    ),
    0,
    1
  );
  return { linearThreat, oppThreat };
}

function computeSelfComboChance(core, P) {
  const comboNear = safeNum(core?.selfComboNear, 0);
  return clamp(
    sigmoid(
      safeNum(P.goDecisionComboSigmoidK, 1.2) *
      (comboNear - safeNum(P.goDecisionComboSigmoidX0, 1.0))
    ),
    0,
    1
  );
}

function computeDeckRisk(deckCount, P) {
  let deckRisk = safeNum(P.goDecisionDeckRiskBase, 0.2);
  if (deckCount <= safeNum(P.goDecisionDeckRiskHighCut, 10)) {
    deckRisk = safeNum(P.goDecisionDeckRiskHigh, 1.0);
  } else if (deckCount <= safeNum(P.goDecisionDeckRiskMidCut, 20)) {
    deckRisk = safeNum(P.goDecisionDeckRiskMid, 0.5);
  }
  return deckRisk;
}

function computeExpectedGoValue(core, inference, P) {
  const lead = safeNum(core?.lead);
  const deckCount = safeNum(core?.deckCount);
  const selfPi = safeNum(core?.selfPi);
  const { linearThreat, oppThreat } = computeLinearAndSigmoidThreat(inference, P);
  const selfComboChance = computeSelfComboChance(core, P);
  const deckRisk = computeDeckRisk(deckCount, P);

  const expectedGoValue =
    lead * safeNum(P.goDecisionScoreWeight, 0.1) +
    selfComboChance * safeNum(P.goDecisionComboWeight, 0.26) +
    selfPi * safeNum(P.goDecisionPiWeight, 0.05) -
    oppThreat * safeNum(P.goDecisionThreatWeight, 0.45) -
    deckRisk * safeNum(P.goDecisionDeckWeight, 0.2);

  return {
    expectedGoValue,
    linearThreat,
    oppThreat,
    selfComboChance,
    deckRisk
  };
}

function computeRiskRewardRatio(core, inference, P, precomputed = null) {
  const lead = safeNum(core?.lead);
  const selfPi = safeNum(core?.selfPi);
  const ev = precomputed || computeExpectedGoValue(core, inference, P);
  const rewardScore =
    safeNum(ev?.selfComboChance) * safeNum(P.goDecisionAggRewardComboMul, 0.35) +
    lead * safeNum(P.goDecisionAggRewardLeadMul, 0.12) +
    selfPi * safeNum(P.goDecisionAggRewardPiMul, 0.05);
  const riskScore =
    safeNum(ev?.oppThreat) * safeNum(P.goDecisionAggRiskThreatMul, 0.4) +
    safeNum(ev?.deckRisk) * safeNum(P.goDecisionAggRiskDeckMul, 0.2);
  const ratio = rewardScore / Math.max(safeNum(P.goDecisionAggRiskDenomMin, 0.01), riskScore);
  return { rewardScore, riskScore, ratio };
}

// Conservative hard-stop guards to prevent high-cost GO failures.
function enforceHardStopGuards(core, inference, P, precomputed = null) {
  const deckCount = safeNum(core?.deckCount);
  const lead = safeNum(core?.lead);
  const selfPi = safeNum(core?.selfPi);
  const ev = precomputed || computeExpectedGoValue(core, inference, P);
  const oppThreat = safeNum(ev?.oppThreat);
  const selfComboChance = safeNum(ev?.selfComboChance);

  if (oppThreat >= safeNum(P.goHardStopOppThreatCut, 0.85)) {
    return { hardStop: true, reason: "HIGH_OPP_THREAT" };
  }
  if (
    deckCount <= safeNum(P.goHardStopVeryLateDeckCut, 4) &&
    oppThreat >= safeNum(P.goHardStopVeryLateOppThreatCut, 0.6)
  ) {
    return { hardStop: true, reason: "VERY_LATE_DECK_HIGH_THREAT" };
  }
  if (
    selfPi < safeNum(P.goHardStopLowPiCut, 3) &&
    lead <= safeNum(P.goHardStopLowPiLeadMax, 0) &&
    oppThreat >= safeNum(P.goHardStopLowPiThreatCut, 0.6)
  ) {
    return { hardStop: true, reason: "LOW_PI_HIGH_THREAT" };
  }
  if (
    selfComboChance < safeNum(P.goHardStopLowComboCut, 0.4) &&
    oppThreat >= safeNum(P.goHardStopLowComboThreatCut, 0.7)
  ) {
    return { hardStop: true, reason: "LOW_COMBO_HIGH_THREAT" };
  }
  return { hardStop: false, reason: "" };
}

// Dynamic GO decision:
// expectedGoValue = lead + combo + selfPi - oppThreat - deckRisk
// threshold is adjusted by deck/lead/opponent score.
function decideGoDynamic(core, inference, P) {
  const lead = safeNum(core?.lead);
  const deckCount = safeNum(core?.deckCount);
  const oppScore = safeNum(core?.oppScore);
  const selfPi = safeNum(core?.selfPi);
  const oppGoCount = safeNum(core?.oppGoCount, 0);
  const ev = computeExpectedGoValue(core, inference, P);
  const expectedGoValue = safeNum(ev?.expectedGoValue, 0);
  const linearThreat = safeNum(ev?.linearThreat, 0);
  const oppThreat = safeNum(ev?.oppThreat, 0);
  const selfComboChance = safeNum(ev?.selfComboChance, 0);
  const deckRisk = safeNum(ev?.deckRisk, safeNum(P.goDecisionDeckRiskBase, 0.2));

  let threshold = safeNum(P.goDecisionThresholdBase, 0.0);
  if (deckCount <= safeNum(P.goDecisionDeckRiskHighCut, 10)) {
    threshold += safeNum(P.goDecisionThresholdLowDeckUp, 0.2);
  }
  if (deckCount <= safeNum(P.goDecisionLateDeckCut, 6)) {
    threshold += safeNum(P.goDecisionThresholdVeryLateDeckUp, 0.12);
  }
  if (lead >= safeNum(P.goDecisionLeadAggressiveCut, 7)) {
    threshold -= safeNum(P.goDecisionThresholdLeadDown, 0.2);
  }

  // Extensible rule-table style threshold adjustments.
  const ruleHits = [];
  const ruleTable = [
    {
      id: "late_highthreat_smalllead",
      when:
        deckCount <= safeNum(P.goDecisionRuleDeckCut, 10) &&
        oppThreat >= safeNum(P.goDecisionRuleThreatCut, 0.6) &&
        lead <= safeNum(P.goDecisionRuleLeadMax, 2),
      delta: safeNum(P.goDecisionRuleLateHighThreatThresholdUp, 0.25)
    },
    {
      id: "opp_score_high",
      when: oppScore >= safeNum(P.goDecisionOppScoreDangerCut, 6),
      delta: safeNum(P.goDecisionRuleOppScoreThresholdUp, 0.3)
    },
    {
      id: "opp_go_pressure",
      when: oppGoCount >= safeNum(P.goDecisionOppGoCountCut, 1),
      delta: safeNum(P.goDecisionRuleOppGoThresholdUp, 0.15)
    },
    {
      id: "opp_go_highthreat",
      when:
        oppGoCount >= safeNum(P.goDecisionOppGoCountCut, 1) &&
        oppThreat >= safeNum(P.goDecisionRule2ThreatCut, 0.55) &&
        lead <= safeNum(P.goDecisionRule2LeadMax, 3),
      delta: safeNum(P.goDecisionRule2ThresholdUp, 0.15)
    },
    {
      id: "very_late_threat",
      when:
        deckCount <= safeNum(P.goDecisionRule3DeckCut, 6) &&
        oppThreat >= safeNum(P.goDecisionRule3ThreatCut, 0.45),
      delta: safeNum(P.goDecisionRule3ThresholdUp, 0.1)
    },
    {
      id: "strong_combo_low_threat",
      when:
        selfComboChance >= safeNum(P.goDecisionRule4ComboCut, 0.8) &&
        oppThreat <= safeNum(P.goDecisionRule4ThreatCut, 0.4),
      delta: -safeNum(P.goDecisionRule4ThresholdDown, 0.12)
    },
    {
      id: "lead_advantage_moderate_threat",
      when:
        lead >= safeNum(P.goDecisionRule5LeadCut, 5) &&
        oppThreat <= safeNum(P.goDecisionRule5ThreatCut, 0.5),
      delta: -safeNum(P.goDecisionRule5ThresholdDown, 0.1)
    }
  ];
  for (const rule of ruleTable) {
    if (!rule.when) continue;
    threshold += safeNum(rule.delta, 0);
    ruleHits.push(rule.id);
  }

  // Aggressive correction: lower threshold when reward-risk profile is favorable.
  const rr = computeRiskRewardRatio(core, inference, P, ev);
  const riskRewardRatio = safeNum(rr?.ratio, 0);
  const ruleHitsAggressive = [];
  if (riskRewardRatio >= safeNum(P.goDecisionAggRatioCut, 2.5)) {
    threshold -= safeNum(P.goDecisionAggRatioThresholdDown, 0.08);
    ruleHitsAggressive.push("AGG:risk_reward_ratio");
  }
  if (selfComboChance >= safeNum(P.goDecisionAggComboCut, 0.8)) {
    threshold -= safeNum(P.goDecisionAggComboThresholdDown, 0.07);
    ruleHitsAggressive.push("AGG:combo_high");
  }
  if (lead >= safeNum(P.goDecisionAggLeadCut, 6)) {
    threshold -= safeNum(P.goDecisionAggLeadThresholdDown, 0.07);
    ruleHitsAggressive.push("AGG:lead_high");
  }
  if (oppThreat <= safeNum(P.goDecisionAggThreatLowCut, 0.25)) {
    threshold -= safeNum(P.goDecisionAggThreatThresholdDown, 0.05);
    ruleHitsAggressive.push("AGG:threat_low");
  }

  const margin = Math.max(0, safeNum(P.goDecisionMargin, 0.18));
  if (Math.abs(expectedGoValue - threshold) <= margin) {
    return {
      decision: false,
      expectedGoValue,
      threshold,
      oppThreat,
      linearThreat,
      selfComboChance,
      ruleHits,
      ruleHitsAggressive,
      riskRewardRatio,
      margin,
      hardStopReason: "",
      reason: "ambiguous_margin"
    };
  }

  const hardStop = enforceHardStopGuards(core, inference, P, ev);
  if (hardStop.hardStop) {
    return {
      decision: false,
      expectedGoValue,
      threshold,
      oppThreat,
      linearThreat,
      selfComboChance,
      ruleHits,
      ruleHitsAggressive,
      riskRewardRatio,
      margin,
      hardStopReason: String(hardStop.reason || "HARD_STOP"),
      reason: "hard_stop_guard"
    };
  }

  return {
    decision: expectedGoValue > threshold,
    expectedGoValue,
    threshold,
    oppThreat,
    linearThreat,
    selfComboChance,
    ruleHits,
    ruleHitsAggressive,
    riskRewardRatio,
    margin,
    hardStopReason: "",
    reason: "dynamic_threshold"
  };
}

function logAmbiguousGoDecision(core, inference, dynamicGo, finalValue, finalThreshold, P) {
  if (safeNum(P?.goDecisionAmbiguousLogEnabled, 1) <= 0) return;
  const sampleRate = clamp(safeNum(P?.goDecisionAmbiguousLogSampleRate, 0.2), 0, 1);
  if (sampleRate <= 0 || Math.random() > sampleRate) return;
  const payload = {
    tag: "AntiCLGoAmbiguous",
    expectedGoValue: safeNum(dynamicGo?.expectedGoValue),
    threshold: safeNum(dynamicGo?.threshold),
    finalValue: safeNum(finalValue),
    finalThreshold: safeNum(finalThreshold),
    margin: safeNum(dynamicGo?.margin, safeNum(P?.goDecisionMargin, 0.18)),
    lead: safeNum(core?.lead),
    deckCount: safeNum(core?.deckCount),
    oppThreat: safeNum(dynamicGo?.oppThreat, safeNum(inference?.maxDanger)),
    linearThreat: safeNum(dynamicGo?.linearThreat),
    selfComboChance: safeNum(dynamicGo?.selfComboChance),
    selfComboNear: safeNum(core?.selfComboNear),
    maxDanger: safeNum(inference?.maxDanger),
    ruleHits: Array.isArray(dynamicGo?.ruleHits) ? dynamicGo.ruleHits : [],
    ruleHitsAggressive: Array.isArray(dynamicGo?.ruleHitsAggressive) ? dynamicGo.ruleHitsAggressive : [],
    riskRewardRatio: safeNum(dynamicGo?.riskRewardRatio)
  };
  console.log(`[AntiCL][GO_AMBIGUOUS] ${JSON.stringify(payload)}`);
}

function logGoDecision(core, inference, dynamicGo, finalValue, finalThreshold, decision, P, reason = "") {
  if (safeNum(P?.goDecisionLogEnabled, 0) <= 0) return;
  const sampleRate = clamp(safeNum(P?.goDecisionLogSampleRate, 0.05), 0, 1);
  if (sampleRate <= 0 || Math.random() > sampleRate) return;
  const payload = {
    tag: "AntiCLGoDecision",
    decision: Boolean(decision) ? "GO" : "STOP",
    reason: String(reason || ""),
    expectedGoValue: safeNum(dynamicGo?.expectedGoValue),
    threshold: safeNum(dynamicGo?.threshold),
    finalValue: safeNum(finalValue),
    finalThreshold: safeNum(finalThreshold),
    lead: safeNum(core?.lead),
    deckCount: safeNum(core?.deckCount),
    oppThreat: safeNum(dynamicGo?.oppThreat, safeNum(inference?.maxDanger)),
    selfComboChance: safeNum(dynamicGo?.selfComboChance),
    selfPi: safeNum(core?.selfPi),
    oppPi: safeNum(core?.oppPi),
    ruleHits: Array.isArray(dynamicGo?.ruleHits) ? dynamicGo.ruleHits : [],
    ruleHitsAggressive: Array.isArray(dynamicGo?.ruleHitsAggressive) ? dynamicGo.ruleHitsAggressive : [],
    riskRewardRatio: safeNum(dynamicGo?.riskRewardRatio)
  };
  console.log(`[AntiCL][GO_DECISION] ${JSON.stringify(payload)}`);
}

// Emergency brake:
// - lethal opponent threat
// - pibak exposure while currently leading
function shouldStopEmergency(core, inference, P) {
  if (safeNum(inference?.maxDanger) >= safeNum(P.goEmergencyDangerCut, 1.0)) return true;
  if (
    safeNum(core?.oppPi) >= safeNum(P.goEmergencyOppPiCut, 9) &&
    safeNum(core?.selfPi) < safeNum(P.goEmergencyMyPiCut, 5) &&
    safeNum(core?.lead) > safeNum(P.goEmergencyLeadMin, 0)
  ) {
    return true;
  }
  return false;
}

// ============================================================================
// Section 8) Exported decision surface
// ============================================================================
export function rankHandCardsAntiCL(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const me = state?.players?.[playerKey];
  if (!Array.isArray(me?.hand) || me.hand.length <= 0) return [];
  const P = { ...DEFAULT_PARAMS, ...(params || {}) };
  const inference = buildInference(state, playerKey, P);
  const ctx = buildContext(state, playerKey, deps, P, inference);

  const ranked = me.hand.map((card) => evaluateCard(state, playerKey, card, deps, P, inference, ctx));
  ranked.sort(compareRank);
  return ranked;
}

export function chooseMatchHeuristicAntiCL(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const ids = asList(state?.pendingMatch?.boardCardIds);
  if (ids.length <= 0) return null;
  const P = { ...DEFAULT_PARAMS, ...(params || {}) };
  const inference = buildInference(state, playerKey, P);
  const ctx = buildContext(state, playerKey, deps, P, inference);

  const candidates = asList(state?.board).filter((c) => ids.includes(c?.id));
  if (candidates.length <= 0) return null;

  let best = null;
  for (const card of candidates) {
    const month = safeNum(card?.month);
    const ownJokbo = getJokboTargetPressure(state?.players?.[playerKey], month);
    const oppJokbo = getJokboTargetPressure(state?.players?.[otherPlayerKey(playerKey)], month);
    let score =
      safeNum(deps?.cardCaptureValue?.(card)) +
      piValue(card, deps) * safeNum(P.piMul, 4.6);
    score += ownJokbo.nearFinish * safeNum(P.jokboNearFinishBonus, 4.5);
    score += ownJokbo.build * safeNum(P.jokboBuildBonus, 1.0);
    score += oppJokbo.nearFinish * safeNum(P.oppJokboBlockBonus, 2.1);
    const jokboUrgency = safeNum(inference?.jokboUrgencyByMonth?.[month], 0);
    if (jokboUrgency > safeNum(P.blockHardUrgencyCut, 1.0)) {
      score += safeNum(P.blockUrgencyMul, 6.5) * safeNum(P.blockHardUrgencyMul, 10.0) * jokboUrgency;
    }
    if (
      safeNum(inference?.topJokboUrgency, 0) > safeNum(P.blockHardUrgencyCut, 1.0) &&
      month === safeNum(inference?.topDangerMonth)
    ) {
      score += safeNum(P.blockCriticalBonus, 60.0);
    }
    if (inference?.dangerCards?.has(month)) score += safeNum(P.blockUrgencyMul, 6.5);
    if (HIGH_VALUE_MONTHS.includes(month)) score += safeNum(P.highValueMonthBonus, 0.7);
    if (card?.isBonus || month === 0) score += safeNum(P.bonusCardPriorityBonus, 2.2);
    score -= calculatePreemptiveRisk(inference, month, P);
    score += safeNum(deps?.monthStrategicPriority?.(month), 0) * 0.2;
    score *= safeNum(ctx.defenseMul, 1.0);

    if (!best || score > best.score) best = { id: card?.id || null, score };
  }
  return best?.id || null;
}

export function shouldGoAntiCL(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const P = { ...DEFAULT_PARAMS, ...(params || {}) };
  if (typeof deps?.canBankruptOpponentByStop === "function" && deps.canBankruptOpponentByStop(state, playerKey)) {
    return false;
  }

  const me = state?.players?.[playerKey];
  const inference = buildInference(state, playerKey, P);
  const ctx = buildContext(state, playerKey, deps, P, inference);
  const core = extractGoCore(state, playerKey, deps, P, inference, ctx);
  const hardStop = (reason) => {
    logGoDecision(core, inference, null, Number.NaN, Number.NaN, false, P, reason);
    return false;
  };
  // Priority contract: emergency/hard-stop guards always run before dynamic GO scoring.
  if (shouldStopEmergency(core, inference, P)) return hardStop("emergency_stop");
  const boardMonths = new Set(asList(state?.board).map((c) => safeNum(c?.month)));
  const handPotential = asList(me?.hand).reduce((n, c) => {
    const month = safeNum(c?.month);
    const comboChance = safeNum(deps?.ownComboOpportunityScore?.(state, playerKey, month), 0) > 0;
    return n + (boardMonths.has(month) || comboChance ? 1 : 0);
  }, 0);

  if (core.selfPi < core.minPi) return hardStop("min_pi_guard");
  if (
    safeNum(P.goHardNoPotentialStop, 1) > 0 &&
    handPotential <= 0 &&
    core.deckCount <= safeNum(P.goHardNoPotentialDeckCut, 6)
  ) {
    return hardStop("no_potential_guard");
  }
  if (
    safeNum(P.goHardLateLeadLockEnable, 1) > 0 &&
    core.deckCount <= safeNum(P.goHardLateLeadLockDeckCut, 4) &&
    core.lead >= safeNum(P.goHardLateLeadLockLeadCut, 7)
  ) {
    return hardStop("late_lead_lock_guard");
  }

  if (!core.desperate && core.oppCanStop && core.threat >= safeNum(P.goHardThreatCut, 0.92)) {
    return hardStop("opp_can_stop_threat_guard");
  }
  if (!core.desperate && safeNum(inference?.maxDanger) >= safeNum(P.goHardDangerCut, 0.72) && core.lead >= -1) {
    return hardStop("danger_guard");
  }
  if (
    safeNum(state?.players?.[otherPlayerKey(playerKey)]?.goCount) > 0 &&
    core.lead <= 0 &&
    core.threat >= 0.8 &&
    core.oneAwayProb >= 0.45
  ) {
    return hardStop("opp_go_pressure_guard");
  }

  const dynamicGo = decideGoDynamic(core, inference, P);
  if (dynamicGo?.hardStopReason) {
    return hardStop(`dynamic_${String(dynamicGo.hardStopReason).toLowerCase()}`);
  }
  const finalValue = safeNum(dynamicGo?.expectedGoValue, 0);
  const finalThreshold = safeNum(dynamicGo?.threshold, 0);
  const margin = Math.max(0, safeNum(P.goDecisionMargin, 0.18));
  if (Math.abs(finalValue - finalThreshold) <= margin) {
    logAmbiguousGoDecision(core, inference, dynamicGo, finalValue, finalThreshold, P);
    logGoDecision(core, inference, dynamicGo, finalValue, finalThreshold, false, P, "ambiguous_margin_stop");
    return false;
  }
  const decision = finalValue > finalThreshold;
  logGoDecision(core, inference, dynamicGo, finalValue, finalThreshold, decision, P, "dynamic_threshold");
  return decision;
}

export function selectBombMonthAntiCL(state, _playerKey, months, deps, params = DEFAULT_PARAMS) {
  const P = { ...DEFAULT_PARAMS, ...(params || {}) };
  const list = asList(months);
  if (list.length <= 0) return null;
  let best = null;
  for (const month of list) {
    const score =
      safeNum(deps?.monthBoardGain?.(state, month), 0) +
      (safeNum(deps?.isHighImpactBomb?.(state, _playerKey, month)?.highImpact) > 0 ? 1.0 : 0);
    if (!best || score > best.score) best = { month, score };
  }
  if (best && best.score >= safeNum(P.bombBoardGainMin, 0.8)) return best.month;
  return best?.month ?? null;
}

export function shouldBombAntiCL(state, playerKey, bombMonths, deps, params = DEFAULT_PARAMS) {
  const P = { ...DEFAULT_PARAMS, ...(params || {}) };
  const list = asList(bombMonths);
  if (list.length <= 0) return false;
  const me = state?.players?.[playerKey];
  const opp = state?.players?.[otherPlayerKey(playerKey)];
  const scoreDiff = safeNum(me?.score) - safeNum(opp?.score);
  const oppPi = asList(opp?.captured?.junk).length;
  const inference = buildInference(state, playerKey, P);

  if (
    oppPi >= safeNum(P.bombOppPiWindowLow, 6) &&
    oppPi <= safeNum(P.bombOppPiWindowHigh, 7)
  ) {
    return true;
  }
  if (scoreDiff > 0 && safeNum(inference?.maxDanger) >= safeNum(P.bombHighThreatBlock, 0.9)) {
    return false;
  }

  if (scoreDiff <= safeNum(P.bombTrailEnableScoreDiff, -1)) return true;
  const pick = selectBombMonthAntiCL(state, playerKey, list, deps, P);
  if (pick == null) return false;
  const gain = safeNum(deps?.monthBoardGain?.(state, pick), 0);
  const highImpact = Boolean(deps?.isHighImpactBomb?.(state, playerKey, pick)?.highImpact);
  return highImpact || gain >= safeNum(P.bombBoardGainMin, 0.8);
}

export function decideShakingAntiCL(state, playerKey, shakingMonths, deps, params = DEFAULT_PARAMS) {
  const P = { ...DEFAULT_PARAMS, ...(params || {}) };
  const list = asList(shakingMonths);
  if (list.length <= 0) return { allow: false, month: null };
  const me = state?.players?.[playerKey];
  const opp = state?.players?.[otherPlayerKey(playerKey)];
  const inference = buildInference(state, playerKey, P);

  const trailing = safeNum(me?.score) < safeNum(opp?.score);
  const oppKwang = asList(opp?.captured?.kwang).length;
  const highRiskMonth = safeNum(P.shakingHighRiskMonth, 12);

  if (
    safeNum(P.shakingStopLossEnableTrail, 1) > 0 &&
    trailing &&
    safeNum(inference?.maxDanger) >= safeNum(P.shakingStopLossDangerCut, 0.8)
  ) {
    return { allow: false, month: null };
  }
  if (
    oppKwang >= safeNum(P.shakingOppKwangCut, 2) &&
    safeNum(inference?.probs?.[highRiskMonth]) >= safeNum(P.shakingHighRiskProbCut, 0.5)
  ) {
    return { allow: false, month: null };
  }

  let best = null;
  for (const month of list) {
    const immediate = safeNum(deps?.shakingImmediateGainScore?.(state, playerKey, month), 0);
    const monthRisk = safeNum(inference?.dangerLevel?.[month], 0);
    const probRisk = safeNum(inference?.probs?.[month], 0);
    const score = immediate - monthRisk * 1.2 - probRisk * 0.8;
    if (!best || score > best.score) best = { month, score };
  }
  return { allow: true, month: best?.month ?? list[0] };
}

export function shouldPresidentStopAntiCL(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const P = { ...DEFAULT_PARAMS, ...(params || {}) };
  const me = state?.players?.[playerKey];
  const opp = state?.players?.[otherPlayerKey(playerKey)];
  const lead = safeNum(me?.score) - safeNum(opp?.score);
  const deckCount = safeNum(state?.deckCount, asList(state?.deck).length);
  const carry = Math.max(0, safeNum(me?.score) - 6);
  if (lead >= safeNum(P.presidentStopLead, 1)) return true;
  if (deckCount <= safeNum(P.presidentLateDeckCut, 7) && lead > 0) return true;

  const inference = buildInference(state, playerKey, P);
  if (safeNum(inference?.maxDanger) >= safeNum(P.presidentStopThreatCut, 0.65)) return true;
  if (
    carry <= safeNum(P.presidentCarryStopMax, 1) &&
    safeNum(inference?.maxDanger) >= safeNum(P.presidentStopThreatCut, 0.65) * 0.85
  ) {
    return true;
  }
  return false;
}

export function chooseGukjinHeuristicAntiCL(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const P = { ...DEFAULT_PARAMS, ...(params || {}) };
  const me = state?.players?.[playerKey];
  const fiveCount = asList(me?.captured?.five).length;
  const opp = state?.players?.[otherPlayerKey(playerKey)];
  const trailing = safeNum(me?.score) < safeNum(opp?.score);
  const inference = buildInference(state, playerKey, P);
  if (trailing && safeNum(inference?.maxDanger) >= 0.7) return "junk";
  if (fiveCount >= safeNum(P.gukjinFiveCut, 4)) return "five";
  return "junk";
}

// Backward-compatible aliases expected by heuristicPolicyEngine.
export {
  rankHandCardsAntiCL as rankHandCards,
  chooseMatchHeuristicAntiCL as chooseMatch,
  shouldGoAntiCL as shouldGo,
  selectBombMonthAntiCL as selectBombMonth,
  shouldBombAntiCL as shouldBomb,
  decideShakingAntiCL as decideShaking,
  shouldPresidentStopAntiCL as shouldPresidentStop,
  chooseGukjinHeuristicAntiCL as chooseGukjin
};
