// heuristicGPT.js - Matgo Heuristic GPT (rule-tree v1)
import { shouldGoCL as shouldGoCLBase } from "./heuristicCL.js";
/* ============================================================================
 * Heuristic GPT (no rollout)
 * - Layer 1: hard safety gates
 * - Layer 2: risk-adjusted utility
 * - Layer 3: deterministic decision tree
 * - exported decisions: rank / match / go / bomb / shaking / president / gukjin
 * ========================================================================== */

const GUKJIN_CARD_ID = "I0";
const DOUBLE_PI_MONTHS = Object.freeze([11, 12, 13]);
const TRACE_STORE = new Map();
const CORE_GO_FEATURE_KEYS = Object.freeze([
  "myScore",
  "oppScore",
  "lead",
  "selfPi",
  "oppPi",
  "selfFive",
  "oppFive",
  "goCount",
  "carry",
  "deckCount",
  "threat",
  "oneAwayProb",
  "selfJokboTotal",
  "selfJokboOneAway",
  "oppJokboTotal",
  "oppJokboOneAway",
  "secondMover",
  "desperate"
]);

// Tunable parameter set consumed by runtime + optimizer scripts.
export const DEFAULT_PARAMS = {
  // phase

  // profile weights
  attackBase: 1.0,
  defenseBase: 1.0,
  riskBase: 1.0,
  tempoBase: 1.0,
  secondMoverTempoBoost: 0.05,

  // card utility
  noMatchBase: -4.0792449011398091,
  kwangCaptureBonus: 4.4,
  fiveCaptureBonus: 3.2,
  ribbonCaptureBonus: 1.4,
  selfPiWindowMul: 1.4,
  oppPiWindowMul: 1.2,
  doublePiBonus: 3.2,
  blockNoMatchPenalty: 2.0,
  firstTurnPlanBonus: 3.5,
  knownMonthSafeBonus: 1.2,
  unknownMonthPenalty: 1.0,
  trailPiTempoMul: 0.9,
  leadNoMatchTempoPenalty: 1.1,
  endgameSafeDiscardBonus: 0.8,
  endgameUnknownPenalty: 1.2,
  releaseRiskFloor: 0.55,
  pukOpportunityMul: 1.0,
  doublePiNoMatchHoldPenalty: 2.6,
  doublePiMonthPairHoldPenalty: 1.8,
  doublePiMonthTripleHoldPenalty: 1.3,
  phaseMidPiMul: 1.0,
  phaseEarlyTempoMul: 1.08,
  phaseMidTempoMul: 1.0,
  phaseLateTempoMul: 0.96,
  phaseEndTempoMul: 0.92,
  phaseEarlyRiskMul: 0.94,
  phaseMidRiskMul: 1.0,
  phaseEarlyBlockMul: 0.95,
  phaseMidBlockMul: 1.0,

  // choose-match
  chooseMatchBaseMul: 1.0,
  chooseMatchPiMul: 4.0,
  chooseMatchKwangBonus: 4.5,
  chooseMatchFiveBonus: 3.0,
  chooseMatchRibbonBonus: 1.6,
  chooseMatchRibbonVsJunkBonus: 0.8,
  chooseMatchJunkVsRibbonPenalty: 0.0,
  chooseMatchBlockMul: 1.7,
  chooseMatchComboMul: 2.4,
  comboFinishBirds: 28.0,
  comboFinishRed: 25.0,
  comboFinishBlue: 25.0,
  comboFinishPlain: 24.0,
  comboFinishKwang: 30.0,
  ribbonFourBonus: 30.0,
  fiveFourBonus: 32.0,
  handMongBakFiveBonus: 26.0,
  handMongBakPiPenalty: 7.0,
  discardLivePiPenalty: 18.0,
  discardLivePiPenaltyLate: 28.0,
  discardDoublePiLivePenalty: 14.0,
  discardDoublePiLivePenaltyLate: 22.0,
  discardDeadDoublePiBonus: 6.0,
  discardComboHoldPenalty: 26.0,
  discardComboHoldPenaltyLate: 38.0,
  discardOneAwayPenalty: 24.0,
  discardOneAwayPenaltyLate: 36.0,
  discardBlockPenalty: 14.0,
  discardBlockPenaltyLate: 22.0,
  discardBonusPiBonus: 16.0,
  lockedMonthPenalty: 4.0,
  secondMoverBlockBonus: 2.2,
  secondMoverPiBonus: 1.4,
  tieScoreEpsilon: 0.000001,
  ribbonComboBuildRedBonus: 0.6,
  ribbonComboBuildBlueBonus: 0.5,
  ribbonComboBuildPlainBonus: 0.4,
  playMatchKnownMonthMul: 0.0,
  playRibbonMatchBonus: 0.0,
  playRibbonMatchWhenJunkAvailableBonus: 0.0,
  playJunkMatchWhenRibbonAvailablePenalty: 0.0,
  playJunkOnlyCapturePenalty: 0.0,
  playNoMatchWhenAnyMatchPenalty: 0.0,
  playSafeNoMatchWhenAnyMatchPenalty: 0.0,
  playNoMatchJunkWhenAnyMatchPenalty: 0.0,
  playNoMatchDoublePiWhenAnyMatchPenalty: 0.0,

  // go model (hard gates)
  goHardThreatCut: 1.08,
  goHardOppLeadGrace: 1,
  goHardGoCountThreatCut: 0.72,
  goDesperateThreatCap: 0.62,
  goDesperateOneAwayCap: 48,
  goHardRiskOppHighPiPenalty: 0.08,
  goHardRiskLeadSafetyRelief: 0.12,

  // go model (utility)
  goUpsideTrailBonus: 0.12,
  goRiskGoCountMul: 0.09,
  goRiskLateDeckBonus: 0.1,
  stopLeadMul: 0.07,
  stopCarryMul: 0.1,
  stopTenBonus: 0.16,
  goCoreThreatClamp: 1.8,
  goCoreOneAwayClamp: 100,

  // go model (threshold)
  goBaseThreshold: -0.125,
  goSecondTrailBonus: 0.04,
  goRallyPiWindowBonus: 0.02,
  goRallySecondBonus: 0.01,
  goRallyTrailBonus: 0.02,
  goRallyEndDeckBonus: 0.0,
  goSoftHighPiThreatCap: 0.9,
  goSoftHighPiOneAwayCap: 66,
  goSoftHighPiMargin: 0.03,
  goSoftTrailHighPiMargin: 0.01,
  goSoftValueMargin: 0.0,
  goSecondChaseThresholdRelief: 0.08,
  goSecondChaseNearMargin: 0.06,
  goSecondChasePiMargin: 1,
  goSecondChaseThreatCap: 0.82,
  goSecondChaseOneAwayCap: 70,
  goSecondChaseSwingCap: 0.52,
  goSecondChaseHardRiskRelaxMargin: 0.08,

  // go hard safe-stop + light penalties
  goHardSafeStopEnabled: 1,
  goHardSafeStopMinScore: 7,
  goHardSafeStopDeckCut: 9,
  goHardSafeStopLeadMin: 1,
  goLiteScoreDiffMul: 0.04,
  goLiteSelfCanStopPenalty: 0.01,
  goLiteSafeAttackThreatCap: 0.58,
  goLiteSafeAttackOneAwayCap: 45,
  goLiteSafeAttackDeckMin: 5,
  goSafeGoSwingProbCut: 0.05,

  // optimizer compatibility knobs
  goScoreDiffBonus: 0.01,
  goDeckLowBonus: 0.02,
  goUnseeHighPiPenalty: 0.04,

  // bomb
  bombImmediateMul: 0.9,
  bombBoardGainMul: 0.8,
  bombHighImpactBonus: 2.8,
  bombTrailBonus: 1.0,
  bombRiskMul: 1.0,
  bombThreshold: 3.0,
  bombDefenseThreshold: 4.5,

  // shaking
  shakeImpactBonus: 0.65,
  shakePiLineBonus: 0.4,
  shakeDirectGwangBonus: 0.4,
  shakeKnownLowBonus: 0.22,
  shakeKnownHighPenalty: 0.18,
  shakeTrailingBonus: 0.2,
  shakeFirstPlanBonus: 0.25,

  // president and gukjin
  presidentStopLead: 3,
  presidentCarryStopMax: 1,
  presidentThreatStop: 1.0,
  gukjinScoreDiffMul: 1.0,
  gukjinPiDiffMul: 0.22,
  gukjinMongBakBonus: 1.8,
  gukjinMongRiskPenalty: 2.2,

  // trial 114 tuned params
  blockBase: 3.6389093404465638,
  blockThreatMul: 2.925693242333967,
  blockUrgencyMul: 1.6657156721216619,
  captureGainMul: 0.8685039095454038,
  comboOpportunityMul: 5.2576241793437317,
  dangerMatchMul: 1.254023424815037,
  dangerNoMatchMul: 0.60006369148644045,
  desperateAttackBoost: 0.18574008728141175,
  desperateRiskDown: 0.18141389141866276,
  desperateTempoBoost: 0.0300786761264036,
  feedRiskMatchMul: 1.7141622867335182,
  feedRiskNoMatchMul: 1.1521460372219654,
  goHardGoCountCap: 4,
  goHardJokboOneAwayCountCut: 1,
  goHardJokboOneAwayCut: 68,
  goHardJokboOneAwaySwingCut: 0.45964118755905231,
  goHardLateOneAwayCut: 54,
  goHardOppScoreCut: 8,
  goHardRiskOneAwayMul: 0.67869959500383537,
  goHardRiskOppStopPenalty: 0.43208757051963176,
  goHardRiskThreatMul: 0.21781498975205768,
  goHardRiskThreshold: 0.66,
  goHardThreatDeckCut: 10,
  goLiteLatePenalty: 0.11,
  goLiteOneAwayPenaltyMul: 0.11608197553448872,
  goLiteOppCanStopPenalty: 0.22,
  goLiteSafeAttackBonus: 0.15494875441940156,
  goLiteThreatPenaltyMul: 0.13560896875251968,
  goLookaheadThresholdMul: 0.059459872467595147,
  goMinPi: 5,
  goMinPiDesperate: 5,
  goMinPiSecondTrailingDelta: 2,
  goRiskOneAwayMul: 0.17410107037517836,
  goRiskOppJokboMul: 0.34749337695693389,
  goRiskOppOneAwayMul: 0.13405521881051077,
  goRiskPressureMul: 0.37751193247462522,
  goSafeGoBonus: 0.26164359701018652,
  goThresholdLeadUp: 0.14,
  goThresholdPressureUp: 0.0023769481568353054,
  goThresholdTrailDown: 0.135,
  goUpsideOneAwayMul: 0.11046416616873096,
  goUpsidePiMul: 0.068238432502096907,
  goUpsideScoreMul: 0.072204438069289792,
  goUpsideSelfJokboMul: 0.8700789315322861,
  goUtilityRiskWeight: 1.3197840792723667,
  goUtilityStopWeight: 0.95646418729412319,
  goUtilityThresholdWeight: 1.5189777909051363,
  goUtilityUpsideWeight: 1.7797317565163624,
  highPressureDefenseBoost: 0.20342301221416864,
  highPressureRiskBoost: 0.36360774045397382,
  junkPiMul: 4.1130591120449651,
  leadingDefenseBoost: 0.16160416259015861,
  leadingRiskBoost: 0.20077336223817188,
  matchOneBase: 3.750945296601202,
  matchThreeBase: 10.228089029646991,
  matchTwoBase: 7.7121591289308578,
  phaseEarlyDeck: 14,
  phaseEarlyPiMul: 1.3153566274342017,
  phaseEndBlockMul: 1.7398642134212745,
  phaseEndDeck: 4,
  phaseEndPiMul: 0.54813788557403864,
  phaseEndRiskMul: 1.2088700937680157,
  phaseLateBlockMul: 2.0967140913210622,
  phaseLateDeck: 6,
  phaseLatePiMul: 0.677089810720508,
  phaseLateRiskMul: 1.9490415108333872,
  pukRiskMul: 2.2761449473327269,
  rankReleaseRiskPenaltyMul: 32.151302962826065,
  releaseRiskMul: 1.6777627628700409,
  releaseRiskOppScoreExpMul: 0.44282546439954162,
  selfPukRiskExpMul: 0.1277655427730493,
  selfPukRiskMul: 0.51006792818624569,
  selfPukRiskPatternBonus: 0.32796251692829625,
  shakeComboMul: 0.506422480544727,
  shakeImmediateMul: 1.7635366633648264,
  shakeLeadThresholdUp: 0.16094138990062612,
  shakePressureThresholdUp: 0.015514153919548149,
  shakeRiskMul: 0.12869190687210985,
  shakeThreshold: 0.72983558045318309,
  trailingAttackBoost: 0.22829206981691025,
  trailingTempoBoost: 0.16126360939155543,

  // structural v3
  seatSecondAttackDown: 0.12,
  seatSecondDefenseUp: 0.22,
  seatSecondRiskUp: 0.22,
  seatSecondTempoDown: 0.08,
  seatFirstTrailAttackUp: 0.06,
  cardSecondBlockThreatBonus: 1.2,
  cardSecondReleaseRiskMul: 1.35,
  cardSecondNoMatchRiskMul: 0.2,
  cardFirstTrailTempoBonus: 0.35,
  goModeSafeThresholdUp: 0.06,
  goModeChaseThresholdDown: 0.04,
  goHardSecondChaseBlockEnabled: 1,
  goHardSecondChaseThreatCut: 0.8,
  goHardSecondChaseLateDeckCut: 7,
  goHardCarryOppStopCarryMin: 2,
  goHardCarryOppStopThreatCut: 0.85,
  goHardCarryOppStopLateDeckCut: 6,

  // decision trace
  decisionTraceEnabled: 0,
  decisionTraceLimit: 80
};

// ---------------------------------------------------------------------------
// Primitive helpers
// ---------------------------------------------------------------------------
function safeNum(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function mergedParams(params) {
  return { ...DEFAULT_PARAMS, ...(params || {}) };
}

// ---------------------------------------------------------------------------
// Decision trace helpers (debug / explainability)
// ---------------------------------------------------------------------------
function shouldTrace(P, deps) {
  if (safeNum(P?.decisionTraceEnabled, 0) > 0) return true;
  return typeof deps?.onHeuristicGPTTrace === "function";
}

function traceKey(decisionType, playerKey) {
  return `${String(decisionType || "unknown")}:${String(playerKey || "unknown")}`;
}

function beginTrace(decisionType, playerKey, P, deps) {
  if (!shouldTrace(P, deps)) return null;
  return {
    decisionType,
    playerKey,
    at: Date.now(),
    layers: []
  };
}

function traceLayer(trace, layer, detail) {
  if (!trace) return;
  trace.layers.push({ layer, ...(detail || {}) });
}

function pushTrace(trace, decision, reason, P, deps) {
  if (!trace) return decision;
  trace.decision = Boolean(decision);
  trace.reason = String(reason || "");

  const limit = Math.max(1, Math.floor(safeNum(P?.decisionTraceLimit, 80)));
  const key = traceKey(trace.decisionType, trace.playerKey);
  const list = TRACE_STORE.get(key) || [];
  list.push(trace);
  if (list.length > limit) {
    list.splice(0, list.length - limit);
  }
  TRACE_STORE.set(key, list);

  if (typeof deps?.onHeuristicGPTTrace === "function") {
    deps.onHeuristicGPTTrace(trace);
  }
  return decision;
}

// ---------------------------------------------------------------------------
// State adapters and card-map utilities
// ---------------------------------------------------------------------------
function otherPlayerKey(deps, playerKey) {
  if (typeof deps?.otherPlayerKey === "function") return deps.otherPlayerKey(playerKey);
  return playerKey === "human" ? "ai" : "human";
}

function asMonthMap(value) {
  if (value instanceof Map) return value;
  const out = new Map();
  if (!value || typeof value !== "object") return out;
  for (const [k, v] of Object.entries(value)) {
    const month = Number(k);
    if (!Number.isInteger(month) || month < 1 || month > 12) continue;
    out.set(month, safeNum(v));
  }
  return out;
}

function asMonthSet(value) {
  if (value instanceof Set) return value;
  const out = new Set();
  if (Array.isArray(value)) {
    for (const x of value) {
      const m = Number(x);
      if (Number.isInteger(m) && m >= 1 && m <= 12) out.add(m);
    }
  }
  return out;
}

function countByMonth(cards) {
  const out = new Map();
  for (const card of cards || []) {
    const month = Number(card?.month);
    if (!Number.isInteger(month) || month < 1 || month > 12) continue;
    out.set(month, safeNum(out.get(month)) + 1);
  }
  return out;
}

function hasCardId(cards, id) {
  return (cards || []).some((card) => card?.id === id);
}

function hasGukjinInCaptured(captured) {
  if (!captured) return false;
  for (const category of ["kwang", "five", "ribbon", "junk"]) {
    if (hasCardId(captured?.[category] || [], GUKJIN_CARD_ID)) return true;
  }
  return false;
}

function rawFiveCountIncludingCapturedGukjin(player) {
  const fives = player?.captured?.five || [];
  const fiveCount = fives.length;
  if (!player?.captured) return fiveCount;
  const hasPendingGukjinAsFive = fives.some(
    (card) => card?.id === GUKJIN_CARD_ID && !card?.gukjinTransformed
  );
  if (hasPendingGukjinAsFive) return Math.max(0, fiveCount - 1);
  if (hasCardId(fives, GUKJIN_CARD_ID)) return fiveCount;
  return hasGukjinInCaptured(player.captured) ? fiveCount + 1 : fiveCount;
}

function boardByMonth(state, deps) {
  const raw = typeof deps?.boardMatchesByMonth === "function" ? deps.boardMatchesByMonth(state) : null;
  if (raw instanceof Map) return raw;
  const out = new Map();
  for (const card of state?.board || []) {
    const month = Number(card?.month);
    if (!Number.isInteger(month) || month < 1 || month > 12) continue;
    if (!out.has(month)) out.set(month, []);
    out.get(month).push(card);
  }
  return out;
}

function monthCountsWithDeps(deps, cards) {
  const raw = typeof deps?.monthCounts === "function" ? deps.monthCounts(cards || []) : null;
  if (raw instanceof Map) return raw;
  return countByMonth(cards || []);
}

function capturedMonthCountsWithDeps(state, deps) {
  const raw = typeof deps?.capturedMonthCounts === "function" ? deps.capturedMonthCounts(state) : null;
  if (raw instanceof Map) return raw;
  return new Map();
}

// ---------------------------------------------------------------------------
// Threat/profile construction
// ---------------------------------------------------------------------------
function classifyPhase(deckCount, P) {
  if (deckCount <= P.phaseEndDeck) return "end";
  if (deckCount <= P.phaseLateDeck) return "late";
  if (deckCount >= P.phaseEarlyDeck) return "early";
  return "mid";
}

function isSecondMover(state, playerKey) {
  const first = state?.startingTurnKey;
  if (first !== "human" && first !== "ai") return false;
  return first !== playerKey;
}

function piValue(card, deps) {
  if (!card) return 0;
  if (card.id === GUKJIN_CARD_ID) return 2;
  if (card.category !== "junk") return 0;
  if (typeof deps?.junkPiValue === "function") {
    const v = safeNum(deps.junkPiValue(card), 1);
    return Math.max(0, v);
  }
  const fallback = safeNum(card?.piValue, 1);
  return Math.max(0, fallback);
}

function isDoublePi(card, deps) {
  return piValue(card, deps) >= 2;
}

function categoryCaptureBonus(cards, P) {
  let bonus = 0;
  for (const card of cards || []) {
    if (card?.category === "kwang") bonus += P.kwangCaptureBonus;
    else if (card?.category === "five") bonus += P.fiveCaptureBonus;
    else if (card?.category === "ribbon") bonus += P.ribbonCaptureBonus;
  }
  return bonus;
}

function hasComboTag(card, tag) {
  return Array.isArray(card?.comboTags) && card.comboTags.includes(tag);
}

function countComboTag(cards, tag) {
  return (cards || []).reduce((acc, card) => acc + (hasComboTag(card, tag) ? 1 : 0), 0);
}

function comboCounts(player) {
  return {
    red: countComboTag(player?.captured?.ribbon || [], "redRibbons"),
    blue: countComboTag(player?.captured?.ribbon || [], "blueRibbons"),
    plain: countComboTag(player?.captured?.ribbon || [], "plainRibbons"),
    birds: countComboTag(player?.captured?.five || [], "fiveBirds"),
    kwang: (player?.captured?.kwang || []).length
  };
}

function ownComboFinishBonus(combo, cards, P) {
  let bonus = 0;
  if (safeNum(combo?.birds) >= 2 && (cards || []).some((card) => hasComboTag(card, "fiveBirds"))) {
    bonus += safeNum(P?.comboFinishBirds, 0);
  }
  if (safeNum(combo?.red) >= 2 && (cards || []).some((card) => hasComboTag(card, "redRibbons"))) {
    bonus += safeNum(P?.comboFinishRed, 0);
  }
  if (safeNum(combo?.blue) >= 2 && (cards || []).some((card) => hasComboTag(card, "blueRibbons"))) {
    bonus += safeNum(P?.comboFinishBlue, 0);
  }
  if (safeNum(combo?.plain) >= 2 && (cards || []).some((card) => hasComboTag(card, "plainRibbons"))) {
    bonus += safeNum(P?.comboFinishPlain, 0);
  }
  if (safeNum(combo?.kwang) >= 2 && (cards || []).some((card) => card?.category === "kwang")) {
    bonus += safeNum(P?.comboFinishKwang, 0);
  }
  return bonus;
}

function ribbonComboBuildBonus(combo, cards, P) {
  let bonus = 0;
  if (safeNum(combo?.red) === 1 && (cards || []).some((card) => hasComboTag(card, "redRibbons"))) {
    bonus += safeNum(P?.ribbonComboBuildRedBonus, 0);
  }
  if (safeNum(combo?.blue) === 1 && (cards || []).some((card) => hasComboTag(card, "blueRibbons"))) {
    bonus += safeNum(P?.ribbonComboBuildBlueBonus, 0);
  }
  if (safeNum(combo?.plain) === 1 && (cards || []).some((card) => hasComboTag(card, "plainRibbons"))) {
    bonus += safeNum(P?.ribbonComboBuildPlainBonus, 0);
  }
  return bonus;
}

function getLiveDoublePiMonths(state, deps) {
  const capturedMonths = new Set();
  for (const key of ["human", "ai"]) {
    for (const card of state?.players?.[key]?.captured?.junk || []) {
      const month = Number(card?.month);
      if (DOUBLE_PI_MONTHS.includes(month) && isDoublePi(card, deps)) {
        capturedMonths.add(month);
      }
    }
  }
  return new Set(DOUBLE_PI_MONTHS.filter((month) => !capturedMonths.has(month)));
}

function getComboHoldMonths(state, playerKey, deps) {
  const selfPlayer = state?.players?.[playerKey];
  const oppPlayer = state?.players?.[otherPlayerKey(deps, playerKey)];
  const hold = new Set();
  const addMonths = (value) => {
    for (const month of asMonthSet(value)) hold.add(month);
  };
  if (typeof deps?.blockingMonthsAgainst === "function") {
    addMonths(deps.blockingMonthsAgainst(selfPlayer, oppPlayer));
    addMonths(deps.blockingMonthsAgainst(oppPlayer, selfPlayer));
  }
  return hold;
}

function discardTieOrder(card, deps, livePi) {
  if (card?.bonus?.stealPi) return 6;
  if (isDoublePi(card, deps) && livePi) return 1;
  return { five: 5, ribbon: 4, kwang: 3 }[String(card?.category || "")] ?? 2;
}

function analyzePresidentHandPressure(player, deps) {
  let bonusCount = 0;
  let doublePiCount = 0;
  for (const card of player?.hand || []) {
    if (Number(card?.bonus?.stealPi || 0) > 0) bonusCount += 1;
    else if (isDoublePi(card, deps)) doublePiCount += 1;
  }
  return {
    bonusCount,
    doublePiCount,
    weightedPressure: bonusCount * 2 + doublePiCount
  };
}

function countAlternativeMatchCards(state, playerKey, excludedMonths, deps) {
  const excluded = asMonthSet(excludedMonths);
  const byMonth = boardByMonth(state, deps);
  let count = 0;
  for (const card of state?.players?.[playerKey]?.hand || []) {
    const month = Number(card?.month);
    if (excluded.has(month)) continue;
    if ((byMonth.get(month) || []).length > 0) count += 1;
  }
  return count;
}

function summarizeBombMonth(state, playerKey, month, deps) {
  const player = state?.players?.[playerKey];
  const monthHand = (player?.hand || []).filter((card) => Number(card?.month) === Number(month));
  const monthBoard = (state?.board || []).filter((card) => Number(card?.month) === Number(month));
  const all = monthHand.concat(monthBoard);
  return {
    bonusSteal: all.reduce((sum, card) => sum + Number(card?.bonus?.stealPi || 0), 0),
    doublePiCount: all.reduce((sum, card) => sum + (isDoublePi(card, deps) ? 1 : 0), 0),
    highValueCount: all.reduce(
      (sum, card) => sum + (card?.category === "kwang" || card?.category === "five" ? 1 : 0),
      0
    )
  };
}

function collectThreatSignals(state, playerKey, deps, P) {
  const jokbo = typeof deps?.checkOpponentJokboProgress === "function"
    ? deps.checkOpponentJokboProgress(state, playerKey)
    : null;
  const jokboThreat = safeNum(jokbo?.threat);
  const progThreat = typeof deps?.opponentThreatScore === "function"
    ? safeNum(deps.opponentThreatScore(state, playerKey))
    : 0;
  const nextThreat = typeof deps?.nextTurnThreatScore === "function"
    ? safeNum(deps.nextTurnThreatScore(state, playerKey))
    : 0;
  const oppKey = otherPlayerKey(deps, playerKey);
  const boardThreat = typeof deps?.boardHighValueThreatForPlayer === "function"
    ? safeNum(deps.boardHighValueThreatForPlayer(state, oppKey))
    : 0;
  const matchable = typeof deps?.matchableMonthCountForPlayer === "function"
    ? safeNum(deps.matchableMonthCountForPlayer(state, oppKey))
    : 0;
  const deckCount = safeNum(state?.deck?.length);

  const threat = clamp(
    progThreat * 0.4 + jokboThreat * 0.3 + nextThreat * 0.2 + boardThreat * 0.08 + matchable * 0.05,
    0,
    1.8
  );

  let oneAwayProb = progThreat * 34 + jokboThreat * 30 + nextThreat * 20 + boardThreat * 8 + matchable * 4;
  if (deckCount <= P.phaseLateDeck) oneAwayProb += 8;
  if (deckCount <= P.phaseEndDeck) oneAwayProb += 8;
  oneAwayProb = clamp(oneAwayProb, 0, 100);

  return {
    threat,
    oneAwayProb,
    jokboThreat,
    monthUrgency: asMonthMap(jokbo?.monthUrgency),
    deckCount
  };
}

function buildProfile(state, playerKey, deps, P) {
  const ctx = typeof deps?.analyzeGameContext === "function"
    ? deps.analyzeGameContext(state, playerKey)
    : {};
  const myScore = safeNum(ctx?.myScore);
  const oppScore = safeNum(ctx?.oppScore);
  const lead = myScore - oppScore;
  const trailing = lead < 0;
  const leading = lead > 0;
  const second = isSecondMover(state, playerKey);
  const signals = collectThreatSignals(state, playerKey, deps, P);
  const phase = classifyPhase(signals.deckCount, P);
  const goldRisk = typeof deps?.goldRiskProfile === "function"
    ? (deps.goldRiskProfile(state, playerKey) || { selfLow: false, oppLow: false })
    : { selfLow: false, oppLow: false };
  const desperate = Boolean(goldRisk?.selfLow && !goldRisk?.oppLow);

  let aggression = P.attackBase;
  let defense = P.defenseBase;
  let risk = P.riskBase;
  let tempo = P.tempoBase;

  if (trailing) {
    aggression += P.trailingAttackBoost;
    tempo += P.trailingTempoBoost;
  }
  if (leading) {
    defense += P.leadingDefenseBoost;
    risk += P.leadingRiskBoost;
  }
  if (signals.threat >= 0.72) {
    defense += P.highPressureDefenseBoost;
    risk += P.highPressureRiskBoost;
  }
  if (second) tempo += P.secondMoverTempoBoost;
  if (phase === "end") {
    defense += 0.08;
    risk += 0.1;
  }
  if (desperate) {
    aggression += P.desperateAttackBoost;
    tempo += P.desperateTempoBoost;
    risk = Math.max(0.6, risk - P.desperateRiskDown);
  }

  // Structural split: second-mover and first-mover use different policy posture.
  if (second) {
    aggression = Math.max(0.5, aggression - safeNum(P.seatSecondAttackDown, 0.08));
    defense += safeNum(P.seatSecondDefenseUp, 0.16);
    risk += safeNum(P.seatSecondRiskUp, 0.14);
    if (!trailing && !desperate) {
      tempo = Math.max(0.6, tempo - safeNum(P.seatSecondTempoDown, 0.05));
    }
  } else if (trailing) {
    aggression += safeNum(P.seatFirstTrailAttackUp, 0.06);
  }

  return {
    ctx,
    oppKey: otherPlayerKey(deps, playerKey),
    myScore,
    oppScore,
    lead,
    trailing,
    leading,
    second,
    seatPolicy: second ? "second" : "first",
    phase,
    desperate,
    signals,
    aggression,
    defense,
    risk,
    tempo
  };
}

// ---------------------------------------------------------------------------
// Rank/choose helpers
// ---------------------------------------------------------------------------
function buildRankingCache(state, playerKey, deps, profile) {
  const player = state?.players?.[playerKey] || {};
  const board = boardByMonth(state, deps);
  const handCountByMonth = monthCountsWithDeps(deps, player?.hand || []);
  const capturedByMonth = capturedMonthCountsWithDeps(state, deps);
  const oppPlayer = state?.players?.[profile.oppKey];
  const blockMonths = asMonthSet(
    typeof deps?.blockingMonthsAgainst === "function"
      ? deps.blockingMonthsAgainst(oppPlayer, player)
      : null
  );
  const blockUrgency = asMonthMap(
    typeof deps?.blockingUrgencyByMonth === "function"
      ? deps.blockingUrgencyByMonth(oppPlayer, player)
      : null
  );

  const planRaw = typeof deps?.getFirstTurnDoublePiPlan === "function"
    ? deps.getFirstTurnDoublePiPlan(state, playerKey)
    : null;

  const firstTurnPlan = {
    active: Boolean(planRaw?.active),
    months: asMonthSet(planRaw?.months)
  };

  return {
    board,
    boardCountByMonth: monthCountsWithDeps(deps, state?.board || []),
    handCountByMonth,
    capturedByMonth,
    blockMonths,
    blockUrgency,
    firstTurnPlan,
    comboOpportunityByMonth: new Map(),
    oppImmediateGainByMonth: new Map(),
    dangerMonthRiskByMonth: new Map()
  };
}

function buildHandPackageContext(state, playerKey, deps, profile, cache) {
  const player = state?.players?.[playerKey] || {};
  const oppPlayer = state?.players?.[profile?.oppKey] || {};
  const selfPi = safeNum(profile?.ctx?.selfPi, safeNum(deps?.capturedCountByCategory?.(player, "junk")));
  const oppPi = safeNum(profile?.ctx?.oppPi, safeNum(deps?.capturedCountByCategory?.(oppPlayer, "junk")));
  const matchableCategoryCounts = { any: 0, kwang: 0, five: 0, ribbon: 0, junk: 0 };
  for (const card of player?.hand || []) {
    const month = Number(card?.month);
    const boardMatches = cache?.board?.get(month) || [];
    if (boardMatches.length <= 0) continue;
    matchableCategoryCounts.any += 1;
    const category = String(card?.category || "");
    if (Object.prototype.hasOwnProperty.call(matchableCategoryCounts, category)) {
      matchableCategoryCounts[category] += 1;
    }
  }
  return {
    player,
    oppPlayer,
    selfPi,
    oppPi,
    combo: comboCounts(player),
    ribbonCount: (player?.captured?.ribbon || []).length,
    fiveCount: (player?.captured?.five || []).length,
    mongBak: safeNum(profile?.ctx?.selfFive) <= 0 && safeNum(profile?.ctx?.oppFive) >= 7,
    liveDoublePiMonths: getLiveDoublePiMonths(state, deps),
    comboHoldMonths: getComboHoldMonths(state, playerKey, deps),
    matchableCategoryCounts,
    hasAnyMatchablePlay: matchableCategoryCounts.any > 0,
    cache
  };
}

function countKnownMonth(cache, month) {
  return (
    safeNum(cache?.boardCountByMonth?.get(month)) +
    safeNum(cache?.handCountByMonth?.get(month)) +
    safeNum(cache?.capturedByMonth?.get(month))
  );
}

function monthUrgencyInfo(month, profile, cache) {
  return {
    blockUrgency: safeNum(cache?.blockUrgency?.get(month)),
    monthUrgencyRaw: safeNum(profile?.signals?.monthUrgency?.get(month)),
    normalizedUrgency: Math.max(
      safeNum(cache?.blockUrgency?.get(month)),
      safeNum(profile?.signals?.monthUrgency?.get(month)) / 10
    )
  };
}

function capturePlanScore(state, playerKey, month, capturedCards, deps, P, profile, cache, handCtx) {
  const phaseScale = cardPhaseScales(profile.phase, P);
  const captureGain = (capturedCards || []).reduce((acc, card) => acc + safeNum(deps?.cardCaptureValue?.(card)), 0);
  const piGain = (capturedCards || []).reduce((acc, card) => acc + piValue(card, deps), 0);
  const monthPriority = safeNum(deps?.monthStrategicPriority?.(month));

  let comboOpportunity = cache?.comboOpportunityByMonth?.get(month);
  if (comboOpportunity == null) {
    comboOpportunity = safeNum(deps?.ownComboOpportunityScore?.(state, playerKey, month));
    cache?.comboOpportunityByMonth?.set(month, comboOpportunity);
  }

  const urgency = monthUrgencyInfo(month, profile, cache);

  let immediate = captureGain * safeNum(P?.captureGainMul, 1.0);
  immediate += categoryCaptureBonus(capturedCards, P);
  immediate += piGain * safeNum(P?.junkPiMul, 0) * phaseScale.piMul;
  if (handCtx.selfPi >= 7 && handCtx.selfPi <= 9) immediate += piGain * safeNum(P?.selfPiWindowMul, 0);
  if (handCtx.oppPi <= 5) immediate += piGain * safeNum(P?.oppPiWindowMul, 0);
  if ((capturedCards || []).some((card) => isDoublePi(card, deps))) immediate += safeNum(P?.doublePiBonus, 0);
  immediate += comboOpportunity * safeNum(P?.comboOpportunityMul, 0);
  immediate += ownComboFinishBonus(handCtx.combo, capturedCards, P);
  immediate += ribbonComboBuildBonus(handCtx.combo, capturedCards, P);
  if (handCtx.ribbonCount >= 4 && (capturedCards || []).some((card) => card?.category === "ribbon")) {
    immediate += safeNum(P?.ribbonFourBonus, 0);
  }
  if (handCtx.fiveCount >= 4 && (capturedCards || []).some((card) => card?.category === "five")) {
    immediate += safeNum(P?.fiveFourBonus, 0);
  }
  if (handCtx.mongBak) {
    if ((capturedCards || []).some((card) => card?.category === "five")) {
      immediate += safeNum(P?.handMongBakFiveBonus, 0);
    } else if (piGain > 0) {
      immediate -= safeNum(P?.handMongBakPiPenalty, 0);
    }
  }
  if (cache?.firstTurnPlan?.active && cache.firstTurnPlan.months.has(month)) {
    immediate += safeNum(P?.firstTurnPlanBonus, 0);
  }
  if (profile.trailing && piGain > 0) {
    immediate += piGain * safeNum(P?.trailPiTempoMul, 0);
  }
  if (profile.second && profile.trailing && piGain > 0) {
    immediate += piGain * safeNum(P?.secondMoverPiBonus, 0);
  }
  if ((capturedCards || []).length > 0 && (capturedCards || []).every((card) => card?.category === "junk")) {
    immediate -= safeNum(P?.playJunkOnlyCapturePenalty, 0);
  }
  immediate += monthPriority * 0.22;

  let block = 0;
  if (cache?.blockMonths?.has(month)) {
    block += safeNum(P?.blockBase, 0);
    block += urgency.normalizedUrgency * safeNum(P?.blockUrgencyMul, 0);
    block += safeNum(profile?.signals?.threat) * safeNum(P?.blockThreatMul, 0);
    if (profile.second) block += safeNum(P?.secondMoverBlockBonus, 0);
  }

  return { immediate, block, piGain, monthPriority, comboOpportunity };
}

function discardPlanScore(card, month, deps, P, profile, cache, handCtx) {
  const knownMonth = countKnownMonth(cache, month);
  const late = profile.phase === "late" || profile.phase === "end";
  const urgency = monthUrgencyInfo(month, profile, cache);
  const liveDoublePi = handCtx.liveDoublePiMonths.has(month);
  const sameMonth = safeNum(cache?.handCountByMonth?.get(month));

  let tempo = 0;
  if (knownMonth >= 3) tempo += safeNum(P?.knownMonthSafeBonus, 0);
  else if (knownMonth <= 1) tempo -= safeNum(P?.unknownMonthPenalty, 0);

  if (card?.bonus?.stealPi) tempo += safeNum(P?.discardBonusPiBonus, 0);

  if (liveDoublePi) {
    tempo -= late ? safeNum(P?.discardLivePiPenaltyLate, 0) : safeNum(P?.discardLivePiPenalty, 0);
  }
  if (isDoublePi(card, deps) && liveDoublePi) {
    tempo -= late ? safeNum(P?.discardDoublePiLivePenaltyLate, 0) : safeNum(P?.discardDoublePiLivePenalty, 0);
  } else if (isDoublePi(card, deps) && !liveDoublePi) {
    tempo += safeNum(P?.discardDeadDoublePiBonus, 0);
  }

  if (handCtx.comboHoldMonths.has(month)) {
    tempo -= late ? safeNum(P?.discardComboHoldPenaltyLate, 0) : safeNum(P?.discardComboHoldPenalty, 0);
  }
  if (urgency.blockUrgency >= 3 || urgency.monthUrgencyRaw >= 24) {
    tempo -= late ? safeNum(P?.discardOneAwayPenaltyLate, 0) : safeNum(P?.discardOneAwayPenalty, 0);
  } else if (urgency.blockUrgency >= 2 || urgency.monthUrgencyRaw >= 18) {
    tempo -= late ? safeNum(P?.discardBlockPenaltyLate, 0) : safeNum(P?.discardBlockPenalty, 0);
  }

  if (profile.leading) tempo -= safeNum(P?.leadNoMatchTempoPenalty, 0);
  if (profile.phase === "end") {
    if (knownMonth >= 3) tempo += safeNum(P?.endgameSafeDiscardBonus, 0);
    else tempo -= safeNum(P?.endgameUnknownPenalty, 0);
  }

  let holdPenalty = 0;
  if (isDoublePi(card, deps)) {
    holdPenalty += safeNum(P?.doublePiNoMatchHoldPenalty, 0) * 0.35;
    if (sameMonth >= 2) holdPenalty += safeNum(P?.doublePiMonthPairHoldPenalty, 0) * 0.35;
    if (sameMonth >= 3) holdPenalty += safeNum(P?.doublePiMonthTripleHoldPenalty, 0) * 0.35;
  }
  if (safeNum(cache?.capturedByMonth?.get(month)) >= 2 && knownMonth >= 3) {
    holdPenalty += safeNum(P?.lockedMonthPenalty, 0);
  }

  tempo += discardTieOrder(card, deps, liveDoublePi) * 1.8;
  return { tempo, holdPenalty, safeDiscardHint: knownMonth };
}

// Tie-break signature used when primary score is effectively equal.
function rankTieBreak(entry, P) {
  const category = String(entry?.card?.category || "");
  const categoryRank = category === "kwang" ? 3 : category === "five" ? 2 : category === "ribbon" ? 1 : 0;
  const monthPriority = safeNum(entry?.monthPriority);
  const piGain = safeNum(entry?.piGain);
  const matchCount = safeNum(entry?.matches);
  const safeDiscard = safeNum(entry?.safeDiscardHint);
  const releaseRisk = safeNum(entry?.releaseRisk);
  const id = String(entry?.card?.id || "");
  const releaseRiskPenaltyMul = Math.max(0, safeNum(P?.rankReleaseRiskPenaltyMul, 20.0));

  // Base rule: capture quality first, safety second, then deterministic id order.
  const signature =
    categoryRank * 1000000 +
    matchCount * 10000 +
    piGain * 1000 +
    monthPriority * 100 +
    safeDiscard * 10 -
    releaseRisk * releaseRiskPenaltyMul;
  return { signature, id };
}

function compareRankEntries(a, b, P) {
  const eps = Math.max(0, safeNum(P?.tieScoreEpsilon, 0.000001));
  const delta = safeNum(b?.score) - safeNum(a?.score);
  if (Math.abs(delta) > eps) return delta;

  const aa = rankTieBreak(a, P);
  const bb = rankTieBreak(b, P);
  if (bb.signature !== aa.signature) return bb.signature - aa.signature;
  return aa.id.localeCompare(bb.id);
}

function compareMatchChoice(a, b, P) {
  const eps = Math.max(0, safeNum(P?.tieScoreEpsilon, 0.000001));
  const delta = safeNum(b?.score) - safeNum(a?.score);
  if (Math.abs(delta) > eps) return delta;
  if (safeNum(b?.pi) !== safeNum(a?.pi)) return safeNum(b?.pi) - safeNum(a?.pi);
  if (safeNum(b?.monthPriority) !== safeNum(a?.monthPriority)) {
    return safeNum(b?.monthPriority) - safeNum(a?.monthPriority);
  }
  return String(a?.id || "").localeCompare(String(b?.id || ""));
}

// ---------------------------------------------------------------------------
// GO model helpers
// ---------------------------------------------------------------------------
function extractGoCoreFeatures(state, playerKey, deps, profile, P) {
  const player = state?.players?.[playerKey];
  const oppPlayer = state?.players?.[profile?.oppKey];
  const myScore = safeNum(profile?.myScore);
  const oppScore = safeNum(profile?.oppScore);
  const lead = myScore - oppScore;
  const selfPi = safeNum(profile?.ctx?.selfPi, safeNum(deps?.capturedCountByCategory?.(player, "junk")));
  const oppPi = safeNum(profile?.ctx?.oppPi, safeNum(deps?.capturedCountByCategory?.(oppPlayer, "junk")));
  const selfFive = safeNum(profile?.ctx?.selfFive);
  const oppFive = safeNum(profile?.ctx?.oppFive);
  const goCount = safeNum(player?.goCount);
  const carry = safeNum(state?.carryOverMultiplier, 1);
  const deckCount = safeNum(profile?.signals?.deckCount);
  const threat = clamp(safeNum(profile?.signals?.threat), 0, Math.max(0.1, safeNum(P?.goCoreThreatClamp, 1.8)));
  const oneAwayProb = clamp(safeNum(profile?.signals?.oneAwayProb), 0, Math.max(1, safeNum(P?.goCoreOneAwayClamp, 100)));

  const selfJokbo = typeof deps?.estimateJokboExpectedPotential === "function"
    ? (deps.estimateJokboExpectedPotential(state, playerKey, profile?.oppKey) || { total: 0, oneAwayCount: 0 })
    : { total: 0, oneAwayCount: 0 };
  const oppJokbo = typeof deps?.estimateOpponentJokboExpectedPotential === "function"
    ? (deps.estimateOpponentJokboExpectedPotential(state, playerKey) || { total: 0, oneAwayCount: 0 })
    : { total: 0, oneAwayCount: 0 };

  return {
    myScore,
    oppScore,
    lead,
    selfPi,
    oppPi,
    selfFive,
    oppFive,
    goCount,
    carry,
    deckCount,
    threat,
    oneAwayProb,
    selfJokboTotal: safeNum(selfJokbo?.total),
    selfJokboOneAway: safeNum(selfJokbo?.oneAwayCount),
    oppJokboTotal: safeNum(oppJokbo?.total),
    oppJokboOneAway: safeNum(oppJokbo?.oneAwayCount),
    secondMover: profile?.second ? 1 : 0,
    desperate: profile?.desperate ? 1 : 0
  };
}

function buildGoFlags(core, P) {
  const lead = safeNum(core?.lead);
  const trailing = lead < 0;
  const leading = lead > 0;
  const desperate = safeNum(core?.desperate) > 0;
  const second = safeNum(core?.secondMover) > 0;
  const selfCanStop = safeNum(core?.myScore) >= safeNum(P?.goHardSafeStopMinScore, 7);
  const oppCanStop = safeNum(core?.oppScore) >= safeNum(P?.goHardSafeStopMinScore, 7);
  let minPi = desperate ? safeNum(P?.goMinPiDesperate, 4) : safeNum(P?.goMinPi, 5);
  if (second && trailing) minPi -= safeNum(P?.goMinPiSecondTrailingDelta, 1);
  minPi = Math.max(3, minPi);

  return {
    lead,
    trailing,
    leading,
    desperate,
    second,
    selfCanStop,
    oppCanStop,
    minPi
  };
}

function coreFeatureSnapshot(core) {
  const out = {};
  for (const key of CORE_GO_FEATURE_KEYS) {
    out[key] = safeNum(core?.[key]);
  }
  return out;
}

// Layer-1 hard-risk model (single-turn static risk).
function calcHardStopRiskScore(core, flags, P) {
  const oneAwayNorm = safeNum(core?.oneAwayProb) / 100;
  const threat = safeNum(core?.threat);
  const lead = safeNum(flags?.lead);
  let riskScore =
    threat * safeNum(P.goHardRiskThreatMul, 0.36) +
    oneAwayNorm * safeNum(P.goHardRiskOneAwayMul, 0.34) +
    (flags?.oppCanStop ? safeNum(P.goHardRiskOppStopPenalty, 0.26) : 0) +
    (safeNum(core?.oppPi) >= 9 ? safeNum(P.goHardRiskOppHighPiPenalty, 0.08) : 0);

  if (lead >= 2 && safeNum(core?.selfPi) >= 8) {
    riskScore -= safeNum(P.goHardRiskLeadSafetyRelief, 0.12);
  }
  if (safeNum(core?.goCount) >= safeNum(P.goHardGoCountCap, 3)) {
    riskScore += 0.05;
  }
  return riskScore;
}

function classifyGoMode(core, flags) {
  if (flags?.desperate || flags?.trailing) return "chase";
  if (flags?.leading || flags?.selfCanStop || safeNum(core?.selfPi) >= 8) return "safe";
  return "neutral";
}

function isSecondChaseWindow(core, flags, P, swingProb) {
  return Boolean(
    flags?.second &&
    flags?.trailing &&
    !flags?.oppCanStop &&
    safeNum(core?.selfPi) >= safeNum(flags?.minPi) + Math.max(0, Math.floor(safeNum(P?.goSecondChasePiMargin, 1))) &&
    safeNum(core?.threat) <= safeNum(P?.goSecondChaseThreatCap, 0.72) &&
    safeNum(core?.oneAwayProb) <= safeNum(P?.goSecondChaseOneAwayCap, 62) &&
    safeNum(swingProb) <= safeNum(P?.goSecondChaseSwingCap, 0.45)
  );
}

function isSafeAttackWindow(core, flags, P, swingProb) {
  return Boolean(
    flags?.selfCanStop &&
    !flags?.oppCanStop &&
    safeNum(flags?.lead) <= 3 &&
    safeNum(core?.deckCount) >= safeNum(P?.goLiteSafeAttackDeckMin, 5) &&
    safeNum(core?.threat) <= safeNum(P?.goLiteSafeAttackThreatCap, 0.58) &&
    safeNum(core?.oneAwayProb) <= safeNum(P?.goLiteSafeAttackOneAwayCap, 45) &&
    safeNum(swingProb) <= safeNum(P?.goSafeGoSwingProbCut, 0.05) + 0.08
  );
}

// ---------------------------------------------------------------------------
// Card scoring helpers
// ---------------------------------------------------------------------------
function cardPhaseScales(phase, P) {
  if (phase === "early") {
    return {
      piMul: safeNum(P.phaseEarlyPiMul, 1.25),
      tempoMul: safeNum(P.phaseEarlyTempoMul, 1.08),
      riskMul: safeNum(P.phaseEarlyRiskMul, 0.94),
      blockMul: safeNum(P.phaseEarlyBlockMul, 0.95)
    };
  }
  if (phase === "late") {
    return {
      piMul: safeNum(P.phaseLatePiMul, 0.95),
      tempoMul: safeNum(P.phaseLateTempoMul, 0.96),
      riskMul: safeNum(P.phaseLateRiskMul, 1.15),
      blockMul: safeNum(P.phaseLateBlockMul, 1.35)
    };
  }
  if (phase === "end") {
    return {
      piMul: safeNum(P.phaseEndPiMul, 0.85),
      tempoMul: safeNum(P.phaseEndTempoMul, 0.92),
      riskMul: safeNum(P.phaseEndRiskMul, 1.35),
      blockMul: safeNum(P.phaseEndBlockMul, 2.0)
    };
  }
  return {
    piMul: safeNum(P.phaseMidPiMul, 1.0),
    tempoMul: safeNum(P.phaseMidTempoMul, 1.0),
    riskMul: safeNum(P.phaseMidRiskMul, 1.0),
    blockMul: safeNum(P.phaseMidBlockMul, 1.0)
  };
}

function estimateSelfPukRisk(month, cache, profile, P) {
  const boardCnt = safeNum(cache?.boardCountByMonth?.get(month));
  const handCnt = safeNum(cache?.handCountByMonth?.get(month));
  const capturedCnt = safeNum(cache?.capturedByMonth?.get(month));
  const known = boardCnt + handCnt + capturedCnt;
  const unseen = Math.max(0, 4 - known);
  const oppScore = safeNum(profile?.oppScore);

  let base = 0;
  if (boardCnt === 1 && handCnt >= 2) {
    base += 0.52 + Math.max(0, handCnt - 2) * 0.1;
    base += safeNum(P.selfPukRiskPatternBonus, 0.22);
  } else if (boardCnt === 0 && handCnt >= 3) {
    base += 0.28;
  }

  if (unseen <= 1) base += 0.22;
  else if (unseen === 2) base += 0.12;

  if (profile?.phase === "late") base += 0.08;
  if (profile?.phase === "end") base += 0.14;

  const oppPressure = clamp((oppScore - 4) / 6, 0, 2.0);
  const expMul = safeNum(P.selfPukRiskExpMul, 0.42);
  const risk = base * (1 + Math.expm1(oppPressure) * expMul);
  return clamp(risk, 0, 3.2);
}

function estimateNextTurnSwingProb(core, flags, P) {
  const threatNorm = clamp(safeNum(core?.threat) / 1.8, 0, 1);
  const oneAwayNorm = clamp(safeNum(core?.oneAwayProb) / 100, 0, 1);
  const jokboNorm = clamp(safeNum(core?.oppJokboOneAway) / 2, 0, 1);
  let prob =
    threatNorm * 0.45 +
    oneAwayNorm * 0.35 +
    jokboNorm * 0.20;
  if (safeNum(core?.deckCount) <= safeNum(P.phaseLateDeck, 6)) {
    prob += 0.06;
  }
  if (safeNum(core?.oppScore) >= 6) {
    prob += 0.08;
  }
  if (flags?.oppCanStop) prob += 0.03;
  return clamp(prob, 0, 1);
}

// Main card evaluator for hand ranking.
function evaluateCard(state, playerKey, card, deps, P, profile, cache, handCtx) {
  const month = Number(card?.month);
  const matches = cache.board.get(month) || [];
  const knownRatio = clamp(countKnownMonth(cache, month) / 4, 0, 1);
  const phaseScale = cardPhaseScales(profile.phase, P);
  const capturePlan = matches.length > 0
    ? capturePlanScore(state, playerKey, month, [card, ...matches], deps, P, profile, cache, handCtx)
    : null;
  const discardPlan = matches.length === 0
    ? discardPlanScore(card, month, deps, P, profile, cache, handCtx)
    : { tempo: 0, holdPenalty: 0, safeDiscardHint: 0 };

  const matchBase =
    matches.length === 0 ? safeNum(P.noMatchBase) :
    matches.length === 1 ? safeNum(P.matchOneBase) :
    matches.length === 2 ? safeNum(P.matchTwoBase) :
    safeNum(P.matchThreeBase);

  const piGain = safeNum(capturePlan?.piGain);
  const monthPriority = safeNum(capturePlan?.monthPriority, safeNum(deps?.monthStrategicPriority?.(month)));
  const comboOpportunity = safeNum(capturePlan?.comboOpportunity);

  let immediate = matchBase + safeNum(capturePlan?.immediate);
  let deny = safeNum(capturePlan?.block);
  if (matches.length === 0 && cache.blockMonths.has(month)) {
    deny -= safeNum(P.blockNoMatchPenalty, 0);
  }
  if (profile.second) {
    const oneAwayNorm = clamp(safeNum(profile.signals.oneAwayProb) / 100, 0, 1);
    deny += oneAwayNorm * safeNum(P.cardSecondBlockThreatBonus, 0.9);
  }

  let tempo = safeNum(discardPlan?.tempo);
  if (!profile.second && profile.trailing && comboOpportunity > 0) {
    tempo += comboOpportunity * safeNum(P.cardFirstTrailTempoBonus, 0.35);
  }
  if (matches.length > 0) {
    immediate += knownRatio * safeNum(P?.playMatchKnownMonthMul, 0);
    if (card?.category === "ribbon") {
      immediate += safeNum(P?.playRibbonMatchBonus, 0);
      if (safeNum(handCtx?.matchableCategoryCounts?.junk) > 0) {
        immediate += safeNum(P?.playRibbonMatchWhenJunkAvailableBonus, 0);
      }
    } else if (card?.category === "junk" && safeNum(handCtx?.matchableCategoryCounts?.ribbon) > 0) {
      immediate -= safeNum(P?.playJunkMatchWhenRibbonAvailablePenalty, 0);
    }
  } else if (handCtx?.hasAnyMatchablePlay) {
    tempo -= safeNum(P?.playNoMatchWhenAnyMatchPenalty, 0);
    tempo -= knownRatio * safeNum(P?.playSafeNoMatchWhenAnyMatchPenalty, 0);
    if (card?.category === "junk") {
      tempo -= safeNum(P?.playNoMatchJunkWhenAnyMatchPenalty, 0);
    }
    if (isDoublePi(card, deps)) {
      tempo -= safeNum(P?.playNoMatchDoublePiWhenAnyMatchPenalty, 0);
    }
  }

  let risk = 0;
  let feedRisk = cache.oppImmediateGainByMonth.get(month);
  if (feedRisk == null) {
    feedRisk = safeNum(deps?.estimateOpponentImmediateGainIfDiscard?.(state, playerKey, month));
    cache.oppImmediateGainByMonth.set(month, feedRisk);
  }
  risk += feedRisk * (matches.length === 0 ? P.feedRiskNoMatchMul : P.feedRiskMatchMul);

  let dangerMonthRisk = cache.dangerMonthRiskByMonth.get(month);
  if (dangerMonthRisk == null) {
    dangerMonthRisk = safeNum(deps?.estimateDangerMonthRisk?.(
      state,
      playerKey,
      month,
      cache.boardCountByMonth,
      cache.handCountByMonth,
      cache.capturedByMonth
    ));
    cache.dangerMonthRiskByMonth.set(month, dangerMonthRisk);
  }
  risk += dangerMonthRisk * (matches.length === 0 ? P.dangerNoMatchMul : P.dangerMatchMul);

  const releaseRisk = safeNum(deps?.estimateReleasePunishProb?.(state, playerKey, month, profile.signals.jokboThreat, profile.ctx));
  if (matches.length === 0 && releaseRisk >= P.releaseRiskFloor) {
    const oppScoreExp = 1 + Math.expm1(clamp((profile.oppScore - 4) / 6, 0, 2.0)) * safeNum(P.releaseRiskOppScoreExpMul, 0.34);
    risk += (releaseRisk - P.releaseRiskFloor) * P.releaseRiskMul * oppScoreExp;
    if (profile.second) {
      risk += (releaseRisk - P.releaseRiskFloor) * safeNum(P.cardSecondReleaseRiskMul, 1.1);
    }
  }

  const pukRisk = safeNum(deps?.isRiskOfPuk?.(state, playerKey, card, cache.boardCountByMonth, cache.handCountByMonth));
  if (pukRisk > 0) risk += pukRisk * P.pukRiskMul;
  else if (pukRisk < 0) immediate += -pukRisk * P.pukOpportunityMul;

  const selfPukRisk = estimateSelfPukRisk(month, cache, profile, P);
  risk += selfPukRisk * safeNum(P.selfPukRiskMul, 1.0);

  let holdPenalty = 0;
  if (matches.length === 0 && isDoublePi(card, deps)) {
    holdPenalty += P.doublePiNoMatchHoldPenalty;
    const sameMonth = safeNum(cache.handCountByMonth.get(month));
    if (sameMonth >= 2) holdPenalty += P.doublePiMonthPairHoldPenalty;
    if (sameMonth >= 3) holdPenalty += P.doublePiMonthTripleHoldPenalty;
  }

  let seatRiskMul = 1.0;
  if (profile.second && matches.length === 0) {
    seatRiskMul += safeNum(P.cardSecondNoMatchRiskMul, 0.12);
  }

  const score =
    immediate * profile.aggression +
    deny * profile.defense * phaseScale.blockMul +
    tempo * profile.tempo * phaseScale.tempoMul -
    risk * profile.risk * phaseScale.riskMul * seatRiskMul -
    safeNum(discardPlan?.holdPenalty);

  return {
    card,
    score,
    matches: matches.length,
    piGain,
    monthPriority,
    safeDiscardHint: safeNum(discardPlan?.safeDiscardHint),
    releaseRisk
  };
}

// ---------------------------------------------------------------------------
// Policy exports
// ---------------------------------------------------------------------------
export function rankHandCardsGPT(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const player = state?.players?.[playerKey];
  if (!Array.isArray(player?.hand) || player.hand.length <= 0) return [];

  const P = mergedParams(params);
  const profile = buildProfile(state, playerKey, deps, P);
  const cache = buildRankingCache(state, playerKey, deps, profile);
  const handCtx = buildHandPackageContext(state, playerKey, deps, profile, cache);

  const ranked = player.hand.map((card) => evaluateCard(state, playerKey, card, deps, P, profile, cache, handCtx));
  ranked.sort((a, b) => compareRankEntries(a, b, P));
  return ranked;
}

// Resolve pending two-match board choice.
export function chooseMatchHeuristicGPT(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const ids = state?.pendingMatch?.boardCardIds || [];
  if (!Array.isArray(ids) || ids.length <= 0) return null;

  const P = mergedParams(params);
  const profile = buildProfile(state, playerKey, deps, P);
  const cache = buildRankingCache(state, playerKey, deps, profile);
  const handCtx = buildHandPackageContext(state, playerKey, deps, profile, cache);

  const candidates = [];
  for (const card of (state?.board || []).filter((c) => ids.includes(c?.id))) {
    const month = Number(card?.month);
    const pi = piValue(card, deps);
    const monthPriority = safeNum(deps?.monthStrategicPriority?.(month));
    const urgency = monthUrgencyInfo(month, profile, cache);
    const comboFinish = ownComboFinishBonus(handCtx.combo, [card], P);
    let comboOpportunity = cache.comboOpportunityByMonth.get(month);
    if (comboOpportunity == null) {
      comboOpportunity = safeNum(deps?.ownComboOpportunityScore?.(state, playerKey, month));
      cache.comboOpportunityByMonth.set(month, comboOpportunity);
    }

    let score = safeNum(deps?.cardCaptureValue?.(card)) * P.chooseMatchBaseMul + pi * P.chooseMatchPiMul;
    if (card?.category === "kwang") score += P.chooseMatchKwangBonus;
    if (card?.category === "five") score += P.chooseMatchFiveBonus;
    if (card?.category === "ribbon") score += P.chooseMatchRibbonBonus;

    const ribbonBuildBonus = ribbonComboBuildBonus(handCtx.combo, [card], P);
    score += comboOpportunity * P.chooseMatchComboMul;
    score += comboFinish;
    score += ribbonBuildBonus;
    if (cache.blockMonths.has(month)) {
      score += (urgency.normalizedUrgency + safeNum(profile.signals.threat)) * P.chooseMatchBlockMul;
      if (profile.second) score += safeNum(P.secondMoverBlockBonus, 0);
    }
    if (handCtx.ribbonCount >= 4 && card?.category === "ribbon") {
      score += safeNum(P.ribbonFourBonus, 0);
    }
    if (handCtx.fiveCount >= 4 && card?.category === "five") {
      score += safeNum(P.fiveFourBonus, 0);
    }
    if (handCtx.mongBak) {
      if (card?.category === "five") score += safeNum(P.handMongBakFiveBonus, 0);
      else if (pi > 0) score -= safeNum(P.handMongBakPiPenalty, 0);
    }
    score += monthPriority * 0.22;
    const category = String(card?.category || "");
    candidates.push({ id: card?.id || null, score, pi, monthPriority, category, ribbonBuildBonus });
  }

  const hasRibbonBuildCandidate = candidates.some(
    (candidate) => candidate.category === "ribbon" && safeNum(candidate.ribbonBuildBonus) > 0
  );
  const hasJunkCandidate = candidates.some((candidate) => candidate.category === "junk");
  const hasHighValueCandidate = candidates.some(
    (candidate) => candidate.category === "kwang" || candidate.category === "five"
  );
  if (hasRibbonBuildCandidate && hasJunkCandidate && !hasHighValueCandidate) {
    for (const candidate of candidates) {
      if (candidate.category === "ribbon" && safeNum(candidate.ribbonBuildBonus) > 0) {
        candidate.score += safeNum(P?.chooseMatchRibbonVsJunkBonus, 0);
      } else if (candidate.category === "junk") {
        candidate.score -= safeNum(P?.chooseMatchJunkVsRibbonPenalty, 0);
      }
    }
  }

  if (candidates.length <= 0) return null;
  candidates.sort((a, b) => compareMatchChoice(a, b, P));
  return candidates[0].id || null;
}

// GO decision tree:
// - layer0 precheck
// - layer1 hard stop gates
// - layer2 utility vs threshold
// - layer3 final soft gates
export function shouldGoGPT(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const P = mergedParams(params);
  const trace = beginTrace("go", playerKey, P, deps);
  const profile = buildProfile(state, playerKey, deps, P);
  const core = extractGoCoreFeatures(state, playerKey, deps, profile, P);
  const flags = buildGoFlags(core, P);
  const swingProb = estimateNextTurnSwingProb(core, flags, P);
  const clDecision = shouldGoCLBase(state, playerKey, deps);

  traceLayer(trace, "layer0_base_cl", {
    clDecision: clDecision ? 1 : 0,
    features: coreFeatureSnapshot(core),
    minPi: flags.minPi,
    swingProb
  });

  if (clDecision) {
    return pushTrace(trace, true, "cl_base_pass", P, deps);
  }

  if (typeof deps?.canBankruptOpponentByStop === "function" && deps.canBankruptOpponentByStop(state, playerKey)) {
    return pushTrace(trace, false, "bankrupt_stop", P, deps);
  }

  if (core.selfPi < flags.minPi) {
    return pushTrace(trace, false, "min_pi_gate", P, deps);
  }

  if (flags.oppCanStop && core.oppScore >= 7 && core.threat >= 0.55) {
    return pushTrace(trace, false, "opp_stop_risk_lock", P, deps);
  }

  const ultraSafePush =
    !flags.oppCanStop &&
    core.selfPi >= Math.max(8, flags.minPi + 2) &&
    core.selfJokboOneAway >= 1 &&
    core.threat <= 0.28 &&
    core.oneAwayProb <= 22 &&
    swingProb <= 0.1;
  if (ultraSafePush) {
    traceLayer(trace, "layer1_overlay", { mode: "ultra_safe_push", decision: 1 });
    return pushTrace(trace, true, "ultra_safe_push", P, deps);
  }

  const safeAttackOverlay =
    isSafeAttackWindow(core, flags, P, swingProb) &&
    core.selfPi >= Math.max(7, flags.minPi + 1) &&
    core.threat <= 0.38 &&
    core.oneAwayProb <= 28 &&
    swingProb <= 0.12 &&
    (
      core.selfJokboOneAway >= 1 ||
      core.selfJokboTotal >= core.oppJokboTotal + 0.8 ||
      core.myScore >= 5
    );
  if (safeAttackOverlay) {
    traceLayer(trace, "layer1_overlay", { mode: "safe_attack_overlay", decision: 1 });
    return pushTrace(trace, true, "safe_attack_overlay", P, deps);
  }

  const chaseOverlay =
    isSecondChaseWindow(core, flags, P, swingProb) &&
    core.selfPi >= flags.minPi + 1 &&
    core.threat <= 0.48 &&
    core.oneAwayProb <= 36 &&
    swingProb <= 0.18 &&
    (
      core.selfJokboOneAway >= 1 ||
      core.selfJokboTotal >= core.oppJokboTotal + 0.5 ||
      core.myScore + 1 <= core.oppScore
    );
  if (chaseOverlay) {
    traceLayer(trace, "layer1_overlay", { mode: "second_chase_overlay", decision: 1 });
    return pushTrace(trace, true, "second_chase_overlay", P, deps);
  }

  return pushTrace(trace, false, "cl_base_fail", P, deps);
}

// Choose best bomb month by explicit steal/ssangpi/quick-score priority.
export function selectBombMonthGPT(state, playerKey, bombMonths, deps) {
  if (!Array.isArray(bombMonths) || bombMonths.length <= 0) return null;
  const ctx = typeof deps?.analyzeGameContext === "function"
    ? deps.analyzeGameContext(state, playerKey)
    : {};
  let best = bombMonths[0];
  let bestScore = -Infinity;
  for (const month of bombMonths) {
    const impact = typeof deps?.isHighImpactBomb === "function"
      ? deps.isHighImpactBomb(state, playerKey, month)
      : null;
    const summary = summarizeBombMonth(state, playerKey, month, deps);
    const boardGain = safeNum(deps?.monthBoardGain?.(state, month));
    const immediateGain = safeNum(impact?.immediateGain);
    const quickGoPush = safeNum(ctx?.myScore) <= 2 && immediateGain >= 6 ? 1 : 0;
    const score =
      summary.bonusSteal * 3.5 +
      summary.doublePiCount * 2.6 +
      summary.highValueCount * 0.9 +
      boardGain * 0.8 +
      immediateGain * 0.55 +
      (impact?.highImpact ? 2.4 : 0) +
      (quickGoPush ? 1.6 : 0);
    if (score > bestScore) {
      bestScore = score;
      best = month;
    }
  }
  return best;
}

// Bomb decision gate.
export function shouldBombGPT(state, playerKey, bombMonths, deps, params = DEFAULT_PARAMS) {
  if (!Array.isArray(bombMonths) || bombMonths.length <= 0) return false;
  const planRaw = typeof deps?.getFirstTurnDoublePiPlan === "function"
    ? deps.getFirstTurnDoublePiPlan(state, playerKey)
    : null;
  const planMonths = asMonthSet(planRaw?.months);
  if (Boolean(planRaw?.active) && bombMonths.some((m) => planMonths.has(Number(m)))) return true;

  const month = selectBombMonthGPT(state, playerKey, bombMonths, deps);
  if (month == null) return false;

  const ctx = typeof deps?.analyzeGameContext === "function"
    ? deps.analyzeGameContext(state, playerKey)
    : {};
  const impact = typeof deps?.isHighImpactBomb === "function"
    ? deps.isHighImpactBomb(state, playerKey, month)
    : null;
  const summary = summarizeBombMonth(state, playerKey, month, deps);
  const immediateGain = safeNum(impact?.immediateGain);
  const explosiveNow =
    summary.bonusSteal > 0 ||
    summary.doublePiCount > 0 ||
    Boolean(impact?.highImpact) ||
    (safeNum(ctx?.myScore) <= 2 && immediateGain >= 6);
  if (explosiveNow) return true;

  const alternativeMatches = countAlternativeMatchCards(state, playerKey, bombMonths, deps);
  if (alternativeMatches > 0) return false;

  return true;
}

// Shaking decision + selected month payload.
export function decideShakingGPT(state, playerKey, shakingMonths, deps, params = DEFAULT_PARAMS) {
  if (!Array.isArray(shakingMonths) || shakingMonths.length <= 0) {
    return { allow: false, month: null, score: -Infinity };
  }

  const P = mergedParams(params);
  const profile = buildProfile(state, playerKey, deps, P);
  const livePiMonths = getLiveDoublePiMonths(state, deps);
  const comboHoldMonths = getComboHoldMonths(state, playerKey, deps);
  const deckCount = safeNum(profile?.signals?.deckCount, safeNum(state?.deck?.length));
  const lateReveal = deckCount <= 8;
  const endReveal = deckCount <= 5;
  const bbm = typeof deps?.boardMatchesByMonth === "function" ? deps.boardMatchesByMonth(state) : null;
  if (
    bbm &&
    (state.players?.[playerKey]?.hand || []).some((card) => (bbm.get(Number(card?.month || 0)) || []).length > 0)
  ) {
    return { allow: false, month: null, score: -Infinity };
  }
  const myScore = safeNum(profile?.myScore);
  const oppScore = safeNum(profile?.oppScore);
  if (oppScore >= 5 && oppScore >= myScore + 2) {
    return { allow: false, month: null, score: -Infinity };
  }
  const planRaw = typeof deps?.getFirstTurnDoublePiPlan === "function"
    ? deps.getFirstTurnDoublePiPlan(state, playerKey)
    : null;
  const planMonths = asMonthSet(planRaw?.months);

  let best = { allow: false, month: null, score: -Infinity, highImpact: false };
  for (const month of shakingMonths) {
    const immediate = safeNum(deps?.shakingImmediateGainScore?.(state, playerKey, month));
    const combo = safeNum(deps?.ownComboOpportunityScore?.(state, playerKey, month));
    const impact = typeof deps?.isHighImpactShaking === "function"
      ? deps.isHighImpactShaking(state, playerKey, month)
      : null;
    const known = safeNum(deps?.countKnownMonthCards?.(state, month));
    const monthNum = Number(month);
    const monthCards = (state.players?.[playerKey]?.hand || []).filter((card) => Number(card?.month) === monthNum);
    const monthPiPayload = monthCards.reduce((sum, card) => sum + piValue(card, deps), 0);
    const monthBonusCount = monthCards.filter((card) => Number(card?.bonus?.stealPi || 0) > 0).length;
    const monthDoublePiCount = monthCards.filter((card) => isDoublePi(card, deps)).length;
    const monthHighCardCount = monthCards.filter((card) => card?.category === "kwang" || card?.category === "five").length;
    const isLivePiMonth = livePiMonths.has(monthNum);
    const isComboHoldMonth = comboHoldMonths.has(monthNum);
    const isPlanMonth = Boolean(planRaw?.active) && planMonths.has(monthNum);
    let keepPenalty = 0;
    keepPenalty += monthPiPayload * 0.08;
    keepPenalty += monthBonusCount * 0.75;
    keepPenalty += monthDoublePiCount * 0.42;
    keepPenalty += monthHighCardCount * 0.28;
    if (isLivePiMonth) keepPenalty += deckCount > safeNum(P.phaseLateDeck, 6) ? 0.28 : 0.12;
    if (isComboHoldMonth) keepPenalty += 0.35;
    if (isPlanMonth) keepPenalty += 0.18;
    if (known <= 2) keepPenalty += 0.12;
    if (lateReveal) keepPenalty += 0.20;
    if (endReveal) keepPenalty += 0.18;
    if (profile.trailing) keepPenalty += 0.22;

    let score =
      immediate * P.shakeImmediateMul +
      combo * P.shakeComboMul +
      (impact?.highImpact ? P.shakeImpactBonus : 0) +
      (impact?.hasDoublePiLine ? P.shakePiLineBonus : 0) +
      (impact?.directThreeGwang ? P.shakeDirectGwangBonus : 0) -
      profile.signals.threat * P.shakeRiskMul -
      keepPenalty;

    if (known >= 4) score += 0.08;

    if (score > best.score) {
      best = {
        allow: false,
        month,
        score,
        highImpact: Boolean(impact?.highImpact)
      };
    }
  }

  let threshold = P.shakeThreshold;
  if (profile.leading) threshold -= 0.04;
  if (profile.signals.threat >= 0.72) threshold += P.shakePressureThresholdUp;
  if (profile.trailing) threshold += 0.10;
  const allowGate = profile.leading || (myScore === oppScore && best.highImpact);

  return { ...best, allow: allowGate && best.score >= threshold };
}

// President hold/stop option gate.
export function shouldPresidentStopGPT(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const ctx = typeof deps?.analyzeGameContext === "function"
    ? deps.analyzeGameContext(state, playerKey)
    : {};
  const player = state?.players?.[playerKey];
  const scoreDiff = safeNum(ctx?.myScore) - safeNum(ctx?.oppScore);
  const oppScore = safeNum(ctx?.oppScore);
  const carry = safeNum(state?.carryOverMultiplier, 1);
  const handPressure = analyzePresidentHandPressure(player, deps);
  const isMidgamePresident = safeNum(state?.turnSeq, 0) > 0;

  if (handPressure.bonusCount >= 2) return false;
  if (handPressure.bonusCount >= 1 && handPressure.doublePiCount >= 2) return false;
  if (handPressure.doublePiCount >= 3) return false;
  if (isMidgamePresident && scoreDiff > 0 && oppScore <= 4) return false;
  if (handPressure.weightedPressure >= 4 && scoreDiff >= 0 && carry <= 1) return false;

  if (scoreDiff < 0) return true;
  if (carry >= 2 && handPressure.weightedPressure < 4) return true;
  if (scoreDiff >= 2 && handPressure.weightedPressure <= 1) return true;
  return false;
}

// Gukjin mode selector.
export function chooseGukjinHeuristicGPT(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const rawSelfFive = rawFiveCountIncludingCapturedGukjin(state?.players?.[playerKey]);
  const rawOppFive = rawFiveCountIncludingCapturedGukjin(state?.players?.[otherPlayerKey(deps, playerKey)]);
  if (rawSelfFive >= 6 && rawOppFive <= 0) return "five";
  return "junk";
}

// ---------------------------------------------------------------------------
// External trace access API
// ---------------------------------------------------------------------------
export function getDecisionTraceGPT(decisionType = null, playerKey = null) {
  const all = [];
  for (const [key, traces] of TRACE_STORE.entries()) {
    const [kind, actor] = key.split(":");
    if (decisionType != null && String(kind) !== String(decisionType)) continue;
    if (playerKey != null && String(actor) !== String(playerKey)) continue;
    for (const t of traces || []) all.push(t);
  }
  all.sort((a, b) => safeNum(a?.at) - safeNum(b?.at));
  return all;
}

// Clear all or filtered traces.
export function clearDecisionTraceGPT(decisionType = null, playerKey = null) {
  if (decisionType == null && playerKey == null) {
    TRACE_STORE.clear();
    return;
  }
  for (const key of [...TRACE_STORE.keys()]) {
    const [kind, actor] = key.split(":");
    if (decisionType != null && String(kind) !== String(decisionType)) continue;
    if (playerKey != null && String(actor) !== String(playerKey)) continue;
    TRACE_STORE.delete(key);
  }
}
