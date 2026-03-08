// heuristicGPT.js - Matgo Heuristic GPT (rule-tree v1)
/* ============================================================================
 * Heuristic GPT (no rollout)
 * - Layer 1: hard safety gates
 * - Layer 2: risk-adjusted utility
 * - Layer 3: deterministic decision tree
 * - exported decisions: rank / match / go / bomb / shaking / president / gukjin
 * ========================================================================== */

const GUKJIN_CARD_ID = "I0";
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
  chooseMatchBlockMul: 1.7,
  chooseMatchComboMul: 2.4,
  tieScoreEpsilon: 0.000001,

  // go model (hard gates)
  goHardThreatCut: 1.0,
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
  goBaseThreshold: -0.088202235346907271,
  goSecondTrailBonus: 0.04,
  goRallyPiWindowBonus: 0.02,
  goRallySecondBonus: 0.01,
  goRallyTrailBonus: 0.02,
  goRallyEndDeckBonus: 0.0,
  goSoftHighPiThreatCap: 0.9,
  goSoftHighPiOneAwayCap: 66,
  goSoftHighPiMargin: 0.06,
  goSoftTrailHighPiMargin: 0.04,
  goSoftValueMargin: 0.02,
  goSecondChaseThresholdRelief: 0.045,
  goSecondChaseNearMargin: 0.03,
  goSecondChasePiMargin: 1,
  goSecondChaseThreatCap: 0.72,
  goSecondChaseOneAwayCap: 62,
  goSecondChaseSwingCap: 0.45,
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
  goHardLateOneAwayCut: 47,
  goHardOppScoreCut: 8,
  goHardRiskOneAwayMul: 0.67869959500383537,
  goHardRiskOppStopPenalty: 0.43208757051963176,
  goHardRiskThreatMul: 0.21781498975205768,
  goHardRiskThreshold: 0.58761093591951763,
  goHardThreatDeckCut: 10,
  goLiteLatePenalty: 0.18125403165573611,
  goLiteOneAwayPenaltyMul: 0.11608197553448872,
  goLiteOppCanStopPenalty: 0.3,
  goLiteSafeAttackBonus: 0.15494875441940156,
  goLiteThreatPenaltyMul: 0.13560896875251968,
  goLookaheadThresholdMul: 0.059459872467595147,
  goMinPi: 6,
  goMinPiDesperate: 6,
  goMinPiSecondTrailingDelta: 2,
  goRiskOneAwayMul: 0.17410107037517836,
  goRiskOppJokboMul: 0.34749337695693389,
  goRiskOppOneAwayMul: 0.13405521881051077,
  goRiskPressureMul: 0.37751193247462522,
  goSafeGoBonus: 0.26164359701018652,
  goThresholdLeadUp: 0.17600484318391654,
  goThresholdPressureUp: 0.0023769481568353054,
  goThresholdTrailDown: 0.10567145021358916,
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
function evaluateCard(state, playerKey, card, deps, P, profile, cache) {
  const month = Number(card?.month);
  const matches = cache.board.get(month) || [];
  const captured = [card, ...matches];
  const phaseScale = cardPhaseScales(profile.phase, P);
  const captureGain = captured.reduce((acc, c) => acc + safeNum(deps?.cardCaptureValue?.(c)), 0);
  const piGain = captured.reduce((acc, c) => acc + piValue(c, deps), 0);
  const monthPriority = safeNum(deps?.monthStrategicPriority?.(month));
  const selfPi = safeNum(profile.ctx?.selfPi, safeNum(deps?.capturedCountByCategory?.(state?.players?.[playerKey], "junk")));
  const oppPi = safeNum(profile.ctx?.oppPi, safeNum(deps?.capturedCountByCategory?.(state?.players?.[profile.oppKey], "junk")));

  let immediate = matches.length === 0
    ? P.noMatchBase
    : matches.length === 1
      ? P.matchOneBase
      : matches.length === 2
        ? P.matchTwoBase
        : P.matchThreeBase;

  immediate += captureGain * P.captureGainMul;
  immediate += categoryCaptureBonus(matches, P);
  immediate += piGain * P.junkPiMul * phaseScale.piMul;
  if (selfPi >= 7 && selfPi <= 9) immediate += piGain * P.selfPiWindowMul;
  if (oppPi <= 5) immediate += piGain * P.oppPiWindowMul;
  if (captured.some((c) => isDoublePi(c, deps))) immediate += P.doublePiBonus;

  let comboOpportunity = cache.comboOpportunityByMonth.get(month);
  if (comboOpportunity == null) {
    comboOpportunity = safeNum(deps?.ownComboOpportunityScore?.(state, playerKey, month));
    cache.comboOpportunityByMonth.set(month, comboOpportunity);
  }
  immediate += comboOpportunity * P.comboOpportunityMul;

  let deny = 0;
  if (cache.blockMonths.has(month)) {
    const urgency = Math.max(safeNum(cache.blockUrgency.get(month)), safeNum(profile.signals.monthUrgency.get(month)) / 10);
    deny += P.blockBase + urgency * P.blockUrgencyMul + profile.signals.threat * P.blockThreatMul;
    if (matches.length === 0) deny -= P.blockNoMatchPenalty;
  }
  if (profile.second) {
    const oneAwayNorm = clamp(safeNum(profile.signals.oneAwayProb) / 100, 0, 1);
    deny += oneAwayNorm * safeNum(P.cardSecondBlockThreatBonus, 0.9);
  }

  let tempo = 0;
  const knownMonth = safeNum(cache.boardCountByMonth.get(month)) + safeNum(cache.handCountByMonth.get(month)) + safeNum(cache.capturedByMonth.get(month));
  if (matches.length === 0) {
    if (knownMonth >= 3) tempo += P.knownMonthSafeBonus;
    else if (knownMonth <= 1) tempo -= P.unknownMonthPenalty;
  }
  if (profile.trailing && piGain > 0) tempo += piGain * P.trailPiTempoMul;
  if (profile.leading && matches.length === 0) tempo -= P.leadNoMatchTempoPenalty;
  if (profile.phase === "end" && matches.length === 0) {
    if (knownMonth >= 3) tempo += P.endgameSafeDiscardBonus;
    else tempo -= P.endgameUnknownPenalty;
  }
  if (!profile.second && profile.trailing && comboOpportunity > 0) {
    tempo += comboOpportunity * safeNum(P.cardFirstTrailTempoBonus, 0.35);
  }
  if (cache.firstTurnPlan.active && cache.firstTurnPlan.months.has(month)) {
    tempo += P.firstTurnPlanBonus;
  }
  tempo += monthPriority * 0.2;

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
    holdPenalty;

  return {
    card,
    score,
    matches: matches.length,
    piGain,
    monthPriority,
    safeDiscardHint: knownMonth,
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

  const ranked = player.hand.map((card) => evaluateCard(state, playerKey, card, deps, P, profile, cache));
  ranked.sort((a, b) => compareRankEntries(a, b, P));
  return ranked;
}

// Resolve pending two-match board choice.
export function chooseMatchHeuristicGPT(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const ids = state?.pendingMatch?.boardCardIds || [];
  if (!Array.isArray(ids) || ids.length <= 0) return null;

  const P = mergedParams(params);
  const profile = buildProfile(state, playerKey, deps, P);
  const oppPlayer = state?.players?.[profile.oppKey];
  const selfPlayer = state?.players?.[playerKey];
  const blockMonths = asMonthSet(
    typeof deps?.blockingMonthsAgainst === "function"
      ? deps.blockingMonthsAgainst(oppPlayer, selfPlayer)
      : null
  );
  const blockUrgency = asMonthMap(
    typeof deps?.blockingUrgencyByMonth === "function"
      ? deps.blockingUrgencyByMonth(oppPlayer, selfPlayer)
      : null
  );

  const candidates = [];
  for (const card of (state?.board || []).filter((c) => ids.includes(c?.id))) {
    const month = Number(card?.month);
    const pi = piValue(card, deps);
    const monthPriority = safeNum(deps?.monthStrategicPriority?.(month));
    let score = safeNum(deps?.cardCaptureValue?.(card)) * P.chooseMatchBaseMul + pi * P.chooseMatchPiMul;
    if (card?.category === "kwang") score += P.chooseMatchKwangBonus;
    if (card?.category === "five") score += P.chooseMatchFiveBonus;
    if (card?.category === "ribbon") score += P.chooseMatchRibbonBonus;

    score += safeNum(deps?.ownComboOpportunityScore?.(state, playerKey, month)) * P.chooseMatchComboMul;
    if (blockMonths.has(month)) {
      const urgency = Math.max(safeNum(blockUrgency.get(month)), safeNum(profile.signals.monthUrgency.get(month)) / 10);
      score += (urgency + profile.signals.threat) * P.chooseMatchBlockMul;
    }
    score += monthPriority * 0.22;
    candidates.push({ id: card?.id || null, score, pi, monthPriority });
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

  if (typeof deps?.canBankruptOpponentByStop === "function" && deps.canBankruptOpponentByStop(state, playerKey)) {
    traceLayer(trace, "layer0_precheck", { canBankruptOpponentByStop: 1 });
    return pushTrace(trace, false, "bankrupt_stop", P, deps);
  }

  const profile = buildProfile(state, playerKey, deps, P);
  const core = extractGoCoreFeatures(state, playerKey, deps, profile, P);
  const flags = buildGoFlags(core, P);

  traceLayer(trace, "layer1_core_features", {
    features: coreFeatureSnapshot(core),
    minPi: flags.minPi
  });

  if (core.selfPi < flags.minPi) {
    traceLayer(trace, "layer1_hard_stop", { reason: "min_pi_gate", selfPi: core.selfPi, minPi: flags.minPi });
    return pushTrace(trace, false, "min_pi_gate", P, deps);
  }

  if (safeNum(P.goHardSafeStopEnabled, 1) > 0 && !flags.desperate) {
    if (
      flags.selfCanStop &&
      !flags.oppCanStop &&
      core.deckCount <= safeNum(P.goHardSafeStopDeckCut, 7) &&
      core.myScore >= core.oppScore + safeNum(P.goHardSafeStopLeadMin, 2)
    ) {
      traceLayer(trace, "layer1_hard_stop", {
        reason: "safe_stop_lock",
        selfCanStop: 1,
        oppCanStop: 0,
        deckCount: core.deckCount,
        lead: flags.lead
      });
      return pushTrace(trace, false, "safe_stop_lock", P, deps);
    }
  }

  const swingProb = estimateNextTurnSwingProb(core, flags, P);
  const hardRiskScore = calcHardStopRiskScore(core, flags, P);
  const goMode = classifyGoMode(core, flags);
  traceLayer(trace, "layer1_hard_risk", {
    swingProb,
    score: hardRiskScore,
    threshold: safeNum(P.goHardRiskThreshold, 0.62),
    mode: goMode
  });

  // Layer 1: deterministic hard-stop cutoffs.
  if (!flags.desperate) {
    let hardReason = "";
    if (core.oppScore >= P.goHardOppScoreCut && core.myScore <= core.oppScore + safeNum(P.goHardOppLeadGrace, 1)) {
      hardReason = "opp_score_cut";
    } else if (
      flags.oppCanStop &&
      core.carry >= safeNum(P.goHardCarryOppStopCarryMin, 2) &&
      core.deckCount <= safeNum(P.goHardCarryOppStopLateDeckCut, 6) &&
      core.threat >= safeNum(P.goHardCarryOppStopThreatCut, 0.85)
    ) {
      hardReason = "carry_opp_stop_hard";
    } else if (
      safeNum(P.goHardSecondChaseBlockEnabled, 1) > 0 &&
      flags.second &&
      goMode === "chase" &&
      flags.oppCanStop &&
      core.deckCount <= safeNum(P.goHardSecondChaseLateDeckCut, 7) &&
      core.threat >= safeNum(P.goHardSecondChaseThreatCut, 0.8)
    ) {
      hardReason = "second_chase_block";
    } else if (
      core.deckCount <= safeNum(P.goHardThreatDeckCut, 10) &&
      (
        core.threat >= safeNum(P.goHardThreatCut, 1.0) ||
        core.oneAwayProb >= safeNum(P.goHardLateOneAwayCut, 47) ||
        (
          core.oppJokboOneAway >= safeNum(P.goHardJokboOneAwayCountCut, 1) &&
          (
            core.oneAwayProb >= safeNum(P.goHardJokboOneAwayCut, 64) ||
            swingProb >= safeNum(P.goHardJokboOneAwaySwingCut, 0.62)
          )
        )
      )
    ) {
      hardReason = "late_risk_cut";
    } else if (
      core.goCount >= safeNum(P.goHardGoCountCap, 4) &&
      core.threat >= safeNum(P.goHardGoCountThreatCut, 0.72)
    ) {
      hardReason = "go_count_threat_cut";
    } else if (hardRiskScore >= safeNum(P.goHardRiskThreshold, 0.62)) {
      hardReason = "hard_risk_score";
    }

    if (
      hardReason &&
      (hardReason === "late_risk_cut" || hardReason === "hard_risk_score") &&
      flags.second &&
      flags.trailing &&
      !flags.oppCanStop &&
      core.selfPi >= flags.minPi + Math.max(0, Math.floor(safeNum(P.goSecondChasePiMargin, 1))) &&
      core.threat <= safeNum(P.goSecondChaseThreatCap, 0.72) &&
      core.oneAwayProb <= safeNum(P.goSecondChaseOneAwayCap, 62) &&
      swingProb <= safeNum(P.goSecondChaseSwingCap, 0.45) &&
      hardRiskScore <= safeNum(P.goHardRiskThreshold, 0.62) + safeNum(P.goSecondChaseHardRiskRelaxMargin, 0.08)
    ) {
      traceLayer(trace, "layer1_hard_relax", {
        relaxed: hardReason,
        mode: "second_chase_relax",
        swingProb,
        hardRiskScore
      });
      hardReason = "";
    }

    if (hardReason) {
      traceLayer(trace, "layer1_hard_stop", { reason: hardReason, swingProb, hardRiskScore, mode: goMode });
      return pushTrace(trace, false, hardReason, P, deps);
    }
  }

  // Layer 2 fast-pass: emergency comeback mode.
  if (
    flags.desperate &&
    core.selfPi >= (flags.minPi + 1) &&
    core.threat <= safeNum(P.goDesperateThreatCap, 0.62) &&
    core.oneAwayProb <= safeNum(P.goDesperateOneAwayCap, 48)
  ) {
    traceLayer(trace, "layer2_desperate", { fastPass: 1 });
    return pushTrace(trace, true, "desperate_fast_go", P, deps);
  }

  // Layer 2: compact value/risk composition.
  const oneAwayNorm = core.oneAwayProb / 100;
  const upside =
    Math.max(0, core.myScore - 6) * P.goUpsideScoreMul +
    core.selfPi * P.goUpsidePiMul +
    core.selfJokboTotal * P.goUpsideSelfJokboMul +
    core.selfJokboOneAway * P.goUpsideOneAwayMul +
    (flags.trailing ? P.goUpsideTrailBonus : 0);

  const risk =
    core.threat * P.goRiskPressureMul +
    oneAwayNorm * P.goRiskOneAwayMul +
    core.oppJokboTotal * P.goRiskOppJokboMul +
    core.oppJokboOneAway * P.goRiskOppOneAwayMul +
    core.goCount * P.goRiskGoCountMul +
    (core.deckCount <= P.phaseLateDeck ? P.goRiskLateDeckBonus : 0);

  const stopValue =
    Math.max(0, flags.lead) * P.stopLeadMul +
    Math.max(0, core.carry - 1) * P.stopCarryMul +
    (core.myScore >= 10 ? P.stopTenBonus : 0);

  let goValue =
    upside * safeNum(P.goUtilityUpsideWeight, 1.0) -
    risk * safeNum(P.goUtilityRiskWeight, 1.0) -
    stopValue * safeNum(P.goUtilityStopWeight, 1.0);

  if (goMode === "safe") {
    goValue -= safeNum(P.goModeSafeThresholdUp, 0.05) * 0.5;
  } else if (goMode === "chase") {
    goValue += safeNum(P.goModeChaseThresholdDown, 0.07) * 0.5;
  }

  goValue -= clamp(flags.lead / 10, -1, 1) * safeNum(P.goLiteScoreDiffMul, 0.04);
  if (flags.second && flags.trailing) goValue += P.goSecondTrailBonus;
  if (core.selfPi >= 8) goValue += P.goRallyPiWindowBonus;
  if (flags.second) goValue += P.goRallySecondBonus;
  if (flags.trailing) goValue += P.goRallyTrailBonus;
  if (core.deckCount <= 6) goValue += P.goRallyEndDeckBonus;
  if (
    core.myScore > 0 &&
    core.oppScore === 0 &&
    swingProb <= safeNum(P.goSafeGoSwingProbCut, 0.05)
  ) {
    goValue += safeNum(P.goSafeGoBonus, 0.14);
  }

  let threshold = P.goBaseThreshold;
  if (flags.leading) threshold += P.goThresholdLeadUp;
  if (flags.trailing) threshold -= P.goThresholdTrailDown;
  if (core.threat >= 0.72) threshold += P.goThresholdPressureUp;

  threshold += core.threat * safeNum(P.goLiteThreatPenaltyMul, 0.05);
  threshold += oneAwayNorm * safeNum(P.goLiteOneAwayPenaltyMul, 0.04);
  threshold += swingProb * safeNum(P.goLookaheadThresholdMul, 0.09);
  if (core.deckCount <= P.phaseLateDeck) {
    threshold += safeNum(P.goLiteLatePenalty, 0.045);
  } else {
    threshold -= safeNum(P.goDeckLowBonus, 0.02);
  }
  if (flags.oppCanStop) threshold += safeNum(P.goLiteOppCanStopPenalty, 0.1);
  if (flags.selfCanStop && !flags.trailing) threshold += safeNum(P.goLiteSelfCanStopPenalty, 0.01);

  threshold += Math.max(0, flags.lead) * safeNum(P.goScoreDiffBonus, 0.01);
  threshold -= Math.max(0, -flags.lead) * safeNum(P.goScoreDiffBonus, 0.01) * 0.5;
  if (core.oppPi >= 9 && core.selfPi <= flags.minPi + 1) threshold += safeNum(P.goUnseeHighPiPenalty, 0.04);

  if (
    flags.second &&
    flags.trailing &&
    !flags.oppCanStop &&
    core.selfPi >= flags.minPi + Math.max(0, Math.floor(safeNum(P.goSecondChasePiMargin, 1))) &&
    core.threat <= safeNum(P.goSecondChaseThreatCap, 0.72) &&
    core.oneAwayProb <= safeNum(P.goSecondChaseOneAwayCap, 62) &&
    swingProb <= safeNum(P.goSecondChaseSwingCap, 0.45)
  ) {
    threshold -= safeNum(P.goSecondChaseThresholdRelief, 0.045);
  }

  threshold *= Math.max(0.1, safeNum(P.goUtilityThresholdWeight, 1.0));

  if (
    flags.selfCanStop &&
    !flags.oppCanStop &&
    core.threat <= safeNum(P.goLiteSafeAttackThreatCap, 0.58) &&
    core.oneAwayProb <= safeNum(P.goLiteSafeAttackOneAwayCap, 45) &&
    core.deckCount >= safeNum(P.goLiteSafeAttackDeckMin, 5)
  ) {
    goValue += safeNum(P.goLiteSafeAttackBonus, 0.06);
  }

  traceLayer(trace, "layer2_utility", {
    mode: goMode,
    upside,
    risk,
    stopValue,
    swingProb,
    goValue,
    threshold,
    margin: goValue - threshold
  });

  if (goValue < threshold) {
    if (
      flags.second &&
      flags.trailing &&
      !flags.oppCanStop &&
      core.selfPi >= flags.minPi + Math.max(0, Math.floor(safeNum(P.goSecondChasePiMargin, 1))) &&
      core.threat <= safeNum(P.goSecondChaseThreatCap, 0.72) &&
      core.oneAwayProb <= safeNum(P.goSecondChaseOneAwayCap, 62) &&
      swingProb <= safeNum(P.goSecondChaseSwingCap, 0.45) &&
      goValue >= threshold - safeNum(P.goSecondChaseNearMargin, 0.03)
    ) {
      traceLayer(trace, "layer3_final", {
        mode: "second_chase_near",
        decision: 1,
        margin: goValue - threshold
      });
      return pushTrace(trace, true, "second_chase_near_pass", P, deps);
    }
    return pushTrace(trace, false, "utility_below_threshold", P, deps);
  }

  // Layer 3: soft guards for high-pi and trailing windows.
  if (core.selfPi >= 9) {
    if (
      core.threat >= safeNum(P.goSoftHighPiThreatCap, 0.9) ||
      core.oneAwayProb >= safeNum(P.goSoftHighPiOneAwayCap, 66)
    ) {
      return pushTrace(trace, false, "soft_cap_high_pi", P, deps);
    }
    const ok = goValue >= threshold + safeNum(P.goSoftHighPiMargin, 0.06);
    traceLayer(trace, "layer3_final", { mode: "high_pi", decision: ok ? 1 : 0 });
    return pushTrace(trace, ok, ok ? "high_pi_margin_pass" : "high_pi_margin_fail", P, deps);
  }

  if (flags.trailing && core.selfPi >= 8) {
    const ok = goValue >= threshold + safeNum(P.goSoftTrailHighPiMargin, 0.04);
    traceLayer(trace, "layer3_final", { mode: "trail_high_pi", decision: ok ? 1 : 0 });
    return pushTrace(trace, ok, ok ? "trail_high_pi_pass" : "trail_high_pi_fail", P, deps);
  }

  if (core.selfPi >= 8) {
    const ok = goValue >= threshold + safeNum(P.goSoftValueMargin, 0.02);
    traceLayer(trace, "layer3_final", { mode: "high_pi_soft", decision: ok ? 1 : 0 });
    return pushTrace(trace, ok, ok ? "high_pi_soft_pass" : "high_pi_soft_fail", P, deps);
  }

  traceLayer(trace, "layer3_final", { mode: "base", decision: 1 });
  return pushTrace(trace, true, "base_pass", P, deps);
}

// Choose best bomb month by immediate board gain.
export function selectBombMonthGPT(state, _playerKey, bombMonths, deps) {
  if (!Array.isArray(bombMonths) || bombMonths.length <= 0) return null;
  let best = bombMonths[0];
  let bestScore = safeNum(deps?.monthBoardGain?.(state, best));
  for (const month of bombMonths.slice(1)) {
    const score = safeNum(deps?.monthBoardGain?.(state, month));
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
  const P = mergedParams(params);
  const profile = buildProfile(state, playerKey, deps, P);

  const planRaw = typeof deps?.getFirstTurnDoublePiPlan === "function"
    ? deps.getFirstTurnDoublePiPlan(state, playerKey)
    : null;
  const planMonths = asMonthSet(planRaw?.months);
  if (Boolean(planRaw?.active) && bombMonths.some((m) => planMonths.has(Number(m)))) return true;

  const month = selectBombMonthGPT(state, playerKey, bombMonths, deps);
  if (month == null) return false;

  const impact = typeof deps?.isHighImpactBomb === "function"
    ? deps.isHighImpactBomb(state, playerKey, month)
    : null;
  const immediate = safeNum(impact?.immediateGain);
  const boardGain = safeNum(deps?.monthBoardGain?.(state, month));
  const value =
    immediate * P.bombImmediateMul +
    boardGain * P.bombBoardGainMul +
    (impact?.highImpact ? P.bombHighImpactBonus : 0) +
    (profile.trailing ? P.bombTrailBonus : 0) -
    profile.signals.threat * P.bombRiskMul;

  if (safeNum(profile.ctx?.defenseOpening) > 0 && !impact?.highImpact) {
    return value >= P.bombDefenseThreshold;
  }
  return value >= P.bombThreshold;
}

// Shaking decision + selected month payload.
export function decideShakingGPT(state, playerKey, shakingMonths, deps, params = DEFAULT_PARAMS) {
  if (!Array.isArray(shakingMonths) || shakingMonths.length <= 0) {
    return { allow: false, month: null, score: -Infinity };
  }

  const P = mergedParams(params);
  const profile = buildProfile(state, playerKey, deps, P);
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

    let score =
      immediate * P.shakeImmediateMul +
      combo * P.shakeComboMul +
      (impact?.highImpact ? P.shakeImpactBonus : 0) +
      (impact?.hasDoublePiLine ? P.shakePiLineBonus : 0) +
      (impact?.directThreeGwang ? P.shakeDirectGwangBonus : 0) -
      profile.signals.threat * P.shakeRiskMul;

    if (profile.trailing) score += P.shakeTrailingBonus;
    if (Boolean(planRaw?.active) && planMonths.has(Number(month))) score += P.shakeFirstPlanBonus;
    if (known <= 2) score += P.shakeKnownLowBonus;
    if (known >= 4) score -= P.shakeKnownHighPenalty;

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
  if (profile.leading) threshold += P.shakeLeadThresholdUp;
  if (profile.signals.threat >= 0.72) threshold += P.shakePressureThresholdUp;
  if (profile.trailing) threshold -= 0.08;

  return { ...best, allow: best.score >= threshold };
}

// President hold/stop option gate.
export function shouldPresidentStopGPT(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const P = mergedParams(params);
  const profile = buildProfile(state, playerKey, deps, P);
  const carry = safeNum(state?.carryOverMultiplier, 1);
  if (profile.lead >= P.presidentStopLead && carry <= P.presidentCarryStopMax) return true;
  if (profile.signals.threat >= P.presidentThreatStop && profile.lead >= 1) return true;
  return false;
}

// Gukjin mode selector.
export function chooseGukjinHeuristicGPT(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const P = mergedParams(params);
  const ctx = typeof deps?.analyzeGameContext === "function"
    ? deps.analyzeGameContext(state, playerKey)
    : {};
  const selfFive = safeNum(ctx?.selfFive);
  const oppFive = safeNum(ctx?.oppFive);

  if (selfFive <= 0 && oppFive >= 6) return "junk";
  if (selfFive >= 7 && oppFive <= 1) return "five";

  const branch = typeof deps?.analyzeGukjinBranches === "function"
    ? deps.analyzeGukjinBranches(state, playerKey)
    : null;

  if (branch?.enabled && Array.isArray(branch.scenarios) && branch.scenarios.length > 0) {
    let bestMode = "junk";
    let bestScore = -Infinity;
    for (const s of branch.scenarios) {
      const score =
        (safeNum(s?.myScore) - safeNum(s?.oppScore)) * P.gukjinScoreDiffMul +
        (safeNum(s?.myPi) - safeNum(s?.oppPi)) * P.gukjinPiDiffMul +
        (s?.canMongBakSelf ? P.gukjinMongBakBonus : 0) -
        (s?.mongRiskSelf ? P.gukjinMongRiskPenalty : 0);
      if (score > bestScore) {
        bestScore = score;
        bestMode = s?.selfMode === "five" ? "five" : "junk";
      }
    }
    return bestMode;
  }

  return selfFive >= 6 ? "five" : "junk";
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
