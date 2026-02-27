// heuristicV6.js - Matgo Heuristic V6
/* ============================================================================
 * Heuristic V6
 * - risk-adjusted expected utility
 * - phase profile + pressure model + optional rollout blending
 * - exported decisions: rank / match / go / bomb / shaking / president / gukjin
 * ========================================================================== */
const GUKJIN_CARD_ID = "I0";

/* 1) Tunable parameter set */
export const DEFAULT_PARAMS = {
  // phase
  phaseEarlyDeck: 15,
  phaseLateDeck: 8,
  phaseEndDeck: 4,

  // profile weights
  attackBase: 1.0,
  defenseBase: 1.0,
  riskBase: 1.0,
  tempoBase: 1.0,
  trailingAttackBoost: 0.18,
  trailingTempoBoost: 0.2,
  leadingDefenseBoost: 0.12,
  leadingRiskBoost: 0.15,
  highPressureDefenseBoost: 0.2,
  highPressureRiskBoost: 0.22,
  secondMoverTempoBoost: 0.08,

  // card utility
  noMatchBase: -6.5,
  matchOneBase: 7.0,
  matchTwoBase: 11.5,
  matchThreeBase: 13.5,
  captureGainMul: 1.15,
  kwangCaptureBonus: 6.0,
  fiveCaptureBonus: 4.4,
  ribbonCaptureBonus: 2.0,
  junkPiMul: 4.8,
  selfPiWindowMul: 1.5,
  oppPiWindowMul: 1.2,
  doublePiBonus: 5.0,
  doublePiNoMatchHoldPenalty: 4.2,
  doublePiMonthPairHoldPenalty: 2.8,
  doublePiMonthTripleHoldPenalty: 2.2,
  doublePiPairMonthHoldPenalty: 1.8,
  doublePiMonthAnchorHoldPenalty: 2.0,
  doublePiHoldRiskRelease: 8.5,
  doublePiHoldRiskReleaseMul: 0.45,
  comboOpportunityMul: 4.2,
  blockBase: 5.5,
  blockUrgencyMul: 1.6,
  blockThreatMul: 3.0,
  blockNoMatchPenalty: 4.8,
  firstTurnPlanBonus: 6.5,
  knownMonthSafeBonus: 2.5,
  unknownMonthPenalty: 1.8,
  trailPiTempoMul: 1.8,
  leadNoMatchTempoPenalty: 2.2,
  endgameSafeDiscardBonus: 1.5,
  endgameUnknownPenalty: 2.2,
  bonusCardUseBase: 3.6,
  bonusCardStealPiMul: 1.7,
  bonusCardExtraTurnTempo: 1.2,
  bonusCardEarlyHoldBias: 1.0,
  bonusCardLateUseBonus: 1.4,
  bonusCardHoldPenaltyMul: 0.12,
  bonusCardRiskMul: 0.18,
  bonusCardOppPiEmptyPenalty: 0.6,

  // risk
  feedRiskNoMatchMul: 4.8,
  feedRiskMatchMul: 1.1,
  dangerNoMatchMul: 2.6,
  dangerMatchMul: 0.8,
  releaseRiskFloor: 0.6,
  releaseRiskMul: 8.0,
  pukRiskMul: 3.5,
  pukOpportunityMul: 1.4,

  // choose-match
  chooseMatchBaseMul: 1.0,
  chooseMatchPiMul: 5.0,
  chooseMatchKwangBonus: 5.0,
  chooseMatchFiveBonus: 3.5,
  chooseMatchRibbonBonus: 1.8,
  chooseMatchBlockMul: 1.8,
  chooseMatchComboMul: 2.8,
  chooseMatchOppShakeMonthBonus: 1.05,

  // go model
  goMinPi: 5,
  goMinPiDesperate: 4,
  goMinPiSecondTrailingDelta: 1,
  goHardThreatCut: 1.0,
  goHardThreatDeckCut: 7,
  goHardOppFiveCut: 7,
  goHardOppScoreCut: 8,
  goHardLateOneAwayCut: 70,
  goHardLateOneAwayDeckCut: 8,
  goHardGoCountCap: 3,
  goHardGoCountThreatCut: 0.72,
  goUpsideScoreMul: 0.1,
  goUpsidePiMul: 0.045,
  goUpsideSelfJokboMul: 0.5,
  goUpsideOneAwayMul: 0.12,
  goUpsideTrailBonus: 0.12,
  goRiskPressureMul: 0.42,
  goRiskOneAwayMul: 0.30,
  goRiskOppJokboMul: 0.34,
  goRiskOppOneAwayMul: 0.08,
  goRiskGoCountMul: 0.11,
  goRiskLateDeckBonus: 0.12,
  stopLeadMul: 0.10,
  stopCarryMul: 0.13,
  stopTenBonus: 0.22,
  goBaseThreshold: 0.10,
  goThresholdLeadUp: 0.10,
  goThresholdTrailDown: 0.03,
  goThresholdPressureUp: 0.08,
  goSecondTrailBonus: 0.05,
  goRallyPiWindowBonus: 0.02,
  goRallySecondBonus: 0.01,
  goRallyTrailBonus: 0.02,
  goRallyEndDeckBonus: 0.00,
  goSoftHighPiThreatCap: 0.90,
  goSoftHighPiOneAwayCap: 66,
  goSoftHighPiMargin: 0.06,
  goSoftTrailHighPiMargin: 0.04,
  goSoftValueMargin: 0.02,

  // rollout (2-ply)
  rolloutEnabled: 1,
  rolloutTopK: 3,
  rolloutMaxSteps: 28,
  rolloutSamples: 5,
  rolloutCardWeight: 0.85,
  rolloutGoWeight: 0.24,
  rolloutGoDeltaCap: 0.4,

  // bomb
  bombImmediateMul: 0.9,
  bombBoardGainMul: 0.8,
  bombHighImpactBonus: 3.2,
  bombTrailBonus: 1.2,
  bombRiskMul: 1.0,
  bombThreshold: 3.5,
  bombDefenseThreshold: 5.0,

  // shaking
  shakeImmediateMul: 1.4,
  shakeComboMul: 1.1,
  shakeImpactBonus: 0.7,
  shakePiLineBonus: 0.45,
  shakeDirectGwangBonus: 0.45,
  shakeKnownLowBonus: 0.25,
  shakeKnownHighPenalty: 0.2,
  shakeRiskMul: 0.55,
  shakeTrailingBonus: 0.22,
  shakeFirstPlanBonus: 0.28,
  shakeThreshold: 0.7,
  shakeLeadThresholdUp: 0.18,
  shakePressureThresholdUp: 0.15,

  // opponent shaking awareness (small tuning)
  oppShakeRecentWindow: 8,
  oppShakeBlockBonus: 1.15,
  oppShakeNoMatchRiskBonus: 0.45,

  // president and gukjin
  presidentStopLead: 3,
  presidentCarryStopMax: 1,
  gukjinScoreDiffMul: 1.0,
  gukjinPiDiffMul: 0.22,
  gukjinMongBakBonus: 1.8,
  gukjinMongRiskPenalty: 2.2
};

/* 2) Core numeric/card helpers */
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

function isSecondMover(state, playerKey) {
  const first = state?.startingTurnKey;
  if (first !== "human" && first !== "ai") return false;
  return first !== playerKey;
}

function piValue(card, deps) {
  if (!card) return 0;
  if (card.id === GUKJIN_CARD_ID) return 2;
  if (card.category !== "junk") return 0;
  return safeNum(deps.junkPiValue?.(card));
}

function isDoublePi(card, deps) {
  return piValue(card, deps) >= 2;
}

function comboTagCount(cards, tag) {
  return (cards || []).reduce((sum, card) => {
    if (Array.isArray(card?.comboTags) && card.comboTags.includes(tag)) return sum + 1;
    return sum;
  }, 0);
}

/* 3) Pressure and phase profile builders */
function calcOpponentPressure(state, playerKey, deps) {
  const opp = deps.otherPlayerKey(playerKey);
  const jokbo = deps.checkOpponentJokboProgress(state, playerKey);
  const prog = safeNum(deps.opponentThreatScore(state, playerKey));
  const next = safeNum(deps.nextTurnThreatScore(state, playerKey));
  const board = safeNum(deps.boardHighValueThreatForPlayer?.(state, opp));
  const matchable = safeNum(deps.matchableMonthCountForPlayer?.(state, opp));
  const deck = safeNum(state.deck?.length);

  const threatRaw = prog * 0.45 + safeNum(jokbo?.threat) * 0.35 + next * 0.2 + board * 0.12 + matchable * 0.04;
  const threat = clamp(threatRaw, 0, 1.6);

  let oneAwayProb = prog * 38 + safeNum(jokbo?.threat) * 32 + next * 20 + board * 10 + matchable * 4;
  if (deck <= 10) oneAwayProb += 6;
  if (deck <= 6) oneAwayProb += 8;
  oneAwayProb = clamp(oneAwayProb, 0, 100);

  return {
    threat,
    oneAwayProb,
    jokboThreat: safeNum(jokbo?.threat),
    monthUrgency: jokbo?.monthUrgency || new Map(),
    matchableMonths: matchable,
    deckCount: deck
  };
}

function classifyPhase(deckCount, myScore, oppScore, P) {
  if (deckCount <= P.phaseEndDeck) return "end";
  if (deckCount <= P.phaseLateDeck) return "late";
  if (deckCount >= P.phaseEarlyDeck && myScore + oppScore <= 9) return "early";
  return "mid";
}

function buildProfile(state, playerKey, deps, P) {
  const ctx = deps.analyzeGameContext(state, playerKey);
  const myScore = safeNum(ctx.myScore);
  const oppScore = safeNum(ctx.oppScore);
  const pressure = calcOpponentPressure(state, playerKey, deps);
  const phase = classifyPhase(pressure.deckCount, myScore, oppScore, P);
  const trailing = myScore < oppScore;
  const leading = myScore > oppScore;
  const second = isSecondMover(state, playerKey);

  let attack = P.attackBase;
  let defense = P.defenseBase;
  let risk = P.riskBase;
  let tempo = P.tempoBase;

  if (trailing) {
    attack += P.trailingAttackBoost;
    tempo += P.trailingTempoBoost;
  }
  if (leading) {
    defense += P.leadingDefenseBoost;
    risk += P.leadingRiskBoost;
  }
  if (pressure.threat >= 0.72) {
    defense += P.highPressureDefenseBoost;
    risk += P.highPressureRiskBoost;
  }
  if (second) tempo += P.secondMoverTempoBoost;
  if (phase === "end") risk += 0.08;
  if (phase === "early") attack += 0.05;

  return {
    ctx,
    pressure,
    phase,
    trailing,
    leading,
    second,
    attack,
    defense,
    risk,
    tempo
  };
}

function captureCategoryBonus(cards, P) {
  let bonus = 0;
  for (const c of cards || []) {
    if (c?.category === "kwang") bonus += P.kwangCaptureBonus;
    else if (c?.category === "five") bonus += P.fiveCaptureBonus;
    else if (c?.category === "ribbon") bonus += P.ribbonCaptureBonus;
  }
  return bonus;
}

function scoreComboOpportunity(state, playerKey, month, deps) {
  return safeNum(deps.ownComboOpportunityScore?.(state, playerKey, month));
}

function getRecentOpponentShaking(state, playerKey, deps, recentWindow) {
  const kibo = state?.kibo || [];
  if (!Array.isArray(kibo) || kibo.length <= 0) {
    return { active: false, month: null, delta: Infinity };
  }
  const opp = deps.otherPlayerKey(playerKey);
  const window = Math.max(1, Math.floor(safeNum(recentWindow, 8)));
  const nowNo = Math.max(0, Math.floor(safeNum(state?.kiboSeq, kibo.length)));

  for (let i = kibo.length - 1; i >= 0; i -= 1) {
    const e = kibo[i];
    if (e?.type !== "shaking_declare") continue;
    if (e?.playerKey !== opp) continue;
    const month = Number(e?.month);
    if (!Number.isInteger(month) || month < 1 || month > 12) {
      return { active: false, month: null, delta: Infinity };
    }
    const evtNo = Math.max(0, Math.floor(safeNum(e?.no, nowNo)));
    const delta = Math.max(0, nowNo - evtNo);
    if (delta > window) return { active: false, month: null, delta };
    return { active: true, month, delta };
  }
  return { active: false, month: null, delta: Infinity };
}

function knownMonthCount(month, boardCountByMonth, handCountByMonth, capturedByMonth) {
  return (
    safeNum(boardCountByMonth.get(month)) +
    safeNum(handCountByMonth.get(month)) +
    safeNum(capturedByMonth.get(month))
  );
}

/* 4) Card utility model (single-card evaluation core) */
function evaluateCardUtility(state, playerKey, card, deps, P, profile, cache) {
  const matches = cache.boardByMonth.get(card.month) || [];
  const captureGain = matches.reduce((sum, m) => sum + safeNum(deps.cardCaptureValue(m)), 0);
  const cardIsDoublePi = isDoublePi(card, deps);
  const bonusStealPi = safeNum(card?.bonus?.stealPi);
  const isBonusCard = bonusStealPi > 0;
  const sameMonthInHand = safeNum(cache.handCountByMonth.get(card.month));
  const sameMonthDoublePiInHand = safeNum(cache.handDoublePiCountByMonth.get(card.month));
  const allCaptured = [card, ...matches];
  const piGain = allCaptured.reduce((sum, c) => sum + piValue(c, deps), 0);
  const selfPi = safeNum(profile.ctx.selfPi, deps.capturedCountByCategory(state.players?.[playerKey], "junk"));
  const opp = deps.otherPlayerKey(playerKey);
  const oppPi = safeNum(profile.ctx.oppPi, deps.capturedCountByCategory(state.players?.[opp], "junk"));

  let immediate =
    matches.length === 0 ? P.noMatchBase :
    matches.length === 1 ? P.matchOneBase :
    matches.length === 2 ? P.matchTwoBase :
    P.matchThreeBase;

  immediate += captureGain * P.captureGainMul;
  immediate += captureCategoryBonus(matches, P);
  immediate += piGain * P.junkPiMul;

  if (selfPi >= 7 && selfPi <= 9) immediate += piGain * P.selfPiWindowMul;
  if (oppPi <= 5) immediate += piGain * P.oppPiWindowMul;
  if (matches.length > 0 && allCaptured.some((c) => isDoublePi(c, deps))) immediate += P.doublePiBonus;
  if (isBonusCard) {
    immediate += P.bonusCardUseBase + bonusStealPi * P.bonusCardStealPiMul;
    if (profile.phase === "late" || profile.phase === "end") immediate += P.bonusCardLateUseBonus;
    if (oppPi <= 0) immediate -= P.bonusCardOppPiEmptyPenalty;
  }

  const comboOpportunity = scoreComboOpportunity(state, playerKey, card.month, deps);
  immediate += comboOpportunity * P.comboOpportunityMul;

  let risk = 0;
  let deny = 0;
  if (cache.blockMonths.has(card.month)) {
    const urgency =
      Math.max(
        safeNum(cache.blockUrgency.get(card.month)),
        safeNum(profile.pressure.monthUrgency.get(card.month)) / 10
      );
    deny += P.blockBase + urgency * P.blockUrgencyMul + profile.pressure.threat * P.blockThreatMul;
    if (matches.length === 0) deny -= P.blockNoMatchPenalty;
  }

  if (cache.recentOppShake.active && cache.recentOppShake.month === card.month) {
    const freshness = clamp(
      1 - safeNum(cache.recentOppShake.delta) / Math.max(1, safeNum(P.oppShakeRecentWindow, 8)),
      0.2,
      1.0
    );
    if (matches.length > 0) {
      deny += P.oppShakeBlockBonus * freshness;
    } else {
      risk += P.oppShakeNoMatchRiskBonus * freshness;
    }
  }

  let tempo = 0;
  const known = knownMonthCount(card.month, cache.boardCountByMonth, cache.handCountByMonth, cache.capturedByMonth);
  if (matches.length === 0) {
    if (known >= 3) tempo += P.knownMonthSafeBonus;
    else if (known <= 1) tempo -= P.unknownMonthPenalty;
  }
  if (profile.trailing && piGain > 0) tempo += piGain * P.trailPiTempoMul;
  if (profile.leading && matches.length === 0) tempo -= P.leadNoMatchTempoPenalty;
  if (profile.phase === "end" && matches.length === 0) {
    if (known >= 3) tempo += P.endgameSafeDiscardBonus;
    else tempo -= P.endgameUnknownPenalty;
  }
  if (isBonusCard) {
    tempo += P.bonusCardExtraTurnTempo;
    if (profile.phase === "early" && oppPi <= 2) tempo -= P.bonusCardEarlyHoldBias;
  }
  if (cache.firstTurnPlan.active && cache.firstTurnPlan.months.has(card.month)) tempo += P.firstTurnPlanBonus;

  const feedRisk = safeNum(deps.estimateOpponentImmediateGainIfDiscard(state, playerKey, card.month));
  risk += feedRisk * (matches.length === 0 ? P.feedRiskNoMatchMul : P.feedRiskMatchMul);

  const danger = safeNum(
    deps.estimateDangerMonthRisk?.(
      state,
      playerKey,
      card.month,
      cache.boardCountByMonth,
      cache.handCountByMonth,
      cache.capturedByMonth
    )
  );
  risk += danger * (matches.length === 0 ? P.dangerNoMatchMul : P.dangerMatchMul);

  const releasePunishProb = safeNum(
    deps.estimateReleasePunishProb?.(
      state,
      playerKey,
      card.month,
      cache.jokboThreat,
      profile.ctx
    )
  );
  if (matches.length === 0 && releasePunishProb >= P.releaseRiskFloor) {
    risk += (releasePunishProb - P.releaseRiskFloor) * P.releaseRiskMul;
  }

  const pukRisk = safeNum(
    deps.isRiskOfPuk(
      state,
      playerKey,
      card,
      cache.boardCountByMonth,
      cache.handCountByMonth
    )
  );
  if (pukRisk > 0) risk += pukRisk * P.pukRiskMul;
  else if (pukRisk < 0) immediate += -pukRisk * P.pukOpportunityMul;

  let holdPenalty = 0;
  if (matches.length === 0) {
    if (cardIsDoublePi) {
      holdPenalty += P.doublePiNoMatchHoldPenalty;
      if (sameMonthInHand >= 2) holdPenalty += P.doublePiMonthPairHoldPenalty;
      if (sameMonthInHand >= 3) holdPenalty += P.doublePiMonthTripleHoldPenalty;
    }
    if (sameMonthDoublePiInHand >= 2) holdPenalty += P.doublePiPairMonthHoldPenalty;
    if (!cardIsDoublePi && sameMonthInHand >= 2 && sameMonthDoublePiInHand >= 1) {
      holdPenalty += P.doublePiMonthAnchorHoldPenalty;
    }
    if (risk >= P.doublePiHoldRiskRelease) {
      holdPenalty *= P.doublePiHoldRiskReleaseMul;
    }
  }
  if (isBonusCard) {
    risk *= P.bonusCardRiskMul;
    holdPenalty *= P.bonusCardHoldPenaltyMul;
  }

  const score =
    immediate * profile.attack +
    deny * profile.defense +
    tempo * profile.tempo -
    risk * profile.risk -
    holdPenalty;

  return {
    card,
    score,
    matches: matches.length,
    releasePunishProb
  };
}

/* 5) Exported policy decisions */
export function rankHandCardsV6(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const player = state.players?.[playerKey];
  if (!player?.hand?.length) return [];

  const P = mergedParams(params);
  const opp = deps.otherPlayerKey(playerKey);
  const profile = buildProfile(state, playerKey, deps, P);
  const jokboThreat = deps.checkOpponentJokboProgress(state, playerKey);

  const cache = {
    boardByMonth: deps.boardMatchesByMonth(state),
    boardCountByMonth: deps.monthCounts(state.board || []),
    handCountByMonth: deps.monthCounts(player.hand || []),
    handDoublePiCountByMonth: (() => {
      const out = new Map();
      for (const c of player.hand || []) {
        if (!isDoublePi(c, deps)) continue;
        out.set(c.month, safeNum(out.get(c.month)) + 1);
      }
      return out;
    })(),
    capturedByMonth: deps.capturedMonthCounts(state),
    blockMonths: deps.blockingMonthsAgainst(state.players?.[opp], player),
    blockUrgency: deps.blockingUrgencyByMonth(state.players?.[opp], player),
    firstTurnPlan: deps.getFirstTurnDoublePiPlan(state, playerKey),
    recentOppShake: getRecentOpponentShaking(state, playerKey, deps, P.oppShakeRecentWindow),
    jokboThreat
  };

  const ranked = player.hand.map((card) => evaluateCardUtility(state, playerKey, card, deps, P, profile, cache));
  ranked.sort((a, b) => b.score - a.score);

  if (safeNum(P.rolloutEnabled, 1) > 0 && typeof deps.rolloutCardValueV6 === "function") {
    const topK = Math.max(1, Math.min(ranked.length, Math.floor(safeNum(P.rolloutTopK, 3))));
    const baseline =
      typeof deps.rolloutStateUtilityV6 === "function" ? safeNum(deps.rolloutStateUtilityV6(state, playerKey)) : 0;
    for (let i = 0; i < topK; i += 1) {
      const cand = ranked[i];
      const rv = deps.rolloutCardValueV6(state, playerKey, cand.card?.id, {
        maxSteps: Math.floor(safeNum(P.rolloutMaxSteps, 28)),
        samples: Math.floor(safeNum(P.rolloutSamples, 5))
      });
      if (Number.isFinite(rv)) {
        cand.score += (rv - baseline) * safeNum(P.rolloutCardWeight, 0.85);
      }
    }
    ranked.sort((a, b) => b.score - a.score);
  }

  return ranked;
}

/* choose-match policy for pending TWO-match selections */
export function chooseMatchHeuristicV6(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const ids = state.pendingMatch?.boardCardIds || [];
  if (!ids.length) return null;

  const P = mergedParams(params);
  const profile = buildProfile(state, playerKey, deps, P);
  const opp = deps.otherPlayerKey(playerKey);
  const blockMonths = deps.blockingMonthsAgainst(state.players?.[opp], state.players?.[playerKey]);
  const blockUrgency = deps.blockingUrgencyByMonth(state.players?.[opp], state.players?.[playerKey]);
  const recentOppShake = getRecentOpponentShaking(state, playerKey, deps, P.oppShakeRecentWindow);

  let bestId = null;
  let bestScore = -Infinity;

  for (const card of (state.board || []).filter((c) => ids.includes(c.id))) {
    const pi = piValue(card, deps);
    let score = safeNum(deps.cardCaptureValue(card)) * P.chooseMatchBaseMul + pi * P.chooseMatchPiMul;
    if (card.category === "kwang") score += P.chooseMatchKwangBonus;
    if (card.category === "five") score += P.chooseMatchFiveBonus;
    if (card.category === "ribbon") score += P.chooseMatchRibbonBonus;
    score += scoreComboOpportunity(state, playerKey, card.month, deps) * P.chooseMatchComboMul;

    if (blockMonths.has(card.month)) {
      const urgency = Math.max(
        safeNum(blockUrgency.get(card.month)),
        safeNum(profile.pressure.monthUrgency.get(card.month)) / 10
      );
      score += (urgency + profile.pressure.threat) * P.chooseMatchBlockMul;
    }
    if (recentOppShake.active && recentOppShake.month === card.month) {
      const freshness = clamp(
        1 - safeNum(recentOppShake.delta) / Math.max(1, safeNum(P.oppShakeRecentWindow, 8)),
        0.2,
        1.0
      );
      score += P.chooseMatchOppShakeMonthBonus * freshness;
    }

    score += safeNum(deps.monthStrategicPriority?.(card.month)) * 0.22;
    if (score > bestScore) {
      bestScore = score;
      bestId = card.id;
    }
  }

  return bestId;
}

/* GO/STOP utility model with optional rollout delta blending */
export function shouldGoV6(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const P = mergedParams(params);
  if (deps.canBankruptOpponentByStop?.(state, playerKey)) return false;

  const player = state.players?.[playerKey];
  const opp = deps.otherPlayerKey(playerKey);
  const oppPlayer = state.players?.[opp];
  const profile = buildProfile(state, playerKey, deps, P);
  const ctx = profile.ctx;
  const pressure = profile.pressure;
  const myScore = safeNum(ctx.myScore);
  const oppScore = safeNum(ctx.oppScore);
  const selfPi = safeNum(ctx.selfPi, deps.capturedCountByCategory(player, "junk"));
  const oppPi = safeNum(ctx.oppPi, deps.capturedCountByCategory(oppPlayer, "junk"));
  const selfFive = safeNum(ctx.selfFive);
  const oppFive = safeNum(ctx.oppFive);
  const goCount = safeNum(player?.goCount);
  const carry = safeNum(state.carryOverMultiplier, 1);
  const goldRisk = deps.goldRiskProfile?.(state, playerKey) || { selfLow: false, oppLow: false };
  const desperateGo = !!goldRisk.selfLow && !goldRisk.oppLow;

  let minPi = desperateGo ? P.goMinPiDesperate : P.goMinPi;
  if (profile.second && profile.trailing) minPi -= P.goMinPiSecondTrailingDelta;
  minPi = Math.max(3, minPi);
  if (selfPi < minPi) return false;

  if (!desperateGo) {
    if (oppScore >= P.goHardOppScoreCut && myScore <= oppScore + 1) return false;
    if (selfFive === 0 && oppFive >= P.goHardOppFiveCut) return false;
    if (pressure.threat >= P.goHardThreatCut && pressure.deckCount <= P.goHardThreatDeckCut) return false;
    if (pressure.oneAwayProb >= P.goHardLateOneAwayCut && pressure.deckCount <= P.goHardLateOneAwayDeckCut) {
      return false;
    }
    if (goCount >= P.goHardGoCountCap && pressure.threat >= P.goHardGoCountThreatCut) return false;
  }

  const selfJokbo = deps.estimateJokboExpectedPotential?.(state, playerKey, opp) || { total: 0, oneAwayCount: 0 };
  const oppJokbo =
    deps.estimateOpponentJokboExpectedPotential?.(state, playerKey) ||
    { total: 0, oneAwayCount: 0 };

  const upside =
    Math.max(0, myScore - 6) * P.goUpsideScoreMul +
    selfPi * P.goUpsidePiMul +
    safeNum(selfJokbo.total) * P.goUpsideSelfJokboMul +
    safeNum(selfJokbo.oneAwayCount) * P.goUpsideOneAwayMul +
    (profile.trailing ? P.goUpsideTrailBonus : 0) +
    (oppPi <= 5 ? 0.03 : 0);

  const oneAwayProbNorm = pressure.oneAwayProb / 100;
  const risk =
    pressure.threat * P.goRiskPressureMul +
    oneAwayProbNorm * P.goRiskOneAwayMul +
    safeNum(oppJokbo.total) * P.goRiskOppJokboMul +
    safeNum(oppJokbo.oneAwayCount) * P.goRiskOppOneAwayMul +
    goCount * P.goRiskGoCountMul +
    (pressure.deckCount <= P.phaseLateDeck ? P.goRiskLateDeckBonus : 0);

  const stopValueRaw =
    Math.max(0, myScore - oppScore) * P.stopLeadMul +
    Math.max(0, carry - 1) * P.stopCarryMul +
    (myScore >= 10 ? P.stopTenBonus : 0);
  const stopValue = stopValueRaw * (profile.leading ? 1 : 0.35);

  let threshold = P.goBaseThreshold;
  if (profile.leading) threshold += P.goThresholdLeadUp;
  if (profile.trailing) threshold -= P.goThresholdTrailDown;
  if (pressure.threat >= 0.72) threshold += P.goThresholdPressureUp;

  let goValue = upside - risk - stopValue;
  if (profile.second && profile.trailing) goValue += P.goSecondTrailBonus;
  if (selfPi >= 8) goValue += P.goRallyPiWindowBonus;
  if (profile.second) goValue += P.goRallySecondBonus;
  if (profile.trailing) goValue += P.goRallyTrailBonus;
  if (pressure.deckCount <= 6) goValue += P.goRallyEndDeckBonus;

  if (safeNum(P.rolloutEnabled, 1) > 0 && typeof deps.rolloutGoStopValueV6 === "function") {
    const goRv = deps.rolloutGoStopValueV6(state, playerKey, true, {
      maxSteps: Math.floor(safeNum(P.rolloutMaxSteps, 28)),
      samples: Math.floor(safeNum(P.rolloutSamples, 5))
    });
    const stopRv = deps.rolloutGoStopValueV6(state, playerKey, false, {
      maxSteps: Math.floor(safeNum(P.rolloutMaxSteps, 28)),
      samples: Math.floor(safeNum(P.rolloutSamples, 5))
    });
    if (Number.isFinite(goRv) && Number.isFinite(stopRv)) {
      const deltaCap = Math.max(0.05, safeNum(P.rolloutGoDeltaCap, 0.4));
      const rolloutDelta = clamp(goRv - stopRv, -deltaCap, deltaCap);
      goValue += rolloutDelta * safeNum(P.rolloutGoWeight, 0.24);
    }
  }

  if (goValue < threshold) return false;

  if (selfPi >= 9) {
    if (
      pressure.threat >= safeNum(P.goSoftHighPiThreatCap, 0.90) ||
      pressure.oneAwayProb >= safeNum(P.goSoftHighPiOneAwayCap, 66)
    ) {
      return false;
    }
    return goValue >= threshold + safeNum(P.goSoftHighPiMargin, 0.06);
  }

  if (profile.trailing && selfPi >= 8 && pressure.threat < 1.0 && pressure.oneAwayProb < 72) {
    return goValue >= threshold + safeNum(P.goSoftTrailHighPiMargin, 0.04);
  }

  if (selfPi >= 8 && pressure.threat < 0.95) {
    return goValue >= threshold + safeNum(P.goSoftValueMargin, 0.02);
  }

  return true;
}

/* Bomb month picker + bomb gate */
export function selectBombMonthV6(state, _playerKey, bombMonths, deps) {
  if (!bombMonths?.length) return null;
  let best = bombMonths[0];
  let bestScore = safeNum(deps.monthBoardGain(state, best));
  for (const month of bombMonths.slice(1)) {
    const score = safeNum(deps.monthBoardGain(state, month));
    if (score > bestScore) {
      bestScore = score;
      best = month;
    }
  }
  return best;
}

export function shouldBombV6(state, playerKey, bombMonths, deps, params = DEFAULT_PARAMS) {
  if (!bombMonths?.length) return false;
  const P = mergedParams(params);
  const profile = buildProfile(state, playerKey, deps, P);
  const ctx = profile.ctx;
  const plan = deps.getFirstTurnDoublePiPlan(state, playerKey);
  if (plan?.active && bombMonths.some((m) => plan.months.has(m))) return true;

  const bestMonth = selectBombMonthV6(state, playerKey, bombMonths, deps);
  if (bestMonth == null) return false;

  const impact = deps.isHighImpactBomb(state, playerKey, bestMonth);
  const immediateGain = safeNum(impact?.immediateGain);
  const boardGain = safeNum(deps.monthBoardGain(state, bestMonth));
  const value =
    immediateGain * P.bombImmediateMul +
    boardGain * P.bombBoardGainMul +
    (impact?.highImpact ? P.bombHighImpactBonus : 0) +
    (profile.trailing ? P.bombTrailBonus : 0) -
    profile.pressure.threat * P.bombRiskMul;

  if (ctx.defenseOpening && !impact?.highImpact) return value >= P.bombDefenseThreshold;
  return value >= P.bombThreshold;
}

export function decideShakingV6(state, playerKey, shakingMonths, deps, params = DEFAULT_PARAMS) {
  if (!shakingMonths?.length) return { allow: false, month: null, score: -Infinity };
  const P = mergedParams(params);
  const profile = buildProfile(state, playerKey, deps, P);
  const plan = deps.getFirstTurnDoublePiPlan(state, playerKey);

  let best = { allow: false, month: null, score: -Infinity, highImpact: false };
  for (const month of shakingMonths) {
    const immediate = safeNum(deps.shakingImmediateGainScore(state, playerKey, month));
    const combo = safeNum(deps.ownComboOpportunityScore(state, playerKey, month));
    const impact = deps.isHighImpactShaking(state, playerKey, month);
    const known = safeNum(deps.countKnownMonthCards?.(state, month));

    let score =
      immediate * P.shakeImmediateMul +
      combo * P.shakeComboMul +
      (impact?.highImpact ? P.shakeImpactBonus : 0) +
      (impact?.hasDoublePiLine ? P.shakePiLineBonus : 0) +
      (impact?.directThreeGwang ? P.shakeDirectGwangBonus : 0) -
      profile.pressure.threat * P.shakeRiskMul;

    if (profile.trailing) score += P.shakeTrailingBonus;
    if (plan?.active && plan.months.has(month)) score += P.shakeFirstPlanBonus;
    if (known <= 2) score += P.shakeKnownLowBonus;
    if (known >= 4) score -= P.shakeKnownHighPenalty;

    if (score > best.score) {
      best = { allow: false, month, score, highImpact: !!impact?.highImpact };
    }
  }

  let threshold = P.shakeThreshold;
  if (profile.leading) threshold += P.shakeLeadThresholdUp;
  if (profile.pressure.threat >= 0.72) threshold += P.shakePressureThresholdUp;
  if (profile.trailing) threshold -= 0.08;

  return { ...best, allow: best.score >= threshold };
}

/* President/Gukjin late decision helpers */
export function shouldPresidentStopV6(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const P = mergedParams(params);
  const ctx = deps.analyzeGameContext(state, playerKey);
  const carry = safeNum(state.carryOverMultiplier, 1);
  const diff = safeNum(ctx.myScore) - safeNum(ctx.oppScore);
  if (diff >= P.presidentStopLead && carry <= P.presidentCarryStopMax) return true;
  return false;
}

export function chooseGukjinHeuristicV6(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const P = mergedParams(params);
  const ctx = deps.analyzeGameContext(state, playerKey);
  const selfFive = safeNum(ctx.selfFive);
  const oppFive = safeNum(ctx.oppFive);
  const branch = deps.analyzeGukjinBranches(state, playerKey);

  if (selfFive <= 0 && oppFive >= 6) return "junk";
  if (selfFive >= 7 && oppFive <= 1) return "five";

  if (branch?.enabled && Array.isArray(branch.scenarios) && branch.scenarios.length) {
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
