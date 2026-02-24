// ============================================================
// heuristicV5.js  –  Matgo Heuristic V5  (정밀 재작성)
// ============================================================
// V4 대비 개선 포인트:
//   1. shouldGoV5  : V4 oppScoreRisk 단계 분기 완전 채택 + 후공 패널티
//   2. rankHandCardsV5 : V4 discard 정책 완전 이식 + 후공 보정 레이어
//   3. decideShakingV5 : V4 "매치 있으면 흔들기 금지" 규칙 반영
//   4. DEFAULT_PARAMS : Optuna 주입 가능한 모든 수치 집약
// ============================================================

// ──────────────────────────────────────────────────────────────
// CONSTANTS
// ──────────────────────────────────────────────────────────────
const GUKJIN_CARD_ID = "I0";
const DOUBLE_PI_MONTHS = Object.freeze([11, 12, 13]);
const BONUS_CARD_ID_SET = Object.freeze(new Set(["M0", "M1"]));
const SSANGPI_WITH_GUKJIN_ID_SET = Object.freeze(new Set(["K1", "L3", GUKJIN_CARD_ID]));
const OPTUNA_TRIAL0_PARAMS = Object.freeze({
  kwangWeight: 4.383881266576942,
  fiveWeight: 6.490321714386611,
  ribbonWeight: 7.158029319177128,
  piWeight: 1.473532659579097,
  doublePiBonus: 5.887881057780191,
  matchOneBase: 7.52508904118817,
  matchTwoBase: 14.795044127401228,
  matchThreeBase: 12.888576070495166,
  noMatchPenalty: 4.809596091334999,
  highValueMatchBonus: 3.7700488763733127,
  feedRiskMul: 4.514706134196945,
  feedRiskMatchMul: 1.054557015161309,
  pukRiskHighMul: 3.9941920096843977,
  pukRiskNormalMul: 3.3292158162053074,
  blockingBonus: 19.077065840867128,
  jokboBlockBonus: 5.089949645091993,
  firstTurnPiPlanBonus: 7.4553377950190445,
  comboBaseBonus: 3.4694194760467023,
  goBaseThreshold: 0.4820388504327037,
  goOppOneAwayGate: 46.96769063203732,
  goScoreDiffBonus: 0.06426611755245973,
  goDeckLowBonus: 0.11698925632759287,
  goUnseeHighPiPenalty: 0.11803494278340058,
  bombImpactMinGain: 0.7916592366243914,
  shakingScoreThreshold: 0.618066666699776,
  shakingImmediateGainMul: 1.9694771624873533,
  shakingComboGainMul: 1.2240695476160273,
  shakingTempoBonusMul: 0.27137723247154383,
  shakingAheadPenalty: 0.294908533278066,
  matchPiGainMul: 6.245538153466629,
  matchKwangBonus: 15.01604247696798,
  matchRibbonBonus: 10.020582065338775,
  matchFiveBonus: 9.766933714061619,
  matchDoublePiBonus: 16.493619158753866,
  matchMongBakFiveBonus: 33.82945327626024,
  goOppScoreGateLow: 3,
  goOppScoreGateHigh: 6,
});

// ──────────────────────────────────────────────────────────────
// DEFAULT PARAMS  (Optuna 튜닝 대상 수치 전부)
// ──────────────────────────────────────────────────────────────
export const DEFAULT_PARAMS = {
  // ── rankHandCards 기본 매치 스코어 ──
  matchZeroBase: -40.0,
  matchOneBase: 48.0,
  matchTwoBase: 56.0,
  matchThreeBase: 62.0,
  captureGainMulThree: 1.15,

  // ── 피 가치 ──
  piGainMul: 4.2,
  piGainSelfHighMul: 1.8,    // 내 피 7~9구간
  piGainOppLowMul: 1.4,      // 상대 피 ≤5
  doublePiMatchBonus: 16.0,
  doublePiMatchExtra: 6.0,
  doublePiNoMatchPenalty: 14.0,

  // ── 콤보 보너스 ──
  comboFinishBirds: 30.0,
  comboFinishRed: 27.0,
  comboFinishBlue: 27.0,
  comboFinishPlain: 27.0,
  comboFinishKwang: 32.0,
  comboBlockBase: 24.0,
  comboBlockUrgencyMul: 0.35,
  comboBlockNextThreatMul: 4.5,
  ribbonFourBonus: 34.0,
  fiveFourBonus: 36.0,

  // ── 몽박 방어 ──
  mongBakFiveBonus: 40.0,
  mongBakPiPenalty: 8.0,

  // ── 무매치 discard 정책 ──
  discardLivePiPenalty: 24.0,
  discardLivePiPenaltyLate: 36.0,
  discardDoublePiLivePenalty: 16.0,
  discardDoublePiLivePenaltyLate: 26.0,
  discardDoublePiDeadBonus: 6.0,
  discardComboHoldPenalty: 44.0,
  discardComboHoldPenaltyLate: 56.0,
  discardOneAwayPenalty: 42.0,
  discardOneAwayPenaltyLate: 58.0,
  discardBlockMedPenalty: 20.0,
  discardBlockMedPenaltyLate: 30.0,
  discardMongBakFivePenalty: 28.0,
  discardBonusPiBonus: 26.0,
  discardKnownMonthBonus: 1.9,
  discardUnknownMonthPenalty: 1.8,

  // ── 먹이/뻑 리스크 ──
  feedRiskNoMatchMul: 5.0,
  feedRiskMatchMul: 1.2,
  pukRiskHighMul: 4.8,
  pukRiskNormalMul: 3.4,

  // ── 첫턴 쌍피 플랜 ──
  firstTurnPiPlanBonus: 5.5,

  // ── 잠긴 월 패널티 ──
  lockedMonthPenalty: 6.0,

  // ── 후공 보정 ──
  // GO 임계값 강화: oppOneAwayProb 게이트를 N포인트 낮춤 (더 쉽게 STOP)
  secondMoverGoGateShrink: 4.0,
  // 카드 선택: 후공 블로킹 추가 점수
  secondMoverBlockBonus: 2.0,
  // 카드 선택: 후공 + 뒤질 때 피 수집 추가 가중
  secondMoverPiBonus: 1.5,

  // ── shouldGo: oppOneAway 임계값 ──
  goOneAwayThreshOpp0: 37,
  goOneAwayThreshOpp0Late: 33,
  goOneAwayThreshOpp1: 34,
  goOneAwayThreshOpp2: 38,
  goOneAwayThreshOpp3: 43,
  goOneAwayThreshOpp4Early: 32,
  goOneAwayThreshOpp4Late: 28,
  // 상대 5+ bigLead 조건
  goBigLeadScoreDiff: 8,
  goBigLeadMinScore: 11,
  goBigLeadOneAwayEarly: 25,
  goBigLeadOneAwayLate: 20,
  goBigLeadJokboThresh: 0.3,
  goBigLeadNextThresh: 0.35,
  // 상대 0 추가 게이트
  goOpp0JokboThresh: 0.37,
  goOpp0NextThresh: 0.47,
  // 상대 1~2 추가 게이트
  goOpp12JokboThresh: 0.35,
  goOpp12NextThresh: 0.45,

  // ── 페이즈 분류 기준 ──
  lateDeckMax: 10,

  // ── Optuna trial 0 baseline (logs/optuna_v5_best.json) ──
  ...OPTUNA_TRIAL0_PARAMS,

  // ── trial0 키를 현재 V5 네이티브 키로 매핑 ──
  matchZeroBase: -safeNum(OPTUNA_TRIAL0_PARAMS.noMatchPenalty) * 10.0,
  piGainMul: safeNum(OPTUNA_TRIAL0_PARAMS.matchPiGainMul),
  doublePiMatchBonus: safeNum(OPTUNA_TRIAL0_PARAMS.matchDoublePiBonus),
  mongBakFiveBonus: safeNum(OPTUNA_TRIAL0_PARAMS.matchMongBakFiveBonus),
  feedRiskNoMatchMul: safeNum(OPTUNA_TRIAL0_PARAMS.feedRiskMul),
  feedRiskMatchMul: safeNum(OPTUNA_TRIAL0_PARAMS.feedRiskMatchMul),
  pukRiskHighMul: safeNum(OPTUNA_TRIAL0_PARAMS.pukRiskHighMul),
  pukRiskNormalMul: safeNum(OPTUNA_TRIAL0_PARAMS.pukRiskNormalMul),
  firstTurnPiPlanBonus: safeNum(OPTUNA_TRIAL0_PARAMS.firstTurnPiPlanBonus),
  comboBlockBase: safeNum(OPTUNA_TRIAL0_PARAMS.blockingBonus),
  comboBlockUrgencyMul: safeNum(OPTUNA_TRIAL0_PARAMS.jokboBlockBonus) * 0.1,
};

// ──────────────────────────────────────────────────────────────
// HELPERS
// ──────────────────────────────────────────────────────────────
function safeNum(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function isSecondMover(state, playerKey) {
  const first = state?.startingTurnKey;
  if (first === "human" || first === "ai") return first !== playerKey;
  return false;
}

function hasComboTag(card, tag) {
  return Array.isArray(card?.comboTags) && card.comboTags.includes(tag);
}

function countComboTag(cards, tag) {
  return (cards || []).reduce((n, c) => (hasComboTag(c, tag) ? n + 1 : n), 0);
}

function comboCounts(player) {
  const ribbons = player?.captured?.ribbon || [];
  const fives   = player?.captured?.five   || [];
  const kwang   = player?.captured?.kwang  || [];
  return {
    red:   countComboTag(ribbons, "redRibbons"),
    blue:  countComboTag(ribbons, "blueRibbons"),
    plain: countComboTag(ribbons, "plainRibbons"),
    birds: countComboTag(fives, "fiveBirds"),
    kwang: kwang.length,
  };
}

function hasCategory(cards, cat) {
  return (cards || []).some((c) => c?.category === cat);
}

function piLikeValue(card, deps) {
  if (!card) return 0;
  if (card.id === GUKJIN_CARD_ID) return 2;
  if (card.category === "junk") return safeNum(deps.junkPiValue(card), 0);
  return 0;
}

function isDoublePiLike(card, deps) {
  if (!card) return false;
  if (card.id === GUKJIN_CARD_ID) return true;
  return card.category === "junk" && safeNum(deps.junkPiValue(card), 0) >= 2;
}

function ownComboFinishBonus(capturedCombo, captureCards, P) {
  let b = 0;
  if (capturedCombo.birds  >= 2 && captureCards.some((c) => hasComboTag(c, "fiveBirds")))    b += P.comboFinishBirds;
  if (capturedCombo.red    >= 2 && captureCards.some((c) => hasComboTag(c, "redRibbons")))   b += P.comboFinishRed;
  if (capturedCombo.blue   >= 2 && captureCards.some((c) => hasComboTag(c, "blueRibbons")))  b += P.comboFinishBlue;
  if (capturedCombo.plain  >= 2 && captureCards.some((c) => hasComboTag(c, "plainRibbons"))) b += P.comboFinishPlain;
  if (capturedCombo.kwang  >= 2 && captureCards.some((c) => c?.category === "kwang"))        b += P.comboFinishKwang;
  return b;
}

function opponentComboBlockBonus(month, jokboThreat, blockMonths, blockUrgency, nextThreat, P) {
  let b = 0;
  const monthUrgency = safeNum(jokboThreat?.monthUrgency?.get(month));
  if (monthUrgency > 0) b += P.comboBlockBase + monthUrgency * P.comboBlockUrgencyMul;
  if (blockMonths?.has(month)) {
    const urgency = safeNum(blockUrgency?.get(month), 2);
    b += urgency >= 3 ? 18 : 10;
    b += nextThreat * P.comboBlockNextThreatMul;
  }
  return b;
}

function getCapturedDoublePiMonths(state) {
  const captured = new Set();
  for (const key of ["human", "ai"]) {
    for (const card of state.players?.[key]?.captured?.junk || []) {
      if (DOUBLE_PI_MONTHS.includes(card?.month) && safeNum(card?.piValue) >= 2) {
        captured.add(card.month);
      }
    }
  }
  return captured;
}

function getLiveDoublePiMonths(state) {
  const captured = getCapturedDoublePiMonths(state);
  const live = new Set();
  for (const m of DOUBLE_PI_MONTHS) { if (!captured.has(m)) live.add(m); }
  return live;
}

function getComboHoldMonths(state, playerKey, deps) {
  const opp = deps.otherPlayerKey(playerKey);
  const hold = new Set();
  for (const m of deps.blockingMonthsAgainst(state.players?.[playerKey], state.players?.[opp])) hold.add(m);
  for (const m of deps.blockingMonthsAgainst(state.players?.[opp], state.players?.[playerKey])) hold.add(m);
  return hold;
}

function discardTieOrderScore(card, deps, monthIsLiveDoublePi) {
  if (card?.bonus?.stealPi) return 6;
  if (isDoublePiLike(card, deps) && monthIsLiveDoublePi) return 1;
  if (card?.category === "five")   return 5;
  if (card?.category === "ribbon") return 4;
  if (card?.category === "kwang")  return 3;
  return 2;
}

function hasCertainJokbo(player) {
  const c = comboCounts(player);
  return c.kwang >= 3 || c.birds >= 3 || c.red >= 3 || c.blue >= 3 || c.plain >= 3;
}

function countUnseenByIdSet(state, _playerKey, idSet) {
  let seen = 0;
  for (const key of ["human", "ai"]) {
    const player = state.players?.[key];
    for (const cat of ["kwang", "five", "ribbon", "junk"]) {
      for (const c of player?.captured?.[cat] || []) if (idSet.has(c?.id)) seen++;
    }
    for (const c of player?.hand || []) if (idSet.has(c?.id)) seen++;
  }
  for (const c of state.board || []) if (idSet.has(c?.id)) seen++;
  return idSet.size - seen;
}

// shouldGo 내부 헬퍼
function _shouldStopForOppFourLike(state, playerKey, deps, bonusThresh, ssangpiThresh) {
  if (countUnseenByIdSet(state, playerKey, BONUS_CARD_ID_SET) >= bonusThresh) return true;
  const jokbo = deps.checkOpponentJokboProgress(state, playerKey);
  // 콤보 위협 존재 시 STOP
  if (safeNum(jokbo?.threat) >= 0.5) return true;
  if (countUnseenByIdSet(state, playerKey, SSANGPI_WITH_GUKJIN_ID_SET) >= ssangpiThresh) return true;
  return false;
}

function _estimateOppOneAwayProb(state, playerKey, deps) {
  const deckCount = safeNum(state.deck?.length);
  const jokbo  = deps.checkOpponentJokboProgress(state, playerKey);
  const next   = safeNum(deps.nextTurnThreatScore(state, playerKey));
  const threat = safeNum(deps.opponentThreatScore(state, playerKey));

  let prob = threat * 35 + safeNum(jokbo?.threat) * 30 + next * 20;
  if (deckCount <= 15) prob += 5;
  if (deckCount <= 10) prob += 8;
  if (deckCount <= 6)  prob += 6;
  if (deckCount <= 3)  prob += 5;

  return {
    oppOneAwayProb: Math.max(0, Math.min(100, prob)),
    jokboThreat: safeNum(jokbo?.threat),
    nextThreat: next,
    deckCount,
  };
}

// ──────────────────────────────────────────────────────────────
// 1. rankHandCardsV5
// ──────────────────────────────────────────────────────────────
export function rankHandCardsV5(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const P = { ...DEFAULT_PARAMS, ...params };
  const player = state.players?.[playerKey];
  if (!player?.hand?.length) return [];

  const opp       = deps.otherPlayerKey(playerKey);
  const oppPlayer = state.players?.[opp];
  const ctx       = deps.analyzeGameContext(state, playerKey);
  const secondMover = isSecondMover(state, playerKey);

  const selfPi = safeNum(ctx.selfPi, deps.capturedCountByCategory(player, "junk"));
  const oppPi  = safeNum(ctx.oppPi,  deps.capturedCountByCategory(oppPlayer, "junk"));
  const nextThreat   = safeNum(deps.nextTurnThreatScore(state, playerKey));
  const jokboThreat  = deps.checkOpponentJokboProgress(state, playerKey);
  const blockMonths  = deps.blockingMonthsAgainst(oppPlayer, player);
  const blockUrgency = deps.blockingUrgencyByMonth(oppPlayer, player);
  const firstTurnPiPlan  = deps.getFirstTurnDoublePiPlan(state, playerKey);
  const boardByMonth     = deps.boardMatchesByMonth(state);
  const boardCountByMonth = deps.monthCounts(state.board || []);
  const handCountByMonth  = deps.monthCounts(player.hand || []);
  const capturedByMonth   = deps.capturedMonthCounts(state);
  const deckCount = safeNum(state.deck?.length);
  const selfCombo       = comboCounts(player);
  const selfRibbonCount = (player?.captured?.ribbon || []).length;
  const selfFiveCount   = (player?.captured?.five   || []).length;
  const mongBakDefenseCritical = safeNum(ctx.selfFive) <= 0 && safeNum(ctx.oppFive) >= 7;
  const liveDoublePiMonths = getLiveDoublePiMonths(state);
  const comboHoldMonths    = getComboHoldMonths(state, playerKey, deps);

  const ranked = player.hand.map((card) => {
    const matches      = boardByMonth.get(card.month) || [];
    const captureCards = [card, ...matches];
    const captureGain  = matches.reduce((sum, c) => sum + deps.cardCaptureValue(c), 0);
    const ownValue     = deps.cardCaptureValue(card);
    const piGain       = captureCards.reduce((sum, c) => sum + piLikeValue(c, deps), 0);
    const doublePiCount = captureCards.filter((c) => isDoublePiLike(c, deps)).length;
    const knownMonth   =
      safeNum(boardCountByMonth.get(card.month)) +
      safeNum(handCountByMonth.get(card.month)) +
      safeNum(capturedByMonth.get(card.month));
    const monthIsLiveDoublePi  = liveDoublePiMonths.has(card.month);
    const monthIsComboHold     = comboHoldMonths.has(card.month);
    const monthBlockUrgency    = safeNum(blockUrgency.get(card.month));
    const monthJokboUrgency    = safeNum(jokboThreat?.monthUrgency?.get(card.month));
    const monthIsOneAwayThreat = monthBlockUrgency >= 3 || monthJokboUrgency >= 24;

    // ── 기본 매치 스코어 ──
    let score = 0;
    if (matches.length === 0) {
      score = P.matchZeroBase - ownValue * 0.9;
    } else if (matches.length === 1) {
      score = P.matchOneBase + captureGain - ownValue * 0.1;
    } else if (matches.length === 2) {
      score = P.matchTwoBase + captureGain;
    } else {
      score = P.matchThreeBase + captureGain * P.captureGainMulThree;
    }
    if (matches.some((m) => m?.category === "kwang" || m?.category === "five" || m?.category === "ribbon")) {
      score += safeNum(P.highValueMatchBonus);
    }

    // ── 피 가치 ──
    score += piGain * P.piGainMul;
    if (selfPi >= 7 && selfPi <= 9) score += piGain * P.piGainSelfHighMul;
    if (oppPi  <= 5)                 score += piGain * P.piGainOppLowMul;

    // ── 쌍피 보너스/패널티 ──
    if (doublePiCount > 0) {
      score += P.doublePiMatchBonus + (doublePiCount - 1) * P.doublePiMatchExtra;
    }
    if (matches.length === 0 && isDoublePiLike(card, deps)) {
      score -= P.doublePiNoMatchPenalty;
    }

    // ── 콤보 완성 + 블로킹 ──
    score += ownComboFinishBonus(selfCombo, captureCards, P);
    score += opponentComboBlockBonus(card.month, jokboThreat, blockMonths, blockUrgency, nextThreat, P);

    // ── 후공 블로킹 추가 가중 ──
    if (secondMover && blockMonths.has(card.month)) {
      score += P.secondMoverBlockBonus;
    }

    // ── 띠/열 4개 이상 ──
    if (selfRibbonCount >= 4 && hasCategory(captureCards, "ribbon")) score += P.ribbonFourBonus;
    if (selfFiveCount   >= 4 && hasCategory(captureCards, "five"))   score += P.fiveFourBonus;

    // ── 몽박 방어 ──
    if (mongBakDefenseCritical) {
      if (hasCategory(captureCards, "five")) score += P.mongBakFiveBonus;
      else if (piGain > 0)                   score -= P.mongBakPiPenalty;
    }

    // ── 무매치 knownMonth 보정 ──
    if (matches.length === 0) {
      if (knownMonth >= 3) score += P.discardKnownMonthBonus;
      else if (knownMonth <= 1) score -= P.discardUnknownMonthPenalty;
    }

    // ── 잠긴 월 패널티 ──
    const capCntMonth = safeNum(capturedByMonth.get(card.month));
    if (matches.length > 0 && doublePiCount === 0 && capCntMonth >= 2 && knownMonth >= 3) {
      score -= P.lockedMonthPenalty;
    }

    // ── 무매치 discard 상세 정책 (V4 완전 이식) ──
    if (matches.length === 0) {
      let ds = 0;
      const late = deckCount <= 8;

      if (card?.bonus?.stealPi) ds += P.discardBonusPiBonus;

      if (monthIsLiveDoublePi)
        ds -= late ? P.discardLivePiPenaltyLate : P.discardLivePiPenalty;

      if (isDoublePiLike(card, deps) && monthIsLiveDoublePi)
        ds -= late ? P.discardDoublePiLivePenaltyLate : P.discardDoublePiLivePenalty;

      if (isDoublePiLike(card, deps) && !monthIsLiveDoublePi)
        ds += P.discardDoublePiDeadBonus;

      if (monthIsComboHold)
        ds -= late ? P.discardComboHoldPenaltyLate : P.discardComboHoldPenalty;

      if (monthIsOneAwayThreat)
        ds -= late ? P.discardOneAwayPenaltyLate : P.discardOneAwayPenalty;
      else if (monthBlockUrgency >= 2 || monthJokboUrgency >= 20)
        ds -= late ? P.discardBlockMedPenaltyLate : P.discardBlockMedPenalty;

      if (mongBakDefenseCritical && card.category === "five") ds -= P.discardMongBakFivePenalty;
      if (mongBakDefenseCritical && card.category === "junk") ds += 5;

      ds += discardTieOrderScore(card, deps, monthIsLiveDoublePi) * 2.2;
      score += ds;
    }

    // ── 먹이/뻑 리스크 ──
    const feedRisk = safeNum(deps.estimateOpponentImmediateGainIfDiscard(state, playerKey, card.month));
    score -= feedRisk * (matches.length === 0 ? P.feedRiskNoMatchMul : P.feedRiskMatchMul);

    const pukRisk = safeNum(deps.isRiskOfPuk(state, playerKey, card, boardCountByMonth, handCountByMonth));
    if (pukRisk > 0) score -= pukRisk * (deckCount <= 10 ? P.pukRiskHighMul : P.pukRiskNormalMul);
    else if (pukRisk < 0) score += -pukRisk * 1.4;

    // ── 첫턴 쌍피 플랜 ──
    if (firstTurnPiPlan.active && firstTurnPiPlan.months.has(card.month)) {
      score += P.firstTurnPiPlanBonus;
    }

    // ── 후공 피 수집 추가 (뒤질 때) ──
    if (secondMover && safeNum(ctx.myScore) < safeNum(ctx.oppScore) && piGain > 0) {
      score += piGain * P.secondMoverPiBonus;
    }

    return { card, score, matches: matches.length };
  });

  ranked.sort((a, b) => b.score - a.score);
  return ranked;
}

// ──────────────────────────────────────────────────────────────
// 2. chooseMatchHeuristicV5
// ──────────────────────────────────────────────────────────────
export function chooseMatchHeuristicV5(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const P = { ...DEFAULT_PARAMS, ...params };
  const ids = state.pendingMatch?.boardCardIds || [];
  if (!ids.length) return null;

  const opp        = deps.otherPlayerKey(playerKey);
  const selfPlayer = state.players?.[playerKey];
  const oppPlayer  = state.players?.[opp];
  const ctx   = deps.analyzeGameContext(state, playerKey);
  const selfPi = safeNum(ctx.selfPi, deps.capturedCountByCategory(selfPlayer, "junk"));
  const oppPi  = safeNum(ctx.oppPi,  deps.capturedCountByCategory(oppPlayer,  "junk"));
  const nextThreat   = safeNum(deps.nextTurnThreatScore(state, playerKey));
  const jokbo        = deps.checkOpponentJokboProgress(state, playerKey);
  const blockMonths  = deps.blockingMonthsAgainst(oppPlayer, selfPlayer);
  const blockUrgency = deps.blockingUrgencyByMonth(oppPlayer, selfPlayer);
  const selfCombo       = comboCounts(selfPlayer);
  const selfRibbonCount = (selfPlayer?.captured?.ribbon || []).length;
  const selfFiveCount   = (selfPlayer?.captured?.five   || []).length;
  const mongBakDefenseCritical = safeNum(ctx.selfFive) <= 0 && safeNum(ctx.oppFive) >= 7;

  const candidates = (state.board || []).filter((c) => ids.includes(c.id));
  if (!candidates.length) return null;

  let best = candidates[0];
  let bestScore = -Infinity;
  const matchPiGainMul = safeNum(P.matchPiGainMul, 4.0);
  const matchKwangBonus = safeNum(P.matchKwangBonus, 8.0);
  const matchRibbonBonus = safeNum(P.matchRibbonBonus, 6.0);
  const matchFiveBonus = safeNum(P.matchFiveBonus, 4.0);
  const matchDoublePiBonus = safeNum(P.matchDoublePiBonus, 14.0);
  const matchMongBakFiveBonus = safeNum(P.matchMongBakFiveBonus, P.mongBakFiveBonus);
  for (const c of candidates) {
    const piGain = piLikeValue(c, deps);
    let score = deps.cardCaptureValue(c) * 0.8;

    score += piGain * matchPiGainMul;
    if (c.category === "kwang")  score += matchKwangBonus;
    if (c.category === "ribbon") score += matchRibbonBonus;
    if (c.category === "five")   score += matchFiveBonus;
    if (selfPi >= 7 && selfPi <= 9) score += piGain * 1.8;
    if (oppPi  <= 5)                 score += piGain * 1.4;
    if (isDoublePiLike(c, deps)) score += matchDoublePiBonus;

    score += ownComboFinishBonus(selfCombo, [c], P);
    score += opponentComboBlockBonus(c.month, jokbo, blockMonths, blockUrgency, nextThreat, P);

    if (selfRibbonCount >= 4 && c.category === "ribbon") score += P.ribbonFourBonus;
    if (selfFiveCount   >= 4 && c.category === "five")   score += P.fiveFourBonus;

    if (mongBakDefenseCritical) {
      if (c.category === "five")  score += matchMongBakFiveBonus;
      else if (piGain > 0)        score -= P.mongBakPiPenalty;
    }

    score += safeNum(deps.monthStrategicPriority?.(c.month)) * 0.25;
    if (ctx.mode === "DESPERATE_DEFENSE" && piGain <= 0) score -= 0.45;

    if (score > bestScore) { bestScore = score; best = c; }
  }
  return best?.id ?? null;
}

// ──────────────────────────────────────────────────────────────
// 3. shouldGoV5
//    V4 oppScoreRisk 단계 분기 완전 채택 + 후공 패널티
// ──────────────────────────────────────────────────────────────
export function shouldGoV5(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const P = { ...DEFAULT_PARAMS, ...params };

  if (deps.canBankruptOpponentByStop?.(state, playerKey)) return false;

  const ctx         = deps.analyzeGameContext(state, playerKey);
  const myScore     = safeNum(ctx.myScore);
  const oppScoreBase = safeNum(ctx.oppScore);
  const threat      = _estimateOppOneAwayProb(state, playerKey, deps);
  const lateGame    = threat.deckCount <= P.lateDeckMax;
  const secondMover = isSecondMover(state, playerKey);
  // 후공 패널티: oppOneAway 임계값을 N포인트 낮춤 → STOP 더 쉽게
  const secG = secondMover ? P.secondMoverGoGateShrink : 0;

  const opp        = deps.otherPlayerKey(playerKey);
  const oppPlayer  = state.players?.[opp];
  const selfPlayer = state.players?.[playerKey];
  const oppPiBase  = safeNum(ctx.oppPi, deps.capturedCountByCategory(oppPlayer, "junk"));
  const certain    = hasCertainJokbo(selfPlayer);

  const unseenBonus   = countUnseenByIdSet(state, playerKey, BONUS_CARD_ID_SET);
  const unseenSsangpi = countUnseenByIdSet(state, playerKey, SSANGPI_WITH_GUKJIN_ID_SET);
  const unseenHighPi  = unseenSsangpi + unseenBonus;

  // ── Gukjin 최악 케이스 ──
  const gukjinBranch = deps.analyzeGukjinBranches?.(state, playerKey);
  let oppScoreRisk = oppScoreBase;
  let oppPiRisk    = oppPiBase;
  if (gukjinBranch?.scenarios?.length) {
    for (const s of gukjinBranch.scenarios) {
      oppScoreRisk = Math.max(oppScoreRisk, safeNum(s?.oppScore));
      oppPiRisk    = Math.max(oppPiRisk,    safeNum(s?.oppPi));
    }
    for (const s of gukjinBranch.scenarios) {
      const sScore = safeNum(s?.oppScore);
      const sPi    = safeNum(s?.oppPi);
      if (sScore >= 6) return false;
      if (sScore >= 5 && !certain) return false;
      if (unseenHighPi >= 2 && sPi >= 7 && !certain) return false;
    }
  }

  const scoreDiffRisk = myScore - oppScoreRisk;
  const oneAwayGateCap = safeNum(P.goOppOneAwayGate, 100);
  const oppScoreGateLow = Math.max(3, Math.min(5, Math.round(safeNum(P.goOppScoreGateLow, 4))));
  const oppScoreGateHigh = Math.max(5, Math.min(7, Math.round(safeNum(P.goOppScoreGateHigh, 6))));
  const quickGoConfidence =
    0.5 +
    scoreDiffRisk * safeNum(P.goScoreDiffBonus, 0) +
    (lateGame ? safeNum(P.goDeckLowBonus, 0) : 0) -
    unseenHighPi * safeNum(P.goUnseeHighPiPenalty, 0);

  if (threat.oppOneAwayProb >= oneAwayGateCap) return false;
  if (!certain && quickGoConfidence < safeNum(P.goBaseThreshold, 0)) return false;

  // ── 쌍피 위험 게이트 ──
  if (unseenHighPi >= 2 && oppPiRisk >= 7 && !certain) return false;

  // ── 1) 상대 6+ → 무조건 STOP ──
  if (oppScoreRisk >= oppScoreGateHigh) return false;

  // ── 2) 상대 5+ → 기본 STOP (bigLead+lowThreat 예외, 후공이면 예외 없음) ──
  if (oppScoreRisk >= oppScoreGateHigh - 1) {
    const shouldStop = _shouldStopForOppFourLike(state, playerKey, deps, 1, 2);
    if (shouldStop) return false;
    if (secondMover) return false; // 후공이면 예외 없이 STOP
    const bigLead  = scoreDiffRisk >= P.goBigLeadScoreDiff && myScore >= P.goBigLeadMinScore;
    const lowThreat =
      threat.oppOneAwayProb < (lateGame ? P.goBigLeadOneAwayLate : P.goBigLeadOneAwayEarly) &&
      threat.jokboThreat    <  P.goBigLeadJokboThresh &&
      threat.nextThreat     <  P.goBigLeadNextThresh;
    return bigLead && lowThreat;
  }

  // ── 3) 상대 4 ──
  if (oppScoreRisk >= oppScoreGateLow) {
    const shouldStop = _shouldStopForOppFourLike(state, playerKey, deps, 2, 3);
    if (shouldStop) return false;
    const gate = (lateGame ? P.goOneAwayThreshOpp4Late : P.goOneAwayThreshOpp4Early) - secG;
    if (threat.oppOneAwayProb >= gate) return false;
    return true;
  }

  // ── 4) 상대 1~3 ──
  if (oppScoreRisk >= 1 && oppScoreRisk <= oppScoreGateLow - 1) {
    if (oppScoreRisk === 3 && threat.jokboThreat >= 0.5) return false;
    const baseGate =
      oppScoreRisk === 3 ? P.goOneAwayThreshOpp3 :
      oppScoreRisk === 2 ? P.goOneAwayThreshOpp2 :
                           P.goOneAwayThreshOpp1;
    const gate = baseGate - (lateGame ? 1 : 0) - secG;
    if (threat.oppOneAwayProb >= gate) return false;

    // 상대 ≤2 추가 위협 게이트
    if (oppScoreRisk <= 2) {
      if (
        threat.jokboThreat >= P.goOpp12JokboThresh ||
        threat.nextThreat  >= P.goOpp12NextThresh  ||
        (lateGame && threat.jokboThreat >= 0.4)
      ) return false;
    }

    // 후공 + 동점/뒤짐 → gate 추가 강화
    if (secondMover && scoreDiffRisk <= 0 && threat.oppOneAwayProb >= baseGate - 5 - secG) return false;
    return true;
  }

  // ── 5) 상대 0 ──
  const gate0 = (lateGame ? P.goOneAwayThreshOpp0Late : P.goOneAwayThreshOpp0) - secG;
  if (threat.oppOneAwayProb >= gate0) return false;
  if (threat.jokboThreat >= P.goOpp0JokboThresh || threat.nextThreat >= P.goOpp0NextThresh) return false;
  if (secondMover && scoreDiffRisk < 0) return false; // 후공 + 뒤지면 GO 금지
  return true;
}

// ──────────────────────────────────────────────────────────────
// 4. shouldBombV5 / selectBombMonthV5
// ──────────────────────────────────────────────────────────────
export function shouldBombV5(state, playerKey, bombMonths, deps, params = DEFAULT_PARAMS) {
  const P = { ...DEFAULT_PARAMS, ...params };
  if (!bombMonths?.length) return false;
  const ctx = deps.analyzeGameContext(state, playerKey);
  const firstTurnPiPlan = deps.getFirstTurnDoublePiPlan(state, playerKey);

  if (firstTurnPiPlan.active && bombMonths.some((m) => firstTurnPiPlan.months.has(m))) return true;

  const bestMonth = _bestBombMonth(state, bombMonths, deps);
  if (bestMonth == null) return false;
  const impact = deps.isHighImpactBomb(state, playerKey, bestMonth);
  const gain   = safeNum(deps.monthBoardGain(state, bestMonth));

  if (impact.highImpact) return true;
  if (ctx.defenseOpening) return false;
  if (ctx.volatilityComeback) {
    if (safeNum(impact.immediateGain) >= safeNum(P.bombImpactMinGain, 4)) return true;
    return gain >= 0;
  }
  if (ctx.nagariDelayMode && safeNum(impact.immediateGain) < 6) return false;
  return gain >= 1.0;
}

export function selectBombMonthV5(state, _playerKey, bombMonths, deps) {
  return _bestBombMonth(state, bombMonths, deps);
}

function _bestBombMonth(state, months, deps) {
  if (!months?.length) return null;
  let best = months[0];
  let bestScore = safeNum(deps.monthBoardGain(state, best));
  for (const m of months.slice(1)) {
    const s = safeNum(deps.monthBoardGain(state, m));
    if (s > bestScore) { best = m; bestScore = s; }
  }
  return best;
}

// ──────────────────────────────────────────────────────────────
// 5. decideShakingV5
//    핵심: 매치 있으면 흔들기 금지 (V4 동일)
// ──────────────────────────────────────────────────────────────
export function decideShakingV5(state, playerKey, shakingMonths, deps, params = DEFAULT_PARAMS) {
  const P = { ...DEFAULT_PARAMS, ...params };
  if (!shakingMonths?.length) return { allow: false, month: null, score: -Infinity };

  // V4 핵심 규칙: 일반 매치 존재 시 흔들기 불가
  const boardByMonth = deps.boardMatchesByMonth(state);
  const player = state.players?.[playerKey];
  const hasAnyMatch = (player?.hand || []).some(
    (card) => (boardByMonth.get(card.month) || []).length > 0
  );
  if (hasAnyMatch) return { allow: false, month: null, score: -Infinity };

  const ctx      = deps.analyzeGameContext(state, playerKey);
  const myScore  = safeNum(ctx.myScore);
  const oppScore = safeNum(ctx.oppScore);
  const liveDoublePiMonths = getLiveDoublePiMonths(state);
  const comboHoldMonths    = getComboHoldMonths(state, playerKey, deps);
  const firstTurnPiPlan    = deps.getFirstTurnDoublePiPlan(state, playerKey);

  // 상대가 크게 앞서면 흔들기 금지
  if (oppScore >= 5 && oppScore >= myScore + 2) return { allow: false, month: null, score: -Infinity };

  let best = { allow: false, month: null, score: -Infinity, highImpact: false };

  for (const month of shakingMonths) {
    const immediateGain = safeNum(deps.shakingImmediateGainScore(state, playerKey, month));
    const comboGain     = safeNum(deps.ownComboOpportunityScore(state, playerKey, month));
    const impact        = deps.isHighImpactShaking(state, playerKey, month);
    const known         = safeNum(deps.countKnownMonthCards(state, month));
    const uncertaintyBonus = known <= 2 ? 0.25 : known >= 4 ? -0.1 : 0;

    let score =
      immediateGain * safeNum(P.shakingImmediateGainMul, 1.35) +
      comboGain * safeNum(P.shakingComboGainMul, 1.15) +
      uncertaintyBonus;
    if (impact?.hasDoublePiLine)  score += 0.35;
    if (impact?.directThreeGwang) score += 0.3;
    if (impact?.highImpact)       score += 0.4;
    if (myScore < oppScore) score += safeNum(P.shakingTempoBonusMul, 0);
    if (liveDoublePiMonths.has(month) && !comboHoldMonths.has(month)) score += 0.55;
    if (comboHoldMonths.has(month)) score -= 0.25;
    if (firstTurnPiPlan.active && firstTurnPiPlan.months.has(month)) score += 0.3;

    if (score > best.score) {
      best = { allow: false, month, score, highImpact: !!impact?.highImpact };
    }
  }

  const preferDoublePiShake =
    best.month != null &&
    liveDoublePiMonths.has(best.month) &&
    !comboHoldMonths.has(best.month);
  const threshold = safeNum(P.shakingScoreThreshold, 0.65);
  const aheadPenalty = myScore > oppScore ? safeNum(P.shakingAheadPenalty, 0) : 0;
  const allow = (myScore > oppScore || preferDoublePiShake) && best.score >= threshold + aheadPenalty;
  return { ...best, allow };
}

// ──────────────────────────────────────────────────────────────
// 6. shouldPresidentStopV5
// ──────────────────────────────────────────────────────────────
export function shouldPresidentStopV5(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const ctx         = deps.analyzeGameContext(state, playerKey);
  const scoreDiff   = safeNum(ctx.myScore) - safeNum(ctx.oppScore);
  const carryOver   = safeNum(state.carryOverMultiplier, 1);
  const secondMover = isSecondMover(state, playerKey);

  if (scoreDiff >= 3 && carryOver <= 1) return true;
  if (scoreDiff <= -1) return false;
  if (secondMover && scoreDiff <= 0) return false;
  if (carryOver >= 2) return false;
  return scoreDiff >= 1;
}

// ──────────────────────────────────────────────────────────────
// 7. chooseGukjinHeuristicV5
// ──────────────────────────────────────────────────────────────
export function chooseGukjinHeuristicV5(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const ctx    = deps.analyzeGameContext(state, playerKey);
  const branch = deps.analyzeGukjinBranches(state, playerKey);
  const selfFive = safeNum(ctx.selfFive);
  const oppFive  = safeNum(ctx.oppFive);

  if (selfFive <= 0 && oppFive >= 6) return "junk";
  if (selfFive >= 7 && oppFive <= 0) return "five";

  if (branch?.enabled && branch.scenarios?.length) {
    const fiveSc = branch.scenarios.find((s) => s?.selfMode === "five");
    const junkSc = branch.scenarios.find((s) => s?.selfMode === "junk");
    if (fiveSc && junkSc) {
      const fiveAdv = safeNum(fiveSc.myScore) - safeNum(fiveSc.oppScore);
      const junkAdv = safeNum(junkSc.myScore) - safeNum(junkSc.oppScore);
      return fiveAdv >= junkAdv ? "five" : "junk";
    }
  }
  return selfFive >= 7 ? "five" : "junk";
}
