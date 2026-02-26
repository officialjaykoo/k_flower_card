export {
  rankHandCardsV5Plus,
  chooseMatchHeuristicV5Plus,
  chooseGukjinHeuristicV5Plus,
  shouldPresidentStopV5Plus,
  shouldGoV5Plus,
  selectBombMonthV5Plus,
  shouldBombV5Plus,
  decideShakingV5Plus,
};

/* ============================================================================
 * Heuristic V5Plus
 * - V5 baseline + targeted upgrades:
 *   1) phase-aware scoring
 *   2) utility-filtered GO gate
 *   3) forward-blocking chooseMatch
 * ========================================================================== */

/* 1) Constants and parameter surface */
const GUKJIN_CARD_ID = "I0";
const DOUBLE_PI_MONTHS = Object.freeze([11, 12, 13]);
const BONUS_CARD_ID_SET = Object.freeze(new Set(["M0", "M1"]));
const SSANGPI_WITH_GUKJIN_ID_SET = Object.freeze(new Set(["K1", "L3", GUKJIN_CARD_ID]));

/* 2) Parameter defaults (V5 base + V5Plus additions) */
export const DEFAULT_PARAMS = {
  // ── 페이즈 경계 ──
  phaseEarlyDeck: 18,          // 덱이 이 이상이면 early
  phaseMidDeck: 10,            // 덱이 이 이상이면 mid, 미만이면 late

  // ── rankHandCards (V5와 동일 기본값) ──
  matchZeroBase: -48.1,
  matchOneBase: 7.53,
  matchTwoBase: 14.80,
  matchThreeBase: 12.89,
  captureGainMulThree: 1.15,
  highValueMatchBonus: 3.77,
  piGainMul: 6.25,
  piGainSelfHighMul: 1.8,
  piGainOppLowMul: 1.4,
  doublePiMatchBonus: 16.49,
  doublePiMatchExtra: 6.0,
  doublePiNoMatchPenalty: 14.0,
  comboFinishBirds: 30.0,  comboFinishRed: 27.0,  comboFinishBlue: 27.0,
  comboFinishPlain: 27.0,  comboFinishKwang: 32.0,
  comboBlockBase: 19.08,
  comboBlockUrgencyMul: 0.51,
  comboBlockNextThreatMul: 4.5,
  ribbonFourBonus: 34.0,   fiveFourBonus: 36.0,
  mongBakFiveBonus: 33.83,
  mongBakPiPenalty: 8.0,
  discardLivePiPenalty: 24.0,       discardLivePiPenaltyLate: 36.0,
  discardDoublePiLivePenalty: 16.0, discardDoublePiLivePenaltyLate: 26.0,
  discardDoublePiDeadBonus: 6.0,
  discardComboHoldPenalty: 44.0,    discardComboHoldPenaltyLate: 56.0,
  discardOneAwayPenalty: 42.0,      discardOneAwayPenaltyLate: 58.0,
  discardBlockMedPenalty: 20.0,     discardBlockMedPenaltyLate: 30.0,
  discardMongBakFivePenalty: 28.0,  discardBonusPiBonus: 26.0,
  discardKnownMonthBonus: 1.9,      discardUnknownMonthPenalty: 1.8,
  feedRiskNoMatchMul: 4.51,
  feedRiskMatchMul: 1.05,
  pukRiskHighMul: 3.99,
  pukRiskNormalMul: 3.33,
  firstTurnPiPlanBonus: 7.46,
  lockedMonthPenalty: 6.0,
  secondMoverGoGateShrink: 4.0,  secondMoverBlockBonus: 2.0,  secondMoverPiBonus: 1.5,

  // ── Phase 보정 배율 (V5Plus 신규) ──
  // early 페이즈: 콤보 구축에 더 집중
  phaseEarlyComboMul: 1.20,     // 콤보 점수 배율 상승
  phaseEarlyBlockMul: 0.90,     // 블로킹은 살짝 완화 (아직 많이 남음)
  phaseEarlyFeedMul: 0.85,      // 먹이 리스크 완화 (아직 여유)
  // mid 페이즈: 기본값
  // late 페이즈: 수비 강화, 기댓값 확보 우선
  phaseLateComboMul: 0.85,      // 콤보 완성이 어려우면 포기
  phaseLateBlockMul: 1.35,      // 블로킹 강화
  phaseLateFeedMul: 1.40,       // 먹이 리스크 증가
  phaseLateDoublePiMul: 1.50,   // 쌍피 가치 증가

  // ── chooseMatch (V5와 동일 기본값) ──
  matchPiGainMul: 6.25,
  matchKwangBonus: 15.02,
  matchRibbonBonus: 10.02,
  matchFiveBonus: 8.0,
  matchDoublePiBonus: 18.0,
  matchMongBakFiveBonus: 33.83,
  // chooseMatch 전방 블로킹 (V5Plus 신규)
  matchFwdBlockMul: 1.45,       // 상대 콤보 차단 선택 시 보너스 배율

  // ── shouldGo (V5Plus: 유틸리티 비교 모델) ──
  // 기본 게이트 (V5와 유사)
  goOppOneAwayGate: 100,
  goScoreDiffBonus: 0.055,
  goDeckLowBonus: 0.08,
  goUnseeHighPiPenalty: 0.08,
  goBaseThreshold: -0.10,
  goOppScoreGateHigh: 6,
  goOppScoreGateLow: 4,
  goBigLeadScoreDiff: 4,
  goBigLeadMinScore: 8,
  goBigLeadOneAwayLate: 35,
  goBigLeadOneAwayEarly: 55,
  goBigLeadJokboThresh: 0.45,
  goBigLeadNextThresh: 0.45,
  goOneAwayThreshOpp4Late: 40,
  goOneAwayThreshOpp4Early: 60,
  goOneAwayThreshOpp3: 65,
  goOneAwayThreshOpp2: 75,
  goOneAwayThreshOpp1: 85,
  goOpp12JokboThresh: 0.55,
  goOpp12NextThresh: 0.55,
  goZeroOppOneAwayLate: 88,
  goZeroOppOneAwayEarly: 95,
  lateDeckMax: 10,
  // Upside/Risk 유틸리티 계수 (V5Plus 신규 - GO 억제 방향)
  goUpsideScoreMul: 0.07,       // 현재 점수 upside
  goUpsidePiMul: 0.030,         // pi upside (낮게)
  goUpsideSelfJokboMul: 0.35,   // 자기 jokbo potential
  goUpsideOneAwayMul: 0.10,     // 자기 one-away 보너스
  goUpsideCarryMul: 0.10,       // carry-over stop 인센티브
  goRiskPressureMul: 0.35,      // 위협 리스크 (높게)
  goRiskOneAwayMul: 0.28,       // one-away 리스크 (높게)
  goRiskOppJokboMul: 0.30,
  goRiskOppOneAwayMul: 0.07,
  goRiskGoCountMul: 0.10,       // go 횟수 리스크 (높게)
  goRiskLateDeckBonus: 0.08,
  goRiskSecondMoverMul: 0.12,   // 후공 리스크 추가
  stopLeadMul: 0.09,
  stopCarryMul: 0.12,
  stopTenBonus: 0.20,
  goUtilityThreshold: 0.10,     // 높은 임계값 = GO 억제 (V5 보다 보수적)

  // ── shouldBomb / selectBombMonth ──
  bombMinPiAdvantage: 1,
  bombOpponentJokboBlock: 0.4,

  // ── decideShaking ──
  shakingImmediateGainMul: 1.35,
  shakingComboGainMul: 1.15,
  shakingTempoBonusMul: 0.28,
  shakingScoreThreshold: 0.65,
  shakingAheadPenalty: 0.05,
};

/* 3) Shared helpers */
function safeNum(v, def = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : Number.isFinite(def) ? def : 0;
}

function otherPlayerKeyFromDeps(playerKey, deps) {
  return typeof deps.otherPlayerKey === "function"
    ? deps.otherPlayerKey(playerKey)
    : playerKey === "human" ? "ai" : "human";
}

function isSecondMover(state, playerKey) {
  return state?.firstMover != null
    ? state.firstMover !== playerKey
    : state?.players?.[playerKey]?.isSecond === true;
}

function piLikeValue(card, deps) {
  if (!card) return 0;
  if (card.bonus?.stealPi) return 3;
  if (typeof deps.cardPiValue === "function") return safeNum(deps.cardPiValue(card));
  if (card.category === "junk") return card.piValue != null ? safeNum(card.piValue) : 1;
  return 0;
}

function isDoublePiLike(card, deps) {
  if (!card) return false;
  if (typeof deps.isDoublePiCard === "function") return deps.isDoublePiCard(card);
  return DOUBLE_PI_MONTHS.includes(card.month) && card.category === "junk";
}

function comboCounts(player) {
  const cap = player?.captured || {};
  const kwang = (cap.kwang || []).length;
  const five = (cap.five || []).filter((c) => !c.isBonusCard).length;
  const ribbon = (cap.ribbon || []).length;
  const birds = ribbon; // approximation for godori tracking
  const red = (cap.ribbon || []).filter((c) => c.comboTags?.includes("red_ribbon")).length;
  const blue = (cap.ribbon || []).filter((c) => c.comboTags?.includes("blue_ribbon")).length;
  const plain = (cap.ribbon || []).filter((c) => c.comboTags?.includes("plain_ribbon")).length;
  return { kwang, five, ribbon, birds, red, blue, plain };
}

function hasCertainJokbo(player) {
  const sc = comboCounts(player);
  return sc.kwang >= 3 || sc.birds >= 3 || sc.red >= 3 || sc.blue >= 3 || sc.plain >= 3;
}

function countUnseen(state, idSet, viewerKey = null) {
  const vk = viewerKey === "human" || viewerKey === "ai" ? viewerKey : null;
  let seen = 0;
  for (const k of ["human", "ai"]) {
    const p = state.players?.[k];
    for (const cat of ["kwang", "five", "ribbon", "junk"]) for (const c of p?.captured?.[cat] || []) if (idSet.has(c?.id)) seen++;
    if (!vk || k === vk) {
      for (const c of p?.hand || []) if (idSet.has(c?.id)) seen++;
    }
  }
  for (const c of state.board || []) if (idSet.has(c?.id)) seen++;
  return Math.max(0, idSet.size - seen);
}

function _stopForOppFour(state, playerKey, deps, bTh, sTh) {
  if (countUnseen(state, BONUS_CARD_ID_SET, playerKey) >= bTh) return true;
  if (safeNum(deps.checkOpponentJokboProgress(state, playerKey)?.threat) >= 0.5) return true;
  return countUnseen(state, SSANGPI_WITH_GUKJIN_ID_SET, playerKey) >= sTh;
}

function _oppOneAway(state, playerKey, deps) {
  const deck = safeNum(state.deck?.length);
  const jokbo = deps.checkOpponentJokboProgress(state, playerKey);
  const next = safeNum(deps.nextTurnThreatScore(state, playerKey));
  let p = safeNum(deps.opponentThreatScore(state, playerKey)) * 35 + safeNum(jokbo?.threat) * 30 + next * 20;
  if (deck <= 15) p += 5; if (deck <= 10) p += 8; if (deck <= 6) p += 6; if (deck <= 3) p += 5;
  return { oppOneAwayProb: Math.max(0, Math.min(100, p)), jokboThreat: safeNum(jokbo?.threat), nextThreat: next, deckCount: deck };
}

function ownComboFinishBonus(sc, cards, P) {
  let bonus = 0;
  for (const c of cards) {
    const cat = c.category;
    if (cat === "kwang" && sc.kwang >= 2) bonus += P.comboFinishKwang;
    if (cat === "five" && sc.five >= 4) bonus += P.comboFinishPlain;
    if (cat === "ribbon") {
      if (sc.ribbon >= 4) bonus += P.comboFinishPlain;
      if (c.comboTags?.includes("red_ribbon") && sc.red >= 2) bonus += P.comboFinishRed;
      if (c.comboTags?.includes("blue_ribbon") && sc.blue >= 2) bonus += P.comboFinishBlue;
      if (c.comboTags?.includes("plain_ribbon") && sc.plain >= 2) bonus += P.comboFinishPlain;
      if (c.comboTags?.includes("godori") && sc.birds >= 2) bonus += P.comboFinishBirds;
    }
  }
  return bonus;
}

function opponentComboBlockBonus(month, jokbo, blkM, blkU, nextT, P) {
  if (!blkM?.has(month)) return 0;
  const urgency = safeNum(blkU?.get(month));
  const threat = safeNum(jokbo?.threat);
  return P.comboBlockBase + urgency * P.comboBlockUrgencyMul + nextT * P.comboBlockNextThreatMul * threat;
}

function getComboHoldMonths(state, playerKey, deps) {
  const hold = new Set();
  const opp = otherPlayerKeyFromDeps(playerKey, deps);
  const blkM = deps.blockingMonthsAgainst(state.players?.[opp], state.players?.[playerKey]);
  if (blkM) for (const m of blkM) hold.add(m);
  return hold;
}

function getLiveDoublePiMonths(state) {
  const live = new Set();
  const all = [...(state.board || []), ...(state.players?.human?.hand || []), ...(state.players?.ai?.hand || [])];
  for (const c of all) if (DOUBLE_PI_MONTHS.includes(c?.month) && c?.category === "junk") live.add(c.month);
  return live;
}

function discardTieOrder(card, deps, livePi) {
  if (card?.bonus?.stealPi) return 6;
  if (isDoublePiLike(card, deps) && livePi) return 1;
  return { five: 5, ribbon: 4, kwang: 3 }[card?.category] ?? 2;
}

/* Phase resolver */
function resolvePhase(deckCount, P) {
  if (deckCount >= safeNum(P.phaseEarlyDeck, 18)) return "early";
  if (deckCount >= safeNum(P.phaseMidDeck, 10)) return "mid";
  return "late";
}

/* 4) Hand ranking (phase-aware V5 baseline) */
function rankHandCardsV5Plus(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const P = { ...DEFAULT_PARAMS, ...params };
  const player = state.players?.[playerKey];
  if (!player?.hand?.length) return [];

  const opp = otherPlayerKeyFromDeps(playerKey, deps);
  const oppPlayer = state.players?.[opp];
  const ctx = deps.analyzeGameContext(state, playerKey);
  const myScore = safeNum(ctx.myScore);
  const oppScore = safeNum(ctx.oppScore);
  const selfPi = safeNum(ctx.selfPi, deps.capturedCountByCategory(player, "junk"));
  const oppPi = safeNum(ctx.oppPi, deps.capturedCountByCategory(oppPlayer, "junk"));
  const nextT = safeNum(deps.nextTurnThreatScore(state, playerKey));
  const jokbo = deps.checkOpponentJokboProgress(state, playerKey);
  const blkM = deps.blockingMonthsAgainst(oppPlayer, player);
  const blkU = deps.blockingUrgencyByMonth(oppPlayer, player);
  const sc = comboCounts(player);
  const rCnt = (player?.captured?.ribbon || []).length;
  const fCnt = (player?.captured?.five || []).filter((c) => !c.isBonusCard).length;
  const mongBak = safeNum(ctx.selfFive) <= 0 && safeNum(ctx.oppFive) >= 7;
  const deckCount = safeNum(state.deck?.length);
  const deck = deckCount;
  const bCnt = (state.board || []).length;
  const hCnt = (player?.hand || []).length;
  const lp = getLiveDoublePiMonths(state);
  const plan = deps.getFirstTurnDoublePiPlan(state, playerKey);
  const sec = isSecondMover(state, playerKey);
  const late = deck <= P.lateDeckMax;
  const phase = resolvePhase(deckCount, P);

  // 페이즈 배율 계산
  const comboMul = phase === "early" ? safeNum(P.phaseEarlyComboMul, 1.20)
    : phase === "late" ? safeNum(P.phaseLateComboMul, 0.85) : 1.0;
  const blockMul = phase === "early" ? safeNum(P.phaseEarlyBlockMul, 0.90)
    : phase === "late" ? safeNum(P.phaseLateBlockMul, 1.35) : 1.0;
  const feedMul = phase === "early" ? safeNum(P.phaseEarlyFeedMul, 0.85)
    : phase === "late" ? safeNum(P.phaseLateFeedMul, 1.40) : 1.0;
  const dpiLateBonus = phase === "late" ? safeNum(P.phaseLateDoublePiMul, 1.50) : 1.0;

  const ranked = player.hand.map((card) => {
    const month = card.month;
    const matches = (state.board || []).filter((b) => b.month === month);
    const matchCnt = matches.length;
    const pi = piLikeValue(card, deps);
    const isDpi = isDoublePiLike(card, deps);

    let score = 0;
    if (matchCnt === 0) {
      score += P.matchZeroBase;
    } else if (matchCnt === 1) {
      score += P.matchOneBase;
      score += deps.cardCaptureValue(matches[0]) * 0.8;
    } else if (matchCnt === 2) {
      score += P.matchTwoBase;
      const bestM = matches.reduce((a, b) => deps.cardCaptureValue(a) >= deps.cardCaptureValue(b) ? a : b);
      score += deps.cardCaptureValue(bestM) * 0.8;
    } else if (matchCnt >= 3) {
      score += P.matchThreeBase;
      const bestM = matches.reduce((a, b) => deps.cardCaptureValue(a) >= deps.cardCaptureValue(b) ? a : b);
      score += deps.cardCaptureValue(bestM) * safeNum(P.captureGainMulThree, 1.15);
      score += P.highValueMatchBonus;
    }

    // 피 가치
    let piGain = pi * P.piGainMul;
    if (selfPi >= 7 && selfPi <= 9) piGain *= P.piGainSelfHighMul;
    if (oppPi <= 5) piGain *= P.piGainOppLowMul;
    if (isDpi) {
      if (matchCnt > 0) {
        piGain += P.doublePiMatchBonus + (lp.has(month) ? P.doublePiMatchExtra : 0);
        // late 페이즈에서 쌍피 더 중요
        if (phase === "late") piGain *= dpiLateBonus;
      } else {
        piGain -= P.doublePiNoMatchPenalty;
      }
    }
    score += piGain;

    // 콤보 완성 (페이즈 배율 적용)
    const cfBonus = ownComboFinishBonus(sc, matchCnt > 0 ? [card, ...matches] : [card], P);
    score += cfBonus * comboMul;

    // 콤보 블로킹 (페이즈 배율 적용)
    const cbBonus = opponentComboBlockBonus(month, jokbo, blkM, blkU, nextT, P);
    score += cbBonus * blockMul;

    // 광/열/띠 추가 보너스
    if (matchCnt > 0) {
      if (card.category === "kwang" && sc.kwang >= 2) score += P.comboFinishKwang;
      if (rCnt >= 4 && card.category === "ribbon") score += P.ribbonFourBonus;
      if (fCnt >= 4 && card.category === "five") score += P.fiveFourBonus;
    }

    // 몽박
    if (mongBak) {
      if (card.category === "five" && matchCnt > 0) score += P.mongBakFiveBonus;
      else if (pi > 0 && matchCnt > 0) score -= P.mongBakPiPenalty;
    }

    // 무매치 버리기 페널티
    if (matchCnt === 0) {
      const livePiCard = isDpi && lp.has(month);
      if (livePiCard) {
        const pen = late ? P.discardDoublePiLivePenaltyLate : P.discardDoublePiLivePenalty;
        score -= pen;
      } else if (lp.has(month)) {
        const pen = late ? P.discardLivePiPenaltyLate : P.discardLivePiPenalty;
        score -= pen;
      } else if (isDpi) {
        score += P.discardDoublePiDeadBonus;
      }

      const ch = getComboHoldMonths(state, playerKey, deps);
      if (ch.has(month)) {
        score -= (late ? P.discardComboHoldPenaltyLate : P.discardComboHoldPenalty);
      }
      if (safeNum(deps.nextTurnThreatScore?.(state, playerKey)) >= 0.5) {
        score -= (late ? P.discardOneAwayPenaltyLate : P.discardOneAwayPenalty);
      }
      if (blkM?.has(month)) {
        score -= (late ? P.discardBlockMedPenaltyLate : P.discardBlockMedPenalty) * blockMul;
      }
      if (mongBak && card.category === "five") score -= P.discardMongBakFivePenalty;
      if (mongBak && card.category === "junk") score += 5;
      score += discardTieOrder(card, deps, lp) * 2.2;
    }

    // 먹이 리스크 (페이즈 배율 적용)
    const feed = safeNum(deps.estimateOpponentImmediateGainIfDiscard(state, playerKey, month));
    score -= feed * (matchCnt === 0 ? P.feedRiskNoMatchMul : P.feedRiskMatchMul) * feedMul;

    // 뻑 리스크
    const puk = safeNum(deps.isRiskOfPuk(state, playerKey, card, bCnt, hCnt));
    if (puk > 0) score -= puk * (deck <= 10 ? P.pukRiskHighMul : P.pukRiskNormalMul);
    else if (puk < 0) score += -puk * 1.4;

    // 첫 턴 계획 보너스
    if (plan.active && plan.months.has(month)) score += P.firstTurnPiPlanBonus;

    // 후공 추격 보너스
    if (sec && myScore < oppScore && pi > 0) score += pi * P.secondMoverPiBonus;

    // 알려진 월 보너스
    const knownCnt = safeNum(deps.countKnownMonthCards?.(state, month));
    if (knownCnt >= 3) score += P.discardKnownMonthBonus;
    else if (knownCnt === 0) score -= P.discardUnknownMonthPenalty;

    // 보너스카드 피 훔치기 카드
    if (card.bonus?.stealPi) score += P.discardBonusPiBonus;

    return { card, score, matches: matchCnt };
  });

  ranked.sort((a, b) => b.score - a.score);
  return ranked;
}

/* 5) Pending match-card selection (forward-blocking) */
function chooseMatchHeuristicV5Plus(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const P = { ...DEFAULT_PARAMS, ...params };
  const ids = state.pendingMatch?.boardCardIds || [];
  if (!ids.length) return null;
  const opp = otherPlayerKeyFromDeps(playerKey, deps);
  const self = state.players?.[playerKey], oppP = state.players?.[opp];
  const ctx = deps.analyzeGameContext(state, playerKey);
  const selfPi = safeNum(ctx.selfPi, deps.capturedCountByCategory(self, "junk"));
  const oppPi = safeNum(ctx.oppPi, deps.capturedCountByCategory(oppP, "junk"));
  const nextT = safeNum(deps.nextTurnThreatScore(state, playerKey));
  const jokbo = deps.checkOpponentJokboProgress(state, playerKey);
  const blkM = deps.blockingMonthsAgainst(oppP, self);
  const blkU = deps.blockingUrgencyByMonth(oppP, self);
  const sc = comboCounts(self);
  const rCnt = (self?.captured?.ribbon || []).length;
  const fCnt = (self?.captured?.five || []).filter((c) => !c.isBonusCard).length;
  const mongBak = safeNum(ctx.selfFive) <= 0 && safeNum(ctx.oppFive) >= 7;
  const deckCount = safeNum(state.deck?.length);
  const phase = resolvePhase(deckCount, P);
  const blockMul = phase === "late" ? safeNum(P.phaseLateBlockMul, 1.35) : 1.0;

  let best = null, bestScore = -Infinity;
  for (const c of (state.board || []).filter((c) => ids.includes(c.id))) {
    const pi = piLikeValue(c, deps);
    let score = deps.cardCaptureValue(c) * 0.8 + pi * P.matchPiGainMul;
    if (c.category === "kwang") score += P.matchKwangBonus;
    if (c.category === "ribbon") score += P.matchRibbonBonus;
    if (c.category === "five") score += P.matchFiveBonus;
    if (selfPi >= 7 && selfPi <= 9) score += pi * 1.8;
    if (oppPi <= 5) score += pi * 1.4;
    if (isDoublePiLike(c, deps)) score += P.matchDoublePiBonus;
    score += ownComboFinishBonus(sc, [c], P);
    // 전방 블로킹: 상대 콤보를 막는 카드 선택 시 가중치 증가
    const cbBonus = opponentComboBlockBonus(c.month, jokbo, blkM, blkU, nextT, P);
    score += cbBonus * blockMul * safeNum(P.matchFwdBlockMul, 1.45);
    if (rCnt >= 4 && c.category === "ribbon") score += P.ribbonFourBonus;
    if (fCnt >= 4 && c.category === "five") score += P.fiveFourBonus;
    if (mongBak) {
      if (c.category === "five") score += P.matchMongBakFiveBonus;
      else if (pi > 0) score -= P.mongBakPiPenalty;
    }
    score += safeNum(deps.monthStrategicPriority?.(c.month)) * 0.25;
    if (ctx.mode === "DESPERATE_DEFENSE" && pi <= 0) score -= 0.45;
    if (score > bestScore) { bestScore = score; best = c; }
  }
  return best?.id ?? null;
}

/* 6) GO/STOP decision (utility-filtered V5 gate) */
function shouldGoV5Plus(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const P = { ...DEFAULT_PARAMS, ...params };

  // 즉시 STOP 조건: 상대 파산 가능
  if (deps.canBankruptOpponentByStop?.(state, playerKey)) return false;

  const ctx = deps.analyzeGameContext(state, playerKey);
  const myScore = safeNum(ctx.myScore);
  const tr = _oppOneAway(state, playerKey, deps);
  const late = tr.deckCount <= P.lateDeckMax;
  const sec = isSecondMover(state, playerKey);
  const secG = sec ? safeNum(P.secondMoverGoGateShrink) : 0;
  const opp = otherPlayerKeyFromDeps(playerKey, deps);
  const oppPiBase = safeNum(ctx.oppPi, deps.capturedCountByCategory(state.players?.[opp], "junk"));
  const selfPi = safeNum(ctx.selfPi, deps.capturedCountByCategory(state.players?.[playerKey], "junk"));
  const certain = hasCertainJokbo(state.players?.[playerKey]);
  const unseenHi = countUnseen(state, SSANGPI_WITH_GUKJIN_ID_SET, playerKey)
    + countUnseen(state, BONUS_CARD_ID_SET, playerKey);

  const gb = deps.analyzeGukjinBranches?.(state, playerKey);
  let oppScoreRisk = safeNum(ctx.oppScore), oppPiRisk = oppPiBase;
  if (gb?.scenarios?.length) {
    for (const s of gb.scenarios) {
      oppScoreRisk = Math.max(oppScoreRisk, safeNum(s?.oppScore));
      oppPiRisk = Math.max(oppPiRisk, safeNum(s?.oppPi));
    }
    for (const s of gb.scenarios) {
      if (safeNum(s?.oppScore) >= 6) return false;
      if (safeNum(s?.oppScore) >= 5 && !certain) return false;
      if (unseenHi >= 2 && safeNum(s?.oppPi) >= 7 && !certain) return false;
    }
  }

  const diff = myScore - oppScoreRisk;
  if (unseenHi >= 2 && oppPiRisk >= 7 && !certain) return false;
  if (tr.oppOneAwayProb >= safeNum(P.goOppOneAwayGate, 100)) return false;

  // 기본 게이트 통과 여부 (V5와 동일 구조)
  if (!certain) {
    const conf = 0.5 + diff * safeNum(P.goScoreDiffBonus)
      + (late ? safeNum(P.goDeckLowBonus) : 0)
      - unseenHi * safeNum(P.goUnseeHighPiPenalty);
    if (conf < safeNum(P.goBaseThreshold)) return false;
  }

  const gH = Math.max(5, Math.min(7, Math.round(safeNum(P.goOppScoreGateHigh, 6))));
  const gL = Math.max(3, Math.min(5, Math.round(safeNum(P.goOppScoreGateLow, 4))));

  if (oppScoreRisk >= gH) return false;

  // ── 분기별 즉시 return (V5와 동일 구조) - 유틸리티 체크 이전에 결정 ──
  if (oppScoreRisk >= gH - 1) {
    if (_stopForOppFour(state, playerKey, deps, 1, 2) || sec) return false;
    const bigLead = diff >= P.goBigLeadScoreDiff && myScore >= P.goBigLeadMinScore;
    const lowThreat = tr.oppOneAwayProb < (late ? P.goBigLeadOneAwayLate : P.goBigLeadOneAwayEarly)
      && tr.jokboThreat < P.goBigLeadJokboThresh && tr.nextThreat < P.goBigLeadNextThresh;
    // V5와 동일: 조건 충족 못하면 즉시 STOP
    if (!bigLead || !lowThreat) return false;
    // 조건 충족 시 유틸리티 필터로 추가 억제
  } else if (oppScoreRisk >= gL) {
    if (_stopForOppFour(state, playerKey, deps, 2, 3)) return false;
    if (tr.oppOneAwayProb >= (late ? P.goOneAwayThreshOpp4Late : P.goOneAwayThreshOpp4Early) - secG) return false;
    // 통과 시 유틸리티 필터 적용
  } else if (oppScoreRisk >= 1) {
    if (oppScoreRisk === 3 && tr.jokboThreat >= 0.5) return false;
    const base = oppScoreRisk === 3 ? P.goOneAwayThreshOpp3
      : oppScoreRisk === 2 ? P.goOneAwayThreshOpp2 : P.goOneAwayThreshOpp1;
    if (tr.oppOneAwayProb >= base - (late ? 1 : 0) - secG) return false;
    if (oppScoreRisk <= 2 && (tr.jokboThreat >= P.goOpp12JokboThresh
      || tr.nextThreat >= P.goOpp12NextThresh
      || (late && tr.jokboThreat >= 0.4))) return false;
    if (sec && diff <= 0 && tr.oppOneAwayProb >= base - 5 - secG) return false;
    // 통과 시 유틸리티 필터 적용
  } else {
    const zThresh = late ? safeNum(P.goZeroOppOneAwayLate, 88) : safeNum(P.goZeroOppOneAwayEarly, 95);
    if (tr.oppOneAwayProb >= zThresh) return false;
    // 통과 시 유틸리티 필터 적용
  }

  // ── V5Plus 추가: 유틸리티 후처리 필터 ──
  // V5 게이트를 통과한 GO 결정 중 일부를 추가로 억제한다.
  // (GO 횟수 감소 방향 - 실패율 개선 목적)
  const carry = safeNum(state.carryOverMultiplier, 1);
  const goCount = safeNum(state.players?.[playerKey]?.goCount, 0);

  const selfJokbo = deps.estimateJokboExpectedPotential?.(state, playerKey, opp) || { total: 0, oneAwayCount: 0 };
  const oppJokbo = deps.estimateOpponentJokboExpectedPotential?.(state, playerKey) || { total: 0, oneAwayCount: 0 };

  const upside =
    Math.max(0, myScore - 6) * P.goUpsideScoreMul +
    selfPi * P.goUpsidePiMul +
    safeNum(selfJokbo.total) * P.goUpsideSelfJokboMul +
    safeNum(selfJokbo.oneAwayCount) * P.goUpsideOneAwayMul +
    (oppPiBase <= 5 ? 0.02 : 0);

  const oneAwayProbNorm = tr.oppOneAwayProb / 100;
  const risk =
    tr.jokboThreat * P.goRiskPressureMul +
    oneAwayProbNorm * P.goRiskOneAwayMul +
    safeNum(oppJokbo.total) * P.goRiskOppJokboMul +
    safeNum(oppJokbo.oneAwayCount) * P.goRiskOppOneAwayMul +
    goCount * P.goRiskGoCountMul +
    (tr.deckCount <= 10 ? P.goRiskLateDeckBonus : 0) +
    (sec ? safeNum(P.goRiskSecondMoverMul, 0.12) : 0);

  const stopValue =
    Math.max(0, diff) * P.stopLeadMul +
    Math.max(0, carry - 1) * P.stopCarryMul +
    (myScore >= 10 ? P.stopTenBonus : 0);

  // carry-over 시 보수적으로
  let utilityThreshold = safeNum(P.goUtilityThreshold, 0.10);
  if (carry >= 2) utilityThreshold += carry * safeNum(P.goUpsideCarryMul, 0.10);

  const utility = upside - risk - stopValue;
  return utility >= utilityThreshold;
}

/* 7) President/Gukjin decisions */
function shouldPresidentStopV5Plus(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const ctx = deps.analyzeGameContext(state, playerKey);
  const diff = safeNum(ctx.myScore) - safeNum(ctx.oppScore);
  const co = safeNum(state.carryOverMultiplier, 1);
  const sec = isSecondMover(state, playerKey);
  if (diff >= 3 && co <= 1) return true;
  if (diff <= -1 || (sec && diff <= 0) || co >= 2) return false;
  return diff >= 1;
}

function chooseGukjinHeuristicV5Plus(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const ctx = deps.analyzeGameContext(state, playerKey);
  const br = deps.analyzeGukjinBranches(state, playerKey);
  const sf = safeNum(ctx.selfFive), of_ = safeNum(ctx.oppFive);
  if (sf <= 0 && of_ >= 6) return "junk";
  if (sf >= 7 && of_ <= 0) return "five";
  if (br?.enabled && br.scenarios?.length) {
    const fSc = br.scenarios.find((s) => s?.selfMode === "five");
    const jSc = br.scenarios.find((s) => s?.selfMode === "junk");
    if (fSc && jSc) return (safeNum(fSc.myScore) - safeNum(fSc.oppScore)) >= (safeNum(jSc.myScore) - safeNum(jSc.oppScore)) ? "five" : "junk";
  }
  return sf >= 7 ? "five" : "junk";
}

/* 8) Bomb decision helpers */
function selectBombMonthV5Plus(state, playerKey, bombMonths, deps, params = DEFAULT_PARAMS) {
  if (!bombMonths?.length) return null;
  const months = [...bombMonths];
  return months.length === 1 ? months[0]
    : months.reduce((b, m) => safeNum(deps.monthBoardGain(state, m)) > safeNum(deps.monthBoardGain(state, b)) ? m : b, months[0]);
}

function shouldBombV5Plus(state, playerKey, bombMonths, deps, params = DEFAULT_PARAMS) {
  if (!bombMonths?.length) return false;
  const P = { ...DEFAULT_PARAMS, ...params };
  const ctx = deps.analyzeGameContext(state, playerKey);
  const myScore = safeNum(ctx.myScore);
  const oppScore = safeNum(ctx.oppScore);
  const jokbo = deps.checkOpponentJokboProgress(state, playerKey);
  // 상대 콤보가 위협적이고 폭탄으로 차단 가능하면 폭탄 선언
  if (safeNum(jokbo?.threat) >= safeNum(P.bombOpponentJokboBlock, 0.4)) {
    const opp = otherPlayerKeyFromDeps(playerKey, deps);
    const oppPi = deps.capturedCountByCategory(state.players?.[opp], "junk");
    const selfPi = deps.capturedCountByCategory(state.players?.[playerKey], "junk");
    if (selfPi - oppPi >= safeNum(P.bombMinPiAdvantage, 1)) return true;
  }
  // 크게 앞서고 있으면 폭탄으로 빠른 종료
  if (myScore - oppScore >= 3 && myScore >= 7) return true;
  return false;
}

/* 9) Shaking decision */
function decideShakingV5Plus(state, playerKey, shakingMonths, deps, params = DEFAULT_PARAMS) {
  const P = { ...DEFAULT_PARAMS, ...params };
  if (!shakingMonths?.length) return { allow: false, month: null, score: -Infinity };
  const bbm = deps.boardMatchesByMonth(state);
  if ((state.players?.[playerKey]?.hand || []).some((c) => (bbm.get(c.month) || []).length > 0))
    return { allow: false, month: null, score: -Infinity };

  const ctx = deps.analyzeGameContext(state, playerKey);
  const my = safeNum(ctx.myScore), opp = safeNum(ctx.oppScore);
  if (opp >= 5 && opp >= my + 2) return { allow: false, month: null, score: -Infinity };

  const livePi = getLiveDoublePiMonths(state);
  const ch = getComboHoldMonths(state, playerKey, deps);
  const plan = deps.getFirstTurnDoublePiPlan(state, playerKey);
  let best = { allow: false, month: null, score: -Infinity, highImpact: false };

  for (const month of shakingMonths) {
    const ig = safeNum(deps.shakingImmediateGainScore(state, playerKey, month));
    const cg = safeNum(deps.ownComboOpportunityScore(state, playerKey, month));
    const imp = deps.isHighImpactShaking(state, playerKey, month);
    const kn = safeNum(deps.countKnownMonthCards(state, month));
    let score = ig * safeNum(P.shakingImmediateGainMul, 1.35)
      + cg * safeNum(P.shakingComboGainMul, 1.15)
      + (kn <= 2 ? 0.25 : kn >= 4 ? -0.1 : 0);
    if (imp?.hasDoublePiLine) score += 0.35;
    if (imp?.directThreeGwang) score += 0.3;
    if (imp?.highImpact) score += 0.4;
    if (my < opp) score += safeNum(P.shakingTempoBonusMul);
    if (livePi.has(month) && !ch.has(month)) score += 0.55;
    if (ch.has(month)) score -= 0.25;
    if (plan.active && plan.months.has(month)) score += 0.3;
    if (score > best.score) best = { allow: false, month, score, highImpact: !!imp?.highImpact };
  }

  const prefDpi = best.month != null && livePi.has(best.month) && !ch.has(best.month);
  const allow = (my > opp || prefDpi)
    && best.score >= safeNum(P.shakingScoreThreshold, 0.65) + (my > opp ? safeNum(P.shakingAheadPenalty) : 0);
  return { ...best, allow };
}
