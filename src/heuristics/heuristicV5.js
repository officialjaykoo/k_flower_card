export {
  rankHandCardsV5,
  chooseMatchHeuristicV5,
  chooseGukjinHeuristicV5,
  shouldPresidentStopV5,
  shouldGoV5,
  selectBombMonthV5,
  shouldBombV5,
  decideShakingV5
};

/* ============================================================================
 * Heuristic V5
 * - Optuna-tuned stable rule policy
 * - score gates first, tactical tie-breakers second
 * - exported decisions: rank / match / go / bomb / shaking / president / gukjin
 * ========================================================================== */

/* 1) Constants and parameter surface */
const GUKJIN_CARD_ID = "I0";
const DOUBLE_PI_MONTHS = Object.freeze([11, 12, 13]);
const BONUS_CARD_ID_SET = Object.freeze(new Set(["M0", "M1"]));
const SSANGPI_WITH_GUKJIN_ID_SET = Object.freeze(new Set(["K1", "L3", GUKJIN_CARD_ID]));

export const DEFAULT_PARAMS = {
  // ── rankHandCards ──
  matchZeroBase: -48.1,        // trial0: -noMatchPenalty*10
  matchOneBase: 7.53,          // trial0
  matchTwoBase: 14.80,         // trial0
  matchThreeBase: 12.89,       // trial0
  captureGainMulThree: 1.15,
  highValueMatchBonus: 3.77,   // trial0
  // Pi value weighting
  piGainMul: 6.25,             // trial0: matchPiGainMul
  piGainSelfHighMul: 1.8,
  piGainOppLowMul: 1.4,
  doublePiMatchBonus: 16.49,   // trial0: matchDoublePiBonus
  doublePiMatchExtra: 6.0,
  doublePiNoMatchPenalty: 14.0,
  // Combo pressure
  comboFinishBirds: 30.0,  comboFinishRed: 27.0,  comboFinishBlue: 27.0,
  comboFinishPlain: 27.0,  comboFinishKwang: 32.0,
  comboBlockBase: 19.08,       // trial0: blockingBonus
  comboBlockUrgencyMul: 0.51,  // trial0: jokboBlockBonus*0.1
  comboBlockNextThreatMul: 4.5,
  ribbonFourBonus: 34.0, fiveFourBonus: 36.0,
  // Mong-bak pressure
  mongBakFiveBonus: 33.83,     // trial0: matchMongBakFiveBonus
  mongBakPiPenalty: 8.0,
  // No-match discard penalties
  discardLivePiPenalty: 24.0,       discardLivePiPenaltyLate: 36.0,
  discardDoublePiLivePenalty: 16.0, discardDoublePiLivePenaltyLate: 26.0,
  discardDoublePiDeadBonus: 6.0,
  discardComboHoldPenalty: 44.0,    discardComboHoldPenaltyLate: 56.0,
  discardOneAwayPenalty: 42.0,      discardOneAwayPenaltyLate: 58.0,
  discardBlockMedPenalty: 20.0,     discardBlockMedPenaltyLate: 30.0,
  discardMongBakFivePenalty: 28.0,  discardBonusPiBonus: 26.0,
  discardKnownMonthBonus: 1.9,      discardUnknownMonthPenalty: 1.8,
  // Feed and puk risk
  feedRiskNoMatchMul: 4.51,    // trial0: feedRiskMul
  feedRiskMatchMul: 1.05,      // trial0
  pukRiskHighMul: 3.99,        // trial0
  pukRiskNormalMul: 3.33,      // trial0
  firstTurnPiPlanBonus: 7.46,  // trial0
  lockedMonthPenalty: 6.0,
  // Second-mover adjustments
  secondMoverGoGateShrink: 4.0, secondMoverBlockBonus: 2.0, secondMoverPiBonus: 1.5,
  // chooseMatch
  matchPiGainMul: 6.25,        // trial0
  matchKwangBonus: 15.02,      // trial0
  matchRibbonBonus: 10.02,     // trial0
  matchFiveBonus: 9.77,        // trial0
  matchDoublePiBonus: 16.49,   // trial0
  matchMongBakFiveBonus: 33.83,// trial0
  // shouldGo
  goBaseThreshold: 0.482,      goOppOneAwayGate: 46.97,
  goScoreDiffBonus: 0.064,     goDeckLowBonus: 0.117,
  goUnseeHighPiPenalty: 0.118, // trial0
  goOppScoreGateLow: 3,        goOppScoreGateHigh: 6,
  goBigLeadScoreDiff: 8,       goBigLeadMinScore: 11,
  goBigLeadOneAwayEarly: 25,   goBigLeadOneAwayLate: 20,
  goBigLeadJokboThresh: 0.3,   goBigLeadNextThresh: 0.35,
  goOneAwayThreshOpp0: 37,     goOneAwayThreshOpp0Late: 33,
  goOneAwayThreshOpp1: 34,     goOneAwayThreshOpp2: 38,     goOneAwayThreshOpp3: 43,
  goOneAwayThreshOpp4Early: 32, goOneAwayThreshOpp4Late: 28,
  goOpp0JokboThresh: 0.37,     goOpp0NextThresh: 0.47,
  goOpp12JokboThresh: 0.35,    goOpp12NextThresh: 0.45,
  lateDeckMax: 10,
  // shouldBomb
  bombImpactMinGain: 0.79,     // trial0
  // decideShaking
  shakingScoreThreshold: 0.618, shakingImmediateGainMul: 1.97,
  shakingComboGainMul: 1.22,    shakingTempoBonusMul: 0.271,
  shakingAheadPenalty: 0.295,   // trial0
};

/* 2) Shared helpers */
function safeNum(v, fb = 0) { const n = Number(v); return Number.isFinite(n) ? n : fb; }
function otherPlayerKeyFromDeps(playerKey, deps) {
  if (typeof deps?.otherPlayerKey === "function") return deps.otherPlayerKey(playerKey);
  return playerKey === "human" ? "ai" : "human";
}
function isSecondMover(state, k) { const f = state?.startingTurnKey; return (f === "human" || f === "ai") ? f !== k : false; }
function hasComboTag(c, tag) { return Array.isArray(c?.comboTags) && c.comboTags.includes(tag); }
function countComboTag(arr, tag) { return (arr || []).reduce((n, c) => hasComboTag(c, tag) ? n + 1 : n, 0); }
function comboCounts(player) {
  return {
    red:   countComboTag(player?.captured?.ribbon || [], "redRibbons"),
    blue:  countComboTag(player?.captured?.ribbon || [], "blueRibbons"),
    plain: countComboTag(player?.captured?.ribbon || [], "plainRibbons"),
    birds: countComboTag(player?.captured?.five   || [], "fiveBirds"),
    kwang: (player?.captured?.kwang || []).length,
  };
}
function hasCategory(cards, cat) { return (cards || []).some((c) => c?.category === cat); }
function piLikeValue(card, deps) {
  if (!card) return 0;
  if (card.id === GUKJIN_CARD_ID) return 2;
  return card.category === "junk" ? safeNum(deps.junkPiValue(card)) : 0;
}
function isDoublePiLike(card, deps) {
  return !!card && (card.id === GUKJIN_CARD_ID || (card.category === "junk" && safeNum(deps.junkPiValue(card)) >= 2));
}
function hasCertainJokbo(player) {
  const c = comboCounts(player);
  return c.kwang >= 3 || c.birds >= 3 || c.red >= 3 || c.blue >= 3 || c.plain >= 3;
}
function ownComboFinishBonus(cc, cards, P) {
  let b = 0;
  if (cc.birds >= 2 && cards.some((c) => hasComboTag(c, "fiveBirds")))    b += P.comboFinishBirds;
  if (cc.red   >= 2 && cards.some((c) => hasComboTag(c, "redRibbons")))   b += P.comboFinishRed;
  if (cc.blue  >= 2 && cards.some((c) => hasComboTag(c, "blueRibbons")))  b += P.comboFinishBlue;
  if (cc.plain >= 2 && cards.some((c) => hasComboTag(c, "plainRibbons"))) b += P.comboFinishPlain;
  if (cc.kwang >= 2 && cards.some((c) => c?.category === "kwang"))        b += P.comboFinishKwang;
  return b;
}
function opponentComboBlockBonus(month, jokbo, blkM, blkU, nextT, P) {
  let b = 0;
  const mu = safeNum(jokbo?.monthUrgency?.get(month));
  if (mu > 0) b += P.comboBlockBase + mu * P.comboBlockUrgencyMul;
  if (blkM?.has(month)) { b += safeNum(blkU?.get(month), 2) >= 3 ? 18 : 10; b += nextT * P.comboBlockNextThreatMul; }
  return b;
}
function getLiveDoublePiMonths(state) {
  const cap = new Set();
  for (const k of ["human", "ai"])
    for (const c of state.players?.[k]?.captured?.junk || [])
      if (DOUBLE_PI_MONTHS.includes(c?.month) && safeNum(c?.piValue) >= 2) cap.add(c.month);
  return new Set(DOUBLE_PI_MONTHS.filter((m) => !cap.has(m)));
}
function getComboHoldMonths(state, playerKey, deps) {
  const opp = otherPlayerKeyFromDeps(playerKey, deps);
  const hold = new Set();
  for (const m of deps.blockingMonthsAgainst(state.players?.[playerKey], state.players?.[opp])) hold.add(m);
  for (const m of deps.blockingMonthsAgainst(state.players?.[opp], state.players?.[playerKey])) hold.add(m);
  return hold;
}
function discardTieOrder(card, deps, livePi) {
  if (card?.bonus?.stealPi) return 6;
  if (isDoublePiLike(card, deps) && livePi) return 1;
  return { five: 5, ribbon: 4, kwang: 3 }[card?.category] ?? 2;
}
function countUnseen(state, idSet, viewerKey = null) {
  const vk = viewerKey === "human" || viewerKey === "ai" ? viewerKey : null;
  let seen = 0;
  for (const k of ["human", "ai"]) {
    const p = state.players?.[k];
    for (const cat of ["kwang","five","ribbon","junk"]) for (const c of p?.captured?.[cat]||[]) if (idSet.has(c?.id)) seen++;
    if (!vk || k === vk) {
      for (const c of p?.hand||[]) if (idSet.has(c?.id)) seen++;
    }
  }
  for (const c of state.board||[]) if (idSet.has(c?.id)) seen++;
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
  const next  = safeNum(deps.nextTurnThreatScore(state, playerKey));
  let p = safeNum(deps.opponentThreatScore(state, playerKey)) * 35 + safeNum(jokbo?.threat) * 30 + next * 20;
  if (deck <= 15) p += 5; if (deck <= 10) p += 8; if (deck <= 6) p += 6; if (deck <= 3) p += 5;
  return { oppOneAwayProb: Math.max(0, Math.min(100, p)), jokboThreat: safeNum(jokbo?.threat), nextThreat: next, deckCount: deck };
}

/* 3) Hand ranking */
function rankHandCardsV5(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const P = { ...DEFAULT_PARAMS, ...params };
  const player = state.players?.[playerKey];
  if (!player?.hand?.length) return [];

  const opp = otherPlayerKeyFromDeps(playerKey, deps);
  const oppPlayer = state.players?.[opp];
  const ctx = deps.analyzeGameContext(state, playerKey);
  const sec = isSecondMover(state, playerKey);
  const selfPi = safeNum(ctx.selfPi, deps.capturedCountByCategory(player, "junk"));
  const oppPi  = safeNum(ctx.oppPi,  deps.capturedCountByCategory(oppPlayer, "junk"));
  const nextT  = safeNum(deps.nextTurnThreatScore(state, playerKey));
  const jokbo  = deps.checkOpponentJokboProgress(state, playerKey);
  const blkM   = deps.blockingMonthsAgainst(oppPlayer, player);
  const blkU   = deps.blockingUrgencyByMonth(oppPlayer, player);
  const plan   = deps.getFirstTurnDoublePiPlan(state, playerKey);
  const bbm    = deps.boardMatchesByMonth(state);
  const bCnt   = deps.monthCounts(state.board || []);
  const hCnt   = deps.monthCounts(player.hand || []);
  const cCnt   = deps.capturedMonthCounts(state);
  const deck   = safeNum(state.deck?.length);
  const late   = deck <= 8;
  const sc     = comboCounts(player);
  const rCnt   = (player?.captured?.ribbon || []).length;
  const fCnt   = (player?.captured?.five   || []).length;
  const mongBak = safeNum(ctx.selfFive) <= 0 && safeNum(ctx.oppFive) >= 7;
  const livePi  = getLiveDoublePiMonths(state);
  const hold    = getComboHoldMonths(state, playerKey, deps);

  const ranked = player.hand.map((card) => {
    const matches  = bbm.get(card.month) || [];
    const capCards = [card, ...matches];
    const capGain  = matches.reduce((s, c) => s + deps.cardCaptureValue(c), 0);
    const ownVal   = deps.cardCaptureValue(card);
    const piGain   = capCards.reduce((s, c) => s + piLikeValue(c, deps), 0);
    const dpiCnt   = capCards.filter((c) => isDoublePiLike(c, deps)).length;
    const known    = safeNum(bCnt.get(card.month)) + safeNum(hCnt.get(card.month)) + safeNum(cCnt.get(card.month));
    const lp       = livePi.has(card.month);
    const ch       = hold.has(card.month);
    const bu       = safeNum(blkU.get(card.month));
    const ju       = safeNum(jokbo?.monthUrgency?.get(card.month));
    const oat      = bu >= 3 || ju >= 24;

    let score =
      matches.length === 0 ? P.matchZeroBase - ownVal * 0.9 :
      matches.length === 1 ? P.matchOneBase + capGain - ownVal * 0.1 :
      matches.length === 2 ? P.matchTwoBase + capGain :
                             P.matchThreeBase + capGain * P.captureGainMulThree;

    if (matches.some((m) => ["kwang","five","ribbon"].includes(m?.category))) score += safeNum(P.highValueMatchBonus);
    score += piGain * P.piGainMul;
    if (selfPi >= 7 && selfPi <= 9) score += piGain * P.piGainSelfHighMul;
    if (oppPi  <= 5)                 score += piGain * P.piGainOppLowMul;
    if (dpiCnt > 0) score += P.doublePiMatchBonus + (dpiCnt - 1) * P.doublePiMatchExtra;
    if (matches.length === 0 && isDoublePiLike(card, deps)) score -= P.doublePiNoMatchPenalty;
    score += ownComboFinishBonus(sc, capCards, P);
    score += opponentComboBlockBonus(card.month, jokbo, blkM, blkU, nextT, P);
    if (sec && blkM.has(card.month)) score += P.secondMoverBlockBonus;
    if (rCnt >= 4 && hasCategory(capCards, "ribbon")) score += P.ribbonFourBonus;
    if (fCnt >= 4 && hasCategory(capCards, "five"))   score += P.fiveFourBonus;
    if (mongBak) {
      if (hasCategory(capCards, "five")) score += P.mongBakFiveBonus;
      else if (piGain > 0)               score -= P.mongBakPiPenalty;
    }
    if (matches.length === 0) {
      if (known >= 3)      score += P.discardKnownMonthBonus;
      else if (known <= 1) score -= P.discardUnknownMonthPenalty;
    }
    if (matches.length > 0 && dpiCnt === 0 && safeNum(cCnt.get(card.month)) >= 2 && known >= 3) score -= P.lockedMonthPenalty;
    if (matches.length === 0) {
      let ds = 0;
      if (card?.bonus?.stealPi)                  ds += P.discardBonusPiBonus;
      if (lp)                                    ds -= late ? P.discardLivePiPenaltyLate       : P.discardLivePiPenalty;
      if (isDoublePiLike(card, deps) && lp)      ds -= late ? P.discardDoublePiLivePenaltyLate : P.discardDoublePiLivePenalty;
      if (isDoublePiLike(card, deps) && !lp)     ds += P.discardDoublePiDeadBonus;
      if (ch)                                    ds -= late ? P.discardComboHoldPenaltyLate    : P.discardComboHoldPenalty;
      if (oat)                                   ds -= late ? P.discardOneAwayPenaltyLate      : P.discardOneAwayPenalty;
      else if (bu >= 2 || ju >= 20)              ds -= late ? P.discardBlockMedPenaltyLate     : P.discardBlockMedPenalty;
      if (mongBak && card.category === "five")   ds -= P.discardMongBakFivePenalty;
      if (mongBak && card.category === "junk")   ds += 5;
      ds += discardTieOrder(card, deps, lp) * 2.2;
      score += ds;
    }
    const feed = safeNum(deps.estimateOpponentImmediateGainIfDiscard(state, playerKey, card.month));
    score -= feed * (matches.length === 0 ? P.feedRiskNoMatchMul : P.feedRiskMatchMul);
    const puk = safeNum(deps.isRiskOfPuk(state, playerKey, card, bCnt, hCnt));
    if (puk > 0)      score -= puk * (deck <= 10 ? P.pukRiskHighMul : P.pukRiskNormalMul);
    else if (puk < 0) score += -puk * 1.4;
    if (plan.active && plan.months.has(card.month)) score += P.firstTurnPiPlanBonus;
    if (sec && safeNum(ctx.myScore) < safeNum(ctx.oppScore) && piGain > 0) score += piGain * P.secondMoverPiBonus;
    return { card, score, matches: matches.length };
  });

  ranked.sort((a, b) => b.score - a.score);
  return ranked;
}

/* 4) Pending match-card selection */
function chooseMatchHeuristicV5(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const P = { ...DEFAULT_PARAMS, ...params };
  const ids = state.pendingMatch?.boardCardIds || [];
  if (!ids.length) return null;
  const opp = otherPlayerKeyFromDeps(playerKey, deps);
  const self = state.players?.[playerKey], oppP = state.players?.[opp];
  const ctx  = deps.analyzeGameContext(state, playerKey);
  const selfPi = safeNum(ctx.selfPi, deps.capturedCountByCategory(self, "junk"));
  const oppPi  = safeNum(ctx.oppPi,  deps.capturedCountByCategory(oppP, "junk"));
  const nextT  = safeNum(deps.nextTurnThreatScore(state, playerKey));
  const jokbo  = deps.checkOpponentJokboProgress(state, playerKey);
  const blkM   = deps.blockingMonthsAgainst(oppP, self);
  const blkU   = deps.blockingUrgencyByMonth(oppP, self);
  const sc     = comboCounts(self);
  const rCnt   = (self?.captured?.ribbon || []).length;
  const fCnt   = (self?.captured?.five   || []).length;
  const mongBak = safeNum(ctx.selfFive) <= 0 && safeNum(ctx.oppFive) >= 7;

  let best = null, bestScore = -Infinity;
  for (const c of (state.board || []).filter((c) => ids.includes(c.id))) {
    const pi = piLikeValue(c, deps);
    let score = deps.cardCaptureValue(c) * 0.8 + pi * P.matchPiGainMul;
    if (c.category === "kwang")  score += P.matchKwangBonus;
    if (c.category === "ribbon") score += P.matchRibbonBonus;
    if (c.category === "five")   score += P.matchFiveBonus;
    if (selfPi >= 7 && selfPi <= 9) score += pi * 1.8;
    if (oppPi  <= 5)                 score += pi * 1.4;
    if (isDoublePiLike(c, deps)) score += P.matchDoublePiBonus;
    score += ownComboFinishBonus(sc, [c], P);
    score += opponentComboBlockBonus(c.month, jokbo, blkM, blkU, nextT, P);
    if (rCnt >= 4 && c.category === "ribbon") score += P.ribbonFourBonus;
    if (fCnt >= 4 && c.category === "five")   score += P.fiveFourBonus;
    if (mongBak) {
      if (c.category === "five")  score += P.matchMongBakFiveBonus;
      else if (pi > 0)            score -= P.mongBakPiPenalty;
    }
    score += safeNum(deps.monthStrategicPriority?.(c.month)) * 0.25;
    if (ctx.mode === "DESPERATE_DEFENSE" && pi <= 0) score -= 0.45;
    if (score > bestScore) { bestScore = score; best = c; }
  }
  return best?.id ?? null;
}

/* 5) GO/STOP decision gate */
function shouldGoV5(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const P = { ...DEFAULT_PARAMS, ...params };
  if (deps.canBankruptOpponentByStop?.(state, playerKey)) return false;

  const ctx = deps.analyzeGameContext(state, playerKey);
  const myScore = safeNum(ctx.myScore);
  const tr  = _oppOneAway(state, playerKey, deps);
  const late = tr.deckCount <= P.lateDeckMax;
  const sec  = isSecondMover(state, playerKey);
  const secG = sec ? safeNum(P.secondMoverGoGateShrink) : 0;
  const opp  = otherPlayerKeyFromDeps(playerKey, deps);
  const oppPiBase = safeNum(ctx.oppPi, deps.capturedCountByCategory(state.players?.[opp], "junk"));
  const certain   = hasCertainJokbo(state.players?.[playerKey]);
  const unseenHi  =
    countUnseen(state, SSANGPI_WITH_GUKJIN_ID_SET, playerKey) +
    countUnseen(state, BONUS_CARD_ID_SET, playerKey);

  const gb = deps.analyzeGukjinBranches?.(state, playerKey);
  let oppScoreRisk = safeNum(ctx.oppScore), oppPiRisk = oppPiBase;
  if (gb?.scenarios?.length) {
    for (const s of gb.scenarios) { oppScoreRisk = Math.max(oppScoreRisk, safeNum(s?.oppScore)); oppPiRisk = Math.max(oppPiRisk, safeNum(s?.oppPi)); }
    for (const s of gb.scenarios) {
      if (safeNum(s?.oppScore) >= 6) return false;
      if (safeNum(s?.oppScore) >= 5 && !certain) return false;
      if (unseenHi >= 2 && safeNum(s?.oppPi) >= 7 && !certain) return false;
    }
  }

  const diff = myScore - oppScoreRisk;
  if (unseenHi >= 2 && oppPiRisk >= 7 && !certain) return false;
  if (tr.oppOneAwayProb >= safeNum(P.goOppOneAwayGate, 100)) return false;
  if (!certain) {
    const conf = 0.5 + diff * safeNum(P.goScoreDiffBonus) + (late ? safeNum(P.goDeckLowBonus) : 0) - unseenHi * safeNum(P.goUnseeHighPiPenalty);
    if (conf < safeNum(P.goBaseThreshold)) return false;
  }

  const gH = Math.max(5, Math.min(7, Math.round(safeNum(P.goOppScoreGateHigh, 6))));
  const gL = Math.max(3, Math.min(5, Math.round(safeNum(P.goOppScoreGateLow,  4))));

  if (oppScoreRisk >= gH) return false;
  if (oppScoreRisk >= gH - 1) {
    if (_stopForOppFour(state, playerKey, deps, 1, 2) || sec) return false;
    const bigLead  = diff >= P.goBigLeadScoreDiff && myScore >= P.goBigLeadMinScore;
    const lowThreat = tr.oppOneAwayProb < (late ? P.goBigLeadOneAwayLate : P.goBigLeadOneAwayEarly) && tr.jokboThreat < P.goBigLeadJokboThresh && tr.nextThreat < P.goBigLeadNextThresh;
    return bigLead && lowThreat;
  }
  if (oppScoreRisk >= gL) {
    if (_stopForOppFour(state, playerKey, deps, 2, 3)) return false;
    return tr.oppOneAwayProb < (late ? P.goOneAwayThreshOpp4Late : P.goOneAwayThreshOpp4Early) - secG;
  }
  if (oppScoreRisk >= 1) {
    if (oppScoreRisk === 3 && tr.jokboThreat >= 0.5) return false;
    const base = oppScoreRisk === 3 ? P.goOneAwayThreshOpp3 : oppScoreRisk === 2 ? P.goOneAwayThreshOpp2 : P.goOneAwayThreshOpp1;
    if (tr.oppOneAwayProb >= base - (late ? 1 : 0) - secG) return false;
    if (oppScoreRisk <= 2 && (tr.jokboThreat >= P.goOpp12JokboThresh || tr.nextThreat >= P.goOpp12NextThresh || (late && tr.jokboThreat >= 0.4))) return false;
    if (sec && diff <= 0 && tr.oppOneAwayProb >= base - 5 - secG) return false;
    return true;
  }
  if (tr.oppOneAwayProb >= (late ? P.goOneAwayThreshOpp0Late : P.goOneAwayThreshOpp0) - secG) return false;
  if (tr.jokboThreat >= P.goOpp0JokboThresh || tr.nextThreat >= P.goOpp0NextThresh) return false;
  if (sec && diff < 0) return false;
  return true;
}

/* 6) Bomb month selection and bomb gate */
function shouldBombV5(state, playerKey, bombMonths, deps, params = DEFAULT_PARAMS) {
  const P = { ...DEFAULT_PARAMS, ...params };
  if (!bombMonths?.length) return false;
  const ctx  = deps.analyzeGameContext(state, playerKey);
  const plan = deps.getFirstTurnDoublePiPlan(state, playerKey);
  if (plan.active && bombMonths.some((m) => plan.months.has(m))) return true;
  const best = _bestBomb(state, bombMonths, deps);
  if (best == null) return false;
  const imp  = deps.isHighImpactBomb(state, playerKey, best);
  const gain = safeNum(deps.monthBoardGain(state, best));
  if (imp.highImpact) return true;
  if (ctx.defenseOpening) return false;
  if (ctx.volatilityComeback) return safeNum(imp.immediateGain) >= safeNum(P.bombImpactMinGain, 4) || gain >= 0;
  if (ctx.nagariDelayMode && safeNum(imp.immediateGain) < 6) return false;
  return gain >= 1.0;
}
function selectBombMonthV5(state, _pk, months, deps) { return _bestBomb(state, months, deps); }
function _bestBomb(state, months, deps) {
  return months?.length ? months.reduce((b, m) => safeNum(deps.monthBoardGain(state, m)) > safeNum(deps.monthBoardGain(state, b)) ? m : b, months[0]) : null;
}

/* 7) Shaking decision */
function decideShakingV5(state, playerKey, shakingMonths, deps, params = DEFAULT_PARAMS) {
  const P = { ...DEFAULT_PARAMS, ...params };
  if (!shakingMonths?.length) return { allow: false, month: null, score: -Infinity };
  const bbm = deps.boardMatchesByMonth(state);
  if ((state.players?.[playerKey]?.hand || []).some((c) => (bbm.get(c.month) || []).length > 0))
    return { allow: false, month: null, score: -Infinity };

  const ctx = deps.analyzeGameContext(state, playerKey);
  const my = safeNum(ctx.myScore), opp = safeNum(ctx.oppScore);
  if (opp >= 5 && opp >= my + 2) return { allow: false, month: null, score: -Infinity };

  const livePi = getLiveDoublePiMonths(state);
  const ch     = getComboHoldMonths(state, playerKey, deps);
  const plan   = deps.getFirstTurnDoublePiPlan(state, playerKey);
  let best = { allow: false, month: null, score: -Infinity, highImpact: false };

  for (const month of shakingMonths) {
    const ig  = safeNum(deps.shakingImmediateGainScore(state, playerKey, month));
    const cg  = safeNum(deps.ownComboOpportunityScore(state, playerKey, month));
    const imp = deps.isHighImpactShaking(state, playerKey, month);
    const kn  = safeNum(deps.countKnownMonthCards(state, month));
    let score = ig * safeNum(P.shakingImmediateGainMul, 1.35) + cg * safeNum(P.shakingComboGainMul, 1.15) + (kn <= 2 ? 0.25 : kn >= 4 ? -0.1 : 0);
    if (imp?.hasDoublePiLine)  score += 0.35;
    if (imp?.directThreeGwang) score += 0.3;
    if (imp?.highImpact)       score += 0.4;
    if (my < opp)              score += safeNum(P.shakingTempoBonusMul);
    if (livePi.has(month) && !ch.has(month)) score += 0.55;
    if (ch.has(month)) score -= 0.25;
    if (plan.active && plan.months.has(month)) score += 0.3;
    if (score > best.score) best = { allow: false, month, score, highImpact: !!imp?.highImpact };
  }

  const prefDpi = best.month != null && livePi.has(best.month) && !ch.has(best.month);
  const allow   = (my > opp || prefDpi) && best.score >= safeNum(P.shakingScoreThreshold, 0.65) + (my > opp ? safeNum(P.shakingAheadPenalty) : 0);
  return { ...best, allow };
}

/* 8) President/Gukjin late decisions */
function shouldPresidentStopV5(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const ctx  = deps.analyzeGameContext(state, playerKey);
  const diff = safeNum(ctx.myScore) - safeNum(ctx.oppScore);
  const co   = safeNum(state.carryOverMultiplier, 1);
  const sec  = isSecondMover(state, playerKey);
  if (diff >= 3 && co <= 1) return true;
  if (diff <= -1 || (sec && diff <= 0) || co >= 2) return false;
  return diff >= 1;
}

function chooseGukjinHeuristicV5(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const ctx = deps.analyzeGameContext(state, playerKey);
  const br  = deps.analyzeGukjinBranches(state, playerKey);
  const sf  = safeNum(ctx.selfFive), of_ = safeNum(ctx.oppFive);
  if (sf <= 0 && of_ >= 6) return "junk";
  if (sf >= 7 && of_ <= 0) return "five";
  if (br?.enabled && br.scenarios?.length) {
    const fSc = br.scenarios.find((s) => s?.selfMode === "five");
    const jSc = br.scenarios.find((s) => s?.selfMode === "junk");
    if (fSc && jSc) return (safeNum(fSc.myScore) - safeNum(fSc.oppScore)) >= (safeNum(jSc.myScore) - safeNum(jSc.oppScore)) ? "five" : "junk";
  }
  return sf >= 7 ? "five" : "junk";
}
