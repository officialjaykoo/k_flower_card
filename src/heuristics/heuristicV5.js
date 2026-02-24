// ============================================================
// heuristicV5.js  –  Matgo Heuristic V5
// ============================================================
// 설계 원칙:
//   1. 모든 수치 상수를 PARAMS 객체로 집약 → Optuna 외부 주입 가능
//   2. 결정마다 통합 expectedValue 기반 스코어링
//   3. 게임 페이즈를 덱 수만이 아닌 멀티팩터로 판단
//   4. V3/V4 대비 분기 단순화, 핵심 로직 명확화
// ============================================================

// ──────────────────────────────────────────────────────────────
// DEFAULT PARAMS  (Optuna로 튜닝할 모든 상수)
// ──────────────────────────────────────────────────────────────
export const DEFAULT_PARAMS = {
  // ── 카드 가치 가중치 ──
  kwangWeight: 8.0,        // 광 기본 가치
  fiveWeight: 5.0,         // 열 기본 가치
  ribbonWeight: 4.0,       // 띠 기본 가치
  piWeight: 1.0,           // 피 기본 가치
  doublePiBonus: 6.0,      // 쌍피 추가 가치

  // ── rankHandCards 스코어링 ──
  matchOneBase: 6.0,       // 매치 1장 기본점
  matchTwoBase: 10.0,      // 매치 2장 기본점
  matchThreeBase: 14.0,    // 매치 3장+ 기본점
  noMatchPenalty: 2.5,     // 무매치 패널티
  highValueMatchBonus: 5.0,// 광/열 매치 시 추가점
  feedRiskMul: 5.5,        // 무매치 상대 먹이 위험 배율
  feedRiskMatchMul: 1.3,   // 매치 있어도 상대 먹이 배율
  pukRiskHighMul: 5.0,     // 뻑 위험 (덱 ≤10) 배율
  pukRiskNormalMul: 3.5,   // 뻑 위험 (일반) 배율
  blockingBonus: 20.0,     // 블로킹 월 보너스
  jokboBlockBonus: 6.0,    // 족보 블로킹 보너스
  firstTurnPiPlanBonus: 5.5,// 첫턴 쌍피 플랜 보너스
  comboBaseBonus: 3.5,     // 콤보 완성 기여 보너스

  // ── shouldGo 임계값 ──
  goBaseThreshold: 0.52,   // GO 기본 확률 임계값 (기댓값 기반)
  goOppScoreGateLow: 3,    // 상대 점수 낮은 게이트
  goOppScoreGateHigh: 5,   // 상대 점수 높은 게이트 (즉시 STOP)
  goOppOneAwayGate: 35,    // oppOneAwayProb STOP 게이트 (%)
  goScoreDiffBonus: 0.06,  // 내 점수 우세당 임계값 완화
  goDeckLowBonus: 0.05,    // 덱 ≤8 시 추가 완화
  goUnseeHighPiPenalty: 0.08, // 미확인 고피 2장 이상 시 임계값 강화

  // ── shouldBomb 임계값 ──
  bombImpactMinGain: 1.0,  // 폭탄 유리한 최소 즉시 이득
  bombHighImpactOverride: true, // highImpact면 무조건 폭탄

  // ── decideShaking 임계값 ──
  shakingScoreThreshold: 0.60,
  shakingImmediateGainMul: 1.25,
  shakingComboGainMul: 1.10,
  shakingTempoBonusMul: 0.45,
  shakingRiskPenaltyMul: 1.0,
  shakingAheadPenalty: 0.20,  // 앞서있을 때 추가 임계 강화

  // ── chooseMatch 스코어링 ──
  matchPiGainMul: 4.0,
  matchKwangBonus: 9.0,
  matchRibbonBonus: 7.0,
  matchFiveBonus: 5.0,
  matchDoublePiBonus: 14.0,
  matchComboFinishMul: 1.5,
  matchBlockBonusMul: 1.0,
  matchMongBakFiveBonus: 42.0, // 몽박 방어 시 열 선택 보너스

  // ── 페이즈 분류 기준 ──
  earlyDeckMin: 20,        // 덱 ≥ 이 값이면 초반
  lateDeckMax: 10,         // 덱 ≤ 이 값이면 후반
  endgameDeckMax: 5,       // 덱 ≤ 이 값이면 엔드게임
};

// ──────────────────────────────────────────────────────────────
// HELPERS
// ──────────────────────────────────────────────────────────────

function safeNum(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

// 게임 페이즈 (멀티팩터)
function getPhase(state, myScore, oppScore, P) {
  const deck = safeNum(state.deck?.length);
  const scoreSum = myScore + oppScore;
  if (deck >= P.earlyDeckMin && scoreSum <= 4) return "early";
  if (deck <= P.endgameDeckMax) return "endgame";
  if (deck <= P.lateDeckMax) return "late";
  return "mid";
}

// 카드 피 가치 (deps 의존)
function piLikeValue(card, deps) {
  return safeNum(deps.junkPiValue?.(card), card?.category === "junk" ? 1 : 0);
}

function isDoublePiLike(card, deps) {
  return piLikeValue(card, deps) >= 2;
}

// 자신 콤보 완성 기여 보너스
function selfComboBonus(selfPlayer, card, P) {
  const captured = selfPlayer?.captured || {};
  const kwangCount = (captured.kwang || []).length;
  const ribbonCount = (captured.ribbon || []).length;
  const fiveCount = (captured.five || []).length;
  let bonus = 0;
  if (card.category === "kwang") {
    if (kwangCount >= 2) bonus += P.comboBaseBonus * (kwangCount - 1);
  }
  if (card.category === "ribbon") {
    if (ribbonCount >= 4) bonus += P.comboBaseBonus * 2;
    else if (ribbonCount >= 3) bonus += P.comboBaseBonus;
  }
  if (card.category === "five") {
    if (fiveCount >= 4) bonus += P.comboBaseBonus * 2;
    else if (fiveCount >= 3) bonus += P.comboBaseBonus;
  }
  return bonus;
}

// 상대 oneAway 확률 추정 (V4 로직 재사용, 단순화)
function estimateOppOneAwayProb(state, playerKey, deps) {
  // deps에 opponentThreatScore가 있으면 활용
  const threat = safeNum(deps.opponentThreatScore?.(state, playerKey));
  const jokbo = safeNum(deps.checkOpponentJokboProgress?.(state, playerKey)?.threat);
  const next = safeNum(deps.nextTurnThreatScore?.(state, playerKey));
  // 0~100 환산
  return clamp((threat * 40 + jokbo * 35 + next * 25), 0, 100);
}

// ──────────────────────────────────────────────────────────────
// 1. rankHandCardsV5
// ──────────────────────────────────────────────────────────────
export function rankHandCardsV5(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const P = { ...DEFAULT_PARAMS, ...params };
  const player = state.players?.[playerKey];
  if (!player?.hand?.length) return [];

  const opp = deps.otherPlayerKey(playerKey);
  const oppPlayer = state.players?.[opp];
  const ctx = deps.analyzeGameContext(state, playerKey);
  const phase = getPhase(state, safeNum(ctx.myScore), safeNum(ctx.oppScore), P);

  const deckCount = safeNum(state.deck?.length);
  const jokbo = deps.checkOpponentJokboProgress(state, playerKey);
  const blockMonths = deps.blockingMonthsAgainst(oppPlayer, player);
  const blockUrgency = deps.blockingUrgencyByMonth(oppPlayer, player);
  const selfPi = safeNum(ctx.selfPi, deps.capturedCountByCategory(player, "junk"));
  const oppPi = safeNum(ctx.oppPi, deps.capturedCountByCategory(oppPlayer, "junk"));
  const nextThreat = safeNum(deps.nextTurnThreatScore(state, playerKey));
  const firstTurnPiPlan = deps.getFirstTurnDoublePiPlan(state, playerKey);
  const boardByMonth = deps.boardMatchesByMonth(state);
  const boardCountByMonth = deps.monthCounts(state.board || []);
  const handCountByMonth = deps.monthCounts(player.hand || []);
  const capturedByMonth = deps.capturedMonthCounts(state);
  const lateOrEnd = phase === "late" || phase === "endgame";

  const ranked = player.hand.map((card) => {
    const matches = boardByMonth.get(card.month) || [];
    const captureGain = matches.reduce((sum, c) => sum + safeNum(deps.cardCaptureValue(c)), 0);
    const selfValue = safeNum(deps.cardCaptureValue(card));

    // ── 기본 매치 스코어 ──
    let score = 0;
    if (matches.length === 0) {
      score = -(P.noMatchPenalty + selfValue * 0.5);
    } else if (matches.length === 1) {
      score = P.matchOneBase + captureGain - selfValue * 0.2;
    } else if (matches.length === 2) {
      score = P.matchTwoBase + captureGain - selfValue * 0.1;
    } else {
      score = P.matchThreeBase + captureGain;
    }

    // 광/열 매치 보너스
    if (matches.some((m) => m.category === "kwang" || m.category === "five")) {
      score += P.highValueMatchBonus;
    }

    // 자기 카드가 고가치면 매치 없어도 보유 고려
    if (matches.length === 0 && (card.category === "kwang" || card.category === "five")) {
      score += selfValue * 0.5; // 패에 있는 것 자체로 부분 가치
    }

    // ── 상대 먹이 위험 ──
    const feedRisk = safeNum(deps.estimateOpponentImmediateGainIfDiscard(state, playerKey, card.month));
    score -= feedRisk * (matches.length === 0 ? P.feedRiskMul : P.feedRiskMatchMul);

    // ── 뻑 위험 ──
    const pukRisk = safeNum(deps.isRiskOfPuk(state, playerKey, card, boardCountByMonth, handCountByMonth));
    if (pukRisk > 0) {
      score -= pukRisk * (deckCount <= 10 ? P.pukRiskHighMul : P.pukRiskNormalMul);
    } else if (pukRisk < 0) {
      score += -pukRisk * 1.3;
    }

    // ── 블로킹 ──
    if (blockMonths.has(card.month)) {
      const urgency = safeNum(blockUrgency.get(card.month), 2);
      const jokboU = safeNum(jokbo.monthUrgency?.get(card.month), 0);
      const blockScore = Math.max(urgency >= 3 ? P.blockingBonus + 4 : P.blockingBonus, jokboU);
      score += blockScore + nextThreat * P.jokboBlockBonus + jokbo.threat * P.jokboBlockBonus;
    }

    // ── 첫턴 쌍피 플랜 ──
    if (firstTurnPiPlan.active && firstTurnPiPlan.months.has(card.month)) {
      score += P.firstTurnPiPlanBonus;
    }

    // ── 피 전략: 내가 앞서있을 때 늦은 게임에서 쌍피 홀드 ──
    if (lateOrEnd && isDoublePiLike(card, deps)) {
      if (selfPi >= 7 && safeNum(ctx.myScore) >= safeNum(ctx.oppScore)) {
        score += 3.5; // 유리할 때 쌍피 보유 가중
      }
    }

    // ── 몽박 방어: 상대가 열 7+ 이고 내가 열 0 ──
    const selfFive = safeNum(ctx.selfFive);
    const oppFive = safeNum(ctx.oppFive);
    if (selfFive === 0 && oppFive >= 7 && card.category === "five" && matches.length > 0) {
      score += 15.0;
    }

    // ── 후반 피 긁기: 상대 피가 낮을 때 쌍피/보너스피 우선 ──
    if (lateOrEnd && oppPi <= 5 && isDoublePiLike(card, deps)) {
      score += 2.5;
    }

    // ── 전략 우선순위 타이브레이커 ──
    const tieBoost = safeNum(deps.monthStrategicPriority?.(card.month)) *
      (matches.length === 0 ? 0.9 : 0.5);
    score += tieBoost;

    return { card, score, matches: matches.length, tieBoost };
  });

  ranked.sort((a, b) => b.score - a.score);

  // 상위 2개 점수 차이 좁을 때 타이브레이커 정밀화
  if (ranked.length >= 2 && Math.abs(ranked[0].score - ranked[1].score) <= 1.0) {
    for (const r of ranked) r.score += safeNum(r.tieBoost) * 1.0;
    ranked.sort((a, b) => b.score - a.score);
  }

  return ranked;
}

// ──────────────────────────────────────────────────────────────
// 2. chooseMatchHeuristicV5
// ──────────────────────────────────────────────────────────────
export function chooseMatchHeuristicV5(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const P = { ...DEFAULT_PARAMS, ...params };
  const ids = state.pendingMatch?.boardCardIds || [];
  if (!ids.length) return null;

  const opp = deps.otherPlayerKey(playerKey);
  const selfPlayer = state.players?.[playerKey];
  const oppPlayer = state.players?.[opp];
  const ctx = deps.analyzeGameContext(state, playerKey);
  const jokbo = deps.checkOpponentJokboProgress(state, playerKey);
  const blockMonths = deps.blockingMonthsAgainst(oppPlayer, selfPlayer);
  const blockUrgency = deps.blockingUrgencyByMonth(oppPlayer, selfPlayer);
  const nextThreat = safeNum(deps.nextTurnThreatScore(state, playerKey));
  const selfPi = safeNum(ctx.selfPi, deps.capturedCountByCategory(selfPlayer, "junk"));
  const oppPi = safeNum(ctx.oppPi, deps.capturedCountByCategory(oppPlayer, "junk"));
  const selfFive = safeNum(ctx.selfFive);
  const oppFive = safeNum(ctx.oppFive);
  const mongBakDefense = selfFive === 0 && oppFive >= 7;

  const candidates = (state.board || []).filter((c) => ids.includes(c.id));
  if (!candidates.length) return null;

  let best = candidates[0];
  let bestScore = -Infinity;

  for (const c of candidates) {
    const piGain = piLikeValue(c, deps);
    let score = safeNum(deps.cardCaptureValue(c)) * 0.8;

    score += piGain * P.matchPiGainMul;
    if (c.category === "kwang") score += P.matchKwangBonus;
    else if (c.category === "ribbon") score += P.matchRibbonBonus;
    else if (c.category === "five") score += P.matchFiveBonus;

    // 쌍피 보너스
    if (isDoublePiLike(c, deps)) score += P.matchDoublePiBonus;

    // 피 구간 보너스
    if (selfPi >= 7 && selfPi <= 9) score += piGain * 1.8;
    if (oppPi <= 5) score += piGain * 1.4;

    // 콤보 완성
    score += selfComboBonus(selfPlayer, c, P) * P.matchComboFinishMul;

    // 블로킹
    if (blockMonths.has(c.month)) {
      const urgency = safeNum(blockUrgency.get(c.month), 2);
      const jokboU = safeNum(jokbo.monthUrgency?.get(c.month), 0);
      score += Math.max(urgency >= 3 ? 24 : 20, jokboU) * P.matchBlockBonusMul;
      score += nextThreat * 4.5;
    }

    // 몽박 방어
    if (mongBakDefense && c.category === "five") score += P.matchMongBakFiveBonus;
    else if (mongBakDefense && piGain > 0) score -= 8;

    score += safeNum(deps.monthStrategicPriority?.(c.month)) * 0.25;

    if (score > bestScore) {
      bestScore = score;
      best = c;
    }
  }

  return best?.id ?? null;
}

// ──────────────────────────────────────────────────────────────
// 3. shouldGoV5  (기댓값 기반)
// ──────────────────────────────────────────────────────────────
export function shouldGoV5(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const P = { ...DEFAULT_PARAMS, ...params };

  // 상대 파산 가능 시 STOP
  if (deps.canBankruptOpponentByStop?.(state, playerKey)) return false;

  const ctx = deps.analyzeGameContext(state, playerKey);
  const myScore = safeNum(ctx.myScore);
  const oppScore = safeNum(ctx.oppScore);
  const deckCount = safeNum(state.deck?.length);
  const lateGame = deckCount <= P.lateDeckMax;
  const endGame = deckCount <= P.endgameDeckMax;

  const opp = deps.otherPlayerKey(playerKey);
  const oppPlayer = state.players?.[opp];
  const oppPiBase = safeNum(ctx.oppPi, deps.capturedCountByCategory(oppPlayer, "junk"));

  // Gukjin 브랜치 위험
  const gukjinBranch = deps.analyzeGukjinBranches?.(state, playerKey);
  const worstOppScore = _worstCaseOppScore(gukjinBranch, oppScore);
  const worstOppPi = _worstCaseOppPi(gukjinBranch, oppPiBase);

  // 즉시 STOP 게이트
  if (worstOppScore >= P.goOppScoreGateHigh) return false;

  // 미확인 고피 위험
  const BONUS_IDS = new Set([43, 44]); // 보너스 피 카드 ID (게임마다 다를 수 있음 - 의존성 사용)
  const SSANGPI_IDS = deps.ssangpiCardIds ? new Set(deps.ssangpiCardIds()) : BONUS_IDS;
  const unseenHighPi = _countUnseen(state, playerKey, SSANGPI_IDS);
  const selfComboSure = _hasCertainJokbo(state.players?.[playerKey]);
  if (unseenHighPi >= 2 && worstOppPi >= 7 && !selfComboSure) return false;

  // 기댓값 기반 GO 판단
  // 내 기대 이득 = 현 점수차 * carryOver
  // 상대 위험 = oppOneAway 확률
  const oppOneAway = estimateOppOneAwayProb(state, playerKey, deps);
  const scoreDiff = myScore - worstOppScore;

  // 동적 임계값
  let threshold = P.goBaseThreshold;
  threshold -= scoreDiff * P.goScoreDiffBonus;       // 앞서있으면 완화
  if (lateGame) threshold -= P.goDeckLowBonus;       // 후반 완화
  if (unseenHighPi >= 2) threshold += P.goUnseeHighPiPenalty; // 미확인 고피 경고
  if (worstOppScore === P.goOppScoreGateLow) threshold += 0.08; // 상대 점수 3 경고
  if (endGame && scoreDiff <= 0) threshold += 0.10;  // 엔드게임 동점/뒤짐 경고

  // oppOneAway 직접 게이트
  if (oppOneAway >= P.goOppOneAwayGate) return false;

  // carryOver 멀티플라이어 있으면 GO 선호
  const carryOver = safeNum(state.carryOverMultiplier, 1);
  if (carryOver >= 2) threshold -= 0.08;

  // 기댓값 추정: 내가 GO해서 이기는 확률이 threshold 이상이면 GO
  // 단순 모델: (1 - oppOneAway/100) × 승리확률
  const winProb = clamp(1 - oppOneAway / 100, 0, 1);
  return winProb >= threshold;
}

function _worstCaseOppScore(gukjinBranch, base) {
  if (!gukjinBranch?.scenarios?.length) return base;
  return Math.max(base, ...gukjinBranch.scenarios.map((s) => safeNum(s?.oppScore, 0)));
}

function _worstCaseOppPi(gukjinBranch, base) {
  if (!gukjinBranch?.scenarios?.length) return base;
  return Math.max(base, ...gukjinBranch.scenarios.map((s) => safeNum(s?.oppPi, 0)));
}

function _countUnseen(state, playerKey, idSet) {
  let seen = 0;
  for (const key of ["human", "ai"]) {
    const player = state.players?.[key];
    for (const cat of ["kwang", "five", "ribbon", "junk"]) {
      for (const c of player?.captured?.[cat] || []) {
        if (idSet.has(c?.id)) seen++;
      }
    }
    for (const c of player?.hand || []) {
      if (idSet.has(c?.id)) seen++;
    }
  }
  for (const c of state.board || []) {
    if (idSet.has(c?.id)) seen++;
  }
  return idSet.size - seen;
}

function _hasCertainJokbo(player) {
  const cap = player?.captured || {};
  const kwang = (cap.kwang || []).length;
  const birds = (cap.five || []).filter((c) => c?.tags?.includes?.("fiveBirds")).length;
  const red = (cap.ribbon || []).filter((c) => c?.tags?.includes?.("redRibbons")).length;
  const blue = (cap.ribbon || []).filter((c) => c?.tags?.includes?.("blueRibbons")).length;
  const plain = (cap.ribbon || []).filter((c) => c?.tags?.includes?.("plainRibbons")).length;
  return kwang >= 3 || birds >= 3 || red >= 3 || blue >= 3 || plain >= 3;
}

// ──────────────────────────────────────────────────────────────
// 4. shouldBombV5
// ──────────────────────────────────────────────────────────────
export function shouldBombV5(state, playerKey, bombMonths, deps, params = DEFAULT_PARAMS) {
  const P = { ...DEFAULT_PARAMS, ...params };
  if (!bombMonths?.length) return false;

  const ctx = deps.analyzeGameContext(state, playerKey);
  const firstTurnPiPlan = deps.getFirstTurnDoublePiPlan(state, playerKey);

  // 첫턴 쌍피 플랜과 겹치면 항상 폭탄
  if (firstTurnPiPlan.active) {
    if ((bombMonths).some((m) => firstTurnPiPlan.months.has(m))) return true;
  }

  const bestMonth = _bestBombMonth(state, bombMonths, deps);
  if (bestMonth == null) return false;

  const impact = deps.isHighImpactBomb(state, playerKey, bestMonth);
  const gain = safeNum(deps.monthBoardGain(state, bestMonth));

  // highImpact 무조건 폭탄
  if (P.bombHighImpactOverride && impact.highImpact) return true;

  // 방어 오프닝: highImpact만 허용
  if (ctx.defenseOpening) return impact.highImpact;

  // 컴백 모드: 관대하게
  if (ctx.volatilityComeback) {
    if (impact.highImpact || safeNum(impact.immediateGain) >= 4) return true;
    return gain >= 0;
  }

  // 나가리 지연 모드: 제한적
  if (ctx.nagariDelayMode && !impact.highImpact && safeNum(impact.immediateGain) < 6) return false;

  return gain >= P.bombImpactMinGain;
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
// ──────────────────────────────────────────────────────────────
export function decideShakingV5(state, playerKey, shakingMonths, deps, params = DEFAULT_PARAMS) {
  const P = { ...DEFAULT_PARAMS, ...params };
  if (!shakingMonths?.length) return { allow: false, month: null, score: -Infinity };

  const ctx = deps.analyzeGameContext(state, playerKey);
  const opp = deps.otherPlayerKey(playerKey);
  const oppPlayer = state.players?.[opp];
  const firstTurnPiPlan = deps.getFirstTurnDoublePiPlan(state, playerKey);
  const oppThreat = safeNum(deps.opponentThreatScore(state, playerKey));
  const jokboThreat = safeNum(deps.checkOpponentJokboProgress(state, playerKey)?.threat);
  const nextThreat = safeNum(deps.nextTurnThreatScore(state, playerKey));
  const myScore = safeNum(ctx.myScore);
  const oppScore = safeNum(ctx.oppScore);
  const aheadOrEven = myScore >= oppScore;
  const trailingBy = Math.max(0, oppScore - myScore);
  const deckCount = safeNum(state.deck?.length);
  const selfPi = safeNum(ctx.selfPi, deps.capturedCountByCategory(state.players?.[playerKey], "junk"));
  const oppPi = safeNum(ctx.oppPi, deps.capturedCountByCategory(oppPlayer, "junk"));

  // 템포 압박
  let tempo = Math.min(2.0, trailingBy * 0.5);
  if (ctx.mode === "DESPERATE_DEFENSE") tempo += 1.0;
  if (safeNum(state.carryOverMultiplier, 1) >= 2) tempo += 0.45;
  if ((oppPlayer?.events?.shaking || 0) + (oppPlayer?.events?.bomb || 0) > 0) tempo += 0.7;
  if ((oppPlayer?.goCount || 0) > 0) tempo += 0.5;
  if ((state.players?.[playerKey]?.goCount || 0) > 0 && trailingBy === 0) tempo -= 0.35;

  // 리스크 패널티
  const riskPenalty = oppThreat * 2.0 + jokboThreat * 1.3 + nextThreat * 1.2;

  let best = { allow: false, month: null, score: -Infinity, highImpact: false };

  for (const month of shakingMonths) {
    let immGain = safeNum(deps.shakingImmediateGainScore(state, playerKey, month));
    if (firstTurnPiPlan.active && firstTurnPiPlan.months.has(month)) immGain += 0.45;
    const comboGain = safeNum(deps.ownComboOpportunityScore(state, playerKey, month));
    const impact = deps.isHighImpactShaking(state, playerKey, month);
    if (impact.directThreeGwang) immGain += 0.55;
    if (impact.hasDoublePiLine) immGain += 0.35;

    // 느리게 가는 패널티 (이득 없고 압박도 없는데 흔들기)
    let slowPenalty = 0;
    if (immGain < 0.75 && comboGain < 0.9 && tempo < 1.2 && aheadOrEven) slowPenalty = 1.15;

    let score = immGain * P.shakingImmediateGainMul
              + comboGain * P.shakingComboGainMul
              + tempo * P.shakingTempoBonusMul
              - riskPenalty * P.shakingRiskPenaltyMul
              - slowPenalty;

    if (trailingBy >= 3) score += 0.35;
    if (ctx.volatilityComeback) {
      score += 0.45;
      if (impact.highImpact) score += 0.22;
    }

    if (score > best.score) {
      best = { allow: false, month, score, highImpact: impact.highImpact, immGain, comboGain };
    }
  }

  // 임계값 결정
  let threshold = P.shakingScoreThreshold;
  if (ctx.mode === "DESPERATE_DEFENSE") threshold -= 0.25;
  if (trailingBy >= 3) threshold -= 0.15;
  if (myScore >= oppScore + 2) threshold += P.shakingAheadPenalty + 0.2;
  if (oppThreat >= 0.7 && aheadOrEven) threshold += 0.35;
  if (deckCount <= 7 && aheadOrEven) threshold += 0.25;
  if (safeNum(state.carryOverMultiplier, 1) >= 2 && aheadOrEven) threshold += 0.2;
  if (selfPi >= 9 && aheadOrEven) threshold += 0.35;
  if (ctx.volatilityComeback) threshold -= 0.3;

  if (best.month != null && firstTurnPiPlan.active && firstTurnPiPlan.months.has(best.month)) {
    threshold -= 0.15;
  }

  // 피 완성 윈도우에서 유리하면 흔들기 보류
  if (selfPi >= 9 && aheadOrEven && best.score < threshold + 0.55) {
    return { ...best, allow: false };
  }

  // 방어 오프닝: highImpact + 충분한 이득만 허용
  if (ctx.defenseOpening && (!best.highImpact || (best.immGain < 1.05 && best.comboGain < 1.35))) {
    return { ...best, allow: false };
  }
  if (ctx.nagariDelayMode && !best.highImpact && best.score < threshold + 0.35) {
    return { ...best, allow: false };
  }

  if (ctx.volatilityComeback && best.month != null) {
    const comebackThreshold = threshold - (best.highImpact ? 0.18 : 0.08);
    return { ...best, allow: best.score >= comebackThreshold };
  }

  return { ...best, allow: best.score >= threshold };
}

// ──────────────────────────────────────────────────────────────
// 6. shouldPresidentStopV5
// ──────────────────────────────────────────────────────────────
export function shouldPresidentStopV5(state, playerKey, deps, params = DEFAULT_PARAMS) {
  // 기본적으로 V4 로직을 유지하되 파라미터화 여지 확보
  const ctx = deps.analyzeGameContext(state, playerKey);
  const myScore = safeNum(ctx.myScore);
  const oppScore = safeNum(ctx.oppScore);
  const scoreDiff = myScore - oppScore;
  const carryOver = safeNum(state.carryOverMultiplier, 1);

  // 크게 앞서있고 carryOver 없으면 STOP (점수 굳히기)
  if (scoreDiff >= 3 && carryOver <= 1) return true;
  // 뒤지고 있으면 HOLD (역전 기회)
  if (scoreDiff <= -1) return false;
  // 동점/소폭 앞서면 carryOver 고려
  if (carryOver >= 2) return false; // 판돈 높으면 계속
  return scoreDiff >= 1; // 소폭 앞서면 STOP
}

// ──────────────────────────────────────────────────────────────
// 7. chooseGukjinHeuristicV5
// ──────────────────────────────────────────────────────────────
export function chooseGukjinHeuristicV5(state, playerKey, deps, params = DEFAULT_PARAMS) {
  const ctx = deps.analyzeGameContext(state, playerKey);
  const branch = deps.analyzeGukjinBranches(state, playerKey);
  const myScore = safeNum(ctx.myScore);
  const oppScore = safeNum(ctx.oppScore);
  const selfFive = safeNum(ctx.selfFive);
  const oppFive = safeNum(ctx.oppFive);

  // 몽박 위험: 상대가 몽박 가능하면 five 선택해 회피
  const mongBakRisk = selfFive <= 0 && oppFive >= 6;
  if (mongBakRisk) return "junk"; // five 선택해서 oppFive 위협 상쇄 불가 → 피로

  // 몽박 찬스: 내가 몽박 가능하면 five
  const mongBakChance = selfFive >= 7 && oppFive <= 0;
  if (mongBakChance) return "five";

  // 브랜치 분석: five가 더 유리한 시나리오 우선
  if (branch?.enabled && branch.scenarios?.length) {
    const fiveScenario = branch.scenarios.find((s) => s?.selfMode === "five");
    const junkScenario = branch.scenarios.find((s) => s?.selfMode === "junk");
    if (fiveScenario && junkScenario) {
      const fiveScore = safeNum(fiveScenario.myScore) - safeNum(fiveScenario.oppScore);
      const junkScore = safeNum(junkScenario.myScore) - safeNum(junkScenario.oppScore);
      return fiveScore >= junkScore ? "five" : "junk";
    }
  }

  // 기본: 열 개수로 판단
  return selfFive >= 7 ? "five" : "junk";
}
