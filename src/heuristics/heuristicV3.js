export {
  rankHandCardsV3,
  chooseMatchHeuristicV3,
  chooseGukjinHeuristicV3,
  shouldPresidentStopV3,
  shouldGoV3,
  selectBombMonthV3,
  shouldBombV3,
  decideShakingV3
};

function rankHandCardsV3(state, playerKey, deps) {
  const player = state.players?.[playerKey];
  if (!player?.hand?.length) return [];

  const {
    analyzeGameContext,
    checkOpponentJokboProgress,
    blockingMonthsAgainst,
    blockingUrgencyByMonth,
    capturedCountByCategory,
    nextTurnThreatScore,
    getFirstTurnDoublePiPlan,
    monthCounts,
    capturedMonthCounts,
    buildDynamicWeights,
    boardMatchesByMonth,
    cardCaptureValue,
    estimateOpponentImmediateGainIfDiscard,
    junkPiValue,
    isRiskOfPuk,
    estimateDangerMonthRisk,
    estimateReleasePunishProb,
    monthStrategicPriority
  } = deps;

  const opp = otherPlayerKeyFromDeps(playerKey, deps);
  const oppPlayer = state.players?.[opp];
  const ctx = analyzeGameContext(state, playerKey);
  const jokboThreat = checkOpponentJokboProgress(state, playerKey);
  const blockMonths = blockingMonthsAgainst(oppPlayer, player);
  const blockUrgency = blockingUrgencyByMonth(oppPlayer, player);
  const oppPi = Number(ctx.oppPi ?? capturedCountByCategory(oppPlayer, "junk"));
  const nextThreat = nextTurnThreatScore(state, playerKey);
  const selfPi = Number(ctx.selfPi ?? capturedCountByCategory(player, "junk"));
  const deckCount = state.deck?.length || 0;
  const mongDanger = ctx.mongDanger || 0;
  const selfFive = ctx.selfFive || 0;
  const oppFive = ctx.oppFive || 0;
  const urgentMongDefense = selfFive === 0 && oppFive >= 6;
  const firstTurnPiPlan = getFirstTurnDoublePiPlan(state, playerKey);
  const boardCountByMonth = monthCounts(state.board || []);
  const handCountByMonth = monthCounts(player.hand || []);
  const capturedByMonth = capturedMonthCounts(state);
  const dyn = buildDynamicWeights(state, playerKey, ctx);
  const defenseOpening = !!ctx.defenseOpening;
  const nagariDelayMode = !!ctx.nagariDelayMode;
  const endgameSafePitch = !!ctx.endgameSafePitch;
  const midgameBlockFocus = !!ctx.midgameBlockFocus;
  const byMonth = boardMatchesByMonth(state);
  const ranked = player.hand.map((card) => {
    const matches = byMonth.get(card.month) || [];
    const captureGain = matches.reduce((sum, c) => sum + cardCaptureValue(c), 0);
    const selfValue = cardCaptureValue(card);
    let score = 0;
    if (matches.length === 0) score = -2 - selfValue;
    else if (matches.length === 1) score = 6 + captureGain - selfValue * 0.2;
    else if (matches.length === 2) score = 9 + captureGain - selfValue * 0.1;
    else score = 12 + captureGain;
    if (matches.some((m) => m.category === "kwang" || m.category === "five")) score += 5;
    if (score > 0) score *= dyn.combo;

    const knownMonth =
      (boardCountByMonth.get(card.month) || 0) +
      (handCountByMonth.get(card.month) || 0) +
      (capturedByMonth.get(card.month) || 0);
    const oppFeedRisk = estimateOpponentImmediateGainIfDiscard(state, playerKey, card.month);

    const fiveCaptureGain = matches.reduce((n, m) => n + (m.category === "five" ? 1 : 0), 0);
    if (card.category === "five" && matches.length > 0) {
      score += (2.6 + fiveCaptureGain * 2.2) * (1 + mongDanger * 0.9);
    }
    if (fiveCaptureGain > 0) {
      score += (1.6 + fiveCaptureGain * 1.4) * (1 + mongDanger * 0.8);
    }
    if (card.category === "five" && matches.length === 0) {
      score -= (0.8 + mongDanger * 2.8) * dyn.hold;
    }
    if (urgentMongDefense && matches.length === 0 && card.category !== "five") {
      score -= 0.45;
    }

    if (defenseOpening) {
      const piGain = matches.reduce((n, m) => n + junkPiValue(m), 0);
      if (card.category === "junk") score += 1.6 * dyn.pi;
      score += piGain * (1.9 * dyn.pi);
      if (matches.length === 0 && card.category !== "junk") score -= 1.15 * dyn.risk;
      if (oppFeedRisk >= 1.35 && matches.length === 0) score -= 1.2 * dyn.safety;
    }

    if (midgameBlockFocus && blockMonths.has(card.month)) {
      if (matches.length > 0) score += 18.0 * dyn.block;
      else score -= 18.0 * dyn.hold;
    }

    if (nagariDelayMode) {
      if (blockMonths.has(card.month) && matches.length > 0) score += 10.0 * dyn.block;
      if (!blockMonths.has(card.month) && matches.length === 0) score -= 3.6 * dyn.safety;
      if (oppFeedRisk > 0.9 && matches.length === 0) score -= 1.5 * dyn.safety;
    }

    if (firstTurnPiPlan.active && firstTurnPiPlan.months.has(card.month)) {
      score += 8.0;
    }

    if (selfPi >= 7 && selfPi <= 8) {
      const piGain = matches.reduce((n, m) => n + junkPiValue(m), 0);
      if (card.category === "junk") score += 3.0 * ctx.piWeight * dyn.pi;
      score += piGain * (3.0 * ctx.piWeight * dyn.pi);
    }
    if (oppPi <= 5) {
      const piGain = matches.reduce((n, m) => n + junkPiValue(m), 0);
      if (card.category === "junk") score += 1.5 * ctx.piWeight * dyn.pi;
      score += piGain * (2.8 * ctx.piWeight * dyn.pi);
      const ownPiValue = junkPiValue(card);
      if (ownPiValue === 2) score += 1.2;
      if (ownPiValue >= 3) score += 1.8;
      if (matches.length === 0 && card.category === "junk") score -= 2.0;
    }
    if (selfPi >= 9) {
      const piGain = matches.reduce((n, m) => n + junkPiValue(m), 0);
      score += piGain * 4.5 * dyn.pi;
      if (card.category === "junk") score += 2.0 * dyn.pi;
    }

    const pukRisk = isRiskOfPuk(state, playerKey, card, boardCountByMonth, handCountByMonth);
    if (pukRisk > 0) score -= pukRisk * (5.2 * ctx.pukPenalty * dyn.risk);
    else if (pukRisk < 0) score += (-pukRisk) * 2.0;

    const dangerRisk = estimateDangerMonthRisk(
      state,
      playerKey,
      card.month,
      boardCountByMonth,
      handCountByMonth,
      capturedByMonth
    );
    if (matches.length === 0) score -= dangerRisk * (3.3 * dyn.safety);
    else if (matches.length === 1 && deckCount <= 8) score -= dangerRisk * (0.75 * dyn.safety);

    if (endgameSafePitch && matches.length === 0) {
      if (knownMonth >= 3) score += 3.1 * dyn.safety;
      else if (knownMonth === 2) score += 0.9 * dyn.safety;
      else score -= 1.0 * dyn.safety;
      score -= oppFeedRisk * (1.55 * dyn.safety);
    }

    if (blockMonths.has(card.month)) {
      const urgency = blockUrgency.get(card.month) || 2;
      const dynamicUrg = Math.max(urgency >= 3 ? 24.0 : 20.0, jokboThreat.monthUrgency.get(card.month) || 0);
      const blockBoost = dynamicUrg * ctx.blockWeight * dyn.block;
      if (matches.length > 0) score += blockBoost + nextThreat * 4.0 + jokboThreat.threat * 4.0;
      else score -= 10.0;
    }

    const releasePunishProb = estimateReleasePunishProb(state, playerKey, card.month, jokboThreat, ctx);
    const hardHold = matches.length === 0 && releasePunishProb >= 0.8;
    if (hardHold) {
      const hardHoldPenalty =
        releasePunishProb >= 0.92 ? 120 : releasePunishProb >= 0.86 ? 95 : 80;
      score -= hardHoldPenalty * dyn.hold;
    } else if (matches.length === 0 && releasePunishProb >= 0.65) {
      const softHoldPenalty = 50 + Math.max(0, releasePunishProb - 0.65) * 70;
      score -= softHoldPenalty * dyn.hold;
    }

    if (ctx.oppGoCount > 0 && matches.length === 0) score -= 2.2;

    const capCntMonth = capturedByMonth.get(card.month) || 0;
    const hasTacticalNeed =
      blockMonths.has(card.month) ||
      (nextThreat > 0.45 && matches.length > 0) ||
      (selfPi >= 9 && matches.some((m) => m.category === "junk")) ||
      (oppPi <= 5 && matches.some((m) => m.category === "junk")) ||
      matches.some((m) => m.category === "kwang" || m.category === "five");
    if (capCntMonth >= 1 && !hasTacticalNeed) {
      score -= capCntMonth >= 2 ? 2.5 : 1.0;
    }

    const lowValueNoMatch = matches.length === 0 && score <= 1.5;
    if (lowValueNoMatch) {
      const dupInHand = (handCountByMonth.get(card.month) || 0) >= 2;
      const capCnt = capturedByMonth.get(card.month) || 0;
      const blockedMonth = blockMonths.has(card.month);
      const pukRiskLocal = isRiskOfPuk(state, playerKey, card, boardCountByMonth, handCountByMonth);
      if (dupInHand && !blockedMonth && pukRiskLocal <= 0.75) {
        score += 2.0;
        if (capCnt >= 2) score += 4.0;
      }
    }
    return {
      card,
      score,
      matches: matches.length,
      uncertainBoost: monthStrategicPriority(card.month),
      hardHold,
      releasePunishProb
    };
  });
  ranked.sort((a, b) => b.score - a.score);

  if (ranked.length >= 2) {
    const top = ranked[0].score;
    const second = ranked[1].score;
    const ambiguous = Math.abs(top - second) <= 1.0;
    if (ambiguous) {
      for (const r of ranked) {
        if (top - r.score <= 3.0) {
          r.score += (r.uncertainBoost || 0) * 1.2;
        }
      }
      ranked.sort((a, b) => b.score - a.score);
    }
  }
  return ranked;
}

// Shared fallback: keep V3 tolerant even if deps.otherPlayerKey is missing.
function otherPlayerKeyFromDeps(playerKey, deps) {
  if (typeof deps?.otherPlayerKey === "function") return deps.otherPlayerKey(playerKey);
  return playerKey === "human" ? "ai" : "human";
}

function chooseMatchHeuristicV3(state, playerKey, deps) {
  const ids = state.pendingMatch?.boardCardIds || [];
  if (!ids.length) return null;

  const opp = otherPlayerKeyFromDeps(playerKey, deps);
  const blockMonths = deps.blockingMonthsAgainst(state.players?.[opp], state.players?.[playerKey]);
  const blockUrgency = deps.blockingUrgencyByMonth(state.players?.[opp], state.players?.[playerKey]);
  const jokboThreat = deps.checkOpponentJokboProgress(state, playerKey);
  const ctx = deps.analyzeGameContext(state, playerKey);
  const mongDanger = ctx.mongDanger || 0;
  const midgameBlockFocus = !!ctx.midgameBlockFocus;
  const defenseOpening = !!ctx.defenseOpening;
  const oppPi = Number(ctx.oppPi ?? deps.capturedCountByCategory(state.players?.[opp], "junk"));
  const selfPi = Number(ctx.selfPi ?? deps.capturedCountByCategory(state.players?.[playerKey], "junk"));
  const nextThreat = deps.nextTurnThreatScore(state, playerKey);
  const candidates = (state.board || []).filter((c) => ids.includes(c.id));
  if (!candidates.length) return null;

  candidates.sort((a, b) => {
    let as = deps.cardCaptureValue(a);
    let bs = deps.cardCaptureValue(b);
    if (blockMonths.has(a.month)) {
      as +=
        Math.max((blockUrgency.get(a.month) || 2) >= 3 ? 24 : 20, jokboThreat.monthUrgency.get(a.month) || 0) *
        ctx.blockWeight;
      as += nextThreat * 4.0 + jokboThreat.threat * 4.0;
    }
    if (blockMonths.has(b.month)) {
      bs +=
        Math.max((blockUrgency.get(b.month) || 2) >= 3 ? 24 : 20, jokboThreat.monthUrgency.get(b.month) || 0) *
        ctx.blockWeight;
      bs += nextThreat * 4.0 + jokboThreat.threat * 4.0;
    }
    if (oppPi <= 5 && a.category === "junk") as += 3;
    if (oppPi <= 5 && b.category === "junk") bs += 3;
    if (selfPi >= 7 && selfPi <= 8 && a.category === "junk") as += 4;
    if (selfPi >= 7 && selfPi <= 8 && b.category === "junk") bs += 4;
    if (a.category === "five") as += 2.2 + mongDanger * 5.0;
    if (b.category === "five") bs += 2.2 + mongDanger * 5.0;
    if (midgameBlockFocus && blockMonths.has(a.month)) as += 12.0;
    if (midgameBlockFocus && blockMonths.has(b.month)) bs += 12.0;
    if (defenseOpening && a.category === "junk") as += 2.1;
    if (defenseOpening && b.category === "junk") bs += 2.1;
    return bs - as;
  });

  return candidates[0].id;
}

function chooseGukjinHeuristicV3(state, playerKey, deps) {
  const branch = deps.analyzeGukjinBranches(state, playerKey);
  if (!branch.enabled || !branch.scenarios.length) return "junk";

  const asFive = branch.scenarios.filter((s) => s.selfMode === "five");
  const asJunk = branch.scenarios.filter((s) => s.selfMode === "junk");

  const mongBakOpportunityAsFive = asFive.some((s) => s.canMongBakSelf);
  const mongBakRiskAsFive = asFive.some((s) => s.mongRiskSelf);
  const mongBakRiskIfJunk = asJunk.some((s) => s.mongRiskSelf);

  if (mongBakOpportunityAsFive || mongBakRiskAsFive || mongBakRiskIfJunk) return "five";
  return "junk";
}

function shouldPresidentStopV3() {
  return false;
}

function shouldGoV3(state, playerKey, deps) {
  const player = state.players?.[playerKey];
  const opp = otherPlayerKeyFromDeps(playerKey, deps);
  const ctx = deps.analyzeGameContext(state, playerKey);
  const jokboThreat = deps.checkOpponentJokboProgress(state, playerKey);
  const goCount = player?.goCount || 0;
  const deckCount = state.deck?.length || 0;
  const myScore = ctx.myScore;
  const oppScore = ctx.oppScore;
  const oppThreat = deps.boardHighValueThreatForPlayer(state, opp);
  const oppProgThreat = deps.opponentThreatScore(state, playerKey);
  const oppNextMatchCount = deps.matchableMonthCountForPlayer(state, opp);
  const oppNextTurnThreat = deps.nextTurnThreatScore(state, playerKey);
  const carry = state.carryOverMultiplier || 1;
  const selfPi = Number(ctx.selfPi ?? deps.capturedCountByCategory(player, "junk"));
  const oppPi = Number(ctx.oppPi ?? deps.capturedCountByCategory(state.players?.[opp], "junk"));
  const selfJokboEV = deps.estimateJokboExpectedPotential(state, playerKey, opp);
  const oppJokboEV = deps.estimateJokboExpectedPotential(state, opp, playerKey);
  const selfFive = ctx.selfFive || 0;
  const oppFive = ctx.oppFive || 0;
  const mongDanger = ctx.mongDanger || 0;
  const isSecond = !!ctx.isSecond;
  const strongLead = myScore >= 10 && oppScore <= 4;
  const goldRisk = deps.goldRiskProfile(state, playerKey);
  const desperateGo = goldRisk.selfLow && !goldRisk.oppLow;
  const conservativeGo = goldRisk.oppLow;
  if (deps.canBankruptOpponentByStop(state, playerKey)) return false;
  if (ctx.mode === "DESPERATE_DEFENSE" && !desperateGo) return false;
  if (ctx.nagariDelayMode && !desperateGo) return false;
  if (selfPi < (desperateGo ? 6 : 7)) return false;
  if (selfFive === 0 && oppFive >= 7 && !desperateGo) return false;
  if (selfFive === 0 && oppFive >= 6 && deckCount <= 5 && !desperateGo) return false;
  if (
    isSecond &&
    !strongLead &&
    oppJokboEV.oneAwayCount >= 1 &&
    deckCount <= 6 &&
    (oppNextTurnThreat >= 0.35 || oppProgThreat >= 0.5) &&
    !desperateGo
  ) {
    return false;
  }
  if (!strongLead && deckCount <= 5 && oppJokboEV.oneAwayCount >= 1 && selfJokboEV.oneAwayCount === 0 && !desperateGo) {
    return false;
  }
  if (mongDanger >= 0.75 && !strongLead && !desperateGo) return false;
  if (
    mongDanger >= 0.6 &&
    (oppThreat || oppProgThreat >= 0.5 || oppNextTurnThreat >= 0.35) &&
    !strongLead &&
    !desperateGo
  ) {
    return false;
  }
  let carryStopBias = 0;
  if (carry >= 4) {
    const lowRisk =
      oppProgThreat < 0.35 &&
      oppNextTurnThreat < 0.25 &&
      jokboThreat.threat < 0.2 &&
      oppJokboEV.total < 0.6 &&
      mongDanger < 0.45 &&
      !(selfFive === 0 && oppFive >= 6);
    const noImmediateCounter = oppNextMatchCount === 0 && deckCount >= 7;
    if (!(strongLead && lowRisk && noImmediateCounter && selfPi >= 9 && oppPi <= 7) && !desperateGo) {
      carryStopBias += 0.22;
    }
  }
  if (!strongLead && selfPi < (desperateGo ? 6 : 7) && oppPi >= 6) return false;
  if (!strongLead && deckCount <= 5 && selfPi < (desperateGo ? 8 : 9)) return false;
  if (
    myScore >= 7 &&
    oppScore >= 5 &&
    (oppThreat ||
      oppProgThreat >= 0.6 ||
      oppNextMatchCount > 0 ||
      oppNextTurnThreat >= 0.43 ||
      jokboThreat.threat >= 0.35) &&
    !desperateGo
  ) {
    return false;
  }
  if (goCount >= 3 && !strongLead && !desperateGo && !deps.isOppVulnerableForBigGo(state, playerKey)) {
    return false;
  }

  const myGainPotential =
    Math.max(0, myScore - 6) * 0.12 +
    (10 - Math.min(10, ctx.deckCount)) * 0.02 +
    (selfPi >= 9 ? 0.2 : 0) +
    selfJokboEV.total * 0.34 +
    selfJokboEV.oneAwayCount * 0.12;
  const oppGainPotential =
    oppProgThreat * 0.65 +
    oppNextTurnThreat * 0.55 +
    jokboThreat.threat * 0.45 +
    mongDanger * 0.55 +
    oppJokboEV.total * 0.4 +
    oppJokboEV.oneAwayCount * 0.14;
  let goMargin = isSecond ? 0.22 : 0.12;
  if (desperateGo) goMargin -= 0.18;
  if (conservativeGo) goMargin += 0.22;
  goMargin -= 0.16;
  if (isSecond && !strongLead && myScore < oppScore) goMargin -= 0.04;
  goMargin += carryStopBias;
  goMargin = Math.max(-0.1, goMargin);
  if (!strongLead && myGainPotential < oppGainPotential + goMargin) return false;
  const lowDeckGate = isSecond ? 7 : 8;
  const lowJokboGate = isSecond ? 0.4 : 0.45;
  if (!strongLead && deckCount <= lowDeckGate && selfJokboEV.total < lowJokboGate && !desperateGo) return false;
  if (isSecond && !strongLead && (oppProgThreat >= 0.4 && oppNextMatchCount > 0) && !desperateGo) return false;
  if ((oppProgThreat >= 0.5 || oppNextTurnThreat >= 0.35) && !desperateGo) return false;
  if (selfFive === 0 && oppFive >= 6 && !desperateGo) return false;
  if (
    (oppNextMatchCount >= 2 ||
      (oppNextMatchCount > 0 &&
        (oppProgThreat >= 0.45 || oppNextTurnThreat >= 0.3 || jokboThreat.threat >= 0.3))) &&
    !strongLead &&
    !desperateGo
  ) {
    return false;
  }
  if (conservativeGo && !strongLead && myGainPotential <= oppGainPotential + goMargin + 0.08) return false;
  if (deps.capturedCountByCategory(state.players?.[opp], "junk") < 6) return !conservativeGo || desperateGo;
  if (oppNextMatchCount === 0 && oppScore <= 4 && deckCount >= (desperateGo ? 4 : 5)) return !conservativeGo || desperateGo;
  return deckCount > (desperateGo ? 4 : 6) && (!conservativeGo || strongLead);
}

function selectBestMonthByGain(state, months, deps) {
  if (!months?.length) return null;
  let best = months[0];
  let bestScore = deps.monthBoardGain(state, best);
  for (const m of months.slice(1)) {
    const score = deps.monthBoardGain(state, m);
    if (score > bestScore) {
      best = m;
      bestScore = score;
    }
  }
  return best;
}

function selectBombMonthV3(state, _playerKey, bombMonths, deps) {
  return selectBestMonthByGain(state, bombMonths, deps);
}

function shouldBombV3(state, playerKey, bombMonths, deps) {
  const ctx = deps.analyzeGameContext(state, playerKey);
  const firstTurnPiPlan = deps.getFirstTurnDoublePiPlan(state, playerKey);
  if (firstTurnPiPlan.active) {
    const target = (bombMonths || []).find((m) => firstTurnPiPlan.months.has(m));
    if (target != null) return true;
  }
  const bestMonth = selectBestMonthByGain(state, bombMonths, deps);
  const bestGain = deps.monthBoardGain(state, bestMonth);
  if (bestMonth == null) return false;
  const impact = deps.isHighImpactBomb(state, playerKey, bestMonth);
  if (ctx.defenseOpening) return impact.highImpact;
  if (ctx.volatilityComeback) {
    if (impact.highImpact) return true;
    if (impact.immediateGain >= 4) return true;
    return bestGain >= 0;
  }
  if (ctx.nagariDelayMode && !impact.highImpact && impact.immediateGain < 6) return false;
  return bestGain >= 1;
}

function shakingTempoPressureScoreV3(state, playerKey, ctx, oppThreat, deps) {
  const opp = otherPlayerKeyFromDeps(playerKey, deps);
  const self = state.players?.[playerKey];
  const oppPlayer = state.players?.[opp];
  const trailingBy = Math.max(0, (ctx.oppScore || 0) - (ctx.myScore || 0));
  let tempo = Math.min(2.0, trailingBy * 0.5);
  if (ctx.mode === "DESPERATE_DEFENSE") tempo += 1.0;
  if ((state.carryOverMultiplier || 1) >= 2) tempo += 0.45;
  if ((oppPlayer?.events?.shaking || 0) + (oppPlayer?.events?.bomb || 0) > 0) tempo += 0.7;
  if ((oppPlayer?.goCount || 0) > 0) tempo += 0.5;
  if ((self?.goCount || 0) > 0 && trailingBy === 0) tempo -= 0.35;
  if (oppThreat >= 0.7 && trailingBy > 0) tempo += 0.35;
  return tempo;
}

function shakingRiskPenaltyV3(state, playerKey, ctx, oppThreat, jokboThreat, nextThreat) {
  const self = state.players?.[playerKey];
  let risk = oppThreat * 2.0 + jokboThreat * 1.3 + nextThreat * 1.2;
  if ((ctx.deckCount || 0) <= 8) risk += 0.6;
  if ((ctx.myScore || 0) >= 7 && (ctx.myScore || 0) >= (ctx.oppScore || 0)) risk += 0.7;
  if ((state.carryOverMultiplier || 1) >= 2) risk += 0.4;
  if ((self?.goCount || 0) > 0) risk += 0.25;
  return risk;
}

function slowPlayPenaltyV3(immediateGain, comboGain, tempoPressure, ctx) {
  const lowPracticalGain = immediateGain < 0.75 && comboGain < 0.9;
  const noTempoNeed = tempoPressure < 1.2;
  const notBehind = (ctx.myScore || 0) >= (ctx.oppScore || 0);
  if (lowPracticalGain && noTempoNeed && notBehind) return 1.15;
  if (lowPracticalGain && noTempoNeed) return 0.65;
  return 0;
}

function decideShakingV3(state, playerKey, shakingMonths, deps) {
  if (!shakingMonths?.length) return { allow: false, month: null, score: -Infinity };

  const ctx = deps.analyzeGameContext(state, playerKey);
  const firstTurnPiPlan = deps.getFirstTurnDoublePiPlan(state, playerKey);
  const oppThreat = deps.opponentThreatScore(state, playerKey);
  const jokboThreat = deps.checkOpponentJokboProgress(state, playerKey).threat;
  const nextThreat = deps.nextTurnThreatScore(state, playerKey);
  const tempoPressure = shakingTempoPressureScoreV3(state, playerKey, ctx, oppThreat, deps);
  const riskPenalty = shakingRiskPenaltyV3(state, playerKey, ctx, oppThreat, jokboThreat, nextThreat);
  const trailingBy = Math.max(0, (ctx.oppScore || 0) - (ctx.myScore || 0));
  const selfPi = deps.capturedCountByCategory(state.players?.[playerKey], "junk");
  const opp = otherPlayerKeyFromDeps(playerKey, deps);
  const oppPi = deps.capturedCountByCategory(state.players?.[opp], "junk");
  const aheadOrEven = (ctx.myScore || 0) >= (ctx.oppScore || 0);
  const piFinishWindow = selfPi >= 9;
  const piPressureWindow = selfPi >= 8 || oppPi <= 6;

  let best = { allow: false, month: null, score: -Infinity, highImpact: false, immediateGain: 0, comboGain: 0 };
  for (const month of shakingMonths) {
    let immediateGain = deps.shakingImmediateGainScore(state, playerKey, month);
    if (firstTurnPiPlan.active && firstTurnPiPlan.months.has(month)) immediateGain += 0.45;
    const comboGain = deps.ownComboOpportunityScore(state, playerKey, month);
    const impact = deps.isHighImpactShaking(state, playerKey, month);
    if (impact.directThreeGwang) immediateGain += 0.55;
    if (impact.hasDoublePiLine) immediateGain += 0.35;
    const slowPenalty = slowPlayPenaltyV3(immediateGain, comboGain, tempoPressure, ctx);
    const monthTieBreak = deps.monthStrategicPriority(month) * 0.25;
    let score = immediateGain * 1.3 + comboGain * 1.15 + tempoPressure - riskPenalty - slowPenalty + monthTieBreak;
    if (trailingBy >= 3) score += 0.35;
    if (ctx.volatilityComeback) {
      score += 0.45;
      if (impact.highImpact) score += 0.22;
    }
    if (score > best.score) {
      best = {
        allow: false,
        month,
        score,
        highImpact: impact.highImpact,
        immediateGain,
        comboGain
      };
    }
  }

  let threshold = 0.65;
  if (ctx.mode === "DESPERATE_DEFENSE") threshold -= 0.25;
  if (trailingBy >= 3) threshold -= 0.15;
  if ((ctx.myScore || 0) >= (ctx.oppScore || 0) + 2) threshold += 0.4;
  if (oppThreat >= 0.7 && (ctx.myScore || 0) >= (ctx.oppScore || 0)) threshold += 0.35;
  if ((ctx.deckCount || 0) <= 7 && (ctx.myScore || 0) >= (ctx.oppScore || 0)) threshold += 0.25;
  if ((state.carryOverMultiplier || 1) >= 2 && (ctx.myScore || 0) >= (ctx.oppScore || 0)) threshold += 0.2;
  if (piPressureWindow && aheadOrEven) threshold += 0.2;
  if (piFinishWindow && aheadOrEven) threshold += 0.35;
  if (oppPi <= 5 && aheadOrEven) threshold += 0.15;
  if (ctx.volatilityComeback) threshold -= 0.3;
  if (best.month != null && firstTurnPiPlan.active && firstTurnPiPlan.months.has(best.month)) threshold -= 0.15;

  if (piFinishWindow && aheadOrEven && tempoPressure < 2.2 && best.score < threshold + 0.55) {
    return { ...best, allow: false };
  }
  if (piPressureWindow && aheadOrEven && oppThreat >= 0.65 && best.score < threshold + 0.35) {
    return { ...best, allow: false };
  }

  if (oppThreat >= 0.8 && (ctx.myScore || 0) >= (ctx.oppScore || 0) + 1 && tempoPressure < 1.6) {
    return { ...best, allow: false };
  }
  if (ctx.defenseOpening && !best.highImpact) return { ...best, allow: false };
  if (ctx.defenseOpening && best.immediateGain < 1.05 && best.comboGain < 1.35) return { ...best, allow: false };
  if (ctx.nagariDelayMode && !best.highImpact && best.score < threshold + 0.35) return { ...best, allow: false };
  if (ctx.volatilityComeback && best.month != null) {
    const comebackThreshold = threshold - (best.highImpact ? 0.18 : 0.08);
    return { ...best, allow: best.score >= comebackThreshold };
  }
  return { ...best, allow: best.score >= threshold };
}
