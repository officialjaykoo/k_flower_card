function safeNumber(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

const GUKJIN_CARD_ID = "I0";
const DOUBLE_PI_MONTHS = Object.freeze([11, 12, 13]);
const BONUS_CARD_ID_SET = new Set(["M0", "M1"]);
const SSANGPI_WITH_GUKJIN_ID_SET = new Set(["K1", "L3", GUKJIN_CARD_ID]);

function hasComboTag(card, tag) {
  return Array.isArray(card?.comboTags) && card.comboTags.includes(tag);
}

function countComboTag(cards, tag) {
  return (cards || []).reduce((count, card) => (hasComboTag(card, tag) ? count + 1 : count), 0);
}

function comboCounts(player) {
  const ribbons = player?.captured?.ribbon || [];
  const fives = player?.captured?.five || [];
  const kwang = player?.captured?.kwang || [];
  return {
    red: countComboTag(ribbons, "redRibbons"),
    blue: countComboTag(ribbons, "blueRibbons"),
    plain: countComboTag(ribbons, "plainRibbons"),
    birds: countComboTag(fives, "fiveBirds"),
    kwang: kwang.length
  };
}

function hasCategory(cards, category) {
  return (cards || []).some((c) => c?.category === category);
}

function isDoublePiLike(card, deps) {
  if (!card) return false;
  if (card.id === GUKJIN_CARD_ID) return true;
  return card.category === "junk" && safeNumber(deps.junkPiValue(card), 0) >= 2;
}

function piLikeValue(card, deps) {
  if (!card) return 0;
  if (card.id === GUKJIN_CARD_ID) return 2;
  if (card.category === "junk") return safeNumber(deps.junkPiValue(card), 0);
  return 0;
}

function ownComboFinishBonus(capturedCombo, captureCards) {
  let bonus = 0;
  if (capturedCombo.birds >= 2 && captureCards.some((c) => hasComboTag(c, "fiveBirds"))) bonus += 30;
  if (capturedCombo.red >= 2 && captureCards.some((c) => hasComboTag(c, "redRibbons"))) bonus += 27;
  if (capturedCombo.blue >= 2 && captureCards.some((c) => hasComboTag(c, "blueRibbons"))) bonus += 27;
  if (capturedCombo.plain >= 2 && captureCards.some((c) => hasComboTag(c, "plainRibbons"))) bonus += 27;
  if (capturedCombo.kwang >= 2 && captureCards.some((c) => c?.category === "kwang")) bonus += 32;
  return bonus;
}

function opponentComboBlockBonus(month, jokboThreat, blockMonths, blockUrgency, nextThreat) {
  let bonus = 0;
  const monthUrgency = safeNumber(jokboThreat?.monthUrgency?.get(month), 0);
  if (monthUrgency > 0) {
    bonus += 24 + monthUrgency * 0.35;
  }
  if (blockMonths?.has(month)) {
    const urgency = blockUrgency?.get(month) || 2;
    bonus += urgency >= 3 ? 18 : 10;
    bonus += nextThreat * 4.5;
  }
  return bonus;
}

function getCapturedDoublePiMonths(state) {
  const captured = new Set();
  for (const key of ["human", "ai"]) {
    const player = state.players?.[key];
    for (const card of player?.captured?.junk || []) {
      if (DOUBLE_PI_MONTHS.includes(card?.month) && safeNumber(card?.piValue, 0) >= 2) {
        captured.add(card.month);
      }
    }
  }
  return captured;
}

function getLiveDoublePiMonths(state) {
  const captured = getCapturedDoublePiMonths(state);
  const live = new Set();
  for (const month of DOUBLE_PI_MONTHS) {
    if (!captured.has(month)) live.add(month);
  }
  return live;
}

function getComboHoldMonths(state, playerKey, deps) {
  const opp = deps.otherPlayerKey(playerKey);
  const selfPlayer = state.players?.[playerKey];
  const oppPlayer = state.players?.[opp];
  const hold = new Set();
  for (const m of deps.blockingMonthsAgainst(selfPlayer, oppPlayer)) hold.add(m);
  for (const m of deps.blockingMonthsAgainst(oppPlayer, selfPlayer)) hold.add(m);
  return hold;
}

function discardTieOrderScore(card, deps, monthIsLiveDoublePi) {
  if (card?.bonus?.stealPi) return 6;
  if (isDoublePiLike(card, deps) && monthIsLiveDoublePi) return 1;
  if (card?.category === "five") return 5; // 열
  if (card?.category === "ribbon") return 4; // 띠
  if (card?.category === "kwang") return 3; // 광
  if (card?.category === "junk") return 2; // 피
  return 2;
}

export function rankHandCardsV4(state, playerKey, deps) {
  const player = state.players?.[playerKey];
  if (!player?.hand?.length) return [];

  const opp = deps.otherPlayerKey(playerKey);
  const oppPlayer = state.players?.[opp];
  const ctx = deps.analyzeGameContext(state, playerKey);
  const selfPi = safeNumber(ctx.selfPi, deps.capturedCountByCategory(player, "junk"));
  const oppPi = safeNumber(ctx.oppPi, deps.capturedCountByCategory(oppPlayer, "junk"));
  const nextThreat = deps.nextTurnThreatScore(state, playerKey);
  const jokboThreat = deps.checkOpponentJokboProgress(state, playerKey);
  const blockMonths = deps.blockingMonthsAgainst(oppPlayer, player);
  const blockUrgency = deps.blockingUrgencyByMonth(oppPlayer, player);
  const firstTurnPiPlan = deps.getFirstTurnDoublePiPlan(state, playerKey);
  const boardByMonth = deps.boardMatchesByMonth(state);
  const boardCountByMonth = deps.monthCounts(state.board || []);
  const handCountByMonth = deps.monthCounts(player.hand || []);
  const capturedByMonth = deps.capturedMonthCounts(state);
  const deckCount = safeNumber(state.deck?.length, 0);
  const selfCombo = comboCounts(player);
  const selfRibbonCount = (player?.captured?.ribbon || []).length;
  const selfFiveCount = (player?.captured?.five || []).length;
  const mongBakDefenseCritical = safeNumber(ctx.selfFive, 0) <= 0 && safeNumber(ctx.oppFive, 0) >= 7;
  const liveDoublePiMonths = getLiveDoublePiMonths(state);
  const comboHoldMonths = getComboHoldMonths(state, playerKey, deps);

  const ranked = player.hand.map((card) => {
    const matches = boardByMonth.get(card.month) || [];
    const captureCards = [card, ...matches];
    const captureGain = matches.reduce((sum, c) => sum + deps.cardCaptureValue(c), 0);
    const ownValue = deps.cardCaptureValue(card);
    const piGain = captureCards.reduce((sum, c) => sum + piLikeValue(c, deps), 0);
    const doublePiCount = captureCards.filter((c) => isDoublePiLike(c, deps)).length;
    const knownMonth =
      safeNumber(boardCountByMonth.get(card.month), 0) +
      safeNumber(handCountByMonth.get(card.month), 0) +
      safeNumber(capturedByMonth.get(card.month), 0);
    const monthIsLiveDoublePi = liveDoublePiMonths.has(card.month);
    const monthIsComboHold = comboHoldMonths.has(card.month);
    const monthBlockUrgency = safeNumber(blockUrgency.get(card.month), 0);
    const monthJokboUrgency = safeNumber(jokboThreat?.monthUrgency?.get(card.month), 0);
    const monthIsOneAwayThreat = monthBlockUrgency >= 3 || monthJokboUrgency >= 24;

    let score = 0;
    if (matches.length === 0) {
      score = -40 - ownValue * 0.9;
    } else if (matches.length === 1) {
      score = 48 + captureGain - ownValue * 0.1;
    } else if (matches.length === 2) {
      score = 56 + captureGain;
    } else {
      score = 62 + captureGain * 1.15;
    }

    // Rule 2: prioritize pi.
    score += piGain * 4.2;
    if (selfPi >= 7 && selfPi <= 9) score += piGain * 1.8;
    if (oppPi <= 5) score += piGain * 1.4;

    // Rule 3: prioritize double-pi (including gukjin card).
    if (doublePiCount > 0) {
      score += 16 + (doublePiCount - 1) * 6;
    }
    if (matches.length === 0 && isDoublePiLike(card, deps)) {
      score -= 14;
    }

    // Rules 5/6: combo completion and combo blocking are higher than pi.
    score += ownComboFinishBonus(selfCombo, captureCards);
    score += opponentComboBlockBonus(card.month, jokboThreat, blockMonths, blockUrgency, nextThreat);

    // Rule 7: if self ribbon/five is already 4, 1 more ribbon/five > pi.
    if (selfRibbonCount >= 4 && hasCategory(captureCards, "ribbon")) {
      score += 34;
    }
    if (selfFiveCount >= 4 && hasCategory(captureCards, "five")) {
      score += 36;
    }

    // Rule 8: mong-bak defense, when opp has 7+ five and self has 0 five.
    if (mongBakDefenseCritical) {
      if (hasCategory(captureCards, "five")) score += 40;
      else if (piGain > 0) score -= 8;
    }

    if (matches.length === 0) {
      if (knownMonth >= 3) score += 1.9;
      else if (knownMonth <= 1) score -= 1.8;
    }

    // Rule 4: de-prioritize non-double-pi lines when month state is already locked.
    const capCntMonth = safeNumber(capturedByMonth.get(card.month), 0);
    if (matches.length > 0 && doublePiCount === 0 && capCntMonth >= 2 && knownMonth >= 3) {
      score -= 6.0;
    }

    if (matches.length === 0) {
      // No-match discard policy:
      // avoid live ssangpi months and combo months; otherwise reverse discard order.
      let discardScore = 0;
      if (card?.bonus?.stealPi) discardScore += 26; // prefer bonus-pi discard first
      if (monthIsLiveDoublePi) discardScore -= deckCount <= 8 ? 36 : 24;
      if (isDoublePiLike(card, deps) && monthIsLiveDoublePi) discardScore -= deckCount <= 8 ? 26 : 16;
      if (isDoublePiLike(card, deps) && !monthIsLiveDoublePi) discardScore += 6;
      if (monthIsComboHold) discardScore -= deckCount <= 8 ? 56 : 44; // hold combo-critical month
      if (monthIsOneAwayThreat) discardScore -= deckCount <= 8 ? 58 : 42;
      else if (monthBlockUrgency >= 2 || monthJokboUrgency >= 20) discardScore -= deckCount <= 8 ? 30 : 20;
      if (mongBakDefenseCritical && card.category === "five") discardScore -= 28;
      if (mongBakDefenseCritical && card.category === "junk") discardScore += 5;

      // Tie order when still similar: five > ribbon > kwang > pi > ssangpi.
      const tieOrder = discardTieOrderScore(card, deps, monthIsLiveDoublePi);
      discardScore += tieOrder * 2.2;
      score += discardScore;
    }

    const feedRisk = deps.estimateOpponentImmediateGainIfDiscard(state, playerKey, card.month);
    score -= feedRisk * (matches.length === 0 ? 5.0 : 1.2);

    const pukRisk = deps.isRiskOfPuk(state, playerKey, card, boardCountByMonth, handCountByMonth);
    if (pukRisk > 0) {
      score -= pukRisk * (deckCount <= 10 ? 4.8 : 3.4);
    } else if (pukRisk < 0) {
      score += -pukRisk * 1.4;
    }

    if (card.category === "five" && matches.length === 0) score -= 1.4;
    if (card.category === "kwang" && matches.length === 0) score -= 0.6;

    if (firstTurnPiPlan.active && firstTurnPiPlan.months.has(card.month)) {
      score += 5.5;
    }

    const uncertainBoost = deps.monthStrategicPriority(card.month);
    score += uncertainBoost * (matches.length === 0 ? 0.8 : 0.4);

    return {
      card,
      score,
      matches: matches.length,
      uncertainBoost
    };
  });

  ranked.sort((a, b) => b.score - a.score);
  if (ranked.length >= 2 && Math.abs(ranked[0].score - ranked[1].score) <= 0.8) {
    for (const item of ranked) item.score += safeNumber(item.uncertainBoost, 0) * 0.9;
    ranked.sort((a, b) => b.score - a.score);
  }
  return ranked;
}

export function chooseMatchHeuristicV4(state, playerKey, deps) {
  const ids = state.pendingMatch?.boardCardIds || [];
  if (!ids.length) return null;

  const opp = deps.otherPlayerKey(playerKey);
  const selfPlayer = state.players?.[playerKey];
  const oppPlayer = state.players?.[opp];
  const ctx = deps.analyzeGameContext(state, playerKey);
  const selfPi = safeNumber(ctx.selfPi, deps.capturedCountByCategory(selfPlayer, "junk"));
  const oppPi = safeNumber(ctx.oppPi, deps.capturedCountByCategory(oppPlayer, "junk"));
  const nextThreat = deps.nextTurnThreatScore(state, playerKey);
  const jokbo = deps.checkOpponentJokboProgress(state, playerKey);
  const blockMonths = deps.blockingMonthsAgainst(oppPlayer, selfPlayer);
  const blockUrgency = deps.blockingUrgencyByMonth(oppPlayer, selfPlayer);
  const selfCombo = comboCounts(selfPlayer);
  const selfRibbonCount = (selfPlayer?.captured?.ribbon || []).length;
  const selfFiveCount = (selfPlayer?.captured?.five || []).length;
  const mongBakDefenseCritical = safeNumber(ctx.selfFive, 0) <= 0 && safeNumber(ctx.oppFive, 0) >= 7;

  const candidates = (state.board || []).filter((c) => ids.includes(c.id));
  if (!candidates.length) return null;

  let best = candidates[0];
  let bestScore = -Infinity;
  for (const c of candidates) {
    let score = deps.cardCaptureValue(c) * 0.8;
    const piGain = piLikeValue(c, deps);

    // Same base structure as hand choice: ssangpi/pi/kwang/ribbon/five
    score += piGain * 4.0;
    if (c.category === "kwang") score += 8.0;
    if (c.category === "ribbon") score += 6.0;
    if (c.category === "five") score += 4.0;
    if (selfPi >= 7 && selfPi <= 9) score += piGain * 1.8;
    if (oppPi <= 5) score += piGain * 1.4;

    if (isDoublePiLike(c, deps)) score += 14;

    score += ownComboFinishBonus(selfCombo, [c]);
    score += opponentComboBlockBonus(c.month, jokbo, blockMonths, blockUrgency, nextThreat);

    if (selfRibbonCount >= 4 && c.category === "ribbon") score += 34;
    if (selfFiveCount >= 4 && c.category === "five") score += 36;

    if (mongBakDefenseCritical) {
      if (c.category === "five") score += 40;
      else if (piGain > 0) score -= 8;
    }

    score += deps.monthStrategicPriority(c.month) * 0.25;
    if (ctx.mode === "DESPERATE_DEFENSE" && piLikeValue(c, deps) <= 0) score -= 0.45;

    if (score > bestScore) {
      bestScore = score;
      best = c;
    }
  }

  return best?.id ?? null;
}

export function chooseGukjinHeuristicV4(state, playerKey, deps) {
  const ctx = deps.analyzeGameContext(state, playerKey);
  const branch = deps.analyzeGukjinBranches(state, playerKey);

  const myScore = safeNumber(ctx?.myScore, 0);
  const oppScore = safeNumber(ctx?.oppScore, 0);
  const selfFive = safeNumber(ctx?.selfFive, 0);
  const oppFive = safeNumber(ctx?.oppFive, 0);

  const losing = myScore < oppScore;
  const winning = myScore > oppScore;

  // Mong-bak state: either can be mong-bak'd or can mong-bak opponent.
  const mongBakRisk = selfFive <= 0 && oppFive >= 6;
  const mongBakChance = selfFive >= 7 && oppFive <= 0;
  const mongBakByBranch = !!branch?.mongRiskAny || !!branch?.mongBakAny;

  if (mongBakRisk || mongBakChance || mongBakByBranch) return "five";
  if (losing && selfFive <= 0) return "five";
  if (winning && selfFive >= 6) return "five";

  return "junk";
}

function isBonusCard(card) {
  return safeNumber(card?.bonus?.stealPi, 0) > 0;
}

function isSsangpiCard(card) {
  if (!card) return false;
  if (card.category !== "junk") return false;
  if (isBonusCard(card)) return false;
  return safeNumber(card?.piValue, 0) >= 2;
}

function countCards(cards, predicate) {
  return (cards || []).reduce((count, card) => (predicate(card) ? count + 1 : count), 0);
}

function collectVisibleCardsForPlayer(state, playerKey) {
  const visible = [];
  const player = state.players?.[playerKey];
  if (player?.hand?.length) visible.push(...player.hand);
  if (state.board?.length) visible.push(...state.board);
  for (const key of ["human", "ai"]) {
    const cap = state.players?.[key]?.captured || {};
    visible.push(...(cap.kwang || []), ...(cap.five || []), ...(cap.ribbon || []), ...(cap.junk || []));
  }
  return visible;
}

function countUnseenByIdSet(state, playerKey, idSet) {
  const seen = new Set();
  for (const card of collectVisibleCardsForPlayer(state, playerKey)) {
    if (!card?.id) continue;
    if (idSet.has(card.id)) seen.add(card.id);
  }
  return Math.max(0, idSet.size - seen.size);
}

function opponentComboThreatProfile(state, playerKey, deps) {
  const oppKey = deps.otherPlayerKey(playerKey);
  const selfPlayer = state.players?.[playerKey];
  const oppPlayer = state.players?.[oppKey];
  if (!selfPlayer || !oppPlayer) {
    return { godori: false, ribbon: false, threeKwang: false, count: 0, any: false };
  }

  const oppRibbons = oppPlayer?.captured?.ribbon || [];
  const selfRibbons = selfPlayer?.captured?.ribbon || [];
  const redThreat = countComboTag(oppRibbons, "redRibbons") >= 2 && countComboTag(selfRibbons, "redRibbons") <= 0;
  const blueThreat =
    countComboTag(oppRibbons, "blueRibbons") >= 2 && countComboTag(selfRibbons, "blueRibbons") <= 0;
  const plainThreat =
    countComboTag(oppRibbons, "plainRibbons") >= 2 && countComboTag(selfRibbons, "plainRibbons") <= 0;
  const ribbonThreat = redThreat || blueThreat || plainThreat;

  const oppFives = oppPlayer?.captured?.five || [];
  const selfFives = selfPlayer?.captured?.five || [];
  const godoriThreat =
    countComboTag(oppFives, "fiveBirds") >= 2 && countComboTag(selfFives, "fiveBirds") <= 0;

  const oppKwangCount = deps.capturedCountByCategory(oppPlayer, "kwang");
  const selfKwangCount = deps.capturedCountByCategory(selfPlayer, "kwang");
  const threeKwangThreat = oppKwangCount >= 2 && selfKwangCount <= 2;

  const count = Number(godoriThreat) + Number(ribbonThreat) + Number(threeKwangThreat);
  return {
    godori: godoriThreat,
    ribbon: ribbonThreat,
    threeKwang: threeKwangThreat,
    count,
    any: count > 0
  };
}

function hasUnanswerablePpukBoardRisk(state, playerKey) {
  const handMonths = new Set(
    (state.players?.[playerKey]?.hand || [])
      .map((c) => Number(c?.month))
      .filter((m) => Number.isInteger(m))
  );
  const boardMonths = new Set(
    (state.board || [])
      .map((c) => Number(c?.month))
      .filter((m) => Number.isInteger(m))
  );

  for (const key of ["human", "ai"]) {
    const ppuk = state.players?.[key]?.ppukState;
    if (!ppuk?.active) continue;
    const month = Number(ppuk.lastMonth);
    if (!Number.isInteger(month)) continue;
    if (!boardMonths.has(month)) continue;
    if (!handMonths.has(month)) return true;
  }
  return false;
}

function hasCategoryEscapeLine(oppHand, boardByMonth, category) {
  for (const handCard of oppHand || []) {
    const matches = boardByMonth.get(handCard?.month) || [];
    if (!matches.length) continue;
    if (handCard?.category === category) return true;
    if (matches.some((c) => c?.category === category)) return true;
  }
  return false;
}

function hasPiEscapeLine(oppHand, boardByMonth, neededPi, deps) {
  for (const handCard of oppHand || []) {
    const matches = boardByMonth.get(handCard?.month) || [];
    if (!matches.length) continue;
    const piGain = piLikeValue(handCard, deps) + matches.reduce((sum, c) => sum + piLikeValue(c, deps), 0);
    if (piGain >= neededPi) return true;
  }
  return false;
}

function opponentBakEscapeLikely(state, playerKey, deps, ctx) {
  const selfPlayer = state.players?.[playerKey];
  const oppKey = deps.otherPlayerKey(playerKey);
  const oppPlayer = state.players?.[oppKey];
  if (!selfPlayer || !oppPlayer) return false;

  const boardByMonth = deps.boardMatchesByMonth(state);
  const selfKwang = deps.capturedCountByCategory(selfPlayer, "kwang");
  const oppKwang = deps.capturedCountByCategory(oppPlayer, "kwang");
  const selfPi = safeNumber(ctx?.selfPi, deps.capturedCountByCategory(selfPlayer, "junk"));
  const oppPi = safeNumber(ctx?.oppPi, deps.capturedCountByCategory(oppPlayer, "junk"));
  const selfFive = safeNumber(ctx?.selfFive, (selfPlayer?.captured?.five || []).length);
  const oppFive = safeNumber(ctx?.oppFive, (oppPlayer?.captured?.five || []).length);
  const oppHand = oppPlayer?.hand || [];

  const oppInGwangBak = selfKwang >= 3 && oppKwang === 0;
  if (oppInGwangBak && hasCategoryEscapeLine(oppHand, boardByMonth, "kwang")) return true;

  const oppInMongBak = selfFive >= 7 && oppFive === 0;
  if (oppInMongBak && hasCategoryEscapeLine(oppHand, boardByMonth, "five")) return true;

  const oppInPiBak = selfPi >= 10 && oppPi >= 1 && oppPi <= 7;
  if (oppInPiBak) {
    const neededPi = Math.max(1, Math.ceil(8 - oppPi));
    if (hasPiEscapeLine(oppHand, boardByMonth, neededPi, deps)) return true;
  }

  return false;
}

function shouldStopForOpponentFourLike(state, playerKey, deps, ctx, options = {}) {
  const bonusUnseenThreshold = safeNumber(options?.bonusUnseenThreshold, 2);
  const ssangpiUnseenThreshold = safeNumber(options?.ssangpiUnseenThreshold, 3);

  // 3-1
  const unseenBonus = countUnseenByIdSet(state, playerKey, BONUS_CARD_ID_SET);
  if (unseenBonus >= bonusUnseenThreshold) return true;

  const comboThreat = opponentComboThreatProfile(state, playerKey, deps);
  // 3-2
  if (comboThreat.any) return true;

  // 3-3
  if (hasUnanswerablePpukBoardRisk(state, playerKey)) return true;

  // 3-4
  if (opponentBakEscapeLikely(state, playerKey, deps, ctx)) return true;

  // 3-5
  const unseenSsangpi = countUnseenByIdSet(state, playerKey, SSANGPI_WITH_GUKJIN_ID_SET);
  if (unseenSsangpi >= ssangpiUnseenThreshold) return true;

  // 3-6 (same as 3-point condition)
  if (comboThreat.count >= 2) return true;

  return false;
}

export function shouldPresidentStopV4(state, playerKey, deps) {
  const ctx = deps.analyzeGameContext(state, playerKey);
  const player = state.players?.[playerKey];
  if (!player) return true;

  const myScore = safeNumber(ctx?.myScore, 0);
  const oppScore = safeNumber(ctx?.oppScore, 0);
  const winning = myScore > oppScore;

  const firstTurnPresident = safeNumber(state?.turnSeq, 0) <= 0;
  const capturedBonusCount = countCards(player.captured?.junk || [], isBonusCard);
  const handBonusCount = countCards(player.hand || [], isBonusCard);
  const handSsangpiCount = countCards(player.hand || [], isSsangpiCard);
  const capturedSsangpiCount = countCards(player.captured?.junk || [], isSsangpiCard);

  // First-turn hold-and-hit condition:
  // (1 AND 2) OR 3
  // 1) bonus is guaranteed (already captured OR in hand)
  // 2) at least one ssangpi in hand
  // 3) total (captured + hand) ssangpi+bonus is at least 3
  const guaranteedBonus = capturedBonusCount >= 1 || handBonusCount >= 1;
  const hasHandSsangpi = handSsangpiCount >= 1;
  const totalSsangpiBonusPotential =
    capturedBonusCount + handBonusCount + capturedSsangpiCount + handSsangpiCount;
  const firstTurnHoldAllowed =
    (guaranteedBonus && hasHandSsangpi) || totalSsangpiBonusPotential >= 3;

  // Mid-game hold is allowed only when currently winning.
  const midGameHoldAllowed = winning;

  const holdAllowed = firstTurnPresident ? firstTurnHoldAllowed : midGameHoldAllowed;
  return !holdAllowed;
}

function estimateOpponentOneAwayProbV4(state, playerKey, deps, ctx) {
  const deckCount = safeNumber(state?.deck?.length, 0);
  const oppScore = safeNumber(ctx?.oppScore, 0);
  const comboThreat = opponentComboThreatProfile(state, playerKey, deps);
  const jokboProfile = deps.checkOpponentJokboProgress(state, playerKey);
  const jokboThreat = safeNumber(jokboProfile?.threat, 0);
  const nextThreat = safeNumber(deps.nextTurnThreatScore(state, playerKey), 0);

  let topMonthUrgency = 0;
  const monthUrgency = jokboProfile?.monthUrgency;
  if (monthUrgency && typeof monthUrgency.values === "function") {
    for (const urgency of monthUrgency.values()) {
      topMonthUrgency = Math.max(topMonthUrgency, safeNumber(urgency, 0));
    }
  }

  let prob =
    comboThreat.count * 24 + // explicit near-completion combo lines
    jokboThreat * 52 +
    nextThreat * 36;

  if (topMonthUrgency >= 24) prob += 12;
  else if (topMonthUrgency >= 20) prob += 7;
  if (oppScore >= 3) prob += 5;

  // Late game weights: same threat should stop more often.
  if (deckCount <= 10) prob += 8;
  if (deckCount <= 6) prob += 6;
  if (deckCount <= 3) prob += 5;

  return {
    oppOneAwayProb: Math.max(0, Math.min(100, prob)),
    comboThreatCount: comboThreat.count,
    jokboThreat,
    nextThreat,
    deckCount
  };
}

function summarizeOppGukjinCase(branch, oppMode) {
  const scenarios = (branch?.scenarios || []).filter((s) => s?.oppMode === oppMode);
  if (!scenarios.length) return null;
  return {
    oppMode,
    oppScore: Math.max(...scenarios.map((s) => safeNumber(s?.oppScore, 0))),
    oppPi: Math.max(...scenarios.map((s) => safeNumber(s?.oppPi, 0))),
    oppFive: Math.max(...scenarios.map((s) => safeNumber(s?.oppFive, 0)))
  };
}

export function shouldGoV4(state, playerKey, deps) {
  if (deps.canBankruptOpponentByStop(state, playerKey)) return false;

  const ctx = deps.analyzeGameContext(state, playerKey);
  const myScore = safeNumber(ctx?.myScore, 0);
  const oppScoreBase = safeNumber(ctx?.oppScore, 0);
  const threat = estimateOpponentOneAwayProbV4(state, playerKey, deps, ctx);
  const lateGame = threat.deckCount <= 10;
  const selfPlayer = state.players?.[playerKey];
  const oppPlayer = state.players?.[deps.otherPlayerKey(playerKey)];
  const oppPiBase = safeNumber(ctx?.oppPi, deps.capturedCountByCategory(oppPlayer, "junk"));
  const selfCombo = comboCounts(selfPlayer);
  const myCertainJokbo =
    selfCombo.kwang >= 3 ||
    selfCombo.birds >= 3 ||
    selfCombo.red >= 3 ||
    selfCombo.blue >= 3 ||
    selfCombo.plain >= 3;
  const unseenBonus = countUnseenByIdSet(state, playerKey, BONUS_CARD_ID_SET);
  const unseenSsangpi = countUnseenByIdSet(state, playerKey, SSANGPI_WITH_GUKJIN_ID_SET);
  const unseenHighPi = unseenSsangpi + unseenBonus;
  let oppScoreRisk = oppScoreBase;
  let oppPiRisk = oppPiBase;
  const gukjinBranch = deps.analyzeGukjinBranches?.(state, playerKey);
  const oppCaseFive = summarizeOppGukjinCase(gukjinBranch, "five");
  const oppCaseJunk = summarizeOppGukjinCase(gukjinBranch, "junk");
  for (const c of [oppCaseFive, oppCaseJunk]) {
    if (!c) continue;
    oppScoreRisk = Math.max(oppScoreRisk, c.oppScore);
    oppPiRisk = Math.max(oppPiRisk, c.oppPi);
  }
  const caseRisk = (c) =>
    !!c &&
    (c.oppScore >= 6 ||
      (c.oppScore >= 5 && !myCertainJokbo) ||
      (unseenHighPi >= 2 && c.oppPi >= 7 && !myCertainJokbo));
  if (caseRisk(oppCaseFive) || caseRisk(oppCaseJunk)) return false;
  const scoreDiffRisk = myScore - oppScoreRisk;

  // Ssangpi risk gate:
  // when high-pi cards are still unseen, opponent pi is already high, and we don't have confirmed jokbo.
  if (unseenHighPi >= 2 && oppPiRisk >= 7 && !myCertainJokbo) return false;

  // 1) Opponent 6+: always STOP.
  if (oppScoreRisk >= 6) return false;

  // 2) Opponent 5+: default STOP.
  // Exception only when big score lead + low threat.
  if (oppScoreRisk >= 5) {
    const shouldStop = shouldStopForOpponentFourLike(state, playerKey, deps, ctx, {
      bonusUnseenThreshold: 1,
      ssangpiUnseenThreshold: 2
    });
    const bigLead = scoreDiffRisk >= 8 && myScore >= 11;
    const lowThreat =
      threat.oppOneAwayProb < (lateGame ? 20 : 25) &&
      threat.jokboThreat < 0.3 &&
      threat.nextThreat < 0.35 &&
      threat.comboThreatCount <= 1;
    return !shouldStop && bigLead && lowThreat;
  }

  // 3) Opponent 4: apply 3-1..3-6.
  if (oppScoreRisk >= 4) {
    const shouldStop = shouldStopForOpponentFourLike(state, playerKey, deps, ctx, {
      bonusUnseenThreshold: 2,
      ssangpiUnseenThreshold: 3
    });
    if (shouldStop) return false;
    if (threat.oppOneAwayProb >= (lateGame ? 28 : 32)) return false;
    return true;
  }

  // 4) Opponent 1~3: strengthen STOP gate with one-away probability.
  if (oppScoreRisk >= 1 && oppScoreRisk <= 3) {
    if (oppScoreRisk === 3 && threat.comboThreatCount >= 2) return false;
    const baseThreshold = oppScoreRisk === 3 ? 43 : oppScoreRisk === 2 ? 38 : 34;
    const threshold = baseThreshold - (lateGame ? 1 : 0);
    if (threat.oppOneAwayProb >= threshold) return false;

    // Opponent <=2: remove auto-GO, require minimum threat check.
    if (
      oppScoreRisk <= 2 &&
      (threat.jokboThreat >= 0.35 || threat.nextThreat >= 0.45 || (lateGame && threat.comboThreatCount >= 2))
    ) {
      return false;
    }
    return true;
  }

  // 5) Opponent 0: no automatic GO.
  if (threat.oppOneAwayProb >= (lateGame ? 33 : 37)) return false;
  if (threat.jokboThreat >= 0.37 || threat.nextThreat >= 0.47) return false;
  return true;
}

function hasOnlyBombMatchOption(state, playerKey, bombMonths, deps) {
  const player = state.players?.[playerKey];
  if (!player?.hand?.length) return false;
  const bombSet = new Set((bombMonths || []).map((m) => Number(m)));
  const boardByMonth = deps.boardMatchesByMonth(state);
  let hasAnyMatch = false;

  for (const card of player.hand) {
    const matches = boardByMonth.get(card.month) || [];
    if (!matches.length) continue;
    hasAnyMatch = true;
    if (!bombSet.has(Number(card.month))) return false;
  }

  return hasAnyMatch;
}

function canStealDoublePiFromOpponent(state, playerKey, deps) {
  const opp = deps.otherPlayerKey(playerKey);
  const oppPlayer = state.players?.[opp];
  if (!oppPlayer) return false;

  const hasDoublePiInJunk = (oppPlayer.captured?.junk || []).some(
    (card) => safeNumber(deps.junkPiValue(card), 0) >= 2
  );
  if (hasDoublePiInJunk) return true;

  if (oppPlayer.gukjinMode === "junk") {
    const hasGukjinInFive = (oppPlayer.captured?.five || []).some(
      (card) => card?.id === GUKJIN_CARD_ID && !card?.gukjinTransformed
    );
    if (hasGukjinInFive) return true;
  }
  return false;
}

export function selectBombMonthV4(state, playerKey, bombMonths, deps) {
  if (!Array.isArray(bombMonths) || !bombMonths.length) return null;

  let bestMonth = bombMonths[0];
  let bestScore = -Infinity;
  for (const month of bombMonths) {
    const gain = safeNumber(deps.monthBoardGain(state, month), 0);
    const impact = deps.isHighImpactBomb(state, playerKey, month);
    const monthCards = (state.board || []).filter((c) => c?.month === month);
    const doublePiPayload = monthCards.some((c) => isDoublePiLike(c, deps)) ? 8 : 0;
    const score =
      gain +
      safeNumber(impact?.immediateGain, 0) * 0.75 +
      (impact?.highImpact ? 2.8 : 0) +
      doublePiPayload;
    if (score > bestScore) {
      bestScore = score;
      bestMonth = month;
    }
  }
  return bestMonth;
}

export function shouldBombV4(state, playerKey, bombMonths, deps) {
  if (!Array.isArray(bombMonths) || !bombMonths.length) return false;

  const ctx = deps.analyzeGameContext(state, playerKey);

  // User rule: bomb only when all are true:
  // can steal double-pi, no practical matching except bomb, and current score is 7.
  const onlyBombMatch = hasOnlyBombMatchOption(state, playerKey, bombMonths, deps);
  const canStealDoublePi = canStealDoublePiFromOpponent(state, playerKey, deps);
  const scoreIsSeven = safeNumber(ctx.myScore, 0) === 7;
  return onlyBombMatch && canStealDoublePi && scoreIsSeven;
}

export function decideShakingV4(state, playerKey, shakingMonths, deps) {
  if (!Array.isArray(shakingMonths) || !shakingMonths.length) {
    return { allow: false, month: null, score: -Infinity, highImpact: false };
  }

  const ctx = deps.analyzeGameContext(state, playerKey);
  const player = state.players?.[playerKey];
  const boardByMonth = deps.boardMatchesByMonth(state);
  const hasAnyMatch = (player?.hand || []).some((card) => (boardByMonth.get(card.month) || []).length > 0);
  const liveDoublePiMonths = getLiveDoublePiMonths(state);
  const comboHoldMonths = getComboHoldMonths(state, playerKey, deps);

  // User rule: only consider shaking when no normal matching exists.
  if (hasAnyMatch) {
    return { allow: false, month: null, score: -Infinity, highImpact: false };
  }

  let best = {
    allow: false,
    month: null,
    score: -Infinity,
    highImpact: false,
    immediateGain: 0,
    comboGain: 0
  };

  for (const month of shakingMonths) {
    const immediateGain = deps.shakingImmediateGainScore(state, playerKey, month);
    const comboGain = deps.ownComboOpportunityScore(state, playerKey, month);
    const impact = deps.isHighImpactShaking(state, playerKey, month);
    const known = deps.countKnownMonthCards(state, month);
    const uncertaintyBonus = known <= 2 ? 0.25 : known >= 4 ? -0.1 : 0;

    let score = immediateGain * 1.35 + comboGain * 1.15 + uncertaintyBonus;
    if (impact?.hasDoublePiLine) score += 0.35;
    if (impact?.directThreeGwang) score += 0.3;
    if (impact?.highImpact) score += 0.4;
    if (liveDoublePiMonths.has(month) && !comboHoldMonths.has(month)) score += 0.55;
    if (comboHoldMonths.has(month)) score -= 0.25;

    if (score > best.score) {
      best = {
        allow: false,
        month,
        score,
        highImpact: !!impact?.highImpact,
        immediateGain,
        comboGain
      };
    }
  }

  const myScore = safeNumber(ctx.myScore, 0);
  const oppScore = safeNumber(ctx.oppScore, 0);
  const oppLeadHard = oppScore >= 5 && oppScore >= myScore + 2;

  if (oppLeadHard) {
    return { ...best, allow: false };
  }

  // Default: YES when ahead. Also allow ssangpi-month shaking when combo-hold is not required.
  const preferDoublePiShake = best.month != null && liveDoublePiMonths.has(best.month) && !comboHoldMonths.has(best.month);
  const allow = myScore > oppScore || preferDoublePiShake;
  return { ...best, allow };
}

