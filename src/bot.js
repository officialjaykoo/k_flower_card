import {
  playTurn,
  getDeclarableShakingMonths,
  declareShaking,
  getDeclarableBombMonths,
  declareBomb,
  chooseGo,
  chooseStop,
  choosePresidentStop,
  choosePresidentHold,
  chooseGukjinMode,
  chooseMatch,
  calculateScore
} from "./gameEngine.js";

const POLICY_HEURISTIC_V3 = "heuristic_v3";

export const BOT_POLICIES = [POLICY_HEURISTIC_V3];

export function botChooseCard(state, playerKey, policy = POLICY_HEURISTIC_V3) {
  const player = state.players[playerKey];
  if (!player || player.hand.length === 0) return null;
  const ranked = rankHandCards(state, playerKey);
  if (ranked.length > 0) {
    return ranked[0].card.id;
  }
  return pickRandom(player.hand)?.id ?? null;
}

export function botPlay(state, playerKey, options = {}) {
  const policy = normalizePolicy(options.policy);
  return botPlaySmart(state, playerKey, policy);
}

export function getHeuristicCardProbabilities(state, playerKey, policy = POLICY_HEURISTIC_V3) {
  if (state.phase !== "playing") return null;
  const ranked = rankHandCards(state, playerKey);
  if (!ranked.length) return null;
  normalizePolicy(policy);
  const temp = 1.6;
  const maxScore = Math.max(...ranked.map((r) => Number(r.score || 0)));
  let z = 0;
  const exps = ranked.map((r) => {
    const v = Math.exp((Number(r.score || 0) - maxScore) / temp);
    z += v;
    return { id: r.card.id, v };
  });
  if (z <= 0) return null;
  const probs = {};
  for (const e of exps) probs[e.id] = e.v / z;
  return probs;
}

function botPlayRandom(state, playerKey) {
  if (state.phase === "gukjin-choice" && state.pendingGukjinChoice?.playerKey === playerKey) {
    return chooseGukjinMode(state, playerKey, Math.random() < 0.5 ? "five" : "junk");
  }

  if (state.phase === "president-choice" && state.pendingPresident?.playerKey === playerKey) {
    return Math.random() < 0.5
      ? choosePresidentStop(state, playerKey)
      : choosePresidentHold(state, playerKey);
  }

  if (state.phase === "select-match" && state.pendingMatch?.playerKey === playerKey) {
    const option = pickRandom(
      state.board.filter((c) => (state.pendingMatch?.boardCardIds || []).includes(c.id))
    );
    return option ? chooseMatch(state, option.id) : state;
  }

  if (state.phase === "go-stop" && state.pendingGoStop === playerKey) {
    return Math.random() < 0.5 ? chooseGo(state, playerKey) : chooseStop(state, playerKey);
  }

  if (state.phase === "playing" && state.currentTurn === playerKey) {
    const bombMonths = getDeclarableBombMonths(state, playerKey);
    if (bombMonths.length > 0 && Math.random() < 0.5) {
      return declareBomb(state, playerKey, pickRandom(bombMonths));
    }
    const shakingMonths = getDeclarableShakingMonths(state, playerKey);
    if (shakingMonths.length > 0 && Math.random() < 0.5) {
      return declareShaking(state, playerKey, pickRandom(shakingMonths));
    }
    const cardId = botChooseCard(state, playerKey, POLICY_HEURISTIC_V3);
    if (!cardId) return state;
    return playTurn(state, cardId);
  }

  return state;
}

function botPlaySmart(state, playerKey, policy) {
  if (state.phase === "gukjin-choice" && state.pendingGukjinChoice?.playerKey === playerKey) {
    return chooseGukjinMode(state, playerKey, chooseGukjinHeuristic(state, playerKey, policy));
  }

  if (state.phase === "president-choice" && state.pendingPresident?.playerKey === playerKey) {
    const shouldStop = shouldPresidentStop(state, playerKey, policy);
    return shouldStop ? choosePresidentStop(state, playerKey) : choosePresidentHold(state, playerKey);
  }

  if (state.phase === "select-match" && state.pendingMatch?.playerKey === playerKey) {
    const choiceId = chooseMatchHeuristic(state, playerKey);
    return choiceId ? chooseMatch(state, choiceId) : state;
  }

  if (state.phase === "go-stop" && state.pendingGoStop === playerKey) {
    return shouldGo(state, playerKey, policy) ? chooseGo(state, playerKey) : chooseStop(state, playerKey);
  }

  if (state.phase === "playing" && state.currentTurn === playerKey) {
    const bombMonths = getDeclarableBombMonths(state, playerKey);
    if (bombMonths.length > 0 && shouldBomb(state, playerKey, bombMonths, policy)) {
      return declareBomb(state, playerKey, selectBestMonth(state, bombMonths));
    }
    const shakingMonths = getDeclarableShakingMonths(state, playerKey);
    if (shakingMonths.length > 0 && shouldShaking(state, playerKey, shakingMonths, policy)) {
      return declareShaking(state, playerKey, selectBestMonth(state, shakingMonths));
    }
    const cardId = botChooseCard(state, playerKey, policy);
    if (!cardId) return state;
    return playTurn(state, cardId);
  }

  return state;
}

function normalizePolicy(policy) {
  const raw = String(policy || POLICY_HEURISTIC_V3).trim().toLowerCase();
  if (raw === "heuristic_v3") {
    return POLICY_HEURISTIC_V3;
  }
  return POLICY_HEURISTIC_V3;
}

function isHeuristicPolicy(policy) {
  return normalizePolicy(policy) === POLICY_HEURISTIC_V3;
}

function capturedCountByCategory(player, category) {
  if (!player?.captured) return 0;
  return (player.captured[category] || []).length;
}

function currentScoreTotal(state, playerKey) {
  const opp = playerKey === "human" ? "ai" : "human";
  const selfScore = calculateScore(state.players[playerKey], state.players[opp], state.ruleKey);
  return Number(selfScore?.total || 0);
}

function comboProgress(player) {
  const ribbons = player?.captured?.ribbon || [];
  const fives = player?.captured?.five || [];
  const ribbonMonths = new Set(ribbons.map((c) => c?.month).filter((m) => Number.isInteger(m)));
  const fiveMonths = new Set(fives.map((c) => c?.month).filter((m) => Number.isInteger(m)));
  const countSet = (months, src) => months.reduce((n, m) => n + (src.has(m) ? 1 : 0), 0);
  return {
    hongdan: countSet([1, 2, 3], ribbonMonths),
    cheongdan: countSet([6, 9, 10], ribbonMonths),
    chodan: countSet([4, 5, 7], ribbonMonths),
    godori: countSet([2, 4, 8], fiveMonths)
  };
}

function analyzeGameContext(state, playerKey) {
  const opp = playerKey === "human" ? "ai" : "human";
  const myScore = currentScoreTotal(state, playerKey);
  const oppScore = currentScoreTotal(state, opp);
  const deckCount = state.deck?.length || 0;
  const oppPi = capturedCountByCategory(state.players?.[opp], "junk");
  let mode = "BALANCED";
  if (oppScore >= 5 && myScore <= 2) mode = "DESPERATE_DEFENSE";
  else if (myScore >= 7 && oppScore <= 3 && deckCount >= 8) mode = "AGGRESSIVE";
  else if (deckCount <= 8) mode = "ENDGAME";
  let blockWeight = 1.0;
  let piWeight = 1.0;
  let pukPenalty = 1.0;
  if (mode === "DESPERATE_DEFENSE") {
    blockWeight = 1.45;
    piWeight = 1.2;
    pukPenalty = 1.35;
  } else if (mode === "AGGRESSIVE") {
    blockWeight = 0.95;
    piWeight = 1.15;
    pukPenalty = 0.9;
  } else if (mode === "ENDGAME") {
    blockWeight = 1.25;
    piWeight = oppPi <= 5 ? 1.4 : 1.2;
    pukPenalty = 1.25;
  }
  return { mode, myScore, oppScore, deckCount, oppPi, blockWeight, piWeight, pukPenalty };
}

function blockingMonthsAgainst(player) {
  const p = comboProgress(player);
  const out = new Set();
  if (p.hongdan >= 2) [1, 2, 3].forEach((m) => out.add(m));
  if (p.cheongdan >= 2) [6, 9, 10].forEach((m) => out.add(m));
  if (p.chodan >= 2) [4, 5, 7].forEach((m) => out.add(m));
  if (p.godori >= 2) [2, 4, 8].forEach((m) => out.add(m));
  return out;
}

function blockingUrgencyByMonth(player) {
  const p = comboProgress(player);
  const urg = new Map();
  const put = (months, level) => {
    for (const m of months) urg.set(m, Math.max(urg.get(m) || 0, level));
  };
  if (p.hongdan >= 2) put([1, 2, 3], p.hongdan >= 3 ? 3 : 2);
  if (p.cheongdan >= 2) put([6, 9, 10], p.cheongdan >= 3 ? 3 : 2);
  if (p.chodan >= 2) put([4, 5, 7], p.chodan >= 3 ? 3 : 2);
  if (p.godori >= 2) put([2, 4, 8], p.godori >= 3 ? 3 : 2);
  return urg;
}

function checkOpponentJokboProgress(state, playerKey) {
  const opp = playerKey === "human" ? "ai" : "human";
  const oppPlayer = state.players?.[opp];
  const boardMonths = new Set((state.board || []).map((c) => c.month));
  const p = comboProgress(oppPlayer);
  const rules = [
    { key: "hongdan", months: [1, 2, 3], got: p.hongdan },
    { key: "cheongdan", months: [6, 9, 10], got: p.cheongdan },
    { key: "chodan", months: [4, 5, 7], got: p.chodan },
    { key: "godori", months: [2, 4, 8], got: p.godori }
  ];
  const monthUrgency = new Map();
  let threat = 0;
  for (const r of rules) {
    const missing = r.months.filter((m) => {
      if (r.key === "godori") {
        const fives = new Set((oppPlayer?.captured?.five || []).map((c) => c?.month));
        return !fives.has(m);
      }
      const ribbons = new Set((oppPlayer?.captured?.ribbon || []).map((c) => c?.month));
      return !ribbons.has(m);
    });
    const near = r.got >= 2;
    const canCompleteSoon = near && missing.some((m) => boardMonths.has(m));
    if (near) {
      threat += canCompleteSoon ? 0.28 : 0.18;
      for (const m of missing) {
        const base = canCompleteSoon ? 30 : 20;
        monthUrgency.set(m, Math.max(monthUrgency.get(m) || 0, base));
      }
    } else if (r.got === 1) {
      threat += 0.05;
    }
  }
  return {
    threat: Math.max(0, Math.min(1, threat)),
    monthUrgency
  };
}

function monthCounts(cards) {
  const m = new Map();
  for (const c of cards || []) {
    if (!c || c.month == null) continue;
    m.set(c.month, (m.get(c.month) || 0) + 1);
  }
  return m;
}

function capturedMonthCounts(state) {
  const m = new Map();
  const pushCards = (cards) => {
    for (const c of cards || []) {
      if (!c || c.month == null) continue;
      m.set(c.month, (m.get(c.month) || 0) + 1);
    }
  };
  for (const key of ["human", "ai"]) {
    const cap = state.players?.[key]?.captured || {};
    pushCards(cap.kwang);
    pushCards(cap.five);
    pushCards(cap.ribbon);
    pushCards(cap.junk);
  }
  return m;
}

function monthStrategicPriority(month) {
  // Limited tie-break priority only for uncertain choices.
  if (month === 8 || month === 3 || month === 12) return 2.8; // S
  if (month === 2 || month === 10 || month === 11) return 2.0; // A
  if (month === 1 || month === 6 || month === 9) return 1.35; // B
  if (month === 4 || month === 5 || month === 7) return 0.9; // C
  return 0.0;
}

function isRiskOfPuk(state, playerKey, card, boardCountByMonth, handCountByMonth) {
  if (!card) return 0;
  const deckCount = state.deck?.length || 0;
  const lateGame = deckCount <= 10;
  const boardCnt = boardCountByMonth.get(card.month) || 0;
  const handCnt = handCountByMonth.get(card.month) || 0;
  const hasMatch = boardCnt > 0;
  if (hasMatch) return 0;
  if (handCnt >= 3) return -0.8; // strategic self-ppuk opportunity
  if (!lateGame) return 0.35;
  if (handCnt <= 1) return 1.0;
  return 0.6;
}

function isFirstTurnForActor(state, playerKey) {
  return (state.turnSeq || 0) === 0 && state.currentTurn === playerKey;
}

function getFirstTurnDoublePiPlan(state, playerKey) {
  if (!isFirstTurnForActor(state, playerKey)) return { active: false, months: new Set() };
  const player = state.players?.[playerKey];
  if (!player?.hand?.length) return { active: false, months: new Set() };
  const byMonth = boardMatchesByMonth(state);
  let hasDoublePiLine = false;
  let hasCompetingHighValue = false;
  const months = new Set();
  for (const card of player.hand) {
    const matches = byMonth.get(card.month) || [];
    if (!matches.length) continue;
    const hasDoublePi = card.doubleJunk || card.tripleJunk || matches.some((m) => m.doubleJunk || m.tripleJunk);
    if (hasDoublePi) {
      hasDoublePiLine = true;
      months.add(card.month);
    }
    const hasHigh = card.category === "kwang" || card.category === "five" || matches.some((m) => m.category === "kwang" || m.category === "five");
    if (hasHigh) hasCompetingHighValue = true;
  }
  return {
    active: hasDoublePiLine && !hasCompetingHighValue,
    months
  };
}

function matchableMonthCountForPlayer(state, playerKey) {
  const handMonths = new Set((state.players?.[playerKey]?.hand || []).map((c) => c.month));
  const boardMonths = new Set((state.board || []).map((c) => c.month));
  let n = 0;
  for (const m of handMonths) if (boardMonths.has(m)) n += 1;
  return n;
}

function nextTurnThreatScore(state, defenderKey) {
  const attacker = defenderKey === "human" ? "ai" : "human";
  const attackerHand = state.players?.[attacker]?.hand || [];
  const board = state.board || [];
  const boardByMonth = new Map();
  for (const c of board) {
    const arr = boardByMonth.get(c.month) || [];
    arr.push(c);
    boardByMonth.set(c.month, arr);
  }
  const attackerCombos = comboProgress(state.players?.[attacker]);
  let score = 0;
  for (const h of attackerHand) {
    const matches = boardByMonth.get(h.month) || [];
    if (!matches.length) continue;
    let local = 0;
    if (h.category === "kwang" || matches.some((m) => m.category === "kwang")) local += 0.32;
    if (h.category === "five" || matches.some((m) => m.category === "five")) local += 0.22;
    const hasJunk = h.category === "junk" || matches.some((m) => m.category === "junk");
    if (hasJunk) local += 0.1;
    if (attackerCombos.godori >= 2 && [2, 4, 8].includes(h.month)) local += 0.28;
    if (attackerCombos.hongdan >= 2 && [1, 2, 3].includes(h.month)) local += 0.22;
    if (attackerCombos.cheongdan >= 2 && [6, 9, 10].includes(h.month)) local += 0.22;
    if (attackerCombos.chodan >= 2 && [4, 5, 7].includes(h.month)) local += 0.22;
    score += local;
  }
  return Math.max(0, Math.min(1, score));
}

function opponentThreatScore(state, playerKey) {
  const opp = playerKey === "human" ? "ai" : "human";
  const oppPlayer = state.players?.[opp];
  const oppScore = currentScoreTotal(state, opp);
  const oppPi = capturedCountByCategory(oppPlayer, "junk");
  const oppGwang = capturedCountByCategory(oppPlayer, "kwang");
  const p = comboProgress(oppPlayer);
  const nextThreat = nextTurnThreatScore(state, playerKey);
  let score = 0;
  score += Math.min(1.0, oppScore / 7.0) * 0.55;
  score += Math.min(1.0, oppPi / 10.0) * 0.15;
  score += Math.min(1.0, oppGwang / 3.0) * 0.1;
  score += (p.hongdan >= 2 || p.cheongdan >= 2 || p.chodan >= 2) ? 0.12 : 0;
  score += p.godori >= 2 ? 0.12 : 0;
  score += nextThreat * 0.28;
  return Math.max(0, Math.min(1, score));
}

function boardHighValueThreatForPlayer(state, playerKey) {
  const handMonths = new Set((state.players?.[playerKey]?.hand || []).map((c) => c.month));
  const blockMonths = blockingMonthsAgainst(state.players?.[playerKey]);
  for (const c of state.board || []) {
    if (!handMonths.has(c.month)) continue;
    if (c.category === "kwang" || c.category === "five") return true;
    if (blockMonths.has(c.month)) return true;
  }
  return false;
}

function isOppVulnerableForBigGo(state, playerKey) {
  const opp = playerKey === "human" ? "ai" : "human";
  const oppPlayer = state.players?.[opp];
  const oppPi = capturedCountByCategory(oppPlayer, "junk");
  const oppGwang = capturedCountByCategory(oppPlayer, "kwang");
  return oppPi <= 5 || oppGwang === 0;
}

function cardCaptureValue(card) {
  if (!card) return 0;
  if (card.category === "kwang") return 6;
  if (card.category === "five") return 4;
  if (card.category === "ribbon") return 2;
  if (card.category === "junk") {
    if (card.tripleJunk) return 3;
    if (card.doubleJunk) return 2;
    return 1;
  }
  if (card.bonus?.stealPi) return 3 + Number(card.bonus.stealPi || 0);
  return 0;
}

function boardMatchesByMonth(state) {
  const map = new Map();
  for (const card of state.board || []) {
    const month = card?.month;
    if (month == null) continue;
    const list = map.get(month) || [];
    list.push(card);
    map.set(month, list);
  }
  return map;
}

function rankHandCards(state, playerKey) {
  const player = state.players?.[playerKey];
  if (!player?.hand?.length) return [];
  const opp = playerKey === "human" ? "ai" : "human";
  const oppPlayer = state.players?.[opp];
  const ctx = analyzeGameContext(state, playerKey);
  const jokboThreat = checkOpponentJokboProgress(state, playerKey);
  const blockMonths = blockingMonthsAgainst(oppPlayer);
  const blockUrgency = blockingUrgencyByMonth(oppPlayer);
  const oppPi = capturedCountByCategory(oppPlayer, "junk");
  const nextThreat = nextTurnThreatScore(state, playerKey);
  const selfPi = capturedCountByCategory(player, "junk");
  const deckCount = state.deck?.length || 0;
  const lateGame = deckCount <= 10;
  const firstTurnPiPlan = getFirstTurnDoublePiPlan(state, playerKey);
  const boardCountByMonth = monthCounts(state.board || []);
  const handCountByMonth = monthCounts(player.hand || []);
  const capturedByMonth = capturedMonthCounts(state);
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

    // First-turn limited tactic: if only clear value is double-pi, prioritize it strongly.
    if (firstTurnPiPlan.active && firstTurnPiPlan.months.has(card.month)) {
      score += 8.0;
    }

    // Priority #1: pi control (finish own pi quickly / force opp pi-bak).
    if (selfPi >= 7 && selfPi <= 8) {
      const piGain = matches.reduce((n, m) => n + (m.category === "junk" ? 1 : 0), 0);
      if (card.category === "junk") score += 3.0 * ctx.piWeight;
      score += piGain * (3.0 * ctx.piWeight);
    }
    if (oppPi <= 5) {
      const piGain = matches.reduce((n, m) => n + (m.category === "junk" ? 1 : 0), 0);
      if (card.category === "junk") score += 1.5 * ctx.piWeight;
      score += piGain * (2.8 * ctx.piWeight);
      if (card.doubleJunk) score += 1.2;
      if (card.tripleJunk) score += 1.8;
      if (matches.length === 0 && card.category === "junk") score -= 2.0; // keep pi pressure resources
    }
    // Dynamic pi finish pressure: 9->10 is critical.
    if (selfPi >= 9) {
      const piGain = matches.reduce((n, m) => n + (m.category === "junk" ? 1 : 0), 0);
      score += piGain * 4.5;
      if (card.category === "junk") score += 2.0;
    }

    // Priority #4: ppuk risk management (late game unmatched months are dangerous).
    const pukRisk = isRiskOfPuk(state, playerKey, card, boardCountByMonth, handCountByMonth);
    if (pukRisk > 0) score -= pukRisk * (5.2 * ctx.pukPenalty);
    else if (pukRisk < 0) score += (-pukRisk) * 2.0;

    // Priority #3: hard blocking against opponent near-combo.
    if (blockMonths.has(card.month)) {
      const urgency = blockUrgency.get(card.month) || 2;
      const dynamicUrg = Math.max(urgency >= 3 ? 24.0 : 20.0, jokboThreat.monthUrgency.get(card.month) || 0);
      const blockBoost = dynamicUrg * ctx.blockWeight;
      if (matches.length > 0) score += blockBoost + nextThreat * 4.0 + jokboThreat.threat * 4.0;
      else score -= 10.0; // hold blocker when cannot safely capture
    }

    // Default deprioritization for months already "claimed" on board by captures.
    // Keep exceptions for tactical needs (combo/block/go-pressure) by checking current signals.
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

    // Fallback cleanup rule:
    // If no good match, prefer discarding one of duplicated hand months.
    // Stronger when two cards of that month are already captured (effectively dead month),
    // except blocking months; also avoid high ppuk-risk months in late game.
    const lowValueNoMatch = matches.length === 0 && score <= 1.5;
    if (lowValueNoMatch) {
      const dupInHand = (handCountByMonth.get(card.month) || 0) >= 2;
      const capCnt = capturedByMonth.get(card.month) || 0;
      const blockedMonth = blockMonths.has(card.month);
      const pukRisk = isRiskOfPuk(state, playerKey, card, boardCountByMonth, handCountByMonth);
      if (dupInHand && !blockedMonth && pukRisk <= 0.75) {
        score += 2.0;
        if (capCnt >= 2) score += 4.0;
      }
    }
    return { card, score, matches: matches.length, uncertainBoost: monthStrategicPriority(card.month) };
  });
  ranked.sort((a, b) => b.score - a.score);

  // Apply month-priority only when choices are effectively tied / uncertain.
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

function chooseMatchHeuristic(state, playerKey) {
  const ids = state.pendingMatch?.boardCardIds || [];
  if (!ids.length) return null;
  const opp = playerKey === "human" ? "ai" : "human";
  const blockMonths = blockingMonthsAgainst(state.players?.[opp]);
  const blockUrgency = blockingUrgencyByMonth(state.players?.[opp]);
  const jokboThreat = checkOpponentJokboProgress(state, playerKey);
  const ctx = analyzeGameContext(state, playerKey);
  const oppPi = capturedCountByCategory(state.players?.[opp], "junk");
  const selfPi = capturedCountByCategory(state.players?.[playerKey], "junk");
  const nextThreat = nextTurnThreatScore(state, playerKey);
  const candidates = (state.board || []).filter((c) => ids.includes(c.id));
  if (!candidates.length) return null;
  candidates.sort((a, b) => {
    let as = cardCaptureValue(a);
    let bs = cardCaptureValue(b);
    if (blockMonths.has(a.month)) {
      as += Math.max((blockUrgency.get(a.month) || 2) >= 3 ? 24 : 20, jokboThreat.monthUrgency.get(a.month) || 0) * ctx.blockWeight;
      as += nextThreat * 4.0 + jokboThreat.threat * 4.0;
    }
    if (blockMonths.has(b.month)) {
      bs += Math.max((blockUrgency.get(b.month) || 2) >= 3 ? 24 : 20, jokboThreat.monthUrgency.get(b.month) || 0) * ctx.blockWeight;
      bs += nextThreat * 4.0 + jokboThreat.threat * 4.0;
    }
    if (oppPi <= 5 && a.category === "junk") as += 3;
    if (oppPi <= 5 && b.category === "junk") bs += 3;
    if (selfPi >= 7 && selfPi <= 8 && a.category === "junk") as += 4;
    if (selfPi >= 7 && selfPi <= 8 && b.category === "junk") bs += 4;
    return bs - as;
  });
  return candidates[0].id;
}

function chooseGukjinHeuristic(state, playerKey, policy) {
  const p = normalizePolicy(policy);
  const fiveCount = state.players?.[playerKey]?.captured?.five?.length || 0;
  if (p === POLICY_HEURISTIC_V3) return fiveCount >= 3 ? "five" : "junk";
  return fiveCount >= 3 ? "five" : "junk";
}

function shouldPresidentStop(state, playerKey, policy) {
  if (isHeuristicPolicy(policy)) return false;
  const deckCount = state.deck?.length || 0;
  return deckCount <= 8 && Math.random() < 0.6;
}

function shouldGo(state, playerKey, policy) {
  const p = normalizePolicy(policy);
  const player = state.players?.[playerKey];
  const opp = playerKey === "human" ? "ai" : "human";
  const ctx = analyzeGameContext(state, playerKey);
  const jokboThreat = checkOpponentJokboProgress(state, playerKey);
  const goCount = player?.goCount || 0;
  const deckCount = state.deck?.length || 0;
  const myScore = ctx.myScore;
  const oppScore = ctx.oppScore;
  const oppThreat = boardHighValueThreatForPlayer(state, opp);
  const oppProgThreat = opponentThreatScore(state, playerKey);
  const oppNextMatchCount = matchableMonthCountForPlayer(state, opp);
  const oppNextTurnThreat = nextTurnThreatScore(state, playerKey);
  // Hard stop layer
  if (ctx.mode === "DESPERATE_DEFENSE") return false;
  if (myScore >= 7 && oppScore >= 5 && (oppThreat || oppProgThreat >= 0.55 || oppNextMatchCount > 0 || oppNextTurnThreat >= 0.38 || jokboThreat.threat >= 0.3)) {
    return false;
  }
  if (goCount >= 3 && !isOppVulnerableForBigGo(state, playerKey)) return false;
  if (p === POLICY_HEURISTIC_V3) {
    if (goCount >= 3) return false;
    // Expected gain layer
    const myGainPotential = Math.max(0, myScore - 6) * 0.12 + (10 - Math.min(10, ctx.deckCount)) * 0.02 + (capturedCountByCategory(state.players?.[playerKey], "junk") >= 9 ? 0.2 : 0);
    const oppGainPotential = oppProgThreat * 0.65 + oppNextTurnThreat * 0.55 + jokboThreat.threat * 0.45;
    if (oppGainPotential > myGainPotential + 0.08) return false;
    if (oppProgThreat >= 0.5 || oppNextTurnThreat >= 0.35) return false;
    // Bak layer
    if (capturedCountByCategory(state.players?.[opp], "junk") < 6) return true;
    if (oppNextMatchCount === 0 && oppScore <= 4 && deckCount >= 5) return true; // safe aggressive go
    return deckCount > 7;
  }
  return false;
}

function monthBoardGain(state, month) {
  const cards = (state.board || []).filter((c) => c.month === month);
  return cards.reduce((sum, c) => sum + cardCaptureValue(c), 0);
}

function selectBestMonth(state, months) {
  if (!months?.length) return null;
  let best = months[0];
  let bestScore = monthBoardGain(state, best);
  for (const m of months.slice(1)) {
    const score = monthBoardGain(state, m);
    if (score > bestScore) {
      best = m;
      bestScore = score;
    }
  }
  return best;
}

function shouldBomb(state, playerKey, bombMonths, policy) {
  const p = normalizePolicy(policy);
  const firstTurnPiPlan = getFirstTurnDoublePiPlan(state, playerKey);
  if (p === POLICY_HEURISTIC_V3 && firstTurnPiPlan.active) {
    const target = bombMonths.find((m) => firstTurnPiPlan.months.has(m));
    if (target != null) return true;
  }
  const bestGain = monthBoardGain(state, selectBestMonth(state, bombMonths));
  if (p === POLICY_HEURISTIC_V3) return bestGain >= 1;
  return bestGain >= 1;
}

function shouldShaking(state, playerKey, shakingMonths, policy) {
  const p = normalizePolicy(policy);
  const firstTurnPiPlan = getFirstTurnDoublePiPlan(state, playerKey);
  if (p === POLICY_HEURISTIC_V3 && firstTurnPiPlan.active) {
    const target = shakingMonths.find((m) => firstTurnPiPlan.months.has(m));
    if (target != null) return true;
  }
  const deckCount = state.deck?.length || 0;
  if (p === POLICY_HEURISTIC_V3) return deckCount > 8;
  return deckCount > 8;
}

function pickRandom(arr) {
  if (!arr || arr.length === 0) return null;
  return arr[Math.floor(Math.random() * arr.length)];
}
