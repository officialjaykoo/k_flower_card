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
import { COMBO_MONTHS, COMBO_MONTH_SETS, countComboTag, missingComboMonths } from "./engine/combos.js";

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
    if (shakingMonths.length > 0) {
      const shakeDecision = decideShaking(state, playerKey, shakingMonths, policy);
      if (shakeDecision.allow && shakeDecision.month != null) {
        return declareShaking(state, playerKey, shakeDecision.month);
      }
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
  if (category === "junk") {
    return (player.captured.junk || []).reduce((sum, card) => sum + junkPiValue(card), 0);
  }
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
  return {
    redRibbons: countComboTag(ribbons, "redRibbons"),
    blueRibbons: countComboTag(ribbons, "blueRibbons"),
    plainRibbons: countComboTag(ribbons, "plainRibbons"),
    fiveBirds: countComboTag(fives, "fiveBirds")
  };
}

function analyzeGameContext(state, playerKey) {
  const opp = playerKey === "human" ? "ai" : "human";
  const myScore = currentScoreTotal(state, playerKey);
  const oppScore = currentScoreTotal(state, opp);
  const deckCount = state.deck?.length || 0;
  const carryOverMultiplier = state.carryOverMultiplier || 1;
  const oppGoCount = state.players?.[opp]?.goCount || 0;
  const selfPi = capturedCountByCategory(state.players?.[playerKey], "junk");
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
  const survivalMode = carryOverMultiplier >= 2;
  const endgameSprint = deckCount <= 8;
  return {
    mode,
    myScore,
    oppScore,
    deckCount,
    selfPi,
    oppPi,
    oppGoCount,
    carryOverMultiplier,
    survivalMode,
    endgameSprint,
    blockWeight,
    piWeight,
    pukPenalty
  };
}

function blockingMonthsAgainst(player) {
  const p = comboProgress(player);
  const out = new Set();
  if (p.redRibbons >= 2) COMBO_MONTHS.redRibbons.forEach((m) => out.add(m));
  if (p.blueRibbons >= 2) COMBO_MONTHS.blueRibbons.forEach((m) => out.add(m));
  if (p.plainRibbons >= 2) COMBO_MONTHS.plainRibbons.forEach((m) => out.add(m));
  if (p.fiveBirds >= 2) COMBO_MONTHS.fiveBirds.forEach((m) => out.add(m));
  return out;
}

function blockingUrgencyByMonth(player) {
  const p = comboProgress(player);
  const urg = new Map();
  const put = (months, level) => {
    for (const m of months) urg.set(m, Math.max(urg.get(m) || 0, level));
  };
  if (p.redRibbons >= 2) put(COMBO_MONTHS.redRibbons, p.redRibbons >= 3 ? 3 : 2);
  if (p.blueRibbons >= 2) put(COMBO_MONTHS.blueRibbons, p.blueRibbons >= 3 ? 3 : 2);
  if (p.plainRibbons >= 2) put(COMBO_MONTHS.plainRibbons, p.plainRibbons >= 3 ? 3 : 2);
  if (p.fiveBirds >= 2) put(COMBO_MONTHS.fiveBirds, p.fiveBirds >= 3 ? 3 : 2);
  return urg;
}

function checkOpponentJokboProgress(state, playerKey) {
  const opp = playerKey === "human" ? "ai" : "human";
  const oppPlayer = state.players?.[opp];
  const boardMonths = new Set((state.board || []).map((c) => c.month));
  const p = comboProgress(oppPlayer);
  const rules = [
    { key: "redRibbons", months: COMBO_MONTHS.redRibbons, got: p.redRibbons },
    { key: "blueRibbons", months: COMBO_MONTHS.blueRibbons, got: p.blueRibbons },
    { key: "plainRibbons", months: COMBO_MONTHS.plainRibbons, got: p.plainRibbons },
    { key: "fiveBirds", months: COMBO_MONTHS.fiveBirds, got: p.fiveBirds }
  ];
  const monthUrgency = new Map();
  let threat = 0;
  for (const r of rules) {
    const sourceCards = r.key === "fiveBirds" ? oppPlayer?.captured?.five || [] : oppPlayer?.captured?.ribbon || [];
    const missing = missingComboMonths(sourceCards, r.key);
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

function clamp01(v) {
  return Math.max(0, Math.min(1, Number(v) || 0));
}

function buildDynamicWeights(state, playerKey, ctx) {
  const opp = playerKey === "human" ? "ai" : "human";
  const oppPlayer = state.players?.[opp];
  const selfPlayer = state.players?.[playerKey];
  const deckCount = ctx.deckCount || 0;
  const carry = ctx.carryOverMultiplier || 1;
  const selfPi = ctx.selfPi ?? capturedCountByCategory(selfPlayer, "junk");
  const oppPi = ctx.oppPi ?? capturedCountByCategory(oppPlayer, "junk");
  const oppGoCount = ctx.oppGoCount ?? (oppPlayer?.goCount || 0);
  const weights = {
    pi: 1.0,
    combo: 1.0,
    block: 1.0,
    risk: 1.0,
    hold: 1.0,
    safety: 1.0
  };

  if (deckCount <= 8) {
    weights.pi *= 1.28;
    weights.combo *= 0.86;
    weights.risk *= 1.15;
    weights.safety *= 1.12;
  }
  if (deckCount <= 5) {
    weights.pi *= 1.25;
    weights.combo *= 0.82;
    weights.block *= 1.1;
    weights.safety *= 1.22;
  }
  if (carry >= 2) {
    // Survival mode: defensive behavior dominates.
    weights.block *= 3.0;
    weights.risk *= 1.45;
    weights.hold *= 1.25;
    weights.safety *= 1.2;
    weights.combo *= 0.78;
  }
  if (oppGoCount > 0) {
    // Break opponent tempo before maximizing own gain.
    weights.block *= 1.55;
    weights.risk *= 1.2;
    weights.hold *= 1.15;
    weights.pi *= 1.12;
  }
  if (selfPi >= 8) weights.pi *= 1.22;
  if (oppPi <= 6) weights.pi *= 1.15;
  return weights;
}

function estimateReleasePunishProb(state, playerKey, month, jokboThreat, ctx) {
  const urgency = jokboThreat?.monthUrgency?.get(month) || 0;
  if (urgency <= 0) return 0;
  const boardCnt = (state.board || []).reduce((n, c) => n + (c?.month === month ? 1 : 0), 0);
  const deckCount = ctx?.deckCount || state.deck?.length || 0;
  const oppScore = ctx?.oppScore || 0;
  const carry = ctx?.carryOverMultiplier || state.carryOverMultiplier || 1;
  const oppGoCount = ctx?.oppGoCount || 0;
  let prob = urgency >= 30 ? 0.72 : 0.58;
  if (boardCnt > 0) prob += 0.08;
  if (deckCount <= 8) prob += 0.08;
  if (deckCount <= 5) prob += 0.06;
  if (oppGoCount > 0) prob += 0.08;
  if (carry >= 2) prob += 0.06;
  if (oppScore >= 6) prob += 0.07;
  return clamp01(prob);
}

function estimateDangerMonthRisk(state, playerKey, month, boardCountByMonth, handCountByMonth, capturedByMonth) {
  const opp = playerKey === "human" ? "ai" : "human";
  const deckCount = state.deck?.length || 0;
  const oppGoCount = state.players?.[opp]?.goCount || 0;
  const boardCnt = boardCountByMonth.get(month) || 0;
  const handCnt = handCountByMonth.get(month) || 0;
  const capturedCnt = capturedByMonth.get(month) || 0;
  const known = boardCnt + handCnt + capturedCnt;
  const unseen = Math.max(0, 4 - known);
  let risk = 0;
  if (unseen <= 1) risk += 0.6;
  else if (unseen === 2) risk += 0.36;
  else if (unseen === 3) risk += 0.16;
  if (boardCnt >= 1) risk += 0.08;
  if (deckCount <= 8) risk += 0.15;
  if (deckCount <= 5) risk += 0.12;
  if (oppGoCount > 0) risk += 0.1;
  if ((state.carryOverMultiplier || 1) >= 2) risk += 0.08;
  return Math.max(0, Math.min(1.25, risk));
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
    const hasDoublePi = junkPiValue(card) >= 2 || matches.some((m) => junkPiValue(m) >= 2);
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
    if (attackerCombos.fiveBirds >= 2 && COMBO_MONTH_SETS.fiveBirds.has(h.month)) local += 0.28;
    if (attackerCombos.redRibbons >= 2 && COMBO_MONTH_SETS.redRibbons.has(h.month)) local += 0.22;
    if (attackerCombos.blueRibbons >= 2 && COMBO_MONTH_SETS.blueRibbons.has(h.month)) local += 0.22;
    if (attackerCombos.plainRibbons >= 2 && COMBO_MONTH_SETS.plainRibbons.has(h.month)) local += 0.22;
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
  score += (p.redRibbons >= 2 || p.blueRibbons >= 2 || p.plainRibbons >= 2) ? 0.12 : 0;
  score += p.fiveBirds >= 2 ? 0.12 : 0;
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

function junkPiValue(card) {
  if (!card || card.category !== "junk") return 0;
  const explicit = Number(card.piValue);
  if (Number.isFinite(explicit) && explicit > 0) return explicit;
  return 1;
}

function cardCaptureValue(card) {
  if (!card) return 0;
  if (card.category === "kwang") return 6;
  if (card.category === "five") return 4;
  if (card.category === "ribbon") return 2;
  if (card.category === "junk") return junkPiValue(card);
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
  const dyn = buildDynamicWeights(state, playerKey, ctx);
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

    // First-turn limited tactic: if only clear value is double-pi, prioritize it strongly.
    if (firstTurnPiPlan.active && firstTurnPiPlan.months.has(card.month)) {
      score += 8.0;
    }

    // Priority #1: pi control (finish own pi quickly / force opp pi-bak).
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
      if (matches.length === 0 && card.category === "junk") score -= 2.0; // keep pi pressure resources
    }
    // Dynamic pi finish pressure: 9->10 is critical.
    if (selfPi >= 9) {
      const piGain = matches.reduce((n, m) => n + junkPiValue(m), 0);
      score += piGain * 4.5 * dyn.pi;
      if (card.category === "junk") score += 2.0 * dyn.pi;
    }

    // Priority #4: ppuk risk management (late game unmatched months are dangerous).
    const pukRisk = isRiskOfPuk(state, playerKey, card, boardCountByMonth, handCountByMonth);
    if (pukRisk > 0) score -= pukRisk * (5.2 * ctx.pukPenalty * dyn.risk);
    else if (pukRisk < 0) score += (-pukRisk) * 2.0;

    // Safety-first dispatcher: risky months are pushed down, especially for pure discard lines.
    const dangerRisk = estimateDangerMonthRisk(state, playerKey, card.month, boardCountByMonth, handCountByMonth, capturedByMonth);
    if (matches.length === 0) score -= dangerRisk * (3.3 * dyn.safety);
    else if (matches.length === 1 && deckCount <= 8) score -= dangerRisk * (0.75 * dyn.safety);

    // Priority #3: hard blocking against opponent near-combo.
    if (blockMonths.has(card.month)) {
      const urgency = blockUrgency.get(card.month) || 2;
      const dynamicUrg = Math.max(urgency >= 3 ? 24.0 : 20.0, jokboThreat.monthUrgency.get(card.month) || 0);
      const blockBoost = dynamicUrg * ctx.blockWeight * dyn.block;
      if (matches.length > 0) score += blockBoost + nextThreat * 4.0 + jokboThreat.threat * 4.0;
      else score -= 10.0; // hold blocker when cannot safely capture
    }

    // Jokbo interceptor: hard-holding for near-certain punish lines.
    const releasePunishProb = estimateReleasePunishProb(state, playerKey, card.month, jokboThreat, ctx);
    const hardHold = matches.length === 0 && releasePunishProb >= 0.8;
    if (hardHold) score -= 80 * dyn.hold;
    else if (matches.length === 0 && releasePunishProb >= 0.65) score -= 8.5 * dyn.hold;

    if (ctx.oppGoCount > 0 && matches.length === 0) score -= 2.2;

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
  const carry = state.carryOverMultiplier || 1;
  const selfPi = capturedCountByCategory(player, "junk");
  const oppPi = capturedCountByCategory(state.players?.[opp], "junk");
  const strongLead = myScore >= 10 && oppScore <= 4;
  // Hard stop layer
  if (ctx.mode === "DESPERATE_DEFENSE") return false;
  if (carry >= 2) {
    // Survival mode in carry-over rounds: stop unless edge is very clear.
    const lowRisk = oppProgThreat < 0.35 && oppNextTurnThreat < 0.25 && jokboThreat.threat < 0.2;
    const noImmediateCounter = oppNextMatchCount === 0 && deckCount >= 7;
    if (!(strongLead && lowRisk && noImmediateCounter && selfPi >= 9 && oppPi <= 7)) {
      return false;
    }
  }
  // Conservative default: avoid over-GO unless pi line is already stable.
  if (!strongLead && selfPi < 9 && oppPi >= 6) return false;
  if (!strongLead && deckCount <= 8 && selfPi < 10) return false;
  if (myScore >= 7 && oppScore >= 5 && (oppThreat || oppProgThreat >= 0.55 || oppNextMatchCount > 0 || oppNextTurnThreat >= 0.38 || jokboThreat.threat >= 0.3)) {
    return false;
  }
  if (goCount >= 3 && !isOppVulnerableForBigGo(state, playerKey)) return false;
  if (p === POLICY_HEURISTIC_V3) {
    if (goCount >= 3) return false;
    if (goCount >= 2 && !strongLead) return false;
    // Expected gain layer
    const myGainPotential = Math.max(0, myScore - 6) * 0.12 + (10 - Math.min(10, ctx.deckCount)) * 0.02 + (capturedCountByCategory(state.players?.[playerKey], "junk") >= 9 ? 0.2 : 0);
    const oppGainPotential = oppProgThreat * 0.65 + oppNextTurnThreat * 0.55 + jokboThreat.threat * 0.45;
    if (!strongLead && myGainPotential < oppGainPotential + 0.18) return false;
    if (oppProgThreat >= 0.45 || oppNextTurnThreat >= 0.3) return false;
    if (oppNextMatchCount > 0 && !strongLead) return false;
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

function countKnownMonthCards(state, month) {
  let count = 0;
  for (const c of state.board || []) if (c?.month === month) count += 1;
  for (const key of ["human", "ai"]) {
    const player = state.players?.[key];
    for (const c of player?.hand || []) if (c?.month === month) count += 1;
    const cap = player?.captured || {};
    for (const cat of ["kwang", "five", "ribbon", "junk"]) {
      for (const c of cap[cat] || []) if (c?.month === month) count += 1;
    }
  }
  return count;
}

function ownComboOpportunityScore(state, playerKey, month) {
  const player = state.players?.[playerKey];
  const p = comboProgress(player);
  let score = 0;
  if (COMBO_MONTH_SETS.redRibbons.has(month)) {
    if (p.redRibbons >= 2) score += 1.1;
    else if (p.redRibbons === 1) score += 0.25;
  }
  if (COMBO_MONTH_SETS.blueRibbons.has(month)) {
    if (p.blueRibbons >= 2) score += 1.1;
    else if (p.blueRibbons === 1) score += 0.25;
  }
  if (COMBO_MONTH_SETS.plainRibbons.has(month)) {
    if (p.plainRibbons >= 2) score += 1.0;
    else if (p.plainRibbons === 1) score += 0.2;
  }
  if (COMBO_MONTH_SETS.fiveBirds.has(month)) {
    if (p.fiveBirds >= 2) score += 1.25;
    else if (p.fiveBirds === 1) score += 0.3;
  }
  return score;
}

function shakingImmediateGainScore(state, playerKey, month) {
  const player = state.players?.[playerKey];
  const monthCards = (player?.hand || []).filter((c) => c.month === month);
  const deckCount = state.deck?.length || 0;
  const known = countKnownMonthCards(state, month);
  const unseen = Math.max(0, 4 - known);
  const hasHighCard = monthCards.some((c) => c.category === "kwang" || c.category === "five");
  const piPayload = monthCards.reduce((sum, c) => sum + junkPiValue(c), 0);
  const flipMatchChance = deckCount > 0 ? Math.min(1, unseen / deckCount) : 0;
  let score = flipMatchChance * (2.1 + piPayload * 0.35 + (hasHighCard ? 0.8 : 0));
  if (monthCards.some((c) => c.category === "junk" && junkPiValue(c) >= 2)) score += 0.25;
  if (unseen === 0) score -= 0.7;
  return score;
}

function shakingTempoPressureScore(state, playerKey, ctx, oppThreat) {
  const opp = playerKey === "human" ? "ai" : "human";
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

function shakingRiskPenalty(state, playerKey, ctx, oppThreat, jokboThreat, nextThreat) {
  const self = state.players?.[playerKey];
  let risk = oppThreat * 2.0 + jokboThreat * 1.3 + nextThreat * 1.2;
  if ((ctx.deckCount || 0) <= 8) risk += 0.6;
  if ((ctx.myScore || 0) >= 7 && (ctx.myScore || 0) >= (ctx.oppScore || 0)) risk += 0.7;
  if ((state.carryOverMultiplier || 1) >= 2) risk += 0.4;
  if ((self?.goCount || 0) > 0) risk += 0.25;
  return risk;
}

function slowPlayPenalty(immediateGain, comboGain, tempoPressure, ctx) {
  const lowPracticalGain = immediateGain < 0.75 && comboGain < 0.9;
  const noTempoNeed = tempoPressure < 1.2;
  const notBehind = (ctx.myScore || 0) >= (ctx.oppScore || 0);
  if (lowPracticalGain && noTempoNeed && notBehind) return 1.15;
  if (lowPracticalGain && noTempoNeed) return 0.65;
  return 0;
}

function decideShaking(state, playerKey, shakingMonths, policy) {
  if (!shakingMonths?.length) return { allow: false, month: null, score: -Infinity };
  const p = normalizePolicy(policy);
  if (p !== POLICY_HEURISTIC_V3) {
    const bestMonth = selectBestMonth(state, shakingMonths);
    const deckCount = state.deck?.length || 0;
    return { allow: deckCount > 8, month: bestMonth, score: deckCount - 8 };
  }

  const ctx = analyzeGameContext(state, playerKey);
  const firstTurnPiPlan = getFirstTurnDoublePiPlan(state, playerKey);
  const oppThreat = opponentThreatScore(state, playerKey);
  const jokboThreat = checkOpponentJokboProgress(state, playerKey).threat;
  const nextThreat = nextTurnThreatScore(state, playerKey);
  const tempoPressure = shakingTempoPressureScore(state, playerKey, ctx, oppThreat);
  const riskPenalty = shakingRiskPenalty(state, playerKey, ctx, oppThreat, jokboThreat, nextThreat);
  const trailingBy = Math.max(0, (ctx.oppScore || 0) - (ctx.myScore || 0));
  const selfPi = capturedCountByCategory(state.players?.[playerKey], "junk");
  const opp = playerKey === "human" ? "ai" : "human";
  const oppPi = capturedCountByCategory(state.players?.[opp], "junk");
  const aheadOrEven = (ctx.myScore || 0) >= (ctx.oppScore || 0);
  const piFinishWindow = selfPi >= 9;
  const piPressureWindow = selfPi >= 8 || oppPi <= 6;

  let best = { allow: false, month: null, score: -Infinity };
  for (const month of shakingMonths) {
    let immediateGain = shakingImmediateGainScore(state, playerKey, month);
    if (firstTurnPiPlan.active && firstTurnPiPlan.months.has(month)) immediateGain += 0.45;
    const comboGain = ownComboOpportunityScore(state, playerKey, month);
    const slowPenalty = slowPlayPenalty(immediateGain, comboGain, tempoPressure, ctx);
    const monthTieBreak = monthStrategicPriority(month) * 0.25;
    let score = immediateGain * 1.3 + comboGain * 1.15 + tempoPressure - riskPenalty - slowPenalty + monthTieBreak;
    if (trailingBy >= 3) score += 0.35;
    if (score > best.score) best = { allow: false, month, score };
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
  if (best.month != null && firstTurnPiPlan.active && firstTurnPiPlan.months.has(best.month)) threshold -= 0.15;

  // Keep shaking possible, but strongly suppress it while pi closing is the top priority.
  if (piFinishWindow && aheadOrEven && tempoPressure < 2.2 && best.score < threshold + 0.55) {
    return { ...best, allow: false };
  }
  if (piPressureWindow && aheadOrEven && oppThreat >= 0.65 && best.score < threshold + 0.35) {
    return { ...best, allow: false };
  }

  if (oppThreat >= 0.8 && (ctx.myScore || 0) >= (ctx.oppScore || 0) + 1 && tempoPressure < 1.6) {
    return { ...best, allow: false };
  }
  return { ...best, allow: best.score >= threshold };
}

function shouldShaking(state, playerKey, shakingMonths, policy) {
  return decideShaking(state, playerKey, shakingMonths, policy).allow;
}

function pickRandom(arr) {
  if (!arr || arr.length === 0) return null;
  return arr[Math.floor(Math.random() * arr.length)];
}
