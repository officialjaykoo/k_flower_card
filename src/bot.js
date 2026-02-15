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
  calculateScore,
  scoringFiveCards
} from "./gameEngine.js";
import { COMBO_MONTHS, COMBO_MONTH_SETS, countComboTag, missingComboMonths } from "./engine/combos.js";

const POLICY_HEURISTIC_V3 = "heuristic_v3";
const GWANG_MONTHS = Object.freeze([1, 3, 8, 11, 12]);
const COMBO_REQUIRED_CATEGORY = Object.freeze({
  redRibbons: "ribbon",
  blueRibbons: "ribbon",
  plainRibbons: "ribbon",
  fiveBirds: "five"
});

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

function scoringFiveCount(player) {
  return scoringFiveCards(player).length;
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

function hasCapturedCategoryMonth(player, category, month) {
  return (player?.captured?.[category] || []).some((c) => c?.month === month);
}

function availableMissingMonths(missingMonths, requiredCategory, defenderPlayer) {
  if (!Array.isArray(missingMonths) || !missingMonths.length) return [];
  if (!defenderPlayer) return missingMonths;
  return missingMonths.filter((month) => !hasCapturedCategoryMonth(defenderPlayer, requiredCategory, month));
}

function missingGwangMonths(player) {
  const own = new Set((player?.captured?.kwang || []).map((c) => c?.month).filter((m) => Number.isInteger(m)));
  return GWANG_MONTHS.filter((m) => !own.has(m));
}

function isSecondMover(state, playerKey) {
  const first = state?.startingTurnKey;
  if (first === "human" || first === "ai") return first !== playerKey;
  return false;
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
  const mong = computeMongRiskProfile(state, playerKey);
  const oppCombo = comboProgress(state.players?.[opp]);
  const oppGwang = capturedCountByCategory(state.players?.[opp], "kwang");
  const isSecond = isSecondMover(state, playerKey);
  const turnSeq = state.turnSeq || 0;
  const defenseOpening = isSecond && turnSeq <= 6;
  const trailing = myScore < oppScore;
  const opponentNearSeven = oppScore >= 6;
  const opponentNearCombo =
    oppCombo.redRibbons >= 2 ||
    oppCombo.blueRibbons >= 2 ||
    oppCombo.plainRibbons >= 2 ||
    oppCombo.fiveBirds >= 2 ||
    oppGwang >= 2;
  const nagariDelayMode = isSecond && trailing && deckCount >= 8 && deckCount <= 12 && opponentNearSeven;
  const endgameSafePitch = deckCount <= 8;
  const midgameBlockFocus = deckCount >= 8 && deckCount <= 12 && opponentNearCombo;

  let mode = "BALANCED";
  if (defenseOpening) mode = "DEFENSE_OPENING";
  else if (oppScore >= 5 && myScore <= 2) mode = "DESPERATE_DEFENSE";
  else if (myScore >= 7 && oppScore <= 3 && deckCount >= 8) mode = "AGGRESSIVE";
  else if (deckCount <= 8) mode = "ENDGAME";
  if (!defenseOpening && mong.stage === "CRITICAL" && myScore <= oppScore + 2) mode = "DESPERATE_DEFENSE";
  let blockWeight = 1.0;
  let piWeight = 1.0;
  let pukPenalty = 1.0;
  if (mode === "DEFENSE_OPENING") {
    blockWeight = 1.35;
    piWeight = 1.35;
    pukPenalty = 1.2;
  } else if (mode === "DESPERATE_DEFENSE") {
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
  if (mong.danger >= 0.7) {
    blockWeight *= 1.18;
    piWeight *= 1.08;
    pukPenalty *= 1.12;
  } else if (mong.danger >= 0.4) {
    blockWeight *= 1.08;
    pukPenalty *= 1.05;
  }
  const survivalMode = carryOverMultiplier >= 2;
  const endgameSprint = deckCount <= 8;
  return {
    mode,
    isSecond,
    turnSeq,
    defenseOpening,
    nagariDelayMode,
    endgameSafePitch,
    midgameBlockFocus,
    opponentNearCombo,
    opponentNearSeven,
    myScore,
    oppScore,
    deckCount,
    selfPi,
    oppPi,
    selfFive: mong.selfFive,
    oppFive: mong.oppFive,
    mongDanger: mong.danger,
    mongStage: mong.stage,
    oppGoCount,
    carryOverMultiplier,
    survivalMode,
    endgameSprint,
    blockWeight,
    piWeight,
    pukPenalty
  };
}

function blockingMonthsAgainst(player, defenderPlayer = null) {
  const p = comboProgress(player);
  const out = new Set();
  const addMissing = (tag, got, sourceCards) => {
    if (got < 2) return;
    const requiredCategory = COMBO_REQUIRED_CATEGORY[tag];
    const missing = missingComboMonths(sourceCards, tag);
    for (const m of availableMissingMonths(missing, requiredCategory, defenderPlayer)) out.add(m);
  };
  addMissing("redRibbons", p.redRibbons, player?.captured?.ribbon || []);
  addMissing("blueRibbons", p.blueRibbons, player?.captured?.ribbon || []);
  addMissing("plainRibbons", p.plainRibbons, player?.captured?.ribbon || []);
  addMissing("fiveBirds", p.fiveBirds, player?.captured?.five || []);

  const gwangCount = capturedCountByCategory(player, "kwang");
  if (gwangCount >= 2) {
    for (const m of availableMissingMonths(missingGwangMonths(player), "kwang", defenderPlayer)) out.add(m);
  }
  return out;
}

function blockingUrgencyByMonth(player, defenderPlayer = null) {
  const p = comboProgress(player);
  const urg = new Map();
  const put = (months, level) => {
    for (const m of months) urg.set(m, Math.max(urg.get(m) || 0, level));
  };
  const putMissing = (tag, got, sourceCards) => {
    if (got < 2) return;
    const requiredCategory = COMBO_REQUIRED_CATEGORY[tag];
    const missing = missingComboMonths(sourceCards, tag);
    put(availableMissingMonths(missing, requiredCategory, defenderPlayer), got >= 3 ? 3 : 2);
  };
  putMissing("redRibbons", p.redRibbons, player?.captured?.ribbon || []);
  putMissing("blueRibbons", p.blueRibbons, player?.captured?.ribbon || []);
  putMissing("plainRibbons", p.plainRibbons, player?.captured?.ribbon || []);
  putMissing("fiveBirds", p.fiveBirds, player?.captured?.five || []);

  const gwangCount = capturedCountByCategory(player, "kwang");
  if (gwangCount >= 2) {
    put(availableMissingMonths(missingGwangMonths(player), "kwang", defenderPlayer), gwangCount >= 3 ? 3 : 2);
  }
  return urg;
}

function checkOpponentJokboProgress(state, playerKey) {
  const opp = playerKey === "human" ? "ai" : "human";
  const oppPlayer = state.players?.[opp];
  const selfPlayer = state.players?.[playerKey];
  const boardMonths = new Set((state.board || []).map((c) => c.month));
  const p = comboProgress(oppPlayer);
  const rules = [
    { key: "redRibbons", got: p.redRibbons, requiredCategory: "ribbon" },
    { key: "blueRibbons", got: p.blueRibbons, requiredCategory: "ribbon" },
    { key: "plainRibbons", got: p.plainRibbons, requiredCategory: "ribbon" },
    { key: "fiveBirds", got: p.fiveBirds, requiredCategory: "five" }
  ];
  const monthUrgency = new Map();
  let threat = 0;
  for (const r of rules) {
    const sourceCards = r.key === "fiveBirds" ? oppPlayer?.captured?.five || [] : oppPlayer?.captured?.ribbon || [];
    const missing = availableMissingMonths(
      missingComboMonths(sourceCards, r.key),
      r.requiredCategory,
      selfPlayer
    );
    const near = r.got >= 2 && missing.length > 0;
    const canCompleteSoon = near && missing.some((m) => boardMonths.has(m));
    if (near) {
      threat += canCompleteSoon ? 0.28 : 0.18;
      for (const m of missing) {
        const base = canCompleteSoon ? 30 : 20;
        monthUrgency.set(m, Math.max(monthUrgency.get(m) || 0, base));
      }
    } else if (r.got === 1 && missing.length > 0) {
      threat += 0.05;
    }
  }

  const oppGwangCount = capturedCountByCategory(oppPlayer, "kwang");
  const missingGwang = availableMissingMonths(missingGwangMonths(oppPlayer), "kwang", selfPlayer);
  if (oppGwangCount >= 2 && missingGwang.length > 0) {
    const canCompleteSoon = missingGwang.some((m) => boardMonths.has(m));
    threat += canCompleteSoon
      ? oppGwangCount >= 3
        ? 0.24
        : 0.18
      : oppGwangCount >= 3
      ? 0.16
      : 0.12;
    for (const m of missingGwang) {
      const base = canCompleteSoon ? (oppGwangCount >= 3 ? 28 : 24) : oppGwangCount >= 3 ? 20 : 16;
      monthUrgency.set(m, Math.max(monthUrgency.get(m) || 0, base));
    }
  } else if (oppGwangCount === 1 && missingGwang.length > 0) {
    threat += 0.03;
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

function computeMongRiskProfile(state, playerKey) {
  const opp = playerKey === "human" ? "ai" : "human";
  const selfFive = scoringFiveCount(state.players?.[playerKey]);
  const oppFive = scoringFiveCount(state.players?.[opp]);
  const deckCount = state.deck?.length || 0;

  let baseDanger = 0;
  if (oppFive >= 7) baseDanger = 1.0;
  else if (oppFive === 6) baseDanger = 0.82;
  else if (oppFive === 5) baseDanger = 0.58;
  else if (oppFive === 4) baseDanger = 0.34;

  // Holding at least one five reduces mong-bak pressure, but does not erase it.
  const guardFactor = selfFive === 0 ? 1.0 : selfFive === 1 ? 0.55 : 0.3;
  let danger = baseDanger * guardFactor;
  if (baseDanger > 0 && deckCount <= 8) danger += 0.08;
  if (baseDanger > 0 && deckCount <= 5) danger += 0.06;

  let stage = "SAFE";
  if (selfFive === 0 && oppFive >= 6) stage = "CRITICAL";
  else if (oppFive >= 6) stage = "HIGH";
  else if (oppFive >= 5) stage = selfFive === 0 ? "ELEVATED" : "WATCH";
  else if (oppFive >= 4) stage = "WATCH";

  return {
    selfFive,
    oppFive,
    danger: clamp01(danger),
    stage
  };
}

function hasMonthCard(cards, month) {
  return (cards || []).some((c) => c?.month === month);
}

function hasMonthCategoryCard(cards, month, category) {
  return (cards || []).some((c) => c?.month === month && c?.category === category);
}

function estimateMonthCaptureChance(state, actorKey, month, requiredCategory) {
  const actor = state.players?.[actorKey];
  const hand = actor?.hand || [];
  const board = state.board || [];
  const deckCount = state.deck?.length || 0;
  const drawFactor = deckCount <= 5 ? 0.68 : deckCount <= 8 ? 0.82 : 1.0;
  const handHasAny = hasMonthCard(hand, month);
  const handHasRequired = hasMonthCategoryCard(hand, month, requiredCategory);
  const boardHasAny = hasMonthCard(board, month);
  const boardRequiredCount = board.filter((c) => c?.month === month && c?.category === requiredCategory).length;
  let chance = 0.12 * drawFactor;

  if (handHasRequired) chance = Math.max(chance, boardHasAny ? 0.64 : 0.5);
  if (boardRequiredCount > 0) {
    chance = Math.max(chance, handHasAny ? 0.78 : 0.42);
  }
  if (handHasAny) chance = Math.max(chance, 0.28);
  if (boardRequiredCount >= 2 && handHasAny) chance = Math.max(chance, 0.86);
  return Math.max(0, Math.min(0.92, chance));
}

function estimateJokboExpectedPotential(state, actorKey, blockerKey) {
  const actor = state.players?.[actorKey];
  const blocker = state.players?.[blockerKey];
  if (!actor) {
    return {
      total: 0,
      nearCompleteCount: 0,
      oneAwayCount: 0
    };
  }

  const p = comboProgress(actor);
  const ribbons = actor.captured?.ribbon || [];
  const fives = actor.captured?.five || [];
  let nearCompleteCount = 0;
  let oneAwayCount = 0;

  const comboPotential = (tag, got, sourceCards, requiredCategory, nearWeight, midWeight) => {
    const missing = availableMissingMonths(missingComboMonths(sourceCards, tag), requiredCategory, blocker);
    if (!missing.length) return 0;
    const probs = missing.map((m) => estimateMonthCaptureChance(state, actorKey, m, requiredCategory));
    const avg = probs.reduce((sum, v) => sum + v, 0) / probs.length;
    const top = Math.max(...probs);
    const baseWeight = got >= 2 ? nearWeight : got === 1 ? midWeight : 0.16;
    let score = baseWeight * (avg * 0.65 + top * 0.35);
    if (got >= 2) nearCompleteCount += 1;
    if (got >= 2 && missing.length === 1) {
      score += 0.18;
      oneAwayCount += 1;
    }
    return score;
  };

  const red = comboPotential("redRibbons", p.redRibbons, ribbons, "ribbon", 1.0, 0.38);
  const blue = comboPotential("blueRibbons", p.blueRibbons, ribbons, "ribbon", 1.0, 0.38);
  const plain = comboPotential("plainRibbons", p.plainRibbons, ribbons, "ribbon", 0.92, 0.35);
  const birds = comboPotential("fiveBirds", p.fiveBirds, fives, "five", 1.12, 0.42);

  const gwangCount = capturedCountByCategory(actor, "kwang");
  const gwMissing = availableMissingMonths(missingGwangMonths(actor), "kwang", blocker);
  let gwang = 0;
  if (gwMissing.length > 0) {
    const probs = gwMissing.map((m) => estimateMonthCaptureChance(state, actorKey, m, "kwang"));
    const avg = probs.reduce((sum, v) => sum + v, 0) / probs.length;
    const top = Math.max(...probs);
    const baseWeight = gwangCount >= 2 ? 1.2 : gwangCount === 1 ? 0.42 : 0.12;
    gwang = baseWeight * (avg * 0.6 + top * 0.4);
    if (gwangCount >= 2) nearCompleteCount += 1;
    if (gwangCount >= 2 && gwMissing.length === 1) {
      gwang += 0.2;
      oneAwayCount += 1;
    }
  }

  const total = Math.max(0, Math.min(3.5, red + blue + plain + birds + gwang));
  return {
    total,
    nearCompleteCount,
    oneAwayCount
  };
}

function buildDynamicWeights(state, playerKey, ctx) {
  const opp = playerKey === "human" ? "ai" : "human";
  const oppPlayer = state.players?.[opp];
  const selfPlayer = state.players?.[playerKey];
  const deckCount = ctx.deckCount || 0;
  const carry = ctx.carryOverMultiplier || 1;
  const selfPi = ctx.selfPi ?? capturedCountByCategory(selfPlayer, "junk");
  const oppPi = ctx.oppPi ?? capturedCountByCategory(oppPlayer, "junk");
  const selfFive = ctx.selfFive ?? scoringFiveCount(selfPlayer);
  const oppFive = ctx.oppFive ?? scoringFiveCount(oppPlayer);
  const mongDanger = ctx.mongDanger ?? 0;
  const oppGoCount = ctx.oppGoCount ?? (oppPlayer?.goCount || 0);
  const weights = {
    pi: 1.0,
    combo: 1.0,
    block: 1.0,
    risk: 1.0,
    hold: 1.0,
    safety: 1.0
  };

  if (ctx.defenseOpening) {
    // Requested second-player opening profile.
    weights.pi *= 1.35;
    weights.combo *= 0.75;
    weights.risk *= 1.2;
    weights.block *= 1.2;
    weights.hold *= 1.15;
    weights.safety *= 1.15;
  }

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
  if (mongDanger >= 0.7) {
    weights.safety *= 1.22;
    weights.risk *= 1.18;
    weights.hold *= 1.12;
    weights.combo *= 0.9;
  } else if (mongDanger >= 0.4) {
    weights.safety *= 1.1;
    weights.risk *= 1.08;
  }
  if (selfFive === 0 && oppFive >= 6) {
    weights.block *= 1.18;
    weights.hold *= 1.08;
  }
  if (ctx.nagariDelayMode) {
    weights.combo *= 0.78;
    weights.block *= 1.25;
    weights.safety *= 1.18;
    weights.hold *= 1.12;
  }
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

function estimateOpponentImmediateGainIfDiscard(state, playerKey, month) {
  const opp = playerKey === "human" ? "ai" : "human";
  const oppHand = state.players?.[opp]?.hand || [];
  const boardMonthCards = (state.board || []).filter((c) => c?.month === month);
  const oppMonthCards = oppHand.filter((c) => c?.month === month);
  if (!oppMonthCards.length) return 0;

  let best = 0;
  for (const c of oppMonthCards) {
    // If we discard this month now, opponent can capture discard + one board card.
    const target = boardMonthCards.length
      ? Math.max(...boardMonthCards.map((b) => cardCaptureValue(b)))
      : 0;
    let local = 0.45 + target * 0.38 + cardCaptureValue(c) * 0.18;
    if (c.category === "kwang" || c.category === "five") local += 0.35;
    best = Math.max(best, local);
  }
  return Math.max(0, Math.min(3.6, best));
}

function isHighImpactBomb(state, playerKey, month) {
  const player = state.players?.[playerKey];
  const monthHand = (player?.hand || []).filter((c) => c?.month === month);
  const monthBoard = (state.board || []).filter((c) => c?.month === month);
  const all = monthHand.concat(monthBoard);
  const hasDoublePiLine = all.some((c) => c?.category === "junk" && junkPiValue(c) >= 2);
  const selfGwang = capturedCountByCategory(player, "kwang");
  const monthHasGwang = all.some((c) => c?.category === "kwang");
  const directThreeGwang = selfGwang >= 2 && monthHasGwang;
  const immediateGain = all.reduce((sum, c) => sum + cardCaptureValue(c), 0);
  return {
    highImpact: hasDoublePiLine || directThreeGwang || immediateGain >= 8,
    hasDoublePiLine,
    directThreeGwang,
    immediateGain
  };
}

function isHighImpactShaking(state, playerKey, month) {
  const player = state.players?.[playerKey];
  const monthHand = (player?.hand || []).filter((c) => c?.month === month);
  const monthBoard = (state.board || []).filter((c) => c?.month === month);
  const all = monthHand.concat(monthBoard);
  const hasDoublePiLine = all.some((c) => c?.category === "junk" && junkPiValue(c) >= 2);
  const selfGwang = capturedCountByCategory(player, "kwang");
  const monthHasGwang = all.some((c) => c?.category === "kwang");
  const directThreeGwang = selfGwang >= 2 && monthHasGwang;
  return {
    highImpact: hasDoublePiLine || directThreeGwang,
    hasDoublePiLine,
    directThreeGwang
  };
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
  const blockerKey = playerKey === "human" ? "ai" : "human";
  const blockMonths = blockingMonthsAgainst(state.players?.[playerKey], state.players?.[blockerKey]);
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
  const blockMonths = blockingMonthsAgainst(oppPlayer, player);
  const blockUrgency = blockingUrgencyByMonth(oppPlayer, player);
  const oppPi = capturedCountByCategory(oppPlayer, "junk");
  const nextThreat = nextTurnThreatScore(state, playerKey);
  const selfPi = capturedCountByCategory(player, "junk");
  const deckCount = state.deck?.length || 0;
  const lateGame = deckCount <= 10;
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
      // Opening as second mover: prioritize pi/ssangpi safety over raw combo growth.
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
      // When trailing as second mover, prefer delaying opponent completion over own greedy lines.
      if (blockMonths.has(card.month) && matches.length > 0) score += 10.0 * dyn.block;
      if (!blockMonths.has(card.month) && matches.length === 0) score -= 3.6 * dyn.safety;
      if (oppFeedRisk > 0.9 && matches.length === 0) score -= 1.5 * dyn.safety;
    }

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

    if (endgameSafePitch && matches.length === 0) {
      if (knownMonth >= 3) score += 3.1 * dyn.safety;
      else if (knownMonth === 2) score += 0.9 * dyn.safety;
      else score -= 1.0 * dyn.safety;
      score -= oppFeedRisk * (1.55 * dyn.safety);
    }

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
  const blockMonths = blockingMonthsAgainst(state.players?.[opp], state.players?.[playerKey]);
  const blockUrgency = blockingUrgencyByMonth(state.players?.[opp], state.players?.[playerKey]);
  const jokboThreat = checkOpponentJokboProgress(state, playerKey);
  const ctx = analyzeGameContext(state, playerKey);
  const mongDanger = ctx.mongDanger || 0;
  const midgameBlockFocus = !!ctx.midgameBlockFocus;
  const defenseOpening = !!ctx.defenseOpening;
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
  const selfJokboEV = estimateJokboExpectedPotential(state, playerKey, opp);
  const oppJokboEV = estimateJokboExpectedPotential(state, opp, playerKey);
  const selfFive = ctx.selfFive || 0;
  const oppFive = ctx.oppFive || 0;
  const mongDanger = ctx.mongDanger || 0;
  const isSecond = !!ctx.isSecond;
  const strongLead = myScore >= 10 && oppScore <= 4;
  // Hard stop layer
  if (ctx.mode === "DESPERATE_DEFENSE") return false;
  if (ctx.nagariDelayMode) return false;
  // Absolute bak-defense rule requested: no aggressive GO before pi safety.
  if (selfPi < 9) return false;
  if (selfFive === 0 && oppFive >= 7) return false;
  if (selfFive === 0 && oppFive >= 6 && deckCount <= 10) return false;
  if (isSecond && !strongLead && oppJokboEV.oneAwayCount >= 1) return false;
  if (!strongLead && deckCount <= 10 && oppJokboEV.oneAwayCount >= 1 && selfJokboEV.oneAwayCount === 0) return false;
  if (mongDanger >= 0.75 && !strongLead) return false;
  if (mongDanger >= 0.6 && (oppThreat || oppProgThreat >= 0.45 || oppNextTurnThreat >= 0.3) && !strongLead) {
    return false;
  }
  if (carry >= 2) {
    // Survival mode in carry-over rounds: stop unless edge is very clear.
    const lowRisk =
      oppProgThreat < 0.35 &&
      oppNextTurnThreat < 0.25 &&
      jokboThreat.threat < 0.2 &&
      oppJokboEV.total < 0.6 &&
      mongDanger < 0.45 &&
      !(selfFive === 0 && oppFive >= 6);
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
    const myGainPotential =
      Math.max(0, myScore - 6) * 0.12 +
      (10 - Math.min(10, ctx.deckCount)) * 0.02 +
      (capturedCountByCategory(state.players?.[playerKey], "junk") >= 9 ? 0.2 : 0) +
      selfJokboEV.total * 0.34 +
      selfJokboEV.oneAwayCount * 0.12;
    const oppGainPotential =
      oppProgThreat * 0.65 +
      oppNextTurnThreat * 0.55 +
      jokboThreat.threat * 0.45 +
      mongDanger * 0.55 +
      oppJokboEV.total * 0.4 +
      oppJokboEV.oneAwayCount * 0.14;
    const goMargin = isSecond ? 0.28 : 0.12;
    if (!strongLead && myGainPotential < oppGainPotential + goMargin) return false;
    if (!strongLead && deckCount <= 8 && selfJokboEV.total < 0.45) return false;
    if (isSecond && !strongLead && (oppProgThreat >= 0.35 || oppNextMatchCount > 0)) return false;
    if (oppProgThreat >= 0.45 || oppNextTurnThreat >= 0.3) return false;
    if (selfFive === 0 && oppFive >= 6) return false;
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
  const ctx = analyzeGameContext(state, playerKey);
  const firstTurnPiPlan = getFirstTurnDoublePiPlan(state, playerKey);
  if (p === POLICY_HEURISTIC_V3 && firstTurnPiPlan.active) {
    const target = bombMonths.find((m) => firstTurnPiPlan.months.has(m));
    if (target != null) return true;
  }
  const bestMonth = selectBestMonth(state, bombMonths);
  const bestGain = monthBoardGain(state, bestMonth);
  if (p === POLICY_HEURISTIC_V3) {
    if (bestMonth == null) return false;
    const impact = isHighImpactBomb(state, playerKey, bestMonth);
    if (ctx.defenseOpening) {
      // Requested: opening-second mode allows bomb only on high immediate impact.
      return impact.highImpact;
    }
    if (ctx.nagariDelayMode && !impact.highImpact && impact.immediateGain < 6) return false;
    return bestGain >= 1;
  }
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

  let best = { allow: false, month: null, score: -Infinity, highImpact: false, immediateGain: 0, comboGain: 0 };
  for (const month of shakingMonths) {
    let immediateGain = shakingImmediateGainScore(state, playerKey, month);
    if (firstTurnPiPlan.active && firstTurnPiPlan.months.has(month)) immediateGain += 0.45;
    const comboGain = ownComboOpportunityScore(state, playerKey, month);
    const impact = isHighImpactShaking(state, playerKey, month);
    if (impact.directThreeGwang) immediateGain += 0.55;
    if (impact.hasDoublePiLine) immediateGain += 0.35;
    const slowPenalty = slowPlayPenalty(immediateGain, comboGain, tempoPressure, ctx);
    const monthTieBreak = monthStrategicPriority(month) * 0.25;
    let score = immediateGain * 1.3 + comboGain * 1.15 + tempoPressure - riskPenalty - slowPenalty + monthTieBreak;
    if (trailingBy >= 3) score += 0.35;
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
  if (ctx.defenseOpening && !best.highImpact) {
    return { ...best, allow: false };
  }
  if (ctx.defenseOpening && best.immediateGain < 1.05 && best.comboGain < 1.35) {
    return { ...best, allow: false };
  }
  if (ctx.nagariDelayMode && !best.highImpact && best.score < threshold + 0.35) {
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
