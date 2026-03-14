import { chooseStop } from "../engine/index.js";
import { missingComboMonths } from "../engine/combos.js";
import { canBankruptOpponentByStopGemini } from "../heuristics/heuristicGemini.js";
import {
  COMBO_REQUIRED_CATEGORY,
  GOLD_RISK_THRESHOLD_RATIO,
  availableMissingMonths,
  boardMatchesByMonth,
  capturedCountByCategory,
  capturedMonthCounts,
  cardCaptureValue,
  clamp01,
  comboProgress,
  countKnownMonthCards,
  currentScoreTotal,
  fiveCountIncludingCapturedGukjin,
  gatherGukjinZoneFlags,
  buildGukjinScenario,
  hasAnyGukjinFlag,
  hasMonthCard,
  hasMonthCategoryCard,
  isSecondMover,
  junkPiValue,
  missingGwangMonths,
  monthBoardGain,
  monthCounts,
  monthStrategicPriority,
  otherPlayerKey,
  ownComboOpportunityScore,
  pickGukjinScenario,
  preferredGukjinModeByFiveCount,
  resolveInitialGoldBase,
  scoringFiveCount,
  selectBestMonth,
  shakingImmediateGainScore,
  summarizeScenarioRange
} from "./heuristicUtils.js";

export function analyzeGukjinBranches(state, playerKey) {
  const zoneFlags = gatherGukjinZoneFlags(state, playerKey);
  const enabled = hasAnyGukjinFlag(zoneFlags);
  if (!enabled) {
    return {
      enabled: false,
      zoneFlags,
      scenarios: []
    };
  }

  const scenarios = [];
  for (const selfMode of ["five", "junk"]) {
    for (const oppMode of ["five", "junk"]) {
      const s = buildGukjinScenario(state, playerKey, selfMode, oppMode, zoneFlags);
      if (s) scenarios.push(s);
    }
  }
  if (!scenarios.length) {
    return {
      enabled: false,
      zoneFlags,
      scenarios: []
    };
  }

  return {
    enabled: true,
    zoneFlags,
    scenarios,
    selfPi: summarizeScenarioRange(scenarios, "selfPi"),
    selfFive: summarizeScenarioRange(scenarios, "selfFive"),
    oppPi: summarizeScenarioRange(scenarios, "oppPi"),
    oppFive: summarizeScenarioRange(scenarios, "oppFive"),
    myScore: summarizeScenarioRange(scenarios, "myScore"),
    oppScore: summarizeScenarioRange(scenarios, "oppScore"),
    mongRiskAny: scenarios.some((s) => s.mongRiskSelf),
    mongBakAny: scenarios.some((s) => s.canMongBakSelf)
  };
}

export function goldRiskProfile(state, playerKey) {
  const opp = playerKey === "human" ? "ai" : "human";
  const initialGold = resolveInitialGoldBase(state);
  const threshold = initialGold * GOLD_RISK_THRESHOLD_RATIO;
  const selfGold = Number(state?.players?.[playerKey]?.gold || 0);
  const oppGold = Number(state?.players?.[opp]?.gold || 0);
  return {
    initialGold,
    threshold,
    selfGold,
    oppGold,
    selfLow: selfGold <= threshold,
    oppLow: oppGold <= threshold
  };
}

export function canBankruptOpponentByStop(state, playerKey) {
  if (state.phase !== "go-stop" || state.pendingGoStop !== playerKey) return false;
  const opp = playerKey === "human" ? "ai" : "human";
  const stopped = chooseStop(state, playerKey);
  const oppGoldAfterStop = Number(stopped?.players?.[opp]?.gold || 0);
  return oppGoldAfterStop <= 0;
}

export function canBankruptOpponentByStopGeminiProxy(state, playerKey) {
  const scoreHint = Number(analyzeGameContext(state, playerKey)?.myScore || 0);
  return canBankruptOpponentByStopGemini(state, playerKey, {
    currentScore: scoreHint,
    fallbackExact: canBankruptOpponentByStop,
    defaultBetAmount: 100,
    safetyMargin: 0.9
  });
}

export function computeMongRiskProfile(state, playerKey) {
  const opp = otherPlayerKey(playerKey);
  const selfPlayer = state.players?.[playerKey];
  const oppPlayer = state.players?.[opp];
  const selfFive = scoringFiveCount(selfPlayer);
  const oppFive = scoringFiveCount(oppPlayer);
  const danger =
    selfFive <= 0 && oppFive >= 6 ? 0.9 :
    selfFive <= 0 && oppFive >= 5 ? 0.6 :
    selfFive <= 1 && oppFive >= 6 ? 0.45 :
    0.0;
  return {
    selfFive,
    oppFive,
    danger,
    stage: danger >= 0.85 ? "CRITICAL" : danger >= 0.4 ? "ELEVATED" : "SAFE"
  };
}

export function analyzeGameContext(state, playerKey) {
  const opp = otherPlayerKey(playerKey);
  let myScore = currentScoreTotal(state, playerKey);
  let oppScore = currentScoreTotal(state, opp);
  const deckCount = state.deck?.length || 0;
  const carryOverMultiplier = state.carryOverMultiplier || 1;
  const oppGoCount = state.players?.[opp]?.goCount || 0;
  const selfPlayer = state.players?.[playerKey];
  const oppPlayer = state.players?.[opp];
  const selfCapturedFiveRaw = fiveCountIncludingCapturedGukjin(selfPlayer);
  const oppCapturedFiveRaw = fiveCountIncludingCapturedGukjin(oppPlayer);
  let selfPi = capturedCountByCategory(state.players?.[playerKey], "junk");
  let oppPi = capturedCountByCategory(state.players?.[opp], "junk");
  let mong = computeMongRiskProfile(state, playerKey);
  let selfFive = mong.selfFive;
  let oppFive = mong.oppFive;
  const gukjinBranch = analyzeGukjinBranches(state, playerKey);
  if (gukjinBranch.enabled) {
    const preferredSelfMode = preferredGukjinModeByFiveCount(selfCapturedFiveRaw);
    const preferredOppMode = preferredGukjinModeByFiveCount(oppCapturedFiveRaw);
    const preferredScenario = pickGukjinScenario(gukjinBranch.scenarios, preferredSelfMode, preferredOppMode);

    if (preferredScenario) {
      myScore = Number(preferredScenario.myScore || 0);
      oppScore = Number(preferredScenario.oppScore || 0);
      selfPi = Number(preferredScenario.selfPi || 0);
      oppPi = Number(preferredScenario.oppPi || 0);
      selfFive = Number(preferredScenario.selfFive || 0);
      oppFive = Number(preferredScenario.oppFive || 0);
    } else {
      myScore = gukjinBranch.myScore.avg;
      oppScore = gukjinBranch.oppScore.avg;
      selfPi = gukjinBranch.selfPi.avg;
      oppPi = gukjinBranch.oppPi.avg;
      selfFive = gukjinBranch.selfFive.min;
      oppFive = gukjinBranch.oppFive.max;
    }
    selfFive = Math.min(selfFive, gukjinBranch.selfFive.min);
    oppFive = Math.max(oppFive, gukjinBranch.oppFive.max);
    const criticalMongRisk = selfFive <= 0 && oppFive >= 6;
    mong = {
      ...mong,
      selfFive,
      oppFive,
      danger: Math.max(mong.danger, criticalMongRisk ? 0.85 : 0),
      stage: criticalMongRisk ? "CRITICAL" : selfFive <= 0 && oppFive >= 5 ? "ELEVATED" : mong.stage
    };
  } else {
    mong = { ...mong, selfFive, oppFive };
  }
  const oppCombo = comboProgress(state.players?.[opp]);
  const oppGwang = capturedCountByCategory(state.players?.[opp], "kwang");
  const isSecond = isSecondMover(state, playerKey);
  const turnSeq = state.turnSeq || 0;
  const defenseOpening = isSecond && turnSeq <= 6;
  const scoreDiff = myScore - oppScore;
  const trailing = scoreDiff < 0;
  const volatilityComeback = isSecond && scoreDiff <= -5;
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
  if (!defenseOpening && (mong.stage === "CRITICAL" || gukjinBranch.mongRiskAny) && myScore <= oppScore + 2) {
    mode = "DESPERATE_DEFENSE";
  }
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
    scoreDiff,
    deckCount,
    selfPi,
    oppPi,
    selfFive,
    oppFive,
    mongDanger: mong.danger,
    mongStage: mong.stage,
    gukjinBranch,
    oppGoCount,
    carryOverMultiplier,
    volatilityComeback,
    survivalMode,
    endgameSprint,
    blockWeight,
    piWeight,
    pukPenalty
  };
}

export function blockingMonthsAgainst(player, defenderPlayer = null) {
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

export function blockingUrgencyByMonth(player, defenderPlayer = null) {
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

export function checkOpponentJokboProgress(state, playerKey) {
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
      for (const m of missing) monthUrgency.set(m, Math.max(monthUrgency.get(m) || 0, canCompleteSoon ? 30 : 20));
    } else if (r.got === 1 && missing.length > 0) {
      threat += 0.05;
    }
  }

  const oppGwangCount = capturedCountByCategory(oppPlayer, "kwang");
  const missingGwang = availableMissingMonths(missingGwangMonths(oppPlayer), "kwang", selfPlayer);
  if (oppGwangCount >= 2 && missingGwang.length > 0) {
    const canCompleteSoon = missingGwang.some((m) => boardMonths.has(m));
    threat += canCompleteSoon ? (oppGwangCount >= 3 ? 0.24 : 0.18) : (oppGwangCount >= 3 ? 0.16 : 0.12);
    for (const m of missingGwang) {
      const base = canCompleteSoon ? (oppGwangCount >= 3 ? 28 : 24) : oppGwangCount >= 3 ? 20 : 16;
      monthUrgency.set(m, Math.max(monthUrgency.get(m) || 0, base));
    }
  } else if (oppGwangCount === 1 && missingGwang.length > 0) {
    threat += 0.03;
  }

  return { threat: Math.max(0, Math.min(1, threat)), monthUrgency };
}

export function getMissingComboCards(state, playerKey) {
  const player = state.players?.[playerKey];
  if (!player) return { ids: [], months: [], imminent: false };
  const blockerKey = otherPlayerKey(playerKey);
  const blocker = state.players?.[blockerKey];
  const board = state.board || [];
  const progress = comboProgress(player);
  const ids = new Set();
  const months = new Set();
  let imminent = false;

  const addRule = (tag, got, sourceCards, requiredCategory) => {
    if (got < 2) return;
    const missing = availableMissingMonths(missingComboMonths(sourceCards, tag), requiredCategory, blocker);
    if (!missing.length) return;
    if (got >= 3) imminent = true;
    for (const month of missing) {
      months.add(month);
      for (const card of board) {
        if (card?.month !== month) continue;
        if (requiredCategory && card?.category !== requiredCategory) continue;
        ids.add(card.id);
      }
    }
  };

  addRule("redRibbons", progress.redRibbons, player?.captured?.ribbon || [], "ribbon");
  addRule("blueRibbons", progress.blueRibbons, player?.captured?.ribbon || [], "ribbon");
  addRule("plainRibbons", progress.plainRibbons, player?.captured?.ribbon || [], "ribbon");
  addRule("fiveBirds", progress.fiveBirds, player?.captured?.five || [], "five");

  const gwangCount = capturedCountByCategory(player, "kwang");
  if (gwangCount >= 2) {
    const missingGwang = availableMissingMonths(missingGwangMonths(player), "kwang", blocker);
    if (gwangCount >= 3 && missingGwang.length > 0) imminent = true;
    for (const month of missingGwang) {
      months.add(month);
      for (const card of board) {
        if (card?.month === month && card?.category === "kwang") ids.add(card.id);
      }
    }
  }

  return { ids: [...ids], months: [...months], imminent };
}

export function estimateMonthCaptureChance(state, actorKey, month, requiredCategory) {
  const hand = state.players?.[actorKey]?.hand || [];
  const board = state.board || [];
  const deckCount = state.deck?.length || 0;
  const drawFactor = deckCount <= 5 ? 0.68 : deckCount <= 8 ? 0.82 : 1.0;
  const handHasAny = hasMonthCard(hand, month);
  const handHasRequired = hasMonthCategoryCard(hand, month, requiredCategory);
  const boardHasAny = hasMonthCard(board, month);
  const boardRequiredCount = board.filter((c) => c?.month === month && c?.category === requiredCategory).length;
  let chance = 0.12 * drawFactor;
  if (handHasRequired) chance = Math.max(chance, boardHasAny ? 0.64 : 0.5);
  if (boardRequiredCount > 0) chance = Math.max(chance, handHasAny ? 0.78 : 0.42);
  if (handHasAny) chance = Math.max(chance, 0.28);
  if (boardRequiredCount >= 2 && handHasAny) chance = Math.max(chance, 0.86);
  return Math.max(0, Math.min(0.92, chance));
}

export function estimateJokboExpectedPotential(state, actorKey, blockerKey) {
  const actor = state.players?.[actorKey];
  const blocker = state.players?.[blockerKey];
  if (!actor) return { total: 0, nearCompleteCount: 0, oneAwayCount: 0 };
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
  return { total, nearCompleteCount, oneAwayCount };
}

export function buildDynamicWeights(state, playerKey, ctx) {
  const deckCount = state?.deck?.length || 0;
  const blockEmphasis = Number(ctx?.blockWeight || 1.0);
  const piEmphasis = Number(ctx?.piWeight || 1.0);
  const pukEmphasis = Number(ctx?.pukPenalty || 1.0);
  const endgameBoost = deckCount <= 8 ? 1.1 : 1.0;
  return {
    blockWeight: blockEmphasis * endgameBoost,
    piWeight: piEmphasis,
    pukPenalty: pukEmphasis,
    comboWeight: ctx?.mode === "AGGRESSIVE" ? 1.1 : 1.0,
    scoreLeadWeight: currentScoreTotal(state, playerKey) >= currentScoreTotal(state, otherPlayerKey(playerKey)) ? 1.0 : 1.08
  };
}

export function estimateReleasePunishProb(state, playerKey, month, jokboThreat, ctx) {
  const oppGain = estimateOpponentImmediateGainIfDiscard(state, playerKey, month);
  const raw = (oppGain / 3.5) * 0.6 + Number(jokboThreat || 0) * 0.25 + (ctx?.midgameBlockFocus ? 0.1 : 0);
  return clamp01(raw);
}

export function estimateDangerMonthRisk(state, playerKey, month, boardCountByMonth, handCountByMonth, capturedByMonth) {
  const boardMap = boardCountByMonth instanceof Map ? boardCountByMonth : monthCounts(state?.board || []);
  const handMap = handCountByMonth instanceof Map ? handCountByMonth : monthCounts(state?.players?.[playerKey]?.hand || []);
  const capturedMap = capturedByMonth instanceof Map ? capturedByMonth : capturedMonthCounts(state);
  const boardCount = Number(boardMap.get(month) || 0);
  const handCount = Number(handMap.get(month) || 0);
  const capturedCount = Number(capturedMap.get(month) || 0);
  const remaining = Math.max(0, 4 - capturedCount - handCount - boardCount);
  return clamp01(boardCount * 0.22 + remaining * 0.12);
}

export function estimateOpponentImmediateGainIfDiscard(state, playerKey, month) {
  const opp = otherPlayerKey(playerKey);
  const oppHand = state.players?.[opp]?.hand || [];
  const boardMonthCards = (state.board || []).filter((c) => c?.month === month);
  const matchable = oppHand.some((c) => c?.month === month) ? 1 : 0;
  const target = boardMonthCards.length ? Math.max(...boardMonthCards.map((b) => cardCaptureValue(b))) : 0;
  let expected = matchable * (0.45 + target * 0.38 + 0.25);
  if (boardMonthCards.some((c) => c?.category === "kwang" || c?.category === "five")) expected += matchable * 0.12;
  return Math.max(0, Math.min(3.6, expected));
}

export function isHighImpactBomb(state, playerKey, month) {
  const gain = monthBoardGain(state, month);
  const comboGain = ownComboOpportunityScore(state, playerKey, month);
  return gain >= 5 || comboGain >= 1.0;
}

export function isHighImpactShaking(state, playerKey, month) {
  return shakingImmediateGainScore(state, playerKey, month) >= 1.4;
}

export function isRiskOfPuk(state, playerKey, card, boardCountByMonth, handCountByMonth) {
  const month = Number(card?.month || 0);
  if (month < 1) return false;
  const boardMap = boardCountByMonth instanceof Map ? boardCountByMonth : monthCounts(state?.board || []);
  const handMap = handCountByMonth instanceof Map ? handCountByMonth : monthCounts(state?.players?.[playerKey]?.hand || []);
  const boardCount = Number(boardMap.get(month) || 0);
  const handCount = Number(handMap.get(month) || 0);
  return boardCount >= 2 && handCount <= 1;
}

export function isFirstTurnForActor(state, playerKey) {
  return state?.turnSeq === 0 && state?.currentTurn === playerKey;
}

export function getFirstTurnDoublePiPlan(state, playerKey) {
  const inactive = { active: false, months: new Set(), cardId: null, month: null };
  if (!isFirstTurnForActor(state, playerKey)) return inactive;
  const hand = state?.players?.[playerKey]?.hand || [];
  for (const card of hand) {
    if (card?.category === "junk" && junkPiValue(card) >= 2) {
      return {
        active: true,
        months: new Set([card.month]),
        cardId: card.id,
        month: card.month,
      };
    }
  }
  return inactive;
}

export function matchableMonthCountForPlayer(state, playerKey) {
  const handMonths = new Set((state.players?.[playerKey]?.hand || []).map((c) => c?.month).filter((m) => Number.isInteger(m)));
  let count = 0;
  for (const month of handMonths) {
    if ((state.board || []).some((c) => c?.month === month)) count += 1;
  }
  return count;
}

export function nextTurnThreatScore(state, defenderKey) {
  const attacker = otherPlayerKey(defenderKey);
  const attackerCombos = comboProgress(state.players?.[attacker]);
  const monthToCards = new Map();
  for (const c of state.board || []) {
    if (!Number.isInteger(c?.month)) continue;
    const arr = monthToCards.get(c.month) || [];
    arr.push(c);
    monthToCards.set(c.month, arr);
  }
  let score = 0;
  for (const [month, cards] of monthToCards.entries()) {
    let local = 0;
    if ((state.players?.[attacker]?.hand || []).some((c) => c?.month === month)) {
      if (cards.some((c) => c?.category === "kwang")) local += 0.3;
      if (cards.some((c) => c?.category === "five")) local += 0.22;
      if (cards.some((c) => c?.category === "junk")) local += 0.1;
      if (attackerCombos.fiveBirds >= 2 && month === 5) local += 0.24;
      if (attackerCombos.redRibbons >= 2 && monthStrategicPriority(month) >= 0.7) local += 0.18;
      if (attackerCombos.blueRibbons >= 2 && monthStrategicPriority(month) >= 0.7) local += 0.18;
      if (attackerCombos.plainRibbons >= 2 && monthStrategicPriority(month) >= 0.7) local += 0.18;
      score += local;
    }
  }
  return Math.max(0, Math.min(1, score));
}

export function opponentThreatScore(state, playerKey) {
  const opp = otherPlayerKey(playerKey);
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
  score += p.redRibbons >= 2 || p.blueRibbons >= 2 || p.plainRibbons >= 2 ? 0.12 : 0;
  score += p.fiveBirds >= 2 ? 0.12 : 0;
  score += nextThreat * 0.28;
  return Math.max(0, Math.min(1, score));
}
