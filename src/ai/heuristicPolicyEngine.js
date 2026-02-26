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
  scoringFiveCards,
  scoringPiCount
} from "../engine/index.js";
import { COMBO_MONTHS, COMBO_MONTH_SETS, countComboTag, missingComboMonths } from "../engine/combos.js";
import { STARTING_GOLD } from "../engine/economy.js";
import { buildDeck } from "../cards.js";
import {
  BOT_POLICIES,
  DEFAULT_BOT_POLICY,
  POLICY_HEURISTIC_V3,
  POLICY_HEURISTIC_V4,
  POLICY_HEURISTIC_V5,
  POLICY_HEURISTIC_V5PLUS,
  POLICY_HEURISTIC_V6,
  POLICY_HEURISTIC_V7,
  normalizeBotPolicy
} from "./policies.js";
import {
  chooseGukjinHeuristicV3,
  chooseMatchHeuristicV3,
  decideShakingV3,
  rankHandCardsV3,
  selectBombMonthV3,
  shouldBombV3,
  shouldGoV3,
  shouldPresidentStopV3
} from "../heuristics/heuristicV3.js";
import {
  chooseGukjinHeuristicV4,
  chooseMatchHeuristicV4,
  decideShakingV4,
  rankHandCardsV4,
  selectBombMonthV4,
  shouldBombV4,
  shouldGoV4,
  shouldPresidentStopV4
} from "../heuristics/heuristicV4.js";
import {
  chooseGukjinHeuristicV5,
  chooseMatchHeuristicV5,
  decideShakingV5,
  rankHandCardsV5,
  selectBombMonthV5,
  shouldBombV5,
  shouldGoV5,
  shouldPresidentStopV5,
  DEFAULT_PARAMS as V5_DEFAULT_PARAMS
} from "../heuristics/heuristicV5.js";
import {
  rankHandCardsV5Plus,
  chooseMatchHeuristicV5Plus,
  chooseGukjinHeuristicV5Plus,
  shouldPresidentStopV5Plus,
  shouldGoV5Plus,
  selectBombMonthV5Plus,
  shouldBombV5Plus,
  decideShakingV5Plus,
  DEFAULT_PARAMS as V5PLUS_DEFAULT_PARAMS
} from "../heuristics/heuristicV5Plus.js";
import {
  chooseGukjinHeuristicV6,
  chooseMatchHeuristicV6,
  decideShakingV6,
  rankHandCardsV6,
  selectBombMonthV6,
  shouldBombV6,
  shouldGoV6,
  shouldPresidentStopV6,
  DEFAULT_PARAMS as V6_DEFAULT_PARAMS
} from "../heuristics/heuristicV6.js";
import {
  chooseGukjinHeuristicV7,
  chooseMatchHeuristicV7,
  decideShakingV7,
  rankHandCardsV7,
  selectBombMonthV7,
  shouldBombV7,
  shouldGoV7,
  shouldPresidentStopV7,
  DEFAULT_PARAMS as V7_DEFAULT_PARAMS
} from "../heuristics/heuristicV7.js";
import { canBankruptOpponentByStopV7 } from "../heuristics/heuristicV7_math.js";

const GWANG_MONTHS = Object.freeze([1, 3, 8, 11, 12]);
const GOLD_RISK_THRESHOLD_RATIO = 0.1;
const GUKJIN_CARD_ID = "I0";
const GUKJIN_ANALYSIS_BOARD_WEIGHT = 0.5;
const COMBO_REQUIRED_CATEGORY = Object.freeze({
  redRibbons: "ribbon",
  blueRibbons: "ribbon",
  plainRibbons: "ribbon",
  fiveBirds: "five"
});
const HIGH_PI_CARD_IDS = Object.freeze(["M0", "M1", "K1", "L3", GUKJIN_CARD_ID]);

function loadV5Params() {
  try {
    const raw = (typeof process !== "undefined" && process.env?.HEURISTIC_V5_PARAMS) || "";
    if (!raw) return { ...V5_DEFAULT_PARAMS };
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return { ...V5_DEFAULT_PARAMS, ...parsed };
    }
  } catch {
    // Fall through to defaults on parse/runtime errors.
  }
  return { ...V5_DEFAULT_PARAMS };
}

function loadV5PlusParams() {
  try {
    const raw = (typeof process !== "undefined" && process.env?.HEURISTIC_V5PLUS_PARAMS) || "";
    if (!raw) return { ...V5PLUS_DEFAULT_PARAMS };
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return { ...V5PLUS_DEFAULT_PARAMS, ...parsed };
    }
  } catch {
    // Fall through to defaults on parse/runtime errors.
  }
  return { ...V5PLUS_DEFAULT_PARAMS };
}

function loadV6Params() {
  try {
    const raw = (typeof process !== "undefined" && process.env?.HEURISTIC_V6_PARAMS) || "";
    if (!raw) return { ...V6_DEFAULT_PARAMS };
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return { ...V6_DEFAULT_PARAMS, ...parsed };
    }
  } catch {
    // Fall through to defaults on parse/runtime errors.
  }
  return { ...V6_DEFAULT_PARAMS };
}

function loadV7Params() {
  try {
    const raw = (typeof process !== "undefined" && process.env?.HEURISTIC_V7_PARAMS) || "";
    if (!raw) return { ...V7_DEFAULT_PARAMS };
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return { ...V7_DEFAULT_PARAMS, ...parsed };
    }
  } catch {
    // Fall through to defaults on parse/runtime errors.
  }
  return { ...V7_DEFAULT_PARAMS };
}

function ssangpiCardIds() {
  return [...HIGH_PI_CARD_IDS];
}

const HEURISTIC_V5_PARAMS = Object.freeze(loadV5Params());
const HEURISTIC_V5PLUS_PARAMS = Object.freeze(loadV5PlusParams());
const HEURISTIC_V6_PARAMS = Object.freeze(loadV6Params());
const HEURISTIC_V7_PARAMS = Object.freeze(loadV7Params());

export {
  BOT_POLICIES,
  DEFAULT_BOT_POLICY,
  POLICY_HEURISTIC_V3,
  POLICY_HEURISTIC_V4,
  POLICY_HEURISTIC_V5,
  POLICY_HEURISTIC_V5PLUS,
  POLICY_HEURISTIC_V6,
  POLICY_HEURISTIC_V7
};

export function botChooseCard(state, playerKey, policy = DEFAULT_BOT_POLICY) {
  const player = state.players[playerKey];
  if (!player || player.hand.length === 0) return null;
  const ranked = rankHandCardsByPolicy(state, playerKey, policy);
  if (ranked.length > 0) {
    return ranked[0].card.id;
  }
  return pickRandom(player.hand)?.id ?? null;
}

export function botPlay(state, playerKey, options = {}) {
  const policy = normalizeBotPolicy(options?.policy);
  return botPlaySmart(state, playerKey, { ...options, policy });
}

export function getHeuristicCardProbabilities(state, playerKey, policy = DEFAULT_BOT_POLICY) {
  if (state.phase !== "playing") return null;
  const ranked = rankHandCardsByPolicy(state, playerKey, policy);
  if (!ranked.length) return null;
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

function chooseShakingCardIdForMonth(state, playerKey, month, policy = DEFAULT_BOT_POLICY) {
  const player = state.players?.[playerKey];
  if (!player?.hand?.length) return null;
  const targetMonth = Number(month);
  const monthCards = player.hand.filter((c) => c && !c.passCard && Number(c.month) === targetMonth);
  if (!monthCards.length) return null;

  const ranked = rankHandCardsByPolicy(state, playerKey, policy);
  if (policy === POLICY_HEURISTIC_V6) {
    const v6Picked = chooseShakingDiscardCardIdV6(state, playerKey, monthCards, ranked);
    if (v6Picked) return v6Picked;
  }

  if (Array.isArray(ranked) && ranked.length > 0) {
    const picked = ranked.find((r) =>
      Number(r?.card?.month) === targetMonth &&
      monthCards.some((c) => c.id === r?.card?.id)
    );
    if (picked?.card?.id) return picked.card.id;
  }

  monthCards.sort((a, b) => cardCaptureValue(b) - cardCaptureValue(a));
  return monthCards[0]?.id || null;
}

function toFiniteNumber(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function chooseShakingDiscardCardIdV6(state, playerKey, monthCards, ranked) {
  if (!monthCards?.length) return null;
  const player = state.players?.[playerKey];
  const oppKey = otherPlayerKey(playerKey);
  const opp = state.players?.[oppKey];
  if (!player || !opp) return monthCards[0]?.id || null;

  const rankScoreById = new Map(
    (ranked || []).map((r) => [r?.card?.id, toFiniteNumber(r?.score)])
  );

  const redCnt = countComboTag(player.captured?.ribbon || [], "redRibbons");
  const blueCnt = countComboTag(player.captured?.ribbon || [], "blueRibbons");
  const plainCnt = countComboTag(player.captured?.ribbon || [], "plainRibbons");
  const birdCnt = countComboTag(player.captured?.five || [], "fiveBirds");

  const myBase = toFiniteNumber(calculateScore(player, opp, state.ruleKey)?.base);
  const oppBase = toFiniteNumber(calculateScore(opp, player, state.ruleKey)?.base);
  const scoreDiff = myBase - oppBase;

  let best = null;
  for (const card of monthCards) {
    const cap = cardCaptureValue(card);
    const pi = junkPiValue(card);
    const rankScore = toFiniteNumber(rankScoreById.get(card.id));
    const tags = Array.isArray(card?.comboTags) ? card.comboTags : [];

    // higher keepScore means "do not discard this card for shaking"
    let keepScore = cap * 2.2 + Math.max(0, rankScore) * 0.16;
    if (card.category === "kwang") keepScore += 7.5;
    else if (card.category === "five") keepScore += 5.0;
    else if (card.category === "ribbon") keepScore += 3.2;
    if (pi >= 2) keepScore += 3.0 + (pi - 2) * 1.4;
    if (card.id === GUKJIN_CARD_ID) keepScore += 4.0;

    if (tags.includes("redRibbons") && redCnt >= 2) keepScore += 2.4;
    if (tags.includes("blueRibbons") && blueCnt >= 2) keepScore += 2.4;
    if (tags.includes("plainRibbons") && plainCnt >= 2) keepScore += 2.4;
    if (tags.includes("fiveBirds") && birdCnt >= 2) keepScore += 2.8;

    if (scoreDiff > 0) keepScore += 0.9;
    else if (scoreDiff < 0) keepScore -= 0.7;

    // prefer throwing low-pi junk when available in shaking month
    if (card.category === "junk" && pi <= 1) keepScore -= 1.2;

    const candidate = {
      id: card.id,
      keepScore,
      cap,
      pi
    };
    if (
      !best ||
      candidate.keepScore < best.keepScore ||
      (candidate.keepScore === best.keepScore && candidate.cap < best.cap) ||
      (candidate.keepScore === best.keepScore && candidate.cap === best.cap && candidate.pi < best.pi)
    ) {
      best = candidate;
    }
  }

  return best?.id || monthCards[0]?.id || null;
}

function botPlaySmart(state, playerKey, options = {}) {
  const policy = normalizeBotPolicy(options?.policy);
  if (state.phase === "gukjin-choice" && state.pendingGukjinChoice?.playerKey === playerKey) {
    return chooseGukjinMode(state, playerKey, chooseGukjinByPolicy(state, playerKey, policy));
  }

  if (state.phase === "president-choice" && state.pendingPresident?.playerKey === playerKey) {
    const shouldStop = shouldPresidentStopByPolicy(state, playerKey, policy);
    return shouldStop ? choosePresidentStop(state, playerKey) : choosePresidentHold(state, playerKey);
  }

  if (state.phase === "select-match" && state.pendingMatch?.playerKey === playerKey) {
    const choiceId = chooseMatchByPolicy(state, playerKey, policy);
    return choiceId ? chooseMatch(state, choiceId) : state;
  }

  if (state.phase === "go-stop" && state.pendingGoStop === playerKey) {
    return shouldGoByPolicy(state, playerKey, policy) ? chooseGo(state, playerKey) : chooseStop(state, playerKey);
  }

  if (state.phase === "playing" && state.currentTurn === playerKey) {
    const bombMonths = getDeclarableBombMonths(state, playerKey);
    if (bombMonths.length > 0 && shouldBombByPolicy(state, playerKey, bombMonths, policy)) {
      const month = selectBombMonthByPolicy(state, playerKey, bombMonths, policy);
      return declareBomb(state, playerKey, month ?? selectBestMonth(state, bombMonths));
    }
    const shakingMonths = getDeclarableShakingMonths(state, playerKey);
    if (shakingMonths.length > 0) {
      const shakeDecision = decideShakingByPolicy(state, playerKey, shakingMonths, policy);
      if (shakeDecision.allow && shakeDecision.month != null) {
        const shakeCardId = chooseShakingCardIdForMonth(state, playerKey, shakeDecision.month, policy);
        const declared = declareShaking(state, playerKey, shakeDecision.month);
        if (!declared || declared === state) return state;
        if (!shakeCardId) return declared;
        const played = playTurn(declared, shakeCardId);
        return played || declared;
      }
    }
    const cardId = botChooseCard(state, playerKey, policy);
    if (!cardId) return state;
    return playTurn(state, cardId);
  }

  return state;
}

function capturedCountByCategory(player, category) {
  if (!player?.captured) return 0;
  if (category === "junk") {
    return scoringPiCount(player);
  }
  return (player.captured[category] || []).length;
}

function scoringFiveCount(player) {
  return scoringFiveCards(player).length;
}

function fiveCountIncludingCapturedGukjin(player) {
  const fiveCount = (player?.captured?.five || []).length;
  if (!player?.captured) return fiveCount;
  if (hasCardId(player.captured.five || [], GUKJIN_CARD_ID)) return fiveCount;
  return hasGukjinInCaptured(player.captured) ? fiveCount + 1 : fiveCount;
}

function otherPlayerKey(playerKey) {
  return playerKey === "human" ? "ai" : "human";
}

function hasCardId(cards, cardId) {
  return (cards || []).some((c) => c?.id === cardId);
}

function hasGukjinInCaptured(captured) {
  if (!captured) return false;
  for (const cat of ["kwang", "five", "ribbon", "junk"]) {
    if (hasCardId(captured[cat] || [], GUKJIN_CARD_ID)) return true;
  }
  return false;
}

function gatherGukjinZoneFlags(state, playerKey) {
  const opp = otherPlayerKey(playerKey);
  const selfPlayer = state.players?.[playerKey];
  const oppPlayer = state.players?.[opp];
  return {
    selfHand: hasCardId(selfPlayer?.hand || [], GUKJIN_CARD_ID),
    selfCaptured: hasGukjinInCaptured(selfPlayer?.captured),
    oppCaptured: hasGukjinInCaptured(oppPlayer?.captured),
    board: hasCardId(state.board || [], GUKJIN_CARD_ID)
  };
}

function hasAnyGukjinFlag(flags) {
  return !!flags && Object.values(flags).some(Boolean);
}

function forceGukjinModeState(state, modeByPlayer = {}) {
  const keys = Object.keys(modeByPlayer || {});
  if (!keys.length) return state;
  let changed = false;
  const nextPlayers = { ...state.players };
  for (const key of keys) {
    const mode = modeByPlayer[key];
    if (mode !== "five" && mode !== "junk") continue;
    const player = state.players?.[key];
    if (!player) continue;
    if (player.gukjinMode === mode) continue;
    nextPlayers[key] = { ...player, gukjinMode: mode };
    changed = true;
  }
  if (!changed) return state;
  return { ...state, players: nextPlayers };
}

function buildGukjinScenario(state, playerKey, selfMode, oppMode, zoneFlags) {
  const opp = otherPlayerKey(playerKey);
  const scenarioState = forceGukjinModeState(state, { [playerKey]: selfMode, [opp]: oppMode });
  const selfPlayer = scenarioState.players?.[playerKey];
  const oppPlayer = scenarioState.players?.[opp];
  if (!selfPlayer || !oppPlayer) return null;

  let selfPi = capturedCountByCategory(selfPlayer, "junk");
  let selfFive = scoringFiveCount(selfPlayer);
  let oppPi = capturedCountByCategory(oppPlayer, "junk");
  let oppFive = scoringFiveCount(oppPlayer);

  // If gukjin is still in hand/board, include both-mode projected value in analysis.
  if (zoneFlags?.selfHand) {
    if (selfMode === "junk") selfPi += 2;
    else selfFive += 1;
  }
  if (zoneFlags?.board) {
    if (selfMode === "junk") selfPi += 2 * GUKJIN_ANALYSIS_BOARD_WEIGHT;
    else selfFive += 1 * GUKJIN_ANALYSIS_BOARD_WEIGHT;
  }

  const selfScoreInfo = calculateScore(selfPlayer, oppPlayer, scenarioState.ruleKey);
  const oppScoreInfo = calculateScore(oppPlayer, selfPlayer, scenarioState.ruleKey);

  return {
    selfMode,
    oppMode,
    selfPi,
    selfFive,
    oppPi,
    oppFive,
    myScore: Number(selfScoreInfo?.total || 0),
    oppScore: Number(oppScoreInfo?.total || 0),
    mongRiskSelf: selfFive <= 0 && oppFive >= 6,
    canMongBakSelf: selfFive >= 7 && oppFive <= 0
  };
}

function summarizeScenarioRange(scenarios, field) {
  if (!scenarios.length) return { min: 0, max: 0, avg: 0 };
  const vals = scenarios.map((s) => Number(s?.[field] || 0));
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const avg = vals.reduce((sum, v) => sum + v, 0) / vals.length;
  return { min, max, avg };
}

function pickGukjinScenario(scenarios, selfMode, oppMode) {
  return (scenarios || []).find((s) => s?.selfMode === selfMode && s?.oppMode === oppMode) || null;
}

function preferredGukjinModeByFiveCount(fiveCount) {
  return Number(fiveCount || 0) >= 7 ? "five" : "junk";
}

function analyzeGukjinBranches(state, playerKey) {
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

function resolveInitialGoldBase(state) {
  const configured = Number(state?.initialGoldBase);
  if (Number.isFinite(configured) && configured > 0) return configured;
  return STARTING_GOLD;
}

function goldRiskProfile(state, playerKey) {
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

function canBankruptOpponentByStop(state, playerKey) {
  if (state.phase !== "go-stop" || state.pendingGoStop !== playerKey) return false;
  const opp = playerKey === "human" ? "ai" : "human";
  const stopped = chooseStop(state, playerKey);
  const oppGoldAfterStop = Number(stopped?.players?.[opp]?.gold || 0);
  return oppGoldAfterStop <= 0;
}

function canBankruptOpponentByStopV7Proxy(state, playerKey) {
  const scoreHint = Number(analyzeGameContext(state, playerKey)?.myScore || 0);
  return canBankruptOpponentByStopV7(state, playerKey, {
    currentScore: scoreHint,
    fallbackExact: canBankruptOpponentByStop,
    defaultBetAmount: 100,
    safetyMargin: 0.9
  });
}

function analyzeGameContext(state, playerKey) {
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
    // Default policy: treat gukjin as junk(ssangpi) unless five count reaches 7+ (including captured gukjin).
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
      // Fallback for safety when preferred scenario is unavailable.
      myScore = gukjinBranch.myScore.avg;
      oppScore = gukjinBranch.oppScore.avg;
      selfPi = gukjinBranch.selfPi.avg;
      oppPi = gukjinBranch.oppPi.avg;
      selfFive = gukjinBranch.selfFive.min;
      oppFive = gukjinBranch.oppFive.max;
    }
    // Defensive baseline: assume lower self five / higher opponent five.
    selfFive = Math.min(selfFive, gukjinBranch.selfFive.min);
    oppFive = Math.max(oppFive, gukjinBranch.oppFive.max);
    const criticalMongRisk = selfFive <= 0 && oppFive >= 6;
    mong = {
      ...mong,
      selfFive,
      oppFive,
      danger: Math.max(mong.danger, criticalMongRisk ? 0.85 : 0),
      stage: criticalMongRisk
        ? "CRITICAL"
        : selfFive <= 0 && oppFive >= 5
        ? "ELEVATED"
        : mong.stage
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

function getMissingComboCards(state, playerKey) {
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
    const missing = availableMissingMonths(
      missingComboMonths(sourceCards, tag),
      requiredCategory,
      blocker
    );
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

  return {
    ids: [...ids],
    months: [...months],
    imminent
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

function normalizeMonthCountMap(value, fallbackCards = []) {
  if (value && typeof value.get === "function") return value;
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const out = new Map();
    for (const [k, v] of Object.entries(value)) {
      const month = Number(k);
      const cnt = Number(v);
      if (!Number.isFinite(month) || !Number.isFinite(cnt)) continue;
      out.set(month, cnt);
    }
    return out;
  }
  return monthCounts(fallbackCards);
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
  const boardCounts = normalizeMonthCountMap(boardCountByMonth, state.board || []);
  const handCounts = normalizeMonthCountMap(handCountByMonth, state.players?.[playerKey]?.hand || []);
  const capturedCounts = normalizeMonthCountMap(capturedByMonth, []);
  const boardCnt = boardCounts.get(month) || 0;
  const handCnt = handCounts.get(month) || 0;
  const capturedCnt = capturedCounts.get(month) || 0;
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
  const boardCounts = normalizeMonthCountMap(boardCountByMonth, state.board || []);
  const handCounts = normalizeMonthCountMap(handCountByMonth, state.players?.[playerKey]?.hand || []);
  const boardCnt = boardCounts.get(card.month) || 0;
  const handCnt = handCounts.get(card.month) || 0;
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

function cloneGameState(state) {
  if (typeof structuredClone === "function") return structuredClone(state);
  return JSON.parse(JSON.stringify(state));
}

function makeHiddenCard(prefix, index) {
  return {
    id: `${prefix}_${index}`,
    month: null,
    category: "hidden"
  };
}

function collectPublicShakingRevealIds(state, targetPlayerKey) {
  const ids = new Set();
  if (!state || !targetPlayerKey) return ids;

  const liveReveal = state.shakingReveal;
  if (liveReveal?.playerKey === targetPlayerKey) {
    for (const c of liveReveal.cards || []) {
      if (typeof c?.id === "string" && c.id) ids.add(c.id);
    }
  }

  for (const entry of state.kibo || []) {
    if (entry?.type !== "shaking_declare") continue;
    if (entry?.playerKey !== targetPlayerKey) continue;
    for (const c of entry?.revealCards || []) {
      if (typeof c?.id === "string" && c.id) ids.add(c.id);
    }
  }
  return ids;
}

function getPublicKnownOpponentHandCards(state, observerKey) {
  const opp = otherPlayerKey(observerKey);
  const revealIds = collectPublicShakingRevealIds(state, opp);
  if (revealIds.size <= 0) return [];
  const oppHand = state?.players?.[opp]?.hand || [];
  return oppHand.filter((c) => typeof c?.id === "string" && revealIds.has(c.id));
}

function createPublicState(state, observerKey) {
  const pub = cloneGameState(state);
  const opp = otherPlayerKey(observerKey);
  const oppHandLen = state?.players?.[opp]?.hand?.length || 0;
  const deckLen = state?.deck?.length || 0;
  const knownOppCards = getPublicKnownOpponentHandCards(state, observerKey).map((c) => ({ ...c }));
  const hiddenOppCount = Math.max(0, oppHandLen - knownOppCards.length);

  if (pub?.players?.[opp]) {
    pub.players[opp].hand = knownOppCards.concat(
      Array.from({ length: hiddenOppCount }, (_, i) => makeHiddenCard(`opp_${opp}`, i))
    );
  }
  pub.deck = Array.from({ length: deckLen }, (_, i) => makeHiddenCard("deck", i));
  return pub;
}

function shuffledCards(cards) {
  const out = [...(cards || [])];
  for (let i = out.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

function determinizeUnknownCards(state, observerKey) {
  const opp = otherPlayerKey(observerKey);
  const oppHandLen = state?.players?.[opp]?.hand?.length || 0;
  const deckLen = state?.deck?.length || 0;
  const fixedKnownOppCards = getPublicKnownOpponentHandCards(state, observerKey).map((c) => ({ ...c }));
  const hiddenOppHandSize = Math.max(0, oppHandLen - fixedKnownOppCards.length);
  const unknownCount = hiddenOppHandSize + deckLen;
  const next = cloneGameState(state);
  if (unknownCount <= 0) return next;

  const knownIds = new Set();
  const collect = (cards = []) => {
    for (const c of cards) {
      const id = c?.id;
      if (typeof id !== "string") continue;
      if (id.startsWith("opp_") || id.startsWith("deck_")) continue;
      knownIds.add(id);
    }
  };

  collect(state.board || []);
  collect(state.players?.[observerKey]?.hand || []);
  collect(fixedKnownOppCards);
  for (const k of ["human", "ai"]) {
    const cap = state.players?.[k]?.captured || {};
    collect(cap.kwang || []);
    collect(cap.five || []);
    collect(cap.ribbon || []);
    collect(cap.junk || []);
  }

  const cardTheme = state?.cardTheme || state?.theme;
  const candidateUnknown = buildDeck(cardTheme).filter((c) => !knownIds.has(c.id));
  let sampledUnknown = shuffledCards(candidateUnknown).slice(0, unknownCount);

  if (sampledUnknown.length < unknownCount) {
    const missing = unknownCount - sampledUnknown.length;
    const fillers = Array.from({ length: missing }, (_, i) => ({
      id: `sample_${i}`,
      month: (i % 12) + 1,
      category: "junk",
      piValue: 1
    }));
    sampledUnknown = sampledUnknown.concat(fillers);
  }

  const sampledOppUnknown = sampledUnknown.slice(0, hiddenOppHandSize);
  if (next?.players?.[opp]) next.players[opp].hand = fixedKnownOppCards.concat(sampledOppUnknown);
  next.deck = sampledUnknown.slice(hiddenOppHandSize);
  return next;
}

function knownMonthCountForObserver(state, observerKey, month) {
  let count = 0;
  for (const c of state.board || []) if (c?.month === month) count += 1;
  const selfPlayer = state.players?.[observerKey];
  for (const c of selfPlayer?.hand || []) if (c?.month === month) count += 1;
  for (const c of getPublicKnownOpponentHandCards(state, observerKey)) if (c?.month === month) count += 1;
  for (const key of ["human", "ai"]) {
    const cap = state.players?.[key]?.captured || {};
    for (const cat of ["kwang", "five", "ribbon", "junk"]) {
      for (const c of cap[cat] || []) if (c?.month === month) count += 1;
    }
  }
  return count;
}

function hiddenPoolStatsForObserver(state, observerKey) {
  const opp = otherPlayerKey(observerKey);
  const oppHandSize = state?.players?.[opp]?.hand?.length || 0;
  const knownOppHandSize = getPublicKnownOpponentHandCards(state, observerKey).length;
  const hiddenOppHandSize = Math.max(0, oppHandSize - knownOppHandSize);
  const deckCount = state?.deck?.length || 0;
  return {
    oppHandSize,
    knownOppHandSize,
    hiddenOppHandSize,
    deckCount,
    hiddenPoolSize: hiddenOppHandSize + deckCount
  };
}

function opponentMonthHoldProbPublic(state, observerKey, month) {
  const knownOppMonthCount = getPublicKnownOpponentHandCards(state, observerKey).filter((c) => c?.month === month).length;
  if (knownOppMonthCount > 0) return 1;

  const known = knownMonthCountForObserver(state, observerKey, month);
  const unseenMonthCards = Math.max(0, 4 - known);
  const { hiddenOppHandSize, hiddenPoolSize } = hiddenPoolStatsForObserver(state, observerKey);
  if (hiddenOppHandSize <= 0 || hiddenPoolSize <= 0 || unseenMonthCards <= 0) return 0;
  const perCardMiss = Math.max(0, 1 - unseenMonthCards / hiddenPoolSize);
  const missAll = Math.pow(perCardMiss, hiddenOppHandSize);
  return clamp01(1 - missAll);
}

function matchableMonthCountForPlayerPublic(state, playerKey, observerKey) {
  if (playerKey === observerKey) return matchableMonthCountForPlayer(state, playerKey);
  const boardMonths = [...new Set((state.board || []).map((c) => c?.month).filter((m) => Number.isInteger(m)))];
  let expected = 0;
  for (const month of boardMonths) {
    expected += opponentMonthHoldProbPublic(state, observerKey, month);
  }
  return expected;
}

function nextTurnThreatScorePublic(state, defenderKey, observerKey) {
  const attacker = otherPlayerKey(defenderKey);
  if (attacker === observerKey) return nextTurnThreatScore(state, defenderKey);

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
    const matchProb = opponentMonthHoldProbPublic(state, observerKey, month);
    if (matchProb <= 0) continue;
    let local = 0;
    if (cards.some((c) => c?.category === "kwang")) local += 0.3;
    if (cards.some((c) => c?.category === "five")) local += 0.22;
    if (cards.some((c) => c?.category === "junk")) local += 0.1;
    if (attackerCombos.fiveBirds >= 2 && COMBO_MONTH_SETS.fiveBirds.has(month)) local += 0.24;
    if (attackerCombos.redRibbons >= 2 && COMBO_MONTH_SETS.redRibbons.has(month)) local += 0.18;
    if (attackerCombos.blueRibbons >= 2 && COMBO_MONTH_SETS.blueRibbons.has(month)) local += 0.18;
    if (attackerCombos.plainRibbons >= 2 && COMBO_MONTH_SETS.plainRibbons.has(month)) local += 0.18;
    score += local * matchProb;
  }
  return Math.max(0, Math.min(1, score));
}

function opponentThreatScorePublic(state, playerKey, observerKey) {
  const opp = otherPlayerKey(playerKey);
  const oppPlayer = state.players?.[opp];
  const oppScore = currentScoreTotal(state, opp);
  const oppPi = capturedCountByCategory(oppPlayer, "junk");
  const oppGwang = capturedCountByCategory(oppPlayer, "kwang");
  const p = comboProgress(oppPlayer);
  const nextThreat = nextTurnThreatScorePublic(state, playerKey, observerKey);
  let score = 0;
  score += Math.min(1.0, oppScore / 7.0) * 0.55;
  score += Math.min(1.0, oppPi / 10.0) * 0.15;
  score += Math.min(1.0, oppGwang / 3.0) * 0.1;
  score += p.redRibbons >= 2 || p.blueRibbons >= 2 || p.plainRibbons >= 2 ? 0.12 : 0;
  score += p.fiveBirds >= 2 ? 0.12 : 0;
  score += nextThreat * 0.28;
  return Math.max(0, Math.min(1, score));
}

function boardHighValueThreatForPlayerPublic(state, playerKey, observerKey) {
  if (playerKey === observerKey) return boardHighValueThreatForPlayer(state, playerKey);
  const blockerKey = otherPlayerKey(playerKey);
  const blockMonths = blockingMonthsAgainst(state.players?.[playerKey], state.players?.[blockerKey]);
  for (const c of state.board || []) {
    const highValue = c?.category === "kwang" || c?.category === "five" || blockMonths.has(c?.month);
    if (!highValue || !Number.isInteger(c?.month)) continue;
    const matchProb = opponentMonthHoldProbPublic(state, observerKey, c.month);
    if (matchProb >= 0.33) return true;
  }
  return false;
}

function estimateOpponentImmediateGainIfDiscardPublic(state, playerKey, month, observerKey) {
  const opp = otherPlayerKey(playerKey);
  if (opp === observerKey) return estimateOpponentImmediateGainIfDiscard(state, playerKey, month);

  const boardMonthCards = (state.board || []).filter((c) => c?.month === month);
  const target = boardMonthCards.length ? Math.max(...boardMonthCards.map((b) => cardCaptureValue(b))) : 0;
  const matchProb = opponentMonthHoldProbPublic(state, observerKey, month);
  let expected = matchProb * (0.45 + target * 0.38 + 0.25);
  if (boardMonthCards.some((c) => c?.category === "kwang" || c?.category === "five")) expected += matchProb * 0.12;
  return Math.max(0, Math.min(3.6, expected));
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

const RANK_HAND_CARD_DEPS = Object.freeze({
  analyzeGameContext,
  checkOpponentJokboProgress,
  getMissingComboCards,
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
});

const HEURISTIC_V3_DEPS = Object.freeze({
  ...RANK_HAND_CARD_DEPS,
  analyzeGukjinBranches,
  boardHighValueThreatForPlayer,
  canBankruptOpponentByStop,
  estimateJokboExpectedPotential,
  goldRiskProfile,
  isHighImpactBomb,
  isHighImpactShaking,
  isOppVulnerableForBigGo,
  matchableMonthCountForPlayer,
  monthBoardGain,
  opponentThreatScore,
  otherPlayerKey,
  ownComboOpportunityScore,
  shakingImmediateGainScore
});

const HEURISTIC_V4_DEPS = Object.freeze({
  analyzeGameContext,
  analyzeGukjinBranches,
  blockingMonthsAgainst,
  blockingUrgencyByMonth,
  canBankruptOpponentByStop,
  capturedCountByCategory,
  capturedMonthCounts,
  cardCaptureValue,
  checkOpponentJokboProgress,
  countKnownMonthCards,
  getFirstTurnDoublePiPlan,
  goldRiskProfile,
  isHighImpactBomb,
  isHighImpactShaking,
  isRiskOfPuk,
  junkPiValue,
  monthBoardGain,
  monthCounts,
  monthStrategicPriority,
  nextTurnThreatScore,
  opponentThreatScore,
  otherPlayerKey,
  ownComboOpportunityScore,
  boardMatchesByMonth,
  estimateOpponentImmediateGainIfDiscard,
  shakingImmediateGainScore
});

const HEURISTIC_V5_DEPS = Object.freeze({
  analyzeGameContext,
  analyzeGukjinBranches,
  blockingMonthsAgainst,
  blockingUrgencyByMonth,
  canBankruptOpponentByStop,
  capturedCountByCategory,
  capturedMonthCounts,
  cardCaptureValue,
  checkOpponentJokboProgress,
  countKnownMonthCards,
  getFirstTurnDoublePiPlan,
  goldRiskProfile,
  isHighImpactBomb,
  isHighImpactShaking,
  isRiskOfPuk,
  junkPiValue,
  monthBoardGain,
  monthCounts,
  monthStrategicPriority,
  nextTurnThreatScore,
  opponentThreatScore,
  otherPlayerKey,
  ownComboOpportunityScore,
  boardMatchesByMonth,
  estimateOpponentImmediateGainIfDiscard,
  shakingImmediateGainScore,
  ssangpiCardIds
});

const HEURISTIC_V6_DEPS = Object.freeze({
  analyzeGameContext,
  analyzeGukjinBranches,
  blockingMonthsAgainst,
  blockingUrgencyByMonth,
  boardHighValueThreatForPlayer,
  canBankruptOpponentByStop,
  capturedCountByCategory,
  capturedMonthCounts,
  cardCaptureValue,
  checkOpponentJokboProgress,
  countKnownMonthCards,
  estimateJokboExpectedPotential,
  estimateOpponentJokboExpectedPotential: (state, playerKey) =>
    estimateJokboExpectedPotential(state, otherPlayerKey(playerKey), playerKey),
  estimateOpponentImmediateGainIfDiscard,
  estimateDangerMonthRisk,
  estimateReleasePunishProb,
  getFirstTurnDoublePiPlan,
  goldRiskProfile,
  isHighImpactBomb,
  isHighImpactShaking,
  isRiskOfPuk,
  junkPiValue,
  matchableMonthCountForPlayer,
  monthBoardGain,
  monthCounts,
  monthStrategicPriority,
  nextTurnThreatScore,
  opponentThreatScore,
  otherPlayerKey,
  ownComboOpportunityScore,
  boardMatchesByMonth,
  shakingImmediateGainScore,
  ssangpiCardIds,
  rolloutStateUtilityV6,
  rolloutCardValueV6,
  rolloutGoStopValueV6
});

const HEURISTIC_V7_DEPS = Object.freeze({
  ...HEURISTIC_V5_DEPS,
  canBankruptOpponentByStop: canBankruptOpponentByStopV7Proxy,
  estimateDangerMonthRisk,
  monthBoardGain,
  opponentThreatScore
});

function createHeuristicV6FairDeps(observerKey) {
  return Object.freeze({
    ...HEURISTIC_V6_DEPS,
    boardHighValueThreatForPlayer: (state, playerKey) =>
      boardHighValueThreatForPlayerPublic(state, playerKey, observerKey),
    estimateOpponentJokboExpectedPotential: (state, playerKey) =>
      estimateJokboExpectedPotential(state, otherPlayerKey(playerKey), playerKey),
    estimateOpponentImmediateGainIfDiscard: (state, playerKey, month) =>
      estimateOpponentImmediateGainIfDiscardPublic(state, playerKey, month, observerKey),
    matchableMonthCountForPlayer: (state, playerKey) =>
      matchableMonthCountForPlayerPublic(state, playerKey, observerKey),
    nextTurnThreatScore: (state, playerKey) => nextTurnThreatScorePublic(state, playerKey, observerKey),
    opponentThreatScore: (state, playerKey) => opponentThreatScorePublic(state, playerKey, observerKey),
    rolloutCardValueV6: (state, playerKey, cardId, options = {}) =>
      rolloutCardValueV6(state, playerKey, cardId, {
        ...options,
        observerKey,
        fairTeacher: true
      }),
    rolloutGoStopValueV6: (state, playerKey, chooseGoFlag, options = {}) =>
      rolloutGoStopValueV6(state, playerKey, chooseGoFlag, {
        ...options,
        observerKey,
        fairTeacher: true
      })
  });
}

function createV6FairDecisionContext(state, playerKey) {
  const publicState = createPublicState(state, playerKey);
  const deps = createHeuristicV6FairDeps(playerKey);
  return { publicState, deps };
}

function createHeuristicPublicState(state, playerKey) {
  return createPublicState(state, playerKey);
}

function resolveHeuristicDecisionContext(state, playerKey, policy = DEFAULT_BOT_POLICY) {
  const resolvedPolicy = normalizeBotPolicy(policy);
  if (resolvedPolicy === POLICY_HEURISTIC_V7) {
    return {
      policy: POLICY_HEURISTIC_V7,
      decisionState: createHeuristicPublicState(state, playerKey),
      deps: HEURISTIC_V7_DEPS,
      params: HEURISTIC_V7_PARAMS
    };
  }
  if (resolvedPolicy === POLICY_HEURISTIC_V6) {
    const fair = createV6FairDecisionContext(state, playerKey);
    return {
      policy: POLICY_HEURISTIC_V6,
      decisionState: fair.publicState,
      deps: fair.deps,
      params: HEURISTIC_V6_PARAMS
    };
  }

  const decisionState = createHeuristicPublicState(state, playerKey);
  if (resolvedPolicy === POLICY_HEURISTIC_V5PLUS) {
    return {
      policy: POLICY_HEURISTIC_V5PLUS,
      decisionState,
      deps: HEURISTIC_V5_DEPS,
      params: HEURISTIC_V5PLUS_PARAMS
    };
  }
  if (resolvedPolicy === POLICY_HEURISTIC_V5) {
    return {
      policy: POLICY_HEURISTIC_V5,
      decisionState,
      deps: HEURISTIC_V5_DEPS,
      params: HEURISTIC_V5_PARAMS
    };
  }
  if (resolvedPolicy === POLICY_HEURISTIC_V4) {
    return {
      policy: POLICY_HEURISTIC_V4,
      decisionState,
      deps: HEURISTIC_V4_DEPS,
      params: null
    };
  }
  return {
    policy: POLICY_HEURISTIC_V3,
    decisionState,
    deps: HEURISTIC_V3_DEPS,
    params: null
  };
}

function dispatchHeuristicPolicyCall(ctx, handlers) {
  if (ctx.policy === POLICY_HEURISTIC_V7 && typeof handlers.v7 === "function") return handlers.v7(ctx);
  if (ctx.policy === POLICY_HEURISTIC_V6 && typeof handlers.v6 === "function") return handlers.v6(ctx);
  if (ctx.policy === POLICY_HEURISTIC_V5PLUS && typeof handlers.v5plus === "function") return handlers.v5plus(ctx);
  if (ctx.policy === POLICY_HEURISTIC_V5 && typeof handlers.v5 === "function") return handlers.v5(ctx);
  if (ctx.policy === POLICY_HEURISTIC_V4 && typeof handlers.v4 === "function") return handlers.v4(ctx);
  return handlers.v3(ctx);
}

function chooseGukjinByPolicy(state, playerKey, policy = DEFAULT_BOT_POLICY) {
  const ctx = resolveHeuristicDecisionContext(state, playerKey, policy);
  return dispatchHeuristicPolicyCall(ctx, {
    v7: ({ decisionState, deps, params }) => chooseGukjinHeuristicV7(decisionState, playerKey, deps, params),
    v6: ({ decisionState, deps, params }) => chooseGukjinHeuristicV6(decisionState, playerKey, deps, params),
    v5plus: ({ decisionState, deps, params }) => chooseGukjinHeuristicV5Plus(decisionState, playerKey, deps, params),
    v5: ({ decisionState, deps, params }) => chooseGukjinHeuristicV5(decisionState, playerKey, deps, params),
    v4: ({ decisionState, deps }) => chooseGukjinHeuristicV4(decisionState, playerKey, deps),
    v3: ({ decisionState, deps }) => chooseGukjinHeuristicV3(decisionState, playerKey, deps)
  });
}

function shouldPresidentStopByPolicy(state, playerKey, policy = DEFAULT_BOT_POLICY) {
  const ctx = resolveHeuristicDecisionContext(state, playerKey, policy);
  return dispatchHeuristicPolicyCall(ctx, {
    v7: ({ decisionState, deps, params }) => shouldPresidentStopV7(decisionState, playerKey, deps, params),
    v6: ({ decisionState, deps, params }) => shouldPresidentStopV6(decisionState, playerKey, deps, params),
    v5plus: ({ decisionState, deps, params }) => shouldPresidentStopV5Plus(decisionState, playerKey, deps, params),
    v5: ({ decisionState, deps, params }) => shouldPresidentStopV5(decisionState, playerKey, deps, params),
    v4: ({ decisionState, deps }) => shouldPresidentStopV4(decisionState, playerKey, deps),
    v3: ({ decisionState, deps }) => shouldPresidentStopV3(decisionState, playerKey, deps)
  });
}

function chooseMatchByPolicy(state, playerKey, policy = DEFAULT_BOT_POLICY) {
  const ctx = resolveHeuristicDecisionContext(state, playerKey, policy);
  return dispatchHeuristicPolicyCall(ctx, {
    v7: ({ decisionState, deps, params }) => chooseMatchHeuristicV7(decisionState, playerKey, deps, params),
    v6: ({ decisionState, deps, params }) => chooseMatchHeuristicV6(decisionState, playerKey, deps, params),
    v5plus: ({ decisionState, deps, params }) => chooseMatchHeuristicV5Plus(decisionState, playerKey, deps, params),
    v5: ({ decisionState, deps, params }) => chooseMatchHeuristicV5(decisionState, playerKey, deps, params),
    v4: ({ decisionState, deps }) => chooseMatchHeuristicV4(decisionState, playerKey, deps),
    v3: ({ decisionState, deps }) => chooseMatchHeuristicV3(decisionState, playerKey, deps)
  });
}

function shouldGoByPolicy(state, playerKey, policy = DEFAULT_BOT_POLICY) {
  const ctx = resolveHeuristicDecisionContext(state, playerKey, policy);
  return dispatchHeuristicPolicyCall(ctx, {
    v7: ({ decisionState, deps, params }) => shouldGoV7(decisionState, playerKey, deps, params),
    v6: ({ decisionState, deps, params }) => shouldGoV6(decisionState, playerKey, deps, params),
    v5plus: ({ decisionState, deps, params }) => shouldGoV5Plus(decisionState, playerKey, deps, params),
    v5: ({ decisionState, deps, params }) => shouldGoV5(decisionState, playerKey, deps, params),
    v4: ({ decisionState, deps }) => shouldGoV4(decisionState, playerKey, deps),
    v3: ({ decisionState, deps }) => shouldGoV3(decisionState, playerKey, deps)
  });
}

function selectBombMonthByPolicy(state, playerKey, bombMonths, policy = DEFAULT_BOT_POLICY) {
  const ctx = resolveHeuristicDecisionContext(state, playerKey, policy);
  return dispatchHeuristicPolicyCall(ctx, {
    v7: ({ decisionState, deps }) => selectBombMonthV7(decisionState, playerKey, bombMonths, deps),
    v6: ({ decisionState, deps }) => selectBombMonthV6(decisionState, playerKey, bombMonths, deps),
    v5plus: ({ decisionState, deps }) => selectBombMonthV5Plus(decisionState, playerKey, bombMonths, deps),
    v5: ({ decisionState, deps }) => selectBombMonthV5(decisionState, playerKey, bombMonths, deps),
    v4: ({ decisionState, deps }) => selectBombMonthV4(decisionState, playerKey, bombMonths, deps),
    v3: ({ decisionState, deps }) => selectBombMonthV3(decisionState, playerKey, bombMonths, deps)
  });
}

function shouldBombByPolicy(state, playerKey, bombMonths, policy = DEFAULT_BOT_POLICY) {
  const ctx = resolveHeuristicDecisionContext(state, playerKey, policy);
  return dispatchHeuristicPolicyCall(ctx, {
    v7: ({ decisionState, deps, params }) => shouldBombV7(decisionState, playerKey, bombMonths, deps, params),
    v6: ({ decisionState, deps, params }) => shouldBombV6(decisionState, playerKey, bombMonths, deps, params),
    v5plus: ({ decisionState, deps, params }) => shouldBombV5Plus(decisionState, playerKey, bombMonths, deps, params),
    v5: ({ decisionState, deps, params }) => shouldBombV5(decisionState, playerKey, bombMonths, deps, params),
    v4: ({ decisionState, deps }) => shouldBombV4(decisionState, playerKey, bombMonths, deps),
    v3: ({ decisionState, deps }) => shouldBombV3(decisionState, playerKey, bombMonths, deps)
  });
}

function decideShakingByPolicy(state, playerKey, shakingMonths, policy = DEFAULT_BOT_POLICY) {
  const ctx = resolveHeuristicDecisionContext(state, playerKey, policy);
  return dispatchHeuristicPolicyCall(ctx, {
    v7: ({ decisionState, deps, params }) => decideShakingV7(decisionState, playerKey, shakingMonths, deps, params),
    v6: ({ decisionState, deps, params }) => decideShakingV6(decisionState, playerKey, shakingMonths, deps, params),
    v5plus: ({ decisionState, deps, params }) => decideShakingV5Plus(decisionState, playerKey, shakingMonths, deps, params),
    v5: ({ decisionState, deps, params }) => decideShakingV5(decisionState, playerKey, shakingMonths, deps, params),
    v4: ({ decisionState, deps }) => decideShakingV4(decisionState, playerKey, shakingMonths, deps),
    v3: ({ decisionState, deps }) => decideShakingV3(decisionState, playerKey, shakingMonths, deps)
  });
}

function rankHandCardsByPolicy(state, playerKey, policy = DEFAULT_BOT_POLICY) {
  const ctx = resolveHeuristicDecisionContext(state, playerKey, policy);
  return dispatchHeuristicPolicyCall(ctx, {
    v7: ({ decisionState, deps, params }) => rankHandCardsV7(decisionState, playerKey, deps, params),
    v6: ({ decisionState, deps, params }) => rankHandCardsV6(decisionState, playerKey, deps, params),
    v5plus: ({ decisionState, deps, params }) => rankHandCardsV5Plus(decisionState, playerKey, deps, params),
    v5: ({ decisionState, deps, params }) => rankHandCardsV5(decisionState, playerKey, deps, params),
    v4: ({ decisionState, deps }) => rankHandCardsV4(decisionState, playerKey, deps),
    v3: ({ decisionState, deps }) => rankHandCardsV3(decisionState, playerKey, deps)
  });
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


function pickRandom(arr) {
  if (!arr || arr.length === 0) return null;
  return arr[Math.floor(Math.random() * arr.length)];
}

function getActionPlayerKeyLocal(state) {
  if (state?.phase === "playing") return state.currentTurn || null;
  if (state?.phase === "go-stop") return state.pendingGoStop || null;
  if (state?.phase === "select-match") return state.pendingMatch?.playerKey || null;
  if (state?.phase === "president-choice") return state.pendingPresident?.playerKey || null;
  if (state?.phase === "shaking-confirm") return state.pendingShakingConfirm?.playerKey || null;
  if (state?.phase === "gukjin-choice") return state.pendingGukjinChoice?.playerKey || null;
  return null;
}

function rolloutStateUtilityV6(state, rootPlayerKey, deps = HEURISTIC_V6_DEPS) {
  if (!state || !rootPlayerKey) return 0;
  const opp = otherPlayerKey(rootPlayerKey);
  const myScore = currentScoreTotal(state, rootPlayerKey);
  const oppScore = currentScoreTotal(state, opp);
  const myPi = capturedCountByCategory(state.players?.[rootPlayerKey], "junk");
  const oppPi = capturedCountByCategory(state.players?.[opp], "junk");
  const myFive = scoringFiveCount(state.players?.[rootPlayerKey]);
  const oppFive = scoringFiveCount(state.players?.[opp]);
  const oppThreat = Number(deps?.opponentThreatScore?.(state, rootPlayerKey) ?? 0);
  const nextThreat = Number(deps?.nextTurnThreatScore?.(state, rootPlayerKey) ?? 0);
  const deckCount = Number(state?.deck?.length || 0);
  const carry = Number(state?.carryOverMultiplier || 1);
  const goldDiff = Number(state?.players?.[rootPlayerKey]?.gold || 0) - Number(state?.players?.[opp]?.gold || 0);

  let utility =
    (myScore - oppScore) * 1.0 +
    (myPi - oppPi) * 0.12 +
    (myFive - oppFive) * 0.2 +
    goldDiff * 0.004 -
    oppThreat * 0.55 -
    nextThreat * 0.35;

  if (carry >= 2) utility -= 0.12;
  if (deckCount <= 6 && myScore < oppScore) utility -= 0.06;
  if (deckCount <= 6 && myScore >= oppScore) utility += 0.04;
  return utility;
}

function rolloutBotPlayV6NoRollout(state, actor) {
  const params = { ...HEURISTIC_V6_PARAMS, rolloutEnabled: 0 };
  const fair = createV6FairDecisionContext(state, actor);
  const decisionState = fair.publicState;
  const fairDeps = fair.deps;
  if (state.phase === "gukjin-choice" && state.pendingGukjinChoice?.playerKey === actor) {
    const mode = chooseGukjinHeuristicV6(decisionState, actor, fairDeps, params);
    return chooseGukjinMode(state, actor, mode);
  }
  if (state.phase === "president-choice" && state.pendingPresident?.playerKey === actor) {
    const stop = shouldPresidentStopV6(decisionState, actor, fairDeps, params);
    return stop ? choosePresidentStop(state, actor) : choosePresidentHold(state, actor);
  }
  if (state.phase === "select-match" && state.pendingMatch?.playerKey === actor) {
    const choiceId = chooseMatchHeuristicV6(decisionState, actor, fairDeps, params);
    return choiceId ? chooseMatch(state, choiceId) : state;
  }
  if (state.phase === "go-stop" && state.pendingGoStop === actor) {
    const go = shouldGoV6(decisionState, actor, fairDeps, params);
    return go ? chooseGo(state, actor) : chooseStop(state, actor);
  }
  if (state.phase === "playing" && state.currentTurn === actor) {
    const ranked = rankHandCardsV6(decisionState, actor, fairDeps, params);
    const cardId = ranked?.[0]?.card?.id;
    return cardId ? playTurn(state, cardId) : state;
  }
  return state;
}

function rolloutAdvanceUntilTurnEnds(state, turnOwner, maxSteps = 28) {
  let next = state;
  for (let step = 0; step < maxSteps; step += 1) {
    if (!next || next.phase === "resolution") break;
    const actor = getActionPlayerKeyLocal(next);
    if (!actor) break;
    const updated = rolloutBotPlayV6NoRollout(next, actor);
    if (!updated || updated === next) break;
    next = updated;
    if (next.phase === "playing" && next.currentTurn && next.currentTurn !== turnOwner) break;
  }
  return next;
}

function rolloutCardValueV6(state, playerKey, cardId, options = {}) {
  if (!state || state.phase !== "playing" || state.currentTurn !== playerKey) return null;
  if (!cardId) return null;
  const maxSteps = Math.max(8, Number(options.maxSteps || 28));
  const samples = Math.max(1, Math.floor(Number(options.samples || 5)));
  const observerKey = options.observerKey || playerKey;
  const utilityDeps = createHeuristicV6FairDeps(playerKey);
  let sum = 0;
  let count = 0;

  for (let i = 0; i < samples; i += 1) {
    const sampled = determinizeUnknownCards(state, observerKey);
    let next = playTurn(sampled, cardId);
    if (!next || next === sampled) continue;
    if (next.phase !== "resolution") {
      next = rolloutAdvanceUntilTurnEnds(next, playerKey, maxSteps);
    }
    if (next.phase !== "resolution") {
      const opp = otherPlayerKey(playerKey);
      const actor = getActionPlayerKeyLocal(next);
      if (actor === opp || (next.phase === "playing" && next.currentTurn === opp)) {
        next = rolloutAdvanceUntilTurnEnds(next, opp, maxSteps);
      }
    }
    const utility = rolloutStateUtilityV6(next, playerKey, utilityDeps);
    if (!Number.isFinite(utility)) continue;
    sum += utility;
    count += 1;
  }

  if (count <= 0) return null;
  return sum / count;
}

function rolloutGoStopValueV6(state, playerKey, chooseGoFlag, options = {}) {
  if (!state || state.phase !== "go-stop" || state.pendingGoStop !== playerKey) return null;
  const maxSteps = Math.max(8, Number(options.maxSteps || 28));
  const samples = Math.max(1, Math.floor(Number(options.samples || 5)));
  const observerKey = options.observerKey || playerKey;
  const utilityDeps = createHeuristicV6FairDeps(playerKey);
  let sum = 0;
  let count = 0;

  for (let i = 0; i < samples; i += 1) {
    const sampled = determinizeUnknownCards(state, observerKey);
    let next = chooseGoFlag ? chooseGo(sampled, playerKey) : chooseStop(sampled, playerKey);
    if (!next || next === sampled) continue;
    if (next.phase !== "resolution") {
      const actor = getActionPlayerKeyLocal(next);
      if (actor === playerKey) {
        next = rolloutAdvanceUntilTurnEnds(next, playerKey, maxSteps);
      }
    }
    if (next.phase !== "resolution") {
      const opp = otherPlayerKey(playerKey);
      const actor = getActionPlayerKeyLocal(next);
      if (actor === opp || (next.phase === "playing" && next.currentTurn === opp)) {
        next = rolloutAdvanceUntilTurnEnds(next, opp, maxSteps);
      }
    }
    const utility = rolloutStateUtilityV6(next, playerKey, utilityDeps);
    if (!Number.isFinite(utility)) continue;
    sum += utility;
    count += 1;
  }

  if (count <= 0) return null;
  return sum / count;
}
