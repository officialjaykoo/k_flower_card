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
} from "../engine/index.js";
import { countComboTag } from "../engine/combos.js";
import {
  DEFAULT_BOT_POLICY,
  POLICY_HEURISTIC_J2,
  POLICY_HEURISTIC_CL,
  POLICY_HEURISTIC_NEXG,
  POLICY_HEURISTIC_GPT,
  POLICY_HEURISTIC_GEMINI,
  normalizeBotPolicy
} from "./policies.js";
import {
  chooseGukjinHeuristicJ2,
  chooseMatchHeuristicJ2,
  decideShakingJ2,
  rankHandCardsJ2,
  selectBombMonthJ2,
  shouldBombJ2,
  shouldGoJ2,
  shouldPresidentStopJ2
} from "../heuristics/heuristicJ2.js";
import {
  chooseGukjinHeuristicCL,
  chooseMatchHeuristicCL,
  decideShakingCL,
  rankHandCardsCL,
  selectBombMonthCL,
  shouldBombCL,
  shouldGoCL,
  shouldPresidentStopCL,
  DEFAULT_PARAMS as V5_DEFAULT_PARAMS
} from "../heuristics/heuristicCL.js";
import {
  rankHandCardsNEXg,
  chooseMatchHeuristicNEXg,
  chooseGukjinHeuristicNEXg,
  shouldPresidentStopNEXg,
  shouldGoNEXg,
  selectBombMonthNEXg,
  shouldBombNEXg,
  decideShakingNEXg,
  DEFAULT_PARAMS as NEXG_DEFAULT_PARAMS
} from "../heuristics/heuristicNEXg.js";
import {
  chooseGukjinHeuristicGPT,
  chooseMatchHeuristicGPT,
  decideShakingGPT,
  rankHandCardsGPT,
  selectBombMonthGPT,
  shouldBombGPT,
  shouldGoGPT,
  shouldPresidentStopGPT,
  DEFAULT_PARAMS as V6_DEFAULT_PARAMS
} from "../heuristics/heuristicGPT.js";
import {
  chooseGukjinHeuristicGemini,
  chooseMatchHeuristicGemini,
  decideShakingGemini,
  rankHandCardsGemini,
  selectBombMonthGemini,
  shouldBombGemini,
  shouldGoGemini,
  shouldPresidentStopGemini,
  DEFAULT_PARAMS as V7_DEFAULT_PARAMS
} from "../heuristics/heuristicGemini.js";
import {
  GUKJIN_CARD_ID,
  boardHighValueThreatForPlayer,
  boardMatchesByMonth,
  cardCaptureValue,
  capturedCountByCategory,
  capturedMonthCounts,
  clamp01,
  countKnownMonthCards,
  junkPiValue,
  monthBoardGain,
  monthCounts,
  monthStrategicPriority,
  otherPlayerKey,
  ownComboOpportunityScore,
  pickRandom,
  selectBestMonth,
  shakingImmediateGainScore
} from "./heuristicUtils.js";
import {
  analyzeGameContext,
  analyzeGukjinBranches,
  blockingMonthsAgainst,
  blockingUrgencyByMonth,
  buildDynamicWeights,
  canBankruptOpponentByStop,
  canBankruptOpponentByStopGeminiProxy,
  checkOpponentJokboProgress,
  estimateDangerMonthRisk,
  estimateJokboExpectedPotential,
  estimateOpponentImmediateGainIfDiscard,
  estimateReleasePunishProb,
  getFirstTurnDoublePiPlan,
  getMissingComboCards,
  goldRiskProfile,
  isHighImpactBomb,
  isHighImpactShaking,
  isRiskOfPuk,
  matchableMonthCountForPlayer,
  nextTurnThreatScore,
  opponentThreatScore
} from "./heuristicAnalysis.js";
import {
  boardHighValueThreatForPlayerPublic,
  createHeuristicGPTFairDeps,
  createObserverSafeHeuristicDeps,
  createPublicState,
  estimateJokboExpectedPotentialForObserver,
  estimateOpponentImmediateGainIfDiscardPublic,
  knownMonthCountForObserver,
  matchableMonthCountForPlayerPublic,
  nextTurnThreatScorePublic,
  opponentThreatScorePublic,
  shakingImmediateGainScoreForObserver
} from "./heuristicPublicState.js";

/* ============================================================================
 * Heuristic policy engine (v3-v7)
 * Quick reading order:
 * 1) Runtime params + exported entry points
 * 2) Shared state/score utilities
 * 3) Fair-teacher public-state helpers (gpt)
 * 4) Policy dispatch (j1/j2/cl/nexg/gpt/gemini)
 * 5) Rollout helpers used by gpt
 * ========================================================================== */
/* 1) Runtime parameter loaders */
function loadHeuristicParams(envVarName, defaultParams) {
  try {
    const raw = (typeof process !== "undefined" && process.env?.[envVarName]) || "";
    if (!raw) return { ...defaultParams };
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return { ...defaultParams, ...parsed };
    }
  } catch {
    // Fall through to defaults on parse/runtime errors.
  }
  return { ...defaultParams };
}

const HEURISTIC_CL_PARAMS = Object.freeze(loadHeuristicParams("HEURISTIC_CL_PARAMS", V5_DEFAULT_PARAMS));
const HEURISTIC_NEXG_PARAMS = Object.freeze(loadHeuristicParams("HEURISTIC_NEXG_PARAMS", NEXG_DEFAULT_PARAMS));
const HEURISTIC_GPT_PARAMS = Object.freeze(loadHeuristicParams("HEURISTIC_GPT_PARAMS", V6_DEFAULT_PARAMS));
const HEURISTIC_GEMINI_PARAMS = Object.freeze(loadHeuristicParams("HEURISTIC_GEMINI_PARAMS", V7_DEFAULT_PARAMS));

export {
  DEFAULT_BOT_POLICY,
  POLICY_HEURISTIC_J2,
  POLICY_HEURISTIC_CL,
  POLICY_HEURISTIC_NEXG,
  POLICY_HEURISTIC_GPT,
  POLICY_HEURISTIC_GEMINI
};

/* 2) Public entry points */
export function botChooseCard(state, playerKey, policy = DEFAULT_BOT_POLICY, heuristicParams = null) {
  const player = state.players[playerKey];
  if (!player || player.hand.length === 0) return null;
  const ranked = rankHandCardsByPolicy(state, playerKey, policy, heuristicParams);
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

function chooseShakingCardIdForMonth(state, playerKey, month, policy = DEFAULT_BOT_POLICY, heuristicParams = null) {
  const player = state.players?.[playerKey];
  if (!player?.hand?.length) return null;
  const targetMonth = Number(month);
  const monthCards = player.hand.filter((c) => c && !c.passCard && Number(c.month) === targetMonth);
  if (!monthCards.length) return null;

  const effectivePolicy = normalizeBotPolicy(policy);
  const ranked = rankHandCardsByPolicy(state, playerKey, effectivePolicy, heuristicParams);
  if (effectivePolicy === POLICY_HEURISTIC_GPT) {
    const v6Picked = chooseShakingDiscardCardIdGPT(state, playerKey, monthCards, ranked);
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

function chooseShakingDiscardCardIdGPT(state, playerKey, monthCards, ranked) {
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
  const heuristicParams =
    options?.heuristicParams && typeof options.heuristicParams === "object"
      ? options.heuristicParams
      : null;
  if (state.phase === "gukjin-choice" && state.pendingGukjinChoice?.playerKey === playerKey) {
    return chooseGukjinMode(state, playerKey, chooseGukjinByPolicy(state, playerKey, policy, heuristicParams));
  }

  if (state.phase === "president-choice" && state.pendingPresident?.playerKey === playerKey) {
    const shouldStop = shouldPresidentStopByPolicy(state, playerKey, policy, heuristicParams);
    return shouldStop ? choosePresidentStop(state, playerKey) : choosePresidentHold(state, playerKey);
  }

  if (state.phase === "select-match" && state.pendingMatch?.playerKey === playerKey) {
    const choiceId = chooseMatchByPolicy(state, playerKey, policy, heuristicParams);
    return choiceId ? chooseMatch(state, choiceId) : state;
  }

  if (state.phase === "go-stop" && state.pendingGoStop === playerKey) {
    return shouldGoByPolicy(state, playerKey, policy, heuristicParams)
      ? chooseGo(state, playerKey)
      : chooseStop(state, playerKey);
  }

  if (state.phase === "playing" && state.currentTurn === playerKey) {
    const bombMonths = getDeclarableBombMonths(state, playerKey);
    if (bombMonths.length > 0 && shouldBombByPolicy(state, playerKey, bombMonths, policy, heuristicParams)) {
      const month = selectBombMonthByPolicy(state, playerKey, bombMonths, policy, heuristicParams);
      return declareBomb(state, playerKey, month ?? selectBestMonth(state, bombMonths));
    }
    const shakingMonths = getDeclarableShakingMonths(state, playerKey);
    if (shakingMonths.length > 0) {
      const shakeDecision = decideShakingByPolicy(state, playerKey, shakingMonths, policy, heuristicParams);
      if (shakeDecision.allow && shakeDecision.month != null) {
        const shakeCardId = chooseShakingCardIdForMonth(
          state,
          playerKey,
          shakeDecision.month,
          policy,
          heuristicParams
        );
        const declared = declareShaking(state, playerKey, shakeDecision.month);
        if (!declared || declared === state) return state;
        if (!shakeCardId) return declared;
        const played = playTurn(declared, shakeCardId);
        return played || declared;
      }
    }
    const cardId = botChooseCard(state, playerKey, policy, heuristicParams);
    if (!cardId) return state;
    return playTurn(state, cardId);
  }

  return state;
}

const HEURISTIC_J2_DEPS = Object.freeze({
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

const HEURISTIC_CL_DEPS = Object.freeze({
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

const HEURISTIC_GPT_DEPS = Object.freeze({
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
  shakingImmediateGainScore
});

const HEURISTIC_GEMINI_DEPS = Object.freeze({
  ...HEURISTIC_CL_DEPS,
  canBankruptOpponentByStop: canBankruptOpponentByStopGeminiProxy,
  estimateDangerMonthRisk,
  monthBoardGain,
  opponentThreatScore
});

const POLICY_HANDLER_KEYS = Object.freeze({
  [POLICY_HEURISTIC_GEMINI]: "gemini",
  [POLICY_HEURISTIC_GPT]: "gpt",
  [POLICY_HEURISTIC_NEXG]: "nexg",
  [POLICY_HEURISTIC_CL]: "cl",
  [POLICY_HEURISTIC_J2]: "j2"
});

/* 5) Policy dispatch layer */
function resolveHeuristicDecisionContext(
  state,
  playerKey,
  policy = DEFAULT_BOT_POLICY,
  heuristicParams = null
) {
  const resolvedPolicy = normalizeBotPolicy(policy);
  const paramsOverride =
    heuristicParams && typeof heuristicParams === "object" && !Array.isArray(heuristicParams)
      ? heuristicParams
      : null;
  const decisionState = createPublicState(state, playerKey);
  if (resolvedPolicy === POLICY_HEURISTIC_GEMINI) {
    return {
      policy: POLICY_HEURISTIC_GEMINI,
      decisionState,
      deps: createObserverSafeHeuristicDeps(HEURISTIC_GEMINI_DEPS, playerKey),
      params: paramsOverride ? { ...HEURISTIC_GEMINI_PARAMS, ...paramsOverride } : HEURISTIC_GEMINI_PARAMS
    };
  }
  if (resolvedPolicy === POLICY_HEURISTIC_GPT) {
    return {
      policy: POLICY_HEURISTIC_GPT,
      decisionState,
      deps: createHeuristicGPTFairDeps(HEURISTIC_GPT_DEPS, playerKey),
      params: paramsOverride ? { ...HEURISTIC_GPT_PARAMS, ...paramsOverride } : HEURISTIC_GPT_PARAMS
    };
  }

  if (resolvedPolicy === POLICY_HEURISTIC_NEXG) {
    return {
      policy: POLICY_HEURISTIC_NEXG,
      decisionState,
      deps: createObserverSafeHeuristicDeps(HEURISTIC_CL_DEPS, playerKey),
      params: paramsOverride ? { ...HEURISTIC_NEXG_PARAMS, ...paramsOverride } : HEURISTIC_NEXG_PARAMS
    };
  }
  if (resolvedPolicy === POLICY_HEURISTIC_CL) {
    return {
      policy: POLICY_HEURISTIC_CL,
      decisionState,
      deps: createObserverSafeHeuristicDeps(HEURISTIC_CL_DEPS, playerKey),
      params: paramsOverride ? { ...HEURISTIC_CL_PARAMS, ...paramsOverride } : HEURISTIC_CL_PARAMS
    };
  }
  if (resolvedPolicy === POLICY_HEURISTIC_J2) {
    return {
      policy: POLICY_HEURISTIC_J2,
      decisionState,
      deps: createObserverSafeHeuristicDeps(HEURISTIC_J2_DEPS, playerKey),
      params: null
    };
  }
  return {
    policy: POLICY_HEURISTIC_CL,
    decisionState,
    deps: createObserverSafeHeuristicDeps(HEURISTIC_CL_DEPS, playerKey),
    params: paramsOverride ? { ...HEURISTIC_CL_PARAMS, ...paramsOverride } : HEURISTIC_CL_PARAMS
  };
}

function dispatchHeuristicPolicyCall(ctx, handlers) {
  const handlerKey = POLICY_HANDLER_KEYS[ctx.policy] || "cl";
  const handler = handlers[handlerKey];
  if (typeof handler === "function") return handler(ctx);
  return handlers.cl(ctx);
}

function callPolicyHandler(state, playerKey, policy, heuristicParams, handlers) {
  const ctx = resolveHeuristicDecisionContext(state, playerKey, policy, heuristicParams);
  return dispatchHeuristicPolicyCall(ctx, handlers);
}

function chooseGukjinByPolicy(state, playerKey, policy = DEFAULT_BOT_POLICY, heuristicParams = null) {
  return callPolicyHandler(state, playerKey, policy, heuristicParams, {
    gemini: ({ decisionState, deps, params }) => chooseGukjinHeuristicGemini(decisionState, playerKey, deps, params),
    gpt: ({ decisionState, deps, params }) => chooseGukjinHeuristicGPT(decisionState, playerKey, deps, params),
    nexg: ({ decisionState, deps, params }) => chooseGukjinHeuristicNEXg(decisionState, playerKey, deps, params),
    cl: ({ decisionState, deps, params }) => chooseGukjinHeuristicCL(decisionState, playerKey, deps, params),
    j2: ({ decisionState, deps }) => chooseGukjinHeuristicJ2(decisionState, playerKey, deps)
  });
}

function shouldPresidentStopByPolicy(state, playerKey, policy = DEFAULT_BOT_POLICY, heuristicParams = null) {
  return callPolicyHandler(state, playerKey, policy, heuristicParams, {
    gemini: ({ decisionState, deps, params }) => shouldPresidentStopGemini(decisionState, playerKey, deps, params),
    gpt: ({ decisionState, deps, params }) => shouldPresidentStopGPT(decisionState, playerKey, deps, params),
    nexg: ({ decisionState, deps, params }) => shouldPresidentStopNEXg(decisionState, playerKey, deps, params),
    cl: ({ decisionState, deps, params }) => shouldPresidentStopCL(decisionState, playerKey, deps, params),
    j2: ({ decisionState, deps }) => shouldPresidentStopJ2(decisionState, playerKey, deps)
  });
}

function chooseMatchByPolicy(state, playerKey, policy = DEFAULT_BOT_POLICY, heuristicParams = null) {
  return callPolicyHandler(state, playerKey, policy, heuristicParams, {
    gemini: ({ decisionState, deps, params }) => chooseMatchHeuristicGemini(decisionState, playerKey, deps, params),
    gpt: ({ decisionState, deps, params }) => chooseMatchHeuristicGPT(decisionState, playerKey, deps, params),
    nexg: ({ decisionState, deps, params }) => chooseMatchHeuristicNEXg(decisionState, playerKey, deps, params),
    cl: ({ decisionState, deps, params }) => chooseMatchHeuristicCL(decisionState, playerKey, deps, params),
    j2: ({ decisionState, deps }) => chooseMatchHeuristicJ2(decisionState, playerKey, deps)
  });
}

function shouldGoByPolicy(state, playerKey, policy = DEFAULT_BOT_POLICY, heuristicParams = null) {
  return callPolicyHandler(state, playerKey, policy, heuristicParams, {
    gemini: ({ decisionState, deps, params }) => shouldGoGemini(decisionState, playerKey, deps, params),
    gpt: ({ decisionState, deps, params }) => shouldGoGPT(decisionState, playerKey, deps, params),
    nexg: ({ decisionState, deps, params }) => shouldGoNEXg(decisionState, playerKey, deps, params),
    cl: ({ decisionState, deps, params }) => shouldGoCL(decisionState, playerKey, deps, params),
    j2: ({ decisionState, deps }) => shouldGoJ2(decisionState, playerKey, deps)
  });
}

function selectBombMonthByPolicy(
  state,
  playerKey,
  bombMonths,
  policy = DEFAULT_BOT_POLICY,
  heuristicParams = null
) {
  return callPolicyHandler(state, playerKey, policy, heuristicParams, {
    gemini: ({ decisionState, deps }) => selectBombMonthGemini(decisionState, playerKey, bombMonths, deps),
    gpt: ({ decisionState, deps }) => selectBombMonthGPT(decisionState, playerKey, bombMonths, deps),
    nexg: ({ decisionState, deps }) => selectBombMonthNEXg(decisionState, playerKey, bombMonths, deps),
    cl: ({ decisionState, deps }) => selectBombMonthCL(decisionState, playerKey, bombMonths, deps),
    j2: ({ decisionState, deps }) => selectBombMonthJ2(decisionState, playerKey, bombMonths, deps)
  });
}

function shouldBombByPolicy(
  state,
  playerKey,
  bombMonths,
  policy = DEFAULT_BOT_POLICY,
  heuristicParams = null
) {
  return callPolicyHandler(state, playerKey, policy, heuristicParams, {
    gemini: ({ decisionState, deps, params }) => shouldBombGemini(decisionState, playerKey, bombMonths, deps, params),
    gpt: ({ decisionState, deps, params }) => shouldBombGPT(decisionState, playerKey, bombMonths, deps, params),
    nexg: ({ decisionState, deps, params }) => shouldBombNEXg(decisionState, playerKey, bombMonths, deps, params),
    cl: ({ decisionState, deps, params }) => shouldBombCL(decisionState, playerKey, bombMonths, deps, params),
    j2: ({ decisionState, deps }) => shouldBombJ2(decisionState, playerKey, bombMonths, deps)
  });
}

function decideShakingByPolicy(
  state,
  playerKey,
  shakingMonths,
  policy = DEFAULT_BOT_POLICY,
  heuristicParams = null
) {
  return callPolicyHandler(state, playerKey, policy, heuristicParams, {
    gemini: ({ decisionState, deps, params }) => decideShakingGemini(decisionState, playerKey, shakingMonths, deps, params),
    gpt: ({ decisionState, deps, params }) => decideShakingGPT(decisionState, playerKey, shakingMonths, deps, params),
    nexg: ({ decisionState, deps, params }) => decideShakingNEXg(decisionState, playerKey, shakingMonths, deps, params),
    cl: ({ decisionState, deps, params }) => decideShakingCL(decisionState, playerKey, shakingMonths, deps, params),
    j2: ({ decisionState, deps }) => decideShakingJ2(decisionState, playerKey, shakingMonths, deps)
  });
}

function rankHandCardsByPolicy(state, playerKey, policy = DEFAULT_BOT_POLICY, heuristicParams = null) {
  return callPolicyHandler(state, playerKey, policy, heuristicParams, {
    gemini: ({ decisionState, deps, params }) => rankHandCardsGemini(decisionState, playerKey, deps, params),
    gpt: ({ decisionState, deps, params }) => rankHandCardsGPT(decisionState, playerKey, deps, params),
    nexg: ({ decisionState, deps, params }) => rankHandCardsNEXg(decisionState, playerKey, deps, params),
    cl: ({ decisionState, deps, params }) => rankHandCardsCL(decisionState, playerKey, deps, params),
    j2: ({ decisionState, deps }) => rankHandCardsJ2(decisionState, playerKey, deps)
  });
}

