import {
  calculateScore,
  scoringPiCount,
  getDeclarableShakingMonths,
  getDeclarableBombMonths,
  chooseGo,
  chooseStop
} from "../engine/index.js";
import { STARTING_GOLD } from "../engine/economy.js";
import {
  applyAction,
  canonicalOptionAction,
  legalCandidatesForDecision,
  normalizeDecisionCandidate,
  parsePlaySpecialCandidate,
  selectPool
} from "./evalCore/sharedGameHelpers.js";

/* ============================================================================
 * NEAT model policy runtime
 * - compile genome once and cache
 * - score legal candidates
 * - convert score to action/state transition
 * ========================================================================== */
const NEAT_MODEL_FORMAT = "neat_python_genome_v1";
const IQN_GO_STOP_RUNTIME_FORMAT = "iqn_go_stop_runtime_v1";
const COMPILED_NEAT_CACHE = new WeakMap();
const GO_STOP_OPTION_ONE_HOT = [1, 0, 0, 0];
const NON_BRIGHT_KWANG_ID = "L0";
const NEAT_COMPACT_FEATURES = 16;
const NEAT_OUT_PLAY = 0;
const NEAT_OUT_MATCH = 1;
const NEAT_OUT_GO = 2;
const NEAT_OUT_STOP = 3;
const NEAT_OUT_SHAKING_YES = 4;
const NEAT_OUT_SHAKING_NO = 5;
const NEAT_OUT_PRESIDENT_STOP = 6;
const NEAT_OUT_PRESIDENT_HOLD = 7;
const NEAT_OUT_GUKJIN_FIVE = 8;
const NEAT_OUT_GUKJIN_JUNK = 9;
const LEGACY_NEAT_COMPACT_FEATURES = 13;
const IQN_GO_STOP_BASE_FEATURES = 16;
const IQN_GO_STOP_PAYLOAD_DIM = 10;

/* 2) Feature extraction helpers */
function clamp01(x) {
  const v = Number(x || 0);
  if (v <= 0) return 0;
  if (v >= 1) return 1;
  return v;
}

function clampRange(x, minValue, maxValue) {
  const v = Number(x || 0);
  const lo = Number(minValue || 0);
  const hi = Number(maxValue || 0);
  if (v <= lo) return lo;
  if (v >= hi) return hi;
  return v;
}

function tanhNorm(x, scale) {
  const s = Math.max(1e-6, Number(scale || 1));
  return Math.tanh(Number(x || 0) / s);
}

function resolveInitialGoldBase(state) {
  const configured = Number(state?.initialGoldBase);
  if (Number.isFinite(configured) && configured > 0) return configured;
  return STARTING_GOLD;
}

function findCardById(cards, cardId) {
  const id = String(cardId || "");
  if (!Array.isArray(cards)) return null;
  return cards.find((c) => String(c?.id || "") === id) || null;
}

function optionCode(action) {
  const special = parsePlaySpecialCandidate(action);
  if (special?.kind === "shake_start") return 0.9;
  if (special?.kind === "bomb") return 0.95;
  const a = canonicalOptionAction(action);
  const map = {
    go: 1,
    stop: 2,
    shaking_yes: 3,
    shaking_no: 4,
    president_stop: 5,
    president_hold: 6,
    five: 7,
    junk: 8
  };
  return Number(map[a] || 0) / 8.0;
}

function resolveDecisionType(sp) {
  const cards = sp?.cards || null;
  const boardCardIds = sp?.boardCardIds || null;
  const options = sp?.options || null;
  return cards ? "play" : boardCardIds ? "match" : options ? "option" : null;
}

function candidateCard(state, actor, decisionType, candidate) {
  if (decisionType === "play") {
    const special = parsePlaySpecialCandidate(candidate);
    if (special?.kind === "shake_start") {
      return findCardById(state?.players?.[actor]?.hand || [], special.cardId);
    }
    if (special?.kind === "bomb") {
      return { id: String(candidate || ""), month: special.month, category: "junk", piValue: 0 };
    }
    return findCardById(state?.players?.[actor]?.hand || [], candidate);
  }
  if (decisionType === "match") {
    return findCardById(state?.board || [], candidate);
  }
  return null;
}

function compactCandidatePiNorm(card) {
  const piValue = Math.max(0, Number(card?.piValue || 0));
  const stealPi = Math.max(0, Number(card?.bonus?.stealPi || 0));
  return clamp01((piValue + stealPi) / 4.0);
}

function countCardsByMonth(cards, month) {
  const targetMonth = Number(month || 0);
  if (!Array.isArray(cards) || targetMonth <= 0) return 0;
  let count = 0;
  for (const card of cards) {
    if (Number(card?.month || 0) === targetMonth) count += 1;
  }
  return count;
}

function hasComboTag(card, tag) {
  return Array.isArray(card?.comboTags) && card.comboTags.includes(tag);
}

function countCapturedComboTag(player, zone, tag) {
  const cards = player?.captured?.[zone] || [];
  if (!Array.isArray(cards)) return 0;
  const seen = new Set();
  let count = 0;
  for (const card of cards) {
    const id = String(card?.id || "");
    if (!id || seen.has(id)) continue;
    seen.add(id);
    if (hasComboTag(card, tag)) count += 1;
  }
  return count;
}

function isDoublePiCard(card) {
  if (!card) return false;
  const id = String(card?.id || "");
  const category = String(card?.category || "");
  const piValue = Number(card?.piValue || 0);
  if (id === "I0") return true;
  return category === "junk" && piValue >= 2;
}

function resolveCandidateMonth(state, actor, decisionType, card) {
  const cardMonth = Number(card?.month || 0);
  if (cardMonth >= 1) return cardMonth;
  if (decisionType === "option") {
    if (state?.phase === "shaking-confirm" && state?.pendingShakingConfirm?.playerKey === actor) {
      const pendingMonth = Number(state?.pendingShakingConfirm?.month || 0);
      if (pendingMonth >= 1) return pendingMonth;
    }
    if (state?.phase === "president-choice" && state?.pendingPresident?.playerKey === actor) {
      const pendingMonth = Number(state?.pendingPresident?.month || 0);
      if (pendingMonth >= 1) return pendingMonth;
    }
  }
  return 0;
}

function matchOpportunityDensity(state, month) {
  const boardMonthCount = countCardsByMonth(state?.board || [], month);
  return clamp01(boardMonthCount / 3.0);
}

function immediateMatchPossible(state, decisionType, month) {
  if (decisionType === "match") return 1;
  if (month <= 0) return 0;
  return countCardsByMonth(state?.board || [], month) > 0 ? 1 : 0;
}

function monthTotalCards(month) {
  const m = Number(month || 0);
  if (m >= 1 && m <= 12) return 4;
  if (m === 13) return 2;
  return 0;
}

function collectPublicShakingRevealIds(state, targetPlayerKey) {
  const ids = new Set();
  if (!state || !targetPlayerKey) return ids;

  const liveReveal = state?.shakingReveal;
  if (liveReveal?.playerKey === targetPlayerKey) {
    for (const card of liveReveal.cards || []) {
      const id = String(card?.id || "");
      if (id) ids.add(id);
    }
  }

  for (const entry of state?.kibo || []) {
    if (entry?.type !== "shaking_declare" || entry?.playerKey !== targetPlayerKey) continue;
    for (const card of entry?.revealCards || []) {
      const id = String(card?.id || "");
      if (id) ids.add(id);
    }
  }
  return ids;
}

function getPublicKnownOpponentHandCards(state, actor) {
  const opp = actor === "human" ? "ai" : "human";
  const revealIds = collectPublicShakingRevealIds(state, opp);
  if (revealIds.size <= 0) return [];
  const oppHand = state?.players?.[opp]?.hand || [];
  return oppHand.filter((card) => revealIds.has(String(card?.id || "")));
}

function collectKnownCardsForMonthRatio(state, actor) {
  const out = [];
  const pushAll = (cards) => {
    if (!Array.isArray(cards)) return;
    for (const card of cards) out.push(card);
  };

  pushAll(state?.board || []);
  pushAll(state?.players?.[actor]?.hand || []);
  for (const side of ["human", "ai"]) {
    const captured = state?.players?.[side]?.captured || {};
    pushAll(captured.kwang || []);
    pushAll(captured.five || []);
    pushAll(captured.ribbon || []);
    pushAll(captured.junk || []);
  }
  pushAll(getPublicKnownOpponentHandCards(state, actor));
  return out;
}

function candidatePublicKnownRatio(state, actor, month) {
  const total = monthTotalCards(month);
  if (total <= 0) return 0;
  const cards = collectKnownCardsForMonthRatio(state, actor);
  const seen = new Set();
  let known = 0;
  for (const card of cards) {
    const id = String(card?.id || "");
    if (!id || seen.has(id)) continue;
    seen.add(id);
    if (Number(card?.month || 0) === Number(month)) known += 1;
  }
  return clamp01(known / total);
}

function uniqueCapturedCards(player, zone) {
  const cards = player?.captured?.[zone] || [];
  if (!Array.isArray(cards)) return [];
  const seen = new Set();
  const out = [];
  for (const card of cards) {
    const id = String(card?.id || "");
    if (!id || seen.has(id)) continue;
    seen.add(id);
    out.push(card);
  }
  return out;
}

function uniqueCapturedCardCount(player, zone) {
  return uniqueCapturedCards(player, zone).length;
}

function compactKwangBaseScore(kwangCards) {
  const count = Array.isArray(kwangCards) ? kwangCards.length : 0;
  if (count < 3) return 0;
  if (count === 3) return kwangCards.some((c) => String(c?.id || "") === NON_BRIGHT_KWANG_ID) ? 2 : 3;
  if (count === 4) return 4;
  return 15;
}

function applyDecisionCandidate(state, actor, decisionType, candidate) {
  return applyAction(state, actor, decisionType, candidate);
}

function maskStateForVisibleComboSimulation(state) {
  if (!state || typeof state !== "object") return state;
  const pendingMatch = state?.pendingMatch
    ? {
        ...state.pendingMatch,
        context: state.pendingMatch?.context
          ? {
              ...state.pendingMatch.context,
              deck: []
            }
          : state.pendingMatch.context
      }
    : state?.pendingMatch ?? null;
  return {
    ...state,
    deck: [],
    pendingMatch
  };
}

function candidateComboGain(state, actor, decisionType, candidate) {
  const beforePlayer = state?.players?.[actor];
  if (!beforePlayer) return 0;

  // Combo completion must not peek at hidden future draws. Simulate only with
  // currently public state by masking the deck (and pending match deck context).
  const visibleState = maskStateForVisibleComboSimulation(state);
  const afterState = applyDecisionCandidate(visibleState, actor, decisionType, candidate);
  const afterPlayer = afterState?.players?.[actor];
  if (!afterPlayer) return 0;

  const beforeGwangCards = uniqueCapturedCards(beforePlayer, "kwang");
  const afterGwangCards = uniqueCapturedCards(afterPlayer, "kwang");
  const kwangGain = Math.max(0, compactKwangBaseScore(afterGwangCards) - compactKwangBaseScore(beforeGwangCards));
  const beforeGodori = countCapturedComboTag(beforePlayer, "five", "fiveBirds");
  const afterGodori = countCapturedComboTag(afterPlayer, "five", "fiveBirds");

  const ribbonTags = ["redRibbons", "blueRibbons", "plainRibbons"];
  const completesDan = ribbonTags.some((tag) => {
    const beforeCount = countCapturedComboTag(beforePlayer, "ribbon", tag);
    const afterCount = countCapturedComboTag(afterPlayer, "ribbon", tag);
    return beforeCount < 3 && afterCount >= 3;
  });

  const raw =
    kwangGain +
    (beforeGodori < 3 && afterGodori >= 3 ? 5 : 0) +
    (completesDan ? 3 : 0);
  return clamp01(raw / 11.0);
}

function decisionAvailabilityFlags(state, actor) {
  if (state?.phase === "shaking-confirm" && state?.pendingShakingConfirm?.playerKey === actor) {
    return { hasShake: 1, hasBomb: 0 };
  }
  if (state?.phase !== "playing" || state?.currentTurn !== actor) {
    return { hasShake: 0, hasBomb: 0 };
  }
  const shakingMonths = getDeclarableShakingMonths(state, actor);
  const bombMonths = getDeclarableBombMonths(state, actor);
  return {
    hasShake: Array.isArray(shakingMonths) && shakingMonths.length > 0 ? 1 : 0,
    hasBomb: Array.isArray(bombMonths) && bombMonths.length > 0 ? 1 : 0
  };
}

function currentMultiplierNorm(state, scoreSelf) {
  const carry = Math.max(1.0, Number(state?.carryOverMultiplier || 1.0));
  const mul = Math.max(1.0, Number(scoreSelf?.multiplier || 1.0));
  const currentMultiplier = mul * carry;
  return clamp01(Math.log2(currentMultiplier) / 4.0);
}

function compactSelfScoreProgressNorm(scoreTotal) {
  const s = Math.max(0, Number(scoreTotal || 0));
  if (s <= 7) {
    return clamp01(0.72 * Math.pow(s / 7.0, 1.35));
  }
  const tail = Math.log2(1 + Math.min(s - 7, 8)) / Math.log2(9);
  return clamp01(0.72 + 0.28 * tail);
}

function compactOppStopPressureNorm(scoreTotal) {
  const s = Math.max(0, Number(scoreTotal || 0));
  if (s <= 0) return 0.0;
  if (s === 1) return 0.05;
  if (s === 2) return 0.1;
  if (s === 3) return 0.15;
  if (s === 4) return 0.3;
  if (s === 5) return 0.6;
  if (s === 6) return 0.9;
  return 1.0;
}

function compactGoStageNorm(goCount) {
  const g = Math.max(0, Number(goCount || 0));
  if (g <= 0) return 0.0;
  if (g === 1) return 0.2;
  if (g === 2) return 0.35;
  if (g === 3) return 0.7;
  if (g === 4) return 0.9;
  return 1.0;
}

function compactGoStopGatedFeatures(state, actor, decisionType, candidate) {
  const isGoStopOption = decisionType === "option" && state?.phase === "go-stop";
  const action = canonicalOptionAction(candidate);
  const opp = actor === "human" ? "ai" : "human";
  return [
    !isGoStopOption ? 0.5 : action === "go" ? 1.0 : action === "stop" ? 0.0 : 0.5,
    isGoStopOption ? compactGoStageNorm(state?.players?.[actor]?.goCount || 0) : 0.0,
    isGoStopOption ? compactGoStageNorm(state?.players?.[opp]?.goCount || 0) : 0.0
  ];
}

function padDeletedTailFeatures(baseFeatures, inputDim) {
  const baseDim = baseFeatures.length;
  if (inputDim === baseDim) return baseFeatures;
  if (inputDim === (baseDim + 1)) return [...baseFeatures, 0.0];
  throw new Error(
    `feature vector size mismatch: expected ${inputDim}, supported=${baseDim},${baseDim + 1}`
  );
}

function buildLegacyCompactFeatureVector(state, actor, decisionType, candidate) {
  const opp = actor === "human" ? "ai" : "human";
  const scoreSelf = calculateScore(state.players[actor], state.players[opp], state.ruleKey);
  const scoreOpp = calculateScore(state.players[opp], state.players[actor], state.ruleKey);
  const card = candidateCard(state, actor, decisionType, candidate);
  const month = resolveCandidateMonth(state, actor, decisionType, card);

  return [
    decisionType === "play" ? 1 : 0,
    decisionType === "match" ? 1 : 0,
    optionCode(candidate),
    tanhNorm((scoreSelf?.total || 0) - (scoreOpp?.total || 0), 10.0),
    compactSelfScoreProgressNorm(scoreSelf?.total || 0),
    compactOppStopPressureNorm(scoreOpp?.total || 0),
    clamp01(currentMultiplierNorm(state, scoreSelf)),
    candidateComboGain(state, actor, decisionType, candidate),
    compactCandidatePiNorm(card),
    immediateMatchPossible(state, decisionType, month),
    candidatePublicKnownRatio(state, actor, month),
    Number(scoreSelf?.total || 0) >= 7 ? 1 : 0,
    Number(scoreOpp?.total || 0) >= 7 ? 1 : 0
  ];
}

function buildCompactFeatureVector(state, actor, decisionType, candidate) {
  return [
    ...buildLegacyCompactFeatureVector(state, actor, decisionType, candidate),
    ...compactGoStopGatedFeatures(state, actor, decisionType, candidate)
  ];
}

function buildLegacyFeatureVector(state, actor, decisionType, candidate, legalCount) {
  const opp = actor === "human" ? "ai" : "human";
  const scoreSelf = calculateScore(state.players[actor], state.players[opp], state.ruleKey);
  const scoreOpp = calculateScore(state.players[opp], state.players[actor], state.ruleKey);

  const phase = String(state.phase || "");
  const card = candidateCard(state, actor, decisionType, candidate);
  const month = resolveCandidateMonth(state, actor, decisionType, card);
  const piValue = Number(card?.piValue || 0);
  const category = String(card?.category || "");
  const selfGwangCount = Number(state?.players?.[actor]?.captured?.kwang?.length || 0);
  const oppGwangCount = Number(state?.players?.[opp]?.captured?.kwang?.length || 0);
  const selfPiCount = Number(scoringPiCount(state.players[actor]) || 0);
  const oppPiCount = Number(scoringPiCount(state.players[opp]) || 0);

  const selfGodori = countCapturedComboTag(state.players?.[actor], "five", "fiveBirds");
  const oppGodori = countCapturedComboTag(state.players?.[opp], "five", "fiveBirds");
  const selfCheongdan = countCapturedComboTag(state.players?.[actor], "ribbon", "blueRibbons");
  const oppCheongdan = countCapturedComboTag(state.players?.[opp], "ribbon", "blueRibbons");
  const selfHongdan = countCapturedComboTag(state.players?.[actor], "ribbon", "redRibbons");
  const oppHongdan = countCapturedComboTag(state.players?.[opp], "ribbon", "redRibbons");
  const selfChodan = countCapturedComboTag(state.players?.[actor], "ribbon", "plainRibbons");
  const oppChodan = countCapturedComboTag(state.players?.[opp], "ribbon", "plainRibbons");

  const selfCanStop = Number(scoreSelf?.total || 0) >= 7 ? 1 : 0;
  const oppCanStop = Number(scoreOpp?.total || 0) >= 7 ? 1 : 0;
  const { hasShake, hasBomb } = decisionAvailabilityFlags(state, actor);

  const baseFeatures = [
    phase === "playing" ? 1 : 0,
    phase === "select-match" ? 1 : 0,
    phase === "go-stop" ? 1 : 0,
    phase === "president-choice" ? 1 : 0,
    phase === "gukjin-choice" ? 1 : 0,
    phase === "shaking-confirm" ? 1 : 0,

    decisionType === "play" ? 1 : 0,
    decisionType === "match" ? 1 : 0,
    decisionType === "option" ? 1 : 0,

    clamp01((state.deck?.length || 0) / 30.0),
    clamp01((state.players?.[actor]?.hand?.length || 0) / 10.0),
    clamp01((state.players?.[opp]?.hand?.length || 0) / 10.0),
    clamp01((state.players?.[actor]?.goCount || 0) / 5.0),
    clamp01((state.players?.[opp]?.goCount || 0) / 5.0),
    tanhNorm((scoreSelf?.total || 0) - (scoreOpp?.total || 0), 10.0),
    tanhNorm(scoreSelf?.total || 0, 10.0),
    clamp01(Number(legalCount || 0) / 10.0),

    clamp01(piValue / 5.0),
    category === "kwang" ? 1 : 0,
    category === "ribbon" ? 1 : 0,
    category === "five" ? 1 : 0,
    category === "junk" ? 1 : 0,
    isDoublePiCard(card) ? 1 : 0,

    matchOpportunityDensity(state, month),
    immediateMatchPossible(state, decisionType, month),
    optionCode(candidate),

    clamp01(selfGwangCount / 5.0),
    clamp01(oppGwangCount / 5.0),
    clamp01(selfPiCount / 20.0),
    clamp01(oppPiCount / 20.0),

    clamp01(selfGodori / 3.0),
    clamp01(oppGodori / 3.0),
    clamp01(selfCheongdan / 3.0),
    clamp01(oppCheongdan / 3.0),
    clamp01(selfHongdan / 3.0),
    clamp01(oppHongdan / 3.0),
    clamp01(selfChodan / 3.0),
    clamp01(oppChodan / 3.0),

    selfCanStop,
    oppCanStop,

    hasShake,
    currentMultiplierNorm(state, scoreSelf),
    hasBomb,

    scoreSelf?.bak?.pi ? 1 : 0,
    scoreSelf?.bak?.gwang ? 1 : 0,
    scoreSelf?.bak?.mongBak ? 1 : 0
  ];

  return baseFeatures;
}

function featureVector(state, actor, decisionType, candidate, legalCount, inputDim) {
  if (inputDim === NEAT_COMPACT_FEATURES) {
    const features = buildCompactFeatureVector(state, actor, decisionType, candidate);
    if (features.length !== NEAT_COMPACT_FEATURES) {
      throw new Error(`compact feature length mismatch: expected ${NEAT_COMPACT_FEATURES}, got ${features.length}`);
    }
    return features;
  }
  if (inputDim === LEGACY_NEAT_COMPACT_FEATURES) {
    const features = buildLegacyCompactFeatureVector(state, actor, decisionType, candidate);
    if (features.length !== LEGACY_NEAT_COMPACT_FEATURES) {
      throw new Error(`legacy compact feature length mismatch: expected ${LEGACY_NEAT_COMPACT_FEATURES}, got ${features.length}`);
    }
    return features;
  }
  throw new Error(
    `feature vector size mismatch: expected ${inputDim}, supported=${NEAT_COMPACT_FEATURES},${LEGACY_NEAT_COMPACT_FEATURES}`
  );
}

function estimateStopValueForIqn(state, actor) {
  const opp = actor === "human" ? "ai" : "human";
  const scoreSelf = calculateScore(state.players[actor], state.players[opp], state.ruleKey);
  const carry = Math.max(1.0, Number(state?.carryOverMultiplier || 1.0));
  const mul = Math.max(1.0, Number(scoreSelf?.multiplier || 1.0));
  const estimatedGoldK = Number(scoreSelf?.total || 0) * mul * carry * 0.1;
  return clampRange(estimatedGoldK, -12.0, 12.0);
}

function buildIqnGoStopPayload(state, actor) {
  const opp = actor === "human" ? "ai" : "human";
  const initialGoldBase = resolveInitialGoldBase(state);
  const selfGold = Number(state?.players?.[actor]?.gold || initialGoldBase);
  const oppGold = Number(state?.players?.[opp]?.gold || initialGoldBase);
  const scoreSelf = calculateScore(state.players[actor], state.players[opp], state.ruleKey);
  const scoreOpp = calculateScore(state.players[opp], state.players[actor], state.ruleKey);
  return [
    0.0,
    estimateStopValueForIqn(state, actor),
    clampRange(Number(state?.carryOverMultiplier || 1.0) / 8.0, 0.0, 2.0),
    clampRange((Number(scoreSelf?.multiplier || 1.0) * Math.max(1.0, Number(state?.carryOverMultiplier || 1.0))) / 16.0, 0.0, 2.0),
    clampRange(Number(scoreSelf?.total || 0) / 20.0, -2.0, 2.0),
    clampRange(Number(scoreOpp?.total || 0) / 20.0, -2.0, 2.0),
    clampRange((selfGold - initialGoldBase) / 1000.0, -12.0, 12.0),
    clampRange((oppGold - initialGoldBase) / 1000.0, -12.0, 12.0),
    clampRange(Number(state?.players?.[actor]?.goCount || 0) / 6.0, 0.0, 2.0),
    clampRange(Number(state?.deck?.length || 0) / 20.0, 0.0, 2.0),
  ];
}

function resolveIqnBaseFeatureDim(runtimeModel) {
  const configured = Number(runtimeModel?.feature_spec?.base_features || 0);
  if (configured === IQN_GO_STOP_BASE_FEATURES) {
    return configured;
  }
  const derived = Number(runtimeModel?.input_dim || 0) - GO_STOP_OPTION_ONE_HOT.length - IQN_GO_STOP_PAYLOAD_DIM;
  if (derived === IQN_GO_STOP_BASE_FEATURES) {
    return derived;
  }
  return IQN_GO_STOP_BASE_FEATURES;
}

function buildIqnGoStopFeatureVector(state, actor, candidate, runtimeModel = null) {
  const baseDim = resolveIqnBaseFeatureDim(runtimeModel);
  const baseFeatures = featureVector(state, actor, "option", candidate, 2, baseDim);
  return [...baseFeatures, ...GO_STOP_OPTION_ONE_HOT, ...buildIqnGoStopPayload(state, actor)];
}

function isFiniteNumberArray(values) {
  return Array.isArray(values) && values.every((v) => Number.isFinite(Number(v)));
}

export function isIqnGoStopRuntimeModel(runtimeModel) {
  if (String(runtimeModel?.format_version || "").trim() !== IQN_GO_STOP_RUNTIME_FORMAT) return false;
  const baseDim = resolveIqnBaseFeatureDim(runtimeModel);
  const expectedInputDim = baseDim + GO_STOP_OPTION_ONE_HOT.length + IQN_GO_STOP_PAYLOAD_DIM;
  if (Number(runtimeModel?.input_dim || 0) !== expectedInputDim) return false;
  if (!Array.isArray(runtimeModel?.encoder_blocks) || runtimeModel.encoder_blocks.length <= 0) return false;
  if (!runtimeModel?.tau_fc?.linear || !runtimeModel?.quantile_head?.hidden || !runtimeModel?.quantile_head?.output) {
    return false;
  }
  return true;
}

function linearForward(layer, inputVec) {
  const weight = Array.isArray(layer?.weight) ? layer.weight : [];
  const bias = Array.isArray(layer?.bias) ? layer.bias : [];
  if (weight.length <= 0) return [];
  const out = new Array(weight.length);
  for (let rowIndex = 0; rowIndex < weight.length; rowIndex += 1) {
    const row = Array.isArray(weight[rowIndex]) ? weight[rowIndex] : [];
    let acc = Number(bias[rowIndex] || 0);
    for (let colIndex = 0; colIndex < row.length; colIndex += 1) {
      acc += Number(row[colIndex] || 0) * Number(inputVec[colIndex] || 0);
    }
    out[rowIndex] = acc;
  }
  return out;
}

function reluForward(values) {
  return values.map((v) => (Number(v || 0) > 0 ? Number(v || 0) : 0));
}

function layerNormForward(layer, inputVec) {
  const values = inputVec.map((v) => Number(v || 0));
  if (values.length <= 0) return [];
  const eps = Math.max(1e-9, Number(layer?.eps || 1e-5));
  const meanValue = values.reduce((sum, v) => sum + v, 0) / values.length;
  let variance = 0;
  for (const value of values) {
    const diff = value - meanValue;
    variance += diff * diff;
  }
  variance /= values.length;
  const denom = Math.sqrt(variance + eps);
  const weight = Array.isArray(layer?.weight) ? layer.weight : [];
  const bias = Array.isArray(layer?.bias) ? layer.bias : [];
  return values.map(
    (value, index) => (((value - meanValue) / denom) * Number(weight[index] || 1)) + Number(bias[index] || 0)
  );
}

function encodeIqnState(runtimeModel, featureVec) {
  let x = featureVec.map((v) => Number(v || 0));
  for (const block of runtimeModel.encoder_blocks || []) {
    x = linearForward(block.linear, x);
    x = reluForward(x);
    x = layerNormForward(block.layer_norm, x);
  }
  return x;
}

function buildFixedTaus(numQuantiles) {
  const count = Math.max(1, Number(numQuantiles || 1));
  const out = [];
  for (let index = 0; index < count; index += 1) {
    out.push((index + 0.5) / count);
  }
  return out;
}

function forwardIqnQuantiles(runtimeModel, featureVec, numQuantiles = null) {
  if (!isIqnGoStopRuntimeModel(runtimeModel)) return [];
  const encoded = encodeIqnState(runtimeModel, featureVec);
  const tauLinear = runtimeModel?.tau_fc?.linear || null;
  const qHidden = runtimeModel?.quantile_head?.hidden || null;
  const qOutput = runtimeModel?.quantile_head?.output || null;
  if (!tauLinear || !qHidden || !qOutput) return [];
  const numCosines = Math.max(1, Number(runtimeModel?.num_cosines || 64));
  const taus = buildFixedTaus(numQuantiles ?? runtimeModel?.num_quantiles_eval ?? 64);
  const quantiles = [];
  for (const tau of taus) {
    const cosineFeatures = [];
    for (let k = 1; k <= numCosines; k += 1) {
      cosineFeatures.push(Math.cos(Math.PI * tau * k));
    }
    const tauEmbed = reluForward(linearForward(tauLinear, cosineFeatures));
    const fused = encoded.map((value, index) => Number(value || 0) * Number(tauEmbed[index] || 0));
    const hidden = reluForward(linearForward(qHidden, fused));
    const output = linearForward(qOutput, hidden);
    quantiles.push(Number(output[0] || 0));
  }
  return quantiles;
}

function summarizeIqnQuantiles(runtimeModel, quantiles) {
  if (!isFiniteNumberArray(quantiles) || quantiles.length <= 0) {
    return { quantiles: [], mean: 0, cvar10: 0, score: 0 };
  }
  const meanValue = quantiles.reduce((sum, v) => sum + Number(v || 0), 0) / quantiles.length;
  const sorted = [...quantiles].sort((a, b) => a - b);
  const cvarAlpha = clampRange(Number(runtimeModel?.cvar_alpha || 0.1), 0.01, 0.5);
  const tailCount = Math.max(1, Math.ceil(sorted.length * cvarAlpha));
  const cvar10 = sorted.slice(0, tailCount).reduce((sum, v) => sum + Number(v || 0), 0) / tailCount;
  const weightMean = Number(runtimeModel?.score_weights?.mean || 0.7);
  const weightCvar = Number(runtimeModel?.score_weights?.cvar10 || 0.3);
  return {
    quantiles: sorted,
    mean: meanValue,
    cvar10,
    score: (weightMean * meanValue) + (weightCvar * cvar10),
  };
}

export function evaluateIqnGoStopDecision(state, actor, runtimeModel) {
  if (!isIqnGoStopRuntimeModel(runtimeModel)) return null;
  if (state?.phase !== "go-stop" || state?.pendingGoStop !== actor) return null;
  const sp = selectPool(state, actor);
  const legal = legalCandidatesForDecision(sp, "option");
  if (!legal.includes("go") || !legal.includes("stop")) return null;

  const goEval = summarizeIqnQuantiles(
    runtimeModel,
    forwardIqnQuantiles(runtimeModel, buildIqnGoStopFeatureVector(state, actor, "go", runtimeModel))
  );
  const stopEval = summarizeIqnQuantiles(
    runtimeModel,
    forwardIqnQuantiles(runtimeModel, buildIqnGoStopFeatureVector(state, actor, "stop", runtimeModel))
  );
  const action = goEval.score > stopEval.score ? "go" : "stop";
  return {
    action,
    legal_candidates: legal,
    go: goEval,
    stop: stopEval,
    margin: Number(goEval.score || 0) - Number(stopEval.score || 0),
  };
}

/* 3) Forward-pass helpers */
function activation(name, x) {
  const n = String(name || "tanh").trim().toLowerCase();
  const v = Number(x || 0);
  if (n === "sigmoid") return 1.0 / (1.0 + Math.exp(-v));
  if (n === "relu") return Math.max(0, v);
  if (n === "identity" || n === "linear") return v;
  if (n === "clamped") return Math.max(-1, Math.min(1, v));
  if (n === "gauss") return Math.exp(-(v * v));
  if (n === "sin") return Math.sin(v);
  if (n === "abs") return Math.abs(v);
  return Math.tanh(v);
}

function aggregate(name, values) {
  const agg = String(name || "sum").trim().toLowerCase();
  if (!values.length) return 0.0;
  if (agg === "sum") return values.reduce((a, b) => a + b, 0);
  if (agg === "mean") return values.reduce((a, b) => a + b, 0) / values.length;
  if (agg === "max") return Math.max(...values);
  if (agg === "min") return Math.min(...values);
  if (agg === "product") return values.reduce((a, b) => a * b, 1.0);
  if (agg === "maxabs") return values.reduce((a, b) => (Math.abs(b) > Math.abs(a) ? b : a), values[0]);
  return values.reduce((a, b) => a + b, 0);
}

function compileNeatPythonGenome(raw) {
  const inputKeys = Array.isArray(raw?.input_keys) ? raw.input_keys.map((x) => Number(x)) : [];
  const outputKeys = Array.isArray(raw?.output_keys) ? raw.output_keys.map((x) => Number(x)) : [];
  const nodesRaw = raw?.nodes && typeof raw.nodes === "object" ? raw.nodes : {};
  const decisionParams = raw?.decision_params && typeof raw.decision_params === "object" ? raw.decision_params : {};

  const nodes = new Map();
  for (const [k, v] of Object.entries(nodesRaw)) {
    const nodeId = Number(v?.node_id ?? k);
    nodes.set(nodeId, {
      node_id: nodeId,
      activation: String(v?.activation || "tanh"),
      aggregation: String(v?.aggregation || "sum"),
      bias: Number(v?.bias || 0),
      response: Number(v?.response || 1)
    });
  }

  for (const outKey of outputKeys) {
    if (!nodes.has(outKey)) {
      nodes.set(outKey, {
        node_id: outKey,
        activation: "tanh",
        aggregation: "sum",
        bias: 0,
        response: 1
      });
    }
  }

  const connections = [];
  for (const item of raw?.connections || []) {
    if (!item?.enabled) continue;
    connections.push({
      in_node: Number(item?.in_node || 0),
      out_node: Number(item?.out_node || 0),
      weight: Number(item?.weight || 0)
    });
  }

  const inputSet = new Set(inputKeys);
  const nonInputSet = new Set([...nodes.keys()].filter((k) => !inputSet.has(k)));
  const indegree = new Map();
  const adjacency = new Map();
  const incoming = new Map();

  for (const node of nonInputSet) {
    indegree.set(node, 0);
    adjacency.set(node, []);
    incoming.set(node, []);
  }

  for (const conn of connections) {
    const outNode = conn.out_node;
    if (!nonInputSet.has(outNode)) continue;
    incoming.get(outNode).push(conn);
    const inNode = conn.in_node;
    if (nonInputSet.has(inNode)) {
      indegree.set(outNode, Number(indegree.get(outNode) || 0) + 1);
      adjacency.get(inNode).push(outNode);
    }
  }

  const queue = [...nonInputSet].filter((n) => Number(indegree.get(n) || 0) === 0).sort((a, b) => a - b);
  const order = [];
  while (queue.length > 0) {
    const node = queue.shift();
    order.push(node);
    const nexts = adjacency.get(node) || [];
    for (const nxt of nexts) {
      const deg = Number(indegree.get(nxt) || 0) - 1;
      indegree.set(nxt, deg);
      if (deg === 0) {
        queue.push(nxt);
        queue.sort((a, b) => a - b);
      }
    }
  }

  return {
    kind: NEAT_MODEL_FORMAT,
    inputKeys,
    outputKeys,
    decisionParams: {
      go_stop_threshold: Number(decisionParams.go_stop_threshold || 0),
      shaking_threshold: Number(decisionParams.shaking_threshold || 0),
      president_threshold: Number(decisionParams.president_threshold || 0),
      gukjin_threshold: Number(decisionParams.gukjin_threshold || 0)
    },
    nodes,
    incoming,
    order: order.length === nonInputSet.size ? order : [...nonInputSet].sort((a, b) => a - b)
  };
}

/* 4) Model compilation/cache */
function isNeatModel(policyModel) {
  return String(policyModel?.format_version || "").trim() === NEAT_MODEL_FORMAT;
}

function getCompiledNeatModel(policyModel) {
  if (!policyModel || !isNeatModel(policyModel)) return null;
  const cached = COMPILED_NEAT_CACHE.get(policyModel);
  if (cached) return cached;
  const compiled = compileNeatPythonGenome(policyModel);
  COMPILED_NEAT_CACHE.set(policyModel, compiled);
  return compiled;
}

/* 5) Scoring/post-processing */
function forward(compiled, inputVec) {
  const values = new Map();
  for (let i = 0; i < compiled.inputKeys.length; i += 1) {
    values.set(Number(compiled.inputKeys[i]), Number(inputVec[i] || 0));
  }

  for (const nodeId of compiled.order) {
    const node = compiled.nodes.get(nodeId) || {
      activation: "tanh",
      aggregation: "sum",
      bias: 0,
      response: 1
    };
    const incoming = compiled.incoming.get(nodeId) || [];
    const terms = incoming.map((conn) => Number(values.get(conn.in_node) || 0) * Number(conn.weight || 0));
    const agg = aggregate(node.aggregation, terms);
    const pre = Number(node.bias || 0) + Number(node.response || 1) * agg;
    values.set(nodeId, activation(node.activation, pre));
  }

  return compiled.outputKeys.map((outKey) => Number(values.get(Number(outKey)) || 0.0));
}

function candidateOutputIndex(compiled, decisionType, candidate) {
  const outputCount = Array.isArray(compiled?.outputKeys) ? compiled.outputKeys.length : 0;
  if (outputCount <= 3) {
    if (decisionType === "option") {
      const action = canonicalOptionAction(candidate);
      if (action === "go") return NEAT_OUT_GO - 1;
      if (action === "stop") return NEAT_OUT_STOP - 1;
    }
    return NEAT_OUT_PLAY;
  }
  if (decisionType === "match") return NEAT_OUT_MATCH;
  if (decisionType === "option") {
    const action = canonicalOptionAction(candidate);
    if (action === "go") return NEAT_OUT_GO;
    if (action === "stop") return NEAT_OUT_STOP;
    if (action === "shaking_yes") return NEAT_OUT_SHAKING_YES;
    if (action === "shaking_no") return NEAT_OUT_SHAKING_NO;
    if (action === "president_stop") return NEAT_OUT_PRESIDENT_STOP;
    if (action === "president_hold") return NEAT_OUT_PRESIDENT_HOLD;
    if (action === "five") return NEAT_OUT_GUKJIN_FIVE;
    if (action === "junk") return NEAT_OUT_GUKJIN_JUNK;
  }
  return NEAT_OUT_PLAY;
}

function scoreCandidateFromOutputs(compiled, outputs, decisionType, candidate) {
  const outputKeys = Array.isArray(compiled?.outputKeys) ? compiled.outputKeys : [];
  if (outputKeys.length <= 1) {
    return Number(outputs[0] || 0.0);
  }
  const index = candidateOutputIndex(compiled, decisionType, candidate);
  if (index < 0 || index >= outputKeys.length) return 0.0;
  return Number(outputs[index] || 0.0);
}

function resolveOptionThresholdKey(candidates) {
  const set = new Set((candidates || []).map((candidate) => canonicalOptionAction(candidate)));
  if (set.has("go") && set.has("stop")) return "go_stop_threshold";
  if (set.has("shaking_yes") && set.has("shaking_no")) return "shaking_threshold";
  if (set.has("president_stop") && set.has("president_hold")) return "president_threshold";
  if (set.has("five") && set.has("junk")) return "gukjin_threshold";
  return null;
}

function pickThresholdOptionCandidate(compiled, scored) {
  if (!compiled || !scored || scored.decisionType !== "option") return null;
  const key = resolveOptionThresholdKey(scored.candidates);
  if (!key) return null;
  const threshold = Number(compiled?.decisionParams?.[key] || 0);
  const normalized = scored.candidates.map((candidate) => canonicalOptionAction(candidate));
  let primary = null;
  let secondary = null;
  if (key === "go_stop_threshold") {
    primary = scored.candidates[normalized.indexOf("go")];
    secondary = scored.candidates[normalized.indexOf("stop")];
  } else if (key === "shaking_threshold") {
    primary = scored.candidates[normalized.indexOf("shaking_yes")];
    secondary = scored.candidates[normalized.indexOf("shaking_no")];
  } else if (key === "president_threshold") {
    primary = scored.candidates[normalized.indexOf("president_stop")];
    secondary = scored.candidates[normalized.indexOf("president_hold")];
  } else if (key === "gukjin_threshold") {
    primary = scored.candidates[normalized.indexOf("five")];
    secondary = scored.candidates[normalized.indexOf("junk")];
  }
  if (primary == null || secondary == null) return null;
  const primaryScore = Number(scored.scores?.[String(primary)] ?? -Infinity);
  const secondaryScore = Number(scored.scores?.[String(secondary)] ?? -Infinity);
  return primaryScore - secondaryScore > threshold ? primary : secondary;
}

function scoreToProbabilityMap(candidates, scoreMap, temperature = 1.0) {
  const temp = Math.max(0.05, Number(temperature || 1.0));
  const scores = candidates.map((c) => Number(scoreMap.get(c) || -Infinity));
  const maxScore = Math.max(...scores);
  const exps = scores.map((s) => Math.exp((s - maxScore) / temp));
  const z = exps.reduce((a, b) => a + b, 0);
  const probs = {};
  if (!(z > 0)) {
    const uniform = candidates.length > 0 ? 1 / candidates.length : 0;
    for (const c of candidates) probs[String(c)] = uniform;
    return probs;
  }
  for (let i = 0; i < candidates.length; i += 1) {
    probs[String(candidates[i])] = exps[i] / z;
  }
  return probs;
}

function pickBestByScore(candidates, scoreMap) {
  let best = null;
  let bestScore = -Infinity;
  for (const c of candidates) {
    const s = Number(scoreMap.get(c) || -Infinity);
    if (s > bestScore) {
      bestScore = s;
      best = c;
    }
  }
  return best;
}

/* 6) Public APIs */
export function getModelCandidateProbabilities(state, actor, policyModel, options = {}) {
  const compiled = getCompiledNeatModel(policyModel);
  if (!compiled) return null;

  const sp = selectPool(state, actor, options);
  const decisionType = resolveDecisionType(sp);
  if (!decisionType) return null;

  const candidates = legalCandidatesForDecision(sp, decisionType);
  if (!candidates.length) return null;

  const inputDim = Number(compiled.inputKeys.length || 0);
  const scoreMap = new Map();
  const scores = {};
  try {
    for (const candidate of candidates) {
      const x = featureVector(state, actor, decisionType, candidate, candidates.length, inputDim);
      const outputs = forward(compiled, x);
      const score = scoreCandidateFromOutputs(compiled, outputs, decisionType, candidate);
      scoreMap.set(candidate, score);
      scores[String(candidate)] = score;
    }
  } catch {
    return null;
  }

  const probs = scoreToProbabilityMap(candidates, scoreMap, Number(policyModel?.softmax_temp || 1.0));
  return { decisionType, candidates, probabilities: probs, scores };
}

export function debugFeatureRows(state, actor, options = {}) {
  const sp = selectPool(state, actor, options);
  const decisionType = resolveDecisionType(sp);
  if (!decisionType) return null;

  const candidates = legalCandidatesForDecision(sp, decisionType);
  if (!candidates.length) return null;

  const compiled = options.policyModel ? getCompiledNeatModel(options.policyModel) : null;
  const inputDim = Math.max(
    1,
    Number(options.inputDim || 0) || Number(compiled?.inputKeys?.length || 0) || NEAT_COMPACT_FEATURES
  );

  const rows = [];
  for (const candidate of candidates) {
    const features = featureVector(state, actor, decisionType, candidate, candidates.length, inputDim);
    let score = null;
    if (compiled) {
      const outputs = forward(compiled, features);
      score = scoreCandidateFromOutputs(compiled, outputs, decisionType, candidate);
    }
    rows.push({
      candidate: String(candidate),
      features,
      hasNaN: features.some((v) => Number.isNaN(Number(v))),
      hasInfinity: features.some((v) => !Number.isFinite(Number(v))),
      minValue: features.reduce((acc, v) => Math.min(acc, Number(v)), Infinity),
      maxValue: features.reduce((acc, v) => Math.max(acc, Number(v)), -Infinity),
      score,
    });
  }

  return {
    actor: String(actor || ""),
    decisionType,
    inputDim,
    legalCount: candidates.length,
    rows,
  };
}

function modelPickCandidate(state, actor, policyModel) {
  const compiled = getCompiledNeatModel(policyModel);
  if (!compiled) return null;
  const scored = getModelCandidateProbabilities(state, actor, policyModel);
  if (!scored) return null;
  const thresholdPicked = pickThresholdOptionCandidate(compiled, scored);
  if (thresholdPicked != null) {
    return { decisionType: scored.decisionType, candidate: thresholdPicked };
  }
  const scoreMap = new Map();
  for (const c of scored.candidates) {
    scoreMap.set(c, Number(scored.scores?.[String(c)] || -Infinity));
  }
  const best = pickBestByScore(scored.candidates, scoreMap);
  if (!best) return null;
  return { decisionType: scored.decisionType, candidate: best };
}

export function modelPolicyPlay(state, actor, policyModel, options = {}) {
  const iqnDecision = evaluateIqnGoStopDecision(state, actor, options?.goStopIqnModel || null);
  if (iqnDecision?.action === "go") return chooseGo(state, actor);
  if (iqnDecision?.action === "stop") return chooseStop(state, actor);
  if (!policyModel || !isNeatModel(policyModel)) return state;
  const picked = modelPickCandidate(state, actor, policyModel);
  if (!picked) return state;

  const sp = selectPool(state, actor);
  const decisionType = resolveDecisionType(sp);
  if (!decisionType) return state;

  const legal = legalCandidatesForDecision(sp, decisionType);
  if (!legal.length) return state;

  let c = normalizeDecisionCandidate(decisionType, picked.candidate);
  if (!legal.includes(String(c))) c = legal[0];
  return applyAction(state, actor, decisionType, c);
}
