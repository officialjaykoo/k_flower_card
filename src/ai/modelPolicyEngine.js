import {
  calculateScore,
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
const GUKJIN_CARD_ID = "I0";
const NON_BRIGHT_KWANG_ID = "L0";
const TOTAL_SSANGPI_VALUE = 13;
const COMBO_THREAT_SPECS = Object.freeze({
  redRibbons: Object.freeze({ zone: "ribbon", tag: "redRibbons", months: Object.freeze([1, 2, 3]), reward: 3, category: "ribbon" }),
  blueRibbons: Object.freeze({ zone: "ribbon", tag: "blueRibbons", months: Object.freeze([6, 9, 10]), reward: 3, category: "ribbon" }),
  plainRibbons: Object.freeze({ zone: "ribbon", tag: "plainRibbons", months: Object.freeze([4, 5, 7]), reward: 3, category: "ribbon" }),
  fiveBirds: Object.freeze({ zone: "five", tag: "fiveBirds", months: Object.freeze([2, 4, 8]), reward: 5, category: "five" }),
  kwang: Object.freeze({ zone: "kwang", tag: null, months: Object.freeze([1, 3, 8, 11, 12]), reward: 0, category: "kwang" })
});
const COMBO_THREAT_KEYS = Object.freeze(Object.keys(COMBO_THREAT_SPECS));
const LEGACY13_FEATURES = 13;
const HAND10_FEATURES = 10;
const MATERIAL10_STAGING_FEATURES = 10;
const POSITION11_FEATURES = 11;
const DEFAULT_FEATURE_PROFILE = "hand10";
const NEAT_OUT_ACTION_SCORE = 0;
const NEAT_OUT_OPTION_BIAS = 1;
const IQN_GO_STOP_BASE_FEATURES = 10;
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

function ssangpiLikeValue(card) {
  if (!card) return 0;
  if (String(card?.id || "") === GUKJIN_CARD_ID) return 2;
  const piValue = Math.max(0, Number(card?.piValue || 0));
  const stealPi = Math.max(0, Number(card?.bonus?.stealPi || 0));
  if (stealPi > 0) return piValue + stealPi;
  if (String(card?.category || "") !== "junk") return 0;
  return piValue >= 2 ? piValue : 0;
}

function piLikeValue(card) {
  if (!card) return 0;
  if (String(card?.id || "") === GUKJIN_CARD_ID) return 2;
  const piValue = Math.max(0, Number(card?.piValue || 0));
  const stealPi = Math.max(0, Number(card?.bonus?.stealPi || 0));
  return Math.max(0, piValue + stealPi);
}

function sumUniqueSsangpiLikeValue(cards) {
  if (!Array.isArray(cards)) return 0;
  const seen = new Set();
  let total = 0;
  for (const card of cards) {
    const id = String(card?.id || "");
    if (!id || seen.has(id)) continue;
    seen.add(id);
    total += ssangpiLikeValue(card);
  }
  return total;
}

function sumUniquePiLikeValue(cards) {
  if (!Array.isArray(cards)) return 0;
  const seen = new Set();
  let total = 0;
  for (const card of cards) {
    const id = String(card?.id || "");
    if (!id || seen.has(id)) continue;
    seen.add(id);
    total += piLikeValue(card);
  }
  return total;
}

function collectCapturedCards(player) {
  const captured = player?.captured || {};
  return []
    .concat(captured.kwang || [])
    .concat(captured.five || [])
    .concat(captured.ribbon || [])
    .concat(captured.junk || []);
}

function selfSsangpiControlNorm(state, actor) {
  const selfPlayer = state?.players?.[actor];
  const handValue = sumUniqueSsangpiLikeValue(selfPlayer?.hand || []);
  const capturedValue = sumUniqueSsangpiLikeValue(collectCapturedCards(selfPlayer));
  return clamp01((handValue + capturedValue) / TOTAL_SSANGPI_VALUE);
}

function ssangpiRevealedRatioNorm(state, actor) {
  const opp = actor === "human" ? "ai" : "human";
  const selfPlayer = state?.players?.[actor];
  const oppPlayer = state?.players?.[opp];
  const revealed =
    sumUniqueSsangpiLikeValue(selfPlayer?.hand || []) +
    sumUniqueSsangpiLikeValue(collectCapturedCards(selfPlayer)) +
    sumUniqueSsangpiLikeValue(collectCapturedCards(oppPlayer)) +
    sumUniqueSsangpiLikeValue(state?.board || []);
  return clamp01(revealed / TOTAL_SSANGPI_VALUE);
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

function comboTargetMatchesSpec(card, spec, month = 0) {
  if (!card || !spec) return false;
  if (String(card?.category || "") !== String(spec.category || "")) return false;
  if (month > 0 && Number(card?.month || 0) !== Number(month)) return false;
  if (spec.tag && !hasComboTag(card, spec.tag)) return false;
  return spec.months.includes(Number(card?.month || 0));
}

function capturedComboMonths(player, spec) {
  const out = new Set();
  for (const card of uniqueCapturedCards(player, spec.zone)) {
    if (comboTargetMatchesSpec(card, spec)) out.add(Number(card?.month || 0));
  }
  return out;
}

function cardsContainComboTarget(cards, spec, month) {
  if (!Array.isArray(cards)) return false;
  return cards.some((card) => comboTargetMatchesSpec(card, spec, month));
}

function resolveComboThreatTargetExposure(state, defenderKey, spec, month) {
  const selfPlayer = state?.players?.[defenderKey];
  if (cardsContainComboTarget(selfPlayer?.captured?.[spec.zone] || [], spec, month)) return 0.0;
  if (cardsContainComboTarget(selfPlayer?.hand || [], spec, month)) return 0.2;
  if (cardsContainComboTarget(getPublicKnownOpponentHandCards(state, defenderKey), spec, month)) return 1.0;
  if (cardsContainComboTarget(state?.board || [], spec, month)) return 0.85;
  return 0.55;
}

function kwangThreatBaseRaw(oppPlayer) {
  const oppKwangCards = uniqueCapturedCards(oppPlayer, "kwang");
  const oppKwangCount = oppKwangCards.length;
  const hasNonBright = oppKwangCards.some((card) => String(card?.id || "") === NON_BRIGHT_KWANG_ID);
  if (oppKwangCount === 2) return hasNonBright ? 2 : 3;
  if (oppKwangCount === 3) return hasNonBright ? 2 : 1;
  if (oppKwangCount === 4) return 11;
  return 0;
}

function comboThreatNormByKey(state, defenderKey, comboKey) {
  const spec = COMBO_THREAT_SPECS[comboKey];
  const attackerKey = defenderKey === "human" ? "ai" : "human";
  const selfPlayer = state?.players?.[defenderKey];
  const oppPlayer = state?.players?.[attackerKey];
  if (!selfPlayer || !oppPlayer || !spec) return 0;

  let baseRaw = 0;
  let missingMonths = [];
  if (comboKey === "kwang") {
    const selfKwangCount = uniqueCapturedCards(selfPlayer, "kwang").length;
    const oppKwangCards = uniqueCapturedCards(oppPlayer, "kwang");
    const oppKwangCount = oppKwangCards.length;
    if (selfKwangCount >= 3 || oppKwangCount < 2 || oppKwangCount >= 5) return 0;
    baseRaw = kwangThreatBaseRaw(oppPlayer);
    if (baseRaw <= 0) return 0;
    const capturedMonths = new Set(oppKwangCards.map((card) => Number(card?.month || 0)).filter((month) => month >= 1));
    missingMonths = spec.months.filter((month) => !capturedMonths.has(month));
  } else {
    const oppCount = countCapturedComboTag(oppPlayer, spec.zone, spec.tag);
    if (oppCount !== 2) return 0;
    baseRaw = Number(spec.reward || 0);
    const capturedMonths = capturedComboMonths(oppPlayer, spec);
    missingMonths = spec.months.filter((month) => !capturedMonths.has(month));
  }

  if (missingMonths.length <= 0) return 0;

  const exposures = missingMonths.map((month) => resolveComboThreatTargetExposure(state, defenderKey, spec, month));
  const liveTargets = exposures.filter((value) => value > 0).length;
  if (liveTargets <= 0) return 0;

  const exposureFactor = Math.max(...exposures);
  const outsFactor = Math.min(1.2, 1.0 + 0.1 * Math.max(0, liveTargets - 1));
  return clamp01((baseRaw * exposureFactor * outsFactor) / 11.0);
}

function comboThreatBreakdown(state, defenderKey) {
  const out = Object.create(null);
  for (const comboKey of COMBO_THREAT_KEYS) {
    out[comboKey] = comboThreatNormByKey(state, defenderKey, comboKey);
  }
  return out;
}

function oppComboThreatNorm(state, defenderKey) {
  let maxThreat = 0;
  const breakdown = comboThreatBreakdown(state, defenderKey);
  for (const comboKey of COMBO_THREAT_KEYS) {
    const value = Number(breakdown[comboKey] || 0);
    if (value > maxThreat) maxThreat = value;
  }
  return maxThreat;
}

function candidateBlockGainNorm(state, actor, decisionType, candidate) {
  const visibleState = maskStateForVisibleComboSimulation(state);
  const before = comboThreatBreakdown(visibleState, actor);
  const afterState = applyDecisionCandidate(visibleState, actor, decisionType, candidate);
  if (!afterState) return 0;
  const after = comboThreatBreakdown(afterState, actor);
  let delta = 0;
  for (const comboKey of COMBO_THREAT_KEYS) {
    delta += Number(before[comboKey] || 0) - Number(after[comboKey] || 0);
  }
  return clampRange(delta, -1.0, 1.0);
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

function decisionContextCode(decisionType) {
  if (decisionType === "play") return 0.0;
  if (decisionType === "match") return 0.5;
  return 1.0;
}

function stopStatusDelta(scoreSelf, scoreOpp) {
  const selfCanStop = Number(scoreSelf?.total || 0) >= 7 ? 1 : 0;
  const oppCanStop = Number(scoreOpp?.total || 0) >= 7 ? 1 : 0;
  return selfCanStop - oppCanStop;
}

function deckRemainingNorm(state) {
  return clamp01(Number(state?.deck?.length || 0) / 30.0);
}

function buildLegacy13FeatureVector(state, actor, decisionType, candidate) {
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

function countHandCards(cards) {
  if (!Array.isArray(cards)) return 0;
  return cards.length;
}

function handRatioDenominator(cards) {
  return Math.max(1, countHandCards(cards));
}

function handCardsForActor(state, actor) {
  return state?.players?.[actor]?.hand || [];
}

function boardCountForMonth(state, month) {
  return countCardsByMonth(state?.board || [], month);
}

function knownCountForMonth(state, actor, month) {
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
  return known;
}

function handMonthCountMap(cards) {
  const out = new Map();
  for (const card of cards || []) {
    const month = Number(card?.month || 0);
    if (month <= 0) continue;
    out.set(month, Number(out.get(month) || 0) + 1);
  }
  return out;
}

function countCapturedMonth(player, month) {
  return countCardsByMonth(collectCapturedCards(player), month);
}

function buildVisibleAfterStateForHandFeatures(state, actor, decisionType, candidate) {
  const visibleState = maskStateForVisibleComboSimulation(state);
  try {
    return applyDecisionCandidate(visibleState, actor, decisionType, candidate) || visibleState;
  } catch {
    return visibleState;
  }
}

function handMatchableRatio(state, actor) {
  const hand = handCardsForActor(state, actor);
  const denom = handRatioDenominator(hand);
  const boardMonths = new Set((state?.board || []).map((card) => Number(card?.month || 0)).filter((month) => month > 0));
  let count = 0;
  for (const card of hand) {
    if (boardMonths.has(Number(card?.month || 0))) count += 1;
  }
  return clamp01(count / denom);
}

function handTripleFlagNorm(state, actor) {
  const counts = handMonthCountMap(handCardsForActor(state, actor));
  for (const value of counts.values()) {
    if (value >= 3) return 1.0;
  }
  return 0.0;
}

function handComboReserveNorm(state, actor) {
  const hand = handCardsForActor(state, actor);
  const denom = handRatioDenominator(hand);
  let count = 0;
  for (const card of hand) {
    const category = String(card?.category || "");
    if (category === "kwang" || category === "five" || category === "ribbon") count += 1;
  }
  return clamp01(count / denom);
}

function handHighValueDensity(state, actor) {
  const hand = handCardsForActor(state, actor);
  const denom = handRatioDenominator(hand);
  let count = 0;
  for (const card of hand) {
    if (String(card?.category || "") === "kwang" || ssangpiLikeValue(card) > 0) count += 1;
  }
  return clamp01(count / denom);
}

function handMonopolyIndex(state, actor) {
  const selfPlayer = state?.players?.[actor];
  const hand = handCardsForActor(state, actor);
  if (!selfPlayer || hand.length <= 0) return 0;
  const handCounts = handMonthCountMap(hand);
  let best = 0;
  for (const month of new Set([...handCounts.keys(), ...collectCapturedCards(selfPlayer).map((card) => Number(card?.month || 0)).filter((month) => month > 0)])) {
    const total = monthTotalCards(month);
    if (total <= 0) continue;
    const held = Number(handCounts.get(month) || 0) + countCapturedMonth(selfPlayer, month);
    best = Math.max(best, held / total);
  }
  return clamp01(best);
}

function isSafeDiscardMonth(state, actor, month) {
  if (boardCountForMonth(state, month) >= 3) return true;
  const total = monthTotalCards(month);
  return total > 0 && knownCountForMonth(state, actor, month) >= total;
}

function handSafeDiscardRatio(state, actor) {
  const hand = handCardsForActor(state, actor);
  const denom = handRatioDenominator(hand);
  let count = 0;
  for (const card of hand) {
    if (isSafeDiscardMonth(state, actor, Number(card?.month || 0))) count += 1;
  }
  return clamp01(count / denom);
}

function oppPiPressureNorm(state, actor) {
  const opp = actor === "human" ? "ai" : "human";
  const oppPlayer = state?.players?.[opp];
  const oppEffectivePi = sumUniquePiLikeValue(collectCapturedCards(oppPlayer));
  return clamp01((oppEffectivePi - 6.0) / 4.0);
}

function collectPublicPiTargetWeightsByMonth(state) {
  const weights = new Map();
  const counts = new Map();
  for (const card of state?.board || []) {
    const value = piLikeValue(card);
    if (value <= 0) continue;
    const month = Number(card?.month || 0);
    if (month <= 0) continue;
    weights.set(month, Number(weights.get(month) || 0) + value);
    counts.set(month, Number(counts.get(month) || 0) + 1);
  }
  return { weights, counts };
}

function handOppPiBlockNorm(state, actor) {
  const pressure = oppPiPressureNorm(state, actor);
  if (pressure <= 0) return 0;

  const { weights, counts } = collectPublicPiTargetWeightsByMonth(state);
  if (weights.size <= 0) return 0;

  const handCounts = handMonthCountMap(handCardsForActor(state, actor));
  let totalTargetValue = 0;
  let blockedValue = 0;

  for (const [month, totalWeight] of weights.entries()) {
    const targetCount = Number(counts.get(month) || 0);
    if (targetCount <= 0 || totalWeight <= 0) continue;
    totalTargetValue += totalWeight;
    const heldCount = Number(handCounts.get(month) || 0);
    if (heldCount <= 0) continue;
    const cappedHeld = Math.min(heldCount, targetCount);
    blockedValue += (totalWeight / targetCount) * cappedHeld;
  }

  if (totalTargetValue <= 0) return 0;
  return clamp01(pressure * clamp01(blockedValue / totalTargetValue));
}

function handMonthDiversity(state, actor) {
  const hand = handCardsForActor(state, actor);
  const denom = handRatioDenominator(hand);
  const uniqueMonths = new Set(hand.map((card) => Number(card?.month || 0)).filter((month) => month > 0));
  return clamp01(uniqueMonths.size / denom);
}

function handHiddenPotential(state, actor) {
  const hand = handCardsForActor(state, actor);
  const denom = handRatioDenominator(hand);
  let count = 0;
  for (const card of hand) {
    if (knownCountForMonth(state, actor, Number(card?.month || 0)) <= 1) count += 1;
  }
  return clamp01(count / denom);
}

function collectOppComboTargetEntries(state, actor) {
  const opp = actor === "human" ? "ai" : "human";
  const selfPlayer = state?.players?.[actor];
  const oppPlayer = state?.players?.[opp];
  const out = [];
  const seen = new Set();
  const pushTarget = (spec, month) => {
    const key = `${spec.zone}:${spec.tag || spec.category}:${month}`;
    if (seen.has(key)) return;
    seen.add(key);
    out.push({ spec, month });
  };

  for (const key of COMBO_THREAT_KEYS) {
    const spec = COMBO_THREAT_SPECS[key];
    const oppMonths = capturedComboMonths(oppPlayer, spec);
    if (key === "kwang") {
      if (uniqueCapturedCards(selfPlayer, "kwang").length >= 3) continue;
      if (oppMonths.size < 2) continue;
    } else if (oppMonths.size < 2) {
      continue;
    }
    for (const month of spec.months) {
      if (oppMonths.has(month)) continue;
      if (cardsContainComboTarget(selfPlayer?.captured?.[spec.zone] || [], spec, month)) continue;
      pushTarget(spec, month);
    }
  }
  return out;
}

function handOppBlockNorm(state, actor) {
  const targets = collectOppComboTargetEntries(state, actor);
  if (targets.length <= 0) return 0;
  const hand = handCardsForActor(state, actor);
  let held = 0;
  for (const target of targets) {
    if (cardsContainComboTarget(hand, target.spec, target.month)) held += 1;
  }
  return clamp01(held / targets.length);
}

function candidateDangerExposure(state, actor, decisionType, candidate) {
  const card = candidateCard(state, actor, decisionType, candidate);
  const month = resolveCandidateMonth(state, actor, decisionType, card);
  if (!card || month <= 0) return 0;
  if (immediateMatchPossible(state, decisionType, month) > 0) return 0;

  let comboRisk = 0;
  for (const target of collectOppComboTargetEntries(state, actor)) {
    if (!comboTargetMatchesSpec(card, target.spec, target.month)) continue;
    const rewardNorm = target.spec?.category === "kwang" ? 1.0 : clamp01(Number(target.spec?.reward || 0) / 5.0);
    comboRisk = Math.max(comboRisk, rewardNorm);
  }

  const piRisk = clamp01(oppPiPressureNorm(state, actor) * compactCandidatePiNorm(card));
  return clamp01(Math.max(comboRisk, piRisk));
}

function positionalAdvantageSigned(state, actor) {
  const selfCombo = selfComboCompletionNorm(state, actor);
  const oppProximity = oppWinProximityNorm(state, actor);
  const selfSafety = selfGoSafetyNorm(state, actor);
  const oppPiThreat = oppSsangpiThreatNorm(state, actor);
  return clampRange(Math.tanh((selfCombo - oppProximity + selfSafety - oppPiThreat) / 2.0), -1.0, 1.0);
}

function globalContextTrigger(state, actor) {
  return clamp01(0.5 + (0.5 * positionalAdvantageSigned(state, actor)));
}

function comboCapturedCount(player, comboKey) {
  const spec = COMBO_THREAT_SPECS[comboKey];
  if (!player || !spec) return 0;
  if (comboKey === "kwang") {
    return Math.min(3, uniqueCapturedCards(player, "kwang").length);
  }
  return Math.min(3, countCapturedComboTag(player, spec.zone, spec.tag));
}

function comboHandCount(player, comboKey) {
  const spec = COMBO_THREAT_SPECS[comboKey];
  if (!player || !spec) return 0;
  let count = 0;
  for (const card of player?.hand || []) {
    if (comboTargetMatchesSpec(card, spec)) count += 1;
  }
  return count;
}

function comboMissingMonths(player, comboKey) {
  const spec = COMBO_THREAT_SPECS[comboKey];
  if (!player || !spec) return [];
  if (comboKey === "kwang") {
    const capturedMonths = new Set(
      uniqueCapturedCards(player, "kwang").map((card) => Number(card?.month || 0)).filter((month) => month >= 1)
    );
    return spec.months.filter((month) => !capturedMonths.has(month));
  }
  const capturedMonths = capturedComboMonths(player, spec);
  return spec.months.filter((month) => !capturedMonths.has(month));
}

function comboProximityNormForPlayer(player, comboKey) {
  const captured = comboCapturedCount(player, comboKey);
  const missing = Math.max(0, 3 - captured);
  if (missing <= 1) return 1.0;
  if (missing === 2) return 0.5;
  return 0.0;
}

function comboCompletionNormForPlayer(player, comboKey) {
  return clamp01(comboCapturedCount(player, comboKey) / 3.0);
}

function comboTurnsEstimate(state, actor, comboKey) {
  const selfPlayer = state?.players?.[actor];
  const spec = COMBO_THREAT_SPECS[comboKey];
  if (!selfPlayer || !spec) return 10;
  const missingMonths = comboMissingMonths(selfPlayer, comboKey);
  if (missingMonths.length <= 0) return 0;

  let turns = 0;
  for (const month of missingMonths) {
    if (cardsContainComboTarget(selfPlayer?.hand || [], spec, month)) {
      turns += 1;
    } else if (cardsContainComboTarget(state?.board || [], spec, month)) {
      turns += 1;
    } else if (knownCountForMonth(state, actor, month) >= monthTotalCards(month)) {
      turns += 4;
    } else {
      turns += 2;
    }
  }
  return Math.min(10, turns);
}

function selfWinPathScore(state, actor) {
  let bestTurns = 10;
  for (const comboKey of COMBO_THREAT_KEYS) {
    bestTurns = Math.min(bestTurns, comboTurnsEstimate(state, actor, comboKey));
  }
  if (bestTurns <= 1) return 1.0;
  if (bestTurns <= 2) return 0.85;
  if (bestTurns <= 3) return 0.65;
  return clamp01(1.0 - (bestTurns / 10.0));
}

function selfMaterialConcentration(state, actor) {
  const selfPlayer = state?.players?.[actor];
  let best = 0;
  for (const comboKey of COMBO_THREAT_KEYS) {
    best = Math.max(best, comboCapturedCount(selfPlayer, comboKey) / 3.0);
  }
  return clamp01(best);
}

function selfComboCompletionNorm(state, actor) {
  const selfPlayer = state?.players?.[actor];
  let best = 0;
  for (const comboKey of COMBO_THREAT_KEYS) {
    const captured = comboCapturedCount(selfPlayer, comboKey);
    if (captured >= 3) return 1.0;
    const handCount = comboHandCount(selfPlayer, comboKey);
    best = Math.max(best, (captured + handCount) / 3.0);
  }
  return clamp01(best);
}

function selfGoSafetyNorm(state, actor) {
  const opp = actor === "human" ? "ai" : "human";
  const selfPlayer = state?.players?.[actor];
  const scoreSelf = calculateScore(state.players[actor], state.players[opp], state.ruleKey);
  const capturedPi = sumUniquePiLikeValue(collectCapturedCards(selfPlayer));
  const handSsangpi = sumUniqueSsangpiLikeValue(selfPlayer?.hand || []);
  const carry = Math.max(1.0, Number(state?.carryOverMultiplier || 1.0));
  const multiplier = Math.max(1.0, Number(scoreSelf?.multiplier || 1.0)) * carry;
  return clamp01((capturedPi + handSsangpi) / (6.0 + (multiplier * 1.5)));
}

function selfTempoAdvantage(state, actor) {
  const hand = handCardsForActor(state, actor);
  let score = 0;
  for (const card of hand) {
    let weight = 0;
    if (String(card?.category || "") === "kwang") weight = 3;
    else if (ssangpiLikeValue(card) > 0) weight = 2;
    else if (String(card?.category || "") === "ribbon" || String(card?.category || "") === "five") weight = 1;
    if (weight <= 0) continue;
    if (boardCountForMonth(state, Number(card?.month || 0)) > 0) score += weight;
  }
  return clamp01(score / Math.max(hand.length * 2, 1));
}

function oppWinProximityNorm(state, actor) {
  const opp = actor === "human" ? "ai" : "human";
  const oppPlayer = state?.players?.[opp];
  let comboBest = 0;
  for (const comboKey of COMBO_THREAT_KEYS) {
    comboBest = Math.max(comboBest, comboProximityNormForPlayer(oppPlayer, comboKey));
  }
  const piProx = clamp01(sumUniquePiLikeValue(collectCapturedCards(oppPlayer)) / 9.0);
  return clamp01(Math.max(comboBest, piProx * 0.8));
}

function oppSsangpiThreatNorm(state, actor) {
  const opp = actor === "human" ? "ai" : "human";
  const oppPlayer = state?.players?.[opp];
  const oppPi = sumUniquePiLikeValue(collectCapturedCards(oppPlayer));
  return clamp01((oppPi - 5.0) / 5.0);
}

function handCriticalBlockNorm(state, actor) {
  const proximity = oppWinProximityNorm(state, actor);
  if (proximity <= 0) return 0;
  const targets = collectOppComboTargetEntries(state, actor);
  if (targets.length <= 0) return 0;
  let held = 0;
  const hand = handCardsForActor(state, actor);
  for (const target of targets) {
    if (cardsContainComboTarget(hand, target.spec, target.month)) held += 1;
  }
  return clamp01(proximity * held / Math.max(targets.length, 1));
}

function oppGoThreatNorm(state, actor) {
  const opp = actor === "human" ? "ai" : "human";
  const scoreOpp = calculateScore(state.players[opp], state.players[actor], state.ruleKey);
  const carry = Math.max(1.0, Number(state?.carryOverMultiplier || 1.0));
  const oppMultiplier = Math.max(1.0, Number(scoreOpp?.multiplier || 1.0)) * carry;
  const oppGoCount = Math.max(0, Number(state?.players?.[opp]?.goCount || 0));
  return clamp01((oppGoCount * oppMultiplier) / 8.0);
}

function boardDangerRatio(state, actor) {
  const board = state?.board || [];
  if (!Array.isArray(board) || board.length <= 0) return 0;
  const targets = collectOppComboTargetEntries(state, actor);
  if (targets.length <= 0) return 0;
  let count = 0;
  for (const card of board) {
    for (const target of targets) {
      if (comboTargetMatchesSpec(card, target.spec, target.month)) {
        count += 1;
        break;
      }
    }
  }
  return clamp01(oppWinProximityNorm(state, actor) * count / Math.max(board.length, 4));
}

function buildPosition11FeatureVector(state, actor, decisionType, candidate) {
  const postState = buildVisibleAfterStateForHandFeatures(state, actor, decisionType, candidate);
  return [
    selfWinPathScore(postState, actor),
    selfMaterialConcentration(postState, actor),
    selfSsangpiControlNorm(postState, actor),
    selfComboCompletionNorm(postState, actor),
    selfGoSafetyNorm(postState, actor),
    selfTempoAdvantage(postState, actor),
    oppWinProximityNorm(postState, actor),
    oppSsangpiThreatNorm(postState, actor),
    handCriticalBlockNorm(postState, actor),
    oppGoThreatNorm(postState, actor),
    boardDangerRatio(postState, actor),
  ];
}

function buildHand10FeatureVector(state, actor, decisionType, candidate) {
  const card = candidateCard(state, actor, decisionType, candidate);
  const month = resolveCandidateMonth(state, actor, decisionType, card);
  const postState = buildVisibleAfterStateForHandFeatures(state, actor, decisionType, candidate);
  const postPiBlock = handOppPiBlockNorm(postState, actor);
  const postComboBlock = handOppBlockNorm(postState, actor);
  return [
    candidateComboGain(state, actor, decisionType, candidate),
    compactCandidatePiNorm(card),
    immediateMatchPossible(state, decisionType, month),
    isSafeDiscardMonth(state, actor, month) ? 1.0 : 0.0,
    candidateDangerExposure(state, actor, decisionType, candidate),
    handMatchableRatio(postState, actor),
    handTripleFlagNorm(postState, actor),
    handHighValueDensity(postState, actor),
    clamp01(Math.max(postPiBlock, postComboBlock)),
    globalContextTrigger(postState, actor)
  ];
}

function buildMaterial10StagingFeatureVector(state, actor, decisionType, candidate) {
  const opp = actor === "human" ? "ai" : "human";
  const scoreSelf = calculateScore(state.players[actor], state.players[opp], state.ruleKey);
  const scoreOpp = calculateScore(state.players[opp], state.players[actor], state.ruleKey);
  const month = resolveCandidateMonth(state, actor, decisionType, candidateCard(state, actor, decisionType, candidate));

  return [
    clamp01(currentMultiplierNorm(state, scoreSelf)),
    candidateComboGain(state, actor, decisionType, candidate),
    oppComboThreatNorm(state, actor),
    candidateBlockGainNorm(state, actor, decisionType, candidate),
    candidatePublicKnownRatio(state, actor, month),
    immediateMatchPossible(state, decisionType, month),
    selfSsangpiControlNorm(state, actor),
    ssangpiRevealedRatioNorm(state, actor),
    compactOppStopPressureNorm(scoreOpp?.total || 0),
    tanhNorm((scoreSelf?.total || 0) - (scoreOpp?.total || 0), 10.0)
  ];
}

function normalizeFeatureProfile(featureSpec, inputDim) {
  const profile = String(featureSpec?.profile || "").trim().toLowerCase();
  if (profile === "material10") return "material10";
  if (profile === "hand10") return "hand10";
  if (profile === "position11") return "position11";
  if (inputDim === LEGACY13_FEATURES) return "legacy13";
  if (inputDim === POSITION11_FEATURES) return "position11";
  return DEFAULT_FEATURE_PROFILE;
}

function featureVector(state, actor, decisionType, candidate, legalCount, inputDim, featureSpec = null) {
  const profile = normalizeFeatureProfile(featureSpec, inputDim);
  let features = null;
  if (profile === "legacy13") {
    if (inputDim !== LEGACY13_FEATURES) {
      throw new Error(
        `feature vector size mismatch: expected ${inputDim}, supported=${LEGACY13_FEATURES}(legacy13),${HAND10_FEATURES}(hand10),${MATERIAL10_STAGING_FEATURES}(material10),${POSITION11_FEATURES}(position11)`
      );
    }
    features = buildLegacy13FeatureVector(state, actor, decisionType, candidate);
  } else if (profile === "hand10") {
    if (inputDim !== HAND10_FEATURES) {
      throw new Error(
        `feature vector size mismatch: expected ${inputDim}, supported=${LEGACY13_FEATURES}(legacy13),${HAND10_FEATURES}(hand10),${MATERIAL10_STAGING_FEATURES}(material10),${POSITION11_FEATURES}(position11)`
      );
    }
    features = buildHand10FeatureVector(state, actor, decisionType, candidate);
  } else if (profile === "position11") {
    if (inputDim !== POSITION11_FEATURES) {
      throw new Error(
        `feature vector size mismatch: expected ${inputDim}, supported=${LEGACY13_FEATURES}(legacy13),${HAND10_FEATURES}(hand10),${MATERIAL10_STAGING_FEATURES}(material10),${POSITION11_FEATURES}(position11)`
      );
    }
    features = buildPosition11FeatureVector(state, actor, decisionType, candidate);
  } else {
    if (inputDim !== MATERIAL10_STAGING_FEATURES) {
      throw new Error(
        `feature vector size mismatch: expected ${inputDim}, supported=${LEGACY13_FEATURES}(legacy13),${HAND10_FEATURES}(hand10),${MATERIAL10_STAGING_FEATURES}(material10),${POSITION11_FEATURES}(position11)`
      );
    }
    features = buildMaterial10StagingFeatureVector(state, actor, decisionType, candidate);
  }
  if (features.length !== inputDim) {
    throw new Error(`compact feature length mismatch: expected ${inputDim}, got ${features.length}`);
  }
  return features;
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
  const decisionParamsRaw = raw?.decision_params && typeof raw.decision_params === "object"
    ? raw.decision_params
    : {};
  const decisionParams = {
    go_stop_threshold: Number(decisionParamsRaw?.go_stop_threshold || 0),
    shaking_threshold: Number(decisionParamsRaw?.shaking_threshold || 0),
    president_threshold: Number(decisionParamsRaw?.president_threshold || 0),
    gukjin_threshold: Number(decisionParamsRaw?.gukjin_threshold || 0)
  };

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
    featureSpec: raw?.feature_spec && typeof raw.feature_spec === "object" ? raw.feature_spec : null,
    decisionParams,
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

function isTwoOutputThresholdModel(compiled) {
  return Array.isArray(compiled?.outputKeys) && compiled.outputKeys.length === 2;
}

function resolvePositiveOptionAction(candidate) {
  const action = canonicalOptionAction(candidate);
  if (action === "go" || action === "stop") return "go";
  if (action === "shaking_yes" || action === "shaking_no") return "shaking_yes";
  if (action === "president_stop" || action === "president_hold") return "president_hold";
  if (action === "five" || action === "junk") return "five";
  return null;
}

function resolveNegativeOptionAction(positiveAction) {
  if (positiveAction === "go") return "stop";
  if (positiveAction === "shaking_yes") return "shaking_no";
  if (positiveAction === "president_hold") return "president_stop";
  if (positiveAction === "five") return "junk";
  return null;
}

function resolveDecisionThreshold(compiled, positiveAction) {
  const params = compiled?.decisionParams || {};
  if (positiveAction === "go") return Number(params.go_stop_threshold || 0);
  if (positiveAction === "shaking_yes") return Number(params.shaking_threshold || 0);
  if (positiveAction === "president_hold") return Number(params.president_threshold || 0);
  if (positiveAction === "five") return Number(params.gukjin_threshold || 0);
  return 0.0;
}

function buildTwoOutputOptionScoreBundle(state, actor, compiled, candidates) {
  const positiveAction = candidates.map((candidate) => resolvePositiveOptionAction(candidate)).find(Boolean) || null;
  if (!positiveAction) return null;
  const negativeAction = resolveNegativeOptionAction(positiveAction);
  if (!negativeAction) return null;
  const inputDim = Number(compiled?.inputKeys?.length || 0);
  const features = featureVector(state, actor, "option", positiveAction, candidates.length, inputDim, compiled?.featureSpec);
  const outputs = forward(compiled, features);
  const optionBias = Number(outputs[NEAT_OUT_OPTION_BIAS] || 0.0);
  const threshold = resolveDecisionThreshold(compiled, positiveAction);
  const scoreMap = new Map();
  for (const candidate of candidates) {
    const action = canonicalOptionAction(candidate);
    if (action === positiveAction) {
      scoreMap.set(candidate, optionBias - threshold);
    } else if (action === negativeAction) {
      scoreMap.set(candidate, threshold - optionBias);
    } else {
      scoreMap.set(candidate, -Infinity);
    }
  }
  return { scoreMap, optionBias, threshold, positiveAction, negativeAction };
}

function candidateOutputIndex(compiled, decisionType, candidate) {
  const outputCount = Array.isArray(compiled?.outputKeys) ? compiled.outputKeys.length : 0;
  if (outputCount === 2) {
    if (decisionType === "option") return NEAT_OUT_OPTION_BIAS;
    return NEAT_OUT_ACTION_SCORE;
  }
  if (outputCount === 1) return 0;
  throw new Error(`unsupported NEAT output count: ${outputCount}`);
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

  const scoreMap = new Map();
  const scores = {};
  try {
    if (decisionType === "option" && isTwoOutputThresholdModel(compiled)) {
      const bundle = buildTwoOutputOptionScoreBundle(state, actor, compiled, candidates);
      if (!bundle) return null;
      for (const candidate of candidates) {
        const score = Number(bundle.scoreMap.get(candidate) || -Infinity);
        scoreMap.set(candidate, score);
        scores[String(candidate)] = score;
      }
      const probs = scoreToProbabilityMap(candidates, scoreMap, Number(policyModel?.softmax_temp || 1.0));
      return {
        decisionType,
        candidates,
        probabilities: probs,
        scores,
        optionBias: bundle.optionBias,
        optionThreshold: bundle.threshold,
        optionPositiveAction: bundle.positiveAction,
        optionNegativeAction: bundle.negativeAction
      };
    }

    const inputDim = Number(compiled.inputKeys.length || 0);
    for (const candidate of candidates) {
      const x = featureVector(state, actor, decisionType, candidate, candidates.length, inputDim, compiled?.featureSpec);
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
    Number(options.inputDim || 0) || Number(compiled?.inputKeys?.length || 0) || HAND10_FEATURES
  );
  const optionBundle = compiled && decisionType === "option" && isTwoOutputThresholdModel(compiled)
    ? buildTwoOutputOptionScoreBundle(state, actor, compiled, candidates)
    : null;

  const rows = [];
  for (const candidate of candidates) {
    const features = featureVector(state, actor, decisionType, candidate, candidates.length, inputDim, compiled?.featureSpec);
    let score = null;
    if (compiled) {
      if (optionBundle) {
        score = Number(optionBundle.scoreMap.get(candidate) || -Infinity);
      } else {
        const outputs = forward(compiled, features);
        score = scoreCandidateFromOutputs(compiled, outputs, decisionType, candidate);
      }
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
    optionBias: optionBundle?.optionBias ?? null,
    optionThreshold: optionBundle?.threshold ?? null,
    optionPositiveAction: optionBundle?.positiveAction ?? null,
    rows,
  };
}

function modelPickCandidate(state, actor, policyModel) {
  const scored = getModelCandidateProbabilities(state, actor, policyModel);
  if (!scored) return null;
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
