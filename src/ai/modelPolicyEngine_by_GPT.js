import {
  calculateScore,
  scoringPiCount,
  getDeclarableShakingMonths,
  getDeclarableBombMonths
} from "../engine/index.js";
import {
  canonicalOptionAction,
  parsePlaySpecialCandidate,
  selectDecisionPool,
  resolveDecisionType,
  legalCandidatesForDecision,
  applyDecisionAction
} from "./decisionRuntime_by_GPT.js";

/* ============================================================================
 * NEAT model policy runtime
 * GPT line copy (independent from modelPolicyEngine.js).
 * Safe to modify for GPT-only feature experiments.
 * - compile genome once and cache
 * - score legal candidates
 * - convert score to action/state transition
 * ========================================================================== */
const NEAT_MODEL_FORMAT = "neat_python_genome_v1";
const COMPILED_NEAT_CACHE = new WeakMap();
const NEAT_COMPACT_FEATURES = 16;
const LEGACY_NEAT_COMPACT_FEATURES = 13;

/* 1) Feature extraction helpers */
function clamp01(x) {
  const v = Number(x || 0);
  if (v <= 0) return 0;
  if (v >= 1) return 1;
  return v;
}

function clampSigned(x, limit = 1) {
  const v = Number(x || 0);
  const cap = Math.max(1e-6, Number(limit || 1));
  if (v <= -cap) return -cap;
  if (v >= cap) return cap;
  return v;
}

function tanhNorm(x, scale) {
  const s = Math.max(1e-6, Number(scale || 1));
  return Math.tanh(Number(x || 0) / s);
}

function mean(values) {
  if (!Array.isArray(values) || values.length <= 0) return 0;
  return values.reduce((acc, v) => acc + Number(v || 0), 0) / values.length;
}

function normalizeDecisionCandidate(decisionType, candidate) {
  if (decisionType === "option") return canonicalOptionAction(candidate);
  return String(candidate || "").trim();
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

function countCardsByMonth(cards, month) {
  const targetMonth = Number(month || 0);
  if (!Array.isArray(cards) || targetMonth <= 0) return 0;
  let count = 0;
  for (const card of cards) {
    if (Number(card?.month || 0) === targetMonth) count += 1;
  }
  return count;
}

function countCapturedZoneUnique(player, zone) {
  const cards = player?.captured?.[zone] || [];
  if (!Array.isArray(cards)) return 0;
  const seen = new Set();
  for (const card of cards) {
    const id = String(card?.id || "");
    if (!id) continue;
    seen.add(id);
  }
  return seen.size;
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
  for (const side of ["human", "ai"]) {
    const captured = state?.players?.[side]?.captured || {};
    pushAll(captured.kwang || []);
    pushAll(captured.five || []);
    pushAll(captured.ribbon || []);
    pushAll(captured.junk || []);
  }
  pushAll(state?.players?.[actor]?.hand || []);
  pushAll(getPublicKnownOpponentHandCards(state, actor));
  return out;
}

function candidateMonthKnownRatio(state, actor, month) {
  const total = monthTotalCards(month);
  if (total <= 0) return 0;
  const cards = collectKnownCardsForMonthRatio(state, actor);
  const seen = new Set();
  let known = 0;
  for (const card of cards) {
    if (!card) continue;
    const id = String(card?.id || "");
    if (!id || seen.has(id)) continue;
    seen.add(id);
    if (Number(card?.month || 0) === Number(month)) known += 1;
  }
  return clamp01(known / total);
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
  return clamp01(((mul * carry) - 1.0) / 15.0);
}

function uniqueCapturedCardCount(player, zone) {
  const cards = player?.captured?.[zone] || [];
  if (!Array.isArray(cards)) return 0;
  const seen = new Set();
  for (const card of cards) {
    const id = String(card?.id || "");
    if (!id) continue;
    seen.add(id);
  }
  return seen.size;
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

  const visibleState = maskStateForVisibleComboSimulation(state);
  const afterState = applyDecisionAction(visibleState, actor, decisionType, candidate);
  const afterPlayer = afterState?.players?.[actor];
  if (!afterPlayer) return 0;

  const beforeGwang = uniqueCapturedCardCount(beforePlayer, "kwang");
  const afterGwang = uniqueCapturedCardCount(afterPlayer, "kwang");
  const beforeGodori = countCapturedComboTag(beforePlayer, "five", "fiveBirds");
  const afterGodori = countCapturedComboTag(afterPlayer, "five", "fiveBirds");
  const ribbonTags = ["redRibbons", "blueRibbons", "plainRibbons"];
  const completesDan = ribbonTags.some((tag) => {
    const beforeCount = countCapturedComboTag(beforePlayer, "ribbon", tag);
    const afterCount = countCapturedComboTag(afterPlayer, "ribbon", tag);
    return beforeCount < 3 && afterCount >= 3;
  });
  const raw =
    (beforeGwang < 3 && afterGwang >= 3 ? 3 : 0) +
    (beforeGodori < 3 && afterGodori >= 3 ? 5 : 0) +
    (completesDan ? 3 : 0);
  return clamp01(raw / 11.0);
}

function buildDecisionBaseContext(state, actor, decisionType, legalCount) {
  const opp = actor === "human" ? "ai" : "human";
  const scoreSelf = calculateScore(state.players[actor], state.players[opp], state.ruleKey);
  const scoreOpp = calculateScore(state.players[opp], state.players[actor], state.ruleKey);
  const { hasShake, hasBomb } = decisionAvailabilityFlags(state, actor);

  return {
    state,
    actor,
    opp,
    decisionType,
    legalCount: Number(legalCount || 0),
    phase: String(state?.phase || ""),
    scoreSelf,
    scoreOpp,
    boardCount: Array.isArray(state?.board) ? state.board.length : 0,
    selfGwangCount: countCapturedZoneUnique(state?.players?.[actor], "kwang"),
    oppGwangCount: countCapturedZoneUnique(state?.players?.[opp], "kwang"),
    selfRibbonCount: countCapturedZoneUnique(state?.players?.[actor], "ribbon"),
    oppRibbonCount: countCapturedZoneUnique(state?.players?.[opp], "ribbon"),
    selfFiveCount: countCapturedZoneUnique(state?.players?.[actor], "five"),
    oppFiveCount: countCapturedZoneUnique(state?.players?.[opp], "five"),
    selfPiCount: Number(scoringPiCount(state.players[actor]) || 0),
    oppPiCount: Number(scoringPiCount(state.players[opp]) || 0),
    selfGodori: countCapturedComboTag(state.players?.[actor], "five", "fiveBirds"),
    oppGodori: countCapturedComboTag(state.players?.[opp], "five", "fiveBirds"),
    selfCheongdan: countCapturedComboTag(state.players?.[actor], "ribbon", "blueRibbons"),
    oppCheongdan: countCapturedComboTag(state.players?.[opp], "ribbon", "blueRibbons"),
    selfHongdan: countCapturedComboTag(state.players?.[actor], "ribbon", "redRibbons"),
    oppHongdan: countCapturedComboTag(state.players?.[opp], "ribbon", "redRibbons"),
    selfChodan: countCapturedComboTag(state.players?.[actor], "ribbon", "plainRibbons"),
    oppChodan: countCapturedComboTag(state.players?.[opp], "ribbon", "plainRibbons"),
    selfCanStop: Number(scoreSelf?.total || 0) >= 7 ? 1 : 0,
    oppCanStop: Number(scoreOpp?.total || 0) >= 7 ? 1 : 0,
    hasShake,
    hasBomb,
    multiplierNorm: currentMultiplierNorm(state, scoreSelf),
    bakPi: scoreSelf?.bak?.pi ? 1 : 0,
    bakGwang: scoreSelf?.bak?.gwang ? 1 : 0,
    bakMongBak: scoreSelf?.bak?.mongBak ? 1 : 0
  };
}

function buildCandidateDescriptor(base, candidate) {
  const state = base.state;
  const actor = base.actor;
  const decisionType = base.decisionType;
  const normalizedCandidate = normalizeDecisionCandidate(decisionType, candidate);
  const card = candidateCard(state, actor, decisionType, normalizedCandidate);
  const month = resolveCandidateMonth(state, actor, decisionType, card);
  const piValue = Number(card?.piValue || 0);
  const category = String(card?.category || "");

  return {
    candidate: normalizedCandidate,
    month,
    piNorm: clamp01(piValue / 5.0),
    comboGain: candidateComboGain(state, actor, decisionType, normalizedCandidate),
    category,
    isKwang: category === "kwang" ? 1 : 0,
    isRibbon: category === "ribbon" ? 1 : 0,
    isFive: category === "five" ? 1 : 0,
    isJunk: category === "junk" ? 1 : 0,
    isDoublePi: isDoublePiCard(card) ? 1 : 0,
    matchDensity: matchOpportunityDensity(state, month),
    immediateMatch: immediateMatchPossible(state, decisionType, month),
    knownRatio: candidateMonthKnownRatio(state, actor, month),
    optionCode: optionCode(normalizedCandidate),
    sameMonthHandCountNorm: clamp01(countCardsByMonth(state?.players?.[actor]?.hand || [], month) / 4.0),
    sameMonthBoardCountNorm: clamp01(countCardsByMonth(state?.board || [], month) / 3.0),
    sameMonthLegalShare: 0,
    categoryShare: 0,
    piAdv: 0,
    matchDensityAdv: 0,
    knownRatioAdv: 0
  };
}

function summarizeCandidateDescriptors(descriptors) {
  const monthCounts = new Map();
  const categoryCounts = new Map();
  for (const desc of descriptors) {
    const monthKey = Number(desc?.month || 0);
    if (monthKey > 0) {
      monthCounts.set(monthKey, Number(monthCounts.get(monthKey) || 0) + 1);
    }
    const categoryKey = String(desc?.category || "");
    if (categoryKey) {
      categoryCounts.set(categoryKey, Number(categoryCounts.get(categoryKey) || 0) + 1);
    }
  }
  return {
    maxPiNorm: descriptors.reduce((best, d) => Math.max(best, Number(d?.piNorm || 0)), 0),
    meanPiNorm: mean(descriptors.map((d) => Number(d?.piNorm || 0))),
    maxMatchDensity: descriptors.reduce((best, d) => Math.max(best, Number(d?.matchDensity || 0)), 0),
    meanMatchDensity: mean(descriptors.map((d) => Number(d?.matchDensity || 0))),
    meanKnownRatio: mean(descriptors.map((d) => Number(d?.knownRatio || 0))),
    monthCounts,
    categoryCounts
  };
}

function enrichCandidateDescriptors(descriptors, stats) {
  const total = Math.max(1, descriptors.length);
  return descriptors.map((desc) => {
    const monthCount = Number(stats.monthCounts.get(Number(desc.month || 0)) || 0);
    const categoryCount = Number(stats.categoryCounts.get(String(desc.category || "")) || 0);
    return {
      ...desc,
      sameMonthLegalShare: clamp01(monthCount / total),
      categoryShare: clamp01(categoryCount / total),
      piAdv: clampSigned(Number(desc.piNorm || 0) - Number(stats.meanPiNorm || 0), 1),
      matchDensityAdv: clampSigned(Number(desc.matchDensity || 0) - Number(stats.meanMatchDensity || 0), 1),
      knownRatioAdv: clampSigned(Number(desc.knownRatio || 0) - Number(stats.meanKnownRatio || 0), 1)
    };
  });
}

function buildLegacyFeatureVector(base, desc) {
  return [
    base.phase === "playing" ? 1 : 0,
    base.phase === "select-match" ? 1 : 0,
    base.phase === "go-stop" ? 1 : 0,
    base.phase === "president-choice" ? 1 : 0,
    base.phase === "gukjin-choice" ? 1 : 0,
    base.phase === "shaking-confirm" ? 1 : 0,

    base.decisionType === "play" ? 1 : 0,
    base.decisionType === "match" ? 1 : 0,
    base.decisionType === "option" ? 1 : 0,

    clamp01((base.state?.deck?.length || 0) / 30.0),
    clamp01((base.state?.players?.[base.actor]?.hand?.length || 0) / 10.0),
    clamp01((base.state?.players?.[base.opp]?.hand?.length || 0) / 10.0),
    clamp01(base.boardCount / 24.0),
    clamp01(base.legalCount / 10.0),
    clamp01((base.state?.players?.[base.actor]?.goCount || 0) / 5.0),
    clamp01((base.state?.players?.[base.opp]?.goCount || 0) / 5.0),
    tanhNorm((base.scoreSelf?.total || 0) - (base.scoreOpp?.total || 0), 10.0),
    tanhNorm(base.scoreSelf?.total || 0, 10.0),
    tanhNorm(base.scoreOpp?.total || 0, 10.0),
    base.selfCanStop,
    base.oppCanStop,
    clamp01(base.multiplierNorm * 1.25),

    clamp01(base.selfPiCount / 20.0),
    clamp01(base.oppPiCount / 20.0),
    clamp01(base.selfGwangCount / 5.0),
    clamp01(base.oppGwangCount / 5.0),
    clamp01(base.selfRibbonCount / 10.0),
    clamp01(base.oppRibbonCount / 10.0),
    clamp01(base.selfFiveCount / 5.0),
    clamp01(base.oppFiveCount / 5.0),

    desc.piNorm,
    desc.isKwang,
    desc.isRibbon,
    desc.isFive,
    desc.isJunk,
    desc.isDoublePi,
    desc.matchDensity,
    desc.immediateMatch,
    desc.knownRatio,
    desc.optionCode
  ];
}

function buildCompactFeatureVector(base, desc) {
  const isGoStopOption = base.decisionType === "option" && base.phase === "go-stop";
  const selfGoCountNorm = isGoStopOption ? clamp01((base.state?.players?.[base.actor]?.goCount || 0) / 5.0) : 0.0;
  const oppGoCountNorm = isGoStopOption ? clamp01((base.state?.players?.[base.opp]?.goCount || 0) / 5.0) : 0.0;
  const isGoCandidate = !isGoStopOption ? 0.5 : desc.candidate === "go" ? 1.0 : desc.candidate === "stop" ? 0.0 : 0.5;
  return [
    base.decisionType === "play" ? 1 : 0,
    base.decisionType === "match" ? 1 : 0,
    desc.optionCode,
    tanhNorm((base.scoreSelf?.total || 0) - (base.scoreOpp?.total || 0), 10.0),
    tanhNorm(base.scoreSelf?.total || 0, 10.0),
    clamp01((base.scoreOpp?.total || 0) / 7.0),
    clamp01(base.multiplierNorm),
    desc.comboGain,
    desc.piNorm,
    desc.immediateMatch,
    desc.knownRatio,
    base.selfCanStop,
    base.oppCanStop,
    isGoCandidate,
    selfGoCountNorm,
    oppGoCountNorm
  ];
}

function buildLegacyCompactFeatureVector(base, desc) {
  return [
    base.decisionType === "play" ? 1 : 0,
    base.decisionType === "match" ? 1 : 0,
    desc.optionCode,
    tanhNorm((base.scoreSelf?.total || 0) - (base.scoreOpp?.total || 0), 10.0),
    tanhNorm(base.scoreSelf?.total || 0, 10.0),
    clamp01((base.scoreOpp?.total || 0) / 7.0),
    clamp01(base.multiplierNorm),
    desc.comboGain,
    desc.piNorm,
    desc.immediateMatch,
    desc.knownRatio,
    base.selfCanStop,
    base.oppCanStop
  ];
}

function buildCoreRichFeatureVector(base, desc) {
  return [
    base.phase === "playing" ? 1 : 0,
    base.phase === "select-match" ? 1 : 0,
    base.phase === "go-stop" ? 1 : 0,
    base.phase === "president-choice" ? 1 : 0,
    base.phase === "gukjin-choice" ? 1 : 0,
    base.phase === "shaking-confirm" ? 1 : 0,

    base.decisionType === "play" ? 1 : 0,
    base.decisionType === "match" ? 1 : 0,
    base.decisionType === "option" ? 1 : 0,

    clamp01((base.state?.deck?.length || 0) / 30.0),
    clamp01((base.state?.players?.[base.actor]?.hand?.length || 0) / 10.0),
    clamp01((base.state?.players?.[base.opp]?.hand?.length || 0) / 10.0),
    clamp01((base.state?.players?.[base.actor]?.goCount || 0) / 5.0),
    clamp01((base.state?.players?.[base.opp]?.goCount || 0) / 5.0),
    tanhNorm((base.scoreSelf?.total || 0) - (base.scoreOpp?.total || 0), 10.0),
    tanhNorm(base.scoreSelf?.total || 0, 10.0),
    clamp01(base.legalCount / 10.0),

    desc.piNorm,
    desc.isKwang,
    desc.isRibbon,
    desc.isFive,
    desc.isJunk,
    desc.isDoublePi,

    desc.matchDensity,
    desc.immediateMatch,
    desc.optionCode,

    clamp01(base.selfGwangCount / 5.0),
    clamp01(base.oppGwangCount / 5.0),
    clamp01(base.selfPiCount / 20.0),
    clamp01(base.oppPiCount / 20.0),

    clamp01(base.selfGodori / 3.0),
    clamp01(base.oppGodori / 3.0),
    clamp01(base.selfCheongdan / 3.0),
    clamp01(base.oppCheongdan / 3.0),
    clamp01(base.selfHongdan / 3.0),
    clamp01(base.oppHongdan / 3.0),
    clamp01(base.selfChodan / 3.0),
    clamp01(base.oppChodan / 3.0),

    base.selfCanStop,
    base.oppCanStop,

    base.hasShake,
    base.multiplierNorm,
    base.hasBomb,

    base.bakPi,
    base.bakGwang,
    base.bakMongBak,

    desc.knownRatio
  ];
}

function buildExtendedFeatureVector(base, desc, stats) {
  return [
    ...buildCoreRichFeatureVector(base, desc),
    clamp01(stats.maxPiNorm),
    desc.piAdv,
    clamp01(stats.maxMatchDensity),
    desc.matchDensityAdv,
    desc.knownRatioAdv,
    desc.sameMonthLegalShare,
    desc.sameMonthHandCountNorm,
    desc.sameMonthBoardCountNorm,
    desc.categoryShare
  ];
}

function buildFeatureVector(base, desc, stats, inputDim) {
  if (inputDim === NEAT_COMPACT_FEATURES) {
    const features = buildCompactFeatureVector(base, desc);
    if (features.length !== NEAT_COMPACT_FEATURES) {
      throw new Error(`compact feature length mismatch: expected ${NEAT_COMPACT_FEATURES}, got ${features.length}`);
    }
    return features;
  }
  if (inputDim === LEGACY_NEAT_COMPACT_FEATURES) {
    const features = buildLegacyCompactFeatureVector(base, desc);
    if (features.length !== LEGACY_NEAT_COMPACT_FEATURES) {
      throw new Error(`legacy compact feature length mismatch: expected ${LEGACY_NEAT_COMPACT_FEATURES}, got ${features.length}`);
    }
    return features;
  }
  if (inputDim === 40) return buildLegacyFeatureVector(base, desc);
  if (inputDim === 47) return buildCoreRichFeatureVector(base, desc);
  if (inputDim === 56) return buildExtendedFeatureVector(base, desc, stats);
  throw new Error(`feature vector size mismatch: expected ${inputDim}, supported=16,13,40,47,56`);
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

  const outKey = compiled.outputKeys.length > 0 ? Number(compiled.outputKeys[0]) : null;
  if (outKey == null) return 0.0;
  return Number(values.get(outKey) || 0.0);
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

function scoreDecisionCandidates(state, actor, policyModel, options = {}) {
  const compiled = getCompiledNeatModel(policyModel);
  if (!compiled) return null;

  const sp = selectDecisionPool(state, actor, options);
  const decisionType = resolveDecisionType(sp);
  if (!decisionType) return null;

  const candidates = legalCandidatesForDecision(sp, decisionType).map((c) => normalizeDecisionCandidate(decisionType, c));
  if (!candidates.length) return null;

  const base = buildDecisionBaseContext(state, actor, decisionType, candidates.length);
  const rawDescriptors = candidates.map((candidate) => buildCandidateDescriptor(base, candidate));
  const stats = summarizeCandidateDescriptors(rawDescriptors);
  const descriptors = enrichCandidateDescriptors(rawDescriptors, stats);
  const inputDim = Number(compiled.inputKeys.length || 0);
  const scoreMap = new Map();
  const scores = {};
  const diagnostics = {};
  const featureMode =
    inputDim === 56
      ? "rich56"
      : inputDim === 47
        ? "rich47"
        : inputDim === 40
          ? "legacy40"
          : inputDim === NEAT_COMPACT_FEATURES
            ? "compact16"
            : inputDim === LEGACY_NEAT_COMPACT_FEATURES
              ? "compact13"
              : "unknown";

  for (const desc of descriptors) {
    const x = buildFeatureVector(base, desc, stats, inputDim);
    const score = forward(compiled, x);
    scoreMap.set(desc.candidate, score);
    scores[String(desc.candidate)] = score;
    diagnostics[String(desc.candidate)] = {
      feature_mode: featureMode,
      combo_gain: desc.comboGain,
      pi_norm: desc.piNorm,
      match_density: desc.matchDensity,
      known_ratio: desc.knownRatio,
      same_month_legal_share: desc.sameMonthLegalShare,
      same_month_hand_count_norm: desc.sameMonthHandCountNorm,
      same_month_board_count_norm: desc.sameMonthBoardCountNorm,
      category_share: desc.categoryShare,
      pi_adv: desc.piAdv,
      match_density_adv: desc.matchDensityAdv,
      known_ratio_adv: desc.knownRatioAdv,
      raw_score: score
    };
  }

  const probs = scoreToProbabilityMap(candidates, scoreMap, Number(policyModel?.softmax_temp || 1.0));
  const chosenCandidate = pickBestByScore(candidates, scoreMap);
  return {
    analysis_version: "gpt_rich_candidate_context_v1",
    decisionType,
    candidates,
    chosenCandidate,
    probabilities: probs,
    scores,
    diagnostics,
    feature_mode: featureMode
  };
}

/* 6) Public APIs */
export function getModelCandidateProbabilities(state, actor, policyModel, options = {}) {
  return scoreDecisionCandidates(state, actor, policyModel, options);
}

export function getModelDecisionAnalysis(state, actor, policyModel, options = {}) {
  return scoreDecisionCandidates(state, actor, policyModel, options);
}

export function applyModelDecisionAnalysis(state, actor, analysis) {
  if (!analysis || !analysis.decisionType) {
    throw new Error(`[modelPolicyEngine_by_GPT] missing analysis payload (actor=${actor}, phase=${String(state?.phase || "")})`);
  }

  const sp = selectDecisionPool(state, actor);
  const decisionType = resolveDecisionType(sp);
  if (!decisionType || decisionType !== analysis.decisionType) {
    throw new Error(
      `[modelPolicyEngine_by_GPT] analysis decision type mismatch (actor=${actor}, phase=${String(state?.phase || "")}, expected=${decisionType}, actual=${String(analysis.decisionType || "")})`
    );
  }

  const legal = legalCandidatesForDecision(sp, decisionType).map((c) => normalizeDecisionCandidate(decisionType, c));
  if (!legal.length) {
    throw new Error(
      `[modelPolicyEngine_by_GPT] legal candidates empty (actor=${actor}, phase=${String(state?.phase || "")}, decisionType=${decisionType})`
    );
  }

  const picked = normalizeDecisionCandidate(decisionType, analysis.chosenCandidate);
  if (!picked || !legal.includes(picked)) {
    throw new Error(
      `[modelPolicyEngine_by_GPT] illegal analysis candidate (actor=${actor}, phase=${String(state?.phase || "")}, decisionType=${decisionType}, candidate=${String(picked)})`
    );
  }

  const next = applyDecisionAction(state, actor, decisionType, picked);
  if (!next || next === state) {
    throw new Error(
      `[modelPolicyEngine_by_GPT] action apply failed (actor=${actor}, phase=${String(state?.phase || "")}, decisionType=${decisionType}, candidate=${String(picked)})`
    );
  }
  return next;
}

export function modelPolicyPlay(state, actor, policyModel) {
  if (!policyModel || !isNeatModel(policyModel)) {
    throw new Error(
      `[modelPolicyEngine_by_GPT] invalid model format (actor=${actor}, phase=${String(state?.phase || "")})`
    );
  }

  const analysis = getModelDecisionAnalysis(state, actor, policyModel);
  if (!analysis) {
    throw new Error(
      `[modelPolicyEngine_by_GPT] model candidate unresolved (actor=${actor}, phase=${String(state?.phase || "")})`
    );
  }

  return applyModelDecisionAnalysis(state, actor, analysis);
}
