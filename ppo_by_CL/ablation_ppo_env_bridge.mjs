#!/usr/bin/env node
// ppo_env_bridge.mjs
// - StdIO JSON bridge between Matgo engine and PPO trainer.
// - Strict CLI/runtime validation + fail-fast errors with context.
// Execution order:
// train_ppo.py/duel_ppo_vs_v5.py -> this bridge(reset/step) -> Matgo engine decision runtime.
//
// Execution Flow Map:
// 1) parseArgs() + buildRuntime(): strict CLI/runtime bootstrap
// 2) handleReset(): initialize episode and return first observation/mask
// 3) handleStep(): apply one policy action, auto-run opponent if needed, compute reward
// 4) stdin command loop: dispatch reset/step/close and stream JSON responses

import {
  initSimulationGame,
  createSeededRng,
  calculateScore,
  scoringPiCount,
  getDeclarableShakingMonths,
  getDeclarableBombMonths,
  ruleSets
} from "../../src/engine/index.js";
import { getActionPlayerKey } from "../../src/engine/runner.js";
import { aiPlay } from "../../src/ai/aiPlay.js";
import { normalizeBotPolicy, resolveBotPolicy } from "../../src/ai/policies.js";
import {
  selectDecisionPool,
  resolveDecisionType,
  legalCandidatesForDecision,
  applyDecisionAction,
  canonicalOptionAction,
  stateProgressKey
} from "../../src/ai/decisionRuntime_by_GPT.js";

const ACTION_DIM = 26;
const PLAY_SLOTS = 10;
const MATCH_SLOTS = 8;
const CARD_FEATURE_DIM = 7;
const PHASE_FEATURE_DIM = 6;
const DECISION_FEATURE_DIM = 3;
const MONTH_BUCKETS = 12;
const MONTH_PROFILE_FEATURE_DIM = MONTH_BUCKETS * 5; // board/self_captured/opp_captured/self_hand/opp_shaked
const STRATEGY_AUX_FEATURE_DIM = 2; // multiplier_risk_norm, go_expected_value_norm
const MACRO_FEATURE_DIM = 41 + MONTH_PROFILE_FEATURE_DIM + STRATEGY_AUX_FEATURE_DIM;
const OPTION_ORDER = Object.freeze([
  "go",
  "stop",
  "shaking_yes",
  "shaking_no",
  "president_stop",
  "president_hold",
  "five",
  "junk"
]);
const OPTION_FEATURE_DIM = OPTION_ORDER.length;
const OBS_DIM =
  PHASE_FEATURE_DIM +
  DECISION_FEATURE_DIM +
  MACRO_FEATURE_DIM +
  PLAY_SLOTS * CARD_FEATURE_DIM +
  MATCH_SLOTS * CARD_FEATURE_DIM +
  OPTION_FEATURE_DIM;

const PHASE1_SHAPING = Object.freeze({
  godoriComplete: 0.18,
  cheongdanComplete: 0.16,
  hongdanComplete: 0.16,
  chodanComplete: 0.16,
  gwang3Complete: 0.20,
  gwang4Complete: 0.32,
  gwang5Complete: 0.50,
  jjob: 0.04,
  ppuk: 0.03,
  yeonPpuk: 0.07,
  ddadak: 0.04,
  jabbeok: 0.04,
  pansseul: 0.05,
  shaking: 0.05,
  bomb: 0.05,
  goScoreUp: 0.12,
  goScoreHold: 0.05,
  goWin: 0.50,
  goLoss: -0.25,
  stopWin: -0.10,
  doublePiGet: 0.03,
  pi10Reached: 0.06,
  piBakRisk: -0.10,
  gwangBakRisk: -0.10,
  piBakSuffered: -0.40,
  gwangBakSuffered: -0.40,
  mongBakSuffered: -0.40
});

// =============================================================================
// Section 1. Console/CLI Validation Helpers
// =============================================================================

function fail(message) {
  throw new Error(String(message || "unknown failure"));
}

function formatConsoleArgs(args) {
  return args
    .map((arg) => {
      if (typeof arg === "string") return arg;
      try {
        return JSON.stringify(arg);
      } catch (_) {
        return String(arg);
      }
    })
    .join(" ");
}

function redirectConsoleToStderr() {
  const forward = (level, args) => {
    const text = formatConsoleArgs(args);
    process.stderr.write(`[bridge:${level}] ${text}\n`);
  };
  console.log = (...args) => forward("log", args);
  console.info = (...args) => forward("info", args);
  console.debug = (...args) => forward("debug", args);
  console.warn = (...args) => forward("warn", args);
  console.error = (...args) => forward("error", args);
}

function silenceConsole() {
  const noop = () => {};
  console.log = noop;
  console.info = noop;
  console.debug = noop;
  console.warn = noop;
  console.error = noop;
}

function configureConsoleForBridge() {
  const mode = String(process.env.PPO_BRIDGE_CONSOLE_MODE || "drop").trim().toLowerCase();
  if (mode === "stderr") {
    redirectConsoleToStderr();
    return;
  }
  silenceConsole();
}

function toPositiveInt(raw, name) {
  const n = Number(raw);
  if (!Number.isFinite(n) || !Number.isInteger(n) || n <= 0) {
    fail(`invalid ${name}: ${raw}`);
  }
  return n;
}

function toNonNegativeInt(raw, name) {
  const n = Number(raw);
  if (!Number.isFinite(n) || !Number.isInteger(n) || n < 0) {
    fail(`invalid ${name}: ${raw}`);
  }
  return n;
}

function toFiniteNumber(raw, name) {
  const n = Number(raw);
  if (!Number.isFinite(n)) {
    fail(`invalid ${name}: ${raw}`);
  }
  return n;
}

function otherActor(actor) {
  if (actor === "human") return "ai";
  if (actor === "ai") return "human";
  fail(`invalid actor: ${actor}`);
}

function normalizeActor(raw, name) {
  const v = String(raw || "").trim().toLowerCase();
  if (v !== "human" && v !== "ai") fail(`invalid ${name}: ${raw}`);
  return v;
}

function normalizeFirstTurnPolicy(raw) {
  const v = String(raw || "").trim().toLowerCase();
  if (v !== "alternate" && v !== "fixed") {
    fail(`invalid --first-turn-policy: ${raw} (allowed: alternate|fixed)`);
  }
  return v;
}

function normalizeTrainingMode(raw) {
  const v = String(raw || "").trim().toLowerCase();
  if (v !== "single_actor" && v !== "selfplay") {
    fail(`invalid --training-mode: ${raw} (allowed: single_actor|selfplay)`);
  }
  return v;
}

// Parses a strict one-format CLI and rejects unknown/missing keys.
function parseArgs(argv) {
  const args = [...argv];
  const out = {
    trainingMode: "",
    phase: null,
    seedBase: "",
    ruleKey: "",
    controlActor: "",
    opponentPolicy: "",
    maxSteps: null,
    rewardScale: null,
    downsidePenaltyScale: null,
    terminalBonusScale: null,
    terminalWinBonus: null,
    terminalLossPenalty: null,
    goActionBonus: null,
    firstTurnPolicy: "",
    fixedFirstTurn: ""
  };

  while (args.length > 0) {
    const raw = String(args.shift() || "");
    if (!raw.startsWith("--")) fail(`unknown argument: ${raw}`);

    const eq = raw.indexOf("=");
    let key = raw;
    let value = "";
    if (eq >= 0) {
      key = raw.slice(0, eq);
      value = raw.slice(eq + 1);
    } else {
      value = String(args.shift() || "");
    }

    if (key === "--training-mode") out.trainingMode = normalizeTrainingMode(value);
    else if (key === "--phase") out.phase = toNonNegativeInt(value, "--phase");
    else if (key === "--seed-base") out.seedBase = String(value || "").trim();
    else if (key === "--rule-key") out.ruleKey = String(value || "").trim();
    else if (key === "--control-actor") out.controlActor = normalizeActor(value, "--control-actor");
    else if (key === "--opponent-policy") out.opponentPolicy = String(value || "").trim();
    else if (key === "--max-steps") out.maxSteps = toPositiveInt(value, "--max-steps");
    else if (key === "--reward-scale") out.rewardScale = toFiniteNumber(value, "--reward-scale");
    else if (key === "--downside-penalty-scale") {
      out.downsidePenaltyScale = toFiniteNumber(value, "--downside-penalty-scale");
    } else if (key === "--terminal-bonus-scale") {
      out.terminalBonusScale = toFiniteNumber(value, "--terminal-bonus-scale");
    } else if (key === "--terminal-win-bonus") {
      out.terminalWinBonus = toFiniteNumber(value, "--terminal-win-bonus");
    } else if (key === "--terminal-loss-penalty") {
      out.terminalLossPenalty = toFiniteNumber(value, "--terminal-loss-penalty");
    } else if (key === "--go-action-bonus") {
      out.goActionBonus = toFiniteNumber(value, "--go-action-bonus");
    } else if (key === "--first-turn-policy") {
      out.firstTurnPolicy = normalizeFirstTurnPolicy(value);
    } else if (key === "--fixed-first-turn") {
      out.fixedFirstTurn = normalizeActor(value, "--fixed-first-turn");
    } else {
      fail(`unknown argument: ${key}`);
    }
  }

  if (!out.trainingMode) fail("--training-mode is required");
  if (out.phase == null) fail("--phase is required");
  if (!out.seedBase) fail("--seed-base is required");
  if (!out.ruleKey) fail("--rule-key is required");
  if (out.maxSteps == null) fail("--max-steps is required");
  if (out.rewardScale == null) fail("--reward-scale is required");
  if (Math.abs(out.rewardScale) <= 0) fail("--reward-scale must be non-zero");
  if (out.downsidePenaltyScale == null) fail("--downside-penalty-scale is required");
  if (out.downsidePenaltyScale < 0) fail("--downside-penalty-scale must be >= 0");
  if (out.terminalBonusScale == null) fail("--terminal-bonus-scale is required");
  if (!Number.isFinite(out.terminalBonusScale)) fail("--terminal-bonus-scale must be finite");
  if (out.terminalWinBonus == null) fail("--terminal-win-bonus is required");
  if (out.terminalWinBonus < 0) fail("--terminal-win-bonus must be >= 0");
  if (out.terminalLossPenalty == null) fail("--terminal-loss-penalty is required");
  if (out.terminalLossPenalty < 0) fail("--terminal-loss-penalty must be >= 0");
  if (out.goActionBonus == null) fail("--go-action-bonus is required");
  if (out.goActionBonus < 0) fail("--go-action-bonus must be >= 0");
  if (!out.firstTurnPolicy) fail("--first-turn-policy is required");
  if (out.firstTurnPolicy === "fixed" && !out.fixedFirstTurn) {
    fail("--fixed-first-turn is required when --first-turn-policy=fixed");
  }
  if (out.trainingMode === "single_actor") {
    if (!out.controlActor) fail("--control-actor is required when --training-mode=single_actor");
    if (!out.opponentPolicy) fail("--opponent-policy is required when --training-mode=single_actor");
  } else {
    if (out.controlActor) {
      fail("--control-actor is not allowed when --training-mode=selfplay");
    }
    if (out.opponentPolicy) {
      fail("--opponent-policy is not allowed when --training-mode=selfplay");
    }
  }

  return out;
}

// =============================================================================
// Section 2. Opponent Policy Sampling Helpers
// =============================================================================

function parseOpponentPolicyPool(raw, argName) {
  const text = String(raw || "").trim();
  if (!text) {
    fail(`${argName} must be non-empty`);
  }
  const parts = text.split("|").map((v) => String(v || "").trim()).filter((v) => v.length > 0);
  if (parts.length <= 0) {
    fail(`${argName} must contain at least one policy`);
  }
  const seen = new Set();
  const entries = [];
  let totalWeight = 0;
  for (const token of parts) {
    const sep = token.indexOf(":");
    if (sep <= 0 || sep >= token.length - 1) {
      fail(`${argName} token must follow POLICY:WEIGHT format, got=${token}`);
    }
    if (token.indexOf(":", sep + 1) >= 0) {
      fail(`${argName} token has multiple ':' separators, got=${token}`);
    }
    const policyRaw = token.slice(0, sep).trim();
    const weightRaw = token.slice(sep + 1).trim();
    const resolved = resolveBotPolicy(policyRaw);
    if (!resolved) {
      fail(`${argName} contains unsupported bot policy: ${policyRaw}`);
    }
    const policy = normalizeBotPolicy(resolved);
    if (seen.has(policy)) {
      fail(`${argName} contains duplicate policy: ${policy}`);
    }
    const weight = Number(weightRaw);
    if (!Number.isFinite(weight) || weight <= 0) {
      fail(`${argName} has invalid weight for policy ${policy}: ${weightRaw}`);
    }
    totalWeight += weight;
    seen.add(policy);
    entries.push({ policy, weight });
  }
  if (!Number.isFinite(totalWeight) || totalWeight <= 0) {
    fail(`${argName} total weight must be > 0`);
  }
  let cumulative = 0;
  const out = [];
  for (const entry of entries) {
    const prob = entry.weight / totalWeight;
    cumulative += prob;
    out.push({
      policy: entry.policy,
      weight: entry.weight,
      prob,
      cumulative
    });
  }
  out[out.length - 1].cumulative = 1.0;
  return out;
}

function sampleOpponentPolicy(runtime, seedText) {
  const pool = runtime.opponentPolicyPool;
  if (!Array.isArray(pool) || pool.length <= 0) {
    fail(`opponent policy pool is empty: seed=${seedText}`);
  }
  const pickRng = createSeededRng(`${seedText}|opp-policy`);
  const r = Number(pickRng());
  if (!Number.isFinite(r)) {
    fail(`opponent policy sampling rng returned non-finite: seed=${seedText}, value=${String(r)}`);
  }
  for (const entry of pool) {
    if (r < Number(entry.cumulative)) {
      return entry;
    }
  }
  return pool[pool.length - 1];
}

// =============================================================================
// Section 3. Observation / Action Context Builders
// =============================================================================

function clamp01(x) {
  const n = Number(x || 0);
  if (n <= 0) return 0;
  if (n >= 1) return 1;
  return n;
}

function tanhNorm(x, scale) {
  const s = Math.max(1e-6, Number(scale || 1));
  return Math.tanh(Number(x || 0) / s);
}

function findCardById(cards, cardId) {
  const target = String(cardId || "");
  if (!target || !Array.isArray(cards)) return null;
  for (const card of cards) {
    if (String(card?.id || "") === target) return card;
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

function decisionAvailabilityFlags(state, actor) {
  if (state?.phase === "shaking-confirm" && state?.pendingShakingConfirm?.playerKey === actor) {
    return { hasShaking: 1, hasBomb: 0 };
  }
  if (state?.phase !== "playing" || state?.currentTurn !== actor) {
    return { hasShaking: 0, hasBomb: 0 };
  }
  const shakingMonths = getDeclarableShakingMonths(state, actor);
  const bombMonths = getDeclarableBombMonths(state, actor);
  return {
    hasShaking: Array.isArray(shakingMonths) && shakingMonths.length > 0 ? 1 : 0,
    hasBomb: Array.isArray(bombMonths) && bombMonths.length > 0 ? 1 : 0
  };
}

function currentMultiplierNorm(state, scoreSelf) {
  const carry = Math.max(1.0, Number(state?.carryOverMultiplier || 1.0));
  const mul = Math.max(1.0, Number(scoreSelf?.multiplier || 1.0));
  return clamp01((mul * carry - 1.0) / 15.0);
}

function scoreGapToStopNorm(scoreTotal, goMinScore) {
  const threshold = Math.max(1, Number(goMinScore || 7));
  const gap = threshold - Number(scoreTotal || 0);
  return clamp01(gap / threshold);
}

function resolveObservationFocusMonth(state, controlActor, actionCtx) {
  const board = state?.board || [];
  if (actionCtx?.decisionType === "match") {
    for (const cardId of actionCtx.candidates || []) {
      const boardCard = findCardById(board, cardId);
      const month = Number(boardCard?.month || 0);
      if (month >= 1) return month;
    }
    return 0;
  }

  if (actionCtx?.decisionType === "option") {
    if (state?.phase === "shaking-confirm" && state?.pendingShakingConfirm?.playerKey === controlActor) {
      const month = Number(state?.pendingShakingConfirm?.month || 0);
      if (month >= 1) return month;
    }
    if (state?.phase === "president-choice" && state?.pendingPresident?.playerKey === controlActor) {
      const month = Number(state?.pendingPresident?.month || 0);
      if (month >= 1) return month;
    }
    return 0;
  }

  if (actionCtx?.decisionType === "play") {
    const hand = state?.players?.[controlActor]?.hand || [];
    let bestMonth = 0;
    let bestMatchCount = -1;
    for (const cardId of actionCtx.candidates || []) {
      const card = findCardById(hand, cardId);
      const month = Number(card?.month || 0);
      if (month <= 0) continue;
      const boardCount = countCardsByMonth(board, month);
      if (boardCount > bestMatchCount) {
        bestMatchCount = boardCount;
        bestMonth = month;
      }
    }
    return bestMonth;
  }

  return 0;
}

function immediateMatchPossible(state, decisionType, focusMonth) {
  if (decisionType === "match") return 1;
  if (focusMonth <= 0) return 0;
  return countCardsByMonth(state?.board || [], focusMonth) > 0 ? 1 : 0;
}

function matchOpportunityDensity(state, focusMonth) {
  return clamp01(countCardsByMonth(state?.board || [], focusMonth) / 3.0);
}

function monthTotalCards(month) {
  const m = Number(month || 0);
  if (m >= 1 && m <= 12) return 4;
  if (m === 13) return 2;
  return 0;
}

function uniqueMonthCounts(cards) {
  const counts = new Array(MONTH_BUCKETS).fill(0);
  if (!Array.isArray(cards)) return counts;
  const seen = new Set();
  for (const card of cards) {
    if (!card || typeof card !== "object") continue;
    const month = Number(card?.month || 0);
    if (!Number.isFinite(month) || month < 1 || month > MONTH_BUCKETS) continue;
    const id = String(card?.id || "");
    if (id) {
      if (seen.has(id)) continue;
      seen.add(id);
    }
    counts[month - 1] += 1;
  }
  return counts;
}

function appendMonthNormFromCards(out, cards, denom) {
  const d = Math.max(1, Number(denom || 4));
  const counts = uniqueMonthCounts(cards);
  for (let m = 0; m < MONTH_BUCKETS; m += 1) {
    out.push(clamp01(counts[m] / d));
  }
}

function collectCapturedCards(player) {
  const out = [];
  const pushAll = (cards) => {
    if (!Array.isArray(cards)) return;
    for (const card of cards) out.push(card);
  };
  const captured = player?.captured || {};
  pushAll(captured.kwang || []);
  pushAll(captured.five || []);
  pushAll(captured.ribbon || []);
  pushAll(captured.junk || []);
  return out;
}

function collectOppShakedMonths(state, oppActor) {
  const months = new Set();
  const addMonth = (rawMonth) => {
    const m = Number(rawMonth || 0);
    if (Number.isInteger(m) && m >= 1 && m <= MONTH_BUCKETS) months.add(m);
  };

  const declared = state?.players?.[oppActor]?.shakingDeclaredMonths;
  if (Array.isArray(declared)) {
    for (const month of declared) addMonth(month);
  }

  const reveal = state?.shakingReveal;
  if (String(reveal?.playerKey || "") === oppActor) {
    addMonth(reveal?.month);
  }

  const kibo = state?.kibo;
  if (Array.isArray(kibo)) {
    for (const evt of kibo) {
      if (String(evt?.type || "") !== "shaking_declare") continue;
      if (String(evt?.playerKey || "") !== oppActor) continue;
      addMonth(evt?.month);
    }
  }
  return months;
}

function appendOppShakedMonthFlags(out, state, oppActor) {
  const months = collectOppShakedMonths(state, oppActor);
  for (let month = 1; month <= MONTH_BUCKETS; month += 1) {
    out.push(months.has(month) ? 1 : 0);
  }
}

function multiplierRiskNorm(state, scoreSelf, scoreOpp, goMinScore) {
  const carry = Math.max(1.0, Number(state?.carryOverMultiplier || 1.0));
  const selfMul = Math.max(1.0, Number(scoreSelf?.multiplier || 1.0));
  const currentMul = selfMul * carry;
  const mulNorm = clamp01((currentMul - 1.0) / 12.0);
  const selfStopReady = 1.0 - scoreGapToStopNorm(Number(scoreSelf?.total || 0), goMinScore);
  const oppStopReady = 1.0 - scoreGapToStopNorm(Number(scoreOpp?.total || 0), goMinScore);
  const leadNorm = clamp01((tanhNorm(Number(scoreSelf?.total || 0) - Number(scoreOpp?.total || 0), 8.0) + 1.0) / 2.0);
  return clamp01(mulNorm * (0.5 + 0.3 * oppStopReady + 0.2 * (1.0 - leadNorm)) * (0.4 + 0.6 * selfStopReady));
}

function goExpectedValueNormPublic(state, controlActor, decisionType, focusMonth, scoreSelf, scoreOpp, goMinScore) {
  const selfScore = Number(scoreSelf?.total || 0);
  const oppScore = Number(scoreOpp?.total || 0);
  const lead = tanhNorm(selfScore - oppScore, 8.0);
  const selfStopReady = 1.0 - scoreGapToStopNorm(selfScore, goMinScore);
  const oppStopReady = 1.0 - scoreGapToStopNorm(oppScore, goMinScore);
  const mulNorm = currentMultiplierNorm(state, scoreSelf);
  const matchDensity = matchOpportunityDensity(state, focusMonth);
  const immediate = immediateMatchPossible(state, decisionType, focusMonth);
  const knownRatio = candidateMonthKnownRatio(state, controlActor, focusMonth);
  // goCount momentum: 이미 go를 더 많이 했을수록 계속 go가 합리적
  const selfGoCount = Number(state?.players?.[controlActor]?.goCount || 0);
  const goMomentum = Math.min(1.0, selfGoCount / 3.0);

  const raw =
    0.30 * lead +
    0.18 * (selfStopReady - oppStopReady) +
    0.17 * mulNorm +
    0.15 * goMomentum +
    0.10 * matchDensity +
    0.05 * immediate +
    0.05 * knownRatio;
  return clamp01((Math.tanh(raw) + 1.0) * 0.5);
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

  // Public by rule: self hand is observable to self.
  pushAll(state?.players?.[actor]?.hand || []);

  // Public by reveal: shaking revealed cards.
  pushAll(state?.shakingReveal?.cards || []);
  const kibo = state?.kibo || [];
  if (Array.isArray(kibo)) {
    for (const evt of kibo) {
      if (String(evt?.type || "") === "shaking_declare") {
        pushAll(evt?.revealCards || []);
      }
    }
  }
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

function cardFeatureVector(card) {
  if (!card || typeof card !== "object") {
    return [0, 0, 0, 0, 0, 0, 0];
  }
  const month = Number(card.month || 0);
  const category = String(card.category || "");
  const piValue = Number(card.piValue || 0);
  return [
    1,
    month >= 1 ? clamp01(month / 12.0) : 0,
    category === "kwang" ? 1 : 0,
    category === "five" ? 1 : 0,
    category === "ribbon" ? 1 : 0,
    category === "junk" ? 1 : 0,
    clamp01(piValue / 5.0)
  ];
}

function appendCardSlots(out, cards, maxSlots) {
  const source = Array.isArray(cards) ? cards : [];
  for (let i = 0; i < maxSlots; i += 1) {
    const c = i < source.length ? source[i] : null;
    const fv = cardFeatureVector(c);
    for (const v of fv) out.push(Number(v));
  }
}

function maskValidCount(mask) {
  let count = 0;
  for (const raw of mask) {
    const v = Number(raw);
    if (v !== 0 && v !== 1) {
      fail(`action mask value must be 0 or 1, got=${String(raw)}`);
    }
    if (v === 1) count += 1;
  }
  return count;
}

function validateStepView(runtime, view, context) {
  if (!view || typeof view !== "object") {
    fail(`invalid step view: ${String(context || "")}`);
  }
  if (!Array.isArray(view.obs)) {
    fail(`obs must be array: ${String(context || "")}`);
  }
  if (!Array.isArray(view.action_mask)) {
    fail(`action_mask must be array: ${String(context || "")}`);
  }
  if (view.obs.length !== OBS_DIM) {
    fail(
      `obs_dim mismatch: expected=${OBS_DIM}, got=${view.obs.length}, seed=${runtime.seedText}, step=${runtime.actionCount}, ctx=${String(context || "")}`
    );
  }
  if (view.action_mask.length !== ACTION_DIM) {
    fail(
      `action_dim mismatch: expected=${ACTION_DIM}, got=${view.action_mask.length}, seed=${runtime.seedText}, step=${runtime.actionCount}, ctx=${String(context || "")}`
    );
  }
  for (const raw of view.obs) {
    if (!Number.isFinite(Number(raw))) {
      fail(`obs has non-finite value: seed=${runtime.seedText}, step=${runtime.actionCount}, ctx=${String(context || "")}`);
    }
  }
  const validActions = maskValidCount(view.action_mask);
  const isTerminal = view.decision_type === "terminal";
  if (isTerminal) {
    if (validActions !== 0) {
      fail(
        `terminal view must have 0 legal actions: got=${validActions}, seed=${runtime.seedText}, step=${runtime.actionCount}, ctx=${String(context || "")}`
      );
    }
  } else if (validActions <= 0) {
    fail(`non-terminal view has empty action mask: seed=${runtime.seedText}, step=${runtime.actionCount}, ctx=${String(context || "")}`);
  }
  if (runtime.obsDim == null) {
    runtime.obsDim = view.obs.length;
  } else if (runtime.obsDim !== view.obs.length) {
    fail(
      `runtime obs_dim drift: expected=${runtime.obsDim}, got=${view.obs.length}, seed=${runtime.seedText}, step=${runtime.actionCount}, ctx=${String(context || "")}`
    );
  }
}

function resolveDecisionContext(state, controlActor) {
  const pool = selectDecisionPool(state, controlActor);
  const decisionType = resolveDecisionType(pool);
  if (!decisionType) {
    fail(`decision_type unresolved: actor=${controlActor}, phase=${String(state?.phase || "")}`);
  }
  const candidates = legalCandidatesForDecision(pool, decisionType);
  if (!Array.isArray(candidates) || candidates.length <= 0) {
    fail(
      `legal candidates empty: actor=${controlActor}, phase=${String(state?.phase || "")}, decisionType=${decisionType}`
    );
  }
  return { pool, decisionType, candidates };
}

function buildActionContext(state, controlActor) {
  const { pool, decisionType, candidates } = resolveDecisionContext(state, controlActor);
  const mask = new Array(ACTION_DIM).fill(0);
  const actionToCandidate = new Map();

  if (decisionType === "play") {
    if (candidates.length > PLAY_SLOTS) {
      fail(`play candidate overflow: ${candidates.length} > ${PLAY_SLOTS}`);
    }
    for (let i = 0; i < candidates.length; i += 1) {
      mask[i] = 1;
      actionToCandidate.set(i, String(candidates[i]));
    }
  } else if (decisionType === "match") {
    if (candidates.length > MATCH_SLOTS) {
      fail(`match candidate overflow: ${candidates.length} > ${MATCH_SLOTS}`);
    }
    for (let i = 0; i < candidates.length; i += 1) {
      const idx = PLAY_SLOTS + i;
      mask[idx] = 1;
      actionToCandidate.set(idx, String(candidates[i]));
    }
  } else if (decisionType === "option") {
    for (const raw of candidates) {
      const option = canonicalOptionAction(raw);
      const pos = OPTION_ORDER.indexOf(option);
      if (pos < 0) {
        fail(`unsupported option candidate: ${option}`);
      }
      const idx = PLAY_SLOTS + MATCH_SLOTS + pos;
      mask[idx] = 1;
      actionToCandidate.set(idx, option);
    }
  } else {
    fail(`unsupported decision type: ${decisionType}`);
  }

  const legalActions = [];
  for (let i = 0; i < mask.length; i += 1) {
    if (mask[i] === 1) legalActions.push(i);
  }
  if (legalActions.length <= 0) {
    fail(`action mask is empty: phase=${String(state?.phase || "")}, decisionType=${decisionType}`);
  }

  return { pool, decisionType, candidates, mask, actionToCandidate, legalActions };
}

function goldDiff(state, controlActor) {
  const opp = otherActor(controlActor);
  const selfGold = Number(state?.players?.[controlActor]?.gold || 0);
  const oppGold = Number(state?.players?.[opp]?.gold || 0);
  return selfGold - oppGold;
}

function uniqueCardCount(cards) {
  if (!Array.isArray(cards)) return 0;
  const seen = new Set();
  let count = 0;
  for (const card of cards) {
    const id = String(card?.id || "");
    if (!id || seen.has(id)) continue;
    seen.add(id);
    count += 1;
  }
  return count;
}

function eventDeltaCount(beforePlayer, afterPlayer, key) {
  const before = Number(beforePlayer?.events?.[key] || 0);
  const after = Number(afterPlayer?.events?.[key] || 0);
  const delta = after - before;
  return delta > 0 ? delta : 0;
}

function countNewDoublePiCards(beforePlayer, afterPlayer) {
  const beforeCards = Array.isArray(beforePlayer?.captured?.junk) ? beforePlayer.captured.junk : [];
  const afterCards = Array.isArray(afterPlayer?.captured?.junk) ? afterPlayer.captured.junk : [];
  const beforeIds = new Set();
  const seenBefore = new Set();
  for (const card of beforeCards) {
    const id = String(card?.id || "");
    if (!id || seenBefore.has(id)) continue;
    seenBefore.add(id);
    beforeIds.add(id);
  }
  const seenAfter = new Set();
  let count = 0;
  for (const card of afterCards) {
    const id = String(card?.id || "");
    if (!id || seenAfter.has(id)) continue;
    seenAfter.add(id);
    if (beforeIds.has(id)) continue;
    const piValue = Number(card?.piValue || 0);
    if (Number.isFinite(piValue) && piValue >= 2) count += 1;
  }
  return count;
}

function isPiBakRiskState(controlPlayer, oppPlayer) {
  const selfPi = Number(scoringPiCount(controlPlayer) || 0);
  const oppPi = Number(scoringPiCount(oppPlayer) || 0);
  return selfPi >= 1 && selfPi <= 7 && oppPi >= 10;
}

function isGwangBakRiskState(controlPlayer, oppPlayer) {
  const selfGwang = uniqueCardCount(controlPlayer?.captured?.kwang || []);
  const oppGwang = uniqueCardCount(oppPlayer?.captured?.kwang || []);
  return selfGwang <= 0 && oppGwang >= 3;
}

function phase1RewardShapingDelta({
  runtime,
  beforeState,
  afterState,
  controlActor,
  done,
  truncated,
  isGoStopDecision,
  selectedOption,
  afterDiff
}) {
  if (Number(runtime?.phase || 0) !== 1 || runtime?.trainingMode !== "single_actor") {
    return { total: 0.0, breakdown: {} };
  }

  const w = PHASE1_SHAPING;
  const beforePlayer = beforeState?.players?.[controlActor];
  const afterPlayer = afterState?.players?.[controlActor];
  const opp = otherActor(controlActor);
  const beforeOpp = beforeState?.players?.[opp];
  const afterOpp = afterState?.players?.[opp];
  if (!beforePlayer || !afterPlayer || !beforeOpp || !afterOpp) {
    fail(
      `phase1 shaping player resolve failed: seed=${runtime.seedText}, step=${runtime.actionCount}, actor=${controlActor}`
    );
  }

  const breakdown = {};
  let total = 0;
  const add = (key, value) => {
    const v = Number(value || 0);
    if (!Number.isFinite(v) || Math.abs(v) <= 0) return;
    breakdown[key] = Number((breakdown[key] || 0) + v);
    total += v;
  };

  const beforeGodori = countCapturedComboTag(beforePlayer, "five", "fiveBirds");
  const afterGodori = countCapturedComboTag(afterPlayer, "five", "fiveBirds");
  if (beforeGodori < 3 && afterGodori >= 3) add("godori_complete", w.godoriComplete);

  const beforeCheong = countCapturedComboTag(beforePlayer, "ribbon", "blueRibbons");
  const afterCheong = countCapturedComboTag(afterPlayer, "ribbon", "blueRibbons");
  if (beforeCheong < 3 && afterCheong >= 3) add("cheongdan_complete", w.cheongdanComplete);

  const beforeHong = countCapturedComboTag(beforePlayer, "ribbon", "redRibbons");
  const afterHong = countCapturedComboTag(afterPlayer, "ribbon", "redRibbons");
  if (beforeHong < 3 && afterHong >= 3) add("hongdan_complete", w.hongdanComplete);

  const beforeCho = countCapturedComboTag(beforePlayer, "ribbon", "plainRibbons");
  const afterCho = countCapturedComboTag(afterPlayer, "ribbon", "plainRibbons");
  if (beforeCho < 3 && afterCho >= 3) add("chodan_complete", w.chodanComplete);

  const beforeGwang = uniqueCardCount(beforePlayer?.captured?.kwang || []);
  const afterGwang = uniqueCardCount(afterPlayer?.captured?.kwang || []);
  if (beforeGwang < 3 && afterGwang >= 3) add("gwang3_complete", w.gwang3Complete);
  if (beforeGwang < 4 && afterGwang >= 4) add("gwang4_complete", w.gwang4Complete);
  if (beforeGwang < 5 && afterGwang >= 5) add("gwang5_complete", w.gwang5Complete);

  const eventWeights = {
    jjob: w.jjob,
    ppuk: w.ppuk,
    yeonPpuk: w.yeonPpuk,
    ddadak: w.ddadak,
    jabbeok: w.jabbeok,
    pansseul: w.pansseul,
    shaking: w.shaking,
    bomb: w.bomb
  };
  for (const [eventKey, weight] of Object.entries(eventWeights)) {
    const delta = eventDeltaCount(beforePlayer, afterPlayer, eventKey);
    if (delta > 0) add(eventKey, weight * delta);
  }

  if (isGoStopDecision && selectedOption === "go") {
    const beforeScoreSelf = Number(calculateScore(beforePlayer, beforeOpp, beforeState?.ruleKey).total || 0);
    const afterScoreSelf = Number(calculateScore(afterPlayer, afterOpp, afterState?.ruleKey).total || 0);
    if (afterScoreSelf > beforeScoreSelf) {
      add("go_score_up", w.goScoreUp);
    } else if (afterScoreSelf >= beforeScoreSelf - 1) {
      add("go_score_hold", w.goScoreHold);
    }
  }

  const beforePi = Number(scoringPiCount(beforePlayer) || 0);
  const afterPi = Number(scoringPiCount(afterPlayer) || 0);
  const newDoublePi = countNewDoublePiCards(beforePlayer, afterPlayer);
  if (newDoublePi > 0) add("double_pi_get", w.doublePiGet * newDoublePi);
  if (beforePi < 10 && afterPi >= 10) add("pi10_reached", w.pi10Reached);

  const beforePiRisk = isPiBakRiskState(beforePlayer, beforeOpp);
  const afterPiRisk = isPiBakRiskState(afterPlayer, afterOpp);
  if (!beforePiRisk && afterPiRisk) add("pibak_risk", w.piBakRisk);

  const beforeGwangRisk = isGwangBakRiskState(beforePlayer, beforeOpp);
  const afterGwangRisk = isGwangBakRiskState(afterPlayer, afterOpp);
  if (!beforeGwangRisk && afterGwangRisk) add("gwangbak_risk", w.gwangBakRisk);

  if (done && !truncated && runtime.controlGoDeclaredInEpisode) {
    if (afterDiff > 0) add("go_win", w.goWin);
    else if (afterDiff < 0) add("go_loss", w.goLoss);
  }

  if (done && !truncated && !runtime.controlGoDeclaredInEpisode && afterDiff > 0) {
    add("stop_win", w.stopWin);
  }

  if (done && !truncated && afterDiff < 0) {
    const winnerKey = String(afterState?.result?.winner || "");
    if (winnerKey === opp) {
      const winnerScore = afterState?.result?.[winnerKey] || {};
      const winnerBak = winnerScore?.bak || {};
      if (winnerBak.pi) add("pibak_suffered", w.piBakSuffered);
      if (winnerBak.gwang) add("gwangbak_suffered", w.gwangBakSuffered);
      if (winnerBak.mongBak) add("mongbak_suffered", w.mongBakSuffered);
    }
  }

  return { total, breakdown };
}

function buildObservation(state, controlActor, actionCtx) {
  const opp = otherActor(controlActor);
  const scoreSelf = calculateScore(state.players[controlActor], state.players[opp], state.ruleKey);
  const scoreOpp = calculateScore(state.players[opp], state.players[controlActor], state.ruleKey);
  const rules = ruleSets[state?.ruleKey];
  if (!rules) {
    fail(`rule set not found for observation: ruleKey=${String(state?.ruleKey || "")}`);
  }
  const goMinScore = Number(rules.goMinScore || 7);
  if (!Number.isFinite(goMinScore) || goMinScore <= 0) {
    fail(`invalid goMinScore in rule set: ruleKey=${String(state?.ruleKey || "")}, goMinScore=${String(rules.goMinScore)}`);
  }
  const phase = String(state?.phase || "");
  const focusMonth = resolveObservationFocusMonth(state, controlActor, actionCtx);
  const legalCountNorm = clamp01(Number(actionCtx?.legalActions?.length || 0) / 10.0);
  const selfCanStop = Number(scoreSelf?.total || 0) >= goMinScore ? 1 : 0;
  const oppCanStop = Number(scoreOpp?.total || 0) >= goMinScore ? 1 : 0;
  const { hasShaking, hasBomb } = decisionAvailabilityFlags(state, controlActor);
  const selfGodori = clamp01(countCapturedComboTag(state.players?.[controlActor], "five", "fiveBirds") / 3.0);
  const selfCheongdan = clamp01(countCapturedComboTag(state.players?.[controlActor], "ribbon", "blueRibbons") / 3.0);
  const selfHongdan = clamp01(countCapturedComboTag(state.players?.[controlActor], "ribbon", "redRibbons") / 3.0);
  const selfChodan = clamp01(countCapturedComboTag(state.players?.[controlActor], "ribbon", "plainRibbons") / 3.0);
  const oppGodori = clamp01(countCapturedComboTag(state.players?.[opp], "five", "fiveBirds") / 3.0);
  const oppCheongdan = clamp01(countCapturedComboTag(state.players?.[opp], "ribbon", "blueRibbons") / 3.0);
  const oppHongdan = clamp01(countCapturedComboTag(state.players?.[opp], "ribbon", "redRibbons") / 3.0);
  const oppChodan = clamp01(countCapturedComboTag(state.players?.[opp], "ribbon", "plainRibbons") / 3.0);
  const out = [];

  // 1) phase one-hot (6)
  out.push(phase === "playing" ? 1 : 0);
  out.push(phase === "select-match" ? 1 : 0);
  out.push(phase === "go-stop" ? 1 : 0);
  out.push(phase === "president-choice" ? 1 : 0);
  out.push(phase === "gukjin-choice" ? 1 : 0);
  out.push(phase === "shaking-confirm" ? 1 : 0);

  // 2) decision type one-hot (3)
  out.push(actionCtx.decisionType === "play" ? 1 : 0);
  out.push(actionCtx.decisionType === "match" ? 1 : 0);
  out.push(actionCtx.decisionType === "option" ? 1 : 0);

  // 3) macro counts/scores/public strategy block (103 total -> 112 cumulative)
  out.push(clamp01(Number(state?.deck?.length || 0) / 30.0));
  out.push(clamp01(Number(state?.players?.[controlActor]?.hand?.length || 0) / 10.0));
  out.push(clamp01(Number(state?.players?.[opp]?.hand?.length || 0) / 10.0));
  out.push(clamp01(Number(state?.board?.length || 0) / 24.0));
  out.push(clamp01(Number(state?.players?.[controlActor]?.goCount || 0) / 5.0));
  out.push(clamp01(Number(state?.players?.[opp]?.goCount || 0) / 5.0));
  out.push(tanhNorm(Number(scoreSelf?.total || 0), 10.0));
  out.push(tanhNorm(Number(scoreOpp?.total || 0), 10.0));
  out.push(tanhNorm(Number(scoreSelf?.total || 0) - Number(scoreOpp?.total || 0), 10.0));

  const selfGold = Number(state?.players?.[controlActor]?.gold || 0);
  const oppGold = Number(state?.players?.[opp]?.gold || 0);
  out.push(tanhNorm(selfGold, 50000.0));
  out.push(tanhNorm(oppGold, 50000.0));
  out.push(tanhNorm(selfGold - oppGold, 25000.0));

  const carryOver = Math.max(1.0, Number(state?.carryOverMultiplier || 1.0));
  out.push(clamp01((carryOver - 1.0) / 12.0));
  out.push(clamp01(Number(state?.turnSeq || 0) / 200.0));

  const selfCap = state?.players?.[controlActor]?.captured || {};
  const oppCap = state?.players?.[opp]?.captured || {};
  out.push(clamp01((selfCap.kwang || []).length / 5.0));
  out.push(clamp01((selfCap.five || []).length / 5.0));
  out.push(clamp01((selfCap.ribbon || []).length / 10.0));
  out.push(clamp01(Number(scoringPiCount(state.players[controlActor]) || 0) / 20.0));
  out.push(clamp01((oppCap.kwang || []).length / 5.0));
  out.push(clamp01((oppCap.five || []).length / 5.0));
  out.push(clamp01((oppCap.ribbon || []).length / 10.0));
  out.push(clamp01(Number(scoringPiCount(state.players[opp]) || 0) / 20.0));

  // 3-b) added high-signal public features (19)
  out.push(selfCanStop);
  out.push(oppCanStop);
  out.push(legalCountNorm);
  out.push(immediateMatchPossible(state, actionCtx.decisionType, focusMonth));
  out.push(matchOpportunityDensity(state, focusMonth));
  out.push(hasBomb);
  out.push(hasShaking);
  out.push(currentMultiplierNorm(state, scoreSelf));
  out.push(selfGodori);
  out.push(selfCheongdan);
  out.push(selfHongdan);
  out.push(selfChodan);
  out.push(oppGodori);
  out.push(oppCheongdan);
  out.push(oppHongdan);
  out.push(oppChodan);
  out.push(scoreGapToStopNorm(Number(scoreSelf?.total || 0), goMinScore));
  out.push(scoreGapToStopNorm(Number(scoreOpp?.total || 0), goMinScore));
  out.push(candidateMonthKnownRatio(state, controlActor, focusMonth));

  // 3-c) public month distributions (48)
  appendMonthNormFromCards(out, state?.board || [], 4.0); // board_month_1..12_norm
  appendMonthNormFromCards(out, collectCapturedCards(state?.players?.[controlActor]), 4.0); // self_captured_month_1..12
  appendMonthNormFromCards(out, collectCapturedCards(state?.players?.[opp]), 4.0); // opp_captured_month_1..12
  appendMonthNormFromCards(out, state?.players?.[controlActor]?.hand || [], 4.0); // self_hand_month_1..12

  // 3-d) opponent shaking-declared months (12), public-only.
  appendOppShakedMonthFlags(out, state, opp);

  // 3-e) public-only strategic auxiliaries (2)
  out.push(multiplierRiskNorm(state, scoreSelf, scoreOpp, goMinScore));
  out.push(goExpectedValueNormPublic(state, controlActor, actionCtx.decisionType, focusMonth, scoreSelf, scoreOpp, goMinScore));

  // 4) hand card slots (10 * 7 = 70)
  appendCardSlots(out, state?.players?.[controlActor]?.hand || [], PLAY_SLOTS);

  // 5) pending match slots (8 * 7 = 56)
  const board = state?.board || [];
  const pendingMatchCards = [];
  if (actionCtx.decisionType === "match") {
    for (const cardId of actionCtx.candidates) {
      pendingMatchCards.push(findCardById(board, cardId));
    }
  }
  appendCardSlots(out, pendingMatchCards, MATCH_SLOTS);

  // 6) option availability (8)
  for (let i = 0; i < OPTION_ORDER.length; i += 1) {
    const idx = PLAY_SLOTS + MATCH_SLOTS + i;
    out.push(actionCtx.mask[idx] === 1 ? 1 : 0);
  }

  for (const v of out) {
    if (!Number.isFinite(v)) {
      fail(`observation contains non-finite value: phase=${phase}`);
    }
  }
  if (out.length !== OBS_DIM) {
    fail(`observation dim mismatch: expected=${OBS_DIM}, got=${out.length}, phase=${phase}`);
  }
  return out;
}

function buildStepView(state, controlActor) {
  const actionCtx = buildActionContext(state, controlActor);
  return {
    obs: buildObservation(state, controlActor, actionCtx),
    action_mask: actionCtx.mask,
    decision_type: actionCtx.decisionType,
    legal_actions: actionCtx.legalActions
  };
}

// =============================================================================
// Section 4. Episode Lifecycle Helpers
// =============================================================================

function resolveFirstTurnKey(runtime) {
  if (runtime.firstTurnPolicy === "fixed") return runtime.fixedFirstTurn;
  return runtime.episodeIndex % 2 === 0 ? "human" : "ai";
}

function progressOrThrow(beforeState, nextState, context) {
  const before = stateProgressKey(beforeState, { includeKiboSeq: true });
  const after = stateProgressKey(nextState, { includeKiboSeq: true });
  if (!nextState || before === after) {
    fail(
      `action resolution failed: seed=${context.seed}, step=${context.step}, actor=${context.actor}, phase=${context.phase}, detail=${context.detail}`
    );
  }
}

function runOpponentUntilControl(runtime, seedText) {
  let guard = 0;
  while (runtime.state.phase !== "resolution") {
    const actor = getActionPlayerKey(runtime.state);
    if (!actor) break;
    if (actor === runtime.controlActor) break;
    const before = runtime.state;
    const next = aiPlay(runtime.state, actor, {
      source: "heuristic",
      heuristicPolicy: runtime.currentOpponentPolicy
    });
    progressOrThrow(before, next, {
      seed: seedText,
      step: runtime.actionCount,
      actor,
      phase: String(before?.phase || ""),
      detail: "opponent_auto"
    });
    runtime.state = next;
    runtime.actionCount += 1;
    guard += 1;
    if (runtime.actionCount > runtime.maxSteps) {
      fail(`episode max step overflow while auto-running opponent: seed=${seedText}, step=${runtime.actionCount}`);
    }
    if (guard > runtime.maxSteps) {
      fail(`opponent auto loop overflow: seed=${seedText}, step=${runtime.actionCount}`);
    }
  }
}

function initEpisode(runtime) {
  const firstTurnKey = resolveFirstTurnKey(runtime);
  const attemptBase = runtime.resetCount;
  const maxAttempts = 20;
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    const seedText = `${runtime.seedBase}|episode=${runtime.episodeIndex}|reset=${attemptBase}|attempt=${attempt}`;
    const rng = createSeededRng(seedText);
    runtime.state = initSimulationGame(runtime.ruleKey, rng, {
      kiboDetail: "lean",
      firstTurnKey
    });
    runtime.seedText = seedText;
    if (runtime.trainingMode === "single_actor") {
      const picked = sampleOpponentPolicy(runtime, seedText);
      runtime.currentOpponentPolicy = picked.policy;
      runtime.currentOpponentPolicyProb = Number(picked.prob);
    } else {
      runtime.currentOpponentPolicy = null;
      runtime.currentOpponentPolicyProb = 0;
    }
    runtime.actionCount = 0;
    runtime.controlGoDeclaredInEpisode = false;
    if (runtime.trainingMode === "single_actor") {
      runOpponentUntilControl(runtime, seedText);
    }
    if (runtime.state.phase !== "resolution") return;
  }
  fail(`failed to initialize non-terminal episode after ${maxAttempts} attempts (seedBase=${runtime.seedBase})`);
}

function buildRuntime(cli) {
  const opponentPolicyPool =
    cli.trainingMode === "single_actor"
      ? parseOpponentPolicyPool(cli.opponentPolicy, "--opponent-policy")
      : [];
  return {
    trainingMode: cli.trainingMode,
    phase: cli.phase,
    seedBase: cli.seedBase,
    ruleKey: cli.ruleKey,
    controlActor: cli.controlActor || null,
    opponentPolicyPool,
    currentOpponentPolicy: opponentPolicyPool.length > 0 ? opponentPolicyPool[0].policy : null,
    currentOpponentPolicyProb: opponentPolicyPool.length > 0 ? Number(opponentPolicyPool[0].prob) : 0,
    maxSteps: cli.maxSteps,
    rewardScale: cli.rewardScale,
    downsidePenaltyScale: cli.downsidePenaltyScale,
    terminalBonusScale: cli.terminalBonusScale,
    terminalWinBonus: cli.terminalWinBonus,
    terminalLossPenalty: cli.terminalLossPenalty,
    goActionBonus: cli.goActionBonus,
    firstTurnPolicy: cli.firstTurnPolicy,
    fixedFirstTurn: cli.fixedFirstTurn || "human",
    state: null,
    seedText: "",
    episodeIndex: 0,
    resetCount: 0,
    actionCount: 0,
    obsDim: null,
    controlGoDeclaredInEpisode: false
  };
}

function ensureControlTurn(runtime) {
  if (runtime.trainingMode !== "single_actor") return;
  if (!runtime.state || runtime.state.phase === "resolution") return;
  const actor = getActionPlayerKey(runtime.state);
  if (actor !== runtime.controlActor) {
    fail(
      `control actor turn mismatch: seed=${runtime.seedText}, step=${runtime.actionCount}, actor=${String(actor || "")}, expected=${runtime.controlActor}, phase=${String(runtime.state?.phase || "")}`
    );
  }
}

// =============================================================================
// Section 5. Command Handlers + Transport
// =============================================================================

function handleReset(runtime, payload) {
  const epRaw = payload && Object.prototype.hasOwnProperty.call(payload, "episode") ? payload.episode : null;
  if (epRaw != null) {
    runtime.episodeIndex = toPositiveInt(Number(epRaw) + 1, "episode") - 1;
  }
  runtime.resetCount += 1;
  initEpisode(runtime);
  if (runtime.trainingMode === "single_actor") {
    ensureControlTurn(runtime);
  }
  const actorToAct = runtime.trainingMode === "single_actor" ? runtime.controlActor : getActionPlayerKey(runtime.state);
  if (!actorToAct) {
    fail(`actor_to_act unresolved on reset: seed=${runtime.seedText}, phase=${String(runtime.state?.phase || "")}`);
  }
  const view = buildStepView(runtime.state, actorToAct);
  validateStepView(runtime, view, "reset");
  return {
    ok: true,
    type: "reset",
    obs: view.obs,
    action_mask: view.action_mask,
    info: {
      seed: runtime.seedText,
      episode: runtime.episodeIndex,
      training_mode: runtime.trainingMode,
      control_actor: runtime.controlActor,
      opponent_policy: runtime.currentOpponentPolicy,
      opponent_policy_prob: runtime.currentOpponentPolicyProb,
      actor_to_act: actorToAct,
      phase: runtime.state.phase,
      decision_type: view.decision_type,
      legal_actions: view.legal_actions,
      action_dim: ACTION_DIM,
      obs_dim: runtime.obsDim
    }
  };
}

function handleStep(runtime, payload) {
  if (!runtime.state) fail("step called before reset");
  if (runtime.trainingMode === "single_actor") {
    ensureControlTurn(runtime);
  }

  const actionRaw = payload ? payload.action : null;
  const actionIndex = toPositiveInt(Number(actionRaw) + 1, "action") - 1;
  if (actionIndex < 0 || actionIndex >= ACTION_DIM) {
    fail(`action out of range: ${actionIndex} (allowed: 0..${ACTION_DIM - 1})`);
  }

  const actingActor = runtime.trainingMode === "single_actor" ? runtime.controlActor : getActionPlayerKey(runtime.state);
  if (!actingActor) {
    fail(`acting actor unresolved: seed=${runtime.seedText}, step=${runtime.actionCount}, phase=${String(runtime.state?.phase || "")}`);
  }

  // Policy takes exactly one legal action, then bridge applies it and auto-plays opponent when needed.
  const beforeState = runtime.state;
  const beforeDiff = goldDiff(beforeState, actingActor);
  const actionCtx = buildActionContext(beforeState, actingActor);
  if (actionCtx.mask[actionIndex] !== 1) {
    fail(
      `illegal action index: seed=${runtime.seedText}, step=${runtime.actionCount}, actor=${actingActor}, phase=${beforeState.phase}, decisionType=${actionCtx.decisionType}, action=${actionIndex}`
    );
  }
  const candidate = actionCtx.actionToCandidate.get(actionIndex);
  if (candidate == null) {
    fail(`candidate resolution failed for action index: ${actionIndex}`);
  }

  const nextState = applyDecisionAction(beforeState, actingActor, actionCtx.decisionType, candidate);
  progressOrThrow(beforeState, nextState, {
    seed: runtime.seedText,
    step: runtime.actionCount,
    actor: actingActor,
    phase: String(beforeState?.phase || ""),
    detail: `policy_action:${actionCtx.decisionType}:${String(candidate)}`
  });
  runtime.state = nextState;
  runtime.actionCount += 1;

  if (runtime.trainingMode === "single_actor") {
    runOpponentUntilControl(runtime, runtime.seedText);
  }

  const truncated = runtime.state.phase !== "resolution" && runtime.actionCount >= runtime.maxSteps;
  const done = runtime.state.phase === "resolution" || truncated;

  const afterDiff = goldDiff(runtime.state, actingActor);
  const rewardGoldDelta = (afterDiff - beforeDiff) * runtime.rewardScale;
  let rewardTerminalBonus = 0;
  let rewardDownsidePenalty = 0;
  let rewardOutcomeWinBonus = 0;
  let rewardOutcomeLossPenalty = 0;
  let rewardGoActionBonus = 0;
  let rewardPhase1ShapingTotal = 0;
  let rewardPhase1ShapingBreakdown = {};
  const isGoStopDecision = actionCtx.decisionType === "option" && String(beforeState?.phase || "") === "go-stop";
  const selectedOption = actionCtx.decisionType === "option" ? String(candidate) : "";
  let reward = rewardGoldDelta;
  if (isGoStopDecision && selectedOption === "go") {
    runtime.controlGoDeclaredInEpisode = true;
    rewardGoActionBonus = runtime.goActionBonus;
    reward += rewardGoActionBonus;
  }
  if (done) {
    rewardTerminalBonus = afterDiff * runtime.terminalBonusScale;
    rewardDownsidePenalty = Math.max(0, -afterDiff) * runtime.downsidePenaltyScale;
    if (afterDiff > 0) rewardOutcomeWinBonus = runtime.terminalWinBonus;
    if (afterDiff < 0) rewardOutcomeLossPenalty = runtime.terminalLossPenalty;
    reward += rewardTerminalBonus;
    reward -= rewardDownsidePenalty;
    reward += rewardOutcomeWinBonus;
    reward -= rewardOutcomeLossPenalty;
  }

  const phase1Shaping = phase1RewardShapingDelta({
    runtime,
    beforeState,
    afterState: runtime.state,
    controlActor: actingActor,
    done,
    truncated,
    isGoStopDecision,
    selectedOption,
    afterDiff
  });
  rewardPhase1ShapingTotal = Number(phase1Shaping.total || 0);
  rewardPhase1ShapingBreakdown = phase1Shaping.breakdown || {};
  reward += rewardPhase1ShapingTotal;

  let view = null;
  if (!done) {
    if (runtime.trainingMode === "single_actor") {
      ensureControlTurn(runtime);
    }
    const actorToAct = runtime.trainingMode === "single_actor" ? runtime.controlActor : getActionPlayerKey(runtime.state);
    if (!actorToAct) {
      fail(`actor_to_act unresolved after step: seed=${runtime.seedText}, step=${runtime.actionCount}, phase=${String(runtime.state?.phase || "")}`);
    }
    view = buildStepView(runtime.state, actorToAct);
    validateStepView(runtime, view, "step");
  } else {
    if (runtime.obsDim == null || runtime.obsDim !== OBS_DIM) {
      fail(
        `terminal obs_dim unresolved: obsDim=${String(runtime.obsDim)}, expected=${OBS_DIM}, seed=${runtime.seedText}, step=${runtime.actionCount}`
      );
    }
    view = {
      obs: new Array(runtime.obsDim).fill(0),
      action_mask: new Array(ACTION_DIM).fill(0),
      decision_type: "terminal",
      legal_actions: []
    };
    validateStepView(runtime, view, "terminal");
  }

  return {
    ok: true,
    type: "step",
    obs: view.obs,
    action_mask: view.action_mask,
    reward,
    done,
    truncated,
    info: {
      seed: runtime.seedText,
      episode: runtime.episodeIndex,
      training_mode: runtime.trainingMode,
      opponent_policy: runtime.currentOpponentPolicy,
      opponent_policy_prob: runtime.currentOpponentPolicyProb,
      step: runtime.actionCount,
      acted_actor: actingActor,
      actor_to_act: done ? null : getActionPlayerKey(runtime.state),
      phase: runtime.state.phase,
      decision_type: view.decision_type,
      legal_actions: view.legal_actions,
      control_gold: Number(runtime.state?.players?.[actingActor]?.gold || 0),
      opponent_gold: Number(runtime.state?.players?.[otherActor(actingActor)]?.gold || 0),
      final_gold_diff: afterDiff,
      reward_gold_delta: rewardGoldDelta,
      reward_terminal_bonus: rewardTerminalBonus,
      reward_downside_penalty: rewardDownsidePenalty,
      reward_outcome_win_bonus: rewardOutcomeWinBonus,
      reward_outcome_loss_penalty: rewardOutcomeLossPenalty,
      reward_go_action_bonus: rewardGoActionBonus,
      reward_phase1_shaping_total: rewardPhase1ShapingTotal,
      reward_phase1_shaping_breakdown: rewardPhase1ShapingBreakdown,
      go_stop_decision: isGoStopDecision,
      go_option_selected: selectedOption
    }
  };
}

function handleClose() {
  return { ok: true, type: "close" };
}

function send(obj) {
  process.stdout.write(`${JSON.stringify(obj)}\n`);
}

// Bridge process bootstrap + stdin command dispatcher.
const cli = parseArgs(process.argv.slice(2));
configureConsoleForBridge();
const runtime = buildRuntime(cli);

process.on("uncaughtException", (err) => {
  const message = err && err.message ? String(err.message) : String(err);
  send({ ok: false, error: message });
  process.exit(1);
});

process.on("unhandledRejection", (err) => {
  const message = err && err.message ? String(err.message) : String(err);
  send({ ok: false, error: message });
  process.exit(1);
});

let didReset = false;
process.stdin.setEncoding("utf8");
let buffer = "";
process.stdin.on("data", (chunk) => {
  buffer += String(chunk || "");
  while (true) {
    const idx = buffer.indexOf("\n");
    if (idx < 0) break;
    const line = buffer.slice(0, idx).trim();
    buffer = buffer.slice(idx + 1);
    if (!line) continue;

    try {
      const msg = JSON.parse(line);
      const cmd = String(msg?.cmd || "").trim().toLowerCase();
      if (!cmd) fail("missing cmd");

      let response = null;
      if (cmd === "reset") {
        response = handleReset(runtime, msg);
        didReset = true;
      } else if (cmd === "step") {
        if (!didReset) fail("step received before reset");
        response = handleStep(runtime, msg);
      } else if (cmd === "close") {
        response = handleClose();
        send(response);
        process.exit(0);
      } else {
        fail(`unknown cmd: ${cmd}`);
      }

      send(response);
    } catch (err) {
      const message = err && err.message ? String(err.message) : String(err);
      send({ ok: false, error: message });
      process.exit(1);
    }
  }
});
