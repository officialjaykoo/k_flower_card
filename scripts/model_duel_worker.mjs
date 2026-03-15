import {
  initSimulationGame,
  startSimulationGame,
  createSeededRng,
  calculateScore,
  scoringPiCount,
  getDeclarableShakingMonths,
  getDeclarableBombMonths,
  declareShaking,
  declareBomb,
  playTurn,
  chooseMatch,
  chooseGo,
  chooseStop,
  chooseShakingYes,
  chooseShakingNo,
  choosePresidentStop,
  choosePresidentHold,
  chooseGukjinMode,
} from "../src/engine/index.js";
import { STARTING_GOLD } from "../src/engine/economy.js";
import { createWriteStream, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { getActionPlayerKey } from "../src/engine/runner.js";
import { resolveBotPolicy } from "../src/ai/policies.js";
import { resolvePlayerSpecCore } from "../src/ai/evalCore/playerSpecCore.js";
import { resolveResolvedPlayerAction } from "../src/ai/evalCore/resolvedPlayerAction.js";
import {
  resolveGoStopIqnRuntime,
  canonicalOptionAction,
  normalizeOptionCandidates,
  parsePlaySpecialCandidate,
  buildPlayingSpecialActions,
  applyPlayCandidate,
  selectPool,
  legalCandidatesForDecision,
  applyAction,
  stateProgressKey,
  normalizeDecisionCandidate,
  randomChoice,
  randomLegalAction,
  startRound,
  continueRound,
  clamp01,
  quantile,
} from "../src/ai/evalCore/sharedGameHelpers.js";

const GUKJIN_CARD_ID = "I0";
const NON_BRIGHT_KWANG_ID = "L0";
const TOTAL_SSANGPI_VALUE = 13;
const COMBO_THREAT_SPECS = Object.freeze({
  redRibbons: Object.freeze({ zone: "ribbon", tag: "redRibbons", months: Object.freeze([1, 2, 3]), reward: 3, category: "ribbon" }),
  blueRibbons: Object.freeze({ zone: "ribbon", tag: "blueRibbons", months: Object.freeze([6, 9, 10]), reward: 3, category: "ribbon" }),
  plainRibbons: Object.freeze({ zone: "ribbon", tag: "plainRibbons", months: Object.freeze([4, 5, 7]), reward: 3, category: "ribbon" }),
  fiveBirds: Object.freeze({ zone: "five", tag: "fiveBirds", months: Object.freeze([2, 4, 8]), reward: 5, category: "five" }),
  kwang: Object.freeze({ zone: "kwang", tag: null, months: Object.freeze([1, 3, 8, 11, 12]), reward: 0, category: "kwang" }),
});
const COMBO_THREAT_KEYS = Object.freeze(Object.keys(COMBO_THREAT_SPECS));
const LEGACY13_FEATURES = 13;
const HAND10_FEATURES = 10;
const MATERIAL10_STAGING_FEATURES = 10;
const POSITION11_FEATURES = 11;

// Pipeline Stage: 3/3 (neat_train.py -> neat_eval_worker.mjs -> model_duel_worker.mjs)
// Execution Flow Map:
// 1) main()
// 2) runDuelRound(): duel loop + decision callback
// 3) featureVectorForCandidate(): dataset feature construction
// 4) inferChosenCandidateFromTransition(): chosen-label reconstruction
//
// File Layout Map (top-down):
// 1) parseArgs()/spec/path helpers
// 2) engine action + feature helpers
// 3) chosen-candidate inference helpers
// 4) round simulation + summary helpers
// 5) main() entrypoint

// =============================================================================
// Section 1. CLI
// =============================================================================
function parseArgs(argv) {
  const args = [...argv];
  const out = {
    humanSpecRaw: "",
    aiSpecRaw: "",
    humanGoStopIqnPath: "",
    aiGoStopIqnPath: "",
    games: 1000,
    seed: "model-duel",
    maxSteps: 600,
    firstTurnPolicy: "alternate",
    fixedFirstTurn: "human",
    continuousSeries: true,
    stdoutFormat: "text",
    kiboDetail: "none",
    kiboOut: "",
    resultOut: "",
    datasetOut: "",
    datasetActor: "all",
    datasetDecisionTypesRaw: "all",
    datasetOptionCandidatesRaw: "all",
    datasetDecisionTypes: null,
    datasetOptionCandidates: null,
    featureProfile: "auto",
  };

  while (args.length > 0) {
    const raw = String(args.shift() || "");
    if (!raw.startsWith("--")) throw new Error(`Unknown argument: ${raw}`);
    const eq = raw.indexOf("=");
    let key = raw;
    let value = "";
    if (eq >= 0) {
      key = raw.slice(0, eq);
      value = raw.slice(eq + 1);
    } else {
      value = String(args.shift() || "");
    }

    if (key === "--human") out.humanSpecRaw = String(value || "").trim();
    else if (key === "--ai") out.aiSpecRaw = String(value || "").trim();
    else if (key === "--human-go-stop-iqn" || key === "--human-go-stop-iqn-model") {
      out.humanGoStopIqnPath = String(value || "").trim();
    }
    else if (key === "--ai-go-stop-iqn" || key === "--ai-go-stop-iqn-model") {
      out.aiGoStopIqnPath = String(value || "").trim();
    }
    else if (key === "--policy-a" || key === "--policy-b") {
      throw new Error(`deprecated option: ${key} (use --human and --ai)`);
    }
    else if (key === "--games") out.games = Math.max(1, Number(value || 1000));
    else if (key === "--seed") out.seed = String(value || "model-duel");
    else if (key === "--max-steps") out.maxSteps = Math.max(20, Number(value || 600));
    else if (key === "--first-turn-policy") out.firstTurnPolicy = String(value || "alternate").trim().toLowerCase();
    else if (key === "--fixed-first-turn") out.fixedFirstTurn = String(value || "human").trim().toLowerCase();
    else if (key === "--continuous-series") out.continuousSeries = parseContinuousSeriesValue(value);
    else if (key === "--stdout-format") out.stdoutFormat = String(value || "text").trim().toLowerCase();
    else if (key === "--kibo-detail") out.kiboDetail = String(value || "none").trim().toLowerCase();
    else if (key === "--kibo-out") out.kiboOut = String(value || "").trim();
    else if (key === "--result-out") out.resultOut = String(value || "").trim();
    else if (key === "--dataset-out") out.datasetOut = String(value || "").trim();
    else if (key === "--dataset-actor") out.datasetActor = String(value || "all").trim().toLowerCase();
    else if (key === "--feature-profile") out.featureProfile = String(value || "auto").trim().toLowerCase();
    else if (key === "--dataset-decision-types") {
      out.datasetDecisionTypesRaw = String(value || "all").trim().toLowerCase();
    }
    else if (key === "--dataset-option-candidates" || key === "--dataset-option-actions") {
      out.datasetOptionCandidatesRaw = String(value || "all").trim().toLowerCase();
    }
    else throw new Error(`Unknown argument: ${key}`);
  }

  if (!out.humanSpecRaw) throw new Error("--human is required");
  if (!out.aiSpecRaw) throw new Error("--ai is required");
  out.games = Math.floor(out.games);

  if (out.firstTurnPolicy !== "alternate" && out.firstTurnPolicy !== "fixed") {
    throw new Error(`invalid --first-turn-policy: ${out.firstTurnPolicy}`);
  }
  if (out.fixedFirstTurn !== "human") {
    throw new Error("--fixed-first-turn is locked to human");
  }
  if (out.kiboDetail !== "none" && out.kiboDetail !== "lean" && out.kiboDetail !== "full") {
    throw new Error(`invalid --kibo-detail: ${out.kiboDetail} (allowed: none, lean, full)`);
  }
  if (out.datasetActor !== "all" && out.datasetActor !== "human" && out.datasetActor !== "ai") {
    throw new Error(`invalid --dataset-actor: ${out.datasetActor} (allowed: all, human, ai)`);
  }
  if (
    out.featureProfile !== "auto" &&
    out.featureProfile !== "hand10" &&
    out.featureProfile !== "material10" &&
    out.featureProfile !== "position11"
  ) {
    throw new Error(`invalid --feature-profile: ${out.featureProfile} (allowed: auto, hand10, material10, position11)`);
  }
  out.datasetDecisionTypes = parseDatasetDecisionTypes(out.datasetDecisionTypesRaw);
  out.datasetOptionCandidates = parseDatasetOptionCandidates(out.datasetOptionCandidatesRaw);
  if (out.datasetOptionCandidates && !out.datasetDecisionTypes) {
    out.datasetDecisionTypes = new Set(["option"]);
  }
  if (out.datasetOptionCandidates && out.datasetDecisionTypes && !out.datasetDecisionTypes.has("option")) {
    throw new Error(
      "--dataset-option-candidates requires --dataset-decision-types to include option (or all)"
    );
  }
  if (out.stdoutFormat !== "text" && out.stdoutFormat !== "json") {
    throw new Error(`invalid --stdout-format: ${out.stdoutFormat} (allowed: text, json)`);
  }

  return out;
}

const DATASET_DECISION_TYPES = new Set(["play", "match", "option"]);
const DATASET_OPTION_CANDIDATES = new Set([
  "go",
  "stop",
  "shaking_yes",
  "shaking_no",
  "president_stop",
  "president_hold",
  "five",
  "junk",
]);

function parseCsvTokens(raw) {
  const text = String(raw || "").trim().toLowerCase();
  if (!text || text === "all") return [];
  return text
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

function parseDatasetDecisionTypes(raw) {
  const items = parseCsvTokens(raw);
  if (items.length <= 0) return null;
  const out = new Set();
  for (const item of items) {
    if (!DATASET_DECISION_TYPES.has(item)) {
      throw new Error(
        `invalid --dataset-decision-types token: ${item} (allowed: all, play, match, option)`
      );
    }
    out.add(item);
  }
  return out;
}

function parseDatasetOptionCandidates(raw) {
  const text = String(raw || "").trim().toLowerCase();
  if (!text || text === "all") return null;
  const normalizedRaw =
    text === "go-stop" || text === "go_stop" || text === "gostop" ? "go,stop" : text;
  const items = parseCsvTokens(normalizedRaw);
  if (items.length <= 0) return null;
  const out = new Set();
  for (const item of items) {
    const canonical = canonicalOptionAction(item);
    if (!DATASET_OPTION_CANDIDATES.has(canonical)) {
      throw new Error(
        `invalid --dataset-option-candidates token: ${item} (allowed: all, go, stop, shaking_yes, shaking_no, president_stop, president_hold, five, junk; alias: go-stop)`
      );
    }
    out.add(canonical);
  }
  return out;
}

function parseContinuousSeriesValue(value) {
  const raw = String(value ?? "1").trim();
  if (raw === "" || raw === "1") return true;
  if (raw === "2") return false;
  throw new Error(`invalid --continuous-series: ${raw} (allowed: 1=true, 2=false)`);
}

function sanitizeFilePart(text) {
  return String(text || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function dateTag() {
  const now = new Date();
  const yyyy = String(now.getFullYear());
  const mm = String(now.getMonth() + 1).padStart(2, "0");
  const dd = String(now.getDate()).padStart(2, "0");
  return `${yyyy}${mm}${dd}`;
}

function resolvePlayerSpec(rawSpec, sideLabel) {
  return resolvePlayerSpecCore(rawSpec, {
    label: `${sideLabel} spec`,
    resolveHeuristic: (token) => resolveBotPolicy(token),
    resolveModel: (token, modelLabel) => resolvePhaseModelSpec(token, modelLabel),
  });
}

function parsePhaseModelToken(rawToken) {
  const m = String(rawToken || "")
    .trim()
    .match(/^phase([0-3])_seed(\d+)(?::(winner_genome))?$/i);
  if (!m) return null;
  const phase = Number(m[1]);
  const seed = Number(m[2]);
  const modelName = String(m[3] || "winner_genome").trim().toLowerCase();
  const outputPrefix = "neat";
  const baseTokenKey = `phase${phase}_seed${seed}`;
  const tokenKey = modelName === "winner_genome" ? baseTokenKey : `${baseTokenKey}:${modelName}`;
  return { phase, seed, outputPrefix, tokenKey, modelName };
}

function resolvePhaseModelSpec(token, sideLabel) {
  const parsedToken = parsePhaseModelToken(token);
  if (!parsedToken) {
    throw new Error(
      `invalid ${sideLabel} spec: ${token} (use policy key, phase0_seed9, or hybrid_play(phase1_seed202,H-CL))`
    );
  }
  const { phase, seed, outputPrefix, tokenKey, modelName } = parsedToken;
  const summaryPath = resolve(`logs/NEAT/${outputPrefix}_phase${phase}_seed${seed}/run_summary.json`);
  if (modelName === "winner_genome" && existsSync(summaryPath)) {
    try {
      const summaryRaw = String(readFileSync(summaryPath, "utf8") || "").replace(/^\uFEFF/, "");
      const summary = JSON.parse(summaryRaw);
      if (String(summary?.winner_repair_status || "") === "summary_repaired_winner_unrecoverable") {
        throw new Error(
          `unrecoverable winner for ${token}: exact best genome was not restorable from saved artifacts`
        );
      }
    } catch (err) {
      if (String(err?.message || err).includes("unrecoverable winner")) {
        throw err;
      }
    }
  }
  const modelPath = resolve(`logs/NEAT/${outputPrefix}_phase${phase}_seed${seed}/models/${modelName}.json`);
  if (!existsSync(modelPath)) {
    throw new Error(`model not found for ${token}: ${modelPath}`);
  }

  let model = null;
  try {
    const raw = String(readFileSync(modelPath, "utf8") || "").replace(/^\uFEFF/, "");
    model = JSON.parse(raw);
  } catch (err) {
    throw new Error(`failed to parse model JSON (${token}): ${modelPath} (${String(err)})`);
  }
  if (String(model?.format_version || "").trim() !== "neat_python_genome_v1") {
    throw new Error(`invalid model format for ${token}: expected neat_python_genome_v1`);
  }

  return {
    input: token,
    kind: "model",
    key: tokenKey,
    label: tokenKey,
    model,
    modelPath,
    phase,
    seed,
  };
}

function modelFeatureProfile(spec) {
  const profile = String(spec?.model?.feature_spec?.profile || "").trim().toLowerCase();
  if (profile === "hand10" || profile === "material10" || profile === "position11") return profile;
  const inputDim = Array.isArray(spec?.model?.input_keys) ? spec.model.input_keys.length : 0;
  if (inputDim === LEGACY13_FEATURES) return "legacy13";
  if (inputDim === POSITION11_FEATURES) return "position11";
  return "";
}

function resolveDatasetFeatureProfile(featureProfileOpt, humanPlayer, aiPlayer) {
  const explicit = String(featureProfileOpt || "auto").trim().toLowerCase();
  if (explicit === "hand10" || explicit === "material10" || explicit === "position11") return explicit;

  const profiles = new Set();
  for (const spec of [humanPlayer, aiPlayer]) {
    const profile = modelFeatureProfile(spec);
    if (profile) profiles.add(profile);
  }
  if (profiles.size > 1) {
    throw new Error(
      `conflicting model feature profiles for dataset auto mode: ${Array.from(profiles).join(", ")}`
    );
  }
  if (profiles.size === 1) {
    return Array.from(profiles)[0];
  }
  return "hand10";
}

function attachGoStopIqnRuntime(playerSpec, runtimePath, sideLabel) {
  const rawPath = String(runtimePath || "").trim();
  if (!rawPath) return playerSpec;
  const loaded = resolveGoStopIqnRuntime(rawPath, sideLabel);
  return {
    ...playerSpec,
    goStopIqnModel: loaded.model,
    goStopIqnPath: loaded.modelPath,
  };
}

function buildAutoOutputDir(humanLabel, aiLabel) {
  const duelKey = `${sanitizeFilePart(humanLabel)}_vs_${sanitizeFilePart(aiLabel)}_${dateTag()}`;
  const outDir = join("logs", "model_duel", duelKey);
  mkdirSync(outDir, { recursive: true });
  return outDir;
}

function buildAutoArtifactPath(outDir, seed, suffix) {
  const stem = sanitizeFilePart(seed) || "model-duel";
  return join(outDir, `${stem}_${suffix}`);
}

function toReportPath(pathValue) {
  const raw = String(pathValue || "").trim();
  if (!raw) return null;
  const rel = relative(process.cwd(), resolve(raw));
  const normalized = String(rel || raw).replace(/\\/g, "/");
  return normalized || null;
}

function setToReportList(setLike) {
  if (!setLike || !(setLike instanceof Set) || setLike.size <= 0) return "all";
  return [...setLike].sort();
}

// =============================================================================
// Section 2. Engine Action Helpers + Feature Helpers
// =============================================================================
let ACTIVE_FEATURE_PROFILE = "hand10";
let ACTIVE_COMPACT_FEATURES = HAND10_FEATURES;

function isGoStopOpportunityDecision(decision) {
  if (!decision || decision.decisionType !== "option") return false;
  if (String(decision?.stateBefore?.phase || "") !== "go-stop") return false;
  const options = normalizeOptionCandidates(decision.candidates || []);
  return options.includes("go") && options.includes("stop");
}

function isPresidentOpportunityDecision(decision) {
  if (!decision || decision.decisionType !== "option") return false;
  if (String(decision?.stateBefore?.phase || "") !== "president-choice") return false;
  const options = normalizeOptionCandidates(decision.candidates || []);
  return options.includes("president_stop") && options.includes("president_hold");
}

function isGukjinOpportunityDecision(decision) {
  if (!decision || decision.decisionType !== "option") return false;
  if (String(decision?.stateBefore?.phase || "") !== "gukjin-choice") return false;
  const options = normalizeOptionCandidates(decision.candidates || []);
  return options.includes("five") && options.includes("junk");
}

function maskStateForVisibleComboSimulation(state) {
  if (!state || typeof state !== "object") return state;
  const pendingMatch = state?.pendingMatch
    ? {
        ...state.pendingMatch,
        context: state.pendingMatch?.context
          ? {
              ...state.pendingMatch.context,
              deck: [],
            }
          : state.pendingMatch.context,
      }
    : state?.pendingMatch ?? null;
  return {
    ...state,
    deck: [],
    pendingMatch,
  };
}

function tanhNorm(x, scale) {
  const s = Math.max(1e-6, Number(scale || 1));
  return Math.tanh(Number(x || 0) / s);
}

function clampRange(x, minValue, maxValue) {
  const v = Number(x || 0);
  if (v <= minValue) return minValue;
  if (v >= maxValue) return maxValue;
  return v;
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
    junk: 8,
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

function candidateComboGain(state, actor, decisionType, candidate) {
  const beforePlayer = state?.players?.[actor];
  if (!beforePlayer) return 0;

  const visibleState = maskStateForVisibleComboSimulation(state);
  const afterState = applyAction(visibleState, actor, decisionType, candidate);
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
  const afterState = applyAction(visibleState, actor, decisionType, candidate);
  if (!afterState) return 0;
  const after = comboThreatBreakdown(afterState, actor);
  let delta = 0;
  for (const comboKey of COMBO_THREAT_KEYS) {
    delta += Number(before[comboKey] || 0) - Number(after[comboKey] || 0);
  }
  return clampRange(delta, -1.0, 1.0);
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
    hasBomb: Array.isArray(bombMonths) && bombMonths.length > 0 ? 1 : 0,
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

function buildLegacy13FeatureVectorForCandidate(state, actor, decisionType, candidate) {
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
    Number(scoreOpp?.total || 0) >= 7 ? 1 : 0,
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
  if (countCardsByMonth(state?.board || [], month) >= 3) return true;
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
    if (countCardsByMonth(state?.board || [], Number(card?.month || 0)) > 0) score += weight;
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

function buildPosition11FeatureVectorForCandidate(state, actor, decisionType, candidate) {
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

function buildHand10FeatureVectorForCandidate(state, actor, decisionType, candidate) {
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
    globalContextTrigger(postState, actor),
  ];
}

function buildMaterial10StagingFeatureVectorForCandidate(state, actor, decisionType, candidate) {
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
    tanhNorm((scoreSelf?.total || 0) - (scoreOpp?.total || 0), 10.0),
  ];
}

function featureVectorForCandidate(state, actor, decisionType, candidate) {
  const features =
    ACTIVE_FEATURE_PROFILE === "legacy13"
      ? buildLegacy13FeatureVectorForCandidate(state, actor, decisionType, candidate)
      : ACTIVE_FEATURE_PROFILE === "material10"
        ? buildMaterial10StagingFeatureVectorForCandidate(state, actor, decisionType, candidate)
        : ACTIVE_FEATURE_PROFILE === "position11"
          ? buildPosition11FeatureVectorForCandidate(state, actor, decisionType, candidate)
          : buildHand10FeatureVectorForCandidate(state, actor, decisionType, candidate);
  if (features.length !== ACTIVE_COMPACT_FEATURES) {
    throw new Error(`compact feature length mismatch: expected ${ACTIVE_COMPACT_FEATURES}, got ${features.length}`);
  }
  return features;
}

function summarizeCapturedPublic(player) {
  return {
    kwang_count: Number(player?.captured?.kwang?.length || 0),
    five_count: Number(player?.captured?.five?.length || 0),
    ribbon_count: Number(player?.captured?.ribbon?.length || 0),
    junk_count: Number(player?.captured?.junk?.length || 0),
    pi_count: Number(scoringPiCount(player) || 0),
    godori_count: countCapturedComboTag(player, "five", "fiveBirds"),
    cheongdan_count: countCapturedComboTag(player, "ribbon", "blueRibbons"),
    hongdan_count: countCapturedComboTag(player, "ribbon", "redRibbons"),
    chodan_count: countCapturedComboTag(player, "ribbon", "plainRibbons"),
    go_count: Number(player?.goCount || 0),
    gukjin_mode: String(player?.gukjinMode || "five"),
  };
}

function buildGoStopPublicSnapshot(state, actor, legalCandidates) {
  const opp = actor === "human" ? "ai" : "human";
  const scoreSelf = calculateScore(state.players[actor], state.players[opp], state.ruleKey);
  const scoreOpp = calculateScore(state.players[opp], state.players[actor], state.ruleKey);
  return {
    snapshot_version: 1,
    phase: "go-stop",
    actor,
    opponent: opp,
    current_turn: String(state?.currentTurn || ""),
    pending_go_stop: String(state?.pendingGoStop || ""),
    turn_seq: Number(state?.turnSeq || 0),
    kibo_seq: Number(state?.kiboSeq || 0),
    deck_count: Number(state?.deck?.length || 0),
    board_count: Number(state?.board?.length || 0),
    hand_count_self: Number(state?.players?.[actor]?.hand?.length || 0),
    hand_count_opp: Number(state?.players?.[opp]?.hand?.length || 0),
    initial_gold_base: Number(resolveInitialGoldBase(state) || STARTING_GOLD),
    self_gold: Number(state?.players?.[actor]?.gold || 0),
    opp_gold: Number(state?.players?.[opp]?.gold || 0),
    carry_over_multiplier: Number(state?.carryOverMultiplier || 1),
    self_score_total: Number(scoreSelf?.total || 0),
    opp_score_total: Number(scoreOpp?.total || 0),
    self_multiplier: Number(scoreSelf?.multiplier || 1),
    opp_multiplier: Number(scoreOpp?.multiplier || 1),
    self_can_stop: Number(scoreSelf?.total || 0) >= 7 ? 1 : 0,
    opp_can_stop: Number(scoreOpp?.total || 0) >= 7 ? 1 : 0,
    self_bak_pi: scoreSelf?.bak?.pi ? 1 : 0,
    self_bak_gwang: scoreSelf?.bak?.gwang ? 1 : 0,
    self_bak_mongbak: scoreSelf?.bak?.mongBak ? 1 : 0,
    opp_bak_pi: scoreOpp?.bak?.pi ? 1 : 0,
    opp_bak_gwang: scoreOpp?.bak?.gwang ? 1 : 0,
    opp_bak_mongbak: scoreOpp?.bak?.mongBak ? 1 : 0,
    self_captured: summarizeCapturedPublic(state?.players?.[actor]),
    opp_captured: summarizeCapturedPublic(state?.players?.[opp]),
    legal_candidates: Array.isArray(legalCandidates) ? legalCandidates.slice() : [],
  };
}

function buildDecisionId(gameIndex, decision) {
  return [
    String(gameIndex),
    String(decision?.step || 0),
    String(decision?.actor || ""),
    String(decision?.decisionType || ""),
    String(decision?.stateBefore?.phase || ""),
  ].join(":");
}

// =============================================================================
// Section 3. Chosen Candidate Inference (for dataset labels)
// =============================================================================
function inferCandidateFromKiboTransition(stateBefore, actor, decisionType, candidates, stateAfter) {
  const beforeLen = Array.isArray(stateBefore?.kibo) ? stateBefore.kibo.length : 0;
  const afterKibo = Array.isArray(stateAfter?.kibo) ? stateAfter.kibo : [];
  if (afterKibo.length <= beforeLen) return null;

  const candidateSet = new Set((candidates || []).map((c) => normalizeDecisionCandidate(decisionType, c)));
  const shakingDeclared =
    decisionType === "play" &&
    afterKibo.slice(beforeLen).some(
      (ev) =>
        String(ev?.type || "").trim().toLowerCase() === "shaking_declare" &&
        String(ev?.playerKey || "") === String(actor || "")
    );
  for (let i = afterKibo.length - 1; i >= beforeLen; i -= 1) {
    const ev = afterKibo[i];
    const evType = String(ev?.type || "").trim().toLowerCase();

    if (decisionType === "option" && String(ev?.playerKey || "") === String(actor || "")) {
      if (evType === "gukjin_mode") {
        const mode = String(ev?.mode || "").trim().toLowerCase();
        if ((mode === "five" || mode === "junk") && candidateSet.has(mode)) return mode;
      }
      if (evType === "president_stop" && candidateSet.has("president_stop")) return "president_stop";
      if (evType === "president_hold" && candidateSet.has("president_hold")) return "president_hold";
    }

    if (evType !== "turn_end") continue;
    if (String(ev?.actor || "") !== String(actor || "")) continue;
    const actionType = String(ev?.action?.type || "").trim().toLowerCase();

    if (decisionType === "play" && actionType === "play") {
      const playedId = String(ev?.action?.card?.id || "").trim();
      const shakingCandidate = `shake_start:${playedId}`;
      if (playedId && shakingDeclared && candidateSet.has(shakingCandidate)) return shakingCandidate;
      if (playedId && candidateSet.has(playedId)) return playedId;
    }
    if (decisionType === "play" && actionType === "declare_bomb") {
      const bombMonth = Number(ev?.action?.month || 0);
      const bombCandidate = `bomb:${bombMonth}`;
      if (bombMonth >= 1 && candidateSet.has(bombCandidate)) return bombCandidate;
    }
    if (decisionType === "match" && actionType === "play") {
      const selectedBoardId = String(ev?.action?.selectedBoardCard?.id || "").trim();
      if (selectedBoardId && candidateSet.has(selectedBoardId)) return selectedBoardId;
    }
    break;
  }
  return null;
}

function inferPlayCandidateFromHandDiff(stateBefore, actor, candidates, stateAfter) {
  const beforeHand = Array.isArray(stateBefore?.players?.[actor]?.hand) ? stateBefore.players[actor].hand : [];
  const afterHand = Array.isArray(stateAfter?.players?.[actor]?.hand) ? stateAfter.players[actor].hand : [];
  const beforeIds = new Set(beforeHand.map((c) => String(c?.id || "")).filter((id) => id.length > 0));
  const afterIds = new Set(afterHand.map((c) => String(c?.id || "")).filter((id) => id.length > 0));
  const removed = [];
  for (const candidate of candidates || []) {
    const id = String(candidate || "").trim();
    if (!id) continue;
    if (beforeIds.has(id) && !afterIds.has(id)) removed.push(id);
  }
  if (removed.length === 1) return removed[0];
  return null;
}

function inferChosenCandidateFromTransition(stateBefore, actor, decisionType, candidates, stateAfter) {
  if (!stateAfter || !Array.isArray(candidates) || !candidates.length) return null;

  // Prefer explicit action traces over simulation replay when available.
  const kiboInferred = inferCandidateFromKiboTransition(
    stateBefore,
    actor,
    decisionType,
    candidates,
    stateAfter
  );
  if (kiboInferred) return normalizeDecisionCandidate(decisionType, kiboInferred);

  // For play decisions, the selected card should disappear from actor hand.
  if (decisionType === "play") {
    const handDiffInferred = inferPlayCandidateFromHandDiff(stateBefore, actor, candidates, stateAfter);
    if (handDiffInferred) return normalizeDecisionCandidate(decisionType, handDiffInferred);
  }

  const target = stateProgressKey(stateAfter);
  for (const candidate of candidates) {
    const simulated = applyAction(stateBefore, actor, decisionType, candidate);
    if (simulated && stateProgressKey(simulated) === target) {
      return normalizeDecisionCandidate(decisionType, candidate);
    }
  }
  return null;
}

function collectTransitionEventCounts(stateBefore, stateAfter) {
  const beforeLen = Array.isArray(stateBefore?.kibo) ? stateBefore.kibo.length : 0;
  const afterKibo = Array.isArray(stateAfter?.kibo) ? stateAfter.kibo : [];
  const out = {
    bomb_declare_count: 0,
    shaking_declare_count: 0,
  };
  for (let i = Math.max(0, beforeLen); i < afterKibo.length; i += 1) {
    const type = String(afterKibo[i]?.type || "");
    if (type === "declare_bomb") out.bomb_declare_count += 1;
    else if (type === "shaking_declare") out.shaking_declare_count += 1;
    else if (type === "turn_end") {
      const actionType = String(afterKibo[i]?.action?.type || "");
      if (actionType === "declare_bomb") out.bomb_declare_count += 1;
    }
  }
  return out;
}

function incrementCounter(map, key) {
  const k = String(key || "");
  if (!k) return;
  map[k] = Number(map[k] || 0) + 1;
}

// =============================================================================
// Section 4. Round Simulation + Summary Helpers
// =============================================================================
function resolveFirstTurnKey(opts, gameIndex) {
  if (opts.firstTurnPolicy === "fixed") return opts.fixedFirstTurn;
  return gameIndex % 2 === 0 ? "ai" : "human";
}

function goldDiffByActor(state, actor) {
  const opp = actor === "human" ? "ai" : "human";
  const selfGold = Number(state?.players?.[actor]?.gold || 0);
  const oppGold = Number(state?.players?.[opp]?.gold || 0);
  return selfGold - oppGold;
}

function runDuelRound(initialState, seed, playerByActor, maxSteps, onDecision = null) {
  const rng = createSeededRng(`${seed}|rng`);
  let state = initialState;
  let steps = 0;

  while (state.phase !== "resolution" && steps < maxSteps) {
    const actor = getActionPlayerKey(state);
    if (!actor) break;

    const before = stateProgressKey(state);
    const sp = selectPool(state, actor);
    const cards = sp.cards || null;
    const boardCardIds = sp.boardCardIds || null;
    const options = sp.options || null;
    const decisionType = cards ? "play" : boardCardIds ? "match" : options ? "option" : null;
    const candidates = decisionType ? legalCandidatesForDecision(sp, decisionType) : [];
    const specialFlags = decisionAvailabilityFlags(state, actor);
    const playerSpec = playerByActor[actor];
    const policy = String(playerSpec?.label || "");
    const action = resolveResolvedPlayerAction(state, actor, playerSpec);
    let actionSource = String(action?.actionSource || "heuristic");
    let next = action?.next || state;

    if (!next || stateProgressKey(next) === before) {
      actionSource = "fallback_random";
      next = randomLegalAction(state, actor, rng);
    }
    if (!next || stateProgressKey(next) === before) {
      throw new Error(
        `action resolution failed after fallback: seed=${seed}, step=${steps}, actor=${actor}, phase=${String(state?.phase || "")}, policy=${policy}, source=${actionSource}`
      );
    }

    if (typeof onDecision === "function" && decisionType && candidates.length > 0) {
      const chosenCandidate = inferChosenCandidateFromTransition(
        state,
        actor,
        decisionType,
        candidates,
        next
      );
      const transitionEvents = collectTransitionEventCounts(state, next);
      onDecision({
        stateBefore: state,
        stateAfter: next,
        actor,
        policy,
        decisionType,
        candidates,
        chosenCandidate,
        specialFlags,
        transitionEvents,
        actionSource,
        step: steps,
      });
    }

    state = next;
    steps += 1;
  }

  return state;
}

function createSeatRecord() {
  return {
    games: 0,
    wins: 0,
    losses: 0,
    draws: 0,
    go_count_total: 0,
    go_game_count: 0,
    go_fail_count: 0,
    go_opportunity_count_total: 0,
    go_opportunity_game_count: 0,
    shaking_count_total: 0,
    shaking_game_count: 0,
    shaking_win_game_count: 0,
    shaking_opportunity_count_total: 0,
    shaking_opportunity_game_count: 0,
    shaking_plain_play_total: 0,
    shaking_skip_other_play_total: 0,
    shaking_no_total: 0,
    bomb_count_total: 0,
    bomb_game_count: 0,
    bomb_win_game_count: 0,
    bomb_opportunity_count_total: 0,
    bomb_opportunity_game_count: 0,
    bomb_plain_play_total: 0,
    bomb_skip_other_play_total: 0,
    president_stop_total: 0,
    president_hold_total: 0,
    president_stop_win_total: 0,
    president_stop_loss_total: 0,
    president_hold_win_total: 0,
    president_hold_loss_total: 0,
    president_opportunity_count_total: 0,
    president_opportunity_game_count: 0,
    gukjin_five_total: 0,
    gukjin_five_mongbak_total: 0,
    gukjin_junk_total: 0,
    gukjin_junk_mongbak_total: 0,
    gukjin_opportunity_count_total: 0,
    gukjin_opportunity_game_count: 0,
    gold_deltas: [],
  };
}

function createRoundSpecialMetrics() {
  return {
    go_opportunity_count: 0,
    shaking_count: 0,
    shaking_opportunity_count: 0,
    shaking_plain_play_count: 0,
    shaking_skip_other_play_count: 0,
    shaking_no_count: 0,
    bomb_count: 0,
    bomb_opportunity_count: 0,
    bomb_plain_play_count: 0,
    bomb_skip_other_play_count: 0,
    president_stop_count: 0,
    president_hold_count: 0,
    president_opportunity_count: 0,
    gukjin_five_count: 0,
    gukjin_five_mongbak_count: 0,
    gukjin_junk_count: 0,
    gukjin_junk_mongbak_count: 0,
    gukjin_opportunity_count: 0,
  };
}

function collectUniqueShakingOpportunitySignatures(state, actor) {
  if (state?.phase !== "playing" || state?.currentTurn !== actor) return [];
  const player = state?.players?.[actor];
  if (!player) return [];
  const declared = new Set(player.shakingDeclaredMonths || []);
  const counts = new Map();
  for (const card of player.hand || []) {
    if (!card || card.passCard) continue;
    const month = Number(card.month || 0);
    if (month < 1 || month > 12) continue;
    const current = counts.get(month) || [];
    current.push(String(card.id || ""));
    counts.set(month, current);
  }
  const signatures = [];
  for (const [month, ids] of counts.entries()) {
    if (ids.length < 3) continue;
    if (declared.has(month)) continue;
    if ((state.board || []).some((card) => Number(card?.month || 0) === month)) continue;
    signatures.push(`${month}:${ids.filter(Boolean).sort().join("|")}`);
  }
  return signatures;
}

function collectUniqueBombOpportunitySignatures(state, actor) {
  if (state?.phase !== "playing" || state?.currentTurn !== actor) return [];
  const player = state?.players?.[actor];
  if (!player) return [];
  const bombMonths = getDeclarableBombMonths(state, actor);
  if (!Array.isArray(bombMonths) || bombMonths.length === 0) return [];
  const signatures = [];
  for (const month of bombMonths) {
    const monthNum = Number(month || 0);
    const handIds = (player.hand || [])
      .filter((card) => card && !card.passCard && Number(card.month || 0) === monthNum)
      .map((card) => String(card.id || ""))
      .filter(Boolean)
      .sort();
    const boardIds = (state.board || [])
      .filter((card) => Number(card?.month || 0) === monthNum)
      .map((card) => String(card?.id || ""))
      .filter(Boolean)
      .sort();
    signatures.push(`${monthNum}:${handIds.join("|")}::${boardIds.join("|")}`);
  }
  return signatures;
}

function parseShakingOpportunitySignature(signature) {
  const raw = String(signature || "");
  const sep = raw.indexOf(":");
  if (sep < 0) return { month: 0, cardIds: [] };
  const month = Number(raw.slice(0, sep));
  const cardIds = raw
    .slice(sep + 1)
    .split("|")
    .map((id) => String(id || "").trim())
    .filter(Boolean);
  return { month, cardIds };
}

function parseBombOpportunitySignature(signature) {
  const raw = String(signature || "");
  const boardSep = raw.indexOf("::");
  const left = boardSep >= 0 ? raw.slice(0, boardSep) : raw;
  const monthSep = left.indexOf(":");
  if (monthSep < 0) return { month: 0 };
  return { month: Number(left.slice(0, monthSep)) };
}

function classifyPlaySpecialOpportunityResolution(
  state,
  actor,
  chosenCandidate,
  shakingSignatures = null,
  bombSignatures = null
) {
  const out = {
    shaking_plain_play_count: 0,
    shaking_skip_other_play_count: 0,
    bomb_plain_play_count: 0,
    bomb_skip_other_play_count: 0,
  };
  if (state?.phase !== "playing" || state?.currentTurn !== actor) return out;

  const normalizedChosen = normalizeDecisionCandidate("play", chosenCandidate);
  if (!normalizedChosen) return out;

  const chosenSpecial = parsePlaySpecialCandidate(normalizedChosen);
  const player = state?.players?.[actor];
  const chosenCard =
    !chosenSpecial && player
      ? (player.hand || []).find((card) => String(card?.id || "") === normalizedChosen) || null
      : null;

  const effectiveShakingSignatures = Array.isArray(shakingSignatures)
    ? shakingSignatures
    : collectUniqueShakingOpportunitySignatures(state, actor);
  const effectiveBombSignatures = Array.isArray(bombSignatures)
    ? bombSignatures
    : collectUniqueBombOpportunitySignatures(state, actor);

  for (const signature of effectiveShakingSignatures) {
    const { cardIds } = parseShakingOpportunitySignature(signature);
    if (
      chosenSpecial?.kind === "shake_start" &&
      cardIds.includes(String(chosenSpecial.cardId || ""))
    ) {
      continue;
    }
    if (chosenCard && cardIds.includes(String(chosenCard.id || ""))) {
      out.shaking_plain_play_count += 1;
    } else {
      out.shaking_skip_other_play_count += 1;
    }
  }

  for (const signature of effectiveBombSignatures) {
    const { month } = parseBombOpportunitySignature(signature);
    if (chosenSpecial?.kind === "bomb" && Number(chosenSpecial.month || 0) === month) continue;
    if (chosenCard && Number(chosenCard.month || 0) === month) {
      out.bomb_plain_play_count += 1;
    } else {
      out.bomb_skip_other_play_count += 1;
    }
  }

  return out;
}

function updateSeatRecord(
  record,
  winner,
  selfActor,
  oppActor,
  goldDelta,
  roundMetrics
) {
  record.games += 1;
  if (winner === selfActor) record.wins += 1;
  else if (winner === oppActor) record.losses += 1;
  else record.draws += 1;
  const goCount = Math.max(0, Number(roundMetrics?.go_count || 0));
  const goOpportunityCount = Math.max(0, Number(roundMetrics?.go_opportunity_count || 0));
  const shakingCount = Math.max(0, Number(roundMetrics?.shaking_count || 0));
  const shakingOpportunityCount = Math.max(0, Number(roundMetrics?.shaking_opportunity_count || 0));
  const shakingPlainPlayCount = Math.max(0, Number(roundMetrics?.shaking_plain_play_count || 0));
  const shakingSkipOtherPlayCount = Math.max(
    0,
    Number(roundMetrics?.shaking_skip_other_play_count || 0)
  );
  const shakingNoCount = Math.max(0, Number(roundMetrics?.shaking_no_count || 0));
  const bombCount = Math.max(0, Number(roundMetrics?.bomb_count || 0));
  const bombOpportunityCount = Math.max(0, Number(roundMetrics?.bomb_opportunity_count || 0));
  const bombPlainPlayCount = Math.max(0, Number(roundMetrics?.bomb_plain_play_count || 0));
  const bombSkipOtherPlayCount = Math.max(
    0,
    Number(roundMetrics?.bomb_skip_other_play_count || 0)
  );
  const presidentStopCount = Math.max(0, Number(roundMetrics?.president_stop_count || 0));
  const presidentHoldCount = Math.max(0, Number(roundMetrics?.president_hold_count || 0));
  const presidentOpportunityCount = Math.max(0, Number(roundMetrics?.president_opportunity_count || 0));
  const gukjinFiveCount = Math.max(0, Number(roundMetrics?.gukjin_five_count || 0));
  const gukjinFiveMongBakCount = Math.max(0, Number(roundMetrics?.gukjin_five_mongbak_count || 0));
  const gukjinJunkCount = Math.max(0, Number(roundMetrics?.gukjin_junk_count || 0));
  const gukjinJunkMongBakCount = Math.max(0, Number(roundMetrics?.gukjin_junk_mongbak_count || 0));
  const gukjinOpportunityCount = Math.max(0, Number(roundMetrics?.gukjin_opportunity_count || 0));
  record.go_count_total += goCount;
  record.go_opportunity_count_total += goOpportunityCount;
  if (goOpportunityCount > 0) {
    record.go_opportunity_game_count += 1;
  }
  if (goCount > 0) {
    record.go_game_count += 1;
    if (winner !== selfActor) {
      record.go_fail_count += 1;
    }
  }
  record.shaking_count_total += shakingCount;
  record.shaking_opportunity_count_total += shakingOpportunityCount;
  record.shaking_plain_play_total += shakingPlainPlayCount;
  record.shaking_skip_other_play_total += shakingSkipOtherPlayCount;
  record.shaking_no_total += shakingNoCount;
  if (shakingCount > 0) record.shaking_game_count += 1;
  if (shakingCount > 0 && winner === selfActor) record.shaking_win_game_count += 1;
  if (shakingOpportunityCount > 0) record.shaking_opportunity_game_count += 1;
  record.bomb_count_total += bombCount;
  record.bomb_opportunity_count_total += bombOpportunityCount;
  record.bomb_plain_play_total += bombPlainPlayCount;
  record.bomb_skip_other_play_total += bombSkipOtherPlayCount;
  if (bombCount > 0) record.bomb_game_count += 1;
  if (bombCount > 0 && winner === selfActor) record.bomb_win_game_count += 1;
  if (bombOpportunityCount > 0) record.bomb_opportunity_game_count += 1;
  record.president_stop_total += presidentStopCount;
  record.president_hold_total += presidentHoldCount;
  if (presidentStopCount > 0) {
    if (winner === selfActor) record.president_stop_win_total += presidentStopCount;
    else if (winner === oppActor) record.president_stop_loss_total += presidentStopCount;
  }
  if (presidentHoldCount > 0) {
    if (winner === selfActor) record.president_hold_win_total += presidentHoldCount;
    else if (winner === oppActor) record.president_hold_loss_total += presidentHoldCount;
  }
  record.president_opportunity_count_total += presidentOpportunityCount;
  if (presidentOpportunityCount > 0) record.president_opportunity_game_count += 1;
  record.gukjin_five_total += gukjinFiveCount;
  record.gukjin_five_mongbak_total += gukjinFiveMongBakCount;
  record.gukjin_junk_total += gukjinJunkCount;
  record.gukjin_junk_mongbak_total += gukjinJunkMongBakCount;
  record.gukjin_opportunity_count_total += gukjinOpportunityCount;
  if (gukjinOpportunityCount > 0) record.gukjin_opportunity_game_count += 1;
  record.gold_deltas.push(Number(goldDelta || 0));
}

function finalizeSeatRecord(record) {
  const games = Number(record?.games || 0);
  const wins = Number(record?.wins || 0);
  const losses = Number(record?.losses || 0);
  const draws = Number(record?.draws || 0);
  const goCountTotal = Number(record?.go_count_total || 0);
  const goGameCount = Number(record?.go_game_count || 0);
  const goFailCount = Number(record?.go_fail_count || 0);
  const goOpportunityCountTotal = Number(record?.go_opportunity_count_total || 0);
  const goOpportunityGameCount = Number(record?.go_opportunity_game_count || 0);
  const shakingCountTotal = Number(record?.shaking_count_total || 0);
  const shakingGameCount = Number(record?.shaking_game_count || 0);
  const shakingWinGameCount = Number(record?.shaking_win_game_count || 0);
  const shakingOpportunityCountTotal = Number(record?.shaking_opportunity_count_total || 0);
  const shakingOpportunityGameCount = Number(record?.shaking_opportunity_game_count || 0);
  const shakingPlainPlayTotal = Number(record?.shaking_plain_play_total || 0);
  const shakingSkipOtherPlayTotal = Number(record?.shaking_skip_other_play_total || 0);
  const shakingNoTotal = Number(record?.shaking_no_total || 0);
  const bombCountTotal = Number(record?.bomb_count_total || 0);
  const bombGameCount = Number(record?.bomb_game_count || 0);
  const bombWinGameCount = Number(record?.bomb_win_game_count || 0);
  const bombOpportunityCountTotal = Number(record?.bomb_opportunity_count_total || 0);
  const bombOpportunityGameCount = Number(record?.bomb_opportunity_game_count || 0);
  const bombPlainPlayTotal = Number(record?.bomb_plain_play_total || 0);
  const bombSkipOtherPlayTotal = Number(record?.bomb_skip_other_play_total || 0);
  const presidentStopTotal = Number(record?.president_stop_total || 0);
  const presidentHoldTotal = Number(record?.president_hold_total || 0);
  const presidentStopWinTotal = Number(record?.president_stop_win_total || 0);
  const presidentStopLossTotal = Number(record?.president_stop_loss_total || 0);
  const presidentHoldWinTotal = Number(record?.president_hold_win_total || 0);
  const presidentHoldLossTotal = Number(record?.president_hold_loss_total || 0);
  const presidentOpportunityCountTotal = Number(record?.president_opportunity_count_total || 0);
  const presidentOpportunityGameCount = Number(record?.president_opportunity_game_count || 0);
  const gukjinFiveTotal = Number(record?.gukjin_five_total || 0);
  const gukjinFiveMongBakTotal = Number(record?.gukjin_five_mongbak_total || 0);
  const gukjinJunkTotal = Number(record?.gukjin_junk_total || 0);
  const gukjinJunkMongBakTotal = Number(record?.gukjin_junk_mongbak_total || 0);
  const gukjinOpportunityCountTotal = Number(record?.gukjin_opportunity_count_total || 0);
  const gukjinOpportunityGameCount = Number(record?.gukjin_opportunity_game_count || 0);
  const deltas = Array.isArray(record?.gold_deltas) ? record.gold_deltas : [];
  const meanGoldDelta = deltas.length > 0 ? deltas.reduce((a, b) => a + b, 0) / deltas.length : 0;
  return {
    games,
    wins,
    losses,
    draws,
    win_rate: games > 0 ? wins / games : 0,
    loss_rate: games > 0 ? losses / games : 0,
    draw_rate: games > 0 ? draws / games : 0,
    go_count_total: goCountTotal,
    go_avg_per_game: games > 0 ? goCountTotal / games : 0,
    go_game_count: goGameCount,
    go_fail_count: goFailCount,
    go_success_count: Math.max(0, goGameCount - goFailCount),
    go_fail_rate: goGameCount > 0 ? goFailCount / goGameCount : 0,
    go_opportunity_count_total: goOpportunityCountTotal,
    go_opportunity_game_count: goOpportunityGameCount,
    go_opportunity_rate: games > 0 ? goOpportunityGameCount / games : 0,
    go_take_rate: goOpportunityCountTotal > 0 ? goCountTotal / goOpportunityCountTotal : 0,
    shaking_count_total: shakingCountTotal,
    shaking_game_count: shakingGameCount,
    shaking_win_game_count: shakingWinGameCount,
    shaking_opportunity_count_total: shakingOpportunityCountTotal,
    shaking_opportunity_game_count: shakingOpportunityGameCount,
    shaking_opportunity_rate: games > 0 ? shakingOpportunityGameCount / games : 0,
    shaking_take_rate: shakingOpportunityCountTotal > 0 ? shakingCountTotal / shakingOpportunityCountTotal : 0,
    shaking_plain_play_total: shakingPlainPlayTotal,
    shaking_skip_other_play_total: shakingSkipOtherPlayTotal,
    shaking_no_total: shakingNoTotal,
    shaking_plain_play_rate:
      shakingOpportunityCountTotal > 0 ? shakingPlainPlayTotal / shakingOpportunityCountTotal : 0,
    shaking_skip_other_play_rate:
      shakingOpportunityCountTotal > 0
        ? shakingSkipOtherPlayTotal / shakingOpportunityCountTotal
        : 0,
    shaking_no_rate: shakingOpportunityCountTotal > 0 ? shakingNoTotal / shakingOpportunityCountTotal : 0,
    bomb_count_total: bombCountTotal,
    bomb_game_count: bombGameCount,
    bomb_win_game_count: bombWinGameCount,
    bomb_opportunity_count_total: bombOpportunityCountTotal,
    bomb_opportunity_game_count: bombOpportunityGameCount,
    bomb_opportunity_rate: games > 0 ? bombOpportunityGameCount / games : 0,
    bomb_take_rate: bombOpportunityCountTotal > 0 ? bombCountTotal / bombOpportunityCountTotal : 0,
    bomb_plain_play_total: bombPlainPlayTotal,
    bomb_skip_other_play_total: bombSkipOtherPlayTotal,
    bomb_plain_play_rate:
      bombOpportunityCountTotal > 0 ? bombPlainPlayTotal / bombOpportunityCountTotal : 0,
    bomb_skip_other_play_rate:
      bombOpportunityCountTotal > 0 ? bombSkipOtherPlayTotal / bombOpportunityCountTotal : 0,
    president_stop_total: presidentStopTotal,
    president_hold_total: presidentHoldTotal,
    president_stop_win_total: presidentStopWinTotal,
    president_stop_loss_total: presidentStopLossTotal,
    president_hold_win_total: presidentHoldWinTotal,
    president_hold_loss_total: presidentHoldLossTotal,
    president_opportunity_count_total: presidentOpportunityCountTotal,
    president_opportunity_game_count: presidentOpportunityGameCount,
    president_stop_rate: presidentOpportunityCountTotal > 0 ? presidentStopTotal / presidentOpportunityCountTotal : 0,
    president_stop_win_rate: presidentStopTotal > 0 ? presidentStopWinTotal / presidentStopTotal : 0,
    president_hold_win_rate: presidentHoldTotal > 0 ? presidentHoldWinTotal / presidentHoldTotal : 0,
    gukjin_five_total: gukjinFiveTotal,
    gukjin_five_mongbak_total: gukjinFiveMongBakTotal,
    gukjin_junk_total: gukjinJunkTotal,
    gukjin_junk_mongbak_total: gukjinJunkMongBakTotal,
    gukjin_opportunity_count_total: gukjinOpportunityCountTotal,
    gukjin_opportunity_game_count: gukjinOpportunityGameCount,
    gukjin_five_rate: gukjinOpportunityCountTotal > 0 ? gukjinFiveTotal / gukjinOpportunityCountTotal : 0,
    gukjin_five_mongbak_rate: gukjinFiveTotal > 0 ? gukjinFiveMongBakTotal / gukjinFiveTotal : 0,
    gukjin_junk_mongbak_rate: gukjinJunkTotal > 0 ? gukjinJunkMongBakTotal / gukjinJunkTotal : 0,
    mean_gold_delta: meanGoldDelta,
    p10_gold_delta: quantile(deltas, 0.1),
    p50_gold_delta: quantile(deltas, 0.5),
    p90_gold_delta: quantile(deltas, 0.9),
  };
}

function buildSeatSplitSummary(firstRecord, secondRecord) {
  const combined = createSeatRecord();
  combined.games = Number(firstRecord.games || 0) + Number(secondRecord.games || 0);
  combined.wins = Number(firstRecord.wins || 0) + Number(secondRecord.wins || 0);
  combined.losses = Number(firstRecord.losses || 0) + Number(secondRecord.losses || 0);
  combined.draws = Number(firstRecord.draws || 0) + Number(secondRecord.draws || 0);
  combined.go_count_total =
    Number(firstRecord.go_count_total || 0) + Number(secondRecord.go_count_total || 0);
  combined.go_game_count =
    Number(firstRecord.go_game_count || 0) + Number(secondRecord.go_game_count || 0);
  combined.go_fail_count =
    Number(firstRecord.go_fail_count || 0) + Number(secondRecord.go_fail_count || 0);
  combined.go_opportunity_count_total =
    Number(firstRecord.go_opportunity_count_total || 0) +
    Number(secondRecord.go_opportunity_count_total || 0);
  combined.go_opportunity_game_count =
    Number(firstRecord.go_opportunity_game_count || 0) +
    Number(secondRecord.go_opportunity_game_count || 0);
  combined.shaking_count_total =
    Number(firstRecord.shaking_count_total || 0) + Number(secondRecord.shaking_count_total || 0);
  combined.shaking_game_count =
    Number(firstRecord.shaking_game_count || 0) + Number(secondRecord.shaking_game_count || 0);
  combined.shaking_win_game_count =
    Number(firstRecord.shaking_win_game_count || 0) + Number(secondRecord.shaking_win_game_count || 0);
  combined.shaking_opportunity_count_total =
    Number(firstRecord.shaking_opportunity_count_total || 0) +
    Number(secondRecord.shaking_opportunity_count_total || 0);
  combined.shaking_opportunity_game_count =
    Number(firstRecord.shaking_opportunity_game_count || 0) +
    Number(secondRecord.shaking_opportunity_game_count || 0);
  combined.shaking_plain_play_total =
    Number(firstRecord.shaking_plain_play_total || 0) +
    Number(secondRecord.shaking_plain_play_total || 0);
  combined.shaking_skip_other_play_total =
    Number(firstRecord.shaking_skip_other_play_total || 0) +
    Number(secondRecord.shaking_skip_other_play_total || 0);
  combined.shaking_no_total =
    Number(firstRecord.shaking_no_total || 0) + Number(secondRecord.shaking_no_total || 0);
  combined.bomb_count_total =
    Number(firstRecord.bomb_count_total || 0) + Number(secondRecord.bomb_count_total || 0);
  combined.bomb_game_count =
    Number(firstRecord.bomb_game_count || 0) + Number(secondRecord.bomb_game_count || 0);
  combined.bomb_win_game_count =
    Number(firstRecord.bomb_win_game_count || 0) + Number(secondRecord.bomb_win_game_count || 0);
  combined.bomb_opportunity_count_total =
    Number(firstRecord.bomb_opportunity_count_total || 0) +
    Number(secondRecord.bomb_opportunity_count_total || 0);
  combined.bomb_opportunity_game_count =
    Number(firstRecord.bomb_opportunity_game_count || 0) +
    Number(secondRecord.bomb_opportunity_game_count || 0);
  combined.bomb_plain_play_total =
    Number(firstRecord.bomb_plain_play_total || 0) +
    Number(secondRecord.bomb_plain_play_total || 0);
  combined.bomb_skip_other_play_total =
    Number(firstRecord.bomb_skip_other_play_total || 0) +
    Number(secondRecord.bomb_skip_other_play_total || 0);
  combined.president_stop_total =
    Number(firstRecord.president_stop_total || 0) + Number(secondRecord.president_stop_total || 0);
  combined.president_hold_total =
    Number(firstRecord.president_hold_total || 0) + Number(secondRecord.president_hold_total || 0);
  combined.president_stop_win_total =
    Number(firstRecord.president_stop_win_total || 0) + Number(secondRecord.president_stop_win_total || 0);
  combined.president_stop_loss_total =
    Number(firstRecord.president_stop_loss_total || 0) + Number(secondRecord.president_stop_loss_total || 0);
  combined.president_hold_win_total =
    Number(firstRecord.president_hold_win_total || 0) + Number(secondRecord.president_hold_win_total || 0);
  combined.president_hold_loss_total =
    Number(firstRecord.president_hold_loss_total || 0) + Number(secondRecord.president_hold_loss_total || 0);
  combined.president_opportunity_count_total =
    Number(firstRecord.president_opportunity_count_total || 0) +
    Number(secondRecord.president_opportunity_count_total || 0);
  combined.president_opportunity_game_count =
    Number(firstRecord.president_opportunity_game_count || 0) +
    Number(secondRecord.president_opportunity_game_count || 0);
  combined.gukjin_five_total =
    Number(firstRecord.gukjin_five_total || 0) + Number(secondRecord.gukjin_five_total || 0);
  combined.gukjin_five_mongbak_total =
    Number(firstRecord.gukjin_five_mongbak_total || 0) +
    Number(secondRecord.gukjin_five_mongbak_total || 0);
  combined.gukjin_junk_total =
    Number(firstRecord.gukjin_junk_total || 0) + Number(secondRecord.gukjin_junk_total || 0);
  combined.gukjin_junk_mongbak_total =
    Number(firstRecord.gukjin_junk_mongbak_total || 0) +
    Number(secondRecord.gukjin_junk_mongbak_total || 0);
  combined.gukjin_opportunity_count_total =
    Number(firstRecord.gukjin_opportunity_count_total || 0) +
    Number(secondRecord.gukjin_opportunity_count_total || 0);
  combined.gukjin_opportunity_game_count =
    Number(firstRecord.gukjin_opportunity_game_count || 0) +
    Number(secondRecord.gukjin_opportunity_game_count || 0);
  combined.gold_deltas = [
    ...(Array.isArray(firstRecord.gold_deltas) ? firstRecord.gold_deltas : []),
    ...(Array.isArray(secondRecord.gold_deltas) ? secondRecord.gold_deltas : []),
  ];
  return {
    when_first: finalizeSeatRecord(firstRecord),
    when_second: finalizeSeatRecord(secondRecord),
    combined: finalizeSeatRecord(combined),
  };
}

function buildConsoleSummary(report) {
  const seatAFirst = report?.seat_split_a?.when_first || {};
  const seatASecond = report?.seat_split_a?.when_second || {};
  const seatBFirst = report?.seat_split_b?.when_first || {};
  const seatBSecond = report?.seat_split_b?.when_second || {};
  return {
    games: Number(report?.games || 0),
    human: String(report?.human || ""),
    ai: String(report?.ai || ""),
    first_turn_policy: String(report?.first_turn_policy || ""),
    fixed_first_turn: report?.fixed_first_turn ?? null,
    continuous_series: !!report?.continuous_series,
    kibo_detail: String(report?.kibo_detail || "none"),
    result_out: String(report?.result_out || ""),
    wins_a: Number(report?.wins_a || 0),
    losses_a: Number(report?.losses_a || 0),
    wins_b: Number(report?.wins_b || 0),
    losses_b: Number(report?.losses_b || 0),
    draws: Number(report?.draws || 0),
    win_rate_a: Number(report?.win_rate_a || 0),
    win_rate_b: Number(report?.win_rate_b || 0),
    draw_rate: Number(report?.draw_rate || 0),
    seat_split_a: {
      when_first: {
        win_rate: Number(seatAFirst.win_rate || 0),
        mean_gold_delta: Number(seatAFirst.mean_gold_delta || 0),
      },
      when_second: {
        win_rate: Number(seatASecond.win_rate || 0),
        mean_gold_delta: Number(seatASecond.mean_gold_delta || 0),
      },
    },
    seat_split_b: {
      when_first: {
        win_rate: Number(seatBFirst.win_rate || 0),
        mean_gold_delta: Number(seatBFirst.mean_gold_delta || 0),
      },
      when_second: {
        win_rate: Number(seatBSecond.win_rate || 0),
        mean_gold_delta: Number(seatBSecond.mean_gold_delta || 0),
      },
    },
    mean_gold_delta_a: Number(report?.mean_gold_delta_a || 0),
    p10_gold_delta_a: Number(report?.p10_gold_delta_a || 0),
    p50_gold_delta_a: Number(report?.p50_gold_delta_a || 0),
    p90_gold_delta_a: Number(report?.p90_gold_delta_a || 0),
    go_count_a: Number(report?.go_count_a || 0),
    go_count_b: Number(report?.go_count_b || 0),
    go_games_a: Number(report?.go_games_a || 0),
    go_games_b: Number(report?.go_games_b || 0),
    go_fail_count_a: Number(report?.go_fail_count_a || 0),
    go_fail_count_b: Number(report?.go_fail_count_b || 0),
    go_fail_rate_a: Number(report?.go_fail_rate_a || 0),
    go_fail_rate_b: Number(report?.go_fail_rate_b || 0),
    go_opportunity_count_a: Number(report?.go_opportunity_count_a || 0),
    go_opportunity_count_b: Number(report?.go_opportunity_count_b || 0),
    go_opportunity_games_a: Number(report?.go_opportunity_games_a || 0),
    go_opportunity_games_b: Number(report?.go_opportunity_games_b || 0),
    go_opportunity_rate_a: Number(report?.go_opportunity_rate_a || 0),
    go_opportunity_rate_b: Number(report?.go_opportunity_rate_b || 0),
    go_take_rate_a: Number(report?.go_take_rate_a || 0),
    go_take_rate_b: Number(report?.go_take_rate_b || 0),
    shaking_count_a: Number(report?.shaking_count_a || 0),
    shaking_count_b: Number(report?.shaking_count_b || 0),
    shaking_games_a: Number(report?.shaking_games_a || 0),
    shaking_games_b: Number(report?.shaking_games_b || 0),
    shaking_win_a: Number(report?.shaking_win_a || 0),
    shaking_win_b: Number(report?.shaking_win_b || 0),
    shaking_opportunity_count_a: Number(report?.shaking_opportunity_count_a || 0),
    shaking_opportunity_count_b: Number(report?.shaking_opportunity_count_b || 0),
    shaking_opportunity_games_a: Number(report?.shaking_opportunity_games_a || 0),
    shaking_opportunity_games_b: Number(report?.shaking_opportunity_games_b || 0),
    shaking_take_rate_a: Number(report?.shaking_take_rate_a || 0),
    shaking_take_rate_b: Number(report?.shaking_take_rate_b || 0),
    shaking_plain_play_a: Number(report?.shaking_plain_play_a || 0),
    shaking_plain_play_b: Number(report?.shaking_plain_play_b || 0),
    shaking_skip_other_play_a: Number(report?.shaking_skip_other_play_a || 0),
    shaking_skip_other_play_b: Number(report?.shaking_skip_other_play_b || 0),
    shaking_no_a: Number(report?.shaking_no_a || 0),
    shaking_no_b: Number(report?.shaking_no_b || 0),
    bomb_count_a: Number(report?.bomb_count_a || 0),
    bomb_count_b: Number(report?.bomb_count_b || 0),
    bomb_games_a: Number(report?.bomb_games_a || 0),
    bomb_games_b: Number(report?.bomb_games_b || 0),
    bomb_win_a: Number(report?.bomb_win_a || 0),
    bomb_win_b: Number(report?.bomb_win_b || 0),
    bomb_opportunity_count_a: Number(report?.bomb_opportunity_count_a || 0),
    bomb_opportunity_count_b: Number(report?.bomb_opportunity_count_b || 0),
    bomb_opportunity_games_a: Number(report?.bomb_opportunity_games_a || 0),
    bomb_opportunity_games_b: Number(report?.bomb_opportunity_games_b || 0),
    bomb_take_rate_a: Number(report?.bomb_take_rate_a || 0),
    bomb_take_rate_b: Number(report?.bomb_take_rate_b || 0),
    bomb_plain_play_a: Number(report?.bomb_plain_play_a || 0),
    bomb_plain_play_b: Number(report?.bomb_plain_play_b || 0),
    bomb_skip_other_play_a: Number(report?.bomb_skip_other_play_a || 0),
    bomb_skip_other_play_b: Number(report?.bomb_skip_other_play_b || 0),
    president_stop_a: Number(report?.president_stop_a || 0),
    president_stop_b: Number(report?.president_stop_b || 0),
    president_hold_a: Number(report?.president_hold_a || 0),
    president_hold_b: Number(report?.president_hold_b || 0),
    president_hold_win_a: Number(report?.president_hold_win_a || 0),
    president_hold_win_b: Number(report?.president_hold_win_b || 0),
    president_opportunity_count_a: Number(report?.president_opportunity_count_a || 0),
    president_opportunity_count_b: Number(report?.president_opportunity_count_b || 0),
    gukjin_five_a: Number(report?.gukjin_five_a || 0),
    gukjin_five_b: Number(report?.gukjin_five_b || 0),
    gukjin_five_mongbak_a: Number(report?.gukjin_five_mongbak_a || 0),
    gukjin_five_mongbak_b: Number(report?.gukjin_five_mongbak_b || 0),
    gukjin_junk_a: Number(report?.gukjin_junk_a || 0),
    gukjin_junk_b: Number(report?.gukjin_junk_b || 0),
    gukjin_opportunity_count_a: Number(report?.gukjin_opportunity_count_a || 0),
    gukjin_opportunity_count_b: Number(report?.gukjin_opportunity_count_b || 0),
    gukjin_five_rate_a: Number(report?.gukjin_five_rate_a || 0),
    gukjin_five_rate_b: Number(report?.gukjin_five_rate_b || 0),
    gukjin_five_mongbak_rate_a: Number(report?.gukjin_five_mongbak_rate_a || 0),
    gukjin_five_mongbak_rate_b: Number(report?.gukjin_five_mongbak_rate_b || 0),
    gukjin_junk_mongbak_a: Number(report?.gukjin_junk_mongbak_a || 0),
    gukjin_junk_mongbak_b: Number(report?.gukjin_junk_mongbak_b || 0),
    gukjin_junk_mongbak_rate_a: Number(report?.gukjin_junk_mongbak_rate_a || 0),
    gukjin_junk_mongbak_rate_b: Number(report?.gukjin_junk_mongbak_rate_b || 0),
    bankrupt: report?.bankrupt || { a_bankrupt_count: 0, b_bankrupt_count: 0 },
  };
}

function formatConsoleSummaryText(summary) {
  const safe = buildConsoleSummary(summary || {});
  const fmtRate = (value) => {
    const n = Number(value || 0);
    return Number.isFinite(n) ? n.toFixed(3) : "0.000";
  };
  const aFirst = safe.seat_split_a?.when_first || {};
  const aSecond = safe.seat_split_a?.when_second || {};
  const bFirst = safe.seat_split_b?.when_first || {};
  const bSecond = safe.seat_split_b?.when_second || {};
  const bankrupt = safe.bankrupt || { a_bankrupt_count: 0, b_bankrupt_count: 0 };
  const lines = [
    "",
    `=== Model Duel (${safe.human} vs ${safe.ai}, games=${safe.games}) ===`,
    `Win/Loss/Draw(A):  ${safe.wins_a} / ${safe.losses_a} / ${safe.draws}  (WR=${fmtRate(safe.win_rate_a)})`,
    `Win/Loss/Draw(B):  ${safe.wins_b} / ${safe.losses_b} / ${safe.draws}  (WR=${fmtRate(safe.win_rate_b)})`,
    `Seat A first:      WR=${fmtRate(aFirst.win_rate)}, mean_gold_delta=${aFirst.mean_gold_delta}`,
    `Seat A second:     WR=${fmtRate(aSecond.win_rate)}, mean_gold_delta=${aSecond.mean_gold_delta}`,
    `Seat B first:      WR=${fmtRate(bFirst.win_rate)}, mean_gold_delta=${bFirst.mean_gold_delta}`,
    `Seat B second:     WR=${fmtRate(bSecond.win_rate)}, mean_gold_delta=${bSecond.mean_gold_delta}`,
    `Gold delta(A):     mean=${safe.mean_gold_delta_a}, p10=${safe.p10_gold_delta_a}, p50=${safe.p50_gold_delta_a}, p90=${safe.p90_gold_delta_a}`,
    `GO A:              games=${safe.go_games_a}, count=${safe.go_count_a}, fail=${safe.go_fail_count_a}, fail_rate=${fmtRate(safe.go_fail_rate_a)}`,
    `GO B:              games=${safe.go_games_b}, count=${safe.go_count_b}, fail=${safe.go_fail_count_b}, fail_rate=${fmtRate(safe.go_fail_rate_b)}`,
    `GO Opp A:          opp_games=${safe.go_opportunity_games_a}, opp_turns=${safe.go_opportunity_count_a}, opp_rate=${fmtRate(safe.go_opportunity_rate_a)}, take_rate=${fmtRate(safe.go_take_rate_a)}`,
    `GO Opp B:          opp_games=${safe.go_opportunity_games_b}, opp_turns=${safe.go_opportunity_count_b}, opp_rate=${fmtRate(safe.go_opportunity_rate_b)}, take_rate=${fmtRate(safe.go_take_rate_b)}`,
    `Shake A:           opp_games=${safe.shaking_opportunity_games_a}, opp_unique=${safe.shaking_opportunity_count_a}, games=${safe.shaking_games_a}, count=${safe.shaking_count_a}, plain=${safe.shaking_plain_play_a}, other=${safe.shaking_skip_other_play_a}, no=${safe.shaking_no_a}, win=${safe.shaking_win_a}, take_rate=${fmtRate(safe.shaking_take_rate_a)}`,
    `Shake B:           opp_games=${safe.shaking_opportunity_games_b}, opp_unique=${safe.shaking_opportunity_count_b}, games=${safe.shaking_games_b}, count=${safe.shaking_count_b}, plain=${safe.shaking_plain_play_b}, other=${safe.shaking_skip_other_play_b}, no=${safe.shaking_no_b}, win=${safe.shaking_win_b}, take_rate=${fmtRate(safe.shaking_take_rate_b)}`,
    `Bomb A:            opp_games=${safe.bomb_opportunity_games_a}, opp_unique=${safe.bomb_opportunity_count_a}, games=${safe.bomb_games_a}, count=${safe.bomb_count_a}, plain=${safe.bomb_plain_play_a}, other=${safe.bomb_skip_other_play_a}, win=${safe.bomb_win_a}, take_rate=${fmtRate(safe.bomb_take_rate_a)}`,
    `Bomb B:            opp_games=${safe.bomb_opportunity_games_b}, opp_unique=${safe.bomb_opportunity_count_b}, games=${safe.bomb_games_b}, count=${safe.bomb_count_b}, plain=${safe.bomb_plain_play_b}, other=${safe.bomb_skip_other_play_b}, win=${safe.bomb_win_b}, take_rate=${fmtRate(safe.bomb_take_rate_b)}`,
    `President A:       opp_total=${safe.president_opportunity_count_a}, hold=${safe.president_hold_a}, hold_win=${safe.president_hold_win_a}, stop=${safe.president_stop_a}`,
    `President B:       opp_total=${safe.president_opportunity_count_b}, hold=${safe.president_hold_b}, hold_win=${safe.president_hold_win_b}, stop=${safe.president_stop_b}`,
    `Gukjin A:          opp_total=${safe.gukjin_opportunity_count_a}, five=${safe.gukjin_five_a}, junk=${safe.gukjin_junk_a}, five_rate=${fmtRate(safe.gukjin_five_rate_a)}, five_mongbak=${safe.gukjin_five_mongbak_a}(${fmtRate(safe.gukjin_five_mongbak_rate_a)}), junk_mongbak=${safe.gukjin_junk_mongbak_a}(${fmtRate(safe.gukjin_junk_mongbak_rate_a)})`,
    `Gukjin B:          opp_total=${safe.gukjin_opportunity_count_b}, five=${safe.gukjin_five_b}, junk=${safe.gukjin_junk_b}, five_rate=${fmtRate(safe.gukjin_five_rate_b)}, five_mongbak=${safe.gukjin_five_mongbak_b}(${fmtRate(safe.gukjin_five_mongbak_rate_b)}), junk_mongbak=${safe.gukjin_junk_mongbak_b}(${fmtRate(safe.gukjin_junk_mongbak_rate_b)})`,
    `Bankrupt:          A=${bankrupt.a_bankrupt_count}, B=${bankrupt.b_bankrupt_count}`,
    `Result file:       ${safe.result_out || ""}`,
    "===========================================================",
    "",
  ];
  return `${lines.join("\n")}\n`;
}

// =============================================================================
// Section 5. Entrypoint
// =============================================================================
export function runModelDuelCli(argv = process.argv.slice(2)) {
  const evalStartMs = Date.now();
  const opts = parseArgs(argv);
  const humanPlayer = attachGoStopIqnRuntime(
    resolvePlayerSpec(opts.humanSpecRaw, "human"),
    opts.humanGoStopIqnPath,
    "human"
  );
  const aiPlayer = attachGoStopIqnRuntime(
    resolvePlayerSpec(opts.aiSpecRaw, "ai"),
    opts.aiGoStopIqnPath,
    "ai"
  );
  ACTIVE_FEATURE_PROFILE = resolveDatasetFeatureProfile(opts.featureProfile, humanPlayer, aiPlayer);
  ACTIVE_COMPACT_FEATURES =
    ACTIVE_FEATURE_PROFILE === "legacy13"
      ? LEGACY13_FEATURES
      : ACTIVE_FEATURE_PROFILE === "material10"
        ? MATERIAL10_STAGING_FEATURES
        : ACTIVE_FEATURE_PROFILE === "position11"
          ? POSITION11_FEATURES
          : HAND10_FEATURES;
  let autoOutputDir = "";
  const ensureAutoOutputDir = () => {
    if (!autoOutputDir) {
      autoOutputDir = buildAutoOutputDir(humanPlayer.label, aiPlayer.label);
    }
    return autoOutputDir;
  };

  if (!opts.resultOut) {
    opts.resultOut = buildAutoArtifactPath(ensureAutoOutputDir(), opts.seed, "result.json");
  }

  if (opts.kiboDetail === "none" && opts.kiboOut) {
    opts.kiboDetail = "lean";
  }
  if (opts.kiboDetail !== "none" && !opts.kiboOut) {
    opts.kiboOut = buildAutoArtifactPath(ensureAutoOutputDir(), opts.seed, "kibo.jsonl");
  }
  if (opts.datasetOut === "auto") {
    opts.datasetOut = buildAutoArtifactPath(ensureAutoOutputDir(), opts.seed, "dataset.jsonl");
  }
  const effectiveKiboDetail = opts.kiboDetail === "none" ? "lean" : opts.kiboDetail;

  let kiboWriter = null;
  if (opts.kiboOut) {
    mkdirSync(dirname(opts.kiboOut), { recursive: true });
    kiboWriter = createWriteStream(opts.kiboOut, { flags: "w", encoding: "utf8" });
  }
  let datasetWriter = null;
  const datasetStats = {
    rows: 0,
    positive_rows: 0,
    decisions: 0,
    unresolved_decisions: 0,
  };
  const unresolvedStats = {
    decisions: 0,
    unresolved_decisions: 0,
    by_actor: {},
    by_policy: {},
    by_decision_type: {},
    by_action_source: {},
  };
  if (opts.datasetOut) {
    mkdirSync(dirname(opts.datasetOut), { recursive: true });
    datasetWriter = createWriteStream(opts.datasetOut, { flags: "w", encoding: "utf8" });
  }

  const actorA = "human";
  const actorB = "ai";
  const playerByActor = {
    [actorA]: humanPlayer,
    [actorB]: aiPlayer,
  };

  let winsA = 0;
  let winsB = 0;
  let draws = 0;
  const goldDeltasA = [];
  const bankrupt = {
    a_bankrupt_count: 0,
    b_bankrupt_count: 0,
  };
  const firstTurnCounts = {
    human: 0,
    ai: 0,
  };
  const seatSplitA = {
    first: createSeatRecord(),
    second: createSeatRecord(),
  };
  const seatSplitB = {
    first: createSeatRecord(),
    second: createSeatRecord(),
  };
  const seriesSession = {
    roundsPlayed: 0,
    previousEndState: null,
  };

  for (let gi = 0; gi < opts.games; gi += 1) {
    const firstTurnKey = resolveFirstTurnKey(opts, gi);
    firstTurnCounts[firstTurnKey] += 1;
    const seed = `${opts.seed}|g=${gi}|first=${firstTurnKey}|sr=${seriesSession.roundsPlayed}`;

    const roundStart = opts.continuousSeries
      ? seriesSession.previousEndState
        ? continueRound(seriesSession.previousEndState, seed, firstTurnKey, effectiveKiboDetail)
        : startRound(seed, firstTurnKey, effectiveKiboDetail)
      : startRound(seed, firstTurnKey, effectiveKiboDetail);
    const roundGoOpportunity = { human: 0, ai: 0 };
    const roundSpecial = {
      human: createRoundSpecialMetrics(),
      ai: createRoundSpecialMetrics(),
    };
    const roundUniqueShakingOpportunity = {
      human: new Set(),
      ai: new Set(),
    };
    const roundUniqueBombOpportunity = {
      human: new Set(),
      ai: new Set(),
    };

    const beforeDiffA = goldDiffByActor(roundStart, actorA);
    const endState = runDuelRound(
      roundStart,
      seed,
      playerByActor,
      Math.max(20, Math.floor(opts.maxSteps)),
      (decision) => {
        if (decision?.actor === "human" || decision?.actor === "ai") {
          const actorMetrics = roundSpecial[decision.actor];
          if (decision?.stateBefore?.phase === "playing") {
            const newShakingSignatures = [];
            for (const signature of collectUniqueShakingOpportunitySignatures(
              decision.stateBefore,
              decision.actor
            )) {
              const seen = roundUniqueShakingOpportunity[decision.actor];
              if (seen.has(signature)) continue;
              seen.add(signature);
              actorMetrics.shaking_opportunity_count += 1;
              newShakingSignatures.push(signature);
            }
            const newBombSignatures = [];
            for (const signature of collectUniqueBombOpportunitySignatures(
              decision.stateBefore,
              decision.actor
            )) {
              const seen = roundUniqueBombOpportunity[decision.actor];
              if (seen.has(signature)) continue;
              seen.add(signature);
              actorMetrics.bomb_opportunity_count += 1;
              newBombSignatures.push(signature);
            }
            const playSpecialResolution = classifyPlaySpecialOpportunityResolution(
              decision.stateBefore,
              decision.actor,
              decision.chosenCandidate,
              newShakingSignatures,
              newBombSignatures
            );
            actorMetrics.shaking_plain_play_count += Number(
              playSpecialResolution.shaking_plain_play_count || 0
            );
            actorMetrics.shaking_skip_other_play_count += Number(
              playSpecialResolution.shaking_skip_other_play_count || 0
            );
            actorMetrics.bomb_plain_play_count += Number(
              playSpecialResolution.bomb_plain_play_count || 0
            );
            actorMetrics.bomb_skip_other_play_count += Number(
              playSpecialResolution.bomb_skip_other_play_count || 0
            );
          }
          actorMetrics.shaking_count += Math.max(0, Number(decision?.transitionEvents?.shaking_declare_count || 0));
          actorMetrics.bomb_count += Math.max(0, Number(decision?.transitionEvents?.bomb_declare_count || 0));
          if (decision.decisionType === "option" && decision.chosenCandidate === "shaking_no") {
            actorMetrics.shaking_no_count += 1;
          }
          if (isPresidentOpportunityDecision(decision)) {
            actorMetrics.president_opportunity_count += 1;
            if (decision.chosenCandidate === "president_stop") actorMetrics.president_stop_count += 1;
            else if (decision.chosenCandidate === "president_hold") actorMetrics.president_hold_count += 1;
          }
          if (isGukjinOpportunityDecision(decision)) {
            actorMetrics.gukjin_opportunity_count += 1;
            if (decision.chosenCandidate === "five") actorMetrics.gukjin_five_count += 1;
            else if (decision.chosenCandidate === "junk") actorMetrics.gukjin_junk_count += 1;
          }
        }
        if (isGoStopOpportunityDecision(decision)) {
          if (decision.actor === "human" || decision.actor === "ai") {
            roundGoOpportunity[decision.actor] += 1;
          }
        }
        if (!datasetWriter) return;
        if (opts.datasetActor !== "all" && decision.actor !== opts.datasetActor) return;
        if (opts.datasetDecisionTypes && !opts.datasetDecisionTypes.has(decision.decisionType)) return;

        let candidates = decision.candidates;
        if (decision.decisionType === "option" && opts.datasetOptionCandidates) {
          candidates = candidates.filter((candidate) =>
            opts.datasetOptionCandidates.has(normalizeDecisionCandidate("option", candidate))
          );
        }
        const legalCount = candidates.length;
        if (legalCount <= 0) return;
        datasetStats.decisions += 1;
        unresolvedStats.decisions += 1;
        const normalizedCandidates = candidates.map((candidate) =>
          normalizeDecisionCandidate(decision.decisionType, candidate)
        );
        const decisionId = buildDecisionId(gi, decision);
        const isGoStopDecision =
          decision.decisionType === "option" &&
          String(decision?.stateBefore?.phase || "") === "go-stop";
        const goStopSnapshot = isGoStopDecision
          ? buildGoStopPublicSnapshot(decision.stateBefore, decision.actor, normalizedCandidates)
          : null;
        const matched = normalizedCandidates.some(
          (candidateNorm) => candidateNorm === decision.chosenCandidate
        );
        const unresolvedFlag = matched ? 0 : 1;
        for (const candidate of candidates) {
          const candidateNorm = normalizeDecisionCandidate(decision.decisionType, candidate);
          const isChosen = candidateNorm === decision.chosenCandidate ? 1 : 0;
          const row = {
            game_index: gi,
            seed,
            first_turn: firstTurnKey,
            step: decision.step,
            actor: decision.actor,
            actor_policy: decision.policy,
            action_source: decision.actionSource,
            decision_type: decision.decisionType,
            legal_count: legalCount,
            decision_id: decisionId,
            candidate: candidateNorm,
            chosen: isChosen,
            chosen_candidate: decision.chosenCandidate,
            unresolved: unresolvedFlag,
            features: featureVectorForCandidate(
              decision.stateBefore,
              decision.actor,
              decision.decisionType,
              candidate
            ),
          };
          if (goStopSnapshot) {
            row.public_snapshot = goStopSnapshot;
            row.state_before_full = decision.stateBefore;
          }
          datasetWriter.write(`${JSON.stringify(row)}\n`);
          datasetStats.rows += 1;
          if (isChosen) datasetStats.positive_rows += 1;
        }
        if (!matched) {
          datasetStats.unresolved_decisions += 1;
          unresolvedStats.unresolved_decisions += 1;
          incrementCounter(unresolvedStats.by_actor, decision.actor);
          incrementCounter(unresolvedStats.by_policy, decision.policy);
          incrementCounter(unresolvedStats.by_decision_type, decision.decisionType);
          incrementCounter(unresolvedStats.by_action_source, decision.actionSource);
        }
      }
    );
    const actorAMongBak = Boolean(endState?.result?.[actorA]?.bak?.mongBak);
    const actorBMongBak = Boolean(endState?.result?.[actorB]?.bak?.mongBak);
    if (roundSpecial[actorA].gukjin_five_count > 0 && actorAMongBak) {
      roundSpecial[actorA].gukjin_five_mongbak_count = 1;
    }
    if (roundSpecial[actorA].gukjin_junk_count > 0 && actorAMongBak) {
      roundSpecial[actorA].gukjin_junk_mongbak_count = 1;
    }
    if (roundSpecial[actorB].gukjin_five_count > 0 && actorBMongBak) {
      roundSpecial[actorB].gukjin_five_mongbak_count = 1;
    }
    if (roundSpecial[actorB].gukjin_junk_count > 0 && actorBMongBak) {
      roundSpecial[actorB].gukjin_junk_mongbak_count = 1;
    }
    const afterDiffA = goldDiffByActor(endState, actorA);
    goldDeltasA.push(afterDiffA - beforeDiffA);

    const goldA = Number(endState?.players?.[actorA]?.gold || 0);
    const goldB = Number(endState?.players?.[actorB]?.gold || 0);
    const goCountA = Math.max(0, Number(endState?.players?.[actorA]?.goCount || 0));
    const goCountB = Math.max(0, Number(endState?.players?.[actorB]?.goCount || 0));
    const goOpportunityCountA = Math.max(0, Number(roundGoOpportunity[actorA] || 0));
    const goOpportunityCountB = Math.max(0, Number(roundGoOpportunity[actorB] || 0));
    const roundMetricsA = {
      ...roundSpecial[actorA],
      go_count: goCountA,
      go_opportunity_count: goOpportunityCountA,
    };
    const roundMetricsB = {
      ...roundSpecial[actorB],
      go_count: goCountB,
      go_opportunity_count: goOpportunityCountB,
    };
    if (goldA <= 0) bankrupt.a_bankrupt_count += 1;
    if (goldB <= 0) bankrupt.b_bankrupt_count += 1;

    if (opts.continuousSeries) {
      seriesSession.previousEndState = endState;
    }
    seriesSession.roundsPlayed += 1;

    const winner = String(endState?.result?.winner || "").trim();
    if (winner === actorA) winsA += 1;
    else if (winner === actorB) winsB += 1;
    else draws += 1;

    if (kiboWriter) {
      const kiboRecord = {
        game_index: gi,
        seed,
        first_turn: firstTurnKey,
        human: humanPlayer.label,
        ai: aiPlayer.label,
        winner,
        result: endState?.result || null,
        kibo_detail: endState?.kiboDetail || effectiveKiboDetail,
        kibo: Array.isArray(endState?.kibo) ? endState.kibo : [],
      };
      kiboWriter.write(`${JSON.stringify(kiboRecord)}\n`);
    }

    const seatAKey = firstTurnKey === actorA ? "first" : "second";
    const seatBKey = firstTurnKey === actorB ? "first" : "second";
    const goldDeltaA = afterDiffA - beforeDiffA;
    updateSeatRecord(
      seatSplitA[seatAKey],
      winner,
      actorA,
      actorB,
      goldDeltaA,
      roundMetricsA
    );
    updateSeatRecord(
      seatSplitB[seatBKey],
      winner,
      actorB,
      actorA,
      -goldDeltaA,
      roundMetricsB
    );
  }

  const games = opts.games;
  const winRateA = winsA / games;
  const winRateB = winsB / games;
  const drawRate = draws / games;
  const meanGoldDeltaA =
    goldDeltasA.length > 0 ? goldDeltasA.reduce((a, b) => a + b, 0) / goldDeltasA.length : 0;
  const lossesA = winsB;
  const lossesB = winsA;
  const splitSummaryA = buildSeatSplitSummary(seatSplitA.first, seatSplitA.second);
  const splitSummaryB = buildSeatSplitSummary(seatSplitB.first, seatSplitB.second);

  const summary = {
    games,
    actor_human: actorA,
    actor_ai: actorB,
    human: humanPlayer.label,
    ai: aiPlayer.label,
    player_human: {
      input: humanPlayer.input,
      kind: humanPlayer.kind,
      key: humanPlayer.key,
      model_path: humanPlayer.modelPath,
      go_stop_iqn_path: humanPlayer.goStopIqnPath || null,
    },
    player_ai: {
      input: aiPlayer.input,
      kind: aiPlayer.kind,
      key: aiPlayer.key,
      model_path: aiPlayer.modelPath,
      go_stop_iqn_path: aiPlayer.goStopIqnPath || null,
    },
    first_turn_policy: opts.firstTurnPolicy,
    fixed_first_turn: opts.firstTurnPolicy === "fixed" ? opts.fixedFirstTurn : null,
    first_turn_counts: firstTurnCounts,
    continuous_series: !!opts.continuousSeries,
    kibo_detail: opts.kiboDetail,
    result_out: toReportPath(opts.resultOut),
    kibo_out: toReportPath(opts.kiboOut),
    dataset_out: toReportPath(opts.datasetOut),
    dataset_feature_profile: ACTIVE_FEATURE_PROFILE,
    dataset_actor: opts.datasetActor,
    dataset_decision_types: setToReportList(opts.datasetDecisionTypes),
    dataset_option_candidates: setToReportList(opts.datasetOptionCandidates),
    dataset_rows: datasetStats.rows,
    dataset_positive_rows: datasetStats.positive_rows,
    dataset_decisions: datasetStats.decisions,
    dataset_unresolved_decisions: datasetStats.unresolved_decisions,
    unresolved_out: null,
    unresolved_limit: null,
    unresolved_rows: unresolvedStats.unresolved_decisions,
    unresolved_decisions: unresolvedStats.unresolved_decisions,
    unresolved_decision_rate:
      unresolvedStats.decisions > 0 ? unresolvedStats.unresolved_decisions / unresolvedStats.decisions : 0,
    unresolved_by_actor: unresolvedStats.by_actor,
    unresolved_by_policy: unresolvedStats.by_policy,
    unresolved_by_decision_type: unresolvedStats.by_decision_type,
    unresolved_by_action_source: unresolvedStats.by_action_source,
    bankrupt,
    session_rounds: {
      total_rounds: seriesSession.roundsPlayed,
    },
    wins_a: winsA,
    losses_a: lossesA,
    wins_b: winsB,
    losses_b: lossesB,
    draws,
    win_rate_a: winRateA,
    win_rate_b: winRateB,
    draw_rate: drawRate,
    go_count_a: splitSummaryA.combined.go_count_total,
    go_count_b: splitSummaryB.combined.go_count_total,
    go_games_a: splitSummaryA.combined.go_game_count,
    go_games_b: splitSummaryB.combined.go_game_count,
    go_fail_count_a: splitSummaryA.combined.go_fail_count,
    go_fail_count_b: splitSummaryB.combined.go_fail_count,
    go_fail_rate_a: splitSummaryA.combined.go_fail_rate,
    go_fail_rate_b: splitSummaryB.combined.go_fail_rate,
    go_opportunity_count_a: splitSummaryA.combined.go_opportunity_count_total,
    go_opportunity_count_b: splitSummaryB.combined.go_opportunity_count_total,
    go_opportunity_games_a: splitSummaryA.combined.go_opportunity_game_count,
    go_opportunity_games_b: splitSummaryB.combined.go_opportunity_game_count,
    go_opportunity_rate_a: splitSummaryA.combined.go_opportunity_rate,
    go_opportunity_rate_b: splitSummaryB.combined.go_opportunity_rate,
    go_take_rate_a: splitSummaryA.combined.go_take_rate,
    go_take_rate_b: splitSummaryB.combined.go_take_rate,
    shaking_count_a: splitSummaryA.combined.shaking_count_total,
    shaking_count_b: splitSummaryB.combined.shaking_count_total,
    shaking_games_a: splitSummaryA.combined.shaking_game_count,
    shaking_games_b: splitSummaryB.combined.shaking_game_count,
    shaking_win_a: splitSummaryA.combined.shaking_win_game_count,
    shaking_win_b: splitSummaryB.combined.shaking_win_game_count,
    shaking_opportunity_count_a: splitSummaryA.combined.shaking_opportunity_count_total,
    shaking_opportunity_count_b: splitSummaryB.combined.shaking_opportunity_count_total,
    shaking_opportunity_games_a: splitSummaryA.combined.shaking_opportunity_game_count,
    shaking_opportunity_games_b: splitSummaryB.combined.shaking_opportunity_game_count,
    shaking_take_rate_a: splitSummaryA.combined.shaking_take_rate,
    shaking_take_rate_b: splitSummaryB.combined.shaking_take_rate,
    shaking_plain_play_a: splitSummaryA.combined.shaking_plain_play_total,
    shaking_plain_play_b: splitSummaryB.combined.shaking_plain_play_total,
    shaking_skip_other_play_a: splitSummaryA.combined.shaking_skip_other_play_total,
    shaking_skip_other_play_b: splitSummaryB.combined.shaking_skip_other_play_total,
    shaking_no_a: splitSummaryA.combined.shaking_no_total,
    shaking_no_b: splitSummaryB.combined.shaking_no_total,
    bomb_count_a: splitSummaryA.combined.bomb_count_total,
    bomb_count_b: splitSummaryB.combined.bomb_count_total,
    bomb_games_a: splitSummaryA.combined.bomb_game_count,
    bomb_games_b: splitSummaryB.combined.bomb_game_count,
    bomb_win_a: splitSummaryA.combined.bomb_win_game_count,
    bomb_win_b: splitSummaryB.combined.bomb_win_game_count,
    bomb_opportunity_count_a: splitSummaryA.combined.bomb_opportunity_count_total,
    bomb_opportunity_count_b: splitSummaryB.combined.bomb_opportunity_count_total,
    bomb_opportunity_games_a: splitSummaryA.combined.bomb_opportunity_game_count,
    bomb_opportunity_games_b: splitSummaryB.combined.bomb_opportunity_game_count,
    bomb_take_rate_a: splitSummaryA.combined.bomb_take_rate,
    bomb_take_rate_b: splitSummaryB.combined.bomb_take_rate,
    bomb_plain_play_a: splitSummaryA.combined.bomb_plain_play_total,
    bomb_plain_play_b: splitSummaryB.combined.bomb_plain_play_total,
    bomb_skip_other_play_a: splitSummaryA.combined.bomb_skip_other_play_total,
    bomb_skip_other_play_b: splitSummaryB.combined.bomb_skip_other_play_total,
    president_stop_a: splitSummaryA.combined.president_stop_total,
    president_stop_b: splitSummaryB.combined.president_stop_total,
    president_hold_a: splitSummaryA.combined.president_hold_total,
    president_hold_b: splitSummaryB.combined.president_hold_total,
    president_stop_win_a: splitSummaryA.combined.president_stop_win_total,
    president_stop_win_b: splitSummaryB.combined.president_stop_win_total,
    president_stop_loss_a: splitSummaryA.combined.president_stop_loss_total,
    president_stop_loss_b: splitSummaryB.combined.president_stop_loss_total,
    president_hold_win_a: splitSummaryA.combined.president_hold_win_total,
    president_hold_win_b: splitSummaryB.combined.president_hold_win_total,
    president_hold_loss_a: splitSummaryA.combined.president_hold_loss_total,
    president_hold_loss_b: splitSummaryB.combined.president_hold_loss_total,
    president_opportunity_count_a: splitSummaryA.combined.president_opportunity_count_total,
    president_opportunity_count_b: splitSummaryB.combined.president_opportunity_count_total,
    president_stop_rate_a: splitSummaryA.combined.president_stop_rate,
    president_stop_rate_b: splitSummaryB.combined.president_stop_rate,
    president_stop_win_rate_a: splitSummaryA.combined.president_stop_win_rate,
    president_stop_win_rate_b: splitSummaryB.combined.president_stop_win_rate,
    president_hold_win_rate_a: splitSummaryA.combined.president_hold_win_rate,
    president_hold_win_rate_b: splitSummaryB.combined.president_hold_win_rate,
    gukjin_five_a: splitSummaryA.combined.gukjin_five_total,
    gukjin_five_b: splitSummaryB.combined.gukjin_five_total,
    gukjin_five_mongbak_a: splitSummaryA.combined.gukjin_five_mongbak_total,
    gukjin_five_mongbak_b: splitSummaryB.combined.gukjin_five_mongbak_total,
    gukjin_junk_a: splitSummaryA.combined.gukjin_junk_total,
    gukjin_junk_b: splitSummaryB.combined.gukjin_junk_total,
    gukjin_junk_mongbak_a: splitSummaryA.combined.gukjin_junk_mongbak_total,
    gukjin_junk_mongbak_b: splitSummaryB.combined.gukjin_junk_mongbak_total,
    gukjin_opportunity_count_a: splitSummaryA.combined.gukjin_opportunity_count_total,
    gukjin_opportunity_count_b: splitSummaryB.combined.gukjin_opportunity_count_total,
    gukjin_five_rate_a: splitSummaryA.combined.gukjin_five_rate,
    gukjin_five_rate_b: splitSummaryB.combined.gukjin_five_rate,
    gukjin_five_mongbak_rate_a: splitSummaryA.combined.gukjin_five_mongbak_rate,
    gukjin_five_mongbak_rate_b: splitSummaryB.combined.gukjin_five_mongbak_rate,
    gukjin_junk_mongbak_rate_a: splitSummaryA.combined.gukjin_junk_mongbak_rate,
    gukjin_junk_mongbak_rate_b: splitSummaryB.combined.gukjin_junk_mongbak_rate,
    mean_gold_delta_a: meanGoldDeltaA,
    p10_gold_delta_a: quantile(goldDeltasA, 0.1),
    p50_gold_delta_a: quantile(goldDeltasA, 0.5),
    p90_gold_delta_a: quantile(goldDeltasA, 0.9),
    seat_split_a: splitSummaryA,
    seat_split_b: splitSummaryB,
    eval_time_ms: Math.max(0, Date.now() - evalStartMs),
  };

  if (kiboWriter) kiboWriter.end();
  if (datasetWriter) datasetWriter.end();

  const reportLine = `${JSON.stringify(summary)}\n`;
  const consoleSummary = buildConsoleSummary(summary);
  mkdirSync(dirname(opts.resultOut), { recursive: true });
  writeFileSync(opts.resultOut, reportLine, { encoding: "utf8" });
  if (opts.stdoutFormat === "json") {
    process.stdout.write(reportLine);
  } else {
    process.stdout.write(formatConsoleSummaryText(consoleSummary));
  }
}

try {
  runModelDuelCli(process.argv.slice(2));
} catch (err) {
  const msg = err && err.stack ? err.stack : String(err);
  process.stderr.write(`${msg}\n`);
  process.exit(1);
}
