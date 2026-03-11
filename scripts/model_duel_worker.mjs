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
import { aiPlay } from "../src/ai/aiPlay.js";
import { resolveBotPolicy } from "../src/ai/policies.js";
import { hybridPolicyPlayDetailed } from "../src/ai/hybridPolicyEngine.js";

// Pipeline Stage: 3/3 (neat_train.py -> neat_eval_worker.mjs -> model_duel_worker.mjs)
// Execution Flow Map:
// 1) main()
// 2) playSingleRound(): duel loop + decision callback
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
  if (Math.floor(out.games) < 1000) {
    throw new Error("this worker requires --games >= 1000");
  }
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
  const token = String(rawSpec || "").trim();
  if (!token) throw new Error(`empty player spec: ${sideLabel}`);

  const hybridOption = parseHybridOptionSpec(token);
  if (hybridOption) {
    const playMatchModelSpec = resolvePhaseModelSpec(hybridOption.playMatchModelToken, `${sideLabel}:play_match_model`);
    const optionModelSpec = resolvePhaseModelSpec(hybridOption.optionModelToken, `${sideLabel}:option_model`);
    return {
      input: token,
      kind: "hybrid_option_model",
      key: `hybrid_option(${playMatchModelSpec.key},${optionModelSpec.key})`,
      label: `hybrid_option(${playMatchModelSpec.label},${optionModelSpec.label})`,
      playMatchModel: playMatchModelSpec.model,
      playMatchModelPath: playMatchModelSpec.modelPath,
      playMatchPhase: playMatchModelSpec.phase,
      playMatchSeed: playMatchModelSpec.seed,
      optionModel: optionModelSpec.model,
      optionModelPath: optionModelSpec.modelPath,
      optionPhase: optionModelSpec.phase,
      optionSeed: optionModelSpec.seed,
    };
  }

  const hybridGo = parseHybridPlayGoSpec(token);
  if (hybridGo) {
    const modelSpec = resolvePhaseModelSpec(hybridGo.modelToken, `${sideLabel}:model`);
    const goStopPolicy = resolveBotPolicy(hybridGo.goStopToken);
    if (!goStopPolicy) {
      throw new Error(
        `invalid ${sideLabel} hybrid go-stop policy: ${hybridGo.goStopToken} (use a policy key from src/ai/policies.js)`
      );
    }
    const heuristicToken = String(hybridGo.heuristicToken || "").trim();
    const heuristicPolicy = heuristicToken ? resolveBotPolicy(heuristicToken) : "";
    if (heuristicToken && !heuristicPolicy) {
      throw new Error(
        `invalid ${sideLabel} hybrid heuristic policy: ${hybridGo.heuristicToken} (use a policy key from src/ai/policies.js)`
      );
    }
    const goStopOnly = !heuristicPolicy;
    return {
      input: token,
      kind: "hybrid_play_model",
      key: goStopOnly
        ? `hybrid_play_go(${modelSpec.key},${goStopPolicy})`
        : `hybrid_play_go(${modelSpec.key},${goStopPolicy},${heuristicPolicy})`,
      label: goStopOnly
        ? `hybrid_play_go(${modelSpec.label},${goStopPolicy})`
        : `hybrid_play_go(${modelSpec.label},${goStopPolicy},${heuristicPolicy})`,
      model: modelSpec.model,
      modelPath: modelSpec.modelPath,
      phase: modelSpec.phase,
      seed: modelSpec.seed,
      heuristicPolicy,
      goStopPolicy,
      goStopOnly,
    };
  }

  const hybrid = parseHybridPlaySpec(token);
  if (hybrid) {
    const modelSpec = resolvePhaseModelSpec(hybrid.modelToken, `${sideLabel}:model`);
    const heuristicPolicy = resolveBotPolicy(hybrid.heuristicToken);
    if (!heuristicPolicy) {
      throw new Error(
        `invalid ${sideLabel} hybrid heuristic policy: ${hybrid.heuristicToken} (use a policy key from src/ai/policies.js)`
      );
    }
    return {
      input: token,
      kind: "hybrid_play_model",
      key: `hybrid_play(${modelSpec.key},${heuristicPolicy})`,
      label: `hybrid_play(${modelSpec.label},${heuristicPolicy})`,
      model: modelSpec.model,
      modelPath: modelSpec.modelPath,
      phase: modelSpec.phase,
      seed: modelSpec.seed,
      heuristicPolicy,
    };
  }

  const resolvedPolicy = resolveBotPolicy(token);
  if (resolvedPolicy) {
    return {
      input: token,
      kind: "heuristic",
      key: resolvedPolicy,
      label: resolvedPolicy,
      model: null,
      modelPath: null,
      phase: null,
      seed: null,
    };
  }

  return resolvePhaseModelSpec(token, sideLabel);
}

function parseHybridPlayGoSpec(token) {
  const m = String(token || "")
    .trim()
    .match(/^hybrid_play_go\(\s*([^,]+)\s*,\s*([^,]+?)(?:\s*,\s*([^)]+)\s*)?\)$/i);
  if (!m) {
    return null;
  }
  return {
    modelToken: String(m[1] || "").trim(),
    goStopToken: String(m[2] || "").trim(),
    heuristicToken: String(m[3] || "").trim(),
  };
}

function parseHybridOptionSpec(token) {
  const m = String(token || "")
    .trim()
    .match(/^hybrid_option\(\s*([^,]+)\s*,\s*([^)]+)\s*\)$/i);
  if (!m) {
    return null;
  }
  return {
    playMatchModelToken: String(m[1] || "").trim(),
    optionModelToken: String(m[2] || "").trim(),
  };
}

function parseHybridPlaySpec(token) {
  const m = String(token || "")
    .trim()
    .match(/^hybrid_play\(\s*([^,]+)\s*,\s*([^)]+)\s*\)$/i);
  if (!m) {
    return null;
  }
  return {
    modelToken: String(m[1] || "").trim(),
    heuristicToken: String(m[2] || "").trim(),
  };
}

function parsePhaseModelToken(rawToken) {
  const m = String(rawToken || "").trim().match(/^(?:(pareto52)_)?phase([0-3])_seed(\d+)$/i);
  if (!m) return null;
  const profile = m[1] ? "pareto52" : "classic";
  const phase = Number(m[2]);
  const seed = Number(m[3]);
  const outputPrefix = profile === "pareto52" ? "neat_pareto52" : "neat";
  const tokenKey = profile === "pareto52" ? `pareto52_phase${phase}_seed${seed}` : `phase${phase}_seed${seed}`;
  return { profile, phase, seed, outputPrefix, tokenKey };
}

function resolvePhaseModelSpec(token, sideLabel) {
  const parsedToken = parsePhaseModelToken(token);
  if (!parsedToken) {
    throw new Error(
      `invalid ${sideLabel} spec: ${token} (use policy key, phase0_seed9, pareto52_phase1_seed601, hybrid_play(phase1_seed202,H-CL), hybrid_play_go(phase1_seed202,H-CL), hybrid_option(phase1_seed208,phase2_seed501), or hybrid_play_go(phase1_seed202,H-NEXg,H-CL))`
    );
  }
  const { phase, seed, outputPrefix, tokenKey } = parsedToken;
  const summaryPath = resolve(`logs/NEAT/${outputPrefix}_phase${phase}_seed${seed}/run_summary.json`);
  if (existsSync(summaryPath)) {
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
  const modelPath = resolve(`logs/NEAT/${outputPrefix}_phase${phase}_seed${seed}/models/winner_genome.json`);
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

function resolveGoStopIqnRuntime(rawPath, sideLabel) {
  const fullPath = resolve(String(rawPath || "").trim());
  if (!existsSync(fullPath)) {
    throw new Error(`go/stop IQN runtime not found for ${sideLabel}: ${fullPath}`);
  }

  let model = null;
  try {
    const raw = String(readFileSync(fullPath, "utf8") || "").replace(/^\uFEFF/, "");
    model = JSON.parse(raw);
  } catch (err) {
    throw new Error(`failed to parse go/stop IQN runtime (${sideLabel}): ${fullPath} (${String(err)})`);
  }
  if (String(model?.format_version || "").trim() !== "iqn_go_stop_runtime_v1") {
    throw new Error(`invalid go/stop IQN runtime format for ${sideLabel}: expected iqn_go_stop_runtime_v1`);
  }

  return {
    model,
    modelPath: fullPath,
  };
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
function canonicalOptionAction(action) {
  const a = String(action || "").trim();
  if (!a) return "";
  const aliases = {
    choose_go: "go",
    choose_stop: "stop",
    choose_shaking_yes: "shaking_yes",
    choose_shaking_no: "shaking_no",
    choose_president_stop: "president_stop",
    choose_president_hold: "president_hold",
    choose_five: "five",
    choose_junk: "junk",
  };
  return aliases[a] || a;
}

function normalizeOptionCandidates(items) {
  if (!Array.isArray(items)) return [];
  const out = [];
  const seen = new Set();
  for (const raw of items) {
    const v = canonicalOptionAction(raw);
    if (!v || seen.has(v)) continue;
    seen.add(v);
    out.push(v);
  }
  return out;
}

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

const PLAY_SPECIAL_SHAKE_PREFIX = "shake_start:";
const PLAY_SPECIAL_BOMB_PREFIX = "bomb:";

function parsePlaySpecialCandidate(candidate) {
  const raw = String(candidate || "").trim();
  if (!raw) return null;
  if (raw.startsWith(PLAY_SPECIAL_SHAKE_PREFIX)) {
    const cardId = raw.slice(PLAY_SPECIAL_SHAKE_PREFIX.length).trim();
    if (!cardId) return null;
    return { kind: "shake_start", cardId };
  }
  if (raw.startsWith(PLAY_SPECIAL_BOMB_PREFIX)) {
    const month = Number(raw.slice(PLAY_SPECIAL_BOMB_PREFIX.length).trim());
    if (!Number.isInteger(month) || month < 1 || month > 12) return null;
    return { kind: "bomb", month };
  }
  return null;
}

function buildPlayingSpecialActions(state, actor) {
  if (state?.phase !== "playing" || state?.currentTurn !== actor) return [];
  const out = [];
  const shakingMonths = new Set(getDeclarableShakingMonths(state, actor));
  if (shakingMonths.size > 0) {
    for (const card of state?.players?.[actor]?.hand || []) {
      const cardId = String(card?.id || "").trim();
      if (!cardId || card?.passCard) continue;
      const month = Number(card?.month || 0);
      if (shakingMonths.has(month)) out.push(`${PLAY_SPECIAL_SHAKE_PREFIX}${cardId}`);
    }
  }
  for (const month of getDeclarableBombMonths(state, actor)) {
    const monthNum = Number(month || 0);
    if (monthNum >= 1 && monthNum <= 12) out.push(`${PLAY_SPECIAL_BOMB_PREFIX}${monthNum}`);
  }
  return out;
}

function applyPlayCandidate(state, actor, candidate) {
  const special = parsePlaySpecialCandidate(candidate);
  if (!special) return playTurn(state, candidate);
  if (special.kind === "shake_start") {
    const selected = findCardById(state?.players?.[actor]?.hand || [], special.cardId);
    const month = Number(selected?.month || 0);
    if (month < 1) return state;
    const declared = declareShaking(state, actor, month);
    if (!declared || declared === state) return state;
    return playTurn(declared, special.cardId) || declared;
  }
  if (special.kind === "bomb") return declareBomb(state, actor, special.month);
  return state;
}

function selectPool(state, actor) {
  if (state.phase === "playing" && state.currentTurn === actor) {
    return {
      cards: (state.players?.[actor]?.hand || []).map((c) => c.id),
      specialActions: buildPlayingSpecialActions(state, actor),
    };
  }
  if (state.phase === "select-match" && state.pendingMatch?.playerKey === actor) {
    return { boardCardIds: state.pendingMatch.boardCardIds || [] };
  }
  if (state.phase === "go-stop" && state.pendingGoStop === actor) {
    return { options: ["go", "stop"] };
  }
  if (state.phase === "president-choice" && state.pendingPresident?.playerKey === actor) {
    return { options: ["president_stop", "president_hold"] };
  }
  if (state.phase === "gukjin-choice" && state.pendingGukjinChoice?.playerKey === actor) {
    return { options: ["five", "junk"] };
  }
  if (state.phase === "shaking-confirm" && state.pendingShakingConfirm?.playerKey === actor) {
    return { options: ["shaking_yes", "shaking_no"] };
  }
  return {};
}

function legalCandidatesForDecision(sp, decisionType) {
  if (decisionType === "play") {
    return [
      ...(sp.cards || []).map((x) => String(x)).filter((x) => x.length > 0),
      ...(sp.specialActions || []).map((x) => String(x)).filter((x) => x.length > 0),
    ];
  }
  if (decisionType === "match") {
    return (sp.boardCardIds || []).map((x) => String(x)).filter((x) => x.length > 0);
  }
  if (decisionType === "option") {
    return normalizeOptionCandidates(sp.options || []);
  }
  return [];
}

function applyAction(state, actor, decisionType, rawAction) {
  let action = String(rawAction || "").trim();
  if (!action) return state;
  if (decisionType === "play") return applyPlayCandidate(state, actor, action);
  if (decisionType === "match") return chooseMatch(state, action);
  if (decisionType !== "option") return state;

  action = canonicalOptionAction(action);
  if (action === "go") return chooseGo(state, actor);
  if (action === "stop") return chooseStop(state, actor);
  if (action === "shaking_yes") return chooseShakingYes(state, actor);
  if (action === "shaking_no") return chooseShakingNo(state, actor);
  if (action === "president_stop") return choosePresidentStop(state, actor);
  if (action === "president_hold") return choosePresidentHold(state, actor);
  if (action === "five" || action === "junk") return chooseGukjinMode(state, actor, action);
  return state;
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

function stateProgressKey(state) {
  if (!state) return "null";
  const hh = Number(state?.players?.human?.hand?.length || 0);
  const ah = Number(state?.players?.ai?.hand?.length || 0);
  const d = Number(state?.deck?.length || 0);
  return [
    String(state.phase || ""),
    String(state.currentTurn || ""),
    String(state.pendingGoStop || ""),
    String(state.pendingMatch?.stage || ""),
    String(state.pendingPresident?.playerKey || ""),
    String(state.pendingShakingConfirm?.playerKey || ""),
    String(state.pendingShakingConfirm?.cardId || ""),
    String(state.pendingShakingConfirm?.month || ""),
    String(state.pendingGukjinChoice?.playerKey || ""),
    String(state.turnSeq || 0),
    String(state.kiboSeq || 0),
    String(hh),
    String(ah),
    String(d),
  ].join("|");
}

function clamp01(x) {
  const v = Number(x || 0);
  if (v <= 0) return 0;
  if (v >= 1) return 1;
  return v;
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

function candidateComboGain(state, actor, decisionType, candidate) {
  const beforePlayer = state?.players?.[actor];
  if (!beforePlayer) return 0;

  const visibleState = maskStateForVisibleComboSimulation(state);
  const afterState = applyAction(visibleState, actor, decisionType, candidate);
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
  return clamp01((currentMultiplier - 1.0) / 15.0);
}

function buildBaseFeatureVectorForCandidate(state, actor, decisionType, candidate, legalCount) {
  const opp = actor === "human" ? "ai" : "human";
  const scoreSelf = calculateScore(state.players[actor], state.players[opp], state.ruleKey);
  const scoreOpp = calculateScore(state.players[opp], state.players[actor], state.ruleKey);
  const card = candidateCard(state, actor, decisionType, candidate);
  const month = resolveCandidateMonth(state, actor, decisionType, card);
  const selfCanStop = Number(scoreSelf?.total || 0) >= 7 ? 1 : 0;
  const oppCanStop = Number(scoreOpp?.total || 0) >= 7 ? 1 : 0;

  return [
    decisionType === "play" ? 1 : 0,
    decisionType === "match" ? 1 : 0,
    optionCode(candidate),
    tanhNorm((scoreSelf?.total || 0) - (scoreOpp?.total || 0), 10.0),
    tanhNorm(scoreSelf?.total || 0, 10.0),
    clamp01((scoreOpp?.total || 0) / 7.0),
    clamp01(currentMultiplierNorm(state, scoreSelf)),
    candidateComboGain(state, actor, decisionType, candidate),
    clamp01(Number(card?.piValue || 0) / 5.0),
    immediateMatchPossible(state, decisionType, month),
    candidatePublicKnownRatio(state, actor, month),
    selfCanStop,
    oppCanStop
  ];
}

function featureVectorForCandidate(state, actor, decisionType, candidate, legalCount) {
  return buildBaseFeatureVectorForCandidate(state, actor, decisionType, candidate, legalCount);
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
function normalizeDecisionCandidate(decisionType, candidate) {
  if (decisionType === "option") return canonicalOptionAction(candidate);
  return String(candidate || "").trim();
}

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
      const shakingCandidate = `${PLAY_SPECIAL_SHAKE_PREFIX}${playedId}`;
      if (playedId && shakingDeclared && candidateSet.has(shakingCandidate)) return shakingCandidate;
      if (playedId && candidateSet.has(playedId)) return playedId;
    }
    if (decisionType === "play" && actionType === "declare_bomb") {
      const bombMonth = Number(ev?.action?.month || 0);
      const bombCandidate = `${PLAY_SPECIAL_BOMB_PREFIX}${bombMonth}`;
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

function randomChoice(arr, rng) {
  if (!arr.length) return null;
  const idx = Math.max(0, Math.min(arr.length - 1, Math.floor(Number(rng() || 0) * arr.length)));
  return arr[idx];
}

function randomLegalAction(state, actor, rng) {
  const sp = selectPool(state, actor);
  const cards = sp.cards || null;
  const boardCardIds = sp.boardCardIds || null;
  const options = sp.options || null;
  const decisionType = cards ? "play" : boardCardIds ? "match" : options ? "option" : null;
  if (!decisionType) return state;
  const candidates = legalCandidatesForDecision(sp, decisionType);
  if (!candidates.length) return state;
  const picked = randomChoice(candidates, rng);
  return applyAction(state, actor, decisionType, picked);
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

function startRound(seed, firstTurnKey, kiboDetail) {
  return initSimulationGame("A", createSeededRng(`${seed}|game`), {
    kiboDetail,
    firstTurnKey,
  });
}

function continueRound(prevEndState, seed, firstTurnKey, kiboDetail) {
  return startSimulationGame(prevEndState, createSeededRng(`${seed}|game`), {
    kiboDetail,
    keepGold: true,
    useCarryOver: true,
    firstTurnKey,
  });
}

function goldDiffByActor(state, actor) {
  const opp = actor === "human" ? "ai" : "human";
  const selfGold = Number(state?.players?.[actor]?.gold || 0);
  const oppGold = Number(state?.players?.[opp]?.gold || 0);
  return selfGold - oppGold;
}

function buildAiPlayOptions(playerSpec) {
  if (playerSpec?.kind === "model" && playerSpec?.model) {
    return {
      source: "model",
      model: playerSpec.model,
      goStopIqnModel: playerSpec.goStopIqnModel || null,
    };
  }
  return { source: "heuristic", heuristicPolicy: String(playerSpec?.key || "") };
}

function resolvePlayerAction(state, actor, playerSpec) {
  if (playerSpec?.kind === "hybrid_option_model") {
    const sp = selectPool(state, actor);
    const cards = sp.cards || null;
    const boardCardIds = sp.boardCardIds || null;
    const options = sp.options || null;
    const decisionType = cards ? "play" : boardCardIds ? "match" : options ? "option" : null;

    if (decisionType === "play" || decisionType === "match") {
      let next = aiPlay(state, actor, {
        source: "model",
        model: playerSpec.playMatchModel,
        goStopIqnModel: playerSpec.goStopIqnModel || null,
      });
      let actionSource = "hybrid_option_play_match_model";
      if (!next || stateProgressKey(next) === stateProgressKey(state)) {
        next = aiPlay(state, actor, {
          source: "model",
          model: playerSpec.optionModel,
          goStopIqnModel: playerSpec.goStopIqnModel || null,
        });
        actionSource = "hybrid_option_play_match_fallback_option_model";
      }
      return {
        next: next || state,
        actionSource,
      };
    }

    return {
      next: aiPlay(state, actor, {
        source: "model",
        model: playerSpec.optionModel,
        goStopIqnModel: playerSpec.goStopIqnModel || null,
      }),
      actionSource: "hybrid_option_option_model",
    };
  }
  if (playerSpec?.kind === "hybrid_play_model") {
    const traced = hybridPolicyPlayDetailed(state, actor, {
      model: playerSpec.model,
      heuristicPolicy: String(playerSpec.heuristicPolicy || ""),
      goStopPolicy: String(playerSpec.goStopPolicy || ""),
      goStopOnly: !!playerSpec.goStopOnly,
      goStopIqnModel: playerSpec.goStopIqnModel || null,
    });
    return {
      next: traced?.next || state,
      actionSource: String(traced?.actionSource || "hybrid_play_model"),
    };
  }
  if (playerSpec?.kind === "model" && playerSpec?.model) {
    return {
      next: aiPlay(state, actor, buildAiPlayOptions(playerSpec)),
      actionSource: "model",
    };
  }
  return {
    next: aiPlay(state, actor, buildAiPlayOptions(playerSpec)),
    actionSource: "heuristic",
  };
}

function playSingleRound(initialState, seed, playerByActor, maxSteps, onDecision = null) {
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
    const action = resolvePlayerAction(state, actor, playerSpec);
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

function quantile(values, q) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.max(0, Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * q)));
  return sorted[idx];
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
    bomb_count_total: 0,
    bomb_game_count: 0,
    bomb_win_game_count: 0,
    bomb_opportunity_count_total: 0,
    bomb_opportunity_game_count: 0,
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
    bomb_count: 0,
    bomb_opportunity_count: 0,
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
  const bombCount = Math.max(0, Number(roundMetrics?.bomb_count || 0));
  const bombOpportunityCount = Math.max(0, Number(roundMetrics?.bomb_opportunity_count || 0));
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
  if (shakingCount > 0) record.shaking_game_count += 1;
  if (shakingCount > 0 && winner === selfActor) record.shaking_win_game_count += 1;
  if (shakingOpportunityCount > 0) record.shaking_opportunity_game_count += 1;
  record.bomb_count_total += bombCount;
  record.bomb_opportunity_count_total += bombOpportunityCount;
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
  const bombCountTotal = Number(record?.bomb_count_total || 0);
  const bombGameCount = Number(record?.bomb_game_count || 0);
  const bombWinGameCount = Number(record?.bomb_win_game_count || 0);
  const bombOpportunityCountTotal = Number(record?.bomb_opportunity_count_total || 0);
  const bombOpportunityGameCount = Number(record?.bomb_opportunity_game_count || 0);
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
    bomb_count_total: bombCountTotal,
    bomb_game_count: bombGameCount,
    bomb_win_game_count: bombWinGameCount,
    bomb_opportunity_count_total: bombOpportunityCountTotal,
    bomb_opportunity_game_count: bombOpportunityGameCount,
    bomb_opportunity_rate: games > 0 ? bombOpportunityGameCount / games : 0,
    bomb_take_rate: bombOpportunityCountTotal > 0 ? bombCountTotal / bombOpportunityCountTotal : 0,
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
  const fmtRate = (value) => {
    const n = Number(value || 0);
    return Number.isFinite(n) ? n.toFixed(3) : "0.000";
  };
  const aFirst = summary?.seat_split_a?.when_first || {};
  const aSecond = summary?.seat_split_a?.when_second || {};
  const bFirst = summary?.seat_split_b?.when_first || {};
  const bSecond = summary?.seat_split_b?.when_second || {};
  const bankrupt = summary?.bankrupt || { a_bankrupt_count: 0, b_bankrupt_count: 0 };
  const lines = [
    "",
    `=== Model Duel (${summary.human} vs ${summary.ai}, games=${summary.games}) ===`,
    `Win/Loss/Draw(A):  ${summary.wins_a} / ${summary.losses_a} / ${summary.draws}  (WR=${fmtRate(summary.win_rate_a)})`,
    `Win/Loss/Draw(B):  ${summary.wins_b} / ${summary.losses_b} / ${summary.draws}  (WR=${fmtRate(summary.win_rate_b)})`,
    `Seat A first:      WR=${fmtRate(aFirst.win_rate)}, mean_gold_delta=${aFirst.mean_gold_delta}`,
    `Seat A second:     WR=${fmtRate(aSecond.win_rate)}, mean_gold_delta=${aSecond.mean_gold_delta}`,
    `Seat B first:      WR=${fmtRate(bFirst.win_rate)}, mean_gold_delta=${bFirst.mean_gold_delta}`,
    `Seat B second:     WR=${fmtRate(bSecond.win_rate)}, mean_gold_delta=${bSecond.mean_gold_delta}`,
    `Gold delta(A):     mean=${summary.mean_gold_delta_a}, p10=${summary.p10_gold_delta_a}, p50=${summary.p50_gold_delta_a}, p90=${summary.p90_gold_delta_a}`,
    `GO A:              games=${summary.go_games_a}, count=${summary.go_count_a}, fail=${summary.go_fail_count_a}, fail_rate=${fmtRate(summary.go_fail_rate_a)}`,
    `GO B:              games=${summary.go_games_b}, count=${summary.go_count_b}, fail=${summary.go_fail_count_b}, fail_rate=${fmtRate(summary.go_fail_rate_b)}`,
    `GO Opp A:          opp_games=${summary.go_opportunity_games_a}, opp_turns=${summary.go_opportunity_count_a}, opp_rate=${fmtRate(summary.go_opportunity_rate_a)}, take_rate=${fmtRate(summary.go_take_rate_a)}`,
    `GO Opp B:          opp_games=${summary.go_opportunity_games_b}, opp_turns=${summary.go_opportunity_count_b}, opp_rate=${fmtRate(summary.go_opportunity_rate_b)}, take_rate=${fmtRate(summary.go_take_rate_b)}`,
    `Shake A:           opp_games=${summary.shaking_opportunity_games_a}, opp_unique=${summary.shaking_opportunity_count_a}, games=${summary.shaking_games_a}, count=${summary.shaking_count_a}, win=${summary.shaking_win_a}, take_rate=${fmtRate(summary.shaking_take_rate_a)}`,
    `Shake B:           opp_games=${summary.shaking_opportunity_games_b}, opp_unique=${summary.shaking_opportunity_count_b}, games=${summary.shaking_games_b}, count=${summary.shaking_count_b}, win=${summary.shaking_win_b}, take_rate=${fmtRate(summary.shaking_take_rate_b)}`,
    `Bomb A:            opp_games=${summary.bomb_opportunity_games_a}, opp_unique=${summary.bomb_opportunity_count_a}, games=${summary.bomb_games_a}, count=${summary.bomb_count_a}, win=${summary.bomb_win_a}, take_rate=${fmtRate(summary.bomb_take_rate_a)}`,
    `Bomb B:            opp_games=${summary.bomb_opportunity_games_b}, opp_unique=${summary.bomb_opportunity_count_b}, games=${summary.bomb_games_b}, count=${summary.bomb_count_b}, win=${summary.bomb_win_b}, take_rate=${fmtRate(summary.bomb_take_rate_b)}`,
    `President A:       opp_total=${summary.president_opportunity_count_a}, hold=${summary.president_hold_a}, hold_win=${summary.president_hold_win_a}, stop=${summary.president_stop_a}`,
    `President B:       opp_total=${summary.president_opportunity_count_b}, hold=${summary.president_hold_b}, hold_win=${summary.president_hold_win_b}, stop=${summary.president_stop_b}`,
    `Gukjin A:          opp_total=${summary.gukjin_opportunity_count_a}, five=${summary.gukjin_five_a}, junk=${summary.gukjin_junk_a}, five_rate=${fmtRate(summary.gukjin_five_rate_a)}, five_mongbak=${summary.gukjin_five_mongbak_a}(${fmtRate(summary.gukjin_five_mongbak_rate_a)}), junk_mongbak=${summary.gukjin_junk_mongbak_a}(${fmtRate(summary.gukjin_junk_mongbak_rate_a)})`,
    `Gukjin B:          opp_total=${summary.gukjin_opportunity_count_b}, five=${summary.gukjin_five_b}, junk=${summary.gukjin_junk_b}, five_rate=${fmtRate(summary.gukjin_five_rate_b)}, five_mongbak=${summary.gukjin_five_mongbak_b}(${fmtRate(summary.gukjin_five_mongbak_rate_b)}), junk_mongbak=${summary.gukjin_junk_mongbak_b}(${fmtRate(summary.gukjin_junk_mongbak_rate_b)})`,
    `Bankrupt:          A=${bankrupt.a_bankrupt_count}, B=${bankrupt.b_bankrupt_count}`,
    `Result file:       ${summary.result_out || ""}`,
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
    const endState = playSingleRound(
      roundStart,
      seed,
      playerByActor,
      Math.max(20, Math.floor(opts.maxSteps)),
      (decision) => {
        if (decision?.actor === "human" || decision?.actor === "ai") {
          const actorMetrics = roundSpecial[decision.actor];
          if (decision?.stateBefore?.phase === "playing") {
            for (const signature of collectUniqueShakingOpportunitySignatures(
              decision.stateBefore,
              decision.actor
            )) {
              const seen = roundUniqueShakingOpportunity[decision.actor];
              if (seen.has(signature)) continue;
              seen.add(signature);
              actorMetrics.shaking_opportunity_count += 1;
            }
            for (const signature of collectUniqueBombOpportunitySignatures(
              decision.stateBefore,
              decision.actor
            )) {
              const seen = roundUniqueBombOpportunity[decision.actor];
              if (seen.has(signature)) continue;
              seen.add(signature);
              actorMetrics.bomb_opportunity_count += 1;
            }
          }
          actorMetrics.shaking_count += Math.max(0, Number(decision?.transitionEvents?.shaking_declare_count || 0));
          actorMetrics.bomb_count += Math.max(0, Number(decision?.transitionEvents?.bomb_declare_count || 0));
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
              candidate,
              legalCount
            ),
          };
          if (goStopSnapshot) {
            row.features13 = Array.isArray(row.features) ? row.features.slice() : [];
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
