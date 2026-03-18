import fs from "node:fs";
import path from "node:path";
import { createSeededRng } from "../src/engine/index.js";
import { getActionPlayerKey } from "../src/engine/runner.js";
import { aiPlay } from "../src/ai/aiPlay.js";
import { resolveBotPolicy } from "../src/ai/policies.js";
import { resolvePlayerSpecCore } from "../src/ai/evalCore/playerSpecCore.js";
import { resolveResolvedPlayerAction } from "../src/ai/evalCore/resolvedPlayerAction.js";
import {
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

// Quick Read Map (top-down):
// 1) main()
// 2) runEvalRound(): per-game simulation loop
// 3) decision inference helpers (imitation counters)
// 4) parseArgs()/state transition helpers

// =============================================================================
// Section 1. CLI
// =============================================================================
function normalizePolicyName(policy) {
  return String(policy || "").trim().toLowerCase();
}

function normalizeControlPolicyMode(mode) {
  const raw = String(mode || "").trim().toLowerCase();
  if (!raw || raw === "pure_model") return "pure_model";
  throw new Error(`invalid --control-policy-mode: ${mode} (allowed: pure_model)`);
}

const OPPONENT_SPEC_CACHE = new Map();
const NEAT_MODEL_FORMAT = "neat_python_genome_v1";
const K_HYPERNEAT_MODEL_FORMAT = "k_hyperneat_executor_v1";
function loadGenomeModel(genomePath, label) {
  const full = path.resolve(String(genomePath || "").trim());
  if (!fs.existsSync(full)) {
    throw new Error(`${label} not found: ${genomePath}`);
  }
  const parsed = JSON.parse(fs.readFileSync(full, "utf8"));
  const formatVersion = String(parsed?.format_version || "").trim();
  if (formatVersion !== NEAT_MODEL_FORMAT) {
    throw new Error(`invalid ${label} format: expected ${NEAT_MODEL_FORMAT}`);
  }
  return {
    model: parsed,
    modelPath: full,
  };
}

function loadSupportedPolicyModel(modelPath, label) {
  const full = path.resolve(String(modelPath || "").trim());
  if (!fs.existsSync(full)) {
    throw new Error(`${label} not found: ${modelPath}`);
  }
  const parsed = JSON.parse(fs.readFileSync(full, "utf8"));
  const formatVersion = String(parsed?.format_version || "").trim();
  if (formatVersion !== NEAT_MODEL_FORMAT && formatVersion !== K_HYPERNEAT_MODEL_FORMAT) {
    throw new Error(
      `invalid ${label} format: expected ${NEAT_MODEL_FORMAT} or ${K_HYPERNEAT_MODEL_FORMAT}`
    );
  }
  return {
    model: parsed,
    modelPath: full,
  };
}

function parsePhaseModelToken(rawToken) {
  const m = String(rawToken || "")
    .trim()
    .match(/^(phase([0-3])_seed(\d+))(?:\:(winner_genome|winner_play_genome|winner_option_genome))?$/i);
  if (!m) return null;
  const phase = Number(m[2]);
  const seed = Number(m[3]);
  const defaultModelName = "winner_genome";
  const modelName = String(m[4] || defaultModelName).trim().toLowerCase();
  const outputPrefix = "neat";
  const baseTokenKey = `phase${phase}_seed${seed}`;
  const tokenKey = modelName === defaultModelName ? baseTokenKey : `${baseTokenKey}:${modelName}`;
  return { phase, seed, outputPrefix, tokenKey, modelName };
}

function parseKHyperneatModelToken(rawToken) {
  const m = String(rawToken || "")
    .trim()
    .match(/^k_hyperneat\(\s*(.+?)\s*\)$/i);
  if (!m) return null;
  const modelPathToken = String(m[1] || "").trim();
  if (!modelPathToken) return null;
  return {
    tokenKey: `k_hyperneat(${modelPathToken})`,
    modelPathToken,
  };
}

function looksLikeJsonModelPath(rawToken) {
  return /\.json$/i.test(String(rawToken || "").trim());
}

function resolvePhaseModelToken(token, label) {
  const parsedToken = parsePhaseModelToken(token);
  if (!parsedToken) {
    return null;
  }
  const { phase, seed, outputPrefix, tokenKey, modelName } = parsedToken;
  const summaryPath = path.resolve(`logs/NEAT/${outputPrefix}_phase${phase}_seed${seed}/run_summary.json`);
  if (modelName === "winner_genome" && fs.existsSync(summaryPath)) {
    try {
      const summaryRaw = String(fs.readFileSync(summaryPath, "utf8") || "").replace(/^\uFEFF/, "");
      const summary = JSON.parse(summaryRaw);
      if (String(summary?.winner_repair_status || "") === "summary_repaired_winner_unrecoverable") {
        throw new Error(
          `unrecoverable winner for phase${phase}_seed${seed}: exact best genome was not restorable from saved artifacts`
        );
      }
    } catch (err) {
      if (String(err?.message || err).includes("unrecoverable winner")) {
        throw err;
      }
    }
  }
  const modelPath = path.resolve(`logs/NEAT/${outputPrefix}_phase${phase}_seed${seed}/models/${modelName}.json`);
  const loaded = loadSupportedPolicyModel(modelPath, label);
  return {
    key: tokenKey,
    label: tokenKey,
    phase,
    seed,
    model: loaded.model,
    modelPath: loaded.modelPath,
  };
}

function resolveKHyperneatModelToken(token, label) {
  const parsedToken = parseKHyperneatModelToken(token);
  if (!parsedToken) return null;
  const loaded = loadSupportedPolicyModel(parsedToken.modelPathToken, label);
  return {
    key: parsedToken.tokenKey,
    label: parsedToken.tokenKey,
    model: loaded.model,
    modelPath: loaded.modelPath,
  };
}

function resolveDirectJsonModelToken(token, label) {
  if (!looksLikeJsonModelPath(token)) return null;
  const loaded = loadSupportedPolicyModel(token, label);
  return {
    key: token,
    label: token,
    model: loaded.model,
    modelPath: loaded.modelPath,
  };
}

function resolveAnyModelToken(token, label) {
  const phaseModel = resolvePhaseModelToken(token, label);
  if (phaseModel) return phaseModel;
  const kHyperneatModel = resolveKHyperneatModelToken(token, label);
  if (kHyperneatModel) return kHyperneatModel;
  const directJsonModel = resolveDirectJsonModelToken(token, label);
  if (directJsonModel) return directJsonModel;
  throw new Error(`invalid ${label}: ${token}`);
}

function resolveOpponentSpec(policyToken, opponentGenomePath) {
  const raw = String(policyToken || "").trim();
  const cacheKey = `${raw}|oppGenome=${String(opponentGenomePath || "").trim()}`;
  if (OPPONENT_SPEC_CACHE.has(cacheKey)) {
    return OPPONENT_SPEC_CACHE.get(cacheKey);
  }

  let resolved = null;
  if (normalizePolicyName(raw) === "genome") {
    const loaded = loadSupportedPolicyModel(opponentGenomePath, "opponent genome");
    resolved = {
      kind: "model",
      key: "genome",
      label: "genome",
      model: loaded.model,
      modelPath: loaded.modelPath,
    };
  } else {
    resolved = resolvePlayerSpecCore(raw, {
      label: "opponent policy",
      resolveHeuristic: (token) => resolveBotPolicy(token),
      resolveModel: (token, modelLabel) => resolveAnyModelToken(token, modelLabel),
    });
  }

  OPPONENT_SPEC_CACHE.set(cacheKey, resolved);
  return resolved;
}

function parseEarlyStopWinRateCutoffs(rawValue, label) {
  if (Array.isArray(rawValue)) {
    return normalizeEarlyStopWinRateCutoffs(rawValue, label);
  }
  const text = String(rawValue || "").trim();
  if (!text) {
    return [];
  }
  let parsed = null;
  try {
    parsed = JSON.parse(text);
  } catch (err) {
    throw new Error(`invalid ${label} JSON: ${String(err && err.message ? err.message : err)}`);
  }
  return normalizeEarlyStopWinRateCutoffs(parsed, label);
}

function normalizeEarlyStopWinRateCutoffs(parsed, label) {
  if (!Array.isArray(parsed)) {
    throw new Error(`${label} must be a JSON array`);
  }
  const out = [];
  const seenGames = new Set();
  for (let i = 0; i < parsed.length; i += 1) {
    const item = parsed[i];
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      throw new Error(`${label} items must be objects`);
    }
    const games = Number(item.games);
    const maxWinRate = Number(item.max_win_rate);
    if (!Number.isInteger(games) || games < 1) {
      throw new Error(`${label}[${i}].games must be an integer >= 1`);
    }
    if (!Number.isFinite(maxWinRate) || maxWinRate < 0 || maxWinRate > 1) {
      throw new Error(`${label}[${i}].max_win_rate must be finite and in [0,1]`);
    }
    if (seenGames.has(games)) {
      throw new Error(`duplicate ${label} games value: ${games}`);
    }
    seenGames.add(games);
    out.push({ games, maxWinRate });
  }
  out.sort((a, b) => a.games - b.games);
  return out;
}

function parseEarlyStopGoTakeRateCutoffs(rawValue, label) {
  if (Array.isArray(rawValue)) {
    return normalizeEarlyStopGoTakeRateCutoffs(rawValue, label);
  }
  const text = String(rawValue || "").trim();
  if (!text) {
    return [];
  }
  let parsed = null;
  try {
    parsed = JSON.parse(text);
  } catch (err) {
    throw new Error(`invalid ${label} JSON: ${String(err && err.message ? err.message : err)}`);
  }
  return normalizeEarlyStopGoTakeRateCutoffs(parsed, label);
}

function normalizeEarlyStopGoTakeRateCutoffs(parsed, label) {
  if (!Array.isArray(parsed)) {
    throw new Error(`${label} must be a JSON array`);
  }
  const out = [];
  const seenGames = new Set();
  for (let i = 0; i < parsed.length; i += 1) {
    const item = parsed[i];
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      throw new Error(`${label} items must be objects`);
    }
    const games = Number(item.games);
    const minGoOpportunityCount = Number(item.min_go_opportunity_count);
    const minGoTakeRate =
      item.min_go_take_rate === null || item.min_go_take_rate === undefined || String(item.min_go_take_rate).trim() === ""
        ? null
        : Number(item.min_go_take_rate);
    const maxGoTakeRate =
      item.max_go_take_rate === null || item.max_go_take_rate === undefined || String(item.max_go_take_rate).trim() === ""
        ? null
        : Number(item.max_go_take_rate);
    if (!Number.isInteger(games) || games < 1) {
      throw new Error(`${label}[${i}].games must be an integer >= 1`);
    }
    if (!Number.isInteger(minGoOpportunityCount) || minGoOpportunityCount < 0) {
      throw new Error(`${label}[${i}].min_go_opportunity_count must be an integer >= 0`);
    }
    if (minGoTakeRate === null && maxGoTakeRate === null) {
      throw new Error(`${label}[${i}] must define min_go_take_rate or max_go_take_rate`);
    }
    for (const [fieldName, value] of [["min_go_take_rate", minGoTakeRate], ["max_go_take_rate", maxGoTakeRate]]) {
      if (value === null) continue;
      if (!Number.isFinite(value) || value < 0 || value > 1) {
        throw new Error(`${label}[${i}].${fieldName} must be finite and in [0,1] or null`);
      }
    }
    if (minGoTakeRate !== null && maxGoTakeRate !== null && minGoTakeRate > maxGoTakeRate) {
      throw new Error(`${label}[${i}].min_go_take_rate must be <= max_go_take_rate`);
    }
    if (seenGames.has(games)) {
      throw new Error(`duplicate ${label} games value: ${games}`);
    }
    seenGames.add(games);
    out.push({ games, minGoOpportunityCount, minGoTakeRate, maxGoTakeRate });
  }
  out.sort((a, b) => a.games - b.games);
  return out;
}

function parseArgs(argv) {
  const args = [...argv];
  const out = {
    genomePath: "",
    opponentGenomePath: "",
    games: 3,
    seed: "neat-python",
    maxSteps: 600,
    opponentPolicy: "",
    opponentPolicyMix: [],
    firstTurnPolicy: "alternate",
    fixedFirstTurn: "human",
    continuousSeries: true,
    kiboOut: "",
    // NOTE:
    // - fitnessGoldScale is used as tanh normalization scale for mean_gold_delta.
    // - fitnessGoldNeutralDelta shifts gold neutral baseline (0-score point).
    // - fitnessWinWeight / fitnessGoldWeight are mapped to:
    //   win / gold component weights (normalized internally).
    fitnessGoldScale: null,
    fitnessGoldNeutralDelta: null,
    fitnessWinWeight: null,
    fitnessGoldWeight: null,
    fitnessWinNeutralRate: null,
    earlyStopWinRateCutoffs: [],
    earlyStopGoTakeRateCutoffs: [],
    controlPolicyMode: "pure_model",
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

    if (key === "--genome") out.genomePath = String(value || "").trim();
    else if (key === "--opponent-genome") out.opponentGenomePath = String(value || "").trim();
    else if (key === "--games") out.games = Math.max(1, Number(value || 0));
    else if (key === "--seed") out.seed = String(value || "neat-python");
    else if (key === "--max-steps") out.maxSteps = Math.max(20, Number(value || 600));
    else if (key === "--opponent-policy") out.opponentPolicy = String(value || "").trim();
    else if (key === "--opponent-policy-mix") {
      let parsed = null;
      try {
        parsed = JSON.parse(String(value || "[]"));
      } catch (err) {
        throw new Error(`invalid --opponent-policy-mix JSON: ${String(err && err.message ? err.message : err)}`);
      }
      if (!Array.isArray(parsed)) {
        throw new Error("--opponent-policy-mix must be a JSON array");
      }
      const mix = [];
      for (const item of parsed) {
        if (!item || typeof item !== "object") {
          throw new Error("--opponent-policy-mix items must be objects");
        }
        const policy = String(item.policy || "").trim();
        const weight = Number(item.weight);
        if (!policy) {
          throw new Error("--opponent-policy-mix item policy is required");
        }
        if (!Number.isFinite(weight) || weight <= 0) {
          throw new Error(`invalid --opponent-policy-mix weight for policy=${policy}`);
        }
        mix.push({ policy, weight });
      }
      if (mix.length <= 0) {
        throw new Error("--opponent-policy-mix must contain at least one entry");
      }
      out.opponentPolicyMix = mix;
    }
    else if (key === "--first-turn-policy") out.firstTurnPolicy = String(value || "alternate").trim().toLowerCase();
    else if (key === "--fixed-first-turn") out.fixedFirstTurn = String(value || "human").trim().toLowerCase();
    else if (key === "--switch-seats") {
      // Backward compatibility: legacy seat switch now maps to first-turn policy.
      out.firstTurnPolicy = String(value || "1").trim() === "0" ? "fixed" : "alternate";
    }
    else if (key === "--continuous-series") out.continuousSeries = !(String(value || "1").trim() === "0");
    else if (key === "--kibo-out") out.kiboOut = String(value || "").trim();
    else if (key === "--fitness-gold-scale") out.fitnessGoldScale = Number(value);
    else if (key === "--fitness-gold-neutral-delta") out.fitnessGoldNeutralDelta = Number(value);
    else if (key === "--fitness-win-weight") out.fitnessWinWeight = Number(value);
    else if (key === "--fitness-gold-weight") out.fitnessGoldWeight = Number(value);
    else if (key === "--fitness-win-neutral-rate") out.fitnessWinNeutralRate = Number(value);
    else if (key === "--early-stop-win-rate-cutoffs") {
      out.earlyStopWinRateCutoffs = parseEarlyStopWinRateCutoffs(value, "--early-stop-win-rate-cutoffs");
    }
    else if (key === "--early-stop-go-take-rate-cutoffs") {
      out.earlyStopGoTakeRateCutoffs = parseEarlyStopGoTakeRateCutoffs(
        value,
        "--early-stop-go-take-rate-cutoffs"
      );
    }
    else if (key === "--control-policy-mode") out.controlPolicyMode = normalizeControlPolicyMode(value);
    else if (key === "--control-heuristic-policy") {
      throw new Error(`deprecated option: ${key} (control fallback modes were removed)`);
    }
    else if (key === "--control-play-match-model") {
      throw new Error(`deprecated option: ${key} (hybrid option control mode was removed)`);
    }
    else throw new Error(`Unknown argument: ${key}`);
  }

  if (!out.genomePath) throw new Error("--genome is required");
  if (out.firstTurnPolicy !== "alternate" && out.firstTurnPolicy !== "fixed") {
    throw new Error(`invalid --first-turn-policy: ${out.firstTurnPolicy}`);
  }
  if (out.fixedFirstTurn !== "human" && out.fixedFirstTurn !== "ai") {
    throw new Error(`invalid --fixed-first-turn: ${out.fixedFirstTurn}`);
  }
  const hasPolicy = String(out.opponentPolicy || "").trim().length > 0;
  const hasPolicyMix = Array.isArray(out.opponentPolicyMix) && out.opponentPolicyMix.length > 0;
  if (!hasPolicy && !hasPolicyMix) {
    throw new Error("--opponent-policy or --opponent-policy-mix is required");
  }
  if (hasPolicy && String(out.opponentPolicy || "").trim().toLowerCase() === "genome" && !out.opponentGenomePath) {
    throw new Error("--opponent-genome is required when --opponent-policy=genome");
  }
  if (!hasPolicy && hasPolicyMix) {
    const hasGenomeInMix = out.opponentPolicyMix.some((x) => String(x.policy || "").trim().toLowerCase() === "genome");
    if (hasGenomeInMix && !out.opponentGenomePath) {
      throw new Error("--opponent-genome is required when --opponent-policy-mix includes genome");
    }
  }
  if (!Number.isFinite(out.fitnessGoldScale) || out.fitnessGoldScale <= 0) {
    throw new Error("--fitness-gold-scale is required and must be > 0");
  }
  if (!Number.isFinite(out.fitnessGoldNeutralDelta)) {
    throw new Error("--fitness-gold-neutral-delta is required and must be finite");
  }
  if (!Number.isFinite(out.fitnessWinWeight) || out.fitnessWinWeight < 0) {
    throw new Error("--fitness-win-weight is required and must be >= 0");
  }
  if (!Number.isFinite(out.fitnessGoldWeight) || out.fitnessGoldWeight < 0) {
    throw new Error("--fitness-gold-weight is required and must be >= 0");
  }
  if ((out.fitnessWinWeight + out.fitnessGoldWeight) <= 0) {
    throw new Error("--fitness-win-weight + --fitness-gold-weight must be > 0");
  }
  if (!Number.isFinite(out.fitnessWinNeutralRate) || out.fitnessWinNeutralRate <= 0 || out.fitnessWinNeutralRate >= 1) {
    throw new Error("--fitness-win-neutral-rate is required and must be in (0,1)");
  }
  return out;
}

function classifyOptionDecision(candidates) {
  const set = new Set((candidates || []).map((c) => String(c || "")));
  if (set.has("go") && set.has("stop")) return "go_stop";
  if (set.has("shaking_yes") && set.has("shaking_no")) return "shaking";
  if (set.has("president_stop") && set.has("president_hold")) return "president";
  if (set.has("five") && set.has("junk")) return "gukjin";
  return null;
}

// =============================================================================
// Section 2. Engine Action Helpers + Feature Helpers
// =============================================================================

// =============================================================================
// Section 3. Decision Inference Helpers
// =============================================================================
function heuristicCandidateForDecision(state, actor, decisionType, candidates, heuristicPolicy) {
  if (!Array.isArray(candidates) || !candidates.length) return null;
  const nextByHeuristic = aiPlay(state, actor, {
    source: "heuristic",
    heuristicPolicy: heuristicPolicy || "H-J2",
  });
  if (!nextByHeuristic || stateProgressKey(nextByHeuristic) === stateProgressKey(state)) {
    return null;
  }
  const target = stateProgressKey(nextByHeuristic);
  for (const c of candidates) {
    const simulated = applyAction(state, actor, decisionType, c);
    if (simulated && stateProgressKey(simulated) === target) {
      return normalizeDecisionCandidate(decisionType, c);
    }
  }
  return null;
}

function inferChosenCandidateFromTransition(stateBefore, actor, decisionType, candidates, stateAfter) {
  if (!stateAfter || !Array.isArray(candidates) || !candidates.length) return null;
  const target = stateProgressKey(stateAfter);
  for (const candidate of candidates) {
    const simulated = applyAction(stateBefore, actor, decisionType, candidate);
    if (simulated && stateProgressKey(simulated) === target) {
      return normalizeDecisionCandidate(decisionType, candidate);
    }
  }
  return null;
}

// =============================================================================
// Section 4. Round Simulation + Metrics
// =============================================================================
function buildWeightedOpponentPlan(opts, totalGames) {
  const fixedPolicy = String(opts.opponentPolicy || "").trim();
  if (fixedPolicy) {
    return [{ policy: fixedPolicy, count: totalGames, order: 0 }];
  }
  if (!Array.isArray(opts.opponentPolicyMix) || opts.opponentPolicyMix.length <= 0) {
    return [];
  }
  const weighted = [];
  let totalWeight = 0;
  for (let i = 0; i < opts.opponentPolicyMix.length; i += 1) {
    const item = opts.opponentPolicyMix[i];
    const policy = String(item?.policy || "").trim();
    const weight = Number(item?.weight || 0);
    if (!policy) continue;
    if (!Number.isFinite(weight) || weight <= 0) continue;
    weighted.push({ policy, weight, order: i });
    totalWeight += weight;
  }
  if (!Number.isFinite(totalWeight) || totalWeight <= 0 || weighted.length <= 0) {
    throw new Error("invalid opponent_policy_mix total weight");
  }

  const plan = weighted.map((entry) => {
    const exact = (totalGames * entry.weight) / totalWeight;
    const count = Math.floor(exact);
    return {
      policy: entry.policy,
      count,
      remainder: exact - count,
      order: entry.order,
    };
  });
  let assigned = 0;
  for (const item of plan) {
    assigned += item.count;
  }
  let remaining = totalGames - assigned;
  if (remaining > 0) {
    const priority = [...plan].sort((a, b) => {
      if (b.remainder !== a.remainder) return b.remainder - a.remainder;
      return a.order - b.order;
    });
    for (let i = 0; i < remaining; i += 1) {
      priority[i % priority.length].count += 1;
    }
  }

  return plan.filter((item) => item.count > 0);
}

function shuffleInPlace(items, rng) {
  for (let i = items.length - 1; i > 0; i -= 1) {
    const j = Math.max(0, Math.min(i, Math.floor(Number(rng() || 0) * (i + 1))));
    if (i === j) continue;
    const tmp = items[i];
    items[i] = items[j];
    items[j] = tmp;
  }
}

function buildEvaluationSchedule(opts, totalGames) {
  const plan = buildWeightedOpponentPlan(opts, totalGames);
  if (plan.length <= 0) {
    throw new Error("resolved empty opponent policy schedule");
  }
  const remaining = plan.map((item) => ({
    policy: item.policy,
    count: Math.max(0, Math.floor(Number(item.count || 0))),
  }));
  const schedule = [];
  while (schedule.length < totalGames) {
    let progressed = false;
    for (const item of remaining) {
      if (item.count <= 0) continue;
      const gi = schedule.length;
      schedule.push({
        opponentPolicy: item.policy,
        firstTurnKey:
          opts.firstTurnPolicy === "fixed"
            ? opts.fixedFirstTurn
            : gi % 2 === 0
              ? "ai"
              : "human",
      });
      item.count -= 1;
      progressed = true;
      if (schedule.length >= totalGames) break;
    }
    if (!progressed) {
      throw new Error("evaluation schedule exhausted before requested game count");
    }
  }
  return schedule;
}

function controlGoldDiff(state, controlActor) {
  const opp = controlActor === "human" ? "ai" : "human";
  const controlGold = Number(state?.players?.[controlActor]?.gold || 0);
  const oppGold = Number(state?.players?.[opp]?.gold || 0);
  return controlGold - oppGold;
}

function runEvalRound(
  initialState,
  controlModel,
  seed,
  controlActor,
  opponentPolicy,
  maxSteps,
  opponentGenomePath,
  controlOptions = {}
) {
  const rng = createSeededRng(`${seed}|rng`);
  let state = initialState;
  const opponentSpec = resolveOpponentSpec(opponentPolicy, opponentGenomePath);
  const controlPolicyMode = normalizeControlPolicyMode(controlOptions.controlPolicyMode || "pure_model");
  const imitation = {
    totals: { play: 0, match: 0, option: 0 },
    matches: { play: 0, match: 0, option: 0 },
  };
  const behaviorDecisions = {
    shaking_opportunities: 0,
    shaking_yes_count: 0,
    president_opportunities: 0,
    president_hold_count: 0,
    gukjin_opportunities: 0,
    gukjin_junk_count: 0,
    gukjin_five_count: 0,
  };
  let goOpportunityCount = 0;

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
    if (
      actor === controlActor &&
      decisionType === "option" &&
      state?.phase === "go-stop" &&
      state?.pendingGoStop === actor
    ) {
      const normalizedOptions = (candidates || []).map((candidate) => canonicalOptionAction(candidate)).filter(Boolean);
      if (normalizedOptions.includes("go") && normalizedOptions.includes("stop")) {
        goOpportunityCount += 1;
      }
    }
    let next = state;
    let controlDecisionOwnedByModel = false;

    if (actor === controlActor) {
      next = (
        resolveResolvedPlayerAction(state, actor, {
          kind: "model",
          model: controlModel,
        })?.next || state
      );
      controlDecisionOwnedByModel = true;
    } else {
      next = resolveResolvedPlayerAction(state, actor, opponentSpec)?.next || state;
    }

    if (!next || stateProgressKey(next) === before) {
      next = randomLegalAction(state, actor, rng);
      if (actor === controlActor) {
        controlDecisionOwnedByModel = false;
      }
    }
    if (!next || stateProgressKey(next) === before) {
      throw new Error(
        `action resolution failed after fallback: seed=${seed}, step=${steps}, actor=${actor}, phase=${String(state?.phase || "")}`
      );
    }

    if (actor === controlActor && controlDecisionOwnedByModel && decisionType && candidates.length > 0) {
      const chosen = inferChosenCandidateFromTransition(
        state,
        actor,
        decisionType,
        candidates,
        next
      );
      if (chosen) {
        if (decisionType === "option") {
          const optionKind = classifyOptionDecision(candidates);
          if (optionKind === "shaking") {
            behaviorDecisions.shaking_opportunities += 1;
            if (chosen === "shaking_yes") {
              behaviorDecisions.shaking_yes_count += 1;
            }
          } else if (optionKind === "president") {
            behaviorDecisions.president_opportunities += 1;
            if (chosen === "president_hold") {
              behaviorDecisions.president_hold_count += 1;
            }
          } else if (optionKind === "gukjin") {
            behaviorDecisions.gukjin_opportunities += 1;
            if (chosen === "junk") {
              behaviorDecisions.gukjin_junk_count += 1;
            } else if (chosen === "five") {
              behaviorDecisions.gukjin_five_count += 1;
            }
          }
        }
        const imitationDecisionType =
          decisionType === "play" || decisionType === "match" || decisionType === "option"
            ? decisionType
            : null;
        if (imitationDecisionType) {
          const imitationRefPolicy =
            opponentSpec.kind === "model"
              ? "H-J2"
              : String(opponentSpec.heuristicPolicy || opponentPolicy || "");
          const refCandidate = heuristicCandidateForDecision(
            state,
            actor,
            decisionType,
            candidates,
            imitationRefPolicy
          );
          if (refCandidate) {
            imitation.totals[imitationDecisionType] += 1;
            if (chosen === refCandidate) {
              imitation.matches[imitationDecisionType] += 1;
            }
          }
        }
      }
    }

    state = next;
    steps += 1;
  }

  return {
    endState: state,
    imitation,
    goOpportunityCount,
    behaviorDecisions,
  };
}

function cloneDecisionCounters(src) {
  return {
    play: Number(src?.play || 0),
    match: Number(src?.match || 0),
    option: Number(src?.option || 0),
  };
}

function buildImitationMetrics(totals, matches) {
  const t = cloneDecisionCounters(totals);
  const m = cloneDecisionCounters(matches);
  const ratio = (num, den) => (den > 0 ? num / den : 0);
  const playRatio = ratio(m.play, t.play);
  const matchRatio = ratio(m.match, t.match);
  const optionRatio = ratio(m.option, t.option);
  const weights = { play: 0.5, match: 0.3, option: 0.2 };
  let weightSum = 0;
  for (const k of ["play", "match", "option"]) {
    if (Number(t[k] || 0) > 0) weightSum += Number(weights[k] || 0);
  }
  const weightedRaw =
    weights.play * playRatio +
    weights.match * matchRatio +
    weights.option * optionRatio;
  const weightedScore = weightSum > 0 ? weightedRaw / weightSum : 0;
  return {
    totals: t,
    matches: m,
    playRatio,
    matchRatio,
    optionRatio,
    weights,
    weightedScore,
  };
}

// =============================================================================
// Section 5. Entrypoint
// =============================================================================
function main() {
  const evalStartMs = Date.now();
  const opts = parseArgs(process.argv.slice(2));
  const full = path.resolve(opts.genomePath);
  if (!fs.existsSync(full)) throw new Error(`genome not found: ${opts.genomePath}`);

  const controlModel = JSON.parse(fs.readFileSync(full, "utf8"));
  {
    const formatVersion = String(controlModel?.format_version || "").trim();
    if (formatVersion !== NEAT_MODEL_FORMAT) {
      throw new Error(`invalid --genome format: expected ${NEAT_MODEL_FORMAT}`);
    }
  }
  if (String(opts.opponentPolicy || "").trim()) {
    resolveOpponentSpec(opts.opponentPolicy, opts.opponentGenomePath);
  }
  for (const item of opts.opponentPolicyMix || []) {
    resolveOpponentSpec(item?.policy, opts.opponentGenomePath);
  }

  const requestedGames = Math.max(1, Math.floor(opts.games));
  const maxSteps = Math.max(20, Math.floor(opts.maxSteps));
  const evaluationSchedule = buildEvaluationSchedule(opts, requestedGames);
  const earlyStopWinRateCutoffs = (opts.earlyStopWinRateCutoffs || []).filter(
    (item) => item.games <= requestedGames
  );
  const earlyStopGoTakeRateCutoffs = (opts.earlyStopGoTakeRateCutoffs || []).filter(
    (item) => item.games <= requestedGames
  );
  const controlActor = "ai";
  const opponentActor = "human";
  let wins = 0;
  let losses = 0;
  let draws = 0;
  const goldDeltas = [];
  const bankrupt = {
    my_bankrupt_count: 0,
    my_inflicted_bankrupt_count: 0,
  };
  const seatStats = {
    first: { games: 0, wins: 0, losses: 0, draws: 0, goldSum: 0 },
    second: { games: 0, wins: 0, losses: 0, draws: 0, goldSum: 0 },
  };
  let goOpportunityCount = 0;
  let goOpportunityGames = 0;
  let goCount = 0;
  let goGames = 0;
  let goFailCount = 0;
  const simImitationTotals = { play: 0, match: 0, option: 0 };
  const simImitationMatches = { play: 0, match: 0, option: 0 };
  const firstTurnCounts = {
    human: 0,
    ai: 0,
  };
  const opponentPolicyCounts = {};
  const seriesSession = {
    roundsPlayed: 0,
    previousEndState: null,
  };
  let completedGames = 0;
  let earlyStop = null;
  let nextEarlyStopCutoffIdx = 0;
  let nextEarlyStopGoTakeRateCutoffIdx = 0;
  const kiboWriter = opts.kiboOut ? fs.createWriteStream(opts.kiboOut, { encoding: "utf8" }) : null;

  try {
    for (let gi = 0; gi < requestedGames; gi += 1) {
      const scheduleItem = evaluationSchedule[gi];
      const firstTurnKey = String(scheduleItem?.firstTurnKey || "");
      const opponentPolicyForGame = String(scheduleItem?.opponentPolicy || "");
      if (!opponentPolicyForGame) {
        throw new Error("resolved empty opponent policy");
      }
      opponentPolicyCounts[opponentPolicyForGame] = Number(opponentPolicyCounts[opponentPolicyForGame] || 0) + 1;
      firstTurnCounts[firstTurnKey] += 1;
      const seed = `${opts.seed}|g=${gi}|first=${firstTurnKey}|sr=${seriesSession.roundsPlayed}`;
      const roundStart = opts.continuousSeries
        ? seriesSession.previousEndState
          ? continueRound(seriesSession.previousEndState, seed, firstTurnKey)
          : startRound(seed, firstTurnKey)
        : startRound(seed, firstTurnKey);
      const beforeGoldDiff = controlGoldDiff(roundStart, controlActor);
      const gameResult = runEvalRound(
        roundStart,
        controlModel,
        seed,
        controlActor,
        opponentPolicyForGame,
        maxSteps,
        opts.opponentGenomePath,
        {
          controlPolicyMode: opts.controlPolicyMode,
        }
      );
      const endState = gameResult?.endState || gameResult;
      const controlGoOpportunityCount = Math.max(0, Number(gameResult?.goOpportunityCount || 0));
      const afterGoldDiff = controlGoldDiff(endState, controlActor);
      const goldDelta = afterGoldDiff - beforeGoldDiff;
      goldDeltas.push(goldDelta);
      const seatKey = firstTurnKey === controlActor ? "first" : "second";
      const seatRecord = seatStats[seatKey];
      seatRecord.games += 1;
      seatRecord.goldSum += goldDelta;
      const controlGold = Number(endState?.players?.[controlActor]?.gold || 0);
      const opponentGold = Number(endState?.players?.[opponentActor]?.gold || 0);
      const controlBankrupt = controlGold <= 0;
      const opponentBankrupt = opponentGold <= 0;
      const controlGoCount = Math.max(0, Number(endState?.players?.[controlActor]?.goCount || 0));
      goOpportunityCount += controlGoOpportunityCount;
      if (controlGoOpportunityCount > 0) {
        goOpportunityGames += 1;
      }
      goCount += controlGoCount;
      if (controlGoCount > 0) {
        goGames += 1;
      }
      if (opponentBankrupt) {
        bankrupt.my_inflicted_bankrupt_count += 1;
      }
      if (controlBankrupt) {
        bankrupt.my_bankrupt_count += 1;
      }

      if (kiboWriter) {
        kiboWriter.write(
          `${JSON.stringify({
            game_index: gi,
            seed,
            first_turn: firstTurnKey,
            control_actor: controlActor,
            opponent_actor: opponentActor,
            opponent_policy: opponentPolicyForGame,
            winner: endState?.result?.winner || "",
            gold_delta: goldDelta,
            control_go_count: controlGoCount,
            result: endState?.result || null,
            kibo_detail: endState?.kiboDetail || "lean",
            kibo: Array.isArray(endState?.kibo) ? endState.kibo : [],
          })}\n`
        );
      }

      if (opts.continuousSeries) {
        seriesSession.previousEndState = endState;
      }
      seriesSession.roundsPlayed += 1;
      completedGames = gi + 1;

      const winner = endState?.result?.winner || "unknown";
      if (winner === controlActor) {
        wins += 1;
        seatRecord.wins += 1;
      }
      else if (winner === opponentActor) {
        losses += 1;
        seatRecord.losses += 1;
      }
      else {
        draws += 1;
        seatRecord.draws += 1;
      }
      if (controlGoCount > 0 && winner !== controlActor) {
        goFailCount += 1;
      }

      const gt = gameResult?.imitation?.totals || {};
      const gm = gameResult?.imitation?.matches || {};
      for (const k of ["play", "match", "option"]) {
        simImitationTotals[k] += Number(gt[k] || 0);
        simImitationMatches[k] += Number(gm[k] || 0);
      }

      while (nextEarlyStopCutoffIdx < earlyStopWinRateCutoffs.length) {
        const cutoff = earlyStopWinRateCutoffs[nextEarlyStopCutoffIdx];
        if (completedGames < cutoff.games) {
          break;
        }
        const currentWinRate = wins / completedGames;
        if (currentWinRate <= cutoff.maxWinRate) {
          earlyStop = {
            reason: "win_rate_cutoff",
            cutoffGames: cutoff.games,
            maxWinRate: cutoff.maxWinRate,
            observedWinRate: currentWinRate,
          };
          break;
        }
        nextEarlyStopCutoffIdx += 1;
      }
      while (nextEarlyStopGoTakeRateCutoffIdx < earlyStopGoTakeRateCutoffs.length) {
        const cutoff = earlyStopGoTakeRateCutoffs[nextEarlyStopGoTakeRateCutoffIdx];
        if (completedGames < cutoff.games) {
          break;
        }
        if (goOpportunityCount >= cutoff.minGoOpportunityCount) {
          const currentGoTakeRate = goOpportunityCount > 0 ? (goCount / goOpportunityCount) : 0;
          const tooLow = cutoff.minGoTakeRate !== null && currentGoTakeRate <= cutoff.minGoTakeRate;
          const tooHigh = cutoff.maxGoTakeRate !== null && currentGoTakeRate >= cutoff.maxGoTakeRate;
          if (tooLow || tooHigh) {
            earlyStop = {
              reason: "go_take_rate_cutoff",
              cutoffGames: cutoff.games,
              minGoOpportunityCount: cutoff.minGoOpportunityCount,
              minGoTakeRate: cutoff.minGoTakeRate,
              maxGoTakeRate: cutoff.maxGoTakeRate,
              observedGoTakeRate: currentGoTakeRate,
              observedGoOpportunityCount: goOpportunityCount,
            };
            break;
          }
        }
        nextEarlyStopGoTakeRateCutoffIdx += 1;
      }
      if (earlyStop) {
        break;
      }
    }
  } finally {
    if (kiboWriter) {
      kiboWriter.end();
    }
  }

  if (completedGames <= 0) {
    throw new Error("evaluation finished without any completed games");
  }

  const games = completedGames;
  const meanGoldDelta = goldDeltas.length > 0 ? goldDeltas.reduce((a, b) => a + b, 0) / goldDeltas.length : 0;
  const winRate = wins / games;
  const lossRate = losses / games;
  const drawRate = draws / games;
  const firstGames = Math.max(0, Number(seatStats.first.games || 0));
  const secondGames = Math.max(0, Number(seatStats.second.games || 0));
  const firstWinRate = firstGames > 0 ? seatStats.first.wins / firstGames : 0;
  const firstLossRate = firstGames > 0 ? seatStats.first.losses / firstGames : 0;
  const firstDrawRate = firstGames > 0 ? seatStats.first.draws / firstGames : 0;
  const secondWinRate = secondGames > 0 ? seatStats.second.wins / secondGames : 0;
  const secondLossRate = secondGames > 0 ? seatStats.second.losses / secondGames : 0;
  const secondDrawRate = secondGames > 0 ? seatStats.second.draws / secondGames : 0;
  const firstMeanGoldDelta = firstGames > 0 ? seatStats.first.goldSum / firstGames : 0;
  const secondMeanGoldDelta = secondGames > 0 ? seatStats.second.goldSum / secondGames : 0;
  const goOpportunityRate = goOpportunityGames / games;
  const goRate = goGames / games;
  const goFailRate = goGames > 0 ? goFailCount / goGames : 0;
  const goTakeRate = goOpportunityCount > 0 ? goCount / goOpportunityCount : 0;
  const fitnessGoldScale = Number(opts.fitnessGoldScale);
  const fitnessGoldNeutralDelta = Number(opts.fitnessGoldNeutralDelta);
  const weightWinRaw = Number(opts.fitnessWinWeight);
  const weightGoldRaw = Number(opts.fitnessGoldWeight);
  const weightRawSum = weightWinRaw + weightGoldRaw;
  const fitnessWinWeight = weightWinRaw / weightRawSum;
  const fitnessGoldWeight = weightGoldRaw / weightRawSum;
  const fitnessWinNeutralRate = Number(opts.fitnessWinNeutralRate);

  const weightedWinRate = (0.48 * firstWinRate) + (0.52 * secondWinRate);
  const weightedLossRate = (0.48 * firstLossRate) + (0.52 * secondLossRate);
  const weightedDrawRate = (0.48 * firstDrawRate) + (0.52 * secondDrawRate);
  const weightedMeanGoldDelta = (0.48 * firstMeanGoldDelta) + (0.52 * secondMeanGoldDelta);
  const goldNorm = Math.tanh((weightedMeanGoldDelta - fitnessGoldNeutralDelta) / fitnessGoldScale);
  const expectedResultRaw =
    clamp01(weightedWinRate) + (0.5 * clamp01(weightedDrawRate)) - clamp01(weightedLossRate);
  const expectedResult = Math.max(-1.0, Math.min(1.0, expectedResultRaw));
  const neutralExpectedResult = (2.0 * fitnessWinNeutralRate) - 1.0;
  let resultNorm = 0.0;
  if (expectedResult >= neutralExpectedResult) {
    const resultUpperSpan = Math.max(1e-9, 1.0 - neutralExpectedResult);
    resultNorm = clamp01((expectedResult - neutralExpectedResult) / resultUpperSpan);
  } else {
    const resultLowerSpan = Math.max(1e-9, neutralExpectedResult + 1.0);
    resultNorm = -clamp01((neutralExpectedResult - expectedResult) / resultLowerSpan);
  }
  const myBankruptRate = bankrupt.my_bankrupt_count / games;
  const inflictedBankruptRate = bankrupt.my_inflicted_bankrupt_count / games;
  const baseFitness =
    (fitnessGoldWeight * goldNorm) +
    (fitnessWinWeight * resultNorm);
  const fitness = baseFitness;

  const simImitation = buildImitationMetrics(simImitationTotals, simImitationMatches);
  const imitationTotals = cloneDecisionCounters(simImitation.totals);
  const imitationMatches = cloneDecisionCounters(simImitation.matches);
  const imitationPlayRatio = Number(simImitation.playRatio || 0);
  const imitationMatchRatio = Number(simImitation.matchRatio || 0);
  const imitationOptionRatio = Number(simImitation.optionRatio || 0);
  const imitationWeights = simImitation.weights || { play: 0.5, match: 0.3, option: 0.2 };
  const imitationWeightedScore = Number(simImitation.weightedScore || 0);

  const summary = {
    games,
    requested_games: requestedGames,
    control_actor: controlActor,
    opponent_actor: opponentActor,
    opponent_policy: opts.opponentPolicy,
    opponent_policy_mix: opts.opponentPolicyMix,
    opponent_policy_counts: opponentPolicyCounts,
    early_stop_win_rate_cutoffs: earlyStopWinRateCutoffs.map((item) => ({
      games: item.games,
      max_win_rate: item.maxWinRate,
    })),
    early_stop_go_take_rate_cutoffs: earlyStopGoTakeRateCutoffs.map((item) => ({
      games: item.games,
      min_go_opportunity_count: item.minGoOpportunityCount,
      min_go_take_rate: item.minGoTakeRate,
      max_go_take_rate: item.maxGoTakeRate,
    })),
    early_stop_triggered: !!earlyStop,
    early_stop_reason: earlyStop?.reason || null,
    early_stop_cutoff_games: earlyStop?.cutoffGames ?? null,
    early_stop_cutoff_max_win_rate: earlyStop?.maxWinRate ?? null,
    early_stop_observed_win_rate: earlyStop?.observedWinRate ?? null,
    early_stop_cutoff_min_go_opportunity_count: earlyStop?.minGoOpportunityCount ?? null,
    early_stop_cutoff_min_go_take_rate: earlyStop?.minGoTakeRate ?? null,
    early_stop_cutoff_max_go_take_rate: earlyStop?.maxGoTakeRate ?? null,
    early_stop_observed_go_take_rate: earlyStop?.observedGoTakeRate ?? null,
    early_stop_observed_go_opportunity_count: earlyStop?.observedGoOpportunityCount ?? null,
    control_policy_mode: opts.controlPolicyMode,
    opponent_eval_tuning: {
      fast_path: false,
      imitation_reference_enabled: true,
      imitation_decision_scope: "all_control_decisions",
      opponent_heuristic_params: null,
    },
    opponent_genome: String(opts.opponentGenomePath || "") || null,
    first_turn_policy: opts.firstTurnPolicy,
    fixed_first_turn: opts.firstTurnPolicy === "fixed" ? opts.fixedFirstTurn : null,
    first_turn_counts: firstTurnCounts,
    continuous_series: !!opts.continuousSeries,
    bankrupt,
    my_bankrupt_rate: myBankruptRate,
    inflicted_bankrupt_rate: inflictedBankruptRate,
    session_rounds: {
      control_actor_series: seriesSession.roundsPlayed,
    },
    wins,
    losses,
    draws,
    win_rate: winRate,
    loss_rate: lossRate,
    draw_rate: drawRate,
    seat_breakdown: {
      first: {
        games: firstGames,
        wins: seatStats.first.wins,
        losses: seatStats.first.losses,
        draws: seatStats.first.draws,
        win_rate: firstWinRate,
        loss_rate: firstLossRate,
        draw_rate: firstDrawRate,
        mean_gold_delta: firstMeanGoldDelta,
      },
      second: {
        games: secondGames,
        wins: seatStats.second.wins,
        losses: seatStats.second.losses,
        draws: seatStats.second.draws,
        win_rate: secondWinRate,
        loss_rate: secondLossRate,
        draw_rate: secondDrawRate,
        mean_gold_delta: secondMeanGoldDelta,
      },
      weighted: {
        win_rate: weightedWinRate,
        loss_rate: weightedLossRate,
        draw_rate: weightedDrawRate,
        mean_gold_delta: weightedMeanGoldDelta,
        win_weights: { first: 0.48, second: 0.52 },
        gold_weights: { first: 0.45, second: 0.55 },
      },
    },
    go_opportunity_count: goOpportunityCount,
    go_opportunity_games: goOpportunityGames,
    go_opportunity_rate: goOpportunityRate,
    go_count: goCount,
    go_fail_count: goFailCount,
    go_fail_rate: goFailRate,
    go_games: goGames,
    go_rate: goRate,
    go_take_rate: goTakeRate,
    mean_gold_delta: meanGoldDelta,
    p10_gold_delta: quantile(goldDeltas, 0.1),
    p50_gold_delta: quantile(goldDeltas, 0.5),
    p90_gold_delta: quantile(goldDeltas, 0.9),
    fitness_gold_scale: fitnessGoldScale,
    fitness_gold_neutral_delta: fitnessGoldNeutralDelta,
    fitness_win_neutral_rate: fitnessWinNeutralRate,
    fitness_win_weight: fitnessWinWeight,
    fitness_gold_weight: fitnessGoldWeight,
    imitation_source: "opponent_policy",
    sim_imitation_weighted_score: Number(simImitation.weightedScore || 0),
    imitation_play_total: imitationTotals.play,
    imitation_play_matches: imitationMatches.play,
    imitation_play_ratio: imitationPlayRatio,
    imitation_match_total: imitationTotals.match,
    imitation_match_matches: imitationMatches.match,
    imitation_match_ratio: imitationMatchRatio,
    imitation_go_stop_total: imitationTotals.option,
    imitation_go_stop_matches: imitationMatches.option,
    imitation_go_stop_ratio: imitationOptionRatio,
    imitation_option_total: imitationTotals.option,
    imitation_option_matches: imitationMatches.option,
    imitation_option_ratio: imitationOptionRatio,
    imitation_weight_play: imitationWeights.play,
    imitation_weight_match: imitationWeights.match,
    imitation_weight_option: imitationWeights.option,
    imitation_weighted_score: imitationWeightedScore,
    fitness_components: {
      gold_norm: goldNorm,
      weighted_gold_delta: weightedMeanGoldDelta,
      gold_neutral_delta: fitnessGoldNeutralDelta,
      win_norm: resultNorm,
      result_norm: resultNorm,
      result_expected: expectedResult,
      result_neutral: neutralExpectedResult,
      weighted_win_rate: weightedWinRate,
      weighted_draw_rate: weightedDrawRate,
      weighted_loss_rate: weightedLossRate,
      win_neutral_rate: fitnessWinNeutralRate,
      base_fitness: baseFitness,
      bankrupt_rates: {
        my: myBankruptRate,
        inflicted: inflictedBankruptRate,
      },
      bankrupt_counts: {
        my: Number(bankrupt.my_bankrupt_count || 0),
        inflicted: Number(bankrupt.my_inflicted_bankrupt_count || 0),
      },
      weights: {
        win: fitnessWinWeight,
        gold: fitnessGoldWeight,
      },
    },
    eval_time_ms: Math.max(0, Date.now() - evalStartMs),
    seed_used: opts.seed,
    eval_ok: true,
    fitness,
  };

  process.stdout.write(`${JSON.stringify(summary)}\n`);
}

try {
  main();
} catch (err) {
  const msg = err && err.stack ? err.stack : String(err);
  process.stderr.write(`${msg}\n`);
  process.exit(1);
}
