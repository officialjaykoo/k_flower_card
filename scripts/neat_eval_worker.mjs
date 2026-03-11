import fs from "node:fs";
import path from "node:path";
import {
  initSimulationGame,
  startSimulationGame,
  createSeededRng,
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
import { getActionPlayerKey } from "../src/engine/runner.js";
import { aiPlay } from "../src/ai/aiPlay.js";
import { hybridPolicyPlayDetailed } from "../src/ai/hybridPolicyEngine.js";
import { resolveBotPolicy } from "../src/ai/policies.js";

// Quick Read Map (top-down):
// 1) main()
// 2) playSingleRound(): per-game simulation loop
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
  if (
    raw === "hybrid_play_match_only" ||
    raw === "hybrid_playmatch_only" ||
    raw === "play_match_only"
  ) {
    return "hybrid_play_match_only";
  }
  if (
    raw === "hybrid_go_stop_only" ||
    raw === "hybrid_gostop_only" ||
    raw === "go_stop_only" ||
    raw === "gostop_only"
  ) {
    return "hybrid_go_stop_only";
  }
  if (
    raw === "hybrid_option_only" ||
    raw === "hybrid_options_only" ||
    raw === "option_only" ||
    raw === "options_only"
  ) {
    return "hybrid_option_only";
  }
  throw new Error(
    `invalid --control-policy-mode: ${mode} (allowed: pure_model, hybrid_play_match_only, hybrid_go_stop_only, hybrid_option_only)`
  );
}

const OPPONENT_SPEC_CACHE = new Map();
const CONTROL_MODEL_SPEC_CACHE = new Map();

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

function loadGenomeModel(genomePath, label) {
  const full = path.resolve(String(genomePath || "").trim());
  if (!fs.existsSync(full)) {
    throw new Error(`${label} not found: ${genomePath}`);
  }
  const parsed = JSON.parse(fs.readFileSync(full, "utf8"));
  if (String(parsed?.format_version || "").trim() !== "neat_python_genome_v1") {
    throw new Error(`invalid ${label} format: expected neat_python_genome_v1`);
  }
  return {
    model: parsed,
    modelPath: full,
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

function resolvePhaseModelToken(token, label) {
  const parsedToken = parsePhaseModelToken(token);
  if (!parsedToken) {
    throw new Error(`invalid ${label}: ${token}`);
  }
  const { phase, seed, outputPrefix, tokenKey } = parsedToken;
  const summaryPath = path.resolve(`logs/NEAT/${outputPrefix}_phase${phase}_seed${seed}/run_summary.json`);
  if (fs.existsSync(summaryPath)) {
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
  const modelPath = path.resolve(`logs/NEAT/${outputPrefix}_phase${phase}_seed${seed}/models/winner_genome.json`);
  const loaded = loadGenomeModel(modelPath, label);
  return {
    key: tokenKey,
    label: tokenKey,
    phase,
    seed,
    model: loaded.model,
    modelPath: loaded.modelPath,
  };
}

function resolveControlPlayMatchModel(token) {
  const raw = String(token || "").trim();
  if (!raw) {
    throw new Error("control play/match model token is empty");
  }
  if (CONTROL_MODEL_SPEC_CACHE.has(raw)) {
    return CONTROL_MODEL_SPEC_CACHE.get(raw);
  }

  let resolved = null;
  const phaseToken = raw.match(/^phase([0-3])_seed(\d+)$/i);
  if (phaseToken) {
    resolved = resolvePhaseModelToken(raw, "control play/match model");
  } else {
    const loaded = loadGenomeModel(raw, "control play/match model");
    resolved = {
      key: raw,
      label: raw,
      phase: null,
      seed: null,
      model: loaded.model,
      modelPath: loaded.modelPath,
    };
  }
  CONTROL_MODEL_SPEC_CACHE.set(raw, resolved);
  return resolved;
}

function resolveOpponentSpec(policyToken, opponentGenomePath) {
  const raw = String(policyToken || "").trim();
  const cacheKey = `${raw}|oppGenome=${String(opponentGenomePath || "").trim()}`;
  if (OPPONENT_SPEC_CACHE.has(cacheKey)) {
    return OPPONENT_SPEC_CACHE.get(cacheKey);
  }

  let resolved = null;
  if (normalizePolicyName(raw) === "genome") {
    const loaded = loadGenomeModel(opponentGenomePath, "opponent genome");
    resolved = {
      kind: "model",
      key: "genome",
      label: "genome",
      model: loaded.model,
      modelPath: loaded.modelPath,
    };
  } else {
    const hybridGo = parseHybridPlayGoSpec(raw);
    if (hybridGo) {
      const modelSpec = resolvePhaseModelToken(hybridGo.modelToken, "hybrid opponent model");
      const goStopPolicy = resolveBotPolicy(hybridGo.goStopToken);
      const heuristicToken = String(hybridGo.heuristicToken || "").trim();
      const heuristicPolicy = heuristicToken ? resolveBotPolicy(heuristicToken) : "";
      if (!goStopPolicy || (heuristicToken && !heuristicPolicy)) {
        throw new Error(`invalid hybrid opponent policy: ${raw}`);
      }
      const goStopOnly = !heuristicPolicy;
      resolved = {
        kind: "hybrid_play_model",
        key: goStopOnly
          ? `hybrid_play_go(${modelSpec.key},${goStopPolicy})`
          : `hybrid_play_go(${modelSpec.key},${goStopPolicy},${heuristicPolicy})`,
        label: goStopOnly
          ? `hybrid_play_go(${modelSpec.label},${goStopPolicy})`
          : `hybrid_play_go(${modelSpec.label},${goStopPolicy},${heuristicPolicy})`,
        model: modelSpec.model,
        modelPath: modelSpec.modelPath,
        heuristicPolicy,
        goStopPolicy,
        goStopOnly,
      };
    } else {
      const hybrid = parseHybridPlaySpec(raw);
      if (hybrid) {
        const modelSpec = resolvePhaseModelToken(hybrid.modelToken, "hybrid opponent model");
        const heuristicPolicy = resolveBotPolicy(hybrid.heuristicToken);
        if (!heuristicPolicy) {
          throw new Error(`invalid hybrid opponent policy: ${raw}`);
        }
        resolved = {
          kind: "hybrid_play_model",
          key: `hybrid_play(${modelSpec.key},${heuristicPolicy})`,
          label: `hybrid_play(${modelSpec.label},${heuristicPolicy})`,
          model: modelSpec.model,
          modelPath: modelSpec.modelPath,
          heuristicPolicy,
          goStopPolicy: "",
        };
      } else {
        const heuristicPolicy = resolveBotPolicy(raw);
        if (!heuristicPolicy) {
          throw new Error(`invalid opponent policy: ${raw}`);
        }
        resolved = {
          kind: "heuristic",
          key: heuristicPolicy,
          label: heuristicPolicy,
          model: null,
          modelPath: null,
          heuristicPolicy,
        };
      }
    }
  }

  OPPONENT_SPEC_CACHE.set(cacheKey, resolved);
  return resolved;
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
    controlPolicyMode: "pure_model",
    controlHeuristicPolicy: "H-CL",
    controlPlayMatchModel: "",
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
    else if (key === "--fitness-gold-scale") out.fitnessGoldScale = Number(value);
    else if (key === "--fitness-gold-neutral-delta") out.fitnessGoldNeutralDelta = Number(value);
    else if (key === "--fitness-win-weight") out.fitnessWinWeight = Number(value);
    else if (key === "--fitness-gold-weight") out.fitnessGoldWeight = Number(value);
    else if (key === "--fitness-win-neutral-rate") out.fitnessWinNeutralRate = Number(value);
    else if (key === "--control-policy-mode") out.controlPolicyMode = normalizeControlPolicyMode(value);
    else if (key === "--control-heuristic-policy") out.controlHeuristicPolicy = String(value || "").trim();
    else if (key === "--control-play-match-model") out.controlPlayMatchModel = String(value || "").trim();
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
  if (
    out.controlPolicyMode === "hybrid_play_match_only" ||
    out.controlPolicyMode === "hybrid_go_stop_only"
  ) {
    const resolved = resolveBotPolicy(out.controlHeuristicPolicy);
    if (!resolved) {
      throw new Error(
        `invalid --control-heuristic-policy: ${out.controlHeuristicPolicy} (use a policy key from src/ai/policies.js)`
      );
    }
    out.controlHeuristicPolicy = resolved;
  }
  if (out.controlPolicyMode === "hybrid_option_only") {
    if (!String(out.controlPlayMatchModel || "").trim()) {
      throw new Error("--control-play-match-model is required when --control-policy-mode=hybrid_option_only");
    }
    resolveControlPlayMatchModel(out.controlPlayMatchModel);
  }
  return out;
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
    const selected = (state?.players?.[actor]?.hand || []).find(
      (card) => String(card?.id || "") === String(special.cardId || "")
    );
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

// =============================================================================
// Section 3. Decision Inference Helpers
// =============================================================================
function normalizeDecisionCandidate(decisionType, candidate) {
  if (decisionType === "option") return canonicalOptionAction(candidate);
  return String(candidate || "").trim();
}

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
function randomChoice(arr, rng) {
  if (!arr.length) return null;
  const idx = Math.max(0, Math.min(arr.length - 1, Math.floor(Number(rng() || 0) * arr.length)));
  return arr[idx];
}

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

  if (opts.firstTurnPolicy === "fixed") {
    const fixedSchedule = [];
    for (const item of plan) {
      for (let i = 0; i < item.count; i += 1) {
        fixedSchedule.push({
          opponentPolicy: item.policy,
          firstTurnKey: opts.fixedFirstTurn,
        });
      }
    }
    shuffleInPlace(fixedSchedule, createSeededRng(`${opts.seed}|schedule|fixed`));
    return fixedSchedule;
  }

  const aiTarget = Math.ceil(totalGames / 2);
  const humanTarget = totalGames - aiTarget;
  const seatPlan = plan.map((item) => {
    const base = Math.floor(item.count / 2);
    return {
      policy: item.policy,
      count: item.count,
      aiFirstCount: base,
      humanFirstCount: base,
      odd: item.count % 2,
      order: item.order,
    };
  });

  let aiAssigned = 0;
  let humanAssigned = 0;
  const oddEntries = [];
  for (const item of seatPlan) {
    aiAssigned += item.aiFirstCount;
    humanAssigned += item.humanFirstCount;
    if (item.odd) {
      oddEntries.push(item);
    }
  }

  let aiRemaining = aiTarget - aiAssigned;
  let humanRemaining = humanTarget - humanAssigned;
  const oddOrder = oddEntries.map((item, idx) => ({
    item,
    idx,
    ticket: Number(createSeededRng(`${opts.seed}|schedule|odd|${item.policy}|${idx}`)() || 0),
  }));
  oddOrder.sort((a, b) => {
    if (a.ticket !== b.ticket) return a.ticket - b.ticket;
    return a.idx - b.idx;
  });

  for (const entry of oddOrder) {
    if (aiRemaining > humanRemaining) {
      entry.item.aiFirstCount += 1;
      aiRemaining -= 1;
    } else if (humanRemaining > 0) {
      entry.item.humanFirstCount += 1;
      humanRemaining -= 1;
    } else {
      entry.item.aiFirstCount += 1;
      aiRemaining -= 1;
    }
  }

  const schedule = [];
  for (const item of seatPlan) {
    for (let i = 0; i < item.aiFirstCount; i += 1) {
      schedule.push({ opponentPolicy: item.policy, firstTurnKey: "ai" });
    }
    for (let i = 0; i < item.humanFirstCount; i += 1) {
      schedule.push({ opponentPolicy: item.policy, firstTurnKey: "human" });
    }
  }

  if (schedule.length !== totalGames) {
    throw new Error(`invalid evaluation schedule length: expected=${totalGames}, actual=${schedule.length}`);
  }
  shuffleInPlace(schedule, createSeededRng(`${opts.seed}|schedule|shuffle`));
  return schedule;
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

function startRound(seed, firstTurnKey) {
  return initSimulationGame("A", createSeededRng(`${seed}|game`), {
    kiboDetail: "lean",
    firstTurnKey,
  });
}

function continueRound(prevEndState, seed, firstTurnKey) {
  return startSimulationGame(prevEndState, createSeededRng(`${seed}|game`), {
    kiboDetail: "lean",
    keepGold: true,
    useCarryOver: true,
    firstTurnKey,
  });
}

function controlGoldDiff(state, controlActor) {
  const opp = controlActor === "human" ? "ai" : "human";
  const controlGold = Number(state?.players?.[controlActor]?.gold || 0);
  const oppGold = Number(state?.players?.[opp]?.gold || 0);
  return controlGold - oppGold;
}

function clamp01(v) {
  const x = Number(v || 0);
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  return x;
}

function playSingleRound(
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
  const controlHeuristicPolicy =
    controlPolicyMode === "hybrid_play_match_only" || controlPolicyMode === "hybrid_go_stop_only"
      ? String(controlOptions.controlHeuristicPolicy || "H-CL")
      : "";
  const controlPlayMatchModel =
    controlPolicyMode === "hybrid_option_only" ? controlOptions.controlPlayMatchModel || null : null;
  const imitation = {
    totals: { play: 0, match: 0, option: 0 },
    matches: { play: 0, match: 0, option: 0 },
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
      if (controlPolicyMode === "hybrid_play_match_only") {
        const traced = hybridPolicyPlayDetailed(state, actor, {
          model: controlModel,
          heuristicPolicy: controlHeuristicPolicy,
          goStopPolicy: controlHeuristicPolicy,
          phasePolicy: controlHeuristicPolicy,
          specialPolicy: controlHeuristicPolicy,
          playFallbackPolicy: controlHeuristicPolicy,
          modelMatchPhase: true,
        });
        next = traced?.next || state;
        const route = String(traced?.route || "");
        controlDecisionOwnedByModel = route === "model_play" || route === "model_match";
      } else if (controlPolicyMode === "hybrid_go_stop_only") {
        const traced = hybridPolicyPlayDetailed(state, actor, {
          model: controlModel,
          heuristicPolicy: controlHeuristicPolicy,
          goStopPolicy: controlHeuristicPolicy,
          goStopOnly: true,
        });
        next = traced?.next || state;
        const route = String(traced?.route || "");
        controlDecisionOwnedByModel = route === "model_non_go_stop";
      } else if (controlPolicyMode === "hybrid_option_only") {
        const ownsPlayMatch = decisionType === "play" || decisionType === "match";
        if (ownsPlayMatch && controlPlayMatchModel) {
          next = aiPlay(state, actor, {
            source: "model",
            model: controlPlayMatchModel,
          });
          controlDecisionOwnedByModel = false;
          if (!next || stateProgressKey(next) === before) {
            next = aiPlay(state, actor, {
              source: "model",
              model: controlModel,
            });
            controlDecisionOwnedByModel = true;
          }
        } else {
          next = aiPlay(state, actor, {
            source: "model",
            model: controlModel,
          });
          controlDecisionOwnedByModel = true;
        }
      } else {
        next = aiPlay(state, actor, {
          source: "model",
          model: controlModel,
        });
        controlDecisionOwnedByModel = true;
      }
    } else {
      if (opponentSpec.kind === "model") {
        next = aiPlay(state, actor, {
          source: "model",
          model: opponentSpec.model,
        });
      } else if (opponentSpec.kind === "hybrid_play_model") {
        const traced = hybridPolicyPlayDetailed(state, actor, {
          model: opponentSpec.model,
          heuristicPolicy: String(opponentSpec.heuristicPolicy || ""),
          goStopPolicy: String(opponentSpec.goStopPolicy || ""),
          goStopOnly: !!opponentSpec.goStopOnly,
        });
        next = traced?.next || state;
      } else {
        next = aiPlay(state, actor, {
          source: "heuristic",
          heuristicPolicy: opponentSpec.heuristicPolicy,
        });
      }
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
          imitation.totals[decisionType] += 1;
          if (chosen === refCandidate) {
            imitation.matches[decisionType] += 1;
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
  };
}

function quantile(values, q) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.max(0, Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * q)));
  return sorted[idx];
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
  if (String(controlModel?.format_version || "").trim() !== "neat_python_genome_v1") {
    throw new Error("invalid --genome format: expected neat_python_genome_v1");
  }
  if (String(opts.opponentPolicy || "").trim()) {
    resolveOpponentSpec(opts.opponentPolicy, opts.opponentGenomePath);
  }
  const controlPlayMatchModelSpec =
    opts.controlPolicyMode === "hybrid_option_only"
      ? resolveControlPlayMatchModel(opts.controlPlayMatchModel)
      : null;
  for (const item of opts.opponentPolicyMix || []) {
    resolveOpponentSpec(item?.policy, opts.opponentGenomePath);
  }

  const games = Math.max(1, Math.floor(opts.games));
  const maxSteps = Math.max(20, Math.floor(opts.maxSteps));
  const evaluationSchedule = buildEvaluationSchedule(opts, games);
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

  for (let gi = 0; gi < games; gi += 1) {
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
    const gameResult = playSingleRound(
      roundStart,
      controlModel,
      seed,
      controlActor,
      opponentPolicyForGame,
      maxSteps,
      opts.opponentGenomePath,
      {
        controlPolicyMode: opts.controlPolicyMode,
        controlHeuristicPolicy: opts.controlHeuristicPolicy,
        controlPlayMatchModel: controlPlayMatchModelSpec?.model || null,
      }
    );
    const endState = gameResult?.endState || gameResult;
    const controlGoOpportunityCount = Math.max(0, Number(gameResult?.goOpportunityCount || 0));
    const afterGoldDiff = controlGoldDiff(endState, controlActor);
    const goldDelta = afterGoldDiff - beforeGoldDiff;
    goldDeltas.push(goldDelta);
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

    if (opts.continuousSeries) {
      seriesSession.previousEndState = endState;
    }
    seriesSession.roundsPlayed += 1;

    const winner = endState?.result?.winner || "unknown";
    if (winner === controlActor) {
      wins += 1;
    }
    else if (winner === opponentActor) losses += 1;
    else draws += 1;
    if (controlGoCount > 0 && winner !== controlActor) {
      goFailCount += 1;
    }

    const gt = gameResult?.imitation?.totals || {};
    const gm = gameResult?.imitation?.matches || {};
    for (const k of ["play", "match", "option"]) {
      simImitationTotals[k] += Number(gt[k] || 0);
      simImitationMatches[k] += Number(gm[k] || 0);
    }
  }

  const meanGoldDelta = goldDeltas.length > 0 ? goldDeltas.reduce((a, b) => a + b, 0) / goldDeltas.length : 0;
  const winRate = wins / games;
  const lossRate = losses / games;
  const drawRate = draws / games;
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

  const goldNorm = Math.tanh((meanGoldDelta - fitnessGoldNeutralDelta) / fitnessGoldScale);
  const expectedResultRaw =
    clamp01(winRate) + (0.5 * clamp01(drawRate)) - clamp01(lossRate);
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
    control_actor: controlActor,
    opponent_actor: opponentActor,
    opponent_policy: opts.opponentPolicy,
    opponent_policy_mix: opts.opponentPolicyMix,
    opponent_policy_counts: opponentPolicyCounts,
    control_policy_mode: opts.controlPolicyMode,
    control_heuristic_policy:
      opts.controlPolicyMode === "hybrid_play_match_only" || opts.controlPolicyMode === "hybrid_go_stop_only"
        ? opts.controlHeuristicPolicy
        : null,
    control_play_match_model:
      opts.controlPolicyMode === "hybrid_option_only"
        ? String(controlPlayMatchModelSpec?.label || opts.controlPlayMatchModel || "")
        : null,
    opponent_eval_tuning: {
      fast_path: false,
      imitation_reference_enabled: true,
      imitation_decision_scope:
        opts.controlPolicyMode === "hybrid_play_match_only"
          ? "model_owned_play_match_only"
          : opts.controlPolicyMode === "hybrid_go_stop_only"
            ? "model_owned_non_go_stop_only"
            : opts.controlPolicyMode === "hybrid_option_only"
              ? "model_owned_option_only"
            : "all_control_decisions",
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
    imitation_option_total: imitationTotals.option,
    imitation_option_matches: imitationMatches.option,
    imitation_option_ratio: imitationOptionRatio,
    imitation_weight_play: imitationWeights.play,
    imitation_weight_match: imitationWeights.match,
    imitation_weight_option: imitationWeights.option,
    imitation_weighted_score: imitationWeightedScore,
    fitness_components: {
      gold_norm: goldNorm,
      gold_neutral_delta: fitnessGoldNeutralDelta,
      win_norm: resultNorm,
      result_norm: resultNorm,
      result_expected: expectedResult,
      result_neutral: neutralExpectedResult,
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
