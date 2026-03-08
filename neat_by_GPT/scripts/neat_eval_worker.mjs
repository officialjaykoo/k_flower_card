// neat_eval_worker.mjs
// - Evaluates one genome against heuristic/mix opponents.
// - Emits one JSON summary line (stdout last line).
// - Fail-fast on missing/invalid required inputs.

import fs from "node:fs";
import path from "node:path";
import {
  initSimulationGame,
  startSimulationGame,
  createSeededRng,
  calculateScore,
  scoringPiCount
} from "../../src/engine/index.js";
import { getActionPlayerKey } from "../../src/engine/runner.js";
import { aiPlay } from "../../src/ai/aiPlay_by_GPT.js";
import { resolveBotPolicy } from "../../src/ai/policies.js";
import {
  canonicalOptionAction,
  selectDecisionPool,
  resolveDecisionType,
  legalCandidatesForDecision,
  applyDecisionAction,
  stateProgressKey
} from "../../src/ai/decisionRuntime_by_GPT.js";
import {
  getModelDecisionAnalysis,
  applyModelDecisionAnalysis
} from "../../src/ai/modelPolicyEngine_by_GPT.js";

// Quick Read Map (top-down):
// 1) parseArgs(): strict CLI parsing + validation
// 2) playSingleRound(): single-game simulation loop
// 3) main(): N-game aggregate -> fitness -> summary JSON

// =============================================================================
// Section 1. CLI
// =============================================================================
function normalizePolicyName(policy) {
  return String(policy || "").trim().toLowerCase();
}

function parseContinuousSeriesValue(value) {
  const raw = String(value ?? "1").trim();
  if (raw === "" || raw === "1") return true;
  if (raw === "2") return false;
  throw new Error(`invalid --continuous-series: ${raw} (allowed: 1=true, 2=false)`);
}

function parseArgs(argv) {
  const args = [...argv];
  const out = {
    genomePath: "",
    opponentGenomePath: "",
    games: 3,
    seed: "neat-python",
    maxSteps: 600,
    fitnessProfile: "phase1",
    teacherPolicy: "",
    opponentPolicy: "",
    opponentPolicyMix: [],
    firstTurnPolicy: "alternate",
    fixedFirstTurn: "human",
    continuousSeries: true,
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
    else if (key === "--fitness-profile") out.fitnessProfile = String(value || "phase1").trim().toLowerCase();
    else if (key === "--teacher-policy") out.teacherPolicy = String(value || "").trim();
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
    else if (key === "--continuous-series") out.continuousSeries = parseContinuousSeriesValue(value);
    else throw new Error(`Unknown argument: ${key}`);
  }

  if (!out.genomePath) throw new Error("--genome is required");
  if (
    out.fitnessProfile !== "phase1" &&
    out.fitnessProfile !== "phase2" &&
    out.fitnessProfile !== "phase3" &&
    out.fitnessProfile !== "focus"
  ) {
    throw new Error(`invalid --fitness-profile: ${out.fitnessProfile}`);
  }
  if (out.firstTurnPolicy !== "alternate" && out.firstTurnPolicy !== "fixed") {
    throw new Error(`invalid --first-turn-policy: ${out.firstTurnPolicy}`);
  }
  if (out.fixedFirstTurn !== "human" && out.fixedFirstTurn !== "ai") {
    throw new Error(`invalid --fixed-first-turn: ${out.fixedFirstTurn}`);
  }
  if (String(out.teacherPolicy || "").trim()) {
    const resolvedTeacherPolicy = resolveBotPolicy(out.teacherPolicy);
    if (!resolvedTeacherPolicy) {
      throw new Error(`invalid --teacher-policy: ${out.teacherPolicy}`);
    }
    out.teacherPolicy = resolvedTeacherPolicy;
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
  return out;
}

// =============================================================================
// Section 2. Round Simulation + Metrics
// =============================================================================
const DECISION_KEYS = ["play", "match", "option"];
const SCENARIO_KEYS = ["leading", "trailing", "pressure", "late", "option"];

function transitionKey(state) {
  return stateProgressKey(state, { includeKiboSeq: true });
}

function normalizeDecisionCandidate(decisionType, candidate) {
  if (decisionType === "option") return canonicalOptionAction(candidate);
  return String(candidate || "").trim();
}

function createDecisionCounters() {
  return { play: 0, match: 0, option: 0 };
}

function cloneDecisionCounters(src) {
  return {
    play: Number(src?.play || 0),
    match: Number(src?.match || 0),
    option: Number(src?.option || 0),
  };
}

function createImitationStats() {
  return {
    totals: createDecisionCounters(),
    matches: createDecisionCounters(),
    teacherProbSums: createDecisionCounters(),
    teacherProbEdges: createDecisionCounters(),
  };
}

function createScenarioAccumulator() {
  const out = {};
  for (const key of SCENARIO_KEYS) {
    out[key] = {
      games: 0,
      decisions: 0,
      resultSum: 0,
      goldDeltaSum: 0,
      localProxySum: 0,
    };
  }
  return out;
}

function createScenarioGameStats() {
  return {
    shards: createScenarioAccumulator(),
    touched: new Set(),
  };
}

function mergeImitationStats(target, src) {
  for (const key of DECISION_KEYS) {
    target.totals[key] += Number(src?.totals?.[key] || 0);
    target.matches[key] += Number(src?.matches?.[key] || 0);
    target.teacherProbSums[key] += Number(src?.teacherProbSums?.[key] || 0);
    target.teacherProbEdges[key] += Number(src?.teacherProbEdges?.[key] || 0);
  }
}

function mergeScenarioGameIntoTotals(target, gameStats, goldDelta, resultValue) {
  for (const key of SCENARIO_KEYS) {
    const shard = gameStats?.shards?.[key];
    if (!shard || Number(shard.decisions || 0) <= 0) continue;
    target[key].games += 1;
    target[key].decisions += Number(shard.decisions || 0);
    target[key].resultSum += Number(resultValue || 0);
    target[key].goldDeltaSum += Number(goldDelta || 0);
    target[key].localProxySum += Number(shard.localProxySum || 0);
  }
}

function heuristicCandidateForDecision(state, actor, decisionType, candidates, heuristicPolicy) {
  if (!heuristicPolicy || !Array.isArray(candidates) || candidates.length <= 0) return null;
  const nextByHeuristic = aiPlay(state, actor, {
    source: "heuristic",
    heuristicPolicy,
  });
  if (!nextByHeuristic || transitionKey(nextByHeuristic) === transitionKey(state)) {
    return null;
  }
  const target = transitionKey(nextByHeuristic);
  for (const candidate of candidates) {
    const simulated = applyDecisionAction(state, actor, decisionType, candidate);
    if (simulated && transitionKey(simulated) === target) {
      return normalizeDecisionCandidate(decisionType, candidate);
    }
  }
  return null;
}

function stateScoreSnapshot(state, actor) {
  const opp = actor === "human" ? "ai" : "human";
  const selfScore = calculateScore(state?.players?.[actor], state?.players?.[opp], state?.ruleKey);
  const oppScore = calculateScore(state?.players?.[opp], state?.players?.[actor], state?.ruleKey);
  const selfTotal = Number(selfScore?.total || 0);
  const oppTotal = Number(oppScore?.total || 0);
  return {
    scoreDiff: selfTotal - oppTotal,
    goldDiff: controlGoldDiff(state, actor),
    piDiff: Number(scoringPiCount(state?.players?.[actor]) || 0) - Number(scoringPiCount(state?.players?.[opp]) || 0),
    selfCanStop: selfTotal >= 7 ? 1 : 0,
    oppCanStop: oppTotal >= 7 ? 1 : 0,
    deckLen: Number(state?.deck?.length || 0),
    selfHandLen: Number(state?.players?.[actor]?.hand?.length || 0),
    oppHandLen: Number(state?.players?.[opp]?.hand?.length || 0),
  };
}

function localTransitionProxy(before, after) {
  return (
    (0.55 * Math.tanh((Number(after?.scoreDiff || 0) - Number(before?.scoreDiff || 0)) / 4.0)) +
    (0.25 * Math.tanh((Number(after?.goldDiff || 0) - Number(before?.goldDiff || 0)) / 2400.0)) +
    (0.20 * Math.tanh((Number(after?.piDiff || 0) - Number(before?.piDiff || 0)) / 3.0))
  );
}

function resolveScenarioTags(state, actor, decisionType) {
  const snapshot = stateScoreSnapshot(state, actor);
  const tags = [];
  if (snapshot.goldDiff >= 1600 || snapshot.scoreDiff >= 3) tags.push("leading");
  if (snapshot.goldDiff <= -1200 || snapshot.scoreDiff <= -2) tags.push("trailing");
  if (
    snapshot.oppCanStop > 0 ||
    snapshot.goldDiff <= -600 ||
    snapshot.scoreDiff <= -1 ||
    snapshot.deckLen <= 10
  ) {
    tags.push("pressure");
  }
  if (snapshot.deckLen <= 8 || snapshot.selfHandLen <= 3 || snapshot.oppHandLen <= 3) {
    tags.push("late");
  }
  if (decisionType === "option") tags.push("option");
  return [...new Set(tags)];
}

function buildImitationMetrics(stats, weights = null) {
  const totals = cloneDecisionCounters(stats?.totals);
  const matches = cloneDecisionCounters(stats?.matches);
  const teacherProbSums = cloneDecisionCounters(stats?.teacherProbSums);
  const teacherProbEdges = cloneDecisionCounters(stats?.teacherProbEdges);
  const chosenWeights = weights || { play: 0.45, match: 0.20, option: 0.35 };
  const out = {
    totals,
    matches,
    teacherProbSums,
    teacherProbEdges,
    weights: {
      play: Number(chosenWeights.play || 0),
      match: Number(chosenWeights.match || 0),
      option: Number(chosenWeights.option || 0),
    },
  };
  let weightedAgreementAcc = 0;
  let weightedProbAcc = 0;
  let weightedEdgeAcc = 0;
  let weightSum = 0;
  for (const key of DECISION_KEYS) {
    const total = Number(totals[key] || 0);
    const agreement = total > 0 ? Number(matches[key] || 0) / total : 0;
    const teacherProb = total > 0 ? Number(teacherProbSums[key] || 0) / total : 0;
    const teacherProbEdge = total > 0 ? Number(teacherProbEdges[key] || 0) / total : 0;
    out[`${key}Ratio`] = agreement;
    out[`${key}TeacherProb`] = teacherProb;
    out[`${key}TeacherProbEdge`] = teacherProbEdge;
    if (total > 0) {
      const weight = Number(out.weights[key] || 0);
      weightedAgreementAcc += weight * agreement;
      weightedProbAcc += weight * teacherProb;
      weightedEdgeAcc += weight * teacherProbEdge;
      weightSum += weight;
    }
  }
  out.weightSum = weightSum;
  out.weightedAgreement = weightSum > 0 ? weightedAgreementAcc / weightSum : 0;
  out.weightedTeacherProb = weightSum > 0 ? weightedProbAcc / weightSum : 0;
  out.weightedTeacherProbEdge = weightSum > 0 ? weightedEdgeAcc / weightSum : 0;
  out.weightedScore = (
    (0.65 * out.weightedAgreement) +
    (0.35 * clamp01(0.5 + out.weightedTeacherProbEdge))
  );
  return out;
}

function buildScenarioMetrics(totals, profile) {
  const shardWeights = profile?.scenarioShardWeights || {};
  const goldWeight = Number(profile?.scenarioGoldWeight || 0);
  const resultWeight = Number(profile?.scenarioResultWeight || 0);
  const localWeight = Number(profile?.scenarioLocalWeight || 0);
  const reliabilityGames = Math.max(1, Number(profile?.scenarioReliabilityGames || 1));
  const reliabilityDecisions = Math.max(1, Number(profile?.scenarioReliabilityDecisions || 1));
  const perShard = {};
  let weightedScoreAcc = 0;
  let weightAcc = 0;

  for (const key of SCENARIO_KEYS) {
    const shard = totals?.[key] || {};
    const games = Number(shard.games || 0);
    const decisions = Number(shard.decisions || 0);
    const meanGoldDelta = games > 0 ? Number(shard.goldDeltaSum || 0) / games : 0;
    const meanResult = games > 0 ? Number(shard.resultSum || 0) / games : 0;
    const meanLocalProxy = decisions > 0 ? Number(shard.localProxySum || 0) / decisions : 0;
    const reliability = Math.sqrt(
      clamp01(games / reliabilityGames) * clamp01(decisions / reliabilityDecisions)
    );
    const shardScore =
      (goldWeight * Math.tanh(meanGoldDelta / 2200.0)) +
      (resultWeight * Math.max(-1, Math.min(1, meanResult))) +
      (localWeight * Math.max(-1, Math.min(1, meanLocalProxy)));
    const shardWeight = Number(shardWeights[key] || 0);
    const effectiveWeight = shardWeight * reliability;
    perShard[key] = {
      games,
      decisions,
      mean_gold_delta: meanGoldDelta,
      mean_result: meanResult,
      mean_local_proxy: meanLocalProxy,
      reliability,
      weight: shardWeight,
      score: shardScore,
      effective_weight: effectiveWeight,
    };
    if (effectiveWeight > 0) {
      weightedScoreAcc += effectiveWeight * shardScore;
      weightAcc += effectiveWeight;
    }
  }

  return {
    weights: shardWeights,
    weightedScore: weightAcc > 0 ? weightedScoreAcc / weightAcc : 0,
    weightSum: weightAcc,
    shards: perShard,
  };
}

function selectOpponentPolicyForGame(opts, gameIndex) {
  const fixedPolicy = String(opts.opponentPolicy || "").trim();
  if (fixedPolicy) {
    return fixedPolicy;
  }
  if (!Array.isArray(opts.opponentPolicyMix) || opts.opponentPolicyMix.length <= 0) {
    return "";
  }
  const tokenRng = createSeededRng(`${opts.seed}|opponent_mix|g=${gameIndex}`);
  const needle = Number(tokenRng() || 0);
  let total = 0;
  for (const item of opts.opponentPolicyMix) {
    total += Number(item.weight || 0);
  }
  if (!Number.isFinite(total) || total <= 0) {
    throw new Error("invalid opponent_policy_mix total weight");
  }
  const target = needle * total;
  let acc = 0;
  for (const item of opts.opponentPolicyMix) {
    acc += Number(item.weight || 0);
    if (target <= acc) {
      return String(item.policy || "").trim();
    }
  }
  return String(opts.opponentPolicyMix[opts.opponentPolicyMix.length - 1].policy || "").trim();
}
function resolveFirstTurnKey(opts, gameIndex) {
  if (opts.firstTurnPolicy === "fixed") return opts.fixedFirstTurn;
  return gameIndex % 2 === 0 ? "ai" : "human";
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

function playSingleRound(
  initialState,
  controlModel,
  seed,
  controlActor,
  opponentPolicy,
  maxSteps,
  opponentModel,
  teacherPolicy = null
) {
  let state = initialState;
  const imitation = createImitationStats();
  const scenarioGameStats = createScenarioGameStats();

  let steps = 0;
  while (state.phase !== "resolution" && steps < maxSteps) {
    const actor = getActionPlayerKey(state);
    if (!actor) break;

    const before = transitionKey(state);
    let next = state;
    let actionSource = "heuristic";

    if (actor === controlActor) {
      actionSource = "model_control";
      const pool = selectDecisionPool(state, actor);
      const decisionType = resolveDecisionType(pool);
      const candidates = decisionType
        ? legalCandidatesForDecision(pool, decisionType).map((candidate) => normalizeDecisionCandidate(decisionType, candidate))
        : [];
      const scenarioTags = decisionType ? resolveScenarioTags(state, actor, decisionType) : [];
      const beforeSnapshot = stateScoreSnapshot(state, actor);
      const analysis = getModelDecisionAnalysis(state, actor, controlModel);
      if (!analysis || !decisionType || candidates.length <= 0) {
        throw new Error(
          `control analysis unresolved: seed=${seed}, step=${steps}, actor=${actor}, phase=${String(state?.phase || "")}`
        );
      }
      next = applyModelDecisionAnalysis(state, actor, analysis);

      const teacherCandidate = heuristicCandidateForDecision(
        state,
        actor,
        decisionType,
        candidates,
        teacherPolicy
      );
      if (teacherCandidate) {
        const chosenCandidate = normalizeDecisionCandidate(decisionType, analysis.chosenCandidate);
        const teacherProb = Number(analysis?.probabilities?.[teacherCandidate] || 0);
        const uniformProb = candidates.length > 0 ? 1 / candidates.length : 0;
        imitation.totals[decisionType] += 1;
        imitation.teacherProbSums[decisionType] += teacherProb;
        imitation.teacherProbEdges[decisionType] += teacherProb - uniformProb;
        if (chosenCandidate === teacherCandidate) {
          imitation.matches[decisionType] += 1;
        }
      }

      if (scenarioTags.length > 0) {
        const afterSnapshot = stateScoreSnapshot(next, actor);
        const proxy = localTransitionProxy(beforeSnapshot, afterSnapshot);
        for (const tag of scenarioTags) {
          scenarioGameStats.touched.add(tag);
          scenarioGameStats.shards[tag].decisions += 1;
          scenarioGameStats.shards[tag].localProxySum += proxy;
        }
      }
    } else if (normalizePolicyName(opponentPolicy) === "genome") {
      actionSource = "model_opponent";
      next = aiPlay(state, actor, {
        source: "model",
        model: opponentModel,
      });
    } else {
      next = aiPlay(state, actor, {
        source: "heuristic",
        heuristicPolicy: opponentPolicy,
      });
    }

    if (!next || transitionKey(next) === before) {
      throw new Error(
        `action resolution failed: seed=${seed}, step=${steps}, actor=${actor}, phase=${String(state?.phase || "")}, policy=${String(opponentPolicy || "")}, source=${actionSource}`
      );
    }

    state = next;
    steps += 1;
  }

  return {
    endState: state,
    imitation,
    scenarioGameStats,
  };
}

function quantile(values, q) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.max(0, Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * q)));
  return sorted[idx];
}

function mean(values) {
  if (!Array.isArray(values) || values.length <= 0) return 0;
  return values.reduce((acc, v) => acc + Number(v || 0), 0) / values.length;
}

function tailMean(values, ratio) {
  if (!Array.isArray(values) || values.length <= 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const r = Math.max(0.01, Math.min(1.0, Number(ratio || 0.1)));
  const n = Math.max(1, Math.ceil(sorted.length * r));
  const slice = sorted.slice(0, n);
  return mean(slice);
}

function downsideSemiDeviation(values, center = 0) {
  if (!Array.isArray(values) || values.length <= 0) return 0;
  const c = Number(center || 0);
  let acc = 0;
  for (const raw of values) {
    const d = Number(raw || 0) - c;
    if (d < 0) acc += d * d;
  }
  return Math.sqrt(acc / values.length);
}

function rateAtOrBelow(values, threshold) {
  if (!Array.isArray(values) || values.length <= 0) return 0;
  const t = Number(threshold || 0);
  let hit = 0;
  for (const raw of values) {
    if (Number(raw || 0) <= t) hit += 1;
  }
  return hit / values.length;
}

function clamp01(v) {
  const x = Number(v || 0);
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  return x;
}
// =============================================================================
// Section 4. Entrypoint
// =============================================================================
function main() {
  // 4-1) Parse options and load genome/opponent models.
  const evalStartMs = Date.now();
  const opts = parseArgs(process.argv.slice(2));
  const full = path.resolve(opts.genomePath);
  if (!fs.existsSync(full)) throw new Error(`genome not found: ${opts.genomePath}`);

  const controlModel = JSON.parse(fs.readFileSync(full, "utf8"));
  if (String(controlModel?.format_version || "").trim() !== "neat_python_genome_v1") {
    throw new Error("invalid --genome format: expected neat_python_genome_v1");
  }
  let opponentModel = null;
  const needsOpponentModel =
    String(opts.opponentPolicy || "").trim().toLowerCase() === "genome" ||
    (Array.isArray(opts.opponentPolicyMix) &&
      opts.opponentPolicyMix.some((item) => String(item?.policy || "").trim().toLowerCase() === "genome"));
  if (needsOpponentModel) {
    const oppFull = path.resolve(opts.opponentGenomePath);
    if (!fs.existsSync(oppFull)) {
      throw new Error(`opponent genome not found: ${opts.opponentGenomePath}`);
    }
    opponentModel = JSON.parse(fs.readFileSync(oppFull, "utf8"));
    if (String(opponentModel?.format_version || "").trim() !== "neat_python_genome_v1") {
      throw new Error("invalid --opponent-genome format: expected neat_python_genome_v1");
    }
  }
  const FITNESS_PROFILES = {
    phase1: {
      modelName: "gold_tail_guard_profile_phase1_v3",
      goldMeanNeutral: -250.0,
      goldMeanScale: 3200.0,
      goldP50Neutral: -800.0,
      goldP50Scale: 3500.0,
      goldP10Neutral: -2600.0,
      goldP10Scale: 5200.0,
      goldCvar10Neutral: -4200.0,
      goldCvar10Scale: 7200.0,
      goldMeanWeight: 0.50,
      goldP50Weight: 0.16,
      goldP10Weight: 0.18,
      goldCvar10Weight: 0.16,
      tieBreakWeight: 0.02,
      tieBreakExpectedNeutral: -0.18,
      bankruptPenaltyWeight: 0.10,
      catastrophicLossThreshold: -7000.0,
      catastrophicLossWeight: 0.08,
      downsideSemiScale: 5200.0,
      downsideSemiWeight: 0.04,
      goFailRateCap: 0.46,
      goFailExcessWeight: 0.03,
      seatGapWeight: 0.03,
      seatGoldGapScale: 4500.0,
      inflictedBankruptBonusWeight: 0.015,
      goZeroForceScore: -3.0,
      teacherPolicy: null,
      imitationWeights: { play: 0.45, match: 0.20, option: 0.35 },
      imitationAgreementWeight: 0.0,
      imitationProbWeight: 0.0,
      scenarioShardWeights: {},
      scenarioGoldWeight: 0.0,
      scenarioResultWeight: 0.0,
      scenarioLocalWeight: 0.0,
      scenarioReliabilityGames: 18,
      scenarioReliabilityDecisions: 24,
    },
    phase2: {
      modelName: "gold_tail_guard_profile_phase2_v3",
      goldMeanNeutral: -100.0,
      goldMeanScale: 3000.0,
      goldP50Neutral: -500.0,
      goldP50Scale: 3200.0,
      goldP10Neutral: -2200.0,
      goldP10Scale: 4800.0,
      goldCvar10Neutral: -2800.0,
      goldCvar10Scale: 6200.0,
      goldMeanWeight: 0.44,
      goldP50Weight: 0.16,
      goldP10Weight: 0.20,
      goldCvar10Weight: 0.20,
      tieBreakWeight: 0.03,
      tieBreakExpectedNeutral: -0.14,
      bankruptPenaltyWeight: 0.14,
      catastrophicLossThreshold: -6400.0,
      catastrophicLossWeight: 0.12,
      downsideSemiScale: 4600.0,
      downsideSemiWeight: 0.05,
      goFailRateCap: 0.42,
      goFailExcessWeight: 0.04,
      seatGapWeight: 0.04,
      seatGoldGapScale: 4200.0,
      inflictedBankruptBonusWeight: 0.015,
      goZeroForceScore: -3.0,
      teacherPolicy: null,
      imitationWeights: { play: 0.45, match: 0.20, option: 0.35 },
      imitationAgreementWeight: 0.0,
      imitationProbWeight: 0.0,
      scenarioShardWeights: {},
      scenarioGoldWeight: 0.0,
      scenarioResultWeight: 0.0,
      scenarioLocalWeight: 0.0,
      scenarioReliabilityGames: 18,
      scenarioReliabilityDecisions: 24,
    },
    phase3: {
      modelName: "gold_tail_guard_profile_phase3_v3",
      goldMeanNeutral: 0.0,
      goldMeanScale: 2800.0,
      goldP50Neutral: 0.0,
      goldP50Scale: 2800.0,
      goldP10Neutral: -1800.0,
      goldP10Scale: 4200.0,
      goldCvar10Neutral: -2200.0,
      goldCvar10Scale: 5000.0,
      goldMeanWeight: 0.34,
      goldP50Weight: 0.14,
      goldP10Weight: 0.22,
      goldCvar10Weight: 0.30,
      tieBreakWeight: 0.04,
      tieBreakExpectedNeutral: -0.08,
      bankruptPenaltyWeight: 0.18,
      catastrophicLossThreshold: -5600.0,
      catastrophicLossWeight: 0.16,
      downsideSemiScale: 4000.0,
      downsideSemiWeight: 0.07,
      goFailRateCap: 0.38,
      goFailExcessWeight: 0.05,
      seatGapWeight: 0.05,
      seatGoldGapScale: 3600.0,
      inflictedBankruptBonusWeight: 0.02,
      goZeroForceScore: -3.2,
      teacherPolicy: null,
      imitationWeights: { play: 0.45, match: 0.20, option: 0.35 },
      imitationAgreementWeight: 0.0,
      imitationProbWeight: 0.0,
      scenarioShardWeights: {},
      scenarioGoldWeight: 0.0,
      scenarioResultWeight: 0.0,
      scenarioLocalWeight: 0.0,
      scenarioReliabilityGames: 18,
      scenarioReliabilityDecisions: 24,
    },
    focus: {
      modelName: "gold_tail_guard_focus_cl_v3_sharded",
      goldMeanNeutral: 0.0,
      goldMeanScale: 2600.0,
      goldP50Neutral: 0.0,
      goldP50Scale: 2600.0,
      goldP10Neutral: -1600.0,
      goldP10Scale: 3800.0,
      goldCvar10Neutral: -2000.0,
      goldCvar10Scale: 4500.0,
      goldMeanWeight: 0.28,
      goldP50Weight: 0.14,
      goldP10Weight: 0.24,
      goldCvar10Weight: 0.34,
      tieBreakWeight: 0.04,
      tieBreakExpectedNeutral: -0.05,
      bankruptPenaltyWeight: 0.20,
      catastrophicLossThreshold: -5000.0,
      catastrophicLossWeight: 0.22,
      downsideSemiScale: 3600.0,
      downsideSemiWeight: 0.08,
      goFailRateCap: 0.36,
      goFailExcessWeight: 0.06,
      seatGapWeight: 0.05,
      seatGoldGapScale: 3200.0,
      inflictedBankruptBonusWeight: 0.012,
      goZeroForceScore: -3.0,
      teacherPolicy: null,
      imitationWeights: { play: 0.45, match: 0.20, option: 0.35 },
      imitationAgreementWeight: 0.05,
      imitationProbWeight: 0.08,
      scenarioShardWeights: {
        leading: 0.08,
        trailing: 0.30,
        pressure: 0.26,
        late: 0.18,
        option: 0.18,
      },
      scenarioGoldWeight: 0.48,
      scenarioResultWeight: 0.20,
      scenarioLocalWeight: 0.32,
      scenarioReliabilityGames: 18,
      scenarioReliabilityDecisions: 24,
    },
  };
  const profile = FITNESS_PROFILES[opts.fitnessProfile];
  if (!profile) {
    throw new Error(`unsupported fitness profile: ${String(opts.fitnessProfile || "")}`);
  }
  const effectiveTeacherPolicy = String(opts.teacherPolicy || profile.teacherPolicy || "").trim() || null;

  const games = Math.max(1, Math.floor(opts.games));
  const maxSteps = Math.max(20, Math.floor(opts.maxSteps));
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
  let goCount = 0;
  let goGames = 0;
  let goFailCount = 0;
  const firstTurnCounts = {
    human: 0,
    ai: 0,
  };
  const opponentPolicyCounts = {};
  const controlSeatStats = {
    first: { games: 0, wins: 0, goldDeltas: [] },
    second: { games: 0, wins: 0, goldDeltas: [] },
  };
  const seriesSession = {
    roundsPlayed: 0,
    previousEndState: null,
  };
  const simImitationStats = createImitationStats();
  const simScenarioTotals = createScenarioAccumulator();

  // 4-2) Run per-game simulations and accumulate metrics.
  for (let gi = 0; gi < games; gi += 1) {
    const firstTurnKey = resolveFirstTurnKey(opts, gi);
    const opponentPolicyForGame = selectOpponentPolicyForGame(opts, gi);
    if (!opponentPolicyForGame) {
      throw new Error("resolved empty opponent policy");
    }
    opponentPolicyCounts[opponentPolicyForGame] = Number(opponentPolicyCounts[opponentPolicyForGame] || 0) + 1;
    firstTurnCounts[firstTurnKey] += 1;
    const controlSeat = firstTurnKey === controlActor ? "first" : "second";
    controlSeatStats[controlSeat].games += 1;
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
      opponentModel,
      effectiveTeacherPolicy
    );
    const endState = gameResult.endState;
    const afterGoldDiff = controlGoldDiff(endState, controlActor);
    const goldDelta = afterGoldDiff - beforeGoldDiff;
    goldDeltas.push(goldDelta);
    controlSeatStats[controlSeat].goldDeltas.push(goldDelta);
    const controlGold = Number(endState?.players?.[controlActor]?.gold || 0);
    const opponentGold = Number(endState?.players?.[opponentActor]?.gold || 0);
    const controlBankrupt = controlGold <= 0;
    const opponentBankrupt = opponentGold <= 0;
    const controlGoCount = Math.max(0, Number(endState?.players?.[controlActor]?.goCount || 0));
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
      controlSeatStats[controlSeat].wins += 1;
    }
    else if (winner === opponentActor) losses += 1;
    else draws += 1;
    if (controlGoCount > 0 && winner !== controlActor) {
      goFailCount += 1;
    }
    mergeImitationStats(simImitationStats, gameResult.imitation);
    const resultValue = winner === controlActor ? 1 : winner === opponentActor ? -1 : 0;
    mergeScenarioGameIntoTotals(simScenarioTotals, gameResult.scenarioGameStats, goldDelta, resultValue);

  }

  // 4-3) Normalize aggregate metrics for fitness computation.
  const meanGoldDelta = goldDeltas.length > 0 ? goldDeltas.reduce((a, b) => a + b, 0) / goldDeltas.length : 0;
  const p10GoldDelta = quantile(goldDeltas, 0.1);
  const p50GoldDelta = quantile(goldDeltas, 0.5);
  const p90GoldDelta = quantile(goldDeltas, 0.9);
  const winRate = wins / games;
  const lossRate = losses / games;
  const drawRate = draws / games;
  const goRate = goGames / games;
  const goFailRate = goGames > 0 ? goFailCount / goGames : 0;
  const FITNESS_GOLD_MEAN_NEUTRAL = profile.goldMeanNeutral;
  const FITNESS_GOLD_MEAN_SCALE = profile.goldMeanScale;
  const FITNESS_GOLD_P50_NEUTRAL = profile.goldP50Neutral;
  const FITNESS_GOLD_P50_SCALE = profile.goldP50Scale;
  const FITNESS_GOLD_P10_NEUTRAL = profile.goldP10Neutral;
  const FITNESS_GOLD_P10_SCALE = profile.goldP10Scale;
  const FITNESS_GOLD_CVAR10_NEUTRAL = profile.goldCvar10Neutral;
  const FITNESS_GOLD_CVAR10_SCALE = profile.goldCvar10Scale;
  const FITNESS_GOLD_MEAN_WEIGHT = profile.goldMeanWeight;
  const FITNESS_GOLD_P50_WEIGHT = profile.goldP50Weight;
  const FITNESS_GOLD_P10_WEIGHT = profile.goldP10Weight;
  const FITNESS_GOLD_CVAR10_WEIGHT = profile.goldCvar10Weight;
  const FITNESS_TIE_BREAK_WEIGHT = profile.tieBreakWeight;
  const FITNESS_TIE_BREAK_EXPECTED_NEUTRAL = profile.tieBreakExpectedNeutral;
  const FITNESS_BANKRUPT_PENALTY_WEIGHT = profile.bankruptPenaltyWeight;
  const FITNESS_CATASTROPHIC_LOSS_THRESHOLD = profile.catastrophicLossThreshold;
  const FITNESS_CATASTROPHIC_LOSS_WEIGHT = profile.catastrophicLossWeight;
  const FITNESS_DOWNSIDE_SEMI_SCALE = profile.downsideSemiScale;
  const FITNESS_DOWNSIDE_SEMI_WEIGHT = profile.downsideSemiWeight;
  const FITNESS_GO_FAIL_RATE_CAP = profile.goFailRateCap;
  const FITNESS_GO_FAIL_EXCESS_WEIGHT = profile.goFailExcessWeight;
  const FITNESS_SEAT_GAP_WEIGHT = profile.seatGapWeight;
  const FITNESS_SEAT_GOLD_GAP_SCALE = profile.seatGoldGapScale;
  const FITNESS_INFLICTED_BANKRUPT_BONUS_WEIGHT = profile.inflictedBankruptBonusWeight;
  const FITNESS_GO_ZERO_FORCE_SCORE = profile.goZeroForceScore;
  const FITNESS_IMITATION_AGREEMENT_WEIGHT = Number(profile.imitationAgreementWeight || 0);
  const FITNESS_IMITATION_PROB_WEIGHT = Number(profile.imitationProbWeight || 0);

  const expectedResultRaw =
    clamp01(winRate) + (0.5 * clamp01(drawRate)) - clamp01(lossRate);
  const expectedResult = Math.max(-1.0, Math.min(1.0, expectedResultRaw));
  const cvar10GoldDelta = tailMean(goldDeltas, 0.1);
  const cvar20GoldDelta = tailMean(goldDeltas, 0.2);
  const catastrophicLossRate = rateAtOrBelow(goldDeltas, FITNESS_CATASTROPHIC_LOSS_THRESHOLD);
  const downsideSemiRaw = downsideSemiDeviation(goldDeltas, meanGoldDelta);
  const downsideSemiNorm = clamp01(downsideSemiRaw / Math.max(1.0, FITNESS_DOWNSIDE_SEMI_SCALE));

  const goldMeanNorm = Math.tanh((meanGoldDelta - FITNESS_GOLD_MEAN_NEUTRAL) / FITNESS_GOLD_MEAN_SCALE);
  const goldP50Norm = Math.tanh((p50GoldDelta - FITNESS_GOLD_P50_NEUTRAL) / FITNESS_GOLD_P50_SCALE);
  const goldP10Norm = Math.tanh((p10GoldDelta - FITNESS_GOLD_P10_NEUTRAL) / FITNESS_GOLD_P10_SCALE);
  const goldCvar10Norm = Math.tanh((cvar10GoldDelta - FITNESS_GOLD_CVAR10_NEUTRAL) / FITNESS_GOLD_CVAR10_SCALE);
  const goldCore =
    (FITNESS_GOLD_MEAN_WEIGHT * goldMeanNorm) +
    (FITNESS_GOLD_P50_WEIGHT * goldP50Norm) +
    (FITNESS_GOLD_P10_WEIGHT * goldP10Norm) +
    (FITNESS_GOLD_CVAR10_WEIGHT * goldCvar10Norm);

  const tieBreak = FITNESS_TIE_BREAK_WEIGHT * (expectedResult - FITNESS_TIE_BREAK_EXPECTED_NEUTRAL);
  const bankruptRate = games > 0 ? clamp01(bankrupt.my_bankrupt_count / games) : 0;
  const bankruptPenalty = FITNESS_BANKRUPT_PENALTY_WEIGHT * bankruptRate;
  const catastrophicLossPenalty = FITNESS_CATASTROPHIC_LOSS_WEIGHT * catastrophicLossRate;
  const downsideSemiPenalty = FITNESS_DOWNSIDE_SEMI_WEIGHT * downsideSemiNorm;
  const goFailExcess = goGames > 0 ? Math.max(0, goFailRate - FITNESS_GO_FAIL_RATE_CAP) : 0;
  const goFailPenalty = FITNESS_GO_FAIL_EXCESS_WEIGHT * goFailExcess;
  const inflictedBankruptRate = games > 0 ? clamp01(bankrupt.my_inflicted_bankrupt_count / games) : 0;
  const inflictedBankruptBonus = FITNESS_INFLICTED_BANKRUPT_BONUS_WEIGHT * inflictedBankruptRate;

  const seatFirstWinRate = controlSeatStats.first.games > 0
    ? controlSeatStats.first.wins / controlSeatStats.first.games
    : 0;
  const seatSecondWinRate = controlSeatStats.second.games > 0
    ? controlSeatStats.second.wins / controlSeatStats.second.games
    : 0;
  const seatWinRateGap = Math.abs(seatFirstWinRate - seatSecondWinRate);
  const seatFirstMeanGoldDelta = mean(controlSeatStats.first.goldDeltas);
  const seatSecondMeanGoldDelta = mean(controlSeatStats.second.goldDeltas);
  const seatMeanGoldGap = Math.abs(seatFirstMeanGoldDelta - seatSecondMeanGoldDelta);
  const seatMeanGoldGapNorm = clamp01(
    Math.tanh(seatMeanGoldGap / Math.max(1.0, Number(FITNESS_SEAT_GOLD_GAP_SCALE || 1.0)))
  );
  const seatGapPenalty = FITNESS_SEAT_GAP_WEIGHT * ((0.6 * seatWinRateGap) + (0.4 * seatMeanGoldGapNorm));
  const simImitation = buildImitationMetrics(simImitationStats, profile.imitationWeights);
  const imitationAgreementScore = Math.max(-1, Math.min(1, (2.0 * Number(simImitation.weightedAgreement || 0)) - 1.0));
  const imitationProbScore = Math.max(-1, Math.min(1, Number(simImitation.weightedTeacherProbEdge || 0) / 0.35));
  const imitationBonus =
    (FITNESS_IMITATION_AGREEMENT_WEIGHT * imitationAgreementScore) +
    (FITNESS_IMITATION_PROB_WEIGHT * imitationProbScore);
  const scenarioMetrics = buildScenarioMetrics(simScenarioTotals, profile);
  const scenarioShardBonus = Number(scenarioMetrics.weightedScore || 0);

  const goZeroForceEnabled = Number.isFinite(FITNESS_GO_ZERO_FORCE_SCORE);
  const goZeroHardFail = goZeroForceEnabled && goGames === 0;
  let fitness = goZeroHardFail
    ? FITNESS_GO_ZERO_FORCE_SCORE
    : (
      goldCore +
      tieBreak -
      bankruptPenalty -
      catastrophicLossPenalty -
      downsideSemiPenalty -
      goFailPenalty -
      seatGapPenalty +
      inflictedBankruptBonus +
      imitationBonus +
      scenarioShardBonus
    );

  // 4-4) Build final summary contract consumed by phase scripts.
  const summary = {
    games,
    control_actor: controlActor,
    opponent_actor: opponentActor,
    opponent_policy: opts.opponentPolicy,
    opponent_policy_mix: opts.opponentPolicyMix,
    opponent_policy_counts: opponentPolicyCounts,
    opponent_eval_tuning: {
      fast_path: false,
      opponent_heuristic_params: null,
      imitation_teacher_policy: effectiveTeacherPolicy,
      scenario_shard_enabled: Object.keys(profile.scenarioShardWeights || {}).length > 0,
    },
    opponent_genome: String(opts.opponentGenomePath || "") || null,
    first_turn_policy: opts.firstTurnPolicy,
    fixed_first_turn: opts.firstTurnPolicy === "fixed" ? opts.fixedFirstTurn : null,
    first_turn_counts: firstTurnCounts,
    continuous_series: !!opts.continuousSeries,
    bankrupt,
    session_rounds: {
      control_actor_series: seriesSession.roundsPlayed,
    },
    wins,
    losses,
    draws,
    win_rate: winRate,
    loss_rate: lossRate,
    draw_rate: drawRate,
    go_count: goCount,
    go_fail_count: goFailCount,
    go_fail_rate: goFailRate,
    go_games: goGames,
    go_rate: goRate,
    mean_gold_delta: meanGoldDelta,
    p10_gold_delta: p10GoldDelta,
    p50_gold_delta: p50GoldDelta,
    p90_gold_delta: p90GoldDelta,
    cvar10_gold_delta: cvar10GoldDelta,
    cvar20_gold_delta: cvar20GoldDelta,
    catastrophic_loss_rate: catastrophicLossRate,
    seat_first_win_rate: seatFirstWinRate,
    seat_second_win_rate: seatSecondWinRate,
    seat_win_rate_gap: seatWinRateGap,
    seat_first_mean_gold_delta: seatFirstMeanGoldDelta,
    seat_second_mean_gold_delta: seatSecondMeanGoldDelta,
    seat_mean_gold_gap: seatMeanGoldGap,
    fitness_model: profile.modelName,
    fitness_profile: opts.fitnessProfile,
    fitness_gold_scale: FITNESS_GOLD_MEAN_SCALE,
    fitness_gold_neutral_delta: FITNESS_GOLD_MEAN_NEUTRAL,
    fitness_gold_p50_scale: FITNESS_GOLD_P50_SCALE,
    fitness_gold_p50_neutral_delta: FITNESS_GOLD_P50_NEUTRAL,
    fitness_gold_p10_scale: FITNESS_GOLD_P10_SCALE,
    fitness_gold_p10_neutral_delta: FITNESS_GOLD_P10_NEUTRAL,
    fitness_gold_cvar10_scale: FITNESS_GOLD_CVAR10_SCALE,
    fitness_gold_cvar10_neutral_delta: FITNESS_GOLD_CVAR10_NEUTRAL,
    fitness_tie_break_expected_neutral: FITNESS_TIE_BREAK_EXPECTED_NEUTRAL,
    fitness_win_weight: FITNESS_TIE_BREAK_WEIGHT,
    fitness_gold_weight: 1.0,
    fitness_bankrupt_penalty_weight: FITNESS_BANKRUPT_PENALTY_WEIGHT,
    fitness_catastrophic_loss_threshold: FITNESS_CATASTROPHIC_LOSS_THRESHOLD,
    fitness_catastrophic_loss_weight: FITNESS_CATASTROPHIC_LOSS_WEIGHT,
    fitness_downside_semi_scale: FITNESS_DOWNSIDE_SEMI_SCALE,
    fitness_downside_semi_weight: FITNESS_DOWNSIDE_SEMI_WEIGHT,
    fitness_go_fail_rate_cap: FITNESS_GO_FAIL_RATE_CAP,
    fitness_go_fail_excess_weight: FITNESS_GO_FAIL_EXCESS_WEIGHT,
    fitness_seat_gap_weight: FITNESS_SEAT_GAP_WEIGHT,
    fitness_seat_gold_gap_scale: FITNESS_SEAT_GOLD_GAP_SCALE,
    fitness_inflicted_bankrupt_bonus_weight: FITNESS_INFLICTED_BANKRUPT_BONUS_WEIGHT,
    imitation_source: effectiveTeacherPolicy,
    imitation_play_total: Number(simImitation.totals.play || 0),
    imitation_play_matches: Number(simImitation.matches.play || 0),
    imitation_play_ratio: Number(simImitation.playRatio || 0),
    imitation_play_teacher_prob: Number(simImitation.playTeacherProb || 0),
    imitation_play_teacher_prob_edge: Number(simImitation.playTeacherProbEdge || 0),
    imitation_match_total: Number(simImitation.totals.match || 0),
    imitation_match_matches: Number(simImitation.matches.match || 0),
    imitation_match_ratio: Number(simImitation.matchRatio || 0),
    imitation_match_teacher_prob: Number(simImitation.matchTeacherProb || 0),
    imitation_match_teacher_prob_edge: Number(simImitation.matchTeacherProbEdge || 0),
    imitation_option_total: Number(simImitation.totals.option || 0),
    imitation_option_matches: Number(simImitation.matches.option || 0),
    imitation_option_ratio: Number(simImitation.optionRatio || 0),
    imitation_option_teacher_prob: Number(simImitation.optionTeacherProb || 0),
    imitation_option_teacher_prob_edge: Number(simImitation.optionTeacherProbEdge || 0),
    imitation_weight_play: Number(simImitation.weights.play || 0),
    imitation_weight_match: Number(simImitation.weights.match || 0),
    imitation_weight_option: Number(simImitation.weights.option || 0),
    imitation_weighted_agreement: Number(simImitation.weightedAgreement || 0),
    imitation_weighted_teacher_prob: Number(simImitation.weightedTeacherProb || 0),
    imitation_weighted_teacher_prob_edge: Number(simImitation.weightedTeacherProbEdge || 0),
    imitation_weighted_score: Number(simImitation.weightedScore || 0),
    scenario_shard_weighted_score: Number(scenarioMetrics.weightedScore || 0),
    scenario_shards: scenarioMetrics.shards,
    fitness_components: {
      gold_mean_norm: goldMeanNorm,
      gold_mean_neutral: FITNESS_GOLD_MEAN_NEUTRAL,
      gold_mean_scale: FITNESS_GOLD_MEAN_SCALE,
      gold_p50_norm: goldP50Norm,
      gold_p50_neutral: FITNESS_GOLD_P50_NEUTRAL,
      gold_p50_scale: FITNESS_GOLD_P50_SCALE,
      gold_p10_norm: goldP10Norm,
      gold_p10_neutral: FITNESS_GOLD_P10_NEUTRAL,
      gold_p10_scale: FITNESS_GOLD_P10_SCALE,
      gold_cvar10_norm: goldCvar10Norm,
      gold_cvar10_neutral: FITNESS_GOLD_CVAR10_NEUTRAL,
      gold_cvar10_scale: FITNESS_GOLD_CVAR10_SCALE,
      cvar10_gold_delta: cvar10GoldDelta,
      cvar20_gold_delta: cvar20GoldDelta,
      gold_core: goldCore,
      gold_core_weight_mean: FITNESS_GOLD_MEAN_WEIGHT,
      gold_core_weight_p50: FITNESS_GOLD_P50_WEIGHT,
      gold_core_weight_p10: FITNESS_GOLD_P10_WEIGHT,
      gold_core_weight_cvar10: FITNESS_GOLD_CVAR10_WEIGHT,
      expected_result: expectedResult,
      tie_break: tieBreak,
      tie_break_weight: FITNESS_TIE_BREAK_WEIGHT,
      tie_break_expected_neutral: FITNESS_TIE_BREAK_EXPECTED_NEUTRAL,
      bankrupt_rate: bankruptRate,
      bankrupt_penalty: bankruptPenalty,
      bankrupt_penalty_weight: FITNESS_BANKRUPT_PENALTY_WEIGHT,
      catastrophic_loss_rate: catastrophicLossRate,
      catastrophic_loss_threshold: FITNESS_CATASTROPHIC_LOSS_THRESHOLD,
      catastrophic_loss_penalty: catastrophicLossPenalty,
      catastrophic_loss_weight: FITNESS_CATASTROPHIC_LOSS_WEIGHT,
      downside_semi_raw: downsideSemiRaw,
      downside_semi_norm: downsideSemiNorm,
      downside_semi_penalty: downsideSemiPenalty,
      downside_semi_weight: FITNESS_DOWNSIDE_SEMI_WEIGHT,
      downside_semi_scale: FITNESS_DOWNSIDE_SEMI_SCALE,
      go_fail_rate_cap: FITNESS_GO_FAIL_RATE_CAP,
      go_fail_excess: goFailExcess,
      go_fail_penalty: goFailPenalty,
      go_fail_excess_weight: FITNESS_GO_FAIL_EXCESS_WEIGHT,
      seat_first_win_rate: seatFirstWinRate,
      seat_second_win_rate: seatSecondWinRate,
      seat_win_rate_gap: seatWinRateGap,
      seat_first_mean_gold_delta: seatFirstMeanGoldDelta,
      seat_second_mean_gold_delta: seatSecondMeanGoldDelta,
      seat_mean_gold_gap: seatMeanGoldGap,
      seat_mean_gold_gap_norm: seatMeanGoldGapNorm,
      seat_gap_penalty: seatGapPenalty,
      seat_gap_weight: FITNESS_SEAT_GAP_WEIGHT,
      seat_gold_gap_scale: FITNESS_SEAT_GOLD_GAP_SCALE,
      inflicted_bankrupt_rate: inflictedBankruptRate,
      inflicted_bankrupt_bonus: inflictedBankruptBonus,
      inflicted_bankrupt_bonus_weight: FITNESS_INFLICTED_BANKRUPT_BONUS_WEIGHT,
      imitation_agreement_score: imitationAgreementScore,
      imitation_prob_score: imitationProbScore,
      imitation_bonus: imitationBonus,
      imitation_agreement_weight: FITNESS_IMITATION_AGREEMENT_WEIGHT,
      imitation_prob_weight: FITNESS_IMITATION_PROB_WEIGHT,
      imitation_teacher_policy: effectiveTeacherPolicy,
      scenario_shard_bonus: scenarioShardBonus,
      scenario_shard_weight_sum: Number(scenarioMetrics.weightSum || 0),
      go_zero_hard_fail: goZeroHardFail,
      go_zero_force_score: FITNESS_GO_ZERO_FORCE_SCORE,
      fitness_profile: opts.fitnessProfile,
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





