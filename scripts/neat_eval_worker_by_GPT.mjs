// neat_eval_worker_by_GPT.mjs
// - Evaluates one genome against heuristic/mix opponents.
// - Emits one JSON summary line (stdout last line).
// - Fail-fast on missing/invalid required inputs.

import fs from "node:fs";
import path from "node:path";
import {
  initSimulationGame,
  startSimulationGame,
  createSeededRng
} from "../src/engine/index.js";
import { getActionPlayerKey } from "../src/engine/runner.js";
import { aiPlay } from "../src/ai/aiPlay_by_GPT.js";
import { stateProgressKey } from "../src/ai/decisionRuntime_by_GPT.js";

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
  if (out.fitnessProfile !== "phase1" && out.fitnessProfile !== "phase2" && out.fitnessProfile !== "phase3") {
    throw new Error(`invalid --fitness-profile: ${out.fitnessProfile}`);
  }
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
  return out;
}

// =============================================================================
// Section 2. Round Simulation + Metrics
// =============================================================================
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
  opponentModel
) {
  let state = initialState;

  let steps = 0;
  while (state.phase !== "resolution" && steps < maxSteps) {
    const actor = getActionPlayerKey(state);
    if (!actor) break;

    const before = stateProgressKey(state, { includeKiboSeq: true });
    let next = state;
    let actionSource = "heuristic";

    if (actor === controlActor) {
      actionSource = "model_control";
      next = aiPlay(state, actor, {
        source: "model",
        model: controlModel,
      });
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

    if (!next || stateProgressKey(next, { includeKiboSeq: true }) === before) {
      throw new Error(
        `action resolution failed: seed=${seed}, step=${steps}, actor=${actor}, phase=${String(state?.phase || "")}, policy=${String(opponentPolicy || "")}, source=${actionSource}`
      );
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
  if (String(opts.opponentPolicy || "").trim().toLowerCase() === "genome") {
    const oppFull = path.resolve(opts.opponentGenomePath);
    if (!fs.existsSync(oppFull)) {
      throw new Error(`opponent genome not found: ${opts.opponentGenomePath}`);
    }
    opponentModel = JSON.parse(fs.readFileSync(oppFull, "utf8"));
    if (String(opponentModel?.format_version || "").trim() !== "neat_python_genome_v1") {
      throw new Error("invalid --opponent-genome format: expected neat_python_genome_v1");
    }
  }

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
  const seriesSession = {
    roundsPlayed: 0,
    previousEndState: null,
  };

  // 4-2) Run per-game simulations and accumulate metrics.
  for (let gi = 0; gi < games; gi += 1) {
    const firstTurnKey = resolveFirstTurnKey(opts, gi);
    const opponentPolicyForGame = selectOpponentPolicyForGame(opts, gi);
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
      opponentModel
    );
    const endState = gameResult;
    const afterGoldDiff = controlGoldDiff(endState, controlActor);
    const goldDelta = afterGoldDiff - beforeGoldDiff;
    goldDeltas.push(goldDelta);
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
    }
    else if (winner === opponentActor) losses += 1;
    else draws += 1;
    if (controlGoCount > 0 && winner !== controlActor) {
      goFailCount += 1;
    }

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
  // Fitness profile (phase-specific fixed constants):
  // 1) GO is removed from additive shaping. Only hard cut remains.
  // 2) Gold core uses mean + median + downside (p10) robustness.
  // 3) Win/loss/draw aggregate is tie-break only.
  // 4) Bankrupt rate is direct downside penalty.
  const FITNESS_PROFILES = {
    phase1: {
      modelName: "gold_core_hard_go_zero_profile_phase1_v2",
      goldMeanNeutral: -250.0,
      goldMeanScale: 3200.0,
      goldP50Neutral: -800.0,
      goldP50Scale: 3500.0,
      goldP10Neutral: -2600.0,
      goldP10Scale: 5200.0,
      goldMeanWeight: 0.65,
      goldP50Weight: 0.20,
      goldP10Weight: 0.15,
      tieBreakWeight: 0.02,
      tieBreakExpectedNeutral: -0.20,
      bankruptPenaltyWeight: 0.10,
      goZeroForceScore: -3.0,
    },
    phase2: {
      modelName: "gold_core_hard_go_zero_profile_phase2_v2",
      goldMeanNeutral: -100.0,
      goldMeanScale: 3000.0,
      goldP50Neutral: -500.0,
      goldP50Scale: 3200.0,
      goldP10Neutral: -2200.0,
      goldP10Scale: 4800.0,
      goldMeanWeight: 0.55,
      goldP50Weight: 0.20,
      goldP10Weight: 0.25,
      tieBreakWeight: 0.03,
      tieBreakExpectedNeutral: -0.16,
      bankruptPenaltyWeight: 0.14,
      goZeroForceScore: -3.0,
    },
    phase3: {
      modelName: "gold_core_hard_go_zero_profile_phase3_v2",
      goldMeanNeutral: 0.0,
      goldMeanScale: 2800.0,
      goldP50Neutral: 0.0,
      goldP50Scale: 2800.0,
      goldP10Neutral: -1800.0,
      goldP10Scale: 4200.0,
      goldMeanWeight: 0.45,
      goldP50Weight: 0.20,
      goldP10Weight: 0.35,
      tieBreakWeight: 0.04,
      tieBreakExpectedNeutral: -0.10,
      bankruptPenaltyWeight: 0.18,
      goZeroForceScore: -3.0,
    },
  };
  const profile = FITNESS_PROFILES[opts.fitnessProfile];
  if (!profile) {
    throw new Error(`unsupported fitness profile: ${String(opts.fitnessProfile || "")}`);
  }
  const FITNESS_GOLD_MEAN_NEUTRAL = profile.goldMeanNeutral;
  const FITNESS_GOLD_MEAN_SCALE = profile.goldMeanScale;
  const FITNESS_GOLD_P50_NEUTRAL = profile.goldP50Neutral;
  const FITNESS_GOLD_P50_SCALE = profile.goldP50Scale;
  const FITNESS_GOLD_P10_NEUTRAL = profile.goldP10Neutral;
  const FITNESS_GOLD_P10_SCALE = profile.goldP10Scale;
  const FITNESS_GOLD_MEAN_WEIGHT = profile.goldMeanWeight;
  const FITNESS_GOLD_P50_WEIGHT = profile.goldP50Weight;
  const FITNESS_GOLD_P10_WEIGHT = profile.goldP10Weight;
  const FITNESS_TIE_BREAK_WEIGHT = profile.tieBreakWeight;
  const FITNESS_TIE_BREAK_EXPECTED_NEUTRAL = profile.tieBreakExpectedNeutral;
  const FITNESS_BANKRUPT_PENALTY_WEIGHT = profile.bankruptPenaltyWeight;
  const FITNESS_GO_ZERO_FORCE_SCORE = profile.goZeroForceScore;

  const expectedResultRaw =
    clamp01(winRate) + (0.5 * clamp01(drawRate)) - clamp01(lossRate);
  const expectedResult = Math.max(-1.0, Math.min(1.0, expectedResultRaw));
  const goldMeanNorm = Math.tanh((meanGoldDelta - FITNESS_GOLD_MEAN_NEUTRAL) / FITNESS_GOLD_MEAN_SCALE);
  const goldP50Norm = Math.tanh((p50GoldDelta - FITNESS_GOLD_P50_NEUTRAL) / FITNESS_GOLD_P50_SCALE);
  const goldP10Norm = Math.tanh((p10GoldDelta - FITNESS_GOLD_P10_NEUTRAL) / FITNESS_GOLD_P10_SCALE);
  const goldCore =
    (FITNESS_GOLD_MEAN_WEIGHT * goldMeanNorm) +
    (FITNESS_GOLD_P50_WEIGHT * goldP50Norm) +
    (FITNESS_GOLD_P10_WEIGHT * goldP10Norm);
  const tieBreak = FITNESS_TIE_BREAK_WEIGHT * (expectedResult - FITNESS_TIE_BREAK_EXPECTED_NEUTRAL);
  const bankruptRate = games > 0 ? clamp01(bankrupt.my_bankrupt_count / games) : 0;
  const bankruptPenalty = FITNESS_BANKRUPT_PENALTY_WEIGHT * bankruptRate;
  const goZeroHardFail = goGames === 0;
  let fitness = goZeroHardFail ? FITNESS_GO_ZERO_FORCE_SCORE : (goldCore + tieBreak - bankruptPenalty);

  // 4-4) Build final summary contract consumed by phase scripts.
  const summary = {
    games,
    control_actor: controlActor,
    opponent_actor: opponentActor,
    opponent_policy: opts.opponentPolicy,
    opponent_policy_mix: opts.opponentPolicyMix,
    opponent_policy_counts: opponentPolicyCounts,
    opponent_eval_tuning: { fast_path: false, opponent_heuristic_params: null },
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
    fitness_model: profile.modelName,
    fitness_profile: opts.fitnessProfile,
    fitness_gold_scale: FITNESS_GOLD_MEAN_SCALE,
    fitness_gold_neutral_delta: FITNESS_GOLD_MEAN_NEUTRAL,
    fitness_gold_p50_scale: FITNESS_GOLD_P50_SCALE,
    fitness_gold_p50_neutral_delta: FITNESS_GOLD_P50_NEUTRAL,
    fitness_gold_p10_scale: FITNESS_GOLD_P10_SCALE,
    fitness_gold_p10_neutral_delta: FITNESS_GOLD_P10_NEUTRAL,
    fitness_tie_break_expected_neutral: FITNESS_TIE_BREAK_EXPECTED_NEUTRAL,
    fitness_win_weight: FITNESS_TIE_BREAK_WEIGHT,
    fitness_gold_weight: 1.0,
    fitness_bankrupt_penalty_weight: FITNESS_BANKRUPT_PENALTY_WEIGHT,
    imitation_weighted_score: 0,
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
      gold_core: goldCore,
      gold_core_weight_mean: FITNESS_GOLD_MEAN_WEIGHT,
      gold_core_weight_p50: FITNESS_GOLD_P50_WEIGHT,
      gold_core_weight_p10: FITNESS_GOLD_P10_WEIGHT,
      expected_result: expectedResult,
      tie_break: tieBreak,
      tie_break_weight: FITNESS_TIE_BREAK_WEIGHT,
      tie_break_expected_neutral: FITNESS_TIE_BREAK_EXPECTED_NEUTRAL,
      bankrupt_rate: bankruptRate,
      bankrupt_penalty: bankruptPenalty,
      bankrupt_penalty_weight: FITNESS_BANKRUPT_PENALTY_WEIGHT,
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




