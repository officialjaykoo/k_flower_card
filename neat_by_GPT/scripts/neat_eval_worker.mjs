// neat_eval_worker.mjs
// - Evaluates one genome against heuristic/mix opponents.
// - Emits one JSON summary line (stdout last line).
// - Fail-fast on missing/invalid required inputs.

import fs from "node:fs";
import path from "node:path";
import {
  initSimulationGame,
  startSimulationGame,
  createSeededRng
} from "../../src/engine/index.js";
import { getActionPlayerKey } from "../../src/engine/runner.js";
import { aiPlay } from "../../src/ai/aiPlay_by_GPT.js";
import { stateProgressKey } from "../../src/ai/decisionRuntime_by_GPT.js";

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
  const controlSeatStats = {
    first: { games: 0, wins: 0, goldDeltas: [] },
    second: { games: 0, wins: 0, goldDeltas: [] },
  };
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
      opponentModel
    );
    const endState = gameResult;
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

  const goZeroHardFail = goGames === 0;
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
      inflictedBankruptBonus
    );

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





