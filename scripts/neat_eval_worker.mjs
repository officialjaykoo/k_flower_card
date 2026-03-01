import fs from "node:fs";
import path from "node:path";
import {
  initSimulationGame,
  startSimulationGame,
  createSeededRng,
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

function parseArgs(argv) {
  const args = [...argv];
  const out = {
    genomePath: "",
    opponentGenomePath: "",
    games: 3,
    seed: "neat-python",
    maxSteps: 600,
    opponentPolicy: "H-V4",
    firstTurnPolicy: "alternate",
    fixedFirstTurn: "human",
    continuousSeries: true,
    // NOTE:
    // - fitnessGoldScale is used as tanh normalization scale for mean_gold_delta.
    // - fitnessWinWeight / fitnessLossWeight / fitnessDrawWeight are mapped to:
    //   win / gold / go component weights (normalized internally).
    fitnessGoldScale: 2500.0,
    fitnessWinWeight: 0.35,
    fitnessLossWeight: 0.50,
    fitnessDrawWeight: 0.15,
    fitnessGoTargetRate: 0.20,
    fitnessGoFailCap: 0.25,
    fitnessGoMinGames: 20,
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
    else if (key === "--opponent-policy") out.opponentPolicy = String(value || "H-V4").trim();
    else if (key === "--first-turn-policy") out.firstTurnPolicy = String(value || "alternate").trim().toLowerCase();
    else if (key === "--fixed-first-turn") out.fixedFirstTurn = String(value || "human").trim().toLowerCase();
    else if (key === "--switch-seats") {
      // Backward compatibility: legacy seat switch now maps to first-turn policy.
      out.firstTurnPolicy = String(value || "1").trim() === "0" ? "fixed" : "alternate";
    }
    else if (key === "--continuous-series") out.continuousSeries = !(String(value || "1").trim() === "0");
    else if (key === "--fitness-gold-scale") out.fitnessGoldScale = Math.max(1.0, Number(value || 2500.0));
    else if (key === "--fitness-win-weight") out.fitnessWinWeight = Number(value || 0.35);
    else if (key === "--fitness-loss-weight") out.fitnessLossWeight = Number(value || 0.50);
    else if (key === "--fitness-draw-weight") out.fitnessDrawWeight = Number(value || 0.15);
    else if (key === "--fitness-go-target-rate") out.fitnessGoTargetRate = Math.max(0.01, Number(value || 0.20));
    else if (key === "--fitness-go-fail-cap") out.fitnessGoFailCap = Math.max(0.01, Number(value || 0.25));
    else if (key === "--fitness-go-min-games") out.fitnessGoMinGames = Math.max(1, Math.floor(Number(value || 20)));
    else throw new Error(`Unknown argument: ${key}`);
  }

  if (!out.genomePath) throw new Error("--genome is required");
  if (out.firstTurnPolicy !== "alternate" && out.firstTurnPolicy !== "fixed") {
    throw new Error(`invalid --first-turn-policy: ${out.firstTurnPolicy}`);
  }
  if (out.fixedFirstTurn !== "human" && out.fixedFirstTurn !== "ai") {
    throw new Error(`invalid --fixed-first-turn: ${out.fixedFirstTurn}`);
  }
  if (String(out.opponentPolicy || "").trim().toLowerCase() === "genome" && !out.opponentGenomePath) {
    throw new Error("--opponent-genome is required when --opponent-policy=genome");
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

function selectPool(state, actor) {
  if (state.phase === "playing" && state.currentTurn === actor) {
    return { cards: (state.players?.[actor]?.hand || []).map((c) => c.id) };
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
    return (sp.cards || []).map((x) => String(x)).filter((x) => x.length > 0);
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
  if (decisionType === "play") return playTurn(state, action);
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
    heuristicPolicy: heuristicPolicy || "H-V4",
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
  const rng = createSeededRng(`${seed}|rng`);
  let state = initialState;
  const imitation = {
    totals: { play: 0, match: 0, option: 0 },
    matches: { play: 0, match: 0, option: 0 },
  };

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
    let next = state;

    if (actor === controlActor) {
      next = aiPlay(state, actor, {
        source: "model",
        model: controlModel,
      });
    } else if (normalizePolicyName(opponentPolicy) === "genome") {
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

    if (!next || stateProgressKey(next) === before) {
      next = randomLegalAction(state, actor, rng);
    }
    if (!next || stateProgressKey(next) === before) {
      throw new Error(
        `action resolution failed after fallback: seed=${seed}, step=${steps}, actor=${actor}, phase=${String(state?.phase || "")}`
      );
    }

    if (actor === controlActor && decisionType && candidates.length > 0) {
      const chosen = inferChosenCandidateFromTransition(
        state,
        actor,
        decisionType,
        candidates,
        next
      );
      if (chosen) {
        const imitationRefPolicy =
          normalizePolicyName(opponentPolicy) === "genome"
            ? "H-V4"
            : opponentPolicy;
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

  return { endState: state, imitation };
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
  const simImitationTotals = { play: 0, match: 0, option: 0 };
  const simImitationMatches = { play: 0, match: 0, option: 0 };
  const firstTurnCounts = {
    human: 0,
    ai: 0,
  };
  const seriesSession = {
    roundsPlayed: 0,
    previousEndState: null,
  };

  for (let gi = 0; gi < games; gi += 1) {
    const firstTurnKey = resolveFirstTurnKey(opts, gi);
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
      opts.opponentPolicy,
      maxSteps,
      opponentModel
    );
    const endState = gameResult?.endState || gameResult;
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
    if (winner === controlActor) wins += 1;
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
  const goRate = goGames / games;
  const goFailRate = goGames > 0 ? goFailCount / goGames : 0;
  const fitnessGoldScaleRaw = Number(opts.fitnessGoldScale);
  const fitnessWinWeightRaw = Number(opts.fitnessWinWeight);
  const fitnessLossWeightRaw = Number(opts.fitnessLossWeight);
  const fitnessDrawWeightRaw = Number(opts.fitnessDrawWeight);
  const fitnessGoTargetRateRaw = Number(opts.fitnessGoTargetRate);
  const fitnessGoFailCapRaw = Number(opts.fitnessGoFailCap);
  const fitnessGoMinGamesRaw = Number(opts.fitnessGoMinGames);
  const fitnessGoldScale = Number.isFinite(fitnessGoldScaleRaw) && fitnessGoldScaleRaw > 0
    ? fitnessGoldScaleRaw
    : 2500.0;
  const weightWinRaw = Number.isFinite(fitnessWinWeightRaw) ? Math.max(0, fitnessWinWeightRaw) : 0.35;
  const weightGoldRaw = Number.isFinite(fitnessLossWeightRaw) ? Math.max(0, fitnessLossWeightRaw) : 0.50;
  const weightGoRaw = Number.isFinite(fitnessDrawWeightRaw) ? Math.max(0, fitnessDrawWeightRaw) : 0.15;
  const weightRawSum = weightWinRaw + weightGoldRaw + weightGoRaw;
  const fitnessWinWeight = weightRawSum > 0 ? weightWinRaw / weightRawSum : 0.35;
  const fitnessLossWeight = weightRawSum > 0 ? weightGoldRaw / weightRawSum : 0.50;
  const fitnessDrawWeight = weightRawSum > 0 ? weightGoRaw / weightRawSum : 0.15;
  const fitnessGoTargetRate =
    Number.isFinite(fitnessGoTargetRateRaw) && fitnessGoTargetRateRaw > 0 ? fitnessGoTargetRateRaw : 0.20;
  const fitnessGoFailCap =
    Number.isFinite(fitnessGoFailCapRaw) && fitnessGoFailCapRaw > 0 ? fitnessGoFailCapRaw : 0.25;
  const fitnessGoMinGames =
    Number.isFinite(fitnessGoMinGamesRaw) && fitnessGoMinGamesRaw > 0 ? Math.floor(fitnessGoMinGamesRaw) : 20;

  // Balanced fitness:
  // - gold term: bounded by tanh to avoid unstable spikes
  // - win term: symmetric [-1, +1]
  // - go term : reward healthy GO usage (presence + quality)
  const goldNorm = Math.tanh(meanGoldDelta / fitnessGoldScale);
  const winNorm = clamp01(winRate) * 2.0 - 1.0;
  const goPresence = clamp01(goRate / fitnessGoTargetRate);
  const goQuality =
    goGames >= fitnessGoMinGames ? clamp01(1.0 - (goFailRate / fitnessGoFailCap)) : 0.0;
  const goTerm01 = (0.2 * goPresence) + (0.8 * goQuality);
  const goNorm = clamp01(goTerm01) * 2.0 - 1.0;

  let fitness =
    (fitnessLossWeight * goldNorm) +
    (fitnessWinWeight * winNorm) +
    (fitnessDrawWeight * goNorm);
  if (goGames === 0) fitness -= 0.35;
  else if (goGames < fitnessGoMinGames) fitness -= 0.15;

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
    opponent_eval_tuning: {
      fast_path: false,
      imitation_reference_enabled: true,
      opponent_heuristic_params: null,
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
    p10_gold_delta: quantile(goldDeltas, 0.1),
    p50_gold_delta: quantile(goldDeltas, 0.5),
    p90_gold_delta: quantile(goldDeltas, 0.9),
    fitness_gold_scale: fitnessGoldScale,
    fitness_win_weight: fitnessWinWeight,
    fitness_loss_weight: fitnessLossWeight,
    fitness_draw_weight: fitnessDrawWeight,
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
      win_norm: winNorm,
      go_presence: goPresence,
      go_quality: goQuality,
      go_norm: goNorm,
      weights: {
        win: fitnessWinWeight,
        gold: fitnessLossWeight,
        go: fitnessDrawWeight,
      },
      go_target_rate: fitnessGoTargetRate,
      go_fail_cap: fitnessGoFailCap,
      go_min_games: fitnessGoMinGames,
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
