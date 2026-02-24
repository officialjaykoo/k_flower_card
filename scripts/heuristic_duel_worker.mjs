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
import { BOT_POLICIES, normalizeBotPolicy } from "../src/ai/policies.js";

function parseArgs(argv) {
  const args = [...argv];
  const out = {
    policyA: "",
    policyB: "",
    games: 1000,
    seed: "heuristic-duel",
    maxSteps: 600,
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

    if (key === "--policy-a") out.policyA = String(value || "").trim().toLowerCase();
    else if (key === "--policy-b") out.policyB = String(value || "").trim().toLowerCase();
    else if (key === "--games") out.games = Math.max(1, Number(value || 1000));
    else if (key === "--seed") out.seed = String(value || "heuristic-duel");
    else if (key === "--max-steps") out.maxSteps = Math.max(20, Number(value || 600));
    else if (key === "--first-turn-policy") out.firstTurnPolicy = String(value || "alternate").trim().toLowerCase();
    else if (key === "--fixed-first-turn") out.fixedFirstTurn = String(value || "human").trim().toLowerCase();
    else if (key === "--continuous-series") out.continuousSeries = !(String(value || "1").trim() === "0");
    else throw new Error(`Unknown argument: ${key}`);
  }

  if (!out.policyA) throw new Error("--policy-a is required");
  if (!out.policyB) throw new Error("--policy-b is required");
  if (!BOT_POLICIES.includes(out.policyA)) {
    throw new Error(`invalid --policy-a: ${out.policyA} (allowed: ${BOT_POLICIES.join(", ")})`);
  }
  if (!BOT_POLICIES.includes(out.policyB)) {
    throw new Error(`invalid --policy-b: ${out.policyB} (allowed: ${BOT_POLICIES.join(", ")})`);
  }
  out.policyA = normalizeBotPolicy(out.policyA);
  out.policyB = normalizeBotPolicy(out.policyB);

  if (Math.floor(out.games) !== 1000) {
    throw new Error("this worker is fixed to --games 1000");
  }
  out.games = 1000;

  if (out.firstTurnPolicy !== "alternate" && out.firstTurnPolicy !== "fixed") {
    throw new Error(`invalid --first-turn-policy: ${out.firstTurnPolicy}`);
  }
  if (out.fixedFirstTurn !== "human" && out.fixedFirstTurn !== "ai") {
    throw new Error(`invalid --fixed-first-turn: ${out.fixedFirstTurn}`);
  }

  return out;
}

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

function goldDiffByActor(state, actor) {
  const opp = actor === "human" ? "ai" : "human";
  const selfGold = Number(state?.players?.[actor]?.gold || 0);
  const oppGold = Number(state?.players?.[opp]?.gold || 0);
  return selfGold - oppGold;
}

function playSingleRound(initialState, seed, policyByActor, maxSteps) {
  const rng = createSeededRng(`${seed}|rng`);
  let state = initialState;
  let steps = 0;

  while (state.phase !== "resolution" && steps < maxSteps) {
    const actor = getActionPlayerKey(state);
    if (!actor) break;

    const before = stateProgressKey(state);
    const policy = policyByActor[actor];
    let next = aiPlay(state, actor, {
      source: "heuristic",
      heuristicPolicy: policy,
    });

    if (!next || stateProgressKey(next) === before) {
      next = randomLegalAction(state, actor, rng);
    }
    if (!next || stateProgressKey(next) === before) {
      break;
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
    gold_deltas: [],
  };
}

function updateSeatRecord(record, winner, selfActor, oppActor, goldDelta, selfGoCount) {
  record.games += 1;
  if (winner === selfActor) record.wins += 1;
  else if (winner === oppActor) record.losses += 1;
  else record.draws += 1;
  const goCount = Math.max(0, Number(selfGoCount || 0));
  record.go_count_total += goCount;
  if (goCount > 0) {
    record.go_game_count += 1;
    if (winner !== selfActor) {
      record.go_fail_count += 1;
    }
  }
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

function main() {
  const evalStartMs = Date.now();
  const opts = parseArgs(process.argv.slice(2));

  const actorA = "human";
  const actorB = "ai";
  const policyByActor = {
    [actorA]: opts.policyA,
    [actorB]: opts.policyB,
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
        ? continueRound(seriesSession.previousEndState, seed, firstTurnKey)
        : startRound(seed, firstTurnKey)
      : startRound(seed, firstTurnKey);

    const beforeDiffA = goldDiffByActor(roundStart, actorA);
    const endState = playSingleRound(
      roundStart,
      seed,
      policyByActor,
      Math.max(20, Math.floor(opts.maxSteps))
    );
    const afterDiffA = goldDiffByActor(endState, actorA);
    goldDeltasA.push(afterDiffA - beforeDiffA);

    const goldA = Number(endState?.players?.[actorA]?.gold || 0);
    const goldB = Number(endState?.players?.[actorB]?.gold || 0);
    const goCountA = Math.max(0, Number(endState?.players?.[actorA]?.goCount || 0));
    const goCountB = Math.max(0, Number(endState?.players?.[actorB]?.goCount || 0));
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

    const seatAKey = firstTurnKey === actorA ? "first" : "second";
    const seatBKey = firstTurnKey === actorB ? "first" : "second";
    const goldDeltaA = afterDiffA - beforeDiffA;
    updateSeatRecord(seatSplitA[seatAKey], winner, actorA, actorB, goldDeltaA, goCountA);
    updateSeatRecord(seatSplitB[seatBKey], winner, actorB, actorA, -goldDeltaA, goCountB);
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
    actor_a: actorA,
    actor_b: actorB,
    policy_a: opts.policyA,
    policy_b: opts.policyB,
    first_turn_policy: opts.firstTurnPolicy,
    fixed_first_turn: opts.firstTurnPolicy === "fixed" ? opts.fixedFirstTurn : null,
    first_turn_counts: firstTurnCounts,
    continuous_series: !!opts.continuousSeries,
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
    mean_gold_delta_a: meanGoldDeltaA,
    p10_gold_delta_a: quantile(goldDeltasA, 0.1),
    p50_gold_delta_a: quantile(goldDeltasA, 0.5),
    p90_gold_delta_a: quantile(goldDeltasA, 0.9),
    seat_split_a: splitSummaryA,
    seat_split_b: splitSummaryB,
    eval_time_ms: Math.max(0, Date.now() - evalStartMs),
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
