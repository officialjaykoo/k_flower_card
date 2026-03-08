import {
  initSimulationGame,
  startSimulationGame,
  createSeededRng
} from "../../src/engine/index.js";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { getActionPlayerKey } from "../../src/engine/runner.js";
import { aiPlay } from "../../src/ai/aiPlay_by_GPT.js";
import { resolveBotPolicy } from "../../src/ai/policies.js";
import { stateProgressKey } from "../../src/ai/decisionRuntime_by_GPT.js";

// GPT-only duel runner (minimal duel/report flow).

function parseArgs(argv) {
  const args = [...argv];
  if (args.includes("--help") || args.includes("-h")) {
    return { help: true };
  }

  const out = {
    humanSpecRaw: "",
    aiSpecRaw: "",
    games: 1000,
    seed: "model-duel-gpt",
    maxSteps: 600,
    firstTurnPolicy: "alternate",
    fixedFirstTurn: "human",
    continuousSeries: true,
    stdoutFormat: "text",
    resultOut: "",
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
    else if (key === "--games") out.games = Math.max(1, Number(value || 1000));
    else if (key === "--seed") out.seed = String(value || "model-duel-gpt").trim();
    else if (key === "--max-steps") out.maxSteps = Math.max(20, Number(value || 600));
    else if (key === "--first-turn-policy") {
      out.firstTurnPolicy = String(value || "alternate").trim().toLowerCase();
    }
    else if (key === "--fixed-first-turn") {
      out.fixedFirstTurn = String(value || "human").trim().toLowerCase();
    }
    else if (key === "--continuous-series") out.continuousSeries = parseContinuousSeriesValue(value);
    else if (key === "--stdout-format") out.stdoutFormat = String(value || "text").trim().toLowerCase();
    else if (key === "--result-out") out.resultOut = String(value || "").trim();
    else {
      throw new Error(
        `Unknown argument: ${key} (allowed: --human, --ai, --games, --seed, --max-steps, --first-turn-policy, --fixed-first-turn, --continuous-series, --stdout-format, --result-out)`
      );
    }
  }

  if (!out.humanSpecRaw) throw new Error("--human is required");
  if (!out.aiSpecRaw) throw new Error("--ai is required");
  if (Math.floor(out.games) < 1000) throw new Error("this worker requires --games >= 1000");
  out.games = Math.floor(out.games);

  if (out.firstTurnPolicy !== "alternate" && out.firstTurnPolicy !== "fixed") {
    throw new Error(`invalid --first-turn-policy: ${out.firstTurnPolicy}`);
  }
  if (out.fixedFirstTurn !== "human") {
    throw new Error("--fixed-first-turn is locked to human");
  }
  if (out.stdoutFormat !== "text" && out.stdoutFormat !== "json") {
    throw new Error(`invalid --stdout-format: ${out.stdoutFormat} (allowed: text, json)`);
  }

  return out;
}

function usageText() {
  return [
    "Usage:",
    "  node neat_by_GPT/scripts/model_duel_worker.mjs --human <spec> --ai <spec> [options]",
    "",
    "Required:",
    "  --human <policy|phaseX_seedY>",
    "  --ai <policy|phaseX_seedY>",
    "",
    "Options:",
    "  --games <N>                 default=1000, minimum=1000",
    "  --seed <tag>                default=model-duel-gpt",
    "  --max-steps <N>             default=600, minimum=20",
    "  --first-turn-policy <mode>  alternate|fixed (default=alternate)",
    "  --fixed-first-turn <actor>  human only (default=human)",
    "  --continuous-series <flag>  1=true, 2=false (default=1)",
    "  --stdout-format <mode>      text|json (default=text)",
    "  --result-out <path>         optional, auto-generated if omitted",
    "",
    "Example:",
    "  node neat_by_GPT/scripts/model_duel_worker.mjs --human H-CL --ai phase1_seed60 --games 1000 --seed gpt_duel_1 --first-turn-policy alternate --continuous-series 1",
  ].join("\n");
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

  const m = token.match(/^phase([0-3])_seed(\d+)$/i);
  if (!m) {
    throw new Error(`invalid ${sideLabel} spec: ${token} (use policy key or phase0_seed9)`);
  }
  const phase = Number(m[1]);
  const seed = Number(m[2]);
  const modelPath = resolve(`logs/NEAT/neat_phase${phase}_seed${seed}/models/winner_genome.json`);
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
    key: `phase${phase}_seed${seed}`,
    label: `phase${phase}_seed${seed}`,
    model,
    modelPath,
    phase,
    seed,
  };
}

function buildAutoOutputDir(humanLabel, aiLabel) {
  const duelKey = `${sanitizeFilePart(humanLabel)}_vs_${sanitizeFilePart(aiLabel)}_${dateTag()}`;
  const outDir = join("logs", "model_duel", duelKey);
  mkdirSync(outDir, { recursive: true });
  return outDir;
}

function buildAutoArtifactPath(outDir, seed, suffix) {
  const stem = sanitizeFilePart(seed) || "model-duel-gpt";
  return join(outDir, `${stem}_${suffix}`);
}

function toReportPath(pathValue) {
  const raw = String(pathValue || "").trim();
  if (!raw) return null;
  const rel = relative(process.cwd(), resolve(raw));
  const normalized = String(rel || raw).replace(/\\/g, "/");
  return normalized || null;
}

function resolveFirstTurnKey(opts, gameIndex) {
  if (opts.firstTurnPolicy === "fixed") return opts.fixedFirstTurn;
  return gameIndex % 2 === 0 ? "ai" : "human";
}

function startRound(seed, firstTurnKey) {
  return initSimulationGame("A", createSeededRng(`${seed}|game`), {
    firstTurnKey,
  });
}

function continueRound(prevEndState, seed, firstTurnKey) {
  return startSimulationGame(prevEndState, createSeededRng(`${seed}|game`), {
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
    return { source: "model", model: playerSpec.model };
  }
  return { source: "heuristic", heuristicPolicy: String(playerSpec?.key || "") };
}

function playSingleRound(initialState, seed, playerByActor, maxSteps) {
  let state = initialState;
  let steps = 0;

  while (state.phase !== "resolution" && steps < maxSteps) {
    const actor = getActionPlayerKey(state);
    if (!actor) break;

    const before = stateProgressKey(state);
    const playerSpec = playerByActor[actor];
    const policy = String(playerSpec?.label || "");
    const actionSource = playerSpec?.kind === "model" ? "model" : "heuristic";
    const next = aiPlay(state, actor, buildAiPlayOptions(playerSpec));

    if (!next || stateProgressKey(next) === before) {
      throw new Error(
        `action resolution failed: seed=${seed}, step=${steps}, actor=${actor}, phase=${String(state?.phase || "")}, policy=${policy}, source=${actionSource}`
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
    if (winner !== selfActor) record.go_fail_count += 1;
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

function buildConsoleSummary(report) {
  const seatAFirst = report?.seat_split_a?.when_first || {};
  const seatASecond = report?.seat_split_a?.when_second || {};
  const seatBFirst = report?.seat_split_b?.when_first || {};
  const seatBSecond = report?.seat_split_b?.when_second || {};
  const bankrupt = report?.bankrupt || { a_bankrupt_count: 0, b_bankrupt_count: 0 };

  return {
    games: Number(report?.games || 0),
    human: String(report?.human || ""),
    ai: String(report?.ai || ""),
    wins_a: Number(report?.wins_a || 0),
    losses_a: Number(report?.losses_a || 0),
    wins_b: Number(report?.wins_b || 0),
    losses_b: Number(report?.losses_b || 0),
    draws: Number(report?.draws || 0),
    win_rate_a: Number(report?.win_rate_a || 0),
    win_rate_b: Number(report?.win_rate_b || 0),
    mean_gold_delta_a: Number(report?.mean_gold_delta_a || 0),
    p10_gold_delta_a: Number(report?.p10_gold_delta_a || 0),
    p50_gold_delta_a: Number(report?.p50_gold_delta_a || 0),
    p90_gold_delta_a: Number(report?.p90_gold_delta_a || 0),
    go_count_a: Number(report?.go_count_a || 0),
    go_games_a: Number(report?.go_games_a || 0),
    go_fail_count_a: Number(report?.go_fail_count_a || 0),
    go_fail_rate_a: Number(report?.go_fail_rate_a || 0),
    go_count_b: Number(report?.go_count_b || 0),
    go_games_b: Number(report?.go_games_b || 0),
    go_fail_count_b: Number(report?.go_fail_count_b || 0),
    go_fail_rate_b: Number(report?.go_fail_rate_b || 0),
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
    bankrupt,
    eval_time_sec: Number(report?.eval_time_ms || 0) / 1000,
    result_out: String(report?.result_out || ""),
  };
}

function formatConsoleSummaryText(summary) {
  const aFirst = summary?.seat_split_a?.when_first || {};
  const aSecond = summary?.seat_split_a?.when_second || {};
  const bFirst = summary?.seat_split_b?.when_first || {};
  const bSecond = summary?.seat_split_b?.when_second || {};
  const bankrupt = summary?.bankrupt || { a_bankrupt_count: 0, b_bankrupt_count: 0 };

  const lines = [
    "",
    `=== Model Duel (${summary.human} vs ${summary.ai}, games=${summary.games}) ===`,
    `Win/Loss/Draw(A):  ${summary.wins_a} / ${summary.losses_a} / ${summary.draws}  (WR=${summary.win_rate_a})`,
    `Win/Loss/Draw(B):  ${summary.wins_b} / ${summary.losses_b} / ${summary.draws}  (WR=${summary.win_rate_b})`,
    `Seat A first:      WR=${aFirst.win_rate}, mean_gold_delta=${aFirst.mean_gold_delta}`,
    `Seat A second:     WR=${aSecond.win_rate}, mean_gold_delta=${aSecond.mean_gold_delta}`,
    `Seat B first:      WR=${bFirst.win_rate}, mean_gold_delta=${bFirst.mean_gold_delta}`,
    `Seat B second:     WR=${bSecond.win_rate}, mean_gold_delta=${bSecond.mean_gold_delta}`,
    `Gold delta(A):     mean=${summary.mean_gold_delta_a}, p10=${summary.p10_gold_delta_a}, p50=${summary.p50_gold_delta_a}, p90=${summary.p90_gold_delta_a}`,
    `GO A:              count=${summary.go_count_a}, games=${summary.go_games_a}, fail=${summary.go_fail_count_a}, fail_rate=${summary.go_fail_rate_a}`,
    `GO B:              count=${summary.go_count_b}, games=${summary.go_games_b}, fail=${summary.go_fail_count_b}, fail_rate=${summary.go_fail_rate_b}`,
    `Bankrupt:          A=${bankrupt.a_bankrupt_count}, B=${bankrupt.b_bankrupt_count}`,
    `Eval time:         ${summary.eval_time_sec}s`,
    `Result file:       ${summary.result_out || ""}`,
    "===========================================================",
    "",
  ];
  return `${lines.join("\n")}\n`;
}

export function runModelDuelCli(argv = process.argv.slice(2)) {
  const evalStartMs = Date.now();
  const opts = parseArgs(argv);
  if (opts?.help) {
    process.stdout.write(`${usageText()}\\n`);
    return;
  }

  const humanPlayer = resolvePlayerSpec(opts.humanSpecRaw, "human");
  const aiPlayer = resolvePlayerSpec(opts.aiSpecRaw, "ai");

  if (!opts.resultOut) {
    const outDir = buildAutoOutputDir(humanPlayer.label, aiPlayer.label);
    opts.resultOut = buildAutoArtifactPath(outDir, opts.seed, "result.json");
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
  const bankrupt = { a_bankrupt_count: 0, b_bankrupt_count: 0 };
  const firstTurnCounts = { human: 0, ai: 0 };
  const seatSplitA = { first: createSeatRecord(), second: createSeatRecord() };
  const seatSplitB = { first: createSeatRecord(), second: createSeatRecord() };
  const seriesSession = { roundsPlayed: 0, previousEndState: null };

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
      playerByActor,
      Math.max(20, Math.floor(opts.maxSteps))
    );
    const afterDiffA = goldDiffByActor(endState, actorA);
    const roundDeltaA = afterDiffA - beforeDiffA;
    goldDeltasA.push(roundDeltaA);

    const goldA = Number(endState?.players?.[actorA]?.gold || 0);
    const goldB = Number(endState?.players?.[actorB]?.gold || 0);
    const goCountA = Math.max(0, Number(endState?.players?.[actorA]?.goCount || 0));
    const goCountB = Math.max(0, Number(endState?.players?.[actorB]?.goCount || 0));
    if (goldA <= 0) bankrupt.a_bankrupt_count += 1;
    if (goldB <= 0) bankrupt.b_bankrupt_count += 1;

    if (opts.continuousSeries) seriesSession.previousEndState = endState;
    seriesSession.roundsPlayed += 1;

    const winner = String(endState?.result?.winner || "").trim();
    if (winner === actorA) winsA += 1;
    else if (winner === actorB) winsB += 1;
    else draws += 1;

    const seatAKey = firstTurnKey === actorA ? "first" : "second";
    const seatBKey = firstTurnKey === actorB ? "first" : "second";
    updateSeatRecord(seatSplitA[seatAKey], winner, actorA, actorB, roundDeltaA, goCountA);
    updateSeatRecord(seatSplitB[seatBKey], winner, actorB, actorA, -roundDeltaA, goCountB);
  }

  const games = opts.games;
  const winRateA = winsA / games;
  const winRateB = winsB / games;
  const drawRate = draws / games;
  const meanGoldDeltaA =
    goldDeltasA.length > 0 ? goldDeltasA.reduce((a, b) => a + b, 0) / goldDeltasA.length : 0;
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
    },
    player_ai: {
      input: aiPlayer.input,
      kind: aiPlayer.kind,
      key: aiPlayer.key,
      model_path: aiPlayer.modelPath,
    },
    first_turn_policy: opts.firstTurnPolicy,
    fixed_first_turn: opts.firstTurnPolicy === "fixed" ? opts.fixedFirstTurn : null,
    first_turn_counts: firstTurnCounts,
    continuous_series: !!opts.continuousSeries,
    result_out: toReportPath(opts.resultOut),
    bankrupt,
    session_rounds: {
      total_rounds: seriesSession.roundsPlayed,
    },
    wins_a: winsA,
    losses_a: winsB,
    wins_b: winsB,
    losses_b: winsA,
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

  const reportLine = `${JSON.stringify(summary)}\n`;
  mkdirSync(dirname(opts.resultOut), { recursive: true });
  writeFileSync(opts.resultOut, reportLine, { encoding: "utf8" });

  if (opts.stdoutFormat === "json") {
    process.stdout.write(reportLine);
    return;
  }

  const consoleSummary = buildConsoleSummary(summary);
  process.stdout.write(formatConsoleSummaryText(consoleSummary));
}

try {
  runModelDuelCli(process.argv.slice(2));
} catch (err) {
  const msg = err && err.stack ? err.stack : String(err);
  process.stderr.write(`${msg}\n`);
  process.exit(1);
}

