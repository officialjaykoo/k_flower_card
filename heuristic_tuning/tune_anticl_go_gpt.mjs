// AntiCL GO sweep tuner for GPT-authored heuristic runs.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve, join } from "node:path";
import { spawn } from "node:child_process";

function safeNum(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function fail(msg) {
  throw new Error(msg);
}

function parseArgs(argv) {
  const args = [...argv];
  const now = new Date();
  const stamp = `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, "0")}${String(now.getDate()).padStart(2, "0")}_${String(now.getHours()).padStart(2, "0")}${String(now.getMinutes()).padStart(2, "0")}${String(now.getSeconds()).padStart(2, "0")}`;

  const out = {
    opponent: "H-CL",
    games: 1000,
    seedBase: "anticl_tune_gpt",
    outDir: `logs/duel/anticl_tune_gpt/tune_${stamp}`,
    ratioCutStart: 2.2,
    ratioCutEnd: 2.8,
    ratioCutStep: 0.2,
    ratioDownStart: 0.04,
    ratioDownEnd: 0.1,
    ratioDownStep: 0.02,
    comboCutStart: 0.75,
    comboCutEnd: 0.85,
    comboCutStep: 0.05,
    comboDownStart: 0.04,
    comboDownEnd: 0.1,
    comboDownStep: 0.02,
    targetGoFailRate: 0.18,
    goFailPenalty: 5.0,
    goldWeight: 0.15,
    goldNormDiv: 5000.0,
    topK: 5,
    maxCandidates: 0,
    dryRun: false
  };

  while (args.length > 0) {
    const key = String(args.shift() || "");
    const value = String(args.shift() || "");

    if (key === "--opponent") out.opponent = value.trim();
    else if (key === "--games") out.games = Math.floor(safeNum(value, out.games));
    else if (key === "--seed-base") out.seedBase = value.trim();
    else if (key === "--out-dir") out.outDir = value.trim();
    else if (key === "--ratio-cut-start") out.ratioCutStart = safeNum(value, out.ratioCutStart);
    else if (key === "--ratio-cut-end") out.ratioCutEnd = safeNum(value, out.ratioCutEnd);
    else if (key === "--ratio-cut-step") out.ratioCutStep = safeNum(value, out.ratioCutStep);
    else if (key === "--ratio-down-start") out.ratioDownStart = safeNum(value, out.ratioDownStart);
    else if (key === "--ratio-down-end") out.ratioDownEnd = safeNum(value, out.ratioDownEnd);
    else if (key === "--ratio-down-step") out.ratioDownStep = safeNum(value, out.ratioDownStep);
    else if (key === "--combo-cut-start") out.comboCutStart = safeNum(value, out.comboCutStart);
    else if (key === "--combo-cut-end") out.comboCutEnd = safeNum(value, out.comboCutEnd);
    else if (key === "--combo-cut-step") out.comboCutStep = safeNum(value, out.comboCutStep);
    else if (key === "--combo-down-start") out.comboDownStart = safeNum(value, out.comboDownStart);
    else if (key === "--combo-down-end") out.comboDownEnd = safeNum(value, out.comboDownEnd);
    else if (key === "--combo-down-step") out.comboDownStep = safeNum(value, out.comboDownStep);
    else if (key === "--target-go-fail-rate") out.targetGoFailRate = safeNum(value, out.targetGoFailRate);
    else if (key === "--go-fail-penalty") out.goFailPenalty = safeNum(value, out.goFailPenalty);
    else if (key === "--gold-weight") out.goldWeight = safeNum(value, out.goldWeight);
    else if (key === "--gold-norm-div") out.goldNormDiv = safeNum(value, out.goldNormDiv);
    else if (key === "--top-k") out.topK = Math.max(1, Math.floor(safeNum(value, out.topK)));
    else if (key === "--max-candidates") out.maxCandidates = Math.max(0, Math.floor(safeNum(value, out.maxCandidates)));
    else if (key === "--dry-run") out.dryRun = value.trim() === "1";
    else fail(`Unknown argument: ${key}`);
  }

  if (!out.opponent) fail("--opponent is required");
  if (!out.seedBase) fail("--seed-base is required");
  if (!out.outDir) fail("--out-dir is required");
  if (out.games < 1000) fail("--games must be >= 1000 (duel worker contract)");

  validateRange("ratio-cut", out.ratioCutStart, out.ratioCutEnd, out.ratioCutStep);
  validateRange("ratio-down", out.ratioDownStart, out.ratioDownEnd, out.ratioDownStep);
  validateRange("combo-cut", out.comboCutStart, out.comboCutEnd, out.comboCutStep);
  validateRange("combo-down", out.comboDownStart, out.comboDownEnd, out.comboDownStep);
  if (out.goldNormDiv <= 0) fail("--gold-norm-div must be > 0");
  return out;
}

function validateRange(label, start, end, step) {
  if (!Number.isFinite(start) || !Number.isFinite(end) || !Number.isFinite(step)) {
    fail(`invalid ${label} range (non-finite number)`);
  }
  if (step <= 0) fail(`${label} step must be > 0`);
  if (end < start) fail(`${label} end must be >= start`);
}

function rangeValues(start, end, step) {
  const out = [];
  const eps = 1e-9;
  for (let x = start; x <= end + eps; x += step) {
    out.push(Number(x.toFixed(8)));
  }
  return out;
}

function buildCandidates(cfg) {
  const ratioCuts = rangeValues(cfg.ratioCutStart, cfg.ratioCutEnd, cfg.ratioCutStep);
  const ratioDowns = rangeValues(cfg.ratioDownStart, cfg.ratioDownEnd, cfg.ratioDownStep);
  const comboCuts = rangeValues(cfg.comboCutStart, cfg.comboCutEnd, cfg.comboCutStep);
  const comboDowns = rangeValues(cfg.comboDownStart, cfg.comboDownEnd, cfg.comboDownStep);
  const candidates = [];
  for (const ratioCut of ratioCuts) {
    for (const ratioDown of ratioDowns) {
      for (const comboCut of comboCuts) {
        for (const comboDown of comboDowns) {
          candidates.push({
            goDecisionAggRatioCut: ratioCut,
            goDecisionAggRatioThresholdDown: ratioDown,
            goDecisionAggComboCut: comboCut,
            goDecisionAggComboThresholdDown: comboDown
          });
        }
      }
    }
  }
  if (cfg.maxCandidates > 0) return candidates.slice(0, cfg.maxCandidates);
  return candidates;
}

function runNodeProcess(args, env, cwd, contextLabel) {
  return new Promise((resolvePromise, rejectPromise) => {
    const child = spawn(process.execPath, args, {
      cwd,
      env,
      stdio: ["ignore", "pipe", "pipe"]
    });

    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (buf) => {
      stdout += String(buf || "");
    });
    child.stderr.on("data", (buf) => {
      stderr += String(buf || "");
    });
    child.on("error", (err) => {
      rejectPromise(new Error(`[${contextLabel}] spawn failed: ${err?.message || err}`));
    });
    child.on("close", (code) => {
      if (code !== 0) {
        const errTail = stderr.trim().split(/\r?\n/).slice(-8).join("\n");
        rejectPromise(
          new Error(`[${contextLabel}] exit=${code}\n${errTail || stdout.trim() || "no output"}`)
        );
        return;
      }
      resolvePromise({ stdout, stderr });
    });
  });
}

function readJson(path, label) {
  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch (err) {
    fail(`[${label}] failed to parse JSON: ${path}\n${err?.message || err}`);
  }
}

function extractAntiMetrics(summary, label) {
  const human = String(summary?.human || "");
  const ai = String(summary?.ai || "");
  const side = human === "H-AntiCL" ? "a" : ai === "H-AntiCL" ? "b" : "";
  if (!side) fail(`[${label}] result does not contain H-AntiCL`);
  const games = safeNum(summary?.games);
  if (games <= 0) fail(`[${label}] invalid games`);
  const wins = safeNum(summary?.[`wins_${side}`]);
  const losses = safeNum(summary?.[`losses_${side}`]);
  const draws = safeNum(summary?.draws);
  const goGames = safeNum(summary?.[`go_games_${side}`]);
  const goFailCount = safeNum(summary?.[`go_fail_count_${side}`]);
  const meanGoldDelta = safeNum(summary?.[`mean_gold_delta_${side}`]);
  return {
    games,
    wins,
    losses,
    draws,
    goGames,
    goFailCount,
    meanGoldDelta
  };
}

function aggregateTwoRuns(a, b) {
  const games = a.games + b.games;
  const wins = a.wins + b.wins;
  const losses = a.losses + b.losses;
  const draws = a.draws + b.draws;
  const goGames = a.goGames + b.goGames;
  const goFailCount = a.goFailCount + b.goFailCount;
  const meanGoldDelta =
    games > 0 ? (a.meanGoldDelta * a.games + b.meanGoldDelta * b.games) / games : 0;
  const winRate = games > 0 ? wins / games : 0;
  const goFailRate = goGames > 0 ? goFailCount / goGames : 0;
  return {
    games,
    wins,
    losses,
    draws,
    goGames,
    goFailCount,
    meanGoldDelta,
    winRate,
    goFailRate
  };
}

function computeObjective(metrics, cfg) {
  const overFail = Math.max(0, metrics.goFailRate - cfg.targetGoFailRate);
  const goldNorm = metrics.meanGoldDelta / cfg.goldNormDiv;
  const score = metrics.winRate - cfg.goFailPenalty * overFail + cfg.goldWeight * goldNorm;
  return { score, overFail, goldNorm };
}

async function runTrial(cfg, candidate, trialIndex, totalTrials, outDir) {
  const trialTag = `trial_${String(trialIndex + 1).padStart(4, "0")}`;
  const trialDir = resolve(join(outDir, "trials", trialTag));
  mkdirSync(trialDir, { recursive: true });

  const antiHumanOut = resolve(join(trialDir, "anti_human.json"));
  const antiAiOut = resolve(join(trialDir, "anti_ai.json"));
  const paramsEnv = JSON.stringify(candidate);
  const baseEnv = { ...process.env, HEURISTIC_ANTICL_PARAMS: paramsEnv };

  const duelAArgs = [
    resolve("scripts/model_duel_worker.mjs"),
    "--human", "H-AntiCL",
    "--ai", cfg.opponent,
    "--games", String(cfg.games),
    "--seed", `${cfg.seedBase}_${trialTag}_a`,
    "--first-turn-policy", "alternate",
    "--continuous-series", "1",
    "--stdout-format", "json",
    "--result-out", antiHumanOut
  ];

  const duelBArgs = [
    resolve("scripts/model_duel_worker.mjs"),
    "--human", cfg.opponent,
    "--ai", "H-AntiCL",
    "--games", String(cfg.games),
    "--seed", `${cfg.seedBase}_${trialTag}_b`,
    "--first-turn-policy", "alternate",
    "--continuous-series", "1",
    "--stdout-format", "json",
    "--result-out", antiAiOut
  ];

  if (cfg.dryRun) {
    return {
      trialTag,
      candidate,
      dryRun: true,
      antiHumanOut,
      antiAiOut
    };
  }

  await Promise.all([
    runNodeProcess(duelAArgs, baseEnv, resolve("."), `${trialTag}:anti_human`),
    runNodeProcess(duelBArgs, baseEnv, resolve("."), `${trialTag}:anti_ai`)
  ]);

  const summaryA = readJson(antiHumanOut, `${trialTag}:anti_human`);
  const summaryB = readJson(antiAiOut, `${trialTag}:anti_ai`);
  const metricsA = extractAntiMetrics(summaryA, `${trialTag}:anti_human`);
  const metricsB = extractAntiMetrics(summaryB, `${trialTag}:anti_ai`);
  const merged = aggregateTwoRuns(metricsA, metricsB);
  const objective = computeObjective(merged, cfg);

  const row = {
    trialIndex: trialIndex + 1,
    totalTrials,
    trialTag,
    candidate,
    metrics: merged,
    objective,
    outputs: {
      antiHumanOut,
      antiAiOut
    }
  };
  writeFileSync(resolve(join(trialDir, "summary.json")), JSON.stringify(row, null, 2), "utf8");
  return row;
}

async function main() {
  const cfg = parseArgs(process.argv.slice(2));
  const outDir = resolve(cfg.outDir);
  mkdirSync(outDir, { recursive: true });
  mkdirSync(resolve(join(outDir, "trials")), { recursive: true });

  const candidates = buildCandidates(cfg);
  if (candidates.length <= 0) fail("No candidates generated");

  console.log("=== AntiCL GO Tune by GPT ===");
  console.log(`out_dir=${outDir}`);
  console.log(`opponent=${cfg.opponent}`);
  console.log(`games_per_duel=${cfg.games}`);
  console.log(`total_candidates=${candidates.length}`);
  if (cfg.dryRun) console.log("mode=dry_run");

  const rows = [];
  for (let i = 0; i < candidates.length; i += 1) {
    const row = await runTrial(cfg, candidates[i], i, candidates.length, outDir);
    rows.push(row);
    if (!cfg.dryRun) {
      console.log(
        `[trial ${i + 1}/${candidates.length}] score=${safeNum(row.objective?.score).toFixed(6)} win_rate=${safeNum(row.metrics?.winRate).toFixed(4)} go_fail_rate=${safeNum(row.metrics?.goFailRate).toFixed(4)} mean_gold=${safeNum(row.metrics?.meanGoldDelta).toFixed(2)}`
      );
    } else {
      console.log(`[trial ${i + 1}/${candidates.length}] dry-run candidate prepared`);
    }
  }

  const sorted = [...rows]
    .filter((r) => !r.dryRun)
    .sort((a, b) => safeNum(b?.objective?.score) - safeNum(a?.objective?.score));
  const best = sorted[0] || null;
  const topK = sorted.slice(0, cfg.topK);

  const summary = {
    generatedAt: new Date().toISOString(),
    config: cfg,
    totalCandidates: candidates.length,
    dryRun: cfg.dryRun,
    best,
    topK,
    allTrials: rows
  };
  const summaryPath = resolve(join(outDir, "tune_summary.json"));
  writeFileSync(summaryPath, JSON.stringify(summary, null, 2), "utf8");
  console.log(`summary=${summaryPath}`);

  if (best?.candidate) {
    const bestParamPath = resolve(join(outDir, "best_params.json"));
    writeFileSync(bestParamPath, JSON.stringify(best.candidate, null, 2), "utf8");
    console.log(`best_params=${bestParamPath}`);
    console.log(
      [
        "BEST",
        `score=${safeNum(best.objective?.score).toFixed(6)}`,
        `win_rate=${safeNum(best.metrics?.winRate).toFixed(4)}`,
        `go_fail_rate=${safeNum(best.metrics?.goFailRate).toFixed(4)}`,
        `mean_gold=${safeNum(best.metrics?.meanGoldDelta).toFixed(2)}`
      ].join("  ")
    );
  }
}

main().catch((err) => {
  console.error(err?.stack || err?.message || err);
  process.exit(1);
});
