// AntiCL duel result analyzer for GPT-authored heuristic tuning runs.
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

function fail(msg) {
  throw new Error(msg);
}

function safeNum(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function parseArgs(argv) {
  const args = [...argv];
  const out = {
    results: [],
    goLogs: [],
    jsonOut: "",
  };

  while (args.length > 0) {
    const key = String(args.shift() || "");
    if (key === "--result") {
      const value = String(args.shift() || "").trim();
      if (!value) fail("--result requires a path");
      out.results.push(value);
      continue;
    }
    if (key === "--go-log") {
      const value = String(args.shift() || "").trim();
      if (!value) fail("--go-log requires a path");
      out.goLogs.push(value);
      continue;
    }
    if (key === "--json-out") {
      const value = String(args.shift() || "").trim();
      if (!value) fail("--json-out requires a path");
      out.jsonOut = value;
      continue;
    }
    fail(`Unknown argument: ${key}`);
  }

  if (out.results.length <= 0) {
    fail("At least one --result is required");
  }
  return out;
}

function readJson(path) {
  const abs = resolve(path);
  try {
    return JSON.parse(readFileSync(abs, "utf8"));
  } catch (err) {
    fail(`Failed to read JSON: ${abs}\n${err?.message || err}`);
  }
}

function extractAntiSide(summary, pathLabel) {
  const human = String(summary?.human || "");
  const ai = String(summary?.ai || "");
  if (human === "H-AntiCL") return "a";
  if (ai === "H-AntiCL") return "b";
  fail(`H-AntiCL not found in result file: ${pathLabel}`);
}

function sideMetric(summary, side, baseKey) {
  return safeNum(summary?.[`${baseKey}_${side}`]);
}

function parseResultSummary(path) {
  const summary = readJson(path);
  const side = extractAntiSide(summary, path);
  const games = safeNum(summary?.games);
  if (games <= 0) fail(`Invalid games in result file: ${path}`);

  const wins = sideMetric(summary, side, "wins");
  const losses = sideMetric(summary, side, "losses");
  const draws = safeNum(summary?.draws);
  const goCount = sideMetric(summary, side, "go_count");
  const goGames = sideMetric(summary, side, "go_games");
  const goFailCount = sideMetric(summary, side, "go_fail_count");
  const meanKey = `mean_gold_delta_${side}`;
  let meanGoldDelta = 0;
  if (Object.prototype.hasOwnProperty.call(summary, meanKey)) {
    meanGoldDelta = safeNum(summary?.[meanKey]);
  } else if (
    side === "b" &&
    Object.prototype.hasOwnProperty.call(summary, "mean_gold_delta_a")
  ) {
    // Some duel summaries only publish side-a gold delta at top-level.
    meanGoldDelta = -safeNum(summary?.mean_gold_delta_a);
  }
  const winRate = sideMetric(summary, side, "win_rate");
  const goFailRate = goGames > 0 ? goFailCount / goGames : 0;

  return {
    path: resolve(path),
    side,
    games,
    wins,
    losses,
    draws,
    winRate,
    goCount,
    goGames,
    goFailCount,
    goFailRate,
    meanGoldDelta
  };
}

function parseGoDecisionLog(path) {
  const abs = resolve(path);
  let text = "";
  try {
    text = readFileSync(abs, "utf8");
  } catch (err) {
    fail(`Failed to read GO log: ${abs}\n${err?.message || err}`);
  }

  const lines = text.split(/\r?\n/);
  const out = {
    path: abs,
    decisionLogs: 0,
    ambiguousLogs: 0,
    hardStops: 0,
    hardStopByReason: {},
    aggressiveTaggedDecisions: 0,
    goDecisions: 0,
    stopDecisions: 0
  };

  for (const line of lines) {
    const row = String(line || "").trim();
    if (!row) continue;

    if (row.startsWith("[AntiCL][GO_DECISION]")) {
      const jsonIdx = row.indexOf("{");
      if (jsonIdx < 0) continue;
      let payload;
      try {
        payload = JSON.parse(row.slice(jsonIdx));
      } catch {
        continue;
      }
      out.decisionLogs += 1;
      const decision = String(payload?.decision || "").toUpperCase();
      if (decision === "GO") out.goDecisions += 1;
      else if (decision === "STOP") out.stopDecisions += 1;
      const aggressive = Array.isArray(payload?.ruleHitsAggressive) && payload.ruleHitsAggressive.length > 0;
      if (aggressive) out.aggressiveTaggedDecisions += 1;
      continue;
    }

    if (row.startsWith("[AntiCL][GO_AMBIGUOUS]")) {
      out.ambiguousLogs += 1;
      continue;
    }

    if (row.startsWith("[AntiCL][GO_HARDSTOP]")) {
      out.hardStops += 1;
      const reason = row.replace("[AntiCL][GO_HARDSTOP]", "").trim().split(/\s+/)[0] || "UNKNOWN";
      out.hardStopByReason[reason] = safeNum(out.hardStopByReason[reason]) + 1;
    }
  }

  return out;
}

function aggregateResultRows(rows) {
  const totals = {
    files: rows.length,
    games: 0,
    wins: 0,
    losses: 0,
    draws: 0,
    goCount: 0,
    goGames: 0,
    goFailCount: 0,
    weightedGoldSum: 0
  };
  for (const row of rows) {
    totals.games += row.games;
    totals.wins += row.wins;
    totals.losses += row.losses;
    totals.draws += row.draws;
    totals.goCount += row.goCount;
    totals.goGames += row.goGames;
    totals.goFailCount += row.goFailCount;
    totals.weightedGoldSum += row.meanGoldDelta * row.games;
  }
  const winRate = totals.games > 0 ? totals.wins / totals.games : 0;
  const goFailRate = totals.goGames > 0 ? totals.goFailCount / totals.goGames : 0;
  const meanGoldDelta = totals.games > 0 ? totals.weightedGoldSum / totals.games : 0;
  return { ...totals, winRate, goFailRate, meanGoldDelta };
}

function aggregateGoLogRows(rows) {
  const total = {
    files: rows.length,
    decisionLogs: 0,
    ambiguousLogs: 0,
    hardStops: 0,
    aggressiveTaggedDecisions: 0,
    goDecisions: 0,
    stopDecisions: 0,
    hardStopByReason: {}
  };
  for (const row of rows) {
    total.decisionLogs += row.decisionLogs;
    total.ambiguousLogs += row.ambiguousLogs;
    total.hardStops += row.hardStops;
    total.aggressiveTaggedDecisions += row.aggressiveTaggedDecisions;
    total.goDecisions += row.goDecisions;
    total.stopDecisions += row.stopDecisions;
    for (const [reason, count] of Object.entries(row.hardStopByReason || {})) {
      total.hardStopByReason[reason] = safeNum(total.hardStopByReason[reason]) + safeNum(count);
    }
  }
  total.aggressiveTagRate =
    total.decisionLogs > 0 ? total.aggressiveTaggedDecisions / total.decisionLogs : 0;
  return total;
}

function printResultRows(rows, totals) {
  console.log("=== AntiCL Result Summary ===");
  for (const row of rows) {
    console.log(
      [
        `file=${row.path}`,
        `games=${row.games}`,
        `win_rate=${row.winRate.toFixed(4)}`,
        `mean_gold=${row.meanGoldDelta.toFixed(2)}`,
        `go_games=${row.goGames}`,
        `go_fail_rate=${row.goFailRate.toFixed(4)}`
      ].join("  ")
    );
  }
  console.log(
    [
      "TOTAL",
      `files=${totals.files}`,
      `games=${totals.games}`,
      `win_rate=${totals.winRate.toFixed(4)}`,
      `mean_gold=${totals.meanGoldDelta.toFixed(2)}`,
      `go_games=${totals.goGames}`,
      `go_fail_rate=${totals.goFailRate.toFixed(4)}`
    ].join("  ")
  );
}

function printGoLogRows(rows, totals) {
  if (rows.length <= 0) {
    console.log("GO decision log analysis: skipped (no --go-log provided)");
    return;
  }
  console.log("=== AntiCL GO Decision Log Summary ===");
  for (const row of rows) {
    console.log(
      [
        `file=${row.path}`,
        `decision_logs=${row.decisionLogs}`,
        `ambiguous=${row.ambiguousLogs}`,
        `hard_stops=${row.hardStops}`,
        `agg_tagged=${row.aggressiveTaggedDecisions}`,
        `go_decisions=${row.goDecisions}`,
        `stop_decisions=${row.stopDecisions}`
      ].join("  ")
    );
  }
  console.log(
    [
      "TOTAL",
      `files=${totals.files}`,
      `decision_logs=${totals.decisionLogs}`,
      `ambiguous=${totals.ambiguousLogs}`,
      `hard_stops=${totals.hardStops}`,
      `agg_tag_rate=${totals.aggressiveTagRate.toFixed(4)}`
    ].join("  ")
  );
  const reasons = Object.entries(totals.hardStopByReason || {}).sort((a, b) => b[1] - a[1]);
  if (reasons.length > 0) {
    console.log("hard_stop_reasons:");
    for (const [reason, count] of reasons) {
      console.log(`  ${reason}: ${count}`);
    }
  }
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const resultRows = args.results.map(parseResultSummary);
  const resultTotals = aggregateResultRows(resultRows);
  const goLogRows = args.goLogs.map(parseGoDecisionLog);
  const goLogTotals = aggregateGoLogRows(goLogRows);

  printResultRows(resultRows, resultTotals);
  printGoLogRows(goLogRows, goLogTotals);

  if (args.jsonOut) {
    const payload = {
      generatedAt: new Date().toISOString(),
      results: {
        rows: resultRows,
        totals: resultTotals
      },
      goLogs: {
        rows: goLogRows,
        totals: goLogTotals
      }
    };
    writeFileSync(resolve(args.jsonOut), JSON.stringify(payload, null, 2), "utf8");
    console.log(`json_out=${resolve(args.jsonOut)}`);
  }
}

main();
