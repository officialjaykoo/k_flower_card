// Optimizer by GPT
// Purpose:
// - Use kibo + dataset to build two practical parameter plans for GO/STOP.
// - Output two directions: defense plan and attack plan.
// - Keep only meaningful candidates (tiny effects are filtered out).
// - Forecast expected impact per plan (gold / wins / GO / fail deltas).
//
// Validation policy:
// - This tool provides directional plans, not final truth.
// - Final acceptance must be based on real 1000-game reruns.

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const DEFAULTS = {
  actor: "human",
  actorPolicy: null,
  paramsFile: "src/heuristics/heuristicV6.js",
  outRoot: null,
  minNetDecisionEv: 1.0,
  minGoldImpact: 200.0,
  minWinImpact: 2.0,
  minGoImpact: 10.0,
  minFailImpact: 2.0
};

const ZSCORE_95 = 1.96;

const RULES = [
  // Defense rules
  {
    direction: "defense",
    mode: "block",
    zoneKey: "first_go_low_edge",
    param: "goBaseThreshold",
    delta: +0.01,
    min: -0.2,
    max: 0.2,
    reason: "Raise threshold in weak first-GO states."
  },
  {
    direction: "defense",
    mode: "block",
    zoneKey: "first_go_low_edge",
    param: "goMinPi",
    delta: +1,
    min: 3,
    max: 12,
    reason: "Require stronger pi before risky first GO."
  },
  {
    direction: "defense",
    mode: "block",
    zoneKey: "opp_can_stop",
    param: "goLiteOppCanStopPenalty",
    delta: +0.02,
    min: 0,
    max: 1.0,
    reason: "Penalize GO more when opponent can stop."
  },
  {
    direction: "defense",
    mode: "block",
    zoneKey: "first_go_deck_low",
    param: "goLiteLatePenalty",
    delta: +0.01,
    min: 0,
    max: 1.0,
    reason: "Suppress late-deck fragile GO entries."
  },
  {
    direction: "defense",
    mode: "block",
    zoneKey: "opp_burst_risk",
    param: "goHardLateOneAwayCut",
    delta: -5,
    min: 30,
    max: 95,
    reason: "Block GO earlier under burst-risk signals."
  },

  // Attack rules
  {
    direction: "attack",
    mode: "expand",
    zoneKey: "go2plus",
    param: "goBaseThreshold",
    delta: -0.01,
    min: -0.2,
    max: 0.2,
    reason: "Lower threshold in empirically strong GO2+ states."
  },
  {
    direction: "attack",
    mode: "expand",
    zoneKey: "go2plus",
    param: "goMinPi",
    delta: -1,
    min: 3,
    max: 12,
    reason: "Relax pi gate for strong continuation contexts."
  },
  {
    direction: "attack",
    mode: "expand",
    zoneKey: "go2plus",
    param: "goScoreDiffBonus",
    delta: -0.005,
    min: 0,
    max: 1.0,
    reason: "Slightly relax score-gap gate in high-EV GO2+."
  },
  {
    direction: "attack",
    mode: "expand",
    zoneKey: "go2plus",
    param: "goDeckLowBonus",
    delta: +0.01,
    min: -1.0,
    max: 1.0,
    reason: "Allow more late continuation in validated GO2+ contexts."
  },
  {
    direction: "attack",
    mode: "expand",
    zoneKey: "go2plus",
    param: "goUnseeHighPiPenalty",
    delta: -0.01,
    min: 0,
    max: 1.0,
    reason: "Reduce over-penalization in high-quality GO2+ zones."
  },
  {
    direction: "attack",
    mode: "expand",
    zoneKey: "go3plus_strong",
    param: "goRiskGoCountMul",
    delta: -0.02,
    min: 0,
    max: 1.0,
    reason: "Preserve profitable GO3+ continuation when risk is controlled."
  }
];

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i += 1) {
    const key = String(argv[i] || "");
    if (!key.startsWith("--")) continue;
    const k = key.slice(2);
    const v = i + 1 < argv.length ? String(argv[i + 1] || "") : "";
    if (v && !v.startsWith("--")) {
      out[k] = v;
      i += 1;
    } else {
      out[k] = true;
    }
  }
  return out;
}

function resolveOutRoot(kiboPath, outRootArg) {
  if (outRootArg != null && String(outRootArg).trim() !== "") return String(outRootArg);
  return join(resolve(dirname(String(kiboPath))), "optimize_gpt");
}

function readJsonl(path) {
  const raw = readFileSync(path, "utf8").replace(/^\uFEFF/, "");
  if (!raw.trim()) return [];
  return raw
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function round(v, d = 3) {
  if (!Number.isFinite(v)) return 0;
  const s = 10 ** d;
  return Math.round(v * s) / s;
}

function pct(v) {
  return `${(Number(v) * 100).toFixed(1)}%`;
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function atanhSafe(x) {
  const c = clamp(Number(x) || 0, -0.999999, 0.999999);
  return 0.5 * Math.log((1 + c) / (1 - c));
}

function quantileSorted(sorted, q) {
  if (!Array.isArray(sorted) || sorted.length <= 0) return 0;
  const qq = clamp(Number(q) || 0, 0, 1);
  const pos = (sorted.length - 1) * qq;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  const t = pos - lo;
  return sorted[lo] * (1 - t) + sorted[hi] * t;
}

function wilsonBounds(successes, n, z = ZSCORE_95) {
  const nn = Math.max(0, Number(n) || 0);
  if (nn <= 0) return { low: 0, high: 1 };
  const ss = clamp(Number(successes) || 0, 0, nn);
  const p = ss / nn;
  const z2 = z * z;
  const den = 1 + z2 / nn;
  const center = (p + z2 / (2 * nn)) / den;
  const spread = (z / den) * Math.sqrt((p * (1 - p)) / nn + z2 / (4 * nn * nn));
  return { low: clamp(center - spread, 0, 1), high: clamp(center + spread, 0, 1) };
}

function calcStats(records) {
  const n = records.length;
  if (n <= 0) {
    return {
      n: 0,
      wins: 0,
      fails: 0,
      winRate: 0,
      failRate: 0,
      failCiLow: 0,
      failCiHigh: 1,
      ev: 0,
      avgWin: 0,
      avgLoss: 0,
      avgLossAbs: 0,
      p90LossAbs: 0
    };
  }
  const wins = records.filter((r) => Number(r.gold) > 0);
  const fails = records.filter((r) => Number(r.gold) <= 0);
  const totalGold = records.reduce((acc, r) => acc + Number(r.gold || 0), 0);
  const avgWin = wins.length > 0 ? wins.reduce((acc, r) => acc + Number(r.gold || 0), 0) / wins.length : 0;
  const avgLoss = fails.length > 0 ? fails.reduce((acc, r) => acc + Number(r.gold || 0), 0) / fails.length : 0;
  const lossAbs = fails
    .map((r) => Math.abs(Number(r.gold || 0)))
    .filter((x) => Number.isFinite(x))
    .sort((a, b) => a - b);
  const ci = wilsonBounds(fails.length, n);
  return {
    n,
    wins: wins.length,
    fails: fails.length,
    winRate: wins.length / n,
    failRate: fails.length / n,
    failCiLow: ci.low,
    failCiHigh: ci.high,
    ev: totalGold / n,
    avgWin,
    avgLoss,
    avgLossAbs: Math.abs(avgLoss),
    p90LossAbs: lossAbs.length > 0 ? quantileSorted(lossAbs, 0.9) : 0
  };
}

function buildPayoutMap(kiboList, actor) {
  const opp = actor === "ai" ? "human" : "ai";
  const out = new Map();
  for (const game of kiboList) {
    const gi = Number(game.game_index);
    let winner = game.winner ?? game.result?.winner ?? null;
    let scores = game.result ?? null;
    const roundEnd = Array.isArray(game.kibo)
      ? [...game.kibo].reverse().find((k) => k?.type === "round_end")
      : null;
    if (roundEnd) {
      winner = roundEnd.winner ?? winner;
      scores = roundEnd.scores ?? scores;
    }
    let gold = 0;
    if (winner === actor) gold = Number(scores?.[actor]?.payoutTotal ?? 0);
    else if (winner === opp) gold = -Number(scores?.[opp]?.payoutTotal ?? 0);
    out.set(gi, gold);
  }
  return out;
}

function calcGameStats(payoutByGame) {
  const vals = [...payoutByGame.values()].map((x) => Number(x) || 0);
  const wins = vals.filter((x) => x > 0);
  const losses = vals.filter((x) => x < 0);
  const draws = vals.filter((x) => x === 0);
  const avgWin = wins.length > 0 ? wins.reduce((a, b) => a + b, 0) / wins.length : 0;
  const avgLossAbs = losses.length > 0 ? Math.abs(losses.reduce((a, b) => a + b, 0) / losses.length) : 0;
  return {
    games: vals.length,
    wins: wins.length,
    losses: losses.length,
    draws: draws.length,
    meanGoldDelta: vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : 0,
    perWinSwing: avgWin + avgLossAbs
  };
}

function decodeFeatures(f) {
  const deck = Math.round((Number(f?.[9]) || 0) * 30);
  const goCount = Math.round((Number(f?.[12]) || 0) * 5) + 1;
  const selfPi = Math.round((Number(f?.[28]) || 0) * 20);
  const oppPi = Math.round((Number(f?.[29]) || 0) * 20);
  const scoreDiff = atanhSafe(Number(f?.[14]) || 0) * 10.0;
  const selfScore = Math.max(0, atanhSafe(Number(f?.[15]) || 0) * 10.0);
  const oppScore = Math.max(0, selfScore - scoreDiff);
  const oppGwang = Math.round((Number(f?.[27]) || 0) * 5);
  const oppGodori = Math.round((Number(f?.[31]) || 0) * 3);
  const oppCheongdan = Math.round((Number(f?.[33]) || 0) * 3);
  const oppHongdan = Math.round((Number(f?.[35]) || 0) * 3);
  const oppChodan = Math.round((Number(f?.[37]) || 0) * 3);
  const oppComboNearCount = [oppGodori, oppCheongdan, oppHongdan, oppChodan].filter((x) => x >= 2).length;
  const oppBurstSignals = (oppGwang >= 2 ? 1 : 0) + (oppComboNearCount > 0 ? 1 : 0) + (oppScore >= 6 ? 1 : 0);
  return {
    deck,
    goCount,
    selfPi,
    oppPi,
    piDiff: selfPi - oppPi,
    scoreDiff,
    selfScore,
    oppScore,
    oppComboNearCount,
    oppBurstSignals,
    selfCanStop: Number(f?.[38]) > 0.5,
    oppCanStop: Number(f?.[39]) > 0.5
  };
}

function collectChosenGoStop(dataset, actor, actorPolicy) {
  return dataset
    .filter((r) => {
      if (r.decision_type !== "option") return false;
      if (r.actor !== actor) return false;
      if (actorPolicy && r.actor_policy !== actorPolicy) return false;
      if (Number(r.chosen) !== 1) return false;
      return r.candidate === "go" || r.candidate === "stop";
    })
    .map((r) => ({
      game: Number(r.game_index),
      action: r.candidate,
      features: r.features
    }));
}

function extractReferencedParamKeys(paramsFile) {
  const src = readFileSync(resolve(paramsFile), "utf8").replace(/^\uFEFF/, "");
  const keys = new Set();
  const re = /\b(?:P|params)\.([A-Za-z_][A-Za-z0-9_]*)\b/g;
  let m;
  while ((m = re.exec(src)) != null) keys.add(String(m[1]));
  return keys;
}

async function loadCurrentParams(paramsFile) {
  const mod = await import(pathToFileURL(resolve(paramsFile)).href);
  if (!mod || typeof mod.DEFAULT_PARAMS !== "object" || !mod.DEFAULT_PARAMS) {
    throw new Error(`DEFAULT_PARAMS not found: ${paramsFile}`);
  }
  return mod.DEFAULT_PARAMS;
}

function zoneFns() {
  return {
    first_go_low_edge: (r) => r.goCount === 1 && (r.deck <= 4 || r.piDiff < 5),
    first_go_deck_low: (r) => r.goCount === 1 && r.deck <= 4,
    first_go_pi_low: (r) => r.goCount === 1 && r.piDiff < 5,
    opp_can_stop: (r) => r.oppCanStop,
    opp_combo_near: (r) => r.oppComboNearCount >= 1,
    opp_burst_risk: (r) => r.oppBurstSignals >= 2,
    go2plus: (r) => r.goCount >= 2,
    go3plus_strong: (r) => r.goCount >= 3 && r.piDiff >= 10
  };
}

function evaluateRule(rule, currentParams, referenced, goRecs, stopRecs, fnMap, global, cfg) {
  if (!(rule.param in currentParams)) return null;
  if (!referenced.has(rule.param)) return null;
  const fn = fnMap[rule.zoneKey];
  if (!fn) return null;
  const zGoRecs = goRecs.filter(fn);
  const zStopRecs = stopRecs.filter(fn);
  const goZ = calcStats(zGoRecs);
  const stopZ = calcStats(zStopRecs);
  if (goZ.n <= 0 && stopZ.n <= 0) return null;

  const confidence = clamp(Math.sqrt((goZ.n + stopZ.n) / 80), 0.2, 1.0);
  const baseFlipRate = 0.35;

  let flipCount = 0;
  let goDelta = 0;
  let failDelta = 0;
  let goldDelta = 0;

  if (rule.mode === "block") {
    if (goZ.n < 5) return null;
    const edgeGold = stopZ.ev - goZ.ev;
    const edgeFail = goZ.failRate - stopZ.failRate;
    const reliableRisk = goZ.failCiLow > global.go.failCiHigh || goZ.n >= 20;
    if (!reliableRisk && edgeGold <= 0) return null;
    if (edgeGold <= 0 && edgeFail <= 0) return null;
    const oppCostMul = clamp(
      global.stop.ev > 1e-9 ? Math.max(goZ.avgLossAbs, goZ.p90LossAbs) / Math.max(global.stop.ev, 1e-9) : 1,
      0.7,
      1.8
    );
    flipCount = goZ.n * baseFlipRate * confidence * oppCostMul;
    goDelta = -flipCount;
    failDelta = -flipCount * Math.max(0, edgeFail);
    goldDelta = flipCount * edgeGold;
  } else {
    if (stopZ.n < 6 || goZ.n < 6) return null;
    const edgeGold = goZ.ev - stopZ.ev;
    if (edgeGold <= 0) return null;
    if (goZ.failRate > Math.min(0.24, global.go.failRate + 0.07)) return null;
    const reliableAttack = goZ.failCiHigh <= Math.max(global.go.failRate, 0.12) || goZ.n >= 20;
    if (!reliableAttack) return null;
    flipCount = stopZ.n * baseFlipRate * confidence;
    goDelta = flipCount;
    failDelta = flipCount * Math.max(0, goZ.failRate - stopZ.failRate);
    goldDelta = flipCount * edgeGold;
  }

  const games = Math.max(1, global.game.games);
  const netDecisionEv = goldDelta / Math.max(1, global.optionN);
  const winDelta = global.game.perWinSwing > 1e-9 ? goldDelta / global.game.perWinSwing : 0;
  const suggested = clamp(
    Number(currentParams[rule.param]) + rule.delta,
    Number.isFinite(rule.min) ? rule.min : Number.NEGATIVE_INFINITY,
    Number.isFinite(rule.max) ? rule.max : Number.POSITIVE_INFINITY
  );
  if (round(suggested, 6) === round(Number(currentParams[rule.param]), 6)) return null;

  const rec = {
    direction: rule.direction,
    mode: rule.mode,
    zone_key: rule.zoneKey,
    zone_go_n: goZ.n,
    zone_stop_n: stopZ.n,
    zone_go_fail_rate: goZ.failRate,
    zone_go_fail_ci_low: goZ.failCiLow,
    zone_go_fail_ci_high: goZ.failCiHigh,
    zone_stop_fail_rate: stopZ.failRate,
    zone_go_ev: goZ.ev,
    zone_stop_ev: stopZ.ev,
    confidence,
    param: rule.param,
    current: Number(currentParams[rule.param]),
    suggested,
    reason: rule.reason,
    expected_total_gold_delta: goldDelta,
    expected_gold_per_game: goldDelta / games,
    expected_win_delta: winDelta,
    expected_win_rate_delta: winDelta / games,
    expected_go_delta_count: goDelta,
    expected_fail_delta_count: failDelta,
    net_decision_ev_delta: netDecisionEv
  };

  const isMeaningful =
    netDecisionEv >= cfg.minNetDecisionEv &&
    (
      Math.abs(rec.expected_total_gold_delta) >= cfg.minGoldImpact ||
      Math.abs(rec.expected_win_delta) >= cfg.minWinImpact ||
      Math.abs(rec.expected_go_delta_count) >= cfg.minGoImpact ||
      Math.abs(rec.expected_fail_delta_count) >= cfg.minFailImpact
    );
  if (!isMeaningful) return null;
  return rec;
}

function dedupeBestByParamDirection(records) {
  const out = new Map();
  for (const r of records) {
    const k = `${r.direction}:${r.param}`;
    const prev = out.get(k);
    if (!prev || Number(r.expected_total_gold_delta) > Number(prev.expected_total_gold_delta)) out.set(k, r);
  }
  return [...out.values()];
}

function buildPlan(records, direction, gameStats) {
  const list = records.filter((r) => r.direction === direction);
  const sorted = [...list].sort((a, b) => {
    if (b.net_decision_ev_delta !== a.net_decision_ev_delta) return b.net_decision_ev_delta - a.net_decision_ev_delta;
    if (b.expected_total_gold_delta !== a.expected_total_gold_delta) {
      return b.expected_total_gold_delta - a.expected_total_gold_delta;
    }
    return a.expected_fail_delta_count - b.expected_fail_delta_count;
  });
  const zoneUse = new Map();
  const scaled = sorted.map((r) => {
    const z = String(r.zone_key || "unknown");
    const used = Number(zoneUse.get(z) || 0);
    // Diminishing-return correction for overlapping same-zone recommendations.
    const scale = clamp(1 / (1 + used * 0.75), 0.4, 1);
    zoneUse.set(z, used + 1);
    return {
      ...r,
      applied_scale: scale,
      expected_total_gold_delta_scaled: Number(r.expected_total_gold_delta || 0) * scale,
      expected_win_delta_scaled: Number(r.expected_win_delta || 0) * scale,
      expected_go_delta_count_scaled: Number(r.expected_go_delta_count || 0) * scale,
      expected_fail_delta_count_scaled: Number(r.expected_fail_delta_count || 0) * scale
    };
  });

  const agg = scaled.reduce(
    (acc, r) => {
      acc.expected_total_gold_delta += Number(r.expected_total_gold_delta_scaled || 0);
      acc.expected_win_delta += Number(r.expected_win_delta_scaled || 0);
      acc.expected_go_delta_count += Number(r.expected_go_delta_count_scaled || 0);
      acc.expected_fail_delta_count += Number(r.expected_fail_delta_count_scaled || 0);
      return acc;
    },
    {
      expected_total_gold_delta: 0,
      expected_win_delta: 0,
      expected_go_delta_count: 0,
      expected_fail_delta_count: 0
    }
  );
  return {
    direction,
    recommendation_count: scaled.length,
    forecast: {
      expected_total_gold_delta: round(agg.expected_total_gold_delta, 3),
      expected_gold_per_game: round(gameStats.games > 0 ? agg.expected_total_gold_delta / gameStats.games : 0, 6),
      expected_win_delta: round(agg.expected_win_delta, 3),
      expected_win_rate_delta: round(gameStats.games > 0 ? agg.expected_win_delta / gameStats.games : 0, 6),
      expected_go_delta_count: round(agg.expected_go_delta_count, 3),
      expected_fail_delta_count: round(agg.expected_fail_delta_count, 3)
    },
    recommendations: scaled.map((r, i) => ({
      rank: i + 1,
      ...Object.fromEntries(Object.entries(r).map(([k, v]) => [k, typeof v === "number" ? round(v, 6) : v]))
    }))
  };
}

function buildSummaryText(payload) {
  const lines = [];
  lines.push("Optimizer by GPT Summary");
  lines.push(`Generated: ${payload.generated_at}`);
  lines.push(`Actor: ${payload.input.actor} (${payload.input.actor_policy})`);
  lines.push(`Kibo: ${payload.input.kibo}`);
  lines.push(`Dataset: ${payload.input.dataset}`);
  lines.push(`Params: ${payload.input.params_file}`);
  lines.push("");
  lines.push("Baseline:");
  lines.push(
    `- games=${payload.baseline.games}, WR=${pct(payload.baseline.win_rate)}, meanGoldDelta=${round(payload.baseline.mean_gold_delta, 3)}`
  );
  lines.push(
    `- GO n=${payload.baseline.go.n}, fail=${pct(payload.baseline.go.fail_rate)}, EV=${round(payload.baseline.go.ev, 3)}`
  );
  lines.push(
    `- STOP n=${payload.baseline.stop.n}, fail=${pct(payload.baseline.stop.fail_rate)}, EV=${round(payload.baseline.stop.ev, 3)}`
  );
  lines.push("");
  lines.push(
    `Filter thresholds: netEV>=${payload.thresholds.min_net_decision_ev}, |gold|>=${payload.thresholds.min_gold_impact}, |win|>=${payload.thresholds.min_win_impact}, |GO|>=${payload.thresholds.min_go_impact}, |fail|>=${payload.thresholds.min_fail_impact}`
  );
  lines.push("");

  const planOrder = ["defense", "attack"];
  for (const key of planOrder) {
    const plan = payload.plans[key];
    lines.push(`${key.toUpperCase()} Plan:`);
    lines.push(
      `- recommendations=${plan.recommendation_count}, expectedGold=${round(
        plan.forecast.expected_total_gold_delta,
        3
      )}, expectedWins=${round(plan.forecast.expected_win_delta, 3)}, GO delta=${round(
        plan.forecast.expected_go_delta_count,
        3
      )}, fail delta=${round(plan.forecast.expected_fail_delta_count, 3)}`
    );
    if (!Array.isArray(plan.recommendations) || plan.recommendations.length <= 0) {
      lines.push("- (no meaningful recommendation)");
    } else {
      for (const r of plan.recommendations) {
        lines.push(
          `- #${r.rank} ${r.param}: ${r.current} -> ${r.suggested} | zone=${r.zone_key} | gold=${round(
            r.expected_total_gold_delta,
            2
          )} | wins=${round(r.expected_win_delta, 2)} | GO=${round(r.expected_go_delta_count, 2)} | fail=${round(
            r.expected_fail_delta_count,
            2
          )}`
        );
        lines.push(`  reason: ${r.reason}`);
      }
    }
    lines.push("");
  }
  return `${lines.join("\n")}\n`;
}

const argv = parseArgs(process.argv.slice(2));
if (argv.help) {
  console.log(`Usage:
node scripts/optimizer_by_gpt.mjs --kibo <kibo.jsonl> --dataset <dataset.jsonl> --actor-policy <POLICY> --params-file <heuristic.js> [--actor human|ai]

Outputs:
  <match_dir>/optimize_gpt/optimizer_gpt_summary.txt
  <match_dir>/optimize_gpt/optimizer_gpt_plan.json
`);
  process.exit(0);
}

const cfg = {
  kibo: String(argv.kibo || ""),
  dataset: String(argv.dataset || ""),
  actor: String(argv.actor || DEFAULTS.actor).toLowerCase(),
  actorPolicy: argv["actor-policy"] == null ? null : String(argv["actor-policy"]),
  paramsFile: String(argv["params-file"] || DEFAULTS.paramsFile),
  outRoot: resolveOutRoot(String(argv.kibo || ""), argv["out-root"] ?? DEFAULTS.outRoot),
  minNetDecisionEv: Number(argv["min-net-decision-ev"] || DEFAULTS.minNetDecisionEv),
  minGoldImpact: Number(argv["min-gold-impact"] || DEFAULTS.minGoldImpact),
  minWinImpact: Number(argv["min-win-impact"] || DEFAULTS.minWinImpact),
  minGoImpact: Number(argv["min-go-impact"] || DEFAULTS.minGoImpact),
  minFailImpact: Number(argv["min-fail-impact"] || DEFAULTS.minFailImpact)
};

if (!cfg.kibo) throw new Error("missing --kibo");
if (!cfg.dataset) throw new Error("missing --dataset");
if (cfg.actor !== "ai" && cfg.actor !== "human") throw new Error(`invalid --actor: ${cfg.actor}`);
if (!cfg.actorPolicy || String(cfg.actorPolicy).trim() === "") throw new Error("missing --actor-policy <POLICY>");
if (!existsSync(cfg.kibo)) throw new Error(`kibo file not found: ${cfg.kibo}`);
if (!existsSync(cfg.dataset)) throw new Error(`dataset file not found: ${cfg.dataset}`);
if (!existsSync(cfg.paramsFile)) throw new Error(`params file not found: ${cfg.paramsFile}`);

const kibo = readJsonl(cfg.kibo);
const dataset = readJsonl(cfg.dataset);
const payoutByGame = buildPayoutMap(kibo, cfg.actor);
const gameStats = calcGameStats(payoutByGame);
const chosen = collectChosenGoStop(dataset, cfg.actor, cfg.actorPolicy);
const allRecords = chosen.map((r) => ({
  game: r.game,
  action: r.action,
  gold: Number(payoutByGame.get(r.game) ?? 0),
  ...decodeFeatures(r.features)
}));

const goRecs = allRecords.filter((r) => r.action === "go");
const stopRecs = allRecords.filter((r) => r.action === "stop");
const goStats = calcStats(goRecs);
const stopStats = calcStats(stopRecs);
const optionN = Math.max(1, goRecs.length + stopRecs.length);

const currentParams = await loadCurrentParams(cfg.paramsFile);
const referenced = extractReferencedParamKeys(cfg.paramsFile);
const zones = zoneFns();
const global = {
  go: goStats,
  stop: stopStats,
  game: gameStats,
  optionN
};

const evaluated = [];
for (const rule of RULES) {
  const rec = evaluateRule(rule, currentParams, referenced, goRecs, stopRecs, zones, global, cfg);
  if (rec) evaluated.push(rec);
}
const unique = dedupeBestByParamDirection(evaluated);

const defensePlan = buildPlan(unique, "defense", gameStats);
const attackPlan = buildPlan(unique, "attack", gameStats);

const payload = {
  generated_at: new Date().toISOString(),
  input: {
    kibo: cfg.kibo,
    dataset: cfg.dataset,
    actor: cfg.actor,
    actor_policy: cfg.actorPolicy,
    params_file: cfg.paramsFile
  },
  thresholds: {
    min_net_decision_ev: cfg.minNetDecisionEv,
    min_gold_impact: cfg.minGoldImpact,
    min_win_impact: cfg.minWinImpact,
    min_go_impact: cfg.minGoImpact,
    min_fail_impact: cfg.minFailImpact
  },
  baseline: {
    games: gameStats.games,
    wins: gameStats.wins,
    losses: gameStats.losses,
    draws: gameStats.draws,
    win_rate: gameStats.games > 0 ? gameStats.wins / gameStats.games : 0,
    mean_gold_delta: gameStats.meanGoldDelta,
    go: {
      n: goStats.n,
      fail_rate: goStats.failRate,
      fail_ci_low: goStats.failCiLow,
      fail_ci_high: goStats.failCiHigh,
      ev: goStats.ev
    },
    stop: {
      n: stopStats.n,
      fail_rate: stopStats.failRate,
      fail_ci_low: stopStats.failCiLow,
      fail_ci_high: stopStats.failCiHigh,
      ev: stopStats.ev
    }
  },
  plans: {
    defense: defensePlan,
    attack: attackPlan
  },
  all_candidates: unique.map((r) => ({
    ...Object.fromEntries(Object.entries(r).map(([k, v]) => [k, typeof v === "number" ? round(v, 6) : v]))
  }))
};

mkdirSync(cfg.outRoot, { recursive: true });
const summaryPath = join(cfg.outRoot, "optimizer_gpt_summary.txt");
const jsonPath = join(cfg.outRoot, "optimizer_gpt_plan.json");
writeFileSync(summaryPath, buildSummaryText(payload), "utf8");
writeFileSync(jsonPath, JSON.stringify(payload, null, 2), "utf8");

console.log("=== Optimizer by GPT ===");
console.log(`run_dir: ${cfg.outRoot}`);
console.log(`summary: ${summaryPath}`);
console.log(`json:    ${jsonPath}`);
console.log(
  `defense=${defensePlan.recommendation_count}, attack=${attackPlan.recommendation_count}, baseline WR=${round(
    (gameStats.games > 0 ? gameStats.wins / gameStats.games : 0) * 100,
    2
  )}%`
);

