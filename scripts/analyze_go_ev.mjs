// GO/STOP analyzer
// Fixed outputs:
// 1) go_stop_summary.txt
// 2) go_stop_param_suggestions.json

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const DEFAULT_INPUT = {
  kibo: "logs/model_duel/v5_vs_v6_go_diag_20260227/h-v5_vs_h-v6_1000_kibo2.jsonl",
  dataset: "logs/model_duel/v5_vs_v6_go_diag_20260227/h-v5_vs_h-v6_1000_dataset2.jsonl",
  actor: "ai",
  actorPolicy: null,
  paramsFile: "src/heuristics/heuristicV6.js",
  outRoot: null
};

// ---------------------------------------------------------------------------
// CLI / IO helpers
// ---------------------------------------------------------------------------
function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i += 1) {
    const a = String(argv[i] || "");
    if (!a.startsWith("--")) continue;
    const k = a.slice(2);
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
  if (outRootArg != null && String(outRootArg).trim() !== "") {
    return String(outRootArg);
  }
  const kd = resolve(dirname(String(kiboPath)));
  return join(kd, "analyze");
}

function readJsonl(path) {
  const raw = readFileSync(path, "utf8").replace(/^\uFEFF/, "");
  if (!raw.trim()) return [];
  return raw
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

// ---------------------------------------------------------------------------
// Numeric helpers
// ---------------------------------------------------------------------------
function round(v, d = 3) {
  if (!Number.isFinite(v)) return 0;
  const s = 10 ** d;
  return Math.round(v * s) / s;
}

function pct(v) {
  return `${(Number(v) * 100).toFixed(1)}%`;
}

function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

// ---------------------------------------------------------------------------
// Core metric helpers
// ---------------------------------------------------------------------------
function calcStats(records) {
  const n = records.length;
  if (!n) {
    return {
      n: 0,
      wins: 0,
      fails: 0,
      winRate: 0,
      failRate: 0,
      avgWin: 0,
      avgLoss: 0,
      ev: 0,
      totalGold: 0,
      breakEvenRate: 0
    };
  }
  const wins = records.filter((r) => Number(r.gold) > 0);
  const fails = records.filter((r) => Number(r.gold) <= 0);
  const totalGold = records.reduce((acc, r) => acc + Number(r.gold || 0), 0);
  const avgWin = wins.length ? wins.reduce((acc, r) => acc + Number(r.gold || 0), 0) / wins.length : 0;
  const avgLoss = fails.length ? fails.reduce((acc, r) => acc + Number(r.gold || 0), 0) / fails.length : 0;
  const den = avgWin - avgLoss;
  const breakEvenRate = den !== 0 ? -avgLoss / den : 0;
  return {
    n,
    wins: wins.length,
    fails: fails.length,
    winRate: wins.length / n,
    failRate: fails.length / n,
    avgWin,
    avgLoss,
    ev: totalGold / n,
    totalGold,
    breakEvenRate
  };
}

// ---------------------------------------------------------------------------
// Kibo / dataset extraction helpers
// ---------------------------------------------------------------------------
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
    if (winner === actor) {
      gold = Number(scores?.[actor]?.payoutTotal ?? 0);
    } else if (winner === opp) {
      gold = -Number(scores?.[opp]?.payoutTotal ?? 0);
    }
    out.set(gi, gold);
  }
  return out;
}

function calcGameGoldStats(payoutByGame) {
  const vals = [...payoutByGame.values()].map((v) => Number(v) || 0);
  const wins = vals.filter((v) => v > 0);
  const losses = vals.filter((v) => v < 0);
  const draws = vals.filter((v) => v === 0);
  const avgWinGold = wins.length > 0 ? wins.reduce((a, b) => a + b, 0) / wins.length : 0;
  const avgLossGoldAbs =
    losses.length > 0 ? Math.abs(losses.reduce((a, b) => a + b, 0) / losses.length) : 0;
  const perWinSwing = avgWinGold + avgLossGoldAbs;
  return {
    games: vals.length,
    wins: wins.length,
    losses: losses.length,
    draws: draws.length,
    avgWinGold,
    avgLossGoldAbs,
    perWinSwing
  };
}

function analyzeKiboDetail(kiboList) {
  const counts = { full: 0, lean: 0, none: 0, unknown: 0 };
  for (const game of kiboList) {
    const d = String(game?.kibo_detail || "unknown").toLowerCase();
    if (Object.prototype.hasOwnProperty.call(counts, d)) {
      counts[d] += 1;
    } else {
      counts.unknown += 1;
    }
  }
  return {
    total: kiboList.length,
    counts
  };
}

function collectKiboGoStopStats(kiboList, actor) {
  let goEvents = 0;
  let stopEvents = 0;
  let go3PlusEvents = 0;
  let goGames = 0;
  let goLoseGames = 0;
  for (const game of kiboList) {
    let actorGoInGame = false;
    const kibo = Array.isArray(game?.kibo) ? game.kibo : [];
    for (const ev of kibo) {
      if (!ev || ev.playerKey !== actor) continue;
      if (ev.type === "go") {
        goEvents += 1;
        actorGoInGame = true;
        if (Number(ev.goCount || 0) >= 3) go3PlusEvents += 1;
      } else if (ev.type === "stop") {
        stopEvents += 1;
      }
    }
    if (actorGoInGame) {
      goGames += 1;
      if (game?.winner && game.winner !== actor) goLoseGames += 1;
    }
  }
  return {
    goEvents,
    stopEvents,
    go3PlusEvents,
    goGames,
    goLoseGames,
    goLoseRateByGame: goGames > 0 ? goLoseGames / goGames : 0
  };
}

// Decode selected model features used by GO/STOP diagnostics.
function decodeFeatures(f) {
  // Feature index mapping follows model_duel_worker.mjs featureVectorForCandidate().
  const deck = Math.round((Number(f?.[9]) || 0) * 30);
  const goCount = Math.round((Number(f?.[12]) || 0) * 5) + 1;
  const selfPi = Math.round((Number(f?.[28]) || 0) * 20);
  const oppPi = Math.round((Number(f?.[29]) || 0) * 20);
  return {
    deck,
    goCount,
    piDiff: selfPi - oppPi,
    oppCanStop: Number(f?.[39]) > 0.5
  };
}

// Keep only chosen GO/STOP option decisions for one actor/policy.
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
      step: Number(r.step),
      actor: r.actor,
      actor_policy: r.actor_policy,
      action: r.candidate,
      features: r.features
    }));
}

// ---------------------------------------------------------------------------
// Zone diagnostics and impact estimation
// ---------------------------------------------------------------------------
function zoneStats(records, key, label, fn, baseline) {
  const sub = records.filter(fn);
  const s = calcStats(sub);
  const failLift = s.failRate - baseline.failRate;
  const evDrop = baseline.ev - s.ev;
  const riskScore =
    Math.max(0, failLift) * 100 + Math.max(0, evDrop) / 10 + (s.n >= 5 ? Math.min(1.5, s.n / 100) : 0);
  return {
    key,
    label,
    n: s.n,
    failRate: s.failRate,
    ev: s.ev,
    winRate: s.winRate,
    evGap: s.ev - baseline.ev,
    riskScore
  };
}

// Simulate a conservative "block this zone from GO" policy.
function blockDelta(records, fn) {
  const all = calcStats(records);
  const blocked = records.filter(fn);
  const kept = records.filter((r) => !fn(r));
  const b = calcStats(blocked);
  const k = calcStats(kept);
  return {
    blockedN: b.n,
    blockedFailRate: b.failRate,
    blockedEV: b.ev,
    keptN: k.n,
    keptFailRate: k.failRate,
    keptEV: k.ev,
    allFailRate: all.failRate,
    evDeltaIfBlocked: k.ev - all.ev,
    failRateDeltaIfBlocked: k.failRate - all.failRate
  };
}

async function loadCurrentParams(paramsFile) {
  const full = resolve(paramsFile);
  const mod = await import(pathToFileURL(full).href);
  if (!mod || typeof mod.DEFAULT_PARAMS !== "object" || !mod.DEFAULT_PARAMS) {
    throw new Error(`DEFAULT_PARAMS not found in ${paramsFile}`);
  }
  return mod.DEFAULT_PARAMS;
}

function extractReferencedParamKeys(paramsFile) {
  const full = resolve(paramsFile);
  const src = readFileSync(full, "utf8").replace(/^\uFEFF/, "");
  const keys = new Set();
  const re = /\b(?:P|params)\.([A-Za-z_][A-Za-z0-9_]*)\b/g;
  let m;
  while ((m = re.exec(src)) != null) {
    keys.add(String(m[1]));
  }
  return keys;
}

function asNum(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function hasOwnParam(params, key) {
  return params != null && Object.prototype.hasOwnProperty.call(params, key);
}

function isTunableParam(currentParams, referencedParamKeys, key) {
  return hasOwnParam(currentParams, key) && referencedParamKeys.has(key);
}

// Build and rank parameter suggestions.
// Ranking priority:
// 1) netDecisionEvDelta
// 2) failMass
// 3) roi
// 4) score (tie-breaker)
function buildSuggestions(currentParams, referencedParamKeys, goRecs, stopRecs, goStats, stopStats, zones, zoneDefs, gameGoldStats) {
  const MIN_NET_DECISION_EV_DELTA = 1.0;
  const ATTACK_FLIP_RATE = 0.35;
  const ATTACK_MAX_FAIL_PER_ADDED_GO = 0.22;
  const zoneByKey = new Map(zones.map((z) => [z.key, z]));
  const suggestions = [];
  const optionN = Math.max(1, goStats.n + stopStats.n);
  const goRate = goStats.n / optionN;
  const stopRate = stopStats.n / optionN;
  const decisionEvBase = goRate * goStats.ev + stopRate * stopStats.ev;

  const add = (entry) => {
    if (!entry) return;
    if (!Number.isFinite(entry.current) || !Number.isFinite(entry.suggested)) return;
    if (round(entry.current, 6) === round(entry.suggested, 6)) return;
    suggestions.push(entry);
  };

  const addAggressiveExpansionCandidate = (cfg) => {
    if (!isTunableParam(currentParams, referencedParamKeys, cfg.param)) return;
    const zoneFn = zoneDefs[cfg.zoneKey];
    const zGo = zoneByKey.get(cfg.zoneKey);
    if (!zoneFn || !zGo || zGo.n < 8) return;
    const stopZone = stopRecs.filter(zoneFn);
    const zStop = calcStats(stopZone);
    if (zStop.n < 6) return;
    if (zGo.ev <= zStop.ev) return;
    if (zGo.failRate > ATTACK_MAX_FAIL_PER_ADDED_GO) return;

    const goDeltaCount = zStop.n * ATTACK_FLIP_RATE;
    const goRateDelta = optionN > 0 ? goDeltaCount / optionN : 0;
    const failDeltaCount = goDeltaCount * zGo.failRate;
    const predictedGoCount = goStats.n + goDeltaCount;
    const predictedFailCount = goStats.fails + failDeltaCount;
    const predictedFailRate = predictedGoCount > 1e-9 ? predictedFailCount / predictedGoCount : goStats.failRate;
    const failRateDelta = predictedFailRate - goStats.failRate;
    const expectedTotalGoldDelta = goDeltaCount * (zGo.ev - zStop.ev);
    const netDecisionEvDelta = optionN > 0 ? expectedTotalGoldDelta / optionN : 0;
    const score = netDecisionEvDelta * 10 + Math.max(0, goStats.failRate - zGo.failRate) * 30;

    add({
      param: cfg.param,
      mode: "aggressive_expand",
      current: cfg.current,
      suggested: cfg.suggested,
      score,
      affectedN: zStop.n,
      estimatedEvDeltaPerGo: zGo.ev - zStop.ev,
      estimatedFailRateDelta: failRateDelta,
      goRateDeltaDirect: goRateDelta,
      goDeltaCountDirect: goDeltaCount,
      failDeltaCountDirect: failDeltaCount,
      netDecisionEvDeltaDirect: netDecisionEvDelta,
      expectedTotalGoldDeltaDirect: expectedTotalGoldDelta,
      reason: cfg.reason,
      expected_impact: cfg.impact,
      evidence: {
        zone_key: cfg.zoneKey,
        zone_label: zGo.label,
        zone_go_n: zGo.n,
        zone_go_fail_rate: round(zGo.failRate, 4),
        zone_go_ev: round(zGo.ev, 2),
        zone_stop_n: zStop.n,
        zone_stop_ev: round(zStop.ev, 2),
        flip_rate: ATTACK_FLIP_RATE
      }
    });
  };

  const zFirstGoLow = zoneByKey.get("first_go_low_edge");
  if (isTunableParam(currentParams, referencedParamKeys, "goMinPi") && zFirstGoLow && zFirstGoLow.n >= 8) {
    const d = blockDelta(goRecs, zoneDefs.first_go_low_edge);
    const current = asNum(currentParams.goMinPi, 6);
    add({
      param: "goMinPi",
      mode: "conservative_block",
      current,
      suggested: Math.min(10, current + 1),
      score: zFirstGoLow.riskScore + Math.max(0, d.evDeltaIfBlocked) * 2,
      affectedN: d.blockedN,
      estimatedEvDeltaPerGo: d.evDeltaIfBlocked,
      estimatedFailRateDelta: d.failRateDeltaIfBlocked,
      reason: "Failures are concentrated in weak first-GO states (deck<=4 or piDiff<5).",
      expected_impact: `Fewer low-quality first GOs, lower fail rate expected (if this zone is fully blocked: EV/GO +${round(
        d.evDeltaIfBlocked,
        2
      )})`,
      evidence: {
        zone_n: zFirstGoLow.n,
        zone_fail_rate: round(zFirstGoLow.failRate, 4),
        zone_ev: round(zFirstGoLow.ev, 2),
        blocked_n: d.blockedN
      }
    });
  }

  const zFirstPiLow = zoneByKey.get("first_go_pi_low");
  if (isTunableParam(currentParams, referencedParamKeys, "goBaseThreshold") && zFirstPiLow && zFirstPiLow.n >= 6) {
    const d = blockDelta(goRecs, zoneDefs.first_go_pi_low);
    const current = asNum(currentParams.goBaseThreshold, 0.03);
    add({
      param: "goBaseThreshold",
      mode: "conservative_block",
      current,
      suggested: Math.min(0.2, current + 0.01),
      score: zFirstPiLow.riskScore + Math.max(0, d.evDeltaIfBlocked) * 0.5,
      affectedN: d.blockedN,
      estimatedEvDeltaPerGo: d.evDeltaIfBlocked,
      estimatedFailRateDelta: d.failRateDeltaIfBlocked,
      reason: "Low-piDiff first-GO has weak EV, so early GO threshold should be raised.",
      expected_impact: "Reduce low-quality early GOs and improve fail rate.",
      evidence: {
        zone_n: zFirstPiLow.n,
        zone_fail_rate: round(zFirstPiLow.failRate, 4),
        zone_ev: round(zFirstPiLow.ev, 2)
      }
    });
  }

  if (isTunableParam(currentParams, referencedParamKeys, "goScoreDiffBonus") && zFirstPiLow && zFirstPiLow.n >= 6) {
    const d = blockDelta(goRecs, zoneDefs.first_go_pi_low);
    const current = asNum(currentParams.goScoreDiffBonus, 0.05);
    add({
      param: "goScoreDiffBonus",
      mode: "conservative_block",
      current,
      suggested: Math.min(0.2, current + 0.005),
      score: zFirstPiLow.riskScore + Math.max(0, d.evDeltaIfBlocked) * 0.6,
      affectedN: d.blockedN,
      estimatedEvDeltaPerGo: d.evDeltaIfBlocked,
      estimatedFailRateDelta: d.failRateDeltaIfBlocked,
      reason: "Low-piDiff 1GO is weak; increase score-diff weight so GO requires stronger lead.",
      expected_impact: "Reduce low-quality early GO with minimal global aggressiveness loss.",
      evidence: {
        zone_n: zFirstPiLow.n,
        zone_fail_rate: round(zFirstPiLow.failRate, 4),
        zone_ev: round(zFirstPiLow.ev, 2)
      }
    });
  }

  const zLateAll = zoneByKey.get("late_all");
  if (isTunableParam(currentParams, referencedParamKeys, "goDeckLowBonus") && zLateAll && zLateAll.n >= 10) {
    const d = blockDelta(goRecs, zoneDefs.late_all);
    const current = asNum(currentParams.goDeckLowBonus, 0.08);
    add({
      param: "goDeckLowBonus",
      mode: "conservative_block",
      current,
      suggested: Math.max(-0.1, current - 0.01),
      score: zLateAll.riskScore + Math.max(0, d.evDeltaIfBlocked),
      affectedN: d.blockedN,
      estimatedEvDeltaPerGo: d.evDeltaIfBlocked,
      estimatedFailRateDelta: d.failRateDeltaIfBlocked,
      reason: "Late-deck GO zone quality is weak; lower late GO bonus to curb risky late pushes.",
      expected_impact: "Improve late-game GO quality and stabilize fail rate.",
      evidence: {
        zone_n: zLateAll.n,
        zone_fail_rate: round(zLateAll.failRate, 4),
        zone_ev: round(zLateAll.ev, 2)
      }
    });
  }

  if (isTunableParam(currentParams, referencedParamKeys, "goUnseeHighPiPenalty") && zFirstGoLow && zFirstGoLow.n >= 8) {
    const d = blockDelta(goRecs, zoneDefs.first_go_low_edge);
    const current = asNum(currentParams.goUnseeHighPiPenalty, 0.08);
    add({
      param: "goUnseeHighPiPenalty",
      mode: "conservative_block",
      current,
      suggested: Math.min(0.5, current + 0.01),
      score: zFirstGoLow.riskScore + Math.max(0, d.evDeltaIfBlocked) * 0.6,
      affectedN: d.blockedN,
      estimatedEvDeltaPerGo: d.evDeltaIfBlocked,
      estimatedFailRateDelta: d.failRateDeltaIfBlocked,
      reason: "Weak 1GO failures suggest hidden upside/risk is underestimated; raise unseen-high-pi penalty.",
      expected_impact: "Suppress fragile GO entries under latent opponent upside.",
      evidence: {
        zone_n: zFirstGoLow.n,
        zone_fail_rate: round(zFirstGoLow.failRate, 4),
        zone_ev: round(zFirstGoLow.ev, 2)
      }
    });
  }

  addAggressiveExpansionCandidate({
    param: "goBaseThreshold",
    current: asNum(currentParams.goBaseThreshold, 0.03),
    suggested: Math.max(-0.2, asNum(currentParams.goBaseThreshold, 0.03) - 0.01),
    zoneKey: "go2plus",
    reason: "GO2+ zone is high-EV and stable; lower GO threshold slightly to unlock profitable GO opportunities.",
    impact: "Increase GO volume in proven strong contexts while keeping fail risk bounded."
  });
  addAggressiveExpansionCandidate({
    param: "goMinPi",
    current: asNum(currentParams.goMinPi, 6),
    suggested: Math.max(4, asNum(currentParams.goMinPi, 6) - 1),
    zoneKey: "go2plus",
    reason: "Strong multi-GO contexts can tolerate lower pi gate.",
    impact: "Allow additional GO entries with positive expected value."
  });
  addAggressiveExpansionCandidate({
    param: "goUnseeHighPiPenalty",
    current: asNum(currentParams.goUnseeHighPiPenalty, 0.08),
    suggested: Math.max(0, asNum(currentParams.goUnseeHighPiPenalty, 0.08) - 0.01),
    zoneKey: "go2plus",
    reason: "Current hidden-risk penalty may be too strict in already validated high-EV zones.",
    impact: "Recover missed high-quality GO chances."
  });
  addAggressiveExpansionCandidate({
    param: "goDeckLowBonus",
    current: asNum(currentParams.goDeckLowBonus, 0.08),
    suggested: Math.min(0.2, asNum(currentParams.goDeckLowBonus, 0.08) + 0.01),
    zoneKey: "go2plus",
    reason: "Late/high-confidence multi-GO states are profitable in this match.",
    impact: "Increase profitable late GO continuation."
  });
  addAggressiveExpansionCandidate({
    param: "goScoreDiffBonus",
    current: asNum(currentParams.goScoreDiffBonus, 0.05),
    suggested: Math.max(0, asNum(currentParams.goScoreDiffBonus, 0.05) - 0.005),
    zoneKey: "go2plus",
    reason: "Score gap gate can be relaxed slightly in zones with strong empirical EV.",
    impact: "Expand GO entries without broad policy distortion."
  });
  addAggressiveExpansionCandidate({
    param: "goRiskGoCountMul",
    current: asNum(currentParams.goRiskGoCountMul, 0.11),
    suggested: Math.max(0.04, asNum(currentParams.goRiskGoCountMul, 0.11) - 0.02),
    zoneKey: "go3plus_strong",
    reason: "GO3+ high-diff zone has strong EV and manageable fail rate.",
    impact: "Permit more profitable high-order GO continuation."
  });

  const bestByParamMode = new Map();
  for (const s of suggestions) {
    const key = `${s.mode}:${s.param}`;
    const prev = bestByParamMode.get(key);
    if (!prev || s.score > prev.score) bestByParamMode.set(key, s);
  }

  const ranked = [...bestByParamMode.values()].map((s) => {
    const affectedN = Number.isFinite(Number(s.affectedN)) ? Number(s.affectedN) : 0;
    const affectedShareGo = goStats.n > 0 ? affectedN / goStats.n : 0;
    const affectedShareDecision = goRate * affectedShareGo;
    const evDelta = asNum(s.estimatedEvDeltaPerGo, 0);
    const mode = String(s.mode || "conservative_block");

    let netDecisionEvDelta = goRate * evDelta;
    let goRateDelta = 0;
    let predictedGoCount = goStats.n;
    let predictedFailCount = goStats.fails;

    if (mode === "conservative_block") {
      const failRateDeltaBlocked = asNum(s.estimatedFailRateDelta, 0);
      netDecisionEvDelta =
        goRate * ((1 - affectedShareGo) * evDelta + affectedShareGo * (stopStats.ev - goStats.ev));
      goRateDelta = -affectedShareDecision;
      predictedGoCount = Math.max(0, goStats.n + goRateDelta * optionN);
      const predictedFailRate = clamp(goStats.failRate + failRateDeltaBlocked, 0, 1);
      predictedFailCount = predictedGoCount * predictedFailRate;
    } else if (
      Number.isFinite(s.netDecisionEvDeltaDirect) &&
      Number.isFinite(s.goRateDeltaDirect) &&
      Number.isFinite(s.failDeltaCountDirect)
    ) {
      netDecisionEvDelta = Number(s.netDecisionEvDeltaDirect);
      goRateDelta = Number(s.goRateDeltaDirect);
      predictedGoCount = Math.max(0, goStats.n + Number(s.goDeltaCountDirect || 0));
      predictedFailCount = Math.max(0, goStats.fails + Number(s.failDeltaCountDirect));
    }

    const goDeltaCount = predictedGoCount - goStats.n;
    const predictedFailRate = predictedGoCount > 1e-9 ? predictedFailCount / predictedGoCount : goStats.failRate;
    const failDelta = predictedFailRate - goStats.failRate;
    const failDeltaCount = predictedFailCount - goStats.fails;
    const failReducedCount = Math.max(0, -failDeltaCount);
    const goCutPerFailReduced =
      goDeltaCount < 0 && failReducedCount > 1e-9 ? Math.abs(goDeltaCount) / failReducedCount : 0;
    const failMass = Math.max(0, -failDelta) * Math.max(0, goRate + Math.min(0, goRateDelta));
    const roi = Math.abs(goRateDelta) > 1e-9 ? netDecisionEvDelta / Math.abs(goRateDelta) : netDecisionEvDelta;
    const expectedTotalGoldDelta = Number.isFinite(s.expectedTotalGoldDeltaDirect)
      ? Number(s.expectedTotalGoldDeltaDirect)
      : netDecisionEvDelta * optionN;
    const expectedWinDeltaGoldBased =
      asNum(gameGoldStats?.perWinSwing, 0) > 1e-9 ? expectedTotalGoldDelta / asNum(gameGoldStats.perWinSwing, 0) : 0;
    const winCapByEvents = Math.max(0, failReducedCount + Math.max(0, goDeltaCount));
    const expectedWinDelta = Math.min(Math.max(0, expectedWinDeltaGoldBased), winCapByEvents);
    const expectedWinRateDelta = asNum(gameGoldStats?.games, 0) > 0 ? expectedWinDelta / gameGoldStats.games : 0;

    return {
      ...s,
      affectedN,
      affectedShareGo,
      affectedShareDecision,
      evDelta,
      failDelta,
      netDecisionEvDelta,
      goRateDelta,
      roi,
      goDeltaCount,
      failDeltaCount,
      failReducedCount,
      goCutPerFailReduced,
      expectedTotalGoldDelta,
      expectedWinDeltaGoldBased,
      expectedWinDelta,
      expectedWinRateDelta,
      failMass
    };
  });

  const positive = ranked.filter((s) => s.netDecisionEvDelta >= MIN_NET_DECISION_EV_DELTA);
  const defensePool = positive.filter((s) => s.mode === "conservative_block");
  const attackPool = positive.filter((s) => (
    s.mode === "aggressive_expand" &&
    s.goDeltaCount > 0 &&
    s.failDeltaCount <= s.goDeltaCount * ATTACK_MAX_FAIL_PER_ADDED_GO
  ));

  const sortDefense = (a, b) => {
    if (b.netDecisionEvDelta !== a.netDecisionEvDelta) return b.netDecisionEvDelta - a.netDecisionEvDelta;
    if (b.failMass !== a.failMass) return b.failMass - a.failMass;
    if (b.roi !== a.roi) return b.roi - a.roi;
    return b.score - a.score;
  };
  const sortAttack = (a, b) => {
    if (b.netDecisionEvDelta !== a.netDecisionEvDelta) return b.netDecisionEvDelta - a.netDecisionEvDelta;
    if (b.roi !== a.roi) return b.roi - a.roi;
    if (a.failDeltaCount !== b.failDeltaCount) return a.failDeltaCount - b.failDeltaCount;
    return b.score - a.score;
  };

  const topDefense = defensePool.sort(sortDefense);
  const topAttack = attackPool.sort(sortAttack);

  const finalize = (list, channel) => {
    const evMassDen = list.reduce((acc, s) => acc + Math.max(0, s.netDecisionEvDelta), 0);
    return list.map((s, idx) => {
      const weight = evMassDen > 0 ? Math.max(0, s.netDecisionEvDelta) / evMassDen : 1 / Math.max(list.length, 1);
      return {
        rank: idx + 1,
        channel,
        param: s.param,
        current: round(s.current, 6),
        suggested: round(s.suggested, 6),
        priority_percent: round(weight * 100, 1),
        affected_n: s.affectedN,
        affected_share: round(s.affectedShareGo, 6),
        affected_share_decision: round(s.affectedShareDecision, 6),
        estimated_ev_delta_per_go: round(s.evDelta, 3),
        estimated_fail_rate_delta: round(s.failDelta, 6),
        estimated_go_rate_delta: round(s.goRateDelta, 6),
        expected_go_delta_count: round(s.goDeltaCount, 3),
        expected_fail_delta_count: round(s.failDeltaCount, 3),
        expected_fail_reduced_count: round(s.failReducedCount, 3),
        go_cut_per_1_fail_reduced: round(s.goCutPerFailReduced, 3),
        net_decision_ev_delta: round(s.netDecisionEvDelta, 6),
        expected_total_gold_delta: round(s.expectedTotalGoldDelta, 3),
        expected_win_delta_gold_based: round(s.expectedWinDeltaGoldBased, 3),
        expected_win_delta: round(s.expectedWinDelta, 3),
        expected_win_rate_delta: round(s.expectedWinRateDelta, 6),
        roi_per_go_rate: round(s.roi, 6),
        decision_ev_base: round(decisionEvBase, 6),
        fail_mass: round(s.failMass, 6),
        reason: s.reason,
        expected_impact: s.expected_impact,
        evidence: s.evidence
      };
    });
  };

  const defense = finalize(topDefense, "defense");
  const attack = finalize(topAttack, "attack");
  const combined = [...defense, ...attack]
    .sort((a, b) => b.net_decision_ev_delta - a.net_decision_ev_delta);

  return {
    combined,
    defense,
    attack,
    minNetDecisionEvDelta: MIN_NET_DECISION_EV_DELTA
  };
}

// ---------------------------------------------------------------------------
// Report rendering
// ---------------------------------------------------------------------------
function buildSummaryText(payload) {
  const lines = [];
  lines.push("GO/STOP Analysis Summary");
  lines.push(`Generated: ${payload.generated_at}`);
  lines.push(`Actor: ${payload.input.actor} (${payload.input.actor_policy || "any"})`);
  lines.push(`Kibo: ${payload.input.kibo}`);
  lines.push(`Dataset: ${payload.input.dataset}`);
  lines.push(
    `Kibo detail: full=${payload.kibo_detail.counts.full}, lean=${payload.kibo_detail.counts.lean}, none=${payload.kibo_detail.counts.none}, unknown=${payload.kibo_detail.counts.unknown}`
  );
  lines.push(
    `Kibo events(actor): GO=${payload.kibo_events.go_events}, STOP=${payload.kibo_events.stop_events}, GO3+=${payload.kibo_events.go3plus_events}, GO lose(game)=${pct(
      payload.kibo_events.go_lose_rate_by_game
    )}`
  );
  lines.push("");
  lines.push(
    `Option decisions: ${payload.extracted.option_decisions} (GO=${payload.extracted.go_decisions}, STOP=${payload.extracted.stop_decisions}, GO rate=${pct(
      payload.metrics.go_choice_rate
    )})`
  );
  lines.push(
    `GO   : n=${payload.metrics.go.n}, WR=${pct(payload.metrics.go.winRate)}, fail=${pct(
      payload.metrics.go.failRate
    )}, EV=${round(payload.metrics.go.ev, 2)}, breakEven=${pct(payload.metrics.go.breakEvenRate)}`
  );
  lines.push(
    `STOP : n=${payload.metrics.stop.n}, WR=${pct(payload.metrics.stop.winRate)}, fail=${pct(
      payload.metrics.stop.failRate
    )}, EV=${round(payload.metrics.stop.ev, 2)}, breakEven=${pct(payload.metrics.stop.breakEvenRate)}`
  );
  lines.push("");
  lines.push("Risk Zones (GO):");
  const zones = Array.isArray(payload.risk_zones) ? payload.risk_zones : [];
  if (zones.length <= 0) {
    lines.push("- (none)");
  } else {
    for (const z of zones) {
      lines.push(
        `- ${z.label}: n=${z.n}, fail=${pct(z.failRate)}, EV=${round(z.ev, 2)}, EV gap=${round(z.evGap, 2)}`
      );
    }
  }
  lines.push("");
  lines.push("Top1 Quick Forecast:");
  const combined = Array.isArray(payload.recommendations) ? payload.recommendations : [];
  const defense = Array.isArray(payload.recommendations_defense) ? payload.recommendations_defense : [];
  const attack = Array.isArray(payload.recommendations_attack) ? payload.recommendations_attack : [];
  if (combined.length <= 0) {
    lines.push("- (no positive-ROI recommendation in current data)");
  } else {
    const games = Number(payload?.game_gold?.games || 0);
    const agg = combined.reduce(
      (acc, r) => {
        acc.expected_total_gold_delta += Number(r.expected_total_gold_delta || 0);
        acc.expected_go_delta_count += Number(r.expected_go_delta_count || 0);
        acc.expected_fail_delta_count += Number(r.expected_fail_delta_count || 0);
        acc.expected_win_delta += Number(r.expected_win_delta || 0);
        acc.expected_win_delta_gold_based += Number(r.expected_win_delta_gold_based || 0);
        return acc;
      },
      {
        expected_total_gold_delta: 0,
        expected_go_delta_count: 0,
        expected_fail_delta_count: 0,
        expected_win_delta: 0,
        expected_win_delta_gold_based: 0
      }
    );
    const failReduced = Math.max(0, -agg.expected_fail_delta_count);
    const goCutPerFailReduced =
      agg.expected_go_delta_count < 0 && failReduced > 1e-9
        ? Math.abs(agg.expected_go_delta_count) / failReduced
        : 0;
    const expectedWinRateDelta = games > 0 ? agg.expected_win_delta / games : 0;
    const paramList = combined.map((x) => `${x.param}(${x.current}->${x.suggested})`).join(", ");
    lines.push(`- apply all recommendations: ${combined.length} params`);
    lines.push(`- params: ${paramList}`);
    lines.push(
      `- expected gold: total ${round(agg.expected_total_gold_delta, 2)}, per game ${round(
        games > 0 ? agg.expected_total_gold_delta / games : 0,
        3
      )}`
    );
    lines.push(
      `- fail/GO tradeoff: fail change ${round(agg.expected_fail_delta_count, 3)}, GO change ${round(
        agg.expected_go_delta_count,
        3
      )}, GO cut per 1 fail reduced ${round(goCutPerFailReduced, 3)}`
    );
    lines.push(
      `- expected wins: ${round(agg.expected_win_delta, 3)} wins (win rate delta ${pct(
        expectedWinRateDelta
      )}, gold-based raw ${round(agg.expected_win_delta_gold_based, 3)})`
    );
  }
  lines.push("");
  const minNetEv = Number(payload?.recommendation_threshold?.min_net_decision_ev_delta || 1);
  lines.push(`Defense Param Suggestions (netEV>=${round(minNetEv, 3)}):`);
  if (defense.length <= 0) {
    lines.push("- (no positive-ROI recommendation in current data)");
  } else {
    for (const s of defense) {
      lines.push(
        `- #${s.rank} ${s.param}: ${s.current} -> ${s.suggested} | weight=${s.priority_percent}% | scope(go)=${pct(
          s.affected_share
        )} | scope(all)=${pct(s.affected_share_decision)} | EVd(go)=${round(
          s.estimated_ev_delta_per_go,
          2
        )} | netEV=${round(s.net_decision_ev_delta, 3)} | ROI=${round(s.roi_per_go_rate, 3)}`
      );
      lines.push(`  reason: ${s.reason}`);
      lines.push(`  impact: ${s.expected_impact}`);
    }
  }
  lines.push("");
  lines.push(`Attack Param Suggestions (netEV>=${round(minNetEv, 3)}):`);
  if (attack.length <= 0) {
    lines.push("- (no positive-ROI recommendation in current data)");
  } else {
    for (const s of attack) {
      lines.push(
        `- #${s.rank} ${s.param}: ${s.current} -> ${s.suggested} | weight=${s.priority_percent}% | scope(go)=${pct(
          s.affected_share
        )} | scope(all)=${pct(s.affected_share_decision)} | EVd(go)=${round(
          s.estimated_ev_delta_per_go,
          2
        )} | netEV=${round(s.net_decision_ev_delta, 3)} | ROI=${round(s.roi_per_go_rate, 3)}`
      );
      lines.push(`  reason: ${s.reason}`);
      lines.push(`  impact: ${s.expected_impact}`);
    }
  }
  return `${lines.join("\n")}\n`;
}

// ---------------------------------------------------------------------------
// Main pipeline
// ---------------------------------------------------------------------------
const argv = parseArgs(process.argv.slice(2));
if (argv.help) {
  console.log(`Usage:
  node scripts/analyze_go_ev.mjs --kibo <kibo.jsonl> --dataset <dataset.jsonl> --actor-policy <POLICY> [--actor ai]

Outputs (fixed):
  <match_dir>/analyze/go_stop_summary.txt
  <match_dir>/analyze/go_stop_param_suggestions.json
  (or --out-root <dir> override)
`);
  process.exit(0);
}

const cfg = {
  kibo: String(argv.kibo || DEFAULT_INPUT.kibo),
  dataset: String(argv.dataset || DEFAULT_INPUT.dataset),
  actor: String(argv.actor || DEFAULT_INPUT.actor).toLowerCase(),
  actorPolicy:
    argv["actor-policy"] == null
      ? null
      : String(argv["actor-policy"]),
  paramsFile: String(argv["params-file"] || DEFAULT_INPUT.paramsFile),
  outRoot: resolveOutRoot(
    String(argv.kibo || DEFAULT_INPUT.kibo),
    argv["out-root"] ?? DEFAULT_INPUT.outRoot
  )
};

if (cfg.actor !== "ai" && cfg.actor !== "human") {
  throw new Error(`invalid --actor: ${cfg.actor} (allowed: ai, human)`);
}
if (cfg.actorPolicy == null || String(cfg.actorPolicy).trim() === "") {
  throw new Error("missing required --actor-policy <POLICY>");
}
if (!existsSync(cfg.kibo)) throw new Error(`kibo file not found: ${cfg.kibo}`);
if (!existsSync(cfg.dataset)) throw new Error(`dataset file not found: ${cfg.dataset}`);
if (!existsSync(cfg.paramsFile)) throw new Error(`params file not found: ${cfg.paramsFile}`);

const kibo = readJsonl(cfg.kibo);
const dataset = readJsonl(cfg.dataset);
const kiboDetail = analyzeKiboDetail(kibo);
if (kiboDetail.total > 0 && kiboDetail.counts.full !== kiboDetail.total) {
  throw new Error(
    `kibo-detail must be full for this analyzer (full=${kiboDetail.counts.full}, total=${kiboDetail.total})`
  );
}
const kiboEventStats = collectKiboGoStopStats(kibo, cfg.actor);
const payoutByGame = buildPayoutMap(kibo, cfg.actor);
const gameGoldStats = calcGameGoldStats(payoutByGame);

const chosen = collectChosenGoStop(dataset, cfg.actor, cfg.actorPolicy);
const allRecs = chosen.map((r) => ({
  game: r.game,
  step: r.step,
  action: r.action,
  gold: Number(payoutByGame.get(r.game) ?? 0),
  ...decodeFeatures(r.features)
}));

const goRecs = allRecs.filter((r) => r.action === "go");
const stopRecs = allRecs.filter((r) => r.action === "stop");
const goStats = calcStats(goRecs);
const stopStats = calcStats(stopRecs);

const zoneDefs = {
  first_go_low_edge: (r) => r.goCount === 1 && (r.deck <= 4 || r.piDiff < 5),
  first_go_deck_low: (r) => r.goCount === 1 && r.deck <= 4,
  first_go_pi_low: (r) => r.goCount === 1 && r.piDiff < 5,
  opp_can_stop: (r) => r.oppCanStop,
  go2plus: (r) => r.goCount >= 2,
  go3plus_strong: (r) => r.goCount >= 3 && r.piDiff >= 10,
  late_all: (r) => r.deck <= 4
};

const zones = [
  zoneStats(goRecs, "first_go_low_edge", "1GO && (deck<=4 || piDiff<5)", zoneDefs.first_go_low_edge, goStats),
  zoneStats(goRecs, "first_go_deck_low", "1GO && deck<=4", zoneDefs.first_go_deck_low, goStats),
  zoneStats(goRecs, "first_go_pi_low", "1GO && piDiff<5", zoneDefs.first_go_pi_low, goStats),
  zoneStats(goRecs, "opp_can_stop", "oppCanStop", zoneDefs.opp_can_stop, goStats),
  zoneStats(goRecs, "go2plus", "GO2+", zoneDefs.go2plus, goStats),
  zoneStats(goRecs, "go3plus_strong", "GO3+ && piDiff>=10", zoneDefs.go3plus_strong, goStats),
  zoneStats(goRecs, "late_all", "deck<=4", zoneDefs.late_all, goStats)
]
  .filter((z) => z.n >= 5)
  .sort((a, b) => b.riskScore - a.riskScore);

const currentParams = await loadCurrentParams(cfg.paramsFile);
const referencedParamKeys = extractReferencedParamKeys(cfg.paramsFile);
if (referencedParamKeys.size === 0) {
  throw new Error(`no parameter references found in params file: ${cfg.paramsFile}`);
}
const suggestionSet = buildSuggestions(
  currentParams,
  referencedParamKeys,
  goRecs,
  stopRecs,
  goStats,
  stopStats,
  zones,
  zoneDefs,
  gameGoldStats
);

const runDir = cfg.outRoot;
mkdirSync(runDir, { recursive: true });

const payload = {
  generated_at: new Date().toISOString(),
  input: {
    kibo: cfg.kibo,
    dataset: cfg.dataset,
    actor: cfg.actor,
    actor_policy: cfg.actorPolicy,
    params_file: cfg.paramsFile
  },
  param_reference: {
    referenced_key_count: referencedParamKeys.size,
    referenced_keys: [...referencedParamKeys].sort()
  },
  extracted: {
    option_decisions: allRecs.length,
    go_decisions: goRecs.length,
    stop_decisions: stopRecs.length
  },
  kibo_detail: kiboDetail,
  kibo_events: {
    go_events: kiboEventStats.goEvents,
    stop_events: kiboEventStats.stopEvents,
    go3plus_events: kiboEventStats.go3PlusEvents,
    go_games: kiboEventStats.goGames,
    go_lose_games: kiboEventStats.goLoseGames,
    go_lose_rate_by_game: round(kiboEventStats.goLoseRateByGame, 6)
  },
  game_gold: {
    games: gameGoldStats.games,
    wins: gameGoldStats.wins,
    losses: gameGoldStats.losses,
    draws: gameGoldStats.draws,
    avg_win_gold: round(gameGoldStats.avgWinGold, 3),
    avg_loss_gold_abs: round(gameGoldStats.avgLossGoldAbs, 3),
    per_win_swing: round(gameGoldStats.perWinSwing, 3)
  },
  metrics: {
    go_choice_rate: allRecs.length > 0 ? goRecs.length / allRecs.length : 0,
    go: {
      ...goStats,
      winRate: round(goStats.winRate, 6),
      failRate: round(goStats.failRate, 6),
      breakEvenRate: round(goStats.breakEvenRate, 6),
      ev: round(goStats.ev, 3)
    },
    stop: {
      ...stopStats,
      winRate: round(stopStats.winRate, 6),
      failRate: round(stopStats.failRate, 6),
      breakEvenRate: round(stopStats.breakEvenRate, 6),
      ev: round(stopStats.ev, 3)
    }
  },
  risk_zones: zones.map((z) => ({
    key: z.key,
    label: z.label,
    n: z.n,
    failRate: round(z.failRate, 6),
    winRate: round(z.winRate, 6),
    ev: round(z.ev, 3),
    evGap: round(z.evGap, 3),
    riskScore: round(z.riskScore, 3)
  })),
  recommendations: suggestionSet.combined,
  recommendations_defense: suggestionSet.defense,
  recommendations_attack: suggestionSet.attack,
  recommendation_threshold: {
    min_net_decision_ev_delta: suggestionSet.minNetDecisionEvDelta
  }
};

const summaryPath = join(runDir, "go_stop_summary.txt");
const suggestionPath = join(runDir, "go_stop_param_suggestions.json");
writeFileSync(summaryPath, buildSummaryText(payload), "utf8");
writeFileSync(suggestionPath, JSON.stringify(payload, null, 2), "utf8");

console.log("=== GO/STOP Analyzer ===");
console.log(`run_dir: ${runDir}`);
console.log(`summary: ${summaryPath}`);
console.log(`json:    ${suggestionPath}`);
console.log(
  `GO n=${goStats.n}, fail=${pct(goStats.failRate)}, EV=${round(goStats.ev, 2)} | STOP n=${stopStats.n}, EV=${round(
    stopStats.ev,
    2
  )}`
);
