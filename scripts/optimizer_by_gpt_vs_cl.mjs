import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

// Port and rewrite notes (no behavior change intended in this comment-focused refactor):
// 1) This file is a new implementation, not a direct copy of optimizer_by_CL.mjs.
// 2) Expand scoring is STOP-data-free in greedy flow (no subStop gate).
// 3) Zone direction is consistently based on baseline_ev - zone_ev.
// 4) Greedy planning re-discovers zones each step and removes selected scope.
// 5) Expand estimates use a conservative weight to reduce overestimation risk.
// 6) Chain/counterfactual report sections were intentionally omitted for a lean runner.
// 7) Output writing enforces UTF-8 with BOM and fail-fast parsing is preserved.
//
// ---------------------------------------------------------------------------
// Section 1: Global constants and zone catalogs
// ---------------------------------------------------------------------------
const VERSION = "1.0.0";
const Z95 = 1.96;
const FLIP_RATE = 0.35;
const EXPAND_FLIP_WEIGHT = 0.5;
const MIN_ZONE_N = 8;
const ATTACK_MAX_FAIL = 0.22;
const DEFAULT_TOP_ZONES = 12;
const DEFAULT_MIN_EV = 1.5;

const GC_DEFS = [
  { id: "gc1", label: "goCount=1", fn: (r) => r.goCount === 1 },
  { id: "gc2", label: "goCount=2", fn: (r) => r.goCount === 2 },
  { id: "gc3p", label: "goCount>=3", fn: (r) => r.goCount >= 3 }
];

const DK_DEFS = [
  { id: "dk4", label: "deck<=4", fn: (r) => r.deck <= 4 },
  { id: "dk8", label: "deck<=8", fn: (r) => r.deck <= 8 },
  { id: "dk12", label: "deck<=12", fn: (r) => r.deck <= 12 },
  { id: "dkAny", label: "any_deck", fn: () => true }
];

const PI_DEFS = [
  { id: "pi5", label: "piDiff<5", fn: (r) => r.piDiff < 5 },
  { id: "pi10", label: "piDiff<10", fn: (r) => r.piDiff < 10 },
  { id: "pi10p", label: "piDiff>=10", fn: (r) => r.piDiff >= 10 },
  { id: "piAny", label: "any_pi", fn: () => true }
];

const EX_DEFS = [
  { id: "ocs", label: "oppCanStop", fn: (r) => r.oppCanStop },
  { id: "obs", label: "oppBurst>=2", fn: (r) => r.oppBurst >= 2 },
  { id: "exAny", label: "any_threat", fn: () => true }
];

// ---------------------------------------------------------------------------
// Section 2: CLI and file I/O helpers
// ---------------------------------------------------------------------------
function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i += 1) {
    const a = String(argv[i] || "");
    if (!a.startsWith("--")) continue;
    const key = a.slice(2);
    const next = argv[i + 1];
    if (next != null && !String(next).startsWith("--")) {
      out[key] = String(next);
      i += 1;
    } else {
      out[key] = true;
    }
  }
  return out;
}

// Read JSONL with per-line fail-fast validation.
function readJsonl(path) {
  const raw = readFileSync(path, "utf8").replace(/^\uFEFF/, "");
  if (!raw.trim()) return [];
  return raw
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0)
    .map((line, idx) => {
      try {
        return JSON.parse(line);
      } catch (err) {
        throw new Error(`invalid JSON at ${path}:${idx + 1} (${err.message})`);
      }
    });
}

// Always write UTF-8 BOM to satisfy repository encoding rules.
function writeUtf8Bom(path, text) {
  const body = String(text).replace(/^\uFEFF/, "");
  writeFileSync(path, `\uFEFF${body}`, "utf8");
}

// ---------------------------------------------------------------------------
// Section 3: Numeric/statistical utilities
// ---------------------------------------------------------------------------
function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function round(v, d = 3) {
  const n = Number(v);
  if (!Number.isFinite(n)) return 0;
  const s = 10 ** d;
  return Math.round(n * s) / s;
}

function pct(v) {
  return `${(Number(v) * 100).toFixed(1)}%`;
}

function fmtSigned(v, d = 0) {
  const n = Number(v) || 0;
  return `${n >= 0 ? "+" : ""}${n.toFixed(d)}`;
}

function mean(arr) {
  if (!Array.isArray(arr) || arr.length <= 0) return 0;
  return arr.reduce((acc, x) => acc + Number(x || 0), 0) / arr.length;
}

function atanhSafe(x) {
  const c = clamp(Number(x) || 0, -0.999999, 0.999999);
  return 0.5 * Math.log((1 + c) / (1 - c));
}

function wilsonCI(k, n) {
  const nn = Math.max(0, Number(n) || 0);
  if (nn <= 0) return { lo: 0, hi: 1 };
  const kk = clamp(Number(k) || 0, 0, nn);
  const p = kk / nn;
  const z2 = Z95 * Z95;
  const den = 1 + z2 / nn;
  const mid = (p + z2 / (2 * nn)) / den;
  const spread = (Z95 / den) * Math.sqrt((p * (1 - p)) / nn + z2 / (4 * nn * nn));
  return { lo: clamp(mid - spread, 0, 1), hi: clamp(mid + spread, 0, 1) };
}

function stats(records) {
  const n = records.length;
  if (n <= 0) {
    return {
      n: 0,
      wins: 0,
      fails: 0,
      winRate: 0,
      failRate: 0,
      failCiLo: 0,
      failCiHi: 1,
      ev: 0,
      avgWin: 0,
      avgLoss: 0,
      avgLossAbs: 0,
      beRate: 0
    };
  }
  const wins = records.filter((r) => Number(r.gold) > 0);
  const fails = records.filter((r) => Number(r.gold) <= 0);
  const total = records.reduce((acc, r) => acc + Number(r.gold || 0), 0);
  const avgWin = wins.length > 0 ? mean(wins.map((r) => Number(r.gold || 0))) : 0;
  const avgLoss = fails.length > 0 ? mean(fails.map((r) => Number(r.gold || 0))) : 0;
  const ci = wilsonCI(fails.length, n);
  const avgLossAbs = Math.abs(avgLoss);
  const beRate = avgWin + avgLossAbs > 1e-9 ? avgLossAbs / (avgWin + avgLossAbs) : 0;
  return {
    n,
    wins: wins.length,
    fails: fails.length,
    winRate: wins.length / n,
    failRate: fails.length / n,
    failCiLo: ci.lo,
    failCiHi: ci.hi,
    ev: total / n,
    avgWin,
    avgLoss,
    avgLossAbs,
    beRate
  };
}

// ---------------------------------------------------------------------------
// Section 4: Feature decoding and record construction
// ---------------------------------------------------------------------------
function decode(features) {
  const f = Array.isArray(features) ? features : [];
  const deck = Math.round((Number(f[9]) || 0) * 30);
  const goCount = Math.round((Number(f[12]) || 0) * 5) + 1;
  const selfPi = Math.round((Number(f[28]) || 0) * 20);
  const oppPi = Math.round((Number(f[29]) || 0) * 20);
  const scoreDiff = atanhSafe(Number(f[14]) || 0) * 10;
  const selfScore = Math.max(0, atanhSafe(Number(f[15]) || 0) * 10);
  const oppScore = Math.max(0, selfScore - scoreDiff);
  const oppGwang = Math.round((Number(f[27]) || 0) * 5);
  const oppGodori = Math.round((Number(f[31]) || 0) * 3);
  const oppCheong = Math.round((Number(f[33]) || 0) * 3);
  const oppHong = Math.round((Number(f[35]) || 0) * 3);
  const oppCho = Math.round((Number(f[37]) || 0) * 3);
  const oppCombo = [oppGodori, oppCheong, oppHong, oppCho].filter((v) => v >= 2).length;
  const oppBurst = (oppGwang >= 2 ? 1 : 0) + (oppCombo > 0 ? 1 : 0) + (oppScore >= 6 ? 1 : 0);
  return {
    deck,
    goCount,
    selfPi,
    oppPi,
    piDiff: selfPi - oppPi,
    selfScore,
    oppScore,
    scoreDiff,
    oppCombo,
    oppBurst,
    selfCanStop: Number(f[38]) > 0.5,
    oppCanStop: Number(f[39]) > 0.5
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
      ? [...game.kibo].reverse().find((ev) => ev?.type === "round_end")
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

function buildDecisionRecords(dataset, actor, actorPolicy, payoutMap) {
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
      gold: Number(payoutMap.get(Number(r.game_index)) ?? 0),
      ...decode(r.features)
    }));
}

// ---------------------------------------------------------------------------
// Section 5: Zone scoring and greedy planning
// ---------------------------------------------------------------------------
function buildZone(goRecs, fn, tags, baseStats, optionN) {
  const sub = goRecs.filter(fn);
  if (sub.length < MIN_ZONE_N) return null;
  const s = stats(sub);
  const evGap = baseStats.ev - s.ev;
  if (Math.abs(evGap) < 1e-9) return null;
  const direction = evGap > 0 ? "block" : "expand";
  if (direction === "expand" && s.failRate > ATTACK_MAX_FAIL) return null;

  const flipWeight = direction === "expand" ? EXPAND_FLIP_WEIGHT : 1.0;
  const flipCount = FLIP_RATE * sub.length * flipWeight;
  const edge = Math.abs(evGap);
  const goldDelta = flipCount * edge;
  const goDelta = direction === "block" ? -flipCount : flipCount;
  const failDelta = direction === "block"
    ? -flipCount * Math.max(0, s.failRate - baseStats.failRate)
    : flipCount * Math.max(0, s.failRate - baseStats.failRate);
  const netEvPerDecision = goldDelta / Math.max(1, optionN);
  const reliable = s.n >= 20 || (direction === "block" ? s.failCiLo > baseStats.failCiHi : s.failCiHi < Math.max(baseStats.failRate, 0.12));

  return {
    key: `${tags.gcId}|${tags.dkId}|${tags.piId}|${tags.exId}`,
    label: `${tags.gc} + ${tags.dk} + ${tags.pi} + ${tags.ex}`,
    gc: tags.gc,
    dk: tags.dk,
    pi: tags.pi,
    ex: tags.ex,
    direction,
    n: s.n,
    ev: s.ev,
    failRate: s.failRate,
    failCiLo: s.failCiLo,
    failCiHi: s.failCiHi,
    evGap,
    goldDelta,
    goDelta,
    failDelta,
    netEvPerDecision,
    reliable,
    fn
  };
}

function enumerateZonesFromBase(goRecs, baseStats, optionN) {
  const zones = [];
  for (const gc of GC_DEFS) {
    for (const dk of DK_DEFS) {
      for (const pi of PI_DEFS) {
        for (const ex of EX_DEFS) {
          const allAny = dk.id === "dkAny" && pi.id === "piAny" && ex.id === "exAny";
          if (allAny) continue;
          const fn = (r) => gc.fn(r) && dk.fn(r) && pi.fn(r) && ex.fn(r);
          const zone = buildZone(
            goRecs,
            fn,
            {
              gcId: gc.id,
              dkId: dk.id,
              piId: pi.id,
              exId: ex.id,
              gc: gc.label,
              dk: dk.label,
              pi: pi.label,
              ex: ex.label
            },
            baseStats,
            optionN
          );
          if (zone) zones.push(zone);
        }
      }
    }
  }
  zones.sort((a, b) => {
    if (b.netEvPerDecision !== a.netEvPerDecision) return b.netEvPerDecision - a.netEvPerDecision;
    if (b.goldDelta !== a.goldDelta) return b.goldDelta - a.goldDelta;
    return b.n - a.n;
  });
  return zones;
}

function buildGreedyPlan(goRecs, optionN, topZones, minEv) {
  const plan = [];
  let pool = [...goRecs];
  let cumGold = 0;
  let cumGo = 0;
  let cumFail = 0;

  for (let step = 1; step <= topZones; step += 1) {
    if (pool.length < MIN_ZONE_N) break;
    const baseStats = stats(pool);
    const zones = enumerateZonesFromBase(pool, baseStats, optionN).filter((z) => z.netEvPerDecision >= minEv);
    if (zones.length <= 0) break;
    const best = zones[0];

    cumGold += best.goldDelta;
    cumGo += best.goDelta;
    cumFail += best.failDelta;

    plan.push({
      step,
      direction: best.direction,
      zoneKey: best.key,
      zoneLabel: best.label,
      gc: best.gc,
      dk: best.dk,
      pi: best.pi,
      ex: best.ex,
      zoneN: best.n,
      zoneEv: round(best.ev, 3),
      zoneFailRate: round(best.failRate, 6),
      evGap: round(best.evGap, 3),
      goldDelta: round(best.goldDelta, 3),
      goDelta: round(best.goDelta, 3),
      failDelta: round(best.failDelta, 3),
      netEvPerDecision: round(best.netEvPerDecision, 6),
      cumGold: round(cumGold, 3),
      cumGoDelta: round(cumGo, 3),
      cumFailDelta: round(cumFail, 3),
      reliable: best.reliable
    });

    pool = pool.filter((r) => !best.fn(r));
  }

  return plan;
}

// ---------------------------------------------------------------------------
// Section 6: Parameter mapping and heuristic file updates
// ---------------------------------------------------------------------------
const PARAM_RULES = [
  { match: (z) => z.direction === "block" && z.gc === "goCount=1" && z.dk === "deck<=4", param: "goBaseThreshold", delta: +0.01, min: -0.2, max: 0.2, reason: "Raise threshold in weak first-GO late-deck states." },
  { match: (z) => z.direction === "block" && z.gc === "goCount=1" && z.pi === "piDiff<5", param: "goMinPi", delta: +1, min: 3, max: 12, reason: "Require stronger pi edge for first GO in low-pi zones." },
  { match: (z) => z.direction === "block" && z.gc === "goCount=1" && z.ex === "oppCanStop", param: "goLiteOppCanStopPenalty", delta: +0.02, min: 0, max: 1.0, reason: "Increase GO penalty when opponent can stop." },
  { match: (z) => z.direction === "block" && z.dk === "deck<=4", param: "goLiteLatePenalty", delta: +0.01, min: 0, max: 1.0, reason: "Suppress fragile late-deck GO entries." },
  { match: (z) => z.direction === "block" && z.ex === "oppBurst>=2", param: "goHardLateOneAwayCut", delta: -5, min: 30, max: 95, reason: "Cut GO earlier under opponent burst risk." },
  { match: (z) => z.direction === "block" && z.gc === "goCount=1" && z.dk === "deck<=8", param: "goUnseeHighPiPenalty", delta: +0.01, min: 0, max: 0.5, reason: "Raise unseen-pi penalty in weak first-GO areas." },
  { match: (z) => z.direction === "block" && z.gc === "goCount=1" && z.pi === "piDiff<10", param: "goScoreDiffBonus", delta: +0.005, min: 0, max: 1.0, reason: "Require stronger score edge before first GO." },
  { match: (z) => z.direction === "block" && z.gc === "goCount=2", param: "goBaseThreshold", delta: +0.005, min: -0.2, max: 0.2, reason: "Slightly raise threshold for weak second-GO zones." },
  { match: (z) => z.direction === "expand" && z.gc === "goCount>=3", param: "goRiskGoCountMul", delta: -0.02, min: 0.02, max: 0.5, reason: "Reduce GO-count risk multiplier in profitable GO3+ zones." },
  { match: (z) => z.direction === "expand" && z.gc === "goCount=2" && z.pi === "piDiff>=10", param: "goBaseThreshold", delta: -0.01, min: -0.2, max: 0.2, reason: "Lower threshold in strong GO2 high-pi zones." },
  { match: (z) => z.direction === "expand" && z.gc === "goCount>=3" && z.pi === "piDiff>=10", param: "goDeckLowBonus", delta: +0.01, min: -1.0, max: 1.0, reason: "Allow more continuation in high-pi GO3+ zones." },
  { match: (z) => z.direction === "expand" && z.gc === "goCount=2", param: "goScoreDiffBonus", delta: -0.005, min: 0, max: 1.0, reason: "Relax score-gap gate in validated GO2 zones." },
  { match: (z) => z.direction === "expand", param: "goUnseeHighPiPenalty", delta: -0.01, min: 0, max: 0.5, reason: "Reduce over-penalization in high-EV expand zones." }
];

function mapZoneToParam(zone, currentParams, referencedKeys) {
  for (const rule of PARAM_RULES) {
    if (!rule.match(zone)) continue;
    if (!(rule.param in currentParams)) continue;
    if (referencedKeys && !referencedKeys.has(rule.param)) continue;
    const cur = Number(currentParams[rule.param]);
    if (!Number.isFinite(cur)) continue;
    const suggested = clamp(cur + rule.delta, rule.min, rule.max);
    if (Math.abs(suggested - cur) < 1e-9) continue;
    return {
      param: rule.param,
      current: cur,
      suggested,
      reason: rule.reason
    };
  }
  return null;
}

function extractReferencedParamKeys(paramsFile) {
  const src = readFileSync(resolve(paramsFile), "utf8").replace(/^\uFEFF/, "");
  const out = new Set();
  const re = /\b(?:P|params)\.([A-Za-z_][A-Za-z0-9_]*)\b/g;
  let m;
  while ((m = re.exec(src)) != null) out.add(String(m[1]));
  return out;
}

async function loadParams(paramsFile) {
  const mod = await import(pathToFileURL(resolve(paramsFile)).href);
  if (!mod?.DEFAULT_PARAMS || typeof mod.DEFAULT_PARAMS !== "object") {
    throw new Error(`DEFAULT_PARAMS not found: ${paramsFile}`);
  }
  return mod.DEFAULT_PARAMS;
}

function applyToFile(paramsFile, suggestions) {
  const srcBuf = readFileSync(resolve(paramsFile));
  const hasBom = srcBuf.length >= 3 && srcBuf[0] === 0xef && srcBuf[1] === 0xbb && srcBuf[2] === 0xbf;
  let src = srcBuf.toString("utf8").replace(/^\uFEFF/, "");
  let replaced = 0;
  for (const s of suggestions) {
    const re = new RegExp(`(\\b${s.param}\\s*:\\s*)([0-9.eE+\\-]+)`, "g");
    const prev = src;
    src = src.replace(re, `$1${s.suggested}`);
    if (src !== prev) replaced += 1;
  }
  const out = hasBom ? `\uFEFF${src}` : src;
  writeFileSync(resolve(paramsFile), out, "utf8");
  return replaced;
}

// ---------------------------------------------------------------------------
// Section 7: Reporting
// ---------------------------------------------------------------------------
function buildSummary(payload) {
  const lines = [];
  lines.push(`Optimizer by GPT (vs Claude) v${payload.version}`);
  lines.push(`Generated: ${payload.generated_at}`);
  lines.push(`Actor: ${payload.input.actor} (${payload.input.actor_policy})`);
  lines.push(`Kibo: ${payload.input.kibo}`);
  lines.push(`Dataset: ${payload.input.dataset}`);
  lines.push(`Params: ${payload.input.params_file}`);
  lines.push("");
  lines.push("Claude Claim Review:");
  lines.push("- Claim 1 accepted: zone direction is based on baseline_ev - zone_ev.");
  lines.push("- Claim 2 fixed in this file: greedy expand does not require STOP records.");
  lines.push("- Claim 3 mitigated: greedy re-discovers zones each step and removes selected scope.");
  lines.push("");
  lines.push("Baseline:");
  lines.push(`- games=${payload.baseline.games}, WR=${pct(payload.baseline.win_rate)}, meanGoldDelta=${round(payload.baseline.mean_gold_delta, 3)}`);
  lines.push(`- GO n=${payload.baseline.go.n}, fail=${pct(payload.baseline.go.fail_rate)}, EV=${round(payload.baseline.go.ev, 3)}`);
  lines.push(`- STOP n=${payload.baseline.stop.n}, fail=${pct(payload.baseline.stop.fail_rate)}, EV=${round(payload.baseline.stop.ev, 3)}`);
  lines.push(`- GO choice rate=${pct(payload.baseline.go_choice_rate)}`);
  lines.push("");
  lines.push(`Settings: min_ev=${payload.settings.min_ev}, top_zones=${payload.settings.top_zones}, flip_rate=${payload.settings.flip_rate}, expand_flip_weight=${payload.settings.expand_flip_weight}`);
  lines.push("");
  lines.push(`Top Zones (first ${Math.min(12, payload.top_zones.length)}):`);
  for (const z of payload.top_zones.slice(0, 12)) {
    lines.push(
      `- ${z.direction.toUpperCase()} n=${z.n}, EV=${z.ev}, fail=${pct(z.failRate)}, edge=${round(Math.abs(z.evGap), 3)}, netEV=${z.netEvPerDecision} | ${z.label}`
    );
  }
  lines.push("");
  lines.push("Greedy Plan:");
  if (!payload.greedy_plan.length) {
    lines.push("- (no zone met threshold)");
  } else {
    for (const step of payload.greedy_plan) {
      lines.push(
        `- #${step.step} ${step.direction.toUpperCase()} gold=${fmtSigned(step.goldDelta, 1)} go=${fmtSigned(step.goDelta, 1)} fail=${fmtSigned(step.failDelta, 1)} cumGold=${fmtSigned(step.cumGold, 1)} | ${step.zoneLabel}`
      );
    }
  }
  lines.push("");
  lines.push("Param Plan:");
  if (!payload.param_plan.length) {
    lines.push("- (no param mapped)");
  } else {
    for (const p of payload.param_plan) {
      lines.push(`- [${p.direction}] ${p.param}: ${round(p.current, 6)} -> ${round(p.suggested, 6)} | gold=${fmtSigned(p.goldDelta, 1)}`);
      lines.push(`  reason: ${p.reason}`);
    }
  }
  lines.push("");
  lines.push("Note: this is directional analysis only. Confirm with a 1000-game run.");
  return `${lines.join("\n")}\n`;
}

// ---------------------------------------------------------------------------
// Section 8: Main execution flow
// ---------------------------------------------------------------------------
const argv = parseArgs(process.argv.slice(2));
if (argv.help) {
  console.log(`Usage:
node scripts/optimizer_by_gpt_vs_cl.mjs --kibo <kibo.jsonl> --dataset <dataset.jsonl> --actor-policy <POLICY> --params-file <heuristic.js> [--actor ai|human] [--out-root <dir>] [--top-zones 12] [--min-ev 1.5] [--apply]
`);
  process.exit(0);
}

const cfg = {
  kibo: String(argv.kibo || ""),
  dataset: String(argv.dataset || ""),
  actor: String(argv.actor || "ai").toLowerCase(),
  actorPolicy: argv["actor-policy"] != null ? String(argv["actor-policy"]) : null,
  paramsFile: String(argv["params-file"] || "src/heuristics/heuristicV5Plus.js"),
  outRoot: argv["out-root"]
    ? String(argv["out-root"])
    : join(resolve(dirname(String(argv.kibo || "."))), "optimize_gpt_vs_cl"),
  topZones: Number(argv["top-zones"] || DEFAULT_TOP_ZONES),
  minEv: Number(argv["min-ev"] || DEFAULT_MIN_EV),
  apply: Boolean(argv.apply)
};

if (!cfg.kibo) throw new Error("missing --kibo");
if (!cfg.dataset) throw new Error("missing --dataset");
if (cfg.actor !== "ai" && cfg.actor !== "human") throw new Error(`invalid --actor: ${cfg.actor}`);
if (!cfg.actorPolicy) throw new Error("missing --actor-policy");
if (!existsSync(cfg.kibo)) throw new Error(`kibo not found: ${cfg.kibo}`);
if (!existsSync(cfg.dataset)) throw new Error(`dataset not found: ${cfg.dataset}`);
if (!existsSync(cfg.paramsFile)) throw new Error(`params-file not found: ${cfg.paramsFile}`);
if (!Number.isFinite(cfg.topZones) || cfg.topZones < 1) throw new Error(`invalid --top-zones: ${cfg.topZones}`);
if (!Number.isFinite(cfg.minEv) || cfg.minEv < 0) throw new Error(`invalid --min-ev: ${cfg.minEv}`);

const kibo = readJsonl(cfg.kibo);
const dataset = readJsonl(cfg.dataset);
const payoutMap = buildPayoutMap(kibo, cfg.actor);
const records = buildDecisionRecords(dataset, cfg.actor, cfg.actorPolicy, payoutMap);
if (!records.length) throw new Error("no matching option decision records");

const goRecs = records.filter((r) => r.action === "go");
const stopRecs = records.filter((r) => r.action === "stop");
if (!goRecs.length) throw new Error("no GO records found for selected actor/policy");

const optionN = Math.max(1, records.length);
const bsGo = stats(goRecs);
const bsStop = stats(stopRecs);
const gameGold = [...payoutMap.values()].map((x) => Number(x) || 0);
const gameWins = gameGold.filter((v) => v > 0).length;

const baseZones = enumerateZonesFromBase(goRecs, bsGo, optionN);
const topZones = baseZones.slice(0, cfg.topZones * 2);
const greedyPlan = buildGreedyPlan(goRecs, optionN, cfg.topZones, cfg.minEv);

const currentParams = await loadParams(cfg.paramsFile);
const referencedKeys = extractReferencedParamKeys(cfg.paramsFile);
const paramPlan = [];
const usedParamDir = new Set();
for (const step of greedyPlan) {
  const mapped = mapZoneToParam(step, currentParams, referencedKeys);
  if (!mapped) continue;
  const key = `${step.direction}:${mapped.param}`;
  if (usedParamDir.has(key)) continue;
  usedParamDir.add(key);
  paramPlan.push({
    rank: paramPlan.length + 1,
    direction: step.direction,
    zoneKey: step.zoneKey,
    zoneLabel: step.zoneLabel,
    ...mapped,
    goldDelta: step.goldDelta,
    goDelta: step.goDelta,
    failDelta: step.failDelta
  });
}

const last = greedyPlan.length ? greedyPlan[greedyPlan.length - 1] : null;
const payload = {
  generated_at: new Date().toISOString(),
  version: VERSION,
  input: {
    kibo: cfg.kibo,
    dataset: cfg.dataset,
    actor: cfg.actor,
    actor_policy: cfg.actorPolicy,
    params_file: cfg.paramsFile
  },
  baseline: {
    games: gameGold.length,
    wins: gameWins,
    win_rate: gameGold.length > 0 ? gameWins / gameGold.length : 0,
    mean_gold_delta: gameGold.length > 0 ? mean(gameGold) : 0,
    go_choice_rate: goRecs.length / optionN,
    go: {
      n: bsGo.n,
      fail_rate: bsGo.failRate,
      fail_ci_low: bsGo.failCiLo,
      fail_ci_high: bsGo.failCiHi,
      ev: bsGo.ev
    },
    stop: {
      n: bsStop.n,
      fail_rate: bsStop.failRate,
      fail_ci_low: bsStop.failCiLo,
      fail_ci_high: bsStop.failCiHi,
      ev: bsStop.ev
    }
  },
  top_zones: topZones.map((z) => ({
    key: z.key,
    label: z.label,
    direction: z.direction,
    n: z.n,
    ev: round(z.ev, 3),
    failRate: round(z.failRate, 6),
    evGap: round(z.evGap, 3),
    goldDelta: round(z.goldDelta, 3),
    goDelta: round(z.goDelta, 3),
    failDelta: round(z.failDelta, 3),
    netEvPerDecision: round(z.netEvPerDecision, 6),
    reliable: z.reliable
  })),
  greedy_plan: greedyPlan,
  param_plan: paramPlan,
  forecast: last
    ? {
      total_gold_delta: last.cumGold,
      total_go_delta: last.cumGoDelta,
      total_fail_delta: last.cumFailDelta
    }
    : {
      total_gold_delta: 0,
      total_go_delta: 0,
      total_fail_delta: 0
    },
  settings: {
    min_ev: cfg.minEv,
    top_zones: cfg.topZones,
    min_zone_n: MIN_ZONE_N,
    flip_rate: FLIP_RATE,
    expand_flip_weight: EXPAND_FLIP_WEIGHT,
    attack_max_fail: ATTACK_MAX_FAIL
  }
};

mkdirSync(cfg.outRoot, { recursive: true });
const summaryPath = join(cfg.outRoot, "optimizer_gpt_vs_cl_summary.txt");
const jsonPath = join(cfg.outRoot, "optimizer_gpt_vs_cl_plan.json");
writeUtf8Bom(summaryPath, buildSummary(payload));
writeUtf8Bom(jsonPath, `${JSON.stringify(payload, null, 2)}\n`);

if (cfg.apply && paramPlan.length > 0) {
  const changed = applyToFile(cfg.paramsFile, paramPlan);
  console.log(`apply: wrote ${changed} param(s) -> ${cfg.paramsFile}`);
}

console.log("=== Optimizer by GPT (vs Claude) ===");
console.log(`run_dir: ${cfg.outRoot}`);
console.log(`summary: ${summaryPath}`);
console.log(`json:    ${jsonPath}`);
console.log(`GO n=${bsGo.n}, STOP n=${bsStop.n}, zones=${baseZones.length}, greedy_steps=${greedyPlan.length}, param_changes=${paramPlan.length}`);
if (last) {
  console.log(`forecast: gold=${fmtSigned(last.cumGold, 1)} go=${fmtSigned(last.cumGoDelta, 1)} fail=${fmtSigned(last.cumFailDelta, 1)}`);
}
