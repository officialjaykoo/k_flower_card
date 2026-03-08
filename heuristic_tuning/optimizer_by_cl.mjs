// optimizer_by_cl.mjs  — Claude's GO/STOP Gold Optimizer
// ═══════════════════════════════════════════════════════════════════════════
//
// Core design differences vs analyze_go_ev / optimizer_by_gpt:
//
//   1. STOP-DATA-FREE DESIGN
//      Both other tools compare GO EV vs STOP EV.
//      But in typical match logs, STOP records are scarce (AI almost always GOs).
//      This tool works entirely from GO records:
//        block signal  = baseline_ev - zone_ev  (zone pulls DOWN avg → block it)
//        expand signal = zone_ev - baseline_ev  (zone pulls UP avg → push more GOs here)
//
//   2. GO CHAIN ANALYSIS  (from kibo)
//      Extracts per-game GO chains. Finds win-rate and EV by chain length.
//      Identifies the "tipping point" GO that caused losing chains.
//
//   3. COUNTERFACTUAL GOLD RECOVERY  (from kibo)
//      For every GO in a losing chain: how much gold would have been saved
//      by stopping at that position? Gives real dollar values, not estimates.
//
//   4. GREEDY SEQUENTIAL PLANNER
//      Re-discovers zones each step → removes selected zone records from pool.
//      Prevents double-counting overlapping zones.
//      expand uses conservative EXPAND_FLIP_WEIGHT to avoid overestimation.
//
//   5. LEAN KIBO COMPATIBLE
//      Dataset-only analysis is possible even without full kibo.
//
// Usage:
//   node heuristic_tuning/optimizer_by_cl.mjs \
//     --kibo        <kibo.jsonl> \
//     --dataset     <dataset.jsonl> \
//     --actor       ai \
//     --actor-policy H-NEXg \
//     --params-file src/heuristics/heuristicNEXg.js \
//     [--out-root   <dir>]       default: <kibo_dir>/optimize_cl/
//     [--top-zones  12]          zones in greedy plan
//     [--min-ev     1.5]         min netEV/decision to include
//     [--apply]                  write param changes to file
//
// Outputs:
//   <out-root>/optimizer_cl_summary.txt
//   <out-root>/optimizer_cl_plan.json
//
// ─── Validation rule ────────────────────────────────────────────────────────
//   All numbers are directional estimates.
//   Final acceptance requires real 1000-game reruns.
// ═══════════════════════════════════════════════════════════════════════════

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

// ---------------------------------------------------------------------------
// Section 1: Global constants and zone catalogs
// ---------------------------------------------------------------------------
const VERSION          = "1.1.0";
const Z95              = 1.96;
const FLIP_RATE        = 0.35;   // fraction of zone records expected to change decision
const EXPAND_FLIP_WEIGHT = 0.5;  // conservative weight for expand estimates (no STOP data)
const MIN_ZONE_N       = 8;      // minimum records for a zone to be considered
const ATTACK_MAX_FAIL  = 0.22;   // max acceptable failRate for expand zones
const DEFAULT_TOP_ZONES = 12;
const DEFAULT_MIN_EV   = 1.5;

const GC_DEFS = [
  { id: "gc1",  label: "goCount=1",  fn: (r) => r.goCount === 1 },
  { id: "gc2",  label: "goCount=2",  fn: (r) => r.goCount === 2 },
  { id: "gc3p", label: "goCount>=3", fn: (r) => r.goCount >= 3  },
];

const DK_DEFS = [
  { id: "dk4",   label: "deck<=4",  fn: (r) => r.deck <= 4  },
  { id: "dk8",   label: "deck<=8",  fn: (r) => r.deck <= 8  },
  { id: "dk12",  label: "deck<=12", fn: (r) => r.deck <= 12 },
  { id: "dkAny", label: "any_deck", fn: ()  => true         },
];

const PI_DEFS = [
  { id: "pi5",   label: "piDiff<5",   fn: (r) => r.piDiff < 5   },
  { id: "pi10",  label: "piDiff<10",  fn: (r) => r.piDiff < 10  },
  { id: "pi10p", label: "piDiff>=10", fn: (r) => r.piDiff >= 10 },
  { id: "piAny", label: "any_pi",     fn: ()  => true            },
];

const EX_DEFS = [
  { id: "ocs",   label: "oppCanStop",  fn: (r) => r.oppCanStop   },
  { id: "obs",   label: "oppBurst>=2", fn: (r) => r.oppBurst >= 2 },
  { id: "exAny", label: "any_threat",  fn: ()  => true            },
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
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const round = (v, d = 3)  => { const n = Number(v); if (!Number.isFinite(n)) return 0; const s = 10 ** d; return Math.round(n * s) / s; };
const pct   = (v) => `${(Number(v) * 100).toFixed(1)}%`;
const fmt   = (v, d = 0)  => { const n = Number(v) || 0; return `${n >= 0 ? "+" : ""}${n.toFixed(d)}`; };
const mean  = (arr) => { if (!Array.isArray(arr) || arr.length <= 0) return 0; return arr.reduce((a, x) => a + Number(x || 0), 0) / arr.length; };
const atanh = (x) => { const c = clamp(Number(x) || 0, -0.999999, 0.999999); return 0.5 * Math.log((1 + c) / (1 - c)); };

function wilsonCI(k, n) {
  const nn = Math.max(0, Number(n) || 0);
  if (nn <= 0) return { lo: 0, hi: 1 };
  const kk = clamp(Number(k) || 0, 0, nn);
  const p  = kk / nn;
  const z2 = Z95 * Z95;
  const den = 1 + z2 / nn;
  const mid = (p + z2 / (2 * nn)) / den;
  const spread = (Z95 / den) * Math.sqrt((p * (1 - p)) / nn + z2 / (4 * nn * nn));
  return { lo: clamp(mid - spread, 0, 1), hi: clamp(mid + spread, 0, 1) };
}

function stats(records) {
  const n = records.length;
  if (n <= 0) {
    return { n: 0, wins: 0, fails: 0, winRate: 0, failRate: 0,
             failCiLo: 0, failCiHi: 1, ev: 0, avgWin: 0, avgLoss: 0, avgLossAbs: 0, beRate: 0 };
  }
  const wins  = records.filter((r) => Number(r.gold) > 0);
  const fails = records.filter((r) => Number(r.gold) <= 0);
  const total = records.reduce((acc, r) => acc + Number(r.gold || 0), 0);
  const avgWin  = wins.length  > 0 ? mean(wins.map((r)  => Number(r.gold || 0))) : 0;
  const avgLoss = fails.length > 0 ? mean(fails.map((r) => Number(r.gold || 0))) : 0;
  const ci = wilsonCI(fails.length, n);
  const avgLossAbs = Math.abs(avgLoss);
  const beRate = avgWin + avgLossAbs > 1e-9 ? avgLossAbs / (avgWin + avgLossAbs) : 0;
  return {
    n,
    wins: wins.length, fails: fails.length,
    winRate: wins.length / n, failRate: fails.length / n,
    failCiLo: ci.lo, failCiHi: ci.hi,
    ev: total / n, avgWin, avgLoss, avgLossAbs, beRate
  };
}

// ---------------------------------------------------------------------------
// Section 4: Feature decoding and record construction
// ---------------------------------------------------------------------------
// Feature index mapping follows model_duel_worker.mjs featureVectorForCandidate()
function decode(features) {
  const f = Array.isArray(features) ? features : [];
  const deck      = Math.round((Number(f[9])  || 0) * 30);
  const goCount   = Math.round((Number(f[12]) || 0) * 5) + 1; // 1-based
  const selfPi    = Math.round((Number(f[28]) || 0) * 20);
  const oppPi     = Math.round((Number(f[29]) || 0) * 20);
  const selfScore = Math.max(0, atanh(Number(f[15]) || 0) * 10);
  const scoreDiff = atanh(Number(f[14]) || 0) * 10;
  const oppScore  = Math.max(0, selfScore - scoreDiff);
  const oppGwang  = Math.round((Number(f[27]) || 0) * 5);
  const oppGodori = Math.round((Number(f[31]) || 0) * 3);
  const oppCheong = Math.round((Number(f[33]) || 0) * 3);
  const oppHong   = Math.round((Number(f[35]) || 0) * 3);
  const oppCho    = Math.round((Number(f[37]) || 0) * 3);
  const oppCombo  = [oppGodori, oppCheong, oppHong, oppCho].filter((v) => v >= 2).length;
  const oppBurst  = (oppGwang >= 2 ? 1 : 0) + (oppCombo > 0 ? 1 : 0) + (oppScore >= 6 ? 1 : 0);
  return {
    deck, goCount, selfPi, oppPi, piDiff: selfPi - oppPi,
    selfScore, oppScore, scoreDiff,
    oppCombo, oppBurst,
    selfCanStop: Number(f[38]) > 0.5,
    oppCanStop:  Number(f[39]) > 0.5,
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
    if (winner === actor)    gold =  Number(scores?.[actor]?.payoutTotal ?? 0);
    else if (winner === opp) gold = -Number(scores?.[opp]?.payoutTotal  ?? 0);
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
      game:   Number(r.game_index),
      action: r.candidate,
      gold:   Number(payoutMap.get(Number(r.game_index)) ?? 0),
      ...decode(r.features),
    }));
}

// ---------------------------------------------------------------------------
// Section 5: GO chain analysis and counterfactual (kibo-based)
// ---------------------------------------------------------------------------
function extractChains(kiboList, actor) {
  const opp = actor === "ai" ? "human" : "ai";
  const chains = [];
  for (const g of kiboList) {
    const kibo = Array.isArray(g.kibo) ? g.kibo : [];
    const re = [...kibo].reverse().find((k) => k?.type === "round_end");
    if (!re) continue;
    const won  = re.winner === actor;
    const gold = won
      ? +(re.scores?.[actor]?.payoutTotal ?? 0)
      : -(+(re.scores?.[opp]?.payoutTotal  ?? 0));
    const gos = [];
    for (const ev of kibo) {
      if (ev?.type !== "go" || ev?.playerKey !== actor) continue;
      const prevT = kibo.filter((k) => k?.type === "turn_end" && k.no < ev.no);
      const prev  = prevT[prevT.length - 1];
      gos.push({ goNo: ev.goCount, deck: prev?.deckCount ?? -1 });
    }
    if (gos.length) chains.push({ game: g.game_index, won, gold, gos, length: gos.length });
  }
  return chains;
}

function counterfactual(chains) {
  const rows = [];
  for (const c of chains.filter((c) => !c.won)) {
    for (let i = 0; i < c.gos.length; i++) {
      rows.push({
        game: c.game, position: i + 1, goNo: c.gos[i].goNo,
        deck: c.gos[i].deck, goldSaved: Math.abs(c.gold),
      });
    }
  }
  const byGoNo = new Map();
  for (const r of rows) {
    if (!byGoNo.has(r.goNo)) byGoNo.set(r.goNo, []);
    byGoNo.get(r.goNo).push(r);
  }
  const table = [];
  for (const [goNo, recs] of [...byGoNo.entries()].sort((a, b) => a[0] - b[0])) {
    const decks = recs.map((r) => r.deck).filter((d) => d >= 0);
    table.push({
      goNo, n: recs.length,
      avgDeck:        round(mean(decks), 1),
      avgGoldSaved:   round(mean(recs.map((r) => r.goldSaved)), 1),
      totalGoldAtRisk: round(recs.reduce((a, r) => a + r.goldSaved, 0), 0),
    });
  }
  return { rows, table };
}

// ---------------------------------------------------------------------------
// Section 6: Zone scoring and greedy planning
// ---------------------------------------------------------------------------
function buildZone(goRecs, fn, tags, baseStats, optionN) {
  const sub = goRecs.filter(fn);
  if (sub.length < MIN_ZONE_N) return null;
  const s = stats(sub);
  const evGap = baseStats.ev - s.ev;
  if (Math.abs(evGap) < 1e-9) return null;
  const direction = evGap > 0 ? "block" : "expand";
  if (direction === "expand" && s.failRate > ATTACK_MAX_FAIL) return null;

  // expand uses conservative EXPAND_FLIP_WEIGHT to avoid overestimation
  // (no direct STOP data — estimate only)
  const flipWeight  = direction === "expand" ? EXPAND_FLIP_WEIGHT : 1.0;
  const flipCount   = FLIP_RATE * sub.length * flipWeight;
  const edge        = Math.abs(evGap);
  const goldDelta   = flipCount * edge;
  const goDelta     = direction === "block" ? -flipCount : flipCount;
  const failDelta   = direction === "block"
    ? -flipCount * Math.max(0, s.failRate - baseStats.failRate)
    :  flipCount * Math.max(0, s.failRate - baseStats.failRate);
  const netEvPerDecision = goldDelta / Math.max(1, optionN);
  const reliable = s.n >= 20 || (
    direction === "block"
      ? s.failCiLo > baseStats.failCiHi
      : s.failCiHi < Math.max(baseStats.failRate, 0.12)
  );

  return {
    key:   `${tags.gcId}|${tags.dkId}|${tags.piId}|${tags.exId}`,
    label: `${tags.gc} ∧ ${tags.dk} ∧ ${tags.pi} ∧ ${tags.ex}`,
    gc: tags.gc, dk: tags.dk, pi: tags.pi, ex: tags.ex,
    direction, n: s.n,
    ev: s.ev, failRate: s.failRate, failCiLo: s.failCiLo, failCiHi: s.failCiHi,
    avgWin: s.avgWin, avgLoss: s.avgLoss,
    evGap, goldDelta, goDelta, failDelta, netEvPerDecision, reliable, fn,
  };
}

function enumerateZonesFromPool(pool, baseStats, optionN) {
  const zones = [];
  for (const gc of GC_DEFS) {
    for (const dk of DK_DEFS) {
      for (const pi of PI_DEFS) {
        for (const ex of EX_DEFS) {
          // skip pure "any" combos (they'd just be the baseline)
          const allAny = dk.id === "dkAny" && pi.id === "piAny" && ex.id === "exAny";
          if (allAny) continue;
          const fn = (r) => gc.fn(r) && dk.fn(r) && pi.fn(r) && ex.fn(r);
          const zone = buildZone(
            pool, fn,
            { gcId: gc.id, dkId: dk.id, piId: pi.id, exId: ex.id,
              gc: gc.label, dk: dk.label, pi: pi.label, ex: ex.label },
            baseStats, optionN
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

// Greedy sequential planner:
//   re-discovers zones each step from remaining pool → removes selected zone records.
//   Prevents double-counting overlapping zones.
function buildGreedyPlan(goRecs, optionN, topZones, minEv) {
  const plan = [];
  let pool = [...goRecs];
  let cumGold = 0, cumGo = 0, cumFail = 0;

  for (let step = 1; step <= topZones; step += 1) {
    if (pool.length < MIN_ZONE_N) break;
    const baseStats = stats(pool);
    const zones = enumerateZonesFromPool(pool, baseStats, optionN)
      .filter((z) => z.netEvPerDecision >= minEv);
    if (zones.length <= 0) break;
    const best = zones[0];

    cumGold += best.goldDelta;
    cumGo   += best.goDelta;
    cumFail += best.failDelta;

    plan.push({
      step,
      direction:  best.direction,
      zoneKey:    best.key,
      zoneLabel:  best.label,
      gc: best.gc, dk: best.dk, pi: best.pi, ex: best.ex,
      zoneN:      best.n,
      zoneEv:     round(best.ev, 3),
      zoneFailRate: round(best.failRate, 6),
      evGap:      round(best.evGap, 3),
      goldDelta:  round(best.goldDelta, 3),
      goDelta:    round(best.goDelta, 3),
      failDelta:  round(best.failDelta, 3),
      netEvPerDecision: round(best.netEvPerDecision, 6),
      cumGold:     round(cumGold, 3),
      cumGoDelta:  round(cumGo,   3),
      cumFailDelta: round(cumFail, 3),
      reliable: best.reliable,
    });

    // remove zone records from pool so next step doesn't double-count
    pool = pool.filter((r) => !best.fn(r));
  }

  return plan;
}

// ---------------------------------------------------------------------------
// Section 7: Parameter mapping and heuristic file updates
// ---------------------------------------------------------------------------
const PARAM_RULES = [
  // BLOCK rules
  { match: (z) => z.direction === "block" && z.gc === "goCount=1" && z.dk === "deck<=4",
    param: "goBaseThreshold",       delta: +0.01,  min: -0.2, max: 0.2,
    reason: "Raise GO threshold: first-GO at deck≤4 is risky (high fail, low EV)." },
  { match: (z) => z.direction === "block" && z.gc === "goCount=1" && z.pi === "piDiff<5",
    param: "goMinPi",               delta: +1,     min: 3,    max: 12,
    reason: "Require stronger pi edge before first GO when piDiff is weak." },
  { match: (z) => z.direction === "block" && z.gc === "goCount=1" && z.ex === "oppCanStop",
    param: "goLiteOppCanStopPenalty", delta: +0.02, min: 0,   max: 1.0,
    reason: "Penalize first-GO more when opponent can already stop." },
  { match: (z) => z.direction === "block" && z.dk === "deck<=4",
    param: "goLiteLatePenalty",     delta: +0.01,  min: 0,    max: 1.0,
    reason: "Suppress late-deck fragile GO via late-penalty increase." },
  { match: (z) => z.direction === "block" && z.ex === "oppBurst>=2",
    param: "goHardLateOneAwayCut",  delta: -5,     min: 30,   max: 95,
    reason: "Block GO earlier under opponent burst-risk signals." },
  { match: (z) => z.direction === "block" && z.gc === "goCount=1" && z.dk === "deck<=8",
    param: "goUnseeHighPiPenalty",  delta: +0.01,  min: 0,    max: 0.5,
    reason: "Raise unseen-pi penalty to curb fragile first-GO entries." },
  { match: (z) => z.direction === "block" && z.gc === "goCount=1" && z.pi === "piDiff<10",
    param: "goScoreDiffBonus",      delta: +0.005, min: 0,    max: 1.0,
    reason: "Require larger score lead for first-GO when piDiff is weak." },
  { match: (z) => z.direction === "block" && z.gc === "goCount=2",
    param: "goBaseThreshold",       delta: +0.005, min: -0.2, max: 0.2,
    reason: "Slight threshold increase for second-GO weak zones." },
  // EXPAND rules
  { match: (z) => z.direction === "expand" && z.gc === "goCount>=3",
    param: "goRiskGoCountMul",      delta: -0.02,  min: 0.02, max: 0.5,
    reason: "Reduce GO-count risk multiplier: GO3+ chains show high EV." },
  { match: (z) => z.direction === "expand" && z.gc === "goCount=2" && z.pi === "piDiff>=10",
    param: "goBaseThreshold",       delta: -0.01,  min: -0.2, max: 0.2,
    reason: "Lower threshold in strong GO2 high-piDiff contexts." },
  { match: (z) => z.direction === "expand" && z.gc === "goCount>=3" && z.pi === "piDiff>=10",
    param: "goDeckLowBonus",        delta: +0.01,  min: -1.0, max: 1.0,
    reason: "Allow more late continuation in high-piDiff GO3+ contexts." },
  { match: (z) => z.direction === "expand" && z.gc === "goCount=2",
    param: "goScoreDiffBonus",      delta: -0.005, min: 0,    max: 1.0,
    reason: "Relax score-gap gate in validated GO2 zone." },
  { match: (z) => z.direction === "expand",
    param: "goUnseeHighPiPenalty",  delta: -0.01,  min: 0,    max: 0.5,
    reason: "Reduce over-penalization in validated high-EV expand zone." },
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
    return { param: rule.param, current: cur, suggested, reason: rule.reason };
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

// Preserve BOM if present in original file.
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
// Section 8: Reporting
// ---------------------------------------------------------------------------
function buildSummary(P) {
  const L  = "─".repeat(72);
  const LL = "═".repeat(72);
  const lines = [];

  lines.push(LL);
  lines.push(`  Optimizer by CL  —  GO/STOP Gold Optimizer  v${P.version}`);
  lines.push(LL);
  lines.push(`  Generated : ${P.generated_at}`);
  lines.push(`  Actor     : ${P.input.actor} / Policy: ${P.input.actor_policy}`);
  lines.push(`  Kibo      : ${P.input.kibo}`);
  lines.push(`  Dataset   : ${P.input.dataset}`);
  lines.push(`  Params    : ${P.input.params_file}`);

  // ── Baseline ──
  lines.push(`\n${L}`);
  lines.push("  BASELINE");
  lines.push(L);
  const B = P.baseline;
  lines.push(`  Games  : ${B.games}  Wins=${B.wins} (${pct(B.games ? B.wins / B.games : 0)})  MeanGold=${round(B.meanGold, 1)}`);
  lines.push(`  GO     : n=${B.go.n}  fail=${pct(B.go.failRate)} [${pct(B.go.failCiLo)}~${pct(B.go.failCiHi)}]  EV=${round(B.go.ev, 1)}  breakEven=${pct(B.go.beRate)}`);
  lines.push(`  STOP   : n=${B.stop.n}  fail=${pct(B.stop.failRate)}  EV=${round(B.stop.ev, 1)}`);
  lines.push(`  GOrate : ${pct(B.goChoiceRate)}  (GO chosen / all option decisions)`);
  lines.push(`  avgWin : ${fmt(B.go.avgWin, 0)}  avgLoss: ${round(B.go.avgLoss, 0)}`);

  // ── Chain analysis ──
  if (P.chain_analysis && P.chain_analysis.totalChains > 0) {
    const CA = P.chain_analysis;
    lines.push(`\n${L}`);
    lines.push("  GO CHAIN ANALYSIS  (kibo-based)");
    lines.push(L);
    lines.push(`  Total chains: ${CA.totalChains}  won=${CA.wonChains}  lost=${CA.lostChains}`);
    lines.push(`  Chain lengths: ${Object.entries(CA.byLength).sort((a, b) => +a[0] - +b[0]).map(([k, v]) => `${k}GO×${v}`).join("  ")}`);
    for (const r of CA.winRateByLength) {
      const bar = "█".repeat(Math.round(r.winRate * 20));
      lines.push(`    ${r.length}GO-chain : n=${r.n}  WR=${pct(r.winRate)}  EV=${fmt(r.ev)}  ${bar}`);
    }

    // ── Counterfactual ──
    const CF = P.counterfactual;
    if (CF && CF.table && CF.table.length) {
      lines.push("");
      lines.push("  COUNTERFACTUAL  — gold at risk per GO position in losing chains");
      lines.push("  GOno  cases  avgDeck  goldSaved  totalAtRisk");
      lines.push("  " + "─".repeat(48));
      for (const r of CF.table) {
        lines.push(
          `   GO#${r.goNo}  ${String(r.n).padStart(4)}   ${String(r.avgDeck).padStart(5)}   ` +
          `${String(r.avgGoldSaved).padStart(8)}   ${String(r.totalGoldAtRisk).padStart(10)}`
        );
      }
    }
  }

  // ── Top zones ──
  lines.push(`\n${L}`);
  lines.push(`  TOP ZONES  (auto-discovered, top ${Math.min(12, P.top_zones.length)} of ${P.top_zones.length})`);
  lines.push(L);
  lines.push("  Dir     n  EV    failRate  goldDelta  netEV   Zone");
  lines.push("  " + "─".repeat(70));
  for (const z of P.top_zones.slice(0, 12)) {
    lines.push(
      `  ${z.direction === "block" ? "BLOCK " : "EXPAND"} ${String(z.n).padStart(3)}` +
      `  ${round(z.ev, 0).toString().padStart(5)}` +
      `  ${pct(z.failRate).padStart(8)}` +
      `  ${fmt(z.goldDelta, 0).padStart(9)}` +
      `  ${round(z.netEvPerDecision, 2).toString().padStart(6)}` +
      `  ${z.label}`
    );
  }

  // ── Greedy plan ──
  lines.push(`\n${L}`);
  lines.push("  GREEDY PLAN  (sequential, non-overlapping, zones re-discovered each step)");
  lines.push(L);
  if (!P.greedy_plan.length) {
    lines.push("  (no zone met the EV threshold — try --min-ev lower)");
  } else {
    lines.push("  #  Dir     EV_zone  goldΔ    goΔ   failΔ   cumGold  Zone");
    lines.push("  " + "─".repeat(70));
    for (const s of P.greedy_plan) {
      lines.push(
        `  ${s.step}  ${s.direction === "block" ? "BLOCK " : "EXPAND"}` +
        `  ${round(s.zoneEv, 0).toString().padStart(7)}` +
        `  ${fmt(s.goldDelta, 0).padStart(7)}` +
        `  ${fmt(s.goDelta, 0).padStart(5)}` +
        `  ${fmt(s.failDelta, 1).padStart(6)}` +
        `  ${fmt(s.cumGold, 0).padStart(8)}` +
        `  ${s.zoneLabel}`
      );
    }
    const last = P.greedy_plan[P.greedy_plan.length - 1];
    lines.push(`\n  FORECAST  goldΔ=${fmt(last.cumGold, 0)}  goΔ=${fmt(last.cumGoDelta, 0)}  failΔ=${fmt(last.cumFailDelta, 1)}`);
  }

  // ── Param recommendations ──
  lines.push(`\n${L}`);
  lines.push("  PARAM RECOMMENDATIONS");
  lines.push(L);
  if (!P.param_plan.length) {
    lines.push("  (no params mapped — check --params-file references P.paramName)");
  } else {
    for (const pr of P.param_plan) {
      const tag = pr.direction === "block" ? "DEFENSE" : "ATTACK ";
      lines.push(`  [${tag}]  ${pr.param}: ${round(pr.current, 6)} → ${round(pr.suggested, 6)}   goldΔ≈${fmt(pr.goldDelta, 0)}`);
      lines.push(`             zone: ${pr.zoneLabel}`);
      lines.push(`             ${pr.reason}`);
    }
  }

  lines.push(`\n${L}`);
  lines.push("  NOTE: all values are estimates. Validate with real 1000-game rerun.");
  lines.push("  NOTE: expand estimates use EXPAND_FLIP_WEIGHT=0.5 (no STOP data — conservative).");
  lines.push(LL);
  return `${lines.join("\n")}\n`;
}

// ---------------------------------------------------------------------------
// Section 9: Main execution flow
// ---------------------------------------------------------------------------
const argv = parseArgs(process.argv.slice(2));

if (argv.help) {
  console.log(`
  node heuristic_tuning/optimizer_by_cl.mjs \\
    --kibo <kibo.jsonl> --dataset <dataset.jsonl> \\
    --actor ai --actor-policy H-NEXg \\
    --params-file src/heuristics/heuristicNEXg.js \\
    [--out-root <dir>] [--top-zones 12] [--min-ev 1.5] [--apply]
`);
  process.exit(0);
}

const cfg = {
  kibo:        String(argv.kibo || ""),
  dataset:     String(argv.dataset || ""),
  actor:       String(argv.actor || "ai").toLowerCase(),
  actorPolicy: argv["actor-policy"] != null ? String(argv["actor-policy"]) : null,
  paramsFile:  String(argv["params-file"] || "src/heuristics/heuristicNEXg.js"),
  outRoot:     argv["out-root"]
    ? String(argv["out-root"])
    : join(resolve(dirname(String(argv.kibo || "."))), "optimize_cl"),
  topZones:    Number(argv["top-zones"] || DEFAULT_TOP_ZONES),
  minEv:       Number(argv["min-ev"]    || DEFAULT_MIN_EV),
  apply:       Boolean(argv.apply),
};

if (!cfg.kibo)        throw new Error("--kibo required");
if (!cfg.dataset)     throw new Error("--dataset required");
if (cfg.actor !== "ai" && cfg.actor !== "human") throw new Error(`invalid --actor: ${cfg.actor}`);
if (!cfg.actorPolicy) throw new Error("--actor-policy required");
if (!existsSync(cfg.kibo))       throw new Error(`kibo not found: ${cfg.kibo}`);
if (!existsSync(cfg.dataset))    throw new Error(`dataset not found: ${cfg.dataset}`);
if (!existsSync(cfg.paramsFile)) throw new Error(`params-file not found: ${cfg.paramsFile}`);
if (!Number.isFinite(cfg.topZones) || cfg.topZones < 1) throw new Error(`invalid --top-zones: ${cfg.topZones}`);
if (!Number.isFinite(cfg.minEv)    || cfg.minEv    < 0) throw new Error(`invalid --min-ev: ${cfg.minEv}`);

// ── load ──
const kibo    = readJsonl(cfg.kibo);
const dataset = readJsonl(cfg.dataset);

const payoutMap = buildPayoutMap(kibo, cfg.actor);
const chains    = extractChains(kibo, cfg.actor);

// ── game-level stats ──
const goldVals = [...payoutMap.values()].map((x) => Number(x) || 0);
const gameWins = goldVals.filter((v) => v > 0).length;
const meanGold = mean(goldVals);

// ── decision records ──
const records  = buildDecisionRecords(dataset, cfg.actor, cfg.actorPolicy, payoutMap);
if (!records.length) throw new Error("no matching option decision records");

const goRecs  = records.filter((r) => r.action === "go");
const stopRecs = records.filter((r) => r.action === "stop");
if (!goRecs.length) throw new Error("no GO records found for selected actor/policy");

const optionN = Math.max(1, records.length);
const bsGo    = stats(goRecs);
const bsStop  = stats(stopRecs);

// ── chain analysis ──
const wonC  = chains.filter((c) => c.won);
const lostC = chains.filter((c) => !c.won);
const byLen = {};
for (const c of chains) byLen[c.length] = (byLen[c.length] || 0) + 1;
const winRateByLength = Object.entries(byLen).map(([len, n]) => {
  const sub = chains.filter((c) => c.length === Number(len));
  const w   = sub.filter((c) => c.won).length;
  return { length: Number(len), n, wins: w, winRate: round(w / n, 4), ev: round(mean(sub.map((c) => c.gold)), 1) };
}).sort((a, b) => a.length - b.length);

// ── counterfactual ──
const cf = counterfactual(chains);

// ── zone discovery (for top_zones report only) ──
const baseZones = enumerateZonesFromPool(goRecs, bsGo, optionN);

// ── greedy plan (re-discovers each step) ──
const greedyPlan = buildGreedyPlan(goRecs, optionN, cfg.topZones, cfg.minEv);

// ── param mapping ──
const currentParams  = await loadParams(cfg.paramsFile);
const referencedKeys = extractReferencedParamKeys(cfg.paramsFile);
const paramPlan = [];
const usedParamDir = new Set();
for (const step of greedyPlan) {
  // find zone by key from baseZones for mapZoneToParam
  const z = baseZones.find((z) => z.key === step.zoneKey) ?? { ...step, gc: step.gc, dk: step.dk, pi: step.pi, ex: step.ex };
  const mapped = mapZoneToParam(z, currentParams, referencedKeys);
  if (!mapped) continue;
  const key = `${step.direction}:${mapped.param}`;
  if (usedParamDir.has(key)) continue;
  usedParamDir.add(key);
  paramPlan.push({
    rank: paramPlan.length + 1,
    direction: step.direction,
    zoneKey:   step.zoneKey,
    zoneLabel: step.zoneLabel,
    ...mapped,
    goldDelta: step.goldDelta,
    goDelta:   step.goDelta,
    failDelta: step.failDelta,
  });
}

// ── payload ──
const last = greedyPlan.length ? greedyPlan[greedyPlan.length - 1] : null;
const payload = {
  generated_at: new Date().toISOString(),
  version:      VERSION,
  input: {
    kibo: cfg.kibo, dataset: cfg.dataset,
    actor: cfg.actor, actor_policy: cfg.actorPolicy, params_file: cfg.paramsFile,
  },
  baseline: {
    games: goldVals.length, wins: gameWins, meanGold: round(meanGold, 2),
    goChoiceRate: round(optionN > 0 ? goRecs.length / optionN : 0, 4),
    go: {
      n: bsGo.n, failRate: round(bsGo.failRate, 4),
      failCiLo: round(bsGo.failCiLo, 4), failCiHi: round(bsGo.failCiHi, 4),
      ev: round(bsGo.ev, 2), avgWin: round(bsGo.avgWin, 2), avgLoss: round(bsGo.avgLoss, 2),
      beRate: round(bsGo.beRate, 4),
    },
    stop: { n: bsStop.n, failRate: round(bsStop.failRate, 4), ev: round(bsStop.ev, 2) },
  },
  chain_analysis: {
    totalChains: chains.length, wonChains: wonC.length, lostChains: lostC.length,
    byLength: byLen, winRateByLength,
  },
  counterfactual: { table: cf.table },
  top_zones: baseZones.slice(0, cfg.topZones * 2).map((z) => ({
    key: z.key, label: z.label, direction: z.direction, n: z.n,
    ev: round(z.ev, 1), failRate: round(z.failRate, 4),
    evGap: round(z.evGap, 1), goldDelta: round(z.goldDelta, 1),
    netEvPerDecision: round(z.netEvPerDecision, 6),
    reliable: z.reliable,
  })),
  greedy_plan: greedyPlan,
  param_plan:  paramPlan,
  forecast: last
    ? { total_gold_delta: last.cumGold, total_go_delta: last.cumGoDelta, total_fail_delta: last.cumFailDelta }
    : { total_gold_delta: 0, total_go_delta: 0, total_fail_delta: 0 },
  settings: {
    min_ev: cfg.minEv, top_zones: cfg.topZones,
    min_zone_n: MIN_ZONE_N, flip_rate: FLIP_RATE,
    expand_flip_weight: EXPAND_FLIP_WEIGHT,
    attack_max_fail: ATTACK_MAX_FAIL,
  },
};

// ── write ──
mkdirSync(cfg.outRoot, { recursive: true });
const summaryPath = join(cfg.outRoot, "optimizer_cl_summary.txt");
const jsonPath    = join(cfg.outRoot, "optimizer_cl_plan.json");
writeUtf8Bom(summaryPath, buildSummary(payload));
writeUtf8Bom(jsonPath, `${JSON.stringify(payload, null, 2)}\n`);

// ── apply ──
if (cfg.apply && paramPlan.length > 0) {
  const changed = applyToFile(cfg.paramsFile, paramPlan);
  console.log(`apply: wrote ${changed} param(s) → ${cfg.paramsFile}`);
}

// ── console ──
console.log("=== Optimizer by CL ===");
console.log(`run_dir : ${cfg.outRoot}`);
console.log(`summary : ${summaryPath}`);
console.log(`json    : ${jsonPath}`);
console.log(`GO n=${bsGo.n}, fail=${pct(bsGo.failRate)}, EV=${round(bsGo.ev, 1)}  |  STOP n=${bsStop.n}`);
console.log(`chains=${chains.length}  zones=${baseZones.length}  greedy_steps=${greedyPlan.length}  param_changes=${paramPlan.length}`);
if (last) console.log(`forecast: goldΔ=${fmt(last.cumGold, 0)}  goΔ=${fmt(last.cumGoDelta, 0)}  failΔ=${fmt(last.cumFailDelta, 1)}`);
