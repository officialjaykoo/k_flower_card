import fs from "node:fs";
import path from "node:path";
import { once } from "node:events";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";
import {
  initGame,
  createSeededRng,
  playTurn,
  calculateScore,
  chooseGo,
  chooseStop,
  chooseShakingYes,
  chooseShakingNo,
  choosePresidentStop,
  choosePresidentHold,
  chooseGukjinMode,
  chooseMatch,
  getDeclarableBombMonths,
  getDeclarableShakingMonths
} from "../src/engine/index.js";
import { STARTING_GOLD } from "../src/engine/economy.js";
import { buildDeck } from "../src/cards.js";
import { BOT_POLICIES } from "../src/ai/policies.js";
import { aiPlay } from "../src/ai/aiPlay.js";
import { getActionPlayerKey } from "../src/engine/runner.js";

// -----------------------------------------------------------------------------
// 1) Process guard
// -----------------------------------------------------------------------------
if (process.env.NO_SIMULATION === "1") {
  console.error("Simulation blocked: NO_SIMULATION=1");
  process.exit(2);
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// -----------------------------------------------------------------------------
// 2) Global constants
// -----------------------------------------------------------------------------
const DEFAULT_LOG_MODE = "train";
const SUPPORTED_POLICIES = new Set(BOT_POLICIES);
const DEFAULT_POLICY = "heuristic_v3";
const DECISION_CACHE_MAX = 200000;
const HASH_CACHE_MAX = 500000;
const GO_STOP_THRESHOLD_BASE_PERMILLE = 1200;
const GO_STOP_THRESHOLD_AHEAD_PERMILLE = 2000;
const GO_STOP_THRESHOLD_BEHIND_PERMILLE = 500;
const MIXED_STRATEGY_ENABLED_DEFAULT = true;
const MIXED_TOP_K_PLAY = 3;
const MIXED_TOP_K_MATCH = 2;
const MIXED_TOP_K_OPTION = 2;
const MIXED_TEMPERATURE_PLAY = 0.9;
const MIXED_TEMPERATURE_MATCH = 0.8;
const MIXED_TEMPERATURE_OPTION = 0.7;
const OPTION_LOOKAHEAD_ENABLED_DEFAULT = true;
const OPTION_LOOKAHEAD_ROLLOUTS_DEFAULT = 4;
const OPTION_LOOKAHEAD_MAX_STEPS_DEFAULT = 180;
const OPTION_LOOKAHEAD_DOWNSIDE_LAMBDA = 0.25;
const IR_TURN_CLIP = 0.3;
const IR_EPISODE_CLIP = 1.0;
const LEARNING_ROLE_ATTACK_SCORE_MIN = 20;
const LEARNING_ROLE_DEFENSE_OPP_SCORE_MAX = 10;
const FULL_DECK = buildDeck();
const RIBBON_COMBO_MONTH_SETS = Object.freeze({
  hongdan: Object.freeze([1, 2, 3]),
  cheongdan: Object.freeze([6, 9, 10]),
  chodan: Object.freeze([4, 5, 7])
});
const GODORI_MONTHS = Object.freeze([2, 4, 8]);
const GWANG_MONTHS = Object.freeze([1, 3, 8, 11, 12]);
const TRACE_SCHEMA_VERSION = 2;
const TRACE_FEATURE_VERSION = 14;
const TRACE_TRIGGER_DICT_VERSION = 2;

const decisionInferenceCache = new Map();
const hashIndexCache = new Map();
const SIDE_MY = "mySide";
const SIDE_YOUR = "yourSide";

// -----------------------------------------------------------------------------
// 3) Generic utilities
// -----------------------------------------------------------------------------
function setBoundedCache(cache, key, value, maxEntries) {
  if (cache.size >= maxEntries) cache.clear();
  cache.set(key, value);
}

function deterministicUnitFromText(text) {
  const digest = crypto.createHash("md5").update(String(text || ""), "utf8").digest("hex");
  const n = Number.parseInt(digest.slice(0, 8), 16);
  return n / 0xffffffff;
}

function softmaxSampleFromScores(items, scoreOf, topK, temperature, entropyKey) {
  const ranked = [...items].sort((a, b) => Number(scoreOf.get(b) || -Infinity) - Number(scoreOf.get(a) || -Infinity));
  const k = Math.max(1, Math.min(ranked.length, Math.floor(Number(topK || ranked.length))));
  const chosen = ranked.slice(0, k);
  if (chosen.length <= 1) return chosen[0];
  const temp = Math.max(0.05, Number(temperature || 1.0));
  const raw = chosen.map((c) => Number(scoreOf.get(c) || -Infinity));
  const maxRaw = Math.max(...raw);
  const exps = raw.map((x) => Math.exp((x - maxRaw) / temp));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  if (!(sumExp > 0)) return chosen[0];
  const u = deterministicUnitFromText(entropyKey);
  let acc = 0;
  for (let i = 0; i < chosen.length; i += 1) {
    acc += exps[i] / sumExp;
    if (u <= acc) return chosen[i];
  }
  return chosen[chosen.length - 1];
}

function actorToSide(actor, firstTurnKey) {
  return actor === firstTurnKey ? SIDE_MY : SIDE_YOUR;
}

function sideToActor(side, firstTurnKey, secondTurnKey) {
  if (side === SIDE_MY) return firstTurnKey;
  if (side === SIDE_YOUR) return secondTurnKey;
  return null;
}

function findOppActor(state, actor) {
  const actorKeys = Object.keys(state?.players || {});
  return actorKeys.find((k) => k !== actor) || null;
}

function secondActorFromFirst(state, firstTurnKey) {
  return Object.keys(state?.players || {}).find((k) => k !== firstTurnKey) || null;
}

function actorPairFromState(state) {
  const actorKeys = Object.keys(state?.players || {});
  if (actorKeys.length !== 2) {
    throw new Error(`expected exactly 2 actors in state.players, got ${actorKeys.length}`);
  }
  return actorKeys;
}

function normalizePolicyInput(raw) {
  const p = String(raw ?? DEFAULT_POLICY).trim().toLowerCase();
  return p || DEFAULT_POLICY;
}

// -----------------------------------------------------------------------------
// 4) CLI parsing
// -----------------------------------------------------------------------------
function parseArgs(argv) {
  const args = [...argv];
  let games = 1000;
  let outArg = null;
  let policyMySide = DEFAULT_POLICY;
  let policyYourSide = DEFAULT_POLICY;
  let policyModelMySide = null;
  let policyModelYourSide = null;
  let fixedSeats = false; // default: switching first-turn plan
  let dedupeStableTurns = false;

  if (args.length > 0 && /^\d+$/.test(args[0])) {
    games = Number(args.shift());
  }

  if (args.length > 0 && !args[0].startsWith("--")) {
    outArg = args.shift();
  }

  while (args.length > 0) {
    const arg = args.shift();
    if (arg === "--policy-my-side" && args.length > 0) {
      policyMySide = normalizePolicyInput(args.shift());
      continue;
    }
    if (arg.startsWith("--policy-my-side=")) {
      policyMySide = normalizePolicyInput(arg.split("=", 2)[1]);
      continue;
    }
    if (arg === "--policy-your-side" && args.length > 0) {
      policyYourSide = normalizePolicyInput(args.shift());
      continue;
    }
    if (arg.startsWith("--policy-your-side=")) {
      policyYourSide = normalizePolicyInput(arg.split("=", 2)[1]);
      continue;
    }
    if (arg === "--policy-model-my-side" && args.length > 0) {
      policyModelMySide = String(args.shift()).trim();
      continue;
    }
    if (arg.startsWith("--policy-model-my-side=")) {
      policyModelMySide = arg.split("=", 2)[1].trim();
      continue;
    }
    if (arg === "--policy-model-your-side" && args.length > 0) {
      policyModelYourSide = String(args.shift()).trim();
      continue;
    }
    if (arg.startsWith("--policy-model-your-side=")) {
      policyModelYourSide = arg.split("=", 2)[1].trim();
      continue;
    }
    if (arg === "--fixed-seats") {
      fixedSeats = true;
      continue;
    }
    if (arg === "--switch-seats") {
      fixedSeats = false;
      continue;
    }
    if (arg === "--dedupe-stable-turns") {
      dedupeStableTurns = true;
      continue;
    }
    if (arg === "--no-dedupe-stable-turns") {
      dedupeStableTurns = false;
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }

  if (!Number.isInteger(games) || games <= 0) {
    throw new Error(`games must be a positive integer. Received: ${games}`);
  }
  if (!fixedSeats && games % 2 !== 0) {
    throw new Error(
      `games must be even in switching mode for 50:50 first-turn split. Received: ${games}`
    );
  }
  if (!SUPPORTED_POLICIES.has(policyMySide)) {
    throw new Error(
      `Unsupported policy for mySide: ${policyMySide}. Use one of: ${[...SUPPORTED_POLICIES].join(", ")}`
    );
  }
  if (!SUPPORTED_POLICIES.has(policyYourSide)) {
    throw new Error(
      `Unsupported policy for yourSide: ${policyYourSide}. Use one of: ${[...SUPPORTED_POLICIES].join(", ")}`
    );
  }

  return {
    games,
    outArg,
    policyMySide,
    policyYourSide,
    policyModelMySide,
    policyModelYourSide,
    fixedSeats,
    dedupeStableTurns
  };
}

// -----------------------------------------------------------------------------
// 5) Runtime bootstrap (args -> config -> models)
// -----------------------------------------------------------------------------
const parsed = parseArgs(process.argv.slice(2));
const games = parsed.games;
const outArg = parsed.outArg;
const logMode = DEFAULT_LOG_MODE;
const policyMySide = parsed.policyMySide;
const policyYourSide = parsed.policyYourSide;
const policyModelMySidePath = parsed.policyModelMySide;
const policyModelYourSidePath = parsed.policyModelYourSide;
const fixedSeats = parsed.fixedSeats;
const traceDedupeStableTurns = parsed.dedupeStableTurns;
const seatMode = fixedSeats ? "fixed" : "switching";
const traceMyTurnOnly = false;
const traceImportantOnly = true;
const traceContextRadius = 0;
const traceGoStopPlus2 = false;
const sideConfig = {
  [SIDE_MY]: {
    fallbackPolicy: policyMySide,
    policyModelPath: policyModelMySidePath
  },
  [SIDE_YOUR]: {
    fallbackPolicy: policyYourSide,
    policyModelPath: policyModelYourSidePath
  }
};
sideConfig[SIDE_MY].policyModel = loadJsonModel(
  sideConfig[SIDE_MY].policyModelPath,
  "policy-model-my-side"
);
sideConfig[SIDE_YOUR].policyModel = loadJsonModel(
  sideConfig[SIDE_YOUR].policyModelPath,
  "policy-model-your-side"
);

const agentLabelBySide = {
  [SIDE_MY]:
    sideConfig[SIDE_MY].policyModel
      ? `model:${path.basename(sideConfig[SIDE_MY].policyModelPath)}`
      : sideConfig[SIDE_MY].fallbackPolicy,
  [SIDE_YOUR]:
    sideConfig[SIDE_YOUR].policyModel
      ? `model:${path.basename(sideConfig[SIDE_YOUR].policyModelPath)}`
      : sideConfig[SIDE_YOUR].fallbackPolicy
};

const stamp = new Date().toISOString().replace(/[:.]/g, "-");
const runId = `run-${stamp}-${crypto.randomBytes(3).toString("hex")}`;
const outPath = outArg || path.resolve(__dirname, "..", "logs", `side-vs-side-${stamp}.jsonl`);
const reportPath = outPath.replace(/\.jsonl$/i, "-report.json");
const sharedCatalogDir = path.resolve(__dirname, "..", "logs", "catalog");
const sharedCatalogPath = path.join(sharedCatalogDir, "cards-catalog.json");

fs.mkdirSync(path.dirname(outPath), { recursive: true });

// -----------------------------------------------------------------------------
// 6) Aggregation / counters
// -----------------------------------------------------------------------------
const aggregate = {
  games,
  completed: 0,
  winners: { mySide: 0, yourSide: 0, draw: 0, unknown: 0 },
  learningRoleCounts: {
    ATTACK: 0,
    DEFENSE: 0,
    NEUTRAL: 0
  },
  bySide: {
    mySideWins: 0,
    yourSideWins: 0,
    draw: 0,
    mySideScoreSum: 0,
    yourSideScoreSum: 0
  },
  economy: {
    mySideGoldSum: 0,
    yourSideGoldSum: 0,
    mySideDeltaSum: 0,
    first1000MySideDeltaSum: 0,
    first1000Games: 0
  },
  bankrupt: {
    mySideInflicted: 0,
    mySideSuffered: 0,
    yourSideInflicted: 0,
    yourSideSuffered: 0,
    resets: 0
  },
  nagari: 0,
  eventTotals: { ppuk: 0, ddadak: 0, jjob: 0, ssul: 0, pansseul: 0 },
  goCalls: 0,
  goEfficiencySum: 0,
  goDecision: {
    declared: 0,
    success: 0
  },
  steals: {
    piTotal: 0,
    goldTotal: 0
  },
  bakEscape: {
    trials: 0,
    escaped: 0
  },
  bakBreakdown: {
    totalLoserCount: 0,
    piBakCount: 0,
    gwangBakCount: 0,
    mongBakCount: 0,
    doubleBakCount: 0
  },
  luckFlipEvents: 0,
  luckHandEvents: 0,
  luckFlipCaptureValue: 0,
  luckHandCaptureValue: 0
};

function classifyLearningRoleMySide(winnerSide, mySideScore, yourSideScore) {
  if (winnerSide === SIDE_MY && Number(mySideScore || 0) >= LEARNING_ROLE_ATTACK_SCORE_MIN) {
    return "ATTACK";
  }
  if (
    winnerSide === SIDE_YOUR &&
    Number(yourSideScore || 0) <= LEARNING_ROLE_DEFENSE_OPP_SCORE_MAX
  ) {
    return "DEFENSE";
  }
  return "NEUTRAL";
}

function addLearningRoleCount(role) {
  if (aggregate.learningRoleCounts[role] == null) return;
  aggregate.learningRoleCounts[role] += 1;
}

// -----------------------------------------------------------------------------
// 7) Game scheduling / catalog helpers
// -----------------------------------------------------------------------------
function createBalancedFirstTurnPlan(totalGames, actorA, actorB) {
  if (totalGames % 2 !== 0) {
    throw new Error(
      `games must be even for exact 50:50 first-turn split. Received: ${totalGames}`
    );
  }
  const half = totalGames / 2;
  const plan = [];
  for (let i = 0; i < half; i += 1) plan.push(actorA);
  for (let i = 0; i < half; i += 1) plan.push(actorB);
  for (let i = plan.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    const tmp = plan[i];
    plan[i] = plan[j];
    plan[j] = tmp;
  }
  return plan;
}

function createFirstTurnPlan(totalGames, actorA, actorB, isFixedSeats) {
  if (isFixedSeats) {
    return Array.from({ length: totalGames }, () => actorA);
  }
  return createBalancedFirstTurnPlan(totalGames, actorA, actorB);
}

function captureWeight(card) {
  if (!card) return 0;
  if (card.category === "kwang") return 6;
  if (card.category === "five") return 4;
  if (card.category === "ribbon") return 2;
  if (card.category === "junk") return 1;
  return 0;
}

function cardTypeCode(card) {
  if (card.category === "kwang") return "K";
  if (card.category === "five") return "F";
  if (card.category === "ribbon") return "R";
  return "J";
}

function piLikeValue(card) {
  if (card.category !== "junk") return 0;
  const explicit = Number(card?.piValue);
  if (Number.isFinite(explicit) && explicit > 0) return explicit;
  return 1;
}

function buildCatalog() {
  const deck = buildDeck();
  const catalog = {};
  deck.forEach((c) => {
    catalog[c.id] = {
      month: c.month,
      category: c.category,
      type: cardTypeCode(c),
      name: c.name,
      label: `${c.month}-${c.name}-${cardTypeCode(c)}`,
      piValue: piLikeValue(c),
      bonusStealPi: c.bonus?.stealPi || 0
    };
  });
  return catalog;
}

function ensureSharedCatalog() {
  fs.mkdirSync(sharedCatalogDir, { recursive: true });
  if (fs.existsSync(sharedCatalogPath)) return sharedCatalogPath;
  fs.writeFileSync(sharedCatalogPath, JSON.stringify(buildCatalog(), null, 2), "utf8");
  return sharedCatalogPath;
}

// -----------------------------------------------------------------------------
// 8) Decision context / state feature extraction
// -----------------------------------------------------------------------------
function selectPool(state, actor, options = {}) {
  const includeCardLists = options.includeCardLists !== false;
  if (state.phase === "playing" && state.currentTurn === actor) {
    const handCards = (state.players?.[actor]?.hand || []).map((c) => c.id);
    const pool = {
      decisionType: "play",
      candidateCount: handCards.length,
      bombMonths: getDeclarableBombMonths(state, actor),
      shakingMonths: getDeclarableShakingMonths(state, actor)
    };
    if (includeCardLists) {
      pool.cards = handCards;
    } else {
      pool.cardCount = handCards.length;
    }
    return pool;
  }
  if (state.phase === "select-match" && state.pendingMatch?.playerKey === actor) {
    const boardCardIds = state.pendingMatch.boardCardIds || [];
    return {
      decisionType: "match",
      candidateCount: boardCardIds.length,
      ...(includeCardLists ? { boardCardIds } : {})
    };
  }
  if (state.phase === "go-stop" && state.pendingGoStop === actor) {
    const optionsList = ["go", "stop"];
    return {
      decisionType: "option",
      candidateCount: optionsList.length,
      ...(includeCardLists ? { options: optionsList } : {})
    };
  }
  if (state.phase === "president-choice" && state.pendingPresident?.playerKey === actor) {
    const optionsList = ["president_stop", "president_hold"];
    return {
      decisionType: "option",
      candidateCount: optionsList.length,
      ...(includeCardLists ? { options: optionsList } : {})
    };
  }
  if (state.phase === "gukjin-choice" && state.pendingGukjinChoice?.playerKey === actor) {
    const optionsList = ["five", "junk"];
    return {
      decisionType: "option",
      candidateCount: optionsList.length,
      ...(includeCardLists ? { options: optionsList } : {})
    };
  }
  if (state.phase === "shaking-confirm" && state.pendingShakingConfirm?.playerKey === actor) {
    const optionsList = ["shaking_yes", "shaking_no"];
    return {
      decisionType: "option",
      candidateCount: optionsList.length,
      ...(includeCardLists ? { options: optionsList } : {})
    };
  }
  return {};
}

function clamp01(value) {
  if (!Number.isFinite(value)) return 0;
  if (value <= 0) return 0;
  if (value >= 1) return 1;
  return value;
}

function clampRange(value, low, high) {
  if (!Number.isFinite(value)) return 0;
  if (value < low) return low;
  if (value > high) return high;
  return value;
}

function flattenCapturedIds(captured) {
  return [
    ...(captured?.kwang || []),
    ...(captured?.five || []),
    ...(captured?.ribbon || []),
    ...(captured?.junk || [])
  ]
    .map((c) => c?.id)
    .filter((id) => typeof id === "string");
}

function countComboMonths(months, ownSet) {
  let n = 0;
  for (const m of months) {
    if (ownSet.has(m)) n += 1;
  }
  return n;
}

function buildJokboStats(captured) {
  const ribbons = captured?.ribbon || [];
  const fives = captured?.five || [];
  const gwangs = captured?.kwang || [];
  const ribbonMonths = new Set(ribbons.map((c) => c?.month).filter((m) => Number.isInteger(m)));
  const fiveMonths = new Set(fives.map((c) => c?.month).filter((m) => Number.isInteger(m)));
  const gwangMonths = new Set(gwangs.map((c) => c?.month).filter((m) => Number.isInteger(m)));
  return {
    ribbonMonths,
    fiveMonths,
    gwangMonths,
    progress: {
      hongdan: countComboMonths(RIBBON_COMBO_MONTH_SETS.hongdan, ribbonMonths),
      cheongdan: countComboMonths(RIBBON_COMBO_MONTH_SETS.cheongdan, ribbonMonths),
      chodan: countComboMonths(RIBBON_COMBO_MONTH_SETS.chodan, ribbonMonths),
      godori: countComboMonths(GODORI_MONTHS, fiveMonths),
      gwang: gwangMonths.size
    }
  };
}

function buildSelfBlockCounts(handCards, jokboSelfStats) {
  const hand = Array.isArray(handCards) ? handCards : [];
  const handRibbonMonths = new Set(
    hand.filter((c) => c?.category === "ribbon").map((c) => c?.month).filter(Number.isInteger)
  );
  const handFiveMonths = new Set(
    hand.filter((c) => c?.category === "five").map((c) => c?.month).filter(Number.isInteger)
  );
  const handGwang = hand.filter((c) => c?.category === "kwang").length;
  return {
    hongdan:
      countComboMonths(RIBBON_COMBO_MONTH_SETS.hongdan, handRibbonMonths) +
      Number(jokboSelfStats?.progress?.hongdan || 0),
    cheongdan:
      countComboMonths(RIBBON_COMBO_MONTH_SETS.cheongdan, handRibbonMonths) +
      Number(jokboSelfStats?.progress?.cheongdan || 0),
    chodan:
      countComboMonths(RIBBON_COMBO_MONTH_SETS.chodan, handRibbonMonths) +
      Number(jokboSelfStats?.progress?.chodan || 0),
    godori:
      countComboMonths(GODORI_MONTHS, handFiveMonths) +
      Number(jokboSelfStats?.progress?.godori || 0),
    gwang: handGwang + Number(jokboSelfStats?.progress?.gwang || 0)
  };
}

function buildVisibleCardIdSet(state, actor, capturedSelf, capturedOpp) {
  const visible = new Set();
  for (const c of state.players?.[actor]?.hand || []) {
    if (c?.id) visible.add(c.id);
  }
  for (const c of state.board || []) {
    if (c?.id) visible.add(c.id);
  }
  for (const id of flattenCapturedIds(capturedSelf)) visible.add(id);
  for (const id of flattenCapturedIds(capturedOpp)) visible.add(id);
  return visible;
}

function unknownMonthCategoryProbability(visibleCardIds, unknownPool, month, category) {
  if (unknownPool <= 0 || !Number.isInteger(month)) return 0;
  let available = 0;
  for (const c of FULL_DECK) {
    if (c?.month !== month) continue;
    if (category && c?.category !== category) continue;
    if (visibleCardIds.has(c?.id)) continue;
    available += 1;
  }
  return clamp01(available / unknownPool);
}

function unknownAnyMonthProbability(visibleCardIds, unknownPool, months, category) {
  if (!Array.isArray(months) || !months.length) return 0;
  let noneProb = 1;
  for (const month of months) {
    const p = unknownMonthCategoryProbability(visibleCardIds, unknownPool, month, category);
    noneProb *= 1 - p;
  }
  return clamp01(1 - noneProb);
}

function computeJokboThreatProbabilities(stats, visibleCardIds, unknownPool) {
  const rules = [
    { months: RIBBON_COMBO_MONTH_SETS.hongdan, set: stats.ribbonMonths, category: "ribbon" },
    { months: RIBBON_COMBO_MONTH_SETS.cheongdan, set: stats.ribbonMonths, category: "ribbon" },
    { months: RIBBON_COMBO_MONTH_SETS.chodan, set: stats.ribbonMonths, category: "ribbon" },
    { months: GODORI_MONTHS, set: stats.fiveMonths, category: "five" }
  ];
  let oneAwayProb = 0;
  let totalProb = 0;
  for (const r of rules) {
    const got = countComboMonths(r.months, r.set);
    const missing = r.months.filter((m) => !r.set.has(m));
    if (!missing.length) {
      oneAwayProb = 1;
      totalProb += 0.9;
      continue;
    }
    const pAny = unknownAnyMonthProbability(visibleCardIds, unknownPool, missing, r.category);
    if (got >= 2) {
      oneAwayProb = Math.max(oneAwayProb, pAny);
      totalProb += pAny * 0.72;
    } else if (got === 1) {
      totalProb += pAny * 0.24;
    }
  }

  const gotGwang = Number(stats?.progress?.gwang || 0);
  const missingGwang = GWANG_MONTHS.filter((m) => !stats.gwangMonths.has(m));
  let gwangThreatProb = 0;
  if (gotGwang >= 2) {
    gwangThreatProb = unknownAnyMonthProbability(visibleCardIds, unknownPool, missingGwang, "kwang");
  } else if (gotGwang === 1) {
    gwangThreatProb =
      unknownAnyMonthProbability(visibleCardIds, unknownPool, missingGwang, "kwang") * 0.3;
  }
  totalProb += gwangThreatProb * 0.6;
  return {
    oneAwayProb: clamp01(oneAwayProb),
    gwangThreatProb: clamp01(gwangThreatProb),
    totalProb: clamp01(totalProb)
  };
}

function decisionContext(state, actor, options = {}) {
  const includeCardLists = options.includeCardLists === true;
  const opp = findOppActor(state, actor);
  if (!opp) {
    throw new Error(`failed to resolve opponent actor for actor=${actor}`);
  }
  const capturedSelf = state.players?.[actor]?.captured || {};
  const capturedOpp = state.players?.[opp]?.captured || {};
  const handSelfCards = state.players?.[actor]?.hand || [];
  const jokboSelfStats = buildJokboStats(capturedSelf);
  const jokboOppStats = buildJokboStats(capturedOpp);
  const comboBlockSelf = buildSelfBlockCounts(handSelfCards, jokboSelfStats);
  const unknownPool = (state.deck?.length || 0) + (state.players?.[opp]?.hand?.length || 0);
  const visibleCardIds = buildVisibleCardIdSet(state, actor, capturedSelf, capturedOpp);
  const selfThreat = computeJokboThreatProbabilities(jokboSelfStats, visibleCardIds, unknownPool);
  const oppThreat = computeJokboThreatProbabilities(jokboOppStats, visibleCardIds, unknownPool);
  const scoreSelf = calculateScore(state.players[actor], state.players[opp], state.ruleKey);
  const scoreOpp = calculateScore(state.players[opp], state.players[actor], state.ruleKey);
  const jokboProgressSelf = jokboSelfStats.progress;
  const jokboProgressOpp = jokboOppStats.progress;
  const context = {
    phase: state.phase,
    turnNoBefore: (state.turnSeq || 0) + 1,
    deckCount: state.deck.length,
    handCountSelf: state.players[actor].hand.length,
    handCountOpp: state.players[opp].hand.length,
    goCountSelf: state.players[actor].goCount || 0,
    goCountOpp: state.players[opp].goCount || 0,
    shakeCountSelf: state.players?.[actor]?.events?.shaking || 0,
    shakeCountOpp: state.players?.[opp]?.events?.shaking || 0,
    bombCountSelf: state.players?.[actor]?.events?.bomb || 0,
    bombCountOpp: state.players?.[opp]?.events?.bomb || 0,
    carryOverMultiplier: Number(state.carryOverMultiplier || 1),
    isFirstAttacker: state.startingTurnKey === actor ? 1 : 0,
    goldSelf: state.players[actor].gold,
    goldOpp: state.players[opp].gold,
    currentScoreSelf: Number(scoreSelf?.total || 0),
    currentScoreOpp: Number(scoreOpp?.total || 0),
    piBakRisk: scoreSelf?.bak?.pi ? 1 : 0,
    gwangBakRisk: scoreSelf?.bak?.gwang ? 1 : 0,
    mongBakRisk: scoreSelf?.bak?.mongBak ? 1 : 0,
    jokboProgressSelf,
    jokboProgressOpp,
    jokboProgressSelfSum: jokboProgressMagnitude(jokboProgressSelf),
    jokboProgressOppSum: jokboProgressMagnitude(jokboProgressOpp),
    jokboOneAwaySelfCount: jokboOneAwayCount(jokboProgressSelf),
    jokboOneAwayOppCount: jokboOneAwayCount(jokboProgressOpp),
    comboBlockSelf,
    selfJokboThreatProb: clamp01(selfThreat.totalProb),
    selfJokboOneAwayProb: clamp01(selfThreat.oneAwayProb),
    selfGwangThreatProb: clamp01(selfThreat.gwangThreatProb),
    oppJokboThreatProb: clamp01(oppThreat.totalProb),
    oppJokboOneAwayProb: clamp01(oppThreat.oneAwayProb),
    oppGwangThreatProb: clamp01(oppThreat.gwangThreatProb),
    goStopDeltaProxy: 0
  };
  context.goStopDeltaProxy = goStopDeltaProxy(context);
  if (includeCardLists) {
    context.handCards = (state.players?.[actor]?.hand || []).map((c) => c.id);
    context.boardCards = (state.board || []).map((c) => c.id);
    context.capturedCardsSelf = flattenCapturedIds(capturedSelf);
    context.capturedCardsOpp = flattenCapturedIds(capturedOpp);
  }
  return context;
}

function jokboCompleted(progress) {
  if (!progress) return false;
  return (
    Number(progress.hongdan || 0) >= 3 ||
    Number(progress.cheongdan || 0) >= 3 ||
    Number(progress.chodan || 0) >= 3 ||
    Number(progress.godori || 0) >= 3
  );
}

function stableHash(token, dim) {
  const digest = crypto.createHash("md5").update(token, "utf8").digest("hex");
  return Number.parseInt(digest.slice(0, 8), 16) % dim;
}

function stableHashCached(token, dim) {
  const key = `${dim}|${token}`;
  const cached = hashIndexCache.get(key);
  if (cached != null) return cached;
  const idx = stableHash(token, dim);
  setBoundedCache(hashIndexCache, key, idx, HASH_CACHE_MAX);
  return idx;
}

function policyContextKeyMode(policyModel) {
  const raw = String(policyModel?.context_key_mode || "").trim().toLowerCase();
  if (raw === "bucketed_v3") return "bucketed_v3";
  if (raw === "bucketed_v2") return "bucketed_v2";
  return "raw_v1";
}

function bucketHandSelf(handSelf) {
  const hs = Math.floor(Number(handSelf || 0));
  if (hs >= 8) return "8p";
  if (hs >= 5) return "5_7";
  if (hs >= 2) return "2_4";
  return "0_1";
}

function bucketHandDiff(handDiff) {
  const hd = Math.floor(Number(handDiff || 0));
  if (hd <= -3) return "n3";
  if (hd <= -1) return "n2_1";
  if (hd === 0) return "z0";
  if (hd <= 2) return "p1_2";
  return "p3";
}

function bucketScoreDiff(scoreDiff) {
  const sd = Number(scoreDiff || 0);
  if (sd <= -10) return "n10";
  if (sd <= -3) return "n9_3";
  if (sd < 3) return "z2";
  if (sd < 10) return "p3_9";
  return "p10";
}

function bucketGoCount(goCount) {
  const g = Math.floor(Number(goCount || 0));
  if (g <= 0) return "0";
  if (g === 1) return "1";
  return "2p";
}

function bucketCandidates(cands) {
  const c = Math.floor(Number(cands || 0));
  if (c <= 1) return "1";
  if (c === 2) return "2";
  if (c === 3) return "3";
  return "4p";
}

function bucketRiskTotal(total) {
  const t = Math.max(0, Math.floor(Number(total || 0)));
  if (t <= 0) return "0";
  if (t === 1) return "1";
  return "2p";
}

function bucketThreatPercent(v) {
  const x = Math.max(0, Math.min(100, Math.floor(Number(v || 0))));
  if (x >= 70) return "h";
  if (x >= 35) return "m";
  return "l";
}

function bucketProgressDelta(delta) {
  const d = Math.floor(Number(delta || 0));
  if (d <= -3) return "n3";
  if (d <= -1) return "n2_1";
  if (d === 0) return "z0";
  if (d <= 2) return "p1_2";
  return "p3";
}

function bucketGoStopSignal(gsdPermille) {
  const x = Math.floor(Number(gsdPermille || 0));
  if (x <= -1800) return "n2";
  if (x <= -600) return "n1";
  if (x < 600) return "z0";
  if (x < 1800) return "p1";
  return "p2";
}

function policyContextKey(trace, decisionType, policyModel = null) {
  const dc = trace.dc || {};
  const sp = trace.sp || {};
  const deckBucket = Math.floor((Number(dc.d || 0) || 0) / 3);
  const rawPhase = dc.p;
  const phaseCode = Number.isFinite(Number(rawPhase))
    ? Math.floor(Number(rawPhase))
    : tracePhaseCode(rawPhase);
  const handSelf = Number(dc.hs || 0) || 0;
  const handDiff = Number(dc.hd || 0) || 0;
  const goSelf = Number(dc.gs || 0) || 0;
  const goOpp = Number(dc.go || 0) || 0;
  const shakeSelf = Math.min(3, Math.floor(Number(dc.ss ?? 0)));
  const shakeOpp = Math.min(3, Math.floor(Number(dc.so ?? 0)));
  const scoreDiff = Number(dc.sd || 0) || 0;
  const bakRiskTotal =
    (Number(dc.rp || 0) ? 1 : 0) +
    (Number(dc.rg || 0) ? 1 : 0) +
    (Number(dc.rm || 0) ? 1 : 0);
  const oppThreat = Math.max(
    Number(dc.ojt || 0) || 0,
    Number(dc.ojo || 0) || 0,
    Number(dc.ogt || 0) || 0
  );
  const selfThreat = Math.max(
    Number(dc.sjt || 0) || 0,
    Number(dc.sjo || 0) || 0,
    Number(dc.sgt || 0) || 0
  );
  const progressDelta = (Number(dc.jps || 0) || 0) - (Number(dc.jpo || 0) || 0);
  const goStopSignal = Number(dc.gsd || 0) || 0;
  let cands = Number(trace.cc ?? sp.candidateCount ?? 0);
  if (!Number.isFinite(cands) || cands <= 0) {
    cands = (sp.cards || sp.boardCardIds || sp.options || []).length;
  }
  cands = Math.max(0, Math.floor(cands));
  const keyMode = policyContextKeyMode(policyModel);
  const hsToken = keyMode === "bucketed_v2" ? bucketHandSelf(handSelf) : String(Math.floor(Number(handSelf || 0)));
  const hdToken = keyMode === "bucketed_v2" ? bucketHandDiff(handDiff) : String(Math.floor(Number(handDiff || 0)));
  const sdToken = keyMode === "bucketed_v2" ? bucketScoreDiff(scoreDiff) : String(Math.floor(scoreDiff));
  const gsToken = keyMode === "bucketed_v2" ? bucketGoCount(goSelf) : String(Math.floor(Number(goSelf || 0)));
  const goToken = keyMode === "bucketed_v2" ? bucketGoCount(goOpp) : String(Math.floor(Number(goOpp || 0)));
  const ccToken = keyMode === "bucketed_v2" ? bucketCandidates(cands) : String(cands);
  const base = [
    `dt=${decisionType}`,
    `ph=${phaseCode}`,
    `o=${trace.o || "?"}`,
    `db=${deckBucket}`,
    `hs=${hsToken}`,
    `hd=${hdToken}`,
    `sd=${sdToken}`,
    `gs=${gsToken}`,
    `go=${goToken}`,
    `ss=${shakeSelf}`,
    `so=${shakeOpp}`,
    `cc=${ccToken}`
  ];
  if (keyMode === "bucketed_v3") {
    base.push(`br=${bucketRiskTotal(bakRiskTotal)}`);
    base.push(`ot=${bucketThreatPercent(oppThreat)}`);
    base.push(`st=${bucketThreatPercent(selfThreat)}`);
    base.push(`jp=${bucketProgressDelta(progressDelta)}`);
    base.push(`gd=${bucketGoStopSignal(goStopSignal)}`);
  }
  return base.join("|");
}

// -----------------------------------------------------------------------------
// 9) Lightweight policy/value inference helpers
// -----------------------------------------------------------------------------
function policyProb(model, sample, choice) {
  const alpha = Number(model?.alpha ?? 1.0);
  const dt = sample.decisionType;
  const candidates = sample.candidates || [];
  const contextKey = sample.contextKey;
  const k = Math.max(1, candidates.length);

  const dtContextCounts = model?.context_counts?.[dt] || {};
  const dtContextTotals = model?.context_totals?.[dt] || {};
  const ctxCounts = dtContextCounts?.[contextKey];
  if (ctxCounts) {
    const total = Number(dtContextTotals?.[contextKey] || 0);
    return (Number(ctxCounts?.[choice] || 0) + alpha) / (total + alpha * k);
  }

  const dtGlobal = model?.global_counts?.[dt] || {};
  let total = 0;
  for (const c of candidates) total += Number(dtGlobal?.[c] || 0);
  return (Number(dtGlobal?.[choice] || 0) + alpha) / (total + alpha * k);
}

function modelCacheKey(actor, cfg, decisionType, contextKey, candidates) {
  const p = cfg.policyModelPath || "-";
  return `${actor}|${p}|${decisionType}|${contextKey}|${candidates.join(",")}`;
}

function optionActionToType(action) {
  const aliases = {
    go: "choose_go",
    stop: "choose_stop",
    shaking_yes: "choose_shaking_yes",
    shaking_no: "choose_shaking_no",
    president_stop: "choose_president_stop",
    president_hold: "choose_president_hold"
  };
  return aliases[action] || action;
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
    choose_junk: "junk"
  };
  return aliases[a] || a;
}

function normalizeOptionCandidates(items) {
  if (!Array.isArray(items)) return [];
  const out = [];
  const seen = new Set();
  for (const raw of items) {
    const v = canonicalOptionAction(raw);
    if (!v) continue;
    if (seen.has(v)) continue;
    seen.add(v);
    out.push(v);
  }
  return out;
}

function modelOptionActionSet(policyModel) {
  const raw = policyModel?.action_vocab?.option_actions;
  if (!Array.isArray(raw) || raw.length === 0) return null;
  const normalized = normalizeOptionCandidates(raw);
  if (!normalized.length) return null;
  return new Set(normalized);
}

function modelDecisionTypesSet(policyModel) {
  const raw = policyModel?.action_vocab?.decision_types;
  if (!Array.isArray(raw) || raw.length === 0) return null;
  const normalized = raw
    .map((x) => String(x || "").trim().toLowerCase())
    .filter((x) => x === "play" || x === "match" || x === "option");
  if (!normalized.length) return null;
  return new Set(normalized);
}

function legalCandidatesForInference(selectionPool, decisionType, policyModel) {
  const sp = selectionPool || {};
  const allowedDecisionTypes = modelDecisionTypesSet(policyModel);
  if (allowedDecisionTypes && !allowedDecisionTypes.has(String(decisionType || "").toLowerCase())) {
    return [];
  }
  if (decisionType === "play") {
    return normalizeLegalActions(sp.cards || [], "play");
  }
  if (decisionType === "match") {
    return normalizeLegalActions(sp.boardCardIds || [], "match");
  }
  if (decisionType === "option") {
    const legal = normalizeOptionCandidates(sp.options || []);
    const allowed = modelOptionActionSet(policyModel);
    if (!allowed) return legal;
    const filtered = legal.filter((x) => allowed.has(x));
    return filtered.length > 0 ? filtered : legal;
  }
  return [];
}

function actionTypeToChoice(actionType) {
  const mapped = optionActionToType(actionType);
  return mapped || String(actionType || "option");
}

function traceOrderFromSide(actorSide) {
  return actorSide === SIDE_MY ? "first" : "second";
}

function cloneStateForRollout(state) {
  if (typeof globalThis.structuredClone === "function") {
    return globalThis.structuredClone(state);
  }
  return JSON.parse(JSON.stringify(state));
}

function evaluateGoldDeltaFromMySide(state) {
  const myActor = state?.startingTurnKey || null;
  const yourActor = secondActorFromFirst(state, myActor);
  if (!myActor || !yourActor) return 0;
  const myGold = Number(state?.players?.[myActor]?.gold || 0);
  const yourGold = Number(state?.players?.[yourActor]?.gold || 0);
  return myGold - yourGold;
}

function buildRuntimeModelConfig(state, actor, cfg) {
  const base = cfg || {};
  return {
    ...base,
    mixedStrategyEnabled:
      base?.mixedStrategyEnabled != null
        ? !!base.mixedStrategyEnabled
        : MIXED_STRATEGY_ENABLED_DEFAULT,
    optionLookaheadEnabled:
      base?.optionLookaheadEnabled != null
        ? !!base.optionLookaheadEnabled
        : OPTION_LOOKAHEAD_ENABLED_DEFAULT,
    optionLookaheadRollouts: Math.max(
      1,
      Math.floor(Number(base?.optionLookaheadRollouts || OPTION_LOOKAHEAD_ROLLOUTS_DEFAULT))
    ),
    optionLookaheadMaxSteps: Math.max(
      20,
      Math.floor(Number(base?.optionLookaheadMaxSteps || OPTION_LOOKAHEAD_MAX_STEPS_DEFAULT))
    ),
  };
}

function rolloutOptionCandidateScore(baseState, actor, cfg, candidate, control = {}) {
  const rollouts = Math.max(1, Number(cfg?.optionLookaheadRollouts || OPTION_LOOKAHEAD_ROLLOUTS_DEFAULT));
  const maxSteps = Math.max(20, Number(cfg?.optionLookaheadMaxSteps || OPTION_LOOKAHEAD_MAX_STEPS_DEFAULT));
  const outcomes = [];
  for (let ri = 0; ri < rollouts; ri += 1) {
    let sim = cloneStateForRollout(baseState);
    sim = applyModelChoice(sim, actor, { decisionType: "option", candidate }, cfg);
    if (!sim) continue;
    let steps = 0;
    while (sim.phase !== "resolution" && steps < maxSteps) {
      const actionActor = getActionPlayerKey(sim);
      if (!actionActor) break;
      const actionSide = actorToSide(actionActor, sim.startingTurnKey);
      const actionCfg = sideConfig[actionSide];
      const next = executeActorTurn(sim, actionActor, actionCfg, {
        ...control,
        disableOptionLookahead: true,
        rolloutNonce: `${control?.rolloutNonce || "r"}:${ri}`,
      });
      if (!next || next === sim) break;
      sim = next;
      steps += 1;
    }
    outcomes.push(evaluateGoldDeltaFromMySide(sim));
  }
  if (!outcomes.length) {
    return { mean: -Infinity, p10: -Infinity, lossRate: 1, score: -Infinity };
  }
  outcomes.sort((a, b) => a - b);
  const n = outcomes.length;
  const p10 = outcomes[Math.max(0, Math.floor((n - 1) * 0.1))];
  const mean = outcomes.reduce((a, b) => a + b, 0) / n;
  const lossRate = outcomes.filter((x) => x < 0).length / n;
  const downside = Math.max(0, -p10) + Math.max(0, -mean) * lossRate;
  const score = mean - OPTION_LOOKAHEAD_DOWNSIDE_LAMBDA * downside;
  return { mean, p10, lossRate, score };
}

function modelSelectCandidate(state, actor, cfg, control = {}) {
  const sp = selectPool(state, actor);
  const cards = Array.isArray(sp.cards) && sp.cards.length > 0 ? sp.cards : null;
  const boardCardIds = Array.isArray(sp.boardCardIds) && sp.boardCardIds.length > 0 ? sp.boardCardIds : null;
  const options = Array.isArray(sp.options) && sp.options.length > 0 ? sp.options : null;
  const decisionType = cards ? "play" : boardCardIds ? "match" : options ? "option" : null;
  if (!decisionType) return null;
  const candidates = legalCandidatesForInference(sp, decisionType, cfg?.policyModel);
  if (!candidates.length) return null;
  const order = actor === state.startingTurnKey ? "first" : "second";
  const actorSide = actorToSide(actor, state.startingTurnKey);
  const dc = decisionContext(state, actor);
  const traceLike = {
    o: order,
    dc: compactDecisionContextForTrace(dc, actorSide),
    sp: { cards, boardCardIds, options }
  };
  const contextKey = policyContextKey(traceLike, decisionType, cfg.policyModel);
  const cacheKey = modelCacheKey(actor, cfg, decisionType, contextKey, candidates);
  const cached = decisionInferenceCache.get(cacheKey);
  if (cached) return { decisionType, candidate: cached, sp };

  const baseSample = { decisionType, candidates, contextKey };
  const policyScoreByCandidate = new Map();
  for (const candidate of candidates) {
    const candidateLabel = decisionType === "option" ? candidate : String(candidate);
    const pp = cfg.policyModel ? policyProb(cfg.policyModel, baseSample, candidateLabel) : 0;
    policyScoreByCandidate.set(candidate, Math.log(Math.max(1e-12, Number(pp || 0))));
  }

  const shouldRunOptionLookahead =
    decisionType === "option" &&
    candidates.length > 1 &&
    !!cfg?.optionLookaheadEnabled &&
    !control?.disableOptionLookahead &&
    (
      (candidates.includes("go") && candidates.includes("stop")) ||
      (candidates.includes("shaking_yes") && candidates.includes("shaking_no")) ||
      (candidates.includes("president_stop") && candidates.includes("president_hold"))
    );
  const combinedScoreByCandidate = new Map(policyScoreByCandidate);
  if (shouldRunOptionLookahead) {
    for (const candidate of candidates) {
      const rollout = rolloutOptionCandidateScore(state, actor, cfg, candidate, control);
      const policyTerm = Number(policyScoreByCandidate.get(candidate) || 0);
      const rolloutTerm = Number(rollout.score || 0) / 1000000.0;
      combinedScoreByCandidate.set(candidate, policyTerm + rolloutTerm);
    }
  }

  const mixedEnabled = !!cfg?.mixedStrategyEnabled && !control?.disableMixedStrategy;
  let bestCandidate = candidates[0];
  if (mixedEnabled) {
    const topK =
      decisionType === "play" ? MIXED_TOP_K_PLAY : decisionType === "match" ? MIXED_TOP_K_MATCH : MIXED_TOP_K_OPTION;
    const temp =
      decisionType === "play"
        ? MIXED_TEMPERATURE_PLAY
        : decisionType === "match"
        ? MIXED_TEMPERATURE_MATCH
        : MIXED_TEMPERATURE_OPTION;
    const entropyKey = [
      runId,
      control?.rolloutNonce || "main",
      actor,
      state.turnSeq || 0,
      decisionType,
      contextKey,
      candidates.join(","),
    ].join("|");
    bestCandidate = softmaxSampleFromScores(
      candidates,
      combinedScoreByCandidate,
      topK,
      temp,
      entropyKey
    );
  } else {
    let bestScore = -Infinity;
    for (const candidate of candidates) {
      const s = Number(combinedScoreByCandidate.get(candidate) || -Infinity);
      if (s > bestScore) {
        bestCandidate = candidate;
        bestScore = s;
      }
    }
  }

  if (decisionType === "option") {
    const hasGo = candidates.includes("go");
    const hasStop = candidates.includes("stop");
    if (hasGo && hasStop && bestCandidate === "go") {
      const scoreDiff = Number(dc.currentScoreSelf || 0) - Number(dc.currentScoreOpp || 0);
      let threshold = GO_STOP_THRESHOLD_BASE_PERMILLE;
      if (scoreDiff > 0) threshold = GO_STOP_THRESHOLD_AHEAD_PERMILLE;
      else if (scoreDiff < 0) threshold = GO_STOP_THRESHOLD_BEHIND_PERMILLE;
      const goStopPermille = Math.round(Number(dc.goStopDeltaProxy || 0) * 1000);
      if (goStopPermille < threshold) {
        bestCandidate = "stop";
      }
    }
  }
  setBoundedCache(decisionInferenceCache, cacheKey, bestCandidate, DECISION_CACHE_MAX);
  return { decisionType, candidate: bestCandidate, sp };
}

function applyModelChoice(state, actor, picked, cfg = null) {
  if (!picked) return state;
  const sp = selectPool(state, actor);
  const cards = Array.isArray(sp.cards) && sp.cards.length > 0 ? sp.cards : null;
  const boardCardIds = Array.isArray(sp.boardCardIds) && sp.boardCardIds.length > 0 ? sp.boardCardIds : null;
  const options = Array.isArray(sp.options) && sp.options.length > 0 ? sp.options : null;
  const decisionType = cards ? "play" : boardCardIds ? "match" : options ? "option" : null;
  if (!decisionType) return state;
  const legal = legalCandidatesForInference(sp, decisionType, cfg?.policyModel || null);
  if (!legal.length) return state;

  let c = picked.candidate;
  if (decisionType === "option") c = canonicalOptionAction(c);
  if (!legal.includes(c)) c = legal[0];

  if (decisionType === "play") return playTurn(state, c);
  if (decisionType === "match") return chooseMatch(state, c);
  if (decisionType !== "option") return state;

  if (c === "go") return chooseGo(state, actor);
  if (c === "stop") return chooseStop(state, actor);
  if (c === "shaking_yes") return chooseShakingYes(state, actor);
  if (c === "shaking_no") return chooseShakingNo(state, actor);
  if (c === "president_stop") return choosePresidentStop(state, actor);
  if (c === "president_hold") return choosePresidentHold(state, actor);
  if (c === "five" || c === "junk") return chooseGukjinMode(state, actor, c);
  return state;
}

function maybePickExplorationChoice(state, actor, cfg) {
  return null;
}

function loadJsonModel(modelPath, label) {
  if (!modelPath) return null;
  const full = path.resolve(modelPath);
  if (!fs.existsSync(full)) {
    throw new Error(`${label} not found: ${modelPath}`);
  }
  return JSON.parse(fs.readFileSync(full, "utf8"));
}

// -----------------------------------------------------------------------------
// 10) Trace shaping and trigger extraction
// -----------------------------------------------------------------------------
function countCandidates(selectionPool) {
  if (!selectionPool) return 0;
  const explicit = Number(selectionPool.candidateCount || 0);
  if (Number.isFinite(explicit) && explicit > 0) return Math.floor(explicit);
  if (Array.isArray(selectionPool.options)) return selectionPool.options.length;
  if (Array.isArray(selectionPool.boardCardIds)) return selectionPool.boardCardIds.length;
  if (Array.isArray(selectionPool.cards)) return selectionPool.cards.length;
  return Number(selectionPool.cardCount || 0);
}

function traceDecisionType(selectionPool, actionType) {
  const dt = String(selectionPool?.decisionType || "");
  if (dt === "play" || dt === "match" || dt === "option") return dt;
  if (Array.isArray(selectionPool?.options)) return "option";
  if (Array.isArray(selectionPool?.boardCardIds)) return "match";
  if (Array.isArray(selectionPool?.cards)) return "play";
  const action = String(actionType || "");
  if (action.startsWith("choose_")) return "option";
  if (action === "choose_match" || action === "select_match") return "match";
  return "play";
}

function traceCandidateCount(selectionPool) {
  return Math.max(0, Math.floor(Number(countCandidates(selectionPool) || 0)));
}

function normalizeLegalActions(items, decisionType) {
  if (!Array.isArray(items)) return [];
  const out = [];
  const seen = new Set();
  for (const raw of items) {
    const v =
      decisionType === "option"
        ? actionAliasForOption(raw)
        : String(raw ?? "").trim();
    if (!v) continue;
    if (seen.has(v)) continue;
    seen.add(v);
    out.push(v);
  }
  return out;
}

function actionAliasForOption(raw) {
  const s = String(raw ?? "").trim();
  if (!s) return "";
  const mapped = optionActionToType(s);
  return String(mapped || s).trim();
}

function optionLegalActionsFromActionType(actionType) {
  const at = String(actionType || "").trim();
  if (at === "choose_go" || at === "choose_stop") return ["go", "stop"];
  if (at === "choose_shaking_yes" || at === "choose_shaking_no") return ["shaking_yes", "shaking_no"];
  if (at === "choose_president_stop" || at === "choose_president_hold")
    return ["president_stop", "president_hold"];
  if (at === "five" || at === "junk") return ["five", "junk"];
  return [];
}

function traceLegalActions(selectionPool, decisionType) {
  const sp = selectionPool || {};
  if (decisionType === "play") return normalizeLegalActions(sp.cards || [], "play");
  if (decisionType === "match") return normalizeLegalActions(sp.boardCardIds || [], "match");
  if (decisionType === "option") return normalizeLegalActions(sp.options || [], "option");
  return [];
}

function inferOptionActionType({
  beforeState,
  nextState,
  actor,
  selectionPool,
  prevKiboSeq
}) {
  const phase = String(beforeState?.phase || "");
  const nextKiboSeq = Number(nextState?.kiboSeq || 0);
  const nextKibo = Array.isArray(nextState?.kibo) ? nextState.kibo : [];
  const added = nextKiboSeq > prevKiboSeq ? nextKibo.slice(prevKiboSeq, nextKiboSeq) : [];
  for (let i = added.length - 1; i >= 0; i -= 1) {
    const type = String(added[i]?.type || "");
    if (type === "go") return "choose_go";
    if (type === "stop") return "choose_stop";
    if (type === "president_stop") return "choose_president_stop";
    if (type === "president_hold") return "choose_president_hold";
    if (type === "gukjin_mode") return added[i]?.mode === "junk" ? "junk" : "five";
    if (type === "shaking_declare") return "choose_shaking_yes";
  }

  if (phase === "go-stop") {
    if (nextState?.phase === "resolution" || nextState?.players?.[actor]?.declaredStop) {
      return "choose_stop";
    }
    return "choose_go";
  }
  if (phase === "president-choice") {
    return nextState?.phase === "resolution" ? "choose_president_stop" : "choose_president_hold";
  }
  if (phase === "gukjin-choice") {
    const mode = String(nextState?.players?.[actor]?.gukjinMode || "");
    if (mode === "junk" || mode === "five") return mode;
  }
  if (phase === "shaking-confirm") {
    const b = Number(beforeState?.players?.[actor]?.events?.shaking || 0);
    const a = Number(nextState?.players?.[actor]?.events?.shaking || 0);
    return a > b ? "choose_shaking_yes" : "choose_shaking_no";
  }

  const opts = Array.isArray(selectionPool?.options) ? selectionPool.options : [];
  return String(opts[0] || "option");
}

function tracePhaseCode(phase) {
  const map = {
    playing: 1,
    "select-match": 2,
    "go-stop": 3,
    "president-choice": 4,
    "gukjin-choice": 5,
    "shaking-confirm": 6,
    resolution: 7
  };
  return Number(map[phase] || 0);
}

function toPercentInt(v) {
  const n = Number(v || 0);
  if (!Number.isFinite(n) || n <= 0) return 0;
  if (n >= 1) return 100;
  return Math.round(n * 100);
}

function toSignedPermille(v, absMax = 3) {
  const n = Number(v || 0);
  if (!Number.isFinite(n)) return 0;
  const c = clampRange(n, -Math.abs(absMax), Math.abs(absMax));
  return Math.round(c * 1000);
}

function specialEventTags(turn) {
  const events = turn?.action?.matchEvents || [];
  const tags = new Set();
  for (const m of events) {
    const tag = String(m?.eventTag || "");
    if (!tag || tag === "NORMAL") continue;
    tags.add(tag);
  }
  return [...tags];
}

function hasBakRiskShift(beforeDc, afterDc) {
  const keys = ["piBakRisk", "gwangBakRisk", "mongBakRisk"];
  return keys.some((k) => Number(beforeDc?.[k] || 0) !== Number(afterDc?.[k] || 0));
}

function hasBombShift(beforeDc, afterDc) {
  const selfDelta = Math.abs(
    Number(afterDc?.bombCountSelf || 0) - Number(beforeDc?.bombCountSelf || 0)
  );
  const oppDelta = Math.abs(
    Number(afterDc?.bombCountOpp || 0) - Number(beforeDc?.bombCountOpp || 0)
  );
  return selfDelta >= 1 || oppDelta >= 1;
}

function bakRiskShiftDirection(beforeDc, afterDc) {
  const keys = ["piBakRisk", "gwangBakRisk", "mongBakRisk"];
  let beforeSum = 0;
  let afterSum = 0;
  for (const k of keys) {
    beforeSum += Number(beforeDc?.[k] || 0) ? 1 : 0;
    afterSum += Number(afterDc?.[k] || 0) ? 1 : 0;
  }
  if (afterSum > beforeSum) return "up";
  if (afterSum < beforeSum) return "down";
  return null;
}

function comboThreatMask(dc) {
  const opp = dc?.jokboProgressOpp || {};
  const block = dc?.comboBlockSelf || {};
  let mask = 0;
  if (Number(opp.hongdan || 0) >= 2 && Number(block.hongdan || 0) === 0) mask |= 1;
  if (Number(opp.cheongdan || 0) >= 2 && Number(block.cheongdan || 0) === 0) mask |= 2;
  if (Number(opp.chodan || 0) >= 2 && Number(block.chodan || 0) === 0) mask |= 4;
  if (Number(opp.godori || 0) >= 2 && Number(block.godori || 0) === 0) mask |= 8;
  return mask;
}

function gwangThreatLevel(dc) {
  const oppGwang = Number(dc?.jokboProgressOpp?.gwang || 0);
  const myGwangBlock = Number(dc?.comboBlockSelf?.gwang || 0);
  if (oppGwang >= 3 && myGwangBlock <= 1) return 2;
  if (oppGwang === 2 && myGwangBlock <= 2) return 1;
  return 0;
}

function comboThreatEntered(beforeDc, afterDc) {
  const beforeMask = comboThreatMask(beforeDc);
  const afterMask = comboThreatMask(afterDc);
  const comboEnter = (afterMask & ~beforeMask) !== 0;
  const beforeGwang = gwangThreatLevel(beforeDc);
  const afterGwang = gwangThreatLevel(afterDc);
  const gwangEnter = afterGwang > beforeGwang;
  return comboEnter || gwangEnter;
}

function pushTrigger(out, name, back, forward) {
  if (!Array.isArray(out) || !name) return;
  if (out.some((tr) => String(tr?.name || "") === name)) return;
  out.push({
    name,
    back: Math.max(0, Math.floor(Number(back || 0))),
    forward: Math.max(0, Math.floor(Number(forward || 0)))
  });
}

function applyTerminalContextTrigger(records) {
  if (!Array.isArray(records) || !records.length) return;
  const idx = Math.max(0, records.length - 2);
  const rec = records[idx];
  if (!rec) return;
  if (!Array.isArray(rec.triggers)) rec.triggers = [];
  pushTrigger(rec.triggers, "terminalContext", 0, 0);
}

function classifyImportantTurnTriggers({
  decisionType,
  turn,
  beforeDc,
  afterDc,
  earlyTurnForced,
  goStopPlus2 = false
}) {
  const out = [];
  const actionType = String(turn?.action?.type || "");
  const phase = String(beforeDc?.phase || "");

  if (earlyTurnForced) {
    out.push({ name: "earlyTurnForced", back: 0, forward: 0 });
  }

  if (decisionType === "option") {
    const isGoStop =
      phase === "go-stop" || actionType === "choose_go" || actionType === "choose_stop";
    if (isGoStop) {
      const isGo = actionType === "choose_go";
      pushTrigger(out, "goStopOption", 1, isGo && goStopPlus2 ? 2 : 1);
    } else if (actionType === "choose_shaking_yes") {
      pushTrigger(out, "shakingYesOption", 0, 1);
    } else {
      pushTrigger(out, "optionTurnOther", 0, 1);
    }
  }

  if (Number(beforeDc?.deckCount || 0) <= 0) {
    pushTrigger(out, "deckEmpty", 1, 0);
  }

  const eventTags = specialEventTags(turn);
  if (eventTags.length) {
    if (eventTags.includes("PPUK")) {
      pushTrigger(out, "specialEventPpuk", 1, 1);
    }
    if (eventTags.some((tag) => tag !== "PPUK")) {
      pushTrigger(out, "specialEventCore", 1, 0);
    }
  }

  const bombDeclared = actionType === "declare_bomb";
  const bombShifted = hasBombShift(beforeDc, afterDc);
  if (bombDeclared) {
    pushTrigger(out, "bombDeclare", 1, 1);
  } else if (bombShifted) {
    pushTrigger(out, "bombCountShift", 0, 1);
  }

  const riskDirection = bakRiskShiftDirection(beforeDc, afterDc);
  if (riskDirection === "up") {
    pushTrigger(out, "riskShiftUp", 1, 0);
  } else if (riskDirection === "down") {
    pushTrigger(out, "riskShiftDown", 0, 1);
  } else if (hasBakRiskShift(beforeDc, afterDc)) {
    pushTrigger(out, "riskShiftMixed", 0, 0);
  }

  if (comboThreatEntered(beforeDc, afterDc)) {
    pushTrigger(out, "comboThreatEnter", 0, 0);
  }

  return out;
}

function keepBySelf(name) {
  const trigger = String(name || "").trim();
  if (!trigger) return null;
  return { kind: "self", trigger };
}

function keepByFrom(anchorTurn, name) {
  const trigger = String(name || "").trim();
  const turn = Math.max(0, Math.floor(Number(anchorTurn || 0)));
  if (!trigger) return null;
  return { kind: "from", fromTurn: turn, trigger };
}

function keepByCtx(anchorTurn) {
  const turn = Math.max(0, Math.floor(Number(anchorTurn || 0)));
  return { kind: "context", fromTurn: turn };
}

function keepByIndexKey(tag) {
  if (!tag || typeof tag !== "object") return "";
  const kind = String(tag.kind || "").trim();
  if (!kind) return "";
  if (kind === "self") return `self|${String(tag.trigger || "").trim()}`;
  if (kind === "from")
    return `from|${Math.max(0, Math.floor(Number(tag.fromTurn || 0)))}|${String(tag.trigger || "").trim()}`;
  if (kind === "context") return `context|${Math.max(0, Math.floor(Number(tag.fromTurn || 0)))}`;
  return `${kind}|${JSON.stringify(tag)}`;
}

function keepBySortKey(tag) {
  if (!tag || typeof tag !== "object") return "";
  const kind = String(tag.kind || "");
  const trigger = String(tag.trigger || "");
  const turn = Math.max(0, Math.floor(Number(tag.fromTurn || 0)));
  return `${kind}|${turn}|${trigger}`;
}

function keepImportantWithContext(records, contextRadius = 0) {
  if (!Array.isArray(records) || !records.length) return [];
  const extraRadius = Math.max(0, Math.floor(Number(contextRadius || 0)));
  let anyTrigger = false;
  let keep = new Array(records.length).fill(false);
  const keepReasons = Array.from({ length: records.length }, () => new Map());
  for (let i = 0; i < records.length; i += 1) {
    const triggers = Array.isArray(records[i]?.triggers) ? records[i].triggers : [];
    if (!triggers.length) continue;
    anyTrigger = true;
    const anchorTurn = Number(records[i]?.data?.t || 0);
    for (const tr of triggers) {
      const triggerName = String(tr?.name || "");
      const back = Math.max(0, Math.floor(Number(tr?.back || 0)));
      const forward = Math.max(0, Math.floor(Number(tr?.forward || 0)));
      const lo = Math.max(0, i - back);
      const hi = Math.min(records.length - 1, i + forward);
      for (let j = lo; j <= hi; j += 1) {
        keep[j] = true;
        if (!triggerName) continue;
        if (j === i) {
          const tag = keepBySelf(triggerName);
          if (tag) keepReasons[j].set(keepByIndexKey(tag), tag);
        } else {
          const tag = keepByFrom(anchorTurn, triggerName);
          if (tag) keepReasons[j].set(keepByIndexKey(tag), tag);
        }
      }
    }
  }
  if (!anyTrigger) {
    return records.map((r) => ({ ...r.data, keepBy: [] }));
  }

  if (extraRadius > 0) {
    const expanded = keep.slice();
    for (let i = 0; i < keep.length; i += 1) {
      if (!keep[i]) continue;
      const anchorTurn = Number(records[i]?.data?.t || 0);
      const lo = Math.max(0, i - extraRadius);
      const hi = Math.min(keep.length - 1, i + extraRadius);
      for (let j = lo; j <= hi; j += 1) {
        expanded[j] = true;
        if (j !== i) {
          const tag = keepByCtx(anchorTurn);
          keepReasons[j].set(keepByIndexKey(tag), tag);
        }
      }
    }
    keep = expanded;
  }

  const out = [];
  for (let i = 0; i < records.length; i += 1) {
    if (!keep[i]) continue;
    out.push({
      ...records[i].data,
      keepBy: [...(keepReasons[i]?.values() || [])].sort((a, b) =>
        keepBySortKey(a).localeCompare(keepBySortKey(b))
      )
    });
  }
  return out;
}

function triggerNames(triggers) {
  const names = new Set();
  for (const tr of Array.isArray(triggers) ? triggers : []) {
    const name = String(tr?.name || "");
    if (!name) continue;
    names.add(name);
  }
  return [...names];
}

function handCountFromTurn(turn, key) {
  if (turn.handsCount?.[key] != null) return turn.handsCount[key];
  if (Array.isArray(turn.hands?.[key])) return turn.hands[key].length;
  return 0;
}

function boardCountFromTurn(turn) {
  if (turn.boardCount != null) return turn.boardCount;
  if (Array.isArray(turn.board)) return turn.board.length;
  return 0;
}

function jokboProgressMagnitude(progress) {
  if (!progress) return 0;
  return (
    Number(progress.hongdan || 0) +
    Number(progress.cheongdan || 0) +
    Number(progress.chodan || 0) +
    Number(progress.godori || 0) +
    Number(progress.gwang || 0)
  );
}

function jokboOneAwayCount(progress) {
  if (!progress) return 0;
  let n = 0;
  if (Number(progress.hongdan || 0) >= 2) n += 1;
  if (Number(progress.cheongdan || 0) >= 2) n += 1;
  if (Number(progress.chodan || 0) >= 2) n += 1;
  if (Number(progress.godori || 0) >= 2) n += 1;
  if (Number(progress.gwang || 0) >= 2) n += 1;
  return n;
}

function goStopDeltaProxy(dc) {
  if (!dc || dc.phase !== "go-stop") return 0;
  const scoreDiff = Number(dc.currentScoreSelf || 0) - Number(dc.currentScoreOpp || 0);
  const carry = Math.max(1, Number(dc.carryOverMultiplier || 1));
  const selfPress =
    Number(dc.selfJokboThreatProb || 0) * 0.75 +
    Number(dc.selfJokboOneAwayProb || 0) * 0.7 +
    Number(dc.selfGwangThreatProb || 0) * 0.55;
  const oppPress =
    Number(dc.oppJokboThreatProb || 0) * 0.95 +
    Number(dc.oppJokboOneAwayProb || 0) * 0.75 +
    Number(dc.oppGwangThreatProb || 0) * 0.6;
  const bakRisk =
    Number(dc.piBakRisk || 0) + Number(dc.gwangBakRisk || 0) + Number(dc.mongBakRisk || 0);
  const goValue = selfPress + Math.max(0, scoreDiff) * 0.07 + Math.max(0, carry - 1) * 0.14;
  const stopValue = oppPress + bakRisk * 0.32;
  return clampRange(goValue - stopValue, -3, 3);
}

function bakRiskShiftReward(beforeDc, afterDc, key, weight) {
  const b = Number(beforeDc?.[key] || 0);
  const a = Number(afterDc?.[key] || 0);
  if (a < b) return Math.abs(weight);
  if (a > b) return -Math.abs(weight);
  return 0;
}

function weightedImmediateReward(turn, jpCompletedNow, beforeDc, afterDc) {
  const t = turn || {};
  const action = t.action || {};
  const captureBySource = action.captureBySource || { hand: [], flip: [] };
  const captured = [...(captureBySource.hand || []), ...(captureBySource.flip || [])];
  let shortTerm = 0;
  let strategic = 0;

  const stolenPi = Number(t.steals?.pi || 0);
  const stolenGold = Number(t.steals?.gold || 0);
  shortTerm += Math.min(0.2, stolenPi * 0.1);
  shortTerm += Math.min(0.1, stolenGold / 1000);

  for (const c of captured) {
    if (!c) continue;
    if (c.category === "kwang") strategic += 0.12;
    else if (c.category === "five") strategic += 0.08;
    else if (c.category === "ribbon") strategic += 0.05;
    else if (c.category === "junk") {
      const pv = piLikeValue(c);
      shortTerm += pv >= 3 ? 0.1 : pv >= 2 ? 0.08 : 0.05;
    }
  }

  const b = beforeDc || {};
  const a = afterDc || {};
  const beforeJokbo = jokboProgressMagnitude(b.jokboProgressSelf);
  const afterJokbo = jokboProgressMagnitude(a.jokboProgressSelf);
  if (afterJokbo > beforeJokbo) {
    strategic += Math.min(0.14, (afterJokbo - beforeJokbo) * 0.05);
  }
  if (jpCompletedNow) strategic += 0.2;

  const beforeOppThreat = Number(b.oppJokboThreatProb || 0);
  const afterOppThreat = Number(a.oppJokboThreatProb || 0);
  const blockDrop = Math.max(0, beforeOppThreat - afterOppThreat);
  if (blockDrop > 0) {
    strategic += Math.min(0.15, blockDrop * 0.25);
  }

  const bakShiftPi = bakRiskShiftReward(b, a, "piBakRisk", 0.1);
  const bakShiftGwang = bakRiskShiftReward(b, a, "gwangBakRisk", 0.12);
  const bakShiftMong = bakRiskShiftReward(b, a, "mongBakRisk", 0.11);
  strategic += bakShiftPi + bakShiftGwang + bakShiftMong;

  const matchEvents = action.matchEvents || [];
  if (matchEvents.some((m) => m?.eventTag === "PPUK")) {
    strategic -= 0.22;
  }

  const raw = shortTerm + strategic;
  const clipped = clampRange(raw, -IR_TURN_CLIP, IR_TURN_CLIP);
  return {
    raw,
    clipped,
    parts: {
      shortTerm,
      strategic,
      blockDrop,
      bakShift: {
        pi: bakShiftPi,
        gwang: bakShiftGwang,
        mong: bakShiftMong
      }
    }
  };
}

function applyEpisodeIrBudget(irValue, actor, totalsByActor, limit = IR_EPISODE_CLIP) {
  const prev = Number(totalsByActor.get(actor) || 0);
  let adjusted = Number(irValue || 0);
  if (adjusted > 0) adjusted = Math.min(adjusted, limit - prev);
  else if (adjusted < 0) adjusted = Math.max(adjusted, -limit - prev);
  adjusted = clampRange(adjusted, -IR_TURN_CLIP, IR_TURN_CLIP);
  totalsByActor.set(actor, clampRange(prev + adjusted, -limit, limit));
  return adjusted;
}

function compactDecisionContextForTrace(dc, actorSide = null) {
  if (!dc) return null;
  const handCountSelf = Math.max(0, Math.floor(Number(dc.handCountSelf || 0)));
  const handCountOpp = Math.max(0, Math.floor(Number(dc.handCountOpp || 0)));
  const shakeSelf = Math.max(0, Math.floor(Number(dc.shakeCountSelf || 0)));
  const shakeOpp = Math.max(0, Math.floor(Number(dc.shakeCountOpp || 0)));
  const currentScoreSelf = Math.floor(Number(dc.currentScoreSelf || 0));
  const currentScoreOpp = Math.floor(Number(dc.currentScoreOpp || 0));
  const scoreDiff = Math.floor(Number(dc.currentScoreSelf || 0) - Number(dc.currentScoreOpp || 0));
  let scoreMy = currentScoreSelf;
  let scoreYour = currentScoreOpp;
  if (actorSide === SIDE_YOUR) {
    scoreMy = currentScoreOpp;
    scoreYour = currentScoreSelf;
  }
  return {
    p: tracePhaseCode(dc.phase),
    d: Math.max(0, Math.floor(Number(dc.deckCount || 0))),
    hs: handCountSelf,
    hd: handCountSelf - handCountOpp,
    gs: Math.max(0, Math.floor(Number(dc.goCountSelf || 0))),
    go: Math.max(0, Math.floor(Number(dc.goCountOpp || 0))),
    rp: Number(dc.piBakRisk || 0) ? 1 : 0,
    rg: Number(dc.gwangBakRisk || 0) ? 1 : 0,
    rm: Number(dc.mongBakRisk || 0) ? 1 : 0,
    ss: shakeSelf,
    so: shakeOpp,
    jps: Math.max(0, Math.floor(Number(dc.jokboProgressSelfSum || 0))),
    jpo: Math.max(0, Math.floor(Number(dc.jokboProgressOppSum || 0))),
    jas: Math.max(0, Math.floor(Number(dc.jokboOneAwaySelfCount || 0))),
    jao: Math.max(0, Math.floor(Number(dc.jokboOneAwayOppCount || 0))),
    sjt: toPercentInt(dc.selfJokboThreatProb),
    sjo: toPercentInt(dc.selfJokboOneAwayProb),
    sgt: toPercentInt(dc.selfGwangThreatProb),
    ojt: toPercentInt(dc.oppJokboThreatProb),
    ojo: toPercentInt(dc.oppJokboOneAwayProb),
    ogt: toPercentInt(dc.oppGwangThreatProb),
    gsd: toSignedPermille(dc.goStopDeltaProxy, 3),
    sm: scoreMy,
    sy: scoreYour,
    sd: scoreDiff
  };
}

function decisionTraceDcSignature(dc) {
  if (!dc || typeof dc !== "object") return "";
  return [
    dc.p,
    dc.d,
    dc.hs,
    dc.hd,
    dc.gs,
    dc.go,
    dc.rp,
    dc.rg,
    dc.rm,
    dc.ss,
    dc.so,
    dc.jps,
    dc.jpo,
    dc.jas,
    dc.jao,
    dc.sjt,
    dc.sjo,
    dc.sgt,
    dc.ojt,
    dc.ojo,
    dc.ogt,
    dc.gsd,
    dc.sm,
    dc.sy,
    dc.sd
  ]
    .map((v) => String(v ?? ""))
    .join("|");
}

function shouldDropStableTurnRecord(record, lastStableKeyByActor) {
  if (!record || typeof record !== "object") return false;
  const keepBy = Array.isArray(record.keepBy) ? record.keepBy : [];
  if (keepBy.length > 0) return false;
  if (Number(record.ir || 0) !== 0) return false;
  const actor = String(record.a || "");
  if (!actor) return false;
  const key = [
    record.cc,
    record.dt,
    record.ch,
    decisionTraceDcSignature(record.dc)
  ]
    .map((v) => String(v ?? ""))
    .join("|");
  const prev = lastStableKeyByActor.get(actor);
  if (prev && prev === key) return true;
  lastStableKeyByActor.set(actor, key);
  return false;
}

function dedupeStableDecisionTrace(trace) {
  if (!Array.isArray(trace) || trace.length <= 1) return Array.isArray(trace) ? trace : [];
  const out = [];
  const lastStableKeyByActor = new Map();
  for (const rec of trace) {
    const actor = String(rec?.a || "");
    if (shouldDropStableTurnRecord(rec, lastStableKeyByActor)) continue;
    if (Array.isArray(rec?.keepBy) && rec.keepBy.length > 0) {
      lastStableKeyByActor.delete(actor);
    } else if (Number(rec?.ir || 0) !== 0) {
      lastStableKeyByActor.delete(actor);
    }
    out.push(rec);
  }
  return out;
}

function buildDecisionTrace(
  kibo,
  firstTurnKey,
  secondTurnKey,
  contextByTurnNo,
  options = {}
) {
  const myTurnOnly = !!options.myTurnOnly;
  const importantOnly = options.importantOnly !== false;
  const contextRadius = Number(options.contextRadius || 0);
  const goStopPlus2 = !!options.goStopPlus2;
  const dedupeStableTurns = !!options.dedupeStableTurns;
  const optionEvents = Array.isArray(options.optionEvents) ? options.optionEvents : [];
  const turns = (kibo || []).filter((e) => e.type === "turn_end");
  const irTotalsByActor = new Map([
    [firstTurnKey, 0],
    [secondTurnKey, 0]
  ]);
  const sideTurnOrdinal = { [SIDE_MY]: 0, [SIDE_YOUR]: 0 };
  const records = [];
  for (const t of turns) {
    const actor = t.actor;
    const actorSide = actorToSide(actor, firstTurnKey);
    if (myTurnOnly && actorSide !== SIDE_MY) continue;
    const action = t.action || {};
    const turnCtx = contextByTurnNo.get(t.turnNo) || null;
    const before = turnCtx?.before || null;
    const after = turnCtx?.after || null;
    const decisionType = traceDecisionType(before?.selectionPool || null, action.type);
    const legalActions = traceLegalActions(before?.selectionPool || null, decisionType);
    const candidateCount = legalActions.length;
    if (candidateCount < 2) continue;
    const compactDc = compactDecisionContextForTrace(before?.decisionContext || null, actorSide);
    const jpPre = before?.decisionContext?.jokboProgressSelf || null;
    const jpPost = after?.decisionContext?.jokboProgressSelf || null;
    const jpCompletedNow = !!(jokboCompleted(jpPost) && !jokboCompleted(jpPre));
    const irEval = weightedImmediateReward(
      t,
      jpCompletedNow,
      before?.decisionContext || null,
      after?.decisionContext || null
    );
    const immediateReward = applyEpisodeIrBudget(irEval.clipped, actor, irTotalsByActor);
    sideTurnOrdinal[actorSide] = Number(sideTurnOrdinal[actorSide] || 0) + 1;
    const earlyTurnForced = sideTurnOrdinal[actorSide] <= 1;
    const triggers = classifyImportantTurnTriggers({
      decisionType,
      turn: t,
      beforeDc: before?.decisionContext || null,
      afterDc: after?.decisionContext || null,
      earlyTurnForced,
      goStopPlus2
    });
    const seq = Number(turnCtx?.seq || t.turnNo || 0);
    const choice =
      decisionType === "play"
        ? action.card?.id || null
        : decisionType === "match"
        ? action.selectedBoardCard?.id || null
        : actionTypeToChoice(action.type || "option");
    records.push({
      seq,
      triggers,
      data: {
        seq,
        t: t.turnNo,
        a: actorSide,
        dt: decisionType,
        cc: candidateCount,
        la: legalActions,
        ch: choice,
        ir: immediateReward,
        dc: compactDc
      }
    });
  }
  for (const ev of optionEvents) {
    const actor = ev?.actor;
    const actorSide = actorToSide(actor, firstTurnKey);
    if (myTurnOnly && actorSide !== SIDE_MY) continue;
    const actionType = String(ev?.actionType || "option");
    const optionActions = normalizeLegalActions(ev?.options || [], "option");
    const legalActions = optionActions.length > 0 ? optionActions : optionLegalActionsFromActionType(actionType);
    const candidateCount = legalActions.length;
    if (candidateCount < 2) continue;
    const beforeDc = ev?.beforeDc || null;
    const afterDc = ev?.afterDc || null;
    const compactDc = compactDecisionContextForTrace(beforeDc, actorSide);
    const turnNo = Math.max(0, Math.floor(Number(ev?.turnNo || 0)));
    const triggerTurn = { action: { type: actionType, matchEvents: [] } };
    const triggers = classifyImportantTurnTriggers({
      decisionType: "option",
      turn: triggerTurn,
      beforeDc,
      afterDc,
      earlyTurnForced: false,
      goStopPlus2
    });
    const seq = Number(ev?.seq || 0);
    records.push({
      seq,
      triggers,
      data: {
        seq,
        t: turnNo,
        a: actorSide,
        dt: "option",
        cc: candidateCount,
        la: legalActions,
        ch: actionTypeToChoice(actionType),
        ir: 0,
        dc: compactDc
      }
    });
  }
  records.sort((x, y) => {
    const sx = Number(x?.seq || 0);
    const sy = Number(y?.seq || 0);
    if (sx !== sy) return sx - sy;
    return Number(x?.data?.t || 0) - Number(y?.data?.t || 0);
  });
  applyTerminalContextTrigger(records);
  let trace = [];
  if (importantOnly) {
    trace = keepImportantWithContext(records, contextRadius);
  } else {
    trace = records.map((r) => {
      const selfTriggers = triggerNames(r?.triggers || [])
        .map((name) => keepBySelf(name))
        .filter(Boolean);
      const unique = new Map();
      for (const tag of selfTriggers) unique.set(keepByIndexKey(tag), tag);
      return {
        ...r.data,
        keepBy: [...unique.values()].sort((a, b) => keepBySortKey(a).localeCompare(keepBySortKey(b)))
      };
    });
  }
  if (dedupeStableTurns) {
    trace = dedupeStableDecisionTrace(trace);
  }
  return trace;
}

// -----------------------------------------------------------------------------
// 11) Aggregation and report builders
// -----------------------------------------------------------------------------
function baseSummary({
  completed,
  winner,
  mySideScore,
  yourSideScore,
  mySideGold,
  yourSideGold
}) {
  aggregate.completed += completed ? 1 : 0;
  if (aggregate.winners[winner] != null) aggregate.winners[winner] += 1;
  else aggregate.winners.unknown += 1;
  if (winner === SIDE_MY) aggregate.bySide.mySideWins += 1;
  else if (winner === SIDE_YOUR) aggregate.bySide.yourSideWins += 1;
  else aggregate.bySide.draw += 1;
  aggregate.bySide.mySideScoreSum += mySideScore;
  aggregate.bySide.yourSideScoreSum += yourSideScore;
  const mySideDelta = mySideGold - yourSideGold;
  aggregate.economy.mySideGoldSum += mySideGold;
  aggregate.economy.yourSideGoldSum += yourSideGold;
  aggregate.economy.mySideDeltaSum += mySideDelta;
  if (aggregate.economy.first1000Games < 1000) {
    aggregate.economy.first1000MySideDeltaSum += mySideDelta;
    aggregate.economy.first1000Games += 1;
  }
  const mySideBankrupt = mySideGold <= 0 ? 1 : 0;
  const yourSideBankrupt = yourSideGold <= 0 ? 1 : 0;
  aggregate.bankrupt.mySideInflicted += yourSideBankrupt;
  aggregate.bankrupt.mySideSuffered += mySideBankrupt;
  aggregate.bankrupt.yourSideInflicted += mySideBankrupt;
  aggregate.bankrupt.yourSideSuffered += yourSideBankrupt;
  if (mySideBankrupt || yourSideBankrupt) {
    aggregate.bankrupt.resets += 1;
  }
}

function fullSummary({
  completed,
  winner,
  mySideScore,
  yourSideScore,
  mySideGold,
  yourSideGold,
  nagari,
  eventFrequency,
  goCalls,
  goEfficiency,
  goDeclared,
  goSuccess,
  totalPiSteals,
  totalGoldSteals,
  loserKey,
  loserBak,
  bakEscaped,
  flipEvents,
  handEvents,
  flipCaptureValue,
  handCaptureValue
}) {
  baseSummary({
    completed,
    winner,
    mySideScore,
    yourSideScore,
    mySideGold,
    yourSideGold
  });
  if (nagari) aggregate.nagari += 1;
  aggregate.eventTotals.ppuk += Number(eventFrequency?.ppuk || 0);
  aggregate.eventTotals.ddadak += Number(eventFrequency?.ddadak || 0);
  aggregate.eventTotals.jjob += Number(eventFrequency?.jjob || 0);
  aggregate.eventTotals.ssul += Number(eventFrequency?.ssul || 0);
  aggregate.eventTotals.pansseul += Number(eventFrequency?.pansseul || 0);
  aggregate.goCalls += goCalls;
  aggregate.goEfficiencySum += goEfficiency;
  aggregate.goDecision.declared += goDeclared;
  aggregate.goDecision.success += goSuccess;
  aggregate.steals.piTotal += totalPiSteals;
  aggregate.steals.goldTotal += totalGoldSteals;
  if (loserKey) {
    aggregate.bakEscape.trials += 1;
    aggregate.bakEscape.escaped += bakEscaped;
    aggregate.bakBreakdown.totalLoserCount += 1;
    const pi = loserBak?.pi ? 1 : 0;
    const gwang = loserBak?.gwang ? 1 : 0;
    const mong = loserBak?.mongBak ? 1 : 0;
    aggregate.bakBreakdown.piBakCount += pi;
    aggregate.bakBreakdown.gwangBakCount += gwang;
    aggregate.bakBreakdown.mongBakCount += mong;
    if (pi + gwang + mong >= 2) aggregate.bakBreakdown.doubleBakCount += 1;
  }
  aggregate.luckFlipEvents += flipEvents;
  aggregate.luckHandEvents += handEvents;
  aggregate.luckFlipCaptureValue += flipCaptureValue;
  aggregate.luckHandCaptureValue += handCaptureValue;
}

function buildReportBase() {
  return {
    logMode,
    seatMode,
    games: aggregate.games,
    completed: aggregate.completed,
    winners: aggregate.winners,
    learningRole: {
      attackCount: aggregate.learningRoleCounts.ATTACK,
      defenseCount: aggregate.learningRoleCounts.DEFENSE,
      neutralCount: aggregate.learningRoleCounts.NEUTRAL,
      attackScoreMin: LEARNING_ROLE_ATTACK_SCORE_MIN,
      defenseOppScoreMax: LEARNING_ROLE_DEFENSE_OPP_SCORE_MAX
    },
    sideStats: {
      mySideWinRate: aggregate.bySide.mySideWins / Math.max(1, aggregate.games),
      yourSideWinRate: aggregate.bySide.yourSideWins / Math.max(1, aggregate.games),
      drawRate: aggregate.bySide.draw / Math.max(1, aggregate.games),
      averageScoreMySide: aggregate.bySide.mySideScoreSum / Math.max(1, aggregate.games),
      averageScoreYourSide: aggregate.bySide.yourSideScoreSum / Math.max(1, aggregate.games)
    },
    economy: {
      averageGoldMySide: aggregate.economy.mySideGoldSum / Math.max(1, aggregate.games),
      averageGoldYourSide: aggregate.economy.yourSideGoldSum / Math.max(1, aggregate.games),
      averageGoldDeltaMySide: aggregate.economy.mySideDeltaSum / Math.max(1, aggregate.games),
      cumulativeGoldDeltaOver1000:
        (aggregate.economy.mySideDeltaSum / Math.max(1, aggregate.games)) * 1000,
      cumulativeGoldDeltaMySideFirst1000: aggregate.economy.first1000MySideDeltaSum
    },
    bankrupt: {
      mySideInflicted: aggregate.bankrupt.mySideInflicted,
      mySideSuffered: aggregate.bankrupt.mySideSuffered,
      mySideDiff: aggregate.bankrupt.mySideInflicted - aggregate.bankrupt.mySideSuffered,
      yourSideInflicted: aggregate.bankrupt.yourSideInflicted,
      yourSideSuffered: aggregate.bankrupt.yourSideSuffered,
      yourSideDiff: aggregate.bankrupt.yourSideInflicted - aggregate.bankrupt.yourSideSuffered,
      resets: aggregate.bankrupt.resets,
      mySideInflictedRate: aggregate.bankrupt.mySideInflicted / Math.max(1, aggregate.games),
      mySideSufferedRate: aggregate.bankrupt.mySideSuffered / Math.max(1, aggregate.games),
      yourSideInflictedRate: aggregate.bankrupt.yourSideInflicted / Math.max(1, aggregate.games),
      yourSideSufferedRate: aggregate.bankrupt.yourSideSuffered / Math.max(1, aggregate.games)
    },
    primaryMetric: "averageGoldDeltaMySide"
  };
}

function buildFullReport() {
  return {
    ...buildReportBase(),
    catalogPath: sharedCatalogPath,
    nagariRate: aggregate.nagari / Math.max(1, aggregate.games),
    eventFrequencyPerGame: {
      ppuk: aggregate.eventTotals.ppuk / Math.max(1, aggregate.games),
      ddadak: aggregate.eventTotals.ddadak / Math.max(1, aggregate.games),
      jjob: aggregate.eventTotals.jjob / Math.max(1, aggregate.games),
      ssul: aggregate.eventTotals.ssul / Math.max(1, aggregate.games),
      pansseul: aggregate.eventTotals.pansseul / Math.max(1, aggregate.games)
    },
    goStopEfficiencyAvg: aggregate.goEfficiencySum / Math.max(1, aggregate.games),
    goDecision: {
      declared: aggregate.goDecision.declared,
      success: aggregate.goDecision.success,
      successRate: aggregate.goDecision.success / Math.max(1, aggregate.goDecision.declared)
    },
    stealEfficiency: {
      averagePiStealPerGame: aggregate.steals.piTotal / Math.max(1, aggregate.games),
      averageGoldStealPerGame: aggregate.steals.goldTotal / Math.max(1, aggregate.games)
    },
    bakEscapeRate: aggregate.bakEscape.escaped / Math.max(1, aggregate.bakEscape.trials),
    bakBreakdown: {
      totalLoserCount: aggregate.bakBreakdown.totalLoserCount,
      piBak:
        aggregate.bakBreakdown.piBakCount /
        Math.max(1, aggregate.bakBreakdown.totalLoserCount),
      gwangBak:
        aggregate.bakBreakdown.gwangBakCount /
        Math.max(1, aggregate.bakBreakdown.totalLoserCount),
      mongBak:
        aggregate.bakBreakdown.mongBakCount /
        Math.max(1, aggregate.bakBreakdown.totalLoserCount),
      doubleBak:
        aggregate.bakBreakdown.doubleBakCount /
        Math.max(1, aggregate.bakBreakdown.totalLoserCount)
    },
    luckSkillIndex: {
      flipEvents: aggregate.luckFlipEvents,
      handEvents: aggregate.luckHandEvents,
      flipRatio: aggregate.luckFlipEvents / Math.max(1, aggregate.luckFlipEvents + aggregate.luckHandEvents),
      weightedCapture: {
        flipValue: aggregate.luckFlipCaptureValue,
        handValue: aggregate.luckHandCaptureValue,
        flipRatio:
          aggregate.luckFlipCaptureValue /
          Math.max(1, aggregate.luckFlipCaptureValue + aggregate.luckHandCaptureValue)
      }
    }
  };
}

// -----------------------------------------------------------------------------
// 12) Runtime IO and turn execution
// -----------------------------------------------------------------------------
async function writeLine(writer, line) {
  if (writer.write(`${JSON.stringify(line)}\n`)) return;
  await once(writer, "drain");
}

function buildPersistedLine({
  gameIndex,
  seed,
  steps,
  firstTurnKey,
  agentLabelBySide,
  winnerSide,
  mySideScore,
  yourSideScore,
  initialGoldMy,
  initialGoldYour,
  mySideGold,
  yourSideGold,
  learningRole,
  decisionTrace
}) {
  return {
    ver: {
      s: TRACE_SCHEMA_VERSION,
      f: TRACE_FEATURE_VERSION,
      t: TRACE_TRIGGER_DICT_VERSION
    },
    game: gameIndex + 1,
    runId,
    seatMode,
    seed,
    steps,
    firstAttackerActor: firstTurnKey,
    policy: {
      [SIDE_MY]: agentLabelBySide[SIDE_MY],
      [SIDE_YOUR]: agentLabelBySide[SIDE_YOUR]
    },
    winner: winnerSide,
    score: {
      [SIDE_MY]: mySideScore,
      [SIDE_YOUR]: yourSideScore
    },
    initialGoldMy,
    initialGoldYour,
    finalGoldMy: mySideGold,
    finalGoldYour: yourSideGold,
    learningRole,
    decision_trace: decisionTrace
  };
}

function executeActorTurn(state, actor, cfg, control = {}) {
  const runtimeCfg = buildRuntimeModelConfig(state, actor, cfg);
  const canUseModel = !!runtimeCfg?.policyModel;
  if (canUseModel) {
    const picked = modelSelectCandidate(state, actor, runtimeCfg, control);
    if (picked) {
      const next = applyModelChoice(state, actor, picked, runtimeCfg);
      if (next !== state) return next;
    }
  }
  const exploratoryPick = maybePickExplorationChoice(state, actor, runtimeCfg);
  if (exploratoryPick) {
    const explored = applyModelChoice(state, actor, exploratoryPick, runtimeCfg);
    if (explored !== state) return explored;
  }
  return aiPlay(state, actor, {
    source: "heuristic",
    heuristicPolicy: runtimeCfg?.fallbackPolicy || DEFAULT_POLICY
  });
}

// -----------------------------------------------------------------------------
// 13) Main game loop
// -----------------------------------------------------------------------------
async function run() {
  const outStream = fs.createWriteStream(outPath, { encoding: "utf8" });
  const probeState = initGame("A", createSeededRng("side-probe-seed"), {
    carryOverMultiplier: 1,
    kiboDetail: "lean"
  });
  const [actorA, actorB] = actorPairFromState(probeState);
  const firstTurnPlan = createFirstTurnPlan(games, actorA, actorB, fixedSeats);
  const sessionGold = {
    [SIDE_MY]: STARTING_GOLD,
    [SIDE_YOUR]: STARTING_GOLD
  };

  for (let i = 0; i < games; i += 1) {
    const seed = `sim-${Date.now()}-${i}-${Math.random().toString(36).slice(2, 8)}`;
    const forcedFirstTurnKey = firstTurnPlan[i];
    const forcedSecondTurnKey = forcedFirstTurnKey === actorA ? actorB : actorA;
    const initialGoldByActor = {
      [forcedFirstTurnKey]: sessionGold[SIDE_MY],
      [forcedSecondTurnKey]: sessionGold[SIDE_YOUR]
    };
    let state = initGame("A", createSeededRng(seed), {
      carryOverMultiplier: 1,
      kiboDetail: "lean",
      firstTurnKey: forcedFirstTurnKey,
      initialGold: initialGoldByActor
    });
    const firstTurnKey = state.startingTurnKey;
    const secondTurnKey = secondActorFromFirst(state, firstTurnKey);
    if (!secondTurnKey) {
      throw new Error("failed to resolve second actor key from state");
    }
    state.players[firstTurnKey].label = "MY-SIDE";
    state.players[secondTurnKey].label = "YOUR-SIDE";
    const initialGoldMy = Number(state.players?.[firstTurnKey]?.gold || 0);
    const initialGoldYour = Number(state.players?.[secondTurnKey]?.gold || 0);

    const contextByTurnNo = new Map();
    const optionTraceEvents = [];
    let steps = 0;
    const maxSteps = 4000;
    while (state.phase !== "resolution" && steps < maxSteps) {
      const actor = getActionPlayerKey(state);
      if (!actor) break;
      const beforeContext = {
        actor,
        decisionContext: decisionContext(state, actor, { includeCardLists: false }),
        selectionPool: selectPool(state, actor, { includeCardLists: true })
      };
      const prevTurnSeq = state.turnSeq || 0;
      const prevKiboSeq = state.kiboSeq || 0;
      const traceSeqBase = (steps + 1) * 2;
      const actorSide = actorToSide(actor, firstTurnKey);
      const next = executeActorTurn(state, actor, sideConfig[actorSide]);
      if (next === state) break;
      const nextTurnSeq = next.turnSeq || 0;
      const afterDc = decisionContext(next, actor, { includeCardLists: false });
      const beforeSp = beforeContext?.selectionPool || null;
      const beforeDecisionType = String(beforeSp?.decisionType || "");
      if (beforeDecisionType === "option") {
        optionTraceEvents.push({
          seq: traceSeqBase,
          turnNo: nextTurnSeq > prevTurnSeq ? nextTurnSeq : prevTurnSeq,
          actor,
          candidateCount: traceCandidateCount(beforeSp),
          options: Array.isArray(beforeSp?.options) ? [...beforeSp.options] : [],
          actionType: inferOptionActionType({
            beforeState: state,
            nextState: next,
            actor,
            selectionPool: beforeSp,
            prevKiboSeq
          }),
          beforeDc: beforeContext?.decisionContext || null,
          afterDc
        });
      }
      if (nextTurnSeq > prevTurnSeq) {
        contextByTurnNo.set(nextTurnSeq, {
          seq: traceSeqBase + 1,
          before: beforeContext,
          after: {
            actor,
            decisionContext: afterDc,
            selectionPool: null
          }
        });
      }
      state = next;
      steps += 1;
    }

    const winner = state.result?.winner || "unknown";
    const kibo = state.kibo || [];
    const traceOptions = {
      myTurnOnly: traceMyTurnOnly,
      importantOnly: traceImportantOnly,
      contextRadius: traceContextRadius,
      goStopPlus2: traceGoStopPlus2,
      dedupeStableTurns: traceDedupeStableTurns,
      optionEvents: optionTraceEvents
    };
    const decisionTrace = buildDecisionTrace(
      kibo,
      firstTurnKey,
      secondTurnKey,
      contextByTurnNo,
      traceOptions
    );
    const mySideActor = firstTurnKey;
    const yourSideActor = secondTurnKey;
    const mySideScore = Number(state.result?.[mySideActor]?.total || 0);
    const yourSideScore = Number(state.result?.[yourSideActor]?.total || 0);
    const mySideGold = Number(state.players?.[mySideActor]?.gold || 0);
    const yourSideGold = Number(state.players?.[yourSideActor]?.gold || 0);
    const mySideBankrupt = mySideGold <= 0;
    const yourSideBankrupt = yourSideGold <= 0;
    const sessionReset = mySideBankrupt || yourSideBankrupt;
    if (sessionReset) {
      sessionGold[SIDE_MY] = STARTING_GOLD;
      sessionGold[SIDE_YOUR] = STARTING_GOLD;
    } else {
      sessionGold[SIDE_MY] = mySideGold;
      sessionGold[SIDE_YOUR] = yourSideGold;
    }

    const winnerSide =
      winner === "draw" || winner === "unknown" ? winner : actorToSide(winner, firstTurnKey);
    const learningRole = classifyLearningRoleMySide(winnerSide, mySideScore, yourSideScore);
    addLearningRoleCount(learningRole);
    const persistedLine = buildPersistedLine({
      gameIndex: i,
      seed,
      steps,
      firstTurnKey,
      agentLabelBySide,
      winnerSide,
      mySideScore,
      yourSideScore,
      initialGoldMy,
      initialGoldYour,
      mySideGold,
      yourSideGold,
      learningRole,
      decisionTrace
    });

    const goCalls = kibo.filter((e) => e.type === "go").length;
    const winnerTotal =
      winnerSide === SIDE_MY
        ? mySideScore
        : winnerSide === SIDE_YOUR
        ? yourSideScore
        : 0;
    const loserTotal =
      winnerSide === SIDE_MY
        ? yourSideScore
        : winnerSide === SIDE_YOUR
        ? mySideScore
        : 0;
    const goEfficiency = goCalls > 0 ? (winnerTotal - loserTotal) / goCalls : 0;

    let flipEvents = 0;
    let handEvents = 0;
    let flipCaptureValue = 0;
    let handCaptureValue = 0;
    let totalPiSteals = 0;
    let totalGoldSteals = 0;
    const turnEnds = (kibo || []).filter((e) => e.type === "turn_end");
    turnEnds.forEach((t) => {
      totalPiSteals += t.steals?.pi || 0;
      totalGoldSteals += t.steals?.gold || 0;
      const cap = t.action?.captureBySource || { hand: [], flip: [] };
      (cap.flip || []).forEach((c) => {
        flipCaptureValue += captureWeight(c);
      });
      (cap.hand || []).forEach((c) => {
        handCaptureValue += captureWeight(c);
      });
    });

    turnEnds.forEach((t) => {
      (t.action?.matchEvents || []).forEach((m) => {
        if (!m?.eventTag || m.eventTag === "NORMAL") return;
        if (m.source === "flip") flipEvents += 1;
        if (m.source === "hand") handEvents += 1;
      });
    });

    const allEvents = {
      [SIDE_MY]: state.players[mySideActor].events,
      [SIDE_YOUR]: state.players[yourSideActor].events
    };

    const goEvents = kibo.filter((e) => e.type === "go");
    const goDeclared = goEvents.length;
    const goSuccess = goEvents.filter((e) => winner !== "draw" && winner === e.playerKey).length;

    const loserSide =
      winnerSide === SIDE_MY ? SIDE_YOUR : winnerSide === SIDE_YOUR ? SIDE_MY : null;
    const loserActor = loserSide ? sideToActor(loserSide, firstTurnKey, secondTurnKey) : null;
    const loserBak = loserActor ? state.result?.[loserActor]?.bak : null;
    const bakEscaped =
      loserBak && !loserBak.gwang && !loserBak.pi && !loserBak.mongBak ? 1 : 0;

    const completed = state.phase === "resolution";
    const nagari = state.result?.nagari || false;
    const eventFrequency = {
      ppuk: (allEvents[SIDE_MY].ppuk || 0) + (allEvents[SIDE_YOUR].ppuk || 0),
      ddadak: (allEvents[SIDE_MY].ddadak || 0) + (allEvents[SIDE_YOUR].ddadak || 0),
      jjob: (allEvents[SIDE_MY].jjob || 0) + (allEvents[SIDE_YOUR].jjob || 0),
      ssul: (allEvents[SIDE_MY].ssul || 0) + (allEvents[SIDE_YOUR].ssul || 0),
      pansseul: (allEvents[SIDE_MY].pansseul || 0) + (allEvents[SIDE_YOUR].pansseul || 0)
    };

    fullSummary({
      completed,
      winner: winnerSide,
      mySideScore,
      yourSideScore,
      mySideGold,
      yourSideGold,
      nagari,
      eventFrequency,
      goCalls,
      goEfficiency,
      goDeclared,
      goSuccess,
      totalPiSteals,
      totalGoldSteals,
      loserKey: loserSide,
      loserBak,
      bakEscaped,
      flipEvents,
      handEvents,
      flipCaptureValue,
      handCaptureValue
    });

    await writeLine(outStream, persistedLine);
  }

  outStream.end();
  await once(outStream, "finish");

  const report = buildFullReport();
  ensureSharedCatalog();
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2), "utf8");
  console.log(`done: ${games} games -> ${outPath}`);
  console.log(`report: ${reportPath}`);
  console.log(`catalog: ${sharedCatalogPath}`);
}

run().catch((err) => {
  console.error(err instanceof Error ? err.message : String(err));
  process.exitCode = 1;
});

