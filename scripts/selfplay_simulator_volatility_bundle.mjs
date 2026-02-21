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

if (process.env.NO_SIMULATION === "1") {
  console.error("Simulation blocked: NO_SIMULATION=1");
  process.exit(2);
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DEFAULT_LOG_MODE = "train";
const SUPPORTED_LOG_MODES = new Set(["train", "delta"]);
const SUPPORTED_POLICIES = new Set(BOT_POLICIES);
const DEFAULT_POLICY = "heuristic_v3";
const DECISION_CACHE_MAX = 200000;
const HASH_CACHE_MAX = 500000;
const TRAIN_EXPLORE_RATE_DEFAULT = 0.006;
const TRAIN_EXPLORE_RATE_COMEBACK_DEFAULT = 0.012;
const IR_TURN_CLIP = 0.3;
const IR_EPISODE_CLIP = 1.0;
const FULL_DECK = buildDeck();
const SIMULATOR_NAME = "volatility_bundle_v1";
const VOLATILITY_EVENT_KEYS = Object.freeze(["ppuk", "jjob", "ddadak", "pansseul", "ssul", "other"]);
const RIBBON_COMBO_MONTH_SETS = Object.freeze({
  hongdan: Object.freeze([1, 2, 3]),
  cheongdan: Object.freeze([6, 9, 10]),
  chodan: Object.freeze([4, 5, 7])
});
const GODORI_MONTHS = Object.freeze([2, 4, 8]);
const GWANG_MONTHS = Object.freeze([1, 3, 8, 11, 12]);

const decisionInferenceCache = new Map();
const hashIndexCache = new Map();
const SIDE_MY = "mySide";
const SIDE_YOUR = "yourSide";

function setBoundedCache(cache, key, value, maxEntries) {
  if (cache.size >= maxEntries) cache.clear();
  cache.set(key, value);
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

function parseArgs(argv) {
  const args = [...argv];
  let games = 1000;
  let outArg = null;
  let logMode = DEFAULT_LOG_MODE;
  let policyMySide = DEFAULT_POLICY;
  let policyYourSide = DEFAULT_POLICY;
  let modelOnly = false;
  let policyModelMySide = null;
  let policyModelYourSide = null;
  let valueModelMySide = null;
  let valueModelYourSide = null;
  let trainExploreRate = TRAIN_EXPLORE_RATE_DEFAULT;
  let trainExploreRateComeback = TRAIN_EXPLORE_RATE_COMEBACK_DEFAULT;
  let traceMyTurnOnly = false;
  let traceImportantOnly = true;
  let traceContextRadius = 0;
  let traceGoStopPlus2 = false;

  if (args.length > 0 && /^\d+$/.test(args[0])) {
    games = Number(args.shift());
  }

  if (args.length > 0 && !args[0].startsWith("--")) {
    outArg = args.shift();
  }

  while (args.length > 0) {
    const arg = args.shift();
    if (arg === "--log-mode" && args.length > 0) {
      logMode = String(args.shift()).trim();
      continue;
    }
    if (arg.startsWith("--log-mode=")) {
      logMode = arg.split("=", 2)[1].trim();
      continue;
    }
    if (SUPPORTED_LOG_MODES.has(arg)) {
      logMode = arg;
      continue;
    }
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
    if (arg === "--value-model-my-side" && args.length > 0) {
      valueModelMySide = String(args.shift()).trim();
      continue;
    }
    if (arg.startsWith("--value-model-my-side=")) {
      valueModelMySide = arg.split("=", 2)[1].trim();
      continue;
    }
    if (arg === "--value-model-your-side" && args.length > 0) {
      valueModelYourSide = String(args.shift()).trim();
      continue;
    }
    if (arg.startsWith("--value-model-your-side=")) {
      valueModelYourSide = arg.split("=", 2)[1].trim();
      continue;
    }
    if (arg === "--model-only" || arg === "--strict-model-only") {
      modelOnly = true;
      continue;
    }
    if (arg === "--train-explore-rate" && args.length > 0) {
      trainExploreRate = Number(args.shift());
      continue;
    }
    if (arg.startsWith("--train-explore-rate=")) {
      trainExploreRate = Number(arg.split("=", 2)[1]);
      continue;
    }
    if (arg === "--train-explore-rate-comeback" && args.length > 0) {
      trainExploreRateComeback = Number(args.shift());
      continue;
    }
    if (arg.startsWith("--train-explore-rate-comeback=")) {
      trainExploreRateComeback = Number(arg.split("=", 2)[1]);
      continue;
    }
    if (arg === "--trace-my-turn-only") {
      traceMyTurnOnly = true;
      continue;
    }
    if (arg === "--trace-all-turns") {
      traceMyTurnOnly = false;
      continue;
    }
    if (arg.startsWith("--trace-my-turn-only=")) {
      const raw = String(arg.split("=", 2)[1] || "").trim().toLowerCase();
      traceMyTurnOnly = !(raw === "0" || raw === "false" || raw === "no" || raw === "off");
      continue;
    }
    if (arg === "--trace-important-only") {
      traceImportantOnly = true;
      continue;
    }
    if (arg === "--trace-all-candidate-turns" || arg === "--trace-no-important-filter") {
      traceImportantOnly = false;
      continue;
    }
    if (arg === "--trace-context-radius" && args.length > 0) {
      traceContextRadius = Number(args.shift());
      continue;
    }
    if (arg.startsWith("--trace-context-radius=")) {
      traceContextRadius = Number(arg.split("=", 2)[1]);
      continue;
    }
    if (arg === "--trace-go-stop-plus2" || arg === "--trace-go-plus2") {
      traceGoStopPlus2 = true;
      continue;
    }
    if (arg === "--trace-no-go-stop-plus2" || arg === "--trace-no-go-plus2") {
      traceGoStopPlus2 = false;
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }

  if (!SUPPORTED_LOG_MODES.has(logMode)) {
    throw new Error(`Unsupported log mode: ${logMode}. Use one of: train, delta`);
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
  if (!Number.isFinite(trainExploreRate) || trainExploreRate < 0 || trainExploreRate > 1) {
    throw new Error(`Invalid --train-explore-rate: ${trainExploreRate}. Expected [0, 1].`);
  }
  if (
    !Number.isFinite(trainExploreRateComeback) ||
    trainExploreRateComeback < 0 ||
    trainExploreRateComeback > 1
  ) {
    throw new Error(
      `Invalid --train-explore-rate-comeback: ${trainExploreRateComeback}. Expected [0, 1].`
    );
  }
  if (!Number.isFinite(traceContextRadius) || traceContextRadius < 0) {
    throw new Error(
      `Invalid --trace-context-radius: ${traceContextRadius}. Expected non-negative integer.`
    );
  }
  traceContextRadius = Math.floor(traceContextRadius);

  return {
    games,
    outArg,
    logMode,
    policyMySide,
    policyYourSide,
    policyModelMySide,
    policyModelYourSide,
    valueModelMySide,
    valueModelYourSide,
    modelOnly,
    trainExploreRate,
    trainExploreRateComeback,
    traceMyTurnOnly,
    traceImportantOnly,
    traceContextRadius,
    traceGoStopPlus2
  };
}

const parsed = parseArgs(process.argv.slice(2));
const games = parsed.games;
const outArg = parsed.outArg;
const logMode = parsed.logMode;
const policyMySide = parsed.policyMySide;
const policyYourSide = parsed.policyYourSide;
const policyModelMySidePath = parsed.policyModelMySide;
const policyModelYourSidePath = parsed.policyModelYourSide;
const valueModelMySidePath = parsed.valueModelMySide;
const valueModelYourSidePath = parsed.valueModelYourSide;
const modelOnly = !!parsed.modelOnly;
const trainExploreRate = parsed.trainExploreRate;
const trainExploreRateComeback = parsed.trainExploreRateComeback;
const traceMyTurnOnly = !!parsed.traceMyTurnOnly;
const traceImportantOnly = parsed.traceImportantOnly !== false;
const traceContextRadius = Number.isFinite(parsed.traceContextRadius)
  ? Math.max(0, Math.floor(parsed.traceContextRadius))
  : 0;
const traceGoStopPlus2 = !!parsed.traceGoStopPlus2;
const isTrainMode = logMode === "train";
const isDeltaMode = logMode === "delta";
const needsDecisionTrace = true;
const useLeanKibo = isTrainMode || isDeltaMode;
const sideConfig = {
  [SIDE_MY]: {
    fallbackPolicy: policyMySide,
    policyModelPath: policyModelMySidePath,
    valueModelPath: valueModelMySidePath
  },
  [SIDE_YOUR]: {
    fallbackPolicy: policyYourSide,
    policyModelPath: policyModelYourSidePath,
    valueModelPath: valueModelYourSidePath
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
sideConfig[SIDE_MY].valueModel = loadJsonModel(
  sideConfig[SIDE_MY].valueModelPath,
  "value-model-my-side"
);
sideConfig[SIDE_YOUR].valueModel = loadJsonModel(
  sideConfig[SIDE_YOUR].valueModelPath,
  "value-model-your-side"
);

if (modelOnly) {
  for (const side of [SIDE_MY, SIDE_YOUR]) {
    if (!sideConfig[side].policyModel && !sideConfig[side].valueModel) {
      throw new Error(
        `model-only requires model for ${side}. Provide --policy-model-${side} and/or --value-model-${side}`
      );
    }
  }
}

const agentLabelBySide = {
  [SIDE_MY]:
    sideConfig[SIDE_MY].policyModel || sideConfig[SIDE_MY].valueModel
      ? `model:${path.basename(
          sideConfig[SIDE_MY].policyModelPath || sideConfig[SIDE_MY].valueModelPath
        )}`
      : sideConfig[SIDE_MY].fallbackPolicy,
  [SIDE_YOUR]:
    sideConfig[SIDE_YOUR].policyModel || sideConfig[SIDE_YOUR].valueModel
      ? `model:${path.basename(
          sideConfig[SIDE_YOUR].policyModelPath || sideConfig[SIDE_YOUR].valueModelPath
        )}`
      : sideConfig[SIDE_YOUR].fallbackPolicy
};

const stamp = new Date().toISOString().replace(/[:.]/g, "-");
const outPath =
  outArg || path.resolve(__dirname, "..", "logs", `volatility-bundle-side-vs-side-${stamp}.jsonl`);
const reportPath = outPath.replace(/\.jsonl$/i, "-report.json");
const sharedCatalogDir = path.resolve(__dirname, "..", "logs", "catalog");
const sharedCatalogPath = path.join(sharedCatalogDir, "cards-catalog.json");
const legacyCatalogPath = path.resolve(__dirname, "..", "logs", "catalog.json");

fs.mkdirSync(path.dirname(outPath), { recursive: true });

const aggregate = {
  games,
  completed: 0,
  winners: { mySide: 0, yourSide: 0, draw: 0, unknown: 0 },
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
  luckHandCaptureValue: 0,
  volatilityBundle: {
    bySide: {
      [SIDE_MY]: createVolatilityAggregateSideStats(),
      [SIDE_YOUR]: createVolatilityAggregateSideStats()
    }
  }
};

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
  if (!fs.existsSync(sharedCatalogPath) && fs.existsSync(legacyCatalogPath)) {
    fs.renameSync(legacyCatalogPath, sharedCatalogPath);
  }
  if (fs.existsSync(sharedCatalogPath)) return sharedCatalogPath;
  fs.writeFileSync(sharedCatalogPath, JSON.stringify(buildCatalog(), null, 2), "utf8");
  return sharedCatalogPath;
}

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
    jokboProgressSelf: jokboSelfStats.progress,
    jokboProgressOpp: jokboOppStats.progress,
    comboBlockSelf,
    selfJokboThreatProb: clamp01(selfThreat.totalProb),
    selfJokboOneAwayProb: clamp01(selfThreat.oneAwayProb),
    selfGwangThreatProb: clamp01(selfThreat.gwangThreatProb),
    oppJokboThreatProb: clamp01(oppThreat.totalProb),
    oppJokboOneAwayProb: clamp01(oppThreat.oneAwayProb),
    oppGwangThreatProb: clamp01(oppThreat.gwangThreatProb)
  };
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

function policyContextKey(trace, decisionType) {
  const dc = trace.dc || {};
  const sp = trace.sp || {};
  const deckBucket = Math.floor((dc.deckCount || 0) / 3);
  const rawPhase = dc.phase;
  const phaseCode = Number.isFinite(Number(rawPhase))
    ? Math.floor(Number(rawPhase))
    : tracePhaseCode(rawPhase);
  const handSelf = dc.handCountSelf || 0;
  const handOpp = dc.handCountOpp || 0;
  const goSelf = dc.goCountSelf || 0;
  const goOpp = dc.goCountOpp || 0;
  const carry = Math.max(1, Math.floor(Number(dc.carryOverMultiplier || 1)));
  const shakeSelf = Math.min(3, Math.floor(Number(dc.shakeCountSelf || 0)));
  const shakeOpp = Math.min(3, Math.floor(Number(dc.shakeCountOpp || 0)));
  let cands = Number(trace.cc ?? sp.candidateCount ?? 0);
  if (!Number.isFinite(cands) || cands <= 0) {
    cands = (sp.cards || sp.boardCardIds || sp.options || []).length;
  }
  cands = Math.max(0, Math.floor(cands));
  return [
    `dt=${decisionType}`,
    `ph=${phaseCode}`,
    `o=${trace.o || "?"}`,
    `db=${deckBucket}`,
    `hs=${handSelf}`,
    `ho=${handOpp}`,
    `gs=${goSelf}`,
    `go=${goOpp}`,
    `cm=${carry}`,
    `ss=${shakeSelf}`,
    `so=${shakeOpp}`,
    `cc=${cands}`
  ].join("|");
}

function policyProb(model, sample, choice) {
  const alpha = Number(model?.alpha ?? 1.0);
  const dt = sample.decisionType;
  const candidates = sample.candidates || [];
  const ck = sample.contextKey;
  const k = Math.max(1, candidates.length);

  const dtContextCounts = model?.context_counts?.[dt] || {};
  const dtContextTotals = model?.context_totals?.[dt] || {};
  const ctxCounts = dtContextCounts?.[ck];
  if (ctxCounts) {
    const total = Number(dtContextTotals?.[ck] || 0);
    return (Number(ctxCounts?.[choice] || 0) + alpha) / (total + alpha * k);
  }

  const dtGlobal = model?.global_counts?.[dt] || {};
  let total = 0;
  for (const c of candidates) total += Number(dtGlobal?.[c] || 0);
  return (Number(dtGlobal?.[choice] || 0) + alpha) / (total + alpha * k);
}

function valueTokens(dc, order, decisionType, actionLabel) {
  const isFirst = order === "first" ? 1 : 0;
  return [
    `phase=${dc.phase || "?"}`,
    `order=${order || "?"}`,
    `decision_type=${decisionType}`,
    `action=${actionLabel || "?"}`,
    `deck_bucket=${Math.floor((dc.deckCount || 0) / 3)}`,
    `self_hand=${Math.floor(dc.handCountSelf || 0)}`,
    `opp_hand=${Math.floor(dc.handCountOpp || 0)}`,
    `self_go=${Math.floor(dc.goCountSelf || 0)}`,
    `opp_go=${Math.floor(dc.goCountOpp || 0)}`,
    `is_first_attacker=${isFirst}`
  ];
}

function valueNumeric(dc, candidateCount, order) {
  const isFirst = order === "first" ? 1 : 0;
  return {
    deck_count: Number(dc.deckCount || 0),
    hand_self: Number(dc.handCountSelf || 0),
    hand_opp: Number(dc.handCountOpp || 0),
    go_self: Number(dc.goCountSelf || 0),
    go_opp: Number(dc.goCountOpp || 0),
    is_first_attacker: isFirst,
    cand_count: Number(candidateCount || 0),
    immediate_reward: 0
  };
}

function valuePredict(model, sample) {
  if (!model || !Array.isArray(model.weights)) return 0;
  const dim = Number(model.dim || 0);
  if (dim <= 0) return 0;
  const w = model.weights;
  const b = Number(model.bias || 0);
  const scale = model.numeric_scale || {};
  let total = b;
  for (const tok of sample.tokens) {
    const i = stableHashCached(`tok:${tok}`, dim);
    total += Number(w[i] || 0);
  }
  for (const [k, v] of Object.entries(sample.numeric || {})) {
    const i = stableHashCached(`num:${k}`, dim);
    const denom = Math.max(1e-9, Number(scale[k] || 1.0));
    total += Number(w[i] || 0) * (Number(v) / denom);
  }
  return total;
}

function modelCacheKey(actor, cfg, decisionType, contextKey, candidates) {
  const p = cfg.policyModelPath || "-";
  const v = cfg.valueModelPath || "-";
  return `${actor}|${p}|${v}|${decisionType}|${contextKey}|${candidates.join(",")}`;
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

function modelSelectCandidate(state, actor, cfg) {
  const sp = selectPool(state, actor);
  const cards = sp.cards || null;
  const boardCardIds = sp.boardCardIds || null;
  const options = sp.options || null;
  const candidates = cards || boardCardIds || options || [];
  if (!candidates.length) return null;

  const decisionType = cards ? "play" : boardCardIds ? "match" : "option";
  const order = actor === state.startingTurnKey ? "first" : "second";
  const dc = decisionContext(state, actor);
  const traceLike = { o: order, dc, sp: { cards, boardCardIds, options } };
  const contextKey = policyContextKey(traceLike, decisionType);
  const cacheKey = modelCacheKey(actor, cfg, decisionType, contextKey, candidates);
  const cached = decisionInferenceCache.get(cacheKey);
  if (cached) return { decisionType, candidate: cached, sp };

  const baseSample = { decisionType, candidates, contextKey };
  const useValue = !!cfg.valueModel;
  const numeric = useValue ? valueNumeric(dc, candidates.length, order) : null;
  const tokens = useValue ? valueTokens(dc, order, decisionType, "?") : null;

  let bestCandidate = candidates[0];
  let bestValue = Number.NEGATIVE_INFINITY;
  let bestPolicy = -1;
  for (const candidate of candidates) {
    const candidateLabel = decisionType === "option" ? candidate : String(candidate);
    const pp = cfg.policyModel ? policyProb(cfg.policyModel, baseSample, candidateLabel) : 0;
    let vs = 0;
    if (useValue) {
      tokens[3] = `action=${candidateLabel}`;
      vs = valuePredict(cfg.valueModel, { tokens, numeric });
    }
    const cmpPrimary = useValue ? vs : pp;
    const tieBreaker = useValue ? pp : vs;
    const currentPrimary = useValue ? bestValue : bestPolicy;
    const better = cmpPrimary > currentPrimary + 1e-12;
    const tieBetter = Math.abs(cmpPrimary - currentPrimary) <= 1e-12 && tieBreaker > (useValue ? bestPolicy : bestValue);
    if (better || tieBetter) {
      bestCandidate = candidate;
      bestValue = vs;
      bestPolicy = pp;
    }
  }
  setBoundedCache(decisionInferenceCache, cacheKey, bestCandidate, DECISION_CACHE_MAX);
  return { decisionType, candidate: bestCandidate, sp };
}

function applyModelChoice(state, actor, picked) {
  if (!picked) return state;
  const c = picked.candidate;
  if (picked.decisionType === "play") return playTurn(state, c);
  if (picked.decisionType === "match") return chooseMatch(state, c);
  if (picked.decisionType !== "option") return state;

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
  if (!isTrainMode || modelOnly) return null;
  const policy = String(cfg?.fallbackPolicy || "").toLowerCase();
  if (!policy.startsWith("heuristic")) return null;

  const dc = decisionContext(state, actor, { includeCardLists: false });
  const scoreDiff = Number(dc.currentScoreSelf || 0) - Number(dc.currentScoreOpp || 0);
  const comebackMode = actor !== state.startingTurnKey && scoreDiff <= -5;
  const exploreRate = comebackMode ? trainExploreRateComeback : trainExploreRate;
  if (Math.random() >= exploreRate) return null;

  const sp = selectPool(state, actor);
  const cards = sp.cards || null;
  const boardCardIds = sp.boardCardIds || null;
  const options = sp.options || null;
  const candidates = cards || boardCardIds || options || [];
  if (!candidates.length) return null;
  const decisionType = cards ? "play" : boardCardIds ? "match" : "option";
  const candidate = candidates[Math.floor(Math.random() * candidates.length)];
  return { decisionType, candidate, exploratory: true };
}

function loadJsonModel(modelPath, label) {
  if (!modelPath) return null;
  const full = path.resolve(modelPath);
  if (!fs.existsSync(full)) {
    throw new Error(`${label} not found: ${modelPath}`);
  }
  return JSON.parse(fs.readFileSync(full, "utf8"));
}

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

function normalizeVolatilityEventTag(tag) {
  const t = String(tag || "").trim().toUpperCase();
  if (!t || t === "NORMAL") return "";
  if (t === "PPUK" || t === "JJOB" || t === "DDADAK" || t === "PANSSEUL" || t === "SSUL") {
    return t;
  }
  return t;
}

function volatilityEventKey(tag) {
  const t = normalizeVolatilityEventTag(tag);
  if (!t) return "";
  if (t === "PPUK") return "ppuk";
  if (t === "JJOB") return "jjob";
  if (t === "DDADAK") return "ddadak";
  if (t === "PANSSEUL") return "pansseul";
  if (t === "SSUL") return "ssul";
  return "other";
}

function normalizedSpecialEventKeys(turn) {
  const out = new Set();
  for (const tag of specialEventTags(turn)) {
    const key = volatilityEventKey(tag);
    if (key) out.add(key);
  }
  return [...out];
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

function scoreSwingFromDecisionContext(beforeDc, afterDc) {
  const beforeDiff = Number(beforeDc?.currentScoreSelf || 0) - Number(beforeDc?.currentScoreOpp || 0);
  const afterDiff = Number(afterDc?.currentScoreSelf || 0) - Number(afterDc?.currentScoreOpp || 0);
  return afterDiff - beforeDiff;
}

function isVolatilityTurnSignal(actionType, eventKeys, bombShifted) {
  return (
    actionType === "choose_shaking_yes" ||
    actionType === "declare_bomb" ||
    !!bombShifted ||
    (Array.isArray(eventKeys) && eventKeys.length > 0)
  );
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
  const beforeLen = rec.triggers.length;
  pushTrigger(rec.triggers, "terminalContext", 0, 0);
  if (rec.triggers.length > beforeLen) {
    if (!Array.isArray(rec.data?.tg)) rec.data.tg = [];
    if (!rec.data.tg.includes("terminalContext")) rec.data.tg.push("terminalContext");
  }
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
  const scoreSwing = scoreSwingFromDecisionContext(beforeDc, afterDc);

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

  const eventKeys = normalizedSpecialEventKeys(turn);
  if (eventKeys.length) {
    if (eventKeys.includes("ppuk")) pushTrigger(out, "specialEventPpuk", 1, 1);
    if (eventKeys.includes("jjob")) pushTrigger(out, "specialEventJjob", 1, 0);
    if (eventKeys.includes("ddadak")) pushTrigger(out, "specialEventDdadak", 1, 0);
    if (eventKeys.includes("pansseul")) pushTrigger(out, "specialEventPansseul", 1, 0);
    if (eventKeys.includes("ssul")) pushTrigger(out, "specialEventSsul", 1, 0);
    if (eventKeys.some((k) => k !== "ppuk")) pushTrigger(out, "specialEventCore", 1, 0);
    if (eventKeys.includes("other")) pushTrigger(out, "specialEventOther", 1, 0);
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

  const volatilityTurn = isVolatilityTurnSignal(actionType, eventKeys, bombShifted);
  if (volatilityTurn) {
    pushTrigger(out, "volatilityTurn", 0, 0);
    if (scoreSwing >= 1 && !eventKeys.includes("ppuk")) {
      pushTrigger(out, "volatilityExploit", 1, 1);
    }
    if (eventKeys.includes("ppuk") || scoreSwing <= -1 || riskDirection === "up") {
      pushTrigger(out, "volatilitySelfHarm", 1, 1);
    }
  }

  return out;
}

function keepImportantWithContext(records, contextRadius = 0) {
  if (!Array.isArray(records) || !records.length) return [];
  const extraRadius = Math.max(0, Math.floor(Number(contextRadius || 0)));
  let anyTrigger = false;
  let keep = new Array(records.length).fill(false);
  for (let i = 0; i < records.length; i += 1) {
    const triggers = Array.isArray(records[i]?.triggers) ? records[i].triggers : [];
    if (!triggers.length) continue;
    anyTrigger = true;
    for (const tr of triggers) {
      const back = Math.max(0, Math.floor(Number(tr?.back || 0)));
      const forward = Math.max(0, Math.floor(Number(tr?.forward || 0)));
      const lo = Math.max(0, i - back);
      const hi = Math.min(records.length - 1, i + forward);
      for (let j = lo; j <= hi; j += 1) keep[j] = true;
    }
  }
  if (!anyTrigger) {
    return records.map((r) => r.data);
  }

  if (extraRadius > 0) {
    const expanded = keep.slice();
    for (let i = 0; i < keep.length; i += 1) {
      if (!keep[i]) continue;
      const lo = Math.max(0, i - extraRadius);
      const hi = Math.min(keep.length - 1, i + extraRadius);
      for (let j = lo; j <= hi; j += 1) expanded[j] = true;
    }
    keep = expanded;
  }

  return records.filter((_, i) => keep[i]).map((r) => r.data);
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

function createVolatilityGameSideStats() {
  return {
    volatilityTurnCount: 0,
    shakingYesCount: 0,
    bombDeclareCount: 0,
    bombCountShiftCount: 0,
    specialEventTurnCount: 0,
    scoreGainCount: 0,
    scoreLossCount: 0,
    scoreFlatCount: 0,
    scoreSwingSum: 0,
    riskUpCount: 0,
    selfHarmCount: 0,
    eventByTag: Object.fromEntries(VOLATILITY_EVENT_KEYS.map((k) => [k, 0]))
  };
}

function createVolatilityAggregateSideStats() {
  return createVolatilityGameSideStats();
}

function analyzeVolatilityBundleFromContext(contextByTurnNo, kibo, firstTurnKey, secondTurnKey) {
  const out = {
    bySide: {
      [SIDE_MY]: createVolatilityGameSideStats(),
      [SIDE_YOUR]: createVolatilityGameSideStats()
    }
  };
  const turns = Array.isArray(kibo) ? kibo.filter((e) => e?.type === "turn_end") : [];
  for (const turn of turns) {
    const actor = turn?.actor;
    if (!actor) continue;
    const side = actorToSide(actor, firstTurnKey);
    const sideStats = out.bySide[side];
    if (!sideStats) continue;

    const turnNo = Number(turn?.turnNo || 0);
    const rec = contextByTurnNo instanceof Map ? contextByTurnNo.get(turnNo) : null;
    const beforeDc = rec?.before?.decisionContext || null;
    const afterDc = rec?.after?.decisionContext || null;
    const actionType = String(turn?.action?.type || "");
    const eventKeys = normalizedSpecialEventKeys(turn);
    const bombShifted = beforeDc && afterDc ? hasBombShift(beforeDc, afterDc) : false;
    if (!isVolatilityTurnSignal(actionType, eventKeys, bombShifted)) continue;

    sideStats.volatilityTurnCount += 1;
    if (actionType === "choose_shaking_yes") sideStats.shakingYesCount += 1;
    if (actionType === "declare_bomb") sideStats.bombDeclareCount += 1;
    if (bombShifted) sideStats.bombCountShiftCount += 1;
    if (eventKeys.length) sideStats.specialEventTurnCount += 1;

    for (const key of eventKeys) {
      if (sideStats.eventByTag[key] == null) sideStats.eventByTag[key] = 0;
      sideStats.eventByTag[key] += 1;
    }

    let scoreSwing = 0;
    if (beforeDc && afterDc) {
      scoreSwing = scoreSwingFromDecisionContext(beforeDc, afterDc);
      sideStats.scoreSwingSum += scoreSwing;
      if (scoreSwing > 0) sideStats.scoreGainCount += 1;
      else if (scoreSwing < 0) sideStats.scoreLossCount += 1;
      else sideStats.scoreFlatCount += 1;
      if (bakRiskShiftDirection(beforeDc, afterDc) === "up") {
        sideStats.riskUpCount += 1;
      }
    }

    const ppukEvent = eventKeys.includes("ppuk");
    const selfHarmBySwing = beforeDc && afterDc ? scoreSwing <= -1 : false;
    const selfHarmByRisk = beforeDc && afterDc ? bakRiskShiftDirection(beforeDc, afterDc) === "up" : false;
    if (ppukEvent || selfHarmBySwing || selfHarmByRisk) {
      sideStats.selfHarmCount += 1;
    }
  }
  return out;
}

function accumulateVolatilityBundleSummary(aggregateRef, volatilityStats) {
  if (!aggregateRef?.volatilityBundle || !volatilityStats?.bySide) return;
  for (const side of [SIDE_MY, SIDE_YOUR]) {
    const src = volatilityStats.bySide[side];
    const dst = aggregateRef.volatilityBundle.bySide[side];
    if (!src || !dst) continue;
    dst.volatilityTurnCount += Number(src.volatilityTurnCount || 0);
    dst.shakingYesCount += Number(src.shakingYesCount || 0);
    dst.bombDeclareCount += Number(src.bombDeclareCount || 0);
    dst.bombCountShiftCount += Number(src.bombCountShiftCount || 0);
    dst.specialEventTurnCount += Number(src.specialEventTurnCount || 0);
    dst.scoreGainCount += Number(src.scoreGainCount || 0);
    dst.scoreLossCount += Number(src.scoreLossCount || 0);
    dst.scoreFlatCount += Number(src.scoreFlatCount || 0);
    dst.scoreSwingSum += Number(src.scoreSwingSum || 0);
    dst.riskUpCount += Number(src.riskUpCount || 0);
    dst.selfHarmCount += Number(src.selfHarmCount || 0);
    for (const key of VOLATILITY_EVENT_KEYS) {
      dst.eventByTag[key] += Number(src.eventByTag?.[key] || 0);
    }
  }
}

function volatilityBundleReportBySide(bundleSide, games) {
  const volTurns = Number(bundleSide?.volatilityTurnCount || 0);
  const scoreGain = Number(bundleSide?.scoreGainCount || 0);
  const scoreLoss = Number(bundleSide?.scoreLossCount || 0);
  const scoreFlat = Number(bundleSide?.scoreFlatCount || 0);
  const selfHarm = Number(bundleSide?.selfHarmCount || 0);
  const riskUp = Number(bundleSide?.riskUpCount || 0);
  const eventByTagPerGame = {};
  const eventByTagCount = {};
  for (const key of VOLATILITY_EVENT_KEYS) {
    const cnt = Number(bundleSide?.eventByTag?.[key] || 0);
    eventByTagCount[key] = cnt;
    eventByTagPerGame[key] = cnt / Math.max(1, games);
  }
  const exploitRate = scoreGain / Math.max(1, volTurns);
  const selfHarmRate = selfHarm / Math.max(1, volTurns);
  return {
    volatilityTurnsPerGame: volTurns / Math.max(1, games),
    shakingYesPerGame: Number(bundleSide?.shakingYesCount || 0) / Math.max(1, games),
    bombDeclarePerGame: Number(bundleSide?.bombDeclareCount || 0) / Math.max(1, games),
    bombShiftPerGame: Number(bundleSide?.bombCountShiftCount || 0) / Math.max(1, games),
    specialEventTurnPerGame: Number(bundleSide?.specialEventTurnCount || 0) / Math.max(1, games),
    eventByTagPerGame,
    positiveOutcomeRate: exploitRate,
    negativeOutcomeRate: scoreLoss / Math.max(1, volTurns),
    flatOutcomeRate: scoreFlat / Math.max(1, volTurns),
    selfHarmRate,
    riskUpAfterVolatilityRate: riskUp / Math.max(1, volTurns),
    exploitMinusSelfHarm: exploitRate - selfHarmRate,
    averageScoreSwingPerVolatilityTurn: volTurns > 0 ? Number(bundleSide?.scoreSwingSum || 0) / volTurns : null,
    counts: {
      volatilityTurnCount: volTurns,
      shakingYesCount: Number(bundleSide?.shakingYesCount || 0),
      bombDeclareCount: Number(bundleSide?.bombDeclareCount || 0),
      bombCountShiftCount: Number(bundleSide?.bombCountShiftCount || 0),
      specialEventTurnCount: Number(bundleSide?.specialEventTurnCount || 0),
      scoreGainCount: scoreGain,
      scoreLossCount: scoreLoss,
      scoreFlatCount: scoreFlat,
      selfHarmCount: selfHarm,
      riskUpCount: riskUp,
      eventByTagCount
    }
  };
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

  const actionType = String(action.type || "");
  const scoreSwing = scoreSwingFromDecisionContext(b, a);
  const riskDirection = bakRiskShiftDirection(b, a);
  const eventKeys = normalizedSpecialEventKeys(t);
  const bombShifted = hasBombShift(b, a);
  let volatility = 0;

  if (actionType === "choose_shaking_yes") {
    if (scoreSwing > 0) volatility += 0.1;
    else if (scoreSwing < 0) volatility -= 0.1;
    else volatility += 0.015;
  }
  if (actionType === "declare_bomb") {
    if (scoreSwing > 0) volatility += 0.14;
    else if (scoreSwing < 0) volatility -= 0.14;
  }
  if (bombShifted && actionType !== "declare_bomb") {
    if (scoreSwing > 0) volatility += 0.05;
    else if (scoreSwing < 0) volatility -= 0.05;
  }

  for (const key of eventKeys) {
    if (key === "ppuk") volatility -= 0.24;
    else if (key === "jjob") volatility += 0.08;
    else if (key === "ddadak") volatility += 0.1;
    else if (key === "pansseul") volatility += 0.12;
    else if (key === "ssul") volatility += 0.07;
    else volatility += 0.03;
  }
  if (riskDirection === "up") volatility -= 0.08;
  if (riskDirection === "down") volatility += 0.05;
  if (scoreSwing <= -2) volatility -= 0.1;
  if (scoreSwing >= 2) volatility += 0.06;
  strategic += volatility;

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
      },
      volatility: {
        actionType,
        scoreSwing,
        riskDirection: riskDirection || "flat",
        eventKeys,
        bombShifted: !!bombShifted,
        reward: volatility
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

function compactDecisionContextForTrace(dc) {
  if (!dc) return null;
  return {
    phase: tracePhaseCode(dc.phase),
    deckCount: Math.max(0, Math.floor(Number(dc.deckCount || 0))),
    handCountSelf: Math.max(0, Math.floor(Number(dc.handCountSelf || 0))),
    handCountOpp: Math.max(0, Math.floor(Number(dc.handCountOpp || 0))),
    goCountSelf: Math.max(0, Math.floor(Number(dc.goCountSelf || 0))),
    goCountOpp: Math.max(0, Math.floor(Number(dc.goCountOpp || 0))),
    carryOverMultiplier: Math.max(1, Math.floor(Number(dc.carryOverMultiplier || 1))),
    piBakRisk: Number(dc.piBakRisk || 0) ? 1 : 0,
    gwangBakRisk: Number(dc.gwangBakRisk || 0) ? 1 : 0,
    mongBakRisk: Number(dc.mongBakRisk || 0) ? 1 : 0,
    oppJokboThreatProb: toPercentInt(dc.oppJokboThreatProb),
    oppJokboOneAwayProb: toPercentInt(dc.oppJokboOneAwayProb),
    oppGwangThreatProb: toPercentInt(dc.oppGwangThreatProb),
    currentScoreSelf: Math.floor(Number(dc.currentScoreSelf || 0)),
    currentScoreOpp: Math.floor(Number(dc.currentScoreOpp || 0))
  };
}

function compactSelectionPoolForTrace(sp) {
  if (!sp) return null;
  const out = {};
  if (Array.isArray(sp.cards) && sp.cards.length) out.cards = sp.cards;
  if (Array.isArray(sp.boardCardIds) && sp.boardCardIds.length) out.boardCardIds = sp.boardCardIds;
  if (Array.isArray(sp.options) && sp.options.length) out.options = sp.options;
  return out;
}

function buildDecisionTrace(
  kibo,
  winner,
  contextByTurnNo,
  firstTurnKey,
  secondTurnKey,
  policyBySideRef,
  options = {}
) {
  const myTurnOnly = !!options.myTurnOnly;
  const importantOnly = options.importantOnly !== false;
  const contextRadius = Number(options.contextRadius || 0);
  const goStopPlus2 = !!options.goStopPlus2;
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
    const candidateCount = traceCandidateCount(before?.selectionPool || null);
    if (candidateCount < 1) continue;
    const compactDc = compactDecisionContextForTrace(before?.decisionContext || null);
    const order = actorSide === SIDE_MY ? "first" : "second";
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
    const tg = triggerNames(triggers);
    records.push({
      seq: Number(turnCtx?.seq || t.turnNo || 0),
      triggers,
      data: {
      t: t.turnNo,
      a: actorSide,
      o: order,
      dt: decisionType,
      cc: candidateCount,
      at: action.type || "unknown",
      c: action.card?.id || null,
      s: action.selectedBoardCard?.id || null,
      ir: immediateReward,
      dc: compactDc,
      ...(tg.length ? { tg } : {})
      }
    });
  }
  for (const ev of optionEvents) {
    const actor = ev?.actor;
    const actorSide = actorToSide(actor, firstTurnKey);
    if (myTurnOnly && actorSide !== SIDE_MY) continue;
    const candidateCount = Math.max(0, Math.floor(Number(ev?.candidateCount || 0)));
    if (candidateCount < 1) continue;
    const actionType = String(ev?.actionType || "option");
    const beforeDc = ev?.beforeDc || null;
    const afterDc = ev?.afterDc || null;
    const compactDc = compactDecisionContextForTrace(beforeDc);
    const turnNo = Math.max(0, Math.floor(Number(ev?.turnNo || 0)));
    const order = actorSide === SIDE_MY ? "first" : "second";
    const triggerTurn = { action: { type: actionType, matchEvents: [] } };
    const triggers = classifyImportantTurnTriggers({
      decisionType: "option",
      turn: triggerTurn,
      beforeDc,
      afterDc,
      earlyTurnForced: false,
      goStopPlus2
    });
    const tg = triggerNames(triggers);
    records.push({
      seq: Number(ev?.seq || 0),
      triggers,
      data: {
        t: turnNo,
        a: actorSide,
        o: order,
        dt: "option",
        cc: candidateCount,
        at: actionType,
        c: null,
        s: null,
        ir: 0,
        dc: compactDc,
        ...(tg.length ? { tg } : {})
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
  return importantOnly ? keepImportantWithContext(records, contextRadius) : records.map((r) => r.data);
}

function buildDecisionTraceTrain(
  kibo,
  winner,
  firstTurnKey,
  secondTurnKey,
  contextByTurnNo,
  options = {}
) {
  const myTurnOnly = !!options.myTurnOnly;
  const importantOnly = options.importantOnly !== false;
  const contextRadius = Number(options.contextRadius || 0);
  const goStopPlus2 = !!options.goStopPlus2;
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
    const candidateCount = traceCandidateCount(before?.selectionPool || null);
    if (candidateCount < 1) continue;
    const compactDc = compactDecisionContextForTrace(before?.decisionContext || null);
    const order = actorSide === SIDE_MY ? "first" : "second";
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
    const tg = triggerNames(triggers);
    records.push({
      seq: Number(turnCtx?.seq || t.turnNo || 0),
      triggers,
      data: {
      t: t.turnNo,
      a: actorSide,
      o: order,
      dt: decisionType,
      cc: candidateCount,
      at: action.type || "unknown",
      c: action.card?.id || null,
      s: action.selectedBoardCard?.id || null,
      ir: immediateReward,
      dc: compactDc,
      ...(tg.length ? { tg } : {})
      }
    });
  }
  for (const ev of optionEvents) {
    const actor = ev?.actor;
    const actorSide = actorToSide(actor, firstTurnKey);
    if (myTurnOnly && actorSide !== SIDE_MY) continue;
    const candidateCount = Math.max(0, Math.floor(Number(ev?.candidateCount || 0)));
    if (candidateCount < 1) continue;
    const actionType = String(ev?.actionType || "option");
    const beforeDc = ev?.beforeDc || null;
    const afterDc = ev?.afterDc || null;
    const compactDc = compactDecisionContextForTrace(beforeDc);
    const turnNo = Math.max(0, Math.floor(Number(ev?.turnNo || 0)));
    const order = actorSide === SIDE_MY ? "first" : "second";
    const triggerTurn = { action: { type: actionType, matchEvents: [] } };
    const triggers = classifyImportantTurnTriggers({
      decisionType: "option",
      turn: triggerTurn,
      beforeDc,
      afterDc,
      earlyTurnForced: false,
      goStopPlus2
    });
    const tg = triggerNames(triggers);
    records.push({
      seq: Number(ev?.seq || 0),
      triggers,
      data: {
        t: turnNo,
        a: actorSide,
        o: order,
        dt: "option",
        cc: candidateCount,
        at: actionType,
        c: null,
        s: null,
        ir: 0,
        dc: compactDc,
        ...(tg.length ? { tg } : {})
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
  return importantOnly ? keepImportantWithContext(records, contextRadius) : records.map((r) => r.data);
}

function buildDecisionTraceDelta(
  kibo,
  winner,
  firstTurnKey,
  secondTurnKey,
  contextByTurnNo,
  policyBySideRef,
  options = {}
) {
  const myTurnOnly = !!options.myTurnOnly;
  const turns = (kibo || []).filter((e) => e.type === "turn_end");
  let prevDeck = null;
  let prevHand = { [firstTurnKey]: null, [secondTurnKey]: null };
  const out = [];
  for (const t of turns) {
    const actor = t.actor;
    const actorSide = actorToSide(actor, firstTurnKey);
    if (myTurnOnly && actorSide !== SIDE_MY) continue;
    const action = t.action || {};
    const deckAfter = t.deckCount ?? 0;
    const handSelfAfter = handCountFromTurn(t, actor);
    const opp = actor === firstTurnKey ? secondTurnKey : firstTurnKey;
    const handOppAfter = handCountFromTurn(t, opp);
    const deckDelta = prevDeck == null ? null : deckAfter - prevDeck;
    const handSelfDelta = prevHand[actor] == null ? null : handSelfAfter - prevHand[actor];
    const handOppDelta = prevHand[opp] == null ? null : handOppAfter - prevHand[opp];
    const turnCtx = contextByTurnNo.get(t.turnNo) || null;
    const before = turnCtx?.before || null;
    const candidateCount = traceCandidateCount(before?.selectionPool || null);
    prevDeck = deckAfter;
    prevHand = {
      [firstTurnKey]: handCountFromTurn(t, firstTurnKey),
      [secondTurnKey]: handCountFromTurn(t, secondTurnKey)
    };
    if (candidateCount < 2) continue;

    out.push({
      t: t.turnNo,
      a: actorSide,
      o: actorSide === SIDE_MY ? "first" : "second",
      at: action.type || "unknown",
      c: action.card?.id || null,
      s: action.selectedBoardCard?.id || null,
      delta: {
        deck: deckDelta,
        handSelf: handSelfDelta,
        handOpp: handOppDelta,
        piSteal: t.steals?.pi || 0,
        goldSteal: t.steals?.gold || 0,
        capHand: (action.captureBySource?.hand || []).map((x) => x.id),
        capFlip: (action.captureBySource?.flip || []).map((x) => x.id),
        events: (action.matchEvents || [])
          .filter((m) => m.eventTag && m.eventTag !== "NORMAL")
          .map((m) => `${m.source}:${m.eventTag}`)
      },
      reasoning: {
        policy: policyBySideRef[actorSide] || DEFAULT_POLICY,
        candidatesCount: candidateCount,
        evaluation: null
      }
    });
  }
  return out;
}

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

function buildTrainReport() {
  return {
    simulator: SIMULATOR_NAME,
    logMode,
    games: aggregate.games,
    completed: aggregate.completed,
    winners: aggregate.winners,
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
    volatilityBundle: {
      [SIDE_MY]: volatilityBundleReportBySide(aggregate.volatilityBundle.bySide[SIDE_MY], aggregate.games),
      [SIDE_YOUR]: volatilityBundleReportBySide(
        aggregate.volatilityBundle.bySide[SIDE_YOUR],
        aggregate.games
      )
    },
    primaryMetric: "averageGoldDeltaMySide"
  };
}

function buildFullReport() {
  return {
    simulator: SIMULATOR_NAME,
    logMode,
    catalogPath: sharedCatalogPath,
    games: aggregate.games,
    completed: aggregate.completed,
    winners: aggregate.winners,
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
    volatilityBundle: {
      [SIDE_MY]: volatilityBundleReportBySide(aggregate.volatilityBundle.bySide[SIDE_MY], aggregate.games),
      [SIDE_YOUR]: volatilityBundleReportBySide(
        aggregate.volatilityBundle.bySide[SIDE_YOUR],
        aggregate.games
      )
    },
    primaryMetric: "averageGoldDeltaMySide",
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

async function writeLine(writer, line) {
  if (writer.write(`${JSON.stringify(line)}\n`)) return;
  await once(writer, "drain");
}

function executeActorTurn(state, actor, cfg) {
  const canUseModel = !!(cfg?.policyModel || cfg?.valueModel);
  if (canUseModel) {
    const picked = modelSelectCandidate(state, actor, cfg);
    if (picked) {
      const next = applyModelChoice(state, actor, picked);
      if (next !== state) return next;
    }
    if (modelOnly) {
      throw new Error(`model-only decision failed for actor=${actor}, phase=${state.phase}`);
    }
  }
  if (modelOnly) {
    throw new Error(`model-only missing model for actor=${actor}`);
  }
  const exploratoryPick = maybePickExplorationChoice(state, actor, cfg);
  if (exploratoryPick) {
    const explored = applyModelChoice(state, actor, exploratoryPick);
    if (explored !== state) return explored;
  }
  return aiPlay(state, actor, {
    source: "heuristic",
    heuristicPolicy: cfg?.fallbackPolicy || DEFAULT_POLICY
  });
}

async function run() {
  const outStream = fs.createWriteStream(outPath, { encoding: "utf8" });
  const probeState = initGame("A", createSeededRng("side-probe-seed"), {
    carryOverMultiplier: 1,
    kiboDetail: "lean"
  });
  const [actorA, actorB] = actorPairFromState(probeState);
  const firstTurnPlan = createBalancedFirstTurnPlan(games, actorA, actorB);
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
      kiboDetail: useLeanKibo ? "lean" : "full",
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

    const contextByTurnNo = needsDecisionTrace ? new Map() : null;
    const optionTraceEvents = needsDecisionTrace ? [] : null;
    let steps = 0;
    const maxSteps = 4000;
    while (state.phase !== "resolution" && steps < maxSteps) {
      const actor = getActionPlayerKey(state);
      if (!actor) break;
      const beforeContext = needsDecisionTrace
        ? {
            actor,
            decisionContext: decisionContext(state, actor, { includeCardLists: false }),
            selectionPool: selectPool(state, actor, { includeCardLists: false })
          }
        : null;
      const prevTurnSeq = state.turnSeq || 0;
      const prevKiboSeq = state.kiboSeq || 0;
      const traceSeqBase = (steps + 1) * 2;
      const actorSide = actorToSide(actor, firstTurnKey);
      const next = executeActorTurn(state, actor, sideConfig[actorSide]);
      if (next === state) break;
      const nextTurnSeq = next.turnSeq || 0;
      if (needsDecisionTrace) {
        const afterDc = decisionContext(next, actor, { includeCardLists: false });
        const beforeSp = beforeContext?.selectionPool || null;
        const beforeDecisionType = String(beforeSp?.decisionType || "");
        if (beforeDecisionType === "option") {
          optionTraceEvents.push({
            seq: traceSeqBase,
            turnNo: nextTurnSeq > prevTurnSeq ? nextTurnSeq : prevTurnSeq,
            actor,
            candidateCount: traceCandidateCount(beforeSp),
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
      }
      state = next;
      steps += 1;
    }

    const winner = state.result?.winner || "unknown";
    const kibo = state.kibo || [];
    const decisionTrace = needsDecisionTrace
      ? isDeltaMode
        ? buildDecisionTraceDelta(
            kibo,
            winner,
            firstTurnKey,
            secondTurnKey,
            contextByTurnNo,
            agentLabelBySide,
            {
              myTurnOnly: traceMyTurnOnly,
              importantOnly: traceImportantOnly,
              contextRadius: traceContextRadius,
              goStopPlus2: traceGoStopPlus2,
              optionEvents: optionTraceEvents
            }
          )
        : isTrainMode
        ? buildDecisionTraceTrain(
            kibo,
            winner,
            firstTurnKey,
            secondTurnKey,
            contextByTurnNo,
            {
              myTurnOnly: traceMyTurnOnly,
              importantOnly: traceImportantOnly,
              contextRadius: traceContextRadius,
              goStopPlus2: traceGoStopPlus2,
              optionEvents: optionTraceEvents
            }
          )
        : buildDecisionTrace(
            kibo,
            winner,
            contextByTurnNo,
            firstTurnKey,
            secondTurnKey,
            agentLabelBySide,
            {
              myTurnOnly: traceMyTurnOnly,
              importantOnly: traceImportantOnly,
              contextRadius: traceContextRadius,
              goStopPlus2: traceGoStopPlus2,
              optionEvents: optionTraceEvents
            }
          )
      : null;
    const volatilityBundleGame =
      needsDecisionTrace && contextByTurnNo
        ? analyzeVolatilityBundleFromContext(contextByTurnNo, kibo, firstTurnKey, secondTurnKey)
        : {
            bySide: {
              [SIDE_MY]: createVolatilityGameSideStats(),
              [SIDE_YOUR]: createVolatilityGameSideStats()
            }
          };
    const mySideActor = firstTurnKey;
    const yourSideActor = secondTurnKey;
    const mySideScore = Number(state.result?.[mySideActor]?.total || 0);
    const yourSideScore = Number(state.result?.[yourSideActor]?.total || 0);
    const mySideGold = Number(state.players?.[mySideActor]?.gold || 0);
    const yourSideGold = Number(state.players?.[yourSideActor]?.gold || 0);
    const mySideBankrupt = mySideGold <= 0;
    const yourSideBankrupt = yourSideGold <= 0;
    const sessionReset = mySideBankrupt || yourSideBankrupt;
    const goldDeltaMy = mySideGold - initialGoldMy;
    const goldDeltaYour = yourSideGold - initialGoldYour;
    const goldDeltaMyRatio = goldDeltaMy / Math.max(1, initialGoldMy);
    const goldDeltaMyNorm = Math.tanh(goldDeltaMyRatio);

    if (sessionReset) {
      sessionGold[SIDE_MY] = STARTING_GOLD;
      sessionGold[SIDE_YOUR] = STARTING_GOLD;
    } else {
      sessionGold[SIDE_MY] = mySideGold;
      sessionGold[SIDE_YOUR] = yourSideGold;
    }

    const winnerSide =
      winner === "draw" || winner === "unknown" ? winner : actorToSide(winner, firstTurnKey);

    if (isTrainMode) {
      const line = {
        simulator: SIMULATOR_NAME,
        game: i + 1,
        seed,
        steps,
        completed: state.phase === "resolution",
        logMode,
        firstAttackerSide: SIDE_MY,
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
        goldDeltaMy,
        goldDeltaYour,
        goldDeltaMyRatio,
        goldDeltaMyNorm,
        gold: {
          [SIDE_MY]: mySideGold,
          [SIDE_YOUR]: yourSideGold
        },
        bankrupt: {
          [SIDE_MY]: mySideBankrupt,
          [SIDE_YOUR]: yourSideBankrupt
        },
        sessionReset,
        volatilityBundle: volatilityBundleGame,
        decision_trace: decisionTrace
      };
      baseSummary({
        completed: line.completed,
        winner: line.winner,
        mySideScore,
        yourSideScore,
        mySideGold,
        yourSideGold
      });
      accumulateVolatilityBundleSummary(aggregate, volatilityBundleGame);
      await writeLine(outStream, line);
      continue;
    }

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
    const policyRef = {
      [SIDE_MY]: agentLabelBySide[SIDE_MY],
      [SIDE_YOUR]: agentLabelBySide[SIDE_YOUR]
    };
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
    accumulateVolatilityBundleSummary(aggregate, volatilityBundleGame);

    const persistedLine = {
      simulator: SIMULATOR_NAME,
      game: i + 1,
      seed,
      steps,
      completed,
      logMode,
      firstAttackerSide: SIDE_MY,
      firstAttackerActor: firstTurnKey,
      policy: policyRef,
      winner: winnerSide,
      score: {
        [SIDE_MY]: mySideScore,
        [SIDE_YOUR]: yourSideScore
      },
      initialGoldMy,
      initialGoldYour,
      finalGoldMy: mySideGold,
      finalGoldYour: yourSideGold,
      goldDeltaMy,
      goldDeltaYour,
      goldDeltaMyRatio,
      goldDeltaMyNorm,
      gold: {
        [SIDE_MY]: mySideGold,
        [SIDE_YOUR]: yourSideGold
      },
      bankrupt: {
        [SIDE_MY]: mySideBankrupt,
        [SIDE_YOUR]: yourSideBankrupt
      },
      sessionReset,
      volatilityBundle: volatilityBundleGame,
      decision_trace: decisionTrace
    };

    await writeLine(outStream, persistedLine);
  }

  outStream.end();
  await once(outStream, "finish");

  const report = isTrainMode ? buildTrainReport() : buildFullReport();
  if (!isTrainMode) ensureSharedCatalog();
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2), "utf8");
  console.log(`done: ${games} games -> ${outPath}`);
  console.log(`report: ${reportPath}`);
  if (!isTrainMode) {
    console.log(`catalog: ${sharedCatalogPath}`);
  }
}

run().catch((err) => {
  console.error(err instanceof Error ? err.message : String(err));
  process.exitCode = 1;
});

