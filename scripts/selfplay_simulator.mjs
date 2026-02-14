import fs from "node:fs";
import path from "node:path";
import { once } from "node:events";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";
import {
  initGame,
  createSeededRng,
  playTurn,
  chooseGo,
  chooseStop,
  choosePresidentStop,
  choosePresidentHold,
  chooseGukjinMode,
  chooseMatch,
  getDeclarableBombMonths,
  getDeclarableShakingMonths
} from "../src/gameEngine.js";
import { buildDeck } from "../src/cards.js";
import { BOT_POLICIES, botPlay } from "../src/bot.js";
import { getActionPlayerKey } from "../src/engineRunner.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DEFAULT_LOG_MODE = "compact";
const SUPPORTED_LOG_MODES = new Set(["compact", "delta", "train"]);
const SUPPORTED_POLICIES = new Set(BOT_POLICIES);
const DEFAULT_POLICY = "random";
const DECISION_CACHE_MAX = 200000;
const HASH_CACHE_MAX = 500000;

const decisionInferenceCache = new Map();
const hashIndexCache = new Map();

function setBoundedCache(cache, key, value, maxEntries) {
  if (cache.size >= maxEntries) cache.clear();
  cache.set(key, value);
}

function normalizePolicyInput(raw) {
  const p = String(raw || DEFAULT_POLICY).trim().toLowerCase();
  if (p === "heuristic" || p === "smart") return "heuristic_v1";
  if (p === "smart_v2") return "heuristic_v2";
  if (p === "random-plus" || p === "plus" || p === "semi_random" || p === "random_plus") {
    return "random_v2";
  }
  return p;
}

function parseArgs(argv) {
  const args = [...argv];
  let games = 1000;
  let outArg = null;
  let logMode = DEFAULT_LOG_MODE;
  let policyHuman = DEFAULT_POLICY;
  let policyAi = DEFAULT_POLICY;
  let modelOnly = false;
  let policyModelHuman = null;
  let policyModelAi = null;
  let valueModelHuman = null;
  let valueModelAi = null;

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
    if (arg === "--policy" && args.length > 0) {
      const p = normalizePolicyInput(args.shift());
      policyHuman = p;
      policyAi = p;
      continue;
    }
    if (arg.startsWith("--policy=")) {
      const p = normalizePolicyInput(arg.split("=", 2)[1]);
      policyHuman = p;
      policyAi = p;
      continue;
    }
    if (arg === "--policy-human" && args.length > 0) {
      policyHuman = normalizePolicyInput(args.shift());
      continue;
    }
    if (arg.startsWith("--policy-human=")) {
      policyHuman = normalizePolicyInput(arg.split("=", 2)[1]);
      continue;
    }
    if (arg === "--policy-ai" && args.length > 0) {
      policyAi = normalizePolicyInput(args.shift());
      continue;
    }
    if (arg.startsWith("--policy-ai=")) {
      policyAi = normalizePolicyInput(arg.split("=", 2)[1]);
      continue;
    }
    if (arg === "--policy-model-human" && args.length > 0) {
      policyModelHuman = String(args.shift()).trim();
      continue;
    }
    if (arg.startsWith("--policy-model-human=")) {
      policyModelHuman = arg.split("=", 2)[1].trim();
      continue;
    }
    if (arg === "--policy-model-ai" && args.length > 0) {
      policyModelAi = String(args.shift()).trim();
      continue;
    }
    if (arg.startsWith("--policy-model-ai=")) {
      policyModelAi = arg.split("=", 2)[1].trim();
      continue;
    }
    if (arg === "--policy-model-a" && args.length > 0) {
      policyModelHuman = String(args.shift()).trim();
      continue;
    }
    if (arg.startsWith("--policy-model-a=")) {
      policyModelHuman = arg.split("=", 2)[1].trim();
      continue;
    }
    if (arg === "--policy-model-b" && args.length > 0) {
      policyModelAi = String(args.shift()).trim();
      continue;
    }
    if (arg.startsWith("--policy-model-b=")) {
      policyModelAi = arg.split("=", 2)[1].trim();
      continue;
    }
    if (arg === "--value-model-human" && args.length > 0) {
      valueModelHuman = String(args.shift()).trim();
      continue;
    }
    if (arg.startsWith("--value-model-human=")) {
      valueModelHuman = arg.split("=", 2)[1].trim();
      continue;
    }
    if (arg === "--value-model-ai" && args.length > 0) {
      valueModelAi = String(args.shift()).trim();
      continue;
    }
    if (arg.startsWith("--value-model-ai=")) {
      valueModelAi = arg.split("=", 2)[1].trim();
      continue;
    }
    if (arg === "--value-model-a" && args.length > 0) {
      valueModelHuman = String(args.shift()).trim();
      continue;
    }
    if (arg.startsWith("--value-model-a=")) {
      valueModelHuman = arg.split("=", 2)[1].trim();
      continue;
    }
    if (arg === "--value-model-b" && args.length > 0) {
      valueModelAi = String(args.shift()).trim();
      continue;
    }
    if (arg.startsWith("--value-model-b=")) {
      valueModelAi = arg.split("=", 2)[1].trim();
      continue;
    }
    if (arg === "--model-only" || arg === "--strict-model-only") {
      modelOnly = true;
      continue;
    }
  }

  if (!SUPPORTED_LOG_MODES.has(logMode)) {
    throw new Error(`Unsupported log mode: ${logMode}. Use one of: compact, delta, train`);
  }
  if (!SUPPORTED_POLICIES.has(policyHuman)) {
    throw new Error(`Unsupported policy-human: ${policyHuman}. Use one of: ${[...SUPPORTED_POLICIES].join(", ")}`);
  }
  if (!SUPPORTED_POLICIES.has(policyAi)) {
    throw new Error(`Unsupported policy-ai: ${policyAi}. Use one of: ${[...SUPPORTED_POLICIES].join(", ")}`);
  }

  return {
    games,
    outArg,
    logMode,
    policyHuman,
    policyAi,
    policyModelHuman,
    policyModelAi,
    valueModelHuman,
    valueModelAi,
    modelOnly
  };
}

const parsed = parseArgs(process.argv.slice(2));
const games = parsed.games;
const outArg = parsed.outArg;
const logMode = parsed.logMode;
const policyHuman = parsed.policyHuman;
const policyAi = parsed.policyAi;
const policyModelHumanPath = parsed.policyModelHuman;
const policyModelAiPath = parsed.policyModelAi;
const valueModelHumanPath = parsed.valueModelHuman;
const valueModelAiPath = parsed.valueModelAi;
const modelOnly = !!parsed.modelOnly;
const isTrainMode = logMode === "train";
const isDeltaMode = logMode === "delta";
const useLeanKibo = isTrainMode || isDeltaMode;
const actorConfig = {
  human: {
    fallbackPolicy: policyHuman,
    policyModelPath: policyModelHumanPath,
    valueModelPath: valueModelHumanPath
  },
  ai: {
    fallbackPolicy: policyAi,
    policyModelPath: policyModelAiPath,
    valueModelPath: valueModelAiPath
  }
};
actorConfig.human.policyModel = loadJsonModel(actorConfig.human.policyModelPath, "policy-model-human");
actorConfig.ai.policyModel = loadJsonModel(actorConfig.ai.policyModelPath, "policy-model-ai");
actorConfig.human.valueModel = loadJsonModel(actorConfig.human.valueModelPath, "value-model-human");
actorConfig.ai.valueModel = loadJsonModel(actorConfig.ai.valueModelPath, "value-model-ai");

if (modelOnly) {
  for (const actor of ["human", "ai"]) {
    if (!actorConfig[actor].policyModel && !actorConfig[actor].valueModel) {
      throw new Error(`model-only requires model for ${actor}. Provide --policy-model-${actor} and/or --value-model-${actor}`);
    }
  }
}

const agentLabelByActor = {
  human: actorConfig.human.policyModel || actorConfig.human.valueModel
    ? `model:${path.basename(actorConfig.human.policyModelPath || actorConfig.human.valueModelPath)}`
    : actorConfig.human.fallbackPolicy,
  ai: actorConfig.ai.policyModel || actorConfig.ai.valueModel
    ? `model:${path.basename(actorConfig.ai.policyModelPath || actorConfig.ai.valueModelPath)}`
    : actorConfig.ai.fallbackPolicy
};

const stamp = new Date().toISOString().replace(/[:.]/g, "-");
const outPath = outArg || path.resolve(__dirname, "..", "logs", `ai-vs-ai-${stamp}.jsonl`);
const reportPath = outPath.replace(/\.jsonl$/i, "-report.json");
const sharedCatalogDir = path.resolve(__dirname, "..", "logs", "catalog");
const sharedCatalogPath = path.join(sharedCatalogDir, "cards-catalog.json");
const legacyCatalogPath = path.resolve(__dirname, "..", "logs", "catalog.json");

fs.mkdirSync(path.dirname(outPath), { recursive: true });

const aggregate = {
  games,
  completed: 0,
  winners: { human: 0, ai: 0, draw: 0, unknown: 0 },
  byTurnOrder: {
    firstWins: 0,
    secondWins: 0,
    draw: 0,
    firstScoreSum: 0,
    secondScoreSum: 0
  },
  economy: {
    humanGoldSum: 0,
    aiGoldSum: 0,
    humanDeltaSum: 0,
    firstGoldSum: 0,
    secondGoldSum: 0,
    firstDeltaSum: 0,
    secondDeltaSum: 0,
    first1000HumanDeltaSum: 0,
    first1000Games: 0
  },
  nagari: 0,
  eventTotals: { ppuk: 0, ddadak: 0, jjob: 0, ssul: 0, ttak: 0 },
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

function createBalancedFirstTurnPlan(totalGames) {
  if (totalGames % 2 !== 0) {
    throw new Error(
      `games must be even for exact 50:50 first-turn split. Received: ${totalGames}`
    );
  }
  const half = totalGames / 2;
  const plan = [];
  for (let i = 0; i < half; i += 1) plan.push("human");
  for (let i = 0; i < half; i += 1) plan.push("ai");
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
  if (card.tripleJunk) return 3;
  if (card.doubleJunk) return 2;
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

function selectPool(state, actor) {
  if (state.phase === "playing" && state.currentTurn === actor) {
    return {
      cards: (state.players?.[actor]?.hand || []).map((c) => c.id),
      bombMonths: getDeclarableBombMonths(state, actor),
      shakingMonths: getDeclarableShakingMonths(state, actor)
    };
  }
  if (state.phase === "select-match" && state.pendingMatch?.playerKey === actor) {
    return {
      boardCardIds: state.pendingMatch.boardCardIds || []
    };
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
  if (state.phase === "kung-choice" && state.pendingKung?.playerKey === actor) {
    return { options: ["kung_use", "kung_pass"] };
  }
  return {};
}

function decisionContext(state, actor) {
  const opp = actor === "human" ? "ai" : "human";
  return {
    phase: state.phase,
    turnNoBefore: (state.turnSeq || 0) + 1,
    deckCount: state.deck.length,
    handCountSelf: state.players[actor].hand.length,
    handCountOpp: state.players[opp].hand.length,
    goCountSelf: state.players[actor].goCount || 0,
    goCountOpp: state.players[opp].goCount || 0,
    goldSelf: state.players[actor].gold,
    goldOpp: state.players[opp].gold
  };
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
  const handSelf = dc.handCountSelf || 0;
  const handOpp = dc.handCountOpp || 0;
  const goSelf = dc.goCountSelf || 0;
  const goOpp = dc.goCountOpp || 0;
  const cands = (sp.cards || sp.boardCardIds || sp.options || []).length;
  return [
    `dt=${decisionType}`,
    `ph=${dc.phase || "?"}`,
    `o=${trace.o || "?"}`,
    `db=${deckBucket}`,
    `hs=${handSelf}`,
    `ho=${handOpp}`,
    `gs=${goSelf}`,
    `go=${goOpp}`,
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
  return [
    `phase=${dc.phase || "?"}`,
    `order=${order || "?"}`,
    `decision_type=${decisionType}`,
    `action=${actionLabel || "?"}`,
    `deck_bucket=${Math.floor((dc.deckCount || 0) / 3)}`,
    `self_hand=${Math.floor(dc.handCountSelf || 0)}`,
    `opp_hand=${Math.floor(dc.handCountOpp || 0)}`,
    `self_go=${Math.floor(dc.goCountSelf || 0)}`,
    `opp_go=${Math.floor(dc.goCountOpp || 0)}`
  ];
}

function valueNumeric(dc, candidateCount) {
  return {
    deck_count: Number(dc.deckCount || 0),
    hand_self: Number(dc.handCountSelf || 0),
    hand_opp: Number(dc.handCountOpp || 0),
    go_self: Number(dc.goCountSelf || 0),
    go_opp: Number(dc.goCountOpp || 0),
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
    kung_use: "choose_kung_use",
    kung_pass: "choose_kung_pass",
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
  const numeric = useValue ? valueNumeric(dc, candidates.length) : null;
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
  if (c === "president_stop") return choosePresidentStop(state, actor);
  if (c === "president_hold") return choosePresidentHold(state, actor);
  if (c === "five" || c === "junk") return chooseGukjinMode(state, actor, c);
  return state;
}

function loadJsonModel(modelPath, label) {
  if (!modelPath) return null;
  const full = path.resolve(modelPath);
  if (!fs.existsSync(full)) {
    throw new Error(`${label} not found: ${modelPath}`);
  }
  return JSON.parse(fs.readFileSync(full, "utf8"));
}

function compactInitial(initialDeal) {
  if (!initialDeal) return null;
  return {
    firstTurn: initialDeal.firstTurn,
    hands:
      initialDeal.hands != null
        ? {
            human: (initialDeal.hands?.human || []).map((c) => c.id),
            ai: (initialDeal.hands?.ai || []).map((c) => c.id)
          }
        : null,
    handsCount: initialDeal.handsCount || null,
    board: initialDeal.board != null ? (initialDeal.board || []).map((c) => c.id) : null,
    boardCount:
      initialDeal.boardCount != null
        ? initialDeal.boardCount
        : Array.isArray(initialDeal.board)
        ? initialDeal.board.length
        : 0,
    deckCount:
      initialDeal.deckCount != null
        ? initialDeal.deckCount
        : Array.isArray(initialDeal.deck)
        ? initialDeal.deck.length
        : 0
  };
}

function countCandidates(selectionPool) {
  if (!selectionPool) return 0;
  if (Array.isArray(selectionPool.options)) return selectionPool.options.length;
  if (Array.isArray(selectionPool.boardCardIds)) return selectionPool.boardCardIds.length;
  const cards = selectionPool.cards?.length || 0;
  const bomb = selectionPool.bombMonths?.length || 0;
  const shaking = selectionPool.shakingMonths?.length || 0;
  return cards + bomb + shaking;
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

function buildDecisionTrace(kibo, winner, contextByTurnNo, firstTurnKey, policyByActorRef) {
  const turns = (kibo || []).filter((e) => e.type === "turn_end");
  return turns.map((t) => {
    const actor = t.actor;
    const action = t.action || {};
    const matchEvents = action.matchEvents || [];
    const captureBySource = action.captureBySource || { hand: [], flip: [] };
    const immediateReward = (t.steals?.pi || 0) + (t.steals?.gold || 0) / 100;
    const terminalReward = winner === "draw" ? 0 : winner === actor ? 1 : -1;
    const before = contextByTurnNo.get(t.turnNo) || null;
    const opp = actor === "human" ? "ai" : "human";

    return {
      t: t.turnNo,
      a: actor,
      o: actor === firstTurnKey ? "first" : "second",
      at: action.type || "unknown",
      c: action.card?.id || null,
      s: action.selectedBoardCard?.id || null,
      ir: immediateReward,
      tr: terminalReward,
      dc: before?.decisionContext || null,
      sp: before?.selectionPool || null,
      reasoning: {
        policy: policyByActorRef[actor] || DEFAULT_POLICY,
        candidatesCount: countCandidates(before?.selectionPool),
        evaluation: null
      },
      cap: {
        hand: (captureBySource.hand || []).map((c) => c.id),
        flip: (captureBySource.flip || []).map((c) => c.id)
      },
      fv: {
        deckCountAfter: t.deckCount ?? 0,
        boardCountAfter: boardCountFromTurn(t),
        handCountSelfAfter: handCountFromTurn(t, actor),
        handCountOppAfter: handCountFromTurn(t, opp),
        eventCounts: t.events || {},
        matchEvents
      }
    };
  });
}

function buildDecisionTraceTrain(kibo, winner, firstTurnKey, contextByTurnNo, policyByActorRef) {
  const turns = (kibo || []).filter((e) => e.type === "turn_end");
  return turns.map((t) => {
    const actor = t.actor;
    const action = t.action || {};
    const immediateReward = (t.steals?.pi || 0) + (t.steals?.gold || 0) / 100;
    const terminalReward = winner === "draw" ? 0 : winner === actor ? 1 : -1;
    const before = contextByTurnNo.get(t.turnNo) || null;
    return {
      t: t.turnNo,
      a: actor,
      o: actor === firstTurnKey ? "first" : "second",
      at: action.type || "unknown",
      c: action.card?.id || null,
      s: action.selectedBoardCard?.id || null,
      ir: immediateReward,
      tr: terminalReward,
      dc: before?.decisionContext || null,
      sp: before?.selectionPool || null,
      policy: policyByActorRef[actor] || DEFAULT_POLICY
    };
  });
}

function buildDecisionTraceDelta(kibo, winner, firstTurnKey, contextByTurnNo, policyByActorRef) {
  const turns = (kibo || []).filter((e) => e.type === "turn_end");
  let prevDeck = null;
  let prevHand = { human: null, ai: null };
  return turns.map((t) => {
    const actor = t.actor;
    const action = t.action || {};
    const terminalReward = winner === "draw" ? 0 : winner === actor ? 1 : -1;
    const deckAfter = t.deckCount ?? 0;
    const handSelfAfter = handCountFromTurn(t, actor);
    const opp = actor === "human" ? "ai" : "human";
    const handOppAfter = handCountFromTurn(t, opp);
    const deckDelta = prevDeck == null ? null : deckAfter - prevDeck;
    const handSelfDelta = prevHand[actor] == null ? null : handSelfAfter - prevHand[actor];
    const handOppDelta = prevHand[opp] == null ? null : handOppAfter - prevHand[opp];
    const before = contextByTurnNo.get(t.turnNo) || null;
    prevDeck = deckAfter;
    prevHand = {
      human: handCountFromTurn(t, "human"),
      ai: handCountFromTurn(t, "ai")
    };

    return {
      t: t.turnNo,
      a: actor,
      o: actor === firstTurnKey ? "first" : "second",
      at: action.type || "unknown",
      c: action.card?.id || null,
      s: action.selectedBoardCard?.id || null,
      tr: terminalReward,
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
        policy: policyByActorRef[actor] || DEFAULT_POLICY,
        candidatesCount: countCandidates(before?.selectionPool),
        evaluation: null
      }
    };
  });
}

function baseSummary(line, winnerTurnOrder, firstScore, secondScore, firstGold, secondGold) {
  aggregate.completed += line.completed ? 1 : 0;
  if (aggregate.winners[line.winner] != null) aggregate.winners[line.winner] += 1;
  else aggregate.winners.unknown += 1;
  if (winnerTurnOrder === "first") aggregate.byTurnOrder.firstWins += 1;
  else if (winnerTurnOrder === "second") aggregate.byTurnOrder.secondWins += 1;
  else aggregate.byTurnOrder.draw += 1;
  aggregate.byTurnOrder.firstScoreSum += firstScore;
  aggregate.byTurnOrder.secondScoreSum += secondScore;
  const humanGold = Number(line?.gold?.human ?? 0);
  const aiGold = Number(line?.gold?.ai ?? 0);
  const humanDelta = humanGold - aiGold;
  aggregate.economy.humanGoldSum += humanGold;
  aggregate.economy.aiGoldSum += aiGold;
  aggregate.economy.humanDeltaSum += humanDelta;
  aggregate.economy.firstGoldSum += Number(firstGold || 0);
  aggregate.economy.secondGoldSum += Number(secondGold || 0);
  aggregate.economy.firstDeltaSum += Number(firstGold || 0) - Number(secondGold || 0);
  aggregate.economy.secondDeltaSum += Number(secondGold || 0) - Number(firstGold || 0);
  if (aggregate.economy.first1000Games < 1000) {
    aggregate.economy.first1000HumanDeltaSum += humanDelta;
    aggregate.economy.first1000Games += 1;
  }
}

function fullSummary({
  line,
  winnerTurnOrder,
  firstScore,
  secondScore,
  goCalls,
  goEfficiency,
  totalPiSteals,
  totalGoldSteals,
  loserKey,
  loserBak,
  bakEscaped,
  flipEvents,
  handEvents,
  flipCaptureValue,
  handCaptureValue,
  firstGold,
  secondGold
}) {
  baseSummary(line, winnerTurnOrder, firstScore, secondScore, firstGold, secondGold);
  if (line.nagari) aggregate.nagari += 1;
  aggregate.eventTotals.ppuk += line.eventFrequency.ppuk;
  aggregate.eventTotals.ddadak += line.eventFrequency.ddadak;
  aggregate.eventTotals.jjob += line.eventFrequency.jjob;
  aggregate.eventTotals.ssul += line.eventFrequency.ssul;
  aggregate.eventTotals.ttak += line.eventFrequency.ttak;
  aggregate.goCalls += goCalls;
  aggregate.goEfficiencySum += goEfficiency;
  aggregate.goDecision.declared += line.goDecision.declared;
  aggregate.goDecision.success += line.goDecision.success;
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
    logMode,
    games: aggregate.games,
    completed: aggregate.completed,
    winners: aggregate.winners,
    turnOrder: {
      firstWinRate: aggregate.byTurnOrder.firstWins / Math.max(1, aggregate.games),
      secondWinRate: aggregate.byTurnOrder.secondWins / Math.max(1, aggregate.games),
      drawRate: aggregate.byTurnOrder.draw / Math.max(1, aggregate.games),
      averageScoreFirst: aggregate.byTurnOrder.firstScoreSum / Math.max(1, aggregate.games),
      averageScoreSecond: aggregate.byTurnOrder.secondScoreSum / Math.max(1, aggregate.games)
    },
    economy: {
      averageGoldHuman: aggregate.economy.humanGoldSum / Math.max(1, aggregate.games),
      averageGoldAi: aggregate.economy.aiGoldSum / Math.max(1, aggregate.games),
      averageGoldDeltaHuman: aggregate.economy.humanDeltaSum / Math.max(1, aggregate.games),
      averageGoldFirst: aggregate.economy.firstGoldSum / Math.max(1, aggregate.games),
      averageGoldSecond: aggregate.economy.secondGoldSum / Math.max(1, aggregate.games),
      averageGoldDeltaFirst: aggregate.economy.firstDeltaSum / Math.max(1, aggregate.games),
      cumulativeGoldDeltaOver1000:
        (aggregate.economy.humanDeltaSum / Math.max(1, aggregate.games)) * 1000,
      cumulativeGoldDeltaFirst1000: aggregate.economy.first1000HumanDeltaSum
    },
    primaryMetric: "averageGoldDeltaHuman"
  };
}

function buildFullReport() {
  return {
    logMode,
    catalogPath: sharedCatalogPath,
    games: aggregate.games,
    completed: aggregate.completed,
    winners: aggregate.winners,
    turnOrder: {
      firstWinRate: aggregate.byTurnOrder.firstWins / Math.max(1, aggregate.games),
      secondWinRate: aggregate.byTurnOrder.secondWins / Math.max(1, aggregate.games),
      drawRate: aggregate.byTurnOrder.draw / Math.max(1, aggregate.games),
      averageScoreFirst: aggregate.byTurnOrder.firstScoreSum / Math.max(1, aggregate.games),
      averageScoreSecond: aggregate.byTurnOrder.secondScoreSum / Math.max(1, aggregate.games)
    },
    economy: {
      averageGoldHuman: aggregate.economy.humanGoldSum / Math.max(1, aggregate.games),
      averageGoldAi: aggregate.economy.aiGoldSum / Math.max(1, aggregate.games),
      averageGoldDeltaHuman: aggregate.economy.humanDeltaSum / Math.max(1, aggregate.games),
      averageGoldFirst: aggregate.economy.firstGoldSum / Math.max(1, aggregate.games),
      averageGoldSecond: aggregate.economy.secondGoldSum / Math.max(1, aggregate.games),
      averageGoldDeltaFirst: aggregate.economy.firstDeltaSum / Math.max(1, aggregate.games),
      cumulativeGoldDeltaOver1000:
        (aggregate.economy.humanDeltaSum / Math.max(1, aggregate.games)) * 1000,
      cumulativeGoldDeltaFirst1000: aggregate.economy.first1000HumanDeltaSum
    },
    primaryMetric: "averageGoldDeltaHuman",
    nagariRate: aggregate.nagari / Math.max(1, aggregate.games),
    eventFrequencyPerGame: {
      ppuk: aggregate.eventTotals.ppuk / Math.max(1, aggregate.games),
      ddadak: aggregate.eventTotals.ddadak / Math.max(1, aggregate.games),
      jjob: aggregate.eventTotals.jjob / Math.max(1, aggregate.games),
      ssul: aggregate.eventTotals.ssul / Math.max(1, aggregate.games),
      ttak: aggregate.eventTotals.ttak / Math.max(1, aggregate.games)
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

function executeActorTurn(state, actor) {
  const cfg = actorConfig[actor];
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
  return botPlay(state, actor, { policy: cfg?.fallbackPolicy || DEFAULT_POLICY });
}

async function run() {
  const outStream = fs.createWriteStream(outPath, { encoding: "utf8" });
  const firstTurnPlan = createBalancedFirstTurnPlan(games);

  for (let i = 0; i < games; i += 1) {
    const seed = `sim-${Date.now()}-${i}-${Math.random().toString(36).slice(2, 8)}`;
    const forcedFirstTurnKey = firstTurnPlan[i];
    let state = initGame("A", createSeededRng(seed), {
      carryOverMultiplier: 1,
      kiboDetail: useLeanKibo ? "lean" : "full",
      firstTurnKey: forcedFirstTurnKey
    });
    state.players.human.label = "AI-1";
    state.players.ai.label = "AI-2";
    const firstTurnKey = state.startingTurnKey;
    const secondTurnKey = firstTurnKey === "human" ? "ai" : "human";

    const contextByTurnNo = new Map();
    let steps = 0;
    const maxSteps = 4000;
    while (state.phase !== "resolution" && steps < maxSteps) {
      const actor = getActionPlayerKey(state);
      if (!actor) break;
      const beforeContext = {
        actor,
        decisionContext: decisionContext(state, actor),
        selectionPool: selectPool(state, actor)
      };
      const prevTurnSeq = state.turnSeq || 0;
      const next = executeActorTurn(state, actor);
      if (next === state) break;
      const nextTurnSeq = next.turnSeq || 0;
      if (nextTurnSeq > prevTurnSeq) {
        contextByTurnNo.set(nextTurnSeq, beforeContext);
      }
      state = next;
      steps += 1;
    }

    const winner = state.result?.winner || "unknown";
    const kibo = state.kibo || [];
    const decisionTrace = isDeltaMode
      ? buildDecisionTraceDelta(kibo, winner, firstTurnKey, contextByTurnNo, agentLabelByActor)
      : isTrainMode
      ? buildDecisionTraceTrain(kibo, winner, firstTurnKey, contextByTurnNo, agentLabelByActor)
      : buildDecisionTrace(kibo, winner, contextByTurnNo, firstTurnKey, agentLabelByActor);
    const firstScore = firstTurnKey === "human" ? state.result?.human?.total || 0 : state.result?.ai?.total || 0;
    const secondScore = secondTurnKey === "human" ? state.result?.human?.total || 0 : state.result?.ai?.total || 0;
    const winnerTurnOrder =
      winner === "draw" || winner === "unknown"
        ? "draw"
        : winner === firstTurnKey
        ? "first"
        : "second";
    const humanGold = Number(state.players?.human?.gold || 0);
    const aiGold = Number(state.players?.ai?.gold || 0);
    const firstGold = firstTurnKey === "human" ? humanGold : aiGold;
    const secondGold = secondTurnKey === "human" ? humanGold : aiGold;

    if (isTrainMode) {
      const line = {
        game: i + 1,
        seed,
        steps,
        completed: state.phase === "resolution",
        logMode,
        firstTurn: firstTurnKey,
        secondTurn: secondTurnKey,
        policy: agentLabelByActor,
        winner,
        score: {
          human: state.result?.human?.total ?? null,
          ai: state.result?.ai?.total ?? null
        },
        gold: {
          human: humanGold,
          ai: aiGold
        },
        decision_trace: decisionTrace
      };
      baseSummary(line, winnerTurnOrder, firstScore, secondScore, firstGold, secondGold);
      await writeLine(outStream, line);
      continue;
    }

    const goCalls = kibo.filter((e) => e.type === "go").length;
    const winnerTotal = winner === "human" ? state.result?.human?.total || 0 : winner === "ai" ? state.result?.ai?.total || 0 : 0;
    const loserTotal = winner === "human" ? state.result?.ai?.total || 0 : winner === "ai" ? state.result?.human?.total || 0 : 0;
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
      human: state.players.human.events,
      ai: state.players.ai.events
    };

    const goEvents = kibo.filter((e) => e.type === "go");
    const goDeclared = goEvents.length;
    const goSuccess = goEvents.filter((e) => winner !== "draw" && winner === e.playerKey).length;

    const loserKey = winner === "human" ? "ai" : winner === "ai" ? "human" : null;
    const loserBak = loserKey ? state.result?.[loserKey]?.bak : null;
    const bakEscaped =
      loserBak && !loserBak.gwang && !loserBak.pi && !loserBak.mongBak ? 1 : 0;

    const line = {
      game: i + 1,
      seed,
      steps,
      completed: state.phase === "resolution",
      logMode,
      firstTurn: firstTurnKey,
      secondTurn: secondTurnKey,
      policy: agentLabelByActor,
      winner,
      winnerTurnOrder,
      nagari: state.result?.nagari || false,
      nagariReasons: state.result?.nagariReasons || [],
      score: {
        human: state.result?.human?.total ?? null,
        ai: state.result?.ai?.total ?? null,
        first: firstScore,
        second: secondScore
      },
      gold: {
        human: humanGold,
        ai: aiGold,
        first: firstGold,
        second: secondGold
      },
      goCalls,
      goStopEfficiency: goEfficiency,
      goDecision: {
        declared: goDeclared,
        success: goSuccess
      },
      steals: {
        piTotal: totalPiSteals,
        goldTotal: totalGoldSteals
      },
      luckSkill: {
        flipEvents,
        handEvents,
        flipRatio: flipEvents / Math.max(1, flipEvents + handEvents),
        weightedCapture: {
          flipValue: flipCaptureValue,
          handValue: handCaptureValue,
          flipRatio: flipCaptureValue / Math.max(1, flipCaptureValue + handCaptureValue)
        }
      },
      eventFrequency: {
        ppuk: (allEvents.human.ppuk || 0) + (allEvents.ai.ppuk || 0),
        ddadak: (allEvents.human.ddadak || 0) + (allEvents.ai.ddadak || 0),
        jjob: (allEvents.human.jjob || 0) + (allEvents.ai.jjob || 0),
        ssul: (allEvents.human.ssul || 0) + (allEvents.ai.ssul || 0),
        ttak: (allEvents.human.ttak || 0) + (allEvents.ai.ttak || 0)
      },
      bakEscape: loserKey ? { loser: loserKey, escaped: !!bakEscaped } : null,
      catalogPath: sharedCatalogPath,
      initial: compactInitial(kibo.find((e) => e.type === "initial_deal")),
      decision_trace: decisionTrace
    };

    fullSummary({
      line,
      winnerTurnOrder,
      firstScore,
      secondScore,
      goCalls,
      goEfficiency,
      totalPiSteals,
      totalGoldSteals,
      loserKey,
      loserBak,
      bakEscaped,
      flipEvents,
      handEvents,
      flipCaptureValue,
      handCaptureValue,
      firstGold,
      secondGold
    });

    await writeLine(outStream, line);
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
