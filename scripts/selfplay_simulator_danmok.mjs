import fs from "node:fs";
import path from "node:path";
import { once } from "node:events";
import { fileURLToPath } from "node:url";
import {
  initGame,
  createSeededRng,
  calculateScore,
  getDeclarableBombMonths,
  getDeclarableShakingMonths
} from "../src/gameEngine.js";
import { buildDeck } from "../src/cards.js";
import { BOT_POLICIES, botPlay } from "../src/bot.js";
import { getActionPlayerKey } from "../src/engineRunner.js";
import { STARTING_GOLD } from "../src/engine/economy.js";

if (process.env.NO_SIMULATION === "1") {
  console.error("Simulation blocked: NO_SIMULATION=1");
  process.exit(2);
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const SIDE_MY = "mySide";
const SIDE_YOUR = "yourSide";
const DEFAULT_POLICY = "heuristic_v3";
const DEFAULT_LOG_MODE = "train";
const SUPPORTED_LOG_MODES = new Set(["train"]);
const SUPPORTED_POLICIES = new Set(BOT_POLICIES);

const FULL_DECK = buildDeck();
const CARD_BY_ID = new Map(FULL_DECK.map((c) => [c.id, c]));
const CARD_INDEX_BY_ID = new Map(FULL_DECK.map((c, i) => [c.id, i]));

function normalizePolicyInput(raw) {
  const p = String(raw ?? DEFAULT_POLICY).trim().toLowerCase();
  return p || DEFAULT_POLICY;
}

function parseArgs(argv) {
  const args = [...argv];
  let games = 1000;
  let outArg = null;
  let policyMySide = DEFAULT_POLICY;
  let policyYourSide = DEFAULT_POLICY;
  let logMode = DEFAULT_LOG_MODE;
  let traceMyTurnOnly = false;

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
    if (arg === "--log-mode" && args.length > 0) {
      logMode = String(args.shift()).trim();
      continue;
    }
    if (arg.startsWith("--log-mode=")) {
      logMode = arg.split("=", 2)[1].trim();
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
    throw new Error(`Unknown argument: ${arg}`);
  }

  if (!SUPPORTED_LOG_MODES.has(logMode)) {
    throw new Error(`Unsupported log mode: ${logMode}. Use one of: train`);
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
    logMode,
    traceMyTurnOnly
  };
}

function actorToSide(actor, firstTurnKey) {
  return actor === firstTurnKey ? SIDE_MY : SIDE_YOUR;
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

function createBalancedFirstTurnPlan(totalGames, actorA, actorB) {
  if (totalGames % 2 !== 0) {
    throw new Error(`games must be even for exact 50:50 first-turn split. Received: ${totalGames}`);
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

function selectPool(state, actor) {
  if (state.phase === "playing" && state.currentTurn === actor) {
    const cards = (state.players?.[actor]?.hand || []).map((c) => c.id);
    return {
      decisionType: "play",
      candidateCount: cards.length,
      cards,
      bombMonths: getDeclarableBombMonths(state, actor),
      shakingMonths: getDeclarableShakingMonths(state, actor)
    };
  }
  if (state.phase === "select-match" && state.pendingMatch?.playerKey === actor) {
    const boardCardIds = state.pendingMatch.boardCardIds || [];
    return {
      decisionType: "match",
      candidateCount: boardCardIds.length,
      boardCardIds
    };
  }
  if (state.phase === "go-stop" && state.pendingGoStop === actor) {
    return {
      decisionType: "option",
      candidateCount: 2,
      options: ["go", "stop"]
    };
  }
  if (state.phase === "president-choice" && state.pendingPresident?.playerKey === actor) {
    return {
      decisionType: "option",
      candidateCount: 2,
      options: ["president_stop", "president_hold"]
    };
  }
  if (state.phase === "gukjin-choice" && state.pendingGukjinChoice?.playerKey === actor) {
    return {
      decisionType: "option",
      candidateCount: 2,
      options: ["five", "junk"]
    };
  }
  if (state.phase === "shaking-confirm" && state.pendingShakingConfirm?.playerKey === actor) {
    return {
      decisionType: "option",
      candidateCount: 2,
      options: ["shaking_yes", "shaking_no"]
    };
  }
  return { decisionType: "play", candidateCount: 0 };
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
  const explicit = Number(selectionPool?.candidateCount || 0);
  if (Number.isFinite(explicit) && explicit > 0) return Math.floor(explicit);
  if (Array.isArray(selectionPool?.options)) return selectionPool.options.length;
  if (Array.isArray(selectionPool?.boardCardIds)) return selectionPool.boardCardIds.length;
  if (Array.isArray(selectionPool?.cards)) return selectionPool.cards.length;
  return 0;
}

function inferOptionActionType({ beforeState, nextState, actor, selectionPool, prevKiboSeq }) {
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

function monthMaskFromIds(ids) {
  let mask = 0;
  for (const id of ids || []) {
    const c = CARD_BY_ID.get(id);
    const m = Number(c?.month || 0);
    if (m >= 1 && m <= 12) mask |= 1 << (m - 1);
  }
  return mask;
}

function kindCountsFromIds(ids) {
  const out = { kwang: 0, five: 0, ribbon: 0, junk: 0 };
  for (const id of ids || []) {
    const cat = CARD_BY_ID.get(id)?.category;
    if (cat === "kwang") out.kwang += 1;
    else if (cat === "five") out.five += 1;
    else if (cat === "ribbon") out.ribbon += 1;
    else if (cat === "junk") out.junk += 1;
  }
  return out;
}

function idsFromSelectionPool(sp) {
  if (Array.isArray(sp?.cards)) return sp.cards;
  if (Array.isArray(sp?.boardCardIds)) return sp.boardCardIds;
  return [];
}

function safeCalculateScore(state, actor) {
  if (!actor || !state?.players?.[actor]) {
    return { total: 0, bak: { pi: false, gwang: false, mongBak: false } };
  }
  try {
    return (
      calculateScore(state, actor) || {
        total: 0,
        bak: { pi: false, gwang: false, mongBak: false }
      }
    );
  } catch {
    return { total: 0, bak: { pi: false, gwang: false, mongBak: false } };
  }
}

function decisionContextLite(state, actor, selectionPool) {
  const opp = Object.keys(state?.players || {}).find((k) => k !== actor) || null;
  const self = state?.players?.[actor] || {};
  const other = opp ? state?.players?.[opp] || {} : {};

  const scoreSelf = safeCalculateScore(state, actor);
  const scoreOpp = safeCalculateScore(state, opp);

  const handIds = (self.hand || []).map((c) => c.id);
  const boardIds = (state.board || []).map((c) => c.id);
  const candidateIds = idsFromSelectionPool(selectionPool);
  const candidateKinds = kindCountsFromIds(candidateIds);

  return {
    phase: tracePhaseCode(state.phase),
    deckCount: Array.isArray(state.deck) ? state.deck.length : 0,
    handCountSelf: Array.isArray(self.hand) ? self.hand.length : 0,
    handCountOpp: Array.isArray(other.hand) ? other.hand.length : 0,
    goCountSelf: Number(self.goCount || 0),
    goCountOpp: Number(other.goCount || 0),
    shakeCountSelf: Number(self.events?.shaking || 0),
    shakeCountOpp: Number(other.events?.shaking || 0),
    carryOverMultiplier: Math.max(1, Number(state.carryOverMultiplier || 1)),
    piBakRisk: scoreSelf?.bak?.pi ? 1 : 0,
    gwangBakRisk: scoreSelf?.bak?.gwang ? 1 : 0,
    mongBakRisk: scoreSelf?.bak?.mongBak ? 1 : 0,
    currentScoreSelf: Number(scoreSelf?.total || 0),
    currentScoreOpp: Number(scoreOpp?.total || 0),
    selfHandMonthMask: monthMaskFromIds(handIds),
    boardMonthMask: monthMaskFromIds(boardIds),
    candidateMonthMask: monthMaskFromIds(candidateIds),
    candidateKwangCount: candidateKinds.kwang,
    candidateFiveCount: candidateKinds.five,
    candidateRibbonCount: candidateKinds.ribbon,
    candidateJunkCount: candidateKinds.junk
  };
}

function findTurnEndEvent(kibo, turnNo, actor) {
  if (!Array.isArray(kibo)) return null;
  for (let i = kibo.length - 1; i >= 0; i -= 1) {
    const e = kibo[i];
    if (e?.type !== "turn_end") continue;
    if (Number(e?.turnNo || 0) !== Number(turnNo || 0)) continue;
    if (actor && e?.actor !== actor) continue;
    return e;
  }
  return null;
}

async function writeLine(writer, line) {
  if (writer.write(`${JSON.stringify(line)}\n`)) return;
  await once(writer, "drain");
}

function buildDefaultOutPath() {
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  return path.resolve(__dirname, "..", "logs", `side-vs-side-trace-${stamp}.jsonl`);
}

function createAggregate(games) {
  return {
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
    }
  };
}

function accumulateSummary(aggregate, { completed, winner, mySideScore, yourSideScore, mySideGold, yourSideGold }) {
  aggregate.completed += completed ? 1 : 0;
  if (aggregate.winners[winner] != null) aggregate.winners[winner] += 1;
  else aggregate.winners.unknown += 1;
  if (winner === SIDE_MY) aggregate.bySide.mySideWins += 1;
  else if (winner === SIDE_YOUR) aggregate.bySide.yourSideWins += 1;
  else aggregate.bySide.draw += 1;

  aggregate.bySide.mySideScoreSum += mySideScore;
  aggregate.bySide.yourSideScoreSum += yourSideScore;

  const myDelta = mySideGold - yourSideGold;
  aggregate.economy.mySideGoldSum += mySideGold;
  aggregate.economy.yourSideGoldSum += yourSideGold;
  aggregate.economy.mySideDeltaSum += myDelta;
  if (aggregate.economy.first1000Games < 1000) {
    aggregate.economy.first1000MySideDeltaSum += myDelta;
    aggregate.economy.first1000Games += 1;
  }

  const myBankrupt = mySideGold <= 0 ? 1 : 0;
  const yourBankrupt = yourSideGold <= 0 ? 1 : 0;
  aggregate.bankrupt.mySideInflicted += yourBankrupt;
  aggregate.bankrupt.mySideSuffered += myBankrupt;
  aggregate.bankrupt.yourSideInflicted += myBankrupt;
  aggregate.bankrupt.yourSideSuffered += yourBankrupt;
  if (myBankrupt || yourBankrupt) {
    aggregate.bankrupt.resets += 1;
  }
}

function buildReport(logMode, aggregate) {
  return {
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
    primaryMetric: "averageGoldDeltaMySide"
  };
}

async function run() {
  const parsed = parseArgs(process.argv.slice(2));
  const games = parsed.games;
  const outPath = parsed.outArg || buildDefaultOutPath();
  const reportPath = outPath.replace(/\.jsonl$/i, "-report.json");

  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  const outStream = fs.createWriteStream(outPath, { encoding: "utf8" });

  const probeState = initGame("A", createSeededRng("trace-sim-probe"), {
    carryOverMultiplier: 1,
    kiboDetail: "lean"
  });
  const [actorA, actorB] = actorPairFromState(probeState);
  const firstTurnPlan = createBalancedFirstTurnPlan(games, actorA, actorB);

  const aggregate = createAggregate(games);
  const sessionGold = {
    [SIDE_MY]: STARTING_GOLD,
    [SIDE_YOUR]: STARTING_GOLD
  };

  for (let i = 0; i < games; i += 1) {
    const seed = `trace-${Date.now()}-${i}-${Math.random().toString(36).slice(2, 8)}`;
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

    const decisionTrace = [];
    let steps = 0;
    const maxSteps = 4000;

    while (state.phase !== "resolution" && steps < maxSteps) {
      const actor = getActionPlayerKey(state);
      if (!actor) break;

      const actorSide = actorToSide(actor, firstTurnKey);
      if (parsed.traceMyTurnOnly && actorSide !== SIDE_MY) {
        const next = botPlay(state, actor, {
          policy: actorSide === SIDE_MY ? parsed.policyMySide : parsed.policyYourSide
        });
        if (next === state) break;
        state = next;
        steps += 1;
        continue;
      }

      const beforeSp = selectPool(state, actor);
      const beforeDc = decisionContextLite(state, actor, beforeSp);
      const decisionType = traceDecisionType(beforeSp, null);
      const candidateCount = traceCandidateCount(beforeSp);

      const prevTurnSeq = Number(state.turnSeq || 0);
      const prevKiboSeq = Number(state.kiboSeq || 0);
      const next = botPlay(state, actor, {
        policy: actorSide === SIDE_MY ? parsed.policyMySide : parsed.policyYourSide
      });
      if (next === state) break;

      const nextTurnSeq = Number(next.turnSeq || 0);
      if (candidateCount >= 1) {
        const order = actorSide === SIDE_MY ? "first" : "second";
        const turnNo = nextTurnSeq > prevTurnSeq ? nextTurnSeq : Math.max(1, prevTurnSeq);
        let at = "unknown";
        let c = null;
        let s = null;
        if (decisionType === "option") {
          at = inferOptionActionType({
            beforeState: state,
            nextState: next,
            actor,
            selectionPool: beforeSp,
            prevKiboSeq
          });
        } else {
          const te = findTurnEndEvent(next.kibo, turnNo, actor);
          at = String(te?.action?.type || "unknown");
          c = te?.action?.card?.id || null;
          s = te?.action?.selectedBoardCard?.id || null;
        }

        const trace = {
          t: turnNo,
          a: actorSide,
          o: order,
          dt: decisionType,
          cc: candidateCount,
          at,
          c,
          s,
          ir: 0,
          dc: beforeDc
        };
        if (c && CARD_INDEX_BY_ID.has(c)) trace.ci = CARD_INDEX_BY_ID.get(c);
        if (s && CARD_INDEX_BY_ID.has(s)) trace.si = CARD_INDEX_BY_ID.get(s);
        decisionTrace.push(trace);
      }

      state = next;
      steps += 1;
    }

    const winner = state.result?.winner || "unknown";
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

    const line = {
      game: i + 1,
      seed,
      steps,
      completed: state.phase === "resolution",
      logMode: parsed.logMode,
      firstAttackerSide: SIDE_MY,
      firstAttackerActor: firstTurnKey,
      policy: {
        [SIDE_MY]: parsed.policyMySide,
        [SIDE_YOUR]: parsed.policyYourSide
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
      decision_trace: decisionTrace
    };

    accumulateSummary(aggregate, {
      completed: line.completed,
      winner: line.winner,
      mySideScore,
      yourSideScore,
      mySideGold,
      yourSideGold
    });

    await writeLine(outStream, line);
  }

  outStream.end();
  await once(outStream, "finish");

  const report = buildReport(parsed.logMode, aggregate);
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2), "utf8");
  console.log(`done: ${games} games -> ${outPath}`);
  console.log(`report: ${reportPath}`);
}

run().catch((err) => {
  console.error(err instanceof Error ? err.stack || err.message : String(err));
  process.exitCode = 1;
});
