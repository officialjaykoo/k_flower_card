import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  initGame,
  createSeededRng,
  getDeclarableBombMonths,
  getDeclarableShakingMonths
} from "../src/gameEngine.js";
import { buildDeck } from "../src/cards.js";
import { botPlay } from "../src/bot.js";
import { getActionPlayerKey } from "../src/engineRunner.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const games = Number(process.argv[2] || 1000);
const outArg = process.argv[3];
const modeArg = process.argv[4];
const stamp = new Date().toISOString().replace(/[:.]/g, "-");
const outPath = outArg || path.resolve(__dirname, "..", "logs", `ai-vs-ai-${stamp}.jsonl`);
const reportPath = outPath.replace(/\.jsonl$/i, "-report.json");
const sharedCatalogDir = path.resolve(__dirname, "..", "logs", "catalog");
const sharedCatalogPath = path.join(sharedCatalogDir, "cards-catalog.json");
const legacyCatalogPath = path.resolve(__dirname, "..", "logs", "catalog.json");
const logMode = modeArg === "--log-mode=delta" || modeArg === "delta" ? "delta" : "compact";

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

function compactInitial(initialDeal) {
  if (!initialDeal) return null;
  return {
    firstTurn: initialDeal.firstTurn,
    hands: {
      human: (initialDeal.hands?.human || []).map((c) => c.id),
      ai: (initialDeal.hands?.ai || []).map((c) => c.id)
    },
    board: (initialDeal.board || []).map((c) => c.id),
    deckCount: (initialDeal.deck || []).length
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

function buildDecisionTrace(kibo, winner, contextByTurnNo, firstTurnKey) {
  const turns = (kibo || []).filter((e) => e.type === "turn_end");
  return turns.map((t) => {
    const actor = t.actor;
    const action = t.action || {};
    const matchEvents = action.matchEvents || [];
    const captureBySource = action.captureBySource || { hand: [], flip: [] };
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
      reasoning: {
        policy: "random",
        candidatesCount: countCandidates(before?.selectionPool),
        evaluation: null
      },
      cap: {
        hand: (captureBySource.hand || []).map((c) => c.id),
        flip: (captureBySource.flip || []).map((c) => c.id)
      },
      fv: {
        deckCountAfter: t.deckCount ?? 0,
        boardCountAfter: (t.board || []).length,
        handCountSelfAfter: (t.hands?.[actor] || []).length,
        handCountOppAfter: actor === "human" ? (t.hands?.ai || []).length : (t.hands?.human || []).length,
        eventCounts: t.events || {},
        matchEvents
      }
    };
  });
}

function buildDecisionTraceDelta(kibo, winner, firstTurnKey, contextByTurnNo) {
  const turns = (kibo || []).filter((e) => e.type === "turn_end");
  let prevDeck = null;
  let prevHand = { human: null, ai: null };
  return turns.map((t) => {
    const actor = t.actor;
    const action = t.action || {};
    const terminalReward = winner === "draw" ? 0 : winner === actor ? 1 : -1;
    const deckAfter = t.deckCount ?? 0;
    const handSelfAfter = (t.hands?.[actor] || []).length;
    const opp = actor === "human" ? "ai" : "human";
    const handOppAfter = (t.hands?.[opp] || []).length;
    const deckDelta = prevDeck == null ? null : deckAfter - prevDeck;
    const handSelfDelta = prevHand[actor] == null ? null : handSelfAfter - prevHand[actor];
    const handOppDelta = prevHand[opp] == null ? null : handOppAfter - prevHand[opp];
    const before = contextByTurnNo.get(t.turnNo) || null;
    prevDeck = deckAfter;
    prevHand = { human: (t.hands?.human || []).length, ai: (t.hands?.ai || []).length };

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
        policy: "random",
        candidatesCount: countCandidates(before?.selectionPool),
        evaluation: null
      }
    };
  });
}

for (let i = 0; i < games; i += 1) {
  const seed = `sim-${Date.now()}-${i}-${Math.random().toString(36).slice(2, 8)}`;
  let state = initGame("A", createSeededRng(seed), { carryOverMultiplier: 1 });
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
    const next = botPlay(state, actor);
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
  const decisionTrace =
    logMode === "delta"
      ? buildDecisionTraceDelta(kibo, winner, firstTurnKey, contextByTurnNo)
      : buildDecisionTrace(kibo, winner, contextByTurnNo, firstTurnKey);
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

  const firstScore = firstTurnKey === "human" ? state.result?.human?.total || 0 : state.result?.ai?.total || 0;
  const secondScore = secondTurnKey === "human" ? state.result?.human?.total || 0 : state.result?.ai?.total || 0;
  const winnerTurnOrder =
    winner === "draw" || winner === "unknown"
      ? "draw"
      : winner === firstTurnKey
      ? "first"
      : "second";

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

  aggregate.completed += line.completed ? 1 : 0;
  if (aggregate.winners[winner] != null) aggregate.winners[winner] += 1;
  else aggregate.winners.unknown += 1;
  if (line.nagari) aggregate.nagari += 1;
  if (winnerTurnOrder === "first") aggregate.byTurnOrder.firstWins += 1;
  else if (winnerTurnOrder === "second") aggregate.byTurnOrder.secondWins += 1;
  else aggregate.byTurnOrder.draw += 1;
  aggregate.byTurnOrder.firstScoreSum += firstScore;
  aggregate.byTurnOrder.secondScoreSum += secondScore;

  aggregate.eventTotals.ppuk += line.eventFrequency.ppuk;
  aggregate.eventTotals.ddadak += line.eventFrequency.ddadak;
  aggregate.eventTotals.jjob += line.eventFrequency.jjob;
  aggregate.eventTotals.ssul += line.eventFrequency.ssul;
  aggregate.eventTotals.ttak += line.eventFrequency.ttak;
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

  fs.appendFileSync(outPath, `${JSON.stringify(line)}\n`, "utf8");
}

const report = {
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

ensureSharedCatalog();
fs.writeFileSync(reportPath, JSON.stringify(report, null, 2), "utf8");
console.log(`done: ${games} games -> ${outPath}`);
console.log(`report: ${reportPath}`);
console.log(`catalog: ${sharedCatalogPath}`);
