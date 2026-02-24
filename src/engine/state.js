import { buildDeck, normalizeCardTheme, DEFAULT_CARD_THEME, shuffle } from "../cards.js";
import { ruleSets } from "./rules.js";
import { calculateScore, calculateBaseScore, isGukjinCard } from "./scoring.js";
import { POINT_GOLD_UNIT, STARTING_GOLD, settleRoundGold } from "./economy.js";
import { resolveMatch } from "./matching.js";
import { resolveRound } from "./resolution.js";
import {
  decideFirstTurn,
  emptyPlayer,
  normalizeOpeningHands,
  normalizeOpeningBoard,
  findPresidentMonth
} from "./opening.js";
import {
  pushCaptured,
  needsChoice,
  bestMatchCard,
  stealPiFromOpponent
} from "./capturesEvents.js";
import { clearExpiredReveal, ensurePassCardFor } from "./turnFlow.js";
import { DEFAULT_LANGUAGE, translate as i18nTranslate } from "../ui/i18n/i18n.js";
import {
  packCard,
  finalizeTurn,
  continueAfterTurnIfNeeded,
  getDeclarableShakingMonths,
  getDeclarableBombMonths,
  getShakingReveal as selectShakingReveal
} from "./finalizeTurn.js";
export { getDeclarableShakingMonths, getDeclarableBombMonths };

function normalizeEngineLanguage(language) {
  return language === "en" ? "en" : DEFAULT_LANGUAGE;
}

function txLang(language, key, params = {}, fallback = "") {
  const lang = normalizeEngineLanguage(language);
  return i18nTranslate(lang, key, params, fallback);
}

function tx(state, key, params = {}, fallback = "") {
  return txLang(state?.language, key, params, fallback);
}

// ============================================================================
// SECTION 1. SETUP
// ----------------------------------------------------------------------------
// Game initialization and seeded RNG
// ============================================================================
export function createSeededRng(seedText = "") {
  let h = 1779033703 ^ seedText.length;
  for (let i = 0; i < seedText.length; i += 1) {
    h = Math.imul(h ^ seedText.charCodeAt(i), 3432918353);
    h = (h << 13) | (h >>> 19);
  }
  h = Math.imul(h ^ (h >>> 16), 2246822507);
  h = Math.imul(h ^ (h >>> 13), 3266489909);
  h ^= h >>> 16;

  return function seededRandom() {
    let t = (h += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function initGame(ruleKey = "A", seedRng = Math.random, options = {}) {
  let carryOverMultiplier = options.carryOverMultiplier ?? 1;
  const language = normalizeEngineLanguage(options.language);
  const cardTheme = normalizeCardTheme(options.cardTheme || DEFAULT_CARD_THEME);
  const kiboDetail = options.kiboDetail === "lean" ? "lean" : "full";
  const initialGoldBase =
    Number(options.initialGoldBase) > 0 ? Number(options.initialGoldBase) : STARTING_GOLD;
  const fixedFirstTurnKey =
    options.firstTurnKey === "human" || options.firstTurnKey === "ai"
      ? options.firstTurnKey
      : null;
  const carryLogs = [];

  while (true) {
    const deck = shuffle(buildDeck(cardTheme), seedRng);
    const players = {
      human: emptyPlayer(txLang(language, "player.human")),
      ai: emptyPlayer("AI")
    };
    if (options.initialGold?.human != null) {
      players.human.gold =
        Number(options.initialGold.human) > 0 ? options.initialGold.human : STARTING_GOLD;
    }
    if (options.initialGold?.ai != null) {
      players.ai.gold = Number(options.initialGold.ai) > 0 ? options.initialGold.ai : STARTING_GOLD;
    }
    if (options.initialMoney?.human != null) {
      players.human.gold =
        Number(options.initialMoney.human) > 0 ? options.initialMoney.human : STARTING_GOLD;
    }
    if (options.initialMoney?.ai != null) {
      players.ai.gold = Number(options.initialMoney.ai) > 0 ? options.initialMoney.ai : STARTING_GOLD;
    }

    players.human.hand = deck.slice(0, 10);
    players.ai.hand = deck.slice(10, 20);
    let board = deck.slice(20, 28);
    let remain = deck.slice(28);
    const firstTurnInfo = fixedFirstTurnKey
      ? {
          winnerKey: fixedFirstTurnKey,
          log: txLang(language, "label.firstTurn", {
            starter: fixedFirstTurnKey === "human" ? txLang(language, "player.human") : "AI"
          })
        }
      : decideFirstTurn(players.human.hand[0], players.ai.hand[0], seedRng);
    const initLog = [];
    initLog.push(firstTurnInfo.log);

    // Opening normalization for bonus cards (hand/board)
    remain = normalizeOpeningHands(players, remain, initLog);
    ({ board, remain } = normalizeOpeningBoard(
      board,
      remain,
      players[firstTurnInfo.winnerKey],
      initLog
    ));

    // Board president (4 cards of same month): starter instant win (10-point end)
    const boardPresident = findPresidentMonth(board);
    if (boardPresident !== null) {
      const winnerKey = firstTurnInfo.winnerKey;
      const winnerLabel = players[winnerKey].label;
      const baseScore = 10;
      const settled = settleRoundGold(players, winnerKey, baseScore);
      const openLog = [txLang(language, "log.gameStartRule", { ruleName: ruleSets[ruleKey].name })];
      const log = [...openLog, ...carryLogs, ...initLog];
      log.push(txLang(language, "log.boardPresidentWin", { month: boardPresident, winner: winnerLabel }));
      log.push(
        txLang(language, "log.roundGoldSettle", {
          winner: winnerLabel,
          requested: settled.requested,
          paid: settled.paid
        })
      );
      log.push(...settled.log);

      const human =
        winnerKey === "human"
          ? {
              base: baseScore,
              multiplier: 1,
              total: baseScore,
              payoutTotal: baseScore,
              bak: { gwang: false, pi: false, mongBak: false, multiplier: 1 },
              breakdown: { boardPresident: true, goBonus: 0 }
            }
          : {
              base: 0,
              multiplier: 1,
              total: 0,
              payoutTotal: 0,
              bak: { gwang: false, pi: false, mongBak: false, multiplier: 1 },
              breakdown: { goBonus: 0 }
            };
      const ai =
        winnerKey === "ai"
          ? {
              base: baseScore,
              multiplier: 1,
              total: baseScore,
              payoutTotal: baseScore,
              bak: { gwang: false, pi: false, mongBak: false, multiplier: 1 },
              breakdown: { boardPresident: true, goBonus: 0 }
            }
          : {
              base: 0,
              multiplier: 1,
              total: 0,
              payoutTotal: 0,
              bak: { gwang: false, pi: false, mongBak: false, multiplier: 1 },
              breakdown: { goBonus: 0 }
            };

      return {
        deck: remain,
        board,
        players: settled.updatedPlayers,
        currentTurn: winnerKey === "human" ? "ai" : "human",
        startingTurnKey: firstTurnInfo.winnerKey,
        language,
        cardTheme,
        initialGoldBase,
        phase: "resolution",
        pendingGoStop: null,
        pendingMatch: null,
        pendingShakingConfirm: null,
        pendingGukjinChoice: null,
        pendingPresident: null,
        shakingReveal: null,
        actionReveal: null,
        carryOverMultiplier,
        nextCarryOverMultiplier: 1,
        log,
        turnSeq: 0,
        kiboSeq: 2,
        passCardCounter: 0,
        kiboDetail,
        kibo: [
          {
            no: 1,
            type: "initial_deal",
            firstTurn: firstTurnInfo.winnerKey,
            ...(kiboDetail === "full"
              ? {
                  hands: {
                    human: players.human.hand.map(packCard),
                    ai: players.ai.hand.map(packCard)
                  },
                  board: board.map(packCard),
                  deck: remain.map(packCard)
                }
              : {
                  handsCount: {
                    human: players.human.hand.length,
                    ai: players.ai.hand.length
                  },
                  boardCount: board.length,
                  deckCount: remain.length
                })
          },
          {
            no: 2,
            type: "board_president_stop",
            winner: winnerKey,
            month: boardPresident,
            payout: baseScore
          }
        ],
        ruleKey,
        result: {
          human,
          ai,
          winner: winnerKey,
          nagari: false,
          nagariReasons: [],
          gold: {
            requested: settled.requested,
            paid: settled.paid,
            unitPerPoint: POINT_GOLD_UNIT
          }
        }
      };
    }

    let phase = "playing";
    let pendingPresident = null;
    const handPresident = findPresidentMonth(players[firstTurnInfo.winnerKey].hand);
    if (handPresident !== null) {
      phase = "president-choice";
      pendingPresident = {
        playerKey: firstTurnInfo.winnerKey,
        month: handPresident
      };
      initLog.push(
        txLang(language, "log.handPresidentAvailable", {
          player: players[firstTurnInfo.winnerKey].label,
          month: handPresident
        })
      );
    }

    const openLog = [txLang(language, "log.gameStartRule", { ruleName: ruleSets[ruleKey].name })];
    if (carryOverMultiplier > 1) {
      openLog.push(txLang(language, "log.carryOverMultiplier", { multiplier: carryOverMultiplier }));
    }

    return {
      deck: remain,
      board,
      players,
      currentTurn: firstTurnInfo.winnerKey,
      startingTurnKey: firstTurnInfo.winnerKey,
      language,
      cardTheme,
      initialGoldBase,
      phase,
      pendingGoStop: null,
      pendingMatch: null,
      pendingShakingConfirm: null,
      pendingGukjinChoice: null,
      pendingPresident,
      shakingReveal: null,
      actionReveal: null,
      carryOverMultiplier,
      nextCarryOverMultiplier: 1,
      log: [...openLog, ...carryLogs, ...initLog],
      turnSeq: 0,
      kiboSeq: 1,
      passCardCounter: 0,
      kiboDetail,
      kibo: [
        {
          no: 1,
          type: "initial_deal",
          firstTurn: firstTurnInfo.winnerKey,
          ...(kiboDetail === "full"
            ? {
                hands: {
                  human: players.human.hand.map(packCard),
                  ai: players.ai.hand.map(packCard)
                },
                board: board.map(packCard),
                deck: remain.map(packCard)
              }
            : {
                handsCount: {
                  human: players.human.hand.length,
                  ai: players.ai.hand.length
                },
                boardCount: board.length,
                deckCount: remain.length
              })
        }
      ],
      ruleKey,
      result: null
    };
  }
}

function isPlayerKey(key) {
  return key === "human" || key === "ai";
}

function deriveFirstTurnFromPrevious(previousState) {
  const previousWinner = previousState?.result?.winner;
  if (isPlayerKey(previousWinner)) return previousWinner;

  const wasNagari = !!previousState?.result?.nagari;
  const previousStarter = previousState?.startingTurnKey;
  if (wasNagari && isPlayerKey(previousStarter)) return previousStarter;

  return isPlayerKey(previousStarter) ? previousStarter : null;
}

export function startGameFromState(previousState, seedRng = Math.random, options = {}) {
  const nextOptions = { ...options };
  const explicitRuleKey = typeof nextOptions.ruleKey === "string" ? nextOptions.ruleKey : "";
  const ruleKey = explicitRuleKey || previousState?.ruleKey || "A";
  const keepGold = nextOptions.keepGold !== false;
  const useCarryOver = nextOptions.useCarryOver !== false;

  const explicitFirstTurnKey = isPlayerKey(nextOptions.firstTurnKey) ? nextOptions.firstTurnKey : null;
  const derivedFirstTurnKey = deriveFirstTurnFromPrevious(previousState);

  const hasExplicitInitialGold = nextOptions.initialGold?.human != null || nextOptions.initialGold?.ai != null;
  if (!hasExplicitInitialGold && keepGold && previousState?.players) {
    nextOptions.initialGold = {
      human: Number(previousState.players?.human?.gold ?? STARTING_GOLD),
      ai: Number(previousState.players?.ai?.gold ?? STARTING_GOLD)
    };
  }

  if (nextOptions.carryOverMultiplier == null) {
    nextOptions.carryOverMultiplier = useCarryOver
      ? Number(previousState?.nextCarryOverMultiplier || 1)
      : 1;
  }

  nextOptions.firstTurnKey = explicitFirstTurnKey || derivedFirstTurnKey;

  delete nextOptions.ruleKey;
  delete nextOptions.keepGold;
  delete nextOptions.useCarryOver;

  return initGame(ruleKey, seedRng, nextOptions);
}

export function initSimulationGame(ruleKey = "A", seedRng = Math.random, options = {}) {
  const nextOptions = { ...options };
  if (!isPlayerKey(nextOptions.firstTurnKey)) {
    throw new Error("initSimulationGame requires options.firstTurnKey ('human' or 'ai').");
  }
  return initGame(ruleKey, seedRng, nextOptions);
}

export function startSimulationGame(previousState, seedRng = Math.random, options = {}) {
  const nextOptions = { ...options };
  if (!isPlayerKey(nextOptions.firstTurnKey)) {
    throw new Error("startSimulationGame requires options.firstTurnKey ('human' or 'ai').");
  }
  if (nextOptions.keepGold == null) nextOptions.keepGold = true;
  if (nextOptions.useCarryOver == null) nextOptions.useCarryOver = true;
  return startGameFromState(previousState, seedRng, nextOptions);
}




// ============================================================================
// SECTION 2. CORE ACTIONS
// ----------------------------------------------------------------------------
// Core turn entry points and match-choice action
// ============================================================================
export function playTurn(state, cardId) {
  if (state.phase !== "playing") return state;
  state = clearExpiredReveal(state);

  const currentKey = state.currentTurn;
  const player = state.players[currentKey];
  const idx = player.hand.findIndex((c) => c.id === cardId);
  if (idx < 0) return state;

  const playedCard = player.hand[idx];
  const startedWithLastHand = player.hand.length === 1;
  const isLastHandTurn = startedWithLastHand;
  if (playedCard.passCard) {
    const hand = player.hand.filter((c) => c.id !== playedCard.id);
    const captured = {
      ...player.captured,
      kwang: player.captured.kwang.slice(),
      five: player.captured.five.slice(),
      ribbon: player.captured.ribbon.slice(),
      junk: player.captured.junk.slice()
    };
    const events = { ...player.events };
    const log = state.log.concat(`${player.label}: Dummy pass card used (consume card + flip once)`);
    const newlyCaptured = [];

    const flipResult = runFlipPhase({
      state,
      currentKey,
      playedCard,
      isLastHandTurn,
      board: state.board.slice(),
      deck: state.deck.slice(),
      hand,
      captured,
      events,
      log,
      newlyCaptured,
      pendingSteal: 0,
      pendingBonusFlips: []
    });
    if (flipResult.pendingState) return flipResult.pendingState;

    return finalizeTurn({
      state,
      currentKey,
      hand,
      captured,
      events,
      deck: flipResult.deck,
      board: flipResult.board,
      log: flipResult.log,
      newlyCaptured: flipResult.newlyCaptured,
      pendingSteal: flipResult.pendingSteal,
      heldBonusCards: flipResult.heldBonusOnPpuk || [],
      isLastHandTurn,
      turnMeta: {
        type: "pass",
        card: packCard(playedCard),
        flips: flipResult.flips || [],
        matchEvents: flipResult.matchEvents || [],
        captureBySource: {
          hand: [],
          flip: (flipResult.capturedFromFlip || []).map(packCard)
        }
      }
    });
  }

  const hand = player.hand.slice();
  hand.splice(idx, 1);

  if (playedCard.bonus?.stealPi) {
    const captured = { ...player.captured };
    captured.kwang = captured.kwang.slice();
    captured.five = captured.five.slice();
    captured.ribbon = captured.ribbon.slice();
    captured.junk = captured.junk.slice();
    pushCaptured(captured, playedCard);

    let nextDeck = state.deck.slice();
    let drawnToHand = null;
    if (nextDeck.length > 0) {
      drawnToHand = nextDeck[0];
      nextDeck = nextDeck.slice(1);
      hand.push(drawnToHand);
    }

    const events = { ...player.events };
    let nextLog = state.log
      .concat(
        tx(state, "log.bonusUse", {
          player: player.label,
          cardName: playedCard.name,
          stealPi: playedCard.bonus.stealPi
        })
      )
      .concat(
        drawnToHand
          ? tx(state, "log.bonusDraw", {
              player: player.label,
              month: drawnToHand.month,
              cardName: drawnToHand.name
            })
          : tx(state, "log.bonusDrawFail", { player: player.label })
      );

    let nextPlayers = {
      ...state.players,
      [currentKey]: {
        ...player,
        hand,
        captured,
        events
      }
    };

    // Bonus steal applies only when this is not the last-hand turn.
    if (!startedWithLastHand && playedCard.bonus.stealPi > 0) {
      const { updatedPlayers, stealLog } = stealPiFromOpponent(nextPlayers, currentKey, playedCard.bonus.stealPi);
      nextPlayers = updatedPlayers;
      nextLog = nextLog.concat(stealLog);
    }

    const bonusPresidentMonth = findPresidentMonth(hand);
    if (bonusPresidentMonth !== null) {
      return {
        ...state,
        players: nextPlayers,
        deck: nextDeck,
        board: state.board.slice(),
        phase: "president-choice",
        pendingMatch: null,
        pendingGoStop: null,
        pendingShakingConfirm: null,
        pendingGukjinChoice: null,
        pendingPresident: { playerKey: currentKey, month: bonusPresidentMonth },
        actionReveal: {
          type: "bonus",
          title: tx(state, "reveal.bonus.title"),
          message: tx(state, "reveal.bonus.toPresident", { player: player.label }),
          cards: [packCard(playedCard)],
          expiresAt: Date.now() + 2000
        },
        log: nextLog.concat(
          tx(state, "log.bonusPresidentAvailable", {
            player: player.label,
            month: bonusPresidentMonth
          })
        ),
        kiboSeq: (state.kiboSeq || 0) + 1,
        kibo: (state.kibo || []).concat({
          no: (state.kiboSeq || 0) + 1,
          type: "bonus_use",
          playerKey: currentKey,
          card: packCard(playedCard),
          drawToHand: drawnToHand ? packCard(drawnToHand) : null
        })
      };
    }

    return {
      ...state,
      players: nextPlayers,
      deck: nextDeck,
      board: state.board.slice(),
      phase: "playing",
      pendingMatch: null,
      pendingGoStop: null,
      pendingShakingConfirm: null,
      pendingGukjinChoice: null,
      actionReveal: {
        type: "bonus",
        title: tx(state, "reveal.bonus.title"),
        message: tx(state, "reveal.bonus.extraHand", { player: player.label }),
        cards: [packCard(playedCard)],
        expiresAt: Date.now() + 2000
      },
      log: nextLog.concat(tx(state, "log.bonusContinueTurn", { player: player.label })),
      kiboSeq: (state.kiboSeq || 0) + 1,
      kibo: (state.kibo || []).concat({
        no: (state.kiboSeq || 0) + 1,
        type: "bonus_use",
        playerKey: currentKey,
        card: packCard(playedCard),
        drawToHand: drawnToHand ? packCard(drawnToHand) : null
      })
    };
  }


  const remainingSameMonthCount = hand.filter((c) => c.month === playedCard.month).length;
  const presidentChainArmed =
    !!player.presidentHold &&
    player.presidentHoldMonth === playedCard.month &&
    remainingSameMonthCount === 3;

  let board = state.board.slice();
  const captured = { ...player.captured };
  let events = { ...player.events };
  let log = state.log.slice();
  const newlyCaptured = [];
  const capturedFromHand = [];
  let pendingSteal = 0;
  const matchEvents = [];
  const handMatch = resolveMatch({
    card: playedCard,
    board,
    source: "hand",
    isLastHandTurn,
    playedMonth: playedCard.month
  });

  if (handMatch.type === "NONE") {
    board.push(playedCard);
    log.push(tx(state, "log.playNoMatch", { player: player.label, month: playedCard.month }));
  } else if (handMatch.type === "ONE") {
    const matched = handMatch.matches[0];
    board = board.filter((c) => c.id !== matched.id);
    pushCaptured(captured, matched);
    pushCaptured(captured, playedCard);
    newlyCaptured.push(matched, playedCard);
    capturedFromHand.push(matched, playedCard);
    log.push(tx(state, "log.playOneMatchCapture", { player: player.label, month: playedCard.month }));
    if (!isLastHandTurn && presidentChainArmed) {
      events.shaking += 1;
      log.push(tx(state, "log.presidentChainShake", { player: player.label, month: playedCard.month }));
    }
  } else if (handMatch.type === "TWO") {
    if (handMatch.needsChoice) {
      return {
        ...state,
        phase: "select-match",
        pendingMatch: {
          stage: "player",
          playerKey: currentKey,
          cardId,
          boardCardIds: handMatch.matches.map((c) => c.id),
          presidentChainArmed,
          message: tx(state, "pending.match.player", { player: player.label })
        },
        log: state.log.concat(
          tx(state, "log.playTwoMatchWait", { player: player.label, month: playedCard.month })
        )
      };
    }
    const matched = bestMatchCard(handMatch.matches);
    board = board.filter((c) => c.id !== matched.id);
    pushCaptured(captured, matched);
    pushCaptured(captured, playedCard);
    newlyCaptured.push(matched, playedCard);
    capturedFromHand.push(matched, playedCard);
    log.push(tx(state, "log.playTwoMatchAuto", { player: player.label, month: playedCard.month }));
    if (!isLastHandTurn && presidentChainArmed) {
      events.shaking += 1;
      log.push(tx(state, "log.presidentChainShake", { player: player.label, month: playedCard.month }));
    }
  } else {
    board = board.filter((c) => c.month !== playedCard.month);
    handMatch.matches.forEach((m) => {
      pushCaptured(captured, m);
      newlyCaptured.push(m);
      capturedFromHand.push(m);
    });
    pushCaptured(captured, playedCard);
    newlyCaptured.push(playedCard);
    capturedFromHand.push(playedCard);
    if (handMatch.eventTag === "PANSSEUL") events.pansseul = (events.pansseul || 0) + 1;
    log.push(tx(state, "log.playSweepCapture", { player: player.label, month: playedCard.month }));
    if (!isLastHandTurn) {
      pendingSteal += 1;
      log.push(tx(state, "log.pansseulReserved", { player: player.label }));
    }
    if (!isLastHandTurn && presidentChainArmed) {
      events.shaking += 1;
      log.push(tx(state, "log.presidentChainShake", { player: player.label, month: playedCard.month }));
    }
  }
  matchEvents.push({ source: "hand", eventTag: handMatch.eventTag, type: handMatch.type });

  const flipResult = runFlipPhase({
    state,
    currentKey,
    playedCard,
    isLastHandTurn,
    board,
    deck: state.deck.slice(),
    hand,
    captured,
    events,
    log,
    newlyCaptured,
    pendingSteal,
    pendingBonusFlips: []
  });

  if (flipResult.pendingState) return flipResult.pendingState;

  return finalizeTurn({
    state,
    currentKey,
    hand,
    captured,
    events,
    deck: flipResult.deck,
    board: flipResult.board,
    log: flipResult.log,
    newlyCaptured: flipResult.newlyCaptured,
    pendingSteal: flipResult.pendingSteal,
    heldBonusCards: flipResult.heldBonusOnPpuk || [],
    isLastHandTurn,
    turnMeta: {
      type: "play",
      card: packCard(playedCard),
      flips: flipResult.flips || [],
      matchEvents: matchEvents.concat(flipResult.matchEvents || []),
      captureBySource: {
        hand: capturedFromHand.map(packCard),
        flip: (flipResult.capturedFromFlip || []).map(packCard)
      }
    }
  });
}

export function chooseMatch(state, boardCardId) {
  if (state.phase !== "select-match" || !state.pendingMatch) return state;

  const pending = state.pendingMatch;
  if (state.currentTurn !== pending.playerKey) return state;
  if (!pending.boardCardIds.includes(boardCardId)) return state;

  if (pending.stage === "player") {
    return resolvePlayerMatchChoice(state, boardCardId);
  }
  if (pending.stage === "flip") {
    return resolveFlipMatchChoice(state, boardCardId);
  }

  return state;
}


// ============================================================================
// SECTION 3. EVENT ACTIONS
// ----------------------------------------------------------------------------
// Special event actions: shaking, bomb, go/stop, president, gukjin
// ============================================================================
export function declareShaking(state, playerKey, month) {
  if (state.phase !== "playing" || state.currentTurn !== playerKey) return state;
  const monthNum = Number(month);
  const available = getDeclarableShakingMonths(state, playerKey);
  if (!available.includes(monthNum)) return state;

  const player = state.players[playerKey];
  const nextDeclared = [...(player.shakingDeclaredMonths || []), monthNum];
  const revealCards = player.hand.filter((c) => c.month === monthNum).slice(0, 3);
  const nextPlayers = {
    ...state.players,
    [playerKey]: {
      ...player,
      shakingDeclaredMonths: nextDeclared,
      events: { ...player.events, shaking: player.events.shaking + 1 }
    }
  };

  return {
    ...state,
    players: nextPlayers,
    shakingReveal: {
      playerKey,
      month: monthNum,
      cards: revealCards.map(packCard),
      expiresAt: Date.now() + 2000
    },
    actionReveal: {
      type: "shaking",
      title: tx(state, "reveal.shaking.title"),
      message: tx(state, "reveal.shaking.message", { player: player.label }),
      cards: revealCards.map(packCard),
      expiresAt: Date.now() + 2000
    },
    log: state.log.concat(tx(state, "log.shakingDeclare", { player: player.label, month: monthNum })),
    kibo: state.kibo.concat({
      no: (state.kiboSeq || 0) + 1,
      type: "shaking_declare",
      playerKey,
      month: monthNum,
      revealCards: revealCards.map(packCard),
      revealMs: 2000
    }),
    kiboSeq: (state.kiboSeq || 0) + 1
  };
}


export function declareBomb(state, playerKey, month) {
  if (state.phase !== "playing" || state.currentTurn !== playerKey) return state;
  const monthNum = Number(month);
  const bombable = getDeclarableBombMonths(state, playerKey);
  if (!bombable.includes(monthNum)) return state;

  const player = state.players[playerKey];
  const monthCards = player.hand.filter((c) => c.month === monthNum);
  const matched = state.board.find((c) => c.month === monthNum);
  if (!matched || monthCards.length < 3) return state;

  const hand = player.hand.filter((c) => c.month !== monthNum);
  const board = state.board.filter((c) => c.id !== matched.id);
  const captured = {
    ...player.captured,
    kwang: player.captured.kwang.slice(),
    five: player.captured.five.slice(),
    ribbon: player.captured.ribbon.slice(),
    junk: player.captured.junk.slice()
  };
  const events = { ...player.events, bomb: player.events.bomb + 1 };
  const log = state.log.concat(
    tx(state, "log.bombDeclare", { player: player.label, month: monthNum, count: monthCards.length })
  );
  const newlyCaptured = [];
  const capturedFromHand = [];
  pushCaptured(captured, matched);
  newlyCaptured.push(matched);
  capturedFromHand.push(matched);
  monthCards.forEach((c) => {
    pushCaptured(captured, c);
    newlyCaptured.push(c);
    capturedFromHand.push(c);
  });

  const flipResult = runFlipPhase({
    state,
    currentKey: playerKey,
    playedCard: { id: `bomb-declare-${monthNum}`, month: monthNum, category: "junk", name: "Bomb Declare" },
    isLastHandTurn: hand.length === 0,
    board,
    deck: state.deck.slice(),
    hand,
    captured,
    events,
    log,
    newlyCaptured,
    pendingSteal: 1
  });
  if (flipResult.pendingState) return flipResult.pendingState;
  const bombState = {
    ...state,
    actionReveal: {
      type: "bomb",
      title: tx(state, "reveal.bomb.title"),
      message: tx(state, "reveal.bomb.message", { player: player.label }),
      cards: [matched, ...monthCards].map(packCard),
      expiresAt: Date.now() + 2000
    }
  };
  return finalizeTurn({
    state: bombState,
    currentKey: playerKey,
    hand,
    captured,
    events,
    deck: flipResult.deck,
    board: flipResult.board,
    log: flipResult.log,
    newlyCaptured: flipResult.newlyCaptured,
    pendingSteal: flipResult.pendingSteal,
    turnMeta: {
      type: "declare_bomb",
      month: monthNum,
      captured: [packCard(matched), ...monthCards.map(packCard)],
      flips: flipResult.flips || [],
      captureBySource: {
        hand: capturedFromHand.map(packCard),
        flip: (flipResult.capturedFromFlip || []).map(packCard)
      }
    }
  });
}

export function askShakingConfirm(state, playerKey, cardId) {
  if (state.phase !== "playing" || state.currentTurn !== playerKey) return state;
  const player = state.players[playerKey];
  if (!player) return state;
  const selected = player.hand.find((c) => c.id === cardId);
  if (!selected) return state;
  const available = getDeclarableShakingMonths(state, playerKey);
  if (!available.includes(selected.month)) return state;

  return {
    ...state,
    phase: "shaking-confirm",
    pendingShakingConfirm: {
      playerKey,
      cardId,
      month: selected.month
    }
  };
}

export function chooseShakingYes(state, playerKey) {
  if (state.phase !== "shaking-confirm" || state.pendingShakingConfirm?.playerKey !== playerKey) {
    return state;
  }
  const { cardId, month } = state.pendingShakingConfirm;
  const resumed = {
    ...state,
    phase: "playing",
    pendingShakingConfirm: null
  };
  const declared = declareShaking(resumed, playerKey, month);
  return playTurn(declared, cardId);
}

export function chooseShakingNo(state, playerKey) {
  if (state.phase !== "shaking-confirm" || state.pendingShakingConfirm?.playerKey !== playerKey) {
    return state;
  }
  const { cardId } = state.pendingShakingConfirm;
  const resumed = {
    ...state,
    phase: "playing",
    pendingShakingConfirm: null
  };
  return playTurn(resumed, cardId);
}

export function getShakingReveal(state, now = Date.now()) {
  return selectShakingReveal(state, now);
}

export function chooseGo(state, playerKey) {
  if (state.phase !== "go-stop" || state.pendingGoStop !== playerKey) return state;

  const player = state.players[playerKey];
  const opponentKey = playerKey === "human" ? "ai" : "human";
  const rules = ruleSets[state.ruleKey];
  const baseInfo = calculateBaseScore(state.players[playerKey], rules);
  const nextPlayers = {
    ...state.players,
    [playerKey]: {
      ...player,
      goCount: player.goCount + 1,
      lastGoBase: baseInfo.base
    }
  };

  const log = state.log.concat(tx(state, "log.goDeclare", { player: player.label, goCount: player.goCount + 1 }));

  const nextState = {
    ...state,
    players: nextPlayers,
    currentTurn: opponentKey,
    phase: "playing",
    pendingGoStop: null,
    pendingMatch: null,
    pendingShakingConfirm: null,
    log,
    kiboSeq: (state.kiboSeq || 0) + 1,
    kibo: (state.kibo || []).concat({
      no: (state.kiboSeq || 0) + 1,
      type: "go",
      playerKey,
      goCount: player.goCount + 1
    })
  };
  return ensurePassCardFor(nextState, nextState.currentTurn);
}

export function chooseStop(state, playerKey) {
  if (state.phase !== "go-stop" || state.pendingGoStop !== playerKey) return state;

  const player = state.players[playerKey];
  const nextPlayers = {
    ...state.players,
    [playerKey]: { ...player, declaredStop: true }
  };

  const log = state.log.concat(tx(state, "log.stopDeclare", { player: player.label }));
  let nextState = {
    ...state,
    players: nextPlayers,
    phase: "resolution",
    pendingGoStop: null,
    pendingMatch: null,
    pendingShakingConfirm: null,
    log,
    kiboSeq: (state.kiboSeq || 0) + 1,
    kibo: (state.kibo || []).concat({
      no: (state.kiboSeq || 0) + 1,
      type: "stop",
      playerKey
    })
  };

  nextState = resolveRound(nextState, playerKey);
  return nextState;
}

export function choosePresidentStop(state, playerKey) {
  if (state.phase !== "president-choice" || state.pendingPresident?.playerKey !== playerKey) {
    return state;
  }

  const baseScore = 10;
  const payout = baseScore * (state.carryOverMultiplier || 1);
  let log = state.log.concat(
    tx(state, "log.presidentStopChoose", { player: state.players[playerKey].label, payout })
  );
  const settled = settleRoundGold(state.players, playerKey, payout);
  log = log.concat(
    tx(state, "log.roundGoldSettle", {
      winner: state.players[playerKey].label,
      requested: settled.requested,
      paid: settled.paid
    })
  );

  const human =
    playerKey === "human"
      ? {
          base: baseScore,
          multiplier: state.carryOverMultiplier || 1,
          total: payout,
          bak: { gwang: false, pi: false, mongBak: false, multiplier: 1 },
          breakdown: { presidentStop: true, goBonus: 0 }
        }
      : {
          base: 0,
          multiplier: 1,
          total: 0,
          bak: { gwang: false, pi: false, mongBak: false, multiplier: 1 },
          breakdown: { goBonus: 0 }
        };
  const ai =
    playerKey === "ai"
      ? {
          base: baseScore,
          multiplier: state.carryOverMultiplier || 1,
          total: payout,
          bak: { gwang: false, pi: false, mongBak: false, multiplier: 1 },
          breakdown: { presidentStop: true, goBonus: 0 }
        }
      : {
          base: 0,
          multiplier: 1,
          total: 0,
          bak: { gwang: false, pi: false, mongBak: false, multiplier: 1 },
          breakdown: { goBonus: 0 }
        };

  return {
    ...state,
    phase: "resolution",
    pendingPresident: null,
    pendingGoStop: null,
    pendingMatch: null,
    pendingShakingConfirm: null,
    nextCarryOverMultiplier: 1,
    players: settled.updatedPlayers,
    kiboSeq: (state.kiboSeq || 0) + 1,
    kibo: (state.kibo || []).concat({
      no: (state.kiboSeq || 0) + 1,
      type: "president_stop",
      playerKey,
      payout
    }),
    result: {
      human,
      ai,
      winner: playerKey,
      nagari: false,
      nagariReasons: [],
      gold: {
        requested: settled.requested,
        paid: settled.paid,
        unitPerPoint: POINT_GOLD_UNIT
      }
    },
    log
  };
}

export function choosePresidentHold(state, playerKey) {
  if (state.phase !== "president-choice" || state.pendingPresident?.playerKey !== playerKey) {
    return state;
  }
  const player = state.players[playerKey];
  const month = state.pendingPresident?.month ?? null;
  const nextPlayers = {
    ...state.players,
    [playerKey]: { ...player, presidentHold: true, presidentHoldMonth: month }
  };
  const log = state.log.concat(
    tx(state, "log.presidentHoldChoose", { player: state.players[playerKey].label, month })
  );
  return {
    ...state,
    players: nextPlayers,
    phase: "playing",
    pendingPresident: null,
    pendingShakingConfirm: null,
    log,
    kiboSeq: (state.kiboSeq || 0) + 1,
    kibo: (state.kibo || []).concat({
      no: (state.kiboSeq || 0) + 1,
      type: "president_hold",
      playerKey,
      month
    })
  };
}

export function chooseGukjinMode(state, playerKey, mode) {
  if (state.phase !== "gukjin-choice" || state.pendingGukjinChoice?.playerKey !== playerKey) {
    return state;
  }
  if (!state.players[playerKey]) return state;
  if (mode !== "five" && mode !== "junk") return state;

  const player = state.players[playerKey];
  if (player.gukjinLocked) return state;
  const captured = {
    ...player.captured,
    kwang: (player.captured.kwang || []).slice(),
    five: (player.captured.five || []).slice(),
    ribbon: (player.captured.ribbon || []).slice(),
    junk: (player.captured.junk || []).slice()
  };
  if (mode === "junk") {
    const gukjinIdx = captured.five.findIndex(
      (card) => isGukjinCard(card) && card.category === "five" && !card.gukjinTransformed
    );
    if (gukjinIdx >= 0) {
      const [gukjinCard] = captured.five.splice(gukjinIdx, 1);
      captured.junk.push({
        ...gukjinCard,
        category: "junk",
        piValue: 2,
        gukjinTransformed: true,
        name: `${gukjinCard.name}${tx(state, "label.gukjinSuffix")}`
      });
    }
  } else if (mode === "five") {
    const gukjinJunkIdx = captured.junk.findIndex(
      (card) => isGukjinCard(card) && card.category === "junk"
    );
    if (gukjinJunkIdx >= 0) {
      const [gukjinCard] = captured.junk.splice(gukjinJunkIdx, 1);
      captured.five.push({
        ...gukjinCard,
        category: "five",
        piValue: undefined,
        gukjinTransformed: false,
        name: String(gukjinCard.name || "").replace(tx(state, "label.gukjinSuffix"), "")
      });
    }
  }
  const nextPlayers = {
    ...state.players,
    [playerKey]: { ...player, captured, gukjinMode: mode, gukjinLocked: true }
  };
  const modeLabel = mode === "junk" ? tx(state, "label.gukjin.junk") : tx(state, "label.gukjin.five");
  const log = state.log.concat(tx(state, "log.gukjinModeChoose", { player: player.label, modeLabel }));

  const nextState = {
    ...state,
    players: nextPlayers,
    phase: "playing",
    pendingGukjinChoice: null,
    pendingShakingConfirm: null,
    log,
    kiboSeq: (state.kiboSeq || 0) + 1,
    kibo: (state.kibo || []).concat({
      no: (state.kiboSeq || 0) + 1,
      type: "gukjin_mode",
      playerKey,
      mode
    })
  };
  return continueAfterTurnIfNeeded(nextState, playerKey);
}











// ============================================================================
// SECTION 4. HELPERS
// ----------------------------------------------------------------------------
// Internal helper logic used by core/event actions
// ============================================================================
function runFlipPhase(context) {
  let {
    board,
    deck,
    hand,
    captured,
    events,
    log,
    newlyCaptured,
    pendingSteal = 0,
    pendingBonusFlips = []
  } = context;
  let heldBonusOnPpuk = [];
  const flips = [];
  const matchEvents = [];
  const capturedFromFlip = [];
  const { state, currentKey, playedCard, isLastHandTurn = false } = context;
  while (deck.length > 0) {
    const flip = deck[0];
    deck = deck.slice(1);
    flips.push(packCard(flip));

    if (flip.bonus?.stealPi) {
      pendingBonusFlips = pendingBonusFlips.concat(flip);
      log.push(tx(state, "log.flipHoldForRedraw", { cardName: flip.name }));
      continue;
    }

    const flipMatch = resolveMatch({
      card: flip,
      board,
      source: "flip",
      isLastHandTurn,
      playedMonth: playedCard.month
    });
    matchEvents.push({ source: "flip", eventTag: flipMatch.eventTag, type: flipMatch.type });

    if (flipMatch.type === "NONE") {
      if (pendingBonusFlips.length > 0) {
        pendingBonusFlips.forEach((b) => {
          pushCaptured(captured, b);
          newlyCaptured.push(b);
          capturedFromFlip.push(b);
        });
        log.push(tx(state, "log.flipPendingBonusConfirmed", { count: pendingBonusFlips.length }));
        pendingBonusFlips = [];
      }
      board.push(flip);
      log.push(tx(state, "log.flipPlaceBoard", { month: flip.month }));
      if (flipMatch.eventTag === "JJOB") {
        events.jjob += 1;
        pendingSteal += 1;
        log.push(tx(state, "log.flipJjobReserved"));
      }
      return {
        board,
        deck,
        captured,
        events,
        log,
        newlyCaptured,
        pendingSteal,
        flips,
        matchEvents,
        capturedFromFlip
      };
    }

    if (flipMatch.type === "ONE") {
      if (pendingBonusFlips.length > 0) {
        pendingBonusFlips.forEach((b) => {
          pushCaptured(captured, b);
          newlyCaptured.push(b);
          capturedFromFlip.push(b);
        });
        log.push(tx(state, "log.flipPendingBonusConfirmed", { count: pendingBonusFlips.length }));
        pendingBonusFlips = [];
      }
      const matched = flipMatch.matches[0];
      board = board.filter((c) => c.id !== matched.id);
      pushCaptured(captured, flip);
      pushCaptured(captured, matched);
      newlyCaptured.push(flip, matched);
      capturedFromFlip.push(flip, matched);
      log.push(tx(state, "log.flipMatchCapture", { month: flip.month }));
      if (flipMatch.eventTag === "DDADAK") {
        events.ddadak = (events.ddadak || 0) + 1;
        pendingSteal += 1;
        log.push(tx(state, "log.flipDdadakReserved"));
      }
      return {
        board,
        deck,
        captured,
        events,
        log,
        newlyCaptured,
        pendingSteal,
        flips,
        matchEvents,
        capturedFromFlip
      };
    }

    if (flipMatch.type === "TWO" && flipMatch.needsChoice) {
      return {
        pendingState: {
          ...state,
          phase: "select-match",
          pendingMatch: {
            stage: "flip",
            playerKey: currentKey,
            boardCardIds: flipMatch.matches.map((c) => c.id),
            message: tx(state, "pending.match.flip", { player: state.players[currentKey].label }),
            context: {
              hand,
              captured,
              events,
              deck,
              board,
              log,
              newlyCaptured,
              flipCard: flip,
              pendingSteal,
              flips,
              pendingBonusFlips,
              matchEvents,
              capturedFromFlip
            }
          },
          log: log.concat(tx(state, "log.flipTwoMatchWait", { month: flip.month }))
        }
      };
    }

    if (flipMatch.type === "TWO") {
      if (!isLastHandTurn && pendingBonusFlips.length > 0) {
        log.push(tx(state, "log.ppukHoldBonus", { count: pendingBonusFlips.length }));
        heldBonusOnPpuk = heldBonusOnPpuk.concat(pendingBonusFlips);
        pendingBonusFlips = [];
      } else if (pendingBonusFlips.length > 0) {
        pendingBonusFlips.forEach((b) => {
          pushCaptured(captured, b);
          newlyCaptured.push(b);
          capturedFromFlip.push(b);
        });
        log.push(tx(state, "log.flipPendingBonusConfirmed", { count: pendingBonusFlips.length }));
        pendingBonusFlips = [];
      }
      if (flipMatch.eventTag === "PPUK") events.ppuk += 1;
      const matched = bestMatchCard(flipMatch.matches);
      board = board.filter((c) => c.id !== matched.id);
      pushCaptured(captured, flip);
      pushCaptured(captured, matched);
      newlyCaptured.push(flip, matched);
      capturedFromFlip.push(flip, matched);
      log.push(tx(state, "log.flipMultiMatchCapture", { month: flip.month }));
      return {
        board,
        deck,
        captured,
        events,
        log,
        newlyCaptured,
        pendingSteal,
        flips,
        heldBonusOnPpuk,
        matchEvents,
        capturedFromFlip
      };
    }

    if (flipMatch.eventTag === "PPUK") events.ppuk += 1;
    if (!isLastHandTurn && pendingBonusFlips.length > 0) {
      log.push(tx(state, "log.ppukHoldBonus", { count: pendingBonusFlips.length }));
      heldBonusOnPpuk = heldBonusOnPpuk.concat(pendingBonusFlips);
      pendingBonusFlips = [];
    } else if (pendingBonusFlips.length > 0) {
      pendingBonusFlips.forEach((b) => {
        pushCaptured(captured, b);
        newlyCaptured.push(b);
        capturedFromFlip.push(b);
      });
      log.push(tx(state, "log.flipPendingBonusConfirmed", { count: pendingBonusFlips.length }));
      pendingBonusFlips = [];
    }
    board = board.filter((c) => c.month !== flip.month);
    flipMatch.matches.forEach((m) => {
      pushCaptured(captured, m);
      newlyCaptured.push(m);
      capturedFromFlip.push(m);
    });
    pushCaptured(captured, flip);
    newlyCaptured.push(flip);
    capturedFromFlip.push(flip);
    log.push(tx(state, "log.flipFourCapture", { month: flip.month }));
    return {
      board,
      deck,
      captured,
      events,
      log,
      newlyCaptured,
      pendingSteal,
      flips,
      heldBonusOnPpuk,
      matchEvents,
      capturedFromFlip
    };
  }

  if (pendingBonusFlips.length > 0) {
    pendingBonusFlips.forEach((b) => {
      pushCaptured(captured, b);
      newlyCaptured.push(b);
      capturedFromFlip.push(b);
    });
    log.push(tx(state, "log.flipPendingBonusConfirmed", { count: pendingBonusFlips.length }));
    pendingBonusFlips = [];
  }
  return {
    board,
    deck,
    captured,
    events,
    log,
    newlyCaptured,
    pendingSteal,
    flips,
    heldBonusOnPpuk,
    matchEvents,
    capturedFromFlip
  };
}

function resolvePlayerMatchChoice(state, boardCardId) {
  const { playerKey, cardId, presidentChainArmed = false } = state.pendingMatch;
  const player = state.players[playerKey];
  const idx = player.hand.findIndex((c) => c.id === cardId);
  if (idx < 0) return state;

  const playedCard = player.hand[idx];
  const startedWithLastHand = player.hand.length === 1;
  const hand = player.hand.slice();
  hand.splice(idx, 1);

  const board = state.board.slice();
  const selected = board.find((c) => c.id === boardCardId);
  if (!selected) return state;

  let boardAfter = board.filter((c) => c.id !== selected.id);
  const captured = { ...player.captured };
  const events = { ...player.events };
  let log = state.log.concat(
    tx(state, "log.selectCapture", { player: player.label, month: playedCard.month, cardName: selected.name })
  );
  const newlyCaptured = [playedCard, selected];
  const capturedFromHand = [playedCard, selected];
  let pendingSteal = 0;
  pushCaptured(captured, playedCard);
  pushCaptured(captured, selected);
  if (!startedWithLastHand && presidentChainArmed) {
    events.shaking = (events.shaking || 0) + 1;
    log = log.concat(tx(state, "log.presidentChainShake", { player: player.label, month: playedCard.month }));
  }

  const flipResult = runFlipPhase({
    state,
    currentKey: playerKey,
    playedCard,
    isLastHandTurn: startedWithLastHand,
    board: boardAfter,
    deck: state.deck.slice(),
    hand,
    captured,
    events,
    log,
    newlyCaptured,
    pendingSteal,
    pendingBonusFlips: []
  });

  if (flipResult.pendingState) return flipResult.pendingState;

  return finalizeTurn({
    state,
    currentKey: playerKey,
    hand,
    captured,
    events,
    deck: flipResult.deck,
    board: flipResult.board,
    log: flipResult.log,
    newlyCaptured: flipResult.newlyCaptured,
    pendingSteal: flipResult.pendingSteal,
    heldBonusCards: flipResult.heldBonusOnPpuk || [],
    isLastHandTurn: startedWithLastHand,
    turnMeta: {
      type: "play",
      card: packCard(playedCard),
      selectedBoardCard: packCard(selected),
      flips: flipResult.flips || [],
      matchEvents: [{ source: "hand", eventTag: "NORMAL", type: "TWO" }].concat(
        flipResult.matchEvents || []
      ),
      captureBySource: {
        hand: capturedFromHand.map(packCard),
        flip: (flipResult.capturedFromFlip || []).map(packCard)
      }
    }
  });
}

function resolveFlipMatchChoice(state, boardCardId) {
  const { playerKey, context } = state.pendingMatch;
  if (!context) return state;

  const {
    hand,
    captured,
    events,
    deck,
    board,
    log,
    newlyCaptured,
    flipCard,
    pendingSteal,
    flips,
    pendingBonusFlips = [],
    matchEvents = [],
    capturedFromFlip = []
  } = context;

  const selected = board.find((c) => c.id === boardCardId);
  if (!selected) return state;

  const nextCaptured = {
    ...captured,
    kwang: captured.kwang.slice(),
    five: captured.five.slice(),
    ribbon: captured.ribbon.slice(),
    junk: captured.junk.slice()
  };
  const isLastHandTurn = hand.length === 0;
  const nextEvents = { ...events, ppuk: isLastHandTurn ? events.ppuk : events.ppuk + 1 };
  const nextNewlyCaptured = newlyCaptured.slice();
  let nextLog = log.slice();
  let bonusQueue = pendingBonusFlips.slice();
  if (!isLastHandTurn && bonusQueue.length > 0) {
    nextLog.push(tx(state, "log.ppukSelectHoldBonus", { count: bonusQueue.length }));
    bonusQueue = [];
  } else if (bonusQueue.length > 0) {
    bonusQueue.forEach((b) => {
      pushCaptured(nextCaptured, b);
      nextNewlyCaptured.push(b);
      capturedFromFlip.push(b);
    });
    nextLog.push(tx(state, "log.flipPendingBonusConfirmed", { count: bonusQueue.length }));
    bonusQueue = [];
  }

  pushCaptured(nextCaptured, flipCard);
  pushCaptured(nextCaptured, selected);
  nextNewlyCaptured.push(flipCard, selected);
  capturedFromFlip.push(flipCard, selected);
  const nextBoard = board.filter((c) => c.id !== selected.id);
  nextLog.push(tx(state, "log.flipSelectCapture", { month: flipCard.month, cardName: selected.name }));

  return finalizeTurn({
    state,
    currentKey: playerKey,
    hand,
    captured: nextCaptured,
    events: nextEvents,
    deck,
    board: nextBoard,
    log: nextLog,
    newlyCaptured: nextNewlyCaptured,
    pendingSteal,
    heldBonusCards: !isLastHandTurn ? pendingBonusFlips : [],
    isLastHandTurn,
    turnMeta: {
      type: "flip-select",
      card: packCard(flipCard),
      selectedBoardCard: packCard(selected),
      flips: flips || [packCard(flipCard)],
      matchEvents,
      captureBySource: {
        hand: [],
        flip: capturedFromFlip.map(packCard)
      }
    }
  });
}


