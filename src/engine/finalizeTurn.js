import { ruleSets } from "./rules.js";
import { calculateScore, calculateBaseScore, isGukjinCard } from "./scoring.js";
import { pointsToGold, stealGoldFromOpponent } from "./economy.js";
import { resolveRound } from "./resolution.js";
import { findPresidentMonth } from "./opening.js";
import {
  pushCaptured,
  stealPiFromOpponent
} from "./capturesEvents.js";
import { clearExpiredReveal, ensurePassCardFor } from "./turnFlow.js";

/* ============================================================================
 * Turn finalization pipeline
 * - normalize card zones
 * - resolve ppuk/steal/economy side effects
 * - transition to next phase (go-stop / resolution / next turn)
 * ========================================================================== */

/* 1) Card-zone normalization helpers */
export function normalizeUniqueCardZones(players, board, deck) {
  const seen = new Set();

  const dedupeReal = (cards = []) => {
    const next = [];
    cards.forEach((card) => {
      if (!card?.id || seen.has(card.id)) return;
      seen.add(card.id);
      next.push(card);
    });
    return next;
  };

  const dedupeHand = (cards = []) => {
    const passSeen = new Set();
    const next = [];
    cards.forEach((card) => {
      if (!card) return;
      if (card.passCard) {
        if (!card.id || passSeen.has(card.id)) return;
        passSeen.add(card.id);
        next.push(card);
        return;
      }
      if (!card.id || seen.has(card.id)) return;
      seen.add(card.id);
      next.push(card);
    });
    return next;
  };

  // Priority order: hand -> captured -> board -> deck
  const nextHumanHand = dedupeHand(players.human.hand || []);
  const nextAiHand = dedupeHand(players.ai.hand || []);

  const nextHumanCaptured = {
    ...players.human.captured,
    kwang: dedupeReal(players.human.captured?.kwang || []),
    five: dedupeReal(players.human.captured?.five || []),
    ribbon: dedupeReal(players.human.captured?.ribbon || []),
    junk: dedupeReal(players.human.captured?.junk || [])
  };

  const nextAiCaptured = {
    ...players.ai.captured,
    kwang: dedupeReal(players.ai.captured?.kwang || []),
    five: dedupeReal(players.ai.captured?.five || []),
    ribbon: dedupeReal(players.ai.captured?.ribbon || []),
    junk: dedupeReal(players.ai.captured?.junk || [])
  };

  const nextBoard = dedupeReal(board || []);
  const nextDeck = dedupeReal(deck || []);

  return {
    players: {
      ...players,
      human: {
        ...players.human,
        hand: nextHumanHand,
        captured: nextHumanCaptured
      },
      ai: {
        ...players.ai,
        hand: nextAiHand,
        captured: nextAiCaptured
      }
    },
    board: nextBoard,
    deck: nextDeck
  };
}

export function packCard(card) {
  return {
    id: card.id,
    month: card.month,
    category: card.category,
    name: card.name,
    asset: card.asset || null,
    passCard: !!card.passCard
  };
}

/* 2) Declarable action discovery */
export function getDeclarableShakingMonths(state, playerKey) {
  if (state.phase !== "playing" || state.currentTurn !== playerKey) return [];
  const player = state.players[playerKey];
  const declared = new Set(player.shakingDeclaredMonths || []);
  const counts = {};
  player.hand.forEach((c) => {
    if (!c || c.passCard) return;
    if (c.month < 1 || c.month > 12) return;
    counts[c.month] = (counts[c.month] || 0) + 1;
  });
  return Object.entries(counts)
    .map(([m, count]) => ({ month: Number(m), count }))
    .filter((x) => x.count >= 3 && !declared.has(x.month))
    .filter((x) => state.board.every((b) => b.month !== x.month))
    .map((x) => x.month);
}

export function getDeclarableBombMonths(state, playerKey) {
  if (state.phase !== "playing" || state.currentTurn !== playerKey) return [];
  const player = state.players[playerKey];
  const counts = {};
  player.hand.forEach((c) => {
    if (!c || c.passCard) return;
    if (c.month < 1 || c.month > 12) return;
    counts[c.month] = (counts[c.month] || 0) + 1;
  });
  return Object.entries(counts)
    .map(([m, count]) => ({ month: Number(m), count }))
    .filter((x) => x.count >= 3)
    .filter((x) => state.board.filter((b) => b.month === x.month).length === 1)
    .map((x) => x.month);
}

export function getShakingReveal(state, now) {
  if (state.actionReveal && state.actionReveal.expiresAt > now) return state.actionReveal;
  if (!state.shakingReveal) return null;
  if (state.shakingReveal.expiresAt <= now) return null;
  return state.shakingReveal;
}

/* 3) Gukjin scoring-time decision helpers */
function hasPendingGukjinChoice(player) {
  if (!player || player.gukjinLocked) return false;
  return (player.captured?.five || []).some((card) => isGukjinCard(card) && !card.gukjinTransformed);
}

function scoreWithGukjinMode(player, opponent, ruleKey, mode) {
  return calculateScore({ ...player, gukjinMode: mode }, opponent, ruleKey);
}

function baseWithGukjinMode(player, ruleKey, mode) {
  const rules = ruleSets[ruleKey];
  return calculateBaseScore({ ...player, gukjinMode: mode }, rules).base;
}

function shouldPromptGukjinChoiceAtScoring(state, playerKey) {
  const player = state.players?.[playerKey];
  if (!hasPendingGukjinChoice(player)) return false;

  const opponentKey = playerKey === "human" ? "ai" : "human";
  const opponent = state.players?.[opponentKey];
  if (!opponent) return false;

  const rules = ruleSets[state.ruleKey];
  if (!rules) return false;

  const scoreAsFive = scoreWithGukjinMode(player, opponent, state.ruleKey, "five");
  const scoreAsJunk = scoreWithGukjinMode(player, opponent, state.ruleKey, "junk");
  const bakAsFive = scoreAsFive.bak || {};
  const bakAsJunk = scoreAsJunk.bak || {};
  const hasScoringDifference =
    scoreAsFive.base !== scoreAsJunk.base ||
    scoreAsFive.total !== scoreAsJunk.total ||
    scoreAsFive.multiplier !== scoreAsJunk.multiplier ||
    scoreAsFive.payoutTotal !== scoreAsJunk.payoutTotal ||
    !!bakAsFive.gwang !== !!bakAsJunk.gwang ||
    !!bakAsFive.pi !== !!bakAsJunk.pi ||
    !!bakAsFive.mongBak !== !!bakAsJunk.mongBak;

  const handCount = (player.hand || []).length;
  const lastGoBase = player.lastGoBase || 0;
  const hasGo = (player.goCount || 0) > 0;
  const baseAsFive = baseWithGukjinMode(player, state.ruleKey, "five");
  const baseAsJunk = baseWithGukjinMode(player, state.ruleKey, "junk");
  const isRaisedFive = baseAsFive > lastGoBase;
  const isRaisedJunk = baseAsJunk > lastGoBase;

  const canGoStopAsFive =
    rules.useEarlyStop &&
    baseAsFive >= rules.goMinScore &&
    isRaisedFive &&
    handCount > 0;
  const canGoStopAsJunk =
    rules.useEarlyStop &&
    baseAsJunk >= rules.goMinScore &&
    isRaisedJunk &&
    handCount > 0;

  if (canGoStopAsFive !== canGoStopAsJunk) return true;
  if ((canGoStopAsFive || canGoStopAsJunk) && hasScoringDifference) return true;

  const canAutoResolveAfterGoAsFive = hasGo && handCount === 0 && isRaisedFive;
  const canAutoResolveAfterGoAsJunk = hasGo && handCount === 0 && isRaisedJunk;
  if (canAutoResolveAfterGoAsFive !== canAutoResolveAfterGoAsJunk) return true;
  if ((canAutoResolveAfterGoAsFive || canAutoResolveAfterGoAsJunk) && hasScoringDifference) return true;

  const bothHandsEmpty =
    (state.players?.human?.hand || []).length === 0 &&
    (state.players?.ai?.hand || []).length === 0;

  if (bothHandsEmpty) return hasScoringDifference;

  return false;
}

/* 4) Main finalize pipeline */
export function finalizeTurn({
  state,
  currentKey,
  hand,
  captured,
  events,
  deck,
  board,
  log,
  newlyCaptured,
  pendingSteal = 0,
  heldBonusCards = [],
  isLastHandTurn = false,
  turnMeta = null
}) {
  const nextPlayerKey = currentKey === "human" ? "ai" : "human";
  const prevPlayer = state.players[currentKey];
  const prevOpponent = state.players[nextPlayerKey];
  const ppukOccurred = (events.ppuk || 0) > (prevPlayer.events.ppuk || 0);
  const capturedAny = newlyCaptured.length > 0;

  const nextEvents = { ...events };
  const nextPlayerPatch = {};
  let goldSteal = 0;
  let extraSteal = pendingSteal;
  let nextLog = log.slice();
  const prevPpukState = prevPlayer.ppukState || {
    active: false,
    streak: 0,
    lastTurnNo: 0,
    lastSource: null,
    lastMonth: null
  };
  const prevOppPpukState = prevOpponent.ppukState || {
    active: false,
    streak: 0,
    lastTurnNo: 0,
    lastSource: null,
    lastMonth: null
  };
  const currentTurnNo = (state.turnSeq || 0) + 1;
  const prevHeldBonus = (prevPlayer.heldBonusCards || []).slice();
  let nextHeldBonus = prevHeldBonus.concat(heldBonusCards || []);
  let nextOpponentPatch = null;

  const clearPpukState = (ppukState) => ({
    active: false,
    streak: 0,
    lastTurnNo: currentTurnNo,
    lastSource: ppukState?.lastSource || null,
    lastMonth: ppukState?.lastMonth || null
  });

  const applyPpukEat = (ownerKey) => {
    const ownerIsSelf = ownerKey === currentKey;
    const ownerLabel = ownerIsSelf ? "self-ppuk" : "opponent-ppuk";
    const ownerHeldBonus = ownerIsSelf
      ? nextHeldBonus
      : (state.players[ownerKey]?.heldBonusCards || []).slice();
    const ownerPpukState = ownerIsSelf ? prevPpukState : prevOppPpukState;

    // Unified rule: self-ppuk eat and opponent-ppuk eat are treated identically.
    nextEvents.jabbeok = (nextEvents.jabbeok || 0) + 1;
    extraSteal += 1;
    nextLog.push(`${prevPlayer.label}: ${ownerLabel} conversion succeeded (reserve steal 1 pi)`);

    if (ownerHeldBonus.length > 0) {
      ownerHeldBonus.forEach((b) => pushCaptured(captured, b));
      const bonusSteal = ownerHeldBonus.reduce((sum, c) => sum + (c.bonus?.stealPi || 0), 0);
      extraSteal += bonusSteal;
      nextLog.push(
        `${prevPlayer.label}: recovered ${ownerHeldBonus.length} held ppuk bonus cards (extra steal ${bonusSteal})`
      );
    }

    if (ownerIsSelf) {
      nextHeldBonus = [];
      nextPlayerPatch.ppukState = clearPpukState(ownerPpukState);
    } else {
      nextOpponentPatch = {
        ...(nextOpponentPatch || {}),
        ppukState: clearPpukState(ownerPpukState),
        heldBonusCards: []
      };
    }
  };

  if (ppukOccurred) {
    const nextStreak = (prevPpukState.streak || 0) + 1;
    nextPlayerPatch.ppukState = {
      active: true,
      streak: nextStreak,
      lastTurnNo: currentTurnNo,
      lastSource: turnMeta?.matchEvents?.some((e) => e.source === "flip" && e.eventTag === "PPUK")
        ? "FLIP"
        : "HAND",
      lastMonth: turnMeta?.card?.month ?? null
    };

    const turnCount = prevPlayer.turnCount || 0;
    if (turnCount === 0) {
      goldSteal += pointsToGold(7);
      nextLog.push(`${prevPlayer.label}: first ppuk reward (7 points, ${pointsToGold(7)} gold)`);
    } else if (turnCount === 1 && nextStreak >= 2) {
      goldSteal += pointsToGold(14);
      nextLog.push(`${prevPlayer.label}: two-ppuk streak reward (14 points, ${pointsToGold(14)} gold)`);
      nextEvents.yeonPpuk = (nextEvents.yeonPpuk || 0) + 1;
    } else if (turnCount === 2 && nextStreak >= 3) {
      goldSteal += pointsToGold(21);
      nextLog.push(`${prevPlayer.label}: three-ppuk streak reward (21 points, ${pointsToGold(21)} gold)`);
      nextEvents.yeonPpuk = (nextEvents.yeonPpuk || 0) + 1;
    }
  } else {
    nextPlayerPatch.ppukState = { ...prevPpukState };
  }

  if (capturedAny && !ppukOccurred && prevPpukState.active) {
    applyPpukEat(currentKey);
  }
  if (capturedAny && prevOppPpukState.active) {
    applyPpukEat(nextPlayerKey);
  }

  if (!isLastHandTurn && board.length === 0 && capturedAny) {
    nextEvents.ssul = (nextEvents.ssul || 0) + 1;
    extraSteal += 1;
    nextLog.push(`${prevPlayer.label}: board sweep triggered (reserve steal 1 pi)`);
  }

  let nextPlayers = {
    ...state.players,
    [currentKey]: {
      ...state.players[currentKey],
      ...nextPlayerPatch,
      turnCount: (prevPlayer.turnCount || 0) + 1,
      heldBonusCards: nextHeldBonus,
      hand,
      captured,
      events: nextEvents
    }
  };
  if (nextOpponentPatch) {
    nextPlayers = {
      ...nextPlayers,
      [nextPlayerKey]: {
        ...nextPlayers[nextPlayerKey],
        ...nextOpponentPatch
      }
    };
  }

  let nextState = {
    ...state,
    phase: "playing",
    pendingMatch: null,
    pendingGoStop: null,
    pendingShakingConfirm: null,
    pendingGukjinChoice: null,
    result: null,
    deck,
    board,
    players: nextPlayers,
    currentTurn: nextPlayerKey,
    log: nextLog
  };

  if (isLastHandTurn) {
    extraSteal = 0;
  }

  if (extraSteal > 0) {
    const { updatedPlayers, stealLog } = stealPiFromOpponent(nextState.players, currentKey, extraSteal);
    nextState = {
      ...nextState,
      players: updatedPlayers,
      log: nextState.log.concat(stealLog)
    };
  }

  if (goldSteal > 0) {
    const goldResult = stealGoldFromOpponent(nextState.players, currentKey, goldSteal);
    nextState = {
      ...nextState,
      players: goldResult.updatedPlayers,
      log: nextState.log.concat(goldResult.log)
    };
  }

  // Always normalize after steal/economy side-effects before returning.
  const uniq = normalizeUniqueCardZones(nextState.players, nextState.board, nextState.deck);
  nextState = {
    ...nextState,
    players: uniq.players,
    board: uniq.board,
    deck: uniq.deck
  };

  const turnNo = (nextState.turnSeq || 0) + 1;
  const kiboNo = (nextState.kiboSeq || 0) + 1;
  const isLeanKibo = nextState.kiboDetail === "lean";
  nextState = {
    ...nextState,
    turnSeq: turnNo,
    kiboSeq: kiboNo,
    kibo: (nextState.kibo || []).concat({
      no: kiboNo,
      type: "turn_end",
      turnNo,
      actor: currentKey,
      action: turnMeta,
      deckCount: nextState.deck.length,
      ...(isLeanKibo
        ? {
            boardCount: nextState.board.length,
            handsCount: {
              human: nextState.players.human.hand.length,
              ai: nextState.players.ai.hand.length
            }
          }
        : {
            board: nextState.board.map(packCard),
            hands: {
              human: nextState.players.human.hand.map(packCard),
              ai: nextState.players.ai.hand.map(packCard)
            }
          }),
      steals: { pi: extraSteal, gold: goldSteal },
      heldBonus: nextState.players[currentKey].heldBonusCards?.map(packCard) || [],
      events: { ...nextEvents },
      ppukState: { ...(nextState.players[currentKey].ppukState || {}) }
    })
  };

  if ((nextState.players[currentKey].events.ppuk || 0) >= 3) {
    nextState = resolveRound(nextState, currentKey);
    return nextState;
  }

  if (shouldPromptGukjinChoiceAtScoring(nextState, currentKey)) {
    return {
      ...nextState,
      phase: "gukjin-choice",
      pendingGukjinChoice: { playerKey: currentKey },
      log: nextState.log.concat(
        `${nextState.players[currentKey].label}: choose gukjin (September five) scoring mode`
      )
    };
  }

  return continueAfterTurnIfNeeded(nextState, currentKey);
}

/* 5) Post-turn flow gate */
export function continueAfterTurnIfNeeded(state, justPlayedKey) {
  state = clearExpiredReveal(state);
  const opponentKey = justPlayedKey === "human" ? "ai" : "human";
  const rules = ruleSets[state.ruleKey];
  const baseInfo = calculateBaseScore(state.players[justPlayedKey], rules);
  const currentBase = baseInfo.base;
  const playerAfterTurn = state.players[justPlayedKey];
  const isRaisedSinceLastGo = currentBase > playerAfterTurn.lastGoBase;
  const handCountAfterTurn = state.players[justPlayedKey].hand.length;

  // Last-hand rule:
  // If the acting player has no cards left and satisfies Go/Stop scoring condition,
  // skip prompt and auto-stop the round immediately.
  if (
    handCountAfterTurn === 0 &&
    rules.useEarlyStop &&
    currentBase >= rules.goMinScore &&
    isRaisedSinceLastGo
  ) {
    return resolveRound(
      {
        ...state,
        players: {
          ...state.players,
          [justPlayedKey]: {
            ...state.players[justPlayedKey],
            declaredStop: true
          }
        }
      },
      justPlayedKey
    );
  }

  if (
    rules.useEarlyStop &&
    currentBase >= rules.goMinScore &&
    isRaisedSinceLastGo &&
    handCountAfterTurn > 0
  ) {
    return {
      ...state,
      currentTurn: justPlayedKey,
      phase: "go-stop",
      pendingGoStop: justPlayedKey
    };
  }

  const bothHandsEmpty =
    state.players.human.hand.length === 0 && state.players.ai.hand.length === 0;

  if (bothHandsEmpty) {
    return resolveRound(state, justPlayedKey);
  }

  // On the actor's first turn, if hand president condition (4 cards same month) is met,
  // enter the president-choice phase immediately.
  const actorKey = state.currentTurn;
  const actor = state.players[actorKey];
  if (
    state.phase === "playing" &&
    !state.pendingPresident &&
    actor &&
    (actor.turnCount || 0) === 0 &&
    !actor.presidentHold
  ) {
    const month = findPresidentMonth(actor.hand || []);
    if (month !== null) {
      return {
        ...state,
        phase: "president-choice",
        pendingPresident: { playerKey: actorKey, month },
        log: state.log.concat(
          `${actor.label}: first-turn hand president (${month} month x4) - choose 10-point stop or hold`
        )
      };
    }
  }

  return ensurePassCardFor(state, state.currentTurn);
}

