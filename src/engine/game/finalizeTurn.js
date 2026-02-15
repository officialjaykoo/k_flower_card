import { ruleSets } from "../rules.js";
import { calculateScore, isGukjinCard } from "../scoring.js";
import { pointsToGold, stealGoldFromOpponent } from "../economy.js";
import { resolveRound } from "../resolution.js";
import { findPresidentMonth } from "../opening.js";
import {
  pushCaptured,
  stealPiFromOpponent
} from "../capturesEvents.js";
import { clearExpiredReveal, ensurePassCardFor } from "../turnFlow.js";
import {
  normalizeUniqueCardZones,
  packCard
} from "./normalize.js";

function hasPendingGukjinChoice(player) {
  if (!player || player.gukjinLocked) return false;
  return (player.captured?.five || []).some((card) => isGukjinCard(card) && !card.gukjinTransformed);
}

function scoreWithGukjinMode(player, opponent, ruleKey, mode) {
  return calculateScore({ ...player, gukjinMode: mode }, opponent, ruleKey);
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
  const isRaisedFive = scoreAsFive.base > lastGoBase;
  const isRaisedJunk = scoreAsJunk.base > lastGoBase;

  const canGoStopAsFive =
    rules.useEarlyStop &&
    scoreAsFive.base >= rules.goMinScore &&
    isRaisedFive &&
    handCount > 0;
  const canGoStopAsJunk =
    rules.useEarlyStop &&
    scoreAsJunk.base >= rules.goMinScore &&
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
    const ownerLabel = ownerIsSelf ? "자뻑" : "상대뻑";
    const ownerHeldBonus = ownerIsSelf
      ? nextHeldBonus
      : (state.players[ownerKey]?.heldBonusCards || []).slice();
    const ownerPpukState = ownerIsSelf ? prevPpukState : prevOppPpukState;

    // Unified rule: self-ppuk eat and opponent-ppuk eat are treated identically.
    nextEvents.jabbeok = (nextEvents.jabbeok || 0) + 1;
    extraSteal += 1;
    nextLog.push(`${prevPlayer.label}: ${ownerLabel} 먹기 성공 (상대 피 1장 강탈 예약)`);

    if (ownerHeldBonus.length > 0) {
      ownerHeldBonus.forEach((b) => pushCaptured(captured, b));
      const bonusSteal = ownerHeldBonus.reduce((sum, c) => sum + (c.bonus?.stealPi || 0), 0);
      extraSteal += bonusSteal;
      nextLog.push(
        `${prevPlayer.label}: 뻑 보류 보너스 ${ownerHeldBonus.length}장 회수 (추가 강탈 ${bonusSteal})`
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
      nextLog.push(`${prevPlayer.label}: 첫뻑 보상(7점, ${pointsToGold(7)}골드)`);
    } else if (turnCount === 1 && nextStreak >= 2) {
      goldSteal += pointsToGold(14);
      nextLog.push(`${prevPlayer.label}: 2연뻑 보상(14점, ${pointsToGold(14)}골드)`);
      nextEvents.yeonPpuk = (nextEvents.yeonPpuk || 0) + 1;
    } else if (turnCount === 2 && nextStreak >= 3) {
      goldSteal += pointsToGold(21);
      nextLog.push(`${prevPlayer.label}: 3연뻑 보상(21점, ${pointsToGold(21)}골드)`);
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
    nextLog.push(`${prevPlayer.label}: 판쓸 발생 (상대 피 1장 강탈 예약)`);
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
        `${nextState.players[currentKey].label}: 국진(9월 열) 점수 판정 시점 선택`
      )
    };
  }

  return continueAfterTurnIfNeeded(nextState, currentKey);
}

export function continueAfterTurnIfNeeded(state, justPlayedKey) {
  state = clearExpiredReveal(state);
  const opponentKey = justPlayedKey === "human" ? "ai" : "human";
  const rules = ruleSets[state.ruleKey];
  const scoreInfo = calculateScore(
    state.players[justPlayedKey],
    state.players[opponentKey],
    state.ruleKey
  );
  const playerAfterTurn = state.players[justPlayedKey];
  const isRaisedSinceLastGo = scoreInfo.base > playerAfterTurn.lastGoBase;
  if (state.players[justPlayedKey].goCount > 0 && state.players[justPlayedKey].hand.length === 0) {
    if (isRaisedSinceLastGo) {
      // GO 이후 손패 소진 시 마지막 GO 기준점보다 점수가 오르면 자동 종료.
      return resolveRound(state, justPlayedKey);
    }
  }

  if (
    rules.useEarlyStop &&
    scoreInfo.base >= rules.goMinScore &&
    isRaisedSinceLastGo &&
    state.players[justPlayedKey].hand.length > 0
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

  // 자기 첫 턴 시작 시 손패 대통령(동월 4장)이면 즉시 선언/선택 단계로 진입.
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
          `${actor.label}: 첫 턴 손패 대통령(${month}월 4장) - 10점 종료/들고치기 선택`
        )
      };
    }
  }

  return ensurePassCardFor(state, state.currentTurn);
}

