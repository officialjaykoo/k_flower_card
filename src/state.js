import { buildDeck, shuffle } from "./cards.js";
import { ruleSets } from "./engine/rules.js";
import { calculateScore, isGukjinCard } from "./engine/scoring.js";
import { POINT_GOLD_UNIT, STARTING_GOLD, settleRoundGold } from "./engine/economy.js";
import { resolveMatch } from "./engine/matching.js";
import { resolveRound } from "./engine/resolution.js";
import {
  decideFirstTurn,
  emptyPlayer,
  normalizeOpeningHands,
  normalizeOpeningBoard,
  findPresidentMonth
} from "./engine/opening.js";
import {
  pushCaptured,
  needsChoice,
  bestMatchCard,
  stealPiFromOpponent
} from "./engine/capturesEvents.js";
import { clearExpiredReveal, ensurePassCardFor } from "./engine/turnFlow.js";
import { packCard } from "./engine/game/normalize.js";
import {
  finalizeTurn,
  continueAfterTurnIfNeeded
} from "./engine/game/finalizeTurn.js";
import {
  getDeclarableShakingMonths,
  getDeclarableBombMonths,
  getShakingReveal as selectShakingReveal
} from "./engine/game/selectors.js";
export { getDeclarableShakingMonths, getDeclarableBombMonths };
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
  const kiboDetail = options.kiboDetail === "lean" ? "lean" : "full";
  const fixedFirstTurnKey =
    options.firstTurnKey === "human" || options.firstTurnKey === "ai"
      ? options.firstTurnKey
      : null;
  const carryLogs = [];

  while (true) {
    const deck = shuffle(buildDeck(), seedRng);
    const players = {
      human: emptyPlayer("플레이어"),
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
          log: `선턴 유지: ${fixedFirstTurnKey === "human" ? "플레이어" : "AI"} 선`
        }
      : decideFirstTurn(players.human.hand[0], players.ai.hand[0], seedRng);
    const initLog = [];
    initLog.push(firstTurnInfo.log);

    // 보너스 카드 시작 보정(손패/바닥패)
    remain = normalizeOpeningHands(players, remain, initLog);
    ({ board, remain } = normalizeOpeningBoard(
      board,
      remain,
      players[firstTurnInfo.winnerKey],
      initLog
    ));

    // 바닥 대통령(같은 월 4장): 선공 즉시 승리(10점 종료).
    const boardPresident = findPresidentMonth(board);
    if (boardPresident !== null) {
      const winnerKey = firstTurnInfo.winnerKey;
      const winnerLabel = players[winnerKey].label;
      const baseScore = 10;
      const settled = settleRoundGold(players, winnerKey, baseScore);
      const openLog = [`게임 시작 - 룰: ${ruleSets[ruleKey].name}`];
      const log = [...openLog, ...carryLogs, ...initLog];
      log.push(`바닥 대통령(${boardPresident}월 4장): ${winnerLabel} 선공 즉시 승리(10점 종료)`);
      log.push(
        `라운드 정산(골드): ${winnerLabel} 요구 ${settled.requested}골드 / 수령 ${settled.paid}골드`
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
        `${players[firstTurnInfo.winnerKey].label} 손패 대통령(${handPresident}월 4장): 선턴은 10점 종료 또는 들고치기 선택 가능`
      );
    }

    const openLog = [`게임 시작 - 룰: ${ruleSets[ruleKey].name}`];
    if (carryOverMultiplier > 1) {
      openLog.push(`적용 배수: x${carryOverMultiplier}`);
    }

    return {
      deck: remain,
      board,
      players,
      currentTurn: firstTurnInfo.winnerKey,
      startingTurnKey: firstTurnInfo.winnerKey,
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
      title: "흔들기",
      message: `${player.label}님 흔들기 했습니다.`,
      cards: revealCards.map(packCard),
      expiresAt: Date.now() + 2000
    },
    log: state.log.concat(`${player.label}: 흔들기 선언 (${monthNum}월 공개 2초, 카드 보류 가능)`),
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
    `${player.label}: 폭탄 선언 (${monthNum}월 ${monthCards.length}장 + 바닥 1장 획득)`
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
      title: "폭탄",
      message: `${player.label}님 폭탄 선언했습니다.`,
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
      .concat(`${player.label}: 보너스 카드 사용 (${playedCard.name}, 피 ${playedCard.bonus.stealPi}장 강탈)`)
      .concat(
        drawnToHand
          ? `${player.label}: 보너스 보상으로 손패 1장 보충 (${drawnToHand.month}월 ${drawnToHand.name})`
          : `${player.label}: 보너스 보상 보충 실패 (덱 부족)`
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

    // 마지막 손패 턴이 아니면 보너스 강탈 1장을 처리한다.
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
          title: "보너스 패 사용",
          message: `${player.label}님 보너스 패를 사용했습니다. 대통령 선택으로 진행합니다.`,
          cards: [packCard(playedCard)],
          expiresAt: Date.now() + 2000
        },
        log: nextLog.concat(
          `${player.label}: 보너스 보충으로 손패 대통령(${bonusPresidentMonth}월 4장) 성립 - 10점 종료/들고치기 선택`
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
        title: "보너스 패 사용",
        message: `${player.label}님 보너스 패를 사용했습니다. 같은 턴에서 손패 1장을 더 선택하세요.`,
        cards: [packCard(playedCard)],
        expiresAt: Date.now() + 2000
      },
      log: nextLog.concat(`${player.label}: 보너스 사용 후 같은 턴 계속 진행 (손패 1장 추가 선택)`),
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
    log.push(`${player.label}: ${playedCard.month}월 카드 투입 (매치 없음)`);
  } else if (handMatch.type === "ONE") {
    const matched = handMatch.matches[0];
    board = board.filter((c) => c.id !== matched.id);
    pushCaptured(captured, matched);
    pushCaptured(captured, playedCard);
    newlyCaptured.push(matched, playedCard);
    capturedFromHand.push(matched, playedCard);
    log.push(`${player.label}: ${playedCard.month}월 매치 캡처`);
    if (!isLastHandTurn && presidentChainArmed) {
      events.shaking += 1;
      log.push(`${player.label}: 대통령 들고치기 연계 성공 (${playedCard.month}월) - 흔들기 처리`);
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
          message: `${player.label}: 낸 카드와 매치할 보드 카드 1장을 선택하세요`
        },
        log: state.log.concat(`${player.label}: ${playedCard.month}월 2장 매치 - 카드 선택 대기`)
      };
    }
    const matched = bestMatchCard(handMatch.matches);
    board = board.filter((c) => c.id !== matched.id);
    pushCaptured(captured, matched);
    pushCaptured(captured, playedCard);
    newlyCaptured.push(matched, playedCard);
    capturedFromHand.push(matched, playedCard);
    log.push(`${player.label}: ${playedCard.month}월 자동 매치 캡처`);
    if (!isLastHandTurn && presidentChainArmed) {
      events.shaking += 1;
      log.push(`${player.label}: 대통령 들고치기 연계 성공 (${playedCard.month}월) - 흔들기 처리`);
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
    if (handMatch.eventTag === "TTAK") events.ttak += 1;
    log.push(`${player.label}: ${playedCard.month}월 싹쓸이 캡처`);
    if (!isLastHandTurn) {
      pendingSteal += 1;
      log.push(`${player.label}: 판쓸 발생 (상대 피 1장 강탈 예약)`);
    }
    if (!isLastHandTurn && presidentChainArmed) {
      events.shaking += 1;
      log.push(`${player.label}: 대통령 들고치기 연계 성공 (${playedCard.month}월) - 흔들기 처리`);
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
      log.push(`뒤집기: ${flip.name} 보류(재뒤집기 후 확정)`);
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
        log.push(`보류 보너스 ${pendingBonusFlips.length}장 획득 확정`);
        pendingBonusFlips = [];
      }
      board.push(flip);
      log.push(`뒤집기: ${flip.month}월 보드 배치`);
      if (flipMatch.eventTag === "JJOB") {
        events.jjob += 1;
        pendingSteal += 1;
        log.push("쪽 발생 (+1, 상대 피 1장 강탈 예약)");
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
        log.push(`보류 보너스 ${pendingBonusFlips.length}장 획득 확정`);
        pendingBonusFlips = [];
      }
      const matched = flipMatch.matches[0];
      board = board.filter((c) => c.id !== matched.id);
      pushCaptured(captured, flip);
      pushCaptured(captured, matched);
      newlyCaptured.push(flip, matched);
      capturedFromFlip.push(flip, matched);
      log.push(`뒤집기: ${flip.month}월 매치 캡처`);
      if (flipMatch.eventTag === "DDADAK") {
        events.ddadak = (events.ddadak || 0) + 1;
        pendingSteal += 1;
        log.push(`따닥 발생 (상대 피 1장 강탈 예약)`);
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
            message: `${state.players[currentKey].label}: 뒤집은 카드와 매치할 보드 카드 1장을 선택하세요`,
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
          log: log.concat(`뒤집기: ${flip.month}월 2장 매치 - 카드 선택 대기`)
        }
      };
    }

    if (flipMatch.type === "TWO") {
      if (!isLastHandTurn && pendingBonusFlips.length > 0) {
        log.push(`뻑 발생: 보류 보너스 ${pendingBonusFlips.length}장 홀딩`);
        heldBonusOnPpuk = heldBonusOnPpuk.concat(pendingBonusFlips);
        pendingBonusFlips = [];
      } else if (pendingBonusFlips.length > 0) {
        pendingBonusFlips.forEach((b) => {
          pushCaptured(captured, b);
          newlyCaptured.push(b);
          capturedFromFlip.push(b);
        });
        log.push(`보류 보너스 ${pendingBonusFlips.length}장 획득 확정`);
        pendingBonusFlips = [];
      }
      if (flipMatch.eventTag === "PPUK") events.ppuk += 1;
      const matched = bestMatchCard(flipMatch.matches);
      board = board.filter((c) => c.id !== matched.id);
      pushCaptured(captured, flip);
      pushCaptured(captured, matched);
      newlyCaptured.push(flip, matched);
      capturedFromFlip.push(flip, matched);
      log.push(`뒤집기: ${flip.month}월 다중 매치 선택 캡처`);
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
      log.push(`뻑 발생: 보류 보너스 ${pendingBonusFlips.length}장 홀딩`);
      heldBonusOnPpuk = heldBonusOnPpuk.concat(pendingBonusFlips);
      pendingBonusFlips = [];
    } else if (pendingBonusFlips.length > 0) {
      pendingBonusFlips.forEach((b) => {
        pushCaptured(captured, b);
        newlyCaptured.push(b);
        capturedFromFlip.push(b);
      });
      log.push(`보류 보너스 ${pendingBonusFlips.length}장 획득 확정`);
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
    log.push(`뒤집기: ${flip.month}월 4장 캡처`);
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
    log.push(`보류 보너스 ${pendingBonusFlips.length}장 획득 확정`);
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
    `${player.label}: ${playedCard.month}월 선택 캡처 (${selected.name})`
  );
  const newlyCaptured = [playedCard, selected];
  const capturedFromHand = [playedCard, selected];
  let pendingSteal = 0;
  pushCaptured(captured, playedCard);
  pushCaptured(captured, selected);
  if (!startedWithLastHand && presidentChainArmed) {
    events.shaking = (events.shaking || 0) + 1;
    log = log.concat(`${player.label}: 대통령 들고치기 연계 성공 (${playedCard.month}월) - 흔들기 처리`);
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
    nextLog.push(`뻑 선택 발생: 보류 보너스 ${bonusQueue.length}장 홀딩`);
    bonusQueue = [];
  } else if (bonusQueue.length > 0) {
    bonusQueue.forEach((b) => {
      pushCaptured(nextCaptured, b);
      nextNewlyCaptured.push(b);
      capturedFromFlip.push(b);
    });
    nextLog.push(`보류 보너스 ${bonusQueue.length}장 획득 확정`);
    bonusQueue = [];
  }

  pushCaptured(nextCaptured, flipCard);
  pushCaptured(nextCaptured, selected);
  nextNewlyCaptured.push(flipCard, selected);
  capturedFromFlip.push(flipCard, selected);
  const nextBoard = board.filter((c) => c.id !== selected.id);
  nextLog.push(`뒤집기: ${flipCard.month}월 선택 캡처 (${selected.name})`);

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


export function chooseGo(state, playerKey) {
  if (state.phase !== "go-stop" || state.pendingGoStop !== playerKey) return state;

  const player = state.players[playerKey];
  const opponentKey = playerKey === "human" ? "ai" : "human";
  const scoreInfo = calculateScore(state.players[playerKey], state.players[opponentKey], state.ruleKey);
  const nextPlayers = {
    ...state.players,
    [playerKey]: {
      ...player,
      goCount: player.goCount + 1,
      lastGoBase: scoreInfo.base
    }
  };

  const log = state.log.concat(`${player.label}: GO 선언 (${player.goCount + 1}고)`);

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

  const log = state.log.concat(`${player.label}: STOP 선언`);
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
    `${state.players[playerKey].label}: 손패 대통령 즉시 종료 선택 (${payout}점)`
  );
  const settled = settleRoundGold(state.players, playerKey, payout);
  log = log.concat(
    `라운드 정산(골드): ${state.players[playerKey].label} 요구 ${settled.requested}골드 / 수령 ${settled.paid}골드`
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
    `${state.players[playerKey].label}: 손패 대통령(${month}월) 들고치기 선택`
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
        name: `${gukjinCard.name} (국진피)`
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
        name: String(gukjinCard.name || "").replace(" (국진피)", "")
      });
    }
  }
  const nextPlayers = {
    ...state.players,
    [playerKey]: { ...player, captured, gukjinMode: mode, gukjinLocked: true }
  };
  const label = mode === "junk" ? "쌍피" : "열";
  const log = state.log.concat(`${player.label}: 국진(9월 열) ${label} 처리 선택(낙장불입)`);

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









