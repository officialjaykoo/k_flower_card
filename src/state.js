import { buildDeck, shuffle, groupByMonth } from "./cards.js";
import { ruleSets } from "./engine/rules.js";
import { calculateScore } from "./engine/scoring.js";
import { POINT_GOLD_UNIT, stealGoldFromOpponent, settleRoundGold } from "./engine/economy.js";
import { resolveMatch } from "./engine/matching.js";
import { resolveRound } from "./engine/resolution.js";
import {
  decideFirstTurn,
  findKungMonth,
  emptyPlayer,
  normalizeOpeningHands,
  normalizeOpeningBoard,
  findPresidentMonth
} from "./engine/opening.js";
import {
  pushCaptured,
  needsChoice,
  bestMatchCard,
  shouldPromptGukjinChoice,
  stealPiFromOpponent
} from "./engine/capturesEvents.js";
import { clearExpiredReveal, ensurePassCardFor } from "./engine/turnFlow.js";
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
    if (options.initialGold?.human != null) players.human.gold = options.initialGold.human;
    if (options.initialGold?.ai != null) players.ai.gold = options.initialGold.ai;
    if (options.initialMoney?.human != null) players.human.gold = options.initialMoney.human;
    if (options.initialMoney?.ai != null) players.ai.gold = options.initialMoney.ai;

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

    // 바닥 대통령(같은 월 4장): 해당 판 무효, 다음 판 배수 x2.
    const boardPresident = findPresidentMonth(board);
    if (boardPresident !== null) {
      carryOverMultiplier *= 2;
      carryLogs.push(
        `바닥 대통령(${boardPresident}월 4장): 판 무효, 다음 판 배수 x${carryOverMultiplier}`
      );
      continue;
    }

    let phase = "playing";
    let pendingPresident = null;
    let pendingKung = null;
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
    } else {
      const kungMonth = findKungMonth(players[firstTurnInfo.winnerKey].hand, board);
      if (kungMonth !== null) {
        phase = "kung-choice";
        pendingKung = {
          playerKey: firstTurnInfo.winnerKey,
          month: kungMonth
        };
        initLog.push(
          `${players[firstTurnInfo.winnerKey].label}: 쿵 가능(${kungMonth}월 3장+바닥 1장) - 쿵/패스 선택`
        );
      }
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
      pendingGukjinChoice: null,
      pendingPresident,
      pendingKung,
      shakingReveal: null,
      carryOverMultiplier,
      nextCarryOverMultiplier: 1,
      log: [...openLog, ...carryLogs, ...initLog],
      turnSeq: 0,
      kiboSeq: 1,
      kibo: [
        {
          no: 1,
          type: "initial_deal",
          firstTurn: firstTurnInfo.winnerKey,
          hands: {
            human: players.human.hand.map(packCard),
            ai: players.ai.hand.map(packCard)
          },
          board: board.map(packCard),
          deck: remain.map(packCard)
        }
      ],
      ruleKey,
      result: null
    };
  }
}

function packCard(card) {
  return {
    id: card.id,
    month: card.month,
    category: card.category,
    name: card.name,
    passCard: !!card.passCard
  };
}

export function getDeclarableShakingMonths(state, playerKey) {
  if (state.phase !== "playing" || state.currentTurn !== playerKey) return [];
  const player = state.players[playerKey];
  const declared = new Set(player.shakingDeclaredMonths || []);
  const counts = {};
  player.hand.forEach((c) => {
    if (c.month <= 12) counts[c.month] = (counts[c.month] || 0) + 1;
  });
  return Object.entries(counts)
    .map(([m, count]) => ({ month: Number(m), count }))
    .filter((x) => x.count >= 3 && !declared.has(x.month))
    .filter((x) => state.board.every((b) => b.month !== x.month))
    .map((x) => x.month);
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

export function getDeclarableBombMonths(state, playerKey) {
  if (state.phase !== "playing" || state.currentTurn !== playerKey) return [];
  const player = state.players[playerKey];
  const counts = {};
  player.hand.forEach((c) => {
    if (c.month <= 12) counts[c.month] = (counts[c.month] || 0) + 1;
  });
  return Object.entries(counts)
    .map(([m, count]) => ({ month: Number(m), count }))
    .filter((x) => x.count >= 3)
    .filter((x) => state.board.filter((b) => b.month === x.month).length === 1)
    .map((x) => x.month);
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

export function getShakingReveal(state, now = Date.now()) {
  if (!state.shakingReveal) return null;
  if (state.shakingReveal.expiresAt <= now) return null;
  return state.shakingReveal;
}

export function playTurn(state, cardId) {
  if (state.phase !== "playing") return state;
  state = clearExpiredReveal(state);

  const currentKey = state.currentTurn;
  const player = state.players[currentKey];
  const idx = player.hand.findIndex((c) => c.id === cardId);
  if (idx < 0) return state;

  const playedCard = player.hand[idx];
  const isLastHandTurn = player.hand.length === 1;
  if (playedCard.passCard) {
    return finalizeTurn({
      state,
      currentKey,
      hand: player.hand.filter((c) => c.id !== playedCard.id),
      captured: { ...player.captured },
      events: { ...player.events },
      deck: state.deck,
      board: state.board.slice(),
      log: state.log.concat(`${player.label}: 패스 카드 사용 (턴 넘김)`),
      newlyCaptured: [],
      pendingSteal: 0,
      isLastHandTurn,
      turnMeta: { type: "pass", card: packCard(playedCard) }
    });
  }
  const hand = player.hand.slice();
  hand.splice(idx, 1);
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
    if (presidentChainArmed) {
      events.shaking += 1;
      events.bomb += 1;
      pendingSteal += 1;
      log.push(`${player.label}: 대통령 들고치기 연계 성공 (${playedCard.month}월) - 흔들기+폭탄 처리`);
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
    if (presidentChainArmed) {
      events.shaking += 1;
      events.bomb += 1;
      pendingSteal += 1;
      log.push(`${player.label}: 대통령 들고치기 연계 성공 (${playedCard.month}월) - 흔들기+폭탄 처리`);
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
      log.push(`${player.label}: 쓸 발생 (상대 피 1장 강탈 예약)`);
    }
    if (presidentChainArmed) {
      events.shaking += 1;
      events.bomb += 1;
      pendingSteal += 1;
      log.push(`${player.label}: 대통령 들고치기 연계 성공 (${playedCard.month}월) - 흔들기+폭탄 처리`);
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
        log.push(`쪽 발생 (+${ruleSets[state.ruleKey].jjobBonus}, 상대 피 1장 강탈 예약)`);
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
  if (presidentChainArmed) {
    events.shaking = (events.shaking || 0) + 1;
    events.bomb = (events.bomb || 0) + 1;
    pendingSteal += 1;
    log = log.concat(`${player.label}: 대통령 들고치기 연계 성공 (${playedCard.month}월) - 흔들기+폭탄 처리`);
  }

  const flipResult = runFlipPhase({
    state,
    currentKey: playerKey,
    playedCard,
    isLastHandTurn: hand.length === 0,
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
    isLastHandTurn: hand.length === 0,
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

function finalizeTurn({
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
  const currentTurnNo = (state.turnSeq || 0) + 1;
  const prevHeldBonus = (prevPlayer.heldBonusCards || []).slice();
  let nextHeldBonus = prevHeldBonus.concat(heldBonusCards || []);

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
    if ((prevPlayer.turnCount || 0) === 0) {
      goldSteal += 3;
      nextLog.push(`${prevPlayer.label}: 첫뻑 보상(골드 3)`);
    }
    if (nextStreak >= 2) {
      nextEvents.yeonPpuk = (nextEvents.yeonPpuk || 0) + 1;
      goldSteal += 3;
      nextLog.push(`${prevPlayer.label}: 연뻑 보상(골드 3)`);
    }
  } else if (prevPpukState.active && capturedAny) {
    nextEvents.jabbeok = (nextEvents.jabbeok || 0) + 1;
    nextPlayerPatch.ppukState = {
      active: false,
      streak: 0,
      lastTurnNo: currentTurnNo,
      lastSource: prevPpukState.lastSource,
      lastMonth: prevPpukState.lastMonth
    };
    extraSteal += 1;
    nextLog.push(`${prevPlayer.label}: 자뻑 먹기 성공 (상대 피 1장 강탈 예약)`);
    if (nextHeldBonus.length > 0) {
      nextHeldBonus.forEach((b) => {
        pushCaptured(captured, b);
      });
      const bonusSteal = nextHeldBonus.reduce((sum, c) => sum + (c.bonus?.stealPi || 0), 0);
      extraSteal += bonusSteal;
      nextLog.push(
        `${prevPlayer.label}: 홀딩 보너스 ${nextHeldBonus.length}장 회수 (추가 강탈 ${bonusSteal})`
      );
      nextHeldBonus = [];
    }
  } else {
    nextPlayerPatch.ppukState = { ...prevPpukState };
  }

  if (!isLastHandTurn && board.length === 0 && capturedAny) {
    nextEvents.ssul = (nextEvents.ssul || 0) + 1;
    extraSteal += 1;
    nextLog.push(`${prevPlayer.label}: 쓸 발생 (상대 피 1장 강탈 예약)`);
  }

  const nextPlayers = {
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

  let nextState = {
    ...state,
    phase: "playing",
    pendingMatch: null,
    pendingGoStop: null,
    pendingGukjinChoice: null,
    pendingKung: null,
    result: null,
    deck,
    board,
    players: nextPlayers,
    currentTurn: nextPlayerKey,
    log: nextLog
  };

  if (newlyCaptured.some((c) => c.bonus?.stealPi)) {
    const stealCount = newlyCaptured.reduce((sum, c) => sum + (c.bonus?.stealPi || 0), 0);
    extraSteal += stealCount;
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

  const turnNo = (nextState.turnSeq || 0) + 1;
  const kiboNo = (nextState.kiboSeq || 0) + 1;
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
      board: nextState.board.map(packCard),
      hands: {
        human: nextState.players.human.hand.map(packCard),
        ai: nextState.players.ai.hand.map(packCard)
      },
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

  if (shouldPromptGukjinChoice(nextState.players[currentKey])) {
    return {
      ...nextState,
      phase: "gukjin-choice",
      pendingGukjinChoice: { playerKey: currentKey },
      log: nextState.log.concat(
        `${nextState.players[currentKey].label}: 국진(9월 열) 첫 소유 - 열/쌍피 선택`
      )
    };
  }

  return continueAfterTurnIfNeeded(nextState, currentKey);
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
    phase: "playing",
    pendingGoStop: null,
    pendingMatch: null,
    pendingKung: null,
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
    pendingKung: null,
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

export function chooseKungUse(state, playerKey) {
  if (state.phase !== "kung-choice" || state.pendingKung?.playerKey !== playerKey) return state;
  const month = state.pendingKung.month;
  const player = state.players[playerKey];
  const boardCard = state.board.find((c) => c.month === month);
  if (!boardCard) {
    return chooseKungPass(state, playerKey);
  }

  const handMonthCards = player.hand.filter((c) => c.month === month).slice(0, 3);
  if (handMonthCards.length < 3) {
    return chooseKungPass(state, playerKey);
  }

  const nextHand = player.hand.filter((c) => !handMonthCards.some((h) => h.id === c.id));
  const nextBoard = state.board.filter((c) => c.id !== boardCard.id);
  const nextCaptured = {
    ...player.captured,
    kwang: player.captured.kwang.slice(),
    five: player.captured.five.slice(),
    ribbon: player.captured.ribbon.slice(),
    junk: player.captured.junk.slice()
  };
  [boardCard, ...handMonthCards].forEach((c) => pushCaptured(nextCaptured, c));
  const nextEvents = { ...player.events, kung: (player.events.kung || 0) + 1 };

  const nextPlayers = {
    ...state.players,
    [playerKey]: {
      ...player,
      hand: nextHand,
      captured: nextCaptured,
      events: nextEvents
    }
  };
  const log = state.log.concat(
    `${player.label}: 쿵 사용 (${month}월 3장+바닥 1장 획득, 상대 피 1장 강탈)`
  );
  const { updatedPlayers, stealLog } = stealPiFromOpponent(nextPlayers, playerKey, 1);
  return {
    ...state,
    players: updatedPlayers,
    board: nextBoard,
    phase: "playing",
    pendingKung: null,
    log: log.concat(stealLog),
    kiboSeq: (state.kiboSeq || 0) + 1,
    kibo: (state.kibo || []).concat({
      no: (state.kiboSeq || 0) + 1,
      type: "kung_use",
      playerKey,
      month
    })
  };
}

export function chooseKungPass(state, playerKey) {
  if (state.phase !== "kung-choice" || state.pendingKung?.playerKey !== playerKey) return state;
  const log = state.log.concat(`${state.players[playerKey].label}: 쿵 패스`);
  return {
    ...state,
    phase: "playing",
    pendingKung: null,
    log,
    kiboSeq: (state.kiboSeq || 0) + 1,
    kibo: (state.kibo || []).concat({
      no: (state.kiboSeq || 0) + 1,
      type: "kung_pass",
      playerKey
    })
  };
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
    pendingKung: null,
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
    pendingKung: null,
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
  const nextPlayers = {
    ...state.players,
    [playerKey]: { ...player, gukjinMode: mode, gukjinLocked: true }
  };
  const label = mode === "junk" ? "쌍피" : "열";
  const log = state.log.concat(`${player.label}: 국진(9월 열) ${label} 처리 선택(낙장불입)`);

  const nextState = {
    ...state,
    players: nextPlayers,
    phase: "playing",
    pendingGukjinChoice: null,
    pendingKung: null,
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

function continueAfterTurnIfNeeded(state, justPlayedKey) {
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
  if (
    rules.useEarlyStop &&
    scoreInfo.base >= rules.goMinScore &&
    isRaisedSinceLastGo &&
    state.players[justPlayedKey].hand.length > 0
  ) {
    return {
      ...state,
      phase: "go-stop",
      pendingGoStop: justPlayedKey
    };
  }

  const bothHandsEmpty =
    state.players.human.hand.length === 0 && state.players.ai.hand.length === 0;

  if (bothHandsEmpty) {
    return resolveRound(state, justPlayedKey);
  }

  // 누구나 자기 첫 턴 시작 시 손패 대통령(동월 4장)이면 즉시 선언/선택 단계로 진입.
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

export function estimateRemaining(state) {
  const seen = [
    ...state.board,
    ...state.players.human.hand,
    ...state.players.ai.hand,
    ...state.players.human.captured.kwang,
    ...state.players.human.captured.five,
    ...state.players.human.captured.ribbon,
    ...state.players.human.captured.junk,
    ...state.players.ai.captured.kwang,
    ...state.players.ai.captured.five,
    ...state.players.ai.captured.ribbon,
    ...state.players.ai.captured.junk
  ];

  const groupedSeen = groupByMonth(seen);
  const result = {};
  for (let m = 1; m <= 12; m += 1) {
    const seenCount = groupedSeen[m]?.length ?? 0;
    result[m] = 4 - seenCount;
  }
  return result;
}






