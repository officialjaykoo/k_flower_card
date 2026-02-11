import { STARTING_GOLD } from "./economy.js";
import { pushCaptured } from "./capturesEvents.js";

export function decideFirstTurn(humanCard, aiCard, rng = Math.random) {
  const hour = new Date().getHours();
  const isNight = hour >= 18 || hour < 6;

  let winnerKey = "human";
  let reason = "";
  if (humanCard.month === aiCard.month) {
    winnerKey = rng() < 0.5 ? "human" : "ai";
    reason = "동월 랜덤";
  } else if (isNight) {
    winnerKey = humanCard.month < aiCard.month ? "human" : "ai";
    reason = "밤일낮장(밤: 낮은 월 우선)";
  } else {
    winnerKey = humanCard.month > aiCard.month ? "human" : "ai";
    reason = "밤일낮장(낮: 높은 월 우선)";
  }

  const winnerLabel = winnerKey === "human" ? "플레이어" : "AI";
  const log =
    `선턴 결정 [${isNight ? "밤" : "낮"}]: ` +
    `플레이어 ${humanCard.month}월 vs AI ${aiCard.month}월 -> ${winnerLabel} 선 (${reason})`;

  return { winnerKey, log };
}

export function findKungMonth(hand, board) {
  const counts = {};
  for (const c of hand) {
    if (c.month > 12) continue;
    counts[c.month] = (counts[c.month] || 0) + 1;
  }
  const boardMonths = new Set(board.filter((c) => c.month <= 12).map((c) => c.month));
  for (const [monthText, count] of Object.entries(counts)) {
    const month = Number(monthText);
    if (count >= 3 && boardMonths.has(month)) return month;
  }
  return null;
}

export function emptyPlayer(label) {
  return {
    label,
    hand: [],
    captured: { kwang: [], ribbon: [], five: [], junk: [] },
    gukjinMode: "five",
    gukjinLocked: false,
    goCount: 0,
    turnCount: 0,
    lastGoBase: 0,
    presidentHold: false,
    presidentHoldMonth: null,
    shakingDeclaredMonths: [],
    heldBonusCards: [],
    ppukState: {
      active: false,
      streak: 0,
      lastTurnNo: 0,
      lastSource: null,
      lastMonth: null
    },
    gold: STARTING_GOLD,
    declaredStop: false,
    score: 0,
    events: {
      ttak: 0,
      ppuk: 0,
      jjob: 0,
      shaking: 0,
      bomb: 0,
      ddadak: 0,
      ssul: 0,
      jabbeok: 0,
      yeonPpuk: 0,
      kung: 0
    }
  };
}

export function normalizeOpeningHands(players, remain, initLog) {
  let nextRemain = remain.slice();
  ["human", "ai"].forEach((key) => {
    const player = players[key];
    const keep = [];
    const pulled = [];
    player.hand.forEach((card) => {
      if (card.bonus?.stealPi) {
        pulled.push(card);
      } else {
        keep.push(card);
      }
    });
    if (pulled.length > 0) {
      pulled.forEach((card) => {
        pushCaptured(player.captured, card);
        initLog.push(`시작 보정: ${player.label} 손패 ${card.name} 즉시 획득`);
      });
      while (keep.length < 10 && nextRemain.length > 0) {
        keep.push(nextRemain[0]);
        nextRemain = nextRemain.slice(1);
      }
      player.hand = keep;
    }
  });
  return nextRemain;
}

export function normalizeOpeningBoard(board, remain, firstPlayer, initLog) {
  let nextBoard = board.slice();
  let nextRemain = remain.slice();
  let changed = true;
  while (changed) {
    changed = false;
    const bonus = nextBoard.filter((c) => c.bonus?.stealPi);
    if (bonus.length === 0) break;
    changed = true;
    bonus.forEach((card) => {
      pushCaptured(firstPlayer.captured, card);
      initLog.push(`시작 보정: 바닥 ${card.name}는 선 플레이어가 획득`);
    });
    nextBoard = nextBoard.filter((c) => !c.bonus?.stealPi);
    while (nextBoard.length < 8 && nextRemain.length > 0) {
      nextBoard.push(nextRemain[0]);
      nextRemain = nextRemain.slice(1);
    }
  }
  return { board: nextBoard, remain: nextRemain };
}

export function findPresidentMonth(cards) {
  const counts = {};
  for (const card of cards) {
    if (card.month > 12) continue;
    counts[card.month] = (counts[card.month] || 0) + 1;
    if (counts[card.month] >= 4) return card.month;
  }
  return null;
}
