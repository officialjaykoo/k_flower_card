import { STARTING_GOLD } from "./economy.js";
import { pushCaptured } from "./capturesEvents.js";

/* ============================================================================
 * Opening helpers
 * - first-turn decision
 * - initial player template
 * - opening board/hand normalization
 * ========================================================================== */

/* 1) First-turn decision */
export function decideFirstTurn(humanCard, aiCard, rng = Math.random) {
  const hour = new Date().getHours();
  const isNight = hour >= 18 || hour < 6;

  let winnerKey = "human";
  let reason = "";
  if (humanCard.month === aiCard.month) {
    winnerKey = rng() < 0.5 ? "human" : "ai";
    reason = "same month random";
  } else if (isNight) {
    winnerKey = humanCard.month < aiCard.month ? "human" : "ai";
    reason = "night rule: lower month starts";
  } else {
    winnerKey = humanCard.month > aiCard.month ? "human" : "ai";
    reason = "day rule: higher month starts";
  }

  const winnerLabel = winnerKey === "human" ? "Player" : "AI";
  const log =
    `Starter decided [${isNight ? "night" : "day"}]: ` +
    `Player ${humanCard.month} vs AI ${aiCard.month} -> ${winnerLabel} (${reason})`;

  return { winnerKey, log };
}

/* 2) Player template */
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
      pansseul: 0,
      ppuk: 0,
      jjob: 0,
      shaking: 0,
      bomb: 0,
      ddadak: 0,
      ssul: 0,
      jabbeok: 0,
      yeonPpuk: 0
    }
  };
}

/* 3) Opening normalization */
export function normalizeOpeningHands(players, remain, initLog) {
  // Opening bonus cards are not auto-triggered in hand phase.
  // The effect is applied only when the player actually acquires the card.
  return remain.slice();
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
      initLog.push(`Opening adjust: ${firstPlayer.label} captures board bonus card ${card.name}`);
    });
    nextBoard = nextBoard.filter((c) => !c.bonus?.stealPi);
    while (nextBoard.length < 8 && nextRemain.length > 0) {
      nextBoard.push(nextRemain[0]);
      nextRemain = nextRemain.slice(1);
    }
  }
  return { board: nextBoard, remain: nextRemain };
}

/* 4) Opening special checks */
export function findPresidentMonth(cards) {
  const counts = {};
  for (const card of cards) {
    // President check only counts normal month cards (1..12).
    // Excludes pass/dummy (month 0) and bonus cards (month 13).
    if (!card || card.passCard) continue;
    if (card.month < 1 || card.month > 12) continue;
    counts[card.month] = (counts[card.month] || 0) + 1;
    if (counts[card.month] >= 4) return card.month;
  }
  return null;
}
