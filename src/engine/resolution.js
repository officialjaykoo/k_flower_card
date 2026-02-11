import { calculateScore } from "./scoring.js";
import { POINT_GOLD_UNIT, settleRoundGold } from "./economy.js";

function isFailedGo(player, currentBase) {
  return player.goCount > 0 && currentBase <= player.lastGoBase;
}

export function resolveRound(state, stopperKey) {
  let humanScore = calculateScore(state.players.human, state.players.ai, state.ruleKey);
  let aiScore = calculateScore(state.players.ai, state.players.human, state.ruleKey);
  const humanPpukWin = (state.players.human.events.ppuk || 0) >= 3;
  const aiPpukWin = (state.players.ai.events.ppuk || 0) >= 3;

  if (humanPpukWin || aiPpukWin) {
    if (humanPpukWin && !aiPpukWin) {
      humanScore = { ...humanScore, base: 7, multiplier: 1, total: 7 };
      aiScore = { ...aiScore, base: 0, multiplier: 1, total: 0 };
    } else if (aiPpukWin && !humanPpukWin) {
      aiScore = { ...aiScore, base: 7, multiplier: 1, total: 7 };
      humanScore = { ...humanScore, base: 0, multiplier: 1, total: 0 };
    }
  }

  const winner =
    humanPpukWin && !aiPpukWin
      ? "human"
      : aiPpukWin && !humanPpukWin
      ? "ai"
      : humanScore.total === aiScore.total
      ? "draw"
      : humanScore.total > aiScore.total
      ? "human"
      : "ai";

  const nagariReasons = [];
  if (winner === "draw") nagariReasons.push("무승부");
  if (humanScore.base <= 0 && aiScore.base <= 0) nagariReasons.push("양측 무득점");
  if (isFailedGo(state.players.human, humanScore.base)) nagariReasons.push("플레이어 고 실패");
  if (isFailedGo(state.players.ai, aiScore.base)) nagariReasons.push("AI 고 실패");
  if (humanPpukWin || aiPpukWin) nagariReasons.length = 0;
  const nagari = nagariReasons.length > 0;

  const carry = state.carryOverMultiplier || 1;
  let nextCarryOverMultiplier = 1;
  let resolvedWinner = winner;

  if (nagari) {
    nextCarryOverMultiplier = carry * 2;
    resolvedWinner = "draw";
  } else {
    if (resolvedWinner !== "draw" && carry > 1) {
      if (resolvedWinner === "human") {
        humanScore = {
          ...humanScore,
          multiplier: humanScore.multiplier * carry,
          total: humanScore.total * carry
        };
      } else {
        aiScore = {
          ...aiScore,
          multiplier: aiScore.multiplier * carry,
          total: aiScore.total * carry
        };
      }
    }

    if (resolvedWinner !== "draw") {
      const winnerPlayer = state.players[resolvedWinner];
      const hasShakingBombWin = winnerPlayer.events.shaking > 0 || winnerPlayer.events.bomb > 0;
      if (winnerPlayer.presidentHold && hasShakingBombWin) {
        if (resolvedWinner === "human") {
          humanScore = {
            ...humanScore,
            multiplier: humanScore.multiplier * 4,
            total: humanScore.total * 4
          };
        } else {
          aiScore = {
            ...aiScore,
            multiplier: aiScore.multiplier * 4,
            total: aiScore.total * 4
          };
        }
      }
    }
  }

  const settled =
    !nagari && (resolvedWinner === "human" || resolvedWinner === "ai")
      ? settleRoundGold(
          state.players,
          resolvedWinner,
          resolvedWinner === "human" ? humanScore.total : aiScore.total
        )
      : { updatedPlayers: state.players, log: [], requested: 0, paid: 0 };

  const log = state.log
    .concat(`라운드 정산: 플레이어 ${humanScore.total} / AI ${aiScore.total} (승자: ${resolvedWinner})`)
    .concat(
      !nagari && (resolvedWinner === "human" || resolvedWinner === "ai")
        ? [
            `라운드 정산(골드): ${state.players[resolvedWinner].label} 요구 ${settled.requested}골드 / 수령 ${settled.paid}골드`
          ]
        : []
    )
    .concat(settled.log)
    .concat(
      nagari
        ? [`나가리 판(${nagariReasons.join(", ")}): 다음 판 배수 x${nextCarryOverMultiplier}`]
        : []
    );

  return {
    ...state,
    phase: "resolution",
    currentTurn: stopperKey === "human" ? "ai" : "human",
    nextCarryOverMultiplier,
    players: settled.updatedPlayers,
    kiboSeq: (state.kiboSeq || 0) + 1,
    kibo: (state.kibo || []).concat({
      no: (state.kiboSeq || 0) + 1,
      type: "round_end",
      winner: resolvedWinner,
      nagari,
      nagariReasons,
      scores: {
        human: humanScore,
        ai: aiScore
      }
    }),
    result: {
      human: humanScore,
      ai: aiScore,
      winner: resolvedWinner,
      nagari,
      nagariReasons,
      gold: {
        requested: settled.requested,
        paid: settled.paid,
        unitPerPoint: POINT_GOLD_UNIT
      }
    },
    log
  };
}
