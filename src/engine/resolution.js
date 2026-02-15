import { calculateScore } from "./scoring.js";
import { POINT_GOLD_UNIT, settleRoundGold } from "./economy.js";
import { ruleSets } from "./rules.js";

function isFailedGo(player, currentBase) {
  return player.goCount > 0 && currentBase <= player.lastGoBase;
}

export function resolveRound(state, stopperKey) {
  const rules = ruleSets[state.ruleKey];
  const hasStopper =
    (stopperKey === "human" || stopperKey === "ai") && !!state.players?.[stopperKey]?.declaredStop;
  let humanScore = calculateScore(state.players.human, state.players.ai, state.ruleKey);
  let aiScore = calculateScore(state.players.ai, state.players.human, state.ruleKey);
  const humanPpukWin = (state.players.human.events.ppuk || 0) >= 3;
  const aiPpukWin = (state.players.ai.events.ppuk || 0) >= 3;

  if (humanPpukWin || aiPpukWin) {
    if (humanPpukWin && !aiPpukWin) {
      humanScore = { ...humanScore, base: 7, multiplier: 1, total: 7, payoutTotal: 7 };
      aiScore = { ...aiScore, base: 0, multiplier: 1, total: 0, payoutTotal: 0 };
    } else if (aiPpukWin && !humanPpukWin) {
      aiScore = { ...aiScore, base: 7, multiplier: 1, total: 7, payoutTotal: 7 };
      humanScore = { ...humanScore, base: 0, multiplier: 1, total: 0, payoutTotal: 0 };
    }
  }

  let winner =
    humanPpukWin && !aiPpukWin
      ? "human"
      : aiPpukWin && !humanPpukWin
      ? "ai"
      : humanScore.total === aiScore.total
      ? "draw"
      : humanScore.total > aiScore.total
      ? "human"
      : "ai";

  // STOP 선언으로 끝난 판은 동점/근소차 여부와 무관하게 선언자가 이긴다.
  if (!humanPpukWin && !aiPpukWin && hasStopper) {
    winner = stopperKey;
  }

  const humanFailedGo = isFailedGo(state.players.human, humanScore.base);
  const aiFailedGo = isFailedGo(state.players.ai, aiScore.base);

  let unresolvedFailedGo = [];
  if (!humanPpukWin && !aiPpukWin) {
    if (humanFailedGo && !aiFailedGo) {
      if (aiScore.base >= rules.goMinScore) winner = "ai";
      else unresolvedFailedGo.push("player");
    } else if (aiFailedGo && !humanFailedGo) {
      if (humanScore.base >= rules.goMinScore) winner = "human";
      else unresolvedFailedGo.push("ai");
    } else if (humanFailedGo && aiFailedGo) {
      unresolvedFailedGo = ["player", "ai"];
      winner = "draw";
    }
  }

  const nagariReasons = [];
  if (winner === "draw") nagariReasons.push("무승부");
  if (humanScore.base <= 0 && aiScore.base <= 0) nagariReasons.push("양측 무득점");
  if (unresolvedFailedGo.includes("player")) nagariReasons.push("플레이어 GO 실패");
  if (unresolvedFailedGo.includes("ai")) nagariReasons.push("AI GO 실패");
  if (humanPpukWin || aiPpukWin) nagariReasons.length = 0;
  const nagari = nagariReasons.length > 0;

  const carry = state.carryOverMultiplier || 1;
  let nextCarryOverMultiplier = 1;
  let resolvedWinner = winner;
  let dokbakApplied = false;

  if (nagari) {
    nextCarryOverMultiplier = carry * 2;
    resolvedWinner = "draw";
  } else {
    if (resolvedWinner !== "draw" && carry > 1) {
      if (resolvedWinner === "human") {
        humanScore = {
          ...humanScore,
          multiplier: humanScore.multiplier * carry,
          payoutTotal: humanScore.payoutTotal * carry
        };
      } else {
        aiScore = {
          ...aiScore,
          multiplier: aiScore.multiplier * carry,
          payoutTotal: aiScore.payoutTotal * carry
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
            payoutTotal: humanScore.payoutTotal * 4
          };
        } else {
          aiScore = {
            ...aiScore,
            multiplier: aiScore.multiplier * 4,
            payoutTotal: aiScore.payoutTotal * 4
          };
        }
      }
    }

    // 독박: 고한 플레이어가 다시 고/스톱 전에 상대 STOP으로 지면 2배 배상.
    if (resolvedWinner !== "draw" && hasStopper) {
      const loserKey = resolvedWinner === "human" ? "ai" : "human";
      const loserPlayer = state.players[loserKey];
      if ((loserPlayer?.goCount || 0) > 0) {
        dokbakApplied = true;
        if (resolvedWinner === "human") {
          humanScore = {
            ...humanScore,
            multiplier: humanScore.multiplier * 2,
            payoutTotal: humanScore.payoutTotal * 2,
            bak: { ...(humanScore.bak || {}), dokbak: true }
          };
        } else {
          aiScore = {
            ...aiScore,
            multiplier: aiScore.multiplier * 2,
            payoutTotal: aiScore.payoutTotal * 2,
            bak: { ...(aiScore.bak || {}), dokbak: true }
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
          resolvedWinner === "human"
            ? humanScore.payoutTotal || humanScore.total
            : aiScore.payoutTotal || aiScore.total
        )
      : { updatedPlayers: state.players, log: [], requested: 0, paid: 0 };

  const log = state.log
    .concat(`라운드 정산: 플레이어 점수 ${humanScore.total} / AI 점수 ${aiScore.total} (승자: ${resolvedWinner})`)
    .concat(`정산 포인트: 플레이어 ${humanScore.payoutTotal || humanScore.total} / AI ${aiScore.payoutTotal || aiScore.total}`)
    .concat(
      !nagari && (resolvedWinner === "human" || resolvedWinner === "ai")
        ? [
            `라운드 정산(골드): ${state.players[resolvedWinner].label} 요구 ${settled.requested}골드 / 수령 ${settled.paid}골드`
          ]
        : []
    )
    .concat(settled.log)
    .concat(dokbakApplied ? ["독박 적용: STOP 승리 배상 2배"] : [])
    .concat(nagari ? [`나가리(${nagariReasons.join(", ")}): 다음 판 배수 x${nextCarryOverMultiplier}`] : []);

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
