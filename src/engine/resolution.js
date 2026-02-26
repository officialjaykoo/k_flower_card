import { calculateScore, calculateBaseScore, isGukjinCard } from "./scoring.js";
import { POINT_GOLD_UNIT, settleRoundGold } from "./economy.js";
import { ruleSets } from "./rules.js";

/* ============================================================================
 * Round resolution pipeline
 * - score/bak calculation
 * - STOP/GO-fail/nagari handling
 * - carry-over and gold settlement
 * ========================================================================== */

/* 1) Helper predicates */
function isFailedGo(player, currentBase) {
  return player.goCount > 0 && currentBase <= player.lastGoBase;
}

/* 2) Gukjin transform helpers (STOP defensive auto-convert) */
function hasTransformableGukjinFive(player) {
  return (player?.captured?.five || []).some(
    (card) => isGukjinCard(card) && card?.category === "five" && !card?.gukjinTransformed
  );
}

function transformGukjinFiveToJunk(player) {
  const captured = {
    ...player.captured,
    kwang: (player.captured?.kwang || []).slice(),
    five: (player.captured?.five || []).slice(),
    ribbon: (player.captured?.ribbon || []).slice(),
    junk: (player.captured?.junk || []).slice()
  };
  const gukjinIdx = captured.five.findIndex(
    (card) => isGukjinCard(card) && card?.category === "five" && !card?.gukjinTransformed
  );
  if (gukjinIdx < 0) return null;
  const [gukjinCard] = captured.five.splice(gukjinIdx, 1);
  captured.junk.push({
    ...gukjinCard,
    category: "junk",
    piValue: 2,
    gukjinTransformed: true,
    name: `${gukjinCard.name} (Gukjin Pi)`
  });
  return {
    ...player,
    captured,
    gukjinMode: "junk",
    gukjinLocked: true
  };
}

export function resolveRound(state, stopperKey) {
  const rules = ruleSets[state.ruleKey];
  const hasStopper =
    (stopperKey === "human" || stopperKey === "ai") && !!state.players?.[stopperKey]?.declaredStop;
  let workingState = state;
  const autoGukjinLogs = [];

  /* 3) STOP-triggered auto gukjin transform (defense against pi-bak) */
  if (hasStopper) {
    const loserKey = stopperKey === "human" ? "ai" : "human";
    const winnerPlayer = workingState.players?.[stopperKey];
    const loserPlayer = workingState.players?.[loserKey];
    if (winnerPlayer && loserPlayer && hasTransformableGukjinFive(loserPlayer)) {
      const winnerScoreBefore = calculateScore(winnerPlayer, loserPlayer, workingState.ruleKey);
      if (winnerScoreBefore?.bak?.pi) {
        const switchedLoser = transformGukjinFiveToJunk(loserPlayer);
        if (switchedLoser) {
          workingState = {
            ...workingState,
            players: {
              ...workingState.players,
              [loserKey]: switchedLoser
            }
          };
          autoGukjinLogs.push(`${loserPlayer.label}: auto-converted gukjin to double-pi to defend against pi-bak`);
        }
      }
    }
  }

  /* 4) Base score calculation (+ ppuk instant-win override) */
  let humanScore = calculateScore(workingState.players.human, workingState.players.ai, workingState.ruleKey);
  let aiScore = calculateScore(workingState.players.ai, workingState.players.human, workingState.ruleKey);
  const humanBaseOnly = calculateBaseScore(workingState.players.human, rules).base;
  const aiBaseOnly = calculateBaseScore(workingState.players.ai, rules).base;
  const humanPpukWin = (workingState.players.human.events.ppuk || 0) >= 3;
  const aiPpukWin = (workingState.players.ai.events.ppuk || 0) >= 3;

  if (humanPpukWin || aiPpukWin) {
    if (humanPpukWin && !aiPpukWin) {
      humanScore = { ...humanScore, base: 7, multiplier: 1, total: 7, payoutTotal: 7 };
      aiScore = { ...aiScore, base: 0, multiplier: 1, total: 0, payoutTotal: 0 };
    } else if (aiPpukWin && !humanPpukWin) {
      aiScore = { ...aiScore, base: 7, multiplier: 1, total: 7, payoutTotal: 7 };
      humanScore = { ...humanScore, base: 0, multiplier: 1, total: 0, payoutTotal: 0 };
    }
  }

  /* 5) Winner decision (ppuk > stopper override > score) */
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

  // If the round ended by STOP declaration, the stopper wins regardless of tie/close-score state.
  if (!humanPpukWin && !aiPpukWin && hasStopper) {
    winner = stopperKey;
  }

  /* 6) GO-fail adjustments */
  const humanFailedGo = isFailedGo(workingState.players.human, humanBaseOnly);
  const aiFailedGo = isFailedGo(workingState.players.ai, aiBaseOnly);

  let unresolvedFailedGo = [];
  if (!humanPpukWin && !aiPpukWin) {
    if (humanFailedGo && !aiFailedGo) {
      if (aiBaseOnly >= rules.goMinScore) winner = "ai";
      else unresolvedFailedGo.push("player");
    } else if (aiFailedGo && !humanFailedGo) {
      if (humanBaseOnly >= rules.goMinScore) winner = "human";
      else unresolvedFailedGo.push("ai");
    } else if (humanFailedGo && aiFailedGo) {
      unresolvedFailedGo = ["player", "ai"];
      winner = "draw";
    }
  }

  /* 7) Nagari detection */
  const nagariReasons = [];
  if (winner === "draw") nagariReasons.push("draw");
  if (humanScore.base <= 0 && aiScore.base <= 0) nagariReasons.push("no score on both sides");
  if (unresolvedFailedGo.includes("player")) nagariReasons.push("player GO failed");
  if (unresolvedFailedGo.includes("ai")) nagariReasons.push("AI GO failed");
  if (humanPpukWin || aiPpukWin) nagariReasons.length = 0;
  const nagari = nagariReasons.length > 0;

  const carry = workingState.carryOverMultiplier || 1;
  let nextCarryOverMultiplier = 1;
  let resolvedWinner = winner;
  let dokbakApplied = false;

  /* 8) Multiplier effects: carry-over / president hold / dokbak */
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
      const winnerPlayer = workingState.players[resolvedWinner];
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

    // Dokbak: if a player who already declared GO loses by opponent STOP before next go/stop, pay double.
    if (resolvedWinner !== "draw" && hasStopper) {
      const loserKey = resolvedWinner === "human" ? "ai" : "human";
      const loserPlayer = workingState.players[loserKey];
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

  /* 9) Gold settlement and round-end log assembly */
  const settled =
    !nagari && (resolvedWinner === "human" || resolvedWinner === "ai")
      ? settleRoundGold(
          workingState.players,
          resolvedWinner,
          resolvedWinner === "human"
            ? humanScore.payoutTotal || humanScore.total
            : aiScore.payoutTotal || aiScore.total
        )
      : { updatedPlayers: workingState.players, log: [], requested: 0, paid: 0 };

  const log = (workingState.log || [])
    .concat(autoGukjinLogs)
    .concat(`Round settle: Player score ${humanScore.total} / AI score ${aiScore.total} (winner: ${resolvedWinner})`)
    .concat(
      `Settle points: Player ${humanScore.payoutTotal || humanScore.total} / AI ${aiScore.payoutTotal || aiScore.total}`
    )
    .concat(
      !nagari && (resolvedWinner === "human" || resolvedWinner === "ai")
        ? [
            `Round settle (gold): ${workingState.players[resolvedWinner].label} requested ${settled.requested} gold / received ${settled.paid} gold`
          ]
        : []
    )
    .concat(settled.log)
    .concat(dokbakApplied ? ["Dokbak applied: STOP win compensation x2"] : [])
    .concat(nagari ? [`Nagari (${nagariReasons.join(", ")}): next round multiplier x${nextCarryOverMultiplier}`] : []);

  /* 10) Final resolution state */
  return {
    ...workingState,
    phase: "resolution",
    currentTurn: stopperKey === "human" ? "ai" : "human",
    nextCarryOverMultiplier,
    players: settled.updatedPlayers,
    kiboSeq: (workingState.kiboSeq || 0) + 1,
    kibo: (workingState.kibo || []).concat({
      no: (workingState.kiboSeq || 0) + 1,
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
