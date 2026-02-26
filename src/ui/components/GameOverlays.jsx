import { useEffect, useMemo, useRef, useState } from "react";
import CardView from "./CardView.jsx";

const CHOICE_LIMIT_MS = 10000;

/* ============================================================================
 * Overlay controller
 * - timed human choices (match/go-stop/president/gukjin)
 * - round/reveal modal layers
 * ========================================================================== */

function isHumanMatchSelection(state, ui, participantType) {
  return (
    state.phase === "select-match" &&
    state.pendingMatch?.playerKey &&
    participantType(ui, state.pendingMatch.playerKey) === "human"
  );
}

function isHumanGoStop(state, ui, participantType) {
  return (
    state.phase === "go-stop" &&
    state.pendingGoStop &&
    participantType(ui, state.pendingGoStop) === "human"
  );
}

function isHumanPresidentChoice(state, ui, participantType) {
  return (
    state.phase === "president-choice" &&
    state.pendingPresident?.playerKey &&
    participantType(ui, state.pendingPresident.playerKey) === "human"
  );
}

function isHumanGukjinChoice(state, ui, participantType) {
  return (
    state.phase === "gukjin-choice" &&
    state.pendingGukjinChoice?.playerKey &&
    participantType(ui, state.pendingGukjinChoice.playerKey) === "human"
  );
}

function buildTimedChoiceKey(state, ui, participantType) {
  if (isHumanMatchSelection(state, ui, participantType)) {
    return `match:${state.pendingMatch.playerKey}:${(state.pendingMatch.boardCardIds || []).join(",")}`;
  }
  if (isHumanGoStop(state, ui, participantType)) {
    return `gostop:${state.pendingGoStop}`;
  }
  if (isHumanPresidentChoice(state, ui, participantType)) {
    return `president:${state.pendingPresident.playerKey}:${state.pendingPresident.month}`;
  }
  if (isHumanGukjinChoice(state, ui, participantType)) {
    return `gukjin:${state.pendingGukjinChoice.playerKey}`;
  }
  return null;
}

export default function GameOverlays({
  state,
  ui,
  reveal,
  participantType,
  onChooseMatch,
  onChooseGo,
  onChooseStop,
  onChoosePresidentStop,
  onChoosePresidentHold,
  onChooseGukjinMode,
  onChooseShakingYes,
  onChooseShakingNo,
  onStartSpecifiedGame,
  onStartRandomGame,
  t
}) {
  /* 1) Timed choice bookkeeping */
  const timedChoiceKey = useMemo(
    () => buildTimedChoiceKey(state, ui, participantType),
    [state, ui, participantType]
  );

  const [deadlineAt, setDeadlineAt] = useState(null);
  const [tick, setTick] = useState(Date.now());
  const autoHandledKeyRef = useRef(null);

  // Reset deadline whenever a new timed prompt appears.
  useEffect(() => {
    if (!timedChoiceKey) {
      setDeadlineAt(null);
      return;
    }
    setDeadlineAt(Date.now() + CHOICE_LIMIT_MS);
  }, [timedChoiceKey]);

  // Keep countdown fresh while deadline is active.
  useEffect(() => {
    if (!deadlineAt) return;
    const timer = setInterval(() => setTick(Date.now()), 200);
    return () => clearInterval(timer);
  }, [deadlineAt]);

  const remainSec = deadlineAt ? Math.max(0, Math.ceil((deadlineAt - tick) / 1000)) : null;

  // Auto-stop only for go/stop timeout.
  useEffect(() => {
    if (!timedChoiceKey) {
      autoHandledKeyRef.current = null;
      return;
    }
    if (!timedChoiceKey.startsWith("gostop:")) return;
    if (remainSec === null || remainSec > 0) return;
    if (autoHandledKeyRef.current === timedChoiceKey) return;
    autoHandledKeyRef.current = timedChoiceKey;
    if (state.phase === "go-stop" && state.pendingGoStop) {
      onChooseStop(state.pendingGoStop);
    }
  }, [timedChoiceKey, remainSec, state.phase, state.pendingGoStop, onChooseStop]);

  const winnerKey = state.result?.winner;
  const winnerScore = winnerKey === "human" ? state.result?.human : winnerKey === "ai" ? state.result?.ai : null;
  const resultCauseText = state.result?.nagari
    ? t("overlay.result.nagari", { reasons: (state.result.nagariReasons || []).join(", ") })
    : winnerScore?.breakdown?.presidentStop
    ? t("overlay.result.president")
    : t("overlay.result.normal");

  /* 2) Render overlays by phase */
  return (
    <>
      {reveal && (
        <div className="result-overlay result-overlay-passive">
          <div className="panel result-panel">
            <div className="section-title">{reveal.title || t("overlay.notice")}</div>
            <div className="meta">
              {reveal.message ||
                (reveal.playerKey
                  ? t("overlay.revealMessage", {
                      player: state.players[reveal.playerKey].label,
                      month: reveal.month || ""
                    })
                  : "")}
            </div>
            <div className="hand">{(reveal.cards || []).map((c) => <CardView key={`rv-${c.id}`} card={c} t={t} />)}</div>
          </div>
        </div>
      )}

      {isHumanMatchSelection(state, ui, participantType) && (
          <div className="result-overlay result-overlay-match-top">
            <div className="panel result-panel">
              <div className="section-title">{t("overlay.selectCard.title")}</div>
              <div className="meta">{state.pendingMatch?.message || t("overlay.selectCard.defaultMessage")}</div>
              <div className="meta">{t("overlay.remainingTime", { seconds: remainSec ?? 10 })}</div>
              <div className="hand">
                {state.board
                  .filter((c) => (state.pendingMatch?.boardCardIds || []).includes(c.id))
                  .map((c) => (
                    <CardView key={`m-${c.id}`} card={c} interactive onClick={() => onChooseMatch(c.id)} t={t} />
                  ))}
              </div>
            </div>
          </div>
        )}

      {isHumanGoStop(state, ui, participantType) && (
        <div className="result-overlay">
          <div className="panel result-panel">
            <div className="section-title">{t("overlay.goStop.title")}</div>
            <div className="meta">{t("overlay.remainingTime", { seconds: remainSec ?? 10 })}</div>
            <div className="control-row">
              <button onClick={() => onChooseGo(state.pendingGoStop)}>Go</button>
              <button onClick={() => onChooseStop(state.pendingGoStop)}>Stop</button>
            </div>
          </div>
        </div>
      )}

      {isHumanPresidentChoice(state, ui, participantType) && (
          <div className="result-overlay">
            <div className="panel result-panel">
              <div className="section-title">{t("overlay.president.title")}</div>
              <div className="meta">{t("overlay.remainingTime", { seconds: remainSec ?? 10 })}</div>
              <div className="control-row">
                <button onClick={() => onChoosePresidentStop(state.pendingPresident.playerKey)}>{t("overlay.president.stop")}</button>
                <button onClick={() => onChoosePresidentHold(state.pendingPresident.playerKey)}>{t("overlay.president.hold")}</button>
              </div>
            </div>
          </div>
        )}

      {state.phase === "shaking-confirm" &&
        state.pendingShakingConfirm?.playerKey &&
        participantType(ui, state.pendingShakingConfirm.playerKey) === "human" && (
          <div className="result-overlay">
            <div className="panel result-panel">
              <div className="section-title">{t("overlay.shaking.title")}</div>
              <div className="meta">
                {t("overlay.shaking.confirm", {
                  player: state.players[state.pendingShakingConfirm.playerKey].label
                })}
              </div>
              <div className="control-row">
                <button onClick={() => onChooseShakingYes(state.pendingShakingConfirm.playerKey)}>{t("overlay.yes")}</button>
                <button onClick={() => onChooseShakingNo(state.pendingShakingConfirm.playerKey)}>{t("overlay.no")}</button>
              </div>
            </div>
          </div>
        )}

      {isHumanGukjinChoice(state, ui, participantType) && (
          <div className="result-overlay">
            <div className="panel result-panel">
              <div className="section-title">{t("overlay.gukjin.title")}</div>
              <div className="meta">{t("overlay.remainingTime", { seconds: remainSec ?? 10 })}</div>
              <div className="control-row">
                <button onClick={() => onChooseGukjinMode(state.pendingGukjinChoice.playerKey, "five")}>{t("overlay.gukjin.five")}</button>
                <button onClick={() => onChooseGukjinMode(state.pendingGukjinChoice.playerKey, "junk")}>{t("overlay.gukjin.junk")}</button>
              </div>
            </div>
          </div>
        )}

      {state.phase === "resolution" && state.result && (
        <div className="result-overlay">
          <div className="panel result-panel">
            <div className="section-title">{t("overlay.roundEnd.title")}</div>
            <div className="meta">
              {state.result.winner === "human"
                ? t("overlay.result.win", { player: state.players.human.label })
                : state.result.winner === "ai"
                ? t("overlay.result.win", { player: state.players.ai.label })
                : t("overlay.result.draw")}
            </div>
            <div className="meta">{resultCauseText}</div>
            <div className="meta">{state.players.human.label} {state.result.human.total} / {state.players.ai.label} {state.result.ai.total}</div>
            <div className="control-row">
              <button onClick={onStartSpecifiedGame}>{t("overlay.button.startSpecified")}</button>
              <button onClick={onStartRandomGame}>{t("overlay.button.startRandom")}</button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
