import { useEffect, useMemo, useRef, useState } from "react";
import CardView from "./CardView.jsx";

const CHOICE_LIMIT_MS = 10000;

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
  onStartRandomGame
}) {
  const timedChoiceKey = useMemo(() => {
    if (state.phase === "select-match" && state.pendingMatch?.playerKey && participantType(ui, state.pendingMatch.playerKey) === "human") {
      return `match:${state.pendingMatch.playerKey}:${(state.pendingMatch.boardCardIds || []).join(",")}`;
    }
    if (state.phase === "go-stop" && state.pendingGoStop && participantType(ui, state.pendingGoStop) === "human") {
      return `gostop:${state.pendingGoStop}`;
    }
    if (state.phase === "president-choice" && state.pendingPresident?.playerKey && participantType(ui, state.pendingPresident.playerKey) === "human") {
      return `president:${state.pendingPresident.playerKey}:${state.pendingPresident.month}`;
    }
    if (state.phase === "gukjin-choice" && state.pendingGukjinChoice?.playerKey && participantType(ui, state.pendingGukjinChoice.playerKey) === "human") {
      return `gukjin:${state.pendingGukjinChoice.playerKey}`;
    }
    return null;
  }, [state, ui, participantType]);

  const [deadlineAt, setDeadlineAt] = useState(null);
  const [tick, setTick] = useState(Date.now());
  const autoHandledKeyRef = useRef(null);

  useEffect(() => {
    if (!timedChoiceKey) {
      setDeadlineAt(null);
      return;
    }
    setDeadlineAt(Date.now() + CHOICE_LIMIT_MS);
  }, [timedChoiceKey]);

  useEffect(() => {
    if (!deadlineAt) return;
    const t = setInterval(() => setTick(Date.now()), 200);
    return () => clearInterval(t);
  }, [deadlineAt]);

  const remainSec = deadlineAt ? Math.max(0, Math.ceil((deadlineAt - tick) / 1000)) : null;

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
  const winnerScore =
    winnerKey === "human" ? state.result?.human : winnerKey === "ai" ? state.result?.ai : null;
  const resultCauseText = state.result?.nagari
    ? `나가리: ${(state.result.nagariReasons || []).join(", ")}`
    : winnerScore?.breakdown?.presidentStop
    ? "대통령!"
    : "일반 종료";

  return (
    <>
      {reveal && (
        <div className="result-overlay result-overlay-passive">
          <div className="panel result-panel">
            <div className="section-title">{reveal.title || "알림"}</div>
            <div className="meta">
              {reveal.message ||
                (reveal.playerKey
                  ? `${state.players[reveal.playerKey].label} ${reveal.month || ""}월 카드 공개`
                  : "")}
            </div>
            <div className="hand">{(reveal.cards || []).map((c) => <CardView key={`rv-${c.id}`} card={c} />)}</div>
          </div>
        </div>
      )}


      {state.phase === "select-match" && state.pendingMatch?.playerKey && participantType(ui, state.pendingMatch.playerKey) === "human" && (
        <div className="result-overlay">
          <div className="panel result-panel">
            <div className="section-title">카드 선택</div>
            <div className="meta">{state.pendingMatch?.message || "보드 카드 1장을 선택하세요."}</div>
            <div className="meta">남은 시간: {remainSec ?? 10}초</div>
            <div className="hand">
              {state.board
                .filter((c) => (state.pendingMatch?.boardCardIds || []).includes(c.id))
                .map((c) => (
                  <CardView key={`m-${c.id}`} card={c} interactive onClick={() => onChooseMatch(c.id)} />
                ))}
            </div>
          </div>
        </div>
      )}

      {state.phase === "go-stop" && state.pendingGoStop && participantType(ui, state.pendingGoStop) === "human" && (
        <div className="result-overlay">
          <div className="panel result-panel">
            <div className="section-title">Go / Stop 선택</div>
            <div className="meta">남은 시간: {remainSec ?? 10}초</div>
            <div className="control-row">
              <button onClick={() => onChooseGo(state.pendingGoStop)}>Go</button>
              <button onClick={() => onChooseStop(state.pendingGoStop)}>Stop</button>
            </div>
          </div>
        </div>
      )}

      {state.phase === "president-choice" && state.pendingPresident?.playerKey && participantType(ui, state.pendingPresident.playerKey) === "human" && (
        <div className="result-overlay">
          <div className="panel result-panel">
            <div className="section-title">대통령 선택</div>
            <div className="meta">남은 시간: {remainSec ?? 10}초</div>
            <div className="control-row">
              <button onClick={() => onChoosePresidentStop(state.pendingPresident.playerKey)}>10점 종료</button>
              <button onClick={() => onChoosePresidentHold(state.pendingPresident.playerKey)}>들고치기</button>
            </div>
          </div>
        </div>
      )}

      {state.phase === "shaking-confirm" &&
        state.pendingShakingConfirm?.playerKey &&
        participantType(ui, state.pendingShakingConfirm.playerKey) === "human" && (
          <div className="result-overlay">
            <div className="panel result-panel">
              <div className="section-title">흔들기 선택</div>
              <div className="meta">{state.players[state.pendingShakingConfirm.playerKey].label}: 흔들기 선언할까요?</div>
              <div className="control-row">
                <button onClick={() => onChooseShakingYes(state.pendingShakingConfirm.playerKey)}>예</button>
                <button onClick={() => onChooseShakingNo(state.pendingShakingConfirm.playerKey)}>아니요</button>
              </div>
            </div>
          </div>
        )}

      {state.phase === "gukjin-choice" && state.pendingGukjinChoice?.playerKey && participantType(ui, state.pendingGukjinChoice.playerKey) === "human" && (
        <div className="result-overlay">
          <div className="panel result-panel">
            <div className="section-title">국진 선택</div>
            <div className="meta">남은 시간: {remainSec ?? 10}초</div>
            <div className="control-row">
              <button onClick={() => onChooseGukjinMode(state.pendingGukjinChoice.playerKey, "five")}>열로 확정</button>
              <button onClick={() => onChooseGukjinMode(state.pendingGukjinChoice.playerKey, "junk")}>쌍피로 확정</button>
            </div>
          </div>
        </div>
      )}

      {state.phase === "resolution" && state.result && (
        <div className="result-overlay">
          <div className="panel result-panel">
            <div className="section-title">라운드 종료</div>
            <div className="meta">
              {state.result.winner === "human"
                ? `${state.players.human.label} 승리`
                : state.result.winner === "ai"
                ? `${state.players.ai.label} 승리`
                : "무승부"}
            </div>
            <div className="meta">{resultCauseText}</div>
            <div className="meta">{state.players.human.label} {state.result.human.total} / {state.players.ai.label} {state.result.ai.total}</div>
            <div className="control-row">
              <button onClick={onStartSpecifiedGame}>같은패로 게임시작</button>
              <button onClick={onStartRandomGame}>새 게임시작</button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
