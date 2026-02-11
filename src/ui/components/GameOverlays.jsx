import CardView from "./CardView.jsx";

export default function GameOverlays({
  state,
  ui,
  reveal,
  participantType,
  onChooseMatch,
  updateNextFirstTurnFromResult,
  setUi,
  startGame,
  randomSeed
}) {
  return (
    <>
      {reveal && (
        <div className="result-overlay">
          <div className="panel result-panel">
            <div className="section-title">흔들기 공개</div>
            <div className="meta">{state.players[reveal.playerKey].label} {reveal.month}월 3장 공개</div>
            <div className="hand">{reveal.cards.map((c) => <CardView key={`rv-${c.id}`} card={c} />)}</div>
          </div>
        </div>
      )}

      {state.phase === "select-match" && state.pendingMatch?.playerKey && participantType(ui, state.pendingMatch.playerKey) === "human" && (
        <div className="result-overlay">
          <div className="panel result-panel">
            <div className="section-title">카드 선택</div>
            <div className="meta">{state.pendingMatch?.message || "보드 카드 1장을 선택하세요."}</div>
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
            <div className="meta">{state.result.nagari ? `나가리: ${(state.result.nagariReasons || []).join(", ")}` : "나가리 아님"}</div>
            <div className="meta">{state.players.human.label} {state.result.human.total} / {state.players.ai.label} {state.result.ai.total}</div>
            <div className="control-row">
              <button
                onClick={() => {
                  updateNextFirstTurnFromResult();
                  setUi((u) => ({ ...u, carryOverMultiplier: state.nextCarryOverMultiplier || 1 }));
                  setTimeout(
                    () =>
                      startGame({
                        keepCarry: true,
                        keepStarter: true,
                        keepGold: true,
                        firstTurnKey: ui.nextFirstTurnKey
                      }),
                    0
                  );
                }}
              >
                같은 시드로 다시
              </button>
              <button
                onClick={() => {
                  updateNextFirstTurnFromResult();
                  setUi((u) => ({ ...u, seed: randomSeed(), carryOverMultiplier: state.nextCarryOverMultiplier || 1 }));
                  setTimeout(
                    () =>
                      startGame({
                        keepCarry: true,
                        keepStarter: true,
                        keepGold: true,
                        firstTurnKey: ui.nextFirstTurnKey
                      }),
                    0
                  );
                }}
              >
                랜덤 새 판
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
