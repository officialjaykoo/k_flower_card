import CardView, { cardBackView } from "./CardView.jsx";

export default function GameBoard({
  state,
  ruleName,
  ui,
  locked,
  participantType,
  onChooseMatch,
  onPlayCard,
  hScore,
  aScore,
  sortCards
}) {
  const renderHand = (playerKey, clickable) => {
    const player = state.players[playerKey];
    const cards = ui.sortHand ? sortCards(player.hand) : player.hand.slice();
    if (playerKey === "ai" && participantType(ui, "ai") === "ai" && !ui.revealAiHand) {
      return cards.map((_, i) => <div key={`b-${i}`}>{cardBackView()}</div>);
    }
    return cards.map((card) => {
      const canSelect = clickable && state.currentTurn === playerKey && state.phase === "playing" && !locked;
      return (
        <CardView
          key={card.id}
          card={card}
          interactive={canSelect}
          onClick={() => canSelect && onPlayCard(card.id)}
        />
      );
    });
  };

  return (
    <div className="board">
      <h2 className="section-title">맞고 데모 ({ruleName || "통합 단일룰"})</h2>
      <div className="meta">
        {state.phase === "kung-choice"
          ? "진행 상태: 쿵 선택"
          : state.phase === "gukjin-choice"
          ? "진행 상태: 국진 선택"
          : state.phase === "president-choice"
          ? "진행 상태: 대통령 선택"
          : state.phase === "select-match"
          ? "진행 상태: 매치 선택"
          : state.phase === "go-stop"
          ? "진행 상태: Go/Stop 선택"
          : state.phase === "resolution"
          ? "진행 상태: 라운드 종료"
          : "진행 상태: 플레이 중"}
      </div>

      <h3 className="section-title">중앙 보드</h3>
      <div className="center-row">
        {state.board.map((card) => {
          const selectable =
            state.phase === "select-match" &&
            state.pendingMatch?.playerKey &&
            participantType(ui, state.pendingMatch.playerKey) === "human" &&
            (state.pendingMatch.boardCardIds || []).includes(card.id) &&
            !locked;
          return (
            <CardView
              key={card.id}
              card={card}
              interactive={selectable}
              onClick={() => selectable && onChooseMatch(card.id)}
            />
          );
        })}
      </div>

      <h3 className="section-title">{state.players.ai.label} 손패</h3>
      <div className="hand">{renderHand("ai", participantType(ui, "ai") === "human")}</div>

      <h3 className="section-title">{state.players.human.label} 손패</h3>
      <div className="hand">{renderHand("human", participantType(ui, "human") === "human")}</div>

      <h3 className="section-title">획득 카드</h3>
      {["kwang", "five", "ribbon", "junk"].map((k) => (
        <div key={k} style={{ marginBottom: 8 }}>
          <strong>{k === "kwang" ? "광" : k === "five" ? "열끗/고도리" : k === "ribbon" ? "띠" : "피"}</strong>
          <span className="tag">{state.players.human.label} {state.players.human.captured[k].length}</span>
          <span className="tag">{state.players.ai.label} {state.players.ai.captured[k].length}</span>
          <div className="captured">
            {state.players.human.captured[k].map((c) => (
              <CardView key={`h-${k}-${c.id}`} card={c} />
            ))}
          </div>
          <div className="captured">
            {state.players.ai.captured[k].map((c) => (
              <CardView key={`a-${k}-${c.id}`} card={c} />
            ))}
          </div>
        </div>
      ))}

      <div className="meta">
        현재 점수 - {state.players.human.label}: {hScore.base} (x{hScore.multiplier}) / {state.players.ai.label}: {aScore.base} (x{aScore.multiplier}) | 골드: {state.players.human.gold} / {state.players.ai.gold}
      </div>
    </div>
  );
}
