import CardView, { cardBackView } from "./CardView.jsx";
import { packJunkRows } from "../../shared/junkLayout.js";
import { sumPiValues } from "../../engine/scoring.js";

export default function GameBoard({
  state,
  ui,
  locked,
  participantType,
  onChooseMatch,
  onPlayCard,
  hScore,
  aScore,
  sortCards,
  replayModeEnabled,
  replaySourceLabel,
  replayPlayers,
  replayFrames,
  replayIdx,
  replayFrame,
  formatActionText,
  formatEventsText,
  openingPick,
  onOpeningPick,
  onReplayToggle,
  onReplayPrev,
  onReplayNext,
  onReplayAutoToggle,
  onReplaySeek,
  onReplayIntervalChange
}) {
  const myKey = "human";
  const opponentKey = "ai";

  const capturedTypes = [
    { key: "kwang", label: "광" },
    { key: "five", label: "열" },
    { key: "ribbon", label: "띠" },
    { key: "junk", label: "피" }
  ];

  const turnRole = (playerKey) => (state.startingTurnKey === playerKey ? "선" : "후");
  const boardCardsSorted = sortCards(state.board);
  const splitAt = Math.ceil(boardCardsSorted.length / 2);
  const boardLeft = boardCardsSorted.slice(0, splitAt);
  const boardRight = boardCardsSorted.slice(splitAt);

  const playerJunkRows = (playerKey) => packJunkRows(state.players[playerKey].captured.junk || []).length;
  const compactLaneMode = playerJunkRows("human") >= 5 || playerJunkRows("ai") >= 5;
  const formatGold = (value) => Number(value || 0).toLocaleString("ko-KR");

  const countSetCards = (player, category, months) =>
    (player.captured?.[category] || []).filter((c) => months.includes(c.month)).length;

  const getEmergencyAlerts = () => {
    const me = state.players[myKey];
    const opponent = state.players[opponentKey];
    const alerts = [];

    if ((state.carryOverMultiplier || 1) >= 2) {
      alerts.push({
        key: "carry-over-multiplier",
        label: `이번 판 누적배수 X${state.carryOverMultiplier}`
      });
    }

    const oneAwaySets = [
      { key: "godori", label: "고도리 비상", category: "five", months: [2, 4, 8] },
      { key: "cheong", label: "청단 비상", category: "ribbon", months: [6, 9, 10] },
      { key: "hong", label: "홍단 비상", category: "ribbon", months: [1, 2, 3] },
      { key: "cho", label: "초단 비상", category: "ribbon", months: [4, 5, 7] }
    ];

    oneAwaySets.forEach((setInfo) => {
      const myCount = countSetCards(me, setInfo.category, setInfo.months);
      const opponentCount = countSetCards(opponent, setInfo.category, setInfo.months);
      if (myCount === 0 && opponentCount === 2) {
        alerts.push({ key: setInfo.key, label: setInfo.label });
      }
    });

    if ((opponent.goCount || 0) > 0) {
      if (aScore?.bak?.gwang) alerts.push({ key: "bak-gwang", label: "광박 비상" });
      if (aScore?.bak?.mongBak) alerts.push({ key: "bak-mong", label: "멍박 비상" });
      if (aScore?.bak?.pi) alerts.push({ key: "bak-pi", label: "피박 비상" });
    }
    return alerts;
  };

  const emergencyAlerts = getEmergencyAlerts();

  const renderHand = (playerKey, clickable) => {
    const player = state.players[playerKey];
    const cards = sortCards(player.hand);
    const boardMonths = new Set(state.board.map((b) => b.month));
    if (playerKey === "ai" && participantType(ui, "ai") === "ai" && !ui.revealAiHand) {
      return cards.map((_, i) => <div key={`b-${i}`}>{cardBackView()}</div>);
    }

    return cards.map((card) => {
      const canSelect = clickable && state.currentTurn === playerKey && state.phase === "playing" && !locked;
      const monthMatched = playerKey === "human" && !card.passCard && boardMonths.has(card.month);
      return (
        <div key={card.id} className={`hand-card-wrap${monthMatched ? " month-matched" : ""}`}>
          <CardView
            card={card}
            interactive={canSelect}
            onClick={() => canSelect && onPlayCard(card.id)}
          />
          {monthMatched ? <span className="hand-month-badge">M</span> : null}
        </div>
      );
    });
  };

  const renderCapturedPanel = (playerKey) => (
    <div className="capture-panel-grid">
      {capturedTypes.map(({ key }) => {
        const cards = state.players[playerKey].captured[key];
        const seen = new Set();
        const uniqueCards = cards.filter((c) => {
          if (!c || seen.has(c.id)) return false;
          seen.add(c.id);
          return true;
        });
        const rows =
          key === "junk"
            ? packJunkRows(uniqueCards)
            : Array.from({ length: Math.ceil(uniqueCards.length / 5) }, (_, i) => uniqueCards.slice(i * 5, i * 5 + 5));
        const displayCount = key === "junk" ? sumPiValues(uniqueCards) : uniqueCards.length;
        return (
          <div key={`${playerKey}-${key}`} className="capture-zone">
            <span className="capture-zone-count">{displayCount}</span>
            <div
              className={`captured-stack rows-${rows.length}${
                rows.length >= 5 ? " rows-5plus" : rows.length >= 3 ? " rows-ge3" : ""
              }`}
            >
              {rows.map((rowCards, rowIdx) => (
                <div key={`${playerKey}-${key}-row-${rowIdx}`} className="captured-row">
                  {rowCards.map((c) => (
                    <div key={`${playerKey}-${key}-${c.id}`} className="stack-item">
                      <CardView card={c} />
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );

  const renderBoardBank = (cards, side) => {
    const monthMap = new Map();
    cards.forEach((card) => {
      if (!monthMap.has(card.month)) monthMap.set(card.month, []);
      monthMap.get(card.month).push(card);
    });
    const months = Array.from(monthMap.keys()).sort((a, b) => a - b);
    const arcMid = (months.length - 1) / 2;
    return (
      <div className={`board-bank board-bank-${side}`}>
        {months.map((month, idx) => {
          const distance = Math.abs(idx - arcMid);
          const normalized = arcMid > 0 ? distance / arcMid : 0;
          const towardDeck = Math.round((1 - normalized) * 14);
          const vertical = Math.round(distance * 2);
          const xOffset = side === "left" ? towardDeck : -towardDeck;
          return (
            <div
              key={`${side}-${month}`}
              className="month-group"
              style={{
                "--arc-x": `${xOffset}px`,
                "--arc-y": `${vertical}px`
              }}
            >
              {monthMap.get(month).map((card) => {
              const selectable =
                !openingPick?.active &&
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
          );
        })}
      </div>
    );
  };

  const renderPlayerLane = (playerKey) => {
    const laneRoleClass = playerKey === "ai" ? "lane-opponent" : "lane-player";
    return (
      <section className={`lane lane-row ${laneRoleClass}`}>
        <div className="lane-capture-wrap">
          {renderCapturedPanel(playerKey)}
        </div>

        <div className="lane-hand-wrap">
          <div className="hand hand-grid">{renderHand(playerKey, participantType(ui, playerKey) === "human")}</div>
        </div>
      </section>
    );
  };

  const renderCenterStatusCard = (playerKey) => {
    const isActive = state.currentTurn === playerKey && state.phase !== "resolution";
    const player = state.players[playerKey];
    const label = player.label;
    const score = playerKey === "human" ? hScore : aScore;
    const opponentScore = playerKey === "human" ? aScore : hScore;
    const flags = [];
    if ((player.goCount || 0) >= 3) flags.push(`${player.goCount}고`);
    if ((player.events?.shaking || 0) > 0) flags.push(`흔들기 x${2 ** player.events.shaking}`);
    if ((player.events?.bomb || 0) > 0) flags.push(`폭탄 x${2 ** player.events.bomb}`);
    if (opponentScore?.bak?.gwang) flags.push("광박");
    if (opponentScore?.bak?.mongBak) flags.push("멍박");
    if (opponentScore?.bak?.pi) flags.push("피박");
    return (
      <div key={`center-status-${playerKey}`} className={`status-card compact ${isActive ? "turn-active" : ""}`}>
        <div className="status-head">
          <div className="status-name">{label}</div>
          <div className="turn-tag">{turnRole(playerKey)}</div>
        </div>
        <div className="status-meta">점수 {score.base}{score.multiplier > 1 ? ` x${score.multiplier}` : ""}</div>
        <div className="status-meta">골드 {formatGold(player.gold)}</div>
        <div className="status-meta">{isActive ? "내 차례" : "대기"}</div>
        {flags.length > 0 ? <div className="status-flags">{flags.join(" / ")}</div> : null}
      </div>
    );
  };


  return (
    <div className="board table-theme layout-debug-rainbow">
      <div className={`table-field${compactLaneMode ? " lanes-compact5" : ""}`}>
        {renderPlayerLane("ai")}

        <section className="center-field">
          <aside className="center-emergency">
            <div className="emergency-title">비상</div>
            {emergencyAlerts.length > 0 ? (
              <div className="emergency-list">
                {emergencyAlerts.map((item) => (
                  <div key={item.key} className="emergency-item">
                    {item.label}
                  </div>
                ))}
              </div>
            ) : (
              <div className="emergency-empty">안전</div>
            )}
          </aside>

          <div className="center-main">
            {openingPick?.active ? (
              <div className="opening-pick-wrap">
                <div className="opening-title">첫 게임 선/후 결정 - 바닥패 1장 선택</div>
                <div className="opening-grid">
                  {state.board.map((card) => {
                    const selectedByHuman = openingPick.humanCard?.id === card.id;
                    const selectedByAi = openingPick.aiCard?.id === card.id;
                    const reveal = selectedByHuman || selectedByAi;
                    return (
                      <button
                        key={`pick-${card.id}`}
                        className={`opening-slot ${reveal ? "picked-choice" : ""}`}
                        onClick={() => !openingPick.humanCard && onOpeningPick(card.id)}
                        disabled={!!openingPick.humanCard}
                        type="button"
                      >
                        {reveal ? (
                          <>
                            <CardView card={card} />
                            {selectedByHuman ? <span className="opening-owner-tag">내 선택</span> : null}
                            {selectedByAi ? <span className="opening-owner-tag">상대 선택</span> : null}
                          </>
                        ) : (
                          <div className="opening-back">{cardBackView()}</div>
                        )}
                      </button>
                    );
                  })}
                </div>
                {(openingPick.humanCard || openingPick.aiCard) && (
                  <div className="opening-picked-meta">
                    <span>나: {openingPick.humanCard ? `${openingPick.humanCard.month}월 ${openingPick.humanCard.name}` : "-"}</span>
                    <span>상대: {openingPick.aiCard ? `${openingPick.aiCard.month}월 ${openingPick.aiCard.name}` : "-"}</span>
                    <span>선턴: {openingPick.winnerKey === "human" ? state.players.human.label : state.players.ai.label}</span>
                  </div>
                )}
              </div>
            ) : (
              <>
                {renderBoardBank(boardLeft, "left")}
                <div className="deck-stack">
                  <img src="/cards/deck-stack.svg" alt="deck stack" />
                </div>
                {renderBoardBank(boardRight, "right")}
              </>
            )}
          </div>
          <div className="center-side-status">
            {renderCenterStatusCard("ai")}
            {renderCenterStatusCard("human")}
          </div>
        </section>

        {renderPlayerLane("human")}
      </div>

      {replayModeEnabled && (
        <div className="replay-dock">
          <div className="replay-head">
            <div className="replay-title">턴 리플레이</div>
            <div className="meta">소스: {replaySourceLabel || "불러온 기보"}</div>
          </div>

          {replayFrame ? (
            <>
              <div className="meta">
                상태: {ui.replay.enabled ? "켜짐" : "꺼짐"} / 프레임 {replayIdx + 1} / {replayFrames.length} / 행동자: {replayFrame.actor ? replayPlayers[replayFrame.actor].label : "-"}
              </div>
              <div className="meta">행동: {formatActionText(replayFrame.action)} | 이벤트: {formatEventsText(replayFrame.events)}</div>
              <div className="control-row">
                <button onClick={onReplayToggle}>{ui.replay.enabled ? "리플레이 종료" : "리플레이 시작"}</button>
                <button disabled={!ui.replay.enabled || replayIdx <= 0} onClick={onReplayPrev}>이전 턴</button>
                <button disabled={!ui.replay.enabled || replayIdx >= replayFrames.length - 1} onClick={onReplayNext}>다음 턴</button>
                <button disabled={!ui.replay.enabled || replayFrames.length <= 1} onClick={onReplayAutoToggle}>
                  {ui.replay.autoPlay ? "자동재생 정지" : "자동재생"}
                </button>
              </div>
              <input
                type="range"
                min={0}
                max={Math.max(0, replayFrames.length - 1)}
                value={replayIdx}
                disabled={!ui.replay.enabled}
                onChange={(e) => onReplaySeek(e.target.value)}
              />
              <div className="control-row">
                <div className="meta">자동재생 간격</div>
                <select
                  value={ui.replay.intervalMs}
                  disabled={!ui.replay.enabled}
                  onChange={(e) => onReplayIntervalChange(e.target.value)}
                >
                  <option value={500}>0.5s</option>
                  <option value={900}>0.9s</option>
                  <option value={1300}>1.3s</option>
                  <option value={1800}>1.8s</option>
                </select>
              </div>
            </>
          ) : (
            <div className="meta">기보가 없어 리플레이를 시작할 수 없습니다.</div>
          )}
        </div>
      )}
    </div>
  );
}
