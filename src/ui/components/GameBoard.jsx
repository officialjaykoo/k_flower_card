import { useRef } from "react";
import CardView, { cardBackView } from "./CardView.jsx";
import { POINT_GOLD_UNIT } from "../../engine/economy.js";
import { buildDeck } from "../../cards.js";

export default function GameBoard({
  state,
  ui,
  setUi,
  locked,
  participantType,
  startGame,
  onChooseMatch,
  onPlayCard,
  onStartSpecifiedGame,
  onStartRandomGame,
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
  modelOptions,
  aiPlayProbMap,
  openingPick,
  onOpeningPick,
  onReplayToggle,
  onReplayPrev,
  onReplayNext,
  onReplayAutoToggle,
  onReplaySeek,
  onReplayIntervalChange,
  onLoadReplay,
  onClearReplay
}) {
  const allCards = buildDeck();
  const order = { kwang: 0, five: 1, ribbon: 2, junk: 3 };
  const monthCards = allCards
    .filter((c) => c.month >= 1 && c.month <= 12)
    .sort((a, b) => a.month - b.month || order[a.category] - order[b.category] || a.id.localeCompare(b.id));
  const bonusCards = allCards.filter((c) => c.month === 13);
  const replayFileInputRef = useRef(null);

  const capturedTypes = [
    { key: "kwang" },
    { key: "five" },
    { key: "ribbon" },
    { key: "junk" }
  ];
  const CAPTURE_ZONE_WIDTH = {
    normal: 230,
    junk: 440,
    state: 80
  };
  const CAPTURE_CARD_WIDTH = 44;

  const turnRole = (playerKey) => (state.startingTurnKey === playerKey ? "선" : "후");
  const boardCardsSorted = sortCards(state.board);

  const formatGold = (value) => Number(value || 0).toLocaleString("ko-KR");

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
          {playerKey === "ai" && typeof aiPlayProbMap?.[card.id] === "number" ? (
            <span className="hand-prob-badge">{`${(aiPlayProbMap[card.id] * 100).toFixed(1)}%`}</span>
          ) : null}
          {monthMatched ? <span className="hand-month-badge">매치</span> : null}
        </div>
      );
    });
  };

  const renderCapturedPanel = (playerKey) => (
    <div
      className="capture-panel-grid"
      style={{
        "--capture-zone-w": `${CAPTURE_ZONE_WIDTH.normal}px`,
        "--capture-junk-zone-w": `${CAPTURE_ZONE_WIDTH.junk}px`,
        "--capture-state-zone-w": `${CAPTURE_ZONE_WIDTH.state}px`
      }}
    >
      {capturedTypes.map(({ key }) => {
        const player = state.players[playerKey];
        const baseCards = state.players[playerKey].captured[key];
        const gukjinFromFive =
          (state.players[playerKey].captured.five || []).filter((card) => card.month === 9 && card.category === "five") || [];
        const cards =
          player.gukjinMode === "junk"
            ? key === "five"
              ? baseCards.filter((card) => !(card.month === 9 && card.category === "five"))
              : key === "junk"
              ? [...baseCards, ...gukjinFromFive]
              : baseCards
            : baseCards;
        const seen = new Set();
        const uniqueCards = cards.filter((c) => {
          if (!c || seen.has(c.id)) return false;
          seen.add(c.id);
          return true;
        });
        const lineCards = uniqueCards;
        const zoneWidth = key === "junk" ? CAPTURE_ZONE_WIDTH.junk : CAPTURE_ZONE_WIDTH.normal;
        const sideReserve = key === "junk" ? 72 : 16;
        const availableWidth = Math.max(60, zoneWidth - sideReserve);
        const cardWidth = CAPTURE_CARD_WIDTH;
        const count = lineCards.length;
        const step =
          count <= 1
            ? cardWidth
            : Math.max(2, Math.min(cardWidth, Math.floor((availableWidth - cardWidth) / (count - 1))));
        const stackOverlap = Math.max(0, cardWidth - step);
        const scoreInfo = playerKey === "human" ? hScore : aScore;
        const zoneScoreLabel =
          key === "kwang"
            ? `${scoreInfo?.breakdown?.kwangBase || 0}점`
            : key === "five"
            ? `${(scoreInfo?.breakdown?.fiveBase || 0) + (scoreInfo?.breakdown?.fiveSetBonus || 0)}점`
            : key === "ribbon"
            ? `${(scoreInfo?.breakdown?.ribbonBase || 0) + (scoreInfo?.breakdown?.ribbonSetBonus || 0)}점`
            : key === "junk"
            ? (scoreInfo?.breakdown?.piCount || 0) < 10
              ? `(${scoreInfo?.breakdown?.piCount || 0}/10)`
              : `${scoreInfo?.breakdown?.junkBase || 0}점`
            : null;
        return (
          <div key={`${playerKey}-${key}`} className={`capture-zone capture-zone-${key}`}>
            <div className="captured-stack">
              <div
                className="captured-row dynamic-overlap"
                style={{
                  "--stack-overlap": `${stackOverlap}px`
                }}
              >
                {lineCards.map((c) => (
                  <div key={`${playerKey}-${key}-${c.id}`} className="stack-item">
                    <CardView card={c} />
                  </div>
                ))}
              </div>
            </div>
            {zoneScoreLabel ? <span className="capture-zone-score-badge">{zoneScoreLabel}</span> : null}
          </div>
        );
      })}
      <div className="capture-zone capture-zone-state-panel" aria-label="획득피 상태">
        <div className="capture-zone-state">
          <div className="capture-state-row">
            <span className="capture-state-label">GO</span>
            <span className="capture-state-value">{state.players[playerKey].goCount || 0}</span>
          </div>
          <div className="capture-state-row">
            <span className="capture-state-label">흔들</span>
            <span className="capture-state-value">{state.players[playerKey].events?.shaking || 0}</span>
          </div>
          <div className="capture-state-row">
            <span className="capture-state-label">뻑</span>
            <span className="capture-state-value">{state.players[playerKey].events?.ppuk || 0}</span>
          </div>
        </div>
      </div>
    </div>
  );

  const renderBoardOrbit = () => {
    const monthMap = new Map();
    boardCardsSorted.forEach((card) => {
      if (!monthMap.has(card.month)) monthMap.set(card.month, []);
      monthMap.get(card.month).push(card);
    });
    const months = Array.from(monthMap.keys()).sort((a, b) => a - b);

    return (
      <div className="board-orbit">
        {months.map((month, idx) => {
          const total = Math.max(1, months.length);
          const angleDeg = -90 + (360 / total) * idx;
          const angle = (angleDeg * Math.PI) / 180;
          // Long horizontal ellipse around the deck center.
          const radiusX = 170 + total * 16;
          const radiusY = 70 + total * 6;
          const xOffset = Math.round(Math.cos(angle) * radiusX);
          const yOffset = Math.round(Math.sin(angle) * radiusY);

          return (
            <div
              key={`orbit-${month}`}
              className="orbit-month-group"
              style={{
                "--orbit-x": `${xOffset}px`,
                "--orbit-y": `${yOffset}px`
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
                  <div key={card.id} className="orbit-card-wrap">
                    <CardView
                      card={card}
                      interactive={selectable}
                      onClick={() => selectable && onChooseMatch(card.id)}
                    />
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
    );
  };

  const renderCaptureLane = (playerKey) => {
    const laneRoleClass = playerKey === "ai" ? "lane-opponent" : "lane-player player-capture-lane";
    return (
      <section className={`lane-row ${laneRoleClass}`}>
        <div className="lane-capture-wrap">
          {renderCapturedPanel(playerKey)}
        </div>
      </section>
    );
  };

  const renderPlayerHandLane = () => (
    <section className="lane-row player-hand-lane">
      <div className="lane-hand-wrap">
        <div className="hand hand-grid">{renderHand("human", participantType(ui, "human") === "human")}</div>
      </div>
    </section>
  );

  const renderCenterStatusCard = (playerKey) => {
    const isActive = state.currentTurn === playerKey && state.phase !== "resolution";
    const player = state.players[playerKey];
    const isAiSlot = participantType(ui, playerKey) === "ai";
    return (
      <div key={`center-status-${playerKey}`} className={`status-card compact ${isActive ? "turn-active" : ""}`}>
        <div className="status-head">
          <div className="turn-tag">{turnRole(playerKey)}</div>
        </div>
        <div className="status-name">
          {isAiSlot ? (
            <select
              value={ui.modelPicks?.[playerKey] || (playerKey === "human" ? "sendol" : "dolbaram")}
              onChange={(e) =>
                setUi((u) => ({
                  ...u,
                  modelPicks: { ...(u.modelPicks || {}), [playerKey]: e.target.value }
                }))
              }
            >
              {(modelOptions || []).map((opt) => (
                <option key={`status-model-${playerKey}-${opt.value}`} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          ) : (
            player.label
          )}
        </div>
        <div className="status-meta">골드 {formatGold(player.gold)}</div>
      </div>
    );
  };

  const openCardGuidePopup = () => {
    const popup = window.open("", "kflower-card-guide", "width=980,height=760,resizable=yes,scrollbars=yes");
    if (!popup) {
      window.alert("팝업이 차단되었습니다. 팝업 허용 후 다시 시도해 주세요.");
      return;
    }

    const renderCards = (cards, labelFn) =>
      cards
        .map(
          (card) => `
            <div class="item">
              <img src="${card.asset}" alt="${card.name}" />
              <div class="label">${labelFn(card)}</div>
            </div>
          `
        )
        .join("");

    const extraCards = [
      { id: "guide-pass", name: "Dummy Pass", asset: "/cards/pass.svg", label: "더미패" },
      { id: "guide-back", name: "Back", asset: "/cards/back.svg", label: "카드 뒷면" },
      { id: "guide-deck", name: "Deck", asset: "/cards/deck-stack.svg", label: "덱" }
    ];

    popup.document.write(`
      <!doctype html>
      <html lang="ko">
        <head>
          <meta charset="UTF-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1.0" />
          <title>패보기</title>
          <style>
            body { margin: 0; padding: 16px; font-family: "Noto Sans KR", sans-serif; background: #143d16; color: #ecffd6; }
            h1 { margin: 0 0 10px; font-size: 22px; }
            h2 { margin: 14px 0 8px; font-size: 16px; }
            .meta { font-size: 12px; color: #d5f6bf; margin-bottom: 8px; }
            .grid { display: grid; gap: 8px; }
            .grid-main { grid-template-columns: repeat(auto-fill, minmax(84px, 1fr)); }
            .grid-small { grid-template-columns: repeat(auto-fill, minmax(84px, 1fr)); }
            .item { display: grid; justify-items: center; gap: 4px; background: rgba(12, 62, 19, 0.45); border: 1px solid rgba(220, 255, 183, 0.25); border-radius: 8px; padding: 6px; }
            .item img { width: 54px; height: 78px; object-fit: contain; }
            .label { font-size: 11px; color: #e8ffd2; text-align: center; }
          </style>
        </head>
        <body>
          <h1>패보기</h1>
          <div class="meta">1~12월 기본패와 보너스/상태 카드를 확인할 수 있습니다.</div>
          <h2>1~12월 기본패</h2>
          <div class="grid grid-main">${renderCards(monthCards, (card) => `${card.month}월 ${card.category}`)}</div>
          <h2>특수/상태 카드</h2>
          <div class="grid grid-small">
            ${renderCards(bonusCards, () => "보너스패")}
            ${extraCards
              .map(
                (card) => `
                  <div class="item">
                    <img src="${card.asset}" alt="${card.name}" />
                    <div class="label">${card.label}</div>
                  </div>
                `
              )
              .join("")}
          </div>
        </body>
      </html>
    `);
    popup.document.close();
    popup.focus();
  };

  const openRulesPopup = () => {
    const popup = window.open("./rules/index.html", "kflower-rules", "width=980,height=760,resizable=yes,scrollbars=yes");
    if (!popup) {
      window.alert("팝업이 차단되었습니다. 팝업 허용 후 다시 시도해 주세요.");
      return;
    }
    popup.focus();
  };

  const escapeHtml = (text) =>
    String(text ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");

  const openGameLogPopup = () => {
    const popup = window.open("", "kflower-game-log", "width=980,height=760,resizable=yes,scrollbars=yes");
    if (!popup) {
      window.alert("팝업이 차단되었습니다. 팝업 허용 후 다시 시도해 주세요.");
      return;
    }
    const hiddenLogPatterns = ["게임 시작 - 룰", "선공 후보:", "매치 캡처", "카드 선택"];
    const visibleLogs = (state.log || []).filter((line) => !hiddenLogPatterns.some((pattern) => String(line).includes(pattern)));
    const logsHtml =
      visibleLogs.length > 0
        ? visibleLogs.map((line) => `<div class="log-line">${escapeHtml(line)}</div>`).join("")
        : '<div class="empty">표시할 로그가 없습니다.</div>';
    popup.document.write(`
      <!doctype html>
      <html lang="ko">
        <head>
          <meta charset="UTF-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1.0" />
          <title>게임로그</title>
          <style>
            body { margin: 0; padding: 16px; font-family: "Noto Sans KR", sans-serif; background: #0f2f12; color: #eaffd5; }
            h1 { margin: 0 0 10px; font-size: 22px; }
            .meta { font-size: 12px; color: #bee4b3; margin-bottom: 10px; }
            .log-wrap { border: 1px solid rgba(141, 218, 255, 0.55); border-radius: 8px; background: rgba(5, 22, 7, 0.72); padding: 8px; }
            .log-line { padding: 4px 6px; border-bottom: 1px solid rgba(130, 180, 138, 0.26); font-size: 13px; line-height: 1.35; white-space: pre-wrap; word-break: break-word; }
            .log-line:last-child { border-bottom: 0; }
            .empty { padding: 8px; color: #b8c9b4; font-size: 13px; }
          </style>
        </head>
        <body>
          <h1>게임로그</h1>
          <div class="meta">현재 판 로그 ${visibleLogs.length}줄</div>
          <div class="log-wrap">${logsHtml}</div>
        </body>
      </html>
    `);
    popup.document.close();
    popup.focus();
  };

  const extractReplayEntry = (parsed) => {
    if (Array.isArray(parsed)) {
      for (let i = parsed.length - 1; i >= 0; i -= 1) {
        const item = parsed[i];
        if (item && Array.isArray(item.kibo)) return item;
      }
      return null;
    }
    if (parsed && Array.isArray(parsed.kibo)) return parsed;
    return null;
  };

  const loadReplayFromFile = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = JSON.parse(String(reader.result || ""));
        const entry = extractReplayEntry(parsed);
        if (!entry) {
          window.alert("기보 형식을 찾지 못했습니다. kibo 배열이 있는 JSON 파일을 선택해 주세요.");
          return;
        }
        onLoadReplay(entry, file.name);
      } catch {
        window.alert("JSON 파싱에 실패했습니다. 파일 내용을 확인해 주세요.");
      } finally {
        if (replayFileInputRef.current) replayFileInputRef.current.value = "";
      }
    };
    reader.readAsText(file, "utf-8");
  };

  const exportKiboJson = () => {
    const snapshot = [
      {
        recordedAt: new Date().toISOString(),
        seed: ui.seed,
        ruleKey: state.ruleKey,
        participants: { ...ui.participants },
        winner: state.result?.winner || null,
        nagari: state.result?.nagari || false,
        nagariReasons: state.result?.nagariReasons || [],
        totals: {
          human: state.result?.human?.total ?? null,
          ai: state.result?.ai?.total ?? null
        },
        startingTurnKey: state.startingTurnKey,
        endingTurnKey: state.currentTurn,
        log: state.log.slice(),
        kibo: state.kibo || []
      }
    ];
    const blob = new Blob([JSON.stringify(snapshot, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    const ts = new Date().toISOString().replace(/[:.]/g, "-");
    a.href = URL.createObjectURL(blob);
    a.download = `kibo-${ts}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(a.href);
  };

  const resetKiboLogs = () => {
    onClearReplay();
    window.alert("브라우저 저장은 사용하지 않습니다. 현재 불러온 기보만 초기화했습니다.");
  };

  const renderRightControlsPanel = () => (
    <div className="panel controls right-controls-panel">
      <input
        ref={replayFileInputRef}
        type="file"
        accept=".json,application/json"
        style={{ display: "none" }}
        onChange={loadReplayFromFile}
      />
      <div className="controls-section">
        <div className="control-row">
          <button onClick={openRulesPopup}>룰페이지 열기</button>
          <button onClick={openCardGuidePopup}>패보기</button>
          <button onClick={openGameLogPopup}>게임로그</button>
        </div>
      </div>
      <div className="controls-section">
        <select
          value={
            participantType(ui, "human") === "human" && participantType(ui, "ai") === "ai"
              ? "hva"
              : participantType(ui, "human") === "human" && participantType(ui, "ai") === "human"
              ? "hvh"
              : "ava"
          }
          onChange={(e) => {
            const value = e.target.value;
            setUi((u) => ({
              ...u,
              participants:
                value === "hvh"
                  ? { human: "human", ai: "human" }
                  : value === "ava"
                  ? { human: "ai", ai: "ai" }
                  : { human: "human", ai: "ai" }
            }));
            setTimeout(() => startGame(), 0);
          }}
        >
          <option value="hva">사람 vs AI</option>
          <option value="hvh">사람 vs 사람</option>
          <option value="ava">AI vs AI</option>
        </select>
      </div>
      <div className="controls-section">
        <div className="control-row seed-row">
          <input
            className="seed-input"
            value={ui.seed}
            onChange={(e) => setUi((u) => ({ ...u, seed: e.target.value }))}
          />
          <button className="btn-start-specified" onClick={onStartSpecifiedGame}>지정 게임시작</button>
        </div>
        <div className="control-row">
          <button onClick={onStartRandomGame}>새 게임시작</button>
        </div>
      </div>
      <div className="controls-section">
        <div className="control-row">
          <button onClick={() => replayFileInputRef.current?.click()}>기보 불러오기</button>
          <button onClick={exportKiboJson}>기보 내보내기</button>
          <button onClick={resetKiboLogs}>기보 초기화</button>
        </div>
      </div>
      <div className="controls-section">
        <div className="toggle-list">
          <label className="toggle-item">
            <span>진행 속도</span>
            <select
              value={ui.speedMode || "fast"}
              onChange={(e) => setUi((u) => ({ ...u, speedMode: e.target.value }))}
            >
              <option value="fast">Fast (즉시)</option>
              <option value="visual">Visual (턴 애니메이션)</option>
            </select>
          </label>
          <label className="toggle-item">
            <span>Visual 지연</span>
            <select
              value={ui.visualDelayMs || 400}
              onChange={(e) => setUi((u) => ({ ...u, visualDelayMs: Number(e.target.value) }))}
              disabled={(ui.speedMode || "fast") !== "visual"}
            >
              <option value={150}>0.15s</option>
              <option value={300}>0.30s</option>
              <option value={400}>0.40s</option>
              <option value={700}>0.70s</option>
              <option value={1000}>1.00s</option>
            </select>
          </label>
          <label className="toggle-item">
            <input
              type="checkbox"
              checked={ui.revealAiHand}
              onChange={(e) => setUi((u) => ({ ...u, revealAiHand: e.target.checked }))}
            />
            <span>AI 패 공개</span>
          </label>
          <label className="toggle-item">
            <input
              type="checkbox"
              checked={ui.sortHand}
              onChange={(e) => setUi((u) => ({ ...u, sortHand: e.target.checked }))}
            />
            <span>패 정렬</span>
          </label>
        </div>
      </div>
    </div>
  );


  return (
    <div className="board table-theme">
      <div className="table-field">
        <div className="play-column">
          {renderCaptureLane("ai")}
          <section className="center-field">
            <div className="center-main">
              <div className="center-total-score center-total-score-opponent">{aScore?.total || 0}점</div>
              <div className="center-total-score-unit">X{POINT_GOLD_UNIT} GOLD</div>
              {(state.carryOverMultiplier || 1) >= 2 ? (
                <div className="center-round-multiplier">이번 판 X{state.carryOverMultiplier}</div>
              ) : null}
              <div className="center-total-score center-total-score-player">{hScore?.total || 0}점</div>
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
                          className={`opening-slot ${reveal ? "picked-choice" : ""}${
                            selectedByHuman ? " picked-human" : ""
                          }${selectedByAi ? " picked-ai" : ""}`}
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
                  <div className="center-orbit-field">
                    {renderBoardOrbit()}
                    <div className="deck-stack deck-stack-center">
                      <img src="/cards/deck-stack.svg" alt="deck stack" />
                    </div>
                  </div>
                  <div className="center-opponent-hand">
                    <div className="lane-hand-wrap">
                      <div className="hand hand-grid">{renderHand("ai", false)}</div>
                    </div>
                  </div>
                </>
              )}
            </div>
          </section>
          {renderCaptureLane("human")}
          {renderPlayerHandLane()}
        </div>
        <aside className="side-column">
          {renderCenterStatusCard("ai")}
          {renderRightControlsPanel()}
          {renderCenterStatusCard("human")}
        </aside>
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






