import { useEffect, useRef, useState } from "react";
import CardView, { cardBackView } from "./CardView.jsx";
import GameBoardControlsPanel from "./GameBoardControlsPanel.jsx";
import { openCardGuidePopup, openGameLogPopup, openRulesPopup } from "./gameBoardPopups.js";
import { POINT_GOLD_UNIT } from "../../engine/economy.js";
import { isGukjinCard } from "../../engine/scoring.js";
import { buildDeck } from "../../cards.js";

const FIXED_HAND_SLOT_COUNT = 10;

function buildInitialFixedSlots(hand = []) {
  const slots = Array(FIXED_HAND_SLOT_COUNT).fill(null);
  hand.slice(0, FIXED_HAND_SLOT_COUNT).forEach((card, idx) => {
    slots[idx] = card?.id || null;
  });
  return slots;
}

function sameFixedSlots(a = [], b = []) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

function syncFixedHandSlots(prevSlots, hand = []) {
  const slots = Array.isArray(prevSlots)
    ? prevSlots.slice(0, FIXED_HAND_SLOT_COUNT)
    : Array(FIXED_HAND_SLOT_COUNT).fill(null);
  while (slots.length < FIXED_HAND_SLOT_COUNT) slots.push(null);

  const handIds = hand.map((card) => card.id);
  const handSet = new Set(handIds);

  for (let i = 0; i < slots.length; i += 1) {
    const slotId = slots[i];
    if (slotId && !handSet.has(slotId)) slots[i] = null;
  }

  handIds.forEach((id) => {
    if (slots.includes(id)) return;
    const emptyIdx = slots.indexOf(null);
    if (emptyIdx >= 0) slots[emptyIdx] = id;
  });

  return slots;
}

export default function GameBoard({
  state,
  ui,
  setUi,
  locked,
  participantType,
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
  onClearReplay,
  t,
  supportedLanguages = ["ko", "en"]
}) {
  const language = ui.language || "ko";
  const allCards = buildDeck();
  const order = { kwang: 0, five: 1, ribbon: 2, junk: 3 };
  const monthCards = allCards
    .filter((c) => c.month >= 1 && c.month <= 12)
    .sort((a, b) => a.month - b.month || order[a.category] - order[b.category] || a.id.localeCompare(b.id));
  const bonusCards = allCards.filter((c) => c.month === 13);

  const capturedTypes = [{ key: "kwang" }, { key: "five" }, { key: "ribbon" }, { key: "junk" }];
  const CAPTURE_ZONE_WIDTH = {
    normal: 230,
    junk: 430,
    state: 90
  };
  const CAPTURE_CARD_WIDTH = 44;

  const turnRole = (playerKey) =>
    state.startingTurnKey === playerKey ? t("board.turn.first") : t("board.turn.second");
  const boardCardsSorted = sortCards(state.board);

  const formatGold = (value) =>
    Number(value || 0).toLocaleString(language === "en" ? "en-US" : "ko-KR");
  const formatMonthLabel = (month) => (language === "en" ? "M" + month : month + "\uC6D4");

  const [fixedHandSlots, setFixedHandSlots] = useState(() => ({
    human: buildInitialFixedSlots(sortCards(state.players?.human?.hand || [])),
    ai: buildInitialFixedSlots(sortCards(state.players?.ai?.hand || []))
  }));
  const lastFixedHandLayoutNonceRef = useRef(ui.handLayoutNonce || 0);

  useEffect(() => {
    if (!ui.fixedHand) return;
    const handLayoutNonce = ui.handLayoutNonce || 0;
    const isNewGame = handLayoutNonce !== lastFixedHandLayoutNonceRef.current;
    lastFixedHandLayoutNonceRef.current = handLayoutNonce;

    setFixedHandSlots((prev) => {
      if (isNewGame) {
        const initial = {
          human: buildInitialFixedSlots(sortCards(state.players.human.hand || [])),
          ai: buildInitialFixedSlots(sortCards(state.players.ai.hand || []))
        };
        const unchanged =
          sameFixedSlots(prev?.human || [], initial.human) &&
          sameFixedSlots(prev?.ai || [], initial.ai);
        return unchanged ? prev : initial;
      }
      const next = {
        human: syncFixedHandSlots(prev?.human, state.players.human.hand || []),
        ai: syncFixedHandSlots(prev?.ai, state.players.ai.hand || [])
      };
      const unchanged =
        sameFixedSlots(prev?.human || [], next.human) &&
        sameFixedSlots(prev?.ai || [], next.ai);
      return unchanged ? prev : next;
    });
  }, [ui.fixedHand, ui.handLayoutNonce, state.players.human.hand, state.players.ai.hand, sortCards]);

  const renderHand = (playerKey, clickable) => {
    const player = state.players[playerKey];
    const cards = player.hand.slice();
    const cardsById = new Map(player.hand.map((card) => [card.id, card]));
    const boardMonths = new Set(state.board.map((b) => b.month));
    const hideAiCards = playerKey === "ai" && participantType(ui, "ai") === "ai" && !ui.revealAiHand;

    if (ui.fixedHand) {
      const slots = fixedHandSlots[playerKey] || buildInitialFixedSlots([]);
      return slots.map((slotId, idx) => {
        if (!slotId) {
          return <div key={playerKey + "-empty-" + idx} className="hand-card-wrap hand-card-empty-slot" />;
        }
        const card = cardsById.get(slotId);
        if (!card) {
          return <div key={playerKey + "-missing-" + idx} className="hand-card-wrap hand-card-empty-slot" />;
        }
        const canSelect = clickable && state.currentTurn === playerKey && state.phase === "playing" && !locked;
        const monthMatched = playerKey === "human" && !card.passCard && boardMonths.has(card.month);
        return (
          <div key={playerKey + "-slot-" + idx} className={"hand-card-wrap" + (monthMatched ? " month-matched" : "")}>
            {hideAiCards ? (
              cardBackView()
            ) : (
              <CardView
                card={card}
                interactive={canSelect}
                onClick={() => canSelect && onPlayCard(card.id)}
                t={t}
              />
            )}
            {playerKey === "ai" && typeof aiPlayProbMap?.[card.id] === "number" ? (
              <span className="hand-prob-badge">{(aiPlayProbMap[card.id] * 100).toFixed(1) + "%"}</span>
            ) : null}
            {monthMatched ? <span className="hand-month-badge">{t("board.badge.match")}</span> : null}
          </div>
        );
      });
    }

    const normalCards = sortCards(cards.filter((card) => !card.passCard));
    const dummyCards = cards.filter((card) => card.passCard);
    const orderedCards = normalCards.concat(dummyCards);

    if (hideAiCards) {
      return orderedCards.map((_, i) => <div key={"b-" + i}>{cardBackView()}</div>);
    }

    return orderedCards.map((card) => {
      const canSelect = clickable && state.currentTurn === playerKey && state.phase === "playing" && !locked;
      const monthMatched = playerKey === "human" && !card.passCard && boardMonths.has(card.month);
      return (
        <div key={card.id} className={"hand-card-wrap" + (monthMatched ? " month-matched" : "")}>
          <CardView
            card={card}
            interactive={canSelect}
            onClick={() => canSelect && onPlayCard(card.id)}
            t={t}
          />
          {playerKey === "ai" && typeof aiPlayProbMap?.[card.id] === "number" ? (
            <span className="hand-prob-badge">{(aiPlayProbMap[card.id] * 100).toFixed(1) + "%"}</span>
          ) : null}
          {monthMatched ? <span className="hand-month-badge">{t("board.badge.match")}</span> : null}
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
          (state.players[playerKey].captured.five || []).filter((card) => card.category === "five" && isGukjinCard(card)) || [];
        const cards =
          player.gukjinMode === "junk"
            ? key === "five"
              ? baseCards.filter((card) => !(card.category === "five" && isGukjinCard(card)))
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
        const zoneScoreConfig =
          key === "kwang"
            ? { points: scoreInfo?.breakdown?.kwangBase || 0, count, target: 3 }
            : key === "five"
            ? {
                points: (scoreInfo?.breakdown?.fiveBase || 0) + (scoreInfo?.breakdown?.fiveSetBonus || 0),
                count,
                target: 5
              }
            : key === "ribbon"
            ? {
                points: (scoreInfo?.breakdown?.ribbonBase || 0) + (scoreInfo?.breakdown?.ribbonSetBonus || 0),
                count,
                target: 5
              }
            : key === "junk"
            ? {
                points: scoreInfo?.breakdown?.junkBase || 0,
                count: scoreInfo?.breakdown?.piCount || 0,
                target: 10
              }
            : null;
        const zoneScoreLabel = zoneScoreConfig
          ? zoneScoreConfig.points >= 1
            ? t("board.point", { value: zoneScoreConfig.points })
            : `(${zoneScoreConfig.count}/${zoneScoreConfig.target})`
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
                    <CardView card={c} t={t} />
                  </div>
                ))}
              </div>
            </div>
            {zoneScoreLabel ? <span className="capture-zone-score-badge">{zoneScoreLabel}</span> : null}
          </div>
        );
      })}
      <div className="capture-zone capture-zone-state-panel" aria-label={t("board.captureState.aria")}>
        <div className="capture-zone-state">
          <div className="capture-state-row">
            <span className="capture-state-label">GO</span>
            <span className="capture-state-value">{state.players[playerKey].goCount || 0}</span>
          </div>
          <div className="capture-state-row">
            <span className="capture-state-label">{t("board.captureState.shakeBomb")}</span>
            <span className="capture-state-value">
              {(state.players[playerKey].events?.shaking || 0)}/{(state.players[playerKey].events?.bomb || 0)}
            </span>
          </div>
          <div className="capture-state-row">
            <span className="capture-state-label">{t("board.captureState.ppuk")}</span>
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
                      t={t}
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
        <div className="lane-capture-wrap">{renderCapturedPanel(playerKey)}</div>
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
        <div className="status-meta">{t("board.gold")} {formatGold(player.gold)}</div>
      </div>
    );
  };

  const handleOpenCardGuidePopup = () =>
    openCardGuidePopup({ monthCards, bonusCards, language });
  const handleOpenRulesPopup = () => openRulesPopup({ language });
  const handleOpenGameLogPopup = () => openGameLogPopup({ log: state.log || [], language });

  return (
    <div className="board table-theme">
      <div className="table-field">
        <div className="play-column">
          {renderCaptureLane("ai")}
          <section className="center-field">
            <div className="center-main">
              <div className="center-total-score center-total-score-opponent">
                {t("board.point", { value: aScore?.total || 0 })}
              </div>
              <div className="center-total-score-unit">X{POINT_GOLD_UNIT} GOLD</div>
              {(state.carryOverMultiplier || 1) >= 2 ? (
                <div className="center-round-multiplier">
                  {t("board.roundMultiplier", { value: state.carryOverMultiplier })}
                </div>
              ) : null}
              <div className="center-total-score center-total-score-player">
                {t("board.point", { value: hScore?.total || 0 })}
              </div>
              {openingPick?.active ? (
                <div className="opening-pick-wrap">
                  <div className="opening-title">{t("board.opening.title")}</div>
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
                              <CardView card={card} t={t} />
                              {selectedByHuman ? <span className="opening-owner-tag">{t("board.opening.myPick")}</span> : null}
                              {selectedByAi ? <span className="opening-owner-tag">{t("board.opening.opponentPick")}</span> : null}
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
                      <span>
                        {t("board.opening.me")}:{" "}
                        {openingPick.humanCard
                          ? `${formatMonthLabel(openingPick.humanCard.month)} ${openingPick.humanCard.name}`
                          : "-"}
                      </span>
                      <span>
                        {t("board.opening.opponent")}:{" "}
                        {openingPick.aiCard
                          ? `${formatMonthLabel(openingPick.aiCard.month)} ${openingPick.aiCard.name}`
                          : "-"}
                      </span>
                      <span>
                        {t("board.opening.starter")}:{" "}
                        {openingPick.winnerKey === "human"
                          ? state.players.human.label
                          : state.players.ai.label}
                      </span>
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
          <GameBoardControlsPanel
            state={state}
            ui={ui}
            setUi={setUi}
            participantType={participantType}
            onStartSpecifiedGame={onStartSpecifiedGame}
            onStartRandomGame={onStartRandomGame}
            onLoadReplay={onLoadReplay}
            onClearReplay={onClearReplay}
            onOpenRulesPopup={handleOpenRulesPopup}
            onOpenCardGuidePopup={handleOpenCardGuidePopup}
            onOpenGameLogPopup={handleOpenGameLogPopup}
            t={t}
            supportedLanguages={supportedLanguages}
          />
          {renderCenterStatusCard("human")}
        </aside>
      </div>

      {replayModeEnabled && (
        <div className="replay-dock">
          <div className="replay-head">
            <div className="replay-title">{t("board.replay.title")}</div>
            <div className="meta">
              {t("board.replay.source")}: {replaySourceLabel || t("replay.loadedSourceDefault")}
            </div>
          </div>

          {replayFrame ? (
            <>
              <div className="meta">
                {t("board.replay.state")}: {ui.replay.enabled ? t("board.replay.stateOn") : t("board.replay.stateOff")} /{" "}
                {t("board.replay.frame")} {replayIdx + 1} / {replayFrames.length} /{" "}
                {t("board.replay.actor")}: {replayFrame.actor ? replayPlayers[replayFrame.actor].label : "-"}
              </div>
              <div className="meta">
                {t("board.replay.action")}: {formatActionText(replayFrame.action)} | {t("board.replay.event")}:{" "}
                {formatEventsText(replayFrame.events)}
              </div>
              <div className="control-row">
                <button onClick={onReplayToggle}>
                  {ui.replay.enabled ? t("board.replay.btn.stop") : t("board.replay.btn.start")}
                </button>
                <button disabled={!ui.replay.enabled || replayIdx <= 0} onClick={onReplayPrev}>
                  {t("board.replay.btn.prev")}
                </button>
                <button disabled={!ui.replay.enabled || replayIdx >= replayFrames.length - 1} onClick={onReplayNext}>
                  {t("board.replay.btn.next")}
                </button>
                <button disabled={!ui.replay.enabled || replayFrames.length <= 1} onClick={onReplayAutoToggle}>
                  {ui.replay.autoPlay ? t("board.replay.btn.autoStop") : t("board.replay.btn.autoStart")}
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
                <div className="meta">{t("board.replay.interval")}</div>
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
            <div className="meta">{t("board.replay.empty")}</div>
          )}
        </div>
      )}
    </div>
  );
}




