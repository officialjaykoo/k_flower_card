import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import CardView, { cardBackView } from "./CardView.jsx";
import GameBoardControlsPanel from "./GameBoardControlsPanel.jsx";
import { openCardGuidePopup, openGameLogPopup, openRulesPopup } from "./gameBoardPopups.js";
import { POINT_GOLD_UNIT } from "../../engine/economy.js";
import { isGukjinCard } from "../../engine/scoring.js";
import { buildDeck, buildCardUiAssetPath, DEFAULT_CARD_THEME } from "../../cards.js";
import { DEFAULT_BOT_POLICY } from "../../ai/policies.js";

/* ============================================================================
 * Main board view
 * - center orbit/hand/capture rendering
 * - controls panel composition
 * - replay dock rendering
 * ========================================================================== */

const FIXED_HAND_SLOT_COUNT = 10;
const CARD_CATEGORY_ORDER = Object.freeze({ kwang: 0, five: 1, ribbon: 2, junk: 3 });
const CAPTURED_TYPES = Object.freeze([{ key: "kwang" }, { key: "five" }, { key: "ribbon" }, { key: "junk" }]);
const CAPTURE_ZONE_WIDTH = Object.freeze({
  normal: 230,
  junk: 430,
  state: 90
});
const CAPTURE_CARD_WIDTH = 44;
const MATCH_STACK_OFFSET = Object.freeze({ x: -20, y: -16 });

/* 1) Fixed-hand slot helpers */
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

function uniqueCardsById(cards = []) {
  const seen = new Set();
  const result = [];
  cards.forEach((card) => {
    if (!card?.id || seen.has(card.id)) return;
    seen.add(card.id);
    result.push(card);
  });
  return result;
}

function centerOfRect(rect) {
  return {
    x: rect.left + rect.width / 2,
    y: rect.top + rect.height / 2
  };
}

function computeMonthAnchor(month, centerRect, isEmpty = false, laneOffset = 0) {
  const m = Number(month);
  if (!Number.isFinite(m) || m < 1 || m > 12) {
    return {
      x: centerRect.left + centerRect.width / 2 + laneOffset * 18,
      y: centerRect.top + centerRect.height / 2 + (isEmpty ? 36 : 0)
    };
  }
  const idx = m - 1;
  const angle = ((-90 + idx * 30) * Math.PI) / 180;
  const radiusX = Math.max(120, Math.min(centerRect.width * 0.33, 360));
  const radiusY = Math.max(70, Math.min(centerRect.height * 0.24, 180));
  return {
    x: centerRect.left + centerRect.width / 2 + Math.cos(angle) * radiusX + laneOffset * 16,
    y: centerRect.top + centerRect.height / 2 + Math.sin(angle) * radiusY + (isEmpty ? 26 : 0)
  };
}

function captureBoardSnapshot(boardRoot, centerRoot) {
  if (!boardRoot || !centerRoot) return null;
  const centerRect = centerRoot.getBoundingClientRect();
  const hand = { human: new Map(), ai: new Map() };
  const boardById = new Map();
  const boardByMonth = new Map();
  const handAnchor = { human: null, ai: null };
  const captureZoneByPlayerCategory = new Map();
  const capturePanelCenter = new Map();
  let deckCenter = null;

  const readCenter = (el) => centerOfRect(el.getBoundingClientRect());

  boardRoot.querySelectorAll('[data-zone="hand-human"][data-card-id]').forEach((el) => {
    hand.human.set(String(el.dataset.cardId), readCenter(el));
  });
  boardRoot.querySelectorAll('[data-zone="hand-ai"][data-card-id]').forEach((el) => {
    hand.ai.set(String(el.dataset.cardId), readCenter(el));
  });
  boardRoot.querySelectorAll('[data-zone="board-card"][data-card-id]').forEach((el) => {
    const id = String(el.dataset.cardId || "");
    const month = Number(el.dataset.month || 0);
    const point = readCenter(el);
    if (id) boardById.set(id, point);
    if (Number.isFinite(month)) {
      if (!boardByMonth.has(month)) boardByMonth.set(month, []);
      boardByMonth.get(month).push(point);
    }
  });

  const humanLane = boardRoot.querySelector(".player-hand-lane .hand-grid");
  if (humanLane) handAnchor.human = readCenter(humanLane);
  const aiLane = boardRoot.querySelector(".center-opponent-hand .hand-grid");
  if (aiLane) handAnchor.ai = readCenter(aiLane);
  boardRoot.querySelectorAll('[data-zone="capture-panel"][data-player-key]').forEach((el) => {
    const playerKey = String(el.dataset.playerKey || "");
    if (!playerKey) return;
    capturePanelCenter.set(playerKey, readCenter(el));
  });
  boardRoot
    .querySelectorAll('[data-zone="capture-zone"][data-player-key][data-capture-key]')
    .forEach((el) => {
      const playerKey = String(el.dataset.playerKey || "");
      const captureKey = String(el.dataset.captureKey || "");
      if (!playerKey || !captureKey) return;
      captureZoneByPlayerCategory.set(`${playerKey}:${captureKey}`, readCenter(el));
    });
  const deckEl = boardRoot.querySelector('[data-zone="deck-center"]');
  if (deckEl) deckCenter = readCenter(deckEl);

  return {
    centerRect,
    hand,
    boardById,
    boardByMonth,
    handAnchor,
    captureZoneByPlayerCategory,
    capturePanelCenter,
    deckCenter
  };
}

function captureKeyForCard(card) {
  if (!card) return "junk";
  const category = card.category || "junk";
  if (category === "kwang") return "kwang";
  if (category === "five") return "five";
  if (category === "ribbon") return "ribbon";
  return "junk";
}

function speedFxSpec(speedMode) {
  if (speedMode === "visual") {
    return {
      playDuration: 860,
      flipDuration: 820,
      landingDuration: 760,
      pauseAfterPlayHit: 980,
      pauseAfterPlayMiss: 680,
      pauseAfterFlip: 580,
      betweenFlips: 260,
      collectDuration: 680,
      collectStagger: 0,
      settleHold: 620
    };
  }
  return {
    playDuration: 520,
    flipDuration: 480,
    landingDuration: 460,
    pauseAfterPlayHit: 560,
    pauseAfterPlayMiss: 360,
    pauseAfterFlip: 340,
    betweenFlips: 140,
    collectDuration: 420,
    collectStagger: 0,
    settleHold: 380
  };
}

export default function GameBoard({
  state,
  ui,
  setUi,
  locked,
  onTurnAnimationBusyChange,
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
  /* 2) Derived board/runtime values */
  const language = ui.language || "ko";
  const cardTheme = ui.cardTheme || DEFAULT_CARD_THEME;
  const allCards = buildDeck(cardTheme);
  const monthCards = allCards
    .filter((c) => c.month >= 1 && c.month <= 12)
    .sort(
      (a, b) =>
        a.month - b.month ||
        CARD_CATEGORY_ORDER[a.category] - CARD_CATEGORY_ORDER[b.category] ||
        a.id.localeCompare(b.id)
    );
  const bonusCards = allCards.filter((c) => c.month === 13);

  const turnRole = (playerKey) =>
    state.startingTurnKey === playerKey ? t("board.turn.first") : t("board.turn.second");
  const boardRealCardIdSet = useMemo(() => new Set((state.board || []).map((card) => card.id)), [state.board]);
  const latestTurnEntry = useMemo(() => {
    const kibo = state.kibo || [];
    for (let i = kibo.length - 1; i >= 0; i -= 1) {
      if (kibo[i]?.type === "turn_end") return kibo[i];
    }
    return null;
  }, [state.kibo]);
  const latestTurnSummary = useMemo(() => {
    const action = latestTurnEntry?.action || null;
    const playedCard = action?.card || null;
    const flips = uniqueCardsById(action?.flips || []);
    const flipIds = new Set(flips.map((card) => card.id));
    const captureBySource = action?.captureBySource || {};
    const handCaptures = uniqueCardsById(captureBySource.hand || []);
    const flipCaptures = uniqueCardsById(captureBySource.flip || []);
    const handMatches = handCaptures.filter((card) => card.id !== playedCard?.id);
    const flipMatches = flipCaptures.filter((card) => !flipIds.has(card.id));
    const captureSourceById = new Map();
    handCaptures.forEach((card) => {
      if (card?.id) captureSourceById.set(card.id, "hand");
    });
    flipCaptures.forEach((card) => {
      if (card?.id) captureSourceById.set(card.id, "flip");
    });
    return {
      actor: latestTurnEntry?.actor || null,
      turnNo: latestTurnEntry?.turnNo || 0,
      action,
      playedCard,
      flips,
      handMatches,
      flipMatches,
      captureSourceById
    };
  }, [latestTurnEntry]);
  const [traceAnimating, setTraceAnimating] = useState(false);
  const lastTraceTurnNoRef = useRef(0);
  const lastFxTurnNoRef = useRef(0);
  const lastFxModeRef = useRef(null);
  const boardRootRef = useRef(null);
  const centerMainRef = useRef(null);
  const snapshotPrevRef = useRef(null);
  const snapshotCurrentRef = useRef(null);
  const fxClearTimerRef = useRef(null);
  const fxRunTokenRef = useRef(0);
  const [flightFxItems, setFlightFxItems] = useState([]);
  const [landingFxItems, setLandingFxItems] = useState([]);
  const [collectFxItems, setCollectFxItems] = useState([]);
  const [hiddenCapturedCardIds, setHiddenCapturedCardIds] = useState([]);
  const hiddenCapturedIdSet = useMemo(() => new Set(hiddenCapturedCardIds), [hiddenCapturedCardIds]);
  const [hiddenBoardCardIds, setHiddenBoardCardIds] = useState([]);
  const hiddenBoardCardIdSet = useMemo(() => new Set(hiddenBoardCardIds), [hiddenBoardCardIds]);
  const boardBeforeTurnRef = useRef(state.board || []);
  const [boardFrozenCards, setBoardFrozenCards] = useState(null);
  const boardCardsRenderSorted = useMemo(() => {
    const source = Array.isArray(boardFrozenCards) ? boardFrozenCards : state.board || [];
    return sortCards(source);
  }, [state.board, boardFrozenCards, sortCards]);

  useEffect(
    () => () => {
      if (fxClearTimerRef.current) clearTimeout(fxClearTimerRef.current);
      fxRunTokenRef.current += 1;
    },
    []
  );

  const turnFxBusy =
    flightFxItems.length > 0 ||
    landingFxItems.length > 0 ||
    collectFxItems.length > 0 ||
    hiddenCapturedCardIds.length > 0 ||
    hiddenBoardCardIds.length > 0 ||
    Array.isArray(boardFrozenCards);

  useEffect(() => {
    if (typeof onTurnAnimationBusyChange !== "function") return;
    onTurnAnimationBusyChange(turnFxBusy);
  }, [turnFxBusy, onTurnAnimationBusyChange]);

  const formatGold = (value) =>
    Number(value || 0).toLocaleString(language === "en" ? "en-US" : "ko-KR");
  const formatMonthLabel = (month) => (language === "en" ? "M" + month : month + "\uC6D4");

  const [fixedHandSlots, setFixedHandSlots] = useState(() => ({
    human: buildInitialFixedSlots(sortCards(state.players?.human?.hand || [])),
    ai: buildInitialFixedSlots(sortCards(state.players?.ai?.hand || []))
  }));
  const lastFixedHandLayoutNonceRef = useRef(ui.handLayoutNonce || 0);

  // Keep fixed slots stable across turns, but rebuild on new-game nonce.
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

  useEffect(() => {
    const turnNo = latestTurnSummary.turnNo || 0;
    if (!turnNo) return;
    if (turnNo === lastTraceTurnNoRef.current) return;
    lastTraceTurnNoRef.current = turnNo;
    setTraceAnimating(true);
    const timeoutMs = (ui.speedMode || "visual") === "visual" ? 2100 : 1200;
    const timer = setTimeout(() => setTraceAnimating(false), timeoutMs);
    return () => clearTimeout(timer);
  }, [latestTurnSummary.turnNo, ui.speedMode]);

  useLayoutEffect(() => {
    const nextSnapshot = captureBoardSnapshot(boardRootRef.current, centerMainRef.current);
    if (!nextSnapshot) return;
    snapshotPrevRef.current = snapshotCurrentRef.current;
    snapshotCurrentRef.current = nextSnapshot;
  });

  useEffect(() => {
    const turnNo = latestTurnSummary.turnNo || 0;
    if (!turnNo) return;
    const speedMode = ui.speedMode || "visual";
    if (turnNo === lastFxTurnNoRef.current && speedMode === lastFxModeRef.current) return;
    lastFxTurnNoRef.current = turnNo;
    lastFxModeRef.current = speedMode;

    fxRunTokenRef.current += 1;
    const runToken = fxRunTokenRef.current;

    if (fxClearTimerRef.current) {
      clearTimeout(fxClearTimerRef.current);
      fxClearTimerRef.current = null;
    }

    const clearTransientFx = () => {
      setFlightFxItems([]);
      setLandingFxItems([]);
      setCollectFxItems([]);
    };
    const clearTurnFxState = () => {
      clearTransientFx();
      setHiddenCapturedCardIds([]);
      setHiddenBoardCardIds([]);
      setBoardFrozenCards(null);
    };
    if (speedMode !== "visual") {
      clearTurnFxState();
      return;
    }

    const prevSnapshot = snapshotPrevRef.current;
    const currentSnapshot = snapshotCurrentRef.current;
    const centerRect = currentSnapshot?.centerRect || prevSnapshot?.centerRect || null;
    if (!centerRect) return;

    const isRunActive = () => runToken === fxRunTokenRef.current;
    const spec = speedFxSpec(speedMode);
    const toLocalPoint = (point) => ({
      x: Math.round(point.x - centerRect.left),
      y: Math.round(point.y - centerRect.top)
    });
    const action = latestTurnSummary.action || null;
    const actionType = action?.type || "";
    const actorKey = latestTurnSummary.actor === "ai" ? "ai" : "human";
    const handCaptureCards = uniqueCardsById(action?.captureBySource?.hand || []);
    const flipCaptureCards = uniqueCardsById(action?.captureBySource?.flip || []);
    const flipMatchEvents = (action?.matchEvents || []).filter((evt) => evt?.source === "flip");
    const ppukTriggered = flipMatchEvents.some((evt) => evt?.eventTag === "PPUK");

    const resolveSourcePoint = (entry) => {
      if (entry.kind === "play") {
        const actor = latestTurnSummary.actor === "ai" ? "ai" : "human";
        const fromHand =
          prevSnapshot?.hand?.[actor]?.get(entry.card.id) ||
          currentSnapshot?.hand?.[actor]?.get(entry.card.id) ||
          prevSnapshot?.handAnchor?.[actor] ||
          currentSnapshot?.handAnchor?.[actor];
        return fromHand || centerOfRect(centerRect);
      }
      const deckPoint =
        prevSnapshot?.deckCenter ||
        currentSnapshot?.deckCenter ||
        computeMonthAnchor(entry.card.month, centerRect, false, 0);
      return deckPoint;
    };

    const resolveTargetPoint = (entry, order) => {
      if (entry.targetCardId) {
        const point =
          prevSnapshot?.boardById?.get(entry.targetCardId) ||
          currentSnapshot?.boardById?.get(entry.targetCardId);
        if (point) return point;
      }
      const monthList =
        prevSnapshot?.boardByMonth?.get(entry.month) ||
        currentSnapshot?.boardByMonth?.get(entry.month) ||
        null;
      if (monthList?.length) return monthList[0];
      return computeMonthAnchor(entry.month, centerRect, entry.targetType === "empty", order);
    };

    const resolveCapturePoint = (card, order = 0) => {
      const captureKey = captureKeyForCard(card);
      const mapped =
        prevSnapshot?.captureZoneByPlayerCategory?.get(`${actorKey}:${captureKey}`) ||
        currentSnapshot?.captureZoneByPlayerCategory?.get(`${actorKey}:${captureKey}`);
      if (mapped) return mapped;
      const panelCenter =
        prevSnapshot?.capturePanelCenter?.get(actorKey) || currentSnapshot?.capturePanelCenter?.get(actorKey);
      if (panelCenter) return panelCenter;
      return computeMonthAnchor(card?.month, centerRect, true, order);
    };

    const plans = [];
    if (actionType === "play" && latestTurnSummary.playedCard) {
      const isHit = latestTurnSummary.handMatches.length > 0;
      plans.push({
        kind: "play",
        card: latestTurnSummary.playedCard,
        month: latestTurnSummary.playedCard.month,
        targetType: isHit ? "hit" : "empty",
        targetCardId: latestTurnSummary.handMatches[0]?.id || null
      });
    }
    latestTurnSummary.flips.forEach((flipCard, idx) => {
      const evtType = flipMatchEvents[idx]?.type || null;
      const byEvent = evtType ? evtType !== "NONE" : null;
      const byMonth = latestTurnSummary.flipMatches.some((card) => card.month === flipCard.month);
      const isHit = byEvent == null ? byMonth : byEvent;
      const sameMonthMatches = latestTurnSummary.flipMatches.filter((card) => card.month === flipCard.month);
      const matchedCard = sameMonthMatches[0] || null;
      plans.push({
        kind: "flip",
        card: flipCard,
        month: flipCard.month,
        targetType: isHit ? "hit" : "empty",
        targetCardId: isHit ? matchedCard?.id || null : null
      });
    });

    if (!plans.length) {
      clearTurnFxState();
      return;
    }

    const capturedCardIdsAll = uniqueCardsById(handCaptureCards.concat(flipCaptureCards)).map((card) => card.id);
    clearTransientFx();
    setHiddenCapturedCardIds(capturedCardIdsAll);
    setHiddenBoardCardIds([]);
    setBoardFrozenCards((boardBeforeTurnRef.current || []).slice());

    const impactPointByCardId = new Map();
    const queue = [];
    const pushQueueStep = (apply, waitMs = 0) => {
      queue.push({
        apply,
        waitMs: Math.max(0, Math.round(Number(waitMs) || 0))
      });
    };

    plans.forEach((entry, idx) => {
      const fromAbs = resolveSourcePoint(entry);
      const targetAbs = resolveTargetPoint(entry, idx);
      const toAbs =
        entry.targetType === "hit"
          ? { x: targetAbs.x + MATCH_STACK_OFFSET.x, y: targetAbs.y + MATCH_STACK_OFFSET.y }
          : targetAbs;
      const duration = entry.kind === "play" ? spec.playDuration : spec.flipDuration;
      const landingDelay = Math.max(0, duration - 140);
      const stepEndMs = Math.max(duration, landingDelay + spec.landingDuration);
      impactPointByCardId.set(entry.card.id, toAbs);
      const flightItem = {
        id: `fx-flight-${turnNo}-${entry.kind}-${entry.card.id}-${idx}`,
        kind: entry.kind,
        card: entry.card,
        targetType: entry.targetType,
        holdUntilCollect: entry.kind === "play" && entry.targetType === "hit",
        tiltDeg: entry.targetType === "hit" ? -9 : 0,
        duration,
        delay: 0,
        from: toLocalPoint(fromAbs),
        delta: {
          x: Math.round(toAbs.x - fromAbs.x),
          y: Math.round(toAbs.y - fromAbs.y)
        }
      };
      const landingItem = {
        id: `fx-landing-${turnNo}-${entry.card.id}-${idx}`,
        targetType: entry.targetType,
        duration: spec.landingDuration,
        delay: landingDelay,
        point: toLocalPoint(toAbs)
      };
      pushQueueStep(() => {
        if (!isRunActive()) return;
        setFlightFxItems((prevItems) => {
          const held = Array.isArray(prevItems)
            ? prevItems.filter((item) => item?.holdUntilCollect)
            : [];
          return held.concat(flightItem);
        });
        setLandingFxItems([landingItem]);
      }, stepEndMs);
      const pauseMs =
        idx === plans.length - 1
          ? spec.pauseAfterFlip
          : entry.kind === "play"
          ? entry.targetType === "hit"
            ? spec.pauseAfterPlayHit
            : spec.pauseAfterPlayMiss
          : spec.betweenFlips;
      pushQueueStep(() => {
        if (!isRunActive()) return;
        setFlightFxItems((prevItems) =>
          Array.isArray(prevItems) ? prevItems.filter((item) => item?.holdUntilCollect) : []
        );
        setLandingFxItems([]);
        if (entry.kind === "play" && entry.targetType !== "hit") {
          setBoardFrozenCards((prevCards) => {
            const nextBase = Array.isArray(prevCards) ? prevCards.slice() : [];
            if (nextBase.some((card) => card?.id === entry.card.id)) return prevCards;
            nextBase.push(entry.card);
            return sortCards(uniqueCardsById(nextBase));
          });
        }
      }, pauseMs);
    });

    const collectCardsAll = ppukTriggered
      ? []
      : uniqueCardsById(handCaptureCards.concat(flipCaptureCards));
    const collectCardIds = collectCardsAll.map((card) => card.id);
    setHiddenCapturedCardIds(collectCardIds);
    if (collectCardsAll.length > 0) {
      const collects = [];
      collectCardsAll.forEach((card, idx) => {
        const collectFromAbs =
          prevSnapshot?.boardById?.get(card.id) ||
          currentSnapshot?.boardById?.get(card.id) ||
          impactPointByCardId.get(card.id) ||
          centerOfRect(centerRect);
        const collectToAbs = resolveCapturePoint(card, idx);
        collects.push({
          id: `fx-collect-${turnNo}-${card.id}-${idx}`,
          card,
          duration: spec.collectDuration,
          delay: idx * spec.collectStagger,
          from: toLocalPoint(collectFromAbs),
          delta: {
            x: Math.round(collectToAbs.x - collectFromAbs.x),
            y: Math.round(collectToAbs.y - collectFromAbs.y)
          }
        });
      });
      const collectWaitMs = spec.collectDuration + Math.max(0, (collects.length - 1) * spec.collectStagger);
      pushQueueStep(() => {
        if (!isRunActive()) return;
        setFlightFxItems((prevItems) =>
          Array.isArray(prevItems) ? prevItems.filter((item) => !item?.holdUntilCollect) : []
        );
        setHiddenCapturedCardIds(collectCardIds);
        setHiddenBoardCardIds(collectCardIds);
        setCollectFxItems(collects);
      }, collectWaitMs);
      pushQueueStep(() => {
        if (!isRunActive()) return;
        setCollectFxItems([]);
      }, spec.settleHold);
    } else {
      pushQueueStep(() => {
        if (!isRunActive()) return;
        setHiddenBoardCardIds([]);
      }, spec.settleHold);
    }

    pushQueueStep(() => {
      if (!isRunActive()) return;
      clearTransientFx();
      // Handoff to the real post-turn board first.
      setBoardFrozenCards(sortCards(state.board || []));
    }, 24);
    pushQueueStep(() => {
      if (!isRunActive()) return;
      setBoardFrozenCards(null);
    }, 16);
    pushQueueStep(() => {
      if (!isRunActive()) return;
      setHiddenCapturedCardIds([]);
      setHiddenBoardCardIds([]);
    }, 0);

    const runQueueStep = (idx) => {
      if (!isRunActive()) return;
      if (idx >= queue.length) return;
      const step = queue[idx];
      step.apply();
      if (!isRunActive()) return;
      if (idx >= queue.length - 1) return;
      if (step.waitMs <= 0) {
        runQueueStep(idx + 1);
        return;
      }
      fxClearTimerRef.current = setTimeout(() => {
        fxClearTimerRef.current = null;
        runQueueStep(idx + 1);
      }, step.waitMs);
    };

    runQueueStep(0);

    return () => {
      if (runToken === fxRunTokenRef.current) {
        fxRunTokenRef.current += 1;
      }
      if (fxClearTimerRef.current) {
        clearTimeout(fxClearTimerRef.current);
        fxClearTimerRef.current = null;
      }
    };
  }, [latestTurnSummary, ui.speedMode]);

  useEffect(() => {
    boardBeforeTurnRef.current = (state.board || []).slice();
  }, [state.board]);

  /* 3) Render helpers */
  const renderHand = (playerKey, clickable) => {
    const player = state.players[playerKey];
    const cards = player.hand.slice();
    const cardsById = new Map(player.hand.map((card) => [card.id, card]));
    const boardMonths = new Set(state.board.map((b) => b.month));
    const capturedMonths = new Set();
    ["human", "ai"].forEach((key) => {
      const captured = state.players?.[key]?.captured || {};
      ["kwang", "five", "ribbon", "junk"].forEach((category) => {
        (captured[category] || []).forEach((card) => {
          if (card?.month >= 1 && card?.month <= 12) capturedMonths.add(card.month);
        });
      });
    });
    const handMonthCounts = cards.reduce((acc, card) => {
      if (!card || card.passCard || card.month < 1 || card.month > 12) return acc;
      acc.set(card.month, (acc.get(card.month) || 0) + 1);
      return acc;
    }, new Map());
    const grayBorderMonths = new Set(
      Array.from(handMonthCounts.entries())
        .filter(([month, count]) => count >= 2 && capturedMonths.has(month))
        .map(([month]) => month)
    );
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
        const grayBorder = playerKey === "human" && !card.passCard && grayBorderMonths.has(card.month);
        return (
          <div
            key={playerKey + "-slot-" + idx}
            className={
              "hand-card-wrap" +
              (monthMatched ? " month-matched" : "") +
              (grayBorder ? " month-gray-border" : "")
            }
            data-zone={`hand-${playerKey}`}
            data-player-key={playerKey}
            data-card-id={card.id}
            data-month={card.month}
          >
            {hideAiCards ? (
              cardBackView(cardTheme)
            ) : (
              <CardView
                card={card}
                interactive={canSelect}
                onClick={() => canSelect && onPlayCard(card.id)}
                t={t}
                theme={cardTheme}
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
      return orderedCards.map((card, i) => (
        <div
          key={"b-" + i}
          className="hand-card-wrap"
          data-zone={`hand-${playerKey}`}
          data-player-key={playerKey}
          data-card-id={card.id}
          data-month={card.month}
        >
          {cardBackView(cardTheme)}
        </div>
      ));
    }

    return orderedCards.map((card) => {
      const canSelect = clickable && state.currentTurn === playerKey && state.phase === "playing" && !locked;
      const monthMatched = playerKey === "human" && !card.passCard && boardMonths.has(card.month);
      const grayBorder = playerKey === "human" && !card.passCard && grayBorderMonths.has(card.month);
      return (
        <div
          key={card.id}
          className={
            "hand-card-wrap" +
            (monthMatched ? " month-matched" : "") +
            (grayBorder ? " month-gray-border" : "")
          }
          data-zone={`hand-${playerKey}`}
          data-player-key={playerKey}
          data-card-id={card.id}
          data-month={card.month}
        >
          <CardView
            card={card}
            interactive={canSelect}
            onClick={() => canSelect && onPlayCard(card.id)}
            t={t}
            theme={cardTheme}
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
      data-zone="capture-panel"
      data-player-key={playerKey}
      style={{
        "--capture-zone-w": `${CAPTURE_ZONE_WIDTH.normal}px`,
        "--capture-junk-zone-w": `${CAPTURE_ZONE_WIDTH.junk}px`,
        "--capture-state-zone-w": `${CAPTURE_ZONE_WIDTH.state}px`
      }}
    >
      {CAPTURED_TYPES.map(({ key }) => {
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
        const lineCards = uniqueCards.filter((card) => !hiddenCapturedIdSet.has(card.id));
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
          <div
            key={`${playerKey}-${key}`}
            className={`capture-zone capture-zone-${key}`}
            data-zone="capture-zone"
            data-player-key={playerKey}
            data-capture-key={key}
          >
            <div className="captured-stack">
              <div
                className="captured-row dynamic-overlap"
                style={{
                  "--stack-overlap": `${stackOverlap}px`
                }}
              >
                {lineCards.map((c) => {
                  const source = latestTurnSummary.captureSourceById.get(c.id);
                  const classes = ["stack-item"];
                  if (source === "hand") classes.push("captured-recent-hand");
                  if (source === "flip") classes.push("captured-recent-flip");
                  if (source && traceAnimating) classes.push("captured-recent-enter");
                  return (
                    <div key={`${playerKey}-${key}-${c.id}`} className={classes.join(" ")}>
                      <CardView card={c} t={t} theme={cardTheme} />
                    </div>
                  );
                })}
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
    boardCardsRenderSorted
      .filter((card) => !hiddenBoardCardIdSet.has(card.id))
      .forEach((card) => {
        if (!monthMap.has(card.month)) monthMap.set(card.month, []);
        monthMap.get(card.month).push(card);
      });
    const months = Array.from({ length: 12 }, (_, i) => i + 1).filter((month) => monthMap.has(month));

    return (
      <div className="board-orbit">
        {months.map((month) => {
          const angleDeg = -90 + (month - 1) * 30;
          const angle = (angleDeg * Math.PI) / 180;
          const radiusX = 320;
          const radiusY = 124;
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
                const isLiveBoardCard = boardRealCardIdSet.has(card.id);
                const selectable =
                  isLiveBoardCard &&
                  !openingPick?.active &&
                  state.phase === "select-match" &&
                  state.pendingMatch?.playerKey &&
                  participantType(ui, state.pendingMatch.playerKey) === "human" &&
                  (state.pendingMatch.boardCardIds || []).includes(card.id) &&
                  !locked;
                return (
                  <div
                    key={card.id}
                    className="orbit-card-wrap"
                    data-zone="board-card"
                    data-card-id={card.id}
                    data-month={card.month}
                  >
                    <CardView
                      card={card}
                      interactive={selectable}
                      onClick={() => selectable && onChooseMatch(card.id)}
                      t={t}
                      theme={cardTheme}
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
              value={ui.modelPicks?.[playerKey] || DEFAULT_BOT_POLICY}
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

  /* 4) Popup handlers */
  const handleOpenCardGuidePopup = () =>
    openCardGuidePopup({ monthCards, bonusCards, language });
  const handleOpenRulesPopup = () => openRulesPopup({ language });
  const handleOpenGameLogPopup = () => openGameLogPopup({ log: state.log || [], language });

  /* 5) Layout render */
  return (
    <div className={`board table-theme speed-${ui.speedMode || "visual"}`} ref={boardRootRef}>
      <div className="table-field">
        <div className="play-column">
          {!openingPick?.active ? renderCaptureLane("ai") : null}
          <section className="center-field">
            <div className="center-main" ref={centerMainRef}>
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
              {!openingPick?.active ? (
                <div className="turn-fx-layer" aria-hidden="true">
                  {landingFxItems.map((item) => (
                    <div
                      key={item.id}
                      className={`turn-landing-ring ${
                        item.targetType === "hit" ? "turn-landing-ring-hit" : "turn-landing-ring-empty"
                      }`}
                      style={{
                        left: `${item.point.x}px`,
                        top: `${item.point.y}px`,
                        "--fx-delay": `${item.delay}ms`,
                        "--fx-dur": `${item.duration}ms`
                      }}
                    />
                  ))}
                  {flightFxItems.map((item) => (
                    <div
                      key={item.id}
                      className={`turn-flight-card ${
                        item.kind === "play" ? "turn-flight-play" : "turn-flight-flip"
                      } ${item.targetType === "hit" ? "turn-flight-target-hit" : "turn-flight-target-empty"}`}
                      style={{
                        left: `${item.from.x}px`,
                        top: `${item.from.y}px`,
                        "--fx-delay": `${item.delay}ms`,
                        "--fx-dur": `${item.duration}ms`,
                        "--fx-dx": `${item.delta.x}px`,
                        "--fx-dy": `${item.delta.y}px`,
                        "--fx-tilt": `${item.tiltDeg}deg`
                      }}
                    >
                      <CardView card={item.card} t={t} theme={cardTheme} />
                    </div>
                  ))}
                  {collectFxItems.map((item) => (
                    <div
                      key={item.id}
                      className="turn-collect-card"
                      style={{
                        left: `${item.from.x}px`,
                        top: `${item.from.y}px`,
                        "--fx-delay": `${item.delay}ms`,
                        "--fx-dur": `${item.duration}ms`,
                        "--fx-cdx": `${item.delta.x}px`,
                        "--fx-cdy": `${item.delta.y}px`
                      }}
                    >
                      <CardView card={item.card} t={t} theme={cardTheme} />
                    </div>
                  ))}
                </div>
              ) : null}
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
                              <CardView card={card} t={t} theme={cardTheme} />
                              {selectedByHuman ? <span className="opening-owner-tag">{t("board.opening.myPick")}</span> : null}
                              {selectedByAi ? <span className="opening-owner-tag">{t("board.opening.opponentPick")}</span> : null}
                            </>
                          ) : (
                            <div className="opening-back">{cardBackView(cardTheme)}</div>
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
                    <div className="deck-stack deck-stack-center" data-zone="deck-center">
                      <img src={buildCardUiAssetPath("deck-stack.svg", cardTheme)} alt="deck stack" />
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
          {!openingPick?.active ? renderCaptureLane("human") : null}
          {!openingPick?.active ? renderPlayerHandLane() : null}
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




