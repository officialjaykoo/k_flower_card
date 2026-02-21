import { useEffect, useMemo, useState } from "react";
import {
  initGame,
  playTurn,
  getShakingReveal,
  calculateScore,
  getDeclarableShakingMonths,
  askShakingConfirm,
  chooseShakingYes,
  chooseShakingNo,
  getDeclarableBombMonths,
  declareBomb,
  chooseGo,
  chooseStop,
  choosePresidentStop,
  choosePresidentHold,
  chooseGukjinMode,
  chooseMatch,
  createSeededRng
} from "./engine/index.js";
import { DEFAULT_BOT_POLICY } from "./ai/policies.js";
import { getActionPlayerKey } from "./engine/runner.js";
import { formatActionText, formatEventsText } from "./ui/utils/replay.js";
import { isBotPlayer, participantType, randomSeed, sortCards } from "./ui/utils/common.js";
import { DEFAULT_CARD_THEME } from "./cards.js";
import { DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES, makeTranslator, translate } from "./ui/i18n/i18n.js";
import { useAiRuntime } from "./app/useAiRuntime.js";
import { useReplayController } from "./app/useReplayController.js";
import GameBoard from "./ui/components/GameBoard.jsx";
import GameOverlays from "./ui/components/GameOverlays.jsx";
import "../styles.css";

export default function App() {
  const [ui, setUi] = useState(() => {
    return {
      language: DEFAULT_LANGUAGE,
      revealAiHand: true,
      fixedHand: true,
      handLayoutNonce: 0,
      seed: randomSeed(),
      participants: { human: "human", ai: "ai" },
      modelPicks: { human: DEFAULT_BOT_POLICY, ai: DEFAULT_BOT_POLICY },
      lastRecordedRoundKey: null,
      speedMode: "fast",
      visualDelayMs: 400,
      cardTheme: DEFAULT_CARD_THEME,
      replay: { enabled: false, turnIndex: 0, autoPlay: false, intervalMs: 900 }
    };
  });

  const t = useMemo(() => makeTranslator(ui.language || DEFAULT_LANGUAGE), [ui.language]);
  const [state, setState] = useState(() =>
    initGame("A", createSeededRng(randomSeed()), {
      carryOverMultiplier: 1,
      language: DEFAULT_LANGUAGE
    })
  );
  const [openingPick, setOpeningPick] = useState({
    active: true,
    humanCard: null,
    aiCard: null,
    winnerKey: null
  });

  const {
    modelOptions,
    applyParticipantLabels,
    chooseBotAction,
    runAuto,
    aiPlayProbMap
  } = useAiRuntime({
    ui,
    state,
    translateFn: translate,
    participantTypeFn: participantType,
    isBotPlayerFn: isBotPlayer
  });

  const {
    replayModeEnabled,
    replaySourceLabel,
    replayPlayers,
    replayFrames,
    replayIdx,
    replayFrame,
    onReplayToggle,
    onReplayPrev,
    onReplayNext,
    onReplayAutoToggle,
    onReplaySeek,
    onReplayIntervalChange,
    onLoadReplay,
    onClearReplay
  } = useReplayController({ ui, setUi, state, t });

  useEffect(() => {
    document.documentElement.lang = ui.language || DEFAULT_LANGUAGE;
  }, [ui.language]);

  useEffect(() => {
    setState((prev) => applyParticipantLabels(prev));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ui.participants, ui.modelPicks?.human, ui.modelPicks?.ai, ui.language]);

  useEffect(() => {
    if (openingPick.active) return;
    setState((prev) => runAuto(applyParticipantLabels(prev)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [openingPick.active]);

  useEffect(() => {
    const reveal = getShakingReveal(state);
    if (!reveal) return;
    const ms = Math.max(30, reveal.expiresAt - Date.now());
    const timer = setTimeout(() => setState((nextState) => ({ ...nextState })), ms);
    return () => clearTimeout(timer);
  }, [state]);

  useEffect(() => {
    if (state.phase !== "resolution" || !state.result) return;
    const roundKey = [
      ui.seed,
      state.result.winner,
      state.result.human.total,
      state.result.ai.total,
      state.result.nagari ? 1 : 0,
      state.log.length
    ].join("|");
    if (ui.lastRecordedRoundKey === roundKey) return;

    setUi((prevUi) => ({ ...prevUi, lastRecordedRoundKey: roundKey }));
  }, [state, ui]);

  useEffect(() => {
    if (openingPick.active) return;
    if (ui.speedMode !== "visual") return;
    if (state.phase === "resolution") return;
    const actor = getActionPlayerKey(state);
    if (!actor || !isBotPlayer(ui, actor)) return;
    const timer = setTimeout(() => {
      setState((prev) => {
        const stepActor = getActionPlayerKey(prev);
        if (!stepActor || !isBotPlayer(ui, stepActor)) return prev;
        const next = chooseBotAction(prev, stepActor, ui);
        return next === prev ? prev : next;
      });
    }, ui.visualDelayMs);
    return () => clearTimeout(timer);
  }, [state, ui.speedMode, ui.visualDelayMs, ui.participants, openingPick.active, chooseBotAction]);

  const startGame = (opts = {}) => {
    const keepGold = opts.keepGold ?? true;
    const seedBase = opts.seedOverride != null ? String(opts.seedOverride) : ui.seed;
    const seed = (seedBase || "").trim() || randomSeed();
    const firstTurnKey = opts.firstTurnKey ?? null;
    const carryOverMultiplier = opts.carryOverMultiplier ?? state.nextCarryOverMultiplier ?? 1;
    const initialGold = keepGold
      ? { human: state.players.human.gold, ai: state.players.ai.gold }
      : undefined;

    let next = initGame(state.ruleKey, createSeededRng(seed), {
      carryOverMultiplier,
      firstTurnKey,
      initialGold,
      cardTheme: ui.cardTheme || DEFAULT_CARD_THEME,
      language: ui.language || DEFAULT_LANGUAGE
    });
    const nextUi = {
      ...ui,
      handLayoutNonce: (ui.handLayoutNonce || 0) + 1,
      seed,
      lastRecordedRoundKey: null,
      replay: { ...ui.replay, enabled: false, autoPlay: false, turnIndex: 0 }
    };
    next = applyParticipantLabels(next, nextUi);
    next = runAuto(next, nextUi);
    setUi(nextUi);
    setState(next);
  };

  const decideStarterByPickedCards = (humanCard, aiCard) => {
    if (!humanCard || !aiCard) return "human";
    const hour = new Date().getHours();
    const isNight = hour >= 18 || hour < 6;
    if (humanCard.month === aiCard.month) {
      return Math.random() < 0.5 ? "human" : "ai";
    }
    if (isNight) {
      return humanCard.month < aiCard.month ? "human" : "ai";
    }
    return humanCard.month > aiCard.month ? "human" : "ai";
  };

  const onOpeningPick = (cardId) => {
    if (!openingPick.active || openingPick.humanCard) return;
    const humanCard = state.board.find((c) => c.id === cardId);
    if (!humanCard) return;
    const remain = state.board.filter((c) => c.id !== cardId);
    if (remain.length === 0) return;
    const aiCard = remain[Math.floor(Math.random() * remain.length)];
    const winnerKey = decideStarterByPickedCards(humanCard, aiCard);

    setOpeningPick({
      active: true,
      humanCard,
      aiCard,
      winnerKey
    });

    setTimeout(() => {
      setOpeningPick({
        active: false,
        humanCard,
        aiCard,
        winnerKey
      });
      startGame({ firstTurnKey: winnerKey });
    }, 1000);
  };

  const reveal = getShakingReveal(state);
  const locked = openingPick.active;

  const onStartSpecifiedGame = () => {
    startGame();
  };

  const onStartRandomGame = () => {
    const nextSeed = randomSeed();
    setUi((prevUi) => ({ ...prevUi, seed: nextSeed }));
    setTimeout(() => startGame({ seedOverride: nextSeed }), 0);
  };

  const hScore = calculateScore(state.players.human, state.players.ai, state.ruleKey);
  const aScore = calculateScore(state.players.ai, state.players.human, state.ruleKey);

  const dispatchGameAction = (actionType, payload, reducer, options = {}) => {
    const { auto = false } = options;
    setState((prev) => {
      let next = reducer(prev);
      if (auto) next = runAuto(next);
      const nextNo = (next.uiActionLog?.length || 0) + 1;
      const entry = {
        no: nextNo,
        actionType,
        payload,
        timestamp: Date.now(),
        phaseBefore: prev.phase,
        phaseAfter: next.phase,
        turnBefore: prev.currentTurn,
        turnAfter: next.currentTurn
      };
      return { ...next, uiActionLog: [...(next.uiActionLog || []), entry] };
    });
  };

  const onChooseMatch = (id) =>
    dispatchGameAction("CHOOSE_MATCH", { boardCardId: id }, (prev) => chooseMatch(prev, id), {
      auto: true
    });

  const onPlayCard = (cardId) =>
    dispatchGameAction(
      "PLAY_CARD",
      { cardId },
      (prev) => {
        if (prev.phase === "playing") {
          const actor = prev.currentTurn;
          if (participantType(ui, actor) === "human") {
            const selected = prev.players[actor]?.hand?.find((c) => c.id === cardId);
            if (selected) {
              const bombMonths = getDeclarableBombMonths(prev, actor);
              if (bombMonths.includes(selected.month)) {
                return declareBomb(prev, actor, selected.month);
              }
              const shakingMonths = getDeclarableShakingMonths(prev, actor);
              if (shakingMonths.includes(selected.month)) {
                return askShakingConfirm(prev, actor, cardId);
              }
            }
          }
        }
        return playTurn(prev, cardId);
      },
      { auto: true }
    );

  const onChooseGo = (playerKey) =>
    dispatchGameAction("CHOOSE_GO", { playerKey }, (prev) => chooseGo(prev, playerKey), {
      auto: true
    });

  const onChooseStop = (playerKey) =>
    dispatchGameAction("CHOOSE_STOP", { playerKey }, (prev) => chooseStop(prev, playerKey), {
      auto: false
    });

  const onChoosePresidentStop = (playerKey) =>
    dispatchGameAction(
      "PRESIDENT_STOP",
      { playerKey },
      (prev) => choosePresidentStop(prev, playerKey),
      { auto: true }
    );

  const onChoosePresidentHold = (playerKey) =>
    dispatchGameAction(
      "PRESIDENT_HOLD",
      { playerKey },
      (prev) => choosePresidentHold(prev, playerKey),
      { auto: true }
    );

  const onChooseGukjinMode = (playerKey, mode) =>
    dispatchGameAction(
      "GUKJIN_MODE",
      { playerKey, mode },
      (prev) => chooseGukjinMode(prev, playerKey, mode),
      { auto: true }
    );

  const onChooseShakingYes = (playerKey) =>
    dispatchGameAction("SHAKING_YES", { playerKey }, (prev) => chooseShakingYes(prev, playerKey), {
      auto: true
    });

  const onChooseShakingNo = (playerKey) =>
    dispatchGameAction("SHAKING_NO", { playerKey }, (prev) => chooseShakingNo(prev, playerKey), {
      auto: true
    });

  return (
    <div className="layout">
      <div className="board-wrap">
        <GameBoard
          state={state}
          ui={ui}
          setUi={setUi}
          locked={locked}
          participantType={participantType}
          onChooseMatch={onChooseMatch}
          onPlayCard={onPlayCard}
          onStartSpecifiedGame={onStartSpecifiedGame}
          onStartRandomGame={onStartRandomGame}
          hScore={hScore}
          aScore={aScore}
          sortCards={sortCards}
          replayModeEnabled={replayModeEnabled}
          replaySourceLabel={replaySourceLabel}
          replayPlayers={replayPlayers}
          replayFrames={replayFrames}
          replayIdx={replayIdx}
          replayFrame={replayFrame}
          formatActionText={(action) => formatActionText(action, t)}
          formatEventsText={formatEventsText}
          modelOptions={modelOptions}
          aiPlayProbMap={aiPlayProbMap}
          openingPick={openingPick}
          onOpeningPick={onOpeningPick}
          onReplayToggle={onReplayToggle}
          onReplayPrev={onReplayPrev}
          onReplayNext={onReplayNext}
          onReplayAutoToggle={onReplayAutoToggle}
          onReplaySeek={onReplaySeek}
          onReplayIntervalChange={onReplayIntervalChange}
          onLoadReplay={onLoadReplay}
          onClearReplay={onClearReplay}
          t={t}
          supportedLanguages={SUPPORTED_LANGUAGES}
        />

        <GameOverlays
          state={state}
          ui={ui}
          reveal={reveal}
          participantType={participantType}
          onChooseMatch={onChooseMatch}
          onChooseGo={onChooseGo}
          onChooseStop={onChooseStop}
          onChoosePresidentStop={onChoosePresidentStop}
          onChoosePresidentHold={onChoosePresidentHold}
          onChooseGukjinMode={onChooseGukjinMode}
          onChooseShakingYes={onChooseShakingYes}
          onChooseShakingNo={onChooseShakingNo}
          onStartSpecifiedGame={onStartSpecifiedGame}
          onStartRandomGame={onStartRandomGame}
          t={t}
        />
      </div>
    </div>
  );
}
