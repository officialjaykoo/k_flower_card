import { useEffect, useMemo, useRef, useState } from "react";
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
} from "./gameEngine.js";
import { botPlay } from "./bot.js";
import { advanceAutoTurns, getActionPlayerKey } from "./engineRunner.js";
import { buildReplayFrames, formatActionText, formatEventsText } from "./ui/utils/replay.js";
import {
  isBotPlayer,
  participantType,
  randomSeed,
  safeLoadJson,
  sortCards
} from "./ui/utils/common.js";
import GameBoard from "./ui/components/GameBoard.jsx";
import SidebarPanels from "./ui/components/SidebarPanels.jsx";
import GameOverlays from "./ui/components/GameOverlays.jsx";
import "../styles.css";

export default function App() {
  const [ui, setUi] = useState({
    revealAiHand: true,
    sortHand: true,
    seed: randomSeed(),
    carryOverMultiplier: 1,
    lastWinnerKey: null,
    nextFirstTurnKey: null,
    participants: { human: "human", ai: "ai" },
    lastRecordedRoundKey: null,
    speedMode: "fast",
    visualDelayMs: 400,
    replay: { enabled: false, turnIndex: 0, autoPlay: false, intervalMs: 900 }
  });

  const [state, setState] = useState(() => initGame("A", createSeededRng(randomSeed()), { carryOverMultiplier: 1 }));
  const [loadedReplay, setLoadedReplay] = useState(null);
  const [openingPick, setOpeningPick] = useState({
    active: true,
    humanCard: null,
    aiCard: null,
    winnerKey: null
  });
  const timerRef = useRef(null);

  const applyParticipantLabels = (s, u = ui) => {
    const humanLabel = participantType(u, "human") === "ai" ? "AI-1" : "플레이어1";
    const aiLabel = participantType(u, "ai") === "ai" ? "AI-2" : "플레이어2";
    s.players.human.label = humanLabel;
    s.players.ai.label = aiLabel;
    return s;
  };

  const runAuto = (s, u = ui, forceFast = false) => {
    if (!forceFast && u.speedMode === "visual") return s;
    return advanceAutoTurns(
      s,
      (playerKey) => isBotPlayer(u, playerKey),
      (ss, playerKey) => botPlay(ss, playerKey)
    );
  };

  useEffect(() => {
    if (openingPick.active) return;
    setState((prev) => runAuto(applyParticipantLabels(prev)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [openingPick.active]);

  useEffect(() => {
    const reveal = getShakingReveal(state);
    if (!reveal) return;
    const ms = Math.max(30, reveal.expiresAt - Date.now());
    const t = setTimeout(() => setState((s) => ({ ...s })), ms);
    return () => clearTimeout(t);
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

    const entry = {
      recordedAt: new Date().toISOString(),
      seed: ui.seed,
      ruleKey: state.ruleKey,
      participants: { ...ui.participants },
      winner: state.result.winner,
      nagari: state.result.nagari,
      nagariReasons: state.result.nagariReasons || [],
      totals: { human: state.result.human.total, ai: state.result.ai.total },
      startingTurnKey: state.startingTurnKey,
      endingTurnKey: state.currentTurn,
      log: state.log.slice(),
      kibo: state.kibo || []
    };
    const key = "kflower_game_logs";
    const prev = safeLoadJson(key, []);
    prev.push(entry);
    localStorage.setItem(key, JSON.stringify(prev.slice(-500)));
    setUi((u) => ({ ...u, lastRecordedRoundKey: roundKey }));
  }, [state, ui]);

  const replaySource = loadedReplay
    ? {
        kibo: loadedReplay.kibo || [],
        players: loadedReplay.players || state.players
      }
    : state;
  const replayFrames = useMemo(() => buildReplayFrames(replaySource), [replaySource]);
  const replayIdx = replayFrames.length
    ? Math.max(0, Math.min(replayFrames.length - 1, ui.replay.turnIndex))
    : 0;
  const replayFrame = replayFrames[replayIdx] || null;

  useEffect(() => {
    if (!ui.replay.autoPlay || !ui.replay.enabled) return;
    const frames = replayFrames;
    if (frames.length <= 1) return;
    timerRef.current = setInterval(() => {
      setUi((u) => {
        const next = Math.min(frames.length - 1, u.replay.turnIndex + 1);
        const ended = next >= frames.length - 1;
        return {
          ...u,
          replay: {
            ...u.replay,
            turnIndex: next,
            autoPlay: ended ? false : u.replay.autoPlay
          }
        };
      });
    }, ui.replay.intervalMs);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      timerRef.current = null;
    };
  }, [ui.replay.autoPlay, ui.replay.enabled, ui.replay.intervalMs, replayFrames]);

  useEffect(() => {
    if (openingPick.active) return;
    if (ui.speedMode !== "visual") return;
    if (state.phase === "resolution") return;
    const actor = getActionPlayerKey(state);
    if (!actor || !isBotPlayer(ui, actor)) return;
    const t = setTimeout(() => {
      setState((prev) => {
        const stepActor = getActionPlayerKey(prev);
        if (!stepActor || !isBotPlayer(ui, stepActor)) return prev;
        const next = botPlay(prev, stepActor);
        return next === prev ? prev : next;
      });
    }, ui.visualDelayMs);
    return () => clearTimeout(t);
  }, [state, ui.speedMode, ui.visualDelayMs, ui.participants]);

  const startGame = (opts = {}) => {
    const keepCarry = opts.keepCarry ?? false;
    const keepStarter = opts.keepStarter ?? false;
    const keepGold = opts.keepGold ?? false;
    const seedBase = opts.seedOverride != null ? String(opts.seedOverride) : ui.seed;
    const seed = (seedBase || "").trim() || randomSeed();
    const carryOverMultiplier = keepCarry ? ui.carryOverMultiplier : 1;
    const nextFirstTurnKey = keepStarter ? ui.nextFirstTurnKey : null;
    const firstTurnKey = opts.firstTurnKey ?? nextFirstTurnKey;
    const initialGold = keepGold
      ? { human: state.players.human.gold, ai: state.players.ai.gold }
      : undefined;

    let next = initGame(state.ruleKey, createSeededRng(seed), {
      carryOverMultiplier,
      firstTurnKey,
      initialGold
    });
    const nextUi = {
      ...ui,
      seed,
      carryOverMultiplier,
      nextFirstTurnKey,
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
      setUi((u) => ({ ...u, nextFirstTurnKey: winnerKey, lastWinnerKey: winnerKey }));
      startGame({ firstTurnKey: winnerKey });
    }, 900);
  };

  const reveal = getShakingReveal(state);
  const locked = openingPick.active;

  const onStartSpecifiedGame = () => {
    startGame();
  };

  const onStartRandomGame = () => {
    const nextSeed = randomSeed();
    setUi((u) => ({ ...u, seed: nextSeed, lastWinnerKey: null }));
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
    dispatchGameAction("PRESIDENT_STOP", { playerKey }, (prev) => choosePresidentStop(prev, playerKey), {
      auto: true
    });
  const onChoosePresidentHold = (playerKey) =>
    dispatchGameAction("PRESIDENT_HOLD", { playerKey }, (prev) => choosePresidentHold(prev, playerKey), {
      auto: true
    });
  const onChooseGukjinMode = (playerKey, mode) =>
    dispatchGameAction("GUKJIN_MODE", { playerKey, mode }, (prev) => chooseGukjinMode(prev, playerKey, mode), {
      auto: true
    });
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
          locked={locked}
          participantType={participantType}
          onChooseMatch={onChooseMatch}
          onPlayCard={onPlayCard}
          hScore={hScore}
          aScore={aScore}
          sortCards={sortCards}
          replayModeEnabled={!!loadedReplay}
          replaySourceLabel={loadedReplay?.label || null}
          replayPlayers={loadedReplay?.players || state.players}
          replayFrames={replayFrames}
          replayIdx={replayIdx}
          replayFrame={replayFrame}
          formatActionText={formatActionText}
          formatEventsText={formatEventsText}
          openingPick={openingPick}
          onOpeningPick={onOpeningPick}
          onReplayToggle={() =>
            setUi((u) => ({ ...u, replay: { ...u.replay, enabled: !u.replay.enabled, autoPlay: false } }))
          }
          onReplayPrev={() =>
            setUi((u) => ({ ...u, replay: { ...u.replay, turnIndex: Math.max(0, u.replay.turnIndex - 1) } }))
          }
          onReplayNext={() =>
            setUi((u) => ({
              ...u,
              replay: { ...u.replay, turnIndex: Math.min(replayFrames.length - 1, u.replay.turnIndex + 1) }
            }))
          }
          onReplayAutoToggle={() =>
            setUi((u) => ({ ...u, replay: { ...u.replay, autoPlay: !u.replay.autoPlay } }))
          }
          onReplaySeek={(idx) =>
            setUi((u) => ({ ...u, replay: { ...u.replay, turnIndex: Number(idx) } }))
          }
          onReplayIntervalChange={(ms) =>
            setUi((u) => ({ ...u, replay: { ...u.replay, intervalMs: Number(ms) } }))
          }
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
        />
      </div>

      <SidebarPanels
        state={state}
        ui={ui}
        setUi={setUi}
        startGame={startGame}
        onStartSpecifiedGame={onStartSpecifiedGame}
        onStartRandomGame={onStartRandomGame}
        participantType={participantType}
        safeLoadJson={safeLoadJson}
        replayModeEnabled={!!loadedReplay}
        onLoadReplay={(entry, label = "불러온 기보") => {
          setLoadedReplay({
            label,
            kibo: Array.isArray(entry?.kibo) ? entry.kibo : [],
            players: entry?.players || state.players
          });
          setUi((u) => ({
            ...u,
            replay: { ...u.replay, enabled: true, autoPlay: false, turnIndex: 0 }
          }));
        }}
        onClearReplay={() => {
          setLoadedReplay(null);
          setUi((u) => ({
            ...u,
            replay: { ...u.replay, enabled: false, autoPlay: false, turnIndex: 0 }
          }));
        }}
      />

    </div>
  );
}
