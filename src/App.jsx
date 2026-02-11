import { useEffect, useMemo, useRef, useState } from "react";
import {
  initGame,
  playTurn,
  getDeclarableShakingMonths,
  declareShaking,
  getShakingReveal,
  getDeclarableBombMonths,
  declareBomb,
  calculateScore,
  estimateRemaining,
  ruleSets,
  chooseGo,
  chooseStop,
  chooseKungUse,
  chooseKungPass,
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
    setState((prev) => runAuto(applyParticipantLabels(prev)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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

  useEffect(() => {
    if (!ui.replay.autoPlay || !ui.replay.enabled) return;
    const frames = buildReplayFrames(state);
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
  }, [ui.replay.autoPlay, ui.replay.enabled, ui.replay.intervalMs, state]);

  useEffect(() => {
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
    const seed = (ui.seed || "").trim() || randomSeed();
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

  const updateNextFirstTurnFromResult = () => {
    const resultWinner = state.result?.winner;
    const isValidWinner = resultWinner === "human" || resultWinner === "ai";
    setUi((u) => {
      if (state.result?.nagari) return { ...u, nextFirstTurnKey: u.lastWinnerKey || state.startingTurnKey || "human" };
      if (isValidWinner) return { ...u, lastWinnerKey: resultWinner, nextFirstTurnKey: resultWinner };
      return { ...u, nextFirstTurnKey: u.lastWinnerKey || state.startingTurnKey || "human" };
    });
  };

  const reveal = getShakingReveal(state);
  const locked = !!reveal;
  const replayFrames = useMemo(() => buildReplayFrames(state), [state]);
  const replayIdx = Math.max(0, Math.min(replayFrames.length - 1, ui.replay.turnIndex));
  const replayFrame = replayFrames[replayIdx] || null;

  const hScore = calculateScore(state.players.human, state.players.ai, state.ruleKey);
  const aScore = calculateScore(state.players.ai, state.players.human, state.ruleKey);
  const remain = estimateRemaining(state);

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
    dispatchGameAction("PLAY_CARD", { cardId }, (prev) => playTurn(prev, cardId), { auto: true });

  return (
    <div className="layout">
      <GameBoard
        state={state}
        ruleName={ruleSets[state.ruleKey].name}
        ui={ui}
        locked={locked}
        participantType={participantType}
        onChooseMatch={onChooseMatch}
        onPlayCard={onPlayCard}
        hScore={hScore}
        aScore={aScore}
        sortCards={sortCards}
      />

      <SidebarPanels
        state={state}
        ui={ui}
        setUi={setUi}
        setState={setState}
        startGame={startGame}
        runAuto={runAuto}
        randomSeed={randomSeed}
        participantType={participantType}
        safeLoadJson={safeLoadJson}
        replayFrames={replayFrames}
        replayIdx={replayIdx}
        replayFrame={replayFrame}
        formatActionText={formatActionText}
        formatEventsText={formatEventsText}
        remain={remain}
        actions={{
          getDeclarableShakingMonths,
          declareShaking: (month) =>
            dispatchGameAction(
              "DECLARE_SHAKING",
              { month, actor: state.currentTurn },
              (prev) => declareShaking(prev, prev.currentTurn, month),
              { auto: false }
            ),
          getDeclarableBombMonths,
          declareBomb: (month) =>
            dispatchGameAction(
              "DECLARE_BOMB",
              { month, actor: state.currentTurn },
              (prev) => declareBomb(prev, prev.currentTurn, month),
              { auto: true }
            ),
          chooseGo: (playerKey) =>
            dispatchGameAction("CHOOSE_GO", { playerKey }, (prev) => chooseGo(prev, playerKey), {
              auto: true
            }),
          chooseStop: (playerKey) =>
            dispatchGameAction("CHOOSE_STOP", { playerKey }, (prev) => chooseStop(prev, playerKey), {
              auto: false
            }),
          choosePresidentStop: (playerKey) =>
            dispatchGameAction(
              "PRESIDENT_STOP",
              { playerKey },
              (prev) => choosePresidentStop(prev, playerKey),
              { auto: true }
            ),
          choosePresidentHold: (playerKey) =>
            dispatchGameAction(
              "PRESIDENT_HOLD",
              { playerKey },
              (prev) => choosePresidentHold(prev, playerKey),
              { auto: true }
            ),
          chooseKungUse: (playerKey) =>
            dispatchGameAction("KUNG_USE", { playerKey }, (prev) => chooseKungUse(prev, playerKey), {
              auto: true
            }),
          chooseKungPass: (playerKey) =>
            dispatchGameAction(
              "KUNG_PASS",
              { playerKey },
              (prev) => chooseKungPass(prev, playerKey),
              { auto: true }
            ),
          chooseGukjinMode: (playerKey, mode) =>
            dispatchGameAction(
              "GUKJIN_MODE",
              { playerKey, mode },
              (prev) => chooseGukjinMode(prev, playerKey, mode),
              { auto: true }
            )
        }}
      />

      <GameOverlays
        state={state}
        ui={ui}
        reveal={reveal}
        participantType={participantType}
        onChooseMatch={onChooseMatch}
        updateNextFirstTurnFromResult={updateNextFirstTurnFromResult}
        setUi={setUi}
        startGame={startGame}
        randomSeed={randomSeed}
      />
    </div>
  );
}
