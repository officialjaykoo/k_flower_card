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
import { botPlay, getHeuristicCardProbabilities } from "./bot.js";
import { getModelCandidateProbabilities, modelPolicyPlay } from "./modelPolicyBot.js";
import { advanceAutoTurns, getActionPlayerKey } from "./engineRunner.js";
import { buildReplayFrames, formatActionText, formatEventsText } from "./ui/utils/replay.js";
import { isBotPlayer, participantType, randomSeed, sortCards } from "./ui/utils/common.js";
import { DEFAULT_CARD_THEME } from "./cards.js";
import { DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES, makeTranslator, translate } from "./ui/i18n/index.js";
import GameBoard from "./ui/components/GameBoard.jsx";
import GameOverlays from "./ui/components/GameOverlays.jsx";
import "../styles.css";

const MODEL_CATALOG = Object.freeze({
  sendol: { labelKey: "model.sendol", kind: "policy_model", policyPath: "/models/policy-sendol.json" },
  dolbaram: { labelKey: "model.dolbaram", kind: "policy_model", policyPath: "/models/policy-dolbaram-v5.json" },
  heuristic_v3: { labelKey: "model.heuristicV3", kind: "bot_policy", botPolicy: "heuristic_v3" }
});

function getModelLabel(pick, language) {
  const cfg = MODEL_CATALOG[pick] || null;
  if (!cfg) return String(pick || "AI");
  return translate(language, cfg.labelKey, {}, String(pick));
}

function buildModelOptions(language) {
  return Object.entries(MODEL_CATALOG).map(([value, config]) => ({
    value,
    label: translate(language, config.labelKey, {}, value)
  }));
}

export default function App() {
  const policyModelRef = useRef({ human: null, ai: null });
  const [modelVersion, setModelVersion] = useState(0);
  const [ui, setUi] = useState(() => {
    return {
      language: DEFAULT_LANGUAGE,
      revealAiHand: true,
      fixedHand: true,
      handLayoutNonce: 0,
      seed: randomSeed(),
      participants: { human: "human", ai: "ai" },
      modelPicks: { human: "sendol", ai: "dolbaram" },
      lastRecordedRoundKey: null,
      speedMode: "fast",
      visualDelayMs: 400,
      cardTheme: DEFAULT_CARD_THEME,
      replay: { enabled: false, turnIndex: 0, autoPlay: false, intervalMs: 900 }
    };
  });

  const t = useMemo(() => makeTranslator(ui.language || DEFAULT_LANGUAGE), [ui.language]);
  const modelOptions = useMemo(() => buildModelOptions(ui.language || DEFAULT_LANGUAGE), [ui.language]);
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
    const lang = u.language || DEFAULT_LANGUAGE;
    const humanModelLabel = getModelLabel(u.modelPicks?.human, lang) || "AI-1";
    const aiModelLabel = getModelLabel(u.modelPicks?.ai, lang) || "AI-2";
    const humanLabel =
      participantType(u, "human") === "ai" ? humanModelLabel : translate(lang, "player.player1");
    const aiLabel =
      participantType(u, "ai") === "ai" ? aiModelLabel : translate(lang, "player.player2");
    s.players.human.label = humanLabel;
    s.players.ai.label = aiLabel;
    return s;
  };

  const chooseBotAction = (ss, playerKey, u = ui) => {
    if (participantType(u, playerKey) === "ai") {
      const pick = u.modelPicks?.[playerKey];
      const cfg = MODEL_CATALOG[pick] || null;
      if (cfg?.kind === "bot_policy") {
        return botPlay(ss, playerKey, { policy: cfg.botPolicy || "heuristic_v3" });
      }
      const model = policyModelRef.current[playerKey];
      if (model) {
        const next = modelPolicyPlay(ss, playerKey, model);
        if (next !== ss) return next;
      }
    }
    return botPlay(ss, playerKey, { policy: "heuristic_v3" });
  };

  const runAuto = (s, u = ui, forceFast = false) => {
    if (!forceFast && u.speedMode === "visual") return s;
    return advanceAutoTurns(
      s,
      (playerKey) => isBotPlayer(u, playerKey),
      (ss, playerKey) => chooseBotAction(ss, playerKey, u)
    );
  };

  useEffect(() => {
    document.documentElement.lang = ui.language || DEFAULT_LANGUAGE;
  }, [ui.language]);

  useEffect(() => {
    let mounted = true;
    const loadFor = async (slot) => {
      const pick = ui.modelPicks?.[slot];
      const cfg = MODEL_CATALOG[pick] || null;
      if (!cfg || cfg.kind !== "policy_model") {
        policyModelRef.current[slot] = null;
        return;
      }
      const path = cfg.policyPath;
      if (!path) {
        policyModelRef.current[slot] = null;
        return;
      }
      try {
        const r = await fetch(path);
        policyModelRef.current[slot] = r.ok ? await r.json() : null;
      } catch {
        policyModelRef.current[slot] = null;
      }
    };
    Promise.all([loadFor("human"), loadFor("ai")])
      .then(() => setModelVersion((v) => v + 1))
      .catch(() => {
        if (!mounted) return;
        policyModelRef.current.human = null;
        policyModelRef.current.ai = null;
        setModelVersion((v) => v + 1);
      });
    return () => {
      mounted = false;
    };
  }, [ui.modelPicks?.human, ui.modelPicks?.ai]);

  useEffect(() => {
    setState((prev) => {
      const next = {
        ...prev,
        players: {
          ...prev.players,
          human: { ...prev.players.human },
          ai: { ...prev.players.ai }
        }
      };
      return applyParticipantLabels(next);
    });
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
    const timer = setTimeout(() => setState((s) => ({ ...s })), ms);
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

  const aiPlayProbMap = useMemo(() => {
    if (state.phase !== "playing") return null;
    if (participantType(ui, "ai") !== "ai") return null;
    const fallbackPolicy = "heuristic_v3";
    const fallbackProbs = () => getHeuristicCardProbabilities(state, "ai", fallbackPolicy);
    const aiPick = ui.modelPicks?.ai;
    const aiCfg = MODEL_CATALOG[aiPick] || null;
    if (aiCfg?.kind === "bot_policy") {
      return getHeuristicCardProbabilities(state, "ai", aiCfg.botPolicy || fallbackPolicy);
    }
    if (aiCfg?.kind === "policy_model") {
      const model = policyModelRef.current.ai;
      if (model) {
        const scored = getModelCandidateProbabilities(state, "ai", model, { previewPlay: true });
        if (scored && scored.decisionType === "play") {
          return scored.probabilities || null;
        }
      }
      return fallbackProbs();
    }
    return fallbackProbs();
  }, [state, ui.participants, ui.modelPicks?.ai, modelVersion]);

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
    const timer = setTimeout(() => {
      setState((prev) => {
        const stepActor = getActionPlayerKey(prev);
        if (!stepActor || !isBotPlayer(ui, stepActor)) return prev;
        const next = chooseBotAction(prev, stepActor, ui);
        return next === prev ? prev : next;
      });
    }, ui.visualDelayMs);
    return () => clearTimeout(timer);
  }, [state, ui.speedMode, ui.visualDelayMs, ui.participants]);

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
      cardTheme: ui.cardTheme || DEFAULT_CARD_THEME
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
    setUi((u) => ({ ...u, seed: nextSeed }));
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
          replayModeEnabled={!!loadedReplay}
          replaySourceLabel={loadedReplay?.label || null}
          replayPlayers={loadedReplay?.players || state.players}
          replayFrames={replayFrames}
          replayIdx={replayIdx}
          replayFrame={replayFrame}
          formatActionText={(action) => formatActionText(action, t)}
          formatEventsText={formatEventsText}
          modelOptions={modelOptions}
          aiPlayProbMap={aiPlayProbMap}
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
          onLoadReplay={(entry, label = t("replay.loadedSourceDefault")) => {
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
