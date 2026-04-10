import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  DEFAULT_BOT_POLICY,
  MODEL_CATALOG,
  buildModelOptions,
  getModelLabel
} from "../ai/policies.js";
import { aiPlay, getAiPlayProbabilities } from "../ai/aiPlay.js";
import { createModelRuntimeContext } from "../ai/modelPolicyEngine.js";
import { advanceAutoTurns } from "../engine/runner.js";
import { DEFAULT_LANGUAGE } from "../ui/i18n/i18n.js";

/* ============================================================================
 * AI runtime hook
 * - policy/model selection
 * - model loading cache
 * - auto-turn execution
 * ========================================================================== */

function createEmptyModelSlots() {
  return { single: null, attack: null, defense: null };
}

function createRuntimeSlots() {
  return {
    single: createModelRuntimeContext(),
    attack: createModelRuntimeContext(),
    defense: createModelRuntimeContext(),
  };
}

async function fetchPolicyModel(path) {
  if (!path) return null;
  try {
    const r = await fetch(path);
    return r.ok ? await r.json() : null;
  } catch {
    return null;
  }
}

// Compatibility fallback for legacy policy_model_moe: use fixed priority.
function pickStaticMoeModel(slotModels) {
  return slotModels.attack || slotModels.defense || slotModels.single || null;
}

function pickStaticMoeSlotKey(slotModels) {
  if (slotModels.attack) return "attack";
  if (slotModels.defense) return "defense";
  if (slotModels.single) return "single";
  return null;
}

function resolveSelectedModelRuntime(pick, slotModels, slotRuntime) {
  const cfg = MODEL_CATALOG[pick] || null;
  if (cfg?.kind === "policy_model") {
    return {
      model: slotModels.single || slotModels.attack || slotModels.defense || null,
      runtimeCtx: slotRuntime.single,
    };
  }
  if (cfg?.kind === "policy_model_moe") {
    const runtimeSlotKey = pickStaticMoeSlotKey(slotModels);
    return {
      model: pickStaticMoeModel(slotModels),
      runtimeCtx: runtimeSlotKey ? slotRuntime[runtimeSlotKey] : null,
    };
  }
  return { model: null, runtimeCtx: null };
}

export function useAiRuntime({ ui, state, translateFn, participantTypeFn, isBotPlayerFn }) {
  /* 1) Runtime refs/state */
  const policyModelRef = useRef({
    human: createEmptyModelSlots(),
    ai: createEmptyModelSlots()
  });
  const policyRuntimeRef = useRef({
    human: createRuntimeSlots(),
    ai: createRuntimeSlots(),
  });
  const [modelVersion, setModelVersion] = useState(0);

  /* 2) UI model options */
  const modelOptions = useMemo(
    () => buildModelOptions(ui.language || DEFAULT_LANGUAGE, translateFn),
    [ui.language, translateFn]
  );

  /* 3) Label mapping */
  const applyParticipantLabels = useCallback(
    (gameState, runtimeUi = ui) => {
      const lang = runtimeUi.language || DEFAULT_LANGUAGE;
      const humanModelLabel =
        getModelLabel(runtimeUi.modelPicks?.human, lang, translateFn) || "AI-1";
      const aiModelLabel = getModelLabel(runtimeUi.modelPicks?.ai, lang, translateFn) || "AI-2";
      const humanLabel =
        participantTypeFn(runtimeUi, "human") === "ai"
          ? humanModelLabel
          : translateFn(lang, "player.player1");
      const aiLabel =
        participantTypeFn(runtimeUi, "ai") === "ai"
          ? aiModelLabel
          : translateFn(lang, "player.player2");
      const prevHuman = gameState?.players?.human || {};
      const prevAi = gameState?.players?.ai || {};
      if (prevHuman.label === humanLabel && prevAi.label === aiLabel) {
        return gameState;
      }
      return {
        ...gameState,
        players: {
          ...gameState.players,
          human: { ...prevHuman, label: humanLabel },
          ai: { ...prevAi, label: aiLabel }
        }
      };
    },
    [ui, translateFn, participantTypeFn]
  );

  /* 4) Per-player execution option resolver */
  const resolveAiExecutionOptions = useCallback(
    (runtimeUi, playerKey, gameState = state) => {
      const pick = runtimeUi.modelPicks?.[playerKey];
      const cfg = MODEL_CATALOG[pick] || null;
      const slotModels = policyModelRef.current[playerKey] || createEmptyModelSlots();
      const slotRuntime = policyRuntimeRef.current[playerKey] || createRuntimeSlots();
      const opponentKey = playerKey === "human" ? "ai" : "human";
      const opponentSlots = resolveSelectedModelRuntime(
        runtimeUi.modelPicks?.[opponentKey],
        policyModelRef.current[opponentKey] || createEmptyModelSlots(),
        policyRuntimeRef.current[opponentKey] || createRuntimeSlots()
      );

      if (cfg?.kind === "policy_model") {
        return {
          source: "model",
          model: slotModels.single || slotModels.attack || slotModels.defense || null,
          runtimeCtx: slotRuntime.single,
          opponentModel: opponentSlots.model,
          opponentRuntimeCtx: opponentSlots.runtimeCtx,
          heuristicPolicy: DEFAULT_BOT_POLICY
        };
      }

      if (cfg?.kind === "policy_model_moe") {
        const runtimeSlotKey = pickStaticMoeSlotKey(slotModels);
        return {
          source: "model",
          model: pickStaticMoeModel(slotModels),
          runtimeCtx: runtimeSlotKey ? slotRuntime[runtimeSlotKey] : null,
          opponentModel: opponentSlots.model,
          opponentRuntimeCtx: opponentSlots.runtimeCtx,
          heuristicPolicy: DEFAULT_BOT_POLICY
        };
      }

      return {
        source: "heuristic",
        model: null,
        heuristicPolicy: cfg?.botPolicy || DEFAULT_BOT_POLICY
      };
    },
    [state]
  );

  /* 5) Action executors */
  const chooseBotAction = useCallback(
    (gameState, playerKey, runtimeUi = ui) => {
      const options = resolveAiExecutionOptions(runtimeUi, playerKey, gameState);
      return aiPlay(gameState, playerKey, options);
    },
    [ui, resolveAiExecutionOptions]
  );

  const runAuto = useCallback(
    (gameState, runtimeUi = ui, forceFast = false) => {
      if (!forceFast && runtimeUi.speedMode === "visual") return gameState;
      return advanceAutoTurns(
        gameState,
        (playerKey) => isBotPlayerFn(runtimeUi, playerKey),
        (nextState, playerKey) => chooseBotAction(nextState, playerKey, runtimeUi)
      );
    },
    [ui, chooseBotAction, isBotPlayerFn]
  );

  /* 6) Model loading */
  useEffect(() => {
    let mounted = true;

    const loadFor = async (slot) => {
      const pick = ui.modelPicks?.[slot];
      const cfg = MODEL_CATALOG[pick] || null;
      if (!cfg) {
        policyModelRef.current[slot] = createEmptyModelSlots();
        policyRuntimeRef.current[slot] = createRuntimeSlots();
        return;
      }
      if (cfg.kind === "policy_model") {
        const single = await fetchPolicyModel(cfg.policyPath);
        policyModelRef.current[slot] = { single, attack: null, defense: null };
        policyRuntimeRef.current[slot] = createRuntimeSlots();
        return;
      }
      if (cfg.kind === "policy_model_moe") {
        const [attack, defense] = await Promise.all([
          fetchPolicyModel(cfg.policyPathAttack),
          fetchPolicyModel(cfg.policyPathDefense)
        ]);
        policyModelRef.current[slot] = { single: null, attack, defense };
        policyRuntimeRef.current[slot] = createRuntimeSlots();
        return;
      }
      policyModelRef.current[slot] = createEmptyModelSlots();
      policyRuntimeRef.current[slot] = createRuntimeSlots();
    };

    Promise.all([loadFor("human"), loadFor("ai")])
      .then(() => {
        if (!mounted) return;
        setModelVersion((v) => v + 1);
      })
      .catch(() => {
        if (!mounted) return;
        policyModelRef.current.human = createEmptyModelSlots();
        policyModelRef.current.ai = createEmptyModelSlots();
        policyRuntimeRef.current.human = createRuntimeSlots();
        policyRuntimeRef.current.ai = createRuntimeSlots();
        setModelVersion((v) => v + 1);
      });

    return () => {
      mounted = false;
    };
  }, [ui.modelPicks?.human, ui.modelPicks?.ai]);

  /* 7) AI hand probability preview */
  const aiPlayProbMap = useMemo(() => {
    if (state.phase !== "playing") return null;
    if (participantTypeFn(ui, "ai") !== "ai") return null;
    const options = resolveAiExecutionOptions(ui, "ai", state);
    return getAiPlayProbabilities(state, "ai", { ...options, previewPlay: true });
  }, [state, ui.participants, ui.modelPicks?.ai, modelVersion, participantTypeFn, resolveAiExecutionOptions]);

  return {
    modelOptions,
    applyParticipantLabels,
    chooseBotAction,
    runAuto,
    aiPlayProbMap
  };
}
