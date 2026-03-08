import { playTurn } from "../engine/index.js";
import { DEFAULT_BOT_POLICY, normalizeBotPolicy } from "./policies.js";
import { botPlay } from "./heuristicPolicyEngine.js";
import { modelPolicyPlay as runModelPolicyPlay } from "./modelPolicyEngine.js";

function transitionKey(state) {
  if (!state) return "null";
  const hh = Number(state?.players?.human?.hand?.length || 0);
  const ah = Number(state?.players?.ai?.hand?.length || 0);
  const d = Number(state?.deck?.length || 0);
  return [
    String(state.phase || ""),
    String(state.currentTurn || ""),
    String(state.pendingGoStop || ""),
    String(state.pendingMatch?.stage || ""),
    String(state.pendingPresident?.playerKey || ""),
    String(state.pendingShakingConfirm?.playerKey || ""),
    String(state.pendingGukjinChoice?.playerKey || ""),
    String(state.turnSeq || 0),
    String(state.kiboSeq || 0),
    String(hh),
    String(ah),
    String(d),
  ].join("|");
}

function isMoved(beforeState, nextState) {
  return !!nextState && transitionKey(nextState) !== transitionKey(beforeState);
}

function isPlayingTurn(state, actor) {
  return state?.phase === "playing" && state?.currentTurn === actor;
}

function isGoStopTurn(state, actor) {
  return state?.phase === "go-stop" && state?.pendingGoStop === actor;
}

function isMatchTurn(state, actor) {
  return state?.phase === "select-match" && state?.pendingMatch?.playerKey === actor;
}

function resolvesToDirectHandPlay(state, actor, targetState) {
  if (!isMoved(state, targetState)) return false;
  const hand = state?.players?.[actor]?.hand || [];
  const targetKey = transitionKey(targetState);
  for (const card of hand) {
    const cardId = String(card?.id || "");
    if (!cardId) continue;
    const next = playTurn(state, cardId);
    if (next && transitionKey(next) === targetKey) {
      return true;
    }
  }
  return false;
}

function resolveHeuristicInputs(options = {}) {
  const heuristicPolicy = normalizeBotPolicy(options.heuristicPolicy || DEFAULT_BOT_POLICY);
  return {
    heuristicPolicy,
    goStopPolicy: normalizeBotPolicy(options.goStopPolicy || heuristicPolicy),
    phasePolicy: normalizeBotPolicy(options.phasePolicy || heuristicPolicy),
    specialPolicy: normalizeBotPolicy(options.specialPolicy || heuristicPolicy),
    playFallbackPolicy: normalizeBotPolicy(options.playFallbackPolicy || heuristicPolicy),
    heuristicParams:
      options.heuristicParams && typeof options.heuristicParams === "object"
        ? options.heuristicParams
        : null,
  };
}

function runHeuristicPolicy(state, actor, policy, heuristicParams) {
  return botPlay(state, actor, {
    policy,
    heuristicParams,
  });
}

export function hybridPolicyPlayDetailed(state, actor, options = {}) {
  const model = options.model || null;
  const heuristicParams =
    options.heuristicParams && typeof options.heuristicParams === "object"
      ? options.heuristicParams
      : null;
  const goStopOnly = !!options.goStopOnly;
  const modelMatchPhase = !!options.modelMatchPhase;
  const goStopPolicy = normalizeBotPolicy(options.goStopPolicy || options.heuristicPolicy || DEFAULT_BOT_POLICY);

  if (isGoStopTurn(state, actor)) {
    const goStopNext = runHeuristicPolicy(state, actor, goStopPolicy, heuristicParams);
    if (isMoved(state, goStopNext)) {
      return {
        next: goStopNext,
        actionSource: "hybrid_heuristic_go_stop",
        route: "heuristic_go_stop",
      };
    }
    if (goStopOnly) {
      return {
        next: state,
        actionSource: "hybrid_go_stop_unresolved",
        route: "heuristic_go_stop_unresolved",
      };
    }
  }

  if (goStopOnly) {
    if (model) {
      const modelNext = runModelPolicyPlay(state, actor, model);
      if (isMoved(state, modelNext)) {
        return {
          next: modelNext,
          actionSource: "hybrid_model_non_go_stop",
          route: "model_non_go_stop",
        };
      }
    }
    return {
      next: state,
      actionSource: "hybrid_unresolved",
      route: "unresolved",
    };
  }

  const {
    heuristicPolicy,
    phasePolicy,
    specialPolicy,
    playFallbackPolicy,
  } = resolveHeuristicInputs(options);
  if (isGoStopTurn(state, actor)) {
    const fallbackNext =
      goStopPolicy === heuristicPolicy
        ? state
        : runHeuristicPolicy(state, actor, heuristicPolicy, heuristicParams);
    if (isMoved(state, fallbackNext)) {
      return {
        next: fallbackNext,
        actionSource: "hybrid_heuristic_go_stop_fallback",
        route: "heuristic_go_stop_fallback",
      };
    }
  }

  if (modelMatchPhase && isMatchTurn(state, actor)) {
    if (model) {
      const modelNext = runModelPolicyPlay(state, actor, model);
      if (isMoved(state, modelNext)) {
        return {
          next: modelNext,
          actionSource: "hybrid_model_match",
          route: "model_match",
        };
      }
    }

    const matchFallbackNext = runHeuristicPolicy(state, actor, phasePolicy, heuristicParams);
    if (isMoved(state, matchFallbackNext)) {
      return {
        next: matchFallbackNext,
        actionSource: "hybrid_heuristic_match_fallback",
        route: "heuristic_match_fallback",
      };
    }

    return {
      next: state,
      actionSource: "hybrid_match_unresolved",
      route: "match_unresolved",
    };
  }

  if (!isPlayingTurn(state, actor)) {
    const phaseNext = runHeuristicPolicy(state, actor, phasePolicy, heuristicParams);
    return {
      next: phaseNext,
      actionSource: "hybrid_heuristic_phase",
      route: "heuristic_phase",
    };
  }

  const specialNext = runHeuristicPolicy(state, actor, specialPolicy, heuristicParams);
  if (isMoved(state, specialNext) && !resolvesToDirectHandPlay(state, actor, specialNext)) {
    return {
      next: specialNext,
      actionSource: "hybrid_heuristic_special",
      route: "heuristic_special",
    };
  }

  if (model) {
    const modelNext = runModelPolicyPlay(state, actor, model);
    if (isMoved(state, modelNext)) {
      return {
        next: modelNext,
        actionSource: "hybrid_model_play",
        route: "model_play",
      };
    }
  }

  const fallbackNext = runHeuristicPolicy(state, actor, playFallbackPolicy, heuristicParams);
  if (isMoved(state, fallbackNext)) {
    return {
      next: fallbackNext,
      actionSource: "hybrid_heuristic_play_fallback",
      route: "heuristic_play_fallback",
    };
  }

  return {
    next: state,
    actionSource: "hybrid_unresolved",
    route: "unresolved",
  };
}

export function hybridPolicyPlay(state, actor, options = {}) {
  return hybridPolicyPlayDetailed(state, actor, options).next;
}
