import { aiPlay, aiPlayAsync } from "../aiPlay.js";
import { hybridPolicyPlayDetailed } from "../hybridPolicyEngine.js";

export function buildAiPlayOptions(playerSpec) {
  if (playerSpec?.model) {
    return {
      source: "model",
      model: playerSpec.model,
      runtimeCtx: playerSpec.runtimeCtx || null,
      opponentModel: playerSpec.opponentModel || null,
      opponentRuntimeCtx: playerSpec.opponentRuntimeCtx || null,
      nativeInferenceBackend: playerSpec.nativeInferenceBackend || "off",
      nativeInferenceStats: playerSpec.nativeInferenceStats || null,
    };
  }
  return {
    source: "heuristic",
    heuristicPolicy: String(playerSpec?.heuristicPolicy || playerSpec?.key || ""),
  };
}

export function resolveResolvedPlayerAction(state, actor, playerSpec) {
  if (playerSpec?.kind === "hybrid_play") {
    const traced = hybridPolicyPlayDetailed(state, actor, {
      model: playerSpec.model,
      heuristicPolicy: String(playerSpec.heuristicPolicy || ""),
      runtimeCtx: playerSpec.runtimeCtx || null,
      opponentModel: playerSpec.opponentModel || null,
      opponentRuntimeCtx: playerSpec.opponentRuntimeCtx || null,
    });
    return {
      next: traced?.next || state,
      actionSource: String(traced?.actionSource || "hybrid_play"),
      route: String(traced?.route || ""),
    };
  }

  if (playerSpec?.model) {
    return {
      next: aiPlay(state, actor, buildAiPlayOptions(playerSpec)),
      actionSource: "model",
      route: "model",
    };
  }

  return {
    next: aiPlay(state, actor, buildAiPlayOptions(playerSpec)),
    actionSource: "heuristic",
    route: "heuristic",
  };
}

export async function resolveResolvedPlayerActionAsync(state, actor, playerSpec) {
  if (playerSpec?.kind === "hybrid_play") {
    return resolveResolvedPlayerAction(state, actor, playerSpec);
  }

  if (playerSpec?.model) {
    if (playerSpec?.nativeInferenceStats && typeof playerSpec.nativeInferenceStats === "object") {
      playerSpec.nativeInferenceStats.resolve_calls =
        Number(playerSpec.nativeInferenceStats.resolve_calls || 0) + 1;
    }
    return {
      next: await aiPlayAsync(state, actor, buildAiPlayOptions(playerSpec)),
      actionSource: "model",
      route: "model",
    };
  }

  return {
    next: await aiPlayAsync(state, actor, buildAiPlayOptions(playerSpec)),
    actionSource: "heuristic",
    route: "heuristic",
  };
}
