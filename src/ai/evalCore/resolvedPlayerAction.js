import { aiPlay } from "../aiPlay.js";
import { hybridPolicyPlayDetailed } from "../hybridPolicyEngine.js";

export function buildAiPlayOptions(playerSpec) {
  if (playerSpec?.model) {
    return {
      source: "model",
      model: playerSpec.model,
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
