import { DEFAULT_BOT_POLICY, normalizeBotPolicy } from "./policies.js";
import { botPlay, getHeuristicCardProbabilities } from "./heuristicPolicyEngine.js";
import {
  getModelCandidateProbabilities,
  modelPolicyPlay as runModelPolicyPlay
} from "./modelPolicyEngine.js";

function normalizeSource(source) {
  const normalized = String(source || "heuristic")
    .trim()
    .toLowerCase();
  return normalized === "model" ? "model" : "heuristic";
}

function resolveHeuristicPolicy(options = {}) {
  return normalizeBotPolicy(options.heuristicPolicy || DEFAULT_BOT_POLICY);
}

export function aiPlay(state, actor, options = {}) {
  const source = normalizeSource(options.source);
  const heuristicPolicy = resolveHeuristicPolicy(options);
  if (source === "model") {
    const next = options.model ? runModelPolicyPlay(state, actor, options.model) : state;
    if (next !== state) return next;
  }
  return botPlay(state, actor, { policy: heuristicPolicy });
}

export function getAiPlayProbabilities(state, actor, options = {}) {
  const source = normalizeSource(options.source);
  const heuristicPolicy = resolveHeuristicPolicy(options);
  if (source === "model") {
    if (options.model) {
      const scored = getModelCandidateProbabilities(state, actor, options.model, {
        previewPlay: !!options.previewPlay
      });
      if (scored && scored.decisionType === "play") {
        return scored.probabilities || null;
      }
    }
  }
  return getHeuristicCardProbabilities(state, actor, heuristicPolicy);
}
