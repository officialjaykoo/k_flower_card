import { DEFAULT_BOT_POLICY, normalizeBotPolicy } from "./policies.js";
import { botPlay, getHeuristicCardProbabilities } from "./heuristicPolicyEngine.js";
import {
  getModelCandidateProbabilities,
  modelPolicyPlay as runModelPolicyPlay
} from "./modelPolicyEngine_by_GPT.js";

/*
 * GPT line AI play entry module
 * - source: heuristic | model
 * - fail-fast: model source must resolve by model action
 */

// Main AI action resolver used by GPT game loops.
export function aiPlay(state, actor, options = {}) {
  const source = normalizeSource(options.source);
  const heuristicPolicy = resolveHeuristicPolicy(options);
  if (source === "model") {
    if (!options.model) {
      throw new Error(`[aiPlay_by_GPT] source=model requires options.model (actor=${actor}, phase=${state?.phase || "unknown"})`);
    }
    const next = runModelPolicyPlay(state, actor, options.model);
    if (next === state) {
      throw new Error(`[aiPlay_by_GPT] model action unresolved (actor=${actor}, phase=${state?.phase || "unknown"})`);
    }
    return next;
  }
  return botPlay(state, actor, {
    policy: heuristicPolicy,
    heuristicParams: options.heuristicParams && typeof options.heuristicParams === "object"
      ? options.heuristicParams
      : null
  });
}

// Probability preview API for hand-card UI overlays.
export function getAiPlayProbabilities(state, actor, options = {}) {
  const source = normalizeSource(options.source);
  const heuristicPolicy = resolveHeuristicPolicy(options);
  if (source === "model") {
    if (!options.model) {
      throw new Error(`[aiPlay_by_GPT] source=model requires options.model for probability preview (actor=${actor}, phase=${state?.phase || "unknown"})`);
    }
    const scored = getModelCandidateProbabilities(state, actor, options.model, {
      previewPlay: !!options.previewPlay
    });
    if (scored && scored.decisionType === "play") {
      return scored.probabilities || null;
    }
    return null;
  }
  return getHeuristicCardProbabilities(state, actor, heuristicPolicy);
}

// Normalize runtime source selector.
function normalizeSource(source) {
  const normalized = String(source || "heuristic")
    .trim()
    .toLowerCase();
  return normalized === "model" ? "model" : "heuristic";
}

// Resolve heuristic policy with default + canonical normalization.
function resolveHeuristicPolicy(options = {}) {
  return normalizeBotPolicy(options.heuristicPolicy || DEFAULT_BOT_POLICY);
}

