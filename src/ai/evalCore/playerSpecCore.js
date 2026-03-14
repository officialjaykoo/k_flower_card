import { parseHybridPlaySpec } from "./sharedGameHelpers.js";

function isLegacyUnsupportedSpec(token) {
  const raw = String(token || "").trim().toLowerCase();
  return raw.startsWith("hybrid_option(") || raw.startsWith("hybrid_play_go(");
}

export function resolvePlayerSpecCore(rawSpec, options = {}) {
  const token = String(rawSpec || "").trim();
  const label = String(options.label || "player spec");
  if (!token) {
    throw new Error(`empty ${label}`);
  }

  if (isLegacyUnsupportedSpec(token)) {
    throw new Error(
      `unsupported ${label}: ${token} (allowed: heuristic, model, hybrid_play(model,heuristic))`
    );
  }

  const resolveHeuristic = options.resolveHeuristic;
  const resolveModel = options.resolveModel;
  if (typeof resolveHeuristic !== "function" || typeof resolveModel !== "function") {
    throw new Error("resolvePlayerSpecCore requires resolveHeuristic and resolveModel callbacks");
  }

  const heuristicPolicy = resolveHeuristic(token);
  if (heuristicPolicy) {
    return {
      input: token,
      kind: "heuristic",
      key: heuristicPolicy,
      label: heuristicPolicy,
      heuristicPolicy,
      model: null,
      modelPath: null,
    };
  }

  const hybrid = parseHybridPlaySpec(token);
  if (hybrid) {
    const modelSpec = resolveModel(hybrid.modelToken, `${label}:model`);
    const fallbackPolicy = resolveHeuristic(hybrid.fallbackToken);
    if (!fallbackPolicy) {
      throw new Error(
        `invalid ${label}: ${token} (hybrid_play fallback must be a heuristic policy key)`
      );
    }
    return {
      ...modelSpec,
      input: token,
      kind: "hybrid_play",
      key: `hybrid_play(${modelSpec.key},${fallbackPolicy})`,
      label: `hybrid_play(${modelSpec.label},${fallbackPolicy})`,
      fallbackKey: fallbackPolicy,
      heuristicPolicy: fallbackPolicy,
    };
  }

  const modelSpec = resolveModel(token, label);
  return {
    ...modelSpec,
    input: token,
    kind: "model",
  };
}
