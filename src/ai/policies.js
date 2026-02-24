export const POLICY_HEURISTIC_V3 = "heuristic_v3";
export const POLICY_HEURISTIC_V4 = "heuristic_v4";
export const POLICY_HEURISTIC_V5 = "heuristic_v5";
export const POLICY_HEURISTIC_V6 = "heuristic_v6";
export const POLICY_NEAT_PHASE2_SEED9 = "neat_phase2_seed9";
export const DEFAULT_BOT_POLICY = POLICY_HEURISTIC_V3;

export const BOT_POLICIES = Object.freeze([POLICY_HEURISTIC_V3, POLICY_HEURISTIC_V4, POLICY_HEURISTIC_V5, POLICY_HEURISTIC_V6]);

export const MODEL_CATALOG = Object.freeze({
  [POLICY_HEURISTIC_V3]: {
    labelKey: "model.heuristicV3",
    kind: "bot_policy",
    botPolicy: POLICY_HEURISTIC_V3
  },
  [POLICY_HEURISTIC_V4]: {
    labelKey: "model.heuristicV4",
    kind: "bot_policy",
    botPolicy: POLICY_HEURISTIC_V4
  },
  [POLICY_HEURISTIC_V5]: {
    labelKey: "model.heuristicV5",
    kind: "bot_policy",
    botPolicy: POLICY_HEURISTIC_V5
  },
  [POLICY_HEURISTIC_V6]: {
    labelKey: "model.heuristicV6",
    kind: "bot_policy",
    botPolicy: POLICY_HEURISTIC_V6
  },
  [POLICY_NEAT_PHASE2_SEED9]: {
    labelKey: "model.neatPhase2Seed9",
    kind: "policy_model",
    policyPath: "/models/neat_phase2_seed9_winner_genome.json"
  }
});

const BOT_POLICY_SET = new Set(BOT_POLICIES);

export function normalizeBotPolicy(policy) {
  const normalized = String(policy || "")
    .trim()
    .toLowerCase();
  if (BOT_POLICY_SET.has(normalized)) return normalized;
  return DEFAULT_BOT_POLICY;
}

export function getModelLabel(pick, language, translateFn) {
  const cfg = MODEL_CATALOG[pick] || null;
  if (!cfg) return String(pick || "AI");
  return translateFn(language, cfg.labelKey, {}, String(pick));
}

export function buildModelOptions(language, translateFn) {
  return Object.entries(MODEL_CATALOG).map(([value, config]) => ({
    value,
    label: translateFn(language, config.labelKey, {}, value)
  }));
}
