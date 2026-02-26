/* ============================================================================
 * 1) Bot/Model policy identifiers
 * ========================================================================== */
export const POLICY_HEURISTIC_V3 = "heuristic_v3";
export const POLICY_HEURISTIC_V4 = "heuristic_v4";
export const POLICY_HEURISTIC_V5 = "heuristic_v5";
export const POLICY_HEURISTIC_V6 = "heuristic_v6";
export const POLICY_HEURISTIC_V7 = "heuristic_v7_gold_digger";
export const POLICY_HEURISTIC_V5PLUS = "heuristic_v5plus";
export const POLICY_NEAT_PHASE2_SEED9 = "neat_phase2_seed9";
export const DEFAULT_BOT_POLICY = POLICY_HEURISTIC_V3;

/* ============================================================================
 * 2) Bot policy whitelist used by runtime validation
 * ========================================================================== */
export const BOT_POLICIES = Object.freeze([
  POLICY_HEURISTIC_V3,
  POLICY_HEURISTIC_V4,
  POLICY_HEURISTIC_V5,
  POLICY_HEURISTIC_V5PLUS,
  POLICY_HEURISTIC_V6,
  POLICY_HEURISTIC_V7
]);

/* ============================================================================
 * 3) Catalog for UI labels and policy loading config
 * ========================================================================== */
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
  [POLICY_HEURISTIC_V5PLUS]: {
    labelKey: "model.heuristicV5Plus",
    kind: "bot_policy",
    botPolicy: POLICY_HEURISTIC_V5PLUS
  },
  [POLICY_HEURISTIC_V6]: {
    labelKey: "model.heuristicV6",
    kind: "bot_policy",
    botPolicy: POLICY_HEURISTIC_V6
  },
  [POLICY_HEURISTIC_V7]: {
    labelKey: "model.heuristicV7",
    kind: "bot_policy",
    botPolicy: POLICY_HEURISTIC_V7
  },
  [POLICY_NEAT_PHASE2_SEED9]: {
    labelKey: "model.neatPhase2Seed9",
    kind: "policy_model",
    policyPath: "/models/neat_phase2_seed9_winner_genome.json"
  }
});

const BOT_POLICY_SET = new Set(BOT_POLICIES);

/* ============================================================================
 * 4) Normalization + UI helpers
 * ========================================================================== */
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
