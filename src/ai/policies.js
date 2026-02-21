export const POLICY_HEURISTIC_V3 = "heuristic_v3";
export const POLICY_HEURISTIC_V4 = "heuristic_v4";
export const POLICY_MODEL_V14_ATTACK = "policy_v14_attack";
export const POLICY_MODEL_V14_DEFENSE = "policy_v14_defense";
export const POLICY_MODEL_V14_MOE = "policy_v14_moe";
export const DEFAULT_BOT_POLICY = POLICY_HEURISTIC_V3;

export const BOT_POLICIES = Object.freeze([POLICY_HEURISTIC_V3, POLICY_HEURISTIC_V4]);

export const MODEL_CATALOG = Object.freeze({
  [POLICY_MODEL_V14_MOE]: {
    labelKey: "model.v14Moe",
    kind: "policy_model_moe",
    policyPathAttack: "/models/policy-danmokv14-attack.json",
    policyPathDefense: "/models/policy-danmokv14-defense.json",
    moe: {
      defenseScoreThreshold: -2,
      attackScoreThreshold: 3,
      riskThreshold: 1,
      oppThreatThreshold: 0.55
    }
  },
  [POLICY_MODEL_V14_ATTACK]: {
    labelKey: "model.v14Attack",
    kind: "policy_model",
    policyPath: "/models/policy-danmokv14-attack.json"
  },
  [POLICY_MODEL_V14_DEFENSE]: {
    labelKey: "model.v14Defense",
    kind: "policy_model",
    policyPath: "/models/policy-danmokv14-defense.json"
  },
  [POLICY_HEURISTIC_V3]: {
    labelKey: "model.heuristicV3",
    kind: "bot_policy",
    botPolicy: POLICY_HEURISTIC_V3
  },
  [POLICY_HEURISTIC_V4]: {
    labelKey: "model.heuristicV4",
    kind: "bot_policy",
    botPolicy: POLICY_HEURISTIC_V4
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
