/* ============================================================================
 * 1) Bot/Model policy identifiers
 * ========================================================================== */
export const POLICY_HEURISTIC_J1 = "H-J1";
export const POLICY_HEURISTIC_J2 = "H-J2";
export const POLICY_HEURISTIC_CL = "H-CL";
export const POLICY_HEURISTIC_ANTICL = "H-AntiCL";
export const POLICY_HEURISTIC_GPT = "H-GPT";
export const POLICY_HEURISTIC_GPT54 = "H-GPT54";
export const POLICY_HEURISTIC_GPT_PLAYMATCH_CL = "H-GPT-PlayMatch+CL";
export const POLICY_HEURISTIC_GPT_PLAYMATCHGO_CL = "H-GPT-PlayMatchGo+CL";
export const POLICY_HEURISTIC_GPT_PLAYMATCHSHAKE_CL = "H-GPT-PlayMatchShake+CL";
export const POLICY_HEURISTIC_GPT_PLAYMATCHGUKJIN_CL = "H-GPT-PlayMatchGukjin+CL";
export const POLICY_HEURISTIC_GPT_PLAYMATCHPRESIDENT_CL = "H-GPT-PlayMatchPresident+CL";
export const POLICY_HEURISTIC_GPT_PLAYMATCHGUKJINPRESIDENT_CL =
  "H-GPT-PlayMatchGukjinPresident+CL";
export const POLICY_HEURISTIC_GPT_ALLBUTGO_CL = "H-GPT-AllButGo+CL";
export const POLICY_HEURISTIC_GPT_PLAYMATCHBPG_CL = "H-GPT-PlayMatchBPG+CL";
export const POLICY_HEURISTIC_GEMINI = "H-Gemini";
export const POLICY_HEURISTIC_NEXG = "H-NEXg";
export const POLICY_NEAT_PHASE2_SEED9 = "N-P2-Seed9";
export const POLICY_NEAT_PHASE2_SEED66 = "N-P2-Seed66";
export const DEFAULT_BOT_POLICY = POLICY_HEURISTIC_CL;

/* ============================================================================
 * 2) Bot policy whitelist used by runtime validation
 * ========================================================================== */
export const BOT_POLICIES = Object.freeze([
  POLICY_HEURISTIC_J1,
  POLICY_HEURISTIC_J2,
  POLICY_HEURISTIC_CL,
  POLICY_HEURISTIC_ANTICL,
  POLICY_HEURISTIC_NEXG,
  POLICY_HEURISTIC_GPT,
  POLICY_HEURISTIC_GPT54,
  POLICY_HEURISTIC_GPT_PLAYMATCH_CL,
  POLICY_HEURISTIC_GPT_PLAYMATCHGO_CL,
  POLICY_HEURISTIC_GPT_PLAYMATCHSHAKE_CL,
  POLICY_HEURISTIC_GPT_PLAYMATCHGUKJIN_CL,
  POLICY_HEURISTIC_GPT_PLAYMATCHPRESIDENT_CL,
  POLICY_HEURISTIC_GPT_PLAYMATCHGUKJINPRESIDENT_CL,
  POLICY_HEURISTIC_GPT_ALLBUTGO_CL,
  POLICY_HEURISTIC_GPT_PLAYMATCHBPG_CL,
  POLICY_HEURISTIC_GEMINI
]);

/* ============================================================================
 * 3) Catalog for UI labels and policy loading config
 * ========================================================================== */
export const MODEL_CATALOG = Object.freeze({
  [POLICY_HEURISTIC_J1]: {
    labelKey: "model.heuristicJ1",
    kind: "bot_policy",
    botPolicy: POLICY_HEURISTIC_J1
  },
  [POLICY_HEURISTIC_J2]: {
    labelKey: "model.heuristicJ2",
    kind: "bot_policy",
    botPolicy: POLICY_HEURISTIC_J2
  },
  [POLICY_HEURISTIC_CL]: {
    labelKey: "model.heuristicCL",
    kind: "bot_policy",
    botPolicy: POLICY_HEURISTIC_CL
  },
  [POLICY_HEURISTIC_ANTICL]: {
    labelKey: "model.heuristicAntiCL",
    kind: "bot_policy",
    botPolicy: POLICY_HEURISTIC_ANTICL
  },
  [POLICY_HEURISTIC_NEXG]: {
    labelKey: "model.heuristicNEXg",
    kind: "bot_policy",
    botPolicy: POLICY_HEURISTIC_NEXG
  },
  [POLICY_HEURISTIC_GPT]: {
    labelKey: "model.heuristicGPT",
    kind: "bot_policy",
    botPolicy: POLICY_HEURISTIC_GPT
  },
  [POLICY_HEURISTIC_GPT54]: {
    kind: "bot_policy",
    botPolicy: POLICY_HEURISTIC_GPT54,
    label: "H-GPT54"
  },
  [POLICY_HEURISTIC_GPT_PLAYMATCH_CL]: {
    kind: "bot_policy",
    botPolicy: POLICY_HEURISTIC_GPT_PLAYMATCH_CL,
    label: "H-GPT-PlayMatch+CL"
  },
  [POLICY_HEURISTIC_GPT_PLAYMATCHGO_CL]: {
    kind: "bot_policy",
    botPolicy: POLICY_HEURISTIC_GPT_PLAYMATCHGO_CL,
    label: "H-GPT-PlayMatchGo+CL"
  },
  [POLICY_HEURISTIC_GPT_PLAYMATCHSHAKE_CL]: {
    kind: "bot_policy",
    botPolicy: POLICY_HEURISTIC_GPT_PLAYMATCHSHAKE_CL,
    label: "H-GPT-PlayMatchShake+CL"
  },
  [POLICY_HEURISTIC_GPT_PLAYMATCHGUKJIN_CL]: {
    kind: "bot_policy",
    botPolicy: POLICY_HEURISTIC_GPT_PLAYMATCHGUKJIN_CL,
    label: "H-GPT-PlayMatchGukjin+CL"
  },
  [POLICY_HEURISTIC_GPT_PLAYMATCHPRESIDENT_CL]: {
    kind: "bot_policy",
    botPolicy: POLICY_HEURISTIC_GPT_PLAYMATCHPRESIDENT_CL,
    label: "H-GPT-PlayMatchPresident+CL"
  },
  [POLICY_HEURISTIC_GPT_PLAYMATCHGUKJINPRESIDENT_CL]: {
    kind: "bot_policy",
    botPolicy: POLICY_HEURISTIC_GPT_PLAYMATCHGUKJINPRESIDENT_CL,
    label: "H-GPT-PlayMatchGukjinPresident+CL"
  },
  [POLICY_HEURISTIC_GPT_ALLBUTGO_CL]: {
    kind: "bot_policy",
    botPolicy: POLICY_HEURISTIC_GPT_ALLBUTGO_CL,
    label: "H-GPT-AllButGo+CL"
  },
  [POLICY_HEURISTIC_GPT_PLAYMATCHBPG_CL]: {
    kind: "bot_policy",
    botPolicy: POLICY_HEURISTIC_GPT_PLAYMATCHBPG_CL,
    label: "H-GPT-PlayMatchBPG+CL"
  },
  [POLICY_HEURISTIC_GEMINI]: {
    labelKey: "model.heuristicGemini",
    kind: "bot_policy",
    botPolicy: POLICY_HEURISTIC_GEMINI
  },
  [POLICY_NEAT_PHASE2_SEED9]: {
    labelKey: "model.neatPhase2Seed9",
    kind: "policy_model",
    policyPath: "/models/neat_phase2_seed9_winner_genome.json"
  },
  [POLICY_NEAT_PHASE2_SEED66]: {
    kind: "policy_model",
    policyPath: "/models/neat_phase2_seed66_winner_genome.json",
    label: "NEAT Phase2 Seed66"
  }
});

const BOT_POLICY_SET = new Set(BOT_POLICIES);
const BOT_POLICY_LOOKUP = new Map(
  BOT_POLICIES.map((policy) => [String(policy).trim().toLowerCase(), policy])
);

/* ============================================================================
 * 4) Normalization + UI helpers
 * ========================================================================== */
export function resolveBotPolicy(policy) {
  const raw = String(policy || "").trim();
  if (!raw) return null;
  const normalized = raw.toLowerCase();
  if (BOT_POLICY_LOOKUP.has(normalized)) return BOT_POLICY_LOOKUP.get(normalized);
  return null;
}

export function normalizeBotPolicy(policy) {
  const resolved = resolveBotPolicy(policy);
  if (resolved && BOT_POLICY_SET.has(resolved)) return resolved;
  return DEFAULT_BOT_POLICY;
}

export function getModelLabel(pick, language, translateFn) {
  const cfg = MODEL_CATALOG[pick] || null;
  if (!cfg) return String(pick || "AI");
  if (cfg.labelKey) return translateFn(language, cfg.labelKey, {}, String(pick));
  return String(cfg.label || pick || "AI");
}

export function buildModelOptions(language, translateFn) {
  return Object.entries(MODEL_CATALOG).map(([value, config]) => ({
    value,
    label: config?.labelKey
      ? translateFn(language, config.labelKey, {}, value)
      : String(config?.label || value)
  }));
}
