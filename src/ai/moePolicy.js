import { calculateScore } from "../engine/index.js";
import { buildDeck } from "../cards.js";

export const MOE_DEFENSE_SCORE_THRESHOLD_DEFAULT = -2;
export const MOE_ATTACK_SCORE_THRESHOLD_DEFAULT = 3;
export const MOE_RISK_THRESHOLD_DEFAULT = 1;
export const MOE_OPP_THREAT_THRESHOLD_DEFAULT = 0.55;

const FULL_DECK = buildDeck();
const RIBBON_COMBO_MONTH_SETS = Object.freeze({
  hongdan: Object.freeze([1, 2, 3]),
  cheongdan: Object.freeze([6, 9, 10]),
  chodan: Object.freeze([4, 5, 7])
});
const GODORI_MONTHS = Object.freeze([2, 4, 8]);
const GWANG_MONTHS = Object.freeze([1, 3, 8, 11, 12]);

function clamp01(value) {
  if (!Number.isFinite(value)) return 0;
  if (value <= 0) return 0;
  if (value >= 1) return 1;
  return value;
}

function findOppActor(state, actor) {
  const actorKeys = Object.keys(state?.players || {});
  return actorKeys.find((k) => k !== actor) || null;
}

function flattenCapturedIds(captured) {
  return [
    ...(captured?.kwang || []),
    ...(captured?.five || []),
    ...(captured?.ribbon || []),
    ...(captured?.junk || [])
  ]
    .map((c) => c?.id)
    .filter((id) => typeof id === "string");
}

function countComboMonths(months, ownSet) {
  let n = 0;
  for (const m of months) {
    if (ownSet.has(m)) n += 1;
  }
  return n;
}

function buildJokboStats(captured) {
  const ribbons = captured?.ribbon || [];
  const fives = captured?.five || [];
  const gwangs = captured?.kwang || [];
  const ribbonMonths = new Set(ribbons.map((c) => c?.month).filter((m) => Number.isInteger(m)));
  const fiveMonths = new Set(fives.map((c) => c?.month).filter((m) => Number.isInteger(m)));
  const gwangMonths = new Set(gwangs.map((c) => c?.month).filter((m) => Number.isInteger(m)));
  return {
    ribbonMonths,
    fiveMonths,
    gwangMonths,
    progress: {
      hongdan: countComboMonths(RIBBON_COMBO_MONTH_SETS.hongdan, ribbonMonths),
      cheongdan: countComboMonths(RIBBON_COMBO_MONTH_SETS.cheongdan, ribbonMonths),
      chodan: countComboMonths(RIBBON_COMBO_MONTH_SETS.chodan, ribbonMonths),
      godori: countComboMonths(GODORI_MONTHS, fiveMonths),
      gwang: gwangMonths.size
    }
  };
}

function buildVisibleCardIdSet(state, actor, capturedSelf, capturedOpp) {
  const visible = new Set();
  for (const c of state.players?.[actor]?.hand || []) {
    if (c?.id) visible.add(c.id);
  }
  for (const c of state.board || []) {
    if (c?.id) visible.add(c.id);
  }
  for (const id of flattenCapturedIds(capturedSelf)) visible.add(id);
  for (const id of flattenCapturedIds(capturedOpp)) visible.add(id);
  return visible;
}

function unknownMonthCategoryProbability(visibleCardIds, unknownPool, month, category) {
  if (unknownPool <= 0 || !Number.isInteger(month)) return 0;
  let available = 0;
  for (const c of FULL_DECK) {
    if (c?.month !== month) continue;
    if (category && c?.category !== category) continue;
    if (visibleCardIds.has(c?.id)) continue;
    available += 1;
  }
  return clamp01(available / unknownPool);
}

function unknownAnyMonthProbability(visibleCardIds, unknownPool, months, category) {
  if (!Array.isArray(months) || !months.length) return 0;
  let noneProb = 1;
  for (const month of months) {
    const p = unknownMonthCategoryProbability(visibleCardIds, unknownPool, month, category);
    noneProb *= 1 - p;
  }
  return clamp01(1 - noneProb);
}

function computeJokboThreatProbabilities(stats, visibleCardIds, unknownPool) {
  const rules = [
    { months: RIBBON_COMBO_MONTH_SETS.hongdan, set: stats.ribbonMonths, category: "ribbon" },
    { months: RIBBON_COMBO_MONTH_SETS.cheongdan, set: stats.ribbonMonths, category: "ribbon" },
    { months: RIBBON_COMBO_MONTH_SETS.chodan, set: stats.ribbonMonths, category: "ribbon" },
    { months: GODORI_MONTHS, set: stats.fiveMonths, category: "five" }
  ];
  let oneAwayProb = 0;
  let totalProb = 0;
  for (const r of rules) {
    const got = countComboMonths(r.months, r.set);
    const missing = r.months.filter((m) => !r.set.has(m));
    if (!missing.length) {
      oneAwayProb = 1;
      totalProb += 0.9;
      continue;
    }
    const pAny = unknownAnyMonthProbability(visibleCardIds, unknownPool, missing, r.category);
    if (got >= 2) {
      oneAwayProb = Math.max(oneAwayProb, pAny);
      totalProb += pAny * 0.72;
    } else if (got === 1) {
      totalProb += pAny * 0.24;
    }
  }

  const gotGwang = Number(stats?.progress?.gwang || 0);
  const missingGwang = GWANG_MONTHS.filter((m) => !stats.gwangMonths.has(m));
  let gwangThreatProb = 0;
  if (gotGwang >= 2) {
    gwangThreatProb = unknownAnyMonthProbability(visibleCardIds, unknownPool, missingGwang, "kwang");
  } else if (gotGwang === 1) {
    gwangThreatProb =
      unknownAnyMonthProbability(visibleCardIds, unknownPool, missingGwang, "kwang") * 0.3;
  }
  totalProb += gwangThreatProb * 0.6;
  return {
    oneAwayProb: clamp01(oneAwayProb),
    gwangThreatProb: clamp01(gwangThreatProb),
    totalProb: clamp01(totalProb)
  };
}

function toFiniteNumber(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

export function buildMoeSelectionContext(state, actor) {
  const opp = findOppActor(state, actor);
  const self = state?.players?.[actor];
  const other = state?.players?.[opp];
  if (!self || !other) {
    return {
      scoreDiff: 0,
      bakRiskCount: 0,
      oppThreat: 0,
      carryOverMultiplier: toFiniteNumber(state?.carryOverMultiplier, 1)
    };
  }

  const selfScore = calculateScore(self, other, state.ruleKey);
  const oppScore = calculateScore(other, self, state.ruleKey);
  const scoreDiff = Number(selfScore?.total || 0) - Number(oppScore?.total || 0);
  const bakRiskCount =
    (selfScore?.bak?.pi ? 1 : 0) +
    (selfScore?.bak?.gwang ? 1 : 0) +
    (selfScore?.bak?.mongBak ? 1 : 0);

  const capturedSelf = self.captured || {};
  const capturedOpp = other.captured || {};
  const jokboOppStats = buildJokboStats(capturedOpp);
  const unknownPool = (state.deck?.length || 0) + (other.hand?.length || 0);
  const visibleCardIds = buildVisibleCardIdSet(state, actor, capturedSelf, capturedOpp);
  const oppThreatProbs = computeJokboThreatProbabilities(jokboOppStats, visibleCardIds, unknownPool);
  const oppThreat = Math.max(
    Number(oppThreatProbs.totalProb || 0),
    Number(oppThreatProbs.oneAwayProb || 0),
    Number(oppThreatProbs.gwangThreatProb || 0)
  );

  return {
    scoreDiff: toFiniteNumber(scoreDiff, 0),
    bakRiskCount: toFiniteNumber(bakRiskCount, 0),
    oppThreat: clamp01(oppThreat),
    carryOverMultiplier: toFiniteNumber(state?.carryOverMultiplier, 1)
  };
}

export function selectMoeRoleByContext(context = {}, moe = {}) {
  const defenseScoreThreshold = toFiniteNumber(
    moe?.defenseScoreThreshold,
    MOE_DEFENSE_SCORE_THRESHOLD_DEFAULT
  );
  const attackScoreThreshold = toFiniteNumber(
    moe?.attackScoreThreshold,
    MOE_ATTACK_SCORE_THRESHOLD_DEFAULT
  );
  const riskThreshold = toFiniteNumber(moe?.riskThreshold, MOE_RISK_THRESHOLD_DEFAULT);
  const oppThreatThreshold = toFiniteNumber(
    moe?.oppThreatThreshold,
    MOE_OPP_THREAT_THRESHOLD_DEFAULT
  );

  const scoreDiff = toFiniteNumber(context?.scoreDiff, 0);
  const bakRiskCount = toFiniteNumber(context?.bakRiskCount, 0);
  const oppThreat = clamp01(toFiniteNumber(context?.oppThreat, 0));
  const carry = toFiniteNumber(context?.carryOverMultiplier, 1);

  const riskScore =
    bakRiskCount + (oppThreat >= oppThreatThreshold ? 1 : 0) + (carry >= 3 ? 1 : 0);
  const useDefense = riskScore >= riskThreshold || scoreDiff <= defenseScoreThreshold;
  const useAttack = !useDefense && scoreDiff >= attackScoreThreshold;
  return useDefense ? "defense" : useAttack ? "attack" : "attack";
}

export function selectMoeRoleFromState(state, actor, moe = {}) {
  return selectMoeRoleByContext(buildMoeSelectionContext(state, actor), moe);
}
