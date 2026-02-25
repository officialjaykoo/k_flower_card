// heuristicV7_math.js - V7 gold-pressure helpers

function resolveOpponentKey(state, playerKey) {
  if (playerKey === "human") return "ai";
  if (playerKey === "ai") return "human";

  const keys = Object.keys(state?.players || {});
  if (keys.length !== 2 || !keys.includes(playerKey)) return null;
  return keys.find((k) => k !== playerKey) || null;
}

function toFinite(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function inferCurrentScore(state, playerKey, options = {}) {
  const fromOption = toFinite(options.currentScore, NaN);
  if (Number.isFinite(fromOption)) return Math.max(0, fromOption);

  const fromPlayerScore = toFinite(state?.players?.[playerKey]?.score, NaN);
  if (Number.isFinite(fromPlayerScore)) return Math.max(0, fromPlayerScore);

  const fromPlayerTotal = toFinite(state?.players?.[playerKey]?.totalScore, NaN);
  if (Number.isFinite(fromPlayerTotal)) return Math.max(0, fromPlayerTotal);

  const fromRootScore = toFinite(state?.score?.[playerKey], NaN);
  if (Number.isFinite(fromRootScore)) return Math.max(0, fromRootScore);

  return NaN;
}

function inferMultiplier(state, playerKey) {
  const player = state?.players?.[playerKey] || {};
  let multiplier = Math.max(1, toFinite(state?.multiplier, 1));

  const carry = Math.max(1, toFinite(state?.carryOverMultiplier, 1));
  multiplier *= carry;

  const goCount = Math.max(0, Math.floor(toFinite(player?.goCount, 0)));
  if (goCount === 3) multiplier *= 2;
  if (goCount === 4) multiplier *= 4;
  if (goCount >= 5) multiplier *= 8;

  const shakingCount = Math.max(0, Math.floor(toFinite(player?.shakingCount, 0)));
  if (shakingCount > 0) multiplier *= 2;

  if (player?.isBomb === true || player?.bombDeclared === true) multiplier *= 2;

  return Math.max(1, multiplier);
}

export function canBankruptOpponentByStopV7(state, playerKey, options = {}) {
  if (!state?.players || !state.players[playerKey]) return false;
  const oppKey = resolveOpponentKey(state, playerKey);
  if (!oppKey) return false;

  const oppGold = toFinite(state?.players?.[oppKey]?.gold, 0);
  if (oppGold <= 0) return true;

  // In explicit go-stop phase, exact simulator result has highest priority.
  if (state?.phase === "go-stop" && typeof options.fallbackExact === "function") {
    const exact = !!options.fallbackExact(state, playerKey);
    if (exact) return true;
  }

  const score = inferCurrentScore(state, playerKey, options);
  if (!Number.isFinite(score) || score <= 0) {
    if (typeof options.fallbackExact === "function") {
      return !!options.fallbackExact(state, playerKey);
    }
    return false;
  }

  const betPerScore = Math.max(1, toFinite(state?.betAmount, toFinite(options.defaultBetAmount, 100)));
  const multiplier = inferMultiplier(state, playerKey);
  const safetyMargin = Math.min(1, Math.max(0.5, toFinite(options.safetyMargin, 0.9)));
  const expectedDamage = score * betPerScore * multiplier;

  return expectedDamage >= oppGold * safetyMargin;
}

// Threshold helper used by shouldGoV7 (Predatory Efficiency).
export function computePrudentGoThresholdV7(options = {}) {
  let threshold = toFinite(options.base, 0.3);
  const myScore = toFinite(options.myScore, 0);
  const deckCount = toFinite(options.deckCount, 0);
  if (myScore >= 7) threshold += 0.05;
  if (deckCount < 6) threshold -= 0.05;
  return Math.max(0.2, Math.min(0.6, threshold));
}

function countComboTag(cards, tag) {
  let n = 0;
  for (const card of cards || []) {
    if (Array.isArray(card?.comboTags) && card.comboTags.includes(tag)) n += 1;
  }
  return n;
}

// Detect "opponent can likely score soon" state (one-two card away pressure).
export function isOpponentImminentV7(state, playerKey, oppProgress = null) {
  const oppKey = resolveOpponentKey(state, playerKey);
  if (!oppKey) return false;
  const opp = state?.players?.[oppKey];
  if (!opp) return false;

  const junkCount = (opp?.captured?.junk || []).length;
  if (junkCount >= 8) return true;

  const ribbons = opp?.captured?.ribbon || [];
  const fives = opp?.captured?.five || [];
  const kwang = opp?.captured?.kwang || [];
  const red = countComboTag(ribbons, "redRibbons");
  const blue = countComboTag(ribbons, "blueRibbons");
  const plain = countComboTag(ribbons, "plainRibbons");
  const birds = countComboTag(fives, "fiveBirds");
  if (red >= 2 || blue >= 2 || plain >= 2 || birds >= 2) return true;
  if (kwang.length >= 2) return true;

  const threat = toFinite(oppProgress?.threat, 0);
  if (threat >= 0.32) return true;

  let topUrgency = 0;
  const monthUrgency = oppProgress?.monthUrgency;
  if (monthUrgency && typeof monthUrgency.values === "function") {
    for (const v of monthUrgency.values()) topUrgency = Math.max(topUrgency, toFinite(v, 0));
  }
  return topUrgency >= 24;
}

export function evaluateGoldPotentialV7(_state, _playerKey, candidateCard) {
  if (!candidateCard) return 0;

  let potential = 0;
  const piValue = toFinite(candidateCard?.piValue, 0);
  const isDoublePi =
    candidateCard?.isDouble === true ||
    candidateCard?.id === "I0" ||
    (candidateCard?.category === "junk" && piValue >= 2);
  const stealPi = toFinite(candidateCard?.bonus?.stealPi, 0);

  if (isDoublePi) potential += 20;
  if (stealPi > 0 || candidateCard?.bonus === true) potential += 30;
  if (candidateCard?.category === "kwang") potential += 12;
  if (candidateCard?.category === "five" || candidateCard?.category === "ribbon") potential += 8;

  return potential;
}
