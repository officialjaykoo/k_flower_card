import { calculateScore, scoringFiveCards, scoringPiCount } from "../engine/index.js";
import { COMBO_MONTH_SETS, countComboTag, missingComboMonths } from "../engine/combos.js";
import { STARTING_GOLD } from "../engine/economy.js";

export const GWANG_MONTHS = Object.freeze([1, 3, 8, 11, 12]);
export const GOLD_RISK_THRESHOLD_RATIO = 0.1;
export const GUKJIN_CARD_ID = "I0";
export const GUKJIN_ANALYSIS_BOARD_WEIGHT = 0.5;
export const COMBO_REQUIRED_CATEGORY = Object.freeze({
  redRibbons: "ribbon",
  blueRibbons: "ribbon",
  plainRibbons: "ribbon",
  fiveBirds: "five"
});
export const HIGH_PI_CARD_IDS = Object.freeze(["M0", "M1", "K1", "L3", GUKJIN_CARD_ID]);

export function capturedCountByCategory(player, category) {
  if (!player?.captured) return 0;
  if (category === "junk") return scoringPiCount(player);
  return (player.captured[category] || []).length;
}

export function scoringFiveCount(player) {
  return scoringFiveCards(player).length;
}

export function hasCardId(cards, cardId) {
  return (cards || []).some((c) => c?.id === cardId);
}

export function hasGukjinInCaptured(captured) {
  if (!captured) return false;
  for (const cat of ["kwang", "five", "ribbon", "junk"]) {
    if (hasCardId(captured[cat] || [], GUKJIN_CARD_ID)) return true;
  }
  return false;
}

export function fiveCountIncludingCapturedGukjin(player) {
  const fiveCount = (player?.captured?.five || []).length;
  if (!player?.captured) return fiveCount;
  if (hasCardId(player.captured.five || [], GUKJIN_CARD_ID)) return fiveCount;
  return hasGukjinInCaptured(player.captured) ? fiveCount + 1 : fiveCount;
}

export function otherPlayerKey(playerKey) {
  return playerKey === "human" ? "ai" : "human";
}

export function gatherGukjinZoneFlags(state, playerKey) {
  const opp = otherPlayerKey(playerKey);
  const selfPlayer = state.players?.[playerKey];
  const oppPlayer = state.players?.[opp];
  return {
    selfHand: hasCardId(selfPlayer?.hand || [], GUKJIN_CARD_ID),
    selfCaptured: hasGukjinInCaptured(selfPlayer?.captured),
    oppCaptured: hasGukjinInCaptured(oppPlayer?.captured),
    board: hasCardId(state.board || [], GUKJIN_CARD_ID)
  };
}

export function hasAnyGukjinFlag(flags) {
  return !!flags && Object.values(flags).some(Boolean);
}

export function forceGukjinModeState(state, modeByPlayer = {}) {
  const keys = Object.keys(modeByPlayer || {});
  if (!keys.length) return state;
  let changed = false;
  const nextPlayers = { ...state.players };
  for (const key of keys) {
    const mode = modeByPlayer[key];
    if (mode !== "five" && mode !== "junk") continue;
    const player = state.players?.[key];
    if (!player) continue;
    if (player.gukjinMode === mode) continue;
    nextPlayers[key] = { ...player, gukjinMode: mode };
    changed = true;
  }
  if (!changed) return state;
  return { ...state, players: nextPlayers };
}

export function buildGukjinScenario(state, playerKey, selfMode, oppMode, zoneFlags) {
  const opp = otherPlayerKey(playerKey);
  const scenarioState = forceGukjinModeState(state, { [playerKey]: selfMode, [opp]: oppMode });
  const selfPlayer = scenarioState.players?.[playerKey];
  const oppPlayer = scenarioState.players?.[opp];
  if (!selfPlayer || !oppPlayer) return null;

  let selfPi = capturedCountByCategory(selfPlayer, "junk");
  let selfFive = scoringFiveCount(selfPlayer);
  let oppPi = capturedCountByCategory(oppPlayer, "junk");
  let oppFive = scoringFiveCount(oppPlayer);

  if (zoneFlags?.selfHand) {
    if (selfMode === "junk") selfPi += 2;
    else selfFive += 1;
  }
  if (zoneFlags?.board) {
    if (selfMode === "junk") selfPi += 2 * GUKJIN_ANALYSIS_BOARD_WEIGHT;
    else selfFive += 1 * GUKJIN_ANALYSIS_BOARD_WEIGHT;
  }

  const selfScoreInfo = calculateScore(selfPlayer, oppPlayer, scenarioState.ruleKey);
  const oppScoreInfo = calculateScore(oppPlayer, selfPlayer, scenarioState.ruleKey);

  return {
    selfMode,
    oppMode,
    selfPi,
    selfFive,
    oppPi,
    oppFive,
    myScore: Number(selfScoreInfo?.total || 0),
    oppScore: Number(oppScoreInfo?.total || 0),
    mongRiskSelf: selfFive <= 0 && oppFive >= 6,
    canMongBakSelf: selfFive >= 7 && oppFive <= 0
  };
}

export function summarizeScenarioRange(scenarios, field) {
  if (!scenarios.length) return { min: 0, max: 0, avg: 0 };
  const vals = scenarios.map((s) => Number(s?.[field] || 0));
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const avg = vals.reduce((sum, v) => sum + v, 0) / vals.length;
  return { min, max, avg };
}

export function pickGukjinScenario(scenarios, selfMode, oppMode) {
  return (scenarios || []).find((s) => s?.selfMode === selfMode && s?.oppMode === oppMode) || null;
}

export function preferredGukjinModeByFiveCount(fiveCount) {
  return Number(fiveCount || 0) >= 7 ? "five" : "junk";
}

export function currentScoreTotal(state, playerKey) {
  const opp = otherPlayerKey(playerKey);
  const selfScore = calculateScore(state.players[playerKey], state.players[opp], state.ruleKey);
  return Number(selfScore?.total || 0);
}

export function comboProgress(player) {
  const ribbons = player?.captured?.ribbon || [];
  const fives = player?.captured?.five || [];
  return {
    redRibbons: countComboTag(ribbons, "redRibbons"),
    blueRibbons: countComboTag(ribbons, "blueRibbons"),
    plainRibbons: countComboTag(ribbons, "plainRibbons"),
    fiveBirds: countComboTag(fives, "fiveBirds")
  };
}

export function hasCapturedCategoryMonth(player, category, month) {
  return (player?.captured?.[category] || []).some((c) => c?.month === month);
}

export function availableMissingMonths(missingMonths, requiredCategory, defenderPlayer) {
  if (!Array.isArray(missingMonths) || !missingMonths.length) return [];
  if (!defenderPlayer) return missingMonths;
  return missingMonths.filter((month) => !hasCapturedCategoryMonth(defenderPlayer, requiredCategory, month));
}

export function missingGwangMonths(player) {
  const own = new Set((player?.captured?.kwang || []).map((c) => c?.month).filter((m) => Number.isInteger(m)));
  return GWANG_MONTHS.filter((m) => !own.has(m));
}

export function isSecondMover(state, playerKey) {
  const first = state?.startingTurnKey;
  if (first === "human" || first === "ai") return first !== playerKey;
  return false;
}

export function resolveInitialGoldBase(state) {
  const configured = Number(state?.initialGoldBase);
  if (Number.isFinite(configured) && configured > 0) return configured;
  return STARTING_GOLD;
}

export function monthCounts(cards) {
  const map = new Map();
  for (const card of cards || []) {
    const month = Number(card?.month || 0);
    if (month < 1) continue;
    map.set(month, Number(map.get(month) || 0) + 1);
  }
  return map;
}

export function normalizeMonthCountMap(value, fallbackCards = []) {
  if (value instanceof Map) return value;
  return monthCounts(fallbackCards);
}

export function capturedMonthCounts(state) {
  const map = new Map();
  for (const side of ["human", "ai"]) {
    const captured = state?.players?.[side]?.captured || {};
    for (const cat of ["kwang", "five", "ribbon", "junk"]) {
      for (const card of captured[cat] || []) {
        const month = Number(card?.month || 0);
        if (month < 1) continue;
        map.set(month, Number(map.get(month) || 0) + 1);
      }
    }
  }
  return map;
}

export function monthStrategicPriority(month) {
  const m = Number(month || 0);
  if (m === 1 || m === 3 || m === 8 || m === 11 || m === 12) return 1.0;
  if (m === 2 || m === 4 || m === 5 || m === 6 || m === 7 || m === 9 || m === 10) return 0.7;
  return 0.5;
}

export function clamp01(v) {
  const x = Number(v || 0);
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  return x;
}

export function hasMonthCard(cards, month) {
  return (cards || []).some((c) => c?.month === month);
}

export function hasMonthCategoryCard(cards, month, category) {
  return (cards || []).some((c) => c?.month === month && c?.category === category);
}

export function boardHighValueThreatForPlayer(state, playerKey, blockingMonthsAgainst) {
  const handMonths = new Set((state.players?.[playerKey]?.hand || []).map((c) => c.month));
  const blockerKey = playerKey === "human" ? "ai" : "human";
  const blockMonths = blockingMonthsAgainst(state.players?.[playerKey], state.players?.[blockerKey]);
  for (const c of state.board || []) {
    if (!handMonths.has(c.month)) continue;
    if (c.category === "kwang" || c.category === "five") return true;
    if (blockMonths.has(c.month)) return true;
  }
  return false;
}

export function isOppVulnerableForBigGo(state, playerKey) {
  const opp = playerKey === "human" ? "ai" : "human";
  const oppPlayer = state.players?.[opp];
  const oppPi = capturedCountByCategory(oppPlayer, "junk");
  const oppGwang = capturedCountByCategory(oppPlayer, "kwang");
  return oppPi <= 5 || oppGwang === 0;
}

export function junkPiValue(card) {
  if (!card || card.category !== "junk") return 0;
  const explicit = Number(card.piValue);
  if (Number.isFinite(explicit) && explicit > 0) return explicit;
  return 1;
}

export function cardCaptureValue(card) {
  if (!card) return 0;
  if (card.category === "kwang") return 6;
  if (card.category === "five") return 4;
  if (card.category === "ribbon") return 2;
  if (card.category === "junk") return junkPiValue(card);
  if (card.bonus?.stealPi) return 3 + Number(card.bonus.stealPi || 0);
  return 0;
}

export function boardMatchesByMonth(state) {
  const map = new Map();
  for (const card of state.board || []) {
    const month = card?.month;
    if (month == null) continue;
    const list = map.get(month) || [];
    list.push(card);
    map.set(month, list);
  }
  return map;
}

export function monthBoardGain(state, month) {
  const cards = (state.board || []).filter((c) => c.month === month);
  return cards.reduce((sum, c) => sum + cardCaptureValue(c), 0);
}

export function selectBestMonth(state, months) {
  if (!months?.length) return null;
  let best = months[0];
  let bestScore = monthBoardGain(state, best);
  for (const m of months.slice(1)) {
    const score = monthBoardGain(state, m);
    if (score > bestScore) {
      best = m;
      bestScore = score;
    }
  }
  return best;
}

export function countKnownMonthCards(state, month) {
  let count = 0;
  for (const c of state.board || []) if (c?.month === month) count += 1;
  for (const key of ["human", "ai"]) {
    const player = state.players?.[key];
    for (const c of player?.hand || []) if (c?.month === month) count += 1;
    const cap = player?.captured || {};
    for (const cat of ["kwang", "five", "ribbon", "junk"]) {
      for (const c of cap[cat] || []) if (c?.month === month) count += 1;
    }
  }
  return count;
}

export function ownComboOpportunityScore(state, playerKey, month) {
  const player = state.players?.[playerKey];
  const p = comboProgress(player);
  let score = 0;
  if (COMBO_MONTH_SETS.redRibbons.has(month)) {
    if (p.redRibbons >= 2) score += 1.1;
    else if (p.redRibbons === 1) score += 0.25;
  }
  if (COMBO_MONTH_SETS.blueRibbons.has(month)) {
    if (p.blueRibbons >= 2) score += 1.1;
    else if (p.blueRibbons === 1) score += 0.25;
  }
  if (COMBO_MONTH_SETS.plainRibbons.has(month)) {
    if (p.plainRibbons >= 2) score += 1.0;
    else if (p.plainRibbons === 1) score += 0.2;
  }
  if (COMBO_MONTH_SETS.fiveBirds.has(month)) {
    if (p.fiveBirds >= 2) score += 1.25;
    else if (p.fiveBirds === 1) score += 0.3;
  }
  return score;
}

export function shakingImmediateGainScore(state, playerKey, month) {
  const player = state.players?.[playerKey];
  const monthCards = (player?.hand || []).filter((c) => c.month === month);
  const deckCount = state.deck?.length || 0;
  const known = countKnownMonthCards(state, month);
  const unseen = Math.max(0, 4 - known);
  const hasHighCard = monthCards.some((c) => c.category === "kwang" || c.category === "five");
  const piPayload = monthCards.reduce((sum, c) => sum + junkPiValue(c), 0);
  const flipMatchChance = deckCount > 0 ? Math.min(1, unseen / deckCount) : 0;
  let score = flipMatchChance * (2.1 + piPayload * 0.35 + (hasHighCard ? 0.8 : 0));
  if (monthCards.some((c) => c.category === "junk" && junkPiValue(c) >= 2)) score += 0.25;
  if (unseen === 0) score -= 0.7;
  return score;
}

export function pickRandom(arr) {
  if (!arr || arr.length === 0) return null;
  return [...arr].sort((a, b) => {
    const ak = String(a?.id ?? a ?? "");
    const bk = String(b?.id ?? b ?? "");
    return ak.localeCompare(bk);
  })[0];
}
