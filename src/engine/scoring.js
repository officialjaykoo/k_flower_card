import { ruleSets } from "./rules.js";
import { countComboTag } from "./combos.js";

/* ============================================================================
 * Scoring engine
 * - base score and payout multiplier
 * - bak detection
 * - pi/gukjin helper utilities
 * ========================================================================== */

const NON_BRIGHT_KWANG_ID = "L0";
const GUKJIN_CARD_ID = "I0";

/* 1) Shared card helpers */
function uniqueCards(cards = []) {
  const seen = new Set();
  return cards.filter((c) => {
    if (!c || seen.has(c.id)) return false;
    seen.add(c.id);
    return true;
  });
}

export function calculateScore(player, opponent, ruleKey) {
  const rules = ruleSets[ruleKey];
  const baseInfo = calculateBaseScore(player, rules);
  const goBonus = player.goCount;

  // base/total is base score + GO bonus; payoutTotal applies multipliers.
  const pointTotal = baseInfo.base + goBonus;

  let payoutMultiplier = 1;
  if (player.goCount >= 3) payoutMultiplier *= 2 ** (player.goCount - 2);
  if (player.events.shaking > 0) payoutMultiplier *= 2 ** player.events.shaking;
  if (player.events.bomb > 0) payoutMultiplier *= 2 ** player.events.bomb;

  const bak = detectBak(player, opponent, ruleKey);
  payoutMultiplier *= bak.multiplier;

  return {
    base: pointTotal,
    multiplier: payoutMultiplier,
    total: pointTotal,
    payoutTotal: pointTotal * payoutMultiplier,
    bak,
    breakdown: { ...baseInfo.breakdown, goBonus }
  };
}


/* 2) Base-score components */
export function calculateBaseScore(player, rules = ruleSets.A) {
  const kwang = uniqueCards(player.captured.kwang || []);
  const ribbon = uniqueCards(player.captured.ribbon || []);
  const fiveCards = scoringFiveCards(player);
  const fiveCount = fiveCards.length;
  const ribbonCount = ribbon.length;
  const junkCount = scoringPiCount(player);

  const kwangBase = kwangBaseScore(kwang);
  const fiveBase = fiveCount >= 5 ? fiveCount - 4 : 0;
  const ribbonBase = ribbonCount >= 5 ? ribbonCount - 4 : 0;
  const junkBase = junkCount >= 10 ? junkCount - 9 : 0;
  const ribbonSetBonus = ribbonBonus(ribbon);
  const fiveSetBonus = fiveBonus(fiveCards);
  const base = kwangBase + fiveBase + ribbonBase + junkBase + ribbonSetBonus + fiveSetBonus;

  return {
    base,
    breakdown: {
      kwangBase,
      fiveBase,
      ribbonBase,
      junkBase,
      ribbonSetBonus,
      fiveSetBonus,
      piCount: junkCount,
      gukjinMode: getGukjinMode(player)
    }
  };
}

function ribbonBonus(ribbons) {
  const hasRed = countComboTag(ribbons, "redRibbons") >= 3;
  const hasBlue = countComboTag(ribbons, "blueRibbons") >= 3;
  const hasPlant = countComboTag(ribbons, "plainRibbons") >= 3;
  let bonus = 0;
  if (hasRed) bonus += 3;
  if (hasBlue) bonus += 3;
  if (hasPlant) bonus += 3;
  return bonus;
}

function fiveBonus(fives) {
  return countComboTag(fives, "fiveBirds") >= 3 ? 5 : 0;
}

function kwangBaseScore(kwangCards) {
  const count = kwangCards.length;
  if (count < 3) return 0;
  if (count === 3) return kwangCards.some((c) => c?.id === NON_BRIGHT_KWANG_ID) ? 2 : 3;
  if (count === 4) return 4;
  return 15;
}

/* 3) Bak detection */
function detectBak(player, opponent, ruleKey) {
  const rules = ruleSets[ruleKey];
  const opponentPi = scoringPiCount(opponent);
  const playerPi = scoringPiCount(player);
  const playerFiveCount = scoringFiveCards(player).length;
  const opponentFiveCount = scoringFiveCards(opponent).length;

  const bak = {
    gwang:
      uniqueCards(player.captured.kwang || []).length >= 3 &&
      uniqueCards(opponent.captured.kwang || []).length === 0,
    // ?쇰컯? 怨??щ?? 臾닿?. ?? ?곷? ?쇨? 0?대㈃ 硫대컯(?쇰컯 硫댁젣).
    pi: opponentPi >= 1 && opponentPi <= 7 && playerPi >= 10,
    // User rule: mongBak is checked on win condition context without GO prerequisite.
    mongBak: opponentFiveCount === 0 && playerFiveCount >= 7
  };

  let multiplier = 1;
  if (bak.gwang) multiplier *= rules.bakMultipliers.gwang;
  if (bak.pi) multiplier *= rules.bakMultipliers.pi;
  if (bak.mongBak) multiplier *= rules.bakMultipliers.mongBak;
  return { ...bak, multiplier };
}

/* 4) Pi/gukjin helpers */
export function piValue(card) {
  const explicit = Number(card?.piValue);
  if (Number.isFinite(explicit) && explicit > 0) return explicit;
  return 1;
}

export function sumPiValues(cards = []) {
  return uniqueCards(cards).reduce((sum, card) => sum + piValue(card), 0);
}

export function isGukjinCard(card) {
  return card?.id === GUKJIN_CARD_ID;
}

export function getGukjinMode(player) {
  return player?.gukjinMode === "junk" ? "junk" : "five";
}

export function scoringFiveCards(player) {
  const fives = uniqueCards(player.captured.five || []);
  if (getGukjinMode(player) === "five") return fives;
  return fives.filter((c) => !(isGukjinCard(c) && !c.gukjinTransformed));
}

export function scoringPiCount(player) {
  const basePi = sumPiValues(player.captured.junk || []);
  if (getGukjinMode(player) !== "junk") return basePi;
  const hasGukjin = uniqueCards(player.captured.five || []).some((c) => isGukjinCard(c) && !c.gukjinTransformed);
  return hasGukjin ? basePi + 2 : basePi;
}
