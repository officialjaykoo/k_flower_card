import { buildDeck } from "../cards.js";

/* ============================================================================
 * Combo metadata/helpers
 * - Centralized combo month maps from deck definition
 * - Card-tag and missing-month utility helpers
 * ========================================================================== */

const COMBO_KEYS = ["redRibbons", "blueRibbons", "plainRibbons", "fiveBirds"];

/* 1) Card/tag helpers */
function uniqueCards(cards = []) {
  const seen = new Set();
  return cards.filter((card) => {
    if (!card) return false;
    const key = card.id || `${card.month}:${card.category}:${card.name || ""}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export function hasComboTag(card, tag) {
  return Array.isArray(card?.comboTags) && card.comboTags.includes(tag);
}

export function countComboTag(cards = [], tag) {
  return uniqueCards(cards).reduce((count, card) => (hasComboTag(card, tag) ? count + 1 : count), 0);
}

/* 2) Combo month maps built once at module load */
function buildComboMonthMap() {
  const map = Object.fromEntries(COMBO_KEYS.map((tag) => [tag, new Set()]));
  for (const card of buildDeck()) {
    const tags = Array.isArray(card.comboTags) ? card.comboTags : [];
    for (const tag of tags) {
      if (map[tag]) map[tag].add(card.month);
    }
  }
  return Object.fromEntries(
    COMBO_KEYS.map((tag) => [tag, Array.from(map[tag]).sort((a, b) => a - b)])
  );
}

const comboMonthMap = buildComboMonthMap();

export const COMBO_MONTHS = Object.freeze(comboMonthMap);

export const COMBO_MONTH_SETS = Object.freeze(
  Object.fromEntries(COMBO_KEYS.map((tag) => [tag, new Set(COMBO_MONTHS[tag])]))
);

/* 3) Public lookup helpers */
export function missingComboMonths(cards = [], tag) {
  const months = COMBO_MONTHS[tag] || [];
  const ownMonths = new Set(
    uniqueCards(cards)
      .filter((card) => hasComboTag(card, tag))
      .map((card) => card.month)
  );
  return months.filter((month) => !ownMonths.has(month));
}
