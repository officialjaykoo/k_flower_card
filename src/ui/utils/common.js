import { buildDeck } from "../../cards.js";

/* ============================================================================
 * Common UI utilities
 * - participant helpers
 * - card hydration/sorting
 * ========================================================================== */

const cardAssetById = new Map(buildDeck().map((c) => [c.id, c.asset]));
const CARD_CATEGORY_ORDER = Object.freeze({ kwang: 0, five: 1, ribbon: 2, junk: 3 });

/* 1) Runtime seed helper */
export function randomSeed() {
  return Math.random().toString(36).slice(2, 10);
}

/* 2) Participant type helpers */
export function participantType(ui, key) {
  return ui.participants[key] === "ai" ? "ai" : "human";
}

export function isBotPlayer(ui, key) {
  return participantType(ui, key) === "ai";
}

/* 3) Card display helpers */
export function hydrateCard(card) {
  if (!card) return null;
  if (card.asset) return card;
  const asset = cardAssetById.get(card.id);
  return asset ? { ...card, asset } : card;
}

export function sortCards(cards) {
  return cards
    .slice()
    .sort(
      (a, b) =>
        a.month - b.month ||
        (CARD_CATEGORY_ORDER[a.category] ?? 9) - (CARD_CATEGORY_ORDER[b.category] ?? 9) ||
        a.id.localeCompare(b.id)
    );
}

