import { buildDeck } from "../../cards.js";

const cardAssetById = new Map(buildDeck().map((c) => [c.id, c.asset]));

export function randomSeed() {
  return Math.random().toString(36).slice(2, 10);
}

export function participantType(ui, key) {
  return ui.participants[key] === "ai" ? "ai" : "human";
}

export function isBotPlayer(ui, key) {
  return participantType(ui, key) === "ai";
}

export function hydrateCard(card) {
  if (!card) return null;
  if (card.asset) return card;
  const asset = cardAssetById.get(card.id);
  return asset ? { ...card, asset } : card;
}

export function sortCards(cards) {
  const categoryOrder = { kwang: 0, five: 1, ribbon: 2, junk: 3 };
  return cards
    .slice()
    .sort(
      (a, b) =>
        a.month - b.month ||
        (categoryOrder[a.category] ?? 9) - (categoryOrder[b.category] ?? 9) ||
        a.id.localeCompare(b.id)
    );
}

