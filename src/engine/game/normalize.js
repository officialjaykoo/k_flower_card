import { flattenPackedJunk, junkRowCount } from "../../shared/junkLayout.js";

export function normalizeUniqueCardZones(players, board, deck) {
  const seen = new Set();

  const dedupeReal = (cards = []) => {
    const next = [];
    cards.forEach((card) => {
      if (!card?.id || seen.has(card.id)) return;
      seen.add(card.id);
      next.push(card);
    });
    return next;
  };

  const dedupeHand = (cards = []) => {
    const passSeen = new Set();
    const next = [];
    cards.forEach((card) => {
      if (!card) return;
      if (card.passCard) {
        if (!card.id || passSeen.has(card.id)) return;
        passSeen.add(card.id);
        next.push(card);
        return;
      }
      if (!card.id || seen.has(card.id)) return;
      seen.add(card.id);
      next.push(card);
    });
    return next;
  };

  // Priority order: hand -> captured -> board -> deck
  const nextHumanHand = dedupeHand(players.human.hand || []);
  const nextAiHand = dedupeHand(players.ai.hand || []);

  const nextHumanCaptured = {
    ...players.human.captured,
    kwang: dedupeReal(players.human.captured?.kwang || []),
    five: dedupeReal(players.human.captured?.five || []),
    ribbon: dedupeReal(players.human.captured?.ribbon || []),
    junk: dedupeReal(players.human.captured?.junk || [])
  };

  const nextAiCaptured = {
    ...players.ai.captured,
    kwang: dedupeReal(players.ai.captured?.kwang || []),
    five: dedupeReal(players.ai.captured?.five || []),
    ribbon: dedupeReal(players.ai.captured?.ribbon || []),
    junk: dedupeReal(players.ai.captured?.junk || [])
  };

  const nextBoard = dedupeReal(board || []);
  const nextDeck = dedupeReal(deck || []);

  return {
    players: {
      ...players,
      human: {
        ...players.human,
        hand: nextHumanHand,
        captured: nextHumanCaptured
      },
      ai: {
        ...players.ai,
        hand: nextAiHand,
        captured: nextAiCaptured
      }
    },
    board: nextBoard,
    deck: nextDeck
  };
}

export function normalizeJunkByRowChange(nextPlayers, prevRowCounts) {
  const normalized = { ...nextPlayers };
  ["human", "ai"].forEach((key) => {
    const player = nextPlayers[key];
    if (!player) return;
    const currentRows = junkRowCount(player.captured?.junk || []);
    if (currentRows !== (prevRowCounts?.[key] ?? currentRows)) {
      normalized[key] = {
        ...player,
        captured: {
          ...player.captured,
          junk: flattenPackedJunk(player.captured?.junk || [])
        }
      };
    }
  });
  return normalized;
}

export function packCard(card) {
  return {
    id: card.id,
    month: card.month,
    category: card.category,
    name: card.name,
    asset: card.asset || null,
    passCard: !!card.passCard
  };
}

export function getPrevJunkRows(players) {
  return {
    human: junkRowCount(players.human.captured?.junk || []),
    ai: junkRowCount(players.ai.captured?.junk || [])
  };
}
