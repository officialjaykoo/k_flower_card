export const STARTING_HAND_SIZE = 10;

export function clearExpiredReveal(state, now = Date.now()) {
  const shakingAlive = state.shakingReveal && state.shakingReveal.expiresAt > now;
  const actionAlive = state.actionReveal && state.actionReveal.expiresAt > now;
  if (shakingAlive && actionAlive) return state;

  const next = { ...state };
  if (!shakingAlive) next.shakingReveal = null;
  if (!actionAlive) next.actionReveal = null;
  return next;
}

function collectExistingCardIds(state) {
  const ids = new Set();
  const push = (cards = []) => {
    cards.forEach((c) => {
      if (c?.id) ids.add(c.id);
    });
  };

  push(state.board || []);
  push(state.deck || []);
  ["human", "ai"].forEach((key) => {
    const p = state.players?.[key];
    if (!p) return;
    push(p.hand || []);
    push(p.captured?.kwang || []);
    push(p.captured?.five || []);
    push(p.captured?.ribbon || []);
    push(p.captured?.junk || []);
  });
  return ids;
}

function dedupeHandNonPass(hand = []) {
  const seen = new Set();
  const result = [];
  hand.forEach((card) => {
    if (!card) return;
    if (card.passCard) {
      result.push(card);
      return;
    }
    if (seen.has(card.id)) return;
    seen.add(card.id);
    result.push(card);
  });
  return result;
}

function makePassCard(playerKey, seed, idx) {
  return {
    id: `pass-${playerKey}-${seed}-${idx}`,
    month: 0,
    category: "junk",
    name: "Pass",
    passCard: true,
    asset: ""
  };
}

export function ensurePassCardFor(state, playerKey) {
  const player = state.players[playerKey];
  if (!player) return state;

  const expectedHandCount = Math.max(0, STARTING_HAND_SIZE - (player.turnCount || 0));
  const hand = dedupeHandNonPass(player.hand || []);

  const normalCards = hand.filter((c) => !c.passCard);
  const passCards = hand.filter((c) => c.passCard);

  let nextHand = normalCards.concat(passCards);

  if (nextHand.length > expectedHandCount) {
    const removeCount = nextHand.length - expectedHandCount;
    const passIdx = [];
    for (let i = nextHand.length - 1; i >= 0; i -= 1) {
      if (nextHand[i].passCard) passIdx.push(i);
      if (passIdx.length >= removeCount) break;
    }
    if (passIdx.length > 0) {
      const drop = new Set(passIdx);
      nextHand = nextHand.filter((_, idx) => !drop.has(idx));
    }
  }

  if (nextHand.length < expectedHandCount) {
    const addCount = expectedHandCount - nextHand.length;
    const seed = `${state.turnSeq || 0}-${state.kiboSeq || 0}-${(state.log || []).length}`;
    const existingIds = collectExistingCardIds(state);
    const added = [];
    let serial = 0;
    while (added.length < addCount) {
      const passCard = makePassCard(playerKey, seed, serial);
      serial += 1;
      if (existingIds.has(passCard.id)) continue;
      existingIds.add(passCard.id);
      added.push(passCard);
    }
    nextHand = nextHand.concat(added);
  }

  const prevHand = player.hand || [];
  if (prevHand.length === nextHand.length && prevHand.every((c, i) => c?.id === nextHand[i]?.id)) {
    return state;
  }

  const nextPlayers = {
    ...state.players,
    [playerKey]: { ...player, hand: nextHand }
  };

  return {
    ...state,
    players: nextPlayers
  };
}
