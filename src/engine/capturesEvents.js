import { piValue, isGukjinCard, getGukjinMode } from "./scoring.js";

/* ============================================================================
 * Capture/event helpers
 * - categorize/append captured cards
 * - steal logic for pi transfer including gukjin-as-junk mode
 * ========================================================================== */

const CAPTURE_ZONES = Object.freeze(["kwang", "five", "ribbon", "junk"]);

/* 1) Capture category helpers */
export function categoryKey(card) {
  switch (card.category) {
    case "kwang":
      return "kwang";
    case "five":
      return "five";
    case "ribbon":
      return "ribbon";
    default:
      return "junk";
  }
}

function hasCapturedCardId(captured, cardId) {
  return CAPTURE_ZONES.some((k) => (captured[k] || []).some((c) => c.id === cardId));
}

export function pushCaptured(captured, card) {
  if (!card || hasCapturedCardId(captured, card.id)) return;
  captured[categoryKey(card)].push(card);
}

export function bestMatchCard(cards) {
  return cards
    .slice()
    .sort((a, b) => piValue(b) - piValue(a) || a.id.localeCompare(b.id))[0];
}

export function needsChoice(cards) {
  if (!cards || cards.length < 2) return false;
  const kind = (c) => c.category;
  return kind(cards[0]) !== kind(cards[1]);
}

/* 2) Pi-steal transfer (supports gukjin transfer in junk mode) */
export function stealPiFromOpponent(players, takerKey, count) {
  const giverKey = takerKey === "human" ? "ai" : "human";
  const taker = { ...players[takerKey], captured: { ...players[takerKey].captured } };
  const giver = { ...players[giverKey], captured: { ...players[giverKey].captured } };

  taker.captured.junk = taker.captured.junk.slice();
  taker.captured.five = taker.captured.five.slice();
  giver.captured.junk = giver.captured.junk.slice();
  giver.captured.five = giver.captured.five.slice();

  const stealLog = [];
  let remaining = count;

  while (remaining > 0) {
    const junkOnlyCandidates = giver.captured.junk.map((card, idx) => {
      const isGukjinPi =
        !!card?.gukjinTransformed ||
        (isGukjinCard(card) && card?.category === "junk");
      return {
        idx,
        card,
        value: piValue(card),
        isGukjin: isGukjinPi,
        source: "junk"
      };
    });
    const candidates = junkOnlyCandidates.slice();

    if (getGukjinMode(giver) === "junk") {
      const gukjinIdx = giver.captured.five.findIndex((card) => isGukjinCard(card) && !card.gukjinTransformed);
      if (gukjinIdx >= 0) {
        candidates.push({
          idx: gukjinIdx,
          card: giver.captured.five[gukjinIdx],
          value: 2,
          isGukjin: true,
          source: "gukjin"
        });
      }
    }

    if (candidates.length === 0) break;

    const stealRank = (x) => {
      if (x.value === 1) return 1; // one-pi
      if (x.value === 2 && !x.isGukjin) return 2; // double-pi
      if (x.isGukjin) return 3; // gukjin
      if (x.value >= 3) return 4; // high-pi
      return 9;
    };

    candidates.sort(
      (a, b) =>
        stealRank(a) - stealRank(b) ||
        b.idx - a.idx ||
        a.card.id.localeCompare(b.card.id)
    );

    const pick = candidates[0];
    let stolen;
    if (pick.source === "gukjin") {
      const [gukjinCard] = giver.captured.five.splice(pick.idx, 1);
      stolen = {
        ...gukjinCard,
        category: "junk",
        piValue: 2,
        gukjinTransformed: true,
        name: `${gukjinCard.name} (Gukjin Pi)`
      };
    } else {
      [stolen] = giver.captured.junk.splice(pick.idx, 1);
    }

    taker.captured.junk.push(stolen);
    stealLog.push(
      `${taker.label}: stole 1 pi from ${giver.label} (${stolen.name}, value ${piValue(stolen)})`
    );
    remaining -= 1;
  }

  return {
    updatedPlayers: { ...players, [takerKey]: taker, [giverKey]: giver },
    stealLog
  };
}

