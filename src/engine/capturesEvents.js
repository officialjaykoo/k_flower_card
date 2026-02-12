import { piValue, isGukjinCard, getGukjinMode } from "./scoring.js";

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
  return ["kwang", "five", "ribbon", "junk"].some((k) => (captured[k] || []).some((c) => c.id === cardId));
}

export function pushCaptured(captured, card) {
  if (!card || hasCapturedCardId(captured, card.id)) return;
  captured[categoryKey(card)].push(card);
}

export function shouldPromptGukjinChoice(player) {
  if (!player || player.gukjinLocked) return false;
  return (player.captured.five || []).some((c) => isGukjinCard(c) && !c.gukjinTransformed);
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
      return {
        idx,
        card,
        value: piValue(card),
        isGukjin: false,
        source: "junk"
      };
    });
    const candidates = junkOnlyCandidates.slice();

    if (getGukjinMode(giver) === "junk") {
      const gukjinIdx = giver.captured.five.findIndex((card) => isGukjinCard(card) && !card.gukjinTransformed);
      if (gukjinIdx >= 0) {
        // 국진쌍피 단독(일반피/쌍피/삼피 없음)인 경우에만 열로 도망 가능.
        if (junkOnlyCandidates.length === 0) {
          giver.captured.five[gukjinIdx] = {
            ...giver.captured.five[gukjinIdx],
            gukjinTransformed: true
          };
          giver.gukjinMode = "five";
          giver.gukjinLocked = true;
          stealLog.push(`${giver.label}: 국진쌍피 단독 보유로 열 전환(강탈 회피)`);
          remaining -= 1;
          continue;
        }
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
      if (x.value === 1) return 1; // 일반피
      if (x.value === 2 && !x.isGukjin) return 2; // 쌍피
      if (x.isGukjin) return 3; // 국진
      if (x.value >= 3) return 4; // 삼피
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
        doubleJunk: true,
        gukjinTransformed: true,
        name: `${gukjinCard.name} (국진피)`
      };
    } else {
      [stolen] = giver.captured.junk.splice(pick.idx, 1);
    }

    taker.captured.junk.push(stolen);
    stealLog.push(`${taker.label}: ${giver.label}에게서 피 1장 강탈 (${stolen.name}, ${piValue(stolen)}피 처리)`);
    remaining -= 1;
  }

  return {
    updatedPlayers: { ...players, [takerKey]: taker, [giverKey]: giver },
    stealLog
  };
}
