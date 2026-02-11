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

export function pushCaptured(captured, card) {
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
    const junkCandidates = giver.captured.junk.map((card, idx) => ({ idx, card, value: piValue(card) }));
    const gukjinIdx = giver.captured.five.findIndex(
      (card) => isGukjinCard(card) && !card.gukjinTransformed
    );
    const canStealGukjinAsPi = getGukjinMode(giver) === "junk" && gukjinIdx >= 0;
    if (junkCandidates.length === 0 && canStealGukjinAsPi) {
      giver.captured.five[gukjinIdx] = {
        ...giver.captured.five[gukjinIdx],
        gukjinTransformed: true
      };
      giver.gukjinMode = "five";
      giver.gukjinLocked = true;
      stealLog.push(
        `${taker.label}: ${giver.label} 국진 피 강탈 시도 -> 강탈 무효, ${giver.label} 열로 고정 전환`
      );
      remaining -= 1;
      continue;
    }

    if (junkCandidates.length === 0) break;

    junkCandidates.sort((a, b) => a.value - b.value || a.card.id.localeCompare(b.card.id));
    const pick = junkCandidates[0];
    const [stolen] = giver.captured.junk.splice(pick.idx, 1);
    taker.captured.junk.push(stolen);
    stealLog.push(
      `${taker.label}: 보너스 효과로 ${giver.label}의 피 1장 강탈 (${stolen.name}, ${piValue(stolen)}피 처리)`
    );

    remaining -= 1;
  }

  return {
    updatedPlayers: { ...players, [takerKey]: taker, [giverKey]: giver },
    stealLog
  };
}
