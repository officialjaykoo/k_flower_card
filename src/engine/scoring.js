import { ruleSets } from "./rules.js";

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

  // base/total은 기본점수+GO 누적값, payoutTotal은 배수(고/흔들기/폭탄/박) 적용 후 금액 계산용.
  const pointTotal = baseInfo.base + goBonus;

  let payoutMultiplier = 1;
  if (player.goCount >= 3) payoutMultiplier *= player.goCount - 1;
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
  const months = new Set(ribbons.map((c) => c.month));
  const hasRed = [1, 2, 3].every((m) => months.has(m));
  const hasBlue = [6, 9, 10].every((m) => months.has(m));
  const hasPlant = [4, 5, 7].every((m) => months.has(m));
  let bonus = 0;
  if (hasRed) bonus += 3;
  if (hasBlue) bonus += 3;
  if (hasPlant) bonus += 3;
  return bonus;
}

function fiveBonus(fives) {
  const months = new Set(fives.map((c) => c.month));
  return [2, 4, 8].every((m) => months.has(m)) ? 5 : 0;
}

function kwangBaseScore(kwangCards) {
  const count = kwangCards.length;
  if (count < 3) return 0;
  if (count === 3) return kwangCards.some((c) => c.month === 10) ? 2 : 3;
  if (count === 4) return 4;
  return 15;
}

function detectBak(player, opponent, ruleKey) {
  const rules = ruleSets[ruleKey];
  const opponentPi = scoringPiCount(opponent);
  const playerPi = scoringPiCount(player);
  const playerFiveCount = scoringFiveCards(player).length;
  const opponentFiveCount = scoringFiveCards(opponent).length;

  const bakEnabled = (player.goCount || 0) > 0;
  const bak = {
    gwang:
      bakEnabled &&
      uniqueCards(player.captured.kwang || []).length >= 3 &&
      uniqueCards(opponent.captured.kwang || []).length === 0,
    pi: bakEnabled && opponentPi <= 7 && playerPi >= 10,
    mongBak: bakEnabled && opponentFiveCount === 0 && playerFiveCount >= 7
  };

  let multiplier = 1;
  if (bak.gwang) multiplier *= rules.bakMultipliers.gwang;
  if (bak.pi) multiplier *= rules.bakMultipliers.pi;
  if (bak.mongBak) multiplier *= rules.bakMultipliers.mongBak;
  return { ...bak, multiplier };
}

export function piValue(card) {
  if (card.tripleJunk) return 3;
  if (card.doubleJunk) return 2;
  return 1;
}

export function sumPiValues(cards = []) {
  return uniqueCards(cards).reduce((sum, card) => sum + piValue(card), 0);
}

export function isGukjinCard(card) {
  return card.month === 9 && card.category === "five";
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
