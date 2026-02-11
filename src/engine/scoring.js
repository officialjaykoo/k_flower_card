import { ruleSets } from "./rules.js";

export function calculateScore(player, opponent, ruleKey) {
  const rules = ruleSets[ruleKey];
  const baseInfo = calculateBaseScore(player, rules);
  const goBonus = player.goCount;

  let multiplier = 1;
  if (player.goCount >= 3) multiplier *= player.goCount - 1;
  if (player.events.shaking > 0) multiplier *= 2 ** player.events.shaking;
  if (player.events.bomb > 0) multiplier *= 2 ** player.events.bomb;
  if (rules.useEarlyStop && opponent.goCount > 0) multiplier *= rules.goStopMultiplier;

  const bak = detectBak(player, opponent, ruleKey);
  multiplier *= bak.multiplier;

  return {
    base: baseInfo.base + goBonus,
    multiplier,
    total: (baseInfo.base + goBonus) * multiplier,
    bak,
    breakdown: { ...baseInfo.breakdown, goBonus }
  };
}

export function calculateBaseScore(player, rules = ruleSets.A) {
  const { kwang, ribbon } = player.captured;
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
  const eventBonus =
    player.events.ttak * rules.ttakBonus +
    player.events.ppuk * rules.ppukBonus +
    player.events.jjob * rules.jjobBonus;

  const base = kwangBase + fiveBase + ribbonBase + junkBase + ribbonSetBonus + fiveSetBonus + eventBonus;

  return {
    base,
    breakdown: {
      kwangBase,
      fiveBase,
      ribbonBase,
      junkBase,
      ribbonSetBonus,
      fiveSetBonus,
      eventBonus,
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

  const bak = {
    gwang: player.captured.kwang.length >= 3 && opponent.captured.kwang.length === 0,
    pi: opponentPi <= 7 && playerPi >= 10,
    mongBak: opponentFiveCount === 0 && playerFiveCount >= 7
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

export function isGukjinCard(card) {
  return card.month === 9 && card.category === "five";
}

export function getGukjinMode(player) {
  return player?.gukjinMode === "junk" ? "junk" : "five";
}

export function scoringFiveCards(player) {
  const fives = player.captured.five || [];
  if (getGukjinMode(player) === "five") return fives;
  return fives.filter((c) => !(isGukjinCard(c) && !c.gukjinTransformed));
}

export function scoringPiCount(player) {
  const basePi = (player.captured.junk || []).reduce((sum, c) => sum + piValue(c), 0);
  if (getGukjinMode(player) !== "junk") return basePi;
  const hasGukjin = (player.captured.five || []).some((c) => isGukjinCard(c) && !c.gukjinTransformed);
  return hasGukjin ? basePi + 2 : basePi;
}
