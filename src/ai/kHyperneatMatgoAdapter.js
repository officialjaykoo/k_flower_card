import { buildDeck } from "../cards.js";
import { calculateBaseScore, calculateScore, isGukjinCard, ruleSets } from "../engine/index.js";
import { parsePlaySpecialCandidate } from "./evalCore/sharedGameHelpers.js";

const CANONICAL_DECK = buildDeck("original");
const CARD_INDEX = new Map(CANONICAL_DECK.map((card, index) => [String(card?.id || ""), index]));
const CARD_META = new Map(
  CANONICAL_DECK.map((card) => [
    String(card?.id || ""),
    Object.freeze({
      id: String(card?.id || ""),
      month: Number(card?.month || 0),
      category: String(card?.category || ""),
      piValue: Number(card?.piValue || 0),
      bonusStealPi: Number(card?.bonus?.stealPi || 0),
      comboTags: Object.freeze([...(card?.comboTags || [])]),
    }),
  ]),
);
const CARD_SCALE = Math.max(1, CANONICAL_DECK.length);
const HAND_FEATURE_PATTERN = /^input_hand_(month|kwang|five|ribbon|junk|pi|bonus|red_ribbons|blue_ribbons|plain_ribbons|five_birds)$/;
const FOCUS_FEATURE_PATTERN = /^input_focus_(month|kwang|five|ribbon|junk|pi|bonus|red_ribbons|blue_ribbons|plain_ribbons|five_birds|board_match_count|board_capture_value|board_single_match|board_multi_match)$/;
const SEMANTIC_BIN_PATTERN = /^input_(board|captured_self|captured_opp)_(month|type|combo)_bin$/;
const NON_BRIGHT_KWANG_ID = "L0";
const BONUS_MONTH = 13;
const TYPE_BIN_KEYS = Object.freeze(["kwang", "five", "ribbon", "pi_value"]);
const COMBO_BIN_KEYS = Object.freeze(["red_ribbons", "blue_ribbons", "plain_ribbons", "five_birds", "kwang_combo"]);
const RULE_SCORE_FEATURES = Object.freeze([
  "self_go_bonus",
  "self_go_multiplier",
  "self_shaking_multiplier",
  "self_bomb_multiplier",
  "self_bak_multiplier",
  "self_pi_bak",
  "self_gwang_bak",
  "self_mong_bak",
  "self_five_base",
  "self_ribbon_base",
  "self_junk_base",
  "self_red_ribbons",
  "self_blue_ribbons",
  "self_plain_ribbons",
  "self_five_birds",
  "self_kwang",
  "opp_go_bonus",
  "opp_go_multiplier",
  "opp_shaking_multiplier",
  "opp_bomb_multiplier",
  "opp_bak_multiplier",
  "opp_pi_bak",
  "opp_gwang_bak",
  "opp_mong_bak",
  "opp_five_base",
  "opp_ribbon_base",
  "opp_junk_base",
  "opp_red_ribbons",
  "opp_blue_ribbons",
  "opp_plain_ribbons",
  "opp_five_birds",
  "opp_kwang",
  "self_go_legal",
  "self_stop_legal",
  "self_go_ready",
  "self_auto_stop_ready",
  "self_failed_go",
  "self_gukjin_mode_junk",
  "self_gukjin_locked",
  "self_pending_gukjin_choice",
  "self_gukjin_junk_better",
  "self_pending_president",
  "self_pending_president_month",
  "self_president_hold",
  "self_president_hold_month",
  "self_president_x4_ready",
  "self_dokbak_risk",
  "self_ppuk",
  "self_jjob",
  "self_jabbeok",
  "self_pansseul",
  "self_ppuk_active",
  "self_ppuk_streak",
  "self_held_bonus_cards",
  "opp_go_legal",
  "opp_stop_legal",
  "opp_go_ready",
  "opp_auto_stop_ready",
  "opp_failed_go",
  "opp_gukjin_mode_junk",
  "opp_gukjin_locked",
  "opp_pending_gukjin_choice",
  "opp_gukjin_junk_better",
  "opp_pending_president",
  "opp_pending_president_month",
  "opp_president_hold",
  "opp_president_hold_month",
  "opp_president_x4_ready",
  "opp_dokbak_risk",
  "opp_ppuk",
  "opp_jjob",
  "opp_jabbeok",
  "opp_pansseul",
  "opp_ppuk_active",
  "opp_ppuk_streak",
  "opp_held_bonus_cards",
  "state_carry_over_multiplier",
  "state_next_carry_over_multiplier",
  "state_last_nagari",
  "state_last_dokbak",
  "state_pending_steal",
  "state_pending_bonus_flips",
]);
const OPTION_SLOT_BINDINGS = Object.freeze([
  Object.freeze({ positive: "go", negative: "stop" }),
  Object.freeze({ positive: "shaking_yes", negative: "shaking_no" }),
  Object.freeze({ positive: "president_hold", negative: "president_stop" }),
  Object.freeze({ positive: "five", negative: "junk" }),
]);
const MONTH_BIN_SCALES = Object.freeze(
  Array.from({ length: BONUS_MONTH }, (_, index) =>
    Math.max(1, CANONICAL_DECK.filter((card) => Number(card?.month || 0) === index + 1).length),
  ),
);
const TYPE_BIN_SCALES = Object.freeze({
  kwang: Math.max(1, CANONICAL_DECK.filter((card) => String(card?.category || "") === "kwang").length),
  five: Math.max(1, CANONICAL_DECK.filter((card) => String(card?.category || "") === "five").length),
  ribbon: Math.max(1, CANONICAL_DECK.filter((card) => String(card?.category || "") === "ribbon").length),
  pi_value: Math.max(1, CANONICAL_DECK.reduce((sum, card) => sum + Number(card?.piValue || 0), 0)),
});
const COMBO_BIN_SCALES = Object.freeze({
  red_ribbons: Math.max(1, CANONICAL_DECK.filter((card) => (card?.comboTags || []).includes("redRibbons")).length),
  blue_ribbons: Math.max(1, CANONICAL_DECK.filter((card) => (card?.comboTags || []).includes("blueRibbons")).length),
  plain_ribbons: Math.max(1, CANONICAL_DECK.filter((card) => (card?.comboTags || []).includes("plainRibbons")).length),
  five_birds: Math.max(1, CANONICAL_DECK.filter((card) => (card?.comboTags || []).includes("fiveBirds")).length),
  kwang_combo: Math.max(1, CANONICAL_DECK.filter((card) => String(card?.category || "") === "kwang").length),
});
const UPSTREAM_CORE_INPUT_FEATURES = Object.freeze([
  "focus_month",
  "focus_five",
  "focus_ribbon",
  "focus_junk",
  "focus_board_match_count",
  "focus_board_capture_value",
  "focus_board_single_match",
  "focus_board_multi_match",
  "board_month_bin_0",
  "board_month_bin_1",
  "board_month_bin_2",
  "board_month_bin_3",
  "board_month_bin_4",
  "board_month_bin_5",
  "board_month_bin_6",
  "board_month_bin_7",
  "board_month_bin_8",
  "board_month_bin_9",
  "board_month_bin_10",
  "board_month_bin_11",
]);

function clamp01(x) {
  const value = Number(x || 0);
  if (value <= 0) return 0;
  if (value >= 1) return 1;
  return value;
}

function normalizeGlobalScore(value) {
  return clamp01(Number(value || 0) / 15.0);
}

function normalizeMultiplier(value) {
  return clamp01(Number(value || 0) / 8.0);
}

function normalizeMonth(value) {
  return clamp01(Number(value || 0) / BONUS_MONTH);
}

function normalizeCount(value, scale = 3.0) {
  return clamp01(Number(value || 0) / Math.max(1.0, Number(scale || 1.0)));
}

function bool01(value) {
  return value ? 1.0 : 0.0;
}

function opponentOf(actor) {
  return String(actor || "") === "human" ? "ai" : "human";
}

function collectCapturedCards(player) {
  const captured = player?.captured || {};
  return []
    .concat(captured.kwang || [])
    .concat(captured.five || [])
    .concat(captured.ribbon || [])
    .concat(captured.junk || []);
}

function getCardMeta(card) {
  const cardId = String(card?.id || card?.cardID || "");
  const canonical = CARD_META.get(cardId);
  const comboTags = Array.isArray(card?.comboTags) && card.comboTags.length
    ? [...card.comboTags]
    : [...(canonical?.comboTags || [])];
  return {
    id: cardId,
    month: Number(card?.month ?? canonical?.month ?? 0),
    category: String(card?.category || canonical?.category || ""),
    piValue: Number(card?.piValue ?? canonical?.piValue ?? 0),
    bonusStealPi: Number(card?.bonus?.stealPi ?? canonical?.bonusStealPi ?? 0),
    comboTags,
  };
}

function cardScalar(card) {
  const cardId = String(card?.id || "");
  const index = CARD_INDEX.get(cardId);
  if (index == null) return 0.0;
  return (index + 1) / CARD_SCALE;
}

function sortCards(cards) {
  return [...(cards || [])].sort((left, right) => {
    const a = CARD_INDEX.get(String(left?.id || "")) ?? Number.MAX_SAFE_INTEGER;
    const b = CARD_INDEX.get(String(right?.id || "")) ?? Number.MAX_SAFE_INTEGER;
    if (a !== b) return a - b;
    return String(left?.id || "").localeCompare(String(right?.id || ""));
  });
}

function hasComboTag(meta, tag) {
  return Array.isArray(meta?.comboTags) && meta.comboTags.includes(tag);
}

function encodeCardFeatureSlots(cards, slotCount, featureName) {
  const sorted = sortCards(cards);
  const values = new Array(Math.max(0, Number(slotCount || 0))).fill(0.0);
  for (let index = 0; index < values.length && index < sorted.length; index += 1) {
    const meta = getCardMeta(sorted[index]);
    if (featureName === "month") {
      values[index] = clamp01(Number(meta.month || 0) / 13.0);
      continue;
    }
    if (featureName === "kwang" || featureName === "five" || featureName === "ribbon" || featureName === "junk") {
      values[index] = meta.category === featureName ? 1.0 : 0.0;
      continue;
    }
    if (featureName === "pi") {
      values[index] = clamp01(Number(meta.piValue || 0) / 3.0);
      continue;
    }
    if (featureName === "bonus") {
      values[index] = Number(meta.bonusStealPi || 0) > 0 ? 1.0 : 0.0;
      continue;
    }
    if (featureName === "red_ribbons") {
      values[index] = hasComboTag(meta, "redRibbons") ? 1.0 : 0.0;
      continue;
    }
    if (featureName === "blue_ribbons") {
      values[index] = hasComboTag(meta, "blueRibbons") ? 1.0 : 0.0;
      continue;
    }
    if (featureName === "plain_ribbons") {
      values[index] = hasComboTag(meta, "plainRibbons") ? 1.0 : 0.0;
      continue;
    }
    if (featureName === "five_birds") {
      values[index] = hasComboTag(meta, "fiveBirds") ? 1.0 : 0.0;
      continue;
    }
  }
  return values;
}

function encodeSingleCardFeatureSlot(card, featureName) {
  const values = encodeCardFeatureSlots(card ? [card] : [], 1, featureName);
  return values.length > 0 ? values : [0.0];
}

function candidateMonth(state, inputContext) {
  const special = parsePlaySpecialCandidate(inputContext?.candidate);
  if (special?.kind === "bomb") {
    return Number(special.month || 0);
  }
  return Number(getCardMeta(inputContext?.candidateCard).month || 0);
}

function boardCardsForCandidate(state, inputContext) {
  const month = candidateMonth(state, inputContext);
  if (month <= 0) {
    return [];
  }
  return (state?.board || []).filter((card) => Number(getCardMeta(card).month || 0) === month);
}

function boardCardValue(card) {
  const meta = getCardMeta(card);
  let value = 0.0;
  if (meta.category === "kwang") value += 1.20;
  if (meta.category === "five") value += 0.95;
  if (meta.category === "ribbon") value += 0.70;
  if (meta.category === "junk") value += 0.20;
  value += clamp01(Number(meta.piValue || 0) / 3.0) * 0.30;
  if (Number(meta.bonusStealPi || 0) > 0) value += 0.25;
  if (hasComboTag(meta, "redRibbons")) value += 0.15;
  if (hasComboTag(meta, "blueRibbons")) value += 0.15;
  if (hasComboTag(meta, "plainRibbons")) value += 0.15;
  if (hasComboTag(meta, "fiveBirds")) value += 0.20;
  return value;
}

function encodeFocusFeatureSlot(state, featureName, inputContext) {
  const candidateCard = inputContext?.candidateCard || null;
  if (
    featureName === "board_match_count"
    || featureName === "board_capture_value"
    || featureName === "board_single_match"
    || featureName === "board_multi_match"
  ) {
    const matched = boardCardsForCandidate(state, inputContext);
    if (featureName === "board_match_count") {
      return [clamp01(matched.length / 4.0)];
    }
    if (featureName === "board_single_match") {
      return [matched.length === 1 ? 1.0 : 0.0];
    }
    if (featureName === "board_multi_match") {
      return [matched.length >= 2 ? 1.0 : 0.0];
    }
    const totalValue = matched.reduce((sum, card) => sum + boardCardValue(card), 0.0);
    return [clamp01(totalValue / 3.0)];
  }
  return encodeSingleCardFeatureSlot(candidateCard, featureName);
}

function encodeDecisionTypeSlots(decisionType, slotCount) {
  const values = new Array(Math.max(0, Number(slotCount || 0))).fill(0.0);
  const kind = String(decisionType || "").trim().toLowerCase();
  if (values.length > 0 && kind === "play") values[0] = 1.0;
  if (values.length > 1 && kind === "match") values[1] = 1.0;
  if (values.length > 2 && kind === "option") values[2] = 1.0;
  return values;
}

function findCardById(cards, cardId) {
  const id = String(cardId || "");
  if (!Array.isArray(cards)) return null;
  return cards.find((card) => String(card?.id || "") === id) || null;
}

function encodeMonthBins(cards, slotCount) {
  const counts = new Array(BONUS_MONTH).fill(0.0);
  for (const card of cards || []) {
    const month = Number(getCardMeta(card).month || 0);
    if (month >= 1 && month <= BONUS_MONTH) {
      counts[month - 1] += 1.0;
    }
  }
  const values = counts.map((count, index) => clamp01(count / MONTH_BIN_SCALES[index]));
  return values.slice(0, Math.max(0, Number(slotCount || 0)));
}

function encodeTypeBins(cards, slotCount) {
  const counts = {
    kwang: 0.0,
    five: 0.0,
    ribbon: 0.0,
    pi_value: 0.0,
  };
  for (const card of cards || []) {
    const meta = getCardMeta(card);
    if (meta.category in counts) {
      counts[meta.category] += 1.0;
    }
    counts.pi_value += Number(meta.piValue || 0);
  }
  return TYPE_BIN_KEYS
    .map((key) => clamp01(Number(counts[key] || 0) / Number(TYPE_BIN_SCALES[key] || 1)))
    .slice(0, Math.max(0, Number(slotCount || 0)));
}

function encodeComboBins(cards, slotCount) {
  const counts = {
    red_ribbons: 0.0,
    blue_ribbons: 0.0,
    plain_ribbons: 0.0,
    five_birds: 0.0,
    kwang_combo: 0.0,
  };
  for (const card of cards || []) {
    const meta = getCardMeta(card);
    if (hasComboTag(meta, "redRibbons")) counts.red_ribbons += 1.0;
    if (hasComboTag(meta, "blueRibbons")) counts.blue_ribbons += 1.0;
    if (hasComboTag(meta, "plainRibbons")) counts.plain_ribbons += 1.0;
    if (hasComboTag(meta, "fiveBirds")) counts.five_birds += 1.0;
    if (meta.category === "kwang") counts.kwang_combo += 1.0;
  }
  return COMBO_BIN_KEYS
    .map((key) => clamp01(Number(counts[key] || 0) / Number(COMBO_BIN_SCALES[key] || 1)))
    .slice(0, Math.max(0, Number(slotCount || 0)));
}

function encodeCardIds(cards, slotCount) {
  const sorted = sortCards(cards);
  const values = new Array(Math.max(0, Number(slotCount || 0))).fill(null);
  for (let index = 0; index < values.length && index < sorted.length; index += 1) {
    values[index] = String(sorted[index]?.id || "") || null;
  }
  return values;
}

function encodePendingMatchCardIds(state, slotCount) {
  const candidateIds = new Set(
    (state?.pendingMatch?.boardCardIds || [])
      .map((id) => String(id || "").trim())
      .filter(Boolean),
  );
  if (candidateIds.size <= 0) {
    return encodeCardIds(state?.board || [], slotCount);
  }
  const candidateCards = (state?.board || []).filter((card) => candidateIds.has(String(card?.id || "")));
  return encodeCardIds(candidateCards, slotCount);
}

function comboCount(cards, tag) {
  let count = 0;
  for (const card of cards || []) {
    if (hasComboTag(getCardMeta(card), tag)) {
      count += 1;
    }
  }
  return count;
}

function ribbonComboScore(cards, tag) {
  return comboCount(cards, tag) >= 3 ? 3 : 0;
}

function fiveBirdsScore(cards) {
  return comboCount(cards, "fiveBirds") >= 3 ? 5 : 0;
}

function fiveBaseScore(cards) {
  const count = Array.isArray(cards) ? cards.length : 0;
  return count >= 5 ? count - 4 : 0;
}

function ribbonBaseScore(cards) {
  const count = Array.isArray(cards) ? cards.length : 0;
  return count >= 5 ? count - 4 : 0;
}

function junkBaseScore(cards) {
  let totalPi = 0;
  for (const card of cards || []) {
    totalPi += Number(getCardMeta(card).piValue || 0);
  }
  return totalPi >= 10 ? totalPi - 9 : 0;
}

function goBonusScore(player) {
  return Math.max(0, Number(player?.goCount || 0));
}

function goMultiplierScore(player) {
  const goCount = Math.max(0, Number(player?.goCount || 0));
  if (goCount < 3) return 1;
  return 2 ** (goCount - 2);
}

function shakingMultiplierScore(player) {
  const shakingCount = Math.max(0, Number(player?.events?.shaking || 0));
  if (shakingCount <= 0) return 1;
  return 2 ** shakingCount;
}

function bombMultiplierScore(player) {
  const bombCount = Math.max(0, Number(player?.events?.bomb || 0));
  if (bombCount <= 0) return 1;
  return 2 ** bombCount;
}

function ruleScoreSnapshot(state, actor) {
  const selfPlayer = state?.players?.[actor];
  const oppPlayer = state?.players?.[opponentOf(actor)];
  if (!selfPlayer || !oppPlayer) {
    return {
      score: null,
      bakMultiplier: 1,
      piBak: 0,
      gwangBak: 0,
      mongBak: 0,
    };
  }
  const score = calculateScore(selfPlayer, oppPlayer, state?.ruleKey);
  return {
    score,
    bakMultiplier: Math.max(1, Number(score?.bak?.multiplier || 1)),
    piBak: score?.bak?.pi ? 1 : 0,
    gwangBak: score?.bak?.gwang ? 1 : 0,
    mongBak: score?.bak?.mongBak ? 1 : 0,
  };
}

function getRulesForState(state) {
  const ruleKey = String(state?.ruleKey || "A");
  return ruleSets?.[ruleKey] || ruleSets?.A || { goMinScore: 7, useEarlyStop: true };
}

function hasTransformableGukjin(player) {
  return (player?.captured?.five || []).some((card) => isGukjinCard(card) && !card?.gukjinTransformed);
}

function compareScorePreference(left, right) {
  const leftPayout = Number(left?.payoutTotal ?? left?.total ?? 0);
  const rightPayout = Number(right?.payoutTotal ?? right?.total ?? 0);
  if (leftPayout !== rightPayout) return leftPayout - rightPayout;
  const leftTotal = Number(left?.total || 0);
  const rightTotal = Number(right?.total || 0);
  if (leftTotal !== rightTotal) return leftTotal - rightTotal;
  const leftBase = Number(left?.base || 0);
  const rightBase = Number(right?.base || 0);
  if (leftBase !== rightBase) return leftBase - rightBase;
  return Number(left?.multiplier || 1) - Number(right?.multiplier || 1);
}

function goStopRuleStatus(state, actor) {
  const player = state?.players?.[actor];
  if (!player) {
    return {
      goLegal: 0,
      stopLegal: 0,
      goReady: 0,
      autoStopReady: 0,
      failedGo: 0,
    };
  }
  const rules = getRulesForState(state);
  const base = Number(calculateBaseScore(player, rules)?.base || 0);
  const raised = base > Number(player?.lastGoBase || 0);
  const handCount = Array.isArray(player?.hand) ? player.hand.length : 0;
  const promptable = !!rules.useEarlyStop && base >= Number(rules.goMinScore || 7) && raised && handCount > 0;
  const autoStop = !!rules.useEarlyStop && base >= Number(rules.goMinScore || 7) && raised && handCount === 0;
  const pending = state?.phase === "go-stop" && state?.pendingGoStop === actor;
  const failedGo = Number(player?.goCount || 0) > 0 && base <= Number(player?.lastGoBase || 0);
  return {
    goLegal: bool01(pending),
    stopLegal: bool01(pending),
    goReady: bool01(promptable),
    autoStopReady: bool01(autoStop),
    failedGo: bool01(failedGo),
  };
}

function gukjinStatus(state, actor) {
  const player = state?.players?.[actor];
  const opponent = state?.players?.[opponentOf(actor)];
  if (!player || !opponent) {
    return {
      modeJunk: 0,
      locked: 0,
      pendingChoice: 0,
      junkBetter: 0,
    };
  }
  const fiveScore = calculateScore({ ...player, gukjinMode: "five" }, opponent, state?.ruleKey);
  const junkScore = calculateScore({ ...player, gukjinMode: "junk" }, opponent, state?.ruleKey);
  return {
    modeJunk: bool01(player?.gukjinMode === "junk"),
    locked: bool01(player?.gukjinLocked),
    pendingChoice: bool01(state?.phase === "gukjin-choice" && state?.pendingGukjinChoice?.playerKey === actor),
    junkBetter: bool01(hasTransformableGukjin(player) && compareScorePreference(junkScore, fiveScore) > 0),
  };
}

function presidentStatus(state, actor) {
  const player = state?.players?.[actor];
  if (!player) {
    return {
      pending: 0,
      pendingMonth: 0,
      hold: 0,
      holdMonth: 0,
      x4Ready: 0,
    };
  }
  const pending = state?.phase === "president-choice" && state?.pendingPresident?.playerKey === actor;
  const hold = !!player?.presidentHold;
  const hasShakingBomb = Number(player?.events?.shaking || 0) > 0 || Number(player?.events?.bomb || 0) > 0;
  return {
    pending: bool01(pending),
    pendingMonth: normalizeMonth(pending ? Number(state?.pendingPresident?.month || 0) : 0),
    hold: bool01(hold),
    holdMonth: normalizeMonth(Number(player?.presidentHoldMonth || 0)),
    x4Ready: bool01(hold && hasShakingBomb),
  };
}

function eventCount(player, key) {
  return Math.max(0, Number(player?.events?.[key] || 0));
}

function heldBonusCount(player) {
  return Array.isArray(player?.heldBonusCards) ? player.heldBonusCards.length : 0;
}

function pendingStealCount(state) {
  return Math.max(
    0,
    Number(state?.pendingSteal || 0),
    Number(state?.pendingMatch?.context?.pendingSteal || 0),
  );
}

function pendingBonusFlipCount(state) {
  const direct = Array.isArray(state?.pendingBonusFlips) ? state.pendingBonusFlips.length : 0;
  const nested = Array.isArray(state?.pendingMatch?.context?.pendingBonusFlips)
    ? state.pendingMatch.context.pendingBonusFlips.length
    : 0;
  return Math.max(direct, nested, 0);
}

function stateOutcomeStatus(state) {
  const result = state?.result || {};
  const humanDokbak = !!result?.human?.bak?.dokbak;
  const aiDokbak = !!result?.ai?.bak?.dokbak;
  return {
    carryOverMultiplier: Math.max(1, Number(state?.carryOverMultiplier || 1)),
    nextCarryOverMultiplier: Math.max(1, Number(state?.nextCarryOverMultiplier || 1)),
    lastNagari: bool01(result?.nagari),
    lastDokbak: bool01(humanDokbak || aiDokbak),
  };
}

function kwangScore(cards) {
  const uniqueIds = new Set();
  const kwangCards = [];
  for (const card of cards || []) {
    const meta = getCardMeta(card);
    if (meta.category !== "kwang") continue;
    if (!meta.id || uniqueIds.has(meta.id)) continue;
    uniqueIds.add(meta.id);
    kwangCards.push(meta);
  }
  const count = kwangCards.length;
  if (count < 3) return 0;
  if (count === 3) {
    return kwangCards.some((card) => card.id === NON_BRIGHT_KWANG_ID) ? 2 : 3;
  }
  if (count === 4) return 4;
  return 15;
}

function buildRuleScoreSlots(state, actor, slotCount) {
  const selfPlayer = state?.players?.[actor];
  const oppPlayer = state?.players?.[opponentOf(actor)];
  const selfRule = ruleScoreSnapshot(state, actor);
  const oppRule = ruleScoreSnapshot(state, opponentOf(actor));
  const selfGo = goStopRuleStatus(state, actor);
  const oppGo = goStopRuleStatus(state, opponentOf(actor));
  const selfGukjin = gukjinStatus(state, actor);
  const oppGukjin = gukjinStatus(state, opponentOf(actor));
  const selfPresident = presidentStatus(state, actor);
  const oppPresident = presidentStatus(state, opponentOf(actor));
  const outcomeState = stateOutcomeStatus(state);
  const selfRibbon = selfPlayer?.captured?.ribbon || [];
  const oppRibbon = oppPlayer?.captured?.ribbon || [];
  const selfFive = selfPlayer?.captured?.five || [];
  const oppFive = oppPlayer?.captured?.five || [];
  const selfKwang = selfPlayer?.captured?.kwang || [];
  const oppKwang = oppPlayer?.captured?.kwang || [];
  const selfJunk = selfPlayer?.captured?.junk || [];
  const oppJunk = oppPlayer?.captured?.junk || [];
  const featureValues = {
    self_go_bonus: normalizeGlobalScore(goBonusScore(selfPlayer)),
    self_go_multiplier: normalizeMultiplier(goMultiplierScore(selfPlayer)),
    self_shaking_multiplier: normalizeMultiplier(shakingMultiplierScore(selfPlayer)),
    self_bomb_multiplier: normalizeMultiplier(bombMultiplierScore(selfPlayer)),
    self_bak_multiplier: normalizeMultiplier(selfRule.bakMultiplier),
    self_pi_bak: clamp01(selfRule.piBak),
    self_gwang_bak: clamp01(selfRule.gwangBak),
    self_mong_bak: clamp01(selfRule.mongBak),
    self_five_base: normalizeGlobalScore(fiveBaseScore(selfFive)),
    self_ribbon_base: normalizeGlobalScore(ribbonBaseScore(selfRibbon)),
    self_junk_base: normalizeGlobalScore(junkBaseScore(selfJunk)),
    self_red_ribbons: normalizeGlobalScore(ribbonComboScore(selfRibbon, "redRibbons")),
    self_blue_ribbons: normalizeGlobalScore(ribbonComboScore(selfRibbon, "blueRibbons")),
    self_plain_ribbons: normalizeGlobalScore(ribbonComboScore(selfRibbon, "plainRibbons")),
    self_five_birds: normalizeGlobalScore(fiveBirdsScore(selfFive)),
    self_kwang: normalizeGlobalScore(kwangScore(selfKwang)),
    opp_go_bonus: normalizeGlobalScore(goBonusScore(oppPlayer)),
    opp_go_multiplier: normalizeMultiplier(goMultiplierScore(oppPlayer)),
    opp_shaking_multiplier: normalizeMultiplier(shakingMultiplierScore(oppPlayer)),
    opp_bomb_multiplier: normalizeMultiplier(bombMultiplierScore(oppPlayer)),
    opp_bak_multiplier: normalizeMultiplier(oppRule.bakMultiplier),
    opp_pi_bak: clamp01(oppRule.piBak),
    opp_gwang_bak: clamp01(oppRule.gwangBak),
    opp_mong_bak: clamp01(oppRule.mongBak),
    opp_five_base: normalizeGlobalScore(fiveBaseScore(oppFive)),
    opp_ribbon_base: normalizeGlobalScore(ribbonBaseScore(oppRibbon)),
    opp_junk_base: normalizeGlobalScore(junkBaseScore(oppJunk)),
    opp_red_ribbons: normalizeGlobalScore(ribbonComboScore(oppRibbon, "redRibbons")),
    opp_blue_ribbons: normalizeGlobalScore(ribbonComboScore(oppRibbon, "blueRibbons")),
    opp_plain_ribbons: normalizeGlobalScore(ribbonComboScore(oppRibbon, "plainRibbons")),
    opp_five_birds: normalizeGlobalScore(fiveBirdsScore(oppFive)),
    opp_kwang: normalizeGlobalScore(kwangScore(oppKwang)),
    self_go_legal: selfGo.goLegal,
    self_stop_legal: selfGo.stopLegal,
    self_go_ready: selfGo.goReady,
    self_auto_stop_ready: selfGo.autoStopReady,
    self_failed_go: selfGo.failedGo,
    self_gukjin_mode_junk: selfGukjin.modeJunk,
    self_gukjin_locked: selfGukjin.locked,
    self_pending_gukjin_choice: selfGukjin.pendingChoice,
    self_gukjin_junk_better: selfGukjin.junkBetter,
    self_pending_president: selfPresident.pending,
    self_pending_president_month: selfPresident.pendingMonth,
    self_president_hold: selfPresident.hold,
    self_president_hold_month: selfPresident.holdMonth,
    self_president_x4_ready: selfPresident.x4Ready,
    self_dokbak_risk: bool01(Number(selfPlayer?.goCount || 0) > 0),
    self_ppuk: normalizeCount(eventCount(selfPlayer, "ppuk"), 3.0),
    self_jjob: normalizeCount(eventCount(selfPlayer, "jjob"), 3.0),
    self_jabbeok: normalizeCount(eventCount(selfPlayer, "jabbeok"), 3.0),
    self_pansseul: normalizeCount(eventCount(selfPlayer, "pansseul"), 3.0),
    self_ppuk_active: bool01(selfPlayer?.ppukState?.active),
    self_ppuk_streak: normalizeCount(Number(selfPlayer?.ppukState?.streak || 0), 3.0),
    self_held_bonus_cards: normalizeCount(heldBonusCount(selfPlayer), 4.0),
    opp_go_legal: oppGo.goLegal,
    opp_stop_legal: oppGo.stopLegal,
    opp_go_ready: oppGo.goReady,
    opp_auto_stop_ready: oppGo.autoStopReady,
    opp_failed_go: oppGo.failedGo,
    opp_gukjin_mode_junk: oppGukjin.modeJunk,
    opp_gukjin_locked: oppGukjin.locked,
    opp_pending_gukjin_choice: oppGukjin.pendingChoice,
    opp_gukjin_junk_better: oppGukjin.junkBetter,
    opp_pending_president: oppPresident.pending,
    opp_pending_president_month: oppPresident.pendingMonth,
    opp_president_hold: oppPresident.hold,
    opp_president_hold_month: oppPresident.holdMonth,
    opp_president_x4_ready: oppPresident.x4Ready,
    opp_dokbak_risk: bool01(Number(oppPlayer?.goCount || 0) > 0),
    opp_ppuk: normalizeCount(eventCount(oppPlayer, "ppuk"), 3.0),
    opp_jjob: normalizeCount(eventCount(oppPlayer, "jjob"), 3.0),
    opp_jabbeok: normalizeCount(eventCount(oppPlayer, "jabbeok"), 3.0),
    opp_pansseul: normalizeCount(eventCount(oppPlayer, "pansseul"), 3.0),
    opp_ppuk_active: bool01(oppPlayer?.ppukState?.active),
    opp_ppuk_streak: normalizeCount(Number(oppPlayer?.ppukState?.streak || 0), 3.0),
    opp_held_bonus_cards: normalizeCount(heldBonusCount(oppPlayer), 4.0),
    state_carry_over_multiplier: normalizeMultiplier(outcomeState.carryOverMultiplier),
    state_next_carry_over_multiplier: normalizeMultiplier(outcomeState.nextCarryOverMultiplier),
    state_last_nagari: outcomeState.lastNagari,
    state_last_dokbak: outcomeState.lastDokbak,
    state_pending_steal: normalizeMultiplier(pendingStealCount(state)),
    state_pending_bonus_flips: normalizeCount(pendingBonusFlipCount(state), 4.0),
  };
  const out = new Array(Math.max(0, Number(slotCount || 0))).fill(0.0);
  for (let index = 0; index < out.length && index < RULE_SCORE_FEATURES.length; index += 1) {
    out[index] = Number(featureValues[RULE_SCORE_FEATURES[index]] || 0.0);
  }
  return out;
}

function cardsForZone(state, actor, zoneName) {
  if (zoneName === "hand") {
    return state?.players?.[actor]?.hand || [];
  }
  if (zoneName === "board") {
    return state?.board || [];
  }
  if (zoneName === "captured_self") {
    return collectCapturedCards(state?.players?.[actor]);
  }
  if (zoneName === "captured_opp") {
    return collectCapturedCards(state?.players?.[opponentOf(actor)]);
  }
  return [];
}

function encodeInputSpec(state, actor, spec, inputContext = null) {
  const kind = String(spec?.kind || "").trim().toLowerCase();
  const slotCount = Number(spec?.slot_count || 0);
  const specIndex = Math.max(0, Number(spec?.index || 0));
  if (kind === "input") {
    if (specIndex === 0 && slotCount >= UPSTREAM_CORE_INPUT_FEATURES.length) {
      const bins = encodeMonthBins(state?.board || [], 12);
      return [
        ...encodeFocusFeatureSlot(state, "month", inputContext).slice(0, 1),
        ...encodeFocusFeatureSlot(state, "five", inputContext).slice(0, 1),
        ...encodeFocusFeatureSlot(state, "ribbon", inputContext).slice(0, 1),
        ...encodeFocusFeatureSlot(state, "junk", inputContext).slice(0, 1),
        ...encodeFocusFeatureSlot(state, "board_match_count", inputContext).slice(0, 1),
        ...encodeFocusFeatureSlot(state, "board_capture_value", inputContext).slice(0, 1),
        ...encodeFocusFeatureSlot(state, "board_single_match", inputContext).slice(0, 1),
        ...encodeFocusFeatureSlot(state, "board_multi_match", inputContext).slice(0, 1),
        ...bins,
      ].slice(0, Math.max(0, slotCount));
    }
    const feature = String(UPSTREAM_CORE_INPUT_FEATURES[specIndex] || "");
    if (feature === "focus_month") {
      return encodeFocusFeatureSlot(state, "month", inputContext).slice(0, Math.max(0, slotCount));
    }
    if (feature === "focus_five") {
      return encodeFocusFeatureSlot(state, "five", inputContext).slice(0, Math.max(0, slotCount));
    }
    if (feature === "focus_ribbon") {
      return encodeFocusFeatureSlot(state, "ribbon", inputContext).slice(0, Math.max(0, slotCount));
    }
    if (feature === "focus_junk") {
      return encodeFocusFeatureSlot(state, "junk", inputContext).slice(0, Math.max(0, slotCount));
    }
    if (feature === "focus_board_match_count") {
      return encodeFocusFeatureSlot(state, "board_match_count", inputContext).slice(0, Math.max(0, slotCount));
    }
    if (feature === "focus_board_capture_value") {
      return encodeFocusFeatureSlot(state, "board_capture_value", inputContext).slice(0, Math.max(0, slotCount));
    }
    if (feature === "focus_board_single_match") {
      return encodeFocusFeatureSlot(state, "board_single_match", inputContext).slice(0, Math.max(0, slotCount));
    }
    if (feature === "focus_board_multi_match") {
      return encodeFocusFeatureSlot(state, "board_multi_match", inputContext).slice(0, Math.max(0, slotCount));
    }
    if (feature.startsWith("board_month_bin_")) {
      const offset = Math.max(0, Number(feature.slice("board_month_bin_".length) || 0));
      const bins = encodeMonthBins(state?.board || [], 12);
      return [Number(bins[offset] || 0.0)].slice(0, Math.max(0, slotCount));
    }
    return new Array(Math.max(0, slotCount)).fill(0.0);
  }
  if (kind === "input_rule_score") {
    return buildRuleScoreSlots(state, actor, slotCount);
  }
  if (kind === "input_decision_type") {
    return encodeDecisionTypeSlots(inputContext?.decisionType || "", slotCount);
  }
  const semanticMatch = kind.match(SEMANTIC_BIN_PATTERN);
  if (semanticMatch) {
    const zoneName = String(semanticMatch[1] || "");
    const binName = String(semanticMatch[2] || "");
    const cards = cardsForZone(state, actor, zoneName);
    if (binName === "month") {
      return encodeMonthBins(cards, slotCount);
    }
    if (binName === "type") {
      return encodeTypeBins(cards, slotCount);
    }
    if (binName === "combo") {
      return encodeComboBins(cards, slotCount);
    }
  }
  const match = kind.match(HAND_FEATURE_PATTERN);
  if (match) {
    const featureName = String(match[1] || "");
    return encodeCardFeatureSlots(state?.players?.[actor]?.hand || [], slotCount, featureName);
  }
  const focusMatch = kind.match(FOCUS_FEATURE_PATTERN);
  if (focusMatch) {
    const featureName = String(focusMatch[1] || "");
    return encodeFocusFeatureSlot(state, featureName, inputContext);
  }
  return new Array(slotCount).fill(0.0);
}

export function isMatgoMinimalAdapter(runtime) {
  return String(runtime?.adapter?.kind || "").trim() === "matgo_minimal_v1";
}

export function encodeMatgoStateToKHyperneatInputs(state, actor, runtime, inputContext = null) {
  if (!isMatgoMinimalAdapter(runtime)) {
    throw new Error("K-HyperNEAT runtime is missing matgo_minimal_v1 adapter metadata");
  }
  const values = [];
  for (const spec of runtime.adapter.inputs || []) {
    values.push(...encodeInputSpec(state, actor, spec, inputContext));
  }
  return values;
}

export function resolveMatgoCandidateCard(state, actor, decisionType, candidate) {
  if (decisionType === "play") {
    const special = parsePlaySpecialCandidate(candidate);
    if (special?.kind === "shake_start") {
      return findCardById(state?.players?.[actor]?.hand || [], special.cardId);
    }
    if (special?.kind === "bomb") {
      return { id: String(candidate || ""), month: special.month, category: "junk", piValue: 0 };
    }
    return findCardById(state?.players?.[actor]?.hand || [], candidate);
  }
  if (decisionType === "match") {
    return findCardById(state?.board || [], candidate);
  }
  return null;
}

export function encodeMatgoFocusedCandidateInputs(state, actor, runtime, decisionType, candidate) {
  return encodeMatgoStateToKHyperneatInputs(state, actor, runtime, {
    decisionType,
    candidate,
    candidateCard: resolveMatgoCandidateCard(state, actor, decisionType, candidate),
  });
}

export function getMatgoKHyperneatOutputBindings(state, actor, runtime) {
  if (!isMatgoMinimalAdapter(runtime)) {
    throw new Error("K-HyperNEAT runtime is missing matgo_minimal_v1 adapter metadata");
  }

  const bindings = {
    playIndex: null,
    matchIndex: null,
    optionPairs: [],
  };

  let cursor = 0;
  for (const spec of runtime.adapter.outputs || []) {
    const kind = String(spec?.kind || "").trim().toLowerCase();
    const specIndex = Math.max(0, Number(spec?.index || 0));
    const slotCount = Math.max(0, Number(spec?.slot_count || 0));
    if (kind === "output") {
      if (specIndex === 0 && slotCount >= 6) {
        bindings.playIndex = cursor;
        bindings.matchIndex = cursor + 1;
        for (let offset = 0; offset < 4; offset += 1) {
          bindings.optionPairs.push({
            outputIndex: cursor + 2 + offset,
            positive: OPTION_SLOT_BINDINGS[offset].positive,
            negative: OPTION_SLOT_BINDINGS[offset].negative,
          });
        }
      } else if (specIndex === 0 && slotCount > 0) {
        bindings.playIndex = cursor;
      }
    } else if (kind === "output_play") {
      if (slotCount > 0) {
        bindings.playIndex = cursor;
      }
    } else if (kind === "output_match") {
      if (slotCount > 0) {
        bindings.matchIndex = cursor;
      }
    } else if (kind === "output_option") {
      for (
        let offset = 0;
        offset < slotCount && (specIndex + offset) < OPTION_SLOT_BINDINGS.length;
        offset += 1
      ) {
        bindings.optionPairs.push({
          outputIndex: cursor + offset,
          positive: OPTION_SLOT_BINDINGS[specIndex + offset].positive,
          negative: OPTION_SLOT_BINDINGS[specIndex + offset].negative,
        });
      }
    }
    cursor += slotCount;
  }

  return bindings;
}
