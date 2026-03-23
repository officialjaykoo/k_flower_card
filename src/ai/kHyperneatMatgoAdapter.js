import { buildDeck } from "../cards.js";
import { parsePlaySpecialCandidate } from "./evalCore/sharedGameHelpers.js";

const CANONICAL_DECK = buildDeck("original");
const CARD_META = new Map(
  CANONICAL_DECK.map((card, index) => [
    String(card?.id || ""),
    Object.freeze({
      index,
      id: String(card?.id || ""),
      month: Number(card?.month || 0),
      category: String(card?.category || ""),
      piValue: Number(card?.piValue || 0),
      bonusStealPi: Number(card?.bonus?.stealPi || 0),
      comboTags: Object.freeze([...(card?.comboTags || [])]),
    }),
  ]),
);

const BONUS_MONTH = 13;
const UPSTREAM_CORE_INPUT_COUNT = 20;
const UPSTREAM_CORE_OUTPUT_COUNT = 6;
const UPSTREAM_INPUT_INDEX = Object.freeze({
  MONTH: 0,
  FIVE: 1,
  RIBBON: 2,
  JUNK: 3,
  BOARD_MATCH_COUNT: 4,
  BOARD_CAPTURE_VALUE: 5,
  BOARD_SINGLE_MATCH: 6,
  BOARD_MULTI_MATCH: 7,
  BOARD_MONTH_BIN_START: 8,
});
const UPSTREAM_OUTPUT_INDEX = Object.freeze({
  PLAY: 0,
  MATCH: 1,
  OPTION_START: 2,
});
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

function clamp01(x) {
  const value = Number(x || 0);
  if (value <= 0) return 0;
  if (value >= 1) return 1;
  return value;
}

function getCardMeta(card) {
  const cardId = String(card?.id || card?.cardID || "");
  const canonical = CARD_META.get(cardId);
  return {
    id: cardId,
    month: Number(card?.month ?? canonical?.month ?? 0),
    category: String(card?.category || canonical?.category || ""),
    piValue: Number(card?.piValue ?? canonical?.piValue ?? 0),
    bonusStealPi: Number(card?.bonus?.stealPi ?? canonical?.bonusStealPi ?? 0),
    comboTags: Array.isArray(card?.comboTags) && card.comboTags.length
      ? [...card.comboTags]
      : [...(canonical?.comboTags || [])],
  };
}

function findCardById(cards, cardId) {
  const id = String(cardId || "");
  if (!Array.isArray(cards)) return null;
  return cards.find((card) => String(card?.id || "") === id) || null;
}

function resolveMatgoCandidateCard(state, actor, decisionType, candidate) {
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

function candidateMonth(inputContext) {
  const special = parsePlaySpecialCandidate(inputContext?.candidate);
  if (special?.kind === "bomb") {
    return Number(special.month || 0);
  }
  return Number(getCardMeta(inputContext?.candidateCard).month || 0);
}

function boardCardsForCandidate(state, inputContext) {
  const month = candidateMonth(inputContext);
  if (month <= 0) {
    return [];
  }
  return (state?.board || []).filter((card) => Number(getCardMeta(card).month || 0) === month);
}

function hasComboTag(meta, tag) {
  return Array.isArray(meta?.comboTags) && meta.comboTags.includes(tag);
}

function boardCardValue(card) {
  const meta = getCardMeta(card);
  let value = 0.0;
  if (meta.category === "kwang") value += 1.2;
  if (meta.category === "five") value += 0.95;
  if (meta.category === "ribbon") value += 0.7;
  if (meta.category === "junk") value += 0.2;
  value += clamp01(Number(meta.piValue || 0) / 3.0) * 0.3;
  if (Number(meta.bonusStealPi || 0) > 0) value += 0.25;
  if (hasComboTag(meta, "redRibbons")) value += 0.15;
  if (hasComboTag(meta, "blueRibbons")) value += 0.15;
  if (hasComboTag(meta, "plainRibbons")) value += 0.15;
  if (hasComboTag(meta, "fiveBirds")) value += 0.2;
  return value;
}

function encodeMonthBins(cards) {
  const counts = new Array(12).fill(0.0);
  for (const card of cards || []) {
    const month = Number(getCardMeta(card).month || 0);
    if (month >= 1 && month <= 12) {
      counts[month - 1] += 1.0;
    }
  }
  return counts.map((count, index) => clamp01(count / MONTH_BIN_SCALES[index]));
}

function encodeUpstreamInputVector(state, inputContext) {
  const values = new Array(UPSTREAM_CORE_INPUT_COUNT).fill(0.0);
  const meta = getCardMeta(inputContext?.candidateCard);
  const matched = boardCardsForCandidate(state, inputContext);
  const boardBins = encodeMonthBins(state?.board || []);
  values[UPSTREAM_INPUT_INDEX.MONTH] = clamp01(Number(meta.month || 0) / BONUS_MONTH);
  values[UPSTREAM_INPUT_INDEX.FIVE] = meta.category === "five" ? 1.0 : 0.0;
  values[UPSTREAM_INPUT_INDEX.RIBBON] = meta.category === "ribbon" ? 1.0 : 0.0;
  values[UPSTREAM_INPUT_INDEX.JUNK] = meta.category === "junk" ? 1.0 : 0.0;
  values[UPSTREAM_INPUT_INDEX.BOARD_MATCH_COUNT] = clamp01(matched.length / 4.0);
  values[UPSTREAM_INPUT_INDEX.BOARD_CAPTURE_VALUE] = clamp01(
    matched.reduce((sum, card) => sum + boardCardValue(card), 0.0) / 3.0,
  );
  values[UPSTREAM_INPUT_INDEX.BOARD_SINGLE_MATCH] = matched.length === 1 ? 1.0 : 0.0;
  values[UPSTREAM_INPUT_INDEX.BOARD_MULTI_MATCH] = matched.length >= 2 ? 1.0 : 0.0;
  for (let index = 0; index < boardBins.length; index += 1) {
    values[UPSTREAM_INPUT_INDEX.BOARD_MONTH_BIN_START + index] = Number(boardBins[index] || 0.0);
  }
  return values;
}

function encodeInputSpec(state, spec, inputContext = null) {
  const kind = String(spec?.kind || "").trim().toLowerCase();
  const slotCount = Math.max(0, Number(spec?.slot_count || 0));
  const specIndex = Math.max(0, Number(spec?.index || 0));
  if (kind !== "input" || specIndex !== 0) {
    return new Array(slotCount).fill(0.0);
  }
  return encodeUpstreamInputVector(state, inputContext).slice(0, slotCount);
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
    values.push(...encodeInputSpec(state, spec, inputContext));
  }
  return values;
}

export function encodeMatgoFocusedCandidateInputs(state, actor, runtime, decisionType, candidate) {
  return encodeMatgoStateToKHyperneatInputs(state, actor, runtime, {
    decisionType,
    candidate,
    candidateCard: resolveMatgoCandidateCard(state, actor, decisionType, candidate),
  });
}

export function getMatgoKHyperneatOutputBindings(state, actor, runtime) {
  void state;
  void actor;
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
    if (kind === "output" && specIndex === 0 && slotCount >= UPSTREAM_CORE_OUTPUT_COUNT) {
      bindings.playIndex = cursor + UPSTREAM_OUTPUT_INDEX.PLAY;
      bindings.matchIndex = cursor + UPSTREAM_OUTPUT_INDEX.MATCH;
      for (let offset = 0; offset < 4; offset += 1) {
        bindings.optionPairs.push({
          outputIndex: cursor + UPSTREAM_OUTPUT_INDEX.OPTION_START + offset,
          positive: OPTION_SLOT_BINDINGS[offset].positive,
          negative: OPTION_SLOT_BINDINGS[offset].negative,
        });
      }
    }
    cursor += slotCount;
  }

  return bindings;
}
