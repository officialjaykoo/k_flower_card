import { buildDeck } from "../cards.js";
import { calculateScore } from "../engine/index.js";

const CANONICAL_DECK = buildDeck("original");
const CARD_INDEX = new Map(CANONICAL_DECK.map((card, index) => [String(card?.id || ""), index]));
const CARD_SCALE = Math.max(1, CANONICAL_DECK.length);
const OPTION_SLOT_BINDINGS = Object.freeze([
  Object.freeze({ positive: "go", negative: "stop" }),
  Object.freeze({ positive: "shaking_yes", negative: "shaking_no" }),
  Object.freeze({ positive: "president_hold", negative: "president_stop" }),
  Object.freeze({ positive: "five", negative: "junk" }),
]);
const PHASE_CODE = Object.freeze({
  playing: 0.1,
  "select-match": 0.25,
  "shaking-confirm": 0.4,
  "go-stop": 0.6,
  "president-choice": 0.8,
  "gukjin-choice": 1.0,
});

function clamp01(x) {
  const value = Number(x || 0);
  if (value <= 0) return 0;
  if (value >= 1) return 1;
  return value;
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

function encodeCardSlots(cards, slotCount) {
  const sorted = sortCards(cards);
  const values = new Array(Math.max(0, Number(slotCount || 0))).fill(0.0);
  for (let index = 0; index < values.length && index < sorted.length; index += 1) {
    values[index] = cardScalar(sorted[index]);
  }
  return values;
}

function encodeCardIds(cards, slotCount) {
  const sorted = sortCards(cards);
  const values = new Array(Math.max(0, Number(slotCount || 0))).fill(null);
  for (let index = 0; index < values.length && index < sorted.length; index += 1) {
    values[index] = String(sorted[index]?.id || "") || null;
  }
  return values;
}

function scoreTotalNorm(state, actor) {
  const selfKey = String(actor || "");
  const oppKey = opponentOf(selfKey);
  const selfPlayer = state?.players?.[selfKey];
  const oppPlayer = state?.players?.[oppKey];
  if (!selfPlayer || !oppPlayer) return 0.0;
  const score = calculateScore(selfPlayer, oppPlayer, state?.ruleKey);
  return clamp01(Number(score?.total || 0) / 15.0);
}

function buildPublicSlots(state, actor, slotCount) {
  const values = [
    String(state?.currentTurn || "") === String(actor || "") ? 1.0 : 0.0,
    Number(PHASE_CODE[String(state?.phase || "")] || 0.0),
    clamp01(Number(state?.deck?.length || 0) / 30.0),
    clamp01(Number(state?.board?.length || 0) / 12.0),
    scoreTotalNorm(state, actor),
    scoreTotalNorm(state, opponentOf(actor)),
  ];
  const out = new Array(Math.max(0, Number(slotCount || 0))).fill(0.0);
  for (let index = 0; index < out.length && index < values.length; index += 1) {
    out[index] = values[index];
  }
  return out;
}

function encodeInputSpec(state, actor, spec) {
  const kind = String(spec?.kind || "").trim().toLowerCase();
  const slotCount = Number(spec?.slot_count || 0);
  if (kind === "input_public") {
    return buildPublicSlots(state, actor, slotCount);
  }
  if (kind === "input_hand") {
    return encodeCardSlots(state?.players?.[actor]?.hand || [], slotCount);
  }
  if (kind === "input_board") {
    return encodeCardSlots(state?.board || [], slotCount);
  }
  if (kind === "input_captured_self") {
    return encodeCardSlots(collectCapturedCards(state?.players?.[actor]), slotCount);
  }
  if (kind === "input_captured_opp") {
    return encodeCardSlots(collectCapturedCards(state?.players?.[opponentOf(actor)]), slotCount);
  }
  return new Array(slotCount).fill(0.0);
}

export function isMatgoMinimalAdapter(runtime) {
  return String(runtime?.adapter?.kind || "").trim() === "matgo_minimal_v1";
}

export function encodeMatgoStateToKHyperneatInputs(state, actor, runtime) {
  if (!isMatgoMinimalAdapter(runtime)) {
    throw new Error("K-HyperNEAT runtime is missing matgo_minimal_v1 adapter metadata");
  }
  const values = [];
  for (const spec of runtime.adapter.inputs || []) {
    values.push(...encodeInputSpec(state, actor, spec));
  }
  return values;
}

export function getMatgoKHyperneatOutputBindings(state, actor, runtime) {
  if (!isMatgoMinimalAdapter(runtime)) {
    throw new Error("K-HyperNEAT runtime is missing matgo_minimal_v1 adapter metadata");
  }

  const bindings = {
    playSlots: [],
    matchSlots: [],
    optionPairs: [],
  };

  let cursor = 0;
  for (const spec of runtime.adapter.outputs || []) {
    const kind = String(spec?.kind || "").trim().toLowerCase();
    const slotCount = Math.max(0, Number(spec?.slot_count || 0));
    if (kind === "output_play") {
      const ids = encodeCardIds(state?.players?.[actor]?.hand || [], slotCount);
      bindings.playSlots.push(...ids.map((cardId, offset) => ({
        cardId,
        outputIndex: cursor + offset,
      })));
    } else if (kind === "output_match") {
      const ids = encodeCardIds(state?.board || [], slotCount);
      bindings.matchSlots.push(...ids.map((cardId, offset) => ({
        cardId,
        outputIndex: cursor + offset,
      })));
    } else if (kind === "output_option") {
      for (let offset = 0; offset < slotCount && offset < OPTION_SLOT_BINDINGS.length; offset += 1) {
        bindings.optionPairs.push({
          outputIndex: cursor + offset,
          positive: OPTION_SLOT_BINDINGS[offset].positive,
          negative: OPTION_SLOT_BINDINGS[offset].negative,
        });
      }
    }
    cursor += slotCount;
  }

  return bindings;
}
