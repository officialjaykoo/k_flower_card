import { DEFAULT_LANGUAGE, makeTranslator } from "../i18n/i18n.js";
import { hydrateCard } from "./common.js";

/* ============================================================================
 * Replay utilities
 * - kibo -> replay frame transformation
 * - replay action/event formatters
 * ========================================================================== */

function hydrateHands(hands) {
  return {
    human: (hands?.human || []).map(hydrateCard),
    ai: (hands?.ai || []).map(hydrateCard)
  };
}

function buildInitialFrame(initial) {
  return {
    type: "initial",
    turnNo: 0,
    actor: null,
    action: { type: "initial" },
    board: (initial.board || []).map(hydrateCard),
    hands: hydrateHands(initial.hands),
    deckCount: Array.isArray(initial.deck) ? initial.deck.length : 0,
    events: null,
    steals: { pi: 0, gold: 0 }
  };
}

function buildTurnEndFrame(entry, idx) {
  return {
    type: "turn_end",
    turnNo: entry.turnNo ?? idx + 1,
    actor: entry.actor,
    action: entry.action || null,
    board: (entry.board || []).map(hydrateCard),
    hands: hydrateHands(entry.hands),
    deckCount: entry.deckCount ?? 0,
    events: entry.events || null,
    steals: entry.steals || { pi: 0, gold: 0 }
  };
}

/* 1) Build replay frames from serialized kibo */
export function buildReplayFrames(state) {
  const kibo = state.kibo || [];
  const initial = kibo.find((e) => e.type === "initial_deal");
  const frames = [];

  if (initial) {
    frames.push(buildInitialFrame(initial));
  }

  const turns = kibo.filter((e) => e.type === "turn_end");
  turns.forEach((e, idx) => {
    frames.push(buildTurnEndFrame(e, idx));
  });

  return frames;
}

/* 2) Action/event text formatting */
export function formatActionText(action, t = null) {
  const tr = t || makeTranslator(DEFAULT_LANGUAGE);
  if (!action) return "-";
  if (action.type === "initial") return tr("replay.action.initial");
  if (action.type === "pass") return tr("replay.action.pass");
  if (action.type === "play") {
    return tr("replay.action.play", { month: action.card?.month || "?", name: action.card?.name || "" });
  }
  if (action.type === "declare_bomb") {
    return tr("replay.action.declareBomb", { month: action.month });
  }
  if (action.type === "flip-select") {
    return tr("replay.action.flipSelect", { month: action.card?.month || "?" });
  }
  return action.type || tr("replay.action.unknown");
}

export function formatEventsText(events) {
  if (!events) return "-";
  return [
    `pansseul:${events.pansseul || 0}`,
    `ppuk:${events.ppuk || 0}`,
    `jjob:${events.jjob || 0}`,
    `ddadak:${events.ddadak || 0}`,
    `jabbeok:${events.jabbeok || 0}`,
    `yeonPpuk:${events.yeonPpuk || 0}`,
    `shaking:${events.shaking || 0}`,
    `bomb:${events.bomb || 0}`
  ].join(" / ");
}
