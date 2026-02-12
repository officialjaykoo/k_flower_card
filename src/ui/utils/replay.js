import { hydrateCard } from "./common.js";

export function buildReplayFrames(state) {
  const kibo = state.kibo || [];
  const initial = kibo.find((e) => e.type === "initial_deal");
  const frames = [];
  if (initial) {
    frames.push({
      type: "initial",
      turnNo: 0,
      actor: null,
      action: { type: "initial" },
      board: (initial.board || []).map(hydrateCard),
      hands: {
        human: (initial.hands?.human || []).map(hydrateCard),
        ai: (initial.hands?.ai || []).map(hydrateCard)
      },
      deckCount: Array.isArray(initial.deck) ? initial.deck.length : 0,
      events: null,
      steals: { pi: 0, gold: 0 }
    });
  }

  const turns = kibo.filter((e) => e.type === "turn_end");
  turns.forEach((e, idx) => {
    frames.push({
      type: "turn_end",
      turnNo: e.turnNo ?? idx + 1,
      actor: e.actor,
      action: e.action || null,
      board: (e.board || []).map(hydrateCard),
      hands: {
        human: (e.hands?.human || []).map(hydrateCard),
        ai: (e.hands?.ai || []).map(hydrateCard)
      },
      deckCount: e.deckCount ?? 0,
      events: e.events || null,
      steals: e.steals || { pi: 0, gold: 0 }
    });
  });
  return frames;
}

export function formatActionText(action) {
  if (!action) return "-";
  if (action.type === "initial") return "초기 배분";
  if (action.type === "pass") return "패스";
  if (action.type === "play") return `카드 플레이 (${action.card?.month || "?"}월 ${action.card?.name || ""})`;
  if (action.type === "declare_bomb") return `폭탄 선언 (${action.month}월)`;
  if (action.type === "flip-select") return `뒤집기 선택 캡처 (${action.card?.month || "?"}월)`;
  return action.type || "행동";
}

export function formatEventsText(events) {
  if (!events) return "-";
  return [
    `ttak:${events.ttak || 0}`,
    `ppuk:${events.ppuk || 0}`,
    `jjob:${events.jjob || 0}`,
    `ddadak:${events.ddadak || 0}`,
    `ssul:${events.ssul || 0}`,
    `jabbeok:${events.jabbeok || 0}`,
    `yeonPpuk:${events.yeonPpuk || 0}`,
    `shaking:${events.shaking || 0}`,
    `bomb:${events.bomb || 0}`
  ].join(" / ");
}

