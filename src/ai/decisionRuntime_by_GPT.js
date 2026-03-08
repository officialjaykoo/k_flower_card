import {
  playTurn,
  chooseMatch,
  chooseGo,
  chooseStop,
  chooseShakingYes,
  chooseShakingNo,
  choosePresidentStop,
  choosePresidentHold,
  chooseGukjinMode,
  getDeclarableShakingMonths,
  getDeclarableBombMonths,
  declareShaking,
  declareBomb
} from "../engine/index.js";

const PLAY_SPECIAL_SHAKE_PREFIX = "shake_start:";
const PLAY_SPECIAL_BOMB_PREFIX = "bomb:";

const OPTION_ALIASES = {
  choose_go: "go",
  choose_stop: "stop",
  choose_shaking_yes: "shaking_yes",
  choose_shaking_no: "shaking_no",
  choose_president_stop: "president_stop",
  choose_president_hold: "president_hold",
  choose_five: "five",
  choose_junk: "junk"
};

export function canonicalOptionAction(action) {
  const a = String(action || "").trim();
  if (!a) return "";
  return OPTION_ALIASES[a] || a;
}

export function normalizeOptionCandidates(items) {
  if (!Array.isArray(items)) return [];
  const out = [];
  const seen = new Set();
  for (const raw of items) {
    const v = canonicalOptionAction(raw);
    if (!v || seen.has(v)) continue;
    seen.add(v);
    out.push(v);
  }
  return out;
}

export function parsePlaySpecialCandidate(candidate) {
  const raw = String(candidate || "").trim();
  if (!raw) return null;
  if (raw.startsWith(PLAY_SPECIAL_SHAKE_PREFIX)) {
    const cardId = raw.slice(PLAY_SPECIAL_SHAKE_PREFIX.length).trim();
    if (!cardId) return null;
    return { kind: "shake_start", cardId };
  }
  if (raw.startsWith(PLAY_SPECIAL_BOMB_PREFIX)) {
    const month = Number(raw.slice(PLAY_SPECIAL_BOMB_PREFIX.length).trim());
    if (!Number.isInteger(month) || month < 1 || month > 12) return null;
    return { kind: "bomb", month };
  }
  return null;
}

function findCardById(cards, cardId) {
  const id = String(cardId || "");
  if (!Array.isArray(cards)) return null;
  return cards.find((c) => String(c?.id || "") === id) || null;
}

function buildPlayingSpecialActions(state, actor) {
  if (state?.phase !== "playing" || state?.currentTurn !== actor) return [];
  const out = [];
  const shakingMonths = new Set(getDeclarableShakingMonths(state, actor));
  if (shakingMonths.size > 0) {
    for (const card of state?.players?.[actor]?.hand || []) {
      const cardId = String(card?.id || "").trim();
      if (!cardId || card?.passCard) continue;
      const month = Number(card?.month || 0);
      if (shakingMonths.has(month)) out.push(`${PLAY_SPECIAL_SHAKE_PREFIX}${cardId}`);
    }
  }
  for (const month of getDeclarableBombMonths(state, actor)) {
    const monthNum = Number(month || 0);
    if (monthNum >= 1 && monthNum <= 12) out.push(`${PLAY_SPECIAL_BOMB_PREFIX}${monthNum}`);
  }
  return out;
}

function applyPlayDecisionCandidate(state, actor, candidate) {
  const special = parsePlaySpecialCandidate(candidate);
  if (!special) return playTurn(state, candidate);
  if (special.kind === "shake_start") {
    const selected = findCardById(state?.players?.[actor]?.hand || [], special.cardId);
    const month = Number(selected?.month || 0);
    if (month < 1) return state;
    const declared = declareShaking(state, actor, month);
    if (!declared || declared === state) return state;
    return playTurn(declared, special.cardId) || declared;
  }
  if (special.kind === "bomb") return declareBomb(state, actor, special.month);
  return state;
}

export function selectDecisionPool(state, actor, options = {}) {
  const previewPlay = !!options.previewPlay;
  if (state.phase === "playing" && (state.currentTurn === actor || previewPlay)) {
    return {
      cards: (state.players?.[actor]?.hand || []).map((c) => c.id),
      specialActions: buildPlayingSpecialActions(state, actor)
    };
  }
  if (state.phase === "select-match" && state.pendingMatch?.playerKey === actor) {
    return { boardCardIds: state.pendingMatch.boardCardIds || [] };
  }
  if (state.phase === "go-stop" && state.pendingGoStop === actor) {
    return { options: ["go", "stop"] };
  }
  if (state.phase === "president-choice" && state.pendingPresident?.playerKey === actor) {
    return { options: ["president_stop", "president_hold"] };
  }
  if (state.phase === "gukjin-choice" && state.pendingGukjinChoice?.playerKey === actor) {
    return { options: ["five", "junk"] };
  }
  if (state.phase === "shaking-confirm" && state.pendingShakingConfirm?.playerKey === actor) {
    return { options: ["shaking_yes", "shaking_no"] };
  }
  return {};
}

export function resolveDecisionType(pool) {
  const cards = pool.cards || null;
  const boardCardIds = pool.boardCardIds || null;
  const options = pool.options || null;
  if (cards) return "play";
  if (boardCardIds) return "match";
  if (options) return "option";
  return null;
}

export function legalCandidatesForDecision(pool, decisionType) {
  if (decisionType === "play") {
    return [
      ...(pool.cards || []).map((x) => String(x)).filter((x) => x.length > 0),
      ...(pool.specialActions || []).map((x) => String(x)).filter((x) => x.length > 0)
    ];
  }
  if (decisionType === "match") {
    return (pool.boardCardIds || []).map((x) => String(x)).filter((x) => x.length > 0);
  }
  if (decisionType === "option") {
    return normalizeOptionCandidates(pool.options || []);
  }
  return [];
}

export function applyDecisionAction(state, actor, decisionType, rawAction) {
  let action = String(rawAction || "").trim();
  if (!action) return state;

  if (decisionType === "play") return applyPlayDecisionCandidate(state, actor, action);
  if (decisionType === "match") return chooseMatch(state, action);
  if (decisionType !== "option") return state;

  action = canonicalOptionAction(action);
  if (action === "go") return chooseGo(state, actor);
  if (action === "stop") return chooseStop(state, actor);
  if (action === "shaking_yes") return chooseShakingYes(state, actor);
  if (action === "shaking_no") return chooseShakingNo(state, actor);
  if (action === "president_stop") return choosePresidentStop(state, actor);
  if (action === "president_hold") return choosePresidentHold(state, actor);
  if (action === "five" || action === "junk") return chooseGukjinMode(state, actor, action);
  return state;
}

export function stateProgressKey(state, options = {}) {
  if (!state) return "null";
  const includeKiboSeq = !!options.includeKiboSeq;
  const hh = Number(state?.players?.human?.hand?.length || 0);
  const ah = Number(state?.players?.ai?.hand?.length || 0);
  const d = Number(state?.deck?.length || 0);
  const parts = [
    String(state.phase || ""),
    String(state.currentTurn || ""),
    String(state.pendingGoStop || ""),
    String(state.pendingMatch?.stage || ""),
    String(state.pendingPresident?.playerKey || ""),
    String(state.pendingShakingConfirm?.playerKey || ""),
    String(state.pendingShakingConfirm?.cardId || ""),
    String(state.pendingShakingConfirm?.month || ""),
    String(state.pendingGukjinChoice?.playerKey || ""),
    String(state.turnSeq || 0)
  ];
  if (includeKiboSeq) {
    parts.push(String(state.kiboSeq || 0));
  }
  parts.push(String(hh), String(ah), String(d));
  return parts.join("|");
}
