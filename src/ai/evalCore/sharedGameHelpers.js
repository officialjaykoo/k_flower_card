import {
  initSimulationGame,
  startSimulationGame,
  createSeededRng,
  getDeclarableShakingMonths,
  getDeclarableBombMonths,
  declareShaking,
  declareBomb,
  playTurn,
  chooseMatch,
  chooseGo,
  chooseStop,
  chooseShakingYes,
  chooseShakingNo,
  choosePresidentStop,
  choosePresidentHold,
  chooseGukjinMode,
} from "../../engine/index.js";

export function parseHybridPlayGoSpec(token) {
  const m = String(token || "")
    .trim()
    .match(/^hybrid_play_go\(\s*([^,]+)\s*,\s*([^,]+?)(?:\s*,\s*([^)]+)\s*)?\)$/i);
  if (!m) {
    return null;
  }
  return {
    modelToken: String(m[1] || "").trim(),
    goStopToken: String(m[2] || "").trim(),
    heuristicToken: String(m[3] || "").trim(),
  };
}

export function parseHybridPlaySpec(token) {
  const m = String(token || "")
    .trim()
    .match(/^hybrid_play\(\s*([^,]+)\s*,\s*([^)]+)\s*\)$/i);
  if (!m) {
    return null;
  }
  return {
    modelToken: String(m[1] || "").trim(),
    fallbackToken: String(m[2] || "").trim(),
  };
}

export function canonicalOptionAction(action) {
  const a = String(action || "").trim();
  if (!a) return "";
  const aliases = {
    choose_go: "go",
    choose_stop: "stop",
    choose_shaking_yes: "shaking_yes",
    choose_shaking_no: "shaking_no",
    choose_president_stop: "president_stop",
    choose_president_hold: "president_hold",
    choose_five: "five",
    choose_junk: "junk",
  };
  return aliases[a] || a;
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

const PLAY_SPECIAL_SHAKE_PREFIX = "shake_start:";
const PLAY_SPECIAL_BOMB_PREFIX = "bomb:";

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

export function buildPlayingSpecialActions(state, actor) {
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

export function applyPlayCandidate(state, actor, candidate) {
  const special = parsePlaySpecialCandidate(candidate);
  if (!special) return playTurn(state, candidate);
  if (special.kind === "shake_start") {
    const selected = (state?.players?.[actor]?.hand || []).find(
      (card) => String(card?.id || "") === String(special.cardId || "")
    );
    const month = Number(selected?.month || 0);
    if (month < 1) return state;
    const declared = declareShaking(state, actor, month);
    if (!declared || declared === state) return state;
    return playTurn(declared, special.cardId) || declared;
  }
  if (special.kind === "bomb") return declareBomb(state, actor, special.month);
  return state;
}

export function selectPool(state, actor) {
  if (state.phase === "playing" && state.currentTurn === actor) {
    return {
      cards: (state.players?.[actor]?.hand || []).map((c) => c.id),
      specialActions: buildPlayingSpecialActions(state, actor),
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

export function legalCandidatesForDecision(sp, decisionType) {
  if (decisionType === "play") {
    return [
      ...(sp.cards || []).map((x) => String(x)).filter((x) => x.length > 0),
      ...(sp.specialActions || []).map((x) => String(x)).filter((x) => x.length > 0),
    ];
  }
  if (decisionType === "match") {
    return (sp.boardCardIds || []).map((x) => String(x)).filter((x) => x.length > 0);
  }
  if (decisionType === "option") {
    return normalizeOptionCandidates(sp.options || []);
  }
  return [];
}

export function applyAction(state, actor, decisionType, rawAction) {
  let action = String(rawAction || "").trim();
  if (!action) return state;
  if (decisionType === "play") return applyPlayCandidate(state, actor, action);
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

export function stateProgressKey(state) {
  if (!state) return "null";
  const hh = Number(state?.players?.human?.hand?.length || 0);
  const ah = Number(state?.players?.ai?.hand?.length || 0);
  const d = Number(state?.deck?.length || 0);
  return [
    String(state.phase || ""),
    String(state.currentTurn || ""),
    String(state.pendingGoStop || ""),
    String(state.pendingMatch?.stage || ""),
    String(state.pendingPresident?.playerKey || ""),
    String(state.pendingShakingConfirm?.playerKey || ""),
    String(state.pendingShakingConfirm?.cardId || ""),
    String(state.pendingShakingConfirm?.month || ""),
    String(state.pendingGukjinChoice?.playerKey || ""),
    String(state.turnSeq || 0),
    String(state.kiboSeq || 0),
    String(hh),
    String(ah),
    String(d),
  ].join("|");
}

export function normalizeDecisionCandidate(decisionType, candidate) {
  if (decisionType === "option") return canonicalOptionAction(candidate);
  return String(candidate || "").trim();
}

export function randomChoice(arr, rng) {
  if (!arr.length) return null;
  const idx = Math.max(0, Math.min(arr.length - 1, Math.floor(Number(rng() || 0) * arr.length)));
  return arr[idx];
}

export function randomLegalAction(state, actor, rng) {
  const sp = selectPool(state, actor);
  const cards = sp.cards || null;
  const boardCardIds = sp.boardCardIds || null;
  const options = sp.options || null;
  const decisionType = cards ? "play" : boardCardIds ? "match" : options ? "option" : null;
  if (!decisionType) return state;
  const candidates = legalCandidatesForDecision(sp, decisionType);
  if (!candidates.length) return state;
  const picked = randomChoice(candidates, rng);
  return applyAction(state, actor, decisionType, picked);
}

export function startRound(seed, firstTurnKey, kiboDetail = "lean") {
  return initSimulationGame("A", createSeededRng(`${seed}|game`), {
    kiboDetail,
    firstTurnKey,
  });
}

export function continueRound(prevEndState, seed, firstTurnKey, kiboDetail = "lean") {
  return startSimulationGame(prevEndState, createSeededRng(`${seed}|game`), {
    kiboDetail,
    keepGold: true,
    useCarryOver: true,
    firstTurnKey,
  });
}

export function clamp01(v) {
  const x = Number(v || 0);
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  return x;
}

export function quantile(values, q) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.max(0, Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * q)));
  return sorted[idx];
}
