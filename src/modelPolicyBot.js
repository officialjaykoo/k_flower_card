import {
  playTurn,
  chooseGo,
  chooseStop,
  chooseShakingYes,
  chooseShakingNo,
  choosePresidentStop,
  choosePresidentHold,
  chooseGukjinMode,
  chooseMatch
} from "./gameEngine.js";

function selectPool(state, actor, options = {}) {
  const previewPlay = !!options.previewPlay;
  if (state.phase === "playing" && (state.currentTurn === actor || previewPlay)) {
    return { cards: (state.players?.[actor]?.hand || []).map((c) => c.id) };
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

function decisionContext(state, actor) {
  const opp = actor === "human" ? "ai" : "human";
  return {
    phase: state.phase,
    turnNoBefore: (state.turnSeq || 0) + 1,
    deckCount: state.deck.length,
    handCountSelf: state.players[actor].hand.length,
    handCountOpp: state.players[opp].hand.length,
    goCountSelf: state.players[actor].goCount || 0,
    goCountOpp: state.players[opp].goCount || 0,
    shakeCountSelf: state.players?.[actor]?.events?.shaking || 0,
    shakeCountOpp: state.players?.[opp]?.events?.shaking || 0,
    carryOverMultiplier: Number(state.carryOverMultiplier || 1),
    isFirstAttacker: state.startingTurnKey === actor ? 1 : 0
  };
}

function tracePhaseCode(phase) {
  const map = {
    playing: 1,
    "select-match": 2,
    "go-stop": 3,
    "president-choice": 4,
    "gukjin-choice": 5,
    "shaking-confirm": 6,
    resolution: 7
  };
  return Number(map[phase] || 0);
}

function policyContextKey(trace, decisionType) {
  const dc = trace.dc || {};
  const sp = trace.sp || {};
  const deckBucket = Math.floor((dc.deckCount || 0) / 3);
  const rawPhase = dc.phase;
  const phaseCode = Number.isFinite(Number(rawPhase))
    ? Math.floor(Number(rawPhase))
    : tracePhaseCode(rawPhase);
  const handSelf = dc.handCountSelf || 0;
  const handOpp = dc.handCountOpp || 0;
  const goSelf = dc.goCountSelf || 0;
  const goOpp = dc.goCountOpp || 0;
  const carry = Math.max(1, Math.floor(Number(dc.carryOverMultiplier || 1)));
  const shakeSelf = Math.min(3, Math.floor(Number(dc.shakeCountSelf || 0)));
  const shakeOpp = Math.min(3, Math.floor(Number(dc.shakeCountOpp || 0)));
  const cands = (sp.cards || sp.boardCardIds || sp.options || []).length;
  return [
    `dt=${decisionType}`,
    `ph=${phaseCode}`,
    `o=${trace.o || "?"}`,
    `db=${deckBucket}`,
    `hs=${handSelf}`,
    `ho=${handOpp}`,
    `gs=${goSelf}`,
    `go=${goOpp}`,
    `cm=${carry}`,
    `ss=${shakeSelf}`,
    `so=${shakeOpp}`,
    `cc=${cands}`
  ].join("|");
}

function policyProb(model, sample, choice) {
  const alpha = Number(model?.alpha ?? 1.0);
  const dt = sample.decisionType;
  const candidates = sample.candidates || [];
  const ck = sample.contextKey;
  const k = Math.max(1, candidates.length);

  const dtContextCounts = model?.context_counts?.[dt] || {};
  const dtContextTotals = model?.context_totals?.[dt] || {};
  const ctxCounts = dtContextCounts?.[ck];
  if (ctxCounts) {
    const total = Number(dtContextTotals?.[ck] || 0);
    return (Number(ctxCounts?.[choice] || 0) + alpha) / (total + alpha * k);
  }

  const dtGlobal = model?.global_counts?.[dt] || {};
  let total = 0;
  for (const c of candidates) total += Number(dtGlobal?.[c] || 0);
  return (Number(dtGlobal?.[choice] || 0) + alpha) / (total + alpha * k);
}

function modelPickCandidate(state, actor, policyModel) {
  const scored = getModelCandidateProbabilities(state, actor, policyModel);
  if (!scored) return null;
  let best = scored.candidates[0];
  let bestP = -1;
  for (const c of scored.candidates) {
    const p = Number(scored.probabilities?.[String(c)] || 0);
    if (p > bestP) {
      best = c;
      bestP = p;
    }
  }
  return { decisionType: scored.decisionType, candidate: best };
}

export function getModelCandidateProbabilities(state, actor, policyModel, options = {}) {
  if (!policyModel) return null;
  const sp = selectPool(state, actor, options);
  const cards = sp.cards || null;
  const boardCardIds = sp.boardCardIds || null;
  const optionCandidates = sp.options || null;
  const candidates = cards || boardCardIds || optionCandidates || [];
  if (!candidates.length) return null;

  const decisionType = cards ? "play" : boardCardIds ? "match" : "option";
  const order = actor === state.startingTurnKey ? "first" : "second";
  const dc = decisionContext(state, actor);
  const contextKey = policyContextKey({ o: order, dc, sp: { cards, boardCardIds, options: optionCandidates } }, decisionType);
  const baseSample = { decisionType, candidates, contextKey };
  const probabilities = {};
  for (const c of candidates) {
    const label = decisionType === "option" ? c : String(c);
    const p = policyProb(policyModel, baseSample, label);
    probabilities[String(c)] = p;
  }
  return { decisionType, candidates, probabilities };
}

export function modelPolicyPlay(state, actor, policyModel) {
  if (!policyModel) return state;
  const picked = modelPickCandidate(state, actor, policyModel);
  if (!picked) return state;
  const c = picked.candidate;
  if (picked.decisionType === "play") return playTurn(state, c);
  if (picked.decisionType === "match") return chooseMatch(state, c);
  if (picked.decisionType !== "option") return state;
  if (c === "go") return chooseGo(state, actor);
  if (c === "stop") return chooseStop(state, actor);
  if (c === "shaking_yes") return chooseShakingYes(state, actor);
  if (c === "shaking_no") return chooseShakingNo(state, actor);
  if (c === "president_stop") return choosePresidentStop(state, actor);
  if (c === "president_hold") return choosePresidentHold(state, actor);
  if (c === "five" || c === "junk") return chooseGukjinMode(state, actor, c);
  return state;
}
