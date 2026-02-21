import {
  calculateScore,
  playTurn,
  chooseGo,
  chooseStop,
  chooseShakingYes,
  chooseShakingNo,
  choosePresidentStop,
  choosePresidentHold,
  chooseGukjinMode,
  chooseMatch
} from "../engine/index.js";

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
  const actorKeys = Object.keys(state?.players || {});
  const opp = actorKeys.find((k) => k !== actor) || (actor === "human" ? "ai" : "human");
  const scoreSelf = calculateScore(state.players[actor], state.players[opp], state.ruleKey);
  const scoreOpp = calculateScore(state.players[opp], state.players[actor], state.ruleKey);
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
    currentScoreSelf: Number(scoreSelf?.total || 0),
    currentScoreOpp: Number(scoreOpp?.total || 0),
    scoreDiff: Number(scoreSelf?.total || 0) - Number(scoreOpp?.total || 0),
    piBakRisk: scoreSelf?.bak?.pi ? 1 : 0,
    gwangBakRisk: scoreSelf?.bak?.gwang ? 1 : 0,
    mongBakRisk: scoreSelf?.bak?.mongBak ? 1 : 0,
    jokboProgressSelfSum: 0,
    jokboProgressOppSum: 0,
    selfJokboThreatProb: 0,
    selfJokboOneAwayProb: 0,
    selfGwangThreatProb: 0,
    oppJokboThreatProb: 0,
    oppJokboOneAwayProb: 0,
    oppGwangThreatProb: 0,
    goStopDeltaProxy: 0,
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

function policyContextKeyMode(policyModel) {
  const raw = String(policyModel?.context_key_mode || "").trim().toLowerCase();
  if (raw === "bucketed_v3") return "bucketed_v3";
  if (raw === "bucketed_v2") return "bucketed_v2";
  return "raw_v1";
}

function bucketHandSelf(handSelf) {
  const hs = Math.floor(Number(handSelf || 0));
  if (hs >= 8) return "8p";
  if (hs >= 5) return "5_7";
  if (hs >= 2) return "2_4";
  return "0_1";
}

function bucketHandDiff(handDiff) {
  const hd = Math.floor(Number(handDiff || 0));
  if (hd <= -3) return "n3";
  if (hd <= -1) return "n2_1";
  if (hd === 0) return "z0";
  if (hd <= 2) return "p1_2";
  return "p3";
}

function bucketScoreDiff(scoreDiff) {
  const sd = Number(scoreDiff || 0);
  if (sd <= -10) return "n10";
  if (sd <= -3) return "n9_3";
  if (sd < 3) return "z2";
  if (sd < 10) return "p3_9";
  return "p10";
}

function bucketGoCount(goCount) {
  const g = Math.floor(Number(goCount || 0));
  if (g <= 0) return "0";
  if (g === 1) return "1";
  return "2p";
}

function bucketCandidates(cands) {
  const c = Math.floor(Number(cands || 0));
  if (c <= 1) return "1";
  if (c === 2) return "2";
  if (c === 3) return "3";
  return "4p";
}

function bucketRiskTotal(total) {
  const t = Math.max(0, Math.floor(Number(total || 0)));
  if (t <= 0) return "0";
  if (t === 1) return "1";
  return "2p";
}

function bucketThreatPercent(v) {
  const x = Math.max(0, Math.min(100, Math.floor(Number(v || 0))));
  if (x >= 70) return "h";
  if (x >= 35) return "m";
  return "l";
}

function bucketProgressDelta(delta) {
  const d = Math.floor(Number(delta || 0));
  if (d <= -3) return "n3";
  if (d <= -1) return "n2_1";
  if (d === 0) return "z0";
  if (d <= 2) return "p1_2";
  return "p3";
}

function bucketGoStopSignal(gsdPermille) {
  const x = Math.floor(Number(gsdPermille || 0));
  if (x <= -1800) return "n2";
  if (x <= -600) return "n1";
  if (x < 600) return "z0";
  if (x < 1800) return "p1";
  return "p2";
}

function policyContextKey(trace, decisionType, policyModel = null) {
  const dc = trace.dc || {};
  const sp = trace.sp || {};
  const deckBucket = Math.floor((dc.deckCount || 0) / 3);
  const rawPhase = dc.phase;
  const phaseCode = Number.isFinite(Number(rawPhase))
    ? Math.floor(Number(rawPhase))
    : tracePhaseCode(rawPhase);
  const handSelf = dc.handCountSelf || 0;
  const handDiff =
    dc.handCountDiff != null
      ? Number(dc.handCountDiff || 0)
      : Number(handSelf || 0) - Number(dc.handCountOpp || 0);
  const goSelf = dc.goCountSelf || 0;
  const goOpp = dc.goCountOpp || 0;
  const shakeSelf = Math.min(3, Math.floor(Number(dc.shakeCountSelf || 0)));
  const shakeOpp = Math.min(3, Math.floor(Number(dc.shakeCountOpp || 0)));
  const scoreDiff =
    dc.scoreDiff != null
      ? Number(dc.scoreDiff || 0)
      : Number(dc.currentScoreSelf || 0) - Number(dc.currentScoreOpp || 0);
  const bakRiskTotal =
    (Number(dc.piBakRisk || 0) ? 1 : 0) +
    (Number(dc.gwangBakRisk || 0) ? 1 : 0) +
    (Number(dc.mongBakRisk || 0) ? 1 : 0);
  const oppThreat = Math.max(
    Number(dc.oppJokboThreatProb || 0) * 100,
    Number(dc.oppJokboOneAwayProb || 0) * 100,
    Number(dc.oppGwangThreatProb || 0) * 100
  );
  const selfThreat = Math.max(
    Number(dc.selfJokboThreatProb || 0) * 100,
    Number(dc.selfJokboOneAwayProb || 0) * 100,
    Number(dc.selfGwangThreatProb || 0) * 100
  );
  const progressDelta = Number(dc.jokboProgressSelfSum || 0) - Number(dc.jokboProgressOppSum || 0);
  const goStopSignal = Math.round(Number(dc.goStopDeltaProxy || 0) * 1000);
  const cands = (sp.cards || sp.boardCardIds || sp.options || []).length;
  const keyMode = policyContextKeyMode(policyModel);
  const hsToken = keyMode === "bucketed_v2" ? bucketHandSelf(handSelf) : String(Math.floor(Number(handSelf || 0)));
  const hdToken = keyMode === "bucketed_v2" ? bucketHandDiff(handDiff) : String(Math.floor(Number(handDiff || 0)));
  const sdToken = keyMode === "bucketed_v2" ? bucketScoreDiff(scoreDiff) : String(Math.floor(scoreDiff));
  const gsToken = keyMode === "bucketed_v2" ? bucketGoCount(goSelf) : String(Math.floor(Number(goSelf || 0)));
  const goToken = keyMode === "bucketed_v2" ? bucketGoCount(goOpp) : String(Math.floor(Number(goOpp || 0)));
  const ccToken = keyMode === "bucketed_v2" ? bucketCandidates(cands) : String(Math.floor(Number(cands || 0)));
  const base = [
    `dt=${decisionType}`,
    `ph=${phaseCode}`,
    `o=${trace.o || "?"}`,
    `db=${deckBucket}`,
    `hs=${hsToken}`,
    `hd=${hdToken}`,
    `sd=${sdToken}`,
    `gs=${gsToken}`,
    `go=${goToken}`,
    `ss=${shakeSelf}`,
    `so=${shakeOpp}`,
    `cc=${ccToken}`
  ];
  if (keyMode === "bucketed_v3") {
    base.push(`br=${bucketRiskTotal(bakRiskTotal)}`);
    base.push(`ot=${bucketThreatPercent(oppThreat)}`);
    base.push(`st=${bucketThreatPercent(selfThreat)}`);
    base.push(`jp=${bucketProgressDelta(progressDelta)}`);
    base.push(`gd=${bucketGoStopSignal(goStopSignal)}`);
  }
  return base.join("|");
}

function rawNoScorePolicyContextKey(trace, decisionType) {
  const dc = trace.dc || {};
  const sp = trace.sp || {};
  const deckBucket = Math.floor((dc.deckCount || 0) / 3);
  const rawPhase = dc.phase;
  const phaseCode = Number.isFinite(Number(rawPhase))
    ? Math.floor(Number(rawPhase))
    : tracePhaseCode(rawPhase);
  const handSelf = dc.handCountSelf || 0;
  const handDiff =
    dc.handCountDiff != null
      ? Number(dc.handCountDiff || 0)
      : Number(handSelf || 0) - Number(dc.handCountOpp || 0);
  const goSelf = dc.goCountSelf || 0;
  const goOpp = dc.goCountOpp || 0;
  const shakeSelf = Math.min(3, Math.floor(Number(dc.shakeCountSelf || 0)));
  const shakeOpp = Math.min(3, Math.floor(Number(dc.shakeCountOpp || 0)));
  const cands = (sp.cards || sp.boardCardIds || sp.options || []).length;
  return [
    `dt=${decisionType}`,
    `ph=${phaseCode}`,
    `o=${trace.o || "?"}`,
    `db=${deckBucket}`,
    `hs=${Math.floor(Number(handSelf || 0))}`,
    `hd=${Math.floor(Number(handDiff || 0))}`,
    `gs=${Math.floor(Number(goSelf || 0))}`,
    `go=${Math.floor(Number(goOpp || 0))}`,
    `ss=${shakeSelf}`,
    `so=${shakeOpp}`,
    `cc=${Math.floor(Number(cands || 0))}`
  ].join("|");
}

function legacyPolicyContextKey(trace, decisionType) {
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
  const contextKeys = Array.isArray(sample.contextKeys)
    ? sample.contextKeys
    : [sample.contextKey].filter(Boolean);
  const k = Math.max(1, candidates.length);

  const dtContextCounts = model?.context_counts?.[dt] || {};
  const dtContextTotals = model?.context_totals?.[dt] || {};
  for (const ck of contextKeys) {
    const ctxCounts = dtContextCounts?.[ck];
    if (ctxCounts) {
      const total = Number(dtContextTotals?.[ck] || 0);
      return (Number(ctxCounts?.[choice] || 0) + alpha) / (total + alpha * k);
    }
  }

  const dtGlobal = model?.global_counts?.[dt] || {};
  let total = 0;
  for (const c of candidates) total += Number(dtGlobal?.[c] || 0);
  return (Number(dtGlobal?.[choice] || 0) + alpha) / (total + alpha * k);
}

function canonicalOptionAction(action) {
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
    choose_junk: "junk"
  };
  return aliases[a] || a;
}

function normalizeOptionCandidates(items) {
  if (!Array.isArray(items)) return [];
  const out = [];
  const seen = new Set();
  for (const raw of items) {
    const v = canonicalOptionAction(raw);
    if (!v) continue;
    if (seen.has(v)) continue;
    seen.add(v);
    out.push(v);
  }
  return out;
}

function modelOptionActionSet(policyModel) {
  const raw = policyModel?.action_vocab?.option_actions;
  if (!Array.isArray(raw) || raw.length === 0) return null;
  const normalized = normalizeOptionCandidates(raw);
  if (!normalized.length) return null;
  return new Set(normalized);
}

function modelDecisionTypesSet(policyModel) {
  const raw = policyModel?.action_vocab?.decision_types;
  if (!Array.isArray(raw) || raw.length === 0) return null;
  const normalized = raw
    .map((x) => String(x || "").trim().toLowerCase())
    .filter((x) => x === "play" || x === "match" || x === "option");
  if (!normalized.length) return null;
  return new Set(normalized);
}

function legalCandidatesForInference(sp, decisionType, policyModel) {
  const allowedDecisionTypes = modelDecisionTypesSet(policyModel);
  if (allowedDecisionTypes && !allowedDecisionTypes.has(String(decisionType || "").toLowerCase())) {
    return [];
  }
  if (decisionType === "play") {
    return (sp.cards || []).map((x) => String(x)).filter((x) => x.length > 0);
  }
  if (decisionType === "match") {
    return (sp.boardCardIds || []).map((x) => String(x)).filter((x) => x.length > 0);
  }
  if (decisionType === "option") {
    const legal = normalizeOptionCandidates(sp.options || []);
    const allowed = modelOptionActionSet(policyModel);
    if (!allowed) return legal;
    const filtered = legal.filter((x) => allowed.has(x));
    return filtered.length > 0 ? filtered : legal;
  }
  return [];
}

function deterministicUnitFromText(text) {
  const s = String(text || "");
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i += 1) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return (h >>> 0) / 0xffffffff;
}

function softmaxSampleFromScores(items, scoreOf, topK, temperature, entropyKey) {
  const ranked = [...items].sort((a, b) => Number(scoreOf.get(b) || -Infinity) - Number(scoreOf.get(a) || -Infinity));
  const k = Math.max(1, Math.min(ranked.length, Math.floor(Number(topK || ranked.length))));
  const chosen = ranked.slice(0, k);
  if (chosen.length <= 1) return chosen[0];
  const temp = Math.max(0.05, Number(temperature || 1.0));
  const raw = chosen.map((c) => Number(scoreOf.get(c) || -Infinity));
  const maxRaw = Math.max(...raw);
  const exps = raw.map((x) => Math.exp((x - maxRaw) / temp));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  if (!(sumExp > 0)) return chosen[0];
  const u = deterministicUnitFromText(entropyKey);
  let acc = 0;
  for (let i = 0; i < chosen.length; i += 1) {
    acc += exps[i] / sumExp;
    if (u <= acc) return chosen[i];
  }
  return chosen[chosen.length - 1];
}

function mixedStrategyConfig(policyModel) {
  const mode = String(policyModel?.mixed_strategy || "on").trim().toLowerCase();
  return {
    enabled: !(mode === "off" || mode === "false" || mode === "0" || mode === "disabled"),
    topKPlay: Math.max(1, Math.floor(Number(policyModel?.mixed_topk_play || 3))),
    topKMatch: Math.max(1, Math.floor(Number(policyModel?.mixed_topk_match || 2))),
    topKOption: Math.max(1, Math.floor(Number(policyModel?.mixed_topk_option || 2))),
    tempPlay: Math.max(0.05, Number(policyModel?.mixed_temp_play || 0.9)),
    tempMatch: Math.max(0.05, Number(policyModel?.mixed_temp_match || 0.8)),
    tempOption: Math.max(0.05, Number(policyModel?.mixed_temp_option || 0.7)),
  };
}

function modelPickCandidate(state, actor, policyModel) {
  const scored = getModelCandidateProbabilities(state, actor, policyModel);
  if (!scored) return null;
  const cfg = mixedStrategyConfig(policyModel);
  const scoreMap = new Map();
  for (const c of scored.candidates) {
    const p = Number(scored.probabilities?.[String(c)] || 0);
    scoreMap.set(c, Math.log(Math.max(1e-12, p)));
  }
  let best = scored.candidates[0];
  if (cfg.enabled) {
    const decisionType = scored.decisionType;
    const topK = decisionType === "play" ? cfg.topKPlay : decisionType === "match" ? cfg.topKMatch : cfg.topKOption;
    const temp =
      decisionType === "play" ? cfg.tempPlay : decisionType === "match" ? cfg.tempMatch : cfg.tempOption;
    const entropyKey = [
      actor,
      state.turnSeq || 0,
      state.phase || "",
      scored.decisionType || "",
      scored.candidates.join(","),
    ].join("|");
    best = softmaxSampleFromScores(scored.candidates, scoreMap, topK, temp, entropyKey);
  } else {
    let bestScore = -Infinity;
    for (const c of scored.candidates) {
      const s = Number(scoreMap.get(c) || -Infinity);
      if (s > bestScore) {
        best = c;
        bestScore = s;
      }
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
  const decisionType = cards ? "play" : boardCardIds ? "match" : optionCandidates ? "option" : null;
  if (!decisionType) return null;
  const candidates = legalCandidatesForInference(sp, decisionType, policyModel);
  if (!candidates.length) return null;
  const order = actor === state.startingTurnKey ? "first" : "second";
  const dc = decisionContext(state, actor);
  const traceLike = { o: order, dc, sp: { cards, boardCardIds, options: optionCandidates } };
  const contextKey = policyContextKey(traceLike, decisionType, policyModel);
  const rawNoScoreKey = rawNoScorePolicyContextKey(traceLike, decisionType);
  const legacyKey = legacyPolicyContextKey(traceLike, decisionType);
  const contextKeys = Array.from(new Set([contextKey, rawNoScoreKey, legacyKey]));
  const baseSample = { decisionType, candidates, contextKey, contextKeys };
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
  const sp = selectPool(state, actor);
  const cards = sp.cards || null;
  const boardCardIds = sp.boardCardIds || null;
  const optionCandidates = sp.options || null;
  const decisionType = cards ? "play" : boardCardIds ? "match" : optionCandidates ? "option" : null;
  if (!decisionType) return state;
  const legal = legalCandidatesForInference(sp, decisionType, policyModel);
  if (!legal.length) return state;
  let c = picked.candidate;
  if (decisionType === "option") c = canonicalOptionAction(c);
  if (!legal.includes(String(c))) c = legal[0];
  if (decisionType === "play") return playTurn(state, c);
  if (decisionType === "match") return chooseMatch(state, c);
  if (decisionType !== "option") return state;
  if (c === "go") return chooseGo(state, actor);
  if (c === "stop") return chooseStop(state, actor);
  if (c === "shaking_yes") return chooseShakingYes(state, actor);
  if (c === "shaking_no") return chooseShakingNo(state, actor);
  if (c === "president_stop") return choosePresidentStop(state, actor);
  if (c === "president_hold") return choosePresidentHold(state, actor);
  if (c === "five" || c === "junk") return chooseGukjinMode(state, actor, c);
  return state;
}
