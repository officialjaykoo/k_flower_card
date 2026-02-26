import {
  calculateScore,
  scoringPiCount,
  getDeclarableShakingMonths,
  getDeclarableBombMonths,
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

/* ============================================================================
 * NEAT model policy runtime
 * - compile genome once and cache
 * - score legal candidates
 * - convert score to action/state transition
 * ========================================================================== */
const NEAT_MODEL_FORMAT = "neat_python_genome_v1";
const COMPILED_NEAT_CACHE = new WeakMap();

/* 1) Candidate-space normalization */
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
    if (!v || seen.has(v)) continue;
    seen.add(v);
    out.push(v);
  }
  return out;
}

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

function resolveDecisionType(sp) {
  const cards = sp.cards || null;
  const boardCardIds = sp.boardCardIds || null;
  const options = sp.options || null;
  if (cards) return "play";
  if (boardCardIds) return "match";
  if (options) return "option";
  return null;
}

function legalCandidatesForDecision(sp, decisionType) {
  if (decisionType === "play") {
    return (sp.cards || []).map((x) => String(x)).filter((x) => x.length > 0);
  }
  if (decisionType === "match") {
    return (sp.boardCardIds || []).map((x) => String(x)).filter((x) => x.length > 0);
  }
  if (decisionType === "option") {
    return normalizeOptionCandidates(sp.options || []);
  }
  return [];
}

/* 2) Feature extraction helpers */
function clamp01(x) {
  const v = Number(x || 0);
  if (v <= 0) return 0;
  if (v >= 1) return 1;
  return v;
}

function tanhNorm(x, scale) {
  const s = Math.max(1e-6, Number(scale || 1));
  return Math.tanh(Number(x || 0) / s);
}

function findCardById(cards, cardId) {
  const id = String(cardId || "");
  if (!Array.isArray(cards)) return null;
  return cards.find((c) => String(c?.id || "") === id) || null;
}

function optionCode(action) {
  const a = canonicalOptionAction(action);
  const map = {
    go: 1,
    stop: 2,
    shaking_yes: 3,
    shaking_no: 4,
    president_stop: 5,
    president_hold: 6,
    five: 7,
    junk: 8
  };
  return Number(map[a] || 0) / 8.0;
}

function candidateCard(state, actor, decisionType, candidate) {
  if (decisionType === "play") {
    return findCardById(state?.players?.[actor]?.hand || [], candidate);
  }
  if (decisionType === "match") {
    return findCardById(state?.board || [], candidate);
  }
  return null;
}

function countCardsByMonth(cards, month) {
  const targetMonth = Number(month || 0);
  if (!Array.isArray(cards) || targetMonth <= 0) return 0;
  let count = 0;
  for (const card of cards) {
    if (Number(card?.month || 0) === targetMonth) count += 1;
  }
  return count;
}

function hasComboTag(card, tag) {
  return Array.isArray(card?.comboTags) && card.comboTags.includes(tag);
}

function countCapturedComboTag(player, zone, tag) {
  const cards = player?.captured?.[zone] || [];
  if (!Array.isArray(cards)) return 0;
  const seen = new Set();
  let count = 0;
  for (const card of cards) {
    const id = String(card?.id || "");
    if (!id || seen.has(id)) continue;
    seen.add(id);
    if (hasComboTag(card, tag)) count += 1;
  }
  return count;
}

function isDoublePiCard(card) {
  if (!card) return false;
  const id = String(card?.id || "");
  const category = String(card?.category || "");
  const piValue = Number(card?.piValue || 0);
  if (id === "I0") return true;
  return category === "junk" && piValue >= 2;
}

function resolveCandidateMonth(state, actor, decisionType, card) {
  const cardMonth = Number(card?.month || 0);
  if (cardMonth >= 1) return cardMonth;
  if (decisionType === "option") {
    if (state?.phase === "shaking-confirm" && state?.pendingShakingConfirm?.playerKey === actor) {
      const pendingMonth = Number(state?.pendingShakingConfirm?.month || 0);
      if (pendingMonth >= 1) return pendingMonth;
    }
    if (state?.phase === "president-choice" && state?.pendingPresident?.playerKey === actor) {
      const pendingMonth = Number(state?.pendingPresident?.month || 0);
      if (pendingMonth >= 1) return pendingMonth;
    }
  }
  return 0;
}

function matchOpportunityDensity(state, month) {
  const boardMonthCount = countCardsByMonth(state?.board || [], month);
  return clamp01(boardMonthCount / 3.0);
}

function immediateMatchPossible(state, decisionType, month) {
  if (decisionType === "match") return 1;
  if (month <= 0) return 0;
  return countCardsByMonth(state?.board || [], month) > 0 ? 1 : 0;
}

function monthTotalCards(month) {
  const m = Number(month || 0);
  if (m >= 1 && m <= 12) return 4;
  if (m === 13) return 2;
  return 0;
}

function collectKnownCardsForMonthRatio(state, actor) {
  const out = [];
  const pushAll = (cards) => {
    if (!Array.isArray(cards)) return;
    for (const card of cards) out.push(card);
  };
  pushAll(state?.board || []);
  for (const side of ["human", "ai"]) {
    const captured = state?.players?.[side]?.captured || {};
    pushAll(captured.kwang || []);
    pushAll(captured.five || []);
    pushAll(captured.ribbon || []);
    pushAll(captured.junk || []);
  }
  pushAll(state?.players?.[actor]?.hand || []);
  return out;
}

function candidateMonthKnownRatio(state, actor, month) {
  const total = monthTotalCards(month);
  if (total <= 0) return 0;
  const cards = collectKnownCardsForMonthRatio(state, actor);
  const seen = new Set();
  let known = 0;
  for (const card of cards) {
    if (!card) continue;
    const id = String(card.id || "");
    if (!id || seen.has(id)) continue;
    seen.add(id);
    if (Number(card.month || 0) === Number(month)) known += 1;
  }
  return clamp01(known / total);
}

function decisionAvailabilityFlags(state, actor) {
  if (state?.phase === "shaking-confirm" && state?.pendingShakingConfirm?.playerKey === actor) {
    return { hasShake: 1, hasBomb: 0 };
  }
  if (state?.phase !== "playing" || state?.currentTurn !== actor) {
    return { hasShake: 0, hasBomb: 0 };
  }
  const shakingMonths = getDeclarableShakingMonths(state, actor);
  const bombMonths = getDeclarableBombMonths(state, actor);
  return {
    hasShake: Array.isArray(shakingMonths) && shakingMonths.length > 0 ? 1 : 0,
    hasBomb: Array.isArray(bombMonths) && bombMonths.length > 0 ? 1 : 0
  };
}

function currentMultiplierNorm(state, scoreSelf) {
  const carry = Math.max(1.0, Number(state?.carryOverMultiplier || 1.0));
  const mul = Math.max(1.0, Number(scoreSelf?.multiplier || 1.0));
  const currentMultiplier = mul * carry;
  return clamp01((currentMultiplier - 1.0) / 15.0);
}

function featureVector(state, actor, decisionType, candidate, legalCount, inputDim) {
  const opp = actor === "human" ? "ai" : "human";
  const scoreSelf = calculateScore(state.players[actor], state.players[opp], state.ruleKey);
  const scoreOpp = calculateScore(state.players[opp], state.players[actor], state.ruleKey);

  const phase = String(state.phase || "");
  const card = candidateCard(state, actor, decisionType, candidate);
  const month = resolveCandidateMonth(state, actor, decisionType, card);
  const piValue = Number(card?.piValue || 0);
  const category = String(card?.category || "");
  const selfGwangCount = Number(state?.players?.[actor]?.captured?.kwang?.length || 0);
  const oppGwangCount = Number(state?.players?.[opp]?.captured?.kwang?.length || 0);
  const selfPiCount = Number(scoringPiCount(state.players[actor]) || 0);
  const oppPiCount = Number(scoringPiCount(state.players[opp]) || 0);

  const selfGodori = countCapturedComboTag(state.players?.[actor], "five", "fiveBirds");
  const oppGodori = countCapturedComboTag(state.players?.[opp], "five", "fiveBirds");
  const selfCheongdan = countCapturedComboTag(state.players?.[actor], "ribbon", "blueRibbons");
  const oppCheongdan = countCapturedComboTag(state.players?.[opp], "ribbon", "blueRibbons");
  const selfHongdan = countCapturedComboTag(state.players?.[actor], "ribbon", "redRibbons");
  const oppHongdan = countCapturedComboTag(state.players?.[opp], "ribbon", "redRibbons");
  const selfChodan = countCapturedComboTag(state.players?.[actor], "ribbon", "plainRibbons");
  const oppChodan = countCapturedComboTag(state.players?.[opp], "ribbon", "plainRibbons");

  const selfCanStop = Number(scoreSelf?.total || 0) >= 7 ? 1 : 0;
  const oppCanStop = Number(scoreOpp?.total || 0) >= 7 ? 1 : 0;
  const { hasShake, hasBomb } = decisionAvailabilityFlags(state, actor);

  const features = [
    phase === "playing" ? 1 : 0,
    phase === "select-match" ? 1 : 0,
    phase === "go-stop" ? 1 : 0,
    phase === "president-choice" ? 1 : 0,
    phase === "gukjin-choice" ? 1 : 0,
    phase === "shaking-confirm" ? 1 : 0,

    decisionType === "play" ? 1 : 0,
    decisionType === "match" ? 1 : 0,
    decisionType === "option" ? 1 : 0,

    clamp01((state.deck?.length || 0) / 30.0),
    clamp01((state.players?.[actor]?.hand?.length || 0) / 10.0),
    clamp01((state.players?.[opp]?.hand?.length || 0) / 10.0),
    clamp01((state.players?.[actor]?.goCount || 0) / 5.0),
    clamp01((state.players?.[opp]?.goCount || 0) / 5.0),
    tanhNorm((scoreSelf?.total || 0) - (scoreOpp?.total || 0), 10.0),
    tanhNorm(scoreSelf?.total || 0, 10.0),
    clamp01(Number(legalCount || 0) / 10.0),

    clamp01(piValue / 5.0),
    category === "kwang" ? 1 : 0,
    category === "ribbon" ? 1 : 0,
    category === "five" ? 1 : 0,
    category === "junk" ? 1 : 0,
    isDoublePiCard(card) ? 1 : 0,

    matchOpportunityDensity(state, month),
    immediateMatchPossible(state, decisionType, month),
    optionCode(candidate),

    clamp01(selfGwangCount / 5.0),
    clamp01(oppGwangCount / 5.0),
    clamp01(selfPiCount / 20.0),
    clamp01(oppPiCount / 20.0),

    clamp01(selfGodori / 3.0),
    clamp01(oppGodori / 3.0),
    clamp01(selfCheongdan / 3.0),
    clamp01(oppCheongdan / 3.0),
    clamp01(selfHongdan / 3.0),
    clamp01(oppHongdan / 3.0),
    clamp01(selfChodan / 3.0),
    clamp01(oppChodan / 3.0),

    selfCanStop,
    oppCanStop,

    hasShake,
    currentMultiplierNorm(state, scoreSelf),
    hasBomb,

    scoreSelf?.bak?.pi ? 1 : 0,
    scoreSelf?.bak?.gwang ? 1 : 0,
    scoreSelf?.bak?.mongBak ? 1 : 0,

    candidateMonthKnownRatio(state, actor, month)
  ];

  if (inputDim === features.length) return features;
  throw new Error(`feature vector size mismatch: expected ${inputDim}, supported=${features.length}`);
}

/* 3) Forward-pass helpers */
function activation(name, x) {
  const n = String(name || "tanh").trim().toLowerCase();
  const v = Number(x || 0);
  if (n === "sigmoid") return 1.0 / (1.0 + Math.exp(-v));
  if (n === "relu") return Math.max(0, v);
  if (n === "identity" || n === "linear") return v;
  if (n === "clamped") return Math.max(-1, Math.min(1, v));
  if (n === "gauss") return Math.exp(-(v * v));
  if (n === "sin") return Math.sin(v);
  if (n === "abs") return Math.abs(v);
  return Math.tanh(v);
}

function aggregate(name, values) {
  const agg = String(name || "sum").trim().toLowerCase();
  if (!values.length) return 0.0;
  if (agg === "sum") return values.reduce((a, b) => a + b, 0);
  if (agg === "mean") return values.reduce((a, b) => a + b, 0) / values.length;
  if (agg === "max") return Math.max(...values);
  if (agg === "min") return Math.min(...values);
  if (agg === "product") return values.reduce((a, b) => a * b, 1.0);
  if (agg === "maxabs") return values.reduce((a, b) => (Math.abs(b) > Math.abs(a) ? b : a), values[0]);
  return values.reduce((a, b) => a + b, 0);
}

function compileNeatPythonGenome(raw) {
  const inputKeys = Array.isArray(raw?.input_keys) ? raw.input_keys.map((x) => Number(x)) : [];
  const outputKeys = Array.isArray(raw?.output_keys) ? raw.output_keys.map((x) => Number(x)) : [];
  const nodesRaw = raw?.nodes && typeof raw.nodes === "object" ? raw.nodes : {};

  const nodes = new Map();
  for (const [k, v] of Object.entries(nodesRaw)) {
    const nodeId = Number(v?.node_id ?? k);
    nodes.set(nodeId, {
      node_id: nodeId,
      activation: String(v?.activation || "tanh"),
      aggregation: String(v?.aggregation || "sum"),
      bias: Number(v?.bias || 0),
      response: Number(v?.response || 1)
    });
  }

  for (const outKey of outputKeys) {
    if (!nodes.has(outKey)) {
      nodes.set(outKey, {
        node_id: outKey,
        activation: "tanh",
        aggregation: "sum",
        bias: 0,
        response: 1
      });
    }
  }

  const connections = [];
  for (const item of raw?.connections || []) {
    if (!item?.enabled) continue;
    connections.push({
      in_node: Number(item?.in_node || 0),
      out_node: Number(item?.out_node || 0),
      weight: Number(item?.weight || 0)
    });
  }

  const inputSet = new Set(inputKeys);
  const nonInputSet = new Set([...nodes.keys()].filter((k) => !inputSet.has(k)));
  const indegree = new Map();
  const adjacency = new Map();
  const incoming = new Map();

  for (const node of nonInputSet) {
    indegree.set(node, 0);
    adjacency.set(node, []);
    incoming.set(node, []);
  }

  for (const conn of connections) {
    const outNode = conn.out_node;
    if (!nonInputSet.has(outNode)) continue;
    incoming.get(outNode).push(conn);
    const inNode = conn.in_node;
    if (nonInputSet.has(inNode)) {
      indegree.set(outNode, Number(indegree.get(outNode) || 0) + 1);
      adjacency.get(inNode).push(outNode);
    }
  }

  const queue = [...nonInputSet].filter((n) => Number(indegree.get(n) || 0) === 0).sort((a, b) => a - b);
  const order = [];
  while (queue.length > 0) {
    const node = queue.shift();
    order.push(node);
    const nexts = adjacency.get(node) || [];
    for (const nxt of nexts) {
      const deg = Number(indegree.get(nxt) || 0) - 1;
      indegree.set(nxt, deg);
      if (deg === 0) {
        queue.push(nxt);
        queue.sort((a, b) => a - b);
      }
    }
  }

  return {
    kind: NEAT_MODEL_FORMAT,
    inputKeys,
    outputKeys,
    nodes,
    incoming,
    order: order.length === nonInputSet.size ? order : [...nonInputSet].sort((a, b) => a - b)
  };
}

/* 4) Model compilation/cache */
function isNeatModel(policyModel) {
  return String(policyModel?.format_version || "").trim() === NEAT_MODEL_FORMAT;
}

function getCompiledNeatModel(policyModel) {
  if (!policyModel || !isNeatModel(policyModel)) return null;
  const cached = COMPILED_NEAT_CACHE.get(policyModel);
  if (cached) return cached;
  const compiled = compileNeatPythonGenome(policyModel);
  COMPILED_NEAT_CACHE.set(policyModel, compiled);
  return compiled;
}

/* 5) Scoring/post-processing */
function forward(compiled, inputVec) {
  const values = new Map();
  for (let i = 0; i < compiled.inputKeys.length; i += 1) {
    values.set(Number(compiled.inputKeys[i]), Number(inputVec[i] || 0));
  }

  for (const nodeId of compiled.order) {
    const node = compiled.nodes.get(nodeId) || {
      activation: "tanh",
      aggregation: "sum",
      bias: 0,
      response: 1
    };
    const incoming = compiled.incoming.get(nodeId) || [];
    const terms = incoming.map((conn) => Number(values.get(conn.in_node) || 0) * Number(conn.weight || 0));
    const agg = aggregate(node.aggregation, terms);
    const pre = Number(node.bias || 0) + Number(node.response || 1) * agg;
    values.set(nodeId, activation(node.activation, pre));
  }

  const outKey = compiled.outputKeys.length > 0 ? Number(compiled.outputKeys[0]) : null;
  if (outKey == null) return 0.0;
  return Number(values.get(outKey) || 0.0);
}

function scoreToProbabilityMap(candidates, scoreMap, temperature = 1.0) {
  const temp = Math.max(0.05, Number(temperature || 1.0));
  const scores = candidates.map((c) => Number(scoreMap.get(c) || -Infinity));
  const maxScore = Math.max(...scores);
  const exps = scores.map((s) => Math.exp((s - maxScore) / temp));
  const z = exps.reduce((a, b) => a + b, 0);
  const probs = {};
  if (!(z > 0)) {
    const uniform = candidates.length > 0 ? 1 / candidates.length : 0;
    for (const c of candidates) probs[String(c)] = uniform;
    return probs;
  }
  for (let i = 0; i < candidates.length; i += 1) {
    probs[String(candidates[i])] = exps[i] / z;
  }
  return probs;
}

function pickBestByScore(candidates, scoreMap) {
  let best = null;
  let bestScore = -Infinity;
  for (const c of candidates) {
    const s = Number(scoreMap.get(c) || -Infinity);
    if (s > bestScore) {
      bestScore = s;
      best = c;
    }
  }
  return best;
}

/* 6) Public APIs */
export function getModelCandidateProbabilities(state, actor, policyModel, options = {}) {
  const compiled = getCompiledNeatModel(policyModel);
  if (!compiled) return null;

  const sp = selectPool(state, actor, options);
  const decisionType = resolveDecisionType(sp);
  if (!decisionType) return null;

  const candidates = legalCandidatesForDecision(sp, decisionType);
  if (!candidates.length) return null;

  const inputDim = Number(compiled.inputKeys.length || 0);
  const scoreMap = new Map();
  const scores = {};
  try {
    for (const candidate of candidates) {
      const x = featureVector(state, actor, decisionType, candidate, candidates.length, inputDim);
      const score = forward(compiled, x);
      scoreMap.set(candidate, score);
      scores[String(candidate)] = score;
    }
  } catch {
    return null;
  }

  const probs = scoreToProbabilityMap(candidates, scoreMap, Number(policyModel?.softmax_temp || 1.0));
  return { decisionType, candidates, probabilities: probs, scores };
}

function modelPickCandidate(state, actor, policyModel) {
  const scored = getModelCandidateProbabilities(state, actor, policyModel);
  if (!scored) return null;
  const scoreMap = new Map();
  for (const c of scored.candidates) {
    scoreMap.set(c, Number(scored.scores?.[String(c)] || -Infinity));
  }
  const best = pickBestByScore(scored.candidates, scoreMap);
  if (!best) return null;
  return { decisionType: scored.decisionType, candidate: best };
}

export function modelPolicyPlay(state, actor, policyModel) {
  if (!policyModel || !isNeatModel(policyModel)) return state;
  const picked = modelPickCandidate(state, actor, policyModel);
  if (!picked) return state;

  const sp = selectPool(state, actor);
  const decisionType = resolveDecisionType(sp);
  if (!decisionType) return state;

  const legal = legalCandidatesForDecision(sp, decisionType);
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
