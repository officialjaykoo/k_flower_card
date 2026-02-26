import fs from "node:fs";
import path from "node:path";
import {
  initSimulationGame,
  startSimulationGame,
  createSeededRng,
  calculateScore,
  scoringPiCount,
  getDeclarableShakingMonths,
  getDeclarableBombMonths,
  playTurn,
  chooseMatch,
  chooseGo,
  chooseStop,
  chooseShakingYes,
  chooseShakingNo,
  choosePresidentStop,
  choosePresidentHold,
  chooseGukjinMode,
} from "../src/engine/index.js";
import { getActionPlayerKey } from "../src/engine/runner.js";
import { aiPlay } from "../src/ai/aiPlay.js";

// Pipeline Stage: 2/3 (neat_train.py -> neat_eval_worker.mjs -> heuristic_duel_worker.mjs)
// Quick Read Map (top-down):
// 1) main()
// 2) playSingleRound(): per-game simulation loop
// 3) pickAction()/forward()/compileGenome(): NEAT inference path
// 4) featureVector(): input feature construction (47 dims)
// 5) parseArgs()/state transition helpers
// 6) teacher dataset imitation helpers

// =============================================================================
// Section 1. Opponent Tuning + CLI
// =============================================================================
const FAST_V6_OPPONENT_POLICY = "heuristic_v6";
const FAST_V6_HEURISTIC_PARAMS = Object.freeze({
  rolloutTopK: 1,
  rolloutSamples: 2,
  rolloutMaxSteps: 12,
  rolloutCardWeight: 0.7,
  rolloutGoWeight: 0.18,
});

function normalizePolicyName(policy) {
  return String(policy || "").trim().toLowerCase();
}

function buildOpponentEvalTuning(opponentPolicy) {
  const normalized = normalizePolicyName(opponentPolicy);
  const useV6FastPath = normalized === FAST_V6_OPPONENT_POLICY;
  return {
    useV6FastPath,
    disableImitationReference: useV6FastPath,
    opponentHeuristicParams: useV6FastPath ? FAST_V6_HEURISTIC_PARAMS : null,
  };
}

function parseArgs(argv) {
  const args = [...argv];
  const out = {
    genomePath: "",
    opponentGenomePath: "",
    games: 3,
    seed: "neat-python",
    maxSteps: 600,
    opponentPolicy: "heuristic_v4",
    firstTurnPolicy: "alternate",
    fixedFirstTurn: "human",
    continuousSeries: true,
    fitnessGoldScale: 10000.0,
    fitnessWinWeight: 2.5,
    fitnessLossWeight: 1.5,
    fitnessDrawWeight: 0.1,
    teacherDatasetCachePath: "",
    teacherDatasetPath: "",
    teacherKiboPath: "",
  };

  while (args.length > 0) {
    const raw = String(args.shift() || "");
    if (!raw.startsWith("--")) throw new Error(`Unknown argument: ${raw}`);
    const eq = raw.indexOf("=");
    let key = raw;
    let value = "";
    if (eq >= 0) {
      key = raw.slice(0, eq);
      value = raw.slice(eq + 1);
    } else {
      value = String(args.shift() || "");
    }

    if (key === "--genome") out.genomePath = String(value || "").trim();
    else if (key === "--opponent-genome") out.opponentGenomePath = String(value || "").trim();
    else if (key === "--games") out.games = Math.max(1, Number(value || 0));
    else if (key === "--seed") out.seed = String(value || "neat-python");
    else if (key === "--max-steps") out.maxSteps = Math.max(20, Number(value || 600));
    else if (key === "--opponent-policy") out.opponentPolicy = String(value || "heuristic_v4").trim();
    else if (key === "--first-turn-policy") out.firstTurnPolicy = String(value || "alternate").trim().toLowerCase();
    else if (key === "--fixed-first-turn") out.fixedFirstTurn = String(value || "human").trim().toLowerCase();
    else if (key === "--switch-seats") {
      // Backward compatibility: legacy seat switch now maps to first-turn policy.
      out.firstTurnPolicy = String(value || "1").trim() === "0" ? "fixed" : "alternate";
    }
    else if (key === "--continuous-series") out.continuousSeries = !(String(value || "1").trim() === "0");
    else if (key === "--fitness-gold-scale") out.fitnessGoldScale = Math.max(1.0, Number(value || 10000.0));
    else if (key === "--fitness-win-weight") out.fitnessWinWeight = Number(value || 2.5);
    else if (key === "--fitness-loss-weight") out.fitnessLossWeight = Number(value || 1.5);
    else if (key === "--fitness-draw-weight") out.fitnessDrawWeight = Number(value || 0.1);
    else if (key === "--teacher-dataset-cache") out.teacherDatasetCachePath = String(value || "").trim();
    else if (key === "--teacher-dataset-path") out.teacherDatasetPath = String(value || "").trim();
    else if (key === "--teacher-kibo-path") out.teacherKiboPath = String(value || "").trim();
    else throw new Error(`Unknown argument: ${key}`);
  }

  if (!out.genomePath) throw new Error("--genome is required");
  if (out.firstTurnPolicy !== "alternate" && out.firstTurnPolicy !== "fixed") {
    throw new Error(`invalid --first-turn-policy: ${out.firstTurnPolicy}`);
  }
  if (out.fixedFirstTurn !== "human" && out.fixedFirstTurn !== "ai") {
    throw new Error(`invalid --fixed-first-turn: ${out.fixedFirstTurn}`);
  }
  if (String(out.opponentPolicy || "").trim().toLowerCase() === "genome" && !out.opponentGenomePath) {
    throw new Error("--opponent-genome is required when --opponent-policy=genome");
  }
  return out;
}

// =============================================================================
// Section 2. Engine Action Helpers + Feature Helpers
// =============================================================================
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
    choose_junk: "junk",
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

function selectPool(state, actor) {
  if (state.phase === "playing" && state.currentTurn === actor) {
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

function applyAction(state, actor, decisionType, rawAction) {
  let action = String(rawAction || "").trim();
  if (!action) return state;
  if (decisionType === "play") return playTurn(state, action);
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

function stateProgressKey(state) {
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
    String(state.pendingGukjinChoice?.playerKey || ""),
    String(state.turnSeq || 0),
    String(state.kiboSeq || 0),
    String(hh),
    String(ah),
    String(d),
  ].join("|");
}

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
    junk: 8,
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
  if (id === "I0") return true; // Gukjin card
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

  // Public zones
  pushAll(state?.board || []);
  for (const side of ["human", "ai"]) {
    const captured = state?.players?.[side]?.captured || {};
    pushAll(captured.kwang || []);
    pushAll(captured.five || []);
    pushAll(captured.ribbon || []);
    pushAll(captured.junk || []);
  }
  // Self-known only: own hand is allowed, opponent hand is forbidden.
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
    hasBomb: Array.isArray(bombMonths) && bombMonths.length > 0 ? 1 : 0,
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
    // 1-6 phase
    phase === "playing" ? 1 : 0,
    phase === "select-match" ? 1 : 0,
    phase === "go-stop" ? 1 : 0,
    phase === "president-choice" ? 1 : 0,
    phase === "gukjin-choice" ? 1 : 0,
    phase === "shaking-confirm" ? 1 : 0,

    // 7-9 decisionType
    decisionType === "play" ? 1 : 0,
    decisionType === "match" ? 1 : 0,
    decisionType === "option" ? 1 : 0,

    // 10-14 game state
    clamp01((state.deck?.length || 0) / 30.0),
    clamp01((state.players?.[actor]?.hand?.length || 0) / 10.0),
    clamp01((state.players?.[opp]?.hand?.length || 0) / 10.0),
    clamp01((state.players?.[actor]?.goCount || 0) / 5.0),
    clamp01((state.players?.[opp]?.goCount || 0) / 5.0),

    // 15-17 score
    tanhNorm((scoreSelf?.total || 0) - (scoreOpp?.total || 0), 10.0),
    tanhNorm((scoreSelf?.total || 0), 10.0), // self_score_total_norm
    clamp01(Number(legalCount || 0) / 10.0),

    // 18-23 candidate card
    clamp01(piValue / 5.0),
    category === "kwang" ? 1 : 0,
    category === "ribbon" ? 1 : 0,
    category === "five" ? 1 : 0,
    category === "junk" ? 1 : 0,
    isDoublePiCard(card) ? 1 : 0,

    // 24-26 board/matching
    matchOpportunityDensity(state, month),
    immediateMatchPossible(state, decisionType, month),
    optionCode(candidate),

    // 27-30 gwang/pi
    clamp01(selfGwangCount / 5.0),
    clamp01(oppGwangCount / 5.0),
    clamp01(selfPiCount / 20.0),
    clamp01(oppPiCount / 20.0),

    // 31-38 combo progress
    clamp01(selfGodori / 3.0),
    clamp01(oppGodori / 3.0),
    clamp01(selfCheongdan / 3.0),
    clamp01(oppCheongdan / 3.0),
    clamp01(selfHongdan / 3.0),
    clamp01(oppHongdan / 3.0),
    clamp01(selfChodan / 3.0),
    clamp01(oppChodan / 3.0),

    // 39-40 stop thresholds
    selfCanStop,
    oppCanStop,

    // 41-43 multiplier/shake/bomb
    hasShake,
    currentMultiplierNorm(state, scoreSelf),
    hasBomb,

    // 44-46 bak risks
    scoreSelf?.bak?.pi ? 1 : 0,
    scoreSelf?.bak?.gwang ? 1 : 0,
    scoreSelf?.bak?.mongBak ? 1 : 0,

    // 47 month-known ratio (public + self hand only)
    candidateMonthKnownRatio(state, actor, month),
  ];

  if (inputDim === features.length) return features;
  throw new Error(
    `feature vector size mismatch: expected ${inputDim}, supported=${features.length}`
  );
}

// =============================================================================
// Section 3. NEAT Genome Compile + Forward Pass
// =============================================================================
function normalizeDecisionCandidate(decisionType, candidate) {
  if (decisionType === "option") return canonicalOptionAction(candidate);
  return String(candidate || "").trim();
}

function heuristicCandidateForDecision(state, actor, decisionType, candidates, heuristicPolicy) {
  if (!Array.isArray(candidates) || !candidates.length) return null;
  const nextByHeuristic = aiPlay(state, actor, {
    source: "heuristic",
    heuristicPolicy: heuristicPolicy || "heuristic_v4",
  });
  if (!nextByHeuristic || stateProgressKey(nextByHeuristic) === stateProgressKey(state)) {
    return null;
  }
  const target = stateProgressKey(nextByHeuristic);
  for (const c of candidates) {
    const simulated = applyAction(state, actor, decisionType, c);
    if (simulated && stateProgressKey(simulated) === target) {
      return normalizeDecisionCandidate(decisionType, c);
    }
  }
  return null;
}

function activation(name, x) {
  const act = String(name || "tanh").trim().toLowerCase();
  const v = Number(x || 0);
  if (act === "linear" || act === "identity") return v;
  if (act === "relu") return v > 0 ? v : 0;
  if (act === "sigmoid") {
    const z = Math.max(-30, Math.min(30, v));
    return 1.0 / (1.0 + Math.exp(-z));
  }
  if (act === "clamped") {
    if (v > 1.0) return 1.0;
    if (v < -1.0) return -1.0;
    return v;
  }
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
      response: Number(v?.response || 1),
    });
  }

  for (const outKey of outputKeys) {
    if (!nodes.has(outKey)) {
      nodes.set(outKey, {
        node_id: outKey,
        activation: "tanh",
        aggregation: "sum",
        bias: 0,
        response: 1,
      });
    }
  }

  const connections = [];
  for (const item of raw?.connections || []) {
    const enabled = !!item?.enabled;
    if (!enabled) continue;
    connections.push({
      in_node: Number(item?.in_node || 0),
      out_node: Number(item?.out_node || 0),
      weight: Number(item?.weight || 0),
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

  if (order.length !== nonInputSet.size) {
    const fallback = [...nonInputSet].sort((a, b) => a - b);
    return {
      kind: "neat_python_genome_v1",
      inputKeys,
      outputKeys,
      nodes,
      incoming,
      order: fallback,
    };
  }

  return {
    kind: "neat_python_genome_v1",
    inputKeys,
    outputKeys,
    nodes,
    incoming,
    order,
  };
}

function compileGenome(raw) {
  const fmt = String(raw?.format_version || "").trim();
  if (fmt !== "neat_python_genome_v1") {
    throw new Error(`unsupported genome format: ${fmt || "<empty>"}`);
  }
  return compileNeatPythonGenome(raw);
}

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
      response: 1,
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

function pickAction(state, actor, compiled) {
  const sp = selectPool(state, actor);
  const cards = sp.cards || null;
  const boardCardIds = sp.boardCardIds || null;
  const options = sp.options || null;
  const decisionType = cards ? "play" : boardCardIds ? "match" : options ? "option" : null;
  if (!decisionType) return null;

  const candidates = legalCandidatesForDecision(sp, decisionType);
  if (!candidates.length) return null;

  const inputDim = compiled.inputKeys.length;
  let best = candidates[0];
  let bestScore = -Infinity;
  for (const candidate of candidates) {
    const x = featureVector(state, actor, decisionType, candidate, candidates.length, inputDim);
    const score = forward(compiled, x);
    if (score > bestScore) {
      best = candidate;
      bestScore = score;
    }
  }

  return { decisionType, candidate: best, candidates };
}

// =============================================================================
// Section 4. Round Simulation + Metrics
// =============================================================================
function randomChoice(arr, rng) {
  if (!arr.length) return null;
  const idx = Math.max(0, Math.min(arr.length - 1, Math.floor(Number(rng() || 0) * arr.length)));
  return arr[idx];
}

function randomLegalAction(state, actor, rng) {
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

function resolveFirstTurnKey(opts, gameIndex) {
  if (opts.firstTurnPolicy === "fixed") return opts.fixedFirstTurn;
  return gameIndex % 2 === 0 ? "ai" : "human";
}

function startRound(seed, firstTurnKey) {
  return initSimulationGame("A", createSeededRng(`${seed}|game`), {
    kiboDetail: "lean",
    firstTurnKey,
  });
}

function continueRound(prevEndState, seed, firstTurnKey) {
  return startSimulationGame(prevEndState, createSeededRng(`${seed}|game`), {
    kiboDetail: "lean",
    keepGold: true,
    useCarryOver: true,
    firstTurnKey,
  });
}

function controlGoldDiff(state, controlActor) {
  const opp = controlActor === "human" ? "ai" : "human";
  const controlGold = Number(state?.players?.[controlActor]?.gold || 0);
  const oppGold = Number(state?.players?.[opp]?.gold || 0);
  return controlGold - oppGold;
}

function playSingleRound(
  initialState,
  compiled,
  seed,
  controlActor,
  opponentPolicy,
  maxSteps,
  opponentCompiled,
  opponentEvalTuning = null
) {
  const rng = createSeededRng(`${seed}|rng`);
  const disableImitationReference = !!opponentEvalTuning?.disableImitationReference;
  const opponentHeuristicParams =
    opponentEvalTuning?.opponentHeuristicParams &&
    typeof opponentEvalTuning.opponentHeuristicParams === "object"
      ? opponentEvalTuning.opponentHeuristicParams
      : null;
  let state = initialState;
  const imitation = {
    totals: { play: 0, match: 0, option: 0 },
    matches: { play: 0, match: 0, option: 0 },
  };

  let steps = 0;
  while (state.phase !== "resolution" && steps < maxSteps) {
    const actor = getActionPlayerKey(state);
    if (!actor) break;

    const before = stateProgressKey(state);
    let next = state;

    if (actor === controlActor) {
      const picked = pickAction(state, actor, compiled);
      if (picked) {
        if (!disableImitationReference) {
          const imitationRefPolicy =
            normalizePolicyName(opponentPolicy) === "genome"
              ? "heuristic_v4"
              : opponentPolicy;
          const refCandidate = heuristicCandidateForDecision(
            state,
            actor,
            picked.decisionType,
            picked.candidates || [],
            imitationRefPolicy
          );
          if (refCandidate) {
            const key = picked.decisionType;
            imitation.totals[key] += 1;
            const chosen = normalizeDecisionCandidate(picked.decisionType, picked.candidate);
            if (chosen === refCandidate) {
              imitation.matches[key] += 1;
            }
          }
        }
        next = applyAction(state, actor, picked.decisionType, picked.candidate);
      }
    } else if (normalizePolicyName(opponentPolicy) === "genome") {
      const picked = opponentCompiled ? pickAction(state, actor, opponentCompiled) : null;
      if (picked) {
        next = applyAction(state, actor, picked.decisionType, picked.candidate);
      }
    } else {
      next = aiPlay(state, actor, {
        source: "heuristic",
        heuristicPolicy: opponentPolicy,
        heuristicParams: opponentHeuristicParams,
      });
    }

    if (!next || stateProgressKey(next) === before) {
      next = randomLegalAction(state, actor, rng);
    }
    if (!next || stateProgressKey(next) === before) {
      break;
    }

    state = next;
    steps += 1;
  }

  return { endState: state, imitation };
}

function quantile(values, q) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.max(0, Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * q)));
  return sorted[idx];
}

function cloneDecisionCounters(src) {
  return {
    play: Number(src?.play || 0),
    match: Number(src?.match || 0),
    option: Number(src?.option || 0),
  };
}

function buildImitationMetrics(totals, matches) {
  const t = cloneDecisionCounters(totals);
  const m = cloneDecisionCounters(matches);
  const ratio = (num, den) => (den > 0 ? num / den : 0);
  const playRatio = ratio(m.play, t.play);
  const matchRatio = ratio(m.match, t.match);
  const optionRatio = ratio(m.option, t.option);
  const weights = { play: 0.5, match: 0.3, option: 0.2 };
  let weightSum = 0;
  for (const k of ["play", "match", "option"]) {
    if (Number(t[k] || 0) > 0) weightSum += Number(weights[k] || 0);
  }
  const weightedRaw =
    weights.play * playRatio +
    weights.match * matchRatio +
    weights.option * optionRatio;
  const weightedScore = weightSum > 0 ? weightedRaw / weightSum : 0;
  return {
    totals: t,
    matches: m,
    playRatio,
    matchRatio,
    optionRatio,
    weights,
    weightedScore,
  };
}

// =============================================================================
// Section 5. Teacher Dataset Imitation
// =============================================================================
function normalizeTeacherDecisionType(value) {
  const dt = String(value || "").trim().toLowerCase();
  if (dt === "play" || dt === "match" || dt === "option") return dt;
  return "";
}

function loadTeacherDatasetCache(cachePath) {
  const full = path.resolve(cachePath);
  if (!fs.existsSync(full)) {
    throw new Error(`teacher dataset cache not found: ${cachePath}`);
  }
  const raw = JSON.parse(fs.readFileSync(full, "utf8"));
  const decisionsRaw = Array.isArray(raw) ? raw : Array.isArray(raw?.decisions) ? raw.decisions : [];
  const decisions = [];

  for (const item of decisionsRaw) {
    const decisionType = normalizeTeacherDecisionType(item?.decision_type);
    if (!decisionType) continue;
    const chosenCandidate = normalizeDecisionCandidate(decisionType, item?.chosen_candidate);
    const candidateRows = Array.isArray(item?.candidates) ? item.candidates : [];
    const candidates = [];
    const seen = new Set();

    for (const row of candidateRows) {
      const candidate = normalizeDecisionCandidate(decisionType, row?.candidate);
      if (!candidate || seen.has(candidate)) continue;
      const featuresRaw = Array.isArray(row?.features) ? row.features : [];
      if (!featuresRaw.length) continue;
      const features = [];
      let ok = true;
      for (const v of featuresRaw) {
        const n = Number(v);
        if (!Number.isFinite(n)) {
          ok = false;
          break;
        }
        features.push(n);
      }
      if (!ok) continue;
      seen.add(candidate);
      candidates.push({ candidate, features });
    }

    if (!chosenCandidate || !candidates.length) continue;
    if (!candidates.some((c) => c.candidate === chosenCandidate)) continue;
    decisions.push({ decisionType, chosenCandidate, candidates });
  }

  return {
    cachePath: full,
    sourceDatasetPath: String(raw?.source_dataset_path || ""),
    sourceKiboPath: String(raw?.source_kibo_path || ""),
    actorFilter: String(raw?.actor_filter || "all"),
    decisions,
  };
}

function evaluateTeacherDatasetImitation(compiled, teacherCache) {
  const totals = { play: 0, match: 0, option: 0 };
  const matches = { play: 0, match: 0, option: 0 };
  const inputDim = Number(compiled?.inputKeys?.length || 0);
  let decisionsUsed = 0;

  for (const item of teacherCache.decisions || []) {
    const decisionType = normalizeTeacherDecisionType(item?.decisionType);
    if (!decisionType) continue;
    const chosenCandidate = normalizeDecisionCandidate(decisionType, item?.chosenCandidate);
    if (!chosenCandidate) continue;

    const candidates = [];
    for (const row of item?.candidates || []) {
      const candidate = normalizeDecisionCandidate(decisionType, row?.candidate);
      const features = Array.isArray(row?.features) ? row.features : [];
      if (!candidate || features.length !== inputDim) continue;
      candidates.push({ candidate, features });
    }
    if (!candidates.length) continue;
    if (!candidates.some((c) => c.candidate === chosenCandidate)) continue;

    let bestCandidate = candidates[0].candidate;
    let bestScore = -Infinity;
    for (const row of candidates) {
      const score = forward(compiled, row.features);
      if (score > bestScore) {
        bestScore = score;
        bestCandidate = row.candidate;
      }
    }

    totals[decisionType] += 1;
    decisionsUsed += 1;
    if (bestCandidate === chosenCandidate) {
      matches[decisionType] += 1;
    }
  }

  const metrics = buildImitationMetrics(totals, matches);
  return {
    ...metrics,
    decisionCount: decisionsUsed,
    cachePath: teacherCache.cachePath,
    sourceDatasetPath: teacherCache.sourceDatasetPath,
    sourceKiboPath: teacherCache.sourceKiboPath,
    actorFilter: teacherCache.actorFilter,
  };
}

// =============================================================================
// Section 6. Entrypoint
// =============================================================================
function main() {
  const evalStartMs = Date.now();
  const opts = parseArgs(process.argv.slice(2));
  const full = path.resolve(opts.genomePath);
  if (!fs.existsSync(full)) throw new Error(`genome not found: ${opts.genomePath}`);

  const raw = JSON.parse(fs.readFileSync(full, "utf8"));
  const compiled = compileGenome(raw);
  let opponentCompiled = null;
  if (String(opts.opponentPolicy || "").trim().toLowerCase() === "genome") {
    const oppFull = path.resolve(opts.opponentGenomePath);
    if (!fs.existsSync(oppFull)) {
      throw new Error(`opponent genome not found: ${opts.opponentGenomePath}`);
    }
    const oppRaw = JSON.parse(fs.readFileSync(oppFull, "utf8"));
    opponentCompiled = compileGenome(oppRaw);
  }
  let teacherCache = null;
  if (String(opts.teacherDatasetCachePath || "").trim()) {
    teacherCache = loadTeacherDatasetCache(opts.teacherDatasetCachePath);
  }

  const games = Math.max(1, Math.floor(opts.games));
  const opponentEvalTuning = buildOpponentEvalTuning(opts.opponentPolicy);
  const controlActor = "ai";
  const opponentActor = "human";
  let wins = 0;
  let losses = 0;
  let draws = 0;
  const goldDeltas = [];
  const bankrupt = {
    my_bankrupt_count: 0,
    my_inflicted_bankrupt_count: 0,
  };
  let go3PlusCount = 0;
  let myGotBakCount = 0;
  const simImitationTotals = { play: 0, match: 0, option: 0 };
  const simImitationMatches = { play: 0, match: 0, option: 0 };
  const firstTurnCounts = {
    human: 0,
    ai: 0,
  };
  const seriesSession = {
    roundsPlayed: 0,
    previousEndState: null,
  };

  for (let gi = 0; gi < games; gi += 1) {
    const firstTurnKey = resolveFirstTurnKey(opts, gi);
    firstTurnCounts[firstTurnKey] += 1;
    const seed = `${opts.seed}|g=${gi}|first=${firstTurnKey}|sr=${seriesSession.roundsPlayed}`;
    const roundStart = opts.continuousSeries
      ? seriesSession.previousEndState
        ? continueRound(seriesSession.previousEndState, seed, firstTurnKey)
        : startRound(seed, firstTurnKey)
      : startRound(seed, firstTurnKey);
    const beforeGoldDiff = controlGoldDiff(roundStart, controlActor);
    const gameResult = playSingleRound(
      roundStart,
      compiled,
      seed,
      controlActor,
      opts.opponentPolicy,
      Math.max(20, Math.floor(opts.maxSteps)),
      opponentCompiled,
      opponentEvalTuning
    );
    const endState = gameResult?.endState || gameResult;
    const afterGoldDiff = controlGoldDiff(endState, controlActor);
    const goldDelta = afterGoldDiff - beforeGoldDiff;
    goldDeltas.push(goldDelta);
    const controlGold = Number(endState?.players?.[controlActor]?.gold || 0);
    const opponentGold = Number(endState?.players?.[opponentActor]?.gold || 0);
    const controlBankrupt = controlGold <= 0;
    const opponentBankrupt = opponentGold <= 0;
    const controlGoCount = Number(endState?.players?.[controlActor]?.goCount || 0);
    if (controlGoCount >= 3) {
      go3PlusCount += 1;
    }
    const resultWinner = String(endState?.result?.winner || "").trim();
    if (resultWinner === opponentActor) {
      const oppScore = endState?.result?.[opponentActor] || {};
      const bak = oppScore?.bak || {};
      if (bak?.dokbak === true) {
        myGotBakCount += 1;
      }
    }
    if (opponentBankrupt) {
      bankrupt.my_inflicted_bankrupt_count += 1;
    }
    if (controlBankrupt) {
      bankrupt.my_bankrupt_count += 1;
    }
    if (opts.continuousSeries) {
      seriesSession.previousEndState = endState;
    }
    seriesSession.roundsPlayed += 1;

    const winner = endState?.result?.winner || "unknown";
    if (winner === controlActor) wins += 1;
    else if (winner === opponentActor) losses += 1;
    else draws += 1;

    const gt = gameResult?.imitation?.totals || {};
    const gm = gameResult?.imitation?.matches || {};
    for (const k of ["play", "match", "option"]) {
      simImitationTotals[k] += Number(gt[k] || 0);
      simImitationMatches[k] += Number(gm[k] || 0);
    }
  }

  const meanGoldDelta = goldDeltas.length > 0 ? goldDeltas.reduce((a, b) => a + b, 0) / goldDeltas.length : 0;
  const winRate = wins / games;
  const lossRate = losses / games;
  const go3PlusRate = go3PlusCount / games;
  const myGotBakRate = myGotBakCount / games;
  const drawRate = draws / games;
  const fitnessGoldScaleRaw = Number(opts.fitnessGoldScale);
  const fitnessWinWeightRaw = Number(opts.fitnessWinWeight);
  const fitnessLossWeightRaw = Number(opts.fitnessLossWeight);
  const fitnessDrawWeightRaw = Number(opts.fitnessDrawWeight);
  const fitnessGoldScale = Number.isFinite(fitnessGoldScaleRaw) && fitnessGoldScaleRaw > 0
    ? fitnessGoldScaleRaw
    : 10000.0;
  const fitnessWinWeight = Number.isFinite(fitnessWinWeightRaw) ? fitnessWinWeightRaw : 2.5;
  const fitnessLossWeight = Number.isFinite(fitnessLossWeightRaw) ? fitnessLossWeightRaw : 1.5;
  const fitnessDrawWeight = Number.isFinite(fitnessDrawWeightRaw) ? fitnessDrawWeightRaw : 0.1;

  let fitness =
    (meanGoldDelta / fitnessGoldScale) +
    (winRate * fitnessWinWeight) -
    (lossRate * fitnessLossWeight) +
    (drawRate * fitnessDrawWeight);
  if (winRate < 0.45 && meanGoldDelta < 0) {
    fitness -= 0.4 + (0.45 - winRate) * 2.0;
  }

  const simImitation = buildImitationMetrics(simImitationTotals, simImitationMatches);
  let imitationSource = "opponent_policy";
  let activeImitation = simImitation;
  let teacherImitation = null;
  if (teacherCache) {
    teacherImitation = evaluateTeacherDatasetImitation(compiled, teacherCache);
    if (Number(teacherImitation.decisionCount || 0) > 0) {
      imitationSource = "teacher_dataset_cache";
      activeImitation = teacherImitation;
    }
  }
  const imitationTotals = cloneDecisionCounters(activeImitation.totals);
  const imitationMatches = cloneDecisionCounters(activeImitation.matches);
  const imitationPlayRatio = Number(activeImitation.playRatio || 0);
  const imitationMatchRatio = Number(activeImitation.matchRatio || 0);
  const imitationOptionRatio = Number(activeImitation.optionRatio || 0);
  const imitationWeights = activeImitation.weights || { play: 0.5, match: 0.3, option: 0.2 };
  const imitationWeightedScore = Number(activeImitation.weightedScore || 0);

  const summary = {
    games,
    control_actor: controlActor,
    opponent_actor: opponentActor,
    opponent_policy: opts.opponentPolicy,
    opponent_eval_tuning: {
      v6_fast_path: !!opponentEvalTuning.useV6FastPath,
      imitation_reference_enabled: !opponentEvalTuning.disableImitationReference,
      opponent_heuristic_params: opponentEvalTuning.opponentHeuristicParams,
    },
    opponent_genome: String(opts.opponentGenomePath || "") || null,
    first_turn_policy: opts.firstTurnPolicy,
    fixed_first_turn: opts.firstTurnPolicy === "fixed" ? opts.fixedFirstTurn : null,
    first_turn_counts: firstTurnCounts,
    continuous_series: !!opts.continuousSeries,
    bankrupt,
    session_rounds: {
      control_actor_series: seriesSession.roundsPlayed,
    },
    wins,
    losses,
    draws,
    win_rate: winRate,
    loss_rate: lossRate,
    draw_rate: drawRate,
    go3_plus_count: go3PlusCount,
    go3_plus_rate: go3PlusRate,
    my_got_bak_count: myGotBakCount,
    my_got_bak_rate: myGotBakRate,
    mean_gold_delta: meanGoldDelta,
    p10_gold_delta: quantile(goldDeltas, 0.1),
    p50_gold_delta: quantile(goldDeltas, 0.5),
    p90_gold_delta: quantile(goldDeltas, 0.9),
    fitness_gold_scale: fitnessGoldScale,
    fitness_win_weight: fitnessWinWeight,
    fitness_loss_weight: fitnessLossWeight,
    fitness_draw_weight: fitnessDrawWeight,
    imitation_source: imitationSource,
    teacher_dataset_cache: teacherImitation
      ? String(teacherImitation.cachePath || "")
      : (String(opts.teacherDatasetCachePath || "").trim() ? path.resolve(opts.teacherDatasetCachePath) : null),
    teacher_dataset_path: teacherImitation
      ? (String(teacherImitation.sourceDatasetPath || "").trim() || null)
      : (String(opts.teacherDatasetPath || "").trim() || null),
    teacher_kibo_path: teacherImitation
      ? (String(teacherImitation.sourceKiboPath || "").trim() || null)
      : (String(opts.teacherKiboPath || "").trim() || null),
    teacher_dataset_actor: teacherImitation ? String(teacherImitation.actorFilter || "all") : null,
    teacher_dataset_decisions: teacherImitation ? Number(teacherImitation.decisionCount || 0) : 0,
    teacher_imitation_weighted_score: teacherImitation
      ? Number(teacherImitation.weightedScore || 0)
      : null,
    sim_imitation_weighted_score: Number(simImitation.weightedScore || 0),
    imitation_play_total: imitationTotals.play,
    imitation_play_matches: imitationMatches.play,
    imitation_play_ratio: imitationPlayRatio,
    imitation_match_total: imitationTotals.match,
    imitation_match_matches: imitationMatches.match,
    imitation_match_ratio: imitationMatchRatio,
    imitation_option_total: imitationTotals.option,
    imitation_option_matches: imitationMatches.option,
    imitation_option_ratio: imitationOptionRatio,
    imitation_weight_play: imitationWeights.play,
    imitation_weight_match: imitationWeights.match,
    imitation_weight_option: imitationWeights.option,
    imitation_weighted_score: imitationWeightedScore,
    eval_time_ms: Math.max(0, Date.now() - evalStartMs),
    fitness,
  };

  process.stdout.write(`${JSON.stringify(summary)}\n`);
}

try {
  main();
} catch (err) {
  const msg = err && err.stack ? err.stack : String(err);
  process.stderr.write(`${msg}\n`);
  process.exit(1);
}
