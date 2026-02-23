import fs from "node:fs";
import path from "node:path";
import {
  initSimulationGame,
  startSimulationGame,
  createSeededRng,
  calculateScore,
  scoringPiCount,
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

function parseArgs(argv) {
  const args = [...argv];
  const out = {
    genomePath: "",
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
    else throw new Error(`Unknown argument: ${key}`);
  }

  if (!out.genomePath) throw new Error("--genome is required");
  if (out.firstTurnPolicy !== "alternate" && out.firstTurnPolicy !== "fixed") {
    throw new Error(`invalid --first-turn-policy: ${out.firstTurnPolicy}`);
  }
  if (out.fixedFirstTurn !== "human" && out.fixedFirstTurn !== "ai") {
    throw new Error(`invalid --fixed-first-turn: ${out.fixedFirstTurn}`);
  }
  return out;
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

function featureVector(state, actor, decisionType, candidate, legalCount, inputDim) {
  const opp = actor === "human" ? "ai" : "human";
  const scoreSelf = calculateScore(state.players[actor], state.players[opp], state.ruleKey);
  const scoreOpp = calculateScore(state.players[opp], state.players[actor], state.ruleKey);

  const phase = String(state.phase || "");
  const card = candidateCard(state, actor, decisionType, candidate);
  const month = Number(card?.month || 0);
  const piValue = Number(card?.piValue || 0);
  const category = String(card?.category || "");
  const selfGwangCount = Number(state?.players?.[actor]?.captured?.kwang?.length || 0);
  const oppGwangCount = Number(state?.players?.[opp]?.captured?.kwang?.length || 0);
  const selfPiCount = Number(scoringPiCount(state.players[actor]) || 0);
  const oppPiCount = Number(scoringPiCount(state.players[opp]) || 0);
  const bakTotal =
    (scoreSelf?.bak?.pi ? 1 : 0) +
    (scoreSelf?.bak?.gwang ? 1 : 0) +
    (scoreSelf?.bak?.mongBak ? 1 : 0);

  const base = [
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
    clamp01(bakTotal / 3.0),
    clamp01(Number(legalCount || 0) / 10.0),
    clamp01(month / 12.0),
    clamp01(piValue / 5.0),
    optionCode(candidate),
  ];

  const extended = base.concat([
    category === "kwang" ? 1 : 0,
    category === "junk" && piValue >= 2 ? 1 : 0,
    clamp01(selfGwangCount / 5.0),
    clamp01(oppGwangCount / 5.0),
    clamp01(selfPiCount / 20.0),
    clamp01(oppPiCount / 20.0),
  ]);

  if (inputDim === base.length) return base;
  if (inputDim === extended.length) return extended;
  throw new Error(
    `feature vector size mismatch: expected ${inputDim}, supported=${base.length}|${extended.length}`
  );
}

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

function playSingleRound(initialState, compiled, seed, controlActor, opponentPolicy, maxSteps) {
  const rng = createSeededRng(`${seed}|rng`);
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
        const refCandidate = heuristicCandidateForDecision(
          state,
          actor,
          picked.decisionType,
          picked.candidates || [],
          opponentPolicy
        );
        if (refCandidate) {
          const key = picked.decisionType;
          imitation.totals[key] += 1;
          const chosen = normalizeDecisionCandidate(picked.decisionType, picked.candidate);
          if (chosen === refCandidate) {
            imitation.matches[key] += 1;
          }
        }
        next = applyAction(state, actor, picked.decisionType, picked.candidate);
      }
    } else {
      next = aiPlay(state, actor, {
        source: "heuristic",
        heuristicPolicy: opponentPolicy,
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

function main() {
  const evalStartMs = Date.now();
  const opts = parseArgs(process.argv.slice(2));
  const full = path.resolve(opts.genomePath);
  if (!fs.existsSync(full)) throw new Error(`genome not found: ${opts.genomePath}`);

  const raw = JSON.parse(fs.readFileSync(full, "utf8"));
  const compiled = compileGenome(raw);

  const games = Math.max(1, Math.floor(opts.games));
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
  const imitationTotals = { play: 0, match: 0, option: 0 };
  const imitationMatches = { play: 0, match: 0, option: 0 };
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
      Math.max(20, Math.floor(opts.maxSteps))
    );
    const endState = gameResult?.endState || gameResult;
    const afterGoldDiff = controlGoldDiff(endState, controlActor);
    const goldDelta = afterGoldDiff - beforeGoldDiff;
    goldDeltas.push(goldDelta);
    const controlGold = Number(endState?.players?.[controlActor]?.gold || 0);
    const opponentGold = Number(endState?.players?.[opponentActor]?.gold || 0);
    const controlBankrupt = controlGold <= 0;
    const opponentBankrupt = opponentGold <= 0;
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
      imitationTotals[k] += Number(gt[k] || 0);
      imitationMatches[k] += Number(gm[k] || 0);
    }
  }

  const meanGoldDelta = goldDeltas.length > 0 ? goldDeltas.reduce((a, b) => a + b, 0) / goldDeltas.length : 0;
  const winRate = wins / games;
  const lossRate = losses / games;
  const drawRate = draws / games;

  const fitness =
    (meanGoldDelta / Number(opts.fitnessGoldScale || 10000.0)) +
    (winRate * Number(opts.fitnessWinWeight || 0)) -
    (lossRate * Number(opts.fitnessLossWeight || 0)) +
    (drawRate * Number(opts.fitnessDrawWeight || 0));

  const ratio = (num, den) => (den > 0 ? num / den : 0);
  const imitationPlayRatio = ratio(imitationMatches.play, imitationTotals.play);
  const imitationMatchRatio = ratio(imitationMatches.match, imitationTotals.match);
  const imitationOptionRatio = ratio(imitationMatches.option, imitationTotals.option);
  const imitationWeights = { play: 0.5, match: 0.3, option: 0.2 };
  let imitationWeightSum = 0;
  let imitationWeightedRaw = 0;
  for (const k of ["play", "match", "option"]) {
    if (Number(imitationTotals[k] || 0) <= 0) continue;
    imitationWeightSum += imitationWeights[k];
  }
  if (imitationWeightSum > 0) {
    imitationWeightedRaw =
      imitationWeights.play * imitationPlayRatio +
      imitationWeights.match * imitationMatchRatio +
      imitationWeights.option * imitationOptionRatio;
  }
  const imitationWeightedScore = imitationWeightSum > 0 ? imitationWeightedRaw / imitationWeightSum : 0;

  const summary = {
    games,
    control_actor: controlActor,
    opponent_actor: opponentActor,
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
    mean_gold_delta: meanGoldDelta,
    p10_gold_delta: quantile(goldDeltas, 0.1),
    p50_gold_delta: quantile(goldDeltas, 0.5),
    p90_gold_delta: quantile(goldDeltas, 0.9),
    fitness_gold_scale: Number(opts.fitnessGoldScale || 10000.0),
    fitness_win_weight: Number(opts.fitnessWinWeight || 0),
    fitness_loss_weight: Number(opts.fitnessLossWeight || 0),
    fitness_draw_weight: Number(opts.fitnessDrawWeight || 0),
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
