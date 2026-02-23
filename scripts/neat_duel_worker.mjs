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

function parseArgs(argv) {
  const args = [...argv];
  const out = {
    genomeAPath: "",
    genomeBPath: "",
    games: 1000,
    seed: "neat-duel",
    maxSteps: 600,
    firstTurnPolicy: "alternate",
    fixedFirstTurn: "human",
    continuousSeries: true,
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

    if (key === "--genome-a") out.genomeAPath = String(value || "").trim();
    else if (key === "--genome-b") out.genomeBPath = String(value || "").trim();
    else if (key === "--games") out.games = Math.max(1, Number(value || 0));
    else if (key === "--seed") out.seed = String(value || "neat-duel");
    else if (key === "--max-steps") out.maxSteps = Math.max(20, Number(value || 600));
    else if (key === "--first-turn-policy") out.firstTurnPolicy = String(value || "alternate").trim().toLowerCase();
    else if (key === "--fixed-first-turn") out.fixedFirstTurn = String(value || "human").trim().toLowerCase();
    else if (key === "--switch-seats") {
      // Backward compatibility: legacy seat switch now maps to first-turn policy.
      out.firstTurnPolicy = String(value || "1").trim() === "0" ? "fixed" : "alternate";
    }
    else if (key === "--continuous-series") out.continuousSeries = !(String(value || "1").trim() === "0");
    else throw new Error(`Unknown argument: ${key}`);
  }

  if (!out.genomeAPath) throw new Error("--genome-a is required");
  if (!out.genomeBPath) throw new Error("--genome-b is required");
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

  return { decisionType, candidate: best };
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

function goldDiffForA(state, actorForA) {
  const actorForB = actorForA === "human" ? "ai" : "human";
  const goldA = Number(state?.players?.[actorForA]?.gold || 0);
  const goldB = Number(state?.players?.[actorForB]?.gold || 0);
  return goldA - goldB;
}

function playSingleDuelRound(initialState, compiledA, compiledB, seed, actorForA, maxSteps) {
  const actorForB = actorForA === "human" ? "ai" : "human";
  const rng = createSeededRng(`${seed}|rng`);
  let state = initialState;

  let steps = 0;
  while (state.phase !== "resolution" && steps < maxSteps) {
    const actor = getActionPlayerKey(state);
    if (!actor) break;

    const before = stateProgressKey(state);
    let next = state;

    const controller = actor === actorForA ? compiledA : actor === actorForB ? compiledB : null;
    if (controller) {
      const picked = pickAction(state, actor, controller);
      if (picked) next = applyAction(state, actor, picked.decisionType, picked.candidate);
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

  return state;
}

function quantile(values, q) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.max(0, Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * q)));
  return sorted[idx];
}

function main() {
  const opts = parseArgs(process.argv.slice(2));
  const fullA = path.resolve(opts.genomeAPath);
  const fullB = path.resolve(opts.genomeBPath);
  if (!fs.existsSync(fullA)) throw new Error(`genome A not found: ${opts.genomeAPath}`);
  if (!fs.existsSync(fullB)) throw new Error(`genome B not found: ${opts.genomeBPath}`);

  const rawA = JSON.parse(fs.readFileSync(fullA, "utf8"));
  const rawB = JSON.parse(fs.readFileSync(fullB, "utf8"));
  const compiledA = compileGenome(rawA);
  const compiledB = compileGenome(rawB);

  const games = Math.max(1, Math.floor(opts.games));
  const actorForA = "ai";
  const actorForB = "human";
  let winsA = 0;
  let winsB = 0;
  let draws = 0;
  const goldDeltasA = [];
  const bankrupt = {
    my_bankrupt_count: 0,
    my_inflicted_bankrupt_count: 0,
  };
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
    const beforeGoldDiffA = goldDiffForA(roundStart, actorForA);
    const endState = playSingleDuelRound(
      roundStart,
      compiledA,
      compiledB,
      seed,
      actorForA,
      Math.max(20, Math.floor(opts.maxSteps))
    );
    const afterGoldDiffA = goldDiffForA(endState, actorForA);
    goldDeltasA.push(afterGoldDiffA - beforeGoldDiffA);
    const goldA = Number(endState?.players?.[actorForA]?.gold || 0);
    const goldB = Number(endState?.players?.[actorForB]?.gold || 0);
    const aBankrupt = goldA <= 0;
    const bBankrupt = goldB <= 0;
    if (bBankrupt) {
      bankrupt.my_inflicted_bankrupt_count += 1;
    }
    if (aBankrupt) {
      bankrupt.my_bankrupt_count += 1;
    }
    if (opts.continuousSeries) {
      seriesSession.previousEndState = endState;
    }
    seriesSession.roundsPlayed += 1;

    const winner = endState?.result?.winner || "unknown";
    if (winner === actorForA) winsA += 1;
    else if (winner === actorForB) winsB += 1;
    else draws += 1;
  }

  const meanGoldDeltaA = goldDeltasA.length > 0 ? goldDeltasA.reduce((a, b) => a + b, 0) / goldDeltasA.length : 0;
  const winRateA = winsA / games;
  const winRateB = winsB / games;
  const drawRate = draws / games;

  const summary = {
    mode: "neat_duel",
    games,
    actor_a: actorForA,
    actor_b: actorForB,
    first_turn_policy: opts.firstTurnPolicy,
    fixed_first_turn: opts.firstTurnPolicy === "fixed" ? opts.fixedFirstTurn : null,
    first_turn_counts: firstTurnCounts,
    continuous_series: !!opts.continuousSeries,
    bankrupt,
    session_rounds: {
      actor_a_series: seriesSession.roundsPlayed,
    },
    genome_a: fullA,
    genome_b: fullB,
    wins_a: winsA,
    wins_b: winsB,
    draws,
    win_rate_a: winRateA,
    win_rate_b: winRateB,
    draw_rate: drawRate,
    mean_gold_delta_a: meanGoldDeltaA,
    p10_gold_delta_a: quantile(goldDeltasA, 0.1),
    p50_gold_delta_a: quantile(goldDeltasA, 0.5),
    p90_gold_delta_a: quantile(goldDeltasA, 0.9),
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
