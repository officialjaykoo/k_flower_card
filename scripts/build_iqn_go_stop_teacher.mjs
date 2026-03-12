import {
  chooseGo,
  chooseGukjinMode,
  chooseMatch,
  choosePresidentHold,
  choosePresidentStop,
  chooseShakingNo,
  chooseShakingYes,
  chooseStop,
  createSeededRng,
  declareBomb,
  declareShaking,
  getDeclarableBombMonths,
  getDeclarableShakingMonths,
  playTurn,
} from "../src/engine/index.js";
import { getActionPlayerKey } from "../src/engine/runner.js";
import { aiPlay } from "../src/ai/aiPlay.js";
import { hybridPolicyPlayDetailed } from "../src/ai/hybridPolicyEngine.js";
import { resolveBotPolicy } from "../src/ai/policies.js";
import { createReadStream, createWriteStream, existsSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import readline from "node:readline";

function parseArgs(argv) {
  const args = [...argv];
  const out = {
    datasetIn: "",
    out: "",
    teacher: "hybrid_play(phase1_seed208,H-CL)",
    samplesPerAction: 8,
    maxSteps: 600,
    seed: "iqn-go-stop-teacher",
    limit: 0,
    progressEvery: 50,
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

    if (key === "--dataset-in") out.datasetIn = String(value || "").trim();
    else if (key === "--out") out.out = String(value || "").trim();
    else if (key === "--teacher") out.teacher = String(value || "").trim();
    else if (key === "--samples-per-action") out.samplesPerAction = Math.max(1, Math.floor(Number(value || 8)));
    else if (key === "--max-steps") out.maxSteps = Math.max(20, Math.floor(Number(value || 600)));
    else if (key === "--seed") out.seed = String(value || "iqn-go-stop-teacher").trim();
    else if (key === "--limit") out.limit = Math.max(0, Math.floor(Number(value || 0)));
    else if (key === "--progress-every") out.progressEvery = Math.max(1, Math.floor(Number(value || 50)));
    else throw new Error(`Unknown argument: ${key}`);
  }

  if (!out.datasetIn) throw new Error("--dataset-in is required");
  if (!out.out) throw new Error("--out is required");
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

function cloneState(state) {
  if (typeof structuredClone === "function") return structuredClone(state);
  return JSON.parse(JSON.stringify(state));
}

function otherPlayerKey(playerKey) {
  return playerKey === "human" ? "ai" : "human";
}

function parseHybridPlayGoSpec(token) {
  const m = String(token || "")
    .trim()
    .match(/^hybrid_play_go\(\s*([^,]+)\s*,\s*([^,]+?)(?:\s*,\s*([^)]+)\s*)?\)$/i);
  if (!m) return null;
  return {
    modelToken: String(m[1] || "").trim(),
    goStopToken: String(m[2] || "").trim(),
    heuristicToken: String(m[3] || "").trim(),
  };
}

function parseHybridOptionSpec(token) {
  const m = String(token || "")
    .trim()
    .match(/^hybrid_option\(\s*([^,]+)\s*,\s*([^)]+)\s*\)$/i);
  if (!m) return null;
  return {
    playMatchModelToken: String(m[1] || "").trim(),
    optionModelToken: String(m[2] || "").trim(),
  };
}

function parseHybridPlaySpec(token) {
  const m = String(token || "")
    .trim()
    .match(/^hybrid_play\(\s*([^,]+)\s*,\s*([^)]+)\s*\)$/i);
  if (!m) return null;
  return {
    modelToken: String(m[1] || "").trim(),
    heuristicToken: String(m[2] || "").trim(),
  };
}

function parsePhaseModelToken(rawToken) {
  const m = String(rawToken || "").trim().match(/^(?:(pareto52)_)?phase([0-3])_seed(\d+)$/i);
  if (!m) return null;
  const profile = m[1] ? "pareto52" : "classic";
  const phase = Number(m[2]);
  const seed = Number(m[3]);
  const outputPrefix = profile === "pareto52" ? "neat_pareto52" : "neat";
  const tokenKey = profile === "pareto52" ? `pareto52_phase${phase}_seed${seed}` : `phase${phase}_seed${seed}`;
  return { profile, phase, seed, outputPrefix, tokenKey };
}

function resolvePhaseModelSpec(token, sideLabel) {
  const parsedToken = parsePhaseModelToken(token);
  if (!parsedToken) {
    throw new Error(`invalid ${sideLabel} spec: ${token}`);
  }
  const { phase, seed, outputPrefix, tokenKey } = parsedToken;
  const modelPath = resolve(`logs/NEAT/${outputPrefix}_phase${phase}_seed${seed}/models/winner_genome.json`);
  if (!existsSync(modelPath)) {
    throw new Error(`model not found for ${token}: ${modelPath}`);
  }

  let model = null;
  try {
    const raw = String(readFileSync(modelPath, "utf8") || "").replace(/^\uFEFF/, "");
    model = JSON.parse(raw);
  } catch (err) {
    throw new Error(`failed to parse model JSON (${token}): ${modelPath} (${String(err)})`);
  }
  if (String(model?.format_version || "").trim() !== "neat_python_genome_v1") {
    throw new Error(`invalid model format for ${token}: expected neat_python_genome_v1`);
  }

  return {
    input: token,
    kind: "model",
    key: tokenKey,
    label: tokenKey,
    model,
    modelPath,
    phase,
    seed,
  };
}

function resolvePlayerSpec(rawSpec, sideLabel) {
  const token = String(rawSpec || "").trim();
  if (!token) throw new Error(`empty player spec: ${sideLabel}`);

  const hybridOption = parseHybridOptionSpec(token);
  if (hybridOption) {
    const playMatchModelSpec = resolvePhaseModelSpec(hybridOption.playMatchModelToken, `${sideLabel}:play_match_model`);
    const optionModelSpec = resolvePhaseModelSpec(hybridOption.optionModelToken, `${sideLabel}:option_model`);
    return {
      input: token,
      kind: "hybrid_option_model",
      key: `hybrid_option(${playMatchModelSpec.key},${optionModelSpec.key})`,
      label: `hybrid_option(${playMatchModelSpec.label},${optionModelSpec.label})`,
      playMatchModel: playMatchModelSpec.model,
      optionModel: optionModelSpec.model,
    };
  }

  const hybridGo = parseHybridPlayGoSpec(token);
  if (hybridGo) {
    const modelSpec = resolvePhaseModelSpec(hybridGo.modelToken, `${sideLabel}:model`);
    const goStopPolicy = resolveBotPolicy(hybridGo.goStopToken);
    if (!goStopPolicy) throw new Error(`invalid ${sideLabel} hybrid go-stop policy: ${hybridGo.goStopToken}`);
    const heuristicToken = String(hybridGo.heuristicToken || "").trim();
    const heuristicPolicy = heuristicToken ? resolveBotPolicy(heuristicToken) : "";
    if (heuristicToken && !heuristicPolicy) {
      throw new Error(`invalid ${sideLabel} hybrid heuristic policy: ${hybridGo.heuristicToken}`);
    }
    return {
      input: token,
      kind: "hybrid_play_model",
      key: token,
      label: token,
      model: modelSpec.model,
      heuristicPolicy,
      goStopPolicy,
      goStopOnly: !heuristicPolicy,
    };
  }

  const hybrid = parseHybridPlaySpec(token);
  if (hybrid) {
    const modelSpec = resolvePhaseModelSpec(hybrid.modelToken, `${sideLabel}:model`);
    const heuristicPolicy = resolveBotPolicy(hybrid.heuristicToken);
    if (!heuristicPolicy) {
      throw new Error(`invalid ${sideLabel} hybrid heuristic policy: ${hybrid.heuristicToken}`);
    }
    return {
      input: token,
      kind: "hybrid_play_model",
      key: token,
      label: token,
      model: modelSpec.model,
      heuristicPolicy,
    };
  }

  const resolvedPolicy = resolveBotPolicy(token);
  if (resolvedPolicy) {
    return {
      input: token,
      kind: "heuristic",
      key: resolvedPolicy,
      label: resolvedPolicy,
      model: null,
    };
  }

  return resolvePhaseModelSpec(token, sideLabel);
}

const PLAY_SPECIAL_SHAKE_PREFIX = "shake_start:";
const PLAY_SPECIAL_BOMB_PREFIX = "bomb:";

function parsePlaySpecialCandidate(candidate) {
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

function applyPlayCandidate(state, actor, candidate) {
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

function selectPool(state, actor) {
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

function legalCandidatesForDecision(sp, decisionType) {
  if (decisionType === "play") {
    return [
      ...(sp.cards || []).map((x) => String(x)).filter(Boolean),
      ...(sp.specialActions || []).map((x) => String(x)).filter(Boolean),
    ];
  }
  if (decisionType === "match") {
    return (sp.boardCardIds || []).map((x) => String(x)).filter(Boolean);
  }
  if (decisionType === "option") {
    return normalizeOptionCandidates(sp.options || []);
  }
  return [];
}

function applyAction(state, actor, decisionType, rawAction) {
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

function buildAiPlayOptions(playerSpec) {
  if (playerSpec?.kind === "model" && playerSpec?.model) {
    return { source: "model", model: playerSpec.model };
  }
  return { source: "heuristic", heuristicPolicy: String(playerSpec?.key || "") };
}

function resolvePlayerAction(state, actor, playerSpec) {
  if (playerSpec?.kind === "hybrid_option_model") {
    const sp = selectPool(state, actor);
    const cards = sp.cards || null;
    const boardCardIds = sp.boardCardIds || null;
    const options = sp.options || null;
    const decisionType = cards ? "play" : boardCardIds ? "match" : options ? "option" : null;

    if (decisionType === "play" || decisionType === "match") {
      let next = aiPlay(state, actor, {
        source: "model",
        model: playerSpec.playMatchModel,
      });
      let actionSource = "hybrid_option_play_match_model";
      if (!next || stateProgressKey(next) === stateProgressKey(state)) {
        next = aiPlay(state, actor, {
          source: "model",
          model: playerSpec.optionModel,
        });
        actionSource = "hybrid_option_play_match_fallback_option_model";
      }
      return { next: next || state, actionSource };
    }

    return {
      next: aiPlay(state, actor, {
        source: "model",
        model: playerSpec.optionModel,
      }),
      actionSource: "hybrid_option_option_model",
    };
  }

  if (playerSpec?.kind === "hybrid_play_model") {
    const traced = hybridPolicyPlayDetailed(state, actor, {
      model: playerSpec.model,
      heuristicPolicy: String(playerSpec.heuristicPolicy || ""),
      goStopPolicy: String(playerSpec.goStopPolicy || ""),
      goStopOnly: !!playerSpec.goStopOnly,
    });
    return {
      next: traced?.next || state,
      actionSource: String(traced?.actionSource || "hybrid_play_model"),
    };
  }

  if (playerSpec?.kind === "model" && playerSpec?.model) {
    return {
      next: aiPlay(state, actor, buildAiPlayOptions(playerSpec)),
      actionSource: "model",
    };
  }

  return {
    next: aiPlay(state, actor, buildAiPlayOptions(playerSpec)),
    actionSource: "heuristic",
  };
}

function rolloutToResolution(initialState, teacherSpec, seed, maxSteps) {
  const playerByActor = {
    human: teacherSpec,
    ai: teacherSpec,
  };
  const rng = createSeededRng(`${seed}|rollout`);
  let state = cloneState(initialState);
  let steps = 0;

  while (state.phase !== "resolution" && steps < maxSteps) {
    const actor = getActionPlayerKey(state);
    if (!actor) break;
    const before = stateProgressKey(state);
    const action = resolvePlayerAction(state, actor, playerByActor[actor]);
    let next = action?.next || state;
    if (!next || stateProgressKey(next) === before) {
      next = randomLegalAction(state, actor, rng);
    }
    if (!next || stateProgressKey(next) === before) {
      throw new Error(
        `rollout action resolution failed: seed=${seed}, step=${steps}, actor=${actor}, phase=${String(state?.phase || "")}`
      );
    }
    state = next;
    steps += 1;
  }

  return state;
}

function collectPublicShakingRevealIds(state, targetPlayerKey) {
  const ids = new Set();
  if (!state || !targetPlayerKey) return ids;

  const liveReveal = state.shakingReveal;
  if (liveReveal?.playerKey === targetPlayerKey) {
    for (const c of liveReveal.cards || []) {
      if (typeof c?.id === "string" && c.id) ids.add(c.id);
    }
  }

  for (const entry of state.kibo || []) {
    if (entry?.type !== "shaking_declare") continue;
    if (entry?.playerKey !== targetPlayerKey) continue;
    for (const c of entry?.revealCards || []) {
      if (typeof c?.id === "string" && c.id) ids.add(c.id);
    }
  }
  return ids;
}

function getPublicKnownOpponentHandCards(state, observerKey) {
  const opp = otherPlayerKey(observerKey);
  const revealIds = collectPublicShakingRevealIds(state, opp);
  if (revealIds.size <= 0) return [];
  const oppHand = state?.players?.[opp]?.hand || [];
  return oppHand.filter((c) => typeof c?.id === "string" && revealIds.has(c.id));
}

function shuffleInPlace(arr, rng) {
  for (let i = arr.length - 1; i > 0; i -= 1) {
    const j = Math.max(0, Math.min(i, Math.floor(Number(rng() || 0) * (i + 1))));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function determinizeHiddenState(fullState, observerKey, rng) {
  const state = cloneState(fullState);
  const opp = otherPlayerKey(observerKey);
  const knownOppCards = getPublicKnownOpponentHandCards(state, observerKey).map((c) => ({ ...c }));
  const knownIdSet = new Set(knownOppCards.map((c) => String(c?.id || "")));
  const oppHand = Array.isArray(state?.players?.[opp]?.hand) ? state.players[opp].hand : [];
  const hiddenOppCards = oppHand.filter((c) => !knownIdSet.has(String(c?.id || "")));
  const hiddenOppCount = hiddenOppCards.length;
  const hiddenPool = hiddenOppCards.concat(Array.isArray(state?.deck) ? state.deck : []);
  shuffleInPlace(hiddenPool, rng);

  if (state?.players?.[opp]) {
    state.players[opp].hand = knownOppCards.concat(hiddenPool.slice(0, hiddenOppCount));
  }
  state.deck = hiddenPool.slice(hiddenOppCount);
  return state;
}

function goldDiffByActor(state, actor) {
  const opp = otherPlayerKey(actor);
  const selfGold = Number(state?.players?.[actor]?.gold || 0);
  const oppGold = Number(state?.players?.[opp]?.gold || 0);
  return selfGold - oppGold;
}

function clipReturn(value) {
  const v = Number(value || 0);
  if (v <= -12000) return -12000;
  if (v >= 12000) return 12000;
  return v;
}

function mean(values) {
  if (!values.length) return 0;
  let total = 0;
  for (const value of values) total += Number(value || 0);
  return total / values.length;
}

function quantile(values, q) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.max(0, Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * q)));
  return sorted[idx];
}

function cvar(values, q) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const count = Math.max(1, Math.ceil(sorted.length * q));
  return mean(sorted.slice(0, count));
}

const ACTIVE_GO_STOP_BASE_FEATURES = 16;

function resolveTeacherBaseFeatures(row) {
  const feature16 = Array.isArray(row?.features16)
    ? row.features16
    : Array.isArray(row?.features)
      ? row.features
      : null;
  if (!Array.isArray(feature16) || feature16.length < ACTIVE_GO_STOP_BASE_FEATURES) {
    return null;
  }
  return feature16.slice(0, ACTIVE_GO_STOP_BASE_FEATURES);
}

async function loadGoStopDecisionGroups(datasetPath, limit = 0) {
  const groups = new Map();
  const stream = createReadStream(datasetPath, { encoding: "utf8" });
  const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });
  let missingStateRows = 0;

  for await (const line of rl) {
    const text = String(line || "").trim();
    if (!text) continue;
    const row = JSON.parse(text);
    if (String(row?.decision_type || "") !== "option") continue;
    const candidate = canonicalOptionAction(row?.candidate);
    if (candidate !== "go" && candidate !== "stop") continue;
    if (!row?.decision_id) continue;
    if (!row?.state_before_full) {
      missingStateRows += 1;
      continue;
    }

    let group = groups.get(row.decision_id);
    if (!group) {
      group = {
        decision_id: String(row.decision_id),
        game_index: Number(row.game_index || 0),
        seed: String(row.seed || ""),
        first_turn: String(row.first_turn || ""),
        step: Number(row.step || 0),
        actor: String(row.actor || ""),
        actor_policy: String(row.actor_policy || ""),
        chosen_candidate: canonicalOptionAction(row.chosen_candidate),
        public_snapshot: row.public_snapshot || null,
        state_before_full: row.state_before_full,
        features16_by_action: {},
      };
      groups.set(row.decision_id, group);
    }
    const baseFeatures = resolveTeacherBaseFeatures(row);
    if (!Array.isArray(baseFeatures)) continue;
    group.features16_by_action[candidate] = baseFeatures;

    if (limit > 0 && groups.size >= limit && group.features16_by_action.go && group.features16_by_action.stop) {
      // continue reading current stream line-by-line to keep parser simple; stop after exact group cap.
      // The output is deterministic because we only keep the first `limit` decision ids encountered.
    }
  }

  const decisions = [...groups.values()].filter(
    (group) => Array.isArray(group.features16_by_action.go) && Array.isArray(group.features16_by_action.stop)
  );
  return {
    decisions: limit > 0 ? decisions.slice(0, limit) : decisions,
    missingStateRows,
  };
}

function evaluateActionReturn(baseState, actor, action, teacherSpec, rolloutSeed, maxSteps) {
  const beforeKey = stateProgressKey(baseState);
  const beforeDiff = goldDiffByActor(baseState, actor);
  const forced = applyAction(cloneState(baseState), actor, "option", action);
  if (!forced || stateProgressKey(forced) === beforeKey) {
    throw new Error(`forced action did not advance state: actor=${actor}, action=${action}`);
  }
  const endState = rolloutToResolution(forced, teacherSpec, rolloutSeed, maxSteps);
  const afterDiff = goldDiffByActor(endState, actor);
  return clipReturn(afterDiff - beforeDiff) / 1000.0;
}

function buildOutputRow(decision, teacherSpecToken, samplesPerAction, returnsByAction) {
  const goReturns = returnsByAction.go;
  const stopReturns = returnsByAction.stop;
  const goMean = mean(goReturns);
  const stopMean = mean(stopReturns);
  const goCvar10 = cvar(goReturns, 0.1);
  const stopCvar10 = cvar(stopReturns, 0.1);
  const goScore = 0.7 * goMean + 0.3 * goCvar10;
  const stopScore = 0.7 * stopMean + 0.3 * stopCvar10;

  return {
    decision_id: decision.decision_id,
    game_index: decision.game_index,
    seed: decision.seed,
    first_turn: decision.first_turn,
    step: decision.step,
    actor: decision.actor,
    actor_policy: decision.actor_policy,
    chosen_candidate: decision.chosen_candidate,
    teacher_policy: teacherSpecToken,
    samples_per_action: samplesPerAction,
    public_snapshot: decision.public_snapshot,
    features16_by_action: decision.features16_by_action,
    returns: {
      go: goReturns,
      stop: stopReturns,
    },
    stats: {
      go_mean: goMean,
      stop_mean: stopMean,
      go_p10: quantile(goReturns, 0.1),
      stop_p10: quantile(stopReturns, 0.1),
      go_cvar10: goCvar10,
      stop_cvar10: stopCvar10,
      go_score: goScore,
      stop_score: stopScore,
    },
    teacher_choice: goScore > stopScore ? "go" : "stop",
    teacher_margin: goScore - stopScore,
  };
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const teacherSpec = resolvePlayerSpec(opts.teacher, "teacher");
  const { decisions, missingStateRows } = await loadGoStopDecisionGroups(resolve(opts.datasetIn), opts.limit);
  if (decisions.length <= 0) {
    throw new Error(
      `no usable go/stop decisions found in dataset: ${opts.datasetIn} (missing_state_rows=${missingStateRows})`
    );
  }

  mkdirSync(dirname(resolve(opts.out)), { recursive: true });
  const writer = createWriteStream(resolve(opts.out), { flags: "w", encoding: "utf8" });
  let processed = 0;

  for (const decision of decisions) {
    const returnsByAction = { go: [], stop: [] };
    for (let sampleIndex = 0; sampleIndex < opts.samplesPerAction; sampleIndex += 1) {
      const sampleSeed = `${opts.seed}|${decision.decision_id}|sample|${sampleIndex}`;
      const determinized = determinizeHiddenState(
        decision.state_before_full,
        decision.actor,
        createSeededRng(`${sampleSeed}|det`)
      );
      returnsByAction.go.push(
        evaluateActionReturn(
          determinized,
          decision.actor,
          "go",
          teacherSpec,
          `${sampleSeed}|go`,
          opts.maxSteps
        )
      );
      returnsByAction.stop.push(
        evaluateActionReturn(
          determinized,
          decision.actor,
          "stop",
          teacherSpec,
          `${sampleSeed}|stop`,
          opts.maxSteps
        )
      );
    }

    const row = buildOutputRow(decision, opts.teacher, opts.samplesPerAction, returnsByAction);
    writer.write(`${JSON.stringify(row)}\n`);
    processed += 1;
    if (processed % opts.progressEvery === 0) {
      console.log(`processed ${processed}/${decisions.length}`);
    }
  }

  await new Promise((resolveDone, rejectDone) => {
    writer.end((err) => (err ? rejectDone(err) : resolveDone()));
  });

  console.log(
    JSON.stringify({
      dataset_in: resolve(opts.datasetIn),
      out: resolve(opts.out),
      teacher_policy: opts.teacher,
      decisions: decisions.length,
      samples_per_action: opts.samplesPerAction,
      missing_state_rows: missingStateRows,
    })
  );
}

main().catch((err) => {
  console.error(String(err?.stack || err));
  process.exitCode = 1;
});
