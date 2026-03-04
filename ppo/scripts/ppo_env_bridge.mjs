#!/usr/bin/env node
// ppo_env_bridge.mjs
// - StdIO JSON bridge between Matgo engine and PPO trainer.
// - Strict CLI/runtime validation + fail-fast errors with context.

import { initSimulationGame, createSeededRng, calculateScore, scoringPiCount } from "../../src/engine/index.js";
import { getActionPlayerKey } from "../../src/engine/runner.js";
import { aiPlay } from "../../src/ai/aiPlay_by_GPT.js";
import { normalizeBotPolicy } from "../../src/ai/policies.js";
import {
  selectDecisionPool,
  resolveDecisionType,
  legalCandidatesForDecision,
  applyDecisionAction,
  canonicalOptionAction,
  stateProgressKey
} from "../../src/ai/decisionRuntime_by_GPT.js";

const ACTION_DIM = 26;
const OBS_DIM = 165;
const PLAY_SLOTS = 10;
const MATCH_SLOTS = 8;
const OPTION_ORDER = Object.freeze([
  "go",
  "stop",
  "shaking_yes",
  "shaking_no",
  "president_stop",
  "president_hold",
  "five",
  "junk"
]);

function fail(message) {
  throw new Error(String(message || "unknown failure"));
}

function toPositiveInt(raw, name) {
  const n = Number(raw);
  if (!Number.isFinite(n) || !Number.isInteger(n) || n <= 0) {
    fail(`invalid ${name}: ${raw}`);
  }
  return n;
}

function toFiniteNumber(raw, name) {
  const n = Number(raw);
  if (!Number.isFinite(n)) {
    fail(`invalid ${name}: ${raw}`);
  }
  return n;
}

function otherActor(actor) {
  if (actor === "human") return "ai";
  if (actor === "ai") return "human";
  fail(`invalid actor: ${actor}`);
}

function normalizeActor(raw, name) {
  const v = String(raw || "").trim().toLowerCase();
  if (v !== "human" && v !== "ai") fail(`invalid ${name}: ${raw}`);
  return v;
}

function normalizeFirstTurnPolicy(raw) {
  const v = String(raw || "").trim().toLowerCase();
  if (v !== "alternate" && v !== "fixed") {
    fail(`invalid --first-turn-policy: ${raw} (allowed: alternate|fixed)`);
  }
  return v;
}

function normalizeTrainingMode(raw) {
  const v = String(raw || "").trim().toLowerCase();
  if (v !== "single_actor" && v !== "selfplay") {
    fail(`invalid --training-mode: ${raw} (allowed: single_actor|selfplay)`);
  }
  return v;
}

function parseArgs(argv) {
  const args = [...argv];
  const out = {
    trainingMode: "",
    seedBase: "",
    ruleKey: "",
    controlActor: "",
    opponentPolicy: "",
    maxSteps: null,
    rewardScale: null,
    downsidePenaltyScale: null,
    terminalBonusScale: null,
    firstTurnPolicy: "",
    fixedFirstTurn: ""
  };

  while (args.length > 0) {
    const raw = String(args.shift() || "");
    if (!raw.startsWith("--")) fail(`unknown argument: ${raw}`);

    const eq = raw.indexOf("=");
    let key = raw;
    let value = "";
    if (eq >= 0) {
      key = raw.slice(0, eq);
      value = raw.slice(eq + 1);
    } else {
      value = String(args.shift() || "");
    }

    if (key === "--training-mode") out.trainingMode = normalizeTrainingMode(value);
    else if (key === "--seed-base") out.seedBase = String(value || "").trim();
    else if (key === "--rule-key") out.ruleKey = String(value || "").trim();
    else if (key === "--control-actor") out.controlActor = normalizeActor(value, "--control-actor");
    else if (key === "--opponent-policy") out.opponentPolicy = String(value || "").trim();
    else if (key === "--max-steps") out.maxSteps = toPositiveInt(value, "--max-steps");
    else if (key === "--reward-scale") out.rewardScale = toFiniteNumber(value, "--reward-scale");
    else if (key === "--downside-penalty-scale") {
      out.downsidePenaltyScale = toFiniteNumber(value, "--downside-penalty-scale");
    } else if (key === "--terminal-bonus-scale") {
      out.terminalBonusScale = toFiniteNumber(value, "--terminal-bonus-scale");
    } else if (key === "--first-turn-policy") {
      out.firstTurnPolicy = normalizeFirstTurnPolicy(value);
    } else if (key === "--fixed-first-turn") {
      out.fixedFirstTurn = normalizeActor(value, "--fixed-first-turn");
    } else {
      fail(`unknown argument: ${key}`);
    }
  }

  if (!out.trainingMode) fail("--training-mode is required");
  if (!out.seedBase) fail("--seed-base is required");
  if (!out.ruleKey) fail("--rule-key is required");
  if (out.maxSteps == null) fail("--max-steps is required");
  if (out.rewardScale == null) fail("--reward-scale is required");
  if (Math.abs(out.rewardScale) <= 0) fail("--reward-scale must be non-zero");
  if (out.downsidePenaltyScale == null) fail("--downside-penalty-scale is required");
  if (out.downsidePenaltyScale < 0) fail("--downside-penalty-scale must be >= 0");
  if (out.terminalBonusScale == null) fail("--terminal-bonus-scale is required");
  if (!Number.isFinite(out.terminalBonusScale)) fail("--terminal-bonus-scale must be finite");
  if (!out.firstTurnPolicy) fail("--first-turn-policy is required");
  if (out.firstTurnPolicy === "fixed" && !out.fixedFirstTurn) {
    fail("--fixed-first-turn is required when --first-turn-policy=fixed");
  }
  if (out.trainingMode === "single_actor") {
    if (!out.controlActor) fail("--control-actor is required when --training-mode=single_actor");
    if (!out.opponentPolicy) fail("--opponent-policy is required when --training-mode=single_actor");
  } else {
    if (out.controlActor) {
      fail("--control-actor is not allowed when --training-mode=selfplay");
    }
    if (out.opponentPolicy) {
      fail("--opponent-policy is not allowed when --training-mode=selfplay");
    }
  }

  return out;
}

function clamp01(x) {
  const n = Number(x || 0);
  if (n <= 0) return 0;
  if (n >= 1) return 1;
  return n;
}

function tanhNorm(x, scale) {
  const s = Math.max(1e-6, Number(scale || 1));
  return Math.tanh(Number(x || 0) / s);
}

function findCardById(cards, cardId) {
  const target = String(cardId || "");
  if (!target || !Array.isArray(cards)) return null;
  for (const card of cards) {
    if (String(card?.id || "") === target) return card;
  }
  return null;
}

function cardFeatureVector(card) {
  if (!card || typeof card !== "object") {
    return [0, 0, 0, 0, 0, 0, 0];
  }
  const month = Number(card.month || 0);
  const category = String(card.category || "");
  const piValue = Number(card.piValue || 0);
  return [
    1,
    month >= 1 ? clamp01(month / 12.0) : 0,
    category === "kwang" ? 1 : 0,
    category === "five" ? 1 : 0,
    category === "ribbon" ? 1 : 0,
    category === "junk" ? 1 : 0,
    clamp01(piValue / 5.0)
  ];
}

function appendCardSlots(out, cards, maxSlots) {
  const source = Array.isArray(cards) ? cards : [];
  for (let i = 0; i < maxSlots; i += 1) {
    const c = i < source.length ? source[i] : null;
    const fv = cardFeatureVector(c);
    for (const v of fv) out.push(Number(v));
  }
}

function resolveDecisionContext(state, controlActor) {
  const pool = selectDecisionPool(state, controlActor);
  const decisionType = resolveDecisionType(pool);
  if (!decisionType) {
    fail(`decision_type unresolved: actor=${controlActor}, phase=${String(state?.phase || "")}`);
  }
  const candidates = legalCandidatesForDecision(pool, decisionType);
  if (!Array.isArray(candidates) || candidates.length <= 0) {
    fail(
      `legal candidates empty: actor=${controlActor}, phase=${String(state?.phase || "")}, decisionType=${decisionType}`
    );
  }
  return { pool, decisionType, candidates };
}

function buildActionContext(state, controlActor) {
  const { pool, decisionType, candidates } = resolveDecisionContext(state, controlActor);
  const mask = new Array(ACTION_DIM).fill(0);
  const actionToCandidate = new Map();

  if (decisionType === "play") {
    if (candidates.length > PLAY_SLOTS) {
      fail(`play candidate overflow: ${candidates.length} > ${PLAY_SLOTS}`);
    }
    for (let i = 0; i < candidates.length; i += 1) {
      mask[i] = 1;
      actionToCandidate.set(i, String(candidates[i]));
    }
  } else if (decisionType === "match") {
    if (candidates.length > MATCH_SLOTS) {
      fail(`match candidate overflow: ${candidates.length} > ${MATCH_SLOTS}`);
    }
    for (let i = 0; i < candidates.length; i += 1) {
      const idx = PLAY_SLOTS + i;
      mask[idx] = 1;
      actionToCandidate.set(idx, String(candidates[i]));
    }
  } else if (decisionType === "option") {
    for (const raw of candidates) {
      const option = canonicalOptionAction(raw);
      const pos = OPTION_ORDER.indexOf(option);
      if (pos < 0) {
        fail(`unsupported option candidate: ${option}`);
      }
      const idx = PLAY_SLOTS + MATCH_SLOTS + pos;
      mask[idx] = 1;
      actionToCandidate.set(idx, option);
    }
  } else {
    fail(`unsupported decision type: ${decisionType}`);
  }

  const legalActions = [];
  for (let i = 0; i < mask.length; i += 1) {
    if (mask[i] === 1) legalActions.push(i);
  }
  if (legalActions.length <= 0) {
    fail(`action mask is empty: phase=${String(state?.phase || "")}, decisionType=${decisionType}`);
  }

  return { pool, decisionType, candidates, mask, actionToCandidate, legalActions };
}

function goldDiff(state, controlActor) {
  const opp = otherActor(controlActor);
  const selfGold = Number(state?.players?.[controlActor]?.gold || 0);
  const oppGold = Number(state?.players?.[opp]?.gold || 0);
  return selfGold - oppGold;
}

function buildObservation(state, controlActor, actionCtx) {
  const opp = otherActor(controlActor);
  const scoreSelf = calculateScore(state.players[controlActor], state.players[opp], state.ruleKey);
  const scoreOpp = calculateScore(state.players[opp], state.players[controlActor], state.ruleKey);
  const phase = String(state?.phase || "");
  const out = [];

  // 1) phase one-hot (6)
  out.push(phase === "playing" ? 1 : 0);
  out.push(phase === "select-match" ? 1 : 0);
  out.push(phase === "go-stop" ? 1 : 0);
  out.push(phase === "president-choice" ? 1 : 0);
  out.push(phase === "gukjin-choice" ? 1 : 0);
  out.push(phase === "shaking-confirm" ? 1 : 0);

  // 2) decision type one-hot (3)
  out.push(actionCtx.decisionType === "play" ? 1 : 0);
  out.push(actionCtx.decisionType === "match" ? 1 : 0);
  out.push(actionCtx.decisionType === "option" ? 1 : 0);

  // 3) macro counts/scores (22 total -> now 31 cumulative)
  out.push(clamp01(Number(state?.deck?.length || 0) / 30.0));
  out.push(clamp01(Number(state?.players?.[controlActor]?.hand?.length || 0) / 10.0));
  out.push(clamp01(Number(state?.players?.[opp]?.hand?.length || 0) / 10.0));
  out.push(clamp01(Number(state?.board?.length || 0) / 24.0));
  out.push(clamp01(Number(state?.players?.[controlActor]?.goCount || 0) / 5.0));
  out.push(clamp01(Number(state?.players?.[opp]?.goCount || 0) / 5.0));
  out.push(tanhNorm(Number(scoreSelf?.total || 0), 10.0));
  out.push(tanhNorm(Number(scoreOpp?.total || 0), 10.0));
  out.push(tanhNorm(Number(scoreSelf?.total || 0) - Number(scoreOpp?.total || 0), 10.0));

  const selfGold = Number(state?.players?.[controlActor]?.gold || 0);
  const oppGold = Number(state?.players?.[opp]?.gold || 0);
  out.push(tanhNorm(selfGold, 50000.0));
  out.push(tanhNorm(oppGold, 50000.0));
  out.push(tanhNorm(selfGold - oppGold, 25000.0));

  const carryOver = Math.max(1.0, Number(state?.carryOverMultiplier || 1.0));
  out.push(clamp01((carryOver - 1.0) / 12.0));
  out.push(clamp01(Number(state?.turnSeq || 0) / 200.0));

  const selfCap = state?.players?.[controlActor]?.captured || {};
  const oppCap = state?.players?.[opp]?.captured || {};
  out.push(clamp01((selfCap.kwang || []).length / 5.0));
  out.push(clamp01((selfCap.five || []).length / 5.0));
  out.push(clamp01((selfCap.ribbon || []).length / 10.0));
  out.push(clamp01(Number(scoringPiCount(state.players[controlActor]) || 0) / 20.0));
  out.push(clamp01((oppCap.kwang || []).length / 5.0));
  out.push(clamp01((oppCap.five || []).length / 5.0));
  out.push(clamp01((oppCap.ribbon || []).length / 10.0));
  out.push(clamp01(Number(scoringPiCount(state.players[opp]) || 0) / 20.0));

  // 4) hand card slots (10 * 7 = 70)
  appendCardSlots(out, state?.players?.[controlActor]?.hand || [], PLAY_SLOTS);

  // 5) pending match slots (8 * 7 = 56)
  const board = state?.board || [];
  const pendingMatchCards = [];
  if (actionCtx.decisionType === "match") {
    for (const cardId of actionCtx.candidates) {
      pendingMatchCards.push(findCardById(board, cardId));
    }
  }
  appendCardSlots(out, pendingMatchCards, MATCH_SLOTS);

  // 6) option availability (8)
  for (let i = 0; i < OPTION_ORDER.length; i += 1) {
    const idx = PLAY_SLOTS + MATCH_SLOTS + i;
    out.push(actionCtx.mask[idx] === 1 ? 1 : 0);
  }

  for (const v of out) {
    if (!Number.isFinite(v)) {
      fail(`observation contains non-finite value: phase=${phase}`);
    }
  }
  return out;
}

function buildStepView(state, controlActor) {
  const actionCtx = buildActionContext(state, controlActor);
  return {
    obs: buildObservation(state, controlActor, actionCtx),
    action_mask: actionCtx.mask,
    decision_type: actionCtx.decisionType,
    legal_actions: actionCtx.legalActions
  };
}

function resolveFirstTurnKey(runtime) {
  if (runtime.firstTurnPolicy === "fixed") return runtime.fixedFirstTurn;
  return runtime.episodeIndex % 2 === 0 ? "human" : "ai";
}

function progressOrThrow(beforeState, nextState, context) {
  const before = stateProgressKey(beforeState, { includeKiboSeq: true });
  const after = stateProgressKey(nextState, { includeKiboSeq: true });
  if (!nextState || before === after) {
    fail(
      `action resolution failed: seed=${context.seed}, step=${context.step}, actor=${context.actor}, phase=${context.phase}, detail=${context.detail}`
    );
  }
}

function runOpponentUntilControl(runtime, seedText) {
  let guard = 0;
  while (runtime.state.phase !== "resolution") {
    const actor = getActionPlayerKey(runtime.state);
    if (!actor) break;
    if (actor === runtime.controlActor) break;
    const before = runtime.state;
    const next = aiPlay(runtime.state, actor, {
      source: "heuristic",
      heuristicPolicy: runtime.opponentPolicy
    });
    progressOrThrow(before, next, {
      seed: seedText,
      step: runtime.actionCount,
      actor,
      phase: String(before?.phase || ""),
      detail: "opponent_auto"
    });
    runtime.state = next;
    runtime.actionCount += 1;
    guard += 1;
    if (runtime.actionCount > runtime.maxSteps) {
      fail(`episode max step overflow while auto-running opponent: seed=${seedText}, step=${runtime.actionCount}`);
    }
    if (guard > runtime.maxSteps) {
      fail(`opponent auto loop overflow: seed=${seedText}, step=${runtime.actionCount}`);
    }
  }
}

function initEpisode(runtime) {
  const firstTurnKey = resolveFirstTurnKey(runtime);
  const attemptBase = runtime.resetCount;
  const maxAttempts = 20;
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    const seedText = `${runtime.seedBase}|episode=${runtime.episodeIndex}|reset=${attemptBase}|attempt=${attempt}`;
    const rng = createSeededRng(seedText);
    runtime.state = initSimulationGame(runtime.ruleKey, rng, {
      kiboDetail: "lean",
      firstTurnKey
    });
    runtime.seedText = seedText;
    runtime.actionCount = 0;
    if (runtime.trainingMode === "single_actor") {
      runOpponentUntilControl(runtime, seedText);
    }
    if (runtime.state.phase !== "resolution") return;
  }
  fail(`failed to initialize non-terminal episode after ${maxAttempts} attempts (seedBase=${runtime.seedBase})`);
}

function buildRuntime(cli) {
  return {
    trainingMode: cli.trainingMode,
    seedBase: cli.seedBase,
    ruleKey: cli.ruleKey,
    controlActor: cli.controlActor || null,
    opponentPolicy: cli.opponentPolicy ? normalizeBotPolicy(cli.opponentPolicy) : null,
    maxSteps: cli.maxSteps,
    rewardScale: cli.rewardScale,
    downsidePenaltyScale: cli.downsidePenaltyScale,
    terminalBonusScale: cli.terminalBonusScale,
    firstTurnPolicy: cli.firstTurnPolicy,
    fixedFirstTurn: cli.fixedFirstTurn || "human",
    state: null,
    seedText: "",
    episodeIndex: 0,
    resetCount: 0,
    actionCount: 0
  };
}

function ensureControlTurn(runtime) {
  if (runtime.trainingMode !== "single_actor") return;
  if (!runtime.state || runtime.state.phase === "resolution") return;
  const actor = getActionPlayerKey(runtime.state);
  if (actor !== runtime.controlActor) {
    fail(
      `control actor turn mismatch: seed=${runtime.seedText}, step=${runtime.actionCount}, actor=${String(actor || "")}, expected=${runtime.controlActor}, phase=${String(runtime.state?.phase || "")}`
    );
  }
}

function handleReset(runtime, payload) {
  const epRaw = payload && Object.prototype.hasOwnProperty.call(payload, "episode") ? payload.episode : null;
  if (epRaw != null) {
    runtime.episodeIndex = toPositiveInt(Number(epRaw) + 1, "episode") - 1;
  }
  runtime.resetCount += 1;
  initEpisode(runtime);
  if (runtime.trainingMode === "single_actor") {
    ensureControlTurn(runtime);
  }
  const actorToAct = runtime.trainingMode === "single_actor" ? runtime.controlActor : getActionPlayerKey(runtime.state);
  if (!actorToAct) {
    fail(`actor_to_act unresolved on reset: seed=${runtime.seedText}, phase=${String(runtime.state?.phase || "")}`);
  }
  const view = buildStepView(runtime.state, actorToAct);
  return {
    ok: true,
    type: "reset",
    obs: view.obs,
    action_mask: view.action_mask,
    info: {
      seed: runtime.seedText,
      episode: runtime.episodeIndex,
      training_mode: runtime.trainingMode,
      control_actor: runtime.controlActor,
      opponent_policy: runtime.opponentPolicy,
      actor_to_act: actorToAct,
      phase: runtime.state.phase,
      decision_type: view.decision_type,
      legal_actions: view.legal_actions,
      action_dim: ACTION_DIM,
      obs_dim: view.obs.length
    }
  };
}

function handleStep(runtime, payload) {
  if (!runtime.state) fail("step called before reset");
  if (runtime.trainingMode === "single_actor") {
    ensureControlTurn(runtime);
  }

  const actionRaw = payload ? payload.action : null;
  const actionIndex = toPositiveInt(Number(actionRaw) + 1, "action") - 1;
  if (actionIndex < 0 || actionIndex >= ACTION_DIM) {
    fail(`action out of range: ${actionIndex} (allowed: 0..${ACTION_DIM - 1})`);
  }

  const actingActor = runtime.trainingMode === "single_actor" ? runtime.controlActor : getActionPlayerKey(runtime.state);
  if (!actingActor) {
    fail(`acting actor unresolved: seed=${runtime.seedText}, step=${runtime.actionCount}, phase=${String(runtime.state?.phase || "")}`);
  }

  const beforeState = runtime.state;
  const beforeDiff = goldDiff(beforeState, actingActor);
  const actionCtx = buildActionContext(beforeState, actingActor);
  if (actionCtx.mask[actionIndex] !== 1) {
    fail(
      `illegal action index: seed=${runtime.seedText}, step=${runtime.actionCount}, actor=${actingActor}, phase=${beforeState.phase}, decisionType=${actionCtx.decisionType}, action=${actionIndex}`
    );
  }
  const candidate = actionCtx.actionToCandidate.get(actionIndex);
  if (candidate == null) {
    fail(`candidate resolution failed for action index: ${actionIndex}`);
  }

  const nextState = applyDecisionAction(beforeState, actingActor, actionCtx.decisionType, candidate);
  progressOrThrow(beforeState, nextState, {
    seed: runtime.seedText,
    step: runtime.actionCount,
    actor: actingActor,
    phase: String(beforeState?.phase || ""),
    detail: `policy_action:${actionCtx.decisionType}:${String(candidate)}`
  });
  runtime.state = nextState;
  runtime.actionCount += 1;

  if (runtime.trainingMode === "single_actor") {
    runOpponentUntilControl(runtime, runtime.seedText);
  }

  const truncated = runtime.state.phase !== "resolution" && runtime.actionCount >= runtime.maxSteps;
  const done = runtime.state.phase === "resolution" || truncated;

  const afterDiff = goldDiff(runtime.state, actingActor);
  let reward = (afterDiff - beforeDiff) * runtime.rewardScale;
  if (done) {
    reward += afterDiff * runtime.terminalBonusScale;
    reward -= Math.max(0, -afterDiff) * runtime.downsidePenaltyScale;
  }

  let view = null;
  if (!done) {
    if (runtime.trainingMode === "single_actor") {
      ensureControlTurn(runtime);
    }
    const actorToAct = runtime.trainingMode === "single_actor" ? runtime.controlActor : getActionPlayerKey(runtime.state);
    if (!actorToAct) {
      fail(`actor_to_act unresolved after step: seed=${runtime.seedText}, step=${runtime.actionCount}, phase=${String(runtime.state?.phase || "")}`);
    }
    view = buildStepView(runtime.state, actorToAct);
  } else {
    view = {
      obs: new Array(OBS_DIM).fill(0),
      action_mask: new Array(ACTION_DIM).fill(0),
      decision_type: "terminal",
      legal_actions: []
    };
  }

  return {
    ok: true,
    type: "step",
    obs: view.obs,
    action_mask: view.action_mask,
    reward,
    done,
    truncated,
    info: {
      seed: runtime.seedText,
      episode: runtime.episodeIndex,
      training_mode: runtime.trainingMode,
      step: runtime.actionCount,
      acted_actor: actingActor,
      actor_to_act: done ? null : getActionPlayerKey(runtime.state),
      phase: runtime.state.phase,
      decision_type: view.decision_type,
      legal_actions: view.legal_actions,
      control_gold: Number(runtime.state?.players?.[actingActor]?.gold || 0),
      opponent_gold: Number(runtime.state?.players?.[otherActor(actingActor)]?.gold || 0),
      final_gold_diff: afterDiff
    }
  };
}

function handleClose() {
  return { ok: true, type: "close" };
}

function send(obj) {
  process.stdout.write(`${JSON.stringify(obj)}\n`);
}

const cli = parseArgs(process.argv.slice(2));
const runtime = buildRuntime(cli);

process.on("uncaughtException", (err) => {
  const message = err && err.message ? String(err.message) : String(err);
  send({ ok: false, error: message });
  process.exit(1);
});

process.on("unhandledRejection", (err) => {
  const message = err && err.message ? String(err.message) : String(err);
  send({ ok: false, error: message });
  process.exit(1);
});

let didReset = false;
process.stdin.setEncoding("utf8");
let buffer = "";
process.stdin.on("data", (chunk) => {
  buffer += String(chunk || "");
  while (true) {
    const idx = buffer.indexOf("\n");
    if (idx < 0) break;
    const line = buffer.slice(0, idx).trim();
    buffer = buffer.slice(idx + 1);
    if (!line) continue;

    try {
      const msg = JSON.parse(line);
      const cmd = String(msg?.cmd || "").trim().toLowerCase();
      if (!cmd) fail("missing cmd");

      let response = null;
      if (cmd === "reset") {
        response = handleReset(runtime, msg);
        didReset = true;
      } else if (cmd === "step") {
        if (!didReset) fail("step received before reset");
        response = handleStep(runtime, msg);
      } else if (cmd === "close") {
        response = handleClose();
        send(response);
        process.exit(0);
      } else {
        fail(`unknown cmd: ${cmd}`);
      }

      send(response);
    } catch (err) {
      const message = err && err.message ? String(err.message) : String(err);
      send({ ok: false, error: message });
      process.exit(1);
    }
  }
});
