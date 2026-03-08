import {
  initSimulationGame,
  startSimulationGame,
  createSeededRng,
  getDeclarableBombMonths
} from "../../src/engine/index.js";
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { getActionPlayerKey } from "../../src/engine/runner.js";
import { aiPlay } from "../../src/ai/aiPlay_by_GPT.js";
import { resolveBotPolicy } from "../../src/ai/policies.js";
import { hybridPolicyPlayDetailed } from "../../src/ai/hybridPolicyEngine.js";
import {
  applyDecisionAction,
  canonicalOptionAction,
  legalCandidatesForDecision,
  normalizeOptionCandidates,
  resolveDecisionType,
  selectDecisionPool,
  stateProgressKey
} from "../../src/ai/decisionRuntime_by_GPT.js";
import { resolveHybridPlayModelSpec } from "./hybridOpponentSpec.mjs";

// GPT-only duel runner (minimal duel/report flow).
const SCRIPT_FILE = fileURLToPath(import.meta.url);
const SCRIPT_DIR = dirname(SCRIPT_FILE);
const REPO_ROOT = resolve(SCRIPT_DIR, "..", "..");

function parseArgs(argv) {
  const args = [...argv];
  if (args.includes("--help") || args.includes("-h")) {
    return { help: true };
  }

  const out = {
    humanSpecRaw: "",
    aiSpecRaw: "",
    games: 1000,
    seed: "model-duel-gpt",
    maxSteps: 600,
    firstTurnPolicy: "alternate",
    fixedFirstTurn: "human",
    continuousSeries: true,
    stdoutFormat: "text",
    resultOut: "",
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

    if (key === "--human") out.humanSpecRaw = String(value || "").trim();
    else if (key === "--ai") out.aiSpecRaw = String(value || "").trim();
    else if (key === "--games") out.games = Math.max(1, Number(value || 1000));
    else if (key === "--seed") out.seed = String(value || "model-duel-gpt").trim();
    else if (key === "--max-steps") out.maxSteps = Math.max(20, Number(value || 600));
    else if (key === "--first-turn-policy") {
      out.firstTurnPolicy = String(value || "alternate").trim().toLowerCase();
    }
    else if (key === "--fixed-first-turn") {
      out.fixedFirstTurn = String(value || "human").trim().toLowerCase();
    }
    else if (key === "--continuous-series") out.continuousSeries = parseContinuousSeriesValue(value);
    else if (key === "--stdout-format") out.stdoutFormat = String(value || "text").trim().toLowerCase();
    else if (key === "--result-out") out.resultOut = String(value || "").trim();
    else {
      throw new Error(
        `Unknown argument: ${key} (allowed: --human, --ai, --games, --seed, --max-steps, --first-turn-policy, --fixed-first-turn, --continuous-series, --stdout-format, --result-out)`
      );
    }
  }

  if (!out.humanSpecRaw) throw new Error("--human is required");
  if (!out.aiSpecRaw) throw new Error("--ai is required");
  if (Math.floor(out.games) < 1000) throw new Error("this worker requires --games >= 1000");
  out.games = Math.floor(out.games);

  if (out.firstTurnPolicy !== "alternate" && out.firstTurnPolicy !== "fixed") {
    throw new Error(`invalid --first-turn-policy: ${out.firstTurnPolicy}`);
  }
  if (out.fixedFirstTurn !== "human") {
    throw new Error("--fixed-first-turn is locked to human");
  }
  if (out.stdoutFormat !== "text" && out.stdoutFormat !== "json") {
    throw new Error(`invalid --stdout-format: ${out.stdoutFormat} (allowed: text, json)`);
  }

  return out;
}

function usageText() {
  return [
    "Usage:",
    "  node neat_by_GPT/scripts/model_duel_worker.mjs --human <spec> --ai <spec> [options]",
    "",
    "Required:",
    "  --human <policy|gpt_run|model:path/to/winner_genome.json>",
    "  --ai <policy|gpt_run|model:path/to/winner_genome.json>",
    "",
    "Options:",
    "  --games <N>                 default=1000, minimum=1000",
    "  --seed <tag>                default=model-duel-gpt",
    "  --max-steps <N>             default=600, minimum=20",
    "  --first-turn-policy <mode>  alternate|fixed (default=alternate)",
    "  --fixed-first-turn <actor>  human only (default=human)",
    "  --continuous-series <flag>  1=true, 2=false (default=1)",
    "  --stdout-format <mode>      text|json (default=text)",
    "  --result-out <path>         optional, auto-generated if omitted",
    "",
    "Example:",
    "  node neat_by_GPT/scripts/model_duel_worker.mjs --human H-CL --ai runtime_focus_cl_v1_seed9 --games 1000 --seed gpt_duel_1 --first-turn-policy alternate --continuous-series 1",
  ].join("\n");
}

function parseContinuousSeriesValue(value) {
  const raw = String(value ?? "1").trim();
  if (raw === "" || raw === "1") return true;
  if (raw === "2") return false;
  throw new Error(`invalid --continuous-series: ${raw} (allowed: 1=true, 2=false)`);
}

function sanitizeFilePart(text) {
  return String(text || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function dateTag() {
  const now = new Date();
  const yyyy = String(now.getFullYear());
  const mm = String(now.getMonth() + 1).padStart(2, "0");
  const dd = String(now.getDate()).padStart(2, "0");
  return `${yyyy}${mm}${dd}`;
}

function toAbsoluteRepoPath(rawPath) {
  const token = String(rawPath || "").trim();
  if (!token) return "";
  return resolve(REPO_ROOT, token);
}

function tryResolveModelArtifactPath(rawSpec) {
  const token = String(rawSpec || "").trim();
  if (!token) return null;

  const explicitPath = token.startsWith("model:") ? token.slice("model:".length).trim() : token;
  const looksLikePath =
    token.startsWith("model:") ||
    explicitPath.includes("/") ||
    explicitPath.includes("\\") ||
    explicitPath.toLowerCase().endsWith(".json");
  if (looksLikePath) {
    const absolute = toAbsoluteRepoPath(explicitPath);
    if (!existsSync(absolute)) {
      return {
        path: absolute,
        source: "explicit_path",
        exists: false,
      };
    }
    const stat = statSync(absolute);
    if (stat.isDirectory()) {
      const dirCandidate = join(absolute, "models", "winner_genome.json");
      const fileCandidate = join(absolute, "winner_genome.json");
      if (existsSync(dirCandidate)) {
        return { path: dirCandidate, source: "explicit_run_dir", exists: true };
      }
      if (existsSync(fileCandidate)) {
        return { path: fileCandidate, source: "explicit_dir_file", exists: true };
      }
      return {
        path: dirCandidate,
        source: "explicit_run_dir",
        exists: false,
      };
    }
    return { path: absolute, source: "explicit_file", exists: true };
  }

  const runDirCandidate = join(REPO_ROOT, "logs", "NEAT_GPT", token, "models", "winner_genome.json");
  if (existsSync(runDirCandidate)) {
    return { path: runDirCandidate, source: "named_gpt_run", exists: true };
  }
  return null;
}

function loadModelSpec(input, label, modelPath) {
  let model = null;
  try {
    const raw = String(readFileSync(modelPath, "utf8") || "").replace(/^\uFEFF/, "");
    model = JSON.parse(raw);
  } catch (err) {
    throw new Error(`failed to parse model JSON (${input}): ${modelPath} (${String(err)})`);
  }
  if (String(model?.format_version || "").trim() !== "neat_python_genome_v1") {
    throw new Error(`invalid model format for ${input}: expected neat_python_genome_v1`);
  }

  return {
    input,
    kind: "model",
    key: label,
    label,
    model,
    modelPath,
  };
}

function resolvePlayerSpec(rawSpec, sideLabel) {
  const token = String(rawSpec || "").trim();
  if (!token) throw new Error(`empty player spec: ${sideLabel}`);

  const hybridSpec = resolveHybridPlayModelSpec(token, sideLabel);
  if (hybridSpec) {
    return hybridSpec;
  }

  const resolvedPolicy = resolveBotPolicy(token);
  if (resolvedPolicy) {
    return {
      input: token,
      kind: "heuristic",
      key: resolvedPolicy,
      label: resolvedPolicy,
      model: null,
      modelPath: null,
      phase: null,
      seed: null,
    };
  }

  const modelArtifact = tryResolveModelArtifactPath(token);
  if (modelArtifact?.exists) {
    const label =
      modelArtifact.source === "named_gpt_run"
        ? token
        : sanitizeFilePart(token).replace(/^model_/, "") || "gpt_model";
    const loaded = loadModelSpec(token, label, modelArtifact.path);
    return {
      ...loaded,
      phase: null,
      seed: null,
    };
  }
  if (modelArtifact && modelArtifact.exists === false) {
    throw new Error(`model not found for ${token}: ${modelArtifact.path}`);
  }
  throw new Error(
    `invalid ${sideLabel} spec: ${token} (use policy key, GPT run name like runtime_focus_cl_v1_seed9, model:path/to/winner_genome.json, hybrid_play(phase2_seed203,H-CL), hybrid_play_go(phase2_seed203,H-CL), or hybrid_play_go(phase2_seed203,H-NEXg,H-CL))`
  );
}

function buildAutoOutputDir(humanLabel, aiLabel) {
  const duelKey = `${sanitizeFilePart(humanLabel)}_vs_${sanitizeFilePart(aiLabel)}_${dateTag()}`;
  const outDir = join(REPO_ROOT, "logs", "NEAT_GPT", "duels", duelKey);
  mkdirSync(outDir, { recursive: true });
  return outDir;
}

function buildAutoArtifactPath(outDir, seed, suffix) {
  const stem = sanitizeFilePart(seed) || "model-duel-gpt";
  return join(outDir, `${stem}_${suffix}`);
}

function toReportPath(pathValue) {
  const raw = String(pathValue || "").trim();
  if (!raw) return null;
  const rel = relative(REPO_ROOT, resolve(raw));
  const normalized = String(rel || raw).replace(/\\/g, "/");
  return normalized || null;
}

function resolveFirstTurnKey(opts, gameIndex) {
  if (opts.firstTurnPolicy === "fixed") return opts.fixedFirstTurn;
  return gameIndex % 2 === 0 ? "ai" : "human";
}

function startRound(seed, firstTurnKey) {
  return initSimulationGame("A", createSeededRng(`${seed}|game`), {
    firstTurnKey,
  });
}

function continueRound(prevEndState, seed, firstTurnKey) {
  return startSimulationGame(prevEndState, createSeededRng(`${seed}|game`), {
    keepGold: true,
    useCarryOver: true,
    firstTurnKey,
  });
}

function goldDiffByActor(state, actor) {
  const opp = actor === "human" ? "ai" : "human";
  const selfGold = Number(state?.players?.[actor]?.gold || 0);
  const oppGold = Number(state?.players?.[opp]?.gold || 0);
  return selfGold - oppGold;
}

function buildAiPlayOptions(playerSpec) {
  if (playerSpec?.kind === "model" && playerSpec?.model) {
    return { source: "model", model: playerSpec.model };
  }
  return { source: "heuristic", heuristicPolicy: String(playerSpec?.key || "") };
}

function resolvePlayerAction(state, actor, playerSpec) {
  if (playerSpec?.kind === "hybrid_play_model") {
    const traced = hybridPolicyPlayDetailed(state, actor, {
      model: playerSpec.model,
      heuristicPolicy: String(playerSpec.heuristicPolicy || ""),
      goStopPolicy: String(playerSpec.goStopPolicy || playerSpec.heuristicPolicy || ""),
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

const PLAY_SPECIAL_SHAKE_PREFIX = "shake_start:";
const PLAY_SPECIAL_BOMB_PREFIX = "bomb:";

function detailedStateProgressKey(state) {
  return stateProgressKey(state, { includeKiboSeq: true });
}

function normalizeDecisionCandidate(decisionType, candidate) {
  if (decisionType === "option") return canonicalOptionAction(candidate);
  return String(candidate || "").trim();
}

function inferCandidateFromKiboTransition(stateBefore, actor, decisionType, candidates, stateAfter) {
  const beforeLen = Array.isArray(stateBefore?.kibo) ? stateBefore.kibo.length : 0;
  const afterKibo = Array.isArray(stateAfter?.kibo) ? stateAfter.kibo : [];
  if (afterKibo.length <= beforeLen) return null;

  const candidateSet = new Set((candidates || []).map((c) => normalizeDecisionCandidate(decisionType, c)));
  const shakingDeclared =
    decisionType === "play" &&
    afterKibo.slice(beforeLen).some(
      (ev) =>
        String(ev?.type || "").trim().toLowerCase() === "shaking_declare" &&
        String(ev?.playerKey || "") === String(actor || "")
    );
  for (let i = afterKibo.length - 1; i >= beforeLen; i -= 1) {
    const ev = afterKibo[i];
    const evType = String(ev?.type || "").trim().toLowerCase();

    if (decisionType === "option" && String(ev?.playerKey || "") === String(actor || "")) {
      if (evType === "gukjin_mode") {
        const mode = String(ev?.mode || "").trim().toLowerCase();
        if ((mode === "five" || mode === "junk") && candidateSet.has(mode)) return mode;
      }
      if (evType === "president_stop" && candidateSet.has("president_stop")) return "president_stop";
      if (evType === "president_hold" && candidateSet.has("president_hold")) return "president_hold";
    }

    if (evType !== "turn_end") continue;
    if (String(ev?.actor || "") !== String(actor || "")) continue;
    const actionType = String(ev?.action?.type || "").trim().toLowerCase();

    if (decisionType === "play" && actionType === "play") {
      const playedId = String(ev?.action?.card?.id || "").trim();
      const shakingCandidate = `${PLAY_SPECIAL_SHAKE_PREFIX}${playedId}`;
      if (playedId && shakingDeclared && candidateSet.has(shakingCandidate)) return shakingCandidate;
      if (playedId && candidateSet.has(playedId)) return playedId;
    }
    if (decisionType === "play" && actionType === "declare_bomb") {
      const bombMonth = Number(ev?.action?.month || 0);
      const bombCandidate = `${PLAY_SPECIAL_BOMB_PREFIX}${bombMonth}`;
      if (bombMonth >= 1 && candidateSet.has(bombCandidate)) return bombCandidate;
    }
    if (decisionType === "match" && actionType === "play") {
      const selectedBoardId = String(ev?.action?.selectedBoardCard?.id || "").trim();
      if (selectedBoardId && candidateSet.has(selectedBoardId)) return selectedBoardId;
    }
    break;
  }
  return null;
}

function inferPlayCandidateFromHandDiff(stateBefore, actor, candidates, stateAfter) {
  const beforeHand = Array.isArray(stateBefore?.players?.[actor]?.hand) ? stateBefore.players[actor].hand : [];
  const afterHand = Array.isArray(stateAfter?.players?.[actor]?.hand) ? stateAfter.players[actor].hand : [];
  const beforeIds = new Set(beforeHand.map((c) => String(c?.id || "")).filter((id) => id.length > 0));
  const afterIds = new Set(afterHand.map((c) => String(c?.id || "")).filter((id) => id.length > 0));
  const removed = [];
  for (const candidate of candidates || []) {
    const id = String(candidate || "").trim();
    if (!id || id.includes(":")) continue;
    if (beforeIds.has(id) && !afterIds.has(id)) removed.push(id);
  }
  if (removed.length === 1) return removed[0];
  return null;
}

function inferChosenCandidateFromTransition(stateBefore, actor, decisionType, candidates, stateAfter) {
  if (!stateAfter || !Array.isArray(candidates) || !candidates.length) return null;

  const kiboInferred = inferCandidateFromKiboTransition(
    stateBefore,
    actor,
    decisionType,
    candidates,
    stateAfter
  );
  if (kiboInferred) return normalizeDecisionCandidate(decisionType, kiboInferred);

  if (decisionType === "play") {
    const handDiffInferred = inferPlayCandidateFromHandDiff(stateBefore, actor, candidates, stateAfter);
    if (handDiffInferred) return normalizeDecisionCandidate(decisionType, handDiffInferred);
  }

  const target = detailedStateProgressKey(stateAfter);
  for (const candidate of candidates) {
    const simulated = applyDecisionAction(stateBefore, actor, decisionType, candidate);
    if (simulated && detailedStateProgressKey(simulated) === target) {
      return normalizeDecisionCandidate(decisionType, candidate);
    }
  }
  return null;
}

function collectTransitionEventCounts(stateBefore, stateAfter) {
  const beforeLen = Array.isArray(stateBefore?.kibo) ? stateBefore.kibo.length : 0;
  const afterKibo = Array.isArray(stateAfter?.kibo) ? stateAfter.kibo : [];
  const out = {
    bomb_declare_count: 0,
    shaking_declare_count: 0,
  };
  for (let i = Math.max(0, beforeLen); i < afterKibo.length; i += 1) {
    const type = String(afterKibo[i]?.type || "");
    if (type === "declare_bomb") out.bomb_declare_count += 1;
    else if (type === "shaking_declare") out.shaking_declare_count += 1;
    else if (type === "turn_end") {
      const actionType = String(afterKibo[i]?.action?.type || "");
      if (actionType === "declare_bomb") out.bomb_declare_count += 1;
    }
  }
  return out;
}

function isGoStopOpportunityDecision(decision) {
  if (!decision || decision.decisionType !== "option") return false;
  if (String(decision?.stateBefore?.phase || "") !== "go-stop") return false;
  const options = normalizeOptionCandidates(decision.candidates || []);
  return options.includes("go") && options.includes("stop");
}

function isPresidentOpportunityDecision(decision) {
  if (!decision || decision.decisionType !== "option") return false;
  if (String(decision?.stateBefore?.phase || "") !== "president-choice") return false;
  const options = normalizeOptionCandidates(decision.candidates || []);
  return options.includes("president_stop") && options.includes("president_hold");
}

function isGukjinOpportunityDecision(decision) {
  if (!decision || decision.decisionType !== "option") return false;
  if (String(decision?.stateBefore?.phase || "") !== "gukjin-choice") return false;
  const options = normalizeOptionCandidates(decision.candidates || []);
  return options.includes("five") && options.includes("junk");
}

function playSingleRound(initialState, seed, playerByActor, maxSteps, onDecision = null) {
  let state = initialState;
  let steps = 0;

  while (state.phase !== "resolution" && steps < maxSteps) {
    const actor = getActionPlayerKey(state);
    if (!actor) break;

    const before = detailedStateProgressKey(state);
    const pool = selectDecisionPool(state, actor);
    const decisionType = resolveDecisionType(pool);
    const candidates = decisionType ? legalCandidatesForDecision(pool, decisionType) : [];
    const playerSpec = playerByActor[actor];
    const policy = String(playerSpec?.label || "");
    const action = resolvePlayerAction(state, actor, playerSpec);
    const actionSource = String(action?.actionSource || "heuristic");
    const next = action?.next || state;

    if (!next || detailedStateProgressKey(next) === before) {
      throw new Error(
        `action resolution failed: seed=${seed}, step=${steps}, actor=${actor}, phase=${String(state?.phase || "")}, policy=${policy}, source=${actionSource}`
      );
    }

    if (typeof onDecision === "function" && decisionType && candidates.length > 0) {
      const chosenCandidate = inferChosenCandidateFromTransition(
        state,
        actor,
        decisionType,
        candidates,
        next
      );
      const transitionEvents = collectTransitionEventCounts(state, next);
      onDecision({
        stateBefore: state,
        stateAfter: next,
        actor,
        policy,
        decisionType,
        candidates,
        chosenCandidate,
        transitionEvents,
        actionSource,
        step: steps,
      });
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

function createSeatRecord() {
  return {
    games: 0,
    wins: 0,
    losses: 0,
    draws: 0,
    go_count_total: 0,
    go_game_count: 0,
    go_fail_count: 0,
    go_opportunity_count_total: 0,
    go_opportunity_game_count: 0,
    shaking_count_total: 0,
    shaking_game_count: 0,
    shaking_win_game_count: 0,
    shaking_opportunity_count_total: 0,
    shaking_opportunity_game_count: 0,
    bomb_count_total: 0,
    bomb_game_count: 0,
    bomb_win_game_count: 0,
    bomb_opportunity_count_total: 0,
    bomb_opportunity_game_count: 0,
    president_stop_total: 0,
    president_hold_total: 0,
    president_stop_win_total: 0,
    president_stop_loss_total: 0,
    president_hold_win_total: 0,
    president_hold_loss_total: 0,
    president_opportunity_count_total: 0,
    president_opportunity_game_count: 0,
    gukjin_five_total: 0,
    gukjin_five_mongbak_total: 0,
    gukjin_junk_total: 0,
    gukjin_junk_mongbak_total: 0,
    gukjin_opportunity_count_total: 0,
    gukjin_opportunity_game_count: 0,
    gold_deltas: [],
  };
}

function createRoundSpecialMetrics() {
  return {
    go_opportunity_count: 0,
    shaking_count: 0,
    shaking_opportunity_count: 0,
    bomb_count: 0,
    bomb_opportunity_count: 0,
    president_stop_count: 0,
    president_hold_count: 0,
    president_opportunity_count: 0,
    gukjin_five_count: 0,
    gukjin_five_mongbak_count: 0,
    gukjin_junk_count: 0,
    gukjin_junk_mongbak_count: 0,
    gukjin_opportunity_count: 0,
  };
}

function collectUniqueShakingOpportunitySignatures(state, actor) {
  if (state?.phase !== "playing" || state?.currentTurn !== actor) return [];
  const player = state?.players?.[actor];
  if (!player) return [];
  const declared = new Set(player.shakingDeclaredMonths || []);
  const counts = new Map();
  for (const card of player.hand || []) {
    if (!card || card.passCard) continue;
    const month = Number(card.month || 0);
    if (month < 1 || month > 12) continue;
    const current = counts.get(month) || [];
    current.push(String(card.id || ""));
    counts.set(month, current);
  }
  const signatures = [];
  for (const [month, ids] of counts.entries()) {
    if (ids.length < 3) continue;
    if (declared.has(month)) continue;
    if ((state.board || []).some((card) => Number(card?.month || 0) === month)) continue;
    signatures.push(`${month}:${ids.filter(Boolean).sort().join("|")}`);
  }
  return signatures;
}

function collectUniqueBombOpportunitySignatures(state, actor) {
  if (state?.phase !== "playing" || state?.currentTurn !== actor) return [];
  const player = state?.players?.[actor];
  if (!player) return [];
  const bombMonths = getDeclarableBombMonths(state, actor);
  if (!Array.isArray(bombMonths) || bombMonths.length === 0) return [];
  const signatures = [];
  for (const month of bombMonths) {
    const monthNum = Number(month || 0);
    const handIds = (player.hand || [])
      .filter((card) => card && !card.passCard && Number(card.month || 0) === monthNum)
      .map((card) => String(card.id || ""))
      .filter(Boolean)
      .sort();
    const boardIds = (state.board || [])
      .filter((card) => Number(card?.month || 0) === monthNum)
      .map((card) => String(card?.id || ""))
      .filter(Boolean)
      .sort();
    signatures.push(`${monthNum}:${handIds.join("|")}::${boardIds.join("|")}`);
  }
  return signatures;
}

function updateSeatRecord(record, winner, selfActor, oppActor, goldDelta, roundMetrics) {
  record.games += 1;
  if (winner === selfActor) record.wins += 1;
  else if (winner === oppActor) record.losses += 1;
  else record.draws += 1;

  const goCount = Math.max(0, Number(roundMetrics?.go_count || 0));
  const goOpportunityCount = Math.max(0, Number(roundMetrics?.go_opportunity_count || 0));
  const shakingCount = Math.max(0, Number(roundMetrics?.shaking_count || 0));
  const shakingOpportunityCount = Math.max(0, Number(roundMetrics?.shaking_opportunity_count || 0));
  const bombCount = Math.max(0, Number(roundMetrics?.bomb_count || 0));
  const bombOpportunityCount = Math.max(0, Number(roundMetrics?.bomb_opportunity_count || 0));
  const presidentStopCount = Math.max(0, Number(roundMetrics?.president_stop_count || 0));
  const presidentHoldCount = Math.max(0, Number(roundMetrics?.president_hold_count || 0));
  const presidentOpportunityCount = Math.max(0, Number(roundMetrics?.president_opportunity_count || 0));
  const gukjinFiveCount = Math.max(0, Number(roundMetrics?.gukjin_five_count || 0));
  const gukjinFiveMongBakCount = Math.max(0, Number(roundMetrics?.gukjin_five_mongbak_count || 0));
  const gukjinJunkCount = Math.max(0, Number(roundMetrics?.gukjin_junk_count || 0));
  const gukjinJunkMongBakCount = Math.max(0, Number(roundMetrics?.gukjin_junk_mongbak_count || 0));
  const gukjinOpportunityCount = Math.max(0, Number(roundMetrics?.gukjin_opportunity_count || 0));
  record.go_count_total += goCount;
  record.go_opportunity_count_total += goOpportunityCount;
  if (goOpportunityCount > 0) {
    record.go_opportunity_game_count += 1;
  }
  if (goCount > 0) {
    record.go_game_count += 1;
    if (winner !== selfActor) record.go_fail_count += 1;
  }
  record.shaking_count_total += shakingCount;
  record.shaking_opportunity_count_total += shakingOpportunityCount;
  if (shakingCount > 0) record.shaking_game_count += 1;
  if (shakingCount > 0 && winner === selfActor) record.shaking_win_game_count += 1;
  if (shakingOpportunityCount > 0) record.shaking_opportunity_game_count += 1;
  record.bomb_count_total += bombCount;
  record.bomb_opportunity_count_total += bombOpportunityCount;
  if (bombCount > 0) record.bomb_game_count += 1;
  if (bombCount > 0 && winner === selfActor) record.bomb_win_game_count += 1;
  if (bombOpportunityCount > 0) record.bomb_opportunity_game_count += 1;
  record.president_stop_total += presidentStopCount;
  record.president_hold_total += presidentHoldCount;
  if (presidentStopCount > 0) {
    if (winner === selfActor) record.president_stop_win_total += presidentStopCount;
    else if (winner === oppActor) record.president_stop_loss_total += presidentStopCount;
  }
  if (presidentHoldCount > 0) {
    if (winner === selfActor) record.president_hold_win_total += presidentHoldCount;
    else if (winner === oppActor) record.president_hold_loss_total += presidentHoldCount;
  }
  record.president_opportunity_count_total += presidentOpportunityCount;
  if (presidentOpportunityCount > 0) record.president_opportunity_game_count += 1;
  record.gukjin_five_total += gukjinFiveCount;
  record.gukjin_five_mongbak_total += gukjinFiveMongBakCount;
  record.gukjin_junk_total += gukjinJunkCount;
  record.gukjin_junk_mongbak_total += gukjinJunkMongBakCount;
  record.gukjin_opportunity_count_total += gukjinOpportunityCount;
  if (gukjinOpportunityCount > 0) record.gukjin_opportunity_game_count += 1;
  record.gold_deltas.push(Number(goldDelta || 0));
}

function finalizeSeatRecord(record) {
  const games = Number(record?.games || 0);
  const wins = Number(record?.wins || 0);
  const losses = Number(record?.losses || 0);
  const draws = Number(record?.draws || 0);
  const goCountTotal = Number(record?.go_count_total || 0);
  const goGameCount = Number(record?.go_game_count || 0);
  const goFailCount = Number(record?.go_fail_count || 0);
  const goOpportunityCountTotal = Number(record?.go_opportunity_count_total || 0);
  const goOpportunityGameCount = Number(record?.go_opportunity_game_count || 0);
  const shakingCountTotal = Number(record?.shaking_count_total || 0);
  const shakingGameCount = Number(record?.shaking_game_count || 0);
  const shakingWinGameCount = Number(record?.shaking_win_game_count || 0);
  const shakingOpportunityCountTotal = Number(record?.shaking_opportunity_count_total || 0);
  const shakingOpportunityGameCount = Number(record?.shaking_opportunity_game_count || 0);
  const bombCountTotal = Number(record?.bomb_count_total || 0);
  const bombGameCount = Number(record?.bomb_game_count || 0);
  const bombWinGameCount = Number(record?.bomb_win_game_count || 0);
  const bombOpportunityCountTotal = Number(record?.bomb_opportunity_count_total || 0);
  const bombOpportunityGameCount = Number(record?.bomb_opportunity_game_count || 0);
  const presidentStopTotal = Number(record?.president_stop_total || 0);
  const presidentHoldTotal = Number(record?.president_hold_total || 0);
  const presidentStopWinTotal = Number(record?.president_stop_win_total || 0);
  const presidentStopLossTotal = Number(record?.president_stop_loss_total || 0);
  const presidentHoldWinTotal = Number(record?.president_hold_win_total || 0);
  const presidentHoldLossTotal = Number(record?.president_hold_loss_total || 0);
  const presidentOpportunityCountTotal = Number(record?.president_opportunity_count_total || 0);
  const presidentOpportunityGameCount = Number(record?.president_opportunity_game_count || 0);
  const gukjinFiveTotal = Number(record?.gukjin_five_total || 0);
  const gukjinFiveMongBakTotal = Number(record?.gukjin_five_mongbak_total || 0);
  const gukjinJunkTotal = Number(record?.gukjin_junk_total || 0);
  const gukjinJunkMongBakTotal = Number(record?.gukjin_junk_mongbak_total || 0);
  const gukjinOpportunityCountTotal = Number(record?.gukjin_opportunity_count_total || 0);
  const gukjinOpportunityGameCount = Number(record?.gukjin_opportunity_game_count || 0);
  const deltas = Array.isArray(record?.gold_deltas) ? record.gold_deltas : [];
  const meanGoldDelta = deltas.length > 0 ? deltas.reduce((a, b) => a + b, 0) / deltas.length : 0;

  return {
    games,
    wins,
    losses,
    draws,
    win_rate: games > 0 ? wins / games : 0,
    loss_rate: games > 0 ? losses / games : 0,
    draw_rate: games > 0 ? draws / games : 0,
    go_count_total: goCountTotal,
    go_avg_per_game: games > 0 ? goCountTotal / games : 0,
    go_game_count: goGameCount,
    go_fail_count: goFailCount,
    go_success_count: Math.max(0, goGameCount - goFailCount),
    go_fail_rate: goGameCount > 0 ? goFailCount / goGameCount : 0,
    go_opportunity_count_total: goOpportunityCountTotal,
    go_opportunity_game_count: goOpportunityGameCount,
    go_opportunity_rate: games > 0 ? goOpportunityGameCount / games : 0,
    go_take_rate: goOpportunityCountTotal > 0 ? goCountTotal / goOpportunityCountTotal : 0,
    shaking_count_total: shakingCountTotal,
    shaking_game_count: shakingGameCount,
    shaking_win_game_count: shakingWinGameCount,
    shaking_opportunity_count_total: shakingOpportunityCountTotal,
    shaking_opportunity_game_count: shakingOpportunityGameCount,
    shaking_opportunity_rate: games > 0 ? shakingOpportunityGameCount / games : 0,
    shaking_take_rate: shakingOpportunityCountTotal > 0 ? shakingCountTotal / shakingOpportunityCountTotal : 0,
    bomb_count_total: bombCountTotal,
    bomb_game_count: bombGameCount,
    bomb_win_game_count: bombWinGameCount,
    bomb_opportunity_count_total: bombOpportunityCountTotal,
    bomb_opportunity_game_count: bombOpportunityGameCount,
    bomb_opportunity_rate: games > 0 ? bombOpportunityGameCount / games : 0,
    bomb_take_rate: bombOpportunityCountTotal > 0 ? bombCountTotal / bombOpportunityCountTotal : 0,
    president_stop_total: presidentStopTotal,
    president_hold_total: presidentHoldTotal,
    president_stop_win_total: presidentStopWinTotal,
    president_stop_loss_total: presidentStopLossTotal,
    president_hold_win_total: presidentHoldWinTotal,
    president_hold_loss_total: presidentHoldLossTotal,
    president_opportunity_count_total: presidentOpportunityCountTotal,
    president_opportunity_game_count: presidentOpportunityGameCount,
    president_stop_rate: presidentOpportunityCountTotal > 0 ? presidentStopTotal / presidentOpportunityCountTotal : 0,
    president_stop_win_rate: presidentStopTotal > 0 ? presidentStopWinTotal / presidentStopTotal : 0,
    president_hold_win_rate: presidentHoldTotal > 0 ? presidentHoldWinTotal / presidentHoldTotal : 0,
    gukjin_five_total: gukjinFiveTotal,
    gukjin_five_mongbak_total: gukjinFiveMongBakTotal,
    gukjin_junk_total: gukjinJunkTotal,
    gukjin_junk_mongbak_total: gukjinJunkMongBakTotal,
    gukjin_opportunity_count_total: gukjinOpportunityCountTotal,
    gukjin_opportunity_game_count: gukjinOpportunityGameCount,
    gukjin_five_rate: gukjinOpportunityCountTotal > 0 ? gukjinFiveTotal / gukjinOpportunityCountTotal : 0,
    gukjin_five_mongbak_rate: gukjinFiveTotal > 0 ? gukjinFiveMongBakTotal / gukjinFiveTotal : 0,
    gukjin_junk_mongbak_rate: gukjinJunkTotal > 0 ? gukjinJunkMongBakTotal / gukjinJunkTotal : 0,
    mean_gold_delta: meanGoldDelta,
    p10_gold_delta: quantile(deltas, 0.1),
    p50_gold_delta: quantile(deltas, 0.5),
    p90_gold_delta: quantile(deltas, 0.9),
  };
}

function buildSeatSplitSummary(firstRecord, secondRecord) {
  const combined = createSeatRecord();
  combined.games = Number(firstRecord.games || 0) + Number(secondRecord.games || 0);
  combined.wins = Number(firstRecord.wins || 0) + Number(secondRecord.wins || 0);
  combined.losses = Number(firstRecord.losses || 0) + Number(secondRecord.losses || 0);
  combined.draws = Number(firstRecord.draws || 0) + Number(secondRecord.draws || 0);
  combined.go_count_total =
    Number(firstRecord.go_count_total || 0) + Number(secondRecord.go_count_total || 0);
  combined.go_game_count =
    Number(firstRecord.go_game_count || 0) + Number(secondRecord.go_game_count || 0);
  combined.go_fail_count =
    Number(firstRecord.go_fail_count || 0) + Number(secondRecord.go_fail_count || 0);
  combined.go_opportunity_count_total =
    Number(firstRecord.go_opportunity_count_total || 0) +
    Number(secondRecord.go_opportunity_count_total || 0);
  combined.go_opportunity_game_count =
    Number(firstRecord.go_opportunity_game_count || 0) +
    Number(secondRecord.go_opportunity_game_count || 0);
  combined.shaking_count_total =
    Number(firstRecord.shaking_count_total || 0) + Number(secondRecord.shaking_count_total || 0);
  combined.shaking_game_count =
    Number(firstRecord.shaking_game_count || 0) + Number(secondRecord.shaking_game_count || 0);
  combined.shaking_win_game_count =
    Number(firstRecord.shaking_win_game_count || 0) + Number(secondRecord.shaking_win_game_count || 0);
  combined.shaking_opportunity_count_total =
    Number(firstRecord.shaking_opportunity_count_total || 0) +
    Number(secondRecord.shaking_opportunity_count_total || 0);
  combined.shaking_opportunity_game_count =
    Number(firstRecord.shaking_opportunity_game_count || 0) +
    Number(secondRecord.shaking_opportunity_game_count || 0);
  combined.bomb_count_total =
    Number(firstRecord.bomb_count_total || 0) + Number(secondRecord.bomb_count_total || 0);
  combined.bomb_game_count =
    Number(firstRecord.bomb_game_count || 0) + Number(secondRecord.bomb_game_count || 0);
  combined.bomb_win_game_count =
    Number(firstRecord.bomb_win_game_count || 0) + Number(secondRecord.bomb_win_game_count || 0);
  combined.bomb_opportunity_count_total =
    Number(firstRecord.bomb_opportunity_count_total || 0) +
    Number(secondRecord.bomb_opportunity_count_total || 0);
  combined.bomb_opportunity_game_count =
    Number(firstRecord.bomb_opportunity_game_count || 0) +
    Number(secondRecord.bomb_opportunity_game_count || 0);
  combined.president_stop_total =
    Number(firstRecord.president_stop_total || 0) + Number(secondRecord.president_stop_total || 0);
  combined.president_hold_total =
    Number(firstRecord.president_hold_total || 0) + Number(secondRecord.president_hold_total || 0);
  combined.president_stop_win_total =
    Number(firstRecord.president_stop_win_total || 0) + Number(secondRecord.president_stop_win_total || 0);
  combined.president_stop_loss_total =
    Number(firstRecord.president_stop_loss_total || 0) + Number(secondRecord.president_stop_loss_total || 0);
  combined.president_hold_win_total =
    Number(firstRecord.president_hold_win_total || 0) + Number(secondRecord.president_hold_win_total || 0);
  combined.president_hold_loss_total =
    Number(firstRecord.president_hold_loss_total || 0) + Number(secondRecord.president_hold_loss_total || 0);
  combined.president_opportunity_count_total =
    Number(firstRecord.president_opportunity_count_total || 0) +
    Number(secondRecord.president_opportunity_count_total || 0);
  combined.president_opportunity_game_count =
    Number(firstRecord.president_opportunity_game_count || 0) +
    Number(secondRecord.president_opportunity_game_count || 0);
  combined.gukjin_five_total =
    Number(firstRecord.gukjin_five_total || 0) + Number(secondRecord.gukjin_five_total || 0);
  combined.gukjin_five_mongbak_total =
    Number(firstRecord.gukjin_five_mongbak_total || 0) +
    Number(secondRecord.gukjin_five_mongbak_total || 0);
  combined.gukjin_junk_total =
    Number(firstRecord.gukjin_junk_total || 0) + Number(secondRecord.gukjin_junk_total || 0);
  combined.gukjin_junk_mongbak_total =
    Number(firstRecord.gukjin_junk_mongbak_total || 0) +
    Number(secondRecord.gukjin_junk_mongbak_total || 0);
  combined.gukjin_opportunity_count_total =
    Number(firstRecord.gukjin_opportunity_count_total || 0) +
    Number(secondRecord.gukjin_opportunity_count_total || 0);
  combined.gukjin_opportunity_game_count =
    Number(firstRecord.gukjin_opportunity_game_count || 0) +
    Number(secondRecord.gukjin_opportunity_game_count || 0);
  combined.gold_deltas = [
    ...(Array.isArray(firstRecord.gold_deltas) ? firstRecord.gold_deltas : []),
    ...(Array.isArray(secondRecord.gold_deltas) ? secondRecord.gold_deltas : []),
  ];

  return {
    when_first: finalizeSeatRecord(firstRecord),
    when_second: finalizeSeatRecord(secondRecord),
    combined: finalizeSeatRecord(combined),
  };
}

function buildConsoleSummary(report) {
  const seatAFirst = report?.seat_split_a?.when_first || {};
  const seatASecond = report?.seat_split_a?.when_second || {};
  const seatBFirst = report?.seat_split_b?.when_first || {};
  const seatBSecond = report?.seat_split_b?.when_second || {};
  const bankrupt = report?.bankrupt || { a_bankrupt_count: 0, b_bankrupt_count: 0 };

  return {
    games: Number(report?.games || 0),
    human: String(report?.human || ""),
    ai: String(report?.ai || ""),
    wins_a: Number(report?.wins_a || 0),
    losses_a: Number(report?.losses_a || 0),
    wins_b: Number(report?.wins_b || 0),
    losses_b: Number(report?.losses_b || 0),
    draws: Number(report?.draws || 0),
    win_rate_a: Number(report?.win_rate_a || 0),
    win_rate_b: Number(report?.win_rate_b || 0),
    mean_gold_delta_a: Number(report?.mean_gold_delta_a || 0),
    p10_gold_delta_a: Number(report?.p10_gold_delta_a || 0),
    p50_gold_delta_a: Number(report?.p50_gold_delta_a || 0),
    p90_gold_delta_a: Number(report?.p90_gold_delta_a || 0),
    go_count_a: Number(report?.go_count_a || 0),
    go_count_b: Number(report?.go_count_b || 0),
    go_games_a: Number(report?.go_games_a || 0),
    go_games_b: Number(report?.go_games_b || 0),
    go_fail_count_a: Number(report?.go_fail_count_a || 0),
    go_fail_count_b: Number(report?.go_fail_count_b || 0),
    go_fail_rate_a: Number(report?.go_fail_rate_a || 0),
    go_fail_rate_b: Number(report?.go_fail_rate_b || 0),
    go_opportunity_count_a: Number(report?.go_opportunity_count_a || 0),
    go_opportunity_count_b: Number(report?.go_opportunity_count_b || 0),
    go_opportunity_games_a: Number(report?.go_opportunity_games_a || 0),
    go_opportunity_games_b: Number(report?.go_opportunity_games_b || 0),
    go_opportunity_rate_a: Number(report?.go_opportunity_rate_a || 0),
    go_opportunity_rate_b: Number(report?.go_opportunity_rate_b || 0),
    go_take_rate_a: Number(report?.go_take_rate_a || 0),
    go_take_rate_b: Number(report?.go_take_rate_b || 0),
    shaking_count_a: Number(report?.shaking_count_a || 0),
    shaking_count_b: Number(report?.shaking_count_b || 0),
    shaking_games_a: Number(report?.shaking_games_a || 0),
    shaking_games_b: Number(report?.shaking_games_b || 0),
    shaking_win_a: Number(report?.shaking_win_a || 0),
    shaking_win_b: Number(report?.shaking_win_b || 0),
    shaking_opportunity_count_a: Number(report?.shaking_opportunity_count_a || 0),
    shaking_opportunity_count_b: Number(report?.shaking_opportunity_count_b || 0),
    shaking_opportunity_games_a: Number(report?.shaking_opportunity_games_a || 0),
    shaking_opportunity_games_b: Number(report?.shaking_opportunity_games_b || 0),
    shaking_take_rate_a: Number(report?.shaking_take_rate_a || 0),
    shaking_take_rate_b: Number(report?.shaking_take_rate_b || 0),
    bomb_count_a: Number(report?.bomb_count_a || 0),
    bomb_count_b: Number(report?.bomb_count_b || 0),
    bomb_games_a: Number(report?.bomb_games_a || 0),
    bomb_games_b: Number(report?.bomb_games_b || 0),
    bomb_win_a: Number(report?.bomb_win_a || 0),
    bomb_win_b: Number(report?.bomb_win_b || 0),
    bomb_opportunity_count_a: Number(report?.bomb_opportunity_count_a || 0),
    bomb_opportunity_count_b: Number(report?.bomb_opportunity_count_b || 0),
    bomb_opportunity_games_a: Number(report?.bomb_opportunity_games_a || 0),
    bomb_opportunity_games_b: Number(report?.bomb_opportunity_games_b || 0),
    bomb_take_rate_a: Number(report?.bomb_take_rate_a || 0),
    bomb_take_rate_b: Number(report?.bomb_take_rate_b || 0),
    president_stop_a: Number(report?.president_stop_a || 0),
    president_stop_b: Number(report?.president_stop_b || 0),
    president_hold_a: Number(report?.president_hold_a || 0),
    president_hold_b: Number(report?.president_hold_b || 0),
    president_hold_win_a: Number(report?.president_hold_win_a || 0),
    president_hold_win_b: Number(report?.president_hold_win_b || 0),
    president_opportunity_count_a: Number(report?.president_opportunity_count_a || 0),
    president_opportunity_count_b: Number(report?.president_opportunity_count_b || 0),
    gukjin_five_a: Number(report?.gukjin_five_a || 0),
    gukjin_five_b: Number(report?.gukjin_five_b || 0),
    gukjin_five_mongbak_a: Number(report?.gukjin_five_mongbak_a || 0),
    gukjin_five_mongbak_b: Number(report?.gukjin_five_mongbak_b || 0),
    gukjin_junk_a: Number(report?.gukjin_junk_a || 0),
    gukjin_junk_b: Number(report?.gukjin_junk_b || 0),
    gukjin_junk_mongbak_a: Number(report?.gukjin_junk_mongbak_a || 0),
    gukjin_junk_mongbak_b: Number(report?.gukjin_junk_mongbak_b || 0),
    gukjin_opportunity_count_a: Number(report?.gukjin_opportunity_count_a || 0),
    gukjin_opportunity_count_b: Number(report?.gukjin_opportunity_count_b || 0),
    gukjin_five_rate_a: Number(report?.gukjin_five_rate_a || 0),
    gukjin_five_rate_b: Number(report?.gukjin_five_rate_b || 0),
    gukjin_five_mongbak_rate_a: Number(report?.gukjin_five_mongbak_rate_a || 0),
    gukjin_five_mongbak_rate_b: Number(report?.gukjin_five_mongbak_rate_b || 0),
    gukjin_junk_mongbak_rate_a: Number(report?.gukjin_junk_mongbak_rate_a || 0),
    gukjin_junk_mongbak_rate_b: Number(report?.gukjin_junk_mongbak_rate_b || 0),
    seat_split_a: {
      when_first: {
        win_rate: Number(seatAFirst.win_rate || 0),
        mean_gold_delta: Number(seatAFirst.mean_gold_delta || 0),
      },
      when_second: {
        win_rate: Number(seatASecond.win_rate || 0),
        mean_gold_delta: Number(seatASecond.mean_gold_delta || 0),
      },
    },
    seat_split_b: {
      when_first: {
        win_rate: Number(seatBFirst.win_rate || 0),
        mean_gold_delta: Number(seatBFirst.mean_gold_delta || 0),
      },
      when_second: {
        win_rate: Number(seatBSecond.win_rate || 0),
        mean_gold_delta: Number(seatBSecond.mean_gold_delta || 0),
      },
    },
    bankrupt,
    result_out: String(report?.result_out || ""),
  };
}

function formatConsoleSummaryText(summary) {
  const fmtRate = (value) => {
    const n = Number(value || 0);
    return Number.isFinite(n) ? n.toFixed(3) : "0.000";
  };
  const aFirst = summary?.seat_split_a?.when_first || {};
  const aSecond = summary?.seat_split_a?.when_second || {};
  const bFirst = summary?.seat_split_b?.when_first || {};
  const bSecond = summary?.seat_split_b?.when_second || {};
  const bankrupt = summary?.bankrupt || { a_bankrupt_count: 0, b_bankrupt_count: 0 };

  const lines = [
    "",
    `=== Model Duel (${summary.human} vs ${summary.ai}, games=${summary.games}) ===`,
    `Win/Loss/Draw(A):  ${summary.wins_a} / ${summary.losses_a} / ${summary.draws}  (WR=${fmtRate(summary.win_rate_a)})`,
    `Win/Loss/Draw(B):  ${summary.wins_b} / ${summary.losses_b} / ${summary.draws}  (WR=${fmtRate(summary.win_rate_b)})`,
    `Seat A first:      WR=${fmtRate(aFirst.win_rate)}, mean_gold_delta=${aFirst.mean_gold_delta}`,
    `Seat A second:     WR=${fmtRate(aSecond.win_rate)}, mean_gold_delta=${aSecond.mean_gold_delta}`,
    `Seat B first:      WR=${fmtRate(bFirst.win_rate)}, mean_gold_delta=${bFirst.mean_gold_delta}`,
    `Seat B second:     WR=${fmtRate(bSecond.win_rate)}, mean_gold_delta=${bSecond.mean_gold_delta}`,
    `Gold delta(A):     mean=${summary.mean_gold_delta_a}, p10=${summary.p10_gold_delta_a}, p50=${summary.p50_gold_delta_a}, p90=${summary.p90_gold_delta_a}`,
    `GO A:              games=${summary.go_games_a}, count=${summary.go_count_a}, fail=${summary.go_fail_count_a}, fail_rate=${fmtRate(summary.go_fail_rate_a)}`,
    `GO B:              games=${summary.go_games_b}, count=${summary.go_count_b}, fail=${summary.go_fail_count_b}, fail_rate=${fmtRate(summary.go_fail_rate_b)}`,
    `GO Opp A:          opp_games=${summary.go_opportunity_games_a}, opp_turns=${summary.go_opportunity_count_a}, opp_rate=${fmtRate(summary.go_opportunity_rate_a)}, take_rate=${fmtRate(summary.go_take_rate_a)}`,
    `GO Opp B:          opp_games=${summary.go_opportunity_games_b}, opp_turns=${summary.go_opportunity_count_b}, opp_rate=${fmtRate(summary.go_opportunity_rate_b)}, take_rate=${fmtRate(summary.go_take_rate_b)}`,
    `Shake A:           opp_games=${summary.shaking_opportunity_games_a}, opp_unique=${summary.shaking_opportunity_count_a}, games=${summary.shaking_games_a}, count=${summary.shaking_count_a}, win=${summary.shaking_win_a}, take_rate=${fmtRate(summary.shaking_take_rate_a)}`,
    `Shake B:           opp_games=${summary.shaking_opportunity_games_b}, opp_unique=${summary.shaking_opportunity_count_b}, games=${summary.shaking_games_b}, count=${summary.shaking_count_b}, win=${summary.shaking_win_b}, take_rate=${fmtRate(summary.shaking_take_rate_b)}`,
    `Bomb A:            opp_games=${summary.bomb_opportunity_games_a}, opp_unique=${summary.bomb_opportunity_count_a}, games=${summary.bomb_games_a}, count=${summary.bomb_count_a}, win=${summary.bomb_win_a}, take_rate=${fmtRate(summary.bomb_take_rate_a)}`,
    `Bomb B:            opp_games=${summary.bomb_opportunity_games_b}, opp_unique=${summary.bomb_opportunity_count_b}, games=${summary.bomb_games_b}, count=${summary.bomb_count_b}, win=${summary.bomb_win_b}, take_rate=${fmtRate(summary.bomb_take_rate_b)}`,
    `President A:       opp_total=${summary.president_opportunity_count_a}, hold=${summary.president_hold_a}, hold_win=${summary.president_hold_win_a}, stop=${summary.president_stop_a}`,
    `President B:       opp_total=${summary.president_opportunity_count_b}, hold=${summary.president_hold_b}, hold_win=${summary.president_hold_win_b}, stop=${summary.president_stop_b}`,
    `Gukjin A:          opp_total=${summary.gukjin_opportunity_count_a}, five=${summary.gukjin_five_a}, junk=${summary.gukjin_junk_a}, five_rate=${fmtRate(summary.gukjin_five_rate_a)}, five_mongbak=${summary.gukjin_five_mongbak_a}(${fmtRate(summary.gukjin_five_mongbak_rate_a)}), junk_mongbak=${summary.gukjin_junk_mongbak_a}(${fmtRate(summary.gukjin_junk_mongbak_rate_a)})`,
    `Gukjin B:          opp_total=${summary.gukjin_opportunity_count_b}, five=${summary.gukjin_five_b}, junk=${summary.gukjin_junk_b}, five_rate=${fmtRate(summary.gukjin_five_rate_b)}, five_mongbak=${summary.gukjin_five_mongbak_b}(${fmtRate(summary.gukjin_five_mongbak_rate_b)}), junk_mongbak=${summary.gukjin_junk_mongbak_b}(${fmtRate(summary.gukjin_junk_mongbak_rate_b)})`,
    `Bankrupt:          A=${bankrupt.a_bankrupt_count}, B=${bankrupt.b_bankrupt_count}`,
    `Result file:       ${summary.result_out || ""}`,
    "===========================================================",
    "",
  ];
  return `${lines.join("\n")}\n`;
}

export function runModelDuelCli(argv = process.argv.slice(2)) {
  const evalStartMs = Date.now();
  const opts = parseArgs(argv);
  if (opts?.help) {
    process.stdout.write(`${usageText()}\\n`);
    return;
  }

  const humanPlayer = resolvePlayerSpec(opts.humanSpecRaw, "human");
  const aiPlayer = resolvePlayerSpec(opts.aiSpecRaw, "ai");

  if (!opts.resultOut) {
    const outDir = buildAutoOutputDir(humanPlayer.label, aiPlayer.label);
    opts.resultOut = buildAutoArtifactPath(outDir, opts.seed, "result.json");
  }
  else {
    opts.resultOut = toAbsoluteRepoPath(opts.resultOut);
  }

  const actorA = "human";
  const actorB = "ai";
  const playerByActor = {
    [actorA]: humanPlayer,
    [actorB]: aiPlayer,
  };

  let winsA = 0;
  let winsB = 0;
  let draws = 0;
  const goldDeltasA = [];
  const bankrupt = { a_bankrupt_count: 0, b_bankrupt_count: 0 };
  const firstTurnCounts = { human: 0, ai: 0 };
  const seatSplitA = { first: createSeatRecord(), second: createSeatRecord() };
  const seatSplitB = { first: createSeatRecord(), second: createSeatRecord() };
  const seriesSession = { roundsPlayed: 0, previousEndState: null };

  for (let gi = 0; gi < opts.games; gi += 1) {
    const firstTurnKey = resolveFirstTurnKey(opts, gi);
    firstTurnCounts[firstTurnKey] += 1;
    const seed = `${opts.seed}|g=${gi}|first=${firstTurnKey}|sr=${seriesSession.roundsPlayed}`;

    const roundStart = opts.continuousSeries
      ? seriesSession.previousEndState
        ? continueRound(seriesSession.previousEndState, seed, firstTurnKey)
        : startRound(seed, firstTurnKey)
      : startRound(seed, firstTurnKey);
    const roundGoOpportunity = { human: 0, ai: 0 };
    const roundSpecial = {
      human: createRoundSpecialMetrics(),
      ai: createRoundSpecialMetrics(),
    };
    const roundUniqueShakingOpportunity = {
      human: new Set(),
      ai: new Set(),
    };
    const roundUniqueBombOpportunity = {
      human: new Set(),
      ai: new Set(),
    };

    const beforeDiffA = goldDiffByActor(roundStart, actorA);
    const endState = playSingleRound(
      roundStart,
      seed,
      playerByActor,
      Math.max(20, Math.floor(opts.maxSteps)),
      (decision) => {
        if (decision?.actor === "human" || decision?.actor === "ai") {
          const actorMetrics = roundSpecial[decision.actor];
          if (decision?.stateBefore?.phase === "playing") {
            for (const signature of collectUniqueShakingOpportunitySignatures(
              decision.stateBefore,
              decision.actor
            )) {
              const seen = roundUniqueShakingOpportunity[decision.actor];
              if (seen.has(signature)) continue;
              seen.add(signature);
              actorMetrics.shaking_opportunity_count += 1;
            }
            for (const signature of collectUniqueBombOpportunitySignatures(
              decision.stateBefore,
              decision.actor
            )) {
              const seen = roundUniqueBombOpportunity[decision.actor];
              if (seen.has(signature)) continue;
              seen.add(signature);
              actorMetrics.bomb_opportunity_count += 1;
            }
          }
          actorMetrics.shaking_count += Math.max(0, Number(decision?.transitionEvents?.shaking_declare_count || 0));
          actorMetrics.bomb_count += Math.max(0, Number(decision?.transitionEvents?.bomb_declare_count || 0));
          if (isPresidentOpportunityDecision(decision)) {
            actorMetrics.president_opportunity_count += 1;
            if (decision.chosenCandidate === "president_stop") actorMetrics.president_stop_count += 1;
            else if (decision.chosenCandidate === "president_hold") actorMetrics.president_hold_count += 1;
          }
          if (isGukjinOpportunityDecision(decision)) {
            actorMetrics.gukjin_opportunity_count += 1;
            if (decision.chosenCandidate === "five") actorMetrics.gukjin_five_count += 1;
            else if (decision.chosenCandidate === "junk") actorMetrics.gukjin_junk_count += 1;
          }
        }
        if (isGoStopOpportunityDecision(decision)) {
          if (decision.actor === "human" || decision.actor === "ai") {
            roundGoOpportunity[decision.actor] += 1;
          }
        }
      }
    );
    const afterDiffA = goldDiffByActor(endState, actorA);
    const roundDeltaA = afterDiffA - beforeDiffA;
    goldDeltasA.push(roundDeltaA);

    const actorAMongBak = Boolean(endState?.result?.[actorA]?.bak?.mongBak);
    const actorBMongBak = Boolean(endState?.result?.[actorB]?.bak?.mongBak);
    if (roundSpecial[actorA].gukjin_five_count > 0 && actorAMongBak) {
      roundSpecial[actorA].gukjin_five_mongbak_count = 1;
    }
    if (roundSpecial[actorA].gukjin_junk_count > 0 && actorAMongBak) {
      roundSpecial[actorA].gukjin_junk_mongbak_count = 1;
    }
    if (roundSpecial[actorB].gukjin_five_count > 0 && actorBMongBak) {
      roundSpecial[actorB].gukjin_five_mongbak_count = 1;
    }
    if (roundSpecial[actorB].gukjin_junk_count > 0 && actorBMongBak) {
      roundSpecial[actorB].gukjin_junk_mongbak_count = 1;
    }

    const goldA = Number(endState?.players?.[actorA]?.gold || 0);
    const goldB = Number(endState?.players?.[actorB]?.gold || 0);
    const goCountA = Math.max(0, Number(endState?.players?.[actorA]?.goCount || 0));
    const goCountB = Math.max(0, Number(endState?.players?.[actorB]?.goCount || 0));
    const goOpportunityCountA = Math.max(0, Number(roundGoOpportunity[actorA] || 0));
    const goOpportunityCountB = Math.max(0, Number(roundGoOpportunity[actorB] || 0));
    const roundMetricsA = {
      ...roundSpecial[actorA],
      go_count: goCountA,
      go_opportunity_count: goOpportunityCountA,
    };
    const roundMetricsB = {
      ...roundSpecial[actorB],
      go_count: goCountB,
      go_opportunity_count: goOpportunityCountB,
    };
    if (goldA <= 0) bankrupt.a_bankrupt_count += 1;
    if (goldB <= 0) bankrupt.b_bankrupt_count += 1;

    if (opts.continuousSeries) seriesSession.previousEndState = endState;
    seriesSession.roundsPlayed += 1;

    const winner = String(endState?.result?.winner || "").trim();
    if (winner === actorA) winsA += 1;
    else if (winner === actorB) winsB += 1;
    else draws += 1;

    const seatAKey = firstTurnKey === actorA ? "first" : "second";
    const seatBKey = firstTurnKey === actorB ? "first" : "second";
    updateSeatRecord(seatSplitA[seatAKey], winner, actorA, actorB, roundDeltaA, roundMetricsA);
    updateSeatRecord(seatSplitB[seatBKey], winner, actorB, actorA, -roundDeltaA, roundMetricsB);
  }

  const games = opts.games;
  const winRateA = winsA / games;
  const winRateB = winsB / games;
  const drawRate = draws / games;
  const meanGoldDeltaA =
    goldDeltasA.length > 0 ? goldDeltasA.reduce((a, b) => a + b, 0) / goldDeltasA.length : 0;
  const splitSummaryA = buildSeatSplitSummary(seatSplitA.first, seatSplitA.second);
  const splitSummaryB = buildSeatSplitSummary(seatSplitB.first, seatSplitB.second);

  const summary = {
    games,
    actor_human: actorA,
    actor_ai: actorB,
    human: humanPlayer.label,
    ai: aiPlayer.label,
    player_human: {
      input: humanPlayer.input,
      kind: humanPlayer.kind,
      key: humanPlayer.key,
      model_path: toReportPath(humanPlayer.modelPath),
    },
    player_ai: {
      input: aiPlayer.input,
      kind: aiPlayer.kind,
      key: aiPlayer.key,
      model_path: toReportPath(aiPlayer.modelPath),
    },
    first_turn_policy: opts.firstTurnPolicy,
    fixed_first_turn: opts.firstTurnPolicy === "fixed" ? opts.fixedFirstTurn : null,
    first_turn_counts: firstTurnCounts,
    continuous_series: !!opts.continuousSeries,
    result_out: toReportPath(opts.resultOut),
    bankrupt,
    session_rounds: {
      total_rounds: seriesSession.roundsPlayed,
    },
    wins_a: winsA,
    losses_a: winsB,
    wins_b: winsB,
    losses_b: winsA,
    draws,
    win_rate_a: winRateA,
    win_rate_b: winRateB,
    draw_rate: drawRate,
    go_count_a: splitSummaryA.combined.go_count_total,
    go_count_b: splitSummaryB.combined.go_count_total,
    go_games_a: splitSummaryA.combined.go_game_count,
    go_games_b: splitSummaryB.combined.go_game_count,
    go_fail_count_a: splitSummaryA.combined.go_fail_count,
    go_fail_count_b: splitSummaryB.combined.go_fail_count,
    go_fail_rate_a: splitSummaryA.combined.go_fail_rate,
    go_fail_rate_b: splitSummaryB.combined.go_fail_rate,
    go_opportunity_count_a: splitSummaryA.combined.go_opportunity_count_total,
    go_opportunity_count_b: splitSummaryB.combined.go_opportunity_count_total,
    go_opportunity_games_a: splitSummaryA.combined.go_opportunity_game_count,
    go_opportunity_games_b: splitSummaryB.combined.go_opportunity_game_count,
    go_opportunity_rate_a: splitSummaryA.combined.go_opportunity_rate,
    go_opportunity_rate_b: splitSummaryB.combined.go_opportunity_rate,
    go_take_rate_a: splitSummaryA.combined.go_take_rate,
    go_take_rate_b: splitSummaryB.combined.go_take_rate,
    shaking_count_a: splitSummaryA.combined.shaking_count_total,
    shaking_count_b: splitSummaryB.combined.shaking_count_total,
    shaking_games_a: splitSummaryA.combined.shaking_game_count,
    shaking_games_b: splitSummaryB.combined.shaking_game_count,
    shaking_win_a: splitSummaryA.combined.shaking_win_game_count,
    shaking_win_b: splitSummaryB.combined.shaking_win_game_count,
    shaking_opportunity_count_a: splitSummaryA.combined.shaking_opportunity_count_total,
    shaking_opportunity_count_b: splitSummaryB.combined.shaking_opportunity_count_total,
    shaking_opportunity_games_a: splitSummaryA.combined.shaking_opportunity_game_count,
    shaking_opportunity_games_b: splitSummaryB.combined.shaking_opportunity_game_count,
    shaking_take_rate_a: splitSummaryA.combined.shaking_take_rate,
    shaking_take_rate_b: splitSummaryB.combined.shaking_take_rate,
    bomb_count_a: splitSummaryA.combined.bomb_count_total,
    bomb_count_b: splitSummaryB.combined.bomb_count_total,
    bomb_games_a: splitSummaryA.combined.bomb_game_count,
    bomb_games_b: splitSummaryB.combined.bomb_game_count,
    bomb_win_a: splitSummaryA.combined.bomb_win_game_count,
    bomb_win_b: splitSummaryB.combined.bomb_win_game_count,
    bomb_opportunity_count_a: splitSummaryA.combined.bomb_opportunity_count_total,
    bomb_opportunity_count_b: splitSummaryB.combined.bomb_opportunity_count_total,
    bomb_opportunity_games_a: splitSummaryA.combined.bomb_opportunity_game_count,
    bomb_opportunity_games_b: splitSummaryB.combined.bomb_opportunity_game_count,
    bomb_take_rate_a: splitSummaryA.combined.bomb_take_rate,
    bomb_take_rate_b: splitSummaryB.combined.bomb_take_rate,
    president_stop_a: splitSummaryA.combined.president_stop_total,
    president_stop_b: splitSummaryB.combined.president_stop_total,
    president_hold_a: splitSummaryA.combined.president_hold_total,
    president_hold_b: splitSummaryB.combined.president_hold_total,
    president_stop_win_a: splitSummaryA.combined.president_stop_win_total,
    president_stop_win_b: splitSummaryB.combined.president_stop_win_total,
    president_stop_loss_a: splitSummaryA.combined.president_stop_loss_total,
    president_stop_loss_b: splitSummaryB.combined.president_stop_loss_total,
    president_hold_win_a: splitSummaryA.combined.president_hold_win_total,
    president_hold_win_b: splitSummaryB.combined.president_hold_win_total,
    president_hold_loss_a: splitSummaryA.combined.president_hold_loss_total,
    president_hold_loss_b: splitSummaryB.combined.president_hold_loss_total,
    president_opportunity_count_a: splitSummaryA.combined.president_opportunity_count_total,
    president_opportunity_count_b: splitSummaryB.combined.president_opportunity_count_total,
    president_stop_rate_a: splitSummaryA.combined.president_stop_rate,
    president_stop_rate_b: splitSummaryB.combined.president_stop_rate,
    president_stop_win_rate_a: splitSummaryA.combined.president_stop_win_rate,
    president_stop_win_rate_b: splitSummaryB.combined.president_stop_win_rate,
    president_hold_win_rate_a: splitSummaryA.combined.president_hold_win_rate,
    president_hold_win_rate_b: splitSummaryB.combined.president_hold_win_rate,
    gukjin_five_a: splitSummaryA.combined.gukjin_five_total,
    gukjin_five_b: splitSummaryB.combined.gukjin_five_total,
    gukjin_five_mongbak_a: splitSummaryA.combined.gukjin_five_mongbak_total,
    gukjin_five_mongbak_b: splitSummaryB.combined.gukjin_five_mongbak_total,
    gukjin_junk_a: splitSummaryA.combined.gukjin_junk_total,
    gukjin_junk_b: splitSummaryB.combined.gukjin_junk_total,
    gukjin_junk_mongbak_a: splitSummaryA.combined.gukjin_junk_mongbak_total,
    gukjin_junk_mongbak_b: splitSummaryB.combined.gukjin_junk_mongbak_total,
    gukjin_opportunity_count_a: splitSummaryA.combined.gukjin_opportunity_count_total,
    gukjin_opportunity_count_b: splitSummaryB.combined.gukjin_opportunity_count_total,
    gukjin_five_rate_a: splitSummaryA.combined.gukjin_five_rate,
    gukjin_five_rate_b: splitSummaryB.combined.gukjin_five_rate,
    gukjin_five_mongbak_rate_a: splitSummaryA.combined.gukjin_five_mongbak_rate,
    gukjin_five_mongbak_rate_b: splitSummaryB.combined.gukjin_five_mongbak_rate,
    gukjin_junk_mongbak_rate_a: splitSummaryA.combined.gukjin_junk_mongbak_rate,
    gukjin_junk_mongbak_rate_b: splitSummaryB.combined.gukjin_junk_mongbak_rate,
    mean_gold_delta_a: meanGoldDeltaA,
    p10_gold_delta_a: quantile(goldDeltasA, 0.1),
    p50_gold_delta_a: quantile(goldDeltasA, 0.5),
    p90_gold_delta_a: quantile(goldDeltasA, 0.9),
    seat_split_a: splitSummaryA,
    seat_split_b: splitSummaryB,
    eval_time_ms: Math.max(0, Date.now() - evalStartMs),
  };

  const reportLine = `${JSON.stringify(summary)}\n`;
  mkdirSync(dirname(opts.resultOut), { recursive: true });
  writeFileSync(opts.resultOut, reportLine, { encoding: "utf8" });

  if (opts.stdoutFormat === "json") {
    process.stdout.write(reportLine);
    return;
  }

  const consoleSummary = buildConsoleSummary(summary);
  process.stdout.write(formatConsoleSummaryText(consoleSummary));
}

try {
  runModelDuelCli(process.argv.slice(2));
} catch (err) {
  const msg = err && err.stack ? err.stack : String(err);
  process.stderr.write(`${msg}\n`);
  process.exit(1);
}

