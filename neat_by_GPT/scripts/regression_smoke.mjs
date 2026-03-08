import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { legalCandidatesForDecision, selectDecisionPool } from "../../src/ai/decisionRuntime_by_GPT.js";
import { resolveHybridPlayModelSpec } from "./hybridOpponentSpec.mjs";

const SCRIPT_FILE = fileURLToPath(import.meta.url);
const SCRIPT_DIR = path.dirname(SCRIPT_FILE);
const REPO_ROOT = path.resolve(SCRIPT_DIR, "..", "..");

function logStep(text) {
  process.stdout.write(`${text}\n`);
}

function approxEqual(actual, expected, epsilon = 1e-9, message = "values differ") {
  const diff = Math.abs(Number(actual) - Number(expected));
  assert.ok(diff <= epsilon, `${message}: actual=${actual}, expected=${expected}, diff=${diff}`);
}

function repoRelative(absPath) {
  return path.relative(REPO_ROOT, absPath).replace(/\\/g, "/");
}

function pickPreferredExisting(paths) {
  for (const relPath of paths) {
    const fullPath = path.resolve(REPO_ROOT, relPath);
    if (fs.existsSync(fullPath)) return fullPath;
  }
  return "";
}

function findWinnerGenome(rootDir, preferredRelPaths = []) {
  const preferred = pickPreferredExisting(preferredRelPaths);
  if (preferred) return preferred;
  if (!fs.existsSync(rootDir)) return "";

  const stack = [rootDir];
  while (stack.length > 0) {
    const current = stack.pop();
    const entries = fs.readdirSync(current, { withFileTypes: true });
    for (const entry of entries) {
      const nextPath = path.join(current, entry.name);
      if (entry.isDirectory()) {
        if (entry.name.toLowerCase() === "duels") continue;
        stack.push(nextPath);
        continue;
      }
      if (entry.isFile() && entry.name === "winner_genome.json") {
        return nextPath;
      }
    }
  }
  return "";
}

function resolveFixtureGenomes() {
  const gptRoot = path.join(REPO_ROOT, "logs", "NEAT_GPT");
  const mainRoot = path.join(REPO_ROOT, "logs", "NEAT");
  const gptGenome = findWinnerGenome(gptRoot, [
    "logs/NEAT_GPT/neat_phase1_seed90/models/winner_genome.json",
    "logs/NEAT_GPT/runtime_phase1_seed90/models/winner_genome.json",
  ]);
  const mainGenome = findWinnerGenome(mainRoot, [
    "logs/NEAT/neat_phase2_seed203/models/winner_genome.json",
    "logs/NEAT/neat_phase1_seed203/models/winner_genome.json",
  ]);

  if (!gptGenome) {
    throw new Error("GPT winner_genome.json fixture not found under logs/NEAT_GPT");
  }
  if (!mainGenome) {
    throw new Error("main winner_genome.json fixture not found under logs/NEAT");
  }
  return {
    gptGenomeAbs: gptGenome,
    gptGenomeRel: repoRelative(gptGenome),
    mainGenomeAbs: mainGenome,
    mainGenomeRel: repoRelative(mainGenome),
  };
}

function runNodeJson(args) {
  const result = spawnSync(process.execPath, args, {
    cwd: REPO_ROOT,
    encoding: "utf8",
  });
  if (result.status !== 0) {
    throw new Error(
      [
        `command failed: node ${args.join(" ")}`,
        result.stdout?.trim() || "",
        result.stderr?.trim() || "",
      ].filter(Boolean).join("\n")
    );
  }
  const lines = String(result.stdout || "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length <= 0) {
    throw new Error(`command returned empty stdout: node ${args.join(" ")}`);
  }
  return JSON.parse(lines[lines.length - 1]);
}

function testSpecialActionCandidates() {
  const state = {
    phase: "playing",
    currentTurn: "ai",
    board: [
      { id: "B5", month: 5 },
      { id: "B2", month: 2 },
    ],
    players: {
      human: {
        hand: [],
      },
      ai: {
        hand: [
          { id: "S3A", month: 3, passCard: false },
          { id: "S3B", month: 3, passCard: false },
          { id: "S3C", month: 3, passCard: false },
          { id: "B5A", month: 5, passCard: false },
          { id: "B5B", month: 5, passCard: false },
          { id: "B5C", month: 5, passCard: false },
        ],
        shakingDeclaredMonths: [],
      },
    },
  };
  const pool = selectDecisionPool(state, "ai");
  const candidates = legalCandidatesForDecision(pool, "play");

  assert.ok(candidates.includes("shake_start:S3A"), "shake_start candidate missing for first month-3 card");
  assert.ok(candidates.includes("shake_start:S3B"), "shake_start candidate missing for second month-3 card");
  assert.ok(candidates.includes("shake_start:S3C"), "shake_start candidate missing for third month-3 card");
  assert.ok(candidates.includes("bomb:5"), "bomb candidate missing for month 5");
  assert.ok(!candidates.includes("shake_start:B5A"), "shake_start should not be offered on bomb month with board card");
  logStep("[ok] action-space includes shake_start and bomb candidates");
}

function resolveHybridTokens(mainGenomeRel) {
  const phase203Path = path.join(REPO_ROOT, "logs", "NEAT", "neat_phase2_seed203", "models", "winner_genome.json");
  const hybridToken = fs.existsSync(phase203Path)
    ? "hybrid_play(phase2_seed203,H-CL)"
    : `hybrid_play(model:${mainGenomeRel},H-CL)`;
  const hybridGoOnlyToken = fs.existsSync(phase203Path)
    ? "hybrid_play_go(phase2_seed203,H-CL)"
    : `hybrid_play_go(model:${mainGenomeRel},H-CL)`;
  const hybridGoToken = fs.existsSync(phase203Path)
    ? "hybrid_play_go(phase2_seed203,H-NEXg,H-CL)"
    : `hybrid_play_go(model:${mainGenomeRel},H-NEXg,H-CL)`;
  return { hybridToken, hybridGoOnlyToken, hybridGoToken };
}

function testHybridSpecParsing(mainGenomeRel) {
  const { hybridToken, hybridGoOnlyToken, hybridGoToken } = resolveHybridTokens(mainGenomeRel);
  const hybrid = resolveHybridPlayModelSpec(hybridToken, "test");
  assert.equal(hybrid?.kind, "hybrid_play_model");
  assert.equal(hybrid?.heuristicPolicy, "H-CL");
  assert.ok(String(hybrid?.modelPath || "").toLowerCase().endsWith("winner_genome.json"));

  const hybridGoOnly = resolveHybridPlayModelSpec(hybridGoOnlyToken, "test");
  assert.equal(hybridGoOnly?.kind, "hybrid_play_model");
  assert.equal(hybridGoOnly?.goStopPolicy, "H-CL");
  assert.equal(hybridGoOnly?.heuristicPolicy, "");
  assert.equal(hybridGoOnly?.goStopOnly, true);

  const hybridGo = resolveHybridPlayModelSpec(hybridGoToken, "test");
  assert.equal(hybridGo?.kind, "hybrid_play_model");
  assert.equal(hybridGo?.heuristicPolicy, "H-CL");
  assert.equal(hybridGo?.goStopPolicy, "H-NEXg");
  logStep("[ok] hybrid opponent spec parsing works");
  return { hybridToken };
}

function testEvalWorkerHybridAndOverrides(gptGenomeRel, hybridToken) {
  const overrides = {
    goldMeanWeight: 0.11,
    goldP50Weight: 0.12,
    goldP10Weight: 0.13,
    goldCvar10Weight: 0.14,
    tieBreakWeight: 0.19,
    teacherPolicy: "H-CL",
  };
  const summary = runNodeJson([
    "neat_by_GPT/scripts/neat_eval_worker.mjs",
    "--genome", gptGenomeRel,
    "--games", "3",
    "--seed", "gpt_regression_eval_override",
    "--max-steps", "80",
    "--fitness-profile", "phase2",
    "--fitness-overrides", JSON.stringify(overrides),
    "--opponent-policy", hybridToken,
    "--continuous-series", "2",
  ]);

  assert.equal(summary.eval_ok, true);
  assert.equal(summary.opponent_policy, hybridToken);
  assert.equal(summary.opponent_policy_counts[hybridToken], 3);
  assert.equal(summary.imitation_source, "H-CL");
  assert.equal(summary.fitness_overrides.goldMeanWeight, 0.11);
  approxEqual(summary.fitness_tie_break_weight, 0.19, 1e-12, "tie break override mismatch");
  approxEqual(summary.fitness_gold_core_total_weight, 0.5, 1e-12, "gold core total weight mismatch");
  logStep("[ok] eval worker accepts hybrid opponent policy and applies fitness_overrides");
}

function testEvalWorkerHybridMix(gptGenomeRel, hybridToken) {
  const mix = [
    { policy: "H-CL", weight: 1 },
    { policy: hybridToken, weight: 1 },
  ];
  const summary = runNodeJson([
    "neat_by_GPT/scripts/neat_eval_worker.mjs",
    "--genome", gptGenomeRel,
    "--games", "4",
    "--seed", "gpt_regression_eval_mix",
    "--max-steps", "80",
    "--fitness-profile", "phase2",
    "--opponent-policy-mix", JSON.stringify(mix),
    "--continuous-series", "2",
  ]);

  assert.equal(summary.eval_ok, true);
  assert.equal(Array.isArray(summary.opponent_policy_mix), true);
  assert.equal(summary.opponent_policy_mix.length, 2);
  const countTotal = Object.values(summary.opponent_policy_counts || {}).reduce(
    (acc, value) => acc + Number(value || 0),
    0
  );
  assert.equal(countTotal, 4);
  for (const usedPolicy of Object.keys(summary.opponent_policy_counts || {})) {
    assert.ok(
      usedPolicy === "H-CL" || usedPolicy === hybridToken,
      `unexpected policy emitted by mix selection: ${usedPolicy}`
    );
  }
  logStep("[ok] eval worker accepts hybrid opponent mix");
}

function main() {
  const fixtures = resolveFixtureGenomes();
  logStep(`GPT fixture:  ${fixtures.gptGenomeRel}`);
  logStep(`Main fixture: ${fixtures.mainGenomeRel}`);
  testSpecialActionCandidates();
  const { hybridToken } = testHybridSpecParsing(fixtures.mainGenomeRel);
  testEvalWorkerHybridAndOverrides(fixtures.gptGenomeRel, hybridToken);
  testEvalWorkerHybridMix(fixtures.gptGenomeRel, hybridToken);
  logStep("[done] neat_by_GPT regression smoke passed");
}

try {
  main();
} catch (err) {
  const message = err && err.stack ? err.stack : String(err);
  process.stderr.write(`${message}\n`);
  process.exit(1);
}
